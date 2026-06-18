"""
CRUD for the background LLM enrichment queue.

Decouples ingestion from LLM latency: ingest never blocks on an LLM call, it
just enqueues low-confidence / premium / demand-driven domains. The llm_worker
drains the queue asynchronously, highest priority first
(premium > demand > embeddings_missing > low_confidence).
"""

import logging
from typing import Dict, List, Optional

import config
from src.enrichment.namebio import db

logger = logging.getLogger(__name__)


class LLMQueue:
    def __init__(self, conn=None):
        self._own_conn = conn is None
        self.conn = conn or db.connect()

    def enqueue(self, domain: str, queue_reason: str) -> None:
        """
        Add (or re-prioritize) a domain in the queue. One open job per domain.
        If a job already exists, keep the HIGHER priority/reason (so a later
        `demand` hit can escalate a `low_confidence` job).
        """
        domain = domain.lower()
        priority = config.QUEUE_PRIORITY.get(queue_reason, 0)
        with db.cursor(self.conn) as cur:
            cur.execute(
                """
                INSERT INTO llm_enrichment_queue (domain, queue_reason, priority, status)
                VALUES (%s, %s, %s, 'pending')
                ON CONFLICT (domain) DO UPDATE SET
                    queue_reason = CASE
                        WHEN EXCLUDED.priority > llm_enrichment_queue.priority
                        THEN EXCLUDED.queue_reason ELSE llm_enrichment_queue.queue_reason END,
                    priority = GREATEST(llm_enrichment_queue.priority, EXCLUDED.priority),
                    -- re-open a finished/failed job if it's requested again
                    status = CASE
                        WHEN llm_enrichment_queue.status IN ('done', 'failed')
                        THEN 'pending' ELSE llm_enrichment_queue.status END,
                    updated_at = now();
                """,
                (domain, queue_reason, priority),
            )

    def claim_next(self) -> Optional[Dict]:
        """
        Atomically claim the highest-priority pending job (FOR UPDATE SKIP
        LOCKED so multiple workers don't collide). Returns the row or None.
        """
        with db.cursor(self.conn) as cur:
            cur.execute(
                """
                UPDATE llm_enrichment_queue
                   SET status = 'processing', attempts = attempts + 1, updated_at = now()
                 WHERE id = (
                     SELECT id FROM llm_enrichment_queue
                      WHERE status = 'pending'
                      ORDER BY priority DESC, created_at ASC
                      FOR UPDATE SKIP LOCKED
                      LIMIT 1
                 )
                RETURNING *;
                """
            )
            return cur.fetchone()

    def complete(self, domain: str) -> None:
        with db.cursor(self.conn) as cur:
            cur.execute(
                """UPDATE llm_enrichment_queue
                       SET status = 'done', updated_at = now()
                     WHERE domain = %s""",
                (domain.lower(),),
            )

    def fail(self, domain: str, error: str, max_attempts: int = 5) -> None:
        """Mark failed; goes back to pending if under the attempt cap."""
        with db.cursor(self.conn) as cur:
            cur.execute(
                """
                UPDATE llm_enrichment_queue
                   SET status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'pending' END,
                       last_error = %s, updated_at = now()
                 WHERE domain = %s
                """,
                (max_attempts, str(error)[:500], domain.lower()),
            )

    def pending_count(self) -> int:
        with db.cursor(self.conn) as cur:
            cur.execute(
                "SELECT count(*) AS n FROM llm_enrichment_queue WHERE status = 'pending'"
            )
            return cur.fetchone()["n"]

    def close(self):
        if self._own_conn and self.conn:
            self.conn.close()
