"""
Background LLM enrichment worker.

Drains llm_enrichment_queue (highest priority first), runs the EXISTING
LLMEnricher to produce high-quality descriptions + categories for non-obvious
domains, upgrades the domain_enrichment row to source=llm / status=enriched_llm,
bumps enrichment_version, and re-embeds (bumping embedding_version).

This is what keeps ingestion non-blocking: ingest never waits on an LLM call;
the worker does the expensive work asynchronously. Run it as a long-lived
process or a cron:

    python -m src.enrichment.namebio.llm_worker            # drain until empty
    python -m src.enrichment.namebio.llm_worker --max 100  # drain up to N jobs
"""

import argparse
import logging
from typing import Dict, Optional

import config
from src.enrichment.domain_parser import parse_domain
from src.enrichment.llm_enricher import LLMEnricher
from src.enrichment.namebio import db
from src.enrichment.namebio.embedder import Embedder
from src.enrichment.namebio.enrichment_cache import EnrichmentCache
from src.enrichment.namebio.queue import LLMQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("namebio.llm_worker")


# Same prompt the agent uses at query time (kept local so the worker doesn't
# import src.agent.nodes, which would pull in a SupabaseClient at import time).
LLM_PROMPT_TEMPLATE = """
You are a domain branding and analysis expert.

Task:
1) Generate up to TWO distinct, business-oriented descriptions for this domain.
2) Classify the domain into TWO categories (primary and secondary) from a fixed list.

Domain: {domain_name}

PART 1: DESCRIPTIONS
- Consider BOTH the word(s) in the name and the TLD (e.g., .ai, .io, .com).
- Focus on realistic, commercially viable uses.
- Each description MUST be 1-3 sentences and explain HOW the domain could be used.

PART 2: CATEGORY CLASSIFICATION
Choose ONE PRIMARY and ONE SECONDARY (different) category from:
Acronym, Brandable, Combination, Descriptive, Exact match, Geo-specific,
Generic, Service-based, Niche, Keyword, Product-based

Return ONLY raw JSON, starting with {{ and ending with }}:
{{
  "domain": "{domain_name}",
  "primary_category": "...",
  "secondary_category": "...",
  "descriptions": ["...", "..."]
}}
"""


class LLMWorker:
    def __init__(self, enricher=None, cache=None, queue=None, embedder=None, conn=None):
        self.conn = conn or db.connect()
        self.enricher = enricher or LLMEnricher()
        self.cache = cache or EnrichmentCache(conn=self.conn)
        self.queue = queue or LLMQueue(conn=self.conn)
        self.embedder = embedder or Embedder(conn=self.conn)

    def _sale_meta_from_embedding(self, domain: str) -> Dict:
        """Best-effort price/date/platform from any existing embedding row."""
        try:
            with db.cursor(self.conn) as cur:
                cur.execute(
                    f"""SELECT metadata FROM {config.DOMAIN_EMBEDDINGS_TABLE}
                            WHERE metadata->>'domain' = %s LIMIT 1""",
                    (domain,),
                )
                row = cur.fetchone()
                if row and row.get("metadata"):
                    m = row["metadata"]
                    return {
                        "price": m.get("price"),
                        "date": m.get("date"),
                        "platform": m.get("platform"),
                    }
        except Exception as e:  # noqa: BLE001 - metadata is optional
            logger.debug("No prior embedding metadata for %s: %s", domain, e)
        return {"price": None, "date": None, "platform": None}

    def process_one(self) -> bool:
        """Claim and process a single job. Returns False when queue is empty."""
        job = self.queue.claim_next()
        if not job:
            return False

        domain = job["domain"]
        try:
            enriched = self.enricher.enrich_domain(domain, LLM_PROMPT_TEMPLATE)
            parsed = parse_domain(domain)
            descriptions = enriched.get("descriptions", [])

            record = {
                "domain": domain,
                "sld": parsed["sld"],
                "tld": parsed["tld"],
                "primary_category": enriched["primary_category"],
                "secondary_category": enriched["secondary_category"],
                "descriptions": descriptions,
                "source": "llm",
                "status": "enriched_llm",
                "queue_reason": None,
                "embedded": False,
                # versioning: this is now the current enrichment version
                "enrichment_version": config.CURRENT_ENRICHMENT_VERSION,
            }
            # keep existing keywords/tokens/confidence if present
            existing = self.cache.get(domain) or {}
            record["keywords"] = existing.get("keywords") or []
            record["tokens"] = existing.get("tokens") or []
            record["confidence"] = existing.get("confidence")
            self.cache.upsert(record)

            # Re-embed with the upgraded descriptions (bumps embedding_version).
            sale = self._sale_meta_from_embedding(domain)
            embed_row = {
                "domain": domain,
                "tld": parsed["tld"],
                "length": parsed["length"],
                "primary_category": enriched["primary_category"],
                "secondary_category": enriched["secondary_category"],
                "keywords": record["keywords"],
                "descriptions": descriptions,
                "source": "llm",
                "price": sale["price"],
                "date": sale["date"],
                "platform": sale["platform"],
                "has_numbers": parsed["has_numbers"],
            }
            self.embedder.embed_and_upsert([embed_row])
            self.cache.mark_embedded(domain, config.CURRENT_EMBEDDING_VERSION)

            self.queue.complete(domain)
            logger.info("LLM-enriched %s (reason=%s)", domain, job.get("queue_reason"))
            return True

        except Exception as e:  # noqa: BLE001
            logger.exception("LLM enrichment failed for %s: %s", domain, e)
            self.queue.fail(domain, str(e))
            return True  # keep draining other jobs

    def drain(self, max_jobs: Optional[int] = None) -> int:
        processed = 0
        while max_jobs is None or processed < max_jobs:
            if not self.process_one():
                break
            processed += 1
        logger.info("Worker drained %d jobs; %d still pending",
                    processed, self.queue.pending_count())
        return processed

    def close(self):
        self.embedder.close()
        self.conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="LLM enrichment queue worker")
    parser.add_argument("--max", type=int, default=None,
                        help="Max jobs to process (default: drain until empty)")
    args = parser.parse_args(argv)
    worker = LLMWorker()
    try:
        worker.drain(args.max)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
