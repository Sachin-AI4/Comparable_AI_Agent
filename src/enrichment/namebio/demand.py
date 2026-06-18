"""
Demand-driven enqueue hook for the query path.

When a user searches a domain, that is the strongest possible signal the domain
deserves expensive LLM enrichment. This enqueues the searched domain for an LLM
upgrade IF it is not already finalized by the LLM — but it is strictly
best-effort and MUST NEVER break or slow a search: every failure is swallowed.

The search itself proceeds on whatever corpus already exists; this only seeds
future quality improvements.
"""

import logging

logger = logging.getLogger("namebio.demand")


def enqueue_demand(domain: str) -> None:
    """
    Best-effort: queue `domain` for LLM enrichment with reason='demand'.
    Skips if already enriched by the LLM. Never raises.
    """
    if not domain:
        return
    conn = None
    try:
        from src.enrichment.namebio import db
        from src.enrichment.namebio.enrichment_cache import EnrichmentCache
        from src.enrichment.namebio.queue import LLMQueue

        conn = db.connect()
        cache = EnrichmentCache(conn=conn)
        existing = cache.get(domain)
        # Already LLM-finalized -> nothing to gain from re-queuing.
        if existing and existing.get("status") == "enriched_llm":
            return
        LLMQueue(conn=conn).enqueue(domain.lower(), "demand")
        logger.info("Queued %s for LLM enrichment (demand)", domain.lower())
    except Exception as e:  # noqa: BLE001 - demand hook must never break search
        logger.debug("Demand enqueue skipped for %s: %s", domain, e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
