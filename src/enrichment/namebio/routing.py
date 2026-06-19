"""
Pure routing decision: given a rule-engine result and the best sale price for
a domain, decide enrichment status, queue reason, and whether to embed now.

Kept dependency-free (just config) so it is trivially unit-testable without a
DB or network. This encodes the information-theory policy:

    price >= PREMIUM            -> queue for LLM (premium_domain), DON'T embed yet
    confidence >= HIGH          -> accept rule result, embed now, no queue
    MEDIUM <= confidence < HIGH -> accept rule result, embed now, queue lazily
    confidence < MEDIUM         -> queue for LLM (low_confidence), DON'T embed yet
"""

from dataclasses import dataclass
from typing import Optional

import config


@dataclass
class RoutingDecision:
    status: str                 # 'enriched_rule' | 'queued_for_llm'
    queue_reason: Optional[str] # 'premium_domain'|'low_confidence'|None
    embed_now: bool             # embed only finalized content
    enqueue: bool               # add to background LLM queue


def decide(confidence: float, best_price: float) -> RoutingDecision:
    # Premium overrides confidence: high-value names always deserve the LLM.
    # We still do NOT embed the rule result first (avoids double embedding);
    # the worker embeds once the LLM finishes.
    if best_price is not None and best_price >= config.PREMIUM_PRICE_THRESHOLD:
        return RoutingDecision(
            status="queued_for_llm",
            queue_reason="premium_domain",
            embed_now=False,
            enqueue=True,
        )

    if confidence >= config.HIGH_CONFIDENCE:
        return RoutingDecision(
            status="enriched_rule",
            queue_reason=None,
            embed_now=True,
            enqueue=False,
        )

    if confidence >= config.MEDIUM_CONFIDENCE:
        # Good enough to embed now, but flag for a later quality upgrade.
        return RoutingDecision(
            status="enriched_rule",
            queue_reason="low_confidence",
            embed_now=True,
            enqueue=True,
        )

    # Ambiguous/coined name: defer entirely to the LLM, embed only after.
    return RoutingDecision(
        status="queued_for_llm",
        queue_reason="low_confidence",
        embed_now=False,
        enqueue=True,
    )
