"""
Deterministic rule-based domain enrichment (the information-theory router).

Given a domain, produce a category classification, keywords, normalized
tokens, a short description, and a CONFIDENCE score in [0, 1]. The confidence
is what routes computation downstream:

    >= HIGH_CONFIDENCE   -> meaning is obvious; accept rule result (free)
    MEDIUM .. HIGH       -> accept but flag for lazy LLM upgrade
    <  MEDIUM            -> meaning is ambiguous/coined; queue for LLM

The guiding idea: low-information, obvious names (mortgagebroker.com,
cryptoexchange.ai) are cheap to classify deterministically and get high
confidence. High-information, non-obvious coined names (zyro.ai, vanta.ai)
score low and are deferred to the LLM, where the expensive call is justified.

Pure module: no network, no DB, no LLM. Categories map onto the agent's
fixed list in config.DOMAIN_CATEGORIES so they stay compatible with
scoring.compute_category_match.
"""

import re
from typing import Dict, List, Tuple

from src.enrichment.domain_parser import parse_domain

# ---------------------------------------------------------------------------
# Lexicons. Kept intentionally small and curated: enough to recognize common
# real words and industry tokens that drive deterministic confidence, without
# pulling a heavy NLP dependency. Tune freely; this is the "tokenizer" the
# plan refers to and the confidence calibration follows directly from it.
# ---------------------------------------------------------------------------

# Common English words used for greedy word-segmentation of an SLD.
# Ordered longest-first at match time. This is a pragmatic, high-precision
# subset rather than a full dictionary.
_COMMON_WORDS = {
    # generic business / tech
    "app", "apps", "web", "site", "online", "digital", "data", "tech", "soft",
    "software", "cloud", "host", "hosting", "server", "net", "code", "dev",
    "api", "bot", "ai", "smart", "auto", "mobile", "phone", "email", "mail",
    "search", "social", "media", "news", "blog", "video", "music", "photo",
    "image", "game", "games", "play", "stream", "live", "chat", "talk",
    "connect", "link", "share", "cloud", "stack", "lab", "labs", "works",
    "studio", "agency", "group", "team", "hub", "zone", "spot", "world",
    "global", "central", "direct", "express", "prime", "pro", "plus", "max",
    "one", "first", "best", "top", "easy", "quick", "fast", "now", "go",
    # commerce / retail
    "shop", "store", "buy", "sell", "sale", "deal", "deals", "market", "mart",
    "trade", "cart", "price", "discount", "coupon", "outlet", "retail",
    "wholesale", "order", "ship", "shipping", "delivery", "product", "goods",
    # finance
    "pay", "payment", "payments", "bank", "banking", "finance", "financial",
    "money", "cash", "fund", "funds", "capital", "invest", "investment",
    "loan", "loans", "credit", "debit", "mortgage", "insurance", "insure",
    "tax", "taxes", "wallet", "coin", "coins", "token", "crypto", "exchange",
    "trading", "stock", "stocks", "wealth", "asset", "assets",
    # health
    "health", "healthcare", "medical", "med", "meds", "doctor", "doc",
    "clinic", "care", "dental", "dentist", "pharma", "pharmacy", "fitness",
    "gym", "diet", "wellness", "therapy", "vision", "vet",
    # real estate / home
    "home", "homes", "house", "houses", "estate", "realty", "rent", "rental",
    "property", "properties", "land", "build", "builder", "construction",
    "roof", "roofing", "kitchen", "garden", "furniture",
    # travel
    "travel", "trip", "tour", "tours", "flight", "flights", "hotel", "hotels",
    "book", "booking", "cruise", "vacation", "holiday", "rental",
    # education
    "learn", "learning", "edu", "education", "school", "academy", "course",
    "courses", "class", "tutor", "study", "training", "teach", "college",
    "university", "student",
    # food
    "food", "eat", "restaurant", "cafe", "coffee", "kitchen", "recipe",
    "recipes", "cook", "cooking", "menu", "pizza", "burger", "bar", "grill",
    # legal / professional
    "law", "legal", "lawyer", "attorney", "broker", "consult", "consulting",
    "advisor", "advisory", "service", "services", "solution", "solutions",
    "partner", "partners", "expert", "experts", "support",
    # people / generic nouns
    "my", "the", "your", "our", "us", "people", "person", "life", "love",
    "love", "family", "kids", "baby", "pet", "pets", "dog", "cat", "city",
    "town", "place", "point", "house", "club", "guide", "wiki", "info",
    "review", "reviews", "rate", "rating", "compare", "find", "list",
    # gaming / betting
    "bet", "bets", "betting", "casino", "poker", "slots", "win", "lucky",
    "gaming", "esports", "sport", "sports", "fantasy",
    # energy / industry
    "energy", "solar", "power", "electric", "green", "eco", "water", "oil",
    "gas", "mining", "steel",
}

# Industry tokens -> (sector_keyword, category_hint). Presence of these in the
# SLD both supplies a keyword and biases the category. category_hint is one of
# config.DOMAIN_CATEGORIES.
_INDUSTRY_TOKENS = {
    "pay": ("fintech", "Service-based"),
    "payment": ("fintech", "Service-based"),
    "bank": ("fintech", "Service-based"),
    "finance": ("fintech", "Descriptive"),
    "money": ("fintech", "Descriptive"),
    "cash": ("fintech", "Descriptive"),
    "crypto": ("crypto", "Descriptive"),
    "coin": ("crypto", "Descriptive"),
    "token": ("crypto", "Descriptive"),
    "exchange": ("crypto", "Service-based"),
    "trading": ("fintech", "Service-based"),
    "invest": ("fintech", "Service-based"),
    "loan": ("fintech", "Service-based"),
    "credit": ("fintech", "Descriptive"),
    "insurance": ("insurance", "Service-based"),
    "mortgage": ("real-estate", "Descriptive"),
    "health": ("health", "Descriptive"),
    "medical": ("health", "Descriptive"),
    "clinic": ("health", "Service-based"),
    "care": ("health", "Service-based"),
    "pharma": ("health", "Descriptive"),
    "fitness": ("health", "Descriptive"),
    "shop": ("ecommerce", "Service-based"),
    "store": ("ecommerce", "Service-based"),
    "buy": ("ecommerce", "Service-based"),
    "market": ("ecommerce", "Descriptive"),
    "deals": ("ecommerce", "Descriptive"),
    "ai": ("ai", "Product-based"),
    "app": ("saas", "Product-based"),
    "cloud": ("saas", "Product-based"),
    "software": ("saas", "Descriptive"),
    "tech": ("tech", "Descriptive"),
    "data": ("tech", "Descriptive"),
    "game": ("gaming", "Product-based"),
    "casino": ("gaming", "Niche"),
    "poker": ("gaming", "Niche"),
    "bet": ("gaming", "Niche"),
    "law": ("legal", "Service-based"),
    "legal": ("legal", "Service-based"),
    "broker": ("brokerage", "Service-based"),
    "estate": ("real-estate", "Descriptive"),
    "realty": ("real-estate", "Descriptive"),
    "travel": ("travel", "Descriptive"),
    "hotel": ("travel", "Service-based"),
    "solar": ("energy", "Descriptive"),
    "energy": ("energy", "Descriptive"),
    "learn": ("education", "Service-based"),
    "edu": ("education", "Descriptive"),
    "food": ("food", "Descriptive"),
    "coffee": ("food", "Descriptive"),
}

# Vowels for pronounceability heuristics (brandable detection).
_VOWELS = set("aeiou")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _split_explicit(sld: str) -> List[str]:
    """Split on hyphens and CamelCase boundaries; lowercase the result."""
    # CamelCase -> spaces
    spaced = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', sld)
    # hyphens / underscores / digits boundaries -> spaces
    spaced = re.sub(r'[-_]+', ' ', spaced)
    parts = [p.lower() for p in spaced.split() if p]
    return parts


def _greedy_segment(token: str) -> Tuple[List[str], float]:
    """
    Greedy longest-match word segmentation of a single lowercase token against
    _COMMON_WORDS. Returns (words, coverage) where coverage is the fraction of
    characters that were absorbed into recognized words. High coverage => the
    name is composed of real words => high confidence.
    """
    if not token:
        return [], 0.0

    words: List[str] = []
    i = 0
    n = len(token)
    covered = 0
    max_word_len = 12

    while i < n:
        matched = None
        # try longest substring first
        upper = min(n, i + max_word_len)
        for j in range(upper, i, -1):
            cand = token[i:j]
            if len(cand) >= 2 and cand in _COMMON_WORDS:
                matched = cand
                break
        if matched:
            words.append(matched)
            covered += len(matched)
            i += len(matched)
        else:
            i += 1

    coverage = covered / n if n else 0.0
    return words, coverage


def tokenize(sld: str) -> Tuple[List[str], float]:
    """
    Tokenize an SLD into normalized word tokens and return a coverage score.

    Combines explicit boundaries (hyphen/CamelCase) with greedy dictionary
    segmentation of each remaining chunk. Coverage is the character-weighted
    fraction recognized as real words across the whole SLD.
    """
    chunks = _split_explicit(sld) or [sld.lower()]
    all_words: List[str] = []
    total_chars = 0
    covered_chars = 0

    for chunk in chunks:
        total_chars += len(chunk)
        if chunk in _COMMON_WORDS:
            all_words.append(chunk)
            covered_chars += len(chunk)
            continue
        words, cov = _greedy_segment(chunk)
        all_words.extend(words)
        covered_chars += int(round(cov * len(chunk)))

    coverage = covered_chars / total_chars if total_chars else 0.0
    # de-dupe preserving order
    seen = set()
    tokens = []
    for w in all_words:
        if w not in seen:
            seen.add(w)
            tokens.append(w)
    return tokens, coverage


# ---------------------------------------------------------------------------
# Pronounceability (brandable detection)
# ---------------------------------------------------------------------------

def _pronounceability(sld: str) -> float:
    """
    Rough pronounceability in [0,1]. Coined-but-sayable names (zyro, vanta,
    nexa) score moderate; random consonant strings score low. Used only to
    decide between Brandable-with-low-confidence vs reject.
    """
    s = re.sub(r'[^a-z]', '', sld.lower())
    if not s:
        return 0.0
    vowel_ratio = sum(c in _VOWELS for c in s) / len(s)
    # longest consonant run
    longest = run = 0
    for c in s:
        if c not in _VOWELS:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    # ideal vowel ratio ~0.4; penalize long consonant runs
    vowel_score = 1.0 - min(abs(vowel_ratio - 0.4) / 0.4, 1.0)
    run_penalty = min(longest / 5.0, 1.0)
    return max(0.0, vowel_score * (1.0 - 0.5 * run_penalty))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def enrich(domain: str) -> Dict:
    """
    Deterministically enrich a domain.

    Returns a dict:
        {
          "domain", "sld", "tld", "length", "has_numbers",
          "primary_category", "secondary_category",
          "keywords": [...], "tokens": [...],
          "description": str,
          "confidence": float in [0,1],
          "source": "rule",
        }
    """
    parsed = parse_domain(domain)
    sld = parsed["sld"]
    tld = parsed["tld"]
    length = parsed["length"]
    has_numbers = parsed["has_numbers"]
    numeric_pct = parsed["numeric_percentage"]

    tokens, coverage = tokenize(sld)

    # Industry signal: which recognized tokens map to a sector?
    sectors: List[str] = []
    industry_hint = None
    keywords: List[str] = []
    for t in tokens:
        if t in _INDUSTRY_TOKENS:
            sector, hint = _INDUSTRY_TOKENS[t]
            if sector not in sectors:
                sectors.append(sector)
            if industry_hint is None:
                industry_hint = hint
            keywords.append(t)
    # keywords default to recognized tokens
    if not keywords:
        keywords = list(tokens)

    primary, secondary, confidence = _classify(
        sld=sld,
        tokens=tokens,
        coverage=coverage,
        has_numbers=has_numbers,
        numeric_pct=numeric_pct,
        industry_hint=industry_hint,
    )

    description = _describe(sld, tld, tokens, sectors, primary)

    return {
        "domain": domain.lower(),
        "sld": sld,
        "tld": tld,
        "length": length,
        "has_numbers": has_numbers,
        "primary_category": primary,
        "secondary_category": secondary,
        "keywords": keywords,
        "tokens": tokens,
        "description": description,
        "confidence": round(confidence, 3),
        "source": "rule",
    }


def _classify(sld, tokens, coverage, has_numbers, numeric_pct, industry_hint):
    """Return (primary_category, secondary_category, confidence)."""
    n_words = len(tokens)

    # 1) Numeric-heavy SLD -> Generic, high confidence (meaning is "just numbers").
    if numeric_pct >= 0.5:
        return "Generic", "Brandable", 0.85

    # 2) Multi-word dictionary compound with strong coverage -> Descriptive/Keyword.
    #    e.g. mortgagebroker, cryptoexchange, onlinebanking.
    if n_words >= 2 and coverage >= 0.85:
        primary = "Descriptive"
        secondary = industry_hint if industry_hint and industry_hint != primary else "Keyword"
        # more words + full coverage = higher confidence
        conf = min(0.92 + 0.02 * (n_words - 2), 0.97)
        return primary, secondary, conf

    # 3) Single recognized word -> Keyword / Exact match, high confidence.
    if n_words == 1 and coverage >= 0.9:
        primary = "Keyword"
        secondary = industry_hint if industry_hint and industry_hint != primary else "Descriptive"
        return primary, secondary, 0.85

    # 4) Partial coverage with an industry token -> Combination, medium-high.
    if industry_hint and coverage >= 0.5:
        primary = "Combination"
        secondary = industry_hint if industry_hint != primary else "Descriptive"
        return primary, secondary, 0.6

    # 5) Some recognized words but low coverage -> Combination, medium.
    if n_words >= 1 and coverage >= 0.5:
        return "Combination", "Brandable", 0.5

    # 6) No real words -> coined/brandable. Confidence driven by pronounceability.
    pron = _pronounceability(sld)
    if pron >= 0.55:
        # sayable coined name (zyro, vanta) -> Brandable, but LOW confidence:
        # the actual meaning/positioning is non-obvious and deserves the LLM.
        return "Brandable", "Generic", round(0.15 + 0.1 * (pron - 0.55), 3)
    # unpronounceable / random -> still Brandable but very low confidence.
    return "Brandable", "Acronym", 0.1


def _describe(sld, tld, tokens, sectors, primary):
    """Build a short deterministic description for the embed document."""
    if sectors:
        sector_str = " / ".join(sectors)
        return (
            f"{sld}{tld} is a {primary.lower()} domain related to {sector_str}. "
            f"Suitable as a brand or product domain in the {sectors[0]} space."
        )
    if tokens:
        return (
            f"{sld}{tld} is a {primary.lower()} domain built from the term(s) "
            f"{', '.join(tokens)}. Suitable as a descriptive brand domain."
        )
    return (
        f"{sld}{tld} is a short, {primary.lower()} domain. "
        f"A coined, brandable name suitable for a new product or company."
    )
