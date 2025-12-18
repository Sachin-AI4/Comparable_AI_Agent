import os
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = "openai/gpt-5.1"  # OpenRouter model format
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # OpenRouter model format

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "domain_embeddings"

DOMAIN_CATEGORIES = [
    "Acronym",
    "Brandable",
    "Combination",
    "Descriptive",
    "Exact match",
    "Geo-specific",
    "Generic",
    "Service-based",
    "Niche",
    "Keyword",
    "Product-based"
]



TLD_FAMILIES = {
    # FAMILY: LEGACY GOLD
    # Reason: The ultimate benchmark. Comps should only be compared against 
    # other.coms unless the SLD is extremely unique. 
    # Price Band: $2,000 - $10,000,000+
    "legacy_gold": [
        ".com"
    ],
    
    # FAMILY: LEGACY STANDARD
    # Reason: Institutional and network trust. High renewal rates and stable 
    # aftermarket for professional entities.
    # Price Band: $500 - $150,000
    "legacy_standard": [
        ".net", ".org", ".info"
    ],
    
    # FAMILY: TECH INNOVATION ELITE
    # Reason: Highest aftermarket ASPs in 2024-25. Primary target for 
    # venture-backed tech startups.
    # Price Band: $5,000 - $300,000
    # "tech_elite": [
    #     ".ai", ".io"
    # ],
    
    # FAMILY: MODERN TECH & BRANDING
    # Reason: Strong alternatives for modern brands. Favored for mobile apps 
    # and SaaS with slightly lower price ceilings than the elite tier.
    # Price Band: $1,000 - $50,000
    "tech_modern": [
        ".ai", ".io", ".co", ".app", ".dev", ".tech", ".cloud", ".software"
    ],
    
    # FAMILY: CORPORATE & SMB LEGAL
    # Reason: Functional business identifiers. Valuation is driven by 
    # Exact Match company names.
    # Price Band: $500 - $25,000
    "corporate_id": [
        ".inc", ".llc", ".ltd", ".biz", ".company", ".corp", ".holdings"
    ],
    
    # FAMILY: E-COMMERCE & RETAIL
    # Reason: Transaction-focused. Valuation is tied to product search 
    # volume and retail keyword strength.
    # Price Band: $500 - $15,000
    "ecommerce": [
        ".shop", ".store", ".market", ".buy", ".deals", ".solutions", ".services"
    ],
    
    # FAMILY: HIGH-VALUE NICHE (GAMING & BETTING)
    # Reason: Industry-specific hyper-liquidity. Premium names in these 
    # zones can rival.com prices.
    # Price Band: $5,000 - $600,000 (SLD dependent)
    "niche_premium": [
        ".bet", ".gg", ".game", ".tv", ".casino", ".poker"
    ],
    
    # FAMILY: CREATIVE ECONOMY
    # Reason: Agency and personal branding. Visual and descriptive power.
    # Price Band: $500 - $10,000
    "creative": [
        ".design", ".art", ".media", ".studio", ".agency", ".photography", ".news"
    ],
    
    # FAMILY: FINANCE & WEB3
    # Reason: Sector-specific trust and fintech association.
    # Price Band: $1,000 - $100,000
    "finance_web3": [
        ".finance", ".money", ".pay", ".crypto", ".cash", ".bank"
    ],
    
    # FAMILY: GEOGRAPHIC TIER 1 (GLOBAL LIQUIDITY)
    # Reason: Strongest regional economies. High tradeability and consumer trust.
    # Price Band: $1,000 - $100,000 (Local market)
    "geo_tier1": [
        ".de", ".uk", ".ca", ".au", ".fr", ".nl", ".jp", ".us", ".eu"
    ],
    
    # FAMILY: GEOGRAPHIC TIER 2 (EMERGING & PERSONAL)
    # Reason: Rapidly growing digital economies and personal branding hacks.
    # Price Band: $500 - $20,000
    "geo_tier2": [
        ".in", ".cn", ".br", ".me", ".sg", ".hk", ".kr", ".it", ".es", ".ch"
    ],
    
    # FAMILY: MODERN GENERIC (SPECULATIVE VOLUME)
    # Reason: High registration, low renewal, binary valuation (premium or nothing).
    # Price Band: $10 - $5,000 (standard), $50,000+ (outlier premiums)
    "generic_modern": [
        ".xyz", ".online", ".site", ".website", ".space", ".fun", ".life", ".world", ".live", ".digital"
    ]
}


MAX_LENGTH_DIFF = 2
CHROMA_RESULTS_PER_QUERY = 50
FINAL_TOP_K = 10
MIN_SCORE_THRESHOLD = 0.5

# Minimum results threshold for unknown TLD fallback
MIN_RESULTS_THRESHOLD = 10

# Whether to enable TLD fallback for unknown TLDs
ENABLE_TLD_FALLBACK = True

SCORE_WEIGHTS = {
    "semantic": 0.6,
    "category": 0.2,
    "recency": 0.2
}

# Recency decay (in days)
RECENCY_BANDS = [
    (90, 1.0),
    (180, 0.9),
    (365, 0.8),
    (730, 0.6),
    (float('inf'), 0.3)
]