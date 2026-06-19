from typing import Dict
from src.agent.state import AgentState
from src.enrichment.domain_parser import parse_domain
from src.enrichment.llm_enricher import LLMEnricher
from src.enrichment.retrieval.supabase_client import SupabaseClient
from src.enrichment.retrieval.filters import build_where_clause, get_tld_family, apply_numeric_filter
from src.enrichment.retrieval.scoring import score_candidates
from src.enrichment.namebio.demand import enqueue_demand

import config

# Initialize shared resources
llm_enricher = LLMEnricher()
supabase_client = SupabaseClient()

LLM_PROMPT_TEMPLATE = """
You are a domain branding and analysis expert.

Task:
1) Generate up to TWO distinct, business-oriented descriptions for this domain.
2) Classify the domain into TWO categories (primary and secondary) from a fixed list.

Domain: {domain_name}

PART 1: DESCRIPTIONS
- Consider BOTH the word(s) in the name and the TLD (e.g., .ai, .io, .com).
- Focus on realistic, commercially viable uses where a real company or project
  would use this as their main brand or product domain.
- Each description MUST:
  - Be 1–3 sentences.
  - Explain HOW the domain could be used (what product/service, who uses it).
  - Mention the kind of business or sector in natural language (e.g., insurance,
    fintech, SaaS, local retail, ecommerce, healthtech, AI tools, education, etc.).
- Generate 1–2 descriptions that together cover the main plausible uses/angles
  of the domain (do NOT exceed 2 descriptions).

PART 2: CATEGORY CLASSIFICATION
From this list, choose:
- ONE PRIMARY category (strongest fit).
- ONE SECONDARY category (next best fit, must be different from primary).

Valid categories:
- Acronym
- Brandable
- Combination
- Descriptive
- Exact match
- Geo-specific
- Generic
- Service-based
- Niche
- Keyword
- Product-based

IMPORTANT:
- Do NOT include any explanations, comments, or markdown fences.
- Return ONLY raw JSON matching the schema, starting with {{ and ending with }}.

Return STRICT JSON with this schema and nothing else:

{{
  "domain": "{domain_name}",
  "primary_category": "Acronym|Brandable|Combination|Descriptive|Exact match|Geo-specific|Generic|Service-based|Niche|Keyword|Product-based",
  "secondary_category": "Acronym|Brandable|Combination|Descriptive|Exact match|Geo-specific|Generic|Service-based|Niche|Keyword|Product-based",
  "descriptions": [
    "First 1–3 sentence business description mentioning where and how the domain is used.",
    "Second 1–3 sentence business description (if needed), or omit if only one is strong."
  ]
}}
"""

def enrichment_node(state: AgentState) ->Dict:
    """
    Node 1: Extract basic features and call LLM for enrichment. 
    """
    try:
        domain_name = state["input_domain"]

        # Parse domain
        parsed = parse_domain(domain_name)

        # call LLms once for categories + description
        enriched = llm_enricher.enrich_domain(domain_name, LLM_PROMPT_TEMPLATE)

        return {
            "sld": parsed["sld"],
            "tld": parsed["tld"],
            "length": parsed["length"],
            "has_numbers": parsed["has_numbers"],
            "primary_category": enriched["primary_category"],
            "secondary_category": enriched["secondary_category"],
            "descriptions": enriched["descriptions"]
        }
    except Exception as e:
        print(f"[DEBUG] enrichment_node: Exception caught: {e}")
        # For testing, return mock data when LLM fails
        parsed = parse_domain(state["input_domain"])
        mock_descriptions = [
            f"{state['input_domain']} is a premium domain suitable for branding purposes in various industries.",
            f"This domain could be used for a company providing services in technology, healthcare, or business sectors."
        ]
        print(f"[DEBUG] enrichment_node: Returning mock data with {len(mock_descriptions)} descriptions")

        return {
            "sld": parsed["sld"],
            "tld": parsed["tld"],
            "length": parsed["length"],
            "primary_category": "Brandable",
            "secondary_category": "Service-based",
            "descriptions": mock_descriptions,
            "error": f"Enrichment failed (using mock data): {str(e)}"
        }

    
def build_queries_node(state: AgentState) -> Dict:
    """
    Node 2: Build query texts from descriptions.
    """
    descriptions = state.get("descriptions", [])

    # Clean descriptions
    queries = []
    for desc in descriptions:
        cleaned = desc.strip()
        if cleaned:
            queries.append(cleaned)
    return {"queries": queries}


def retrieve_node(state: AgentState) ->Dict:
    """ 
    Node 3: Retrieve candidates from ChromaDB with hard filters.
    Implements tiered fallback for unknown TLDs.
    """
    try:
        queries = state["queries"]
        tld_family = get_tld_family(state["tld"])

        # Build where clause from state (with TLD filter)
        where = build_where_clause(
            tld=state["tld"],
            length=state["length"],
            primary_category=state["primary_category"],
            secondary_category=state["secondary_category"],
            include_tld_filter=True
        )

        # Query ChromaDB for each description
        all_candidates = []
        for query_idx, query in enumerate(queries, start=1):
            candidates = supabase_client.query(
                query_texts=[query],
                where=where,
                n_results=config.CHROMA_RESULTS_PER_QUERY
            )
            # Tag each candidate with which query found it
            for candidate in candidates:
                candidate["query_index"] = query_idx
            
            all_candidates.extend(candidates)

        # New: Apply numeric filter BEFORE fallback Check 
        if config.ENABLE_NUMERIC_FILTER:
            all_candidates = apply_numeric_filter(
               all_candidates,
               input_has_numbers=state.get("has_numbers", False),
               threshold = config.NUMERIC_THRESHOLD
            )
        
        # Check if we have enough results for unknown TLDs
        if config.ENABLE_TLD_FALLBACK and len(all_candidates) < config.MIN_RESULTS_THRESHOLD:
            print(f"\n[INFO] Only {len(all_candidates)} results for unknown TLD '{state['tld']}'")
            print(f"[INFO] Expanding search to all TLDs (removing TLD filter)...")
            
            # Build where clause WITHOUT TLD filter
            where_no_tld = build_where_clause(
                tld=state["tld"],
                length=state["length"],
                primary_category=state["primary_category"],
                secondary_category=state["secondary_category"],
                include_tld_filter=False  # Remove TLD filter
            )
            
            # Retry search without TLD filter
            all_candidates = []
            for query_idx, query in enumerate(queries, start=1):
                candidates = supabase_client.query(
                    query_texts=[query],
                    where=where_no_tld,
                    n_results=config.CHROMA_RESULTS_PER_QUERY
                )
                # Tag each candidate with which query found it
                for candidate in candidates:
                    candidate["query_index"] = query_idx
                
                all_candidates.extend(candidates)

            if config.ENABLE_NUMERIC_FILTER:
                all_candidates = apply_numeric_filter(
                    all_candidates,
                    input_has_numbers=state.get("has_numbers", False),
                    threshold = config.NUMERIC_THRESHOLD
                )
            
            print(f"[INFO] Expanded search found {len(all_candidates)} results across all TLDs\n")

        # Demand signal: a user just searched this domain, so it deserves
        # high-quality LLM enrichment. Best-effort and non-blocking — never
        # let this affect the search result.
        enqueue_demand(state.get("input_domain"))

        return {"raw_candidates": all_candidates}

    except Exception as e:
        return {"error": f"Retrieval failed: {str(e)}"}

def score_node(state: AgentState) -> Dict:
    """ Node 4: Score, rerank, and deduplicate candidates."""

    try:
        candidates = state["raw_candidates"]
        input_descriptions = state.get("descriptions", [])

        scored = score_candidates(
            candidates=candidates,
            input_primary=state["primary_category"],
            input_secondary=state["secondary_category"],
            input_tld=state["tld"],
        )
        
        # ===== MATCHING ANALYSIS LOGGING =====
        print("\n" + "="*80)
        print(f"MATCHING ANALYSIS - {state['input_domain']}")
        print("="*80)
        
        # Show input descriptions first
        print("\nINPUT DESCRIPTIONS:")
        for i, desc in enumerate(input_descriptions, start=1):
            print(f"\n  Query #{i}:")
            print(f"  {desc[:150]}...")
        
        print("\n" + "-"*80)
        print("TOP MATCHES:")
        print("-"*80)
        
        for idx, comp in enumerate(scored, start=1):
            query_idx = comp.get("query_index", 0)
            desc_idx = comp.get("desc_index", 1)
            
            print(f"\n{idx}. {comp['domain']} - ${comp['price']:,.0f}")
            print(f"   Score: {comp['score']:.4f} | Semantic: {comp['semantic_sim']:.4f} | Category: {comp['cat_match']:.2f} | Recency: {comp['recency']:.2f}")
            print(f"   Categories: {comp['primary_category']} / {comp['secondary_category']}")
            print(f"   Date: {comp['date']} | Platform: {comp['platform']}")
            print(f"   >> Matched: Input Query #{query_idx} <-> {comp['domain']} Description #{desc_idx}")
            
            # Show matched description if available
            if comp.get("description"):
                print(f"   📝 Description (Desc #{desc_idx}):")
                # Show full description, wrapped nicely
                desc = comp['description']
                # Split into lines of max 100 chars for readability
                import textwrap
                wrapped = textwrap.fill(desc, width=100, initial_indent='      ', subsequent_indent='      ')
                print(wrapped)
            else:
                print(f"   ⚠️  Warning: No description extracted for this match")
        
        print("\n" + "="*80 + "\n")
        # ===== END LOGGING =====
        
        return {"scored_comparables": scored}

    except Exception as e:
        return {"error": f"Scoring failed: {str(e)}"}

def output_node(state: AgentState) -> Dict:
    """Node 5: Format final output as structured JSON."""
    # Check if there was a real error (not mock data)
    error = state.get("error")
    if error and "using mock data" not in str(error):
        return {"result": {"error": error}}

    # Handle missing keys gracefully
    descriptions = state.get("descriptions", [])
    scored_comparables = state.get("scored_comparables", [])

    result = {
        "input_domain": state.get("input_domain", ""),
        "sld": state.get("sld", ""),
        "tld": state.get("tld", ""),
        "length": state.get("length", 0),
        "primary_category": state.get("primary_category", ""),
        "secondary_category": state.get("secondary_category", ""),
        "descriptions": descriptions,
        "comparables": scored_comparables,
        "total_comparables": len(scored_comparables),
        "confidence": "high" if len(scored_comparables) >= 5 else "low"
    }

    # Include error info if it was mock data
    if error:
        result["warning"] = "Used mock data due to API issues"

    return {"result": result}


    
    

        