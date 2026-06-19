# NameBio → Comparable Agent Integration

Ingests historical domain sales from the **NameBio microservice** into the
Comparable Agent's vector corpus, so the agent can return real sold-domain
comparables. Designed to scale to **~883k sales (3–4k/day)** without an
unaffordable LLM-per-domain backfill.

---

## 1. The core idea — route computation by information content

A deterministic **rule engine** enriches every domain for free and emits a
**confidence score**. That score (plus sale price and real user demand) decides
which small, high-information subset is worth an expensive LLM call. Every
enrichment is cached **permanently and versioned**, so each expensive
computation happens at most once.

```
                          NameBio REST API  (read-only)
                                  │  /namebio/sales?date=…  (paginated)
                                  ▼
                  map saleDate→date, marketplace→platform
                                  ▼
                  dedupe by DOMAIN (enrich once, reuse for all its sales)
                                  ▼
                       enrichment_cache.get(domain)
                       │ hit → reuse (free)
                       │ miss
                       ▼
         rule_engine.enrich(domain) → category, keywords, tokens, confidence
                                  │
   confidence ≥ 0.75 (HIGH) ──────┤→ enriched_rule          → EMBED NOW
   0.45–0.75    (MEDIUM) ─────────┤→ enriched_rule + queue  → EMBED NOW (lazy LLM upgrade)
   confidence < 0.45  (LOW) ──────┤→ queued_for_llm         → DO NOT embed; queue
   price ≥ $10,000 (PREMIUM) ─────┘→ queued_for_llm         → DO NOT embed; queue
                                  ▼
              EMBED ONLY FINALIZED CONTENT (all-MiniLM-L6-v2, 384-dim)
              composite doc → domain_embeddings (idempotent, chunked, durable)
                                  ▼
                  save ingest cursor (resumable backfill)

  BACKGROUND (non-blocking):  llm_worker drains the queue → LLM enrich
                              → upgrade to enriched_llm → re-embed (version bump)

  QUERY PATH:  a user search enqueues that domain with reason='demand'
               (best-effort; never blocks or breaks a search)
```

**Why "embed only finalized content":** low-confidence rows are *not* embedded
on the rule pass — they're embedded once, after the LLM finishes. This avoids
the wasteful rule→embed→llm→re-embed double cost.

---

## 2. What gets written where

| System | Role | Access |
|--------|------|--------|
| **NameBio microservice** (`https://namebio.vps4.auctionhacker.com`) | source of sales | **READ-ONLY** (REST) |
| **Supabase Postgres** (the agent's DB) | vector corpus + ingest state | **WRITES** |

Tables written (see [`sql/001_namebio_integration.sql`](sql/001_namebio_integration.sql)):

- **`domain_enrichment`** — permanent, versioned per-domain enrichment cache
- **`llm_enrichment_queue`** — background LLM work queue (priority-ordered)
- **`ingest_state`** — singleton resumable backfill cursor
- **`domain_embeddings`** — *existing* table; new NameBio rows are tagged
  `metadata.source = 'rule' | 'llm'` (so they're identifiable and removable)

---

## 3. Files

**New**
```
src/enrichment/rule_engine.py            # deterministic categorizer + tokenizer + confidence
src/enrichment/namebio/
  __init__.py
  db.py                                  # shared Supabase connection helper
  client.py                              # NameBio REST client (retry/backoff, field mapping)
  enrichment_cache.py                    # domain_enrichment CRUD (versioned)
  queue.py                               # llm_enrichment_queue CRUD
  routing.py                             # pure confidence-band routing decision
  embedder.py                            # composite doc + batched embed + upsert
  ingest.py                              # backfill/daily orchestrator (dedupe, route, embed, cursor)
  llm_worker.py                          # background queue drainer (LLM + re-embed)
  demand.py                              # query-path demand-enqueue hook
sql/001_namebio_integration.sql          # 3 new tables
tests/test_rule_engine.py
tests/test_namebio_client.py
tests/test_ingest.py
tests/test_enrichment_cache.py
```

**Modified**
```
config.py                # NameBio settings, confidence bands, premium threshold, versions
requirements.txt         # + requests, tenacity
src/agent/nodes.py       # demand-enqueue hook in retrieve_node (non-blocking)
src/enrichment/llm_enricher.py  # tolerant JSON parsing ({{ }}, code fences, prose)
.gitignore               # + .venv/
```

---

## 4. Configuration

All tunables live in [`config.py`](config.py) and are overridable via environment
variables (set these in production — do **not** commit secrets):

| Env var | Default | Meaning |
|---------|---------|---------|
| `NAMEBIO_BASE_URL` | `https://namebio.vps4.auctionhacker.com` | NameBio API base |
| `NAMEBIO_PAGE_SIZE` | `500` | page size for `/namebio/sales` |
| `DOMAIN_EMBEDDINGS_TABLE` | `domainvaluation1.domain_embeddings` | **set to `domain_embeddings`** if that's how your DB resolves it |
| `HIGH_CONFIDENCE` / `MEDIUM_CONFIDENCE` / `LOW_CONFIDENCE` | `0.75` / `0.45` / `0.20` | routing bands |
| `PREMIUM_PRICE_THRESHOLD` | `10000` | sale price forcing LLM enrichment |
| `EMBED_BATCH_SIZE` | `256` | embed/flush chunk size (durability) |
| `CURRENT_ENRICHMENT_VERSION` / `CURRENT_EMBEDDING_VERSION` | `1` / `1` | bump to selectively refresh stale rows |
| `SUPABASE_*` | — | existing DB credentials |
| `OPENROUTER_API_KEY` | — | LLM key (worker only) |

> **Versioning:** to refresh enrichments after improving the rule engine, bump
> `CURRENT_ENRICHMENT_VERSION` and re-run — only rows with
> `enrichment_version < CURRENT` are reprocessed. No full rebuild.

---

## 5. Setup

```bash
# 1. Python 3.11 venv (matches production; torch==2.1.2 has no 3.12+ build)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Apply the DB migration (Supabase SQL editor / psql / the one-liner below)
python -c "from src.enrichment.namebio import db; \
  sql=open('sql/001_namebio_integration.sql',encoding='utf-8').read(); \
  c=db.connect(); cur=c.cursor(); cur.execute(sql); c.commit(); print('migration applied')"

# 3. Set env (.env locally, Coolify in prod). Minimum:
#    SUPABASE_HOST/PORT/DB/USER/PASSWORD, OPENROUTER_API_KEY,
#    NAMEBIO_BASE_URL, DOMAIN_EMBEDDINGS_TABLE=domain_embeddings
```

---

## 6. Running

```bash
# One-time historical backfill (resumable — safe to stop/restart):
python -m src.enrichment.namebio.ingest --backfill --from 2024-01-01 --to 2026-06-18

# Daily incremental (cron; pulls yesterday, mirrors NameBio's own daily job):
python -m src.enrichment.namebio.ingest --daily

# Background LLM worker — drains the queue (LLM enrich + re-embed):
python -m src.enrichment.namebio.llm_worker            # drain until empty
python -m src.enrichment.namebio.llm_worker --max 50   # cap (good for cost control)
```

The backfill is **idempotent** (re-running skips already-enriched domains) and
**resumable** (cursor in `ingest_state`; a crash loses at most one date).

---

## 7. Tests

```bash
python -m pytest tests/ -v        # 31 tests, all mocked (no live DB/API needed)
```

Covers: rule-engine confidence routing, NameBio client field-mapping/pagination/
retry, ingest dedupe + embed-only-finalized + routing, and cache versioning.

---

## 8. Observability / health

```bash
# Enrichment status distribution:
SELECT status, count(*) FROM domain_enrichment GROUP BY status;

# Queue backlog:
SELECT queue_reason, status, count(*) FROM llm_enrichment_queue GROUP BY 1,2;

# NameBio rows now in the corpus:
SELECT count(*) FROM domain_embeddings WHERE metadata->>'source' IN ('rule','llm');

# Backfill progress:
SELECT * FROM ingest_state;
```

---

## 9. Production checklist (before go-live)

- [ ] **Set a real LLM model** in `config.py` — `openai/gpt-5.1` is **not a valid
      model id**. Use a real OpenRouter model (e.g. a current GPT-4o-mini /
      Claude Haiku class model for cheap structured JSON).
- [ ] **Funded OpenRouter key** — the prior prod key was *limit-exceeded* (403).
- [ ] **Rotate any key shared during development** — treat as compromised.
- [ ] **Apply `sql/001_namebio_integration.sql`** to the production Supabase.
- [ ] Confirm `DOMAIN_EMBEDDINGS_TABLE` resolves correctly in prod.
- [ ] Decide whether to keep the ~250 NameBio rows already ingested during
      testing, or purge with:
      `DELETE FROM domain_embeddings WHERE metadata->>'source' IN ('rule','llm');`
- [ ] Do **not** commit `.venv/` or `.env` (both gitignored).

---

## 10. Cost note

Embeddings are **local and free** (sentence-transformers). LLM cost applies
**only** to queued domains (low-confidence / premium / demand) — the rule path
covers the bulk for free, and every LLM result is cached forever. Cap the worker
with `--max` to control spend.

## 11. Known characteristics

- Rule-enriched domains have terse descriptions, so they retrieve with shorter
  semantic reach than existing LLM-rich rows. This is by design — the queue
  upgrades the domains that matter (low-confidence/premium/searched) to rich LLM
  descriptions. Run the `llm_worker` to close that gap over time.
- The demand hook in `retrieve_node` is best-effort and silently no-ops if the
  DB is unreachable — it never affects a live search.
