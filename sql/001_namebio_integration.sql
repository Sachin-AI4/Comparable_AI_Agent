-- =====================================================================
-- 001_namebio_integration.sql
-- NameBio -> Comparable AI Agent integration schema.
--
-- Apply manually (Supabase SQL editor or psql). NOT auto-run by the app.
-- Lives alongside the existing vector table `domainvaluation1.domain_embeddings`
-- (embedding vector(384)) which is NOT modified here.
--
-- Tables:
--   domain_enrichment    permanent, versioned per-domain enrichment cache
--   llm_enrichment_queue background work queue (decouples ingest from LLM latency)
--   ingest_state         resumable backfill cursor (singleton)
-- =====================================================================

-- ---------------------------------------------------------------------
-- domain_enrichment : one row per unique domain, reused across all sales.
-- Enrichment is cached forever and versioned so a future tokenizer/model
-- change can selectively refresh via `WHERE enrichment_version < N`
-- instead of a full table rebuild.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS domain_enrichment (
    domain              text PRIMARY KEY,                 -- full domain, lowercased
    sld                 text,
    tld                 text,                             -- e.g. '.ai'

    primary_category    text,                             -- nullable until enriched
    secondary_category  text,
    keywords            jsonb       DEFAULT '[]'::jsonb,  -- e.g. ["pay"]
    tokens              jsonb       DEFAULT '[]'::jsonb,  -- normalized SLD tokens (reused by retrieval)
    descriptions        jsonb       DEFAULT '[]'::jsonb,  -- list of strings

    confidence          real,                             -- rule-engine confidence [0,1]
    source              text,                             -- 'rule' | 'llm'
    status              text        NOT NULL DEFAULT 'queued_for_llm',
                                                          -- 'enriched_rule' | 'queued_for_llm' | 'enriched_llm'
    queue_reason        text,                             -- 'low_confidence'|'premium_domain'|'embeddings_missing'|'demand'|NULL

    embedded            boolean     NOT NULL DEFAULT false,
    enrichment_version  int         NOT NULL DEFAULT 1,
    embedding_version   int         NOT NULL DEFAULT 1,

    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_domain_enrichment_status
    ON domain_enrichment (status);
CREATE INDEX IF NOT EXISTS idx_domain_enrichment_enr_version
    ON domain_enrichment (enrichment_version);
CREATE INDEX IF NOT EXISTS idx_domain_enrichment_emb_version
    ON domain_enrichment (embedding_version);
CREATE INDEX IF NOT EXISTS idx_domain_enrichment_queue_reason
    ON domain_enrichment (queue_reason);
-- Fast lookup of finalized-but-unembedded rows during ingest.
CREATE INDEX IF NOT EXISTS idx_domain_enrichment_embed_pending
    ON domain_enrichment (embedded, status);


-- ---------------------------------------------------------------------
-- llm_enrichment_queue : background LLM work, drained by llm_worker.
-- One open job per domain. Priority orders the drain: premium > demand
-- > low_confidence (higher number = drained first).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_enrichment_queue (
    id            bigserial PRIMARY KEY,
    domain        text        NOT NULL UNIQUE,
    queue_reason  text        NOT NULL,                  -- why it was queued
    priority      int         NOT NULL DEFAULT 0,        -- premium=30, demand=20, low_confidence=10
    attempts      int         NOT NULL DEFAULT 0,
    status        text        NOT NULL DEFAULT 'pending',-- 'pending'|'processing'|'done'|'failed'
    last_error    text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

-- Drain order: pending first, then by priority desc, oldest first.
CREATE INDEX IF NOT EXISTS idx_llm_queue_drain
    ON llm_enrichment_queue (status, priority DESC, created_at ASC);


-- ---------------------------------------------------------------------
-- ingest_state : singleton (id=1) holding resumable cursors so an
-- 883k-row backfill can stop and resume. Mirrors NameBio's own cursor.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_state (
    id                   int PRIMARY KEY DEFAULT 1,
    last_backfill_cursor date,        -- oldest date processed walking backwards
    last_daily_date      date,        -- most recent date pulled by --daily
    updated_at           timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT ingest_state_singleton CHECK (id = 1)
);

INSERT INTO ingest_state (id) VALUES (1)
ON CONFLICT (id) DO NOTHING;
