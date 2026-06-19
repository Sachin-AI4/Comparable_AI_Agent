"""
Embedding builder + upserter for ingested NameBio domains.

- build_document(meta): composite text that is embedded. Includes domain,
  categories, keywords, and a `Description:` marker so the existing
  description extractor in scoring.py keeps working.
- Embedder.embed_and_upsert(rows): batch-encodes finalized enrichment rows
  with all-MiniLM-L6-v2 (384-dim, matching the live vector(384) column) and
  upserts one vector PER DESCRIPTION into domainvaluation1.domain_embeddings,
  keyed `domain__descN` for idempotency.

Only FINALIZED content reaches here (status enriched_rule / enriched_llm) —
queued_for_llm rows are never embedded, avoiding double embedding work.
"""

import logging
from typing import Dict, List

from sentence_transformers import SentenceTransformer

import config
from src.enrichment.namebio import db

logger = logging.getLogger(__name__)

EXPECTED_DIM = 384


def build_document(meta: Dict, description: str) -> str:
    """
    Build the composite embedding document for one (domain, description).

    Keeps the `Description:` marker because scoring.py extracts the description
    by splitting on it. Keywords/categories up top measurably improve semantic
    retrieval vs embedding the description alone.
    """
    keywords = meta.get("keywords") or []
    kw_str = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
    return (
        f"Domain: {meta.get('domain', '')}\n"
        f"Primary Category: {meta.get('primary_category', '')}\n"
        f"Secondary Category: {meta.get('secondary_category', '')}\n"
        f"Keywords: {kw_str}\n"
        f"Description: {description}"
    )


def _embedding_to_pg(vec: List[float]) -> str:
    """Format a python list as a pgvector literal string."""
    return "[" + ",".join(map(str, vec)) + "]"


class Embedder:
    def __init__(self, model: SentenceTransformer = None, conn=None):
        self.model = model or SentenceTransformer(config.EMBEDDING_MODEL)
        self._own_conn = conn is None
        self.conn = conn or db.connect()
        self.table = config.DOMAIN_EMBEDDINGS_TABLE

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=config.EMBED_BATCH_SIZE,
        ).tolist()
        # Hard guard: the live column is vector(384); never write a mismatch.
        if vectors and len(vectors[0]) != EXPECTED_DIM:
            raise ValueError(
                f"Embedding dim {len(vectors[0])} != expected {EXPECTED_DIM}. "
                f"Model {config.EMBEDDING_MODEL} is incompatible with the "
                f"{self.table} vector(384) column."
            )
        return vectors

    def embed_and_upsert(self, enrichment_rows: List[Dict]) -> int:
        """
        Embed and upsert all descriptions for the given enrichment rows.

        Each row is a dict shaped like a domain_enrichment record (must carry
        domain, tld, length, primary/secondary_category, keywords, descriptions,
        plus per-sale price/date/platform/has_numbers from ingest).

        Returns the number of (domain, description) vectors upserted.
        """
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for row in enrichment_rows:
            descriptions = row.get("descriptions") or []
            if isinstance(descriptions, str):
                descriptions = [descriptions]
            if not descriptions:
                continue

            for desc_idx, desc in enumerate(descriptions, start=1):
                desc = (desc or "").strip()
                if not desc:
                    continue
                meta = {
                    "domain": row["domain"],
                    "tld": row.get("tld"),
                    "length": row.get("length"),
                    "price": row.get("price"),
                    "platform": row.get("platform"),
                    "date": row.get("date"),
                    "primary_category": row.get("primary_category"),
                    "secondary_category": row.get("secondary_category"),
                    "keywords": row.get("keywords") or [],
                    "desc_index": desc_idx,
                    "has_numbers": row.get("has_numbers", False),
                    "source": row.get("source", "rule"),
                }
                ids.append(f"{row['domain']}__desc{desc_idx}")
                documents.append(build_document(meta, desc))
                metadatas.append(meta)

        if not ids:
            return 0

        vectors = self.encode(documents)
        self._upsert(documents, metadatas, vectors)
        return len(ids)

    def _upsert(self, documents, metadatas, vectors):
        """
        Idempotent write keyed on (metadata->>'domain', metadata->>'desc_index').

        The existing `domain_embeddings` table inserts only (content, metadata,
        embedding) — its `id` is DB-generated (see restore_supabase.py), so we
        cannot ON CONFLICT on a `domain__descN` string key. Instead we
        delete-then-insert per (domain, desc_index): re-running ingest replaces
        a domain's vectors rather than duplicating them.
        """
        import json

        # Group rows by domain so we delete each domain's prior vectors once.
        domains = sorted({m["domain"] for m in metadatas})

        with db.cursor(self.conn) as cur:
            # Remove any prior vectors for these domains (all desc indices).
            cur.execute(
                f"""DELETE FROM {self.table}
                        WHERE metadata->>'domain' = ANY(%s)""",
                (domains,),
            )
            insert_sql = f"""
                INSERT INTO {self.table} (content, metadata, embedding)
                VALUES (%s, %s::jsonb, %s::vector);
            """
            for i in range(len(metadatas)):
                cur.execute(
                    insert_sql,
                    (
                        documents[i],
                        json.dumps(metadatas[i]),
                        _embedding_to_pg(vectors[i]),
                    ),
                )

    def close(self):
        if self._own_conn and self.conn:
            self.conn.close()
