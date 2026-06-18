"""
Shared Supabase/PostgreSQL connection helper for the NameBio ingest pipeline.

Uses the same credentials as the existing SupabaseClient
(config.SUPABASE_*). Connections are created on demand; callers own the
lifecycle (use as a context manager or call close()).
"""

import logging
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

import config

logger = logging.getLogger(__name__)


def connect():
    """Open a new psycopg2 connection with RealDictCursor rows."""
    conn = psycopg2.connect(
        host=config.SUPABASE_HOST,
        port=config.SUPABASE_PORT,
        database=config.SUPABASE_DB,
        user=config.SUPABASE_USER,
        password=config.SUPABASE_PASSWORD,
        cursor_factory=RealDictCursor,
    )
    return conn


@contextmanager
def cursor(conn):
    """Transactional cursor: commits on success, rolls back on error."""
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
