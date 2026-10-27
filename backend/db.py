"""
PrismRoute — Database connection & schema initialisation.

Reads DATABASE_URL from the environment (default: local pgvector container)
and exposes helpers used by every other module.
"""

import os
import psycopg2
from pgvector.psycopg2 import register_vector

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/prism_router",
)


def get_connection(register_vec: bool = True):
    """Return a new psycopg2 connection.

    Parameters
    ----------
    register_vec : bool, optional
        If *True* (default), registers the pgvector type on the
        connection so that vector columns are automatically
        converted to/from numpy arrays.
    """
    conn = psycopg2.connect(DATABASE_URL)
    if register_vec:
        register_vector(conn)
    return conn


def init_db() -> None:
    """Create the pgvector extension and all application tables."""

    # --- Step 1: enable the vector extension (needs autocommit) ----------
    conn = get_connection(register_vec=False)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.close()

    # --- Step 2: create tables -------------------------------------------
    conn = get_connection(register_vec=True)
    with conn.cursor() as cur:
        # Table 1 — semantic_cache
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id              SERIAL PRIMARY KEY,
                query_hash      TEXT UNIQUE NOT NULL,
                query_text      TEXT NOT NULL,
                embedding       vector(384) NOT NULL,
                metadata        JSONB NOT NULL,
                response_text   TEXT NOT NULL,
                tier_used       TEXT NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_embedding_hnsw
                ON semantic_cache
             USING hnsw (embedding vector_cosine_ops);
        """)

        # Table 2 — telemetry_logs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id                      SERIAL PRIMARY KEY,
                timestamp               TIMESTAMPTZ DEFAULT NOW(),
                query_text              TEXT NOT NULL,
                cache_status            TEXT NOT NULL,
                selected_tier           TEXT NOT NULL,
                p_small                 FLOAT,
                p_medium                FLOAT,
                p_large                 FLOAT,
                input_tokens            INT NOT NULL,
                expected_output_tokens  INT NOT NULL,
                actual_output_tokens    INT NOT NULL,
                actual_cost_usd         FLOAT NOT NULL,
                baseline_large_cost_usd FLOAT NOT NULL,
                latency_ms              FLOAT NOT NULL,
                fallback_triggered      BOOLEAN DEFAULT FALSE
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_telemetry_ts
                ON telemetry_logs (timestamp DESC);
        """)
    conn.commit()
    conn.close()
