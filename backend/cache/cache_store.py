"""
PrismRoute — Two-layer semantic cache with Safe-Reuse Guard support.

Exposes a clean, function-based API consumed by the router and dashboard.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from backend.db import get_connection
from backend.cache.embedder import embedder


# ---------------------------------------------------------------------------
# 1. Normalise + hash
# ---------------------------------------------------------------------------

def normalize_and_hash(query: str) -> str:
    """Lowercase, collapse whitespace, and return the SHA-256 hex digest."""
    normalised = re.sub(r"\s+", " ", query.lower()).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 2. Exact-hash lookup
# ---------------------------------------------------------------------------

def lookup_exact_hash(query_hash: str) -> Optional[Dict[str, Any]]:
    """Return the cached row (as dict) whose query_hash matches, or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, query_hash, query_text, embedding, metadata,
                       response_text, tier_used, created_at
                  FROM semantic_cache
                 WHERE query_hash = %s
                 LIMIT 1;
                """,
                (query_hash,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cur.description]
            result = dict(zip(cols, row))
            # JSONB comes back as dict already from psycopg2
            if isinstance(result.get("metadata"), str):
                result["metadata"] = json.loads(result["metadata"])
            return result
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. Semantic-candidate search
# ---------------------------------------------------------------------------

def search_semantic_candidates(
    query_vec: np.ndarray,
    min_similarity: float = 0.92,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Return up to *top_k* cache rows with cosine similarity >= *min_similarity*."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, query_hash, query_text, embedding, metadata,
                       response_text, tier_used, created_at,
                       1 - (embedding <=> %s) AS similarity
                  FROM semantic_cache
                 WHERE 1 - (embedding <=> %s) >= %s
                 ORDER BY similarity DESC
                 LIMIT %s;
                """,
                (query_vec, query_vec, min_similarity, top_k),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            results = []
            for row in rows:
                d = dict(zip(cols, row))
                if isinstance(d.get("metadata"), str):
                    d["metadata"] = json.loads(d["metadata"])
                results.append(d)
            return results
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 4. Main cache-check entry-point
# ---------------------------------------------------------------------------

def check_semantic_cache(
    query: str,
    query_metadata: Dict[str, Any],
    safe_reuse_validator: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], tuple[bool, str]]
    ] = None,
    min_similarity: float = 0.80,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Two-layer cache lookup with optional Safe-Reuse Guard.

    Returns a dict with at least:
        status, response_text, tier_used, similarity,
        latency_ms, reason, embedding
    """
    t0 = time.perf_counter()

    # --- Step 1: exact hash -----------------------------------------------
    qhash = normalize_and_hash(query)
    exact = lookup_exact_hash(qhash)
    if exact is not None:
        return {
            "status": "HIT_EXACT",
            "response_text": exact["response_text"],
            "tier_used": exact["tier_used"],
            "similarity": 1.0,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "reason": "Exact query hash matched.",
            "embedding": None,
            "route_to": "CACHE",
        }

    # --- Step 2: compute embedding ----------------------------------------
    query_vec = embedder.encode(query)

    # --- Step 3: semantic candidates --------------------------------------
    # Query candidates with cosine similarity >= 0.80 to apply routing threshold rules:
    # 1. Similarity > 0.90: Semantic cache hit (return cached response if safe)
    # 2. 0.80 <= Similarity <= 0.90: Route to router model
    # 3. Similarity < 0.80: Cache miss -> passes to feature extraction & router model
    candidates = search_semantic_candidates(query_vec, min_similarity=0.80, top_k=top_k)

    if not candidates:
        return {
            "status": "MISS",
            "response_text": None,
            "tier_used": None,
            "similarity": 0.0,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "reason": "Cosine similarity < 0.80. Cache miss, query should be processed and cached at the end.",
            "embedding": query_vec,
            "route_to": "ROUTER",
        }

    top_candidate = candidates[0]
    max_sim = float(top_candidate["similarity"])

    # If cosine similarity score is between 0.80 and 0.90 -> return to router model
    if max_sim <= 0.90:
        return {
            "status": "ROUTE_TO_ROUTER",
            "response_text": None,
            "tier_used": None,
            "similarity": max_sim,
            "matched_query": top_candidate["query_text"],
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "reason": f"Cosine similarity score ({max_sim:.4f}) is between 0.80 and 0.90. Returning to router model.",
            "embedding": query_vec,
            "route_to": "ROUTER",
        }

    # --- Step 4: Safe-Reuse Guard for candidates with similarity > 0.90 ---
    rejection_reasons: List[str] = []

    for cand in candidates:
        cand_sim = float(cand["similarity"])
        if cand_sim <= 0.90:
            continue

        cand_meta = cand["metadata"]

        if safe_reuse_validator is not None:
            safe, reason = safe_reuse_validator(query_metadata, cand_meta)
        else:
            safe = query_metadata == cand_meta
            reason = (
                "Metadata matches."
                if safe
                else "Metadata mismatch (default check)."
            )

        if safe:
            return {
                "status": "HIT_SEMANTIC",
                "response_text": cand["response_text"],
                "tier_used": cand["tier_used"],
                "similarity": cand_sim,
                "matched_query": cand["query_text"],
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "reason": reason,
                "embedding": query_vec,
                "route_to": "CACHE",
            }

        rejection_reasons.append(reason)

    # All candidates above 0.90 failed validation
    top = candidates[0]
    return {
        "status": "REJECTED_UNSAFE",
        "response_text": None,
        "tier_used": None,
        "similarity": float(top["similarity"]),
        "matched_query": top["query_text"],
        "latency_ms": (time.perf_counter() - t0) * 1000,
        "reason": " | ".join(rejection_reasons),
        "embedding": query_vec,
        "route_to": "ROUTER",
    }


# ---------------------------------------------------------------------------
# 5. Store in cache
# ---------------------------------------------------------------------------

def store_in_cache(
    query: str,
    query_vec: Optional[np.ndarray],
    metadata: Dict[str, Any],
    response_text: str,
    tier_used: str,
) -> None:
    """Insert a new entry (skip silently on hash collision)."""
    qhash = normalize_and_hash(query)
    if query_vec is None:
        query_vec = embedder.encode(query)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO semantic_cache
                    (query_hash, query_text, embedding, metadata,
                     response_text, tier_used)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (query_hash) DO NOTHING;
                """,
                (
                    qhash,
                    query,
                    query_vec,
                    json.dumps(metadata),
                    response_text,
                    tier_used,
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 6. Log telemetry
# ---------------------------------------------------------------------------

def log_telemetry(log_data: Dict[str, Any]) -> None:
    """Insert a complete telemetry record."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telemetry_logs
                    (query_text, cache_status, selected_tier,
                     p_small, p_medium, p_large,
                     input_tokens, expected_output_tokens, actual_output_tokens,
                     actual_cost_usd, baseline_large_cost_usd,
                     latency_ms, fallback_triggered)
                VALUES (%s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    log_data["query_text"],
                    log_data["cache_status"],
                    log_data["selected_tier"],
                    log_data.get("p_small"),
                    log_data.get("p_medium"),
                    log_data.get("p_large"),
                    log_data["input_tokens"],
                    log_data["expected_output_tokens"],
                    log_data["actual_output_tokens"],
                    log_data["actual_cost_usd"],
                    log_data["baseline_large_cost_usd"],
                    log_data["latency_ms"],
                    log_data.get("fallback_triggered", False),
                ),
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 7. Dashboard metrics
# ---------------------------------------------------------------------------

def get_dashboard_metrics(limit: int = 50) -> Dict[str, Any]:
    """Aggregate telemetry for the live dashboard."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # --- summary ---
            cur.execute("""
                SELECT
                    COUNT(*)                                        AS total_queries,
                    COALESCE(SUM(actual_cost_usd), 0)              AS total_actual_cost,
                    COALESCE(SUM(baseline_large_cost_usd), 0)      AS total_baseline_cost,
                    COALESCE(AVG(latency_ms), 0)                   AS avg_latency_ms,
                    COUNT(*) FILTER (WHERE cache_status IN
                        ('HIT_EXACT','HIT_SEMANTIC'))              AS cache_hits,
                    COUNT(*) FILTER (WHERE cache_status =
                        'REJECTED_UNSAFE')                         AS unsafe_blocks,
                    COUNT(*) FILTER (WHERE fallback_triggered)     AS fallbacks
                FROM telemetry_logs;
            """)
            row = cur.fetchone()
            total_queries = row[0]
            total_actual_cost = float(row[1])
            total_baseline_cost = float(row[2])
            avg_latency_ms = float(row[3])
            cache_hits = row[4]
            unsafe_blocks = row[5]
            fallbacks = row[6]

            total_saved_usd = total_baseline_cost - total_actual_cost
            savings_pct = (
                (total_saved_usd / total_baseline_cost * 100)
                if total_baseline_cost > 0
                else 0.0
            )
            cache_hit_rate_pct = (
                (cache_hits / total_queries * 100) if total_queries > 0 else 0.0
            )

            # tier distribution
            cur.execute("""
                SELECT selected_tier, COUNT(*)
                  FROM telemetry_logs
                 GROUP BY selected_tier;
            """)
            tier_distribution = {r[0]: r[1] for r in cur.fetchall()}

            # --- recent logs ---
            cur.execute(
                """
                SELECT id, timestamp, query_text, cache_status,
                       selected_tier, p_small, p_medium, p_large,
                       input_tokens, expected_output_tokens,
                       actual_output_tokens, actual_cost_usd,
                       baseline_large_cost_usd, latency_ms,
                       fallback_triggered
                  FROM telemetry_logs
                 ORDER BY timestamp DESC
                 LIMIT %s;
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            recent_logs = [dict(zip(cols, r)) for r in cur.fetchall()]

        return {
            "summary": {
                "total_queries": total_queries,
                "total_actual_cost": total_actual_cost,
                "total_baseline_cost": total_baseline_cost,
                "total_saved_usd": total_saved_usd,
                "savings_pct": savings_pct,
                "avg_latency_ms": avg_latency_ms,
                "cache_hits": cache_hits,
                "cache_hit_rate_pct": cache_hit_rate_pct,
                "unsafe_blocks": unsafe_blocks,
                "fallbacks": fallbacks,
                "tier_distribution": tier_distribution,
            },
            "recent_logs": recent_logs,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 8. Clear cache
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    """Truncate the semantic_cache table."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE semantic_cache RESTART IDENTITY;")
        conn.commit()
    finally:
        conn.close()
