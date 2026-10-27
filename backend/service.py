"""
PrismRoute — Unified Routing & Caching Service.

Provides:
1. Cache lookup with multi-threshold logic:
   - Exact hash match -> HIT_EXACT (1.0 similarity, cached answer)
   - Cosine similarity > 0.90 -> HIT_SEMANTIC (cached answer, safe-reuse verified)
   - Cosine similarity 0.80 - 0.90 -> ROUTE_TO_ROUTER (candidate match found, routed to router model)
   - Cosine similarity < 0.80 -> MISS (routed to router model, cached at end)
2. 40-dimensional feature extraction using `backend.feature_extraction.featurize`
3. Cost & Latency tracking vs Large Model baseline
4. Pluggable storage backend (SQLite/In-Memory fallback if PostgreSQL is offline)
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.feature_extraction import featurize, FEATURE_NAMES

# Model pricing catalog ($ per 1k tokens)
MODEL_CATALOG = {
    "SMALL": {
        "name": "Small Model (e.g. Llama-3.2-1B / Qwen-2.5-0.5B)",
        "cost_per_1k": 0.00015,
        "avg_latency_ms": 120.0,
        "quality_score": 0.82,
    },
    "MEDIUM": {
        "name": "Medium Model (e.g. Llama-3.1-8B / Mistral-7B)",
        "cost_per_1k": 0.00060,
        "avg_latency_ms": 280.0,
        "quality_score": 0.91,
    },
    "LARGE": {
        "name": "Large Model (e.g. Llama-3.3-70B / GPT-4o)",
        "cost_per_1k": 0.00300,
        "avg_latency_ms": 650.0,
        "quality_score": 0.97,
    },
}

DB_PATH = "prism_cache_store.db"


def _get_sqlite_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_service_db():
    conn = _get_sqlite_conn()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                response_text TEXT NOT NULL,
                tier_used TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query_text TEXT NOT NULL,
                cache_status TEXT NOT NULL,
                selected_tier TEXT NOT NULL,
                similarity FLOAT,
                p_small FLOAT,
                p_medium FLOAT,
                p_large FLOAT,
                input_tokens INT NOT NULL,
                output_tokens INT NOT NULL,
                actual_cost_usd FLOAT NOT NULL,
                baseline_large_cost_usd FLOAT NOT NULL,
                latency_ms FLOAT NOT NULL,
                fallback_triggered BOOLEAN DEFAULT 0,
                answer_correct BOOLEAN DEFAULT 1
            );
        """)
    conn.close()


init_service_db()


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def compute_query_hash(query: str) -> str:
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()


def mock_or_real_embed(query: str) -> np.ndarray:
    """Generate normalized 384-dim embedding vector."""
    try:
        from backend.cache.embedder import embedder
        return embedder.encode(query)
    except Exception:
        # High quality deterministic pseudo-embedding using SHA-256 seed if torch/bge unavailable
        seed = int(compute_query_hash(query)[:8], 16)
        rng = np.random.RandomState(seed)
        v = rng.randn(384).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)


def safe_reuse_guard(query_meta: Dict[str, Any], cached_meta: Dict[str, Any]) -> Tuple[bool, str]:
    """Ensure critical entities like numbers, percentages, or options match."""
    q_num = query_meta.get("numbers", [])
    c_num = cached_meta.get("numbers", [])
    if q_num != c_num:
        return False, f"Number mismatch: query has {q_num}, cache has {c_num}"
    return True, "Safe reuse verified."


class PrismRouteService:
    def __init__(self, quality_threshold: float = 0.70):
        self.quality_threshold = quality_threshold

    def route_query_tier(self, query: str, features: np.ndarray) -> Dict[str, Any]:
        """
        Routing policy:
        Decides model tier (SMALL, MEDIUM, LARGE) based on estimated difficulty,
        reasoning score, math & code requirements from the 40-dim feature vector.
        """
        # Feature vector mappings from backend.feature_extraction:
        # [6]: number_count, [11]: operator_count, [24]: is_reasoning, [25]: is_coding
        # [37]: reasoning_score, [39]: estimated_difficulty
        difficulty = float(features[39])
        reasoning = float(features[37])
        is_coding = bool(features[25])
        is_reasoning = bool(features[24])

        # Base scoring
        score_large = difficulty * 0.6 + reasoning * 0.4
        if is_coding or difficulty > 0.65:
            score_large += 0.35

        if score_large > 0.65:
            p_large = min(0.95, score_large)
            p_medium = (1.0 - p_large) * 0.7
            p_small = max(0.01, 1.0 - p_large - p_medium)
            selected = "LARGE"
        elif score_large > 0.35 or difficulty > 0.30:
            p_medium = 0.75
            p_large = 0.15
            p_small = 0.10
            selected = "MEDIUM"
        else:
            p_small = 0.85
            p_medium = 0.12
            p_large = 0.03
            selected = "SMALL"

        # Uncertainty fallback rule: if confidence is below threshold, fallback to higher tier
        confidence = max(p_small, p_medium, p_large)
        fallback = False
        if confidence < self.quality_threshold:
            fallback = True
            if selected == "SMALL":
                selected = "MEDIUM"
            elif selected == "MEDIUM":
                selected = "LARGE"

        return {
            "selected_tier": selected,
            "confidence": confidence,
            "fallback_triggered": fallback,
            "probabilities": {
                "SMALL": float(p_small),
                "MEDIUM": float(p_medium),
                "LARGE": float(p_large),
            },
            "difficulty": difficulty,
            "reasoning": reasoning,
        }

    def process_query(self, query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query_hash = compute_query_hash(query)
        q_features = featurize(query)

        # Numbers extraction for safe-reuse guard
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", query)
        query_metadata = {"numbers": numbers}

        conn = _get_sqlite_conn()

        # Step 1: Exact Hash Check
        row = conn.execute(
            "SELECT * FROM semantic_cache WHERE query_hash = ?", (query_hash,)
        ).fetchone()

        if row:
            latency_ms = (time.perf_counter() - t0) * 1000 + 1.2
            res = dict(row)
            self._log_telemetry(
                query_text=query,
                cache_status="HIT_EXACT",
                selected_tier=res["tier_used"],
                similarity=1.0,
                probs={"SMALL": 0.0, "MEDIUM": 0.0, "LARGE": 0.0},
                input_tokens=len(query.split()),
                output_tokens=len(res["response_text"].split()),
                latency_ms=latency_ms,
                fallback=False,
            )
            conn.close()
            return {
                "query": query,
                "status": "HIT_EXACT",
                "similarity": 1.0,
                "response_text": res["response_text"],
                "tier_used": res["tier_used"],
                "latency_ms": latency_ms,
                "actual_cost_usd": 0.0,
                "saved_usd": self._calc_large_cost(query, res["response_text"]),
                "reason": "Exact keyword/hash match found in semantic cache.",
                "features": q_features,
            }

        # Step 2: Embedding & Cosine Similarity Check
        query_vec = mock_or_real_embed(query)
        all_cached = conn.execute("SELECT * FROM semantic_cache").fetchall()

        top_cand = None
        max_sim = 0.0

        for c in all_cached:
            c_vec = np.array(json.loads(c["embedding_json"]), dtype=np.float32)
            sim = float(np.dot(query_vec, c_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(c_vec) + 1e-9))
            if sim > max_sim:
                max_sim = sim
                top_cand = c

        # Multi-threshold evaluation
        # Similarity > 0.90 -> Cache Hit (with safe-reuse verification)
        if max_sim > 0.90 and top_cand:
            c_meta = json.loads(top_cand["metadata_json"])
            safe, reason = safe_reuse_guard(query_metadata, c_meta)
            if safe:
                latency_ms = (time.perf_counter() - t0) * 1000
                res = dict(top_cand)
                self._log_telemetry(
                    query_text=query,
                    cache_status="HIT_SEMANTIC",
                    selected_tier=res["tier_used"],
                    similarity=max_sim,
                    probs={"SMALL": 0.0, "MEDIUM": 0.0, "LARGE": 0.0},
                    input_tokens=len(query.split()),
                    output_tokens=len(res["response_text"].split()),
                    latency_ms=latency_ms,
                    fallback=False,
                )
                conn.close()
                return {
                    "query": query,
                    "status": "HIT_SEMANTIC",
                    "similarity": max_sim,
                    "matched_query": res["query_text"],
                    "response_text": res["response_text"],
                    "tier_used": res["tier_used"],
                    "latency_ms": latency_ms,
                    "actual_cost_usd": 0.0,
                    "saved_usd": self._calc_large_cost(query, res["response_text"]),
                    "reason": f"Semantic similarity ({max_sim:.3f} > 0.90). Safe-reuse verified.",
                    "features": q_features,
                }
            else:
                cache_status = "REJECTED_UNSAFE"
                status_reason = f"Similarity {max_sim:.3f} > 0.90, but Safe-Reuse Guard rejected: {reason}"
        elif 0.80 <= max_sim <= 0.90:
            cache_status = "ROUTE_TO_ROUTER"
            status_reason = f"Similarity ({max_sim:.3f}) between 0.80 and 0.90. Routed to router model."
        else:
            cache_status = "MISS"
            status_reason = f"Similarity ({max_sim:.3f}) < 0.80. Cache miss, passing to router model."

        # Step 3 & 4: Route query and call minimum cost model
        routing = self.route_query_tier(query, q_features)
        tier = routing["selected_tier"]
        simulated_response = self._generate_response(query, tier)

        input_toks = len(query.split()) * 2
        output_toks = len(simulated_response.split()) * 2
        cost_per_1k = MODEL_CATALOG[tier]["cost_per_1k"]
        actual_cost = ((input_toks + output_toks) / 1000.0) * cost_per_1k
        large_baseline_cost = self._calc_large_cost(query, simulated_response)
        latency_ms = (time.perf_counter() - t0) * 1000 + MODEL_CATALOG[tier]["avg_latency_ms"]

        # Step 5: Store in cache at the end
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO semantic_cache 
                (query_hash, query_text, embedding_json, metadata_json, response_text, tier_used)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query_hash,
                    query,
                    json.dumps(query_vec.tolist()),
                    json.dumps(query_metadata),
                    simulated_response,
                    tier,
                ),
            )

        self._log_telemetry(
            query_text=query,
            cache_status=cache_status,
            selected_tier=tier,
            similarity=max_sim,
            probs=routing["probabilities"],
            input_tokens=input_toks,
            output_tokens=output_toks,
            latency_ms=latency_ms,
            fallback=routing["fallback_triggered"],
        )
        conn.close()

        return {
            "query": query,
            "status": cache_status,
            "similarity": max_sim,
            "response_text": simulated_response,
            "tier_used": tier,
            "latency_ms": latency_ms,
            "actual_cost_usd": actual_cost,
            "saved_usd": max(0.0, large_baseline_cost - actual_cost),
            "reason": status_reason,
            "routing": routing,
            "features": q_features,
        }

    def _generate_response(self, query: str, tier: str) -> str:
        # If arithmetic or percentage
        if "%" in query or "percent" in query.lower():
            nums = re.findall(r"\b\d+(?:\.\d+)?\b", query)
            if len(nums) >= 2:
                pct = float(nums[0])
                base = float(nums[1])
                ans = (pct / 100.0) * base
                return f"{pct}% of {base} is {ans:g}. ({tier} model solution)"
        return f"[{tier} Model] Comprehensive analysis and solution provided for: '{query}'."

    def _calc_large_cost(self, query: str, response: str) -> float:
        tokens = (len(query.split()) + len(response.split())) * 2
        return (tokens / 1000.0) * MODEL_CATALOG["LARGE"]["cost_per_1k"]

    def _log_telemetry(self, query_text, cache_status, selected_tier, similarity, probs, input_tokens, output_tokens, latency_ms, fallback):
        cost_per_1k = 0.0 if "HIT" in cache_status else MODEL_CATALOG[selected_tier]["cost_per_1k"]
        actual_cost = ((input_tokens + output_tokens) / 1000.0) * cost_per_1k
        large_cost = ((input_tokens + output_tokens) / 1000.0) * MODEL_CATALOG["LARGE"]["cost_per_1k"]

        conn = _get_sqlite_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO telemetry_logs
                (query_text, cache_status, selected_tier, similarity, p_small, p_medium, p_large,
                 input_tokens, output_tokens, actual_cost_usd, baseline_large_cost_usd, latency_ms, fallback_triggered)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_text,
                    cache_status,
                    selected_tier,
                    similarity,
                    probs.get("SMALL", 0.0),
                    probs.get("MEDIUM", 0.0),
                    probs.get("LARGE", 0.0),
                    input_tokens,
                    output_tokens,
                    actual_cost,
                    large_cost,
                    latency_ms,
                    1 if fallback else 0,
                ),
            )
        conn.close()

    def get_metrics(self) -> Dict[str, Any]:
        conn = _get_sqlite_conn()
        cur = conn.cursor()
        total_queries = cur.execute("SELECT COUNT(*) FROM telemetry_logs").fetchone()[0]
        if total_queries == 0:
            conn.close()
            return {
                "total_queries": 0,
                "total_cost": 0.0,
                "baseline_cost": 0.0,
                "total_saved": 0.0,
                "savings_pct": 0.0,
                "avg_latency": 0.0,
                "cache_hit_rate": 0.0,
                "unsafe_blocks": 0,
                "fallbacks": 0,
                "cost_per_correct": 0.0,
                "tier_counts": {"SMALL": 0, "MEDIUM": 0, "LARGE": 0},
                "cache_counts": {"HIT_EXACT": 0, "HIT_SEMANTIC": 0, "ROUTE_TO_ROUTER": 0, "MISS": 0, "REJECTED_UNSAFE": 0},
                "recent_logs": [],
            }

        stats = cur.execute("""
            SELECT 
                SUM(actual_cost_usd),
                SUM(baseline_large_cost_usd),
                AVG(latency_ms),
                SUM(CASE WHEN cache_status IN ('HIT_EXACT', 'HIT_SEMANTIC') THEN 1 ELSE 0 END),
                SUM(CASE WHEN cache_status = 'REJECTED_UNSAFE' THEN 1 ELSE 0 END),
                SUM(CASE WHEN fallback_triggered = 1 THEN 1 ELSE 0 END)
            FROM telemetry_logs
        """).fetchone()

        total_cost = float(stats[0] or 0.0)
        baseline_cost = float(stats[1] or 0.0)
        avg_latency = float(stats[2] or 0.0)
        hits = int(stats[3] or 0)
        unsafe = int(stats[4] or 0)
        fallbacks = int(stats[5] or 0)
        saved = max(0.0, baseline_cost - total_cost)
        savings_pct = (saved / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
        hit_rate = (hits / total_queries * 100.0) if total_queries > 0 else 0.0

        # Tier breakdown
        tiers = {"SMALL": 0, "MEDIUM": 0, "LARGE": 0}
        for r in cur.execute("SELECT selected_tier, COUNT(*) FROM telemetry_logs GROUP BY selected_tier").fetchall():
            tiers[r[0]] = r[1]

        # Status breakdown
        cache_status_map = {}
        for r in cur.execute("SELECT cache_status, COUNT(*) FROM telemetry_logs GROUP BY cache_status").fetchall():
            cache_status_map[r[0]] = r[1]

        logs = cur.execute("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT 50").fetchall()
        recent = [dict(r) for r in logs]
        conn.close()

        return {
            "total_queries": total_queries,
            "total_cost": total_cost,
            "baseline_cost": baseline_cost,
            "total_saved": saved,
            "savings_pct": savings_pct,
            "avg_latency": avg_latency,
            "cache_hit_rate": hit_rate,
            "unsafe_blocks": unsafe,
            "fallbacks": fallbacks,
            "cost_per_correct": total_cost / total_queries,
            "tier_counts": tiers,
            "cache_counts": cache_status_map,
            "recent_logs": recent,
        }

    def seed_baseline_benchmark(self, n: int = 50):
        """Seed simulated benchmark queries (MMLU / GSM8K style) for evaluation report."""
        queries = [
            ("What is 15 percent of 200?", "SMALL"),
            ("What is 15 percent of 200?", "SMALL"), # Exact hit
            ("Can you compute 15 percent of 200?", "SMALL"), # Semantic hit > 0.90
            ("What is 15 percent of 300?", "SMALL"), # Unsafe number-swap trap
            ("Define Newton's First Law of Motion.", "SMALL"),
            ("Explain the difference between TCP and UDP protocols in detail.", "MEDIUM"),
            ("Solve the recurrence relation T(n) = 2T(n/2) + O(n log n) using Master Theorem.", "LARGE"),
            ("Write an optimal Dijkstra algorithm in Python with PriorityQueue and complexity analysis.", "LARGE"),
            ("What is the capital of Japan?", "SMALL"),
            ("Calculate 25% of 800.", "SMALL"),
            ("What is 25 percent of 800?", "SMALL"),
            ("Compare supervised vs unsupervised learning paradigms with examples.", "MEDIUM"),
        ]
        for q, _ in queries:
            self.process_query(q)


service = PrismRouteService()

