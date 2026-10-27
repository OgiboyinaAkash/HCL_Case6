"""
PrismRoute — Unified Routing & Caching Service.
Architecture Implementation matching the System Specification:

Workflow:
1. User Query: (text, scope_id)
2. Normalize and hash: SHA-256 of clean text
3. Exact match: Hash lookup in memory (O(1))
   -> hit  -> Return cached (Exact hit)
   -> miss -> Embed and search: Max cosine over cache
4. Cosine similarity branches:
   - Reuse zone (Score >= 0.90):
     -> Safety gates check (Safe-Reuse Guard)
     -> pass -> Return cached (Safety gates pass)
     -> fail -> Falls through as unsafe to Feature Layer
   - Hint zone (Score 0.80 to 0.90):
     -> Reuse its tier (Skip feature layer & neural network)
     -> Direct to Call chosen model
   - Miss (Score < 0.80) or failed safety gates:
     -> Feature layer: extracts 40-value vector
     -> Neural network: 3-class softmax probabilities
     -> Threshold rule: Pick cheapest tier (with confidence thresholding & uncertainty fallback)
     -> Direct to Call chosen model
5. Call chosen model:
   -> Retry, escalate on error
   -> Response uncertainty validation (verbalized uncertainty / arithmetic check)
6. Write to cache and log:
   -> Hash, vector, answer, tier (persisted to DB + in-memory index + telemetry)
7. Return response:
   -> (Answer, cost, decision record)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.feature_extraction import featurize, FEATURE_NAMES
from backend.routing_policy import RoutingPolicy, DEFAULT_PRICE_SHEET
from backend.cache.embedder import embedder

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
                scope_id TEXT NOT NULL DEFAULT 'default',
                query_hash TEXT NOT NULL,
                query_text TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                response_text TEXT NOT NULL,
                tier_used TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(scope_id, query_hash)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scope_id TEXT NOT NULL DEFAULT 'default',
                query_text TEXT NOT NULL,
                cache_status TEXT NOT NULL,
                path_taken TEXT NOT NULL,
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
                escalation_triggered BOOLEAN DEFAULT 0,
                retries INT DEFAULT 0
            );
        """)

        # Auto-migrate columns if tables already existed from earlier runs
        cache_cols = [c[1] for c in conn.execute("PRAGMA table_info(semantic_cache)").fetchall()]
        if "scope_id" not in cache_cols:
            conn.execute("ALTER TABLE semantic_cache ADD COLUMN scope_id TEXT NOT NULL DEFAULT 'default'")

        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_scope_hash ON semantic_cache(scope_id, query_hash)")

        log_cols = [c[1] for c in conn.execute("PRAGMA table_info(telemetry_logs)").fetchall()]
        if "scope_id" not in log_cols:
            conn.execute("ALTER TABLE telemetry_logs ADD COLUMN scope_id TEXT NOT NULL DEFAULT 'default'")
        if "path_taken" not in log_cols:
            conn.execute("ALTER TABLE telemetry_logs ADD COLUMN path_taken TEXT NOT NULL DEFAULT 'ROUTED'")
        if "escalation_triggered" not in log_cols:
            conn.execute("ALTER TABLE telemetry_logs ADD COLUMN escalation_triggered BOOLEAN DEFAULT 0")
        if "retries" not in log_cols:
            conn.execute("ALTER TABLE telemetry_logs ADD COLUMN retries INT DEFAULT 0")

    conn.close()


init_service_db()


def normalize_query(query: str) -> str:
    """Normalizes whitespace and lowercases query."""
    return re.sub(r"\s+", " ", query.lower()).strip()


def compute_query_hash(query: str) -> str:
    """Computes deterministic SHA-256 hash of normalized text."""
    return hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()


def extract_query_metadata(query: str) -> Dict[str, Any]:
    """Extracts entities, numbers, percentages, and options for Safe-Reuse Guard."""
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", query)
    percentages = re.findall(r"\b\d+(?:\.\d+)?\s*%", query)
    options = re.findall(r"\b[A-D]\)", query)
    return {
        "numbers": numbers,
        "percentages": percentages,
        "options": options,
    }


def safe_reuse_guard(query_meta: Dict[str, Any], cached_meta: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Safe-Reuse Guard: Ensures critical numerical values, percentages,
    and multiple-choice options strictly match before reusing a cached answer.
    Prevents traps like '15% of 200' vs '15% of 300'.
    """
    q_num = query_meta.get("numbers", [])
    c_num = cached_meta.get("numbers", [])
    if q_num != c_num:
        return False, f"Number mismatch trap: query has {q_num}, cache has {c_num}"

    q_pct = query_meta.get("percentages", [])
    c_pct = cached_meta.get("percentages", [])
    if q_pct != c_pct:
        return False, f"Percentage mismatch: query has {q_pct}, cache has {c_pct}"

    q_opt = query_meta.get("options", [])
    c_opt = cached_meta.get("options", [])
    if q_opt and c_opt and q_opt != c_opt:
        return False, f"Option set mismatch: query has {q_opt}, cache has {c_opt}"

    return True, "Safe reuse verified."


class PrismRouteService:
    """
    Full PrismRoute workflow engine:
    Implements Exact Match -> Semantic Cosine (Reuse/Hint/Miss) -> Feature Layer -> Neural Network -> Model Call with Retry/Escalate -> Cache & Log.
    """

    def __init__(self, quality_threshold: float = 0.65, model_type: str = "neural_net"):
        self.quality_threshold = quality_threshold
        self.model_type = model_type.lower()
        self.policy = RoutingPolicy(model_type=self.model_type, confidence_threshold=quality_threshold)
        self.embedder = embedder

        # In-Memory Fast Lookup Hash Index: (scope_id, query_hash) -> cached_dict
        self._memory_hash_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._load_memory_cache()

    def _load_memory_cache(self):
        """Populate the O(1) in-memory hash index from persistent storage."""
        conn = _get_sqlite_conn()
        rows = conn.execute("SELECT * FROM semantic_cache").fetchall()
        for r in rows:
            entry = dict(r)
            scope = entry.get("scope_id", "default")
            q_hash = entry["query_hash"]
            self._memory_hash_cache[(scope, q_hash)] = entry
        conn.close()

    def process_query(self, query: str, scope_id: str = "default") -> Dict[str, Any]:
        """
        Executes the end-to-end workflow according to the architecture specification.

        Workflow:
        1. User query (text, scope_id)
        2. Normalize & hash (SHA-256)
        3. Exact match (Hash lookup in memory) -> hit -> Return cached
        4. Embed and search (Max cosine over cache):
           - Reuse zone (Score >= 0.90) -> Safety gates pass -> Return cached
           - Hint zone (Score 0.80 to 0.90) -> Reuse its tier (Skip feature layer)
           - Miss (Score < 0.80) -> Feature layer (40-val) -> Neural net (softmax) -> Threshold rule
        5. Call chosen model (Retry, escalate on error)
        6. Write to cache and log
        7. Return response (Answer, cost, decision record)
        """
        t0 = time.perf_counter()
        clean_text = normalize_query(query)
        query_hash = compute_query_hash(clean_text)
        query_meta = extract_query_metadata(query)

        # -------------------------------------------------------------
        # Step 3: Exact Match (Hash lookup in memory)
        # -------------------------------------------------------------
        cache_key = (scope_id, query_hash)
        if cache_key in self._memory_hash_cache:
            hit_data = self._memory_hash_cache[cache_key]
            latency_ms = (time.perf_counter() - t0) * 1000 + 0.5
            large_cost = self._calc_large_cost(query, hit_data["response_text"])

            self._log_telemetry(
                scope_id=scope_id,
                query_text=query,
                cache_status="HIT_EXACT",
                path_taken="EXACT_HASH_HIT",
                selected_tier=hit_data["tier_used"],
                similarity=1.0,
                probs={"SMALL": 0.0, "MEDIUM": 0.0, "LARGE": 0.0},
                input_tokens=len(query.split()),
                output_tokens=len(hit_data["response_text"].split()),
                latency_ms=latency_ms,
                fallback=False,
                escalation=False,
                retries=0,
            )

            return {
                "answer": hit_data["response_text"],
                "cost": {
                    "actual_cost_usd": 0.0,
                    "baseline_large_cost_usd": large_cost,
                    "saved_usd": large_cost,
                    "savings_pct": 100.0,
                },
                "decision_record": {
                    "query": query,
                    "scope_id": scope_id,
                    "status": "HIT_EXACT",
                    "path_taken": "EXACT_HASH_HIT",
                    "similarity": 1.0,
                    "matched_query": hit_data["query_text"],
                    "selected_tier": hit_data["tier_used"],
                    "probabilities": {"SMALL": 0.0, "MEDIUM": 0.0, "LARGE": 0.0},
                    "skipped_feature_layer": True,
                    "fallback_triggered": False,
                    "escalation_triggered": False,
                    "retries": 0,
                    "latency_ms": latency_ms,
                    "reason": "Exact hash match in memory.",
                },
            }

        # -------------------------------------------------------------
        # Step 4: Embed & Search (Max cosine over cache for scope_id)
        # -------------------------------------------------------------
        query_vec = self.embedder.encode(clean_text)

        conn = _get_sqlite_conn()
        scoped_cached = conn.execute(
            "SELECT * FROM semantic_cache WHERE scope_id = ?", (scope_id,)
        ).fetchall()

        top_cand = None
        max_sim = 0.0

        for c in scoped_cached:
            c_vec = np.array(json.loads(c["embedding_json"]), dtype=np.float32)
            sim = float(np.dot(query_vec, c_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(c_vec) + 1e-9))
            if sim > max_sim:
                max_sim = sim
                top_cand = c

        # Branching based on cosine similarity
        chosen_tier: Optional[str] = None
        routing_info: Dict[str, Any] = {}
        cache_status = "MISS"
        path_taken = "NEURAL_NET_ROUTING"
        skipped_feature_layer = False
        status_reason = ""

        # --- Reuse zone: Score >= 0.90 ---
        if max_sim >= 0.90 and top_cand:
            c_meta = json.loads(top_cand["metadata_json"])
            safe, guard_reason = safe_reuse_guard(query_meta, c_meta)

            if safe:
                # Safety gates pass -> Return cached
                res_data = dict(top_cand)
                latency_ms = (time.perf_counter() - t0) * 1000
                large_cost = self._calc_large_cost(query, res_data["response_text"])

                self._log_telemetry(
                    scope_id=scope_id,
                    query_text=query,
                    cache_status="HIT_SEMANTIC",
                    path_taken="SEMANTIC_CACHE_REUSE",
                    selected_tier=res_data["tier_used"],
                    similarity=max_sim,
                    probs={"SMALL": 0.0, "MEDIUM": 0.0, "LARGE": 0.0},
                    input_tokens=len(query.split()),
                    output_tokens=len(res_data["response_text"].split()),
                    latency_ms=latency_ms,
                    fallback=False,
                    escalation=False,
                    retries=0,
                )
                conn.close()

                return {
                    "answer": res_data["response_text"],
                    "cost": {
                        "actual_cost_usd": 0.0,
                        "baseline_large_cost_usd": large_cost,
                        "saved_usd": large_cost,
                        "savings_pct": 100.0,
                    },
                    "decision_record": {
                        "query": query,
                        "scope_id": scope_id,
                        "status": "HIT_SEMANTIC",
                        "path_taken": "SEMANTIC_CACHE_REUSE",
                        "similarity": max_sim,
                        "matched_query": res_data["query_text"],
                        "selected_tier": res_data["tier_used"],
                        "probabilities": {"SMALL": 0.0, "MEDIUM": 0.0, "LARGE": 0.0},
                        "skipped_feature_layer": True,
                        "fallback_triggered": False,
                        "escalation_triggered": False,
                        "retries": 0,
                        "latency_ms": latency_ms,
                        "reason": f"Semantic match ({max_sim:.3f} >= 0.90) and Safety Gates passed: {guard_reason}",
                    },
                }
            else:
                # Safety gates rejected (e.g. number mismatch trap)
                cache_status = "REJECTED_UNSAFE"
                status_reason = f"Cosine similarity {max_sim:.3f} >= 0.90, but Safety Gates blocked reuse: {guard_reason}"

        # --- Hint zone: Score 0.80 to 0.90 -> Reuse its tier (Skip feature layer) ---
        if 0.80 <= max_sim < 0.90 and top_cand and cache_status != "REJECTED_UNSAFE":
            chosen_tier = top_cand["tier_used"]
            skipped_feature_layer = True
            path_taken = "HINT_ZONE_TIER_REUSE"
            cache_status = "HINT_ZONE"
            status_reason = (
                f"Hint zone ({max_sim:.3f} in [0.80, 0.90)). "
                f"Reusing tier '{chosen_tier}' from candidate query '{top_cand['query_text']}' and skipping feature layer."
            )
            routing_info = {
                "predicted_tier": chosen_tier,
                "selected_tier": chosen_tier,
                "confidence": max_sim,
                "probabilities": {chosen_tier: max_sim},
                "fallback_triggered": False,
                "routing_reason": status_reason,
            }

        # --- Miss: Score < 0.80 (or failed safety gates) -> Feature Layer + Neural Net + Threshold Rule ---
        if chosen_tier is None:
            if cache_status != "REJECTED_UNSAFE":
                cache_status = "MISS"
                status_reason = f"Cosine similarity ({max_sim:.3f}) < 0.80. Passing to Feature Layer and Neural Network."

            # Feature layer: 40-value vector
            q_features = featurize(query)

            # Neural network: 3-class softmax & threshold rule
            routing_info = self.policy.route(q_features, override_threshold=self.quality_threshold)
            chosen_tier = routing_info["selected_tier"]
            path_taken = "NEURAL_NET_ROUTING"

        # -------------------------------------------------------------
        # Step 5: Call Chosen Model (Retry, escalate on error)
        # -------------------------------------------------------------
        model_result = self._call_chosen_model(query=query, tier=chosen_tier, max_retries=2)
        final_tier = model_result["final_tier"]
        response_text = model_result["response_text"]
        retries_used = model_result["retries_used"]
        escalation_triggered = model_result["escalation_triggered"]
        fallback_triggered = routing_info.get("fallback_triggered", False) or model_result["uncertainty_fallback"]

        # Cost & Latency Math
        input_tokens = len(query.split()) * 2
        output_tokens = len(response_text.split()) * 2
        cost_per_1k = MODEL_CATALOG[final_tier]["cost_per_1k"]
        actual_cost = ((input_tokens + output_tokens) / 1000.0) * cost_per_1k
        large_baseline_cost = self._calc_large_cost(query, response_text)
        saved_usd = max(0.0, large_baseline_cost - actual_cost)
        savings_pct = (saved_usd / large_baseline_cost * 100.0) if large_baseline_cost > 0 else 0.0
        latency_ms = (time.perf_counter() - t0) * 1000 + MODEL_CATALOG[final_tier]["avg_latency_ms"]

        # -------------------------------------------------------------
        # Step 6: Write to Cache and Log (Hash, vector, answer, tier)
        # -------------------------------------------------------------
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_cache 
                (scope_id, query_hash, query_text, embedding_json, metadata_json, response_text, tier_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_id,
                    query_hash,
                    query,
                    json.dumps(query_vec.tolist()),
                    json.dumps(query_meta),
                    response_text,
                    final_tier,
                ),
            )

        # Update in-memory hash cache
        self._memory_hash_cache[(scope_id, query_hash)] = {
            "scope_id": scope_id,
            "query_hash": query_hash,
            "query_text": query,
            "embedding_json": json.dumps(query_vec.tolist()),
            "metadata_json": json.dumps(query_meta),
            "response_text": response_text,
            "tier_used": final_tier,
        }

        # Log telemetry
        self._log_telemetry(
            scope_id=scope_id,
            query_text=query,
            cache_status=cache_status,
            path_taken=path_taken,
            selected_tier=final_tier,
            similarity=max_sim,
            probs=routing_info.get("probabilities", {final_tier: 1.0}),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            fallback=fallback_triggered,
            escalation=escalation_triggered,
            retries=retries_used,
        )
        conn.close()

        # -------------------------------------------------------------
        # Step 7: Return Response (Answer, cost, decision record)
        # -------------------------------------------------------------
        return {
            "answer": response_text,
            "cost": {
                "actual_cost_usd": actual_cost,
                "baseline_large_cost_usd": large_baseline_cost,
                "saved_usd": saved_usd,
                "savings_pct": savings_pct,
            },
            "decision_record": {
                "query": query,
                "scope_id": scope_id,
                "status": cache_status,
                "path_taken": path_taken,
                "similarity": max_sim,
                "selected_tier": final_tier,
                "probabilities": routing_info.get("probabilities", {}),
                "skipped_feature_layer": skipped_feature_layer,
                "fallback_triggered": fallback_triggered,
                "escalation_triggered": escalation_triggered,
                "retries": retries_used,
                "latency_ms": latency_ms,
                "reason": status_reason or routing_info.get("routing_reason", ""),
            },
        }

    # =================================================================
    # Step 5 Execution Helper: Call Chosen Model with Retry & Escalate
    # =================================================================
    def _call_chosen_model(self, query: str, tier: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Executes chosen model with retry on error and tier escalation:
        - If an error occurs, retries up to max_retries with backoff.
        - If the tier continues to fail, escalates tier (e.g. SMALL -> MEDIUM -> LARGE).
        - Evaluates response for uncertainty (verbalized uncertainty or math omission).
        """
        current_tier = tier
        retries_used = 0
        escalation_triggered = False
        uncertainty_fallback = False
        response_text = ""

        tier_order = ["SMALL", "MEDIUM", "LARGE"]

        for attempt in range(max_retries + 1):
            try:
                # Simulate generation
                response_text = self._generate_response(query, current_tier)
                break
            except Exception as e:
                retries_used += 1
                if attempt < max_retries:
                    time.sleep(0.05 * (2 ** attempt))
                else:
                    # Escalate to next higher tier on persistent error
                    idx = tier_order.index(current_tier)
                    if idx < len(tier_order) - 1:
                        current_tier = tier_order[idx + 1]
                        escalation_triggered = True
                        response_text = self._generate_response(query, current_tier)

        # Check response uncertainty (verbalized uncertainty markers)
        is_uncertain, reason = self.policy.evaluate_response_uncertainty(response_text, query)
        if is_uncertain and current_tier != "LARGE":
            current_tier = "LARGE"
            uncertainty_fallback = True
            escalation_triggered = True
            response_text = self._generate_response(query, "LARGE")

        return {
            "final_tier": current_tier,
            "response_text": response_text,
            "retries_used": retries_used,
            "escalation_triggered": escalation_triggered,
            "uncertainty_fallback": uncertainty_fallback,
        }

    def _generate_response(self, query: str, tier: str) -> str:
        """Generates domain-aware response tailored to tier capabilities."""
        # Arithmetic / percentage problems
        if "%" in query or "percent" in query.lower():
            nums = re.findall(r"\b\d+(?:\.\d+)?\b", query)
            if len(nums) >= 2:
                pct = float(nums[0])
                base = float(nums[1])
                ans = (pct / 100.0) * base
                return f"{pct}% of {base} is {ans:g}. ({tier} model verified solution)"

        # Simple math
        if re.search(r"\b\d+\s*[\*\+\-\/]\s*\d+\b", query):
            expr = re.search(r"(\d+)\s*([\*\+\-\/])\s*(\d+)", query)
            if expr:
                a, op, b = int(expr.group(1)), expr.group(2), int(expr.group(3))
                val = a * b if op == '*' else (a + b if op == '+' else (a - b if op == '-' else a / b))
                return f"{a} {op} {b} = {val}. ({tier} model computation)"

        return f"[{tier} Model] Comprehensive analysis and solution provided for: '{query}'."

    def _calc_large_cost(self, query: str, response: str) -> float:
        """Calculates reference baseline cost if query had been sent to Large model."""
        tokens = (len(query.split()) + len(response.split())) * 2
        return (tokens / 1000.0) * MODEL_CATALOG["LARGE"]["cost_per_1k"]

    def _log_telemetry(
        self,
        scope_id: str,
        query_text: str,
        cache_status: str,
        path_taken: str,
        selected_tier: str,
        similarity: float,
        probs: Dict[str, float],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        fallback: bool,
        escalation: bool,
        retries: int,
    ):
        """Logs structured telemetry entry to persistent storage."""
        cost_per_1k = 0.0 if "HIT" in cache_status else MODEL_CATALOG[selected_tier]["cost_per_1k"]
        actual_cost = ((input_tokens + output_tokens) / 1000.0) * cost_per_1k
        large_cost = ((input_tokens + output_tokens) / 1000.0) * MODEL_CATALOG["LARGE"]["cost_per_1k"]

        conn = _get_sqlite_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO telemetry_logs
                (scope_id, query_text, cache_status, path_taken, selected_tier, similarity, p_small, p_medium, p_large,
                 input_tokens, output_tokens, actual_cost_usd, baseline_large_cost_usd, latency_ms, 
                 fallback_triggered, escalation_triggered, retries)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope_id,
                    query_text,
                    cache_status,
                    path_taken,
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
                    1 if escalation else 0,
                    retries,
                ),
            )
        conn.close()

    def clear_cache(self, scope_id: Optional[str] = None):
        """Clears cache entries and flushes the in-memory index."""
        conn = _get_sqlite_conn()
        with conn:
            if scope_id:
                conn.execute("DELETE FROM semantic_cache WHERE scope_id = ?", (scope_id,))
                self._memory_hash_cache = {k: v for k, v in self._memory_hash_cache.items() if k[0] != scope_id}
            else:
                conn.execute("DELETE FROM semantic_cache")
                self._memory_hash_cache.clear()
        conn.close()

    def get_metrics(self, scope_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculates comprehensive operational metrics and cost savings."""
        conn = _get_sqlite_conn()
        cur = conn.cursor()

        where_clause = "WHERE scope_id = ?" if scope_id else ""
        params = (scope_id,) if scope_id else ()

        total_queries = cur.execute(f"SELECT COUNT(*) FROM telemetry_logs {where_clause}", params).fetchone()[0]
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
                "fallbacks": 0,
                "escalations": 0,
                "tier_counts": {"SMALL": 0, "MEDIUM": 0, "LARGE": 0},
                "cache_counts": {"HIT_EXACT": 0, "HIT_SEMANTIC": 0, "HINT_ZONE": 0, "MISS": 0, "REJECTED_UNSAFE": 0},
            }

        stats = cur.execute(f"""
            SELECT 
                SUM(actual_cost_usd),
                SUM(baseline_large_cost_usd),
                AVG(latency_ms),
                SUM(CASE WHEN cache_status IN ('HIT_EXACT', 'HIT_SEMANTIC') THEN 1 ELSE 0 END),
                SUM(CASE WHEN fallback_triggered = 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN escalation_triggered = 1 THEN 1 ELSE 0 END)
            FROM telemetry_logs {where_clause}
        """, params).fetchone()

        total_cost = float(stats[0] or 0.0)
        baseline_cost = float(stats[1] or 0.0)
        avg_latency = float(stats[2] or 0.0)
        hits = int(stats[3] or 0)
        fallbacks = int(stats[4] or 0)
        escalations = int(stats[5] or 0)
        saved = max(0.0, baseline_cost - total_cost)
        savings_pct = (saved / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
        hit_rate = (hits / total_queries * 100.0) if total_queries > 0 else 0.0

        tiers = {"SMALL": 0, "MEDIUM": 0, "LARGE": 0}
        for r in cur.execute(f"SELECT selected_tier, COUNT(*) FROM telemetry_logs {where_clause} GROUP BY selected_tier", params).fetchall():
            tiers[r[0]] = r[1]

        cache_counts = {}
        for r in cur.execute(f"SELECT cache_status, COUNT(*) FROM telemetry_logs {where_clause} GROUP BY cache_status", params).fetchall():
            cache_counts[r[0]] = r[1]

        conn.close()
        return {
            "total_queries": total_queries,
            "total_cost": total_cost,
            "baseline_cost": baseline_cost,
            "total_saved": saved,
            "savings_pct": savings_pct,
            "avg_latency": avg_latency,
            "cache_hit_rate": hit_rate,
            "fallbacks": fallbacks,
            "escalations": escalations,
            "tier_counts": tiers,
            "cache_counts": cache_counts,
        }


# Singleton service instance
service = PrismRouteService(model_type="neural_net")


# =====================================================================
# Service Entrypoint & Interactive Query Shell
# =====================================================================
def print_query_result(res: Dict[str, Any]):
    dec = res["decision_record"]
    cost = res["cost"]
    print("\n" + "=" * 72)
    print(f"QUERY: '{dec['query']}' [Scope: {dec['scope_id']}]")
    print("-" * 72)
    print(f"  -> Path Taken:            {dec['path_taken']}")
    print(f"  -> Cache Status:          {dec['status']}")
    print(f"  -> Selected Model Tier:   {dec['selected_tier']}")
    print(f"  -> Skipped Feature Layer: {dec['skipped_feature_layer']}")
    print(f"  -> Cosine Similarity:     {dec['similarity']:.4f}")
    print(f"  -> Latency:               {dec['latency_ms']:.1f} ms")
    print(f"  -> Actual Cost:           ${cost['actual_cost_usd']:.6f} (Savings: {cost['savings_pct']:.1f}%)")
    print(f"  -> Answer:                {res['answer']}")
    if dec.get("reason"):
        print(f"  -> Reason:                {dec['reason']}")
    print("=" * 72)


def print_metrics_dashboard(svc: PrismRouteService):
    m = svc.get_metrics()
    print("\n" + "-" * 72)
    print("LIVE PRISMROUTE METRICS & SAVINGS DASHBOARD")
    print("-" * 72)
    print(f"  Total Queries Processed:  {m['total_queries']}")
    print(f"  Actual Cost Incurred:     ${m['total_cost']:.6f}")
    print(f"  Baseline Cost (Large):    ${m['baseline_cost']:.6f}")
    print(f"  Total Dollars Saved:      ${m['total_saved']:.6f} ({m['savings_pct']:.1f}% savings)")
    print(f"  Average Latency:          {m['avg_latency']:.1f} ms")
    print(f"  Cache Hit Rate:           {m['cache_hit_rate']:.1f}%")
    print(f"  Fallbacks Triggered:      {m['fallbacks']}")
    print(f"  Escalations on Error:     {m['escalations']}")
    print(f"  Tier Distribution:        {m['tier_counts']}")
    print(f"  Cache Status Breakdown:   {m['cache_counts']}")
    print("-" * 72 + "\n")


if __name__ == "__main__":
    import sys
    # Ensure project root is in sys.path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    svc = PrismRouteService(model_type="neural_net")

    # If query was passed via command line arguments
    if len(sys.argv) > 1:
        query_arg = " ".join(sys.argv[1:])
        result = svc.process_query(query_arg)
        print_query_result(result)
        sys.exit(0)

    # Interactive Service Prompt
    print("\n" + "=" * 75)
    print("  PRISMROUTE — COST-AWARE MODEL ROUTER & SEMANTIC CACHE SERVICE")
    print("=" * 75)
    print("[*] Service Initialized with PyTorch Neural Network & Semantic Cache.")
    print("[*] Available commands:")
    print("      - Type any query to route, cache, and execute")
    print("      - Type 'metrics' to view cost savings & operational telemetry")
    print("      - Type 'clear'   to reset the semantic cache")
    print("      - Type 'demo'    to run representative test suite queries")
    print("      - Type 'exit'    or 'quit' to stop the service")
    print("=" * 75)

    while True:
        try:
            user_query = input("\nEnter query > ").strip()
            if not user_query:
                continue

            cmd = user_query.lower()
            if cmd in ("exit", "quit", "q"):
                print("[*] Shutting down PrismRoute service. Goodbye!\n")
                break
            elif cmd == "metrics":
                print_metrics_dashboard(svc)
            elif cmd == "clear":
                svc.clear_cache()
                print("[+] Cache cleared successfully.")
            elif cmd == "demo":
                demo_queries = [
                    "What is 15 percent of 200?",
                    "What is 15 percent of 200?",
                    "Can you calculate 15 percent of 200?",
                    "What is 15 percent of 300?",
                    "What percentage is 15 out of 200 total items?",
                    "Solve the recurrence relation T(n) = 2T(n/2) + n log n with complexity proof.",
                ]
                print("[*] Running demo queries...")
                for dq in demo_queries:
                    res = svc.process_query(dq)
                    print_query_result(res)
            else:
                res = svc.process_query(user_query)
                print_query_result(res)

        except (KeyboardInterrupt, EOFError):
            print("\n[*] Exiting PrismRoute service.")
            break
