"""
PrismRoute Module 1 — End-to-end verification script.

Exercises:
  1. DB init + cache clear
  2. Exact-hash hit
  3. Semantic hit (safe paraphrase)
  4. Rejected-unsafe (number-swap trap)
  5. Telemetry logging + dashboard math
"""

import sys
import json

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── 0. Setup ──────────────────────────────────────────────────────────────
from backend.db import init_db
from backend.cache.cache_store import (
    check_semantic_cache,
    store_in_cache,
    clear_cache,
    log_telemetry,
    get_dashboard_metrics,
)

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {label}")
    else:
        FAIL += 1
        print(f"  ❌ {label}  —  {detail}")


# ── 1. Initialise DB and clear cache ─────────────────────────────────────
print("\n🔧 Initialising database …")
init_db()
clear_cache()
print("   Done.\n")

# ── 2. Store a seed entry ────────────────────────────────────────────────
print("📝 Storing seed query …")
store_in_cache(
    query="What is 15 percent of 200?",
    query_vec=None,            # will be computed automatically
    metadata={"numbers": ["15", "200"]},
    response_text="15% of 200 is 30.",
    tier_used="SMALL",
)
print("   Done.\n")

# ── 3. Exact-hash hit ───────────────────────────────────────────────────
print("🔍 Test 1 — Exact hash hit")
r1 = check_semantic_cache(
    query="What is 15 percent of 200?",
    query_metadata={"numbers": ["15", "200"]},
)
check("status == HIT_EXACT", r1["status"] == "HIT_EXACT", f'got {r1["status"]}')
check("similarity == 1.0",   r1["similarity"] == 1.0,     f'got {r1["similarity"]}')
check("response_text correct", r1["response_text"] == "15% of 200 is 30.",
      f'got {r1["response_text"]}')
check("tier_used == SMALL",  r1["tier_used"] == "SMALL",  f'got {r1["tier_used"]}')
print()

# ── 4. Semantic hit (safe paraphrase) ────────────────────────────────────
print("🔍 Test 2 — Semantic hit (safe paraphrase, same numbers)")
r2 = check_semantic_cache(
    query="Can you calculate 15 percent of 200 for me?",
    query_metadata={"numbers": ["15", "200"]},
)
check("status == HIT_SEMANTIC", r2["status"] == "HIT_SEMANTIC", f'got {r2["status"]}')
check("similarity >= 0.92",    r2["similarity"] >= 0.92,        f'got {r2["similarity"]:.4f}')
check("response_text correct", r2["response_text"] == "15% of 200 is 30.",
      f'got {r2["response_text"]}')
print()

# ── 5. Rejected-unsafe (number-swap trap) ────────────────────────────────
print("🔍 Test 3 — Rejected unsafe (number-swap: 200→300)")
r3 = check_semantic_cache(
    query="Can you calculate 15 percent of 300 for me?",
    query_metadata={"numbers": ["15", "300"]},
    min_similarity=0.80,
)
check("status == REJECTED_UNSAFE", r3["status"] == "REJECTED_UNSAFE",
      f'got {r3["status"]}')
check("response_text is None",    r3["response_text"] is None,
      f'got {r3["response_text"]}')
print()

# ── 6. Telemetry + dashboard math ────────────────────────────────────────
print("📊 Test 4 — Telemetry logging & dashboard metrics")

# First truncate telemetry so the math is deterministic
from backend.db import get_connection
conn = get_connection()
with conn.cursor() as cur:
    cur.execute("TRUNCATE TABLE telemetry_logs RESTART IDENTITY;")
conn.commit()
conn.close()

log_telemetry({
    "query_text":              "What is 15 percent of 200?",
    "cache_status":            "HIT_EXACT",
    "selected_tier":           "SMALL",
    "p_small":                 0.85,
    "p_medium":                0.10,
    "p_large":                 0.05,
    "input_tokens":            12,
    "expected_output_tokens":  20,
    "actual_output_tokens":    18,
    "actual_cost_usd":         0.0001,
    "baseline_large_cost_usd": 0.0050,
    "latency_ms":              42.5,
    "fallback_triggered":      False,
})

log_telemetry({
    "query_text":              "Explain quantum computing.",
    "cache_status":            "MISS",
    "selected_tier":           "LARGE",
    "p_small":                 0.10,
    "p_medium":                0.20,
    "p_large":                 0.70,
    "input_tokens":            8,
    "expected_output_tokens":  200,
    "actual_output_tokens":    190,
    "actual_cost_usd":         0.0100,
    "baseline_large_cost_usd": 0.0100,
    "latency_ms":              350.0,
    "fallback_triggered":      False,
})

m = get_dashboard_metrics()
s = m["summary"]

check("total_queries == 2",      s["total_queries"] == 2,
      f'got {s["total_queries"]}')
check("total_actual_cost ≈ 0.0101",
      abs(s["total_actual_cost"] - 0.0101) < 1e-6,
      f'got {s["total_actual_cost"]}')
check("total_baseline_cost ≈ 0.015",
      abs(s["total_baseline_cost"] - 0.015) < 1e-6,
      f'got {s["total_baseline_cost"]}')
expected_saved = 0.015 - 0.0101
check("total_saved_usd ≈ 0.0049",
      abs(s["total_saved_usd"] - expected_saved) < 1e-6,
      f'got {s["total_saved_usd"]}')
expected_pct = expected_saved / 0.015 * 100
check("savings_pct correct",
      abs(s["savings_pct"] - expected_pct) < 0.1,
      f'got {s["savings_pct"]:.2f}, expected {expected_pct:.2f}')
check("cache_hits == 1",         s["cache_hits"] == 1,
      f'got {s["cache_hits"]}')
check("cache_hit_rate_pct == 50",
      abs(s["cache_hit_rate_pct"] - 50.0) < 0.1,
      f'got {s["cache_hit_rate_pct"]}')
check("tier_distribution has SMALL=1",
      s["tier_distribution"].get("SMALL") == 1,
      f'got {s["tier_distribution"]}')
check("tier_distribution has LARGE=1",
      s["tier_distribution"].get("LARGE") == 1,
      f'got {s["tier_distribution"]}')
check("recent_logs length == 2",
      len(m["recent_logs"]) == 2,
      f'got {len(m["recent_logs"])}')
print()

# ── Summary ──────────────────────────────────────────────────────────────
print("=" * 50)
print(f"  Results:  {PASS} passed, {FAIL} failed")
print("=" * 50)

if FAIL > 0:
    sys.exit(1)
else:
    print("\n🎉 All Module 1 checks passed!\n")
    sys.exit(0)
