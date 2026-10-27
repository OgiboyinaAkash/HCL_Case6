# Module 1: Persistence, Local Embedder & Two-Layer Semantic Cache

> **Project:** PrismRoute — Cost-Aware Model Router (Hackathon Case 6)  
> **Status:** Module 1 Complete, Fully Tested & Verified  
> **Target Audience:** Downstream AI Agents, Pair-Programmers, and Developers integrating Router (Module 2), Guardrails (Module 3), and Dashboard (Module 4).

---

## 1. Executive Summary & Purpose

Module 1 establishes the foundational persistence and caching infrastructure for **PrismRoute**. It provides:
1. **Persistent PostgreSQL Store with `pgvector`**: Stores query embeddings, metadata for guardrail checks, responses, and real-time telemetry logs.
2. **In-Process Singleton Embedder**: Uses `sentence-transformers` with `BAAI/bge-small-en-v1.5` (384-dimensional, normalized float32 embeddings) loaded once into memory.
3. **Two-Layer Semantic Cache with Safe-Reuse Guard**:
   - **Layer 1 (Exact Match)**: O(1) lookup via normalized SHA-256 query hash.
   - **Layer 2 (Semantic Match)**: Approximate Nearest Neighbor (ANN) search via HNSW cosine distance index in pgvector (`min_similarity >= 0.92`), followed by a **Safe-Reuse Guard** validation that compares query metadata (e.g., numbers, MCQ options, entities) to prevent unsafe answers (e.g. number-swap traps like "15% of 200" vs "15% of 300").
4. **Telemetry Logging & Analytics Store**: Persists every routing decision, model tier used, token counts, cost calculations, latency, and fallback states, with an aggregation engine powering live metrics.

---

## 2. System Architecture & Flow

```mermaid
flowchart TD
    UserQuery["Incoming User Query + Extracted Metadata"] --> NormHash["normalize_and_hash(query)"]
    NormHash --> ExactLookup{"lookup_exact_hash(hash)"}

    ExactLookup -- "Hash Matched" --> HitExact["Return status: HIT_EXACT<br/>(similarity: 1.0, latency: ~1ms)"]

    ExactLookup -- "Hash Not Found" --> Embed["LocalEmbedder.encode(query)<br/>BAAI/bge-small-en-v1.5 (384-dim)"]
    Embed --> VectorSearch["search_semantic_candidates(query_vec)<br/>pgvector HNSW Cosine Index (min_sim >= 0.92)"]

    VectorSearch -- "No candidates above threshold" --> Miss["Return status: MISS<br/>(similarity: 0.0, embedding: query_vec)"]

    VectorSearch -- "Candidates Found" --> SafeGuard{"Safe-Reuse Guard<br/>safe_reuse_validator(query_meta, cand_meta)"}
    
    SafeGuard -- "Validation Passed" --> HitSemantic["Return status: HIT_SEMANTIC<br/>(similarity: score, response_text, tier)"]
    SafeGuard -- "All Candidates Failed" --> RejectedUnsafe["Return status: REJECTED_UNSAFE<br/>(rejection reason, top similarity)"]
```

---

## 3. Directory & File Manifest

| File Path | Description / Responsibility |
|---|---|
| `docker-compose.yml` | Container specification for `pgvector/pgvector:pg16` on port `5432`, database `prism_router`. |
| `requirements.txt` | Python dependencies: `psycopg2-binary`, `pgvector`, `sentence-transformers`, `numpy`. |
| `backend/__init__.py` | Package marker for `backend`. |
| `backend/db.py` | Database connection factory (`get_connection`) and schema migration (`init_db`). |
| `backend/cache/__init__.py` | Sub-package marker for cache store and embedder. |
| `backend/cache/embedder.py` | Singleton `LocalEmbedder` class and global `embedder` instance (`BAAI/bge-small-en-v1.5`). |
| `backend/cache/cache_store.py` | Complete public interface for exact/semantic cache lookup, insertion, telemetry logging, and dashboard metrics. |
| `test_my_module.py` | End-to-end automated test suite verifying all 4 states (`HIT_EXACT`, `HIT_SEMANTIC`, `REJECTED_UNSAFE`, `MISS`) and telemetry math. |

---

## 4. Database Schema Specification

### 4.1 Table: `semantic_cache`
Stores cached queries, vectors, metadata for Safe-Reuse Guard, and responses.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

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

-- HNSW Index for fast approximate cosine nearest-neighbor search
CREATE INDEX IF NOT EXISTS idx_cache_embedding_hnsw
    ON semantic_cache
 USING hnsw (embedding vector_cosine_ops);
```

### 4.2 Table: `telemetry_logs`
Logs detailed operational telemetry per query for auditing, cost comparison against baseline, and dashboard reporting.

```sql
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id                      SERIAL PRIMARY KEY,
    timestamp               TIMESTAMPTZ DEFAULT NOW(),
    query_text              TEXT NOT NULL,
    cache_status            TEXT NOT NULL,  -- HIT_EXACT, HIT_SEMANTIC, REJECTED_UNSAFE, MISS
    selected_tier           TEXT NOT NULL,  -- SMALL, MEDIUM, LARGE
    p_small                 FLOAT,          -- Router probability for SMALL tier
    p_medium                FLOAT,          -- Router probability for MEDIUM tier
    p_large                 FLOAT,          -- Router probability for LARGE tier
    input_tokens            INT NOT NULL,
    expected_output_tokens  INT NOT NULL,
    actual_output_tokens    INT NOT NULL,
    actual_cost_usd         FLOAT NOT NULL,
    baseline_large_cost_usd FLOAT NOT NULL, -- Cost if routed to Large Model
    latency_ms              FLOAT NOT NULL,
    fallback_triggered      BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_telemetry_ts
    ON telemetry_logs (timestamp DESC);
```

---

## 5. Public API Reference (`backend.cache.cache_store`)

All downstream modules (Module 2 router, Module 3 guardrails, Module 4 dashboard/API) must interact through these functions:

### 1. `normalize_and_hash(query: str) -> str`
- **Behavior**: Lowercases the query, collapses multiple whitespaces, strips edges, and returns the SHA-256 hexadecimal string.

### 2. `lookup_exact_hash(query_hash: str) -> Optional[Dict[str, Any]]`
- **Behavior**: Direct index lookup in `semantic_cache` by `query_hash`.
- **Returns**: Dictionary with columns or `None`. `metadata` is parsed from JSONB.

### 3. `search_semantic_candidates(query_vec: np.ndarray, min_similarity: float = 0.92, top_k: int = 3) -> List[Dict[str, Any]]`
- **Behavior**: Queries `semantic_cache` using pgvector cosine similarity:
  $$\text{similarity} = 1 - (\text{embedding} \Leftrightarrow \text{query\_vec})$$
- **Returns**: List of dicts ordered by `similarity DESC`.

### 4. `check_semantic_cache(query: str, query_metadata: Dict[str, Any], safe_reuse_validator: Optional[Callable] = None, min_similarity: float = 0.92) -> Dict[str, Any]`
Main entry-point for routing. Returns a dictionary with guaranteed keys:
- `status`: `"HIT_EXACT"`, `"HIT_SEMANTIC"`, `"REJECTED_UNSAFE"`, or `"MISS"`
- `response_text`: `Optional[str]` (contains cached response if HIT)
- `tier_used`: `Optional[str]` (e.g. `"SMALL"`, `"MEDIUM"`, `"LARGE"`)
- `similarity`: `float` (1.0 for HIT_EXACT, 0.0 for MISS, score for SEMANTIC/REJECTED)
- `matched_query`: `Optional[str]` (query text from matching cache row)
- `latency_ms`: `float` (lookup duration in milliseconds)
- `reason`: `str` (explanation of why hit, missed, or rejected)
- `embedding`: `Optional[np.ndarray]` (`None` on HIT_EXACT to avoid redundant embedding compute; contains `(384,)` float32 array on MISS, REJECTED_UNSAFE, and HIT_SEMANTIC so caller does not need to re-encode).

### 5. `store_in_cache(query: str, query_vec: Optional[np.ndarray], metadata: Dict[str, Any], response_text: str, tier_used: str) -> None`
- **Behavior**: Inserts a new response into `semantic_cache`. Automatically computes `query_vec` if `None` is provided. Uses `ON CONFLICT (query_hash) DO NOTHING`.

### 6. `log_telemetry(log_data: Dict[str, Any]) -> None`
- **Behavior**: Inserts a structured telemetry log entry into `telemetry_logs`.

### 7. `get_dashboard_metrics(limit: int = 50) -> Dict[str, Any]`
- **Returns**:
  ```json
  {
    "summary": {
      "total_queries": 100,
      "total_actual_cost": 0.045,
      "total_baseline_cost": 0.500,
      "total_saved_usd": 0.455,
      "savings_pct": 91.0,
      "avg_latency_ms": 38.2,
      "cache_hits": 65,
      "cache_hit_rate_pct": 65.0,
      "unsafe_blocks": 5,
      "fallbacks": 2,
      "tier_distribution": {
        "SMALL": 70,
        "MEDIUM": 20,
        "LARGE": 10
      }
    },
    "recent_logs": [ ... ]
  }
  ```

### 8. `clear_cache() -> None`
- **Behavior**: Executes `TRUNCATE TABLE semantic_cache RESTART IDENTITY;`.

---

## 6. How Downstream AI Agents Should Use This Module

### Scenario A: Router Query Flow
```python
from backend.cache.cache_store import check_semantic_cache, store_in_cache, log_telemetry

# 1. Extract query metadata (numbers, options, entities from Module 3 guard)
query = "What is 15 percent of 200?"
metadata = {"numbers": ["15", "200"]}

# 2. Check Cache
cache_result = check_semantic_cache(query, metadata)

if cache_result["status"] in ("HIT_EXACT", "HIT_SEMANTIC"):
    # Cache hit! Zero LLM cost incurred
    response = cache_result["response_text"]
    selected_tier = cache_result["tier_used"]
    # Log telemetry with actual_cost_usd = 0.0
else:
    # Cache miss or REJECTED_UNSAFE -> Call selected model tier
    # Use cache_result["embedding"] if the router needs the vector, no need to re-encode!
    ...
    # After LLM returns:
    store_in_cache(
        query=query,
        query_vec=cache_result["embedding"],
        metadata=metadata,
        response_text=llm_response,
        tier_used=selected_tier,
    )
```

### Scenario B: Custom Safe-Reuse Validator
Downstream agents can supply a specialized domain validator function into `check_semantic_cache`:
```python
def my_guardrail_validator(query_meta: dict, cached_meta: dict) -> tuple[bool, str]:
    if query_meta.get("numbers") != cached_meta.get("numbers"):
        return False, "Numerical values differ between queries."
    if query_meta.get("options") != cached_meta.get("options"):
        return False, "Multiple-choice options differ."
    return True, "Safe to reuse."

result = check_semantic_cache(
    query=query,
    query_metadata=metadata,
    safe_reuse_validator=my_guardrail_validator,
    min_similarity=0.92,
)
```

---

## 7. Setup & Execution Notes

### Database Connection
`backend/db.py` reads `DATABASE_URL` from the environment:
```bash
# Default if omitted:
DATABASE_URL=postgresql://postgres:password@localhost:5432/prism_router
```

### Starting the Database
- **Option 1 (Docker)**:
  ```bash
  docker compose up -d
  ```
- **Option 2 (Native PostgreSQL with pgvector)**:
  PostgreSQL 16 with pgvector extension `0.8.7` is installed and running as a Windows Service (`pgsql16`) on `localhost:5432`.

### Running Verification Tests
```bash
python test_my_module.py
```
Expected output: **19 passed, 0 failed**.
