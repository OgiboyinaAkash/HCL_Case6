# PrismRoute — Cost-Aware Model Router (Case 6)

## Problem Statement
- **Scenario:** LLM bills grow quickly, and many queries do not need the largest model.
- **Problem:** Build a layer that decides which model to call for each query (small, medium or large), reuses answers through semantic caching, and keeps answer quality above a threshold while minimizing cost.
- **Must build:** A routing policy, a semantic cache with a safe-reuse rule, a cost and latency tracker, and a report of cost per correct answer compared with always using the large model.
- **Bonus:** A router that learns from feedback, and fallback to a larger model when a small model's answer looks uncertain.
- **Data:** Public sources: RouteLLM (open router framework and evaluation approach); MMLU and GSM8K for queries of varying difficulty; GPTCache as a reference for semantic caching.
- **Baseline:** 500 queries with reference answers and the price sheet for available models.

---

## Architecture & Module Documentation

- **[MODULE_1_HANDOFF.md](MODULE_1_HANDOFF.md)**: Complete architecture, database schema, local embedder specifications, two-layer cache API reference, and integration guide for downstream AI agents and developers.

---

## Quickstart (Module 1)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Database:**
   ```bash
   docker compose up -d
   ```
   *(Or connect to a local PostgreSQL instance with pgvector extension enabled)*

3. **Run the Verification Suite:**
   ```bash
   python test_my_module.py
   ```
