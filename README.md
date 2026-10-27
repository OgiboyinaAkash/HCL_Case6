Case 6: Cost-Aware Model Router
Scenario: LLM bills grow quickly, and many queries do not need the largest model.
Problem: Build a layer that decides which model to call for each query (small, medium or large), reuses answers through semantic caching, and keeps answer quality above a threshold while minimizing cost.
Must build: A routing policy, a semantic cache with a safe-reuse rule, a cost and latency tracker, and a report of cost per correct answer compared with always using the large model.
Bonus: A router that learns from feedback, and fallback to a larger model when a small models answer looks uncertain.
Data: Public sources: RouteLLM (open router framework and evaluation approach); MMLU and GSM8K for queries of varying difficulty; GPTCache as a reference for semantic caching.
Also provided at the start: 500 queries with reference answers and the price sheet for the available models.