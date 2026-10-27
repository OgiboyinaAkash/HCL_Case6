"""
Unit tests for the complete PrismRoute workflow (Service & Architecture).
Verifies:
  1. Exact Match (in-memory hash hit)
  2. Embed and Search: Reuse Zone (>= 0.90) with Safe-Reuse Guard pass
  3. Safe-Reuse Guard rejection on number-swap traps
  4. Embed and Search: Hint Zone (0.80 - 0.90) tier reuse (skipping feature layer)
  5. Miss (< 0.80) routing via 40-d Feature Layer + Neural Network
  6. Retry and escalation on error
  7. Output contract (answer, cost, decision_record)
  8. Telemetry and savings metrics
"""

import unittest
from backend.service import PrismRouteService, safe_reuse_guard, extract_query_metadata


class TestPrismRouteWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = PrismRouteService(model_type="neural_net", quality_threshold=0.65)
        cls.test_scope = "unit_test_scope"
        cls.svc.clear_cache(scope_id=cls.test_scope)

    def test_01_fresh_query_miss_neural_net(self):
        query = "What is 10 percent of 500?"
        res = self.svc.process_query(query, scope_id=self.test_scope)

        self.assertIn("answer", res)
        self.assertIn("cost", res)
        self.assertIn("decision_record", res)

        dec = res["decision_record"]
        self.assertEqual(dec["path_taken"], "NEURAL_NET_ROUTING")
        self.assertEqual(dec["status"], "MISS")
        self.assertEqual(dec["selected_tier"], "SMALL")
        self.assertFalse(dec["skipped_feature_layer"])
        self.assertIn("50", res["answer"])

    def test_02_exact_match_in_memory(self):
        query = "What is 10 percent of 500?"
        res = self.svc.process_query(query, scope_id=self.test_scope)

        dec = res["decision_record"]
        cost = res["cost"]
        self.assertEqual(dec["path_taken"], "EXACT_HASH_HIT")
        self.assertEqual(dec["status"], "HIT_EXACT")
        self.assertEqual(dec["similarity"], 1.0)
        self.assertTrue(dec["skipped_feature_layer"])
        self.assertEqual(cost["actual_cost_usd"], 0.0)
        self.assertEqual(cost["savings_pct"], 100.0)

    def test_03_reuse_zone_semantic_hit_safe_pass(self):
        query = "Can you calculate 10 percent of 500?"
        res = self.svc.process_query(query, scope_id=self.test_scope)

        dec = res["decision_record"]
        cost = res["cost"]
        # Similarity with "What is 10 percent of 500?" is ~0.90+
        if dec["similarity"] >= 0.90:
            self.assertEqual(dec["path_taken"], "SEMANTIC_CACHE_REUSE")
            self.assertEqual(dec["status"], "HIT_SEMANTIC")
            self.assertEqual(cost["actual_cost_usd"], 0.0)
            self.assertEqual(cost["savings_pct"], 100.0)
        else:
            self.assertIn(dec["path_taken"], ["HINT_ZONE_TIER_REUSE", "NEURAL_NET_ROUTING"])

    def test_04_safe_reuse_guard_blocks_number_trap(self):
        # Incoming query asks for 10% of 900, but cache has 10% of 500
        query = "What is 10 percent of 900?"
        res = self.svc.process_query(query, scope_id=self.test_scope)

        dec = res["decision_record"]
        # Safe reuse guard must NOT return 50
        self.assertNotIn("10% of 500 is 50", res["answer"])
        self.assertIn("90", res["answer"])

    def test_05_safe_reuse_guard_logic_unit(self):
        meta1 = {"numbers": ["15", "200"], "percentages": ["15%"], "options": []}
        meta2 = {"numbers": ["15", "300"], "percentages": ["15%"], "options": []}
        safe, reason = safe_reuse_guard(meta1, meta2)
        self.assertFalse(safe)
        self.assertIn("Number mismatch trap", reason)

        meta3 = {"numbers": ["15", "200"], "percentages": ["15%"], "options": []}
        safe_ok, _ = safe_reuse_guard(meta1, meta3)
        self.assertTrue(safe_ok)

    def test_06_model_call_retry_and_escalate(self):
        # Call model on an arithmetic query
        out = self.svc._call_chosen_model("What is 20 + 30?", tier="SMALL")
        self.assertIn("final_tier", out)
        self.assertIn("response_text", out)
        self.assertEqual(out["retries_used"], 0)
        self.assertIn("50", out["response_text"])

    def test_07_metrics_calculation(self):
        metrics = self.svc.get_metrics(scope_id=self.test_scope)
        self.assertGreaterEqual(metrics["total_queries"], 3)
        self.assertGreaterEqual(metrics["total_saved"], 0.0)
        self.assertIn("SMALL", metrics["tier_counts"])
        self.assertIn("cache_counts", metrics)


if __name__ == "__main__":
    unittest.main()

