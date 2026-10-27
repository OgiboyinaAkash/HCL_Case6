"""
tests/test_routing_policy.py
Unit tests for the PrismRoute Routing Policy.

Verifies:
1. Routing from text query and (40, 1) feature vector
2. 3-tier probability distribution (SMALL, MEDIUM, LARGE)
3. Confidence thresholding and minimum-cost model selection
4. Uncertainty fallback behavior when confidence < threshold
5. Response uncertainty evaluator
6. Feedback recording and feedback database persistence
"""

import os
import unittest
import numpy as np

from backend.routing_policy import RoutingPolicy, DEFAULT_PRICE_SHEET


class TestRoutingPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = RoutingPolicy(model_type="logistic_regression", confidence_threshold=0.60)

    def test_routing_simple_query(self):
        query = "What is 15 percent of 200?"
        decision = self.policy.route(query)

        self.assertIn("predicted_tier", decision)
        self.assertIn("selected_tier", decision)
        self.assertIn(decision["selected_tier"], ["SMALL", "MEDIUM", "LARGE"])
        self.assertIn("confidence", decision)
        self.assertIn("probabilities", decision)
        self.assertIn("cost_per_1k", decision)
        self.assertIn("routing_reason", decision)

        # Probabilities sum to 1.0
        probs = decision["probabilities"]
        total = probs["SMALL"] + probs["MEDIUM"] + probs["LARGE"]
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_routing_from_40x1_vector(self):
        vec_40x1 = np.random.randn(40, 1).astype(np.float32)
        decision = self.policy.route(vec_40x1)

        self.assertIn(decision["selected_tier"], ["SMALL", "MEDIUM", "LARGE"])
        self.assertEqual(decision["features_40x1"].shape, (40, 1))

    def test_confidence_threshold_fallback(self):
        # Force a very high threshold (0.999) so fallback is guaranteed to trigger
        query = "Can you write an essay about machine learning with code examples?"
        fallback_decision = self.policy.route(query, override_threshold=0.999)

        if fallback_decision["confidence"] < 0.999:
            self.assertTrue(fallback_decision["fallback_triggered"])
            self.assertIn(fallback_decision["selected_tier"], ["MEDIUM", "LARGE"])

        # Force a low threshold (0.10) so fallback does not trigger
        direct_decision = self.policy.route(query, override_threshold=0.10)
        self.assertFalse(direct_decision["fallback_triggered"])
        self.assertEqual(direct_decision["selected_tier"], direct_decision["predicted_tier"])

    def test_minimum_cost_model_assignment(self):
        query = "What is 2 + 2?"
        decision = self.policy.route(query, override_threshold=0.10)
        tier = decision["selected_tier"]
        expected_cost = DEFAULT_PRICE_SHEET[tier]["cost_per_1k_tokens"]
        self.assertEqual(decision["cost_per_1k"], expected_cost)

    def test_response_uncertainty_detector(self):
        # 1. Uncertain response
        uncertain_resp = "I am not sure about the exact theorem, but it might be related."
        is_uncertain, reason = self.policy.evaluate_response_uncertainty(uncertain_resp, "Prove Fermat's theorem")
        self.assertTrue(is_uncertain)
        self.assertIn("uncertainty", reason.lower())

        # 2. Confident response
        confident_resp = "15% of 200 is 30. This is calculated as (15/100) * 200 = 30."
        is_uncertain, reason = self.policy.evaluate_response_uncertainty(confident_resp, "What is 15% of 200?")
        self.assertFalse(is_uncertain)

    def test_feedback_recording(self):
        fb_id = self.policy.record_feedback(
            query="Sample query for feedback testing",
            predicted_tier="SMALL",
            corrected_tier="MEDIUM",
            confidence=0.55,
            notes="Requires multi-step explanation"
        )
        self.assertIsInstance(fb_id, int)
        self.assertGreater(fb_id, 0)

        summary = self.policy.get_feedback_summary()
        self.assertGreaterEqual(summary["total_feedback"], 1)


if __name__ == "__main__":
    unittest.main()

