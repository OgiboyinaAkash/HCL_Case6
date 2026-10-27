"""
Unit tests for QueryRouter and classifier pipeline.
"""

import os
import unittest
import numpy as np

from classifier import QueryRouter, QueryRouterNN, PYTORCH_AVAILABLE, load_and_predict


class TestQueryRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "saved_models/router_nn.pt"
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest("Model file saved_models/router_nn.pt not found. Run classifier.py first.")
        cls.router = QueryRouter.load(cls.model_path)

    def test_network_shapes_if_pytorch(self):
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch is not available.")
        import torch
        net = QueryRouterNN(input_dim=40, num_classes=3)
        net.eval()
        dummy_input = torch.randn(5, 40)
        output = net(dummy_input)
        self.assertEqual(output.shape, (5, 3))

    def test_input_handling_single_string(self):
        query = "What is the capital of France?"
        pred = self.router.predict(query)
        self.assertIn(pred, [0, 1, 2])

        details = self.router.predict_with_details(query)
        self.assertIn("predicted_class", details)
        self.assertIn("tier_name", details)
        self.assertIn(details["tier_name"], ["small", "medium", "large"])
        self.assertIn("p_small", details)
        self.assertIn("p_medium", details)
        self.assertIn("p_large", details)

        total_prob = details["p_small"] + details["p_medium"] + details["p_large"]
        self.assertAlmostEqual(total_prob, 1.0, places=4)

    def test_input_handling_batch_strings(self):
        queries = ["Calculate 15 + 28", "Explain the difference between TCP and UDP"]
        preds = self.router.predict(queries)
        self.assertEqual(len(preds), 2)
        for p in preds:
            self.assertIn(p, [0, 1, 2])

    def test_input_handling_numeric_vectors(self):
        # 1D array
        vec_1d = np.random.randn(40).astype(np.float32)
        pred_1d = self.router.predict(vec_1d)
        self.assertIn(pred_1d, [0, 1, 2])

        # (40, 1) column vector
        vec_col = np.random.randn(40, 1).astype(np.float32)
        pred_col = self.router.predict(vec_col)
        self.assertIn(pred_col, [0, 1, 2])

        # (2, 40) batch
        vec_batch = np.random.randn(2, 40).astype(np.float32)
        pred_batch = self.router.predict(vec_batch)
        self.assertEqual(len(pred_batch), 2)

    def test_save_and_reload(self):
        temp_path = "saved_models/test_temp_model.pt"
        try:
            self.router.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))

            reloaded = QueryRouter.load(temp_path)
            q = "What is 100 divided by 4?"
            orig_res = self.router.predict_with_details(q)
            reloaded_res = reloaded.predict_with_details(q)

            self.assertEqual(orig_res["predicted_class"], reloaded_res["predicted_class"])
            self.assertAlmostEqual(orig_res["p_small"], reloaded_res["p_small"], places=4)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_standalone_load_and_predict(self):
        res = load_and_predict("What is 1 + 1?", model_path=self.model_path)
        self.assertIn(res["predicted_class"], [0, 1, 2])
        self.assertIn(res["tier_name"], ["small", "medium", "large"])


if __name__ == "__main__":
    unittest.main()

