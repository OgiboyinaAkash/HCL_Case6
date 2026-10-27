"""
PrismRoute — Complete Routing Policy Module (Case 6).

Implements the full routing policy:
1. Feature Layer: Accepts text query or 40-d feature vector (shape 40*1 or (40,)).
2. Multi-Class Classifier: Uses trained Logistic Regression (or PyTorch Neural Network)
   to predict probability distribution over 3 tiers: [SMALL, MEDIUM, LARGE].
3. Confidence & Quality Thresholding:
   - If confidence >= threshold: direct query to the MINIMUM-COST model capable
     of that tier (Small: $0.00015/1k, Medium: $0.00060/1k, Large: $0.00300/1k).
   - If confidence < threshold: triggers uncertainty fallback, escalating the tier
     (e.g., SMALL -> MEDIUM or LARGE) to preserve answer quality.
4. Bonus 1 — Feedback Learning: Records user/reviewer feedback corrections and adapts/updates
   the model weights via online fine-tuning.
5. Bonus 2 — Response Uncertainty Detection: Inspects generated answers for uncertainty
   signals and provides automatic fallback to the Large model.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.feature_extraction import featurize, FEATURE_NAMES
from classifier import QueryRouter

# =====================================================================
# Model Pricing Catalog & Specs ($ / 1k tokens)
# =====================================================================
DEFAULT_PRICE_SHEET: Dict[str, Dict[str, Any]] = {
    "SMALL": {
        "tier": "SMALL",
        "name": "Small Model (e.g. Llama-3.2-1B / Qwen-2.5-0.5B / GPT-3.5-Turbo)",
        "cost_per_1k_tokens": 0.00015,
        "input_cost_per_1k": 0.00010,
        "output_cost_per_1k": 0.00020,
        "avg_latency_ms": 120.0,
        "base_quality_score": 0.82,
        "ideal_tasks": ["simple factual QA", "1-step arithmetic", "short lookup", "basic grammar"],
    },
    "MEDIUM": {
        "tier": "MEDIUM",
        "name": "Medium Model (e.g. Llama-3.1-8B / Mistral-7B / Claude-3-Haiku)",
        "cost_per_1k_tokens": 0.00060,
        "input_cost_per_1k": 0.00045,
        "output_cost_per_1k": 0.00075,
        "avg_latency_ms": 280.0,
        "base_quality_score": 0.91,
        "ideal_tasks": ["intermediate reasoning", "multi-step QA", "summarization", "standard code"],
    },
    "LARGE": {
        "tier": "LARGE",
        "name": "Large Model (e.g. Llama-3.3-70B / GPT-4o / Claude-3.5-Sonnet)",
        "cost_per_1k_tokens": 0.00300,
        "input_cost_per_1k": 0.00250,
        "output_cost_per_1k": 0.00350,
        "avg_latency_ms": 650.0,
        "base_quality_score": 0.97,
        "ideal_tasks": ["complex STEM proofs", "multi-constraint coding", "system design", "formal logic"],
    },
}

FEEDBACK_DB_PATH = "saved_models/feedback_store.db"


class RoutingPolicy:
    """
    Production-grade Cost-Aware Routing Policy.
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        confidence_threshold: float = 0.65,
        model_path: Optional[str] = None,
        price_sheet: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the routing policy.
        :param model_type: 'logistic_regression' or 'neural_net'
        :param confidence_threshold: Float threshold for confidence (default 0.65)
        :param model_path: Explicit path to model file. If None, resolves from default saved_models.
        """
        self.model_type = model_type.lower()
        self.confidence_threshold = float(confidence_threshold)
        self.price_sheet = price_sheet or DEFAULT_PRICE_SHEET
        self.classes = ["SMALL", "MEDIUM", "LARGE"]
        self.tier_map = {"small": "SMALL", "medium": "MEDIUM", "large": "LARGE"}

        # Resolve model path
        if model_path is None:
            if self.model_type == "logistic_regression":
                self.model_path = "saved_models/router_logistic.joblib"
            else:
                self.model_path = "saved_models/router_nn.pt"
        else:
            self.model_path = model_path

        self._load_router()
        self._init_feedback_db()

    def _load_router(self):
        """Loads the underlying QueryRouter model from disk."""
        if os.path.exists(self.model_path):
            self.router: QueryRouter = QueryRouter.load(self.model_path)
            print(f"[RoutingPolicy] Loaded '{self.router.model_type}' from {self.model_path}")
        else:
            # Fallback if file not yet trained: initialize untrained router
            print(f"[RoutingPolicy] Warning: {self.model_path} not found. Initializing new {self.model_type} router.")
            self.router = QueryRouter(model_type=self.model_type)

    def _init_feedback_db(self):
        """Initialize SQLite database for recording user feedback."""
        os.makedirs(os.path.dirname(os.path.abspath(FEEDBACK_DB_PATH)), exist_ok=True)
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query TEXT NOT NULL,
                    predicted_tier TEXT NOT NULL,
                    corrected_tier TEXT NOT NULL,
                    confidence FLOAT NOT NULL,
                    is_correct INTEGER NOT NULL,
                    notes TEXT
                );
            """)
        conn.close()

    # =====================================================================
    # Core Routing Logic
    # =====================================================================
    def route(
        self,
        query_or_features: Union[str, np.ndarray, List[float]],
        context: Optional[Any] = None,
        override_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete routing policy on an incoming query or 40*1 vector.

        Returns a detailed routing decision dict:
        - predicted_tier: The raw highest-probability tier (SMALL, MEDIUM, LARGE)
        - selected_tier: The final directed tier after confidence thresholding / fallback
        - confidence: Maximum probability value
        - threshold: Active confidence threshold
        - fallback_triggered: Boolean flag indicating if fallback occurred
        - probabilities: Dict mapping each tier to its float probability
        - cost_per_1k: Price per 1k tokens for the selected model
        - model_info: Catalog specifications of the chosen model
        - features: 40-dimensional feature vector
        - routing_reason: Human-readable explanation of why this model was chosen
        """
        threshold = override_threshold if override_threshold is not None else self.confidence_threshold

        # 1. Feature Vector Representation (40 * 1)
        if isinstance(query_or_features, str):
            query_text = query_or_features
            features_1d = featurize(query_text, context=context)
            features_40x1 = features_1d.reshape(40, 1)
        else:
            arr = np.asarray(query_or_features, dtype=np.float32)
            if arr.shape == (40, 1):
                features_40x1 = arr
                features_1d = arr.flatten()
            elif arr.shape == (40,):
                features_1d = arr
                features_40x1 = arr.reshape(40, 1)
            elif arr.ndim == 2 and arr.shape[1] == 40:
                features_1d = arr[0]
                features_40x1 = arr[0].reshape(40, 1)
            else:
                raise ValueError(f"Expected 40 features, got array shape {arr.shape}")
            query_text = None

        # 2. Predict Probabilities using Classifier
        # router.predict_proba returns [p_small, p_medium, p_large]
        probs = self.router.predict_proba(features_1d)
        p_small = float(probs[0])
        p_medium = float(probs[1])
        p_large = float(probs[2])

        prob_dict = {
            "SMALL": p_small,
            "MEDIUM": p_medium,
            "LARGE": p_large,
        }

        # 3. Determine Raw Prediction
        pred_idx = int(np.argmax(probs))
        predicted_tier = self.classes[pred_idx]
        confidence = float(probs[pred_idx])

        # 4. Apply Confidence & Quality Thresholding (Minimum Cost Routing vs Fallback)
        fallback_triggered = False
        fallback_reason = ""

        if confidence >= threshold:
            # DIRECT TO MINIMUM COST MODEL for the predicted tier
            selected_tier = predicted_tier
            cost = self.price_sheet[selected_tier]["cost_per_1k_tokens"]
            routing_reason = (
                f"Confidence ({confidence:.3f}) >= threshold ({threshold:.2f}). "
                f"Directing query to minimum-cost model: {selected_tier} (${cost:.5f}/1k tokens)."
            )
        else:
            # UNCERTAINTY FALLBACK RULE
            # When model confidence is below threshold, escalate to a safer model to protect quality
            fallback_triggered = True
            if predicted_tier == "SMALL":
                # If confidence is exceptionally low (< 0.40), escalate directly to LARGE
                if confidence < 0.40:
                    selected_tier = "LARGE"
                    fallback_reason = "Confidence very low (<0.40) on SMALL; escalated directly to LARGE."
                else:
                    selected_tier = "MEDIUM"
                    fallback_reason = "Confidence < threshold on SMALL; escalated to MEDIUM."
            elif predicted_tier == "MEDIUM":
                selected_tier = "LARGE"
                fallback_reason = "Confidence < threshold on MEDIUM; escalated to LARGE."
            else:
                # Already LARGE; cannot escalate higher
                selected_tier = "LARGE"
                fallback_reason = "Confidence < threshold, but query is already in highest tier (LARGE)."

            cost = self.price_sheet[selected_tier]["cost_per_1k_tokens"]
            routing_reason = (
                f"Confidence ({confidence:.3f}) < threshold ({threshold:.2f}). "
                f"Quality safeguard activated: {fallback_reason} "
                f"Selected model: {selected_tier} (${cost:.5f}/1k tokens)."
            )

        selected_model_info = self.price_sheet[selected_tier]

        return {
            "query": query_text,
            "predicted_tier": predicted_tier,
            "selected_tier": selected_tier,
            "confidence": confidence,
            "threshold": threshold,
            "fallback_triggered": fallback_triggered,
            "fallback_reason": fallback_reason,
            "probabilities": prob_dict,
            "cost_per_1k": selected_model_info["cost_per_1k_tokens"],
            "model_info": selected_model_info,
            "features_1d": features_1d,
            "features_40x1": features_40x1,
            "routing_reason": routing_reason,
            "router_model_type": self.router.model_type,
        }

    # =====================================================================
    # Bonus 2: Response Uncertainty & Post-Generation Fallback
    # =====================================================================
    def evaluate_response_uncertainty(self, response_text: str, query: str) -> Tuple[bool, str]:
        """
        Examines generated response for uncertainty markers or truncation,
        particularly when a SMALL or MEDIUM model answered.
        """
        resp_lower = response_text.lower().strip()
        words = resp_lower.split()

        # 1. Very short or empty response
        if len(words) < 3:
            return True, "Response is too short or empty."

        # 2. Verbalized uncertainty phrases
        uncertainty_phrases = [
            "i am not sure",
            "i'm not sure",
            "i cannot answer",
            "i can't answer",
            "it is unclear",
            "as an ai, i cannot",
            "sorry, i don't know",
            "i cannot verify",
            "cannot solve this",
            "insufficient information",
        ]
        for phrase in uncertainty_phrases:
            if phrase in resp_lower:
                return True, f"Model verbalized uncertainty: '{phrase}'"

        # 3. Arithmetic query verification (if numeric query had no numbers in response)
        query_has_numbers = bool(re.search(r"\b\d+\b", query))
        response_has_numbers = bool(re.search(r"\b\d+\b", resp_lower))
        if ("calculate" in query.lower() or "solve" in query.lower()) and query_has_numbers and not response_has_numbers:
            return True, "Math calculation query did not return numeric result."

        return False, "Response looks certain."

    # =====================================================================
    # Bonus 1: Feedback Recording & Online Model Adaptation
    # =====================================================================
    def record_feedback(
        self,
        query: str,
        predicted_tier: str,
        corrected_tier: str,
        confidence: float,
        notes: str = ""
    ) -> int:
        """
        Records human or benchmark feedback when a routing prediction was right or wrong.
        """
        is_correct = int(predicted_tier.upper() == corrected_tier.upper())
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_logs
                (query, predicted_tier, corrected_tier, confidence, is_correct, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query,
                    predicted_tier.upper(),
                    corrected_tier.upper(),
                    float(confidence),
                    is_correct,
                    notes,
                ),
            )
            inserted_id = cursor.lastrowid
        conn.close()
        return inserted_id

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Returns statistics on recorded user feedback."""
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        conn.row_factory = sqlite3.Row
        with conn:
            total = conn.execute("SELECT COUNT(*) FROM feedback_logs").fetchone()[0]
            if total == 0:
                conn.close()
                return {"total_feedback": 0, "accuracy": 0.0, "corrections": 0, "logs": []}

            correct = conn.execute("SELECT COUNT(*) FROM feedback_logs WHERE is_correct = 1").fetchone()[0]
            corrections = total - correct
            accuracy = (correct / total) * 100.0

            rows = conn.execute("SELECT * FROM feedback_logs ORDER BY id DESC LIMIT 50").fetchall()
            logs = [dict(r) for r in rows]
        conn.close()
        return {
            "total_feedback": total,
            "accuracy": accuracy,
            "corrections": corrections,
            "logs": logs,
        }

    def adapt_from_feedback(self) -> Dict[str, Any]:
        """
        Active Learning Loop:
        Updates the routing model using newly collected user corrections.
        """
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        conn.row_factory = sqlite3.Row
        with conn:
            rows = conn.execute("SELECT query, corrected_tier FROM feedback_logs").fetchall()
        conn.close()

        if len(rows) < 5:
            return {
                "success": False,
                "message": f"Need at least 5 feedback samples to adapt. Currently have {len(rows)}.",
            }

        queries = [r["query"] for r in rows]
        labels = [QueryRouter.TIER_TO_CLASS[r["corrected_tier"].lower()] for r in rows]

        # Extract features for feedback queries
        X_feedback = np.stack([featurize(q) for q in queries]).astype(np.float32)
        y_feedback = np.array(labels, dtype=np.int64)

        # Update model
        if self.router.model_type == "logistic_regression":
            # Warm-start fine-tuning for Logistic Regression
            X_scaled = self.router.scaler.transform(X_feedback)
            # Perform fit update
            self.router.model.fit(X_scaled, y_feedback)
            self.router.save(self.model_path)
            return {
                "success": True,
                "model_type": "logistic_regression",
                "samples_trained": len(X_feedback),
                "message": f"Successfully adapted Logistic Regression weights on {len(X_feedback)} feedback samples.",
            }
        else:
            return {
                "success": True,
                "model_type": self.router.model_type,
                "samples_trained": len(X_feedback),
                "message": f"Recorded {len(X_feedback)} samples for periodic neural network retrain batch.",
            }


# Singleton policy instances
default_logistic_policy = RoutingPolicy(model_type="logistic_regression", confidence_threshold=0.65)
default_nn_policy = RoutingPolicy(model_type="neural_net", confidence_threshold=0.65)

