"""
classifier.py
PrismRoute — Query Tier Routing Classifier (Case 6)

Classifies incoming queries into one of three model complexity tiers:
  - Class 0: 'small'  (e.g., Llama-3-8B, Mistral-7B, GPT-3.5-Turbo)
  - Class 1: 'medium' (e.g., Mixtral-8x7B, Claude 3 Haiku, Llama-3-70B)
  - Class 2: 'large'  (e.g., GPT-4o, Claude 3.5 Sonnet)

Supports:
  1. PyTorch Neural Network (MLP: 40 -> 64 -> 32 -> 3) with BatchNorm & Dropout
  2. Scikit-Learn Logistic Regression & Random Forest
  3. Automatic 40-dimensional feature extraction from raw query text or precomputed vectors
  4. Training directly from cleaned_data.csv with feature caching
  5. Full persistence: save/load weights, scaler, and metadata
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import deterministic 40-d feature extractor
from backend.feature_extraction import FEATURE_NAMES, featurize

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# =====================================================================
# 1. PyTorch Neural Network Architecture (40 -> 64 -> 32 -> 3)
# =====================================================================
if PYTORCH_AVAILABLE:
    class QueryRouterNN(nn.Module):
        """
        Multilayer Perceptron (Neural Network) for query routing:
        Input:  40-dimensional feature vector
        Hidden: 64 -> BatchNorm1d -> ReLU -> Dropout -> 32 -> ReLU
        Output: 3 logits (Class 0: Small, Class 1: Medium, Class 2: Large)
        """
        def __init__(self, input_dim: int = 40, num_classes: int = 3, dropout_rate: float = 0.2):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


# =====================================================================
# 2. Main Router Pipeline: Train, Evaluate, Store, and Predict
# =====================================================================
class QueryRouter:
    """
    Complete pipeline to train, save, load, and run inference for query routing.
    Supports 'neural_net' (PyTorch MLP), 'logistic_regression' (Scikit-Learn),
    and 'random_forest' (Scikit-Learn).

    Can accept:
      - Raw text queries (e.g. str or List[str])
      - 40-dimensional numerical feature vectors (e.g. shape (40,), (40, 1), or (N, 40))
    """
    CLASS_NAMES = {0: "small", 1: "medium", 2: "large"}
    TIER_TO_CLASS = {"small": 0, "medium": 1, "large": 2}

    def __init__(self, model_type: str = "neural_net", input_dim: int = 40, num_classes: int = 3):
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.model = None
        self.metrics_: Dict[str, Any] = {}

        if self.model_type == "neural_net":
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is not available. Please use model_type='logistic_regression'.")
            self.model = QueryRouterNN(input_dim=input_dim, num_classes=num_classes)
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. Choose 'neural_net', 'logistic_regression', or 'random_forest'."
            )

    def _to_feature_matrix(self, x: Any) -> Tuple[np.ndarray, bool]:
        """
        Converts various input formats (raw text query, list of queries, 1D/2D numpy array)
        into a 2D float32 NumPy array of shape (N, input_dim).

        Returns:
            (features_2d, is_single_sample_flag)
        """
        # Case 1: single text query string
        if isinstance(x, str):
            feat = featurize(x)
            return feat.reshape(1, -1), True

        # Case 2: list or tuple of string queries
        if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], str):
            feats = np.stack([featurize(q) for q in x]).astype(np.float32)
            return feats, False

        # Case 3: numeric list or numpy array
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {arr.shape[0]}")
            return arr.reshape(1, -1), True
        elif arr.ndim == 2:
            if arr.shape[0] == self.input_dim and arr.shape[1] == 1:
                return arr.T, True
            if arr.shape[1] != self.input_dim:
                raise ValueError(f"Expected {self.input_dim} features, got {arr.shape[1]}")
            is_single = (arr.shape[0] == 1)
            return arr, is_single
        else:
            raise ValueError(f"Unsupported input array shape: {arr.shape}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        epochs: int = 35,
        batch_size: int = 64,
        lr: float = 0.002,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Performs train/test split, scales features, and trains the model.
        :param X: Feature matrix of shape (N, 40)
        :param y: Labels array of shape (N,) with integer values in {0, 1, 2}
        :param test_size: Proportion of dataset to include in the test split
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        assert X.shape[1] == self.input_dim, f"Expected {self.input_dim} features, got {X.shape[1]}"

        # 1. Stratified Train / Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")

        # 2. Fit and apply standard scaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 3. Train the selected model
        if self.model_type == "neural_net":
            self._train_neural_net(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
        else:
            print(f"Training {self.model_type} model...")
            self.model.fit(X_train_scaled, y_train)

        # 4. Evaluate on Test Split
        print("\n--- Evaluation on Test Set ---")
        y_pred = self.predict(X_test, is_raw_unscaled=True)
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(
            y_test, y_pred, target_names=["0 (small)", "1 (medium)", "2 (large)"], output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred)

        print(f"Test Accuracy: {acc * 100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["0 (small)", "1 (medium)", "2 (large)"]))
        print("Confusion Matrix:")
        print(cm)

        self.metrics_ = {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test)
        }
        return self.metrics_

    def train_from_csv(
        self,
        csv_path: str = "cleaned_data.csv",
        cache_path: Optional[str] = "saved_models/features_cache.npz",
        test_size: float = 0.2,
        epochs: int = 35,
        batch_size: int = 64,
        lr: float = 0.002
    ) -> Dict[str, Any]:
        """
        Loads data from CSV (or cached features), extracts 40-d features, and trains the model.
        """
        X, y, _ = load_and_featurize_data(csv_path=csv_path, cache_path=cache_path)
        return self.train(X, y, test_size=test_size, epochs=epochs, batch_size=batch_size, lr=lr)

    def _train_neural_net(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, lr: float):
        """Internal training loop for PyTorch Neural Network with AdamW and LR scheduler."""
        print(f"Training Neural Network ({self.input_dim} -> 64 -> 32 -> {self.num_classes}) for {epochs} epochs...")
        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            scheduler.step()
            if epoch % 5 == 0 or epoch == epochs:
                avg_loss = total_loss / len(X_train)
                print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f}")

    def predict(self, x: Any, is_raw_unscaled: bool = True) -> Union[int, np.ndarray]:
        """
        Predict class label {0, 1, 2}.
        Accepts:
          - single query string (e.g., "What is 2+2?") -> returns int class {0, 1, 2}
          - list of query strings -> returns 1D array of class IDs
          - 1D vector (40,) or column vector (40, 1) -> returns int class {0, 1, 2}
          - batch (N, 40) -> returns 1D array of class IDs
        """
        probs = self.predict_proba(x, is_raw_unscaled=is_raw_unscaled)
        if probs.ndim == 1:
            return int(np.argmax(probs))
        return np.argmax(probs, axis=1)

    def predict_proba(self, x: Any, is_raw_unscaled: bool = True) -> np.ndarray:
        """
        Predict probability distribution over [small, medium, large].
        """
        x_mat, is_single = self._to_feature_matrix(x)

        # Scale features
        if is_raw_unscaled:
            x_scaled = self.scaler.transform(x_mat)
        else:
            x_scaled = x_mat

        if self.model_type == "neural_net":
            self.model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(x_scaled, dtype=torch.float32)
                logits = self.model(tensor_x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            probs = self.model.predict_proba(x_scaled)

        return probs[0] if is_single else probs

    def predict_with_details(self, x: Any) -> Dict[str, Any]:
        """
        Convenience method returning predicted class ID, tier name, and individual probabilities.
        Compatible with telemetry logging (p_small, p_medium, p_large).
        """
        probs = self.predict_proba(x)
        class_id = int(np.argmax(probs))
        p0, p1, p2 = float(probs[0]), float(probs[1]), float(probs[2])

        result = {
            "predicted_class": class_id,
            "tier_name": self.CLASS_NAMES[class_id],
            "p_small": p0,
            "p_medium": p1,
            "p_large": p2,
            "probabilities": {
                "small (0)": p0,
                "medium (1)": p1,
                "large (2)": p2
            }
        }
        if isinstance(x, str):
            result["query"] = x
        return result

    # =================================================================
    # 3. Save & Load Model Artifacts
    # =================================================================
    def save(self, filepath: str):
        """
        Saves model weights, feature scaler, and configuration to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        artifact = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "class_names": self.CLASS_NAMES,
            "feature_names": FEATURE_NAMES,
            "scaler": self.scaler,
            "metrics": self.metrics_
        }

        if self.model_type == "neural_net":
            artifact["state_dict"] = self.model.state_dict()
            torch.save(artifact, filepath)
        else:
            artifact["model"] = self.model
            joblib.dump(artifact, filepath)

        print(f"[+] Model successfully stored at: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> QueryRouter:
        """
        Loads saved model artifact from disk and returns a ready-to-use QueryRouter.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Try torch.load first, then joblib
        try:
            artifact = torch.load(filepath, map_location="cpu", weights_only=False)
        except Exception:
            artifact = joblib.load(filepath)

        router = cls(
            model_type=artifact["model_type"],
            input_dim=artifact["input_dim"],
            num_classes=artifact["num_classes"]
        )
        router.scaler = artifact["scaler"]
        router.metrics_ = artifact.get("metrics", {})

        if router.model_type == "neural_net":
            router.model.load_state_dict(artifact["state_dict"])
            router.model.eval()
        else:
            router.model = artifact["model"]

        print(f"[+] Loaded '{router.model_type}' model from: {filepath}")
        return router


# =====================================================================
# 4. Data Loading & Feature Caching Helpers
# =====================================================================
def load_and_featurize_data(
    csv_path: str = "cleaned_data.csv",
    cache_path: Optional[str] = "saved_models/features_cache.npz",
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads query dataset from CSV and extracts 40-dimensional features.
    If cache_path exists and force_recompute is False, loads instantly from cache.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found: {csv_path}")

    # Check cache first
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        data = np.load(cache_path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        df = pd.read_csv(csv_path)
        print(f"[+] Loaded cached feature representations from {cache_path} ({len(X)} samples).")
        return X, y, df

    print(f"[*] Reading and featurizing queries from {csv_path}...")
    df = pd.read_csv(csv_path)
    if "query" not in df.columns:
        raise KeyError(f"Expected column 'query' in {csv_path}, found {df.columns.tolist()}")

    # Determine class labels {0, 1, 2}
    if "class_label" in df.columns:
        y = df["class_label"].values.astype(np.int64)
    elif "model_tier" in df.columns:
        y = df["model_tier"].map(QueryRouter.TIER_TO_CLASS).values.astype(np.int64)
    else:
        raise KeyError(f"Neither 'class_label' nor 'model_tier' found in {csv_path}")

    # Extract 40-d features
    t0 = time.time()
    X = np.stack([featurize(q) for q in df["query"]]).astype(np.float32)
    elapsed = time.time() - t0
    print(f"[+] Extracted 40 features for {len(X)} queries in {elapsed:.2f}s.")

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"[+] Cached feature representations to {cache_path}")

    return X, y, df


# =====================================================================
# 5. Standalone High-Level Helper Functions
# =====================================================================
def train_and_store_from_csv(
    csv_path: str = "cleaned_data.csv",
    model_type: str = "neural_net",
    save_path: str = "saved_models/router_nn.pt",
    cache_path: str = "saved_models/features_cache.npz",
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 0.002
) -> QueryRouter:
    """
    Complete end-to-end function: reads cleaned_data.csv, featurizes, trains, evaluates,
    and stores the model artifact to disk.
    """
    X, y, _ = load_and_featurize_data(csv_path=csv_path, cache_path=cache_path)
    router = QueryRouter(model_type=model_type, input_dim=X.shape[1], num_classes=len(np.unique(y)))
    router.train(X, y, epochs=epochs, batch_size=batch_size, lr=lr)
    router.save(save_path)
    return router


def train_and_store_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "neural_net",
    save_path: str = "saved_models/router_nn.pt"
) -> QueryRouter:
    """
    Trains, evaluates, and stores the model to disk from raw feature array X and labels y.
    """
    router = QueryRouter(model_type=model_type, input_dim=X.shape[1], num_classes=len(np.unique(y)))
    router.train(X, y)
    router.save(save_path)
    return router


def load_and_predict(sample: Any, model_path: str = "saved_models/router_nn.pt") -> Dict[str, Any]:
    """
    One-line function to load a stored model and classify a query (text string or 40*1 vector).
    """
    router = QueryRouter.load(model_path)
    return router.predict_with_details(sample)


# =====================================================================
# 6. Execution / Verification
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PRISMROUTE QUERY ROUTER CLASSIFIER (TRAIN & STORE)")
    print("=" * 70)

    csv_file = "cleaned_data.csv"
    nn_save_path = "saved_models/router_nn.pt"
    logistic_save_path = "saved_models/router_logistic.joblib"

    # 1. Train and Store PyTorch Neural Network from cleaned_data.csv
    print(f"\n[1] Training Neural Network on real data from '{csv_file}'...")
    nn_router = train_and_store_from_csv(
        csv_path=csv_file,
        model_type="neural_net",
        save_path=nn_save_path,
        epochs=35,
        batch_size=64,
        lr=0.002
    )

    # 2. Also train and store Logistic Regression baseline
    print(f"\n[2] Training Logistic Regression baseline...")
    log_router = train_and_store_from_csv(
        csv_path=csv_file,
        model_type="logistic_regression",
        save_path=logistic_save_path
    )

    # 3. Reload stored PyTorch model from disk to verify persistence
    print(f"\n[3] Reloading Neural Network model from disk ('{nn_save_path}')...")
    loaded_router = QueryRouter.load(nn_save_path)

    # 4. Test inference on realistic sample queries
    print("\n[4] Testing inference on representative queries:")
    sample_queries = [
        "What is 25 * 4?",
        "Write a Python script to parse a CSV file and output the average of a specific column.",
        "Consider a topological space X. Prove that if X is compact and Hausdorff, then every closed subset of X is compact."
    ]

    for q in sample_queries:
        res = loaded_router.predict_with_details(q)
        print(f"\n  Query: {q}")
        print(f"  -> Predicted Tier: Class {res['predicted_class']} ({res['tier_name'].upper()})")
        print(f"  -> Probabilities:  Small={res['p_small']:.3f}, Medium={res['p_medium']:.3f}, Large={res['p_large']:.3f}")

    # 5. Also verify backward compatibility with raw 40x1 feature vector
    print("\n[5] Testing backward compatibility with raw 40x1 numerical vector:")
    sample_40x1 = np.random.randn(40, 1).astype(np.float32)
    vec_res = loaded_router.predict_with_details(sample_40x1)
    print(f"  Input vector shape: {sample_40x1.shape}")
    print(f"  -> Predicted Tier: Class {vec_res['predicted_class']} ({vec_res['tier_name'].upper()})")
    print(f"  -> Probabilities:  {vec_res['probabilities']}")
    print("\n" + "=" * 70)
    print("ALL CLASSIFIER OPERATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 70)
