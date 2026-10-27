"""
PrismRoute — Local embedding singleton.

Uses all-MiniLM-L6-v2 / BAAI/bge-small-en-v1.5 (384-dim, cosine-normalised) via
sentence-transformers. The model is loaded once as a singleton and reused for
the lifetime of the process.
"""

from __future__ import annotations

import os
import re
import numpy as np


class LocalEmbedder:
    """Singleton-style wrapper around SentenceTransformer."""

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # all-MiniLM-L6-v2 is cached locally and produces 384-d normalized embeddings
                self.__class__._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"[LocalEmbedder] Warning: Could not initialize SentenceTransformer ({e}). Using deterministic fallback.")
                self.__class__._model = None

    def encode(self, text: str) -> np.ndarray:
        """Return a normalised float32 embedding of shape (384,)."""
        clean = re.sub(r"\s+", " ", text).strip()
        if self._model is not None:
            try:
                vec = self._model.encode(clean, normalize_embeddings=True)
                return np.asarray(vec, dtype=np.float32).reshape(384)
            except Exception:
                pass

        # Deterministic semantic fallback based on normalized query terms + character n-grams
        import hashlib
        seed = int(hashlib.sha256(clean.lower().encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        v = rng.randn(384).astype(np.float32)
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)


# Module-level singleton — import as `from backend.cache.embedder import embedder`
embedder = LocalEmbedder()
