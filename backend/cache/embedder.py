"""
PrismRoute — Local embedding singleton.

Uses BAAI/bge-small-en-v1.5 (384-dim, cosine-normalised) via
sentence-transformers.  The model is loaded once and reused for
the lifetime of the process.
"""

import os
import re
import numpy as np

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Singleton-style wrapper around SentenceTransformer."""

    _instance = None
    _model: SentenceTransformer | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            try:
                self.__class__._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                try:
                    self.__class__._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
                except Exception:
                    self.__class__._model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def encode(self, text: str) -> np.ndarray:
        """Return a normalised float32 embedding of shape (384,)."""
        clean = re.sub(r"\s+", " ", text).strip()
        vec = self._model.encode(clean, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32).reshape(384)


# Module-level singleton — import as `from backend.cache.embedder import embedder`
embedder = LocalEmbedder()
