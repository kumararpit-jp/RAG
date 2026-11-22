"""Sentence-Transformer embeddings (runs locally once the model is cached)."""
from typing import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors -> inner product = cosine
            show_progress_bar=len(texts) > 32,
        )
        return np.asarray(vectors, dtype="float32")
