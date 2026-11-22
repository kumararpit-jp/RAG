import zlib

import numpy as np

from rag.bm25 import tokenize


class HashingEmbedder:
    """Deterministic bag-of-words embedder so tests run without downloading a model."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts):
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in tokenize(text):
                vectors[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        return vectors
