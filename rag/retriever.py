"""Dense retrieval over chunk embeddings using cosine similarity."""
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .chunking import Chunk


def l2_normalize(vectors) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class DenseIndex:
    """Cosine-similarity search: vectors are L2-normalised, so the inner product
    equals cosine similarity. Uses FAISS IndexFlatIP when it is installed and
    falls back to an exact NumPy search otherwise."""

    def __init__(self, embeddings, use_faiss: Optional[bool] = None):
        self.vectors = l2_normalize(embeddings)
        self._faiss_index = None
        if use_faiss is not False:
            try:
                import faiss
            except ImportError:
                if use_faiss:
                    raise
            else:
                self._faiss_index = faiss.IndexFlatIP(self.vectors.shape[1])
                self._faiss_index.add(self.vectors)

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, query_vector, k: int) -> List[Tuple[int, float]]:
        query = l2_normalize(query_vector)
        k = min(k, len(self.vectors))
        if self._faiss_index is not None:
            scores, ids = self._faiss_index.search(query, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])]
        scores = self.vectors @ query[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.index = DenseIndex(embedder.encode([c.text for c in self.chunks]))

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        hits = self.index.search(self.embedder.encode([query]), k)
        return [(self.chunks[i], score) for i, score in hits]
