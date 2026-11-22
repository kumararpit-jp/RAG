"""Dense, keyword (BM25) and hybrid retrieval over document chunks."""
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .bm25 import BM25
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


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    """Merge ranked id lists: score(d) = sum over lists of 1 / (k + rank of d).
    Uses ranks only, so BM25 and cosine scores never need to be put on one scale."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    """Retrieves chunks with dense (embedding) search, BM25 keyword search, or a
    hybrid of both fused with Reciprocal Rank Fusion."""

    MODES = ("dense", "bm25", "hybrid")

    def __init__(self, chunks: Sequence[Chunk], embedder=None, mode: str = "hybrid",
                 candidate_pool: int = 20):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if mode != "bm25" and embedder is None:
            raise ValueError(f"mode '{mode}' needs an embedder")
        self.chunks = list(chunks)
        self.embedder = embedder
        self.mode = mode
        self.candidate_pool = candidate_pool
        texts = [chunk.text for chunk in self.chunks]
        self.dense = DenseIndex(embedder.encode(texts)) if mode != "bm25" else None
        self.bm25 = BM25(texts) if mode != "dense" else None

    def _dense_search(self, query: str, k: int):
        return self.dense.search(self.embedder.encode([query]), k)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.mode == "dense":
            hits = self._dense_search(query, k)
        elif self.mode == "bm25":
            hits = self.bm25.search(query, k)
        else:
            pool = max(k, self.candidate_pool)
            dense_ids = [i for i, _ in self._dense_search(query, pool)]
            keyword_ids = [i for i, _ in self.bm25.search(query, pool)]
            hits = reciprocal_rank_fusion([dense_ids, keyword_ids])[:k]
        return [(self.chunks[i], score) for i, score in hits]
