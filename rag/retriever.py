"""Dense retrieval over chunk embeddings with FAISS."""
from typing import List, Sequence, Tuple

from .chunking import Chunk


class DenseRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder):
        import faiss

        self.chunks = list(chunks)
        self.embedder = embedder
        embeddings = embedder.encode([c.text for c in self.chunks])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 2) -> List[Tuple[Chunk, float]]:
        query_vector = self.embedder.encode([query])
        distances, ids = self.index.search(query_vector, min(k, len(self.chunks)))
        return [(self.chunks[i], float(d)) for i, d in zip(ids[0], distances[0])]
