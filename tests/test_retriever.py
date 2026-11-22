import numpy as np
import pytest

from rag.chunking import Chunk
from rag.retriever import DenseIndex, Retriever, reciprocal_rank_fusion
from tests.helpers import HashingEmbedder

CHUNKS = [
    Chunk("Employees get 24 days of paid annual leave every year.", "leave.txt", 1),
    Chunk("Interns must work from the office during the first three months.", "remote.txt", 1),
    Chunk("Passwords must be at least 12 characters long.", "security.txt", 1),
]


def test_dense_index_uses_cosine_similarity():
    index = DenseIndex(np.array([[10.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), use_faiss=False)
    hits = index.search(np.array([1.0, 0.0]), k=3)
    assert [i for i, _ in hits] == [0, 2, 1]
    assert hits[0][1] == pytest.approx(1.0)  # magnitude does not matter after normalising


def test_dense_index_caps_k_at_corpus_size():
    index = DenseIndex(np.eye(2), use_faiss=False)
    assert len(index.search(np.array([1.0, 0.0]), k=10)) == 2


def test_rrf_rewards_documents_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]])
    assert [doc for doc, _ in fused] == [1, 3, 2]


@pytest.mark.parametrize("mode", Retriever.MODES)
def test_each_mode_finds_the_right_chunk(mode):
    retriever = Retriever(CHUNKS, HashingEmbedder(), mode=mode)
    top_chunk, _ = retriever.retrieve("how long must passwords be", k=2)[0]
    assert top_chunk.source == "security.txt"


def test_dense_modes_require_an_embedder():
    with pytest.raises(ValueError):
        Retriever(CHUNKS, None, mode="hybrid")
    with pytest.raises(ValueError):
        Retriever(CHUNKS, HashingEmbedder(), mode="sparse")
