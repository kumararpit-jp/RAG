from rag.bm25 import BM25, tokenize

DOCS = [
    "The cat sat on the mat.",
    "Dogs and cats are common pets.",
    "Photons can become entangled in quantum experiments.",
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Cat is ON the Mat!") == ["cat", "mat"]


def test_most_relevant_document_ranks_first():
    bm25 = BM25(DOCS)
    assert bm25.search("cat mat", k=3)[0][0] == 0
    assert bm25.search("quantum photons", k=3)[0][0] == 2


def test_documents_without_matching_terms_are_skipped():
    bm25 = BM25(DOCS)
    assert bm25.search("zebra", k=3) == []
    assert [i for i, _ in bm25.search("photons", k=3)] == [2]


def test_rare_terms_weigh_more_than_common_ones():
    bm25 = BM25(["apple banana", "apple cherry", "apple durian"])
    assert bm25.idf["banana"] > bm25.idf["apple"] > 0


def test_empty_corpus():
    assert BM25([]).search("anything", k=3) == []
