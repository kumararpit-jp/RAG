import pytest

from rag.chunking import chunk_pages, split_words
from rag.loader import Page


def test_chunks_overlap_by_requested_words():
    text = " ".join(str(i) for i in range(10))
    assert split_words(text, chunk_size=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_short_and_empty_text():
    assert split_words("a b c", chunk_size=5, overlap=2) == ["a b c"]
    assert split_words("   ", chunk_size=5, overlap=2) == []


def test_every_word_is_covered():
    words = [f"w{i}" for i in range(103)]
    chunks = split_words(" ".join(words), chunk_size=20, overlap=5)
    covered = {w for chunk in chunks for w in chunk.split()}
    assert covered == set(words)
    assert all(len(c.split()) <= 20 for c in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1)])
def test_invalid_arguments(size, overlap):
    with pytest.raises(ValueError):
        split_words("a b c", size, overlap)


def test_chunks_keep_source_and_page():
    pages = [Page("a.pdf", 1, "one two three"), Page("a.pdf", 2, "four five")]
    chunks = chunk_pages(pages, chunk_size=2, overlap=0)
    assert [(c.text, c.page) for c in chunks] == [("one two", 1), ("three", 1), ("four five", 2)]
    assert {c.source for c in chunks} == {"a.pdf"}
