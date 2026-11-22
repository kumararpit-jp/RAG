from pathlib import Path

import pytest

from evaluate import evaluate
from rag.chunking import Chunk
from rag.loader import load_documents
from rag.retriever import Retriever

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_loads_folder_recursively_and_ignores_other_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("alpha beta", encoding="utf-8")
    (tmp_path / "sub" / "a.txt").write_text("gamma", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  ", encoding="utf-8")

    pages = load_documents(tmp_path)
    assert [(p.source, p.text) for p in pages] == [("a.txt", "alpha beta"), ("sub/a.txt", "gamma")]


def test_missing_path_and_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)


def test_example_corpus_loads():
    sources = {p.source for p in load_documents(EXAMPLES / "docs")}
    assert sources == {"leave_policy.txt", "remote_work.txt", "it_security.txt"}


def test_evaluate_computes_hit_rate_and_mrr():
    chunks = [Chunk("cats purr", "cats.txt", 1), Chunk("dogs bark loudly", "dogs.txt", 1)]
    questions = [
        {"question": "why do cats purr", "source": "cats.txt"},  # rank 1
        {"question": "dogs bark", "source": "cats.txt"},         # not retrieved
    ]
    metrics = evaluate(Retriever(chunks, mode="bm25"), questions, k=2)
    assert metrics == {"hit_rate": 0.5, "mrr": 0.5}
