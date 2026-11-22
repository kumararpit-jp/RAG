"""Measure retrieval quality on a labelled question set.

    python evaluate.py examples/docs examples/questions.json --mode all --k 3

questions.json is a list of {"question": ..., "source": <file name>} objects,
optionally with a "page" number. A question counts as a hit when one of the
top-k retrieved chunks comes from the expected source (and page, if given).
"""
import argparse
import json
import sys
from typing import Dict, List

from rag.chunking import chunk_pages
from rag.loader import load_documents
from rag.retriever import Retriever


def is_match(chunk, item: Dict) -> bool:
    return chunk.source == item["source"] and item.get("page", chunk.page) == chunk.page


def evaluate(retriever: Retriever, questions: List[Dict], k: int) -> Dict[str, float]:
    """Return hit-rate@k and mean reciprocal rank (MRR@k)."""
    hits = 0
    reciprocal_ranks = 0.0
    for item in questions:
        results = retriever.retrieve(item["question"], k)
        for rank, (chunk, _) in enumerate(results, start=1):
            if is_match(chunk, item):
                hits += 1
                reciprocal_ranks += 1.0 / rank
                break
    n = len(questions)
    return {"hit_rate": hits / n, "mrr": reciprocal_ranks / n}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with hit-rate@k and MRR.")
    parser.add_argument("docs", help="File or folder of .pdf/.txt documents")
    parser.add_argument("questions", help="JSON file with labelled questions")
    parser.add_argument("--mode", choices=Retriever.MODES + ("all",), default="all")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args(argv)

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    if not questions:
        sys.exit("The question file is empty.")
    chunks = chunk_pages(load_documents(args.docs), args.chunk_size, args.overlap)
    modes = Retriever.MODES if args.mode == "all" else (args.mode,)

    embedder = None
    if any(mode != "bm25" for mode in modes):
        try:
            from rag.embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(args.embed_model)
        except ImportError:
            print("sentence-transformers is not installed; only BM25 will be evaluated.")
            modes = ("bm25",)

    print(f"{len(questions)} questions, {len(chunks)} chunks, k={args.k}\n")
    print(f"{'mode':<8} {'hit-rate@k':>10} {'MRR':>8}")
    for mode in modes:
        metrics = evaluate(Retriever(chunks, embedder, mode=mode), questions, args.k)
        print(f"{mode:<8} {metrics['hit_rate']:>10.2f} {metrics['mrr']:>8.2f}")


if __name__ == "__main__":
    main()
