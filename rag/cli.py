"""Command-line chatbot: python -m rag <document> [options]."""
import argparse
import sys

from .chunking import chunk_pages
from .embeddings import SentenceTransformerEmbedder
from .generator import FlanT5Generator
from .loader import load_file
from .pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Q&A over a local PDF/TXT document (RAG).")
    parser.add_argument("path", help="Path to a .pdf or .txt file")
    parser.add_argument("--top-k", type=int, default=2, help="Chunks passed to the model")
    parser.add_argument("--chunk-size", type=int, default=300, help="Words per chunk")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--llm", default="google/flan-t5-base")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    pages = load_file(args.path)
    chunks = chunk_pages(pages, args.chunk_size)
    if not chunks:
        sys.exit("No text could be extracted from the document.")
    print(f"Split {args.path} into {len(chunks)} chunks. Loading models...")

    rag = RAGPipeline(
        chunks,
        SentenceTransformerEmbedder(args.embed_model),
        FlanT5Generator(args.llm),
        top_k=args.top_k,
    )

    if args.question:
        print("\nAnswer:", rag.ask(args.question).text)
        return

    print("Ready. Ask questions about the document (type 'exit' to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("Answer:", rag.ask(question).text)


if __name__ == "__main__":
    main()
