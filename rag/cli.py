"""Command-line chatbot: python -m rag <file-or-folder> [options]."""
import argparse
import sys

from .chunking import chunk_pages
from .embeddings import SentenceTransformerEmbedder
from .generator import FlanT5Generator
from .loader import load_documents
from .pipeline import Answer, RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Q&A over local PDF/TXT documents (RAG).")
    parser.add_argument("path", help="A .pdf/.txt file or a folder containing them")
    parser.add_argument("--top-k", type=int, default=2, help="Chunks passed to the model")
    parser.add_argument("--chunk-size", type=int, default=300, help="Words per chunk")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--llm", default="google/flan-t5-base")
    parser.add_argument("-q", "--question", help="Ask a single question and exit")
    return parser


def print_answer(answer: Answer) -> None:
    print("\nAnswer:", answer.text)
    print("Sources:")
    for chunk, score in answer.sources:
        print(f"  - {chunk.source} (page {chunk.page}), score {score:.3f}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    try:
        pages = load_documents(args.path)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))
    chunks = chunk_pages(pages, args.chunk_size)
    if not chunks:
        sys.exit("No text could be extracted from the documents.")
    n_files = len({page.source for page in pages})
    print(f"Loaded {n_files} file(s), {len(pages)} page(s) -> {len(chunks)} chunks. Loading models...")

    rag = RAGPipeline(
        chunks,
        SentenceTransformerEmbedder(args.embed_model),
        FlanT5Generator(args.llm),
        top_k=args.top_k,
    )

    if args.question:
        print_answer(rag.ask(args.question))
        return

    print("Ready. Ask questions about your documents (type 'exit' to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print_answer(rag.ask(question))


if __name__ == "__main__":
    main()
