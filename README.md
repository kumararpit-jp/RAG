# Offline Document Q&A Chatbot (RAG)

Ask questions about your own PDF and text files and get answers grounded in them. Everything runs locally: no API keys, and no document text leaves your machine. The embedding model and FLAN-T5 are downloaded from Hugging Face the first time you run it and are cached after that.

## Features

- **Multi-document indexing**: point it at one file or a whole folder of `.pdf` / `.txt` files (searched recursively). Every chunk keeps its source file and page number, and answers list the sources they used.
- **Overlapping chunks**: a sliding window of words (default 200 words with a 40-word overlap), so text cut at a chunk boundary still appears whole in one chunk.
- **Three retrieval modes**:
  - `dense`: Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) with cosine similarity. Vectors are L2-normalised and searched with a FAISS `IndexFlatIP`; if FAISS isn't installed, an exact NumPy search is used instead.
  - `bm25`: Okapi BM25 keyword ranking, written from scratch in `rag/bm25.py`.
  - `hybrid` (default): runs both and merges the two rankings with Reciprocal Rank Fusion. Keyword matches catch exact terms such as names, numbers and codes; embeddings catch paraphrases.
- **Grounded generation**: a local `google/flan-t5-base` model answers from the retrieved chunks only, and is told to say "I don't know" when the answer isn't there.
- **Retrieval evaluation**: `evaluate.py` reports hit-rate@k and MRR for each retrieval mode on a labelled question set.
- **Tests**: pytest unit tests for chunking, BM25, rank fusion, the retrievers, loading and evaluation. They use a small hashing embedder, so they run without downloading any model.

## How it works

```
PDF/TXT files ─► load pages (PyPDF2) ─► overlapping chunks (+ file, page)
                                              │
               ┌──────────────────────────────┴──────────────────────────┐
               ▼                                                         ▼
   embeddings + cosine search (FAISS)                         BM25 keyword search
               └──────────────► Reciprocal Rank Fusion ◄─────────────────┘
                                         │ top-k chunks
                                         ▼
                        FLAN-T5 (context-only prompt) ─► answer + sources
```

## Setup

```bash
git clone <this-repo-url>
cd RAG
pip install -r requirements.txt
```

## Usage

```bash
# interactive chat over a folder (hybrid retrieval)
python -m rag examples/docs

# one question, keyword-only retrieval, 5 chunks of context
python -m rag path/to/report.pdf --mode bm25 --top-k 5 -q "What was the total revenue?"
```

Options: `--mode {dense,bm25,hybrid}`, `--top-k`, `--chunk-size`, `--overlap`, `--embed-model`, `--llm`, `-q/--question`.

The notebook `rag.ipynb` walks through the same pipeline step by step.

## Evaluating retrieval

```bash
python evaluate.py examples/docs examples/questions.json --mode all --k 3
```

`questions.json` is a list of `{"question": ..., "source": "<file name>"}` objects; you can add `"page"` to require an exact page. A question counts as a hit if one of the top-k chunks comes from the expected source. The included example set is tiny, so treat it as a smoke test and build a question set from your own documents to compare modes properly.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project structure

```
rag/
├── loader.py      # PDF/TXT loading, one record per page, folder support
├── chunking.py    # overlapping word chunks with source metadata
├── embeddings.py  # Sentence-Transformer wrapper (normalised vectors)
├── bm25.py        # Okapi BM25 from scratch
├── retriever.py   # cosine index (FAISS / NumPy), RRF, dense/bm25/hybrid retriever
├── generator.py   # prompt building + FLAN-T5
├── pipeline.py    # retrieve-then-generate
└── cli.py         # command-line chatbot (python -m rag)
evaluate.py        # hit-rate@k / MRR evaluation
examples/          # sample documents and labelled questions
tests/             # pytest suite
rag.ipynb          # notebook walkthrough
```

## Limitations

- Scanned (image-only) PDFs have no extractable text; they would need OCR first.
- FLAN-T5 reads at most 512 tokens, so very long contexts are truncated. Keep `--top-k` × `--chunk-size` moderate.
- The index is built in memory each run and isn't saved to disk.
