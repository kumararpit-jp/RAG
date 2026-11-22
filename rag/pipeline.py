"""Retrieve-then-generate pipeline."""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .chunking import Chunk
from .generator import build_prompt
from .retriever import Retriever


@dataclass
class Answer:
    text: str
    sources: List[Tuple[Chunk, float]]


class RAGPipeline:
    def __init__(self, chunks: Sequence[Chunk], embedder, generator,
                 mode: str = "hybrid", top_k: int = 3):
        self.retriever = Retriever(chunks, embedder, mode=mode)
        self.generator = generator
        self.top_k = top_k

    def ask(self, question: str) -> Answer:
        hits = self.retriever.retrieve(question, self.top_k)
        if not hits:
            return Answer("I don't know (no relevant passage found).", [])
        prompt = build_prompt(question, [chunk for chunk, _ in hits])
        return Answer(self.generator.generate(prompt), hits)
