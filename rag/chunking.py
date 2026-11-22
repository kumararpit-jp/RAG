"""Split page text into word chunks that keep their source metadata."""
from dataclasses import dataclass
from typing import Iterable, List

from .loader import Page


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def split_words(text: str, chunk_size: int = 300) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def chunk_pages(pages: Iterable[Page], chunk_size: int = 300) -> List[Chunk]:
    return [
        Chunk(text, page.source, page.page)
        for page in pages
        for text in split_words(page.text, chunk_size)
    ]
