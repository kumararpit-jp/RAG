"""Split page text into overlapping word chunks that keep their source metadata."""
from dataclasses import dataclass
from typing import Iterable, List

from .loader import Page


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def split_words(text: str, chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """Sliding window over words; consecutive chunks share `overlap` words so a
    sentence cut at a boundary still appears whole in one of them."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_pages(pages: Iterable[Page], chunk_size: int = 200, overlap: int = 40) -> List[Chunk]:
    return [
        Chunk(text, page.source, page.page)
        for page in pages
        for text in split_words(page.text, chunk_size, overlap)
    ]
