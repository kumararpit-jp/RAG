"""Load text from local PDF and TXT files, one record per page."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


@dataclass
class Page:
    source: str  # file name the text came from
    page: int    # 1-based page number (always 1 for .txt files)
    text: str


def load_file(path, source: Optional[str] = None) -> List[Page]:
    path = Path(path)
    source = source or path.name
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            # extract_text() returns None for image-only pages
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(Page(source, number, text))
        return pages

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return [Page(source, 1, text)] if text else []

    raise ValueError(f"Unsupported file type '{path.suffix}'. Use .pdf or .txt")


def load_documents(path) -> List[Page]:
    """Load a single file, or every .pdf/.txt file under a folder (recursively)."""
    path = Path(path)
    if path.is_file():
        return load_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"No such file or folder: {path}")

    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No .pdf or .txt files found in {path}")

    pages: List[Page] = []
    for file in files:
        # relative path keeps same-named files in different subfolders apart
        pages.extend(load_file(file, source=file.relative_to(path).as_posix()))
    return pages
