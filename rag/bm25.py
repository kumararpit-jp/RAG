"""Okapi BM25 keyword ranking, implemented without external dependencies."""
import math
import re
from collections import Counter
from typing import List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by can did do does for from has have how i in is it its "
    "of on or that the this to was were what when where which who why will with you".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(tokenize(doc)) for doc in documents]
        self.doc_len = [sum(freqs.values()) for freqs in self.doc_freqs]
        self.n_docs = len(documents)
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0

        df = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        # Lucene-style idf: always positive, even for terms in most documents
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    def scores(self, query: str) -> List[float]:
        terms = [t for t in tokenize(query) if t in self.idf]
        result = []
        for freqs, length in zip(self.doc_freqs, self.doc_len):
            norm = self.k1 * (1 - self.b + self.b * length / self.avgdl) if self.avgdl else self.k1
            score = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            result.append(score)
        return result

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Top-k (doc index, score) pairs; documents with no matching term are skipped."""
        scores = self.scores(query)
        ranked = sorted(range(self.n_docs), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]
