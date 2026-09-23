"""BM25 with the Robertson (non-negative) IDF.

Why not `rank_bm25`: its IDF is `ln((N - n + 0.5) / (n + 0.5) + 1)` under some
parameterisations and can go *negative* when a term appears in more than half
the corpus; the `epsilon * average_idf` floor then distorts small corpora so
badly that the correct document ranks last (measured: scores [-0.366, -0.349]
on a 2-document corpus).

Robertson IDF `ln(1 + (N - n + 0.5)/(n + 0.5))` is provably >= 0 for all
N >= 1, 1 <= n <= N. That property is asserted in the tests.

Author: 晨星
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from ..core.types import Chunk
from .tokenizer import tokenize

_K1 = 1.5
_B = 0.75


class BM25:
    """In-memory BM25 index."""

    def __init__(self, k1: float = _K1, b: float = _B) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._len: list[int] = []
        self._avg_len = 0.0

    # -- index construction ------------------------------------------------
    def add(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            tf = Counter(tokenize(c.text))
            self._chunks.append(c)
            self._tf.append(tf)
            self._len.append(sum(tf.values()))
            for term in tf:
                self._df[term] += 1
        self._recompute_avg()

    def _recompute_avg(self) -> None:
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0

    def clear(self) -> None:
        self._chunks.clear()
        self._tf.clear()
        self._df.clear()
        self._len.clear()
        self._avg_len = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    # -- scoring -----------------------------------------------------------
    def idf(self, term: str) -> float:
        """Robertson IDF, always >= 0."""
        n = len(self._chunks)
        if n == 0:
            return 0.0
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, idx: int) -> float:
        tf = self._tf[idx]
        dl = self._len[idx] or 1
        total = 0.0
        for term in set(tokenize(query)):
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1.0 - self.b + self.b * dl / (self._avg_len or 1.0))
            total += self.idf(term) * (f * (self.k1 + 1.0)) / (denom or 1.0)
        return total

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scored = [(self._chunks[i].span_id, self.score(query, i)) for i in range(len(self._chunks))]
        scored = [s for s in scored if s[1] > 0.0]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:top_k]
