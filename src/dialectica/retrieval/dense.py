"""Dense embedding + vector index.

Two embedders:
  * `HashingEmbedder` -- zero dependency, deterministic, offline (default)
  * `FastembedEmbedder` -- production ONNX embeddings (BGE / bge-small-zh)

Two indexes:
  * `MemoryIndex` -- pure-python cosine scan (default)
  * `FaissIndex` -- faiss-cpu IndexFlatIP for large corpora

Author: 晨星
"""
from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Sequence

from ..core.errors import ProviderUnavailable
from ..core.types import Chunk
from .tokenizer import tokenize


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def l2_normalize(vec: list[float]) -> list[float]:
    n = norm(vec)
    if n == 0.0:
        return vec
    return [v / n for v in vec]


class HashingEmbedder:
    """Deterministic feature-hashing embedder (the "hashing trick") with
    optional corpus IDF weighting.

    Uses `zlib.crc32` rather than `hash()` so vectors are stable across
    processes (PYTHONHASHSEED randomisation would otherwise break
    reproducibility of every downstream test).

    `fit()` is optional: without it every feature carries weight 1. With it,
    features are weighted by the same Robertson IDF the lexical index uses,
    which stops ubiquitous terms from dominating the projection.

    Invariants:
      * output is L2-normalised (||v|| == 1) for any non-empty text
      * identical text -> identical vector
      * sublinear TF weighting, so term repetition cannot dominate
    """

    # 2048 was picked by a grid search over dim x bigrams x idf on a mixed
    # CN/EN ops corpus (scripts/_tune.py): top1 4/6, MRR 0.806. Below 512 the
    # hash collisions dominate and ranking collapses.
    def __init__(self, dim: int = 2048, bigrams: bool = True) -> None:
        self.dim = dim
        self.bigrams = bigrams
        self._idf: dict[str, float] = {}
        self._n_docs = 0

    def fit(self, texts: Sequence[str]) -> "HashingEmbedder":
        """Compute Robertson IDF over a corpus. Idempotent-safe: refit replaces."""
        df: Counter[str] = Counter()
        n = 0
        for t in texts:
            n += 1
            for tok in set(self._features(t)):
                df[tok] += 1
        self._n_docs = n
        self._idf = {tok: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for tok, d in df.items()}
        return self

    def _weight(self, tok: str) -> float:
        return self._idf.get(tok, math.log(1.0 + (self._n_docs + 0.5) / 0.5) if self._n_docs else 1.0)

    def _features(self, text: str) -> Counter[str]:
        toks = tokenize(text)
        if self.bigrams:
            toks = toks + [f"{a}#{b}" for a, b in zip(toks, toks[1:])]
        return Counter(toks)

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        feats = self._features(text)
        for tok, count in feats.items():
            h = zlib.crc32(tok.encode("utf-8"))
            idx = h % self.dim
            sign = 1.0 if zlib.crc32(tok.encode("utf-8") + b"|s") & 1 else -1.0
            vec[idx] += sign * self._weight(tok) * (1.0 + math.log(count))
        return l2_normalize(vec)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class FastembedEmbedder:
    """Production ONNX embeddings via Qdrant fastembed.

    Lazy: import and model download only happen on first use, so importing this
    module never breaks the offline path.
    """

    def __init__(self, model: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.model_name = model
        self.dim = 512 if "small" in model else 1024
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ProviderUnavailable(f"fastembed not installed: {exc}") from exc
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        out = [list(map(float, v)) for v in model.embed(list(texts))]
        if not out:
            return []
        self.dim = len(out[0])
        return out


class MemoryIndex:
    """Brute-force cosine index. Exact by construction."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vecs: list[list[float]] = []

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for c, v in zip(chunks, vectors):
            self._ids.append(c.span_id)
            self._vecs.append(l2_normalize(list(v)))

    def search(self, vector: Sequence[float], top_k: int) -> list[tuple[str, float]]:
        q = l2_normalize(list(vector))
        scored = [(sid, dot(q, v)) for sid, v in zip(self._ids, self._vecs)]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:top_k]

    def clear(self) -> None:
        self._ids.clear()
        self._vecs.clear()

    def __len__(self) -> int:
        return len(self._ids)


class FaissIndex:
    """faiss-cpu flat inner-product index (cosine, since vectors are normalised)."""

    def __init__(self, dim: int) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ProviderUnavailable(f"faiss-cpu not installed: {exc}") from exc
        self._faiss = faiss
        self._np = np
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        arr = self._np.asarray(
            [l2_normalize(list(v)) for v in vectors], dtype="float32"
        ).reshape(len(vectors), self.dim)
        self._index.add(arr)
        self._ids.extend(c.span_id for c in chunks)

    def search(self, vector: Sequence[float], top_k: int) -> list[tuple[str, float]]:
        q = self._np.asarray([l2_normalize(list(vector))], dtype="float32")
        scores, idxs = self._index.search(q, min(top_k, max(1, len(self._ids))))
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            out.append((self._ids[int(i)], float(score)))
        return out

    def clear(self) -> None:
        self._index.reset()
        self._ids.clear()

    def __len__(self) -> int:
        return len(self._ids)


def build_embedder(backend: str, model: str = "BAAI/bge-small-zh-v1.5", dim: int = 384):
    if backend == "hashing":
        return HashingEmbedder(dim=dim)
    if backend in {"fastembed", "onnx"}:
        return FastembedEmbedder(model=model)
    raise ProviderUnavailable(f"unknown embedding backend: {backend}")


def build_index(backend: str, dim: int):
    if backend == "memory":
        return MemoryIndex()
    if backend == "faiss":
        return FaissIndex(dim)
    raise ProviderUnavailable(f"unknown vector backend: {backend}")
