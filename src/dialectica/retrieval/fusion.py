"""Adaptive hybrid fusion (weighted Reciprocal Rank Fusion).

Fixed-alpha hybrid search fails half the time: a *lexical* query ("Errno 10054
WinError") wants BM25, a *semantic* query ("为什么服务会自己停掉") wants dense.
A constant alpha cannot serve both.

This module reads three difficulty signals off the two result lists and derives
the weights per query:

  entropy   -- normalised Shannon entropy of the BM25 top-k score distribution.
               Uniform scores mean BM25 is not discriminating -> downweight it.
  margin    -- dense top-1 minus top-2 cosine. A large gap means the dense side
               is confident -> downweight the lexical side.
  lexical   -- fraction of query tokens that literally occur in the dense top-k
               passages. High overlap means the query is lexical -> boost BM25.

Invariants:
  * w_dense + w_sparse == 1.0, both clamped to [0.1, 0.9]
  * output is a subset of the union of both inputs (no hallucinated ids)
  * deterministic

Author: 晨星
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .tokenizer import tokenize

_ENT_WEIGHT = 0.20
_LEX_WEIGHT = 0.25
_MARGIN_WEIGHT = 0.15
_MIN_W = 0.10
_MAX_W = 0.90


@dataclass(frozen=True)
class FusionSignals:
    entropy: float = 0.0
    margin: float = 0.0
    lexical: float = 0.0
    w_dense: float = 0.5
    w_sparse: float = 0.5


def shannon_entropy(values: Sequence[float]) -> float:
    """Normalised entropy in [0, 1]. Empty / degenerate input -> 0."""
    positive = [v for v in values if v > 0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    if total <= 0:
        return 0.0
    h = -sum((v / total) * math.log(v / total) for v in positive)
    return float(h / math.log(len(positive)))


def score_margin(scores: Sequence[float]) -> float:
    """Top-1 minus top-2, scaled into [0, 1] by a 0.20 reference gap."""
    if len(scores) < 2:
        return 0.0
    ordered = sorted(scores, reverse=True)
    return float(min(1.0, max(0.0, ordered[0] - ordered[1]) / 0.20))


def lexical_overlap(query: str, texts: Sequence[str]) -> float:
    qt = set(tokenize(query))
    if not qt or not texts:
        return 0.0
    blob = set()
    for t in texts:
        blob |= set(tokenize(t))
    return len(qt & blob) / len(qt)


def derive_weights(
    query: str,
    dense_scores: Sequence[float],
    sparse_scores: Sequence[float],
    dense_texts: Sequence[str],
    *,
    adaptive: bool = True,
) -> FusionSignals:
    if not adaptive:
        return FusionSignals(w_dense=0.5, w_sparse=0.5)
    entropy = shannon_entropy(sparse_scores)
    margin = score_margin(dense_scores)
    lexical = lexical_overlap(query, dense_texts)
    w_sparse = 0.5 + _LEX_WEIGHT * lexical - _ENT_WEIGHT * entropy - _MARGIN_WEIGHT * margin
    w_sparse = min(_MAX_W, max(_MIN_W, w_sparse))
    return FusionSignals(
        entropy=entropy,
        margin=margin,
        lexical=lexical,
        w_dense=round(1.0 - w_sparse, 6),
        w_sparse=round(w_sparse, 6),
    )


def weighted_rrf(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float]],
    *,
    k: int = 60,
    w_dense: float = 0.5,
    w_sparse: float = 0.5,
    top_k: int = 8,
) -> list[tuple[str, float]]:
    """Fuse two ranked lists with weighted RRF.

    `score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))`
    """
    fused: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for rank, (sid, _s) in enumerate(dense, start=1):
        fused[sid] = fused.get(sid, 0.0) + w_dense / (k + rank)
        components.setdefault(sid, {"d": 0.0, "s": 0.0})["d"] = w_dense / (k + rank)
    for rank, (sid, _s) in enumerate(sparse, start=1):
        fused[sid] = fused.get(sid, 0.0) + w_sparse / (k + rank)
        components.setdefault(sid, {"d": 0.0, "s": 0.0})["s"] = w_sparse / (k + rank)
    ordered = sorted(fused.items(), key=lambda t: (-t[1], t[0]))
    return ordered[:top_k]
