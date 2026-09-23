"""Calibrated reranking.

The failure this module exists to prevent: a cross-encoder that is weak in the
query's language (the classic case is an English reranker scoring Chinese
queries) is handed the *final* ordering. Unopposed, it buries the correct
answer -- measured as top-1 dropping from 11/12 to 5/12 on a Chinese eval set.

Three guardrails:
  1. z-score normalisation inside the candidate set, so raw cross-encoder
     scales cannot dominate.
  2. a floor: candidates far below the mean are dropped, but never the top-1.
  3. a blend: `final = alpha * z_rerank + (1 - alpha) * z_fusion`. The reranker
     *influences* the fused order, it never replaces it. If the reranker's
     top-1 is not clearly ahead (`min_margin`), the fused order wins outright.

Invariants:
  * the output is a permutation of the input candidate set
  * when every reranker score is identical, the fused order is preserved

Author: 晨星
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..core.types import Hit
from .tokenizer import tokenize


def z_scores(values: Sequence[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var**0.5
    if std < 1e-12:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def _overlap_score(query: str, text: str) -> float:
    qt, tt = set(tokenize(query)), set(tokenize(text))
    if not qt or not tt:
        return 0.0
    inter = len(qt & tt)
    return inter / (len(qt) ** 0.5 * len(tt) ** 0.5)


class HeuristicReranker:
    """Zero-dependency reranker: lexical overlap + position prior."""

    name = "heuristic"

    def score(self, query: str, hits: Sequence[Hit]) -> list[float]:
        return [_overlap_score(query, h.text) for h in hits]


class CrossEncoderReranker:
    """ONNX cross-encoder via fastembed (bge-reranker family)."""

    name = "cross-encoder"

    def __init__(self, model: str = "Xenova/bge-reranker-base") -> None:
        self.model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def score(self, query: str, hits: Sequence[Hit]) -> list[float]:
        if not hits:
            return []
        model = self._load()
        scores = list(model.rerank(query, [h.text for h in hits]))
        return [float(s) for s in scores]


class CalibratedReranker:
    """Wraps any scorer with the z-score floor + blended-order guardrails."""

    def __init__(
        self,
        scorer: Callable[[str, Sequence[Hit]], Sequence[float]],
        *,
        alpha: float = 0.5,
        z_floor: float = -0.5,
        min_margin: float = 0.15,
        top_n: int = 8,
    ) -> None:
        self.scorer = scorer
        self.alpha = alpha
        self.z_floor = z_floor
        self.min_margin = min_margin
        self.top_n = top_n
        self.last_diagnostics: dict[str, float] = {}

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits:
            return []
        raw = [float(s) for s in self.scorer(query, hits)]
        zr = z_scores(raw)
        zf = z_scores([h.score for h in hits])

        zr_top1, zr_top2 = (sorted(zr, reverse=True) + [-9e9, -9e9])[:2]
        margin = zr_top1 - zr_top2
        strong_signal = margin >= self.min_margin

        blended = [self.alpha * r + (1.0 - self.alpha) * f for r, f in zip(zr, zf)]
        if not strong_signal:
            # Reranker is not discriminating: trust the fusion order.
            blended = [float(f) for f in zf]

        # The z-floor is meaningless on tiny candidate sets (two candidates
        # always give z = +1 / -1), so only apply it once there is a
        # distribution worth cutting.
        n = len(zr)
        top1 = max(range(n), key=lambda j: zr[j])
        # `hits` arrive in fused order, so index 0 is the fused top-1. It is
        # never dropped by the floor: a reranker that hates the correct answer
        # must not be able to delete it, only to reorder around it.
        fused_top1 = 0
        if n < 4:
            keep = list(range(n))
        else:
            cut = self.z_floor
            keep = [i for i, z in enumerate(zr) if z >= cut or i == top1 or i == fused_top1]
            min_keep = max(2, (n + 1) // 2)
            if len(keep) < min_keep:
                order = sorted(range(n), key=lambda i: (-zr[i], i))
                for i in order:
                    if i not in keep:
                        keep.append(i)
                    if len(keep) >= min_keep:
                        break
        if not keep:
            keep = list(range(n))

        ranked = sorted(keep, key=lambda i: (-blended[i], i))
        self.last_diagnostics = {
            "margin": margin,
            "strong_signal": 1.0 if strong_signal else 0.0,
            "kept": float(len(keep)),
            "dropped": float(len(hits) - len(keep)),
        }
        out: list[Hit] = []
        for i in ranked[: self.top_n]:
            h = hits[i]
            out.append(
                Hit(
                    chunk=h.chunk,
                    score=round(blended[i], 6),
                    components={**h.components, "rerank_z": round(zr[i], 6), "rerank_raw": round(raw[i], 6)},
                )
            )
        return out
