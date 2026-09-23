"""Calibrated reranker invariants.

The guardrail under test: a reranker must never unilaterally own the final
ordering. When it is not discriminating, the fused order survives.
"""
from __future__ import annotations

from dialectica.core.types import Chunk, Document, Hit
from dialectica.retrieval.rerank import (
    CalibratedReranker,
    HeuristicReranker,
    z_scores,
)


def _hits():
    chunks = [
        Chunk.create(Document.create(t, title=str(i)), t, i, 0, len(t))
        for i, t in enumerate(
            [
                "电池续航时间是八小时，支持快充",
                "充电接口为 type-c 规格",
                "整机重量为一点二千克",
                "屏幕分辨率为 2560x1600",
                "内存配置为十六 GB",
                "硬盘容量为五百一十二 GB",
            ]
        )
    ]
    return [
        Hit(chunk=c, score=0.9 - 0.1 * i, components={"fused": 0.9 - 0.1 * i}) for i, c in enumerate(chunks)
    ]


def test_z_scores_zero_mean_unit_variance():
    z = z_scores([1.0, 2.0, 3.0, 4.0])
    assert abs(sum(z)) < 1e-9
    assert abs(sum(v * v for v in z) / len(z) - 1.0) < 1e-9


def test_z_scores_constant_input_is_all_zero():
    assert z_scores([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_output_is_subset_and_never_drops_fused_top1():
    """The floor may prune, but the fused top-1 always survives."""
    hits = _hits()
    rr = CalibratedReranker(HeuristicReranker().score, top_n=10)
    out = rr.rerank("电池续航", hits)
    ids = {h.span_id for h in out}
    assert ids <= {h.span_id for h in hits}
    assert hits[0].span_id in ids


def test_constant_scorer_preserves_fused_order():
    """No discrimination -> the reranker must not reorder anything."""
    hits = _hits()
    rr = CalibratedReranker(lambda q, hs: [0.3] * len(hs), top_n=10)
    out = rr.rerank("电池续航", hits)
    assert [h.span_id for h in out] == [h.span_id for h in hits]
    assert rr.last_diagnostics["strong_signal"] == 0.0


def test_adversarial_scorer_cannot_bury_the_best_candidate():
    """A reranker that hates the correct answer must not win outright.

    With alpha=0.5 and no clear margin, the fused order survives; the
    adversarial signal only tilts the ordering, it does not invert it.
    """
    hits = _hits()
    correct = hits[0].span_id

    def adversarial(_q, hs):
        return [0.0 if h.span_id == correct else 1.0 for h in hs]

    rr = CalibratedReranker(adversarial, alpha=0.5, top_n=10)
    out = rr.rerank("电池续航", hits)
    assert out[0].span_id == correct


def test_strong_signal_is_respected():
    hits = _hits()

    def strong(_q, hs):
        return [1.0 if h.span_id == hits[2].span_id else 0.0 for h in hs]

    rr = CalibratedReranker(strong, alpha=0.9, min_margin=0.15, top_n=10)
    out = rr.rerank("任意查询", hits)
    assert out[0].span_id == hits[2].span_id


def test_floor_keeps_at_least_half_on_large_sets():
    hits = _hits()

    def skewed(_q, hs):
        return [float(len(hs) - i) for i in range(len(hs))]

    rr = CalibratedReranker(skewed, top_n=10)
    out = rr.rerank("q", hits)
    assert len(out) >= max(2, (len(hits) + 1) // 2)


def test_small_candidate_set_keeps_everything():
    hits = _hits()[:2]
    rr = CalibratedReranker(lambda q, hs: [1.0, 0.0], top_n=10)
    out = rr.rerank("q", hits)
    assert len(out) == 2


def test_heuristic_reranker_prefers_overlap():
    hits = _hits()
    rr = CalibratedReranker(HeuristicReranker().score, top_n=10)
    out = rr.rerank("整机重量是多少", hits)
    assert "重量" in out[0].text
