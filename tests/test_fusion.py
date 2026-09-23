"""Adaptive fusion invariants."""
from __future__ import annotations

from dialectica.retrieval.fusion import (
    derive_weights,
    lexical_overlap,
    score_margin,
    shannon_entropy,
    weighted_rrf,
)


def test_entropy_bounds():
    assert 0.0 <= shannon_entropy([1, 1, 1, 1]) <= 1.0
    assert shannon_entropy([1, 1, 1, 1]) == 1.0  # uniform -> max
    assert shannon_entropy([]) == 0.0
    assert shannon_entropy([5]) == 0.0


def test_entropy_degenerate_is_zero():
    assert shannon_entropy([0, 0, 0]) == 0.0


def test_margin_bounds():
    assert score_margin([]) == 0.0
    assert score_margin([0.9]) == 0.0
    assert score_margin([0.9, 0.1]) == 1.0
    assert 0.0 <= score_margin([0.5, 0.45]) <= 1.0


def test_lexical_overlap_full_and_zero():
    assert lexical_overlap("电池续航", ["电池续航是八小时"]) == 1.0
    assert lexical_overlap("电池续航", ["鲸鱼迁徙"]) == 0.0
    assert lexical_overlap("", ["x"]) == 0.0


def test_weights_sum_to_one():
    s = derive_weights("电池续航", [0.9, 0.5, 0.2], [1.0, 0.9, 0.8], ["电池续航八小时"])
    assert abs(s.w_dense + s.w_sparse - 1.0) < 1e-9


def test_weights_clamped():
    for dense in ([0.99, 0.01, 0.01], [0.2, 0.2, 0.2]):
        for sparse in ([1.0, 1.0, 1.0], [0.1, 0.1, 0.1]):
            s = derive_weights("q", dense, sparse, ["q text"])
            assert 0.10 <= s.w_dense <= 0.90
            assert 0.10 <= s.w_sparse <= 0.90


def test_lexical_query_boosts_sparse():
    """A query whose tokens literally occur in the corpus leans on BM25."""
    lexical = derive_weights("site-packages 写入", [0.4, 0.39], [0.9, 0.2], ["site-packages 写入被拦截"])
    semantic = derive_weights("为什么装不上", [0.8, 0.3], [0.9, 0.2], ["沙箱守卫会拦截写入"])
    assert lexical.w_sparse > semantic.w_sparse


def test_non_adaptive_is_fifty_fifty():
    s = derive_weights("q", [0.9, 0.1], [1.0, 0.5], ["q"], adaptive=False)
    assert s.w_dense == 0.5 and s.w_sparse == 0.5


def test_rrf_is_permutation_of_union():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 2.0), ("d", 1.5), ("a", 1.0)]
    out = weighted_rrf(dense, sparse, w_dense=0.5, w_sparse=0.5, top_k=10)
    assert {sid for sid, _ in out} == {"a", "b", "c", "d"}


def test_rrf_scores_monotonic():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("b", 2.0), ("d", 1.5), ("a", 1.0)]
    out = weighted_rrf(dense, sparse, top_k=4)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)


def test_rrf_weights_move_ranking():
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("b", 2.0), ("a", 1.0)]
    dense_heavy = weighted_rrf(dense, sparse, w_dense=0.9, w_sparse=0.1, top_k=2)
    sparse_heavy = weighted_rrf(dense, sparse, w_dense=0.1, w_sparse=0.9, top_k=2)
    assert dense_heavy[0][0] == "a"
    assert sparse_heavy[0][0] == "b"
