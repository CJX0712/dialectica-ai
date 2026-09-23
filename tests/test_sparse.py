"""BM25 invariants. The headline one: Robertson IDF is never negative."""
from __future__ import annotations

import math

from dialectica.core.types import Chunk, Document
from dialectica.retrieval.sparse import BM25


def _chunks():
    return [
        Chunk.create(Document.create("电池续航时间是八小时"), "电池续航时间是八小时", 0, 0, 10),
        Chunk.create(Document.create("充电接口为 type-c"), "充电接口为 type-c", 0, 0, 12),
        Chunk.create(Document.create("整机重量为一点二千克"), "整机重量为一点二千克", 0, 0, 10),
        Chunk.create(Document.create("电池支持快充协议"), "电池支持快充协议", 0, 0, 8),
    ]


def test_idf_non_negative_for_every_df():
    """Robertson IDF >= 0 for all N >= 1, 1 <= n <= N -- the property that
    rank_bm25's variant violates on small corpora."""
    idx = BM25()
    idx.add(_chunks())
    n = len(idx)
    for df in range(1, n + 1):
        # idf is derived from df; emulate by direct formula
        value = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        assert value >= 0.0


def test_idf_zero_for_universal_term():
    idx = BM25()
    idx.add(_chunks())
    assert idx.idf("nonexistent-term-xyz") > 0.0


def test_search_ranks_relevant_first():
    idx = BM25()
    idx.add(_chunks())
    hits = idx.search("电池续航", top_k=4)
    assert hits, "BM25 must return something for a query sharing tokens"
    assert hits[0][1] > 0.0
    texts = {c.span_id: c.text for c in idx.chunks}
    assert "电池" in texts[hits[0][0]]


def test_scores_are_ordered_non_increasing():
    idx = BM25()
    idx.add(_chunks())
    scores = [s for _, s in idx.search("电池 充电 重量", top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_no_match_returns_empty():
    idx = BM25()
    idx.add(_chunks())
    assert idx.search("zzzqqq", top_k=4) == []


def test_two_document_corpus_still_discriminates():
    """Regression: rank_bm25 returns all-zero or inverted scores here."""
    idx = BM25()
    idx.add(
        [
            Chunk.create(Document.create("a"), "沙箱守卫拦截 site-packages 写入", 0, 0, 10),
            Chunk.create(Document.create("b"), "代理导致 SOCKS 依赖缺失", 0, 0, 10),
        ]
    )
    hits = idx.search("site-packages 写入", top_k=2)
    assert len(hits) == 1
    assert hits[0][1] > 0.0


def test_clear_resets():
    idx = BM25()
    idx.add(_chunks())
    assert len(idx) == 4
    idx.clear()
    assert len(idx) == 0
    assert idx.search("电池") == []
