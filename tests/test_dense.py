"""Embedding + vector index invariants."""
from __future__ import annotations

import math

from dialectica.core.types import Chunk, Document
from dialectica.retrieval.dense import (
    HashingEmbedder,
    MemoryIndex,
    cosine,
    l2_normalize,
)


def test_unit_length():
    emb = HashingEmbedder(dim=96)
    for text in ["电池续航", "hello world", "混合检索 hybrid"]:
        v = emb.embed_one(text)
        n = math.sqrt(sum(x * x for x in v))
        assert abs(n - 1.0) < 1e-9, f"not normalised for {text!r}: {n}"


def test_empty_text_yields_zero_vector():
    """Nothing to hash -> all zeros. Callers must not assume unit length."""
    emb = HashingEmbedder(dim=96)
    assert all(x == 0.0 for x in emb.embed_one(""))


def test_deterministic_across_instances():
    a, b = HashingEmbedder(dim=96), HashingEmbedder(dim=96)
    assert a.embed_one("沙箱守卫 site-packages") == b.embed_one("沙箱守卫 site-packages")


def test_self_similarity_is_one():
    emb = HashingEmbedder(dim=96)
    v = emb.embed_one("retrieval augmented generation")
    assert abs(cosine(v, v) - 1.0) < 1e-9


def test_related_text_more_similar_than_unrelated():
    emb = HashingEmbedder(dim=256)
    q = emb.embed_one("电池续航时间")
    related = emb.embed_one("电池续航是八小时")
    unrelated = emb.embed_one("鲸鱼在深海里迁徙")
    assert cosine(q, related) > cosine(q, unrelated)


def test_l2_normalize_zero_vector_is_safe():
    assert l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_memory_index_exact_top1():
    emb = HashingEmbedder(dim=128)
    idx = MemoryIndex()
    chunks = [
        Chunk.create(Document.create(t, title=str(i)), t, i, 0, len(t))
        for i, t in enumerate(["电池续航八小时", "充电接口是 type-c", "整机重量一点二千克"])
    ]
    idx.add(chunks, emb.embed([c.text for c in chunks]))
    results = idx.search(emb.embed_one("电池续航"), top_k=3)
    assert len(results) == 3
    assert results[0][0] == chunks[0].span_id
    assert results[0][1] >= results[1][1]


def test_memory_index_clear():
    emb = HashingEmbedder(dim=32)
    idx = MemoryIndex()
    chunks = [Chunk.create(Document.create("x"), "x", 0, 0, 1)]
    idx.add(chunks, emb.embed(["x"]))
    assert len(idx) == 1
    idx.clear()
    assert len(idx) == 0


def test_faiss_index_matches_memory_index_ordering():
    """Cross-implementation check: faiss and the pure-python index agree."""
    pytest = __import__("pytest")
    pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    from dialectica.retrieval.dense import FaissIndex

    emb = HashingEmbedder(dim=64)
    texts = ["电池续航八小时", "充电接口 type-c", "整机重量一点二千克", "屏幕分辨率很高"]
    chunks = [Chunk.create(Document.create(t, title=str(i)), t, i, 0, len(t)) for i, t in enumerate(texts)]
    vecs = emb.embed(texts)

    mem = MemoryIndex()
    mem.add(chunks, vecs)
    fai = FaissIndex(64)
    fai.add(chunks, vecs)

    q = emb.embed_one("电池续航")
    a = [sid for sid, _ in mem.search(q, 2)]
    b = [sid for sid, _ in fai.search(q, 2)]
    assert a == b
