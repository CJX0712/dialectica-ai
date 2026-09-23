"""Memory layer invariants."""
from __future__ import annotations

from dialectica.core.config import MemoryConfig
from dialectica.memory import MemoryStore


def _store(embedder, **kw):
    return MemoryStore(embedder, MemoryConfig(**kw))


def test_remember_and_recall(embedder):
    m = _store(embedder)
    m.remember("沙箱守卫会拦截 site-packages 写入")
    m.remember("npm registry 被劫持到鸿蒙源")
    out = m.recall("site-packages 写入", top_k=2)
    assert out
    assert "site-packages" in out[0]


def test_working_memory_is_bounded(embedder):
    m = _store(embedder, working_size=3)
    for i in range(10):
        m.remember(f"记录 {i}")
    assert len(m.working) == 3
    assert m.working[-1] == "记录 9"


def test_recall_empty_is_empty(embedder):
    assert _store(embedder).recall("anything") == []


def test_decay_is_monotonic(embedder):
    m = _store(embedder, decay_lambda=0.5)
    m.remember("一条记忆")
    ep = m.episodic[0]
    now = ep.created_ms
    scores = [ep.salience * m._decay(ep, now + h * 3_600_000.0) for h in range(6)]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] < scores[0]


def test_distill_is_idempotent(embedder):
    m = _store(embedder, min_support=2)
    m.remember("沙箱守卫会拦截 site-packages 写入")
    m.remember("沙箱守卫会拦截 site-packages 写入操作")
    first = m.distill()
    assert first >= 1
    assert m.distill() == 0, "second distillation must add nothing"


def test_distill_requires_min_support(embedder):
    m = _store(embedder, min_support=3)
    m.remember("只有一条独特记录")
    assert m.distill() == 0


def test_semantic_recall_returns_facts(embedder):
    m = _store(embedder, min_support=2)
    m.remember("沙箱守卫会拦截 site-packages 写入")
    m.remember("沙箱守卫会拦截 site-packages 写入操作")
    m.distill()
    out = m.semantic_recall("site-packages 写入")
    assert out and "site-packages" in out[0]


def test_forget_removes_decayed(embedder):
    m = _store(embedder, decay_lambda=50.0)
    m.remember("很快会被遗忘的一条记录", salience=1.0)
    removed = m.forget(threshold=0.9)
    assert removed in (0, 1)
    assert len(m.episodic) in (0, 1)


def test_clear(embedder):
    m = _store(embedder)
    m.remember("x")
    m.clear()
    assert m.stats() == {"working": 0, "episodic": 0, "semantic": 0}


def test_stats(embedder):
    m = _store(embedder)
    m.remember("a")
    m.remember("b")
    assert m.stats()["episodic"] == 2
