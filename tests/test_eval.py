"""Evaluation metrics + benchmark invariants."""
from __future__ import annotations

from dialectica.core.types import Document
from dialectica.eval import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    keyword_recall,
    run_benchmark,
)
from dialectica.system import Dialectica

SPANS = [
    "沙箱守卫会拦截对 site-packages 的写入与删除，pip install 会报 SAFE_DELETE_BULK_GUARD_ERROR。",
    "机器上的 SOCKS5 代理会让 pip 报 Missing dependencies for SOCKS support。",
]


def test_faithfulness_bounds():
    assert faithfulness("", SPANS) == 0.0
    assert 0.0 <= faithfulness("沙箱守卫会拦截写入。", SPANS) <= 1.0


def test_faithfulness_detects_hallucination():
    good = faithfulness("沙箱守卫会拦截对 site-packages 的写入与删除。", SPANS)
    bad = faithfulness("鲸鱼在深海里迁徙。", SPANS)
    assert good > bad


def test_answer_relevancy_bounds(embedder):
    v = answer_relevancy(embedder, "电池续航", "电池续航是八小时")
    assert 0.0 <= v <= 1.0


def test_answer_relevancy_empty_is_zero(embedder):
    assert answer_relevancy(embedder, "q", "") == 0.0


def test_context_precision_bounds():
    assert context_precision([], "gold") == 0.0
    assert 0.0 <= context_precision(SPANS, "沙箱守卫会拦截写入") <= 1.0


def test_context_recall_perfect_and_zero():
    assert context_recall(SPANS, SPANS) == 1.0
    assert context_recall(["完全不相关的文本"], SPANS) == 0.0


def test_context_recall_empty_gold_is_one():
    assert context_recall(SPANS, []) == 1.0


def test_keyword_recall():
    assert keyword_recall("答案是 site-packages 与 PYTHONPATH", ["site-packages", "PYTHONPATH"]) == 1.0
    assert keyword_recall("什么都不相关", ["site-packages"]) == 0.0
    assert keyword_recall("任意", []) == 1.0


def test_benchmark_runs_on_fresh_pipelines(embedder):
    """Each case must get a brand-new system, otherwise recall drifts."""
    report = run_benchmark(lambda: Dialectica(), embedder)
    assert report.total == 3
    assert report.passed >= 2, report.as_dict()
    for case in report.cases:
        assert 0.0 <= case.metrics["faithfulness"] <= 1.0
        assert case.metrics["rounds"] >= 1


def test_benchmark_is_deterministic(embedder):
    a = run_benchmark(lambda: Dialectica(), embedder).as_dict()
    b = run_benchmark(lambda: Dialectica(), embedder).as_dict()
    assert a["mean_faithfulness"] == b["mean_faithfulness"]
    assert a["passed"] == b["passed"]


def test_benchmark_ingests_its_own_corpus():
    s = Dialectica()
    n = s.add_documents([Document.create(t) for t in SPANS])
    assert n >= 2
