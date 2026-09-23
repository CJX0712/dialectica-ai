"""DP-4 protocol invariants."""
from __future__ import annotations

from dialectica.core.config import DebateConfig
from dialectica.core.types import Objection, Verdict
from dialectica.debate import (
    ConvergenceTracker,
    DialecticProtocol,
    filter_valid,
    parse_objections,
    parse_verdicts,
    stalled,
)
from dialectica.llm.mock import MockLLM
from dialectica.retrieval.dense import HashingEmbedder
from dialectica.retrieval.pipeline import RetrievalPipeline


def _protocol(docs, max_sentences: int = 3, **cfg):
    emb = HashingEmbedder(dim=128)
    retriever = RetrievalPipeline(embedder=emb)
    retriever.add_documents(docs)
    proto = DialecticProtocol(
        llm=MockLLM(max_sentences=max_sentences),
        retriever=retriever,
        embedder=emb,
        config=DebateConfig(**cfg),
    )
    return proto


def test_parse_objections_requires_citation():
    text = "OBJECTION | span::abc | 遗漏证据\nOBJECTION | 没有引用任何证据\n"
    objs = parse_objections(text)
    assert len(objs) == 2
    assert objs[0].valid
    assert not objs[1].valid


def test_parse_objections_no_objection_marker():
    assert parse_objections("NO_OBJECTION") == []


def test_filter_valid_drops_unknown_spans():
    objs = [
        Objection.create("真实引用", ["span::known"]),
        Objection.create("伪造引用", ["span::hallucinated"]),
    ]
    valid = filter_valid(objs, ["span::known"])
    assert len(valid) == 1
    assert valid[0].citations == ["span::known"]


def test_parse_verdicts():
    out = parse_verdicts("CLAIM 0 | SUPPORTED | span::a | 0.80\nCLAIM 1 | UNSUPPORTED | none | 0.10")
    assert out[0] == ("SUPPORTED", "span::a")
    assert out[1] == ("UNSUPPORTED", "")


def test_stalled_detects_repeat():
    a = [Objection.create("同一条", ["span::x"])]
    assert stalled(a, a)
    assert not stalled([], a)


def test_convergence_tracker_first_delta_is_one(embedder):
    t = ConvergenceTracker(embedder)
    assert t.delta(0, "第一版答案") == 1.0


def test_convergence_tracker_identical_text_delta_zero(embedder):
    t = ConvergenceTracker(embedder)
    t.delta(0, "同样的文本")
    assert abs(t.delta(1, "同样的文本")) < 1e-9
    assert t.stability() > 0.99


def test_protocol_converges_within_budget(docs):
    proto = _protocol(docs)
    result = proto.run("沙箱里为什么无法安装 Python 包")
    assert result.converged, "protocol must reach a state with no admissible objection"
    assert 1 <= result.round_count <= DebateConfig().max_rounds
    assert result.answer


def test_protocol_stops_once_objections_are_resolved(docs):
    """The last round must carry zero admissible objections."""
    proto = _protocol(docs)
    result = proto.run("沙箱里为什么无法安装 Python 包")
    assert result.converged
    assert not result.rounds[-1].valid_objections


def test_protocol_runs_multiple_rounds_when_critic_objects(docs):
    """A proposer limited to one sentence cannot cover all evidence, so the
    critic objects and the protocol must run a second round."""
    proto = _protocol(docs, max_sentences=1, max_rounds=3)
    result = proto.run("npm 安装依赖时找不到包怎么办")
    assert result.round_count >= 2
    assert result.rounds[0].valid_objections, "round 0 must raise an admissible objection"


def test_every_surviving_claim_is_anchored(docs):
    proto = _protocol(docs)
    result = proto.run("重排模型为什么会让中文检索结果变差")
    assert result.claims, "expected at least one surviving claim"
    for c in result.claims:
        assert c.anchored
        assert c.evidence


def test_answer_contains_no_protocol_scaffolding(docs):
    proto = _protocol(docs)
    result = proto.run("沙箱里为什么无法安装 Python 包")
    assert "已通过证据门禁" not in result.answer
    assert result.answer.strip()


def test_metrics_are_finite(docs):
    proto = _protocol(docs)
    result = proto.run("npm 镜像源被劫持怎么办")
    for key, value in result.metrics.items():
        assert value == value, f"{key} is NaN"
    assert 0.0 <= result.metrics["faithfulness"] <= 1.0


def test_round_records_delta(docs):
    proto = _protocol(docs)
    result = proto.run("沙箱里为什么无法安装 Python 包")
    assert result.rounds[0].delta == 1.0
    assert all(0.0 <= r.delta <= 1.0 for r in result.rounds)


def test_gap_retrieval_disabled_shortens_run(docs):
    proto = _protocol(docs, gap_retrieval=False)
    result = proto.run("沙箱守卫拦截写入怎么办")
    assert all(not r.gap_queries for r in result.rounds)


def test_verdict_enum_values():
    assert Verdict.SUPPORTED.value == "supported"
    assert Verdict.UNSUPPORTED.value == "unsupported"
