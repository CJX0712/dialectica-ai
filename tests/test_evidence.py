"""Evidence anchoring invariants."""
from __future__ import annotations

from dialectica.core.config import EvidenceConfig
from dialectica.core.errors import EvidenceGap
from dialectica.core.types import Chunk, Document, Hit
from dialectica.evidence import (
    EvidenceGate,
    align_claims,
    align_score,
    extract_claims,
    is_claim,
)
from dialectica.evidence.aligner import coverage, jaccard

SPANS = [
    "沙箱守卫会拦截对 site-packages 的写入与删除，pip install 会报 SAFE_DELETE_BULK_GUARD_ERROR。",
    "机器上的 SOCKS5 代理会让 pip 报 Missing dependencies for SOCKS support。",
]


def _hits():
    chunks = [
        Chunk.create(Document.create(t, title=str(i)), t, i, 0, len(t)) for i, t in enumerate(SPANS)
    ]
    return [Hit(chunk=c, score=0.9 - 0.1 * i) for i, c in enumerate(chunks)]


def test_extract_claims_splits_sentences():
    claims = extract_claims("沙箱守卫会拦截写入。代理会导致 SOCKS 报错。")
    assert len(claims) == 2


def test_questions_are_not_claims():
    assert not is_claim("这是为什么呢？")
    assert is_claim("沙箱守卫会拦截写入。")


def test_hedges_are_not_claims():
    assert not is_claim("可能可以用 --target 绕行。")


def test_claim_text_is_substring_of_source():
    text = "沙箱守卫会拦截写入。代理会导致报错。"
    for c in extract_claims(text):
        assert c.text in text


def test_coverage_and_jaccard_bounds():
    a, b = "电池续航八小时", "电池续航"
    assert 0.0 <= coverage(a, b) <= 1.0
    assert 0.0 <= jaccard(a, b) <= 1.0
    assert coverage("电池续航", "电池续航") == 1.0
    assert jaccard("电池续航", "电池续航") == 1.0


def test_align_marks_supported_and_unsupported():
    claims = extract_claims("沙箱守卫会拦截对 site-packages 的写入与删除。鲸鱼在深海里迁徙。")
    aligned = align_claims(claims, _hits(), threshold=0.30)
    assert aligned[0].anchored
    assert not aligned[1].anchored
    assert aligned[1].verdict.value == "unsupported"


def test_anchored_implies_evidence_present():
    claims = extract_claims("沙箱守卫会拦截对 site-packages 的写入与删除。")
    aligned = align_claims(claims, _hits())
    for c in aligned:
        if c.anchored:
            assert c.evidence, "anchored claim with no span id is impossible"


def test_align_score_symmetry():
    assert abs(align_score("a b c", "b c d") - align_score("b c d", "a b c")) < 1e-9


def test_gate_report():
    claims = extract_claims("沙箱守卫会拦截对 site-packages 的写入与删除。鲸鱼在深海里迁徙。")
    aligned = align_claims(claims, _hits())
    gate = EvidenceGate(EvidenceConfig(max_unanchored_ratio=0.5))
    report = gate.evaluate(aligned)
    assert report.total == 2
    assert report.anchored == 1
    assert report.passed


def test_gate_prunes_unanchored():
    claims = extract_claims("沙箱守卫会拦截对 site-packages 的写入与删除。鲸鱼在深海里迁徙。")
    aligned = align_claims(claims, _hits())
    gate = EvidenceGate(EvidenceConfig(prune_unanchored=True))
    kept, _ = gate.apply(aligned)
    assert len(kept) == 1
    assert all(c.anchored for c in kept)


def test_gate_enforce_raises_when_mostly_unsupported():
    claims = extract_claims("鲸鱼在深海里迁徙。企鹅在南极孵蛋。")
    aligned = align_claims(claims, _hits())
    gate = EvidenceGate(EvidenceConfig(max_unanchored_ratio=0.1))
    try:
        gate.enforce(aligned)
        assert False, "expected EvidenceGap"
    except EvidenceGap:
        pass
