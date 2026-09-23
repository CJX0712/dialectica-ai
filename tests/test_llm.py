"""LLM provider contracts."""
from __future__ import annotations

from dialectica.core.config import LLMConfig
from dialectica.core.errors import ConfigError
from dialectica.core.types import Message
from dialectica.llm import build_llm
from dialectica.llm.base import detect_role, extract_context, spans_of
from dialectica.llm.mock import MockLLM


def _ctx(texts):
    return "\n".join(f"span::s{i} | doc :: {t}" for i, t in enumerate(texts))


def test_build_llm_mock():
    assert isinstance(build_llm(LLMConfig(backend="mock")), MockLLM)


def test_build_llm_unknown_backend():
    try:
        build_llm(LLMConfig(backend="nope"))
        assert False
    except ConfigError:
        pass


def test_detect_role():
    assert detect_role([Message.system("ROLE: critic\n...")]) == "critic"
    assert detect_role([Message.system("no marker")]) == "generic"


def test_extract_context():
    body = f"问题：x\n<kb-context>\n{_ctx(['a', 'b'])}\n</kb-context>"
    ctx = extract_context([Message.user(body)])
    assert "span::s0" in ctx
    assert "span::s1" in ctx


def test_spans_of_parses_ids():
    pairs = spans_of(_ctx(["第一条证据", "第二条证据"]))
    assert pairs[0][0] == "span::s0"
    assert "第一条证据" in pairs[0][1]


def test_mock_proposer_is_extractive():
    llm = MockLLM()
    ctx = _ctx(["沙箱守卫会拦截 site-packages 写入。", "代理导致 SOCKS 报错。"])
    out = llm.complete(
        [
            Message.system("ROLE: proposer"),
            Message.user(f"沙箱 site-packages 写入\n<kb-context>\n{ctx}\n</kb-context>"),
        ]
    )
    assert "site-packages" in out.text
    assert all(s in ctx for s in out.text.split(" ") if s)


def test_mock_proposer_no_evidence_marker():
    llm = MockLLM()
    out = llm.complete([Message.system("ROLE: proposer"), Message.user("问题\n<kb-context>\n\n</kb-context>")])
    assert "NO_EVIDENCE" in out.text


def test_mock_factchecker_format():
    llm = MockLLM()
    ctx = _ctx(["沙箱守卫会拦截 site-packages 写入。"])
    out = llm.complete(
        [
            Message.system("ROLE: factchecker"),
            Message.user(f"0. 沙箱守卫会拦截 site-packages 写入。\n<kb-context>\n{ctx}\n</kb-context>"),
        ]
    )
    assert out.text.startswith("CLAIM 0")
    assert "SUPPORTED" in out.text


def test_mock_critic_emits_no_objection_when_fully_covered():
    from dialectica.debate.roles import critic_messages

    llm = MockLLM(max_sentences=3)
    ctx = _ctx(["沙箱守卫会拦截 site-packages 写入。"])
    out = llm.complete(critic_messages("沙箱 site-packages 写入", "沙箱守卫会拦截 site-packages 写入。", ctx))
    assert "NO_OBJECTION" in out.text


def test_mock_critic_objects_when_evidence_is_skipped():
    from dialectica.debate.roles import critic_messages

    llm = MockLLM(max_sentences=1)
    ctx = _ctx(["沙箱守卫会拦截 site-packages 写入。", "代理导致 SOCKS 依赖缺失报错。"])
    out = llm.complete(critic_messages("沙箱 site-packages 写入", "代理导致 SOCKS 依赖缺失报错。", ctx))
    assert out.text.startswith("OBJECTION")
    assert "span::" in out.text


def test_mock_synthesizer_strips_scaffolding():
    llm = MockLLM()
    out = llm.complete(
        [Message.system("ROLE: synthesizer"), Message.user("已通过证据门禁的断言：\n- 第一条断言。\n- 第二条断言。")]
    )
    assert "已通过" not in out.text
    assert "第一条断言。" in out.text


def test_mock_react_returns_observation():
    llm = MockLLM()
    out = llm.complete([Message.system("ROLE: react"), Message.user("问题\nObservation: 84")])
    assert "84" in out.text


def test_mock_respects_stop():
    llm = MockLLM()
    ctx = _ctx(["第一条证据。", "第二条证据。"])
    out = llm.complete(
        [
            Message.system("ROLE: proposer"),
            Message.user(f"证据\n<kb-context>\n{ctx}\n</kb-context>"),
        ],
        stop=["第二条"],
    )
    assert "第二条" not in out.text


def test_mock_is_deterministic():
    msgs = [Message.system("ROLE: proposer"), Message.user("<kb-context>\nspan::s0 | d :: 电池续航八小时。\n</kb-context>")]
    a = MockLLM().complete(msgs).text
    b = MockLLM().complete(msgs).text
    assert a == b
