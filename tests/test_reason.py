"""Deterministic tool routing + ReAct + self-consistency."""
from __future__ import annotations

from dialectica.core.errors import ToolError
from dialectica.core.types import Message
from dialectica.llm.mock import MockLLM
from dialectica.reason import (
    Calculator,
    Clock,
    ReActAgent,
    UnitConverter,
    default_tools,
    find_tool,
    route,
    self_consistent,
)


def test_route_arithmetic():
    r = route("12*(3+4)")
    assert r.kind == "calculator"
    assert r.tool == "calculator"


def test_route_clock():
    assert route("现在几点").kind == "clock"
    assert route("what time is it").kind == "clock"


def test_route_unit():
    assert route("12 km to m").kind == "unit"


def test_route_rag_by_default():
    assert route("沙箱里为什么装不上包").kind == "rag"
    assert route("为什么服务会自己停掉").kind == "rag"


def test_calculator_arithmetic():
    assert Calculator().invoke("12*(3+4)") == "84"
    assert Calculator().invoke("7/2") == "3.5"


def test_calculator_rejects_code_injection():
    calc = Calculator()
    for bad in ["__import__('os').system('echo hi')", "open('/etc/passwd').read()"]:
        try:
            calc.invoke(bad)
            assert False, f"should have rejected {bad}"
        except ToolError:
            pass


def test_calculator_rejects_syntax_error():
    try:
        Calculator().invoke("1 +")
        assert False
    except ToolError:
        pass


def test_clock_format():
    out = Clock(now_fn=lambda: __import__("datetime").datetime(2026, 9, 24, 1, 39, 40)).invoke("")
    assert out == "2026-09-24 01:39:40"


def test_unit_converter():
    assert UnitConverter().invoke("12 km to m") == "12000 m"
    assert UnitConverter().invoke("1 h to min") == "60 min"


def test_unit_converter_rejects_unknown():
    try:
        UnitConverter().invoke("12 foo to bar")
        assert False
    except ToolError:
        pass


def test_default_tools_and_lookup():
    tools = default_tools()
    assert {t.name for t in tools} == {"calculator", "clock", "unit"}
    assert find_tool(tools, "calculator") is not None
    assert find_tool(tools, "nope") is None


def test_react_arithmetic_uses_tool_result_not_model_guess():
    agent = ReActAgent(MockLLM())
    out = agent.run("12*(3+4)")
    assert "84" in out
    assert "72" not in out


def test_react_records_trace():
    agent = ReActAgent(MockLLM())
    agent.run("12*(3+4)")
    assert agent.trace
    assert agent.trace[0]["tool"] == "calculator"
    assert agent.trace[0]["observation"] == "84"


def test_react_falls_back_to_llm_for_rag_query():
    agent = ReActAgent(MockLLM())
    out = agent.run("这是一个需要检索的问题")
    assert isinstance(out, str)


def test_self_consistent_majority():
    class Flip:
        def __init__(self):
            self.n = 0

        def complete(self, messages, **kw):
            self.n += 1
            from dialectica.core.types import Generation

            return Generation(text="A" if self.n % 3 != 0 else "B")

    assert self_consistent(Flip(), [Message.user("q")], n=5) == "A"


def test_self_consistent_single_pass():
    assert self_consistent(MockLLM(), [Message.user("q")], n=1) is not None
