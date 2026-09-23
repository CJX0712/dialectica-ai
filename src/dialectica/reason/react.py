"""ReAct loop with deterministic tool pre-routing.

Order of operations matters: run the tool *first*, inject its result as an
Observation, and instruct the model not to recompute. Letting a small model
re-derive `12*(3+4)` after the calculator already answered is how you get
"72" in the final answer.

Author: 晨星
"""
from __future__ import annotations

from typing import Sequence

from ..core.types import Message
from .router import route
from .tools import default_tools, find_tool

REACT_SYSTEM = """ROLE: react
你是一个可以使用工具的推理体。
当输入中出现 Observation 时，直接把 Observation 的数值作为最终答案，禁止重新计算。
输出格式：Thought: <一句话> / Action: <tool(arg)> 或 Final Answer: <答案>
"""


class ReActAgent:
    def __init__(self, llm, tools=None, max_steps: int = 3) -> None:
        self.llm = llm
        self.tools = tools or default_tools()
        self.max_steps = max_steps
        self.trace: list[dict[str, str]] = []

    def run(self, query: str) -> str:
        decision = route(query)
        if decision.is_tool:
            tool = find_tool(self.tools, decision.tool)
            if tool is not None:
                try:
                    observation = tool.invoke(decision.argument)
                except Exception as exc:
                    observation = f"tool error: {exc}"
                self.trace.append(
                    {"step": "tool", "tool": decision.tool, "argument": decision.argument, "observation": observation}
                )
                messages = [
                    Message.system(REACT_SYSTEM),
                    Message.user(
                        f"问题：{query}\nObservation: {observation}\n"
                        "请直接把 Observation 作为最终答案，不要重新计算。"
                    ),
                ]
                out = self.llm.complete(messages).text.strip()
                return self._clean(out) or observation
        return self._clean(self.llm.complete([Message.system(REACT_SYSTEM), Message.user(query)]).text)

    @staticmethod
    def _clean(text: str) -> str:
        if not text:
            return ""
        for marker in ("Final Answer:", "最终答案：", "答案："):
            if marker in text:
                return text.split(marker, 1)[1].strip()
        return text.strip()


def self_consistent(llm, messages: Sequence[Message], n: int = 3, temperature: float = 0.7) -> str:
    """Sample n completions and return the majority answer."""
    from collections import Counter

    if n <= 1:
        return llm.complete(list(messages), temperature=0.0).text.strip()
    votes = Counter(llm.complete(list(messages), temperature=temperature).text.strip() for _ in range(n))
    return votes.most_common(1)[0][0]
