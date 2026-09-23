"""The four DP-4 roles and their prompt contracts.

ROLE markers are machine-readable (`ROLE: <name>`); the deterministic MockLLM
dispatches on them, and any real model simply obeys them.

The instruction text never contains the literal evidence delimiter
(`<kb-context>`): a prompt that mentions the delimiter inside its instructions
makes the model quote the instruction back as evidence.

Author: 晨星
"""
from __future__ import annotations

from typing import Sequence

from ..core.types import Message, Objection, Role

CTX_OPEN = "<kb-context>"
CTX_CLOSE = "</kb-context>"

PROPOSER_SYSTEM = """ROLE: proposer
你是辩证式推理协议中的「提案者」。
规则：
1. 只能使用证据块中出现的原文片段作答，禁止引入证据块之外的任何事实。
2. 用中文输出 1-3 句断言，每句独立成行，句末用句号。
3. 若证据块为空，只输出：NO_EVIDENCE。
4. 若收到「反对意见」，必须针对被引用证据补充或修正后再输出。
"""

FACTCHECKER_SYSTEM = """ROLE: factchecker
你是辩证式推理协议中的「事实核验者」。
对每一条断言给出裁决，逐行输出，严格使用格式：
CLAIM <序号> | SUPPORTED 或 UNSUPPORTED | <证据span_id> 或 none | <0-1 之间的分数>
只输出裁决行，不要解释。
"""

CRITIC_SYSTEM = """ROLE: critic
你是辩证式推理协议中的「批评者」。
规则：
1. 只有当提案遗漏了高相关证据，或断言与证据冲突时，才可以提出反对。
2. 每一条反对必须引用至少一个证据 span_id，否则该反对无效并被丢弃。
3. 输出格式：OBJECTION | <span_id> | <理由>；若无有效反对，只输出 NO_OBJECTION。
"""

SYNTHESIZER_SYSTEM = """ROLE: synthesizer
你是辩证式推理协议中的「综合者」。
把通过证据门禁的断言整理成最终答案：去重、按逻辑排序、不新增任何事实。
只输出最终答案正文。
"""


def role_system(role: Role) -> Message:
    mapping = {
        Role.PROPOSER: PROPOSER_SYSTEM,
        Role.FACTCHECKER: FACTCHECKER_SYSTEM,
        Role.CRITIC: CRITIC_SYSTEM,
        Role.SYNTHESIZER: SYNTHESIZER_SYSTEM,
    }
    return Message.system(mapping[role])


def with_context(body: str, context_block: str) -> str:
    return f"{body}\n{CTX_OPEN}\n{context_block}\n{CTX_CLOSE}"


def proposer_messages(query: str, context_block: str, objections: Sequence[Objection] = ()) -> list[Message]:
    body = f"问题：{query}\n"
    if objections:
        body += "上一轮收到的有效反对意见：\n"
        for o in objections:
            body += f"OBJECTION | {','.join(o.citations)} | {o.text}\n"
        body += "请补充被遗漏的证据后重新作答。\n"
    return [role_system(Role.PROPOSER), Message.user(with_context(body, context_block))]


def factchecker_messages(claims: Sequence[str], context_block: str) -> list[Message]:
    body = "待核验断言：\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    return [role_system(Role.FACTCHECKER), Message.user(with_context(body, context_block))]


def critic_messages(query: str, proposal: str, context_block: str) -> list[Message]:
    body = f"问题：{query}\n待审查提案：\n{proposal}\n"
    return [role_system(Role.CRITIC), Message.user(with_context(body, context_block))]


def synthesizer_messages(claims: Sequence[str]) -> list[Message]:
    body = "已通过证据门禁的断言：\n" + "\n".join(f"- {c}" for c in claims)
    return [role_system(Role.SYNTHESIZER), Message.user(body)]
