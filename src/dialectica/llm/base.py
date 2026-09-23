"""Shared prompt plumbing for every LLM backend.

Author: 晨星
"""
from __future__ import annotations

import re
from typing import Sequence

from ..core.types import Generation, Message

ROLE_MARKER = re.compile(r"ROLE:\s*([a-zA-Z_]+)")
CTX_OPEN = "<kb-context>"
CTX_CLOSE = "</kb-context>"


def detect_role(messages: Sequence[Message]) -> str:
    """Read the `ROLE: xxx` marker the debate protocol stamps into system prompts."""
    for m in messages:
        if m.role != "system":
            continue
        found = ROLE_MARKER.search(m.content)
        if found:
            return found.group(1).lower()
    return "generic"


def extract_context(messages: Sequence[Message]) -> str:
    """Pull the single `<kb-context>` block, if present."""
    for m in reversed(list(messages)):
        if CTX_OPEN in m.content and CTX_CLOSE in m.content:
            return m.content.split(CTX_OPEN, 1)[1].split(CTX_CLOSE, 1)[0]
    return ""


def last_user(messages: Sequence[Message]) -> str:
    for m in reversed(list(messages)):
        if m.role == "user":
            return m.content
    return ""


def tokenize(text: str) -> list[str]:
    """Language-agnostic tokeniser used by the deterministic providers."""
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def spans_of(context: str) -> list[tuple[str, str]]:
    """Parse `span::xxx | title :: text` lines emitted by the retrieval layer."""
    out: list[tuple[str, str]] = []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("span::"):
            head, _, body = line.partition("|")
            body = body.strip()
            if "::" in body:
                _, _, body = body.partition("::")
            out.append((head.strip(), body.strip()))
        else:
            out.append(("", line))
    return out


def parse_labeled(text: str, label: str) -> str:
    """Read a `标签：值` line out of a protocol message."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(label):
            return s[len(label) :].strip()
    return ""


def parse_proposer_user(user: str) -> tuple[str, list[str]]:
    """(query, objection lines) from the Proposer's user message."""
    query = parse_labeled(user, "问题：") or user
    objections = [ln.strip() for ln in user.splitlines() if ln.strip().startswith("OBJECTION")]
    return query.splitlines()[0].strip(), objections


def parse_critique_user(user: str) -> tuple[str, str]:
    """(query, proposal) from the Critic's user message."""
    query = parse_labeled(user, "问题：") or user
    proposal = ""
    if "待审查提案：" in user:
        proposal = user.split("待审查提案：", 1)[1].split(CTX_OPEN)[0].strip()
    return query.splitlines()[0].strip(), proposal


def overlap(a: str, b: str) -> float:
    """Symmetric token overlap (Jaccard) in [0, 1]."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def coverage(claim: str, span: str) -> float:
    """How much of `claim` is covered by `span`, in [0, 1].

    Use this -- not Jaccard -- to ask "is this evidence already represented in
    the proposal?". A proposal made of three sentences has a Jaccard of only
    ~0.4 against each one, which would make a critic object to evidence it
    already contains.
    """
    ta, tb = set(tokenize(claim)), set(tokenize(span))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def empty_generation(model: str = "mock-1") -> Generation:
    return Generation(text="", model=model, tokens=0)
