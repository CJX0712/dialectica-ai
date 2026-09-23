"""Deterministic zero-dependency LLM.

This is the default provider. It is *extractive*: every sentence it emits is a
verbatim span from the context block, so a pipeline that passes its tests with
MockLLM is provably free of unsupported invention.

It is role-aware: the debate protocol stamps `ROLE: <name>` into the system
prompt and MockLLM answers accordingly.

Author: 晨星
"""
from __future__ import annotations

import re
from typing import Sequence

from ..core.types import Generation, Message
from .base import (
    coverage,
    detect_role,
    extract_context,
    last_user,
    overlap,
    parse_critique_user,
    parse_proposer_user,
    spans_of,
    tokenize,
)

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class MockLLM:
    """Offline deterministic provider. `name` is part of the public contract."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-1",
        max_sentences: int = 3,
        relevance_gate: float = 0.70,
        covered_gate: float = 0.50,
    ) -> None:
        self.model = model
        self.max_sentences = max_sentences
        self.relevance_gate = relevance_gate
        self.covered_gate = covered_gate

    # -- ranking helpers ---------------------------------------------------
    @staticmethod
    def _rank(query: str, cands: Sequence[str]) -> list[tuple[int, float]]:
        scored = [(i, overlap(query, c)) for i, c in enumerate(cands)]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored

    # -- roles -------------------------------------------------------------
    def _propose(self, query: str, context: str) -> str:
        pairs = spans_of(context)
        bodies = [b for _, b in pairs] or sentences(context)
        if not bodies:
            return "NO_EVIDENCE: 语料中没有与该问题相关的证据。"
        ranked = self._rank(query, bodies)
        picked = [bodies[i] for i, s in ranked[: self.max_sentences] if s > 0.0]
        if not picked:
            picked = [bodies[0]]
        return " ".join(picked)

    def _revise(self, query: str, context: str, objections: list[str]) -> str:
        """Address objections by pulling in the spans they cite."""
        pairs = spans_of(context)
        cited = [b for sid, b in pairs if any(sid and sid in o for o in objections)]
        base = self._propose(query, context)
        extra = [c for c in cited if c not in base]
        merged = base + (" " + " ".join(extra[:3]) if extra else "")
        return merged.strip()

    def _factcheck(self, context: str, claims: list[str]) -> str:
        pairs = spans_of(context)
        bodies = [b for _, b in pairs] or sentences(context)
        lines: list[str] = []
        for idx, claim in enumerate(claims):
            best, best_span = 0.0, ""
            for sid, body in pairs or [("", b) for b in bodies]:
                s = overlap(claim, body)
                if s > best:
                    best, best_span = s, sid
            verdict = "SUPPORTED" if best >= 0.30 else "UNSUPPORTED"
            lines.append(f"CLAIM {idx} | {verdict} | {best_span or 'none'} | {best:.2f}")
        return "\n".join(lines) if lines else "NO_CLAIMS"

    def _critique(self, user: str, context: str) -> str:
        """Object when the proposal skips evidence the query actually needs.

        An objection must cite a span, and the span must be close to the best
        one (`relevance_gate * top_relevance`); otherwise a large corpus would
        keep the debate spinning forever.
        """
        query, proposal = parse_critique_user(user)
        pairs = spans_of(context)
        if not pairs or not query:
            return "NO_OBJECTION"
        bodies = [b for _, b in pairs]
        ranked = self._rank(query, bodies)
        if not ranked:
            return "NO_OBJECTION"
        top_relevance = ranked[0][1]
        lines: list[str] = []
        for idx, rel in ranked:
            if top_relevance > 0 and rel < self.relevance_gate * top_relevance:
                continue
            if coverage(bodies[idx], proposal) >= self.covered_gate:
                continue
            sid = pairs[idx][0] if idx < len(pairs) else ""
            if not sid:
                continue
            lines.append(f"OBJECTION | {sid} | 遗漏证据：{bodies[idx][:40]} (score={rel:.2f})")
            if len(lines) >= 3:
                break
        return "\n".join(lines) if lines else "NO_OBJECTION"

    def _synthesize(self, proposal: str) -> str:
        """Strip the protocol scaffolding and return only the claim lines."""
        lines = []
        for ln in proposal.splitlines():
            s = ln.strip()
            if not s or s.startswith("已通过证据门禁的断言"):
                continue
            s = re.sub(r"^\s*(?:[-*•]|\d+[.、)])\s*", "", s)
            if s:
                lines.append(s)
        return " ".join(lines) if lines else proposal.strip()

    # -- protocol ----------------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
    ) -> Generation:
        role = detect_role(messages)
        context = extract_context(messages)
        user = last_user(messages)

        if role == "proposer":
            query, objections = parse_proposer_user(user)
            if objections:
                text = self._revise(query, context, objections)
            else:
                text = self._propose(query, context)
        elif role == "factchecker":
            claims = [
                ln.strip()[2:].strip()
                for ln in user.splitlines()
                if ln.strip().startswith(("-", "*", "•")) or re.match(r"^\d+\.", ln.strip())
            ]
            if not claims:
                claims = sentences(user)
            text = self._factcheck(context, claims)
        elif role == "critic":
            text = self._critique(user, context)
        elif role == "synthesizer":
            text = self._synthesize(user)
        elif role == "react":
            found = re.search(r"Observation:\s*(.+)", user)
            if found:
                text = f"Final Answer: {found.group(1).strip()}"
            elif context:
                text = self._propose(user, context)
            else:
                text = ""
        else:
            text = self._propose(user, context) if context else ""

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s, 1)[0]
        return Generation(text=text.strip(), model=self.model, tokens=len(tokenize(text)))

    def stream(self, messages: Sequence[Message], **kwargs):  # pragma: no cover
        gen = self.complete(messages, **kwargs)
        for chunk in gen.text.split(" "):
            yield chunk + " "
