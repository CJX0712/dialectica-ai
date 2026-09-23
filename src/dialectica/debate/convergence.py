"""Convergence measurement for the dialectic protocol.

`delta` is the semantic distance between two consecutive proposals:
`delta = 1 - cosine(embed(prev), embed(cur))`.

It is a *diagnostic*, not the stopping rule -- the debate stops when no
admissible objection survives. A large delta on the final round tells you the
critic actually changed the answer, which is the whole point.

Author: 晨星
"""
from __future__ import annotations

from typing import Sequence

from ..core.types import Objection
from ..retrieval.dense import cosine


class ConvergenceTracker:
    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._vectors: dict[int, list[float]] = {}
        self.history: list[float] = []

    def delta(self, index: int, text: str) -> float:
        vec = self._embedder.embed_one(text)
        self._vectors[index] = vec
        if index == 0 or (index - 1) not in self._vectors:
            # Round 0 has no predecessor: the sentinel 1.0 means "max change"
            # and is deliberately not part of the stability statistics.
            return 1.0
        d = max(0.0, min(1.0, 1.0 - cosine(self._vectors[index - 1], vec)))
        self.history.append(round(d, 6))
        return d

    def stability(self) -> float:
        """1 - mean(delta) over real transitions. 1.0 means the debate never moved."""
        if not self.history:
            return 1.0
        return round(1.0 - sum(self.history) / len(self.history), 6)

    def reset(self) -> None:
        self._vectors.clear()
        self.history.clear()


def parse_objections(text: str) -> list[Objection]:
    """Parse the Critic's output into `Objection` records.

    Lines without a span citation are still parsed but end up invalid, which is
    exactly how the protocol discards unfounded criticism.
    """
    out: list[Objection] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("NO_OBJECTION"):
            continue
        if not line.startswith("OBJECTION"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        # A well-formed objection carries a citation column; a two-column line
        # is an objection *without* evidence and must survive parsing so the
        # protocol can discard it visibly rather than silently.
        cite_col = parts[1] if len(parts) >= 3 else ""
        cites = [c.strip() for c in cite_col.split(",") if c.strip().startswith("span::")]
        reason = "|".join(parts[2:] if len(parts) >= 3 else parts[1:]).strip()
        out.append(Objection.create(reason, cites))
    return out


def parse_verdicts(text: str) -> dict[int, tuple[str, str]]:
    """Parse FactChecker output: {claim_index: (verdict, span_id)}."""
    out: dict[int, tuple[str, str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.upper().startswith("CLAIM"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0].split()[1])
        except (IndexError, ValueError):
            continue
        verdict = parts[1].upper()
        span = parts[2] if parts[2].lower() != "none" else ""
        out[idx] = (verdict, span)
    return out


def filter_valid(objections: Sequence[Objection], known_spans: Sequence[str]) -> list[Objection]:
    """Keep objections whose citations exist in the current evidence set.

    This is the guardrail against a critic inventing span ids.
    """
    known = set(known_spans)
    valid: list[Objection] = []
    for o in objections:
        cites = [c for c in o.citations if c in known]
        if not cites:
            continue
        valid.append(Objection(objection_id=o.objection_id, text=o.text, citations=cites, severity=o.severity))
    return valid


def stalled(prev: Sequence[Objection], cur: Sequence[Objection]) -> bool:
    """True when two consecutive rounds raise the same objections: no progress."""
    a = sorted((o.text, tuple(o.citations)) for o in prev)
    b = sorted((o.text, tuple(o.citations)) for o in cur)
    return bool(a) and a == b
