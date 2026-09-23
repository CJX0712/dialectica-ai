"""Evidence gate: prune or reject unsupported claims.

`max_unanchored_ratio` is the single knob that decides whether an answer may
leave the system. With `prune_unanchored=True` (default) unsupported claims are
deleted rather than passing through with a warning -- the system prefers a
shorter true answer over a longer plausible one.

Author: 晨星
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core.config import EvidenceConfig
from ..core.errors import EvidenceGap
from ..core.types import Claim


@dataclass(frozen=True)
class GateReport:
    total: int
    anchored: int
    unanchored: int
    ratio: float
    passed: bool
    pruned: list[str] = ()

    def as_dict(self) -> dict[str, float]:
        return {
            "total": float(self.total),
            "anchored": float(self.anchored),
            "unanchored": float(self.unanchored),
            "unanchored_ratio": self.ratio,
            "passed": 1.0 if self.passed else 0.0,
        }


class EvidenceGate:
    def __init__(self, config: EvidenceConfig | None = None) -> None:
        self.config = config or EvidenceConfig()

    def evaluate(self, claims: Sequence[Claim]) -> GateReport:
        total = len(claims)
        anchored = [c for c in claims if c.anchored]
        unanchored = [c for c in claims if not c.anchored]
        ratio = (len(unanchored) / total) if total else 0.0
        passed = ratio <= self.config.max_unanchored_ratio
        return GateReport(
            total=total,
            anchored=len(anchored),
            unanchored=len(unanchored),
            ratio=round(ratio, 6),
            passed=passed,
            pruned=[c.text for c in unanchored],
        )

    def apply(self, claims: Sequence[Claim]) -> tuple[list[Claim], GateReport]:
        """Return the surviving claims plus the audit report."""
        report = self.evaluate(claims)
        if not self.config.prune_unanchored:
            return list(claims), report
        kept = [c for c in claims if c.anchored]
        return kept, report

    def enforce(self, claims: Sequence[Claim]) -> list[Claim]:
        """Strict mode: raise when the answer is mostly unsupported."""
        kept, report = self.apply(claims)
        if not report.passed:
            raise EvidenceGap(
                f"unanchored ratio {report.ratio:.2f} > {self.config.max_unanchored_ratio:.2f}"
            )
        return kept
