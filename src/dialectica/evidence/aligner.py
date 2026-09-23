"""Claim <-> evidence span alignment.

Score combines asymmetric token coverage (how much of the claim is covered by
the span) with symmetric Jaccard overlap, so a long span that merely happens to
share common words cannot validate an unrelated claim.

Author: 晨星
"""
from __future__ import annotations

from typing import Sequence

from ..core.types import Claim, Hit, Verdict
from ..retrieval.tokenizer import tokenize

_COVERAGE_W = 0.6
_JACCARD_W = 0.4


def coverage(claim: str, span: str) -> float:
    ct, st = set(tokenize(claim)), set(tokenize(span))
    if not ct or not st:
        return 0.0
    return len(ct & st) / len(ct)


def jaccard(claim: str, span: str) -> float:
    ct, st = set(tokenize(claim)), set(tokenize(span))
    if not ct or not st:
        return 0.0
    return len(ct & st) / len(ct | st)


def align_score(claim: str, span: str) -> float:
    return _COVERAGE_W * coverage(claim, span) + _JACCARD_W * jaccard(claim, span)


def align_claims(claims: Sequence[Claim], hits: Sequence[Hit], threshold: float = 0.30) -> list[Claim]:
    """Attach the best supporting span to every claim and set its verdict.

    Invariant: a claim is SUPPORTED iff it carries at least one span id, so
    `Claim.anchored` can never be true without evidence.
    """
    for claim in claims:
        best_score, best_span = 0.0, ""
        for hit in hits:
            s = align_score(claim.text, hit.text)
            if s > best_score:
                best_score, best_span = s, hit.span_id
        claim.support = round(best_score, 6)
        if best_score >= threshold and best_span:
            claim.verdict = Verdict.SUPPORTED
            claim.evidence = [best_span]
        else:
            claim.verdict = Verdict.UNSUPPORTED
            claim.evidence = []
    return list(claims)
