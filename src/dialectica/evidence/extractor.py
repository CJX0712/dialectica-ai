"""Claim extraction: text -> list of assertive statements.

Only *assertions* become claims. Questions, hedges and pure connectives are
dropped, otherwise the faithfulness metric is diluted by filler and the gate
becomes meaningless.

Author: 晨星
"""
from __future__ import annotations

import re

from ..core.types import Claim

_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_QUESTION = re.compile(r"[？?]$")
_HEDGE = re.compile(r"^\s*(?:可能|也许|或许|大概|似乎|建议|希望|应该考虑)", re.UNICODE)
_LEAD = re.compile(r"^\s*(?:[-*•]|\d+[.、)])\s*")
_DISCARD = re.compile(r"^\s*(?:综上|总之|以上|也就是说|换句话说)\s*[,，]?\s*$")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def is_claim(sentence: str) -> bool:
    s = _LEAD.sub("", sentence).strip()
    if len(s) < 4:
        return False
    if _QUESTION.search(s):
        return False
    if _HEDGE.match(s):
        return False
    if _DISCARD.match(s):
        return False
    return True


def extract_claims(text: str) -> list[Claim]:
    """Deterministic claim splitter.

    Invariant: every returned claim is a non-empty substring of `text`.
    """
    return [Claim.create(s) for s in split_sentences(text) if is_claim(s)]
