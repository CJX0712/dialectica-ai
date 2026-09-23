"""Language-aware tokeniser for mixed Chinese/English corpora.

Chinese has no whitespace word boundaries, so a naive `split()` destroys
recall. We emit, for CJK runs, both unigrams and character bigrams -- bigrams
carry most of the lexical signal ("电池" / "池续" / "续航") while unigrams keep
recall for short queries.

Author: 晨星
"""
from __future__ import annotations

import re

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_ASCII_WORD = re.compile(r"[a-zA-Z0-9]+(?:[._'-][a-zA-Z0-9]+)*")
_CJK_RUN = re.compile(f"[{_CJK}]+")
_PUNCT = re.compile(r"[\s\u3000]+")

CJK_RANGE = (0x4E00, 0x9FFF)


def is_cjk(ch: str) -> bool:
    return bool(ch) and CJK_RANGE[0] <= ord(ch) <= CJK_RANGE[1]


def tokenize(text: str, *, bigrams: bool = True) -> list[str]:
    """Tokenise `text` into a flat token list.

    Invariants:
      * deterministic (same input -> same output)
      * ASCII tokens are lower-cased
      * a CJK run of length n contributes n unigrams + (n-1) bigrams
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    for word in _ASCII_WORD.findall(lowered):
        tokens.append(word)
    for run in _CJK_RUN.findall(lowered):
        for i, ch in enumerate(run):
            tokens.append(ch)
            if bigrams and i + 1 < len(run):
                tokens.append(run[i : i + 2])
    return tokens


def unique_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def normalize(text: str) -> str:
    return _PUNCT.sub(" ", text).strip()
