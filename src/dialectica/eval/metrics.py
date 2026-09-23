"""Evaluation metrics. Deterministic, embedding-based, no LLM judge.

  faithfulness       anchored claims / total claims
  answer_relevancy   mean cosine(answer sentence, query)
  context_precision  retrieved spans relevant to the gold answer / retrieved
  context_recall     gold spans that were retrieved / gold spans
  convergence        rounds used, and whether the protocol converged

Author: 晨星
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..evidence.aligner import align_score
from ..retrieval.dense import cosine
from ..retrieval.tokenizer import tokenize


@dataclass(frozen=True)
class EvalReport:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    rounds: int
    converged: bool

    def as_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 6),
            "answer_relevancy": round(self.answer_relevancy, 6),
            "context_precision": round(self.context_precision, 6),
            "context_recall": round(self.context_recall, 6),
            "rounds": float(self.rounds),
            "converged": 1.0 if self.converged else 0.0,
        }


def faithfulness(answer: str, contexts: Sequence[str], threshold: float = 0.30) -> float:
    """Fraction of answer sentences that find a supporting span."""
    from ..evidence.extractor import split_sentences

    sents = [s for s in split_sentences(answer) if s.strip()]
    if not sents:
        return 0.0
    hit = 0
    for s in sents:
        if any(align_score(s, c) >= threshold for c in contexts):
            hit += 1
    return hit / len(sents)


def answer_relevancy(embedder, query: str, answer: str) -> float:
    from ..evidence.extractor import split_sentences

    sents = [s for s in split_sentences(answer) if s.strip()]
    if not sents:
        return 0.0
    qv = embedder.embed_one(query)
    vals = [max(0.0, cosine(qv, embedder.embed_one(s))) for s in sents]
    return sum(vals) / len(vals)


def _relevant(span: str, gold: str, threshold: float = 0.30) -> bool:
    return align_score(span, gold) >= threshold


def context_precision(retrieved: Sequence[str], gold_answer: str, threshold: float = 0.30) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if _relevant(r, gold_answer, threshold))
    return hits / len(retrieved)


def context_recall(retrieved: Sequence[str], gold_spans: Sequence[str], threshold: float = 0.55) -> float:
    if not gold_spans:
        return 1.0
    found = 0
    for g in gold_spans:
        if any(align_score(r, g) >= threshold for r in retrieved):
            found += 1
    return found / len(gold_spans)


def keyword_recall(answer: str, gold_keywords: Sequence[str]) -> float:
    """Cheap, language-agnostic guard: do the gold keywords appear verbatim?"""
    if not gold_keywords:
        return 1.0
    toks = set(tokenize(answer))
    hit = sum(1 for k in gold_keywords if set(tokenize(k)) <= toks or k in answer)
    return hit / len(gold_keywords)
