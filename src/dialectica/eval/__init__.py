"""Evaluation: deterministic RAG metrics + built-in regression benchmark.

Author: 晨星
"""
from .benchmark import (
    FIXTURES,
    BenchmarkReport,
    CaseResult,
    run_benchmark,
)
from .metrics import (
    EvalReport,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    keyword_recall,
)

__all__ = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "keyword_recall",
    "EvalReport",
    "FIXTURES",
    "CaseResult",
    "BenchmarkReport",
    "run_benchmark",
]
