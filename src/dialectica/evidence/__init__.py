"""Evidence anchoring: claim extraction, span alignment, faithfulness gate.

Author: 晨星
"""
from .aligner import align_claims, align_score, coverage, jaccard
from .extractor import extract_claims, is_claim, split_sentences
from .gate import EvidenceGate, GateReport

__all__ = [
    "extract_claims",
    "split_sentences",
    "is_claim",
    "align_claims",
    "align_score",
    "coverage",
    "jaccard",
    "EvidenceGate",
    "GateReport",
]
