"""Dialectica -- 辩衡：可自证的辩证式多智能体推理系统。

Deliberative multi-agent reasoning with evidence anchoring and provable
convergence. Local-first, CPU-only, offline-verifiable.

Author: 晨星
"""
from .core.config import Settings
from .core.types import (
    Claim,
    DebateResult,
    DebateRound,
    Document,
    Hit,
    Objection,
    Role,
    Verdict,
)
from .system import Dialectica, build_system

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "Dialectica",
    "build_system",
    "Settings",
    "Document",
    "Hit",
    "Claim",
    "Objection",
    "Verdict",
    "Role",
    "DebateRound",
    "DebateResult",
    "__version__",
    "__author__",
]
