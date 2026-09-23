"""DP-4 dialectic protocol: propose, fact-check, criticise, synthesise.

Author: 晨星
"""
from .convergence import (
    ConvergenceTracker,
    filter_valid,
    parse_objections,
    parse_verdicts,
    stalled,
)
from .protocol import DialecticProtocol
from .roles import (
    CRITIC_SYSTEM,
    FACTCHECKER_SYSTEM,
    PROPOSER_SYSTEM,
    SYNTHESIZER_SYSTEM,
    critic_messages,
    factchecker_messages,
    proposer_messages,
    role_system,
    synthesizer_messages,
)

__all__ = [
    "DialecticProtocol",
    "ConvergenceTracker",
    "filter_valid",
    "parse_objections",
    "parse_verdicts",
    "stalled",
    "PROPOSER_SYSTEM",
    "FACTCHECKER_SYSTEM",
    "CRITIC_SYSTEM",
    "SYNTHESIZER_SYSTEM",
    "role_system",
    "proposer_messages",
    "factchecker_messages",
    "critic_messages",
    "synthesizer_messages",
]
