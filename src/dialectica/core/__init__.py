"""Dialectica core: contracts, types, config, events, errors.

Author: 晨星
"""
from .config import Settings
from .errors import (
    ConfigError,
    DebateDiverged,
    DialecticaError,
    EmptyCorpus,
    EvidenceGap,
    ProviderUnavailable,
    ToolError,
)
from .events import EventBus
from .types import (
    Chunk,
    Claim,
    DebateResult,
    DebateRound,
    Document,
    Episode,
    Fact,
    Generation,
    Hit,
    Message,
    Objection,
    Role,
    Verdict,
)

__all__ = [
    "Settings",
    "EventBus",
    "DialecticaError",
    "ConfigError",
    "ProviderUnavailable",
    "EmptyCorpus",
    "EvidenceGap",
    "DebateDiverged",
    "ToolError",
    "Document",
    "Chunk",
    "Hit",
    "Claim",
    "Objection",
    "Verdict",
    "Role",
    "DebateRound",
    "DebateResult",
    "Message",
    "Generation",
    "Episode",
    "Fact",
]
