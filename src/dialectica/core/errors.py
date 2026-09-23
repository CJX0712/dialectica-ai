"""Error taxonomy. One exception root so the API layer can map cleanly.

Author: 晨星
"""
from __future__ import annotations


class DialecticaError(Exception):
    """Root of the error tree."""

    code = "dialectica_error"
    status = 500


class ConfigError(DialecticaError):
    code = "config_error"
    status = 500


class ProviderUnavailable(DialecticaError):
    code = "provider_unavailable"
    status = 503


class EmptyCorpus(DialecticaError):
    code = "empty_corpus"
    status = 400


class EvidenceGap(DialecticaError):
    """Raised when generation cannot be anchored to any evidence."""

    code = "evidence_gap"
    status = 422


class DebateDiverged(DialecticaError):
    """Raised when the dialectic protocol fails to converge in budget."""

    code = "debate_diverged"
    status = 504


class ToolError(DialecticaError):
    code = "tool_error"
    status = 400
