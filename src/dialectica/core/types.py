"""Core value types. Pure data, zero dependencies, zero business logic.

Author: 晨星
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ms() -> float:
    return time.time() * 1000.0


# --------------------------------------------------------------------------
# Documents / retrieval
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """A source document handed to the ingestion layer."""

    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(text: str, title: str = "", source: str = "", **meta: Any) -> "Document":
        return Document(
            doc_id=new_id("doc"),
            text=text,
            title=title or source or "untitled",
            source=source,
            metadata=dict(meta),
        )


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit. `span_id` is the global evidence handle."""

    chunk_id: str
    span_id: str
    doc_id: str
    text: str
    order: int = 0
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(doc: Document, text: str, order: int, start: int, end: int) -> "Chunk":
        cid = new_id("chk")
        return Chunk(
            chunk_id=cid,
            span_id=f"span::{cid}",
            doc_id=doc.doc_id,
            text=text,
            order=order,
            start=start,
            end=end,
            metadata={"title": doc.title, "source": doc.source, **doc.metadata},
        )


@dataclass(frozen=True)
class Hit:
    """One candidate returned by the retrieval pipeline."""

    chunk: Chunk
    score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    @property
    def span_id(self) -> str:
        return self.chunk.span_id

    @property
    def text(self) -> str:
        return self.chunk.text


# --------------------------------------------------------------------------
# Evidence / claims
# --------------------------------------------------------------------------


class Verdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"


@dataclass
class Claim:
    """A single assertive statement extracted from generated text."""

    claim_id: str
    text: str
    verdict: Verdict = Verdict.UNCERTAIN
    evidence: list[str] = field(default_factory=list)  # span ids
    support: float = 0.0

    @property
    def anchored(self) -> bool:
        return self.verdict is Verdict.SUPPORTED and bool(self.evidence)

    @staticmethod
    def create(text: str) -> "Claim":
        return Claim(claim_id=new_id("clm"), text=text)


# --------------------------------------------------------------------------
# Debate
# --------------------------------------------------------------------------


class Role(str, Enum):
    PROPOSER = "proposer"
    FACTCHECKER = "factchecker"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


@dataclass
class Objection:
    """A critique raised by the Critic.

    An objection is *valid* only when it cites at least one evidence span id
    or quotes a concrete contradiction; otherwise it is discarded by the
    protocol. This is the guardrail that stops an unfounded critic from
    hijacking the debate.
    """

    objection_id: str
    text: str
    citations: list[str] = field(default_factory=list)
    severity: float = 1.0

    @property
    def valid(self) -> bool:
        return bool(self.citations) and bool(self.text.strip())

    @staticmethod
    def create(text: str, citations: list[str] | None = None, severity: float = 1.0) -> "Objection":
        return Objection(
            objection_id=new_id("obj"), text=text, citations=list(citations or []), severity=severity
        )


@dataclass
class DebateRound:
    index: int
    proposal: str = ""
    claims: list[Claim] = field(default_factory=list)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    objections: list[Objection] = field(default_factory=list)
    valid_objections: list[Objection] = field(default_factory=list)
    gap_queries: list[str] = field(default_factory=list)
    delta: float = 1.0  # semantic change vs previous round

    @property
    def unanchored(self) -> list[Claim]:
        return [c for c in self.claims if not c.anchored]


@dataclass
class DebateResult:
    answer: str
    rounds: list[DebateRound] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    converged: bool = False
    trace_id: str = field(default_factory=lambda: new_id("trc"))
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def round_count(self) -> int:
        return len(self.rounds)


# --------------------------------------------------------------------------
# LLM message envelope
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Message:
    role: str  # system | user | assistant
    content: str

    @staticmethod
    def system(content: str) -> "Message":
        return Message("system", content)

    @staticmethod
    def user(content: str) -> "Message":
        return Message("user", content)

    @staticmethod
    def assistant(content: str) -> "Message":
        return Message("assistant", content)


@dataclass(frozen=True)
class Generation:
    text: str
    model: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


@dataclass
class Episode:
    episode_id: str
    text: str
    created_ms: float = field(default_factory=now_ms)
    salience: float = 1.0
    hits: int = 0

    @staticmethod
    def create(text: str, salience: float = 1.0) -> "Episode":
        return Episode(episode_id=new_id("ep"), text=text, salience=salience)


@dataclass
class Fact:
    """Distilled semantic memory entry."""

    fact_id: str
    text: str
    support: int = 1
    created_ms: float = field(default_factory=now_ms)

    @staticmethod
    def create(text: str, support: int = 1) -> "Fact":
        return Fact(fact_id=new_id("fact"), text=text, support=support)
