"""Composition root. Wiring only -- no business logic lives here.

    Dialectica()
      .add_documents([...])
      .debate("...")  ->  DebateResult

Every dependency is injectable; the defaults are the offline zero-dependency
implementations, so `Dialectica()` works with no network, no API key and no
model download.

Author: 晨星
"""
from __future__ import annotations

from typing import Iterable

from .core.config import Settings
from .core.events import EventBus
from .core.types import DebateResult, Document
from .debate.protocol import DialecticProtocol
from .evidence.gate import EvidenceGate
from .llm.factory import build_llm
from .memory.store import MemoryStore
from .reason.react import ReActAgent
from .reason.router import route
from .retrieval.dense import build_embedder
from .retrieval.pipeline import RetrievalPipeline


class Dialectica:
    """The assembled system."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm=None,
        embedder=None,
        retriever=None,
        memory=None,
        bus=None,
    ) -> None:
        self.settings = settings or Settings.load()
        # The trace bus is always on: a run that cannot be replayed cannot be
        # audited, and auditing is the whole point of the protocol.
        self.bus = bus or EventBus()
        self.embedder = embedder or build_embedder(
            self.settings.retrieval.embedding_backend, self.settings.retrieval.embedding_model
        )
        self.llm = llm or build_llm(self.settings.llm)
        self.retriever = retriever or RetrievalPipeline(
            embedder=self.embedder,
            config=self.settings.retrieval,
            rerank_config=self.settings.rerank,
        )
        self.gate = EvidenceGate(self.settings.evidence)
        self.protocol = DialecticProtocol(
            llm=self.llm,
            retriever=self.retriever,
            embedder=self.embedder,
            config=self.settings.debate,
            gate=self.gate,
            bus=self.bus,
        )
        self.memory = memory or MemoryStore(self.embedder, self.settings.memory)
        self.agent = ReActAgent(self.llm)

    # -- ingestion ---------------------------------------------------------
    def add_documents(self, docs: Iterable[Document]) -> int:
        return self.retriever.add_documents(docs)

    def add_texts(self, texts: Iterable[str]) -> int:
        return self.add_documents([Document.create(t) for t in texts])

    def __len__(self) -> int:
        return len(self.retriever)

    # -- inference ---------------------------------------------------------
    def debate(self, query: str, top_k: int | None = None) -> DebateResult:
        decision = route(query)
        if decision.is_tool:
            answer = self.agent.run(query)
            result = DebateResult(answer=answer, converged=True)
            result.metrics = {"rounds": 0.0, "converged": 1.0, "route": 1.0, "faithfulness": 1.0}
            if self.settings.memory.enabled:
                self.memory.remember(f"Q: {query}\nA: {answer}")
            return result

        result = self.protocol.run(query, top_k)
        if self.settings.memory.enabled:
            self.memory.remember(f"Q: {query}\nA: {result.answer}")
        return result

    def ask(self, query: str) -> str:
        return self.debate(query).answer

    # -- ops ---------------------------------------------------------------
    def health(self) -> dict:
        return {
            "status": "ok",
            "chunks": len(self.retriever),
            "llm": getattr(self.llm, "name", "unknown"),
            "embedding": type(self.embedder).__name__,
            "vector": type(self.retriever.vector_index).__name__,
            "memory": self.memory.stats(),
            "settings": {
                "llm_backend": self.settings.llm.backend,
                "embedding_backend": self.settings.retrieval.embedding_backend,
                "vector_backend": self.settings.retrieval.vector_backend,
                "rerank_backend": self.settings.rerank.backend if self.settings.rerank.enabled else "off",
            },
        }


def build_system(settings: Settings | None = None, **overrides) -> Dialectica:
    return Dialectica(settings, **overrides)
