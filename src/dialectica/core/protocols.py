"""Interface contracts.

Every collaboration boundary in Dialectica is declared here as a
`typing.Protocol`. Modules depend on these protocols only -- never on a
concrete implementation -- so each module can be verified in isolation with a
deterministic fake, and the production implementation can be swapped by
configuration.

Author: 晨星
"""
from __future__ import annotations

from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from .types import Chunk, Generation, Hit, Message


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a unit-length vector."""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        """Default single-text convenience path."""
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """Dense nearest-neighbour store."""

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, vector: Sequence[float], top_k: int) -> list[tuple[str, float]]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


@runtime_checkable
class SparseIndex(Protocol):
    """Lexical store (BM25 family)."""

    def add(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-attention or heuristic re-scorer over a candidate set."""

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


@runtime_checkable
class Retriever(Protocol):
    """End-to-end retrieval: query -> ranked evidence."""

    def add_documents(self, docs: Iterable[Any]) -> int: ...

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]: ...


@runtime_checkable
class LLM(Protocol):
    """Text completion provider."""

    name: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
    ) -> Generation: ...

    def stream(self, messages: Sequence[Message], **kwargs: Any) -> Iterable[str]: ...


@runtime_checkable
class Tool(Protocol):
    """A callable capability exposed to the agent."""

    name: str
    description: str

    def invoke(self, argument: str) -> str: ...


@runtime_checkable
class Memory(Protocol):
    """Layered memory: working / episodic / semantic."""

    def remember(self, text: str, *, salience: float = 1.0) -> None: ...

    def recall(self, query: str, top_k: int = 5) -> list[str]: ...

    def distill(self) -> int: ...


@runtime_checkable
class Observer(Protocol):
    """Receives structured lifecycle events."""

    def on_event(self, name: str, payload: dict[str, Any]) -> None: ...
