"""Retrieval pipeline: chunk -> (BM25 | dense) -> adaptive fusion -> calibrated rerank.

Composition root for the retrieval module. Every collaborator is injected, so
the whole module can be unit-tested with fakes.

Author: 晨星
"""
from __future__ import annotations

from typing import Iterable, Sequence

from ..core.config import RerankConfig, RetrievalConfig
from ..core.errors import EmptyCorpus
from ..core.types import Chunk, Document, Hit
from .chunker import chunk_document
from .dense import build_embedder, build_index
from .fusion import derive_weights, weighted_rrf
from .rerank import CalibratedReranker, HeuristicReranker
from .sparse import BM25


class RetrievalPipeline:
    """Hybrid retriever. Implements the `Retriever` protocol."""

    def __init__(
        self,
        *,
        embedder=None,
        vector_index=None,
        sparse=None,
        reranker=None,
        config: RetrievalConfig | None = None,
        rerank_config: RerankConfig | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.rerank_config = rerank_config or RerankConfig()
        self.embedder = embedder or build_embedder(
            self.config.embedding_backend, self.config.embedding_model
        )
        self.sparse = sparse or BM25()
        self._chunks: dict[str, Chunk] = {}
        self.vector_index = vector_index or build_index(
            self.config.vector_backend, getattr(self.embedder, "dim", 384)
        )
        self.reranker = reranker or (
            CalibratedReranker(
                HeuristicReranker().score,
                alpha=self.rerank_config.alpha,
                z_floor=self.rerank_config.z_floor,
                min_margin=self.rerank_config.min_margin,
                top_n=self.rerank_config.top_n,
            )
            if self.rerank_config.enabled
            else None
        )
        self.last_signals: dict[str, float] = {}

    # -- ingestion ---------------------------------------------------------
    def add_documents(self, docs: Iterable[Document]) -> int:
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(
                chunk_document(doc, self.config.chunk_size, self.config.chunk_overlap)
            )
        if not chunks:
            return 0
        # Optional hook: statistical embedders (IDF-weighted hashing) need the
        # corpus before they can encode it meaningfully.
        fit = getattr(self.embedder, "fit", None)
        if callable(fit):
            fit([c.text for c in chunks])
        vectors = self.embedder.embed([c.text for c in chunks])
        self.vector_index.add(chunks, vectors)
        self.sparse.add(chunks)
        for c in chunks:
            self._chunks[c.span_id] = c
        return len(chunks)

    def clear(self) -> None:
        self.vector_index.clear()
        self.sparse.clear()
        self._chunks.clear()

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks.values())

    def chunk_by_span(self, span_id: str) -> Chunk | None:
        return self._chunks.get(span_id)

    # -- retrieval ---------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        if len(self._chunks) == 0:
            raise EmptyCorpus("corpus is empty; ingest documents first")
        k = top_k or self.config.top_k
        pool = max(k * 4, self.config.candidate_pool)

        qvec = self.embedder.embed_one(query)
        dense_pairs = self.vector_index.search(qvec, pool)
        sparse_pairs = self.sparse.search(query, pool)

        dense_texts = [self._chunks[sid].text for sid, _ in dense_pairs if sid in self._chunks]
        signals = derive_weights(
            query,
            [s for _, s in dense_pairs],
            [s for _, s in sparse_pairs],
            dense_texts,
            adaptive=self.config.adaptive,
        )
        self.last_signals = {
            "entropy": signals.entropy,
            "margin": signals.margin,
            "lexical": signals.lexical,
            "w_dense": signals.w_dense,
            "w_sparse": signals.w_sparse,
        }

        fused = weighted_rrf(
            dense_pairs,
            sparse_pairs,
            k=self.config.rrf_k,
            w_dense=signals.w_dense,
            w_sparse=signals.w_sparse,
            top_k=pool,
        )
        hits = [
            Hit(chunk=self._chunks[sid], score=round(score, 6), components={"fused": round(score, 6)})
            for sid, score in fused
            if sid in self._chunks
        ]
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits)
        return hits[:k]

    # -- prompt helpers ----------------------------------------------------
    def context_block(self, hits: Sequence[Hit]) -> str:
        """Render evidence for the LLM.

        The delimiter is a globally unique name (`<kb-context>`). It must never
        appear inside the instruction text -- a literal `<context>` inside a
        prompt template collides with the block and makes the model quote the
        instructions back as if they were evidence.
        """
        lines = []
        for h in hits:
            title = h.chunk.metadata.get("title", "")
            lines.append(f"{h.span_id} | {title} :: {h.text}")
        return "\n".join(lines)
