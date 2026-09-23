"""Retrieval: chunking, BM25, dense embeddings, adaptive fusion, calibrated rerank.

Author: 晨星
"""
from .chunker import chunk_document
from .dense import (
    FaissIndex,
    FastembedEmbedder,
    HashingEmbedder,
    MemoryIndex,
    build_embedder,
    build_index,
    cosine,
)
from .fusion import derive_weights, shannon_entropy, weighted_rrf
from .pipeline import RetrievalPipeline
from .rerank import CalibratedReranker, CrossEncoderReranker, HeuristicReranker, z_scores
from .sparse import BM25
from .tokenizer import tokenize

__all__ = [
    "chunk_document",
    "tokenize",
    "BM25",
    "HashingEmbedder",
    "FastembedEmbedder",
    "MemoryIndex",
    "FaissIndex",
    "build_embedder",
    "build_index",
    "cosine",
    "derive_weights",
    "shannon_entropy",
    "weighted_rrf",
    "HeuristicReranker",
    "CrossEncoderReranker",
    "CalibratedReranker",
    "z_scores",
    "RetrievalPipeline",
]
