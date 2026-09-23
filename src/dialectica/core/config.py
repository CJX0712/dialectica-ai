"""Configuration. Every knob is env-driven; nothing is hard-coded.

Author: 晨星
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"DIALECTICA_{name}", default)


def _f(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _b(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass
class RetrievalConfig:
    chunk_size: int = field(default_factory=lambda: _i("CHUNK_SIZE", 480))
    chunk_overlap: int = field(default_factory=lambda: _i("CHUNK_OVERLAP", 80))
    top_k: int = field(default_factory=lambda: _i("TOP_K", 8))
    candidate_pool: int = field(default_factory=lambda: _i("CANDIDATE_POOL", 40))
    rrf_k: int = field(default_factory=lambda: _i("RRF_K", 60))
    dense_weight: float = field(default_factory=lambda: _f("DENSE_WEIGHT", 0.5))
    adaptive: bool = field(default_factory=lambda: _b("ADAPTIVE_FUSION", True))
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "hashing"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "memory"))


@dataclass
class RerankConfig:
    enabled: bool = field(default_factory=lambda: _b("RERANK_ENABLED", True))
    backend: str = field(default_factory=lambda: _env("RERANK_BACKEND", "heuristic"))
    model: str = field(default_factory=lambda: _env("RERANK_MODEL", "Xenova/bge-reranker-base"))
    alpha: float = field(default_factory=lambda: _f("RERANK_ALPHA", 0.5))
    z_floor: float = field(default_factory=lambda: _f("RERANK_Z_FLOOR", -0.5))
    min_margin: float = field(default_factory=lambda: _f("RERANK_MIN_MARGIN", 0.15))
    top_n: int = field(default_factory=lambda: _i("RERANK_TOP_N", 8))


@dataclass
class EvidenceConfig:
    support_threshold: float = field(default_factory=lambda: _f("SUPPORT_THRESHOLD", 0.30))
    max_unanchored_ratio: float = field(default_factory=lambda: _f("MAX_UNANCHORED_RATIO", 0.20))
    prune_unanchored: bool = field(default_factory=lambda: _b("PRUNE_UNANCHORED", True))


@dataclass
class DebateConfig:
    max_rounds: int = field(default_factory=lambda: _i("MAX_ROUNDS", 3))
    convergence_eps: float = field(default_factory=lambda: _f("CONVERGENCE_EPS", 0.03))
    self_consistency: int = field(default_factory=lambda: _i("SELF_CONSISTENCY", 1))
    gap_retrieval: bool = field(default_factory=lambda: _b("GAP_RETRIEVAL", True))


@dataclass
class LLMConfig:
    backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "mock"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "mock-1"))
    gguf_path: str = field(default_factory=lambda: _env("GGUF_PATH", ""))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", ""))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    temperature: float = field(default_factory=lambda: _f("TEMPERATURE", 0.0))
    max_tokens: int = field(default_factory=lambda: _i("MAX_TOKENS", 512))
    n_ctx: int = field(default_factory=lambda: _i("N_CTX", 4096))
    n_threads: int = field(default_factory=lambda: _i("N_THREADS", 4))


@dataclass
class MemoryConfig:
    enabled: bool = field(default_factory=lambda: _b("MEMORY_ENABLED", True))
    working_size: int = field(default_factory=lambda: _i("WORKING_SIZE", 16))
    decay_lambda: float = field(default_factory=lambda: _f("DECAY_LAMBDA", 0.02))
    cluster_threshold: float = field(default_factory=lambda: _f("CLUSTER_THRESHOLD", 0.86))
    min_support: int = field(default_factory=lambda: _i("MIN_SUPPORT", 2))


@dataclass
class APIConfig:
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _i("PORT", 8000))
    cors_origins: str = field(default_factory=lambda: _env("CORS_ORIGINS", "*"))


@dataclass
class Settings:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    offline: bool = field(default_factory=lambda: _b("OFFLINE", True))

    @classmethod
    def load(cls) -> "Settings":
        return cls()
