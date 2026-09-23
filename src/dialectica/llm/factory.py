"""Provider factory. Selection is configuration, never code change.

Author: 晨星
"""
from __future__ import annotations

from ..core.config import LLMConfig
from ..core.errors import ConfigError
from .mock import MockLLM


def build_llm(cfg: LLMConfig):
    """Instantiate the configured provider.

    `mock` is always available; every other backend raises a typed error when
    its dependency or credential is missing.
    """
    backend = (cfg.backend or "mock").lower()
    if backend == "mock":
        return MockLLM(model=cfg.model or "mock-1")
    if backend in {"llamacpp", "llama_cpp", "gguf"}:
        from .remote import LlamaCppLLM

        return LlamaCppLLM(
            model_path=cfg.gguf_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            model=cfg.model or "gguf",
        )
    if backend == "ollama":
        from .remote import OllamaLLM

        return OllamaLLM(host=cfg.ollama_host, model=cfg.model or "qwen2.5:7b")
    if backend in {"openai", "openai_compat", "vllm"}:
        from .remote import OpenAICompatLLM

        return OpenAICompatLLM(
            base_url=cfg.openai_base_url, api_key=cfg.openai_api_key, model=cfg.model
        )
    raise ConfigError(f"unknown LLM backend: {backend}")
