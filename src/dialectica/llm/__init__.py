"""LLM provider layer: mock (default, offline) + llama.cpp / Ollama / OpenAI.

Author: 晨星
"""
from .factory import build_llm
from .mock import MockLLM

__all__ = ["build_llm", "MockLLM"]
