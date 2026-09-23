"""Production LLM backends: llama.cpp (local GGUF), Ollama, OpenAI-compatible.

All three are optional. Import is lazy and failure is reported as
`ProviderUnavailable` so the offline path never breaks at import time.

Author: 晨星
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from ..core.errors import ProviderUnavailable
from ..core.types import Generation, Message


def _to_payload(messages: Sequence[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class LlamaCppLLM:
    """Local GGUF inference through llama-cpp-python. Fully offline."""

    name = "llamacpp"

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 4,
        model: str = "gguf",
    ) -> None:
        if not model_path:
            raise ProviderUnavailable("DIALECTICA_GGUF_PATH is empty")
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ProviderUnavailable(f"llama-cpp-python not installed: {exc}") from exc
        # Thread count matters: small quantised models are memory-bandwidth
        # bound, so oversubscribing threads makes them dramatically slower.
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=max(1, n_threads), verbose=False)
        self.model = model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
    ) -> Generation:
        out = self._llm.create_chat_completion(
            messages=_to_payload(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stop=list(stop or []),
        )
        text = out["choices"][0]["message"]["content"] or ""
        return Generation(text=text.strip(), model=self.model, raw=out)

    def stream(self, messages: Sequence[Message], **kwargs: Any) -> Iterable[str]:
        for part in self._llm.create_chat_completion(
            messages=_to_payload(messages), stream=True, **kwargs
        ):
            delta = part.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                yield delta["content"]


class OllamaLLM:
    """Ollama's OpenAI-compatible chat endpoint."""

    name = "ollama"

    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen2.5:7b") -> None:
        self.host = host.rstrip("/")
        self.model = model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
    ) -> Generation:
        import httpx  # local import keeps offline path clean

        # Localhost traffic must bypass the machine's SOCKS/HTTP proxy.
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            r = client.post(
                f"{self.host}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": _to_payload(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        return Generation(text=text.strip(), model=self.model, raw=data)

    def stream(self, messages: Sequence[Message], **kwargs: Any) -> Iterable[str]:
        import httpx

        with httpx.Client(timeout=120.0, trust_env=False) as client:
            with client.stream(
                "POST",
                f"{self.host}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": _to_payload(messages),
                    "stream": True,
                    **kwargs,
                },
            ) as r:
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                    except Exception:
                        continue
                    if delta.get("content"):
                        yield delta["content"]


class OpenAICompatLLM:
    """Any OpenAI-compatible endpoint (vLLM, OpenAI, Together, DeepSeek, ...)."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str = "", model: str = "gpt-4o-mini") -> None:
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.model = model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stop: Sequence[str] | None = None,
    ) -> Generation:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": _to_payload(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        return Generation(text=text.strip(), model=self.model, raw=data)

    def stream(self, messages: Sequence[Message], **kwargs: Any) -> Iterable[str]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": _to_payload(messages),
                    "stream": True,
                    **kwargs,
                },
            ) as r:
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"]
                    except Exception:
                        continue
                    if delta.get("content"):
                        yield delta["content"]
