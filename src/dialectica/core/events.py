"""Event bus + deterministic trace recorder.

Author: 晨星
"""
from __future__ import annotations

from typing import Any, Callable

from .types import new_id


class EventBus:
    """Synchronous fan-out. Handlers must not raise."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self.trace: list[dict[str, Any]] = []
        self.trace_id = new_id("trc")

    def subscribe(self, name: str, handler: Callable[[dict[str, Any]], None]) -> None:
        self._handlers.setdefault(name, []).append(handler)

    def emit(self, name: str, **payload: Any) -> None:
        record = {"event": name, "trace_id": self.trace_id, **payload}
        self.trace.append(record)
        for handler in self._handlers.get(name, []):
            try:
                handler(record)
            except Exception:  # pragma: no cover - observers must never break flow
                continue
        for handler in self._handlers.get("*", []):
            try:
                handler(record)
            except Exception:  # pragma: no cover
                continue

    def events(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.trace if r["event"] == name]

    def reset(self) -> None:
        self.trace.clear()
        self.trace_id = new_id("trc")
