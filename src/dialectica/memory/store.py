"""Three-layer memory: working / episodic / semantic.

  working  -- bounded FIFO of the current session's turns
  episodic -- timestamped records with an exponential forgetting curve:
              `score = salience * exp(-lambda * hours_elapsed) * cosine(q, e)`
  semantic -- distilled facts, each carrying a support count; distillation is
              idempotent (running it twice changes nothing the second time)

Author: 晨星
"""
from __future__ import annotations

import math
from collections import deque
from typing import Sequence

from ..core.config import MemoryConfig
from ..core.types import Episode, Fact
from ..retrieval.dense import cosine


class MemoryStore:
    """Implements the `Memory` protocol."""

    def __init__(self, embedder, config: MemoryConfig | None = None) -> None:
        self.embedder = embedder
        self.config = config or MemoryConfig()
        self.working: deque[str] = deque(maxlen=self.config.working_size)
        self.episodic: list[Episode] = []
        self.semantic: list[Fact] = []
        self._vecs: dict[str, list[float]] = {}

    # -- write -------------------------------------------------------------
    def remember(self, text: str, *, salience: float = 1.0) -> None:
        if not text or not text.strip():
            return
        self.working.append(text.strip())
        ep = Episode.create(text.strip(), salience=salience)
        self.episodic.append(ep)
        self._vecs[ep.episode_id] = self.embedder.embed_one(ep.text)

    # -- read --------------------------------------------------------------
    def _decay(self, ep: Episode, now_ms: float) -> float:
        hours = max(0.0, (now_ms - ep.created_ms) / 3_600_000.0)
        return math.exp(-self.config.decay_lambda * hours)

    def recall(self, query: str, top_k: int = 5) -> list[str]:
        if not self.episodic:
            return []
        import time

        now = time.time() * 1000.0
        qv = self.embedder.embed_one(query)
        scored = []
        for ep in self.episodic:
            sim = cosine(qv, self._vecs.get(ep.episode_id, []))
            score = ep.salience * self._decay(ep, now) * (0.5 + 0.5 * max(0.0, sim))
            scored.append((score, ep))
        scored.sort(key=lambda t: (-t[0], t[1].episode_id))
        out = []
        for score, ep in scored[:top_k]:
            ep.hits += 1
            out.append(ep.text)
        return out

    def semantic_recall(self, query: str, top_k: int = 3) -> list[str]:
        if not self.semantic:
            return []
        qv = self.embedder.embed_one(query)
        scored = sorted(
            ((cosine(qv, self.embedder.embed_one(f.text)), f) for f in self.semantic),
            key=lambda t: (-t[0], t[1].fact_id),
        )
        return [f.text for _, f in scored[:top_k]]

    # -- distillation ------------------------------------------------------
    def distill(self) -> int:
        """Cluster episodic records by cosine similarity into semantic facts.

        Invariant: idempotent -- a second call returns 0 when nothing changed.
        """
        added = 0
        used: set[str] = set()
        for i, ep in enumerate(self.episodic):
            if ep.episode_id in used:
                continue
            cluster = [ep]
            used.add(ep.episode_id)
            for other in self.episodic[i + 1 :]:
                if other.episode_id in used:
                    continue
                if cosine(self._vecs[ep.episode_id], self._vecs[other.episode_id]) >= self.config.cluster_threshold:
                    cluster.append(other)
                    used.add(other.episode_id)
            if len(cluster) < self.config.min_support:
                continue
            if self._has_fact(cluster[0].text):
                continue
            fact = Fact.create(cluster[0].text, support=len(cluster))
            self.semantic.append(fact)
            added += 1
        return added

    def _has_fact(self, text: str) -> bool:
        return any(f.text == text for f in self.semantic)

    # -- maintenance -------------------------------------------------------
    def forget(self, threshold: float = 0.05) -> int:
        """Drop episodic records decayed below `threshold`."""
        import time

        now = time.time() * 1000.0
        keep = []
        removed = 0
        for ep in self.episodic:
            if ep.salience * self._decay(ep, now) >= threshold:
                keep.append(ep)
            else:
                self._vecs.pop(ep.episode_id, None)
                removed += 1
        self.episodic = keep
        return removed

    def clear(self) -> None:
        self.working.clear()
        self.episodic.clear()
        self.semantic.clear()
        self._vecs.clear()

    def stats(self) -> dict[str, int]:
        return {
            "working": len(self.working),
            "episodic": len(self.episodic),
            "semantic": len(self.semantic),
        }

    def snapshot(self) -> Sequence[str]:
        return list(self.working)
