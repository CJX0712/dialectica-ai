"""DP-4: the dialectic protocol (Proposer / FactChecker / Critic / Synthesizer).

Control flow
------------
    for round in 0..max_rounds-1:
        proposal  <- Proposer(query, evidence, objections)
        claims    <- extract(proposal)
        verdicts  <- FactChecker(claims, evidence)      # advisory
        aligned   <- align(claims, evidence)            # authoritative
        objections<- filter_valid(Critic(...), known_spans)
        delta     <- ConvergenceTracker.delta(...)
        if no objections: converged, break
        if stalled(objections, prev): break
        if gap_retrieval: evidence += retrieve(unanchored claim text)
    answer <- Synthesizer(gate(claims))

Alignment, not the model's self-report, decides what counts as supported: a
model that declares its own claim SUPPORTED is not evidence.

Author: 晨星
"""
from __future__ import annotations

from typing import Sequence

from ..core.config import DebateConfig, EvidenceConfig
from ..core.types import (
    Claim,
    DebateResult,
    DebateRound,
    Generation,
    Hit,
    Objection,
    Verdict,
)
from ..evidence.aligner import align_claims
from ..evidence.extractor import extract_claims
from ..evidence.gate import EvidenceGate
from ..llm.mock import MockLLM
from .convergence import (
    ConvergenceTracker,
    filter_valid,
    parse_objections,
    parse_verdicts,
    stalled,
)
from .roles import (
    critic_messages,
    factchecker_messages,
    proposer_messages,
    synthesizer_messages,
)


class DialecticProtocol:
    """Orchestrates the four roles. Every collaborator is injected."""

    def __init__(
        self,
        llm,
        retriever,
        embedder,
        *,
        config: DebateConfig | None = None,
        evidence_config: EvidenceConfig | None = None,
        gate: EvidenceGate | None = None,
        bus=None,
    ) -> None:
        self.llm = llm or MockLLM()
        self.retriever = retriever
        self.embedder = embedder
        self.config = config or DebateConfig()
        self.gate = gate or EvidenceGate(evidence_config)
        self.bus = bus
        self.tracker = ConvergenceTracker(embedder)

    # -- helpers -----------------------------------------------------------
    def _emit(self, name: str, **payload) -> None:
        if self.bus is not None:
            self.bus.emit(name, **payload)

    def _complete(self, messages: Sequence) -> str:
        gen: Generation = self.llm.complete(list(messages), temperature=0.0)
        return gen.text or ""

    def _self_consistent(self, messages: Sequence) -> str:
        """Sample N times at T>0 and take the majority string (when N > 1)."""
        if self.config.self_consistency <= 1:
            return self._complete(messages)
        from collections import Counter

        votes = Counter(
            self.llm.complete(list(messages), temperature=0.7).text.strip()
            for _ in range(self.config.self_consistency)
        )
        return votes.most_common(1)[0][0]

    # -- main loop ---------------------------------------------------------
    def run(self, query: str, top_k: int | None = None) -> DebateResult:
        hits: list[Hit] = list(self.retriever.retrieve(query, top_k))
        context = self.retriever.context_block(hits)
        known = [h.span_id for h in hits]
        self._emit("retrieval.done", query=query, hits=len(hits), signals=getattr(self.retriever, "last_signals", {}))

        rounds: list[DebateRound] = []
        prev_objections: list[Objection] = []
        converged = False
        final_claims: list[Claim] = []

        for index in range(self.config.max_rounds):
            proposal = self._self_consistent(proposer_messages(query, context, prev_objections))
            claims = extract_claims(proposal)
            claims = align_claims(claims, hits, self.gate.config.support_threshold)

            verdict_text = self._complete(factchecker_messages([c.text for c in claims], context))
            verdicts = parse_verdicts(verdict_text)
            for i, c in enumerate(claims):
                v, span = verdicts.get(i, ("", ""))
                if v == "SUPPORTED" and span and c.verdict is Verdict.SUPPORTED:
                    c.evidence = [span] if span in known else c.evidence
                elif v == "UNSUPPORTED":
                    # The checker's opinion is advisory: alignment decides.
                    pass

            critique = self._complete(critic_messages(query, proposal, context))
            objections = filter_valid(parse_objections(critique), known)
            delta = self.tracker.delta(index, proposal)

            rnd = DebateRound(
                index=index,
                proposal=proposal,
                claims=claims,
                verdicts={c.claim_id: c.verdict for c in claims},
                objections=parse_objections(critique),
                valid_objections=objections,
                delta=delta,
            )
            rounds.append(rnd)
            self._emit(
                "debate.round",
                index=index,
                claims=len(claims),
                anchored=sum(1 for c in claims if c.anchored),
                objections=len(objections),
                delta=delta,
            )

            if not objections:
                converged = True
                final_claims = claims
                break
            if stalled(prev_objections, objections):
                final_claims = claims
                break

            if self.config.gap_retrieval:
                gaps = [c.text for c in claims if not c.anchored][:2]
                if gaps:
                    extra: list[Hit] = []
                    for g in gaps:
                        try:
                            extra.extend(self.retriever.retrieve(g, top_k))
                        except Exception:
                            continue
                    seen = set(known)
                    for h in extra:
                        if h.span_id not in seen:
                            hits.append(h)
                            seen.add(h.span_id)
                    context = self.retriever.context_block(hits)
                    known = list(seen)
                    rnd.gap_queries = gaps

            prev_objections = objections
            final_claims = claims

        kept, report = self.gate.apply(final_claims)
        answer = self._complete(synthesizer_messages([c.text for c in kept])) if kept else ""

        result = DebateResult(
            answer=answer,
            rounds=rounds,
            hits=hits,
            claims=kept,
            converged=converged,
        )
        result.metrics = {
            "rounds": float(len(rounds)),
            "converged": 1.0 if converged else 0.0,
            "stability": self.tracker.stability(),
            "claims_total": float(report.total),
            "claims_anchored": float(report.anchored),
            "faithfulness": (report.anchored / report.total) if report.total else 0.0,
            "evidence_spans": float(len(hits)),
            **{f"retrieval_{k}": float(v) for k, v in getattr(self.retriever, "last_signals", {}).items()},
        }
        self._emit("debate.done", converged=converged, rounds=len(rounds), faithfulness=result.metrics["faithfulness"])
        return result
