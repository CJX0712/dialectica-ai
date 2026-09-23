"""FastAPI gateway.

Routes are thin: parse -> delegate to `Dialectica` -> serialise. No business
logic here, so the HTTP layer can never disagree with the CLI.

Author: 晨星
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.config import Settings
from ..core.errors import DialecticaError
from ..core.types import Document
from ..eval.benchmark import run_benchmark
from ..system import Dialectica

_DESCRIPTION = """Dialectica (辩衡) -- deliberative multi-agent reasoning.

Every answer is produced by a four-role dialectic protocol, every claim is
anchored to a retrieved evidence span, and the run is fully traceable.
"""


class IngestRequest(BaseModel):
    docs: list[dict[str, Any]] = Field(default_factory=list)
    texts: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    converged: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    rounds: list[dict[str, Any]] = Field(default_factory=list)


def _serialize_claim(c) -> dict[str, Any]:
    return {
        "claim_id": c.claim_id,
        "text": c.text,
        "verdict": c.verdict.value if hasattr(c.verdict, "value") else str(c.verdict),
        "evidence": c.evidence,
        "support": c.support,
        "anchored": c.anchored,
    }


def _serialize_hit(h) -> dict[str, Any]:
    return {
        "span_id": h.span_id,
        "text": h.chunk.text,
        "score": h.score,
        "doc_id": h.chunk.doc_id,
        "title": h.chunk.metadata.get("title", ""),
        "components": h.components,
    }


def _serialize_round(r) -> dict[str, Any]:
    return {
        "index": r.index,
        "proposal": r.proposal,
        "delta": r.delta,
        "claims": [_serialize_claim(c) for c in r.claims],
        "objections": [
            {"text": o.text, "citations": o.citations, "valid": o.valid} for o in r.objections
        ],
        "valid_objections": [
            {"text": o.text, "citations": o.citations} for o in r.valid_objections
        ],
        "gap_queries": r.gap_queries,
    }


def _serialize_result(result) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        converged=result.converged,
        metrics=result.metrics,
        claims=[_serialize_claim(c) for c in result.claims],
        evidence=[_serialize_hit(h) for h in result.hits],
        rounds=[_serialize_round(r) for r in result.rounds],
    )


def create_app(system: Dialectica | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    app = FastAPI(
        title="Dialectica",
        description=_DESCRIPTION,
        version="0.1.0",
        contact={"name": "晨星"},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.api.cors_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    holder: dict[str, Dialectica] = {"system": system or Dialectica(settings)}

    def get_system() -> Dialectica:
        return holder["system"]

    @app.get("/health")
    def health():
        return get_system().health()

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        sys_ = get_system()
        docs: list[Document] = []
        for d in req.docs:
            text = str(d.get("text", ""))
            if not text:
                continue
            docs.append(
                Document.create(
                    text,
                    title=str(d.get("title", "")),
                    source=str(d.get("source", "")),
                )
            )
        for t in req.texts:
            if t:
                docs.append(Document.create(t))
        if not docs:
            raise HTTPException(status_code=400, detail="no documents supplied")
        return {"chunks": sys_.add_documents(docs), "documents": len(docs)}

    @app.post("/reset")
    def reset():
        sys_ = get_system()
        sys_.retriever.clear()
        sys_.memory.clear()
        return {"status": "ok"}

    @app.post("/debate", response_model=QueryResponse)
    def debate(req: QueryRequest):
        sys_ = get_system()
        try:
            result = sys_.debate(req.query, req.top_k)
        except DialecticaError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.code) from exc
        return _serialize_result(result)

    @app.post("/ask")
    def ask(req: QueryRequest):
        sys_ = get_system()
        try:
            return {"answer": sys_.ask(req.query)}
        except DialecticaError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.code) from exc

    @app.post("/debate/stream")
    def debate_stream(req: QueryRequest):
        sys_ = get_system()
        result = sys_.debate(req.query, req.top_k)
        payload = _serialize_result(result).model_dump()

        def gen():
            yield f"event: evidence\ndata: {json.dumps(payload['evidence'], ensure_ascii=False)}\n\n"
            for rnd in payload["rounds"]:
                yield f"event: round\ndata: {json.dumps(rnd, ensure_ascii=False)}\n\n"
            yield f"event: answer\ndata: {json.dumps({'answer': payload['answer'], 'converged': payload['converged'], 'metrics': payload['metrics']}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/trace")
    def trace(limit: int = 200):
        bus = get_system().bus
        if bus is None:
            return {"events": []}
        return {"trace_id": bus.trace_id, "events": bus.trace[-limit:]}

    @app.post("/evaluate")
    def evaluate():
        sys_ = get_system()
        report = run_benchmark(lambda: Dialectica(settings), sys_.embedder)
        return report.as_dict()

    @app.get("/config")
    def config():
        s = get_system().settings
        return {
            "llm_backend": s.llm.backend,
            "embedding_backend": s.retrieval.embedding_backend,
            "vector_backend": s.retrieval.vector_backend,
            "rerank_enabled": s.rerank.enabled,
            "rerank_backend": s.rerank.backend,
            "max_rounds": s.debate.max_rounds,
            "offline": s.offline,
        }

    @app.get("/")
    def index():
        return {
            "name": "Dialectica",
            "version": "0.1.0",
            "author": "晨星",
            "endpoints": [
                "/health",
                "/ingest",
                "/reset",
                "/debate",
                "/debate/stream",
                "/ask",
                "/trace",
                "/evaluate",
                "/config",
            ],
        }

    return app


def build_default_app() -> FastAPI:
    return create_app()


app = build_default_app()


def serve(host: str | None = None, port: int | None = None) -> None:  # pragma: no cover
    import uvicorn

    s = Settings.load()
    uvicorn.run(app, host=host or s.api.host, port=port or s.api.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    serve(os.environ.get("DIALECTICA_HOST"), int(os.environ.get("DIALECTICA_PORT", "8000")))
