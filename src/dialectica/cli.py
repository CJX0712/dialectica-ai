"""Command line interface.

    dialectica serve --port 8000
    dialectica ingest --file corpus.txt
    dialectica ask "沙箱里为什么无法安装 Python 包"
    dialectica debate "..." --json
    dialectica evaluate
    dialectica verify

Author: 晨星
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_corpus(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"corpus not found: {path}")
    if p.suffix.lower() in {".json", ".jsonl"}:
        raw = p.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in data]
        except Exception:
            pass
        return [ln for ln in raw.splitlines() if ln.strip()]
    return [b for b in p.read_text(encoding="utf-8").split("\n\n") if b.strip()]


def _pdf_text(path: str) -> list[str]:
    try:
        from pypdf import PdfReader  # optional dependency
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"pypdf is required for PDF ingestion: {exc}") from exc
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dialectica", description="Dialectica CLI (作者: 晨星)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="start the HTTP gateway")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")

    p_ingest = sub.add_parser("ingest", help="ingest a corpus")
    p_ingest.add_argument("--file", required=True)
    p_ingest.add_argument("--pdf", action="store_true", help="treat the file as PDF")

    p_ask = sub.add_parser("ask", help="ask a question (tool route or debate)")
    p_ask.add_argument("query")

    p_debate = sub.add_parser("debate", help="run the full dialectic protocol")
    p_debate.add_argument("query")
    p_debate.add_argument("--top-k", type=int, default=None)
    p_debate.add_argument("--json", action="store_true")

    sub.add_parser("evaluate", help="run the built-in regression benchmark")
    sub.add_parser("health", help="print system health")

    args = parser.parse_args(argv)

    if args.command == "serve":
        from .api.app import serve

        serve(args.host, args.port)
        return 0

    from .core.types import Document
    from .system import Dialectica

    system = Dialectica()

    if args.command == "ingest":
        texts = _pdf_text(args.file) if args.pdf else _read_corpus(args.file)
        n = system.add_documents([Document.create(t, source=args.file) for t in texts if t.strip()])
        print(f"ingested {n} chunks from {len(texts)} blocks")
        return 0

    if args.command == "ask":
        print(system.ask(args.query))
        return 0

    if args.command == "debate":
        result = system.debate(args.query, args.top_k)
        if args.json:
            print(
                json.dumps(
                    {
                        "answer": result.answer,
                        "converged": result.converged,
                        "metrics": result.metrics,
                        "rounds": [
                            {
                                "index": r.index,
                                "proposal": r.proposal,
                                "delta": r.delta,
                                "objections": [o.text for o in r.valid_objections],
                                "claims": [
                                    {
                                        "text": c.text,
                                        "verdict": c.verdict.value,
                                        "evidence": c.evidence,
                                        "support": c.support,
                                    }
                                    for c in r.claims
                                ],
                            }
                            for r in result.rounds
                        ],
                        "evidence": [
                            {"span_id": h.span_id, "score": h.score, "text": h.chunk.text}
                            for h in result.hits
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"答案：{result.answer}\n")
            print(f"收敛：{result.converged}   轮数：{result.round_count}")
            for key, val in result.metrics.items():
                print(f"  {key}: {val}")
            print("\n证据：")
            for h in result.hits:
                print(f"  [{h.score:.4f}] {h.span_id} {h.chunk.text[:80]}")
        return 0

    if args.command == "evaluate":
        from .eval.benchmark import run_benchmark

        report = run_benchmark(lambda: Dialectica(), system.embedder)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0 if report.passed == report.total else 1

    if args.command == "health":
        print(json.dumps(system.health(), ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
