"""Emit the built-in benchmark report + retrieval diagnostics as JSON.

Used to keep the numbers quoted in README.md honest: they are produced by this
script, not typed by hand.

Author: 晨星
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dialectica.core.config import Settings  # noqa: E402
from dialectica.core.types import Document  # noqa: E402
from dialectica.eval.benchmark import run_benchmark  # noqa: E402
from dialectica.retrieval.dense import HashingEmbedder  # noqa: E402
from dialectica.system import Dialectica  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark.json")


def main() -> int:
    settings = Settings.load()
    embedder = HashingEmbedder(dim=2048)
    report = run_benchmark(lambda: Dialectica(settings), embedder)

    # retrieval diagnostics for one representative query
    probe = Dialectica(settings)
    probe.add_documents(
        [
            Document.create(
                "沙箱守卫会拦截对 site-packages 的写入与删除，pip install 会报 SAFE_DELETE_BULK_GUARD_ERROR。"
                "绕行方式是使用 pip install --target 指定目录，再通过 PYTHONPATH 加载。",
                title="pip",
            ),
            Document.create(
                "机器上的 SOCKS5 代理会让 pip 报 Missing dependencies for SOCKS support，"
                "需要先安装 PySocks 再继续安装其它依赖。",
                title="socks",
            ),
        ]
    )
    result = probe.debate("沙箱里为什么无法安装 Python 包")

    payload = {
        "benchmark": report.as_dict(),
        "probe": {
            "converged": result.converged,
            "rounds": result.round_count,
            "metrics": result.metrics,
            "trace_events": len(probe.bus.trace),
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
