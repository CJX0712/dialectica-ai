"""Quick manual smoke run. Not part of the test suite."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dialectica import Dialectica, Document  # noqa: E402

DOCS = [
    "沙箱守卫会拦截对 site-packages 的写入与删除，pip install 会报 SAFE_DELETE_BULK_GUARD_ERROR。"
    "绕行方式是使用 pip install --target 指定目录，再通过 PYTHONPATH 加载。",
    "机器上的 SOCKS5 代理会让 pip 报 Missing dependencies for SOCKS support，"
    "需要先安装 PySocks 再继续安装其它依赖。",
]


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    buf: list[str] = []

    def log(*parts) -> None:
        line = " ".join(str(p) for p in parts)
        buf.append(line)
        print(line)

    s = Dialectica()
    s.add_documents([Document.create(t, title=f"doc-{i}") for i, t in enumerate(DOCS)])
    r = s.debate("沙箱里为什么无法安装 Python 包")
    log("ANSWER:", r.answer)
    log("CONVERGED:", r.converged, "ROUNDS:", r.round_count)
    log("METRICS:", json.dumps(r.metrics, ensure_ascii=False))
    log("HITS:", len(r.hits))
    for rr in r.rounds:
        log(
            " round",
            rr.index,
            "claims",
            len(rr.claims),
            "anchored",
            sum(1 for c in rr.claims if c.anchored),
            "obj",
            len(rr.valid_objections),
            "delta",
            rr.delta,
        )
    log("HEALTH:", json.dumps(s.health(), ensure_ascii=False))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(buf))
    return 0 if r.converged and r.answer else 1


if __name__ == "__main__":
    raise SystemExit(main())
