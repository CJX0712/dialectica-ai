"""Regression benchmark over a built-in fixture set.

Every case runs on a **fresh pipeline**: reusing a singleton pipeline that
already holds previously ingested documents makes recall non-deterministic
(duplicate doc ids / mixed corpora) and the numbers drift between runs.

Author: 晨星
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..core.types import Document
from .metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

FIXTURES: list[dict] = [
    {
        "query": "沙箱里为什么无法安装 Python 包",
        "docs": [
            "沙箱守卫会拦截对 site-packages 的写入与删除，pip install 会报 SAFE_DELETE_BULK_GUARD_ERROR。"
            "绕行方式是使用 pip install --target 指定目录，再通过 PYTHONPATH 加载。",
            "机器上的 SOCKS5 代理会让 pip 报 Missing dependencies for SOCKS support，"
            "需要先安装 PySocks 再继续安装其它依赖。",
        ],
        "gold_answer": "沙箱守卫拦截 site-packages 写入，需要使用 pip install --target 并配置 PYTHONPATH。",
        "keywords": ["site-packages", "PYTHONPATH", "PySocks"],
    },
    {
        "query": "npm 安装依赖时找不到包怎么办",
        "docs": [
            "npm registry 被指向了鸿蒙源 ohpm.openharmony.cn，缺少通用包。"
            "安装时需要显式指定 --registry=https://registry.npmmirror.com。",
            "esbuild 的 postinstall 会被沙箱拦截，报 EBUSY spawnSync node.exe。"
            "使用 npm install --ignore-scripts 并显式补装平台原生包。",
        ],
        "gold_answer": "npm registry 被劫持到鸿蒙源，需要用 npmmirror 镜像，并对原生包使用 --ignore-scripts 后补装。",
        "keywords": ["registry", "npmmirror", "ignore-scripts"],
    },
    {
        "query": "重排模型为什么会让中文检索结果变差",
        "docs": [
            "英文交叉编码器在中文查询上判别力不足，如果把重排顺序直接当作最终排序，"
            "会把正确答案压下去，实测 top-1 命中率从 11/12 掉到 5/12。",
            "护栏做法是先对重排分数做 z-score 归一化，再与融合分数按权重混合，"
            "并在重排 top-1 边际不足时回退到融合顺序。",
        ],
        "gold_answer": "英文重排器在中文查询上判别力不足，需要用 z-score 归一化和融合加权护栏，边际不足时回退融合顺序。",
        "keywords": ["z-score", "融合", "边际"],
    },
]


@dataclass
class CaseResult:
    query: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.metrics.get("faithfulness", 0.0) >= 0.6
            and self.metrics.get("context_recall", 0.0) >= 0.5
            and self.metrics.get("keyword_recall", 0.0) >= 0.6
        )


@dataclass
class BenchmarkReport:
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def total(self) -> int:
        return len(self.cases)

    def mean(self, key: str) -> float:
        vals = [c.metrics.get(key, 0.0) for c in self.cases]
        return sum(vals) / len(vals) if vals else 0.0

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total": self.total,
            "mean_faithfulness": round(self.mean("faithfulness"), 6),
            "mean_answer_relevancy": round(self.mean("answer_relevancy"), 6),
            "mean_context_precision": round(self.mean("context_precision"), 6),
            "mean_context_recall": round(self.mean("context_recall"), 6),
            "mean_keyword_recall": round(self.mean("keyword_recall"), 6),
            "cases": [{"query": c.query, "metrics": c.metrics, "passed": c.passed} for c in self.cases],
        }


def run_benchmark(
    factory: Callable[[], object],
    embedder,
    fixtures: Sequence[dict] | None = None,
) -> BenchmarkReport:
    """`factory` must return a fresh object exposing `add_documents` + `debate`."""
    report = BenchmarkReport()
    for fx in fixtures or FIXTURES:
        system = factory()
        docs = [Document.create(t, title=f"case-{i}") for i, t in enumerate(fx["docs"])]
        system.add_documents(docs)
        result = system.debate(fx["query"])
        retrieved = [h.text for h in result.hits]
        from .metrics import keyword_recall

        metrics = {
            "faithfulness": faithfulness(result.answer, retrieved),
            "answer_relevancy": answer_relevancy(embedder, fx["query"], result.answer),
            "context_precision": context_precision(retrieved, fx["gold_answer"]),
            "context_recall": context_recall(retrieved, fx["docs"]),
            "keyword_recall": keyword_recall(result.answer, fx["keywords"]),
            "rounds": float(result.round_count),
            "converged": 1.0 if result.converged else 0.0,
        }
        report.cases.append(CaseResult(query=fx["query"], metrics={k: round(v, 6) for k, v in metrics.items()}))
    return report
