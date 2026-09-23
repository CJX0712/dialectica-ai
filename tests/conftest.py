"""Shared fixtures. Everything here is offline and deterministic."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from dialectica.core.config import Settings  # noqa: E402
from dialectica.core.types import Document  # noqa: E402
from dialectica.retrieval.dense import HashingEmbedder  # noqa: E402
from dialectica.system import Dialectica  # noqa: E402

CORPUS = [
    "沙箱守卫会拦截对 site-packages 的写入与删除，pip install 会报 SAFE_DELETE_BULK_GUARD_ERROR。"
    "绕行方式是使用 pip install --target 指定目录，再通过 PYTHONPATH 加载。",
    "机器上的 SOCKS5 代理会让 pip 报 Missing dependencies for SOCKS support，"
    "需要先安装 PySocks 再继续安装其它依赖。",
    "npm registry 被指向了鸿蒙源 ohpm.openharmony.cn，缺少通用包。"
    "安装时需要显式指定 --registry=https://registry.npmmirror.com。",
    "esbuild 的 postinstall 会被沙箱拦截，报 EBUSY spawnSync node.exe。"
    "使用 npm install --ignore-scripts 并显式补装平台原生包。",
    "英文交叉编码器在中文查询上判别力不足，如果把重排顺序直接当作最终排序，"
    "会把正确答案压下去。护栏做法是对重排分数做 z-score 归一化后与融合分数加权混合。",
]


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=128)


@pytest.fixture
def docs():
    return [Document.create(t, title=f"doc-{i}") for i, t in enumerate(CORPUS)]


@pytest.fixture
def settings():
    return Settings.load()


@pytest.fixture
def system(settings, docs):
    s = Dialectica(settings)
    s.add_documents(docs)
    return s
