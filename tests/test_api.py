"""HTTP gateway tests via FastAPI TestClient (no real server, no ports)."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dialectica.api.app import create_app  # noqa: E402
from dialectica.core.config import Settings  # noqa: E402
from dialectica.system import Dialectica  # noqa: E402

CORPUS = [
    "沙箱守卫会拦截对 site-packages 的写入与删除，pip install 会报 SAFE_DELETE_BULK_GUARD_ERROR。",
    "机器上的 SOCKS5 代理会让 pip 报 Missing dependencies for SOCKS support，需要先安装 PySocks。",
]


@pytest.fixture
def client():
    settings = Settings.load()
    system = Dialectica(settings)
    return TestClient(create_app(system, settings))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_lists_endpoints(client):
    assert "/debate" in client.get("/").json()["endpoints"]


def test_ingest_then_debate(client):
    assert client.post("/ingest", json={"texts": CORPUS}).json()["chunks"] >= 2
    r = client.post("/debate", json={"query": "沙箱里为什么无法安装 Python 包"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["evidence"]
    assert body["rounds"]


def test_ingest_empty_rejected(client):
    assert client.post("/ingest", json={"texts": []}).status_code == 400


def test_debate_on_empty_corpus_returns_400(client):
    r = client.post("/debate", json={"query": "任何问题"})
    assert r.status_code == 400


def test_ask(client):
    client.post("/ingest", json={"texts": CORPUS})
    assert client.post("/ask", json={"query": "沙箱守卫拦截写入怎么办"}).json()["answer"]


def test_ask_tool_route(client):
    assert "84" in client.post("/ask", json={"query": "12*(3+4)"}).json()["answer"]


def test_reset(client):
    client.post("/ingest", json={"texts": CORPUS})
    assert client.post("/reset").json()["status"] == "ok"
    assert client.get("/health").json()["chunks"] == 0


def test_trace_endpoint(client):
    client.post("/ingest", json={"texts": CORPUS})
    client.post("/debate", json={"query": "沙箱守卫拦截写入怎么办"})
    body = client.get("/trace").json()
    assert "events" in body


def test_evaluate_endpoint_reports_all_cases(client):
    body = client.post("/evaluate").json()
    assert body["total"] == 3
    assert body["passed"] >= 2


def test_config_endpoint(client):
    body = client.get("/config").json()
    assert "llm_backend" in body


def test_stream_endpoint_emits_events(client):
    client.post("/ingest", json={"texts": CORPUS})
    with client.stream("POST", "/debate/stream", json={"query": "沙箱守卫拦截写入怎么办"}) as r:
        text = "".join(r.iter_text())
    assert "event: evidence" in text
    assert "event: answer" in text
    assert "event: done" in text


def test_claims_carry_evidence(client):
    client.post("/ingest", json={"texts": CORPUS})
    body = client.post("/debate", json={"query": "沙箱守卫拦截写入怎么办"}).json()
    for c in body["claims"]:
        assert c["anchored"]
        assert c["evidence"]


def test_ingest_with_titles(client):
    r = client.post("/ingest", json={"docs": [{"text": CORPUS[0], "title": "pip", "source": "ops"}]})
    assert r.json()["chunks"] >= 1
