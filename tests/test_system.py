"""End-to-end system-level invariants (no HTTP, no network)."""
from __future__ import annotations

from dialectica.core.types import Document
from dialectica.system import Dialectica


def test_health_reports_zero_dependency_defaults():
    h = Dialectica().health()
    assert h["llm"] == "mock"
    assert h["embedding"] == "HashingEmbedder"
    assert h["vector"] == "MemoryIndex"


def test_ingest_then_debate(system):
    result = system.debate("沙箱里为什么无法安装 Python 包")
    assert result.answer
    assert result.converged
    assert result.metrics["faithfulness"] > 0.0


def test_arithmetic_bypasses_debate(system):
    result = system.debate("12*(3+4)")
    assert "84" in result.answer
    assert result.round_count == 0
    assert result.metrics["route"] == 1.0


def test_unit_conversion_route(system):
    assert "12000" in system.ask("12 km to m")


def test_ask_returns_string(system):
    assert isinstance(system.ask("npm 镜像源被劫持怎么办"), str)


def test_memory_records_interaction(system):
    before = system.memory.stats()["episodic"]
    system.debate("沙箱守卫拦截写入怎么办")
    assert system.memory.stats()["episodic"] == before + 1


def test_fresh_system_has_no_leakage_between_instances(docs):
    a = Dialectica()
    a.add_documents(docs)
    b = Dialectica()
    assert len(b) == 0


def test_repeated_identical_query_is_stable(system):
    first = system.debate("沙箱里为什么无法安装 Python 包").answer
    second = system.debate("沙箱里为什么无法安装 Python 包").answer
    assert first == second


def test_add_texts_helper():
    s = Dialectica()
    assert s.add_texts(["一段文本。", "另一段文本。"]) >= 2


def test_len_matches_chunks(docs):
    s = Dialectica()
    s.add_documents(docs)
    assert len(s) == len(s.retriever.chunks)


def test_document_create_defaults():
    d = Document.create("内容")
    assert d.doc_id.startswith("doc_")
    assert d.title == "untitled"
