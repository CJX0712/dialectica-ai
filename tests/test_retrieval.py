"""Chunking + retrieval pipeline invariants."""
from __future__ import annotations

from dialectica.core.errors import EmptyCorpus
from dialectica.core.types import Document
from dialectica.retrieval.chunker import chunk_document
from dialectica.retrieval.pipeline import RetrievalPipeline

LONG = "第一句话讲的是电池续航。\n第二句话讲的是充电接口。\n\n第三段讲的是整机重量与尺寸。\n" * 8


def test_chunks_cover_the_document():
    doc = Document.create(LONG, title="spec")
    chunks = chunk_document(doc, chunk_size=120, overlap=20)
    joined = "".join(c.text for c in chunks)
    for probe in ["电池续航", "充电接口", "整机重量"]:
        assert probe in joined


def test_chunks_are_non_empty_and_ordered():
    doc = Document.create(LONG, title="spec")
    chunks = chunk_document(doc, chunk_size=120, overlap=20)
    assert chunks
    assert all(c.text.strip() for c in chunks)
    assert [c.order for c in chunks] == list(range(len(chunks)))


def test_offsets_are_within_document():
    doc = Document.create(LONG, title="spec")
    chunks = chunk_document(doc, chunk_size=120, overlap=20)
    for c in chunks:
        assert 0 <= c.start <= c.end <= len(doc.text)


def test_span_ids_are_unique():
    doc = Document.create(LONG, title="spec")
    chunks = chunk_document(doc, chunk_size=120, overlap=20)
    ids = [c.span_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_short_document_is_single_chunk():
    doc = Document.create("一句话就够短了。", title="short")
    chunks = chunk_document(doc, chunk_size=480, overlap=80)
    assert len(chunks) == 1


def test_pipeline_raise_on_empty_corpus():
    p = RetrievalPipeline()
    try:
        p.retrieve("anything")
        assert False, "expected EmptyCorpus"
    except EmptyCorpus:
        pass


def test_pipeline_ingest_and_retrieve(docs):
    p = RetrievalPipeline()
    n = p.add_documents(docs)
    assert n > 0
    assert len(p) == n
    hits = p.retrieve("沙箱里为什么装不上 Python 包")
    assert hits
    assert all(h.score == h.score for h in hits)  # no NaN


def test_pipeline_top_k_respected(docs):
    p = RetrievalPipeline()
    p.add_documents(docs)
    hits = p.retrieve("npm 镜像源", top_k=2)
    assert len(hits) <= 2


def test_pipeline_context_block_has_unique_delimiter(docs):
    p = RetrievalPipeline()
    p.add_documents(docs)
    hits = p.retrieve("重排护栏")
    block = p.context_block(hits)
    assert "<kb-context>" not in block
    assert block.count("span::") == len(hits)


def test_pipeline_signals_recorded(docs):
    p = RetrievalPipeline()
    p.add_documents(docs)
    p.retrieve("npm 镜像源")
    assert set(p.last_signals) == {"entropy", "margin", "lexical", "w_dense", "w_sparse"}
    assert abs(p.last_signals["w_dense"] + p.last_signals["w_sparse"] - 1.0) < 1e-6


def test_pipeline_clear(docs):
    p = RetrievalPipeline()
    p.add_documents(docs)
    p.clear()
    assert len(p) == 0
