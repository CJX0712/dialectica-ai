"""Tokeniser invariants."""
from __future__ import annotations

from dialectica.retrieval.tokenizer import is_cjk, tokenize


def test_ascii_lowercased():
    assert "Python" not in tokenize("Python")
    assert "python" in tokenize("Python")


def test_cjk_emits_unigrams_and_bigrams():
    toks = tokenize("电池续航")
    assert "电" in toks and "池" in toks and "续" in toks and "航" in toks
    assert "电池" in toks and "池续" in toks and "续航" in toks


def test_deterministic():
    assert tokenize("混合检索 hybrid search") == tokenize("混合检索 hybrid search")


def test_empty():
    assert tokenize("") == []


def test_ngram_count():
    run = "电池续航"
    toks = [t for t in tokenize(run) if len(t) <= 2]
    # n unigrams + (n-1) bigrams, all CJK here
    assert len(toks) == len(run) + len(run) - 1


def test_is_cjk():
    assert is_cjk("中")
    assert not is_cjk("a")
    assert not is_cjk("")
