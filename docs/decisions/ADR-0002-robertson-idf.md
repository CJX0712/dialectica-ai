# ADR-0002：自研 Robertson IDF，不用 rank_bm25

作者：晨星 · 状态：已接受

## 背景

`rank-bm25` 是最常用的 Python BM25 实现。但在小语料上它的行为不符合预期。

## 实测

2 文档语料，查询命中其中一篇，两篇得分 `[-0.366, -0.349]` —— **正确答案垫底**。

根因：其 IDF 形式在 `df > N/2` 时为负，再被 `epsilon * average_idf` 地板规则扭曲。小语料里"词恰好出现在一半文档"是常态。

## 决策

自写 BM25，IDF 用 Robertson 形式：

```
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

该式对任意 `N >= 1`、`1 <= df <= N` 恒非负（因为 `(N - df + 0.5)/(df + 0.5) > 0`，故 `1 + · > 1`，`ln > 0`）。

## 理由

非负 IDF 是不变量，可以写成单测；负 IDF 的行为依赖地板规则，无法用简单断言守护。

## 代价

- 少了 `rank_bm25` 的社区维护。
- 该包仍保留在 `requirements-prod.txt` 里作为**大语料基准对照**，不参与默认链路。

## 验证

`tests/test_sparse.py::test_idf_non_negative_for_every_df`、`test_two_document_corpus_still_discriminates`。
