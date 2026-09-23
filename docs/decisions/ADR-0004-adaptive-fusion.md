# ADR-0004：混合检索用查询自适应权重，不用固定 α

作者：晨星 · 状态：已接受

## 背景

hybrid search 的常规做法是固定 `α·dense + (1-α)·sparse`。但查询有两种极端：

- **词汇型**：`Errno 10054 WinError` —— 字面命中即答案，dense 会把它糊掉
- **语义型**：`为什么服务会自己停掉` —— 一个字面词都撞不上，BM25 全灭

固定 α 必有一边崩。

## 决策

从两条结果列表上读三个难度信号，逐查询求权重：

```
entropy = 归一化香农熵(BM25 top-k 分数)   越高说明词法侧越不 discriminating
margin  = min(1, (dense_top1 - dense_top2)/0.20)
lexical = |query_token ∩ dense_topk_token| / |query_token|

w_sparse = clamp(0.5 + 0.25·lexical - 0.20·entropy - 0.15·margin, 0.1, 0.9)
```

## 理由

三个信号都是**从结果本身读出来的**，不需要额外的模型或训练数据，且都是纯函数 —— 可测、可复现、可解释。

## 代价

- 三个系数是拍的，没有拟合。后续可以用带标注数据拟合，但拟合会把"可解释"换成"黑箱"，暂时不做。
- 极小语料（< 3 篇）下 entropy 退化，此时权重趋近 0.5，不产生危害。

## 验证

- `test_weights_sum_to_one`
- `test_weights_clamped`
- `test_lexical_query_boosts_sparse`：词汇型查询的 `w_sparse` 严格大于语义型查询
- `test_non_adaptive_is_fifty_fifty`：`ADAPTIVE_FUSION=0` 时回退固定权重
