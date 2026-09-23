# 辩衡 Dialectica · 架构设计

作者：晨星

---

## 1. 设计原则

四条硬约束，贯穿全部 12 个模块：

1. **契约先行。** 一切跨模块协作都声明在 `core/protocols.py` 的 `Protocol` 里。模块只依赖契约，不依赖实现。
2. **默认零依赖。** 每个外部依赖都有一个可注入的零依赖实现（Mock LLM / 哈希嵌入 / 内存索引 / 启发式重排）。因此无网络、无密钥、无数据库也能全绿。
3. **单一职责。** 每个模块一个文件夹，每个文件一个类族，单个文件不超过 300 行。装配只发生在 `system.py`。
4. **可自证。** 每个算法至少有一条可被独立实现交叉验证的不变量，且写成单测。

## 2. 数据流

```
                        ┌──────────────────────────────────────┐
   Documents ──────────▶│ retrieval.chunker                    │
                        │ 结构感知切分（空行 > 标题 > 句子）      │
                        └──────────────┬───────────────────────┘
                                       ▼
                        ┌──────────────────────────────────────┐
                        │ SparseIndex (Robertson BM25)         │
                        │ VectorIndex (HashEmbed / fastembed)  │
                        └──────────────┬───────────────────────┘
                                       ▼
        Query ──────────▶ ┌────────────────────────────────────┐
                          │ fusion.derive_weights              │
                          │  entropy / margin / lexical 三信号   │
                          │  → w_dense, w_sparse（和为 1）       │
                          │ fusion.weighted_rrf                 │
                          └──────────────┬─────────────────────┘
                                         ▼
                          ┌────────────────────────────────────┐
                          │ rerank.CalibratedReranker           │
                          │  z-score → 边际门控 → 融合序加权混合  │
                          └──────────────┬─────────────────────┘
                                         ▼   hits[]（证据跨度）
   ┌──────────────────────────────────────────────────────────────────┐
   │                        debate.DialecticProtocol                  │
   │  ┌──────────┐   ┌─────────────┐   ┌────────┐   ┌─────────────┐  │
   │  │ Proposer │──▶│ FactChecker │──▶│ Critic │──▶│ Synthesizer │  │
   │  └────▲─────┘   └─────────────┘   └────┬───┘   └─────────────┘  │
   │       │                                │                         │
   │       └────── 有效反对 + 证据缺口 ◀──────┘                        │
   └──────────────────────────────┬───────────────────────────────────┘
                                  ▼
              ┌────────────────────────────────────────┐
              │ evidence.extract → align → gate         │
              │  未锚定断言被剪除，绝不降级放行            │
              └──────────────────┬─────────────────────┘
                                 ▼
                          DebateResult（含 trace）
```

## 3. 模块契约

| Protocol | 生产实现 | 零依赖实现 | 注入点 |
|---|---|---|---|
| `Embedder` | `FastembedEmbedder`（ONNX bge） | `HashingEmbedder`（crc32 + IDF） | `RetrievalPipeline(embedder=)` |
| `VectorIndex` | `FaissIndex`（IndexFlatIP） | `MemoryIndex`（精确余弦扫描） | `RetrievalPipeline(vector_index=)` |
| `SparseIndex` | `BM25`（Robertson IDF） | 同一个（已零依赖） | `RetrievalPipeline(sparse=)` |
| `Reranker` | `CrossEncoderReranker` | `HeuristicReranker` | `RetrievalPipeline(reranker=)` |
| `Retriever` | `RetrievalPipeline` | 同上 | `DialecticProtocol(retriever=)` |
| `LLM` | `LlamaCppLLM` / `OllamaLLM` / `OpenAICompatLLM` | `MockLLM` | `DialecticProtocol(llm=)` |
| `Tool` | `Calculator` / `Clock` / `UnitConverter` | 同（纯函数） | `ReActAgent(tools=)` |
| `Memory` | `MemoryStore` | 同（内存 + 遗忘 + 蒸馏） | `Dialectica(memory=)` |
| `Observer` | `EventBus` | 同 | `Dialectica(bus=)` |

## 4. 关键算法

### 4.1 分词（中英混合）

中文没有空格词边界，直接 `split()` 会毁掉召回。策略：ASCII 段按词切并小写；CJK 串同时发出**一元**与**二元**字 gram —— 二元 gram 承载主要词法信号（电池 / 池续 / 续航），一元保短查询召回。

### 4.2 Robertson BM25

```
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))      恒 >= 0
score  = Σ idf(t) · f·(k1+1) / (f + k1·(1 - b + b·dl/avgdl))
```

为什么不用 `rank_bm25`：其 IDF 在 `df > N/2` 时可为负，再被 `epsilon * average_idf` 地板规则扭曲，2 文档语料上实测得分为 `[-0.366, -0.349]`，**正确答案垫底**。Robertson 形式对任意 `1 <= df <= N` 恒非负，该性质写成了单测。

### 4.3 哈希嵌入（离线默认）

```
features(text) = tokenize(text) ∪ {相邻 token 二元组}
w(tok)         = idf(tok) · (1 + ln(tf))
vec[crc32(tok) % dim] += sign(tok) · w(tok)
vec := vec / ||vec||
```

- 用 `zlib.crc32` 而非 `hash()`：后者受 `PYTHONHASHSEED` 随机化影响，会让所有下游测试不可复现。
- `fit()` 是可选钩子：传入语料后按 Robertson IDF 加权。`RetrievalPipeline.add_documents` 会检测并调用。
- `dim=2048` 由网格搜索确定（`dim × bigrams × idf`，中英混合运维语料，top1 4/6，MRR 0.806）。低于 512 时哈希碰撞主导，排序崩塌。

### 4.4 自适应融合

```
entropy = H(BM25 top-k 分数) / ln(k)          ∈ [0,1]，越均匀越差
margin  = min(1, (dense_top1 - dense_top2)/0.20)
lexical = |query_token ∩ dense_topk_token| / |query_token|

w_sparse = clamp(0.5 + 0.25·lexical - 0.20·entropy - 0.15·margin, 0.1, 0.9)
w_dense  = 1 - w_sparse

fused(d) = w_dense/(k + rank_dense(d)) + w_sparse/(k + rank_sparse(d))
```

### 4.5 校准重排

```
zr = z_score(rerank_scores)
zf = z_score(fused_scores)
margin = zr_top1 - zr_top2

if margin < min_margin:  final = zf                     # 重排不 discriminating，整单回退
else:                    final = α·zr + (1-α)·zf

keep = { i | zr[i] >= z_floor } ∪ {rerank_top1} ∪ {fused_top1}
```

`fused_top1` 永不被剪 —— 这是"重排器不能单方面埋掉正确答案"的硬保证。对应单测：构造一个给正确答案打 0 分、其余打 1 分的对抗性打分器，正确答案仍在首位。

### 4.6 证据锚定

```
align(claim, span) = 0.6 · coverage(claim, span) + 0.4 · jaccard(claim, span)
coverage(c, s)     = |tok(c) ∩ tok(s)| / |tok(c)|
```

覆盖度项防止"长段落碰巧共享常用词"就能通过校验。`Claim.anchored` 为真必然携带 span id（单测断言）。

### 4.7 收敛判定

```
delta_i = 1 - cos(embed(proposal_i), embed(proposal_{i-1}))
converged ⇔ 本轮无有效反对
```

反对**有效**当且仅当它引用了当前证据集合里真实存在的 span id —— 伪造 span 的反对被 `filter_valid` 丢弃并在 trace 中留痕。`delta` 是诊断量而非停止条件：末轮 delta 大，恰恰说明批评者真的改了答案。

协议还有一条 `stalled` 短路：连续两轮提出完全相同的反对即判定无进展，停止。

### 4.8 记忆三层

```
working  : 有界 FIFO（最近 N 条当前会话）
episodic : 带时间戳，score = salience · exp(-λ·Δhours) · (0.5 + 0.5·cos(q, e))
semantic : 聚类蒸馏产物，每条带 support 计数
```

`distill()` 幂等：第二次调用返回 0（单测断言），因为已存在的 fact 不会重复加入。

## 5. 分层与依赖方向

```
api ──┐
cli ──┼──▶ system ──▶ debate ──▶ evidence ──▶ retrieval ──▶ core
web ──┘                  │          ▲
                         └──▶ llm   │
                         └──▶ reason│
                         └──▶ memory┘
                         └──▶ eval
```

依赖**单向向下**，`core` 不依赖任何东西。因此 `core` 可被任何模块安全引用，而上层模块之间互不可见。

## 6. 可观测性

每次 `debate()` 在 `EventBus` 上留下结构化事件：

| 事件 | 载荷 |
|---|---|
| `retrieval.done` | query, hits, 融合信号（entropy / margin / lexical / 权重） |
| `debate.round` | index, claims, anchored, objections, delta |
| `debate.done` | converged, rounds, faithfulness |

`GET /trace` 导出完整 trace。`DebateResult.rounds` 保留每一轮的提案原文、断言裁决、反对意见与 gap 查询，可以离线重放。

## 7. 扩展点

| 想换什么 | 怎么做 |
|---|---|
| 向量库（Qdrant / Milvus） | 实现 `VectorIndex` Protocol，注入 `RetrievalPipeline` |
| 词法检索（Elasticsearch） | 实现 `SparseIndex` Protocol |
| 工具（MCP / HTTP） | 实现 `Tool` Protocol，加入 `ReActAgent(tools=)` |
| 记忆后端（Redis / Postgres） | 实现 `Memory` Protocol |
| 评估集 | 给 `run_benchmark` 传自定义 `fixtures` |
