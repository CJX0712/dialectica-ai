# 辩衡 Dialectica · 接口规格

作者：晨星

本文定义所有公开类型与契约。实现以 `src/dialectica/` 为准，本文是规范。

---

## 1. 核心类型（`core/types.py`）

### Document
| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | str | `doc_<12 hex>` |
| `text` | str | 原文 |
| `title` | str | 缺省取 `source`，再缺省 `untitled` |
| `source` | str | 来源标识 |
| `metadata` | dict | 透传到 chunk |

`Document.create(text, title="", source="", **meta)` 工厂。

### Chunk
| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | str | `chk_<12 hex>` |
| `span_id` | str | `span::<chunk_id>`，**全局证据句柄** |
| `doc_id` / `text` / `order` / `start` / `end` | | 文档内偏移 |
| `metadata` | dict | 含 `title` / `source` |

### Hit
`chunk` + `score` + `components: dict[str, float]`（保留 `fused` / `rerank_z` / `rerank_raw` 用于审计）。

### Claim
| 字段 | 类型 | 说明 |
|---|---|---|
| `claim_id` | str | |
| `text` | str | |
| `verdict` | Verdict | `supported` / `unsupported` / `uncertain` |
| `evidence` | list[str] | span id 列表 |
| `support` | float | 对齐分数 |

**不变量**：`Claim.anchored == True` ⟹ `verdict is SUPPORTED and len(evidence) >= 1`。

### Objection
| 字段 | 类型 | 说明 |
|---|---|---|
| `objection_id` | str | |
| `text` | str | 反对理由 |
| `citations` | list[str] | span id 列表 |
| `severity` | float | |

**不变量**：`Objection.valid == bool(citations) and bool(text.strip())`。

### DebateRound / DebateResult

`DebateRound`：`index`, `proposal`, `claims`, `verdicts`, `objections`（原始）, `valid_objections`（过滤后）, `gap_queries`, `delta`。

`DebateResult`：`answer`, `rounds`, `hits`, `claims`（门禁后存活的）, `converged`, `trace_id`, `metrics`。

### Message / Generation

`Message(role, content)`，role ∈ {system, user, assistant}。
`Generation(text, model, tokens, latency_ms, raw)`。

## 2. 协议（`core/protocols.py`）

```python
class Embedder(Protocol):
    dim: int
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_one(self, text: str) -> list[float]: ...

class VectorIndex(Protocol):
    def add(self, chunks, vectors) -> None: ...
    def search(self, vector, top_k) -> list[tuple[str, float]]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...

class SparseIndex(Protocol):
    def add(self, chunks) -> None: ...
    def search(self, query, top_k) -> list[tuple[str, float]]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...

class Reranker(Protocol):
    def rerank(self, query, hits) -> list[Hit]: ...

class Retriever(Protocol):
    def add_documents(self, docs) -> int: ...
    def retrieve(self, query, top_k=None) -> list[Hit]: ...

class LLM(Protocol):
    name: str
    def complete(self, messages, *, temperature=0.0, max_tokens=512, stop=None) -> Generation: ...
    def stream(self, messages, **kwargs) -> Iterable[str]: ...

class Tool(Protocol):
    name: str
    description: str
    def invoke(self, argument: str) -> str: ...

class Memory(Protocol):
    def remember(self, text, *, salience=1.0) -> None: ...
    def recall(self, query, top_k=5) -> list[str]: ...
    def distill(self) -> int: ...

class Observer(Protocol):
    def on_event(self, name: str, payload: dict) -> None: ...
```

所有 Protocol 都是 `@runtime_checkable`，可用 `isinstance` 校验。

## 3. 配置（`core/config.py`）

环境变量前缀 `DIALECTICA_`。布尔量接受 `1/true/yes/on`。

| 变量 | 默认 | 含义 |
|---|---|---|
| `LLM_BACKEND` | `mock` | mock / llamacpp / ollama / openai |
| `LLM_MODEL` | `mock-1` | |
| `GGUF_PATH` | 空 | llamacpp 模型路径 |
| `N_CTX` / `N_THREADS` | 4096 / 4 | llamacpp 上下文与线程 |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | 空 | |
| `EMBEDDING_BACKEND` | `hashing` | hashing / fastembed |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | |
| `VECTOR_BACKEND` | `memory` | memory / faiss |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 480 / 80 | 字符数 |
| `TOP_K` / `CANDIDATE_POOL` / `RRF_K` | 8 / 40 / 60 | |
| `ADAPTIVE_FUSION` | 1 | 关掉则固定 0.5 / 0.5 |
| `RERANK_ENABLED` | 1 | |
| `RERANK_BACKEND` | `heuristic` | heuristic / cross-encoder |
| `RERANK_ALPHA` / `Z_FLOOR` / `MIN_MARGIN` / `TOP_N` | 0.5 / -0.5 / 0.15 / 8 | |
| `SUPPORT_THRESHOLD` | 0.30 | 断言-证据对齐阈值 |
| `MAX_UNANCHORED_RATIO` | 0.20 | 门禁 |
| `PRUNE_UNANCHORED` | 1 | 剪除未锚定断言 |
| `MAX_ROUNDS` | 3 | 辩证轮数上限 |
| `CONVERGENCE_EPS` | 0.03 | 稳定性记录阈值 |
| `SELF_CONSISTENCY` | 1 | >1 时采样投票 |
| `GAP_RETRIEVAL` | 1 | 证据缺口自动补检索 |
| `MEMORY_ENABLED` / `WORKING_SIZE` / `DECAY_LAMBDA` / `CLUSTER_THRESHOLD` / `MIN_SUPPORT` | 1 / 16 / 0.02 / 0.86 / 2 | |
| `HOST` / `PORT` / `CORS_ORIGINS` | 127.0.0.1 / 8000 / `*` | |
| `LOG_LEVEL` | INFO | |

## 4. HTTP 接口

完整 OpenAPI 见 `docs/openapi.yaml`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 后端、嵌入、向量、切片数、记忆统计 |
| POST | `/ingest` | `{docs:[{text,title,source}]}` 或 `{texts:[...]}` → `{chunks, documents}` |
| POST | `/reset` | 清空语料与记忆 |
| POST | `/debate` | `{query, top_k?}` → 完整辩论结果 |
| POST | `/debate/stream` | 同上，SSE：`evidence` → `round`×N → `answer` → `done` |
| POST | `/ask` | `{query}` → `{answer}` |
| GET | `/trace?limit=200` | 结构化事件 |
| POST | `/evaluate` | 内置基准（每次在全新管道上跑） |
| GET | `/config` | 当前生效配置 |
| GET | `/` | 端点清单 |

### 错误码

| 状态码 | code | 触发 |
|---|---|---|
| 400 | `empty_corpus` | 未入库就辩论 |
| 400 | `tool_error` | 工具入参非法 |
| 422 | `evidence_gap` | 未锚定比例超阈值（严格模式） |
| 500 | `config_error` | 未知后端 |
| 503 | `provider_unavailable` | 依赖缺失 |
| 504 | `debate_diverged` | 协议未在预算内收敛 |

## 5. 指标定义

| 指标 | 定义 | 域 |
|---|---|---|
| `faithfulness` | 找到支撑跨度的答案句数 / 答案句总数 | [0,1] |
| `answer_relevancy` | mean cosine(embed(答案句), embed(query)) | [0,1] |
| `context_precision` | 与金标答案相关的检索跨度数 / 检索总数 | [0,1] |
| `context_recall` | 被召回的金标跨度数 / 金标跨度总数 | [0,1] |
| `keyword_recall` | 金标关键词命中数 / 关键词总数 | [0,1] |
| `rounds` / `converged` | 辩证轮数 / 是否收敛 | |
| `stability` | 1 - mean(delta)，1.0 表示辩论全程没动过 | [0,1] |

## 6. 提示词契约

四个角色的 system prompt 均以机器可读标记开头：`ROLE: proposer|factchecker|critic|synthesizer`。

- **证据块分隔符是全局唯一名 `<kb-context>` / `</kb-context>`。** 指令文本中**永不**出现该分隔符字面量 —— 否则模型会把指令当作证据引用回来。
- Critic 输出：`OBJECTION | <span_id> | <理由>` 或 `NO_OBJECTION`
- FactChecker 输出：`CLAIM <i> | SUPPORTED|UNSUPPORTED | <span_id>|none | <0-1>`
- Proposer 输出：1-3 句断言，独立成行
- Synthesizer 输出：最终答案正文

## 7. 不变量清单（全部有对应单测）

| # | 不变量 |
|---|---|
| 1 | Robertson IDF 对 `1 <= df <= N` 恒非负 |
| 2 | 哈希嵌入输出 L2 归一（非空文本），相同输入跨进程一致 |
| 3 | faiss 与内存索引的 top-k 顺序完全一致 |
| 4 | 融合权重 `w_dense + w_sparse == 1`，且各自 ∈ [0.1, 0.9] |
| 5 | 词汇型查询的 `w_sparse` 大于语义型查询 |
| 6 | 重排输出是输入的子集，且融合 top-1 永不被剪 |
| 7 | 重排器完全不 discriminating 时，输出顺序 == 融合顺序 |
| 8 | 对抗性重排器无法把正确答案挤出首位 |
| 9 | `Claim.anchored == True` ⟹ 携带 span id |
| 10 | 伪造 span 的反对被 `filter_valid` 丢弃 |
| 11 | `distill()` 幂等 |
| 12 | 遗忘曲线得分随时间单调不增 |
| 13 | 计算器拒绝代码注入（AST 白名单） |
| 14 | 相同查询在同一管道上两次运行，答案逐字符相同 |
| 15 | 辩论末轮的 `valid_objections` 必为空（收敛定义） |
