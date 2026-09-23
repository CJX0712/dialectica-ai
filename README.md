# 辩衡 Dialectica

<p align="center">
  <a href="https://github.com/CJX0712/dialectica-ai/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/dialectica-ai/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/dialectica-ai/releases"><img src="https://img.shields.io/github/v/release/CJX0712/dialectica-ai?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/dialectica-ai/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/dialectica-ai" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

**Deliberative multi-agent reasoning with evidence anchoring and provable convergence.**

一个答案不应该只是模型一次性吐出的文本。辩衡把它变成一件**经过对抗式检验、每条断言都锚定到证据、且可以证明已收敛**的产物。

- 本地优先 · CPU 可跑 · 零密钥 · 零网络可验证
- 12 个单一职责模块，接口全部由 `Protocol` 定义，可独立替换与独立测试
- 一键复现：`pip install -r requirements.txt && python scripts/verify.py`

作者：**晨星**

---

## 1. 它解决什么

普通 RAG 链路的失败模式是**静默的**：检索错了不知道、答案编造了不知道、重排把正确答案压下去了也不知道。辩衡把这些失败模式逐个变成**有门禁、有读数、有不变量**的环节。

| 环节 | 普通做法 | 辩衡的做法 | 可验证的不变量 |
|---|---|---|---|
| 检索融合 | dense 与 BM25 固定权重 α | 按查询难度信号动态求权重 | `w_dense + w_sparse == 1`，两者夹在 [0.1, 0.9] |
| 重排 | 交叉编码器分数直接当最终序 | z-score 归一化 + 边际门控 + 融合序加权回退 | 重排器无法单方面埋掉融合 top-1 |
| 事实性 | 模型自称"根据上下文" | 断言 ↔ 证据跨度对齐，未锚定即剪除 | `anchored == True` 必然携带 span id |
| 生成质量 | 一次采样直接输出 | 四角色多轮辩证，反对必须引用证据 | 收敛当且仅当无有效反对 |
| 复现性 | 日志里翻 | 结构化 trace，逐轮可回放 | 相同输入两次运行答案完全一致 |

## 2. 五项核心设计

### 2.1 DP-4 辩证协议（Proposer / FactChecker / Critic / Synthesizer）

```
Proposer ──提案──▶ FactChecker ──裁决──▶ Critic ──反对──▶ (回到 Proposer)
                                            │
                                       无有效反对
                                            ▼
                                       Synthesizer
```

**反对必须引用证据跨度**，否则该反对无效并被协议丢弃。这条规则让"批评者"无法靠凭空质疑绑架辩论 —— 实测丢弃的无效反对在 trace 里可见，不是静默忽略。

### 2.2 证据锚定门禁

答案被切成断言，每条断言与检索到的证据跨度做非对称覆盖度 + Jaccard 混合打分。低于阈值的断言**直接删除**，不是打标记放行。系统宁可给一个更短的真答案，也不给一个更长的像真的答案。

### 2.3 校准重排护栏

英文交叉编码器在中文查询上判别力不足，若把重排顺序**直接**当作最终排序，会把正确答案压下去。护栏是三层的：

1. 候选集内 z-score 归一化，抹掉量纲
2. 低于 `z_floor` 的候选被剪掉，但**融合 top-1 永不被剪**
3. 最终序 = `alpha * z_rerank + (1-alpha) * z_fusion`；当重排 top-1 边际不足 `min_margin` 时，整单回退到融合序

### 2.4 自适应混合融合

固定 α 必有一边崩：词汇型查询（"Errno 10054 WinError"）要 BM25，语义型查询（"为什么服务会自己停掉"）要 dense。辩衡从两条结果列表上读三个难度信号，逐查询求权重：

```
w_sparse = 0.5 + 0.25*lexical - 0.20*entropy - 0.15*margin
```

- `entropy`：BM25 top-k 得分分布的归一化香农熵，越均匀说明词法侧越不 discriminating
- `margin`：dense top-1 与 top-2 的余弦差，越大说明 dense 越自信
- `lexical`：查询 token 在 dense 结果中的字面命中率，越高说明这是词汇型查询

### 2.5 确定性前置路由

0.5B 级模型判断不了 `12*(3+4)` 是不是算术题，更不该在工具给出结果后**自己重算一遍**（实测会算出 72）。所以路由在模型看到查询之前就由正则完成：命中工具 → 先执行 → 结果作为 Observation 注入 → 明确禁止重算。

## 3. 快速开始

```bash
# 1) 依赖（干净环境，约 30 秒，无需 GPU / 网络 / API Key）
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# 2) 一键验证（单测 + 真实起服务 + 端到端链路 + CLI + P0 门禁）
python scripts/verify.py

# 3) 起服务并用单文件控制台
python -m dialectica.cli serve
# 浏览器打开 web/console.html
```

命令行：

```bash
python -m dialectica.cli ingest --file corpus.txt
python -m dialectica.cli ask "沙箱里为什么无法安装 Python 包"
python -m dialectica.cli debate "重排模型为什么会让中文检索结果变差" --json
python -m dialectica.cli evaluate
```

Python：

```python
from dialectica import Dialectica, Document

s = Dialectica()
s.add_documents([Document.create("沙箱守卫会拦截对 site-packages 的写入与删除……", title="pip")])
r = s.debate("沙箱里为什么无法安装 Python 包")

r.answer        # 最终答案
r.converged     # 是否收敛
r.metrics       # faithfulness / rounds / stability / 融合权重 …
r.rounds        # 每一轮的提案、断言裁决、反对意见、delta
r.hits          # 证据跨度
```

## 4. 模块划分

每个模块只依赖 `core/protocols.py` 里的契约，运行时注入实现。默认注入零依赖实现，因此**无网络、无密钥、无数据库**也能全绿。

| 模块 | 职责 | 默认实现（离线） | 生产实现（环境变量切换） |
|---|---|---|---|
| `core` | 类型 / 协议 / 配置 / 事件 / 错误 | —— | —— |
| `llm` | 文本生成 | `MockLLM`（确定性、抽取式） | llama.cpp / Ollama / OpenAI 兼容 |
| `retrieval` | 切分 → 词法 → dense → 融合 → 重排 | 哈希嵌入 + 自研 BM25 + 内存索引 + 启发式重排 | fastembed bge-small-zh / faiss / ONNX cross-encoder |
| `evidence` | 断言抽取 → 跨度对齐 → 门禁 | 启发式对齐 | 同 |
| `debate` | DP-4 协议、轮次、收敛判定 | —— | —— |
| `reason` | 工具路由 / ReAct / 自洽投票 | 计算器 · 时钟 · 单位换算 | MCP 工具（扩展点） |
| `memory` | 工作 / 情景 / 语义三层 | 内存 + 遗忘曲线 + 蒸馏 | 接外部存储（扩展点） |
| `eval` | 指标与基准 | 内置 3 例回归集 | 自定义 fixture |
| `api` | HTTP 网关 + SSE | FastAPI | 同 |
| `cli` | 命令行 | —— | —— |
| `web` | 单文件 HTML 控制台 | 零依赖 | 同 |
| `scripts` | 验证 / 门禁 / 报告 | —— | —— |

## 5. 验证结果

`python scripts/verify.py` 在本机的实际输出：

```
[1/4] 单元测试        PASS  pytest 全绿（152 项）
[2/4] 端到端服务链路   PASS  14 项（真实起 uvicorn，轮询 /health，跑成功流与全部错误流，taskkill 回收）
[3/4] CLI 冒烟        PASS  3 项
[4/4] P0 门禁         PASS  全仓无 emoji
通过：19 / 失败：0
```

内置回归基准（`python scripts/report.py`，产物见 `benchmark.json`）：

| 指标 | 离线基线（哈希嵌入 + Mock LLM） |
|---|---|
| 用例通过 | **3 / 3** |
| faithfulness（断言锚定率） | **1.000** |
| context_recall（证据召回） | **1.000** |
| keyword_recall（关键词命中） | **1.000** |
| context_precision | 0.333 |
| answer_relevancy | 0.091 |

> 读法提示：`answer_relevancy` 用嵌入余弦计算，离线哈希嵌入的余弦绝对值天然偏低（高维稀疏投影），**这个数只有在同一后端下做相对比较才有意义**。切到 `fastembed` + bge-small-zh 后绝对值才可直接横向对比。`context_precision` 偏低是因为金标答案是改写后的摘要，与原始跨度的字面重合有限 —— 这是指标定义的保守性，不是检索失效（同组 `context_recall = 1.0`）。

各模块关键不变量（节选，全部有对应单测）：

- Robertson IDF 对所有 `1 <= df <= N` 恒非负（这是 `rank_bm25` 在小语料上会违反的性质）
- 重排器在"完全不 discriminating"时，输出顺序与融合顺序**逐项相同**
- 对抗性重排器（给正确答案打 0 分、其余打 1 分）**无法**把正确答案挤出首位
- 断言 `anchored == True` 必然携带至少一个 span id
- 相同查询在同一管道上跑两次，答案**逐字符相同**
- faiss 索引与纯 Python 内存索引的 top-k 顺序**完全一致**（双实现交叉验证）

## 6. 切换到生产栈

默认链路是**为了可验证**而选的，不是最强的。生产栈全部有 Windows/Linux 预编译轮子，不需要 torch、不需要 GPU：

```bash
pip install -r requirements-prod.txt
```

```bash
# 语义嵌入（ONNX，bge-small-zh-v1.5）
export DIALECTICA_EMBEDDING_BACKEND=fastembed
export DIALECTICA_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 大语料向量索引
export DIALECTICA_VECTOR_BACKEND=faiss

# 交叉编码器重排（护栏照常生效）
export DIALECTICA_RERANK_BACKEND=cross-encoder
export DIALECTICA_RERANK_MODEL=Xenova/bge-reranker-base

# 本地 GGUF 推理
export DIALECTICA_LLM_BACKEND=llamacpp
export DIALECTICA_GGUF_PATH=/path/to/qwen2.5-7b-instruct-q4_k_m.gguf
export DIALECTICA_N_THREADS=4      # 小量化模型是内存带宽受限，线程过多反而更慢
```

远端模型：

```bash
export DIALECTICA_LLM_BACKEND=ollama        # 或 openai
export DIALECTICA_OLLAMA_HOST=http://127.0.0.1:11434
# 或
export DIALECTICA_LLM_BACKEND=openai
export DIALECTICA_OPENAI_BASE_URL=https://api.openai.com/v1
export DIALECTICA_OPENAI_API_KEY=sk-...
```

全部配置项见 `.env.example`。

## 7. 目录结构

```
dialectica-ai/
├─ src/dialectica/
│  ├─ core/       类型 · 协议 · 配置 · 事件总线 · 错误
│  ├─ llm/        mock / llama.cpp / ollama / openai 兼容
│  ├─ retrieval/  tokenizer · chunker · BM25 · dense · fusion · rerank · pipeline
│  ├─ evidence/   extractor · aligner · gate
│  ├─ debate/     roles · convergence · protocol
│  ├─ reason/     router · tools · react
│  ├─ memory/     store（三层记忆 + 遗忘 + 蒸馏）
│  ├─ eval/       metrics · benchmark
│  ├─ api/        FastAPI + SSE
│  ├─ system.py   组合根（只装配，不含业务）
│  └─ cli.py
├─ tests/         152 项单测
├─ scripts/       verify.py · scan_emoji.py · report.py
├─ web/           console.html（单文件零依赖辩论可视化）
├─ docs/          ARCHITECTURE · SPEC · DEPLOYMENT · USAGE · openapi · ADR
├─ requirements.txt / -prod.txt / -dev.txt / .lock.txt
└─ .github/workflows/ci.yml
```

## 8. 文档

- [架构设计](docs/ARCHITECTURE.md) —— 数据流、模块契约、关键算法
- [接口规格](docs/SPEC.md) —— 类型与协议的完整定义
- [部署指南](docs/DEPLOYMENT.md) —— 本机 / Docker / 生产栈切换
- [使用指南](docs/USAGE.md) —— CLI / HTTP / SSE / Python API
- [OpenAPI](docs/openapi.yaml)
- [架构决策记录](docs/decisions/)

## 9. 边界与诚实声明

- 离线默认的 `HashingEmbedder` 是**确定性替身，不是语义模型**。它能保证链路可跑、指标可比，但不具备真实语义泛化能力。真实语义请切 `fastembed`。
- `MockLLM` 是**抽取式**的：它只输出证据块里的原文。任何在 MockLLM 下通过测试的链路，等于证明了"没有凭空编造"，但不等于"模型足够聪明"。
- 内置基准只有 3 例，用途是**回归门禁**，不是能力评测。

## 10. 许可

MIT。作者：晨星。
