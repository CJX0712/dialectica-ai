# 辩衡 Dialectica · 使用指南

作者：晨星

---

## 1. 五分钟上手

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python -m dialectica.cli ingest --file my_corpus.txt
python -m dialectica.cli ask "这里讲了什么"
python -m dialectica.cli serve     # 然后打开 web/console.html
```

## 2. CLI

```
dialectica serve    [--host HOST] [--port PORT]
dialectica ingest   --file PATH [--pdf]
dialectica ask      QUERY
dialectica debate   QUERY [--top-k N] [--json]
dialectica evaluate
dialectica health
```

### ingest

`--file` 支持三种格式：

| 格式 | 切分方式 |
|---|---|
| `.txt` | 按空行分段 |
| `.json` | 数组；元素为字符串，或含 `text` 字段的对象 |
| `.jsonl` | 每行一条 |
| `.pdf` | 加 `--pdf`，按页（需要 `pypdf`） |

### debate

```bash
python -m dialectica.cli debate "重排模型为什么会让中文检索结果变差" --json
```

`--json` 输出完整结构：每轮提案、断言裁决、反对意见、delta，以及证据跨度。

### evaluate

跑内置 3 例回归集，输出 `benchmark.json` 同构的 JSON。全部用例通过时退出码 0，否则 1 —— 可直接接进 CI 门禁。

## 3. Python API

```python
from dialectica import Dialectica, Document

s = Dialectica()
s.add_documents([
    Document.create("沙箱守卫会拦截对 site-packages 的写入与删除……", title="pip", source="ops"),
    Document.create("机器上的 SOCKS5 代理会让 pip 报 Missing dependencies……", title="socks"),
])

r = s.debate("沙箱里为什么无法安装 Python 包")

print(r.answer)
print(r.converged, r.round_count)
print(r.metrics)          # faithfulness / stability / 融合权重 …

for rnd in r.rounds:
    print(f"第 {rnd.index+1} 轮  delta={rnd.delta:.3f}")
    for c in rnd.claims:
        print(f"  [{c.verdict.value}] {c.text[:40]}  {c.evidence}")
    for o in rnd.valid_objections:
        print(f"  反对 {o.citations}: {o.text}")

for h in r.hits:
    print(h.span_id, h.score, h.components)
```

### 注入自己的实现

```python
from dialectica import Dialectica
from dialectica.llm.mock import MockLLM
from dialectica.retrieval.dense import HashingEmbedder
from dialectica.retrieval.pipeline import RetrievalPipeline

s = Dialectica(
    llm=MockLLM(max_sentences=2),
    embedder=HashingEmbedder(dim=1024),
    retriever=RetrievalPipeline(embedder=HashingEmbedder(dim=1024)),
)
```

### 只用某一层

```python
from dialectica.retrieval import RetrievalPipeline
from dialectica.evidence import extract_claims, align_claims, EvidenceGate
from dialectica.debate import DialecticProtocol
```

## 4. HTTP

```bash
# 入库
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["沙箱守卫会拦截对 site-packages 的写入与删除……"]}'

# 辩论
curl -X POST http://127.0.0.1:8000/debate \
  -H "Content-Type: application/json" \
  -d '{"query": "沙箱里为什么无法安装 Python 包", "top_k": 5}'
```

响应结构：

```jsonc
{
  "answer": "……",
  "converged": true,
  "metrics": { "rounds": 1.0, "faithfulness": 1.0, "retrieval_w_dense": 0.771, … },
  "claims":  [{ "text": "…", "verdict": "supported", "evidence": ["span::chk_…"], "anchored": true }],
  "evidence":[{ "span_id": "span::chk_…", "score": 1.08, "text": "…", "components": {…} }],
  "rounds":  [{ "index": 0, "proposal": "…", "delta": 1.0, "claims": […], "objections": […], "valid_objections": [] }]
}
```

### SSE 流式

```bash
curl -N -X POST http://127.0.0.1:8000/debate/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "沙箱守卫拦截写入怎么办"}'
```

事件序列：`evidence` → `round`（每轮一个）→ `answer` → `done`。控制台的"实时模式"就是消费这个流。

## 5. Web 控制台

`web/console.html` 是**单文件零依赖**的：双击打开，或由任意静态服务器托管。

功能：

1. 左上填 API 地址 → 自动健康检查
2. "载入示例语料" → 灌内置 8 条运维知识 → "入库"
3. 输入问题 → "发起辩论"
4. 右侧展示：指标条（轮数 / 收敛 / 忠实度 / 稳定性 / **dense-sparse 自适应权重条**）、最终答案、逐轮辩论卡片（提案 → 断言裁决 → 有效/无效反对）、证据跨度列表
5. 勾选 "SSE 流式" 可看到逐轮推送

## 6. 确定性工具路由

以下问题**不经过辩论**，直接走工具：

| 类型 | 例子 | 路由 |
|---|---|---|
| 算术 | `12*(3+4)`、`7/2` | `calculator`（AST 白名单，拒绝代码注入） |
| 时间 | `现在几点`、`what time is it` | `clock` |
| 单位 | `12 km to m`、`1 h to min` | `unit` |

小模型在工具给出结果后自己重算是常见错误（实测把 `12*(3+4)` 算成 72）。辩衡的做法是**先执行工具，把结果作为 Observation 注入，并明确禁止重算**。

## 7. 记忆

```python
s.memory.remember("用户偏好：回答要带证据", salience=1.0)
s.memory.recall("证据要求", top_k=3)
s.memory.distill()        # 情景 → 语义，幂等
s.memory.forget(threshold=0.05)
s.memory.stats()          # {"working": n, "episodic": n, "semantic": n}
```

## 8. 评估

```python
from dialectica.eval import run_benchmark
from dialectica.system import Dialectica
from dialectica.retrieval.dense import HashingEmbedder

report = run_benchmark(lambda: Dialectica(), HashingEmbedder(dim=2048))
print(report.as_dict())
```

自定义评估集：

```python
fixtures = [
  {
    "query": "…",
    "docs": ["…", "…"],
    "gold_answer": "…",
    "keywords": ["…", "…"],
  }
]
run_benchmark(lambda: Dialectica(), embedder, fixtures)
```

> 每个用例都在**全新管道**上跑。复用已写入数据的单例管道会让 recall 因 doc_id 重复而漂移。

## 9. 常见问题

**Q：答案很短/被剪空了？**
说明大部分断言没通过证据门禁。调低 `DIALECTICA_SUPPORT_THRESHOLD`（默认 0.30）或调高 `DIALECTICA_MAX_UNANCHORED_RATIO`（默认 0.20）。但更好的做法是检查语料里是否真的有答案。

**Q：一轮就收敛，是不是没跑起来？**
不是。收敛的定义是"无有效反对"。证据充分时首轮就该收敛。想观察多轮，把 `MockLLM(max_sentences=1)` 注入进去（提案覆盖不全，批评者必然反对）。

**Q：`answer_relevancy` 只有 0.09？**
离线哈希嵌入的余弦绝对值天然偏低（高维稀疏投影）。这个数只在**同一后端下做相对比较**才有意义。切到 `fastembed` 后绝对值才可直接横向对比。

**Q：切到 fastembed 报网络错误？**
首次运行需要下载 ONNX 模型。离线环境请提前在有网机器上下载缓存，或保持 `EMBEDDING_BACKEND=hashing`。

**Q：改了源码但测试行为没变？**
删掉 `__pycache__`。沙箱环境的 mtime 有时不触发失效。
