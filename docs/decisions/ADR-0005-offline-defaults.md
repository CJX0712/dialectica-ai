# ADR-0005：默认注入零依赖实现，离线可验证是一等需求

作者：晨星 · 状态：已接受

## 背景

AI 系统的"能跑"和"能证明跑对了"是两回事。真实模型栈需要下载权重、需要密钥、需要网络，CI 里跑不了， reviewer 也复现不了。

## 决策

每个外部依赖都提供**可注入的零依赖实现**，并作为默认：

| 契约 | 默认实现 | 性质 |
|---|---|---|
| `LLM` | `MockLLM` | 确定性、**抽取式**（只输出证据块原文） |
| `Embedder` | `HashingEmbedder` | crc32 特征哈希 + 语料 IDF |
| `VectorIndex` | `MemoryIndex` | 精确余弦扫描 |
| `SparseIndex` | `BM25` | 本身零依赖 |
| `Reranker` | `HeuristicReranker` | 字面重合度 |
| `Memory` | `MemoryStore` | 内存 + 遗忘曲线 + 蒸馏 |

## 理由

1. **CI 可跑**：无网络无密钥，19 项端到端检查全绿。
2. **抽取式 Mock 是可证伪的**：链路在 MockLLM 下通过，等于证明了"没有凭空编造"—— 因为 Mock 根本没有编造能力。
3. **确定性**：`crc32` 而非 `hash()`，规避 `PYTHONHASHSEED`；相同输入两次运行逐字符相同。

## 代价与诚实声明

- `HashingEmbedder` 是**确定性替身，不是语义模型**，不具备真实语义泛化能力。
- `MockLLM` 通过测试只证明"没编造"，不证明"够聪明"。
- 离线基线指标（`answer_relevancy ≈ 0.09`）只在同后端下做相对比较才有意义。

生产栈由环境变量切换，见 `docs/DEPLOYMENT.md` 形态 B / C。

## 补充：dim=2048 的来由

`scripts/_tune.py` 在 `dim × bigrams × idf` 网格上搜过（中英混合运维语料，6 查询 / 5 文档）：

| dim | bigrams | idf | top1 | MRR |
|---|---|---|---|---|
| 2048 | yes | yes | 4/6 | **0.806** |
| 512 | yes | no | 4/6 | 0.792 |
| 128 | yes | yes | 3/6 | 0.672 |

低于 512 时哈希碰撞主导，排序崩塌。
