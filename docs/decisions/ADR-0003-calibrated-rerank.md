# ADR-0003：重排器不得拥有最终排序权

作者：晨星 · 状态：已接受

## 背景

交叉编码器重排是 RAG 的标准配方。但当重排器在查询语言上力不从心时（最典型：英文 reranker + 中文查询），把重排顺序直接当最终排序会**无人制衡地**压掉正确答案。

## 实测

中文查询评测集上，打开重排后 top-1 命中从 **11/12 掉到 5/12**。

## 决策

重排分数必须先过三层护栏才进入最终序：

1. **z-score 归一化**：在候选集内抹掉量纲，避免原始分数量级主导
2. **边际门控**：`zr_top1 - zr_top2 < min_margin` 时判定"重排不 discriminating"，整单回退到融合序
3. **加权混合**：否则 `final = α·zr + (1-α)·zf`

外加一条硬保证：**融合 top-1 永不被 z_floor 剪掉**。

## 理由

重排器是**强先验但可能错**的信号，融合序是**弱先验但稳定**的信号。让前者影响后者、而不是取代后者，是这两类信号唯一正确的组合方式。

## 代价

- 重排器在它真正擅长的场景（同语言、判别力强）里会被稀释。α 可调到 0.9（单测 `test_strong_signal_is_respected` 覆盖该路径）。
- 多一个超参 `min_margin`。

## 验证

- `test_constant_scorer_preserves_fused_order`：重排器完全不 discriminating 时输出顺序 == 融合顺序
- `test_adversarial_scorer_cannot_bury_the_best_candidate`：给正确答案打 0 分、其余打 1 分的对抗性打分器，正确答案仍在首位
- `test_output_is_subset_and_never_drops_fused_top1`
