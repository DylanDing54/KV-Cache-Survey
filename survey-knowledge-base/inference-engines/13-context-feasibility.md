# NVIDIA：完整 KV 是否值得长期保存

> 需求可行性 · 企业上下文研究

### You don’t need long context

*You don’t need long context* 研究通过语义压缩和按需读取缩短活跃上下文。FHA 累积完整历史，EoCC 达阈值压缩，SAC 每个动作后更新紧凑状态。它不是 KV 无损压缩或 HDD offload 算法。 [N]

论文表 2 中，Claude Opus 4.5 的 SAC 在两项基准下将每任务峰值上下文中位数降至 8K / 9K，准确率没有下降；GPT 5.6 Sol 的两项准确率则分别下降 2.0 / 3.3 个百分点。附录中 Qwen 3 Coder Next 在 SWE-bench Verified 的 EoCC 16K 总上限 / 12K 阈值下，从 53.0% 降到 3.6%。不能把标题外推到所有模型。

![NVIDIA 原论文 PDF 第 1 页：摘要与图 1](../figures/context-feasibility-1.png)

*NVIDIA 原论文 PDF 第 1 页：摘要与图 1* — [原始来源](https://arxiv.org/abs/2611.05306v1)

### 对最底层 HDD 的可行性影响

本文分析：保存完整 KV 可以省去重复 prefill，但不消除长上下文 decode 成本。语义压缩会改变后续上下文和 KV 前缀，降低旧 KV 的直接复用价值；需要分别测量任务质量、未来复用率与全生命周期成本。

图 1 最高约 7 倍成本差距来自特定上下文长度与 decode 成本口径，并非 HDD 读取成本或统一账单降幅。压缩具有质量风险，不能据此认定所有旧 KV 均可丢弃。

## 本章资料

- **[N]** [Belcak et al. · You don’t need long context，arXiv:2611.05306v1](https://arxiv.org/abs/2611.05306v1)
