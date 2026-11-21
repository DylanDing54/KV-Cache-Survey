# DeepInfer：可行性验证与判定标准

> 推理引擎 · HDD 可行性分析

| 判断层次 | 结论 | 依据与缺口 |
|---|---|---|
| 接口可行 | 部分可行 | 3FS 已开源，但设计文档按“每个存储服务管理若干本地 SSD”来描述，没有提到机械盘，在 HDD 上运行属于官方未覆盖的用法；kvcache-store、取数进程等 KV 缓存系统没有开源，DeepInfer 开源版本身不含 KV 缓存系统，没有可直接挂接 HDD 层的接口 [DS1][DS2][DS12] |
| 性能可行 | 作主持久层不可行；作 SSD 之后的冷层待验证 | 推荐配置为每台 ≥ 40 GB/s 读、每块 SSD 约 2 GB/s（本文估算），读取粒度在 10–100 KiB 量级；HDD 只适合在 SSD 之后承接冷条目，需要验证从 HDD 回填的时延能否被请求排队时间吸收 |
| 生产可行 | 无官方证据 | 官方没有 HDD 部署或测试数据；TTL 已接近最大可达命中率，扩容的边际收益小；官方提到的优化方向是用内存减少 SSD 写入（CXL L2），而不是向下增加机械盘层 [DS2] |

### 验证顺序与通过条件

| 试验 | 方法与判定 |
|---|---|
| 保留期收益 | 按 agent 与 chatbot 分开回放前缀复用间隔，只统计 SSD 淘汰后再次使用的前缀。按节省的 prefill GPU 时间核算收益，同时计入冷层写入、索引与读取成本。 |
| 读路径时限 | 对相同前缀比较 SSD 命中、HDD 命中和重算，分别记录元数据查询、排队、取数与 H2D 时间。只有未被计算掩盖的读延迟仍低于重算成本，且目标负载下的 P99 TTFT 达标，才考虑接入在线路径。 |
| 故障与降级 | 注入缺块、校验失败、慢盘与写回中断。检查过期索引能否撤销，超时能否转入重算，恢复后能否拒绝不完整对象。离线回放通过后，还需用有调度器的原型复测端到端尾延迟。 |

### 资料边界

KV 缓存系统（kvcache-store、取数进程、GC Worker）的源码没有开源，本章对它的描述全部来自 DeepInfer 仓库中的部署 Q&A（截至 2026-11-03 的 v0.2）。Q&A 引用的知乎博文以及 Day 6 的中文版都在知乎上，本次抓取返回 403，没有读到原文。V4.1 每 token 890 B 是依据 HF 配置与 `cache_engine.py` 分配逻辑推算的，没有考虑引擎实际配置下的对齐、Engram 与草稿层，也没有经过运行验证。API 指南中“固定 token 间隔”的具体数值、公共前缀检测算法、10+2 EC 是否已并入 3FS 开源主线，在官方材料中都没有找到。

## 本章资料

- **[DS1]** [DeepSeek 官方 GitHub · deepseek-ai/DeepInfer README（固定提交 a679bb86ce36e883a98a97733492b686e5b9a08e，2026-11-03；含 doc/images/architecture_overview.png，采集于 2026-11-18）](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e)
- **[DS2]** [DeepSeek 官方 GitHub · DeepInfer doc/inference_deployment_qa.md《DeepSeek V4.1 Flash Inference Deployment Q&A》（v0.2，2026-11-03；KV cache storage、3FS、负载均衡、SWA bounded replay，采集于 2026-11-18）](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)
- **[DS12]** [DeepSeek 官方 GitHub · 3FS docs/design_notes.md（组件、CRAQ、chunk engine、USRBIO，采集于 2026-11-18）](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md)
