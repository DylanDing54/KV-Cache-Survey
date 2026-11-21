# DeepInfer：整体架构与 KV 缓存管理

> 推理系统 · DeepSeek 官方文档、开源仓库与技术报告

### 整体架构：入口、分阶段计算与共享存储

DeepSeek V4.1 Flash 的线上服务把视觉编码、提示词处理（Prefill）和逐 token 生成（Decode）部署在可独立扩缩的计算资源上。DeepSeek Recipe 在 API 协议与模型 token 之间转换；Entrypoint 协调请求进入相应阶段。引擎实例执行分配到本阶段的批次；独立的 KV 存储集群向不同 Prefill 实例提供已计算的前缀。下图来自官方部署资料，开源 DeepInfer 仓库提供了其中可独立运行的离线推理核心。 [DS1][DS2]

![DeepInfer 仓库 doc/images/architecture_overview.png：DeepSeek V4.1 Flash 推理系统参考架构。DeepSeek Recipe 负责协议与 token 互转，Entrypoint 分发到 Vision Encoder / Prefill / Decode 三类负载均衡器；蓝色箭头为 KV 数据流：External KVCache Storage（3FS）向 Prefill Service 供数，Decode Service 写入 3FS](../figures/deepseek-1.png)

*DeepInfer 仓库 doc/images/architecture_overview.png：DeepSeek V4.1 Flash 推理系统参考架构。DeepSeek Recipe 负责协议与 token 互转，Entrypoint 分发到 Vision Encoder / Prefill / Decode 三类负载均衡器；蓝色箭头为 KV 数据流：External KVCache Storage（3FS）向 Prefill Service 供数，Decode Service 写入 3FS* — [原始来源](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/images/architecture_overview.png)

| 服务或资源 | 接收的工作 | 决策依据与输出 |
|---|---|---|
| Recipe / Entrypoint | 协议请求、历史消息及可选图像。 | Recipe 将不同协议映射为统一消息格式并编码；入口在 Prefill 路由前查询前缀命中长度。返回时由 Recipe 解析 token 流中的文本、推理边界和工具调用。 |
| Vision Encoder | 含图片的请求需要生成视觉 embedding。 | 官方部署按机器选择编码资源，再在机内按正在处理的图像 patch 数选择 GPU。纯文本请求可直接进入语言模型路径。 |
| Prefill 实例 | 提示词、可复用前缀，以及需要计算的新后缀。 | 普通八卡机承担提示词计算。负载均衡器按上下文长度分桶，各引擎按剩余计算配额取任务；共享存储允许不同机器取得同一前缀。 |
| Decode 实例 | 接收 P→D 传来的序列状态和 KV，持续生成 token。 | 64–128 卡超节点承载较大规模专家并行。路由同时考虑活跃请求数和总 KV 长度，两者分别影响每步批量和注意力开销。 |
| kvcache-store / 3FS | 匹配前缀元数据、读取历史 block KV、后台写回。 | 3FS 以 SSD 保存分页 block KV；滑动窗口注意力（SWA）的状态另由主机内存和目录管理。两类数据沿不同路径恢复。 |

这里有两级调度。集群级路由决定请求交给哪台 Prefill 或 Decode 机器；实例内调度决定本步运行哪些序列、每条处理多长。在 Prefill 端，即使两个请求还有同样多的新 token，已存在的上下文长度也会改变计算耗时。官方实现以新 token 数和已有前缀长度估算成本，再按单步时间预算选择分块大小。 [DS2]

### 硬件对接：超节点内的专家通信、跨节点 KV 与主机回填

**模型计算。**DeepSeek 的 V4.1 Flash 部署在较小的 Prefill 组和较大的 Decode 组上运行：Prefill 已能为每个专家形成较大批量，较小的 EP 组有助于控制通信；Decode 使用 Wide-EP，把专家分摊到更多卡上。路由后的激活要到相应专家所在卡，结果再汇合。负载均衡器跟踪热点并调整冗余专家布局，官方还披露通过 NVLink 搬移组内专家权重。因而逐 token 延迟既受 HBM 访存影响，也受最慢专家和组内同步影响。 [DS2]

**跨机器及存储。**部署 Q&A 对 P/D 互连给出每八张 GPU 约 400 Gb/s 的预算。这个数字描述链路规划。另一份 3FS 资料中的约 40 GiB/s 是存储读吞吐，两者测试对象、方向和单位不同。外部 block KV 命中后，取数进程先将数据从 3FS 放入主机共享内存，随后引擎线程做逐层 H2D。官方解释，细粒度 KV 上 CUDA 拷贝引擎效率较低，GPU 平台改由 RDMA 网卡通过 GDR 执行搬运，以利用 PCIe 并释放 SM。另有 SWA 状态可能从持有它的计算机经 400G VLAN 拉回。 [DS2][DS11][DS12]

**硬件平台边界。**公开仓库列出 NVIDIA SM100 和华为昇腾 950，后者使用 CANN。Q&A 提供了同一功能在不同硬件上的具体差异：GPU 上的 Engram 词表可以分片留在主机内存，由网卡执行单边 RDMA 读取；该版本的 NPU 实现将 Engram 留在设备内存。Engram 是独立于 KV 的模型数据。这一差异也说明，评估 NPU 的 HBM 预算和互连流量时要按该平台实现核算。 [DS1][DS2]

### 一次多轮文本请求：从前缀匹配到返回答案

1. **入口先查元数据。**Recipe 生成 token 序列后，Entrypoint 向 CHECK 角色查询哈希链，确定最长可复用前缀。命中长度必须在路由前已知，这一段时间计入请求启动开销。
2. **Prefill 汇合两条数据流。**命中时，取数进程从 3FS 异步读至主机共享内存，再将数据搬入 HBM；执行端为新后缀计算 KV。无可复用前缀时，从头执行 prefill。取数与排队、计算能重叠多少，决定了命中对首 token 延迟的实际贡献。
3. **Decode 接手。**Prefill 完成后，P→D 路径传递继续生成所需的 KV 与状态。Decode 逐步推进序列，生成 token 交由前端解析并返回。后台写回用于后续请求复用，读入缓存时由引擎校验数据。

在开源代码里可以继续追到单实例的一步：`LLMEngine.add_request()` 接入请求；`step()` 调用 C++ 调度器，取得序列元数据及本轮计划；引擎按计划处理需克隆或清零的状态与块，再通过 `execute_model_step` 驱动 Worker。执行结果进入后处理，供序列继续生成或结束。线上跨实例的路由、缓存查询与 3FS 取数则依据部署 Q&A 描述；离线引擎单步仅覆盖其中的模型执行部分。 [DS17][DS2][DS1]

DeepSeek 官方 GitHub 组织 deepseek-ai 下有仓库 `deepseek-ai/DeepInfer`（2026-10-16 创建，Apache-2.0），README 称它是“驱动 DeepSeek 大规模线上服务的高性能推理框架”，目前只支持 DeepSeek-V4.1-Flash，硬件为 NVIDIA SM100 GPU 与华为昇腾 950 NPU [DS1]。但这次开源只包括离线推理所需的引擎核心。README 明确列出“完整推理服务所需、本次未包含”的三项：**KV 缓存系统**、VE / Prefill / Decode 分离（引擎侧功能大部分已实现）、跨数据并行副本的负载均衡 [DS1]。仓库附带的部署 Q&A 也说明，KV 缓存管理与负载均衡组件与内部基础设施耦合较紧，代码尚未完整开源，相关设计见一篇知乎博文与这份 Q&A [DS2]。因此本章分两部分：引擎内部的 KV 管理依据固定提交 `a679bb86`（2026-11-03）的源码；KV 存储系统依据官方 Q&A、DeepSeek API 文档、open-infra-index 与 3FS 仓库的描述，采集于 2026-11-18。结论先行：DeepSeek API 所说的“硬盘缓存”，在官方材料中落在**基于 SSD 的 3FS 专用存储集群**上，没有任何官方材料提到 HDD。

### 定位与演进：从“不开源引擎”到 DeepInfer

2025 年 4 月，DeepSeek 在《The Path to Open-Sourcing the DeepSeek Inference Engine》中写道：内部推理引擎基于一年多前 fork 的早期 vLLM，已针对 DeepSeek 模型深度定制，又与集群管理等内部设施紧耦合，因此决定不整体开源，改为把可复用的功能拆成独立库、把优化贡献给现有开源项目 [DS10]。2026 年 10 月发布的 DeepInfer 在这一点上前进了一步：README 致谢称它“最初主要构建在 vLLM 之上”，并借鉴了 SGLang 与 TensorRT-LLM；但与 KV 存储相关的部分仍然没有放出 [DS1][DS2]。

与开篇的 V4.1 Flash 架构相比，早期 V3 / R1 部署采用了不同的资源配比。2025 年 2 月 Open Source Week Day 6 公布的 V3 / R1 推理系统是 P/D 分离加跨节点 EP：Prefill 每个部署单元 4 节点（路由专家 EP32、MLA 与共享专家 DP32），Decode 每个部署单元 18 节点（EP144 / DP144），图中外部 KVCache 存储标注为“Optional” [DS9]。V4.1 Flash 的 Q&A 则把系统描述为“E/P/D 分离加专用存储节点”：Prefill 用普通 8 卡机、EP2–8 / DP2–8；Decode 用 64–128 卡超节点、EP64–128（Wide-EP）；视觉编码器单卡部署；暂不使用 TP 与 CP。设计目标第三条直接写明**KV 缓存保留 72 小时以上** [DS2]。

![open-infra-index Day 6《DeepSeek-V3/R1 Inference System Overview》中的在线推理系统图（2025-02）：API Server、Prefill / Decode Load Balancer、各实例内的 Expert-Parallel Load Balancer，以及标注为 Optional 的 External KVCache Storage](../figures/deepseek-2.png)

*open-infra-index Day 6《DeepSeek-V3/R1 Inference System Overview》中的在线推理系统图（2025-02）：API Server、Prefill / Decode Load Balancer、各实例内的 Expert-Parallel Load Balancer，以及标注为 Optional 的 External KVCache Storage* — [原始来源](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)

### KV 有多小：从 MLA 到 V4.1 的压缩 KV

DeepSeek 自己把硬盘缓存的前提归结为 KV 体积。2024-08-02 的 API 公告写道：DeepSeek-V2 提出的 MLA 结构“大大压缩了上下文 KV Cache 的大小，使得存储所需要的传输带宽和存储容量均大幅减少，因此可以缓存到低成本的硬盘上” [DS7]。V2 论文给出的数字是：与 DeepSeek 67B 相比，KV 缓存减少 93.3% [DS14]。按 DeepSeek-V3 官方配置（61 层，`kv_lora_rank` 512，`qk_rope_head_dim` 64，128 个注意力头，`qk_nope_head_dim` 128，`v_head_dim` 128）估算：MLA 每层只缓存 576 维潜向量，BF16 下每 token 共 576 × 2 B × 61 = 70,272 B；同样的头配置若按 MHA 缓存完整 K（192 维）与 V（128 维），每 token 约 4,997,120 B，前者约为后者的 1.4%（本文估算） [DS16][DS15]。Day 6 说明 V3 / R1 线上的核心 MLA 计算使用 BF16 [DS9]。

V4.1 Flash 又往下压了一个数量级。FlashMLA README 给出 V4.1 的两种逐 token 格式：MXFP8 为 528 B（512 B e4m3 数据 + 16 B UE8M0 scale），FP4 为 288 B（256 B e2m1 数据 + 32 B e4m3 scale） [DS5]。Q&A 说明 main KV 用 per-16 e4m3 scale 的 FP4，indexer KV 用 MXFP4 [DS2]。DeepInfer 的 `ModelMetadata` 把块大小固定为 128 token；`cache_engine.py` 按层分配 KV：压缩比为 r 的层每块存 128 / r 个条目，跨层共享 KV 的层不再占用新空间 [DS3]。HF 配置中只有 2、8、14、20 四层是 KV 源层（`kv_source_layer_ids`）。其中 2–19 层压缩比为 2，20–39 层为 1，indexer 维度为 128 [DS4]。据此推算，每个 128-token 块写入 3FS 的 block KV（main KV + indexer K）为 3 × (64 × 288 + 64 × 68) + (128 × 288 + 128 × 68) = 113,920 B，约合**每 token 890 B**（本文估算）。此外，每个序列还有一份与上下文长度无关的 SWA 状态，Q&A 给出的大小为 20 层 × 134 token × 528 B = 1,415,040 B [DS2]。按这一估算，128K 上下文的 block KV 约 111 MiB，同长度的 V3 BF16 MLA 约 8.6 GiB。

![DeepSeek V4.1 Flash 的 KV 组成：40 层中 KV 源层与压缩比分布、每个 128-token 块的 block KV 字节、每序列 SWA 状态，以及与 V3 MLA / MHA 的每 token 字节对比（对数刻度，数值为本文估算；本文根据 DeepSeek 官方资料绘制：HF config.json、DeepInfer cache_engine.py 与 Q&A、FlashMLA README、DeepSeek-V3 配置）](../figures/deepseek-3.svg)

*DeepSeek V4.1 Flash 的 KV 组成：40 层中 KV 源层与压缩比分布、每个 128-token 块的 block KV 字节、每序列 SWA 状态，以及与 V3 MLA / MHA 的每 token 字节对比（对数刻度，数值为本文估算；本文根据 DeepSeek 官方资料绘制：HF config.json、DeepInfer cache_engine.py 与 Q&A、FlashMLA README、DeepSeek-V3 配置）* — [原始来源](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py)

### 三层存放与端到端数据通路

Q&A 用一张表说明 KV 放在哪里：

| 内容 | 介质 | 冗余 | 管理方 |
|---|---|---|---|
| 正在计算的 block KV + state KV | 推理卡 HBM | — | Prefill / Decode 实例 |
| SWA KV | 本机主机内存 | 单副本，不进 3FS | Prefill / Decode 实例；跨机命中经 400G VLAN 拉取 |
| 分页 block KV | 专用 3FS 存储集群 | 应用层不复制，依赖 3FS 自身冗余 | kvcache-store |

来源： [DS2]。之所以不用“本地内存 + 分级存储”，Q&A 给出三点理由：专用存储集群让任何一台 Prefill 机器都能命中任何前缀，避免前缀亲和与负载均衡路由冲突；存储节点可以和计算节点分开扩缩容；KV 不受计算节点故障影响，而计算节点的故障率比存储节点高一个数量级以上 [DS2]。所以 Prefill Load Balancer 只按上下文长度分 4 档（<16K / 16–64K / 64–256K / ≥256K）、按剩余算力配额拉取请求，明确“不做前缀亲和路由” [DS2]。

一次请求的 KV 路径如下图。（1）**匹配**：Entrypoint 向本地 CHECK 角色的 kvcache-store 做哈希链前缀匹配，只查元数据。命中长度必须在路由之前算出，这一步的开销无法隐藏。（2）**取数**：一个独立进程异步把 KV 从 3FS 读进主机共享内存。（3）**swap-in**：引擎进程内的线程把 KV 从共享内存搬进 HBM，目标是逐层 H2D 与前向计算重叠。由于 CUDA copy engine 处理细粒度 KV 块拷贝的性能较差，这一步改用 GDR，由 RDMA 网卡打满 PCIe，不占用 SM [DS2]。（4）**P→D**：SWA KV 与 block KV 走同一条 P 到 D 的传输路径 [DS2]。（5）**写回**：Decode 不做 KV offload，但 decode 阶段产生的 KV 仍会以 write-back 方式写盘以提高命中率。写失败不报告给引擎，读取时由引擎严格校验数据 [DS2]。官方架构图中，指向 3FS 的写入箭头只从 Decode Service 发出，Prefill Service 与 3FS 之间只画了读方向的箭头 [DS1]。Prefill 阶段产生的 KV 是否经 Decode 统一落盘，文中没有明说。

![DeepSeek V4.1 Flash 推理系统的 KV 缓存数据通路：Entrypoint 元数据匹配 → Prefill 取数进程（3FS → 主机共享内存）→ GDR swap-in 到 HBM → P→D 传输 → Decode 写回 3FS；SWA KV 留在主机内存，由 Entrypoint 内存目录定位（本文根据 DeepSeek 官方资料绘制：DeepInfer 部署 Q&A 与架构图）](../figures/deepseek-4.svg)

*DeepSeek V4.1 Flash 推理系统的 KV 缓存数据通路：Entrypoint 元数据匹配 → Prefill 取数进程（3FS → 主机共享内存）→ GDR swap-in 到 HBM → P→D 传输 → Decode 写回 3FS；SWA KV 留在主机内存，由 Entrypoint 内存目录定位（本文根据 DeepSeek 官方资料绘制：DeepInfer 部署 Q&A 与架构图）* — [原始来源](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)

开源引擎内部的 KV 管理相当简单。`csrc/engine/block_manager.hpp` 的 `Blocks` 用数组内嵌的 `prev` / `next` 维护空闲链表，块按 `ref_count` 管理：`allocate()` 从链表头取块，`free()` 在引用计数归零后放回链表尾；`BlockManager.free_blocks_nolock()` 逆序释放，`pin_block()` 供 fork 出的序列共享父序列的块。每个序列另从 state 管理器分得一个 state 槽位存放 SWA 等状态 [DS3]。与 vLLM、SGLang 不同，这里**没有 GPU 内的前缀哈希表**，块释放后也不保留可命中的内容。前缀复用全部交给外部的 kvcache-store 与 3FS，引擎只需知道某个序列已有多少 token 算好（`num_cached_tokens`），然后把命中部分装入分配好的块 [DS3][DS2]。

### 内存系统管理：分页 KV、序列状态与跨请求恢复

DeepInfer V4.1 Flash 同时管理按上下文长度增长的 block KV，以及每个活动序列独占的 state 槽。两类对象的索引、增长方式与可恢复性不同。开源核心能核查设备侧分配与布局；跨机目录、持久缓存和线上回收策略的证据来自 DeepSeek 部署 Q&A。 [DS18][DS19][DS2]

#### 请求准入必须同时拿到两种资源

开源块管理器的 try_allocate_seq 在同一临界区里检查两项条件：block 池能否满足本请求所需页数，启用 state 管理器时能否取得一个序列槽。两者都满足才写入序列的 block_table 和 state_id；暂时资源不足则返回等待，永远超过块池可支持范围则报错。已分配块用引用计数保护，解除最后一个引用后回到空闲链表。由此可以解释一种仅看“剩余 KV 页数”无法诊断的现象：即便还有空页，活动序列已占满 state 槽时，新请求仍不能准入。 [DS18]

#### 同一请求里，哪些字节应该占用哪一级内存

| 对象 | 设备侧组织 | 跨请求保存与恢复 |
|---|---|---|
| 分页 block KV | 按块索引存放压缩 attention KV 和相应的 indexer K。共享 KV 的消费层映射到拥有数据的源层；容量核算须按实际拥有缓存的层计数。 | 3FS 保存分页数据。入口先对前缀做元数据匹配，再异步取入主机共享内存，逐层送入 HBM。有效 block KV 可在 SWA 状态缺失时继续复用。 [DS19][DS2] |
| 每序列的 SWA 窗口 | state 槽前段放跨请求恢复所需的 SWA。Decode 在跨过指定边界时把当前窗口复制到设备端副本区，请求结束时取最近完成的快照。 | 官方披露 V4.1 每序列保存段为 1,415,040 B（20 层 × 134 token × 528 B）。保存段以单副本留在产生它的机器的主机内存，由入口目录定位；跨机命中经 400G VLAN 拉回。源码名 state_size_to_disk 在这条线上并不代表写入 3FS。 [DS19][DS2] |
| 消费层窗口与压缩中间状态 | 运行时仍需设备内存。布局把可省略部分放在 state 槽后段，P→D 交接与跨请求保存分别选择所需区域。 | YOCO 消费层的窗口可由 prefill 重建；压缩状态在与存储粒度对齐的边界处无跨界依赖。因此，整槽字节数不能作为每次缓存保存的写入量。 [DS19][DS2] |

#### 为什么丢失窗口后仍能复用较长的 block KV

部署 Q&A 把两次查找分开：先按 128-token 粒度确定 block KV 的可命中前缀，再查相应边界附近的 SWA 快照。窗口缺失时回退一个块，对末尾 128 个 token 做有界重放以重建近似的 SWA 状态；已命中的 block KV 保留，重放不覆盖它。这使单副本主机状态的丢失可局部恢复，代价落在请求的额外 prefill。此处的重放窗口和近似恢复属于该版本模型的实现。 [DS2]

#### 缓存跨越进程生命周期后，如何判定旧字节还有没有效

DeepSeek 用部署配置、模型权重和用户组共同标识兼容域。权重、块字节布局、tokenizer，或任何会改变 KV 数值的框架更新，都可能让旧条目失去复用资格；团队披露线上未做缓存格式迁移。Decode 异步写回失败不回报给引擎，读入时严格校验。3FS 每个存储节点运行 GC Worker，按水位回收最久未使用的数据；kvcache-store 自身没有独立到期逻辑。因此要区分三个事件：当前 API 请求生成结束、后台产生可被复用的副本、旧副本最终被回收。 [DS2]

### 命中判定、计费与线上命中率

对 API 用户而言，规则经历过一次变化。2024-08-02 硬盘缓存上线时，公告写明：只有从第 0 个 token 起相同的前缀才算重复；缓存以 **64 token** 为一个存储单元，不足 64 token 的内容不缓存；缓存“尽力而为”，不保证 100% 命中；不再使用的缓存一般在几小时到几天内自动清空；每个用户的缓存逻辑上相互隔离。当时命中价格为 0.1 元 / 百万 tokens，未命中为 1 元；128K 输入且大部分重复的请求，实测首 token 延迟从 13 秒降到 500 毫秒 [DS7]。现行的“上下文硬盘缓存”指南不再提 64 token，而是说“受 Sliding Window Attention 机制的影响”，每条缓存前缀是一个独立的完整单元，后续请求必须**完整匹配**某个“缓存前缀单元”才能命中。落盘时机有三种：请求结束位置（用户输入末尾、模型输出末尾各产生一个单元）、系统检测到的公共前缀，以及长输入 / 长输出中按固定 token 间隔截取。缓存构建耗时为秒级，命中情况通过 `usage` 中的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 返回 [DS6]。现行价格页上，`deepseek-flash`（DeepSeek-V4.1-Flash-1017）高峰时段输入命中 0.04 元、未命中 2 元 / 百万 tokens，命中价为未命中价的 2%；`deepseek-v4-pro` 为 0.30 元对 9.0 元。空闲时段价格减半，缓存存储本身不另收费 [DS8][DS7]。

引擎侧的判定规则解释了“完整单元”的由来。Q&A 写明：命中长度由 block KV 决定，最后一块是否需要重算由 SWA KV 决定。对长度为 L 的输入，命中上限为 L − 134（128 窗口 + 6 个推测回退位置）。先查 block KV，得到 128 的整数倍 L1；再查 SWA KV，若 [L1 − 128, L1] 内存在 SWA 快照则命中 L1，否则回退为 L1 − 128。SWA 只在 128 token 整数倍处保存；prefill 只在最后一个 chunk 保存，decode 每跨过一个 128 边界就在显存中复制一次窗口 [DS2]。缺失的 SWA 用“SWA bounded replay”补：只重算命中前缀末尾的 128 token，重算代价与上下文长度和层数无关，block KV 原样复用 [DS2][DS1]。缓存兼容性以“(部署配置, 模型权重, 用户组)”为键，权重、块字节布局、tokenizer 的变化，或改变 KV 数值的框架改动，都会使旧缓存失效；生产中没有做过缓存格式迁移 [DS2]。

![DeepSeek 硬盘缓存的命中判定：上半部分为 API 文档的规则（2024-08 的 64 token 存储单元 → 现行“缓存前缀单元”的三种落盘时机），下半部分为 DeepInfer Q&A 描述的两段式查询（block KV 得 L1，SWA 决定 L1 或 L1−128）与 bounded replay（本文根据 DeepSeek 官方资料绘制：API 指南、2024-08-02 公告、DeepInfer 部署 Q&A）](../figures/deepseek-5.svg)

*DeepSeek 硬盘缓存的命中判定：上半部分为 API 文档的规则（2024-08 的 64 token 存储单元 → 现行“缓存前缀单元”的三种落盘时机），下半部分为 DeepInfer Q&A 描述的两段式查询（block KV 得 L1，SWA 决定 L1 或 L1−128）与 bounded replay（本文根据 DeepSeek 官方资料绘制：API 指南、2024-08-02 公告、DeepInfer 部署 Q&A）* — [原始来源](https://api-docs.deepseek.com/zh-cn/guides/kv_cache)

线上效果方面，Day 6 给出了 V3 / R1 的 24 小时统计（北京时间 2025-02-27 12:00 至 02-28 12:00，含网页、App 与 API 全部请求）：输入 6,080 亿 token，其中 3,420 亿（**56.3%**）命中磁盘 KV 缓存；峰值占用 278 个节点，平均 226.75 个（每节点 8 张 H800）；Prefill 阶段每节点平均输入吞吐约 73.7k tokens/s（含缓存命中） [DS9]。V4.1 的 Q&A 没有给出总命中率，只说明当前 3FS 实现已“非常接近可达到的最大命中率”：agent 流量 72 小时、chatbot 流量 7 天的 TTL 几乎覆盖了所有可复用条目，72 小时还能跨过周末，让下一个工作日的请求复用缓存。chatbot 流量的长尾更重，少量 KV 被反复读取很多次；经验上 TTL 缩到约 3 天会让 chatbot 损失 0.5%–1% 的命中率 [DS2]。Q&A 还提到单集群目前每天服务 155 万亿 token [DS2]。

## 本章资料

- **[DS1]** [DeepSeek 官方 GitHub · deepseek-ai/DeepInfer README（固定提交 a679bb86ce36e883a98a97733492b686e5b9a08e，2026-11-03；含 doc/images/architecture_overview.png，采集于 2026-11-18）](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e)
- **[DS2]** [DeepSeek 官方 GitHub · DeepInfer doc/inference_deployment_qa.md《DeepSeek V4.1 Flash Inference Deployment Q&A》（v0.2，2026-11-03；KV cache storage、3FS、负载均衡、SWA bounded replay，采集于 2026-11-18）](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)
- **[DS3]** [DeepSeek 官方 GitHub · DeepInfer 源码 csrc/engine/block_manager.hpp、deepinfer/worker/cache_engine.py、deepinfer/model_executor/model_metadata_utils.py、deepinfer/hugging_face/config.py（固定提交 a679bb86，采集于 2026-11-18）](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py)
- **[DS4]** [DeepSeek 官方 Hugging Face · deepseek-ai/DeepSeek-V4.1-Flash-1017 config.json（num_hidden_layers、compress_ratios、kv_source_layer_ids、index_head_dim、sliding_window，采集于 2026-11-18）](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json)
- **[DS5]** [DeepSeek 官方 GitHub · deepseek-ai/FlashMLA README（V4.1 KV cache 格式：528 B / 288 B per token，采集于 2026-11-18）](https://github.com/deepseek-ai/FlashMLA)
- **[DS6]** [DeepSeek API 文档 · 上下文硬盘缓存（缓存前缀单元、落盘时机、prompt_cache_hit_tokens / prompt_cache_miss_tokens，采集于 2026-11-18）](https://api-docs.deepseek.com/zh-cn/guides/kv_cache)
- **[DS7]** [DeepSeek API 文档 · 新闻《DeepSeek API 创新采用硬盘缓存，价格再降一个数量级》，2024-08-02](https://api-docs.deepseek.com/zh-cn/news/news0802)
- **[DS8]** [DeepSeek API 文档 · 模型 & 价格（deepseek-flash / deepseek-v4-pro 缓存命中与未命中价格，采集于 2026-11-18）](https://api-docs.deepseek.com/zh-cn/quick_start/pricing)
- **[DS9]** [DeepSeek 官方 GitHub · open-infra-index《Day 6 - One More Thing: DeepSeek-V3/R1 Inference System Overview》，2025-02 Open Source Week](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
- **[DS10]** [DeepSeek 官方 GitHub · open-infra-index《The Path to Open-Sourcing the DeepSeek Inference Engine》，2025-04](https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md)
- **[DS11]** [DeepSeek 官方 GitHub · deepseek-ai/3FS README（Performance · KVCache 读吞吐与 GC IOPS 图，采集于 2026-11-18）](https://github.com/deepseek-ai/3FS)
- **[DS12]** [DeepSeek 官方 GitHub · 3FS docs/design_notes.md（组件、CRAQ、chunk engine、USRBIO，采集于 2026-11-18）](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md)
- **[DS14]** [DeepSeek-AI · DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（arXiv:2405.04434，2024-05）](https://arxiv.org/abs/2405.04434)
- **[DS15]** [DeepSeek-AI · DeepSeek-V3 Technical Report（arXiv:2412.19437，2024-12；§3.4 推理与部署）](https://arxiv.org/abs/2412.19437)
- **[DS16]** [DeepSeek 官方 Hugging Face · deepseek-ai/DeepSeek-V3 config.json（kv_lora_rank、qk_rope_head_dim、num_hidden_layers 等，采集于 2026-11-18）](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)
- **[DS17]** [DeepInfer · LLMEngine：请求接入、C++ 调度、Worker 执行与结果处理（固定提交）](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py)
- **[DS18]** [DeepInfer · BlockManagerMutexAcquire 与 block/state 联合准入（固定提交）](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/block_manager.hpp#L190-L240)
- **[DS19]** [DeepInfer · CacheEngine：层所有权、SWA 与压缩状态的物理布局（固定提交）](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L149-L305)
