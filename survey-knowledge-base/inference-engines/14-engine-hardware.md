# 推理引擎与硬件协同：计算、互连、网络与存储

> 跨引擎分析 · 官方部署、设备文档与容量推算

同一条长提示词，GPU 命中、本机 DRAM 命中、远端 GPU 命中和 HDD 命中都可以在服务指标上记为“缓存命中”，但四者交付第一个 token 所需的工作完全不同。定位硬件瓶颈，要知道命中了多少字节、这些字节由谁搬、经过哪几条共享链路，以及 GPU 何时真正可以使用它们。

### 沿一条请求定位硬件

| 请求阶段 | 引擎对硬件发出的工作 | 暴露出来的瓶颈 |
|---|---|---|
| 协议、分词与准入 | CPU 执行协议转换、分词、路由与组批，主机内存保存队列、哈希与元数据；调度器向设备提交工作。多模态请求还可能需要独立编码器。 | CPU 核饱和、跨 NUMA 内存访问、批次准备过慢都会让设备在两个批次之间等待。先拆分排队、分词、元数据匹配与执行时间。 [VL19][SGI30][DS2] |
| Prefill | 模型权重和 KV 主体驻留设备内存，GPU/NPU 内核计算提示词并写入新的 KV。长提示词提供计算并行度；分块使引擎可在不同请求之间安排工作。 | 短批、长上下文、量化方式和算子实现会改变瓶颈。对照设备时间线中的矩阵运算、注意力、HBM 流量与空闲间隔，仅看利用率均值无法区分。 [VL19][DS2] |
| 并行计算与 Decode | TP 在同层交换部分结果；PP 在相邻阶段传激活；MoE 的 EP 按 token 路由交换激活并汇合；DP 将请求分到不同副本。Decode 逐步生成，单步反复读取权重和历史 KV。 | 小批量时带宽与同步常占较大比例。TP 集合通信、EP 热点、PP 流水气泡及副本间负载差异，都会改变每输出 token 的时间（TBT）。 [VL21][DS2] |
| 冷前缀回填 | 控制面查询命中区间并预留目标槽位；数据面将命中字节从主机、对端设备或存储恢复。CPU 页锁定或内存注册有助于支持相应 DMA 通路，实际要求由驱动和后端决定。 | 首 token 要等到所需层的 KV 可用。记录存储队列、网络队列、主机缓冲和 H2D 等待，另计被其他计算掩盖的部分。 [SGI33][VL22][DS2][HW1] |

![计算节点互连与存储支路的物理拓扑示例。上半图是支持直接设备访问时的一条 P/D 跨机路径；下半图是带主机缓冲的存储路径。部署可选择不同交换网络、协议和后端。本文依据 GPUDirect、SuperPOD 参考架构及引擎文档绘制。](../figures/engine-hardware-1.svg)

*计算节点互连与存储支路的物理拓扑示例。上半图是支持直接设备访问时的一条 P/D 跨机路径；下半图是带主机缓冲的存储路径。部署可选择不同交换网络、协议和后端。本文依据 GPUDirect、SuperPOD 参考架构及引擎文档绘制。* — [原始来源](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)

### 同机和跨机：一条“远端 KV 命中”如何落到硬件

**同机。**CPU 调度线程、主机缓冲、GPU 和网卡可能属于不同的 NUMA 节点或 PCIe 根复合体。GPU 对 GPU 可用 NVLink/NVSwitch 或 PCIe P2P；GPU 与 NIC 的直通能力取决于驱动、内存类型和拓扑。把通信线程绑在离 NIC 很近的 CPU 上，若缓冲区仍分配在另一个 NUMA 节点，主机暂存仍可能多走一次跨插槽互连。验收时应同时记录 CPU 亲和性、内存位置、GPU–NIC 拓扑和每个方向的实测带宽。 [HW1][VL21]

**跨机。**以 vLLM 常规 NIXL 拉取为例，D 端先取得 P 端传输元数据，再向传输库提交所需块。支持 GPUDirect RDMA 时，网卡可访问已注册的设备内存，通过交换网络将有效载荷送入目标缓冲区；软件仍需处理握手、内存注册、传输完成和块的生命周期。若设备内存无法直接访问，后端可能改用主机暂存或其他传输，也可能不支持该组合。一次握手成功只能证明控制面可达，实际通路要从传输日志、端口字节数、主机内存流量与完成事件联合确认。 [VL22][HW1]

**交换网络。**InfiniBand 与基于以太网的 RoCE 都可以承载 RDMA。RoCEv2 可跨三层路由；ECN 可标记拥塞，采用 PFC 的部署还须使路径上的端点及交换机按一致的优先级配置。交换机转发流量；前缀命中由引擎或缓存服务判定。多个 P 节点同时向一个 D 节点发流时，接收端口和上联队列可能先拥塞。不同网卡若共用同一条上联或同一 PCIe 瓶颈，增加端口数也很难线性扩容。NVIDIA B200 SuperPOD 的公开参考设计给出了 rail 对齐与计算、存储网络的具体实例；三个引擎的实际装机拓扑仍需单独核实。 [HW2][HW3]

### 存储：同一个文件路径背后可能有三种设备链路

| 数据路径 | 设备与软件配合 | 需要区分的完成事件 |
|---|---|---|
| 本机 CPU 缓存 → GPU | 主机 DRAM 的 KV 通过 PCIe 或平台支持的主机–设备互连回填。批量、页布局和层级预取影响传输粒度；并发写回会占用反方向带宽和主机内存带宽。 | CPU 查到对象、拷贝提交、设备端可读分别打点。 [SGI33][DS2] |
| 本地或远端 SSD → 计算节点 | 本地 NVMe 涉及文件系统、块队列、控制器和闪存；远端 3FS 还涉及客户端、网络和存储服务。GPUDirect Storage 在受支持的栈上可避免 CPU 暂存，普通文件 API 或启用直接 I/O 本身无法证明用了 GDS。 | 对照页缓存命中、设备读完成、网络到达与 GPU 可用时间；确认页缓存是否把“冷读”变成了内存测试。 [VL9][DS12][HW4] |
| HDD 冷层 → SSD/DRAM → GPU | HBA/阵列控制器、盘队列、寻道与旋转延迟限制作业并发。较大顺序段能摊薄定位开销，但会产生读放大；把热点留在 SSD/DRAM 可以保护前台，代价是后台迁移与回收。 | 分别记录引擎请求的有效字节和阵列实际读取字节，区分热命中、全冷读及重建期间的尾延迟。持久化承诺还取决于同步与掉电语义。 [VL9][VL18][SGF] |

### 把带宽换算成请求预算

下面是用于规模估算的算例。假设一次远端命中必须搬运 **1 GiB**，单方向有效吞吐为 **25 GB/s**，则仅传输的下界为 1,073,741,824 ÷ 25,000,000,000 ≈ **43 ms**。400 Gb/s 端口的标称线速是 50 GB/s；25 GB/s 是本算例假设的应用有效吞吐。若每秒到来 30 个这样的命中，单方向有效载荷约 **32.2 GB/s**，已超过假设链路能力，持续运行就会积压。估算多 rank 负载时应先明确 1 GiB 是所有分片总量还是每个 rank 的流量，并另计复制、读放大及其他并发业务。

同样取 1 GiB，若磁盘以 64 KiB 为一个随机读单元，一次恢复有 **16,384 次 I/O**。假设阵列对这类随机读可持续提供 2,000 IOPS，则仅按 IOPS 计算就至少需要 **8.2 s**。这个换算只适用于给定的请求粒度和有效 IOPS；若相邻页已合并为连续段，应改按实测顺序带宽计算。对流水读回，稳定吞吐由最慢环节决定，首块还要付出启动与排队时间。单次命中有利于 TTFT 的条件是：元数据查询、排队与数据恢复中暴露在关键路径上的时间之和，小于跳过该前缀所节省的 prefill 时间。

### 如何验证引擎确实用到了预期硬件

| 对照试验 | 控制条件与观测量 | 如何解释结果 |
|---|---|---|
| 单实例计算基线 | 固定模型、精度、输入输出长度及并发，区分空缓存与 HBM 命中。记录分词、调度、prefill、逐 token 执行时间，以及设备内存带宽和空闲间隔。 | 基线给出每种前缀长度能省下多少计算，也能定位 CPU 供给不足或 GPU 内存瓶颈。 |
| 改变一条硬件路径 | 同机与跨机 P/D、RDMA 与可用的 TCP 对照、本机与远 NUMA 缓冲分别运行；固定块大小、rank 数和副本路由。记录每条流的方向、有效字节、握手耗时、完成延迟及重试。 | vLLM 的 NCCL 日志可核对模型并行链路；NIXL 需另看 UCX。SGLang 还需核验所选传输后端。DeepInfer 应分别测 3FS 取数和共享内存至 HBM 两段。 [VL21][VL22][SGI32][DS2] |
| 交换机与阵列同时承压 | 叠加多对一 KV 拉取、专家通信和异步写回，观察 NIC 利用率、交换端口队列/丢弃、ECN 标记或 PFC 暂停（如启用）、盘队列与重建流量。对齐引擎时间戳，分别报告命中及未命中的 P50/P99 TTFT 和 TBT。 | 如果空载带宽达标而混合负载 P99 恶化，先找最早积压的环节，再评估分网、限速、调整亲和性、热数据提升或冷读超时。 |
| 故障与降级 | 在测试环境模拟慢盘、缓存对象缺失、网卡路径不可用或单个 rank 退出；记录链路切换、回退重算、输出一致性和队列恢复时间。 | 只有重新分配的 rank、传输完成和缓存索引状态一致时才允许复用。对未公开的部署，报告可测代理指标并注明无法验证的内部步骤。 |

## 本章资料

- **[DS2]** [DeepSeek 官方 GitHub · DeepInfer doc/inference_deployment_qa.md《DeepSeek V4.1 Flash Inference Deployment Q&A》（v0.2，2026-11-03；KV cache storage、3FS、负载均衡、SWA bounded replay，采集于 2026-11-18）](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md)
- **[DS12]** [DeepSeek 官方 GitHub · 3FS docs/design_notes.md（组件、CRAQ、chunk engine、USRBIO，采集于 2026-11-18）](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md)
- **[HW1]** [NVIDIA · GPUDirect RDMA：内存注册、PCIe 层次及平台限制](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
- **[HW2]** [NVIDIA · RDMA over Converged Ethernet：RoCEv2 与网络配置](https://docs.nvidia.com/networking/display/mlnxenv24100700/rdma+over+converged+ethernet+(roce))
- **[HW3]** [NVIDIA DGX SuperPOD B200 参考架构 · Network Fabrics：计算与存储网络、rail 与交换拓扑](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-b200/latest/network-fabrics.html)
- **[HW4]** [NVIDIA · GPUDirect Storage Overview：存储与 GPU 的直接 DMA 路径](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- **[SGF]** [HiCacheFile / HiCacheStorage 原始源码](https://github.com/sgl-project/sglang/blob/c712459e7f8233302fdf0bdd5d4ddcd22aa4b2ab/python/sglang/srt/mem_cache/hicache_storage.py)
- **[SGI30]** [SGLang · 请求处理组件：TokenizerManager、Scheduler、DetokenizerManager（固定提交）](https://github.com/sgl-project/sglang/tree/c712459e7f8233302fdf0bdd5d4ddcd22aa4b2ab/python/sglang/srt/managers)
- **[SGI32]** [SGLang · P/D disaggregation：传输后端、设备选择与部署拓扑（固定提交）](https://github.com/sgl-project/sglang/blob/c712459e7f8233302fdf0bdd5d4ddcd22aa4b2ab/docs/docs/advanced_features/pd_disaggregation.mdx)
- **[SGI33]** [SGLang · HiCache：各层可见性、数据布局、I/O 内核与多 rank 同步（固定提交）](https://github.com/sgl-project/sglang/blob/c712459e7f8233302fdf0bdd5d4ddcd22aa4b2ab/docs/docs/advanced_features/hicache_design.mdx)
- **[VL9]** [vLLM 固定提交源码 · vllm/v1/kv_offload/tiering/fs/manager.py 与 io.py（FileSystemTierManager、batch_store_block / batch_load_block、probe_o_direct）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/fs/manager.py)
- **[VL18]** [WD × Solidigm · KV Cache Offload Across Storage Tiers，September 2026 / Revision 02](https://documents.westerndigital.com/content/dam/doc-library/en_us/assets/public/western-digital/collateral/tech-brief/technical-brief-kv-cache-offload-across-storage-tiers-wd.pdf)
- **[VL19]** [vLLM V1 · Architecture Overview：在线服务进程、并行拓扑与调度预算（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/design/arch_overview.md)
- **[VL21]** [vLLM · 并行扩展、多节点通信与 GPUDirect RDMA 验证（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/serving/parallelism_scaling.md)
- **[VL22]** [vLLM · NixlConnector：侧通道、UCX、设备选择及 P/D 读取方向（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/features/nixl_connector_usage.md)
