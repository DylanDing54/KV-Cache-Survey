# Seagate / SK hynix

> SSD / HDD 混合 KV 联合白皮书

以 SSD 承接活跃阶段，以 HDD 延长闲置 KV 的保留窗口。核心场景是“暂停—恢复—连续多轮”。 [S]

| 项目 | 公开配置 |
|---|---|
| 模型 / GPU | DeepSeek-R1-Distill-Llama-8B / H100 80GB PCIe |
| 存储 | 4 × 3.84TB SK hynix PS1010 SSD；12 × 30TB Seagate Exos HAMR HDD |
| 原始容量 | 15.36TB SSD + 360TB HDD，未扣冗余 |
| 主机内存 / 软件 | 2.0 TiB DRAM；Dynamo KVBM、NIXL、SPDK、LVM、Open CAS |
| 工具 / 主延迟负载 | LMBenchmark；ISL 21K、OSL 100、15 用户、QPS 1、10 turns |

### 测试如何构造冷热转换

1. **预置：**OCF pass-through，使第一轮 KV 绕过 SSD 写入 HDD。
2. **恢复：**切换 write-back，HDD KV 回填 SSD 并加载至 HBM。
3. **连续交互：**后续读取命中 SSD；新增 KV 先写 SSD，再后台写回。

白皮书报告首次恢复后的 SSD 命中率为 100%。这证明了受控路径，而非验证长期交错回访下的自动冷热预测算法。LMBenchmark 来自 LMCache 项目，但实际 KV 管理由 KVBM 执行。

| 结果 | 白皮书报告值 | 解释边界 |
|---|---|---|
| TTFT / 端到端延迟 | 改善 95% / 降低 80% | 对比关闭 prefix caching 与 offload、每轮重算；非对比全 SSD |
| 容量 / 成本 | 约 11× / 约低 75% | 作者对 SSD×4 的方案口径；不是云服务整体 TCO |
| 相对全 SSD 的 TTFT 差距 | 1 turn：7.6s；50 turns：0.35s | 后续 SSD 命中摊薄首次冷恢复成本；非 HDD 读取变为 0.35s |
| I/O 特征 | 128KB 请求；72% 随机写、83% 随机读 | 对应 QPS 0.1 的 I/O 实验，不能与所有延迟图混为同一配置 |

4 / 8 / 12 块 HDD 的扩展结果为实测；更多盘数属于估算。白皮书中的 S3 对照、预取 hint、cuComp 压缩列为 future work。正文笼统的微秒级传输延迟不能当成机械盘冷读延迟。

![Seagate / SK hynix 白皮书原 PDF 第 8 页：SSD / HDD 架构与测试拓扑](../figures/seagate-1.png)

*Seagate / SK hynix 白皮书原 PDF 第 8 页：SSD / HDD 架构与测试拓扑* — [原始来源](https://www.seagate.com/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/)

![白皮书原 PDF 第 13 页：容量、成本与多轮性能图，按原始口径阅读](../figures/seagate-2.png)

*白皮书原 PDF 第 13 页：容量、成本与多轮性能图，按原始口径阅读* — [原始来源](https://www.seagate.com/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/)

### 放置策略：哪些 KV 放到哪一层

这套方案的放置由两层软件决定。上层是 Dynamo KVBM，它管理 KV 块在 GPU HBM、CPU 内存和“磁盘”之间的分配、固定（pinning）、淘汰与复用，白皮书称这些元数据“独立于物理位置” [SWP]。下层是 Open CAS（OCF），负责 SSD 与 HDD 之间的放置。对 KVBM 和 NIXL 来说，NVMe-oF 目标只是一个 EXT4 挂载的文件系统，SSD 与 HDD 的分工发生在这个卷内部，按缓存行进行，不感知前缀或会话 [SWP][PLSG1]。因此，“哪些 KV 放 SSD、哪些放 HDD”并不是 KV 语义层面的决策，而是块缓存策略的结果。

| 维度 | 该方案做法 | 证据类型与引用 |
|---|---|---|
| 层级与介质 | H100 HBM 80GB → 主机 DRAM（GPU 服务器 2.0 TiB；KVBM CPU 层实际配置多大未写）→ NVMe-oF 混合卷：SSD 缓存 4 × 3.84TB（15.36TB），HDD 容量层 12 × 30TB（360TB，RAID 聚合） | 官方明确说明（白皮书第 7–9 页） [SWP] |
| 放置单位 | KVBM 以 KV block 为单位，白皮书未写块大小；存储侧每个 I/O 为 128KB，等于混合卷的 MDTS；OCF 按 cache line 映射（Open CAS 默认 4 KiB），白皮书未写实际设置 | 官方明确（第 11 页）；cache line 大小未公开 [SWP][PLSG2] |
| 准入 | 白皮书没有写长度或热度门槛。首轮的系统提示加问题、之后各轮新增的 KV（上一轮回答加新问题）都会写入，也就是包括 decode 输出的 KV。KVBM 当前文档默认开启“磁盘卸载过滤”：只有频率 ≥ 2 的块才从 CPU 层下沉到磁盘层（`DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER` 可关闭）。白皮书没有说明所用 KVBM 版本，也没有说明是否关闭了这项过滤 | 官方明确（第 9 页）；过滤为 KVBM 文档 v1.4.2 [SWP][PLSG3] |
| 写入时机 | KVBM 文档自述为 write-through 缓存，各层容量须逐层递增。OCF 处于 write-back 模式：数据先写入 SSD 即确认，SSD 需要空间时再转到 HDD，由后台 cleaning 刷回（ALRU 默认在最后一次写入 120 s 后、每 20 s 唤醒一次）。白皮书未写 cleaning 参数和副本数 | 官方明确（第 8、10 页）；参数为 Open CAS 文档默认值 [SWP][PLSG1][PLSG2][PLSG3] |
| 降级与淘汰 | 白皮书说法：容量满时淘汰“最少使用的块”；停顿足够久时，混合卷把不活跃的 KV 迁到 HDD。OCF 的淘汰算法是 LRU（按 cache line）。测试中“首轮 KV 在 HDD”是用 pass-through 模式人为预置的，不是自然老化的结果 | 官方明确（第 5、9、10 页）；LRU 为 OCF 文档 [SWP][PLSG1] |
| 提升与预取 | 命中 HDD 的 KV 读一次后被复制进 SSD，再加载到 HBM，之后的轮次全部命中 SSD。没有预取：Dynamo prefetch hint 被列为未来工作。白皮书未描述超时后回退重算 | 官方明确（第 9、13、15 页） [SWP] |
| 位置选择 | 采用共享 NVMe-oF target，多个 prefill / decode worker 经 NIXL 访问。KV 块位置通过 NATS broker 的发布 / 订阅主题告知其他节点，前端为 KV-aware router（Figure 7） | 官方明确（第 7、9 页） [SWP] |
| 决策者 | KVBM 管理块生命周期；OCF 决定 SSD / HDD；运维人员决定缓存模式切换与 SSD : HDD 容量配比 | 官方明确 [SWP] |
| 选择依据 | 主要是成本和保留时长：SSD 每 TB 成本约为 HDD 的 16 倍，DRAM 每字节约为 NVMe SSD 的 44 倍；目标是把 KV 保留数天到数周。4 块 SSD 并行，以匹配 PCIe 5.0 ×16 的带宽。没有公开逐块的成本模型 | 官方明确（第 5–6、8 页）；成本模型未公开 [SWP] |

**选择逻辑。**白皮书的依据是 I/O 特征加上“突发”假设：用户的会话会开始、暂停、再恢复。写入阶段 72% 是随机写，读回阶段 83% 是随机读 [SWP]，这类随机小块 I/O 由 SSD 承接；HDD 只承担“长时间没人访问、但值得保留”的 KV。每次冷恢复要付一次 HDD 读，之后的多轮对话都由 SSD 服务，所以这笔代价会随轮数摊薄：混合方案与全 SSD 的 TTFT 差距，1 轮时为 7.6 s，50 轮时降到 0.35 s [SWP]。这个取舍对会话长、回访集中的负载有利；如果大量会话只回访一次，每次都要付冷读代价。

**需要注意的边界。**write-back 模式下，最新的 KV 在被 cleaning 刷回之前只存在于 SSD 上，HDD 并不是实时的完整副本。KV 可以重算，所以这个风险可以接受，但应该写明。OCF 还提供 nhit 晋升过滤（只有访问达到 n 次的 core line 才进入缓存）和顺序流旁路（Sequential Cutoff），白皮书没有使用，也没有评估 [PLSG1][PLSG2]。正文中“新增 KV 先写 SSD，再后台写回”的准确含义是 OCF 的后台 cleaning，不是 KVBM 自身的动作。

**例子（本文按公开参数推算）。**DeepSeek-R1-Distill-Llama-8B 有 32 层、8 个 KV 头，head 维度 128，按 BF16 计算每 token 的 KV 约 128 KiB，一个 21K token 的提示约 2.6 GiB。流程如下：第 1 轮在 HBM 完成 prefill，KV 经 KVBM 卸载，在 pass-through 模式下直接写入 HDD；第 2 轮命中前缀，这约 2.6 GiB 从 HDD 读出，同时被插入 SSD，再送到 HBM；本轮约 100 个输出 token 和新问题产生的 KV 写入 SSD，成为 dirty 数据，之后由 cleaning 刷到 HDD；用户长时间离开后，SSD 按 LRU 回收这些缓存行，只剩 HDD 上的副本；用户再回来时，又要付一次 HDD 读。

![Seagate / SK hynix 白皮书原 PDF 第 10 页 Figure 9：首轮在 OCF pass-through 下绕过 SSD、写入 HDD，此后切换为 write-back，SSD 作为缓存，读回与新增 KV 都经过 SSD](../figures/seagate-3.png)

*Seagate / SK hynix 白皮书原 PDF 第 10 页 Figure 9：首轮在 OCF pass-through 下绕过 SSD、写入 HDD，此后切换为 write-back，SSD 作为缓存，读回与新增 KV 都经过 SSD* — [原始来源](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf)

### 硬件配置信息

| 场景 / 方案 | 计算（GPU / NPU 型号 × 数量，HBM 容量） | CPU 与主机内存（DRAM 容量） | 本地存储（SSD / NVMe / HDD 型号、数量、容量） | 网络（RDMA / NIC 型号与带宽） | 远端 / 共享存储（服务名与规格） | 来源 |
|---|---|---|---|---|---|---|
| GPU 服务器（NVMe-oF initiator，跑 DeepSeek-R1-Distill-Llama-8B） | 1 × NVIDIA H100 PCIe，80GB HBM2e | Intel Xeon 6767P；2.0 TiB DDR5 | 未公开 | 到 SSD 服务器 400Gb/s，NVMe-oF over RDMA，GDS；NIC 型号未公开 | NVMe-oF 混合目标（SSD-only 或 SSD + HDD），以 EXT4 挂载 | 白皮书第 7–8 页 [SWP] |
| SSD 服务器（active storage，SPDK + LVM + OCF 缓存） | — | Intel Xeon Gold 6548Y+；503 GiB DDR5 | 4 × SK hynix PS1010 3.84TB（PCIe Gen5，TLC），共 15.36TB | 上联 GPU 服务器 400Gb/s；下联 HDD 服务器 25Gb/s（Figure 6） | 对外提供 NVMe-oF 混合目标 | 白皮书第 7–8 页 [SWP] |
| HDD 服务器（durable storage） | — | Intel Xeon E5-2630 v2；32 GiB DDR3 | 12 × Seagate Exos 30TB（Mozaic HAMR，CMR），共 360TB，RAID 聚合（级别未公开） | 25Gb/s NVMe-oF 到 SSD 服务器（Figure 6） | 由 SPDK 以 NVMe-oF 导出给 SSD 服务器 | 白皮书第 7–8 页 [SWP] |
| HDD 扩展实验（不用 SSD，只测首轮） | 同上 | 同上 | 4 / 8 / 12 块 HDD 为实测，24 / 36 块为估算 | 正文写 36 块以上受网络上限“25GB/s”约束（与 Figure 6 的 25Gb/s 单位不一致，按原文列出） | 同上 | 白皮书第 14 页 [SWP] |
| 全 SSD 对照（SSD × 4） | 同 GPU 服务器 | 同上 | 4 × PS1010 3.84TB | 同上 | SSD-only NVMe-oF 目标 | 白皮书第 10–13 页 [SWP] |

KV 实际落在 H100 HBM、SSD 服务器上的 15.36TB SSD 缓存，以及 HDD 服务器上的 360TB HDD 容量层；GPU 服务器的 2.0 TiB DRAM 可作为 KVBM 的 CPU 层，但白皮书没有写这一层的配置值。HDD 在来源中明确出现，是 NVMe-oF 后端的 durable 层，不直接连接 GPU 服务器。

### 联合白皮书的研究问题与容量假设

SK hynix 的 Kyung Soo Lee、Jin Na Yang 与 Seagate 的 Thomas Prohofsky 研究的是用户带着原有上下文回到推理服务时，系统应当读回 KV，还是再次用 GPU 计算整段前缀。客服、预测维护和跨部门的多智能体工作流可能持续数周。每次请求会把此前的回答和新的问题一起带入下一轮，长期保留 KV 的成本因此取决于并发用户数、上下文长度和两次访问之间的间隔。下文按白皮书 SC135.1-2605US 的表 1 和图 1–22 给出配置、比较对象及结果。[Seagate 发布页](https://www.seagate.com/cn/zh/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/)

**同一张 GPU 能留下多久的历史？** 作者引用 NVIDIA 的案例，智能体工作负载产生的 token 最多可达传统聊天机器人的 15 倍。另据 Microsoft 的 72 GPU 机架达到 110 万 token/s 的案例，作者推算每张 GPU 每天可能产生约 92 TB 的 KV。若持续按这一速率写入，80 GB HBM 约保留 1.2 分钟，1 TB DRAM 约 16 分钟。模型也影响这一换算，白皮书举出的 Qwen2.5-7B 和 72B 每 token 分别需要 56 KB 和 320 KB KV。上述案例用于估算容量，后文的延迟实验用的是 DeepSeek-R1-Distill-Llama-8B。[原文第 3–5 页，图 1–3](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=3) · [第 11 页，模型示例](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=11)

| 原文图 2、16 的介质示例 | 容量 | 按上述生成速率折算的保留时间 |
|---|---|---|
| GPU HBM | 80 GB | 0.02 小时 |
| 主机 DDR5 | 1 TB | 0.27 小时 |
| SSD × 4 | 32 TB | 8.3 小时 |
| HDD × 12 或以它为容量层的混合卷 | 360 TB | 93.9 小时，约 3.9 天 |

图 16 的约 11 倍来自 93.9 与 8.3 小时之比。混合卷的 SSD 用作缓存，不能把缓存容量再次加进 HDD 的保留容量。图 17 对相同容量示例标出的单位价格为全 SSD 0.19 美元/GB、全 HDD 0.024 美元/GB、混合 0.04 美元/GB，作者概括为混合相对全 SSD 节省约 75%。**这两张图的全 SSD 均按 32 TB 作图，表 1 实测平台的四块 SSD 合计 15.36 TB。** 图示价格和时长用于方案比较，测试没有让会话实际闲置 93.9 小时再恢复。[原文第 12 页，图 16、17](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=12)

作者用 Omdia 的 2025 年价格说明把历史全部放在内存里的代价，HBM3E 12Hi 约 14 美元/GB，DDR5 约 8.50 美元/GB，按已部署字节计算 DRAM 约为 NVMe SSD 的 44 倍。Forward Insights 的 2016–2026 年序列则比较 SSD 与 HDD 的每 TB 价格，过去十年 SSD 平均约为 HDD 的 10 倍，2026 年约为 16 倍。两组价格资料、图 17 的方案示例和机房整体拥有成本各有各的口径。[原文第 4–5 页，图 3 及文献 4–6](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=5)

### 从请求路由到共享卷

**计算与存储如何协同。** 图 4–8 从一次前缀命中展开。前端 KV-aware router 用索引定位可复用的块。架构允许 prefill worker 把算好的块提交到共享卷，decode worker 根据块映射按需取回。KVBM 跟踪分配、固定、淘汰和复用，NATS 通过发布/订阅分发位置元数据。真正的 KV 载荷由 NIXL 经各节点所见的 EXT4 挂载目标传送，GDS 与 NVMe-oF/RDMA 使数据能直接进入 GPU 显存。这样，即使某个 worker 退出，其他节点仍可读取已经提交的块。单个 worker 的显存容量也不再决定整份历史的寿命。[原文第 6–9 页，图 4–8](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=7)

**目标卷如何构成。** HDD 服务器用 SPDK 导出磁盘，SSD 服务器把后端聚合成逻辑卷，OCF 在卷前缓存活跃数据，GPU 服务器作为 NVMe-oF 发起端。四块 PCIe Gen5、V-NAND TLC 的 PS1010 并行服务读写，设计目标是匹配 GPU 的 PCIe 5.0 ×16 接口。CMR 写入的 Mozaic HAMR Exos 保存冷块。SPDK 目标程序可在存储服务器、DPU 或云节点上运行。作者还提出为 LVM 共享卷组接入 Seagate In-Drive Mutex，并由 Kubernetes CSI 动态分配卷。这是可选部署方式，三机测试实际采用的介质、CPU 和网络规格见本节上表。[原文第 7–8 页，表 1、图 5、6](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=8)

**软件版本和协议边界。** 表 1 记载 GPU 与 SSD 节点均为 Ubuntu 22.04.5 LTS，内核版本按表原记为 6.8.0.-94-generic。HDD 节点为 Rocky Linux 9.6，内核记为 5.14.0-570.58.el9_6x86_64。GPU 节点使用 CUDA 12.8 和 NVIDIA 驱动 570.86.10。原文将 RDMA 网卡间的硬件传输与经过 TCP、TLS、HTTP、负载均衡器及对象服务的路径比较，前者典型端到端 p99 低于 100 µs，后者的握手和请求处理可能增加数千 µs。这是作者对协议路径的描述。实验中包含机械寻道、读回前缀与服务排队的首 token 时间另见下表。[原文第 7–8 页](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=8)

选用 GDS 的另一项依据来自外部研究。相对 POSIX 读写，作者引述的结果为带宽最高 2.8 倍、CPU 开销降低 3–5 倍。另一项研究报告服务器功耗约低 7 W。这些数字说明为何选择直接数据通路。本白皮书自己的 GPU 利用率和功耗实验使用单独的短输出负载。[原文第 6 页，文献 8、9](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=6)

### LMBenchmark 如何隔离重算、冷恢复与热命中

实验以合成的长输入、短输出多轮对话驱动 KVBM。重算组关闭前缀缓存和卸载，每轮重新 prefill。全 SSD 组同时开启两项功能，KV 可在 HBM 和四块 SSD 间复用。混合组增加 HDD 层，先用 pass-through 把初次提示写进 HDD，再切到 write-back 模拟用户恢复会话。第一次读回会填充 SSD，同一轮交互突发内后续恢复达到 100% SSD 命中。实验控制了初始冷热状态，没有测量真实用户两次回访的间隔分布。图 7 给出可分离 prefill/decode 的架构，图 14、15 的延迟测量实际采用未分离部署。[原文第 9–10 页，图 7–9](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=9) · [第 12 页，图 14、15](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=12)

下表按实验列出负载。ISL、OSL 分别为输入、输出 token 数。除单列说明外，输入为 21K、输出为 100。各图分别改变请求速率、用户数、轮数或输出长度。

| 指标与原图 | 实验条件 | 图中观测 |
|---|---|---|
| 卸载和读回的 I/O 构成（图 10、11） | 15 用户，QPS 0.1 | 卸载时约 72% 随机写、28% 顺序写，单次请求 128 KB，等于该混合卷的最大传输大小 MDTS。读回时约 83% 随机读、17% 顺序读。每轮还会按输出长度产生新增 KV 写入，占总 I/O 不足 1%。作者据此要求后端针对 128 KB 及以上随机访问优化。 |
| 逐轮 TTFT（图 12） | 15 用户，QPS 0.5，观察第 1–15 轮 | 重算约从 2 秒升至 40 秒，随后回落并停在约 20 秒。最初请求积压，GPU 饱和后排队延迟维持高位。客户端收到上一轮答复才发下一轮，实际到达速率下降后积压逐步消退。全 SSD 各轮为 0.16–0.19 秒。 |
| 增加用户数（图 13） | QPS 1，10 轮，依次取 10、20、30、40、50 用户 | 重算 TTFT 分别为 11、62、110、158、209 秒，作者概括为增至约 20 倍。全 SSD 在同五个点为 0.16、0.16、0.17、0.16、0.16 秒。 |
| 混合卷与每轮重算（图 14、15） | 15 用户，QPS 1，10 轮，未分离 prefill/decode | TTFT 从 35.24 秒降至 1.75 秒，约降 95%。图中端到端延迟从 753.33 秒降至 153.57 秒，约降 80%。两项百分比都以重算组为基准。 |
| 一次冷恢复如何被后续轮次摊薄（图 18） | 15 用户，QPS 1，比较 1–50 轮 | 与全 SSD 比较，作者报告平均 TTFT 差距由单轮 7.6 秒缩至 50 轮 0.35 秒。一次 HDD 读回之后，后续多轮由 SSD 服务。 |
| 读带宽随运行时间变化（图 19） | 15 用户，QPS 4 | 运行到 100、300、600、1,000 秒时，全 SSD 分别为 25、25、26、26 GB/s，混合卷分别为 3、17、20、22 GB/s。同期差距从 88% 缩至 15%。 |

[原文第 11 页，图 10–13](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=11) · [第 12 页，图 14–17](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=12) · [第 13 页，图 18、19](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=13)

### 冷恢复的硬盘扩展曲线与 GPU 功耗

为单独观察 HDD 带宽，作者让 I/O 绕过 SSD，用 15 个用户、QPS 1、ISL 21K、OSL 100、运行 50 秒的配置观察单轮恢复。多盘组成一个逻辑卷，将同一批 KV 的读取分布到多块盘。图 20 给出首 token 时间，图 21 同时画出峰值和平均读带宽。[原文第 14 页，图 20、21](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=14)

| HDD 数量 | 数据性质 | 单轮 TTFT（秒） | 峰值 / 平均读带宽（GB/s） |
|---|---|---|---|
| 4 | 实测 | 67.89 | 0.70 / 0.37 |
| 8 | 实测 | 34.43 | 1.10 / 0.75 |
| 12 | 实测 | 19.88 | 1.50 / 1.26 |
| 24 | 作者外推 | 10.08 | 2.55 / 2.55 |
| 36 及以上 | 网络受限时的预测平台 | 8.24 | 3.13 / 3.13 |

从 4 增至 12 块时，峰值与平均带宽的差距由接近一倍缩至约 20%。作者预计从 12 到 24 块可再缩短约一半 TTFT，24 块可满足该模型的吞吐需求，继续加盘最多比 12 块缩短约 60%。原文图 6 的 HDD 链路标为 25 Gb/s，图 20、21 却标为 25 GB/s。图 21 的平台值 3.13 GB/s 与 25 Gb/s 的单位换算相符。盘数选型应以实测的三行及实际链路规格为依据。[原文第 8 页，图 6](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=8) · [第 14 页，图 20、21](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=14)

**只看 prefill 的独立实验。** 图 22 将输出缩短到 2 个 token，保留 21K 输入、15 用户和 QPS 1。重算组的 GPU 利用率为 99.7%，混合卷复用组为 23.8%，相对降幅 76.1%。作者另报告该阶段测得功耗比重算低 52%。空出的计算时间可以用来生成新前缀或服务其他请求。原文没有给出绝对瓦数、每请求能耗或整个集群的节电比例。[原文第 14 页，图 22](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=14)

### 白皮书结论与待验证的部署选择

作者认为，高容量共享卷能让历史前缀跨 worker 和跨会话保留，SSD 承接连续交互，HDD 承接较长的空闲期。扩充 SSD 可以让更大的活跃工作集保持低延迟。扩大 HDD 主要延长保留窗口，多盘并行还可缩短首次恢复。文末提出保留数天、数周乃至数月的部署目标。实际收益取决于 KV 生成速率、可用容量、回访频率，以及首次回读能否比重新 prefill 更快。原文把以下三项列作后续研究。[原文第 12、15 页](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf#page=15)

| 待验证方案 | 作者提出的机制与比较指标 |
|---|---|
| Dynamo 预取提示 | 订阅与后端存储对应的 NATS 主题，在调度、分派 prefill worker 的同时触发 HDD 读回，比较首次恢复等待时间。 |
| 本地或远端 S3 与 NVMe-oF | 比较近端、远端及高性能对象存储与 RDMA 块目标，观察简化部署和扩容所带来的收益，以及对象协议和网络路径增加的延迟。 |
| GPU 侧 cuComp 压缩 | 在 KV 离开 GPU 前压缩，读回时解压，评估传输字节数、占用空间、token 吞吐及 TTFT 之间的取舍。 |

#### 白皮书原图索引

[图 1 多智能体上下文增长](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-1.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 2 各层保留时间](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-2.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 3 介质价格比](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-3.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 4 卸载与前缀命中](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-4.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 5 KV管理层级](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-5.png/_jcr_content/renditions/16-10-medium-640x400.png)

[图 6 三机连线](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-6.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 7 Dynamo与存储栈](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-7.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 8 一次回访的读写过程](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-8.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 9 测试阶段切换](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-9.png/_jcr_content/renditions/16-10-medium-640x400.png)

[图 10 卸载期I/O](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-10.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 11 读回期I/O](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-11.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 12 逐轮TTFT](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-12.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 13 并发用户数](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-13.png/_jcr_content/renditions/16-10-medium-640x400.png)

[图 14 首token延迟](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-14.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 15 端到端延迟](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-15.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 16 保留窗口](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-16.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 17 每GB成本](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-17.png/_jcr_content/renditions/16-10-medium-640x400.png)

[图 18 多轮平均TTFT](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-18.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 19 随时间变化的带宽](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-19.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 20 首轮TTFT与盘数](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-20.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 21 峰值和平均带宽](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-21.png/_jcr_content/renditions/16-10-medium-640x400.png) · [图 22 prefill利用率](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/figure-22.png/_jcr_content/renditions/16-10-medium-640x400.png)

## 本章资料

- **[PLSG1]** [Open CAS 官方文档 · Cache configuration / Cache operations / Promotion policies（cache 模式、LRU 淘汰、nhit 晋升，open-cas.github.io 提交 bb4455a，采集于 2026-11-19）](https://github.com/Open-CAS/open-cas.github.io/blob/bb4455adecc144fa9585e2d42b9d36379bafda61/content/concepts/cache-operations.md)
- **[PLSG2]** [Open CAS Linux 官方文档 · casadm Tool Reference（cache line 大小、cleaning-alru、promotion、seq-cutoff 默认值，提交 bb4455a，采集于 2026-11-19）](https://github.com/Open-CAS/open-cas.github.io/blob/bb4455adecc144fa9585e2d42b9d36379bafda61/content/open-cas-linux/tool-reference.md)
- **[PLSG3]** [NVIDIA Dynamo 官方文档 · KVBM Guide（v1.4.2：write-through、DYN_KVBM_* 容量参数、SSD 寿命保护的磁盘卸载过滤，采集于 2026-11-19）](https://docs.nvidia.com/dynamo/v1.4.2/knowledge-base/modular-components/kvbm/kvbm-guide)
- **[S]** [Seagate × SK hynix · Enabling inference at massive scale with hybrid storage for KV cache offloading](https://www.seagate.com/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/)
- **[SWP]** [Seagate / SK hynix 原始白皮书 · 15 页 · SC135.1-2605US](https://www.seagate.com/content/dam/seagate/assets/resources/enabling-inference-at-massive-scale-with-hybrid-storage-for-kv-cache-offloading/files/kv-caching-white-paper.pdf)
