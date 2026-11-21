# vLLM：与具体硬件的对应

> 推理引擎 · 硬件对应 · vLLM 官方安装文档、硬件插件与容量推算

本章把本组第 1 章所述的 vLLM KV 机制（GPU BlockPool、OffloadingConnector 的 CPU 主层与 fs / obj / p2p 二级层、LMCache、NIXL P/D）逐项对应到加速器、主机内存、网卡、SSD 与 HDD 的型号和规格，不重复软件机制本身，通路解释见 2.5 章。依据的版本：vLLM 主仓库固定提交 `0c80c3a6`（2026-11-18，与本组前三章一致，晚于 2026-11-16 发布的 v0.34.0）；硬件插件按 2026-11-20 的主分支提交固定。文中“本文估算 / 本文分析 / 理论下界”均为推算，公式与假设随文给出；采集于 2026-11-20。 [VL1][VLH25]

### 官方硬件矩阵：主仓库后端与树外插件

vLLM 的安装文档把支持的硬件分成两类：主仓库内置的 GPU（NVIDIA CUDA、AMD ROCm、Intel XPU）与 CPU（x86、Arm、Apple silicon、IBM Z）后端；以及按 Hardware-Pluggable RFC 在主仓库**之外**维护的硬件插件，完整清单链接到 vllm.ai 首页的 Universal Compatibility。首页列出的硬件包括 NVIDIA、AMD、华为昇腾、AWS Neuron、Google TPU、IBM Spyre、Intel Gaudi / XPU、CPU、Apple Silicon、百度昆仑与寒武纪 MLU。 [VLH1][VLH6]

| 后端（固定版本） | 官方列出的硬件 | 驱动 / 软件栈要求 | 成熟度与 KV 相关能力 |
|---|---|---|---|
| CUDA（主仓库 0c80c3a6） | 计算能力 ≥ 7.5，例：T4、A100、L4、H100、B200 | 预编译二进制基于 CUDA 12.9，另提供 12.8 与 13.2 版轮子；Blackwell（B200、GB200）至少需 CUDA 12.8 | 主平台。`OffloadingConnector` 支持；NIXL 在 NVIDIA 上可直接 `pip install nixl` [VLH2][VL3][VL22] |
| ROCm（主仓库） | MI200（gfx90a）、MI300（gfx942）、MI350（gfx950）及部分 Radeon / Ryzen AI | ROCm ≥ 6.3，MI350 需 ≥ 7.0；默认预编译轮子为 ROCm 10.0，仅 Python 3.12 | 主仓库内置。`OffloadingConnector` 支持；NIXL 与 UCX 需通过 ROCm Dockerfile 从源码构建 [VLH3][VL3][VL22] |
| XPU（主仓库） | Intel Data Center GPU、Intel Arc GPU | 依赖 `vllm-xpu-kernels`，Python 3.12；建议 compute runtime 26.18；分布式后端为 xccl（torch ≥ 2.8） | 文档称“初步支持基本推理与服务”，PP 为 beta。`OffloadingConnector` 支持；NIXL 多轮 KV 回传当前只支持 CUDA 设备缓冲，XPU 主机缓冲“计划中” [VLH4][VL3][VL22] |
| CPU（主仓库） | x86（推荐 avx512f，avx2 功能受限）、Arm（需 NEON）、Apple silicon、s390x | FP32 / FP16 / BF16；gcc ≥ 12.3 推荐 | 文档称“基本推理与服务”；不在 `OffloadingConnector` 支持平台之列 [VLH5][VL3] |
| vllm-ascend（提交 8f603de；最新稳定 v0.30.0） | Atlas 800I A2、Atlas 800I A3、Atlas 300I Duo、950PR / 950DT | CANN 9.2.1、PyTorch 2.10.0、torch-npu 2.10.0.post7、Python 3.12 | 社区维护插件，主分支对 vLLM main 与 v0.33.0 做 CI。CPU 卸载复用上游 `OffloadingConnector`；`AscendStoreConnector` 接 Mooncake / Memcache / Yuanrong 做跨节点 KV 池（A2 / A3 / 950） [VLH7][VLH8] |
| TPU：tpu-inference（提交 80cfbb7） | 推荐 v7x（Ironwood）、v5e、v6e；实验 v3、v4、v5p | JAX 路径（正迁往 vllm-jax）；PyTorch 原生 TPU 已拆到 vllm-torchtpu | 官方支持矩阵中 Prefix Caching、KV Cache Offload、基于 DCN 的 P/D 分离为 ✅；multi-host 为“未测试” [VLH9] |
| Neuron：vllm-neuron（0.29.0.1.2.0） | Trn2、Trn3 | vLLM 0.29.0、Neuron SDK 2.33 | Beta。Prefix caching 与 1P1D / xPyD 分离 ✅，**KV cache offloading ❌**；旧版 0.5.3（覆盖 Trn1 / Inf2）为维护态 [VLH10] |
| Gaudi：vllm-gaudi（v0.30.0） | Gaudi 2、Gaudi 3 | Intel Gaudi 软件 1.24.2、PyTorch 2.13 | 官方兼容矩阵最新验证组合为 vLLM v0.30.0；APC 默认开启；提供带 NIXL 的 P/D 构建方式 [VLH11] |

这张表有两点直接影响 KV 分层设计。第一，原生 `OffloadingConnector`（以及其上的 fs / obj / p2p 二级层）只支持 CUDA、ROCm、XPU；Neuron 插件明确不支持 KV 卸载，昇腾与 TPU 则由插件自行实现。第二，插件版本落后主仓库一到数个小版本（Neuron 对齐 0.29.0，Gaudi 与昇腾稳定版对齐 0.30.0），本组第 2 章引用的 fs 层行为不能直接假定在插件上可用。 [VL3][VLH10][VLH11][VLH7]

### KV 层级与数据通路对应到具体硬件

以 8 卡 HGX 节点为基准。`gpu_memory_utilization` 在固定提交中的语义是“每实例可用的设备内存比例”：未指定时，CUDA / ROCm 上默认的可扩展 KV cache 取 1.0 并按实测余量定容，其他情况取 0.92。KV 池大小由 profile 后的余额决定，见本组第 1 章。 [VL14][VL23]

| 层级 / 通路 | 引擎中的组件或参数 | 典型硬件（型号） | 容量 | 带宽 / 延迟 | 来源 |
|---|---|---|---|---|---|
| HBM 设备池 | `BlockPool`、`gpu_memory_utilization`、`--kv-cache-dtype` | H100 SXM / H200 SXM / B200（DGX B200）/ MI300X | 80 / 141 / 180（1,440 GB ÷ 8）/ 192 GB | 3.35 / 4.8 / 8（64 TB/s ÷ 8）/ 5.3 TB/s | [DSH5][VLH15][DSH3][VLH16] |
| 机内 GPU 间（TP / 同机 NIXL） | NCCL；NIXL UCX `cuda_ipc` | NVLink 4（H100 / H200）；Infinity Fabric（MI300X） | — | 900 GB/s；8 条链路、每条峰值 128 GB/s | [DSH5][VLH15][VLH16][VL22] |
| 主机 DRAM 主层（GPU↔CPU DMA） | `OffloadingConnector` 的 `cpu_bytes_to_use` / `--kv-offloading-size`；可选 1 GB 大页与 `--numa-bind` | p5.48xlarge 2,048 GiB；ND96isr_H100_v5 1,900 GiB；DGX B200 2 TB（可配 4 TB） | 上述整机内存，扣除系统与权重加载占用 | PCIe Gen5 x16 标称 128 GB/s（双向）；vLLM 在 H100 上实测单向约 50 GB/s、DMA 双向 83.4 GB/s | [PLAWS3][PLMS6][DSH3][DSH5][VL5][VL3] |
| 本地 NVMe（fs 二级层） | `secondary_tiers: [{"type":"fs"}]`；`SimpleCPUOffloadConnector` disk 模式 | p5 系列 8 × 3,800 GB；DGX B200 8 × 3.84 TB U.2；单盘参考 Solidigm D7-PS1010 | 约 30 TB / 节点 | p5.48xlarge 实例存储标称 4,400,000 / 2,200,000 IOPS（读 / 写）；PS1010 顺序读 14,500 MB/s、4K 随机读 3.3M IOPS | [PLAWS3][DSH3][VLH17][VL9] |
| 跨节点 RDMA（NIXL P/D、p2p 层、Mooncake） | `NixlConnector`（UCX / LIBFABRIC 等）；`type: "p2p"` | Azure：每卡 1 × 400 Gb/s ConnectX-7 InfiniBand；AWS p5：3,200 Gb/s EFA；DGX B200：8 × ConnectX-7 400 Gb/s | — | 每节点合计 3.2 Tb/s ≈ 400 GB/s；单端口 50 GB/s | [PLMS6][PLAWS3][DSH3][VL22][VL3] |
| 共享文件 / 对象存储（fs 远端挂载、obj 层、LMCache L2） | `type: "fs"`（`locality: REMOTE`）；`type: "obj"`（NIXL OBJ，S3 兼容） | llm-d 文档列举的 CephFS、GCP Lustre、IBM Storage Scale、AWS EFS | 与部署规模无关 | 取决于后端，**未能核实**统一规格 | [PLGG1][VL3] |
| HDD 冷层 | fs 层 `root_dir` 指向 HDD 挂载点（见本组第 2 章） | Seagate Exos 30 TB（ST30000NM004K）；Exos E 4U106 机箱 | 30 TB / 盘；106 盘位 | 持续传输 275 MB/s；4K QD16 随机读 / 写 170 / 350 IOPS；平均延迟 4.16 ms | [BH11][KT11] |

需要强调的是，fs / obj / p2p 二级层都“只与主机内存交换数据，不接触加速器内存”，官方博客把这称为经由主机的单一数据通路。因此 NVMe、HDD 与 p2p RDMA 的每一字节在进入 HBM 前还要再走一次 PCIe，上表中 PCIe 一行是所有二级层共同的末段。NIXL P/D 则是 GPU 到 GPU 的直接读取，不经过主机主层。 [VLH18][VL22]

### 代表模型的量化算例

**每 token KV 字节数。**GQA 模型：`2 × num_hidden_layers × num_key_value_heads × head_dim × 字节数`。Llama-3.1-70B-Instruct 的 config 为 80 层、8 个 KV head、hidden 8192 / 64 头 = head_dim 128，BF16 时为 2 × 80 × 8 × 128 × 2 = **327,680 B（320 KiB）/ token**，与 WD / Solidigm 报告中 70B 模型的 320 KiB 一致。MLA 模型：每层缓存一条 `kv_lora_rank + qk_rope_head_dim` 的潜向量。DeepSeek-V3 为 61 层、512 + 64，BF16 时为 61 × 576 × 2 = **70,272 B / token**。vLLM 的 `fp8_ds_mla` 格式为每 token 每层 656 B（512 B FP8 NoPE + 16 B 缩放因子 + 128 B BF16 RoPE），即 61 × 656 = 40,016 B / token。 [VLH14][VL18][DS16][VLH13]

**HBM 去掉权重后能放多少 token（本文估算）。**公式：`每卡 KV 字节 ≈ 0.92 × HBM − 权重 ÷ 卡数`，未扣激活与 CUDA Graph，所以是上界。Llama-70B 的 BF16 权重按 safetensors 索引的 `total_size` 为 141.1 GB，TP=8 时每卡 17.6 GB；每卡只存 1/8 个 KV head，因此每卡每 token 40,960 B。DeepSeek-V3 FP8 权重按 671B 参数 × 1 B 估约 671 GB（不含缩放因子与 MTP 层）；MLA 的 KV 在 TP 各 rank 间复制，不随卡数切分。 [VLH14][VLH19]

| 8 卡节点 | Llama-70B BF16，TP=8：每卡 KV / 节点 token 数 / 32k 前缀数 | DeepSeek-V3 FP8 权重，BF16 MLA KV：每 rank KV / 每 rank token 数 |
|---|---|---|
| 8 × H100 80 GB | 56.0 GB / 约 137 万 / 约 41 个 | 权重 671 GB 超过 0.92 × 640 GB，单节点放不下 |
| 8 × H200 141 GB | 112.1 GB / 约 274 万 / 约 83 个 | 45.8 GB / 约 65 万（`fp8_ds_mla` 时约 114 万） |
| 8 × B200 180 GB | 148.0 GB / 约 361 万 / 约 110 个 | 81.7 GB / 约 116 万 |
| 8 × MI300X 192 GB | 159.0 GB / 约 388 万 / 约 118 个 | 92.8 GB / 约 132 万 |

作为对照，vLLM 官方博客称 DeepSeek-V3 以 TP 方式部署时每张 H200 只剩 34 GB 空闲显存，这是 Wide-EP 与 DP 注意力的动机。实测值低于上表估算，原因是有激活、通信缓冲与图内存要扣除。 [VLH19]

**一个 32k 前缀的大小**：Llama-70B 为 327,680 × 32,768 = 10.74 GB（10.0 GiB）；DeepSeek-V3 BF16 为 2.30 GB（2.14 GiB）。**取回时间的理论下界**：`t = 字节数 ÷ 链路有效带宽`，不计元数据、排队与首块延迟；8 卡并行时假设每卡有独立的 PCIe x16 与 NIC，且主机内存带宽不构成瓶颈。

| 来源层级（假设带宽） | Llama-70B 32k（10.74 GB） | DeepSeek-V3 32k（2.30 GB） |
|---|---|---|
| HBM 内读取（H100 每卡 3.35 TB/s，各卡读自己的分片） | 约 0.4 ms | 约 0.7 ms（每 rank 读完整副本） |
| 主机 DRAM → HBM（每卡 PCIe 实测约 50 GB/s，8 卡并行） | 约 27 ms | 约 46 ms（每 rank 都要一份） |
| 本地 NVMe（8 × PS1010 合计 116 GB/s，再经 PCIe 回填） | 约 93 ms | 约 20 ms |
| 远端 RDMA（节点 8 × 400 Gb/s ≈ 400 GB/s / 单端口 50 GB/s） | 约 27 ms / 215 ms | 约 6 ms / 46 ms |
| 单块 Exos 30 TB 顺序读（275 MB/s） | 约 39 s | 约 8.4 s |
| 单块 HDD 按 fs 层文件随机读（每 64-token 块、每 rank 一个文件，每文件 1 次定位 1/170 s） | 4,096 个 2.5 MiB 文件，约 63 s | 512 个 4.3 MiB 文件（MLA 只存一份），约 11 s |

**与 prefill 重算比较（理论下界）。**Llama-70B 一个 32k 前缀的 prefill 计算量约 `2 × 70.6e9 × 32,768 + 2 × 80 × 8,192 × 32,768² ≈ 6.0 PFLOP`（第二项为注意力项，按非因果计，偏大）。8 张 H100 按页面 BF16 1,979 TFLOPS（含稀疏）折半取 989.5 TFLOPS 稠密峰值计算，即使利用率为 100% 也需要约 0.76 s。DeepSeek-V3 按 37B 激活参数计约 5.1 PFLOP，8 × H200 FP8 稠密峰值（3,958 TFLOPS 含稀疏，折半）下约 0.32 s。实际 TTFT 应高出数倍，以实测为准，本文标“**待实测**”。官方 CPU 卸载博客只给出 Llama-3.1-8B 单卡 H100 上 TTFT 降低 2–22 倍的数据，没有 70B 的对应数字。结论是：DRAM、本地 NVMe、RDMA 的取回时间都比 prefill 小一个数量级以上；单块 HDD 则比重算慢 1–2 个数量级。 [VLH14][DSH5][VLH15][VL5]

### 官方与公开的参考部署

| 出处 | 模型 / 场景 | 硬件（机型、卡数、网络、存储） | 公开结果 |
|---|---|---|---|
| vLLM 博客 KV Offloading Connector（2026-01-08） | Llama-3.1-8B / 3.2-1B，CPU 卸载 | 1 × H100 80 GB HBM3，500 GB DRAM，Xeon Sapphire Rapids 2.1 GHz（限 8 核） | 单向 GPU↔CPU 约 50 GB/s；双向 DMA 83.4 GB/s，自定义内核 68.5 GB/s [VL5] |
| vLLM 博客 Tiered KV Offloading（2026-09-10） | Qwen3.6-35B-A3B，多轮对话（初始 12K token，每轮 +4K，8 轮），并发 64 | 2 × H100（TP=2），fs 二级层位于本地 NVMe；示例 `cpu_bytes_to_use` 为 100 GiB | 超过约 128 个会话、CPU 层也装满后，存储卸载的吞吐超过其他方案两倍以上 [VLH18] |
| vLLM 博客 Large Scale Serving（2025-12-17） | DeepSeek 系列，Wide-EP + DP | CoreWeave H200 集群，InfiniBand、ConnectX-7 网卡，多节点 | 每张 H200 持续 2.2k tokens/s [VLH19] |
| vLLM 博客 Mooncake Store（2026-05-06） | Kimi-2.5 NVFP4，Codex 智能体轨迹，P/D 分离 | GB200 节点：1P1D 共 12 张 GB200（P 为 TP4，D 为 DP8 + EP），扩展到 60 张；Mooncake 客户端经 RDMA 组成 DRAM / SSD 池 | 命中率由 1.7% 升至 92.2%，吞吐 3.8 倍，P50 TTFT 降低 46 倍 [VLH20] |
| vLLM 博客 GB300 DeepSeek（2026-02-13） | DeepSeek-V3.2 / R1，NVFP4，P/D 使用 NixlConnector | GB300 / B300 每卡 288 GB，两卡即可容纳 NVFP4 权重 | EP 作 prefill 的峰值吞吐比 TP2 高约 10%–15% [VLH22] |
| vLLM 博客 DeepSeek-V4.1-Flash PD + Wide-EP（2026-10-23） | SemiAnalysis AgentX | 一个 GB300 NVL72 机柜 | 原文：954K tokens/s per GPU，P90 196 tokens/s/用户，为单节点聚合部署的 7.9 倍 [VLH21] |
| vLLM 文档 llm-d 集成页（固定提交） | 前缀感知路由、P/D、分层卸载 | Llama 3.1 70B 在 AMD MI300X；GPT-OSS 在 B200；分层卸载在 H100 | 输出吞吐 3 倍、TTFT 2 倍；P/D 最高 +70% tokens/s；高并发下分层卸载为 13.9 倍 [VLH23] |
| 云机型（规格页，可承载上述部署） | — | AWS p5.48xlarge：8 × H100（640 GiB）、2,048 GiB 内存、3,200 Gb/s、8 × 3,800 GB NVMe；p5e / p5en：8 × H200（1,128 GiB）。Azure ND96isr_H100_v5：8 × H100、1,900 GiB、每卡 400 Gb/s CX7 InfiniBand、8 块 NVMe 合计 28 TiB | GCP 上 llm-d 1.0 支持 TPU 与 GPU；具体机型规格**未能核实**（Google 域名超时） [PLAWS3][PLMS6][GCP12] |

### 对 HDD 层的硬件含义（本文分析）

接入点沿用本组第 2 章：`OffloadingConnector` + `TieringOffloadingSpec` 的 fs 二级层（或 LMCache FS L2），所有读写都经主机 DRAM 主层中转。由此有三种硬件形态：（a）计算节点或存储服务器经 SAS 直连 JBOD（例如 Exos E 4U106），以本地挂载点作为 `root_dir`；（b）远端 NAS / 并行文件系统挂载为 `locality: REMOTE` 的 fs 层，WD / Solidigm 报告采用的就是 NFS 远端 HDD；（c）HDD 后端的 S3 兼容对象存储接 `obj` 层。三者在引擎看来都只是“二级层”，差别在于盘到主机之间的链路。 [VL3][VL18][KT11]

**读带宽所需盘数**：`N_盘 ≥ R × S ÷ (B_盘 × η)`，其中 R 为每秒需要冷恢复的前缀数，S 为前缀大小，B_盘 = 275 MB/s，η 为顺序效率。按上一节的 fs 文件粒度（TP=8，每文件 2.5 MiB，一次定位约 5.9 ms，传输约 9.5 ms）计，η ≈ 0.62。示例：要让一个 Llama-70B 32k 前缀在 1 s 内从 HDD 读完，需要 10.74 GB ÷ (275 MB/s × 0.62) ≈ **63 块盘**并行；每秒 10 个这样的冷命中需要约 630 块盘，约 6 台 4U106。DeepSeek-V3 MLA 的前缀小 4.7 倍，每个前缀每秒约需 11 块盘就够。 [BH11][KT11]

**写入吸收**：fs 层把进入 CPU 主层的每个块级联写入二级层。假设节点持续产生 20k prefill token/s，Llama-70B 的写流为 20,000 × 327,680 B ≈ 6.6 GB/s，按顺序写也要约 24 块盘。但 fs 层以“每 chunk 每 rank 一个文件 + rename”的方式写，属于 create-heavy 负载。WD / Solidigm 报告测得的 create-heavy 持久写上限约 692 MB/s，因此实际写能力以该类实测为准，需要靠 `offload_prompt_only`、加大 `blocks_per_chunk` 与背压丢写来削减。 [VL18][VL3][VL11]

**前置缓冲**：一台 p5.48xlarge 的 2,048 GiB 内存最多容纳约 205 个 Llama-70B 32k 前缀；DGX B200 的 8 × 3.84 TB NVMe 约 30.7 TB，可容纳约 2,860 个；一块 30 TB Exos 可容纳约 2,790 个，MLA 下约 13,000 个。WD / Solidigm 报告显示，前置 30 GiB DRAM 后，HDD 与 QLC 在测试点上无差异。因此 HDD 的合理位置是 DRAM / NVMe 之后的容量与持久层：命中先提升到 SSD / DRAM，再走 PCIe 回填。 [PLAWS3][DSH3][BH11][VL18]

**受限链路**：一台 4U106 满配的盘侧理论合计为 106 × 275 MB/s ≈ 29 GB/s，与 deepseek-hw 章引用的机箱最大带宽 36 GB/s 同一量级，都低于一个 400 Gb/s 端口（50 GB/s），也低于一条 PCIe Gen5 x16。所以 HDD 层的瓶颈在**盘数与随机定位**，而不在网卡或 PCIe。SAS HBA 与扩展器的规格本文未核实。另外，fs 层的 promotion 没有超时机制（见本组第 2 章）；单盘 39 s 的读时间如果落在请求的关键路径上，会直接表现为 TTFT。部署时需要用 `max_load_tokens` 或外部路由，在“预计读取时间大于重算时间”时放弃加载。 [KT11][VL9][VL3]

### 资料边界

硬件矩阵取自固定提交的安装文档与各插件 README，插件的成熟度标签（Beta、社区维护、实验）均按原文转述，没有在任何加速器上运行验证。HBM 可放 token 数、取回时间与 prefill 下界都是按数据手册峰值推算的理论值；PCIe 采用 vLLM 博客在 H100 上的实测约 50 GB/s，其余链路的有效带宽、SAS HBA 规格及 GCP 机型规格未能核实。DeepSeek-V3 的权重大小按 671B × 1 B 估算，没有使用镜像站给出的索引总字节数。

## 本章资料

- **[BH11]** [Seagate · Exos 数据手册 DS2046.1-2512（12–32 TB 型号随机读写 4K QD16 与持续传输，2025-12，采集于 2026-11-19）](https://www.seagate.com/content/dam/seagate/en/content-fragments/products/datasheets/exos-portfolio-refresh/exos-ds2046.1-2512-en_us_V2_en_AS.pdf)
- **[DS16]** [DeepSeek 官方 Hugging Face · deepseek-ai/DeepSeek-V3 config.json（kv_lora_rank、qk_rope_head_dim、num_hidden_layers 等，采集于 2026-11-18）](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)
- **[DSH3]** [NVIDIA · DGX B200 产品页 Specifications（8× Blackwell GPU、1,440 GB HBM3e / 64 TB/s、NVLink 14.4 TB/s、2 TB 系统内存、8× ConnectX-7 400 Gb/s、8× 3.84 TB NVMe U.2，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- **[DSH5]** [NVIDIA · H100 产品页（Interconnect：PCIe Gen5 128 GB/s；NVLink-C2C 900 GB/s，为 PCIe Gen5 的 7 倍，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/h100/)
- **[GCP12]** [Google Cloud Blog · Announcing llm-d 1.0: Open distributed inference for agents on Google Cloud TPUs and GPUs（2026-10-21）](https://cloud.google.com/blog/topics/ai-infrastructure/llm-d-1-0-an-open-source-stack-for-distributed-inference)
- **[KT11]** [Seagate · Exos E 4U106 数据手册 DS1980-12（2022-05；106 盘位、每 PSU 2000 W、满载 141 kg）](https://www.seagate.com/content/dam/seagate/migrated-assets/www-content/datasheets/pdfs/exos-e-4u106-DS1980-12-2205US-en_US.pdf)
- **[PLAWS3]** [AWS · Amazon EC2 Instance Types：Specifications for accelerated computing instances（p5 / g5 / g6 / g6e 规格，采集于 2026-11-19）](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- **[PLGG1]** [llm-d 官方文档（v1.0）· Architecture › KV Cache Management › KV-Cache Offloading（采集于 2026-11-19）](https://llm-d.ai/docs/architecture/kv-management/kv-offloader)
- **[PLMS6]** [Microsoft Learn · ND H100 v5 sizes series（Standard_ND96isr_H100_v5 规格，采集于 2026-11-19）](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndh100v5-series)
- **[VL1]** [vLLM 官方 GitHub 仓库 · main 分支固定提交 0c80c3a6aa0201d1b4f71915f90c03fd64837cec（2026-11-18；含 kv_connector/factory.py 注册表，采集于 2026-11-18）](https://github.com/vllm-project/vllm/tree/0c80c3a6aa0201d1b4f71915f90c03fd64837cec)
- **[VL3]** [vLLM 官方文档 · KV Offloading Usage Guide（OffloadingConnector、TieringOffloadingSpec、fs / obj / p2p 二级层、On-Disk Layout、max_load_tokens，采集于 2026-11-18）](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- **[VL5]** [vLLM 官方博客 · Inside vLLM's New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput（Or Ozeri、Danny Harnik，IBM Research），2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
- **[VL9]** [vLLM 固定提交源码 · vllm/v1/kv_offload/tiering/fs/manager.py 与 io.py（FileSystemTierManager、batch_store_block / batch_load_block、probe_o_direct）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/fs/manager.py)
- **[VL11]** [vLLM 固定提交源码 · vllm/v1/kv_offload/tiering/backpressure.py（EMABackpressureDetector、DropStorePolicy、ThrottledDropPolicy 及默认水位）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/backpressure.py)
- **[VL14]** [vLLM 固定提交源码 · vllm/config/cache.py（DEFAULT_BLOCK_SIZE、prefix_match_unit、kv_offloading_size / kv_offloading_backend）、config/vllm.py（_post_init_kv_transfer_config）、config/kv_transfer.py（kv_load_failure_policy）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/config/cache.py)
- **[VL18]** [WD × Solidigm · KV Cache Offload Across Storage Tiers，September 2026 / Revision 02](https://documents.westerndigital.com/content/dam/doc-library/en_us/assets/public/western-digital/collateral/tech-brief/technical-brief-kv-cache-offload-across-storage-tiers-wd.pdf)
- **[VL22]** [vLLM · NixlConnector：侧通道、UCX、设备选择及 P/D 读取方向（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/features/nixl_connector_usage.md)
- **[VL23]** [vLLM V1 GPU Worker · determine_available_memory 与显存 profile（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/worker/gpu_worker.py#L716-L826)
- **[VLH1]** [vLLM 固定提交文档 · docs/getting_started/installation/README.md（GPU / CPU 平台与树外硬件插件，采集于 2026-11-20）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/getting_started/installation/README.md)
- **[VLH2]** [vLLM 固定提交文档 · gpu.cuda.inc.md（计算能力 ≥ 7.5、CUDA 12.9 预编译、12.8 / 13.2 轮子、Blackwell 需 CUDA ≥ 12.8，采集于 2026-11-20）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/getting_started/installation/gpu.cuda.inc.md)
- **[VLH3]** [vLLM 固定提交文档 · gpu.rocm.inc.md（MI200 / MI300 / MI350、ROCm ≥ 6.3、ROCm 10.0 轮子，采集于 2026-11-20）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/getting_started/installation/gpu.rocm.inc.md)
- **[VLH4]** [vLLM 固定提交文档 · gpu.xpu.inc.md（Intel Data Center GPU / Arc、vllm-xpu-kernels、compute runtime 26.18、xccl，采集于 2026-11-20）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/getting_started/installation/gpu.xpu.inc.md)
- **[VLH5]** [vLLM 固定提交文档 · cpu.md / cpu.x86.inc.md / cpu.arm.inc.md（avx512f / avx2、NEON、FP32 / FP16 / BF16，采集于 2026-11-20）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/getting_started/installation/cpu.x86.inc.md)
- **[VLH6]** [vLLM 官网首页 · Universal Compatibility – Hardware（采集于 2026-11-20）](https://vllm.ai/#compatibility)
- **[VLH7]** [vllm-project/vllm-ascend README（提交 8f603de4f07b1f60c9486a66f8dc7c09bc765b5f；Atlas A2 / A3 / 300I Duo / 950PR&950DT、CANN 9.2.1、v0.30.0，采集于 2026-11-20）](https://github.com/vllm-project/vllm-ascend/blob/8f603de4f07b1f60c9486a66f8dc7c09bc765b5f/README.md)
- **[VLH8]** [vllm-ascend 文档 · KV Cache Pool (Ascend Store) 与 KV Cache CPU Offload Guide（提交 8f603de，采集于 2026-11-20）](https://github.com/vllm-project/vllm-ascend/blob/8f603de4f07b1f60c9486a66f8dc7c09bc765b5f/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md)
- **[VLH9]** [vllm-project/tpu-inference README 与 docs/includes/core_features.md（提交 80cfbb7697b5cb4484ec1e7681d8ee9bc38db917；推荐 v7x / v5e / v6e，采集于 2026-11-20）](https://github.com/vllm-project/tpu-inference/blob/80cfbb7697b5cb4484ec1e7681d8ee9bc38db917/README.md)
- **[VLH10]** [vllm-project/vllm-neuron README（提交 5b527a1a8e27ba274ec1dcb7155ece6c96ec4761；0.29.0.1.2.0、Neuron SDK 2.33、Trn2 / Trn3、Beta、KV cache offloading 不支持，采集于 2026-11-20）](https://github.com/vllm-project/vllm-neuron/blob/5b527a1a8e27ba274ec1dcb7155ece6c96ec4761/README.md)
- **[VLH11]** [vllm-project/vllm-gaudi · README 与 docs/getting_started/compatibility_matrix.md（提交 2f1aaea017ca975ab1b5fffe037648c3671fb47a；v0.30.0 / Gaudi 软件 1.24.2，采集于 2026-11-20）](https://github.com/vllm-project/vllm-gaudi/blob/2f1aaea017ca975ab1b5fffe037648c3671fb47a/docs/getting_started/compatibility_matrix.md)
- **[VLH13]** [vLLM 固定提交源码 · vllm/v1/attention/backends/mla/flashmla_sparse.py（fp8_ds_mla 每 token 656 B 布局说明）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/attention/backends/mla/flashmla_sparse.py)
- **[VLH14]** [ModelScope 镜像 · LLM-Research/Meta-Llama-3.1-70B-Instruct config.json 与 model.safetensors.index.json（80 层、8 KV head、total_size 141,107,412,992，采集于 2026-11-20）](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B-Instruct/file/view/master/config.json)
- **[VLH15]** [NVIDIA · H200 产品页 Specifications（141 GB HBM3e、4.8 TB/s、NVLink 900 GB/s、PCIe Gen5 128 GB/s，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/h200/)
- **[VLH16]** [AMD · Instinct MI300X 产品页 Specifications（192 GB HBM3、5.3 TB/s、PCIe 5.0 x16、8 条 Infinity Fabric 链路 / 每条 128 GB/s，采集于 2026-11-20）](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- **[VLH17]** [Solidigm · D7-PS1010 产品页（128KB 顺序读 14,500 MB/s、顺序写 10,500 MB/s、4KB 随机读 3.3M IOPS，采集于 2026-11-20）](https://www.solidigm.com/products/data-center/d7/ps1010.html)
- **[VLH18]** [vLLM 官方博客 · Tiered KV Cache Offloading（2026-09-10；2× H100、本地 NVMe fs 层基准）](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading)
- **[VLH19]** [vLLM 官方博客 · vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP（2025-12-17）](https://vllm.ai/blog/2025-12-17-large-scale-serving)
- **[VLH20]** [vLLM 官方博客 · Mooncake Store 集成（2026-05-06；GB200、1P1D 12 卡至 60 卡）](https://vllm.ai/blog/2026-05-06-mooncake-store)
- **[VLH21]** [vLLM 官方博客 · Serving DeepSeek-V4.1-Flash at Scale: vLLM Matches DeepInfer and Scales to GB300 NVL72（2026-10-23）](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)
- **[VLH22]** [vLLM 官方博客 · DeepSeek-V3.2 on GB300: Performance Breakthrough（2026-02-13）](https://vllm.ai/blog/2026-02-13-gb300-deepseek)
- **[VLH23]** [vLLM 固定提交文档 · docs/deployment/integrations/llm-d.md（Performance：MI300X / B200 / H100 代表结果，采集于 2026-11-20）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/deployment/integrations/llm-d.md)
- **[VLH25]** [vLLM GitHub Releases · v0.34.0（2026-11-16，采集于 2026-11-20）](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)
