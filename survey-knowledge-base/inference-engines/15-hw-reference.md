# 硬件参数速查：加速器、互连、网卡、SSD 与 HDD

> 参考速查 · 官方数据手册 / 产品页参数与理论下界换算

本章是 2.5「推理引擎与硬件协同」的参数补充：该章解释了 KV 数据在 HBM、主机内存、网络与存储之间经过哪些通路，本章只把这些通路上常见器件的数据手册级参数列成速查表，并给出“取回 1 GiB KV 的理论下界”换算，供本组三章引擎算例统一引用。表中数值均取自所引官方产品页、数据手册或官方文档（采集于 2026-11-20，KIOXIA CM7-R 页面采集于 2026-11-22）；凡由两个官方数值相除、相乘得到的结果，都标为“本文换算”。

### 口径与单位

- **容量与带宽单位。**厂商带宽一律为十进制：1 GB/s = 10⁹ B/s，1 TB/s = 10¹² B/s，1 Gb/s = 1.25×10⁸ B/s。KV 体积按二进制计：1 GiB = 2³⁰ = 1,073,741,824 B。HDD 手册同时给出 MB/s 与 MiB/s 时，本章用 MB/s 计算。
- **双向与单方向。**NVIDIA 产品页给出的 NVLink、NVLink-C2C 与“PCIe Gen5: 128GB/s”均为每 GPU 双向合计；PCIe Gen5 x16 的单方向原始速率是 32 GT/s × 16 ÷ 8 = 64 GB/s，与 128 GB/s 双向一致。昇腾 950PR 页面直接把 PCIe 5.0 x16 写为“128GB/s 双向”。 [DSH5][HWR5]KV 取回是单方向搬运，下文换算一律按“双向值 ÷ 2”取单方向。
- **“最高”与“典型”。**SSD 顺序读写与随机 IOPS 都是厂商给出的上限（特定容量、队列深度、块大小）；HDD 持续传输率是外圈（OD）最大值，内圈更低。理论下界表直接使用这些上限，因此只能当作不可逾越的下限，不是预期时延。

### 加速器：HBM 与卡间互连

| 加速器（形态） | HBM 容量 | HBM 带宽 | 卡间 / 机架内互连 | 主机接口 | 来源 |
|---|---|---|---|---|---|
| NVIDIA H100 SXM | 80 GB | 3.35 TB/s | NVLink 900 GB/s（第四代） | PCIe Gen5 128 GB/s | [DSH5] |
| NVIDIA H200 SXM | 141 GB HBM3e | 4.8 TB/s | NVLink 900 GB/s | PCIe Gen5 128 GB/s | [SGH23] |
| NVIDIA B200（DGX B200，8 卡） | 整机 1,440 GB HBM3e；每卡 180 GB（本文换算 ÷8） | 整机 64 TB/s；每卡 8 TB/s（本文换算） | NVLink 整机 14.4 TB/s 聚合；第五代 NVLink 每 GPU 1,800 GB/s | 整机 2 TB 系统内存（可配 4 TB） | [DSH3][DSH6] |
| NVIDIA GB200（Grace Blackwell 超级芯片：1 Grace + 2 GPU） | 每超级芯片 372 GB HBM3E；NVL72 整柜 13.4 TB | 每超级芯片 16 TB/s；整柜 576 TB/s | 每超级芯片 NVLink 3.6 TB/s；整柜 130 TB/s | Grace 侧最高 480 GB LPDDR5X / 最高 512 GB/s；GPU 与 Grace 之间为 NVLink-C2C | [DSH4] |
| NVIDIA GB300 NVL72（72 Blackwell Ultra GPU + 36 Grace） | 整柜 GPU 内存 20 TB（“Fast Memory” 37 TB 含 CPU 内存） | 整柜最高 576 TB/s | 整柜 NVLink 130 TB/s | 整柜 17 TB LPDDR5X / 14 TB/s | [HWR1] |
| AMD Instinct MI300X（OAM） | 192 GB HBM3 | 5.3 TB/s（8192-bit × 5.2 Gbps ÷ 8 = 5.325 TB/s） | 8 条 Infinity Fabric 链路，每条峰值 128 GB/s | PCIe 5.0 x16 | [VLH16] |
| AMD Instinct MI325X（OAM） | 256 GB HBM3E | 6 TB/s | 8 条 Infinity Fabric 链路，每条峰值 128 GB/s | PCIe 5.0 x16 | [HWR2] |
| AMD Instinct MI355X（OAM） | 288 GB HBM3E | 8 TB/s | 7 条 Infinity Fabric 链路，scale-up 每条峰值 144 GB/s；另有 1 条 scale-out 链路，峰值 128 GB/s | PCIe 5.0 x16 | [HWR3] |
| AWS Trainium2（单芯片） | 96 GiB | 2.9 TB/s | NeuronLink-v3 每芯片 1.28 TB/s；内存池最多 64 芯片 | DMA 带宽 3.5 TB/s | [HWR4] |
| 华为 Ascend 950PR | 128 / 112 / 84 GB（按 SKU） | 1.6 / 1.4 TB/s | Unified Bus 18 端口 × 112 Gbps，2016 GB/s 双向；UBoE 2 × 400 Gbps | PCIe 5.0 x16，128 GB/s 双向 | [HWR5] |
| 华为昇腾 910B / 910C | 昇腾社区处理器页当前只列 Ascend 950PR / 950DT；910B / 910C 的 HBM 与互连规格未在可访问的官方页面找到，**未能核实**。 | — |  |  |  |
| Google TPU v5e / v6e / Ironwood（TPU7x） | Cloud TPU 文档（cloud.google.com）在采集环境中多次超时，**未能核实**，本表不填数。 | — |  |  |  |

对 KV 容量估算最常用的是“每卡 HBM”一列：H100 80 GB、H200 141 GB、B200 约 180 GB、MI300X 192 GB、MI325X 256 GB、MI355X 288 GB、Trainium2 96 GiB。AWS EC2 文档列出的实例可见内存略低于标称值，例如 p5e / p5en 为 8 × 141 GiB、p6-b200 为 8 × 179 GiB、trn2.48xlarge 为 16 × 512 GiB（每个 Trainium2 加速器设备 512 GiB），引擎算例如果按云实例部署，应以实例文档为准。 [PLAWS3]

### 主机与片间互连：PCIe、NVLink、NVLink-C2C、CXL

| 链路 | 官方给出的速率 | x16 / 每 GPU 双向 | 单方向（本文换算） | 来源 |
|---|---|---|---|---|
| PCIe Gen4 x16 | 16 GT/s（ConnectX-7 Socket Direct 辅助卡以 Gen4 16 GT/s 运行） | 64 GB/s（本文换算） | 32 GB/s 原始 | [HWR7] |
| PCIe Gen5 x16 | 32 GT/s；NVIDIA 产品页标 128 GB/s | 128 GB/s | 64 GB/s 原始；扣 128b/130b 编码约 63.0 GB/s | [DSH5][HWR7] |
| PCIe Gen6 x16 | 64 GT/s（ConnectX-8 默认 Gen6 x16） | 256 GB/s（本文换算） | 128 GB/s 原始 | [HWR8] |
| NVLink 第四代（Hopper） | 每 GPU 900 GB/s，每 GPU 最多 18 条链路 | 900 GB/s | 450 GB/s | [DSH6] |
| NVLink 第五代（Blackwell） | 每 GPU 1,800 GB/s，每 GPU 最多 18 条链路 | 1,800 GB/s | 900 GB/s | [DSH6] |
| NVLink 第六代（Vera Rubin） | 每 GPU 3,000 GB/s，每 GPU 最多 36 条链路；产品页称为 PCIe Gen6 的 12 倍 | 3,000 GB/s | 1,500 GB/s | [DSH6] |
| NVLink-C2C（Grace–Hopper / Grace–Blackwell） | 900 GB/s 一致性接口，产品页称为 PCIe Gen5 的 7 倍 | 900 GB/s | 450 GB/s | [HWR6][DSH5] |
| AMD Infinity Fabric（MI300X / MI325X / MI355X） | 每条链路峰值 128 / 128 / 144 GB/s | 按链路数相加：8×128、8×128、7×144 GB/s（本文换算，不等于任意两卡间带宽） | 依拓扑而定 | [VLH16][HWR2][HWR3] |
| CXL 3.x / 4.0 | CXL 4.0 把速率从 64 GT/s 提升到 128 GT/s，沿用 CXL 3.x 的 256B Flit，并引入原生 x2 宽度与 bundled port | x16：64 GT/s 对应 256 GB/s、128 GT/s 对应 512 GB/s（本文换算，未计 Flit 开销） | 128 / 256 GB/s 原始 | [HWR11] |

PCI-SIG 规范页在采集环境中被 Cloudflare 校验拦截，表中 PCIe 速率取自 NVIDIA 网卡文档与产品页对所用代际的描述，x16 带宽按“GT/s × 16 ÷ 8”换算，没有计入编码以外的 TLP / DLLP 开销。实际 DMA 吞吐还受 Max Payload Size、读请求大小与根复合体实现影响，需实测。

### 网卡、DPU 与云实例网络

| 型号 / 实例 | 端口速率 | 主机接口 | 备注 | 来源 |
|---|---|---|---|---|
| NVIDIA ConnectX-7 | 单口 400 / 200 Gb/s（OSFP），或双口 200 / 100 Gb/s（QSFP112） | PCIe Gen 5.0（32 GT/s）x16；Socket Direct 为 2 × x16 | DGX B200 配 8 个单口 ConnectX-7，每口最高 400 Gb/s | [HWR7][DSH3] |
| NVIDIA ConnectX-8 SuperNIC | InfiniBand 与以太网最高 800 Gb/s | PCIe Gen6 64 GT/s x16；扩展选项走 Gen5 32 GT/s | Socket Direct 把 32 条通道拆成两个 x16，在 PCIe Gen5 / Gen4 服务器上分别提供 800 / 400 Gb/s | [HWR8] |
| NVIDIA BlueField-3 DPU | 以太网与 InfiniBand 最高 400 Gb/s | PCIe Gen 5.0（32 GT/s）x16 | DGX B200 另配 2 个双口 QSFP112 BlueField-3，最高 400 Gb/s | [HWR9][DSH3] |
| Broadcom Thor 2（BCM57608 NIC） | 单 / 双 / 四端口 400 Gb/s，25–400 Gb/s | PCIe Gen 5.0 x16 | RoCEv2，带硬件拥塞控制 | [HWR10] |
| AWS p5.48xlarge / p5e.48xlarge / p5en.48xlarge | 每实例 3200 Gigabit，支持 EFA | — | 8 × H100（640 GiB）/ 8 × H200（1128 GiB）；实例存储 8 × 3800 GB NVMe，随机读 / 写 4,400,000 / 2,200,000 IOPS；p5en EBS 带宽 100,000 Mbps，p5 / p5e 为 80,000 Mbps | [PLAWS3] |
| AWS p6-b300.48xlarge | 每实例 6400 Gigabit | — | 同表列出 8 × 3800 GB NVMe 实例存储 | [PLAWS3] |
| AWS trn2.48xlarge | 16 × 200 Gigabit，支持 EFA | — | 16 × Trainium2；实例存储 4 × 1900 GB NVMe，随机读 / 写 1,720,000 / 720,000 IOPS | [PLAWS3] |

换算到字节：400 Gb/s = 50 GB/s，800 Gb/s = 100 GB/s，p5 的 3200 Gb/s 实例总量 = 400 GB/s。ConnectX-8 的 800 Gb/s 需要 PCIe Gen6 x16 或两个 Gen5 x16；插在单个 Gen5 x16 槽上时，主机侧单方向原始上限约 64 GB/s，低于端口的 100 GB/s。 [HWR8]

### 存储设备：数据中心 NVMe SSD 与大容量 HDD

| SSD 型号 | 接口 | 顺序读 / 写（最高） | 4K 随机读 / 写（最高） | 来源 |
|---|---|---|---|---|
| Solidigm D7-PS1010 | PCIe 5.0 NVMe（E1.S / E3.S / U.2） | 128 KB 顺序读 14,500 MB/s；写 10,500 MB/s | 随机读 3.3M IOPS；随机写本页未列 | [SGH24] |
| KIOXIA CM7-R | PCIe 5.0 / NVMe 2.0 | 单端口 1x4 模式持续 128 KiB 顺序读最高 14,000 MB/s；顺序写依容量 3,500–7,000 MB/s | 持续 4 KiB 随机读最高 2,700K IOPS；随机写最高 310K IOPS | [HWR12] |
| Solidigm D5-P5336（大容量） | 页面未提取 | 页面未提取 | 随机读最高 1005K IOPS；容量最高 122.88 TB | [HWR13] |
| Samsung PM1743、SK hynix PS1010 | 官方产品页为纯 JavaScript 渲染，采集环境中无法取得规格文本，**未能核实**。 | — |  |  |

| HDD 型号 | 容量 / 记录方式 | 最大持续传输率 | 随机 IOPS | 来源 |
|---|---|---|---|---|
| Seagate Exos M（ST32000NM004K 等，DS2046.1） | 32 / 30 / 28 / 24 TB，CMR，7,200 RPM，SATA 6 Gb/s | OD：285 / 275 / 270 / 240 MB/s（272 / 262 / 257 / 229 MiB/s） | 4K QD16 WCD 随机读 / 写 170 / 350 IOPS；平均延迟 4.16 ms | [KI1] |
| Western Digital Ultrastar DC HC690（SATA） | 32 / 31 / 30 / 28 TB，主机管理型 SMR | 269 / 264 / 260 / 256 MB/s（位于容量约 10% 处） | 功耗表随机读 4KB：QD1 81 IO/s、QD4 117 IO/s | [KI5] |
| Toshiba MG11（SATA） | 24–14 TB，7,200 rpm，1 GiB 缓存 | 典型最高 295 MiB/s（系列按容量依次为 295 / 285 / 281 MiB/s） | 本文所引产品资料未给出 | [KT2] |

SSD 与 HDD 的差距主要不在顺序带宽，而在随机 IOPS：单块 PCIe 5.0 SSD 顺序读约为单块 Exos 32 TB 外圈持续速率的 51 倍（14,500 ÷ 285），4K 随机读则约为后者的 19,400 倍（3.3M ÷ 170）。KV 页面若不能合并成大段顺序对象，HDD 层的可用带宽由 IOPS 而不是 MB/s 决定。

### 取回 1 GiB KV 的理论下界

**公式。**带宽型链路：t = 1,073,741,824 B ÷ B（单方向）。IOPS 型访问：t = (1 GiB ÷ I/O 粒度) ÷ IOPS；64 KiB 粒度对应 16,384 次 I/O。阵列：顺序带宽按 n 块盘线性相加，随机 I/O 按 n 块盘均匀分摊。

**假设。**(1) 取各表中的单方向上限，不计协议、软件栈、排队与首块启动时间；(2) 链路独占，无并发写回或其他租户流量；(3) 1 GiB 是经过该链路的总有效字节，若分到多 rank，应按每条链路实际承担的份额重算；(4) 数据已在源端连续存放，SSD / HDD 顺序行不考虑读放大；(5) HDD 随机行用数据手册 4K QD16 的 170 IOPS 作为 64 KiB 读的上限，64 KiB 在 285 MB/s 下还要多 0.23 ms 传输时间，所以实际 IOPS 只会更低。这些都是**理论下界**，结果只说明“不可能更快”。

| 链路 / 介质 | 采用的单方向上限 | 1 GiB 理论下界 | 依据 |
|---|---|---|---|
| HBM（H200，片上读） | 4.8 TB/s | **0.22 ms** | [SGH23] |
| HBM（B200 每卡 / MI355X） | 8 TB/s | **0.13 ms** | [DSH3][HWR3] |
| NVLink 第四代（H100 / H200 对端 GPU） | 450 GB/s（900 ÷ 2） | **2.39 ms** | [DSH6] |
| NVLink 第五代（B200 / GB200 对端 GPU） | 900 GB/s（1,800 ÷ 2） | **1.19 ms** | [DSH6] |
| NVLink-C2C（Grace LPDDR5X → GPU） | 450 GB/s（900 ÷ 2） | **2.39 ms** | [HWR6] |
| PCIe Gen5 x16（主机 DRAM → GPU） | 64 GB/s 原始（扣编码约 63.0 GB/s） | **16.8 ms**（扣编码 17.0 ms） | [DSH5] |
| PCIe Gen6 x16 | 128 GB/s | **8.4 ms** | [HWR8] |
| 400 Gb/s RDMA（ConnectX-7 / BlueField-3 / Thor 2 单口） | 50 GB/s | **21.5 ms** | [HWR7][HWR9][HWR10] |
| 800 Gb/s RDMA（ConnectX-8，Gen6 x16 或 2 × Gen5 x16） | 100 GB/s | **10.7 ms**；若受单个 Gen5 x16 限制则约 16.8 ms | [HWR8] |
| AWS p5 / p5e / p5en 整机 EFA（3200 Gb/s，全部网卡并行） | 400 GB/s | **2.7 ms** | [PLAWS3] |
| 单块 NVMe SSD（D7-PS1010 顺序读） | 14.5 GB/s | **74 ms**（CM7-R 14.0 GB/s 为 77 ms） | [SGH24][HWR12] |
| 4 块 NVMe SSD 条带（4 × D7-PS1010） | 58 GB/s | **18.5 ms**（已接近单个 Gen5 x16 的 64 GB/s） | [SGH24] |
| 单块 HDD 顺序（Exos 32 TB 外圈） | 285 MB/s | **3.77 s**（HC690 32 TB 269 MB/s 为 3.99 s） | [KI1][KI5] |
| 12 块 HDD 顺序条带（12 × Exos 32 TB） | 3.42 GB/s | **314 ms** | [KI1] |
| 单块 HDD 随机 64 KiB（16,384 次 I/O） | ≤ 170 IOPS（4K QD16） | **≥ 96 s**（按 HC690 QD4 117 IO/s 为 140 s） | [KI1][KI5] |
| 12 块 HDD 随机 64 KiB（理想均匀分摊） | ≤ 2,040 IOPS | **≥ 8.0 s** | [KI1] |

读表时注意两点。第一，取回路径的稳定吞吐由最慢一段决定：从 12 盘 HDD 阵列经 400 Gb/s 网络送到 GPU，下界是 314 ms 而不是 21.5 ms；4 块 SSD 直连同一 GPU 时，PCIe Gen5 x16 与盘阵带宽处在同一量级，两者都可能先成为瓶颈。第二，同样 1 GiB，HDD 顺序与随机的下界相差约 25 倍（96 s ÷ 3.77 s），差别全部来自 I/O 粒度；把 KV 页聚合成数十 MiB 以上的连续段，是 HDD 层能否进入秒级以下预算的前提（本文分析）。

### 资料边界

Google Cloud TPU 文档与 PCI-SIG 规范页在采集环境中不可达，昇腾 910B / 910C、Samsung PM1743 与 SK hynix PS1010 的官方页面未能取得规格文本，这些项在表中标为未能核实，没有用二手资料补数。NVLink、C2C 与 PCIe 的单方向值按“双向 ÷ 2”换算，CXL 与 PCIe Gen6 的 x16 带宽按速率换算，均未计协议开销。全部下界只是规模估算的起点，引擎算例的实际毫秒数仍应以实测为准。

## 本章资料

- **[DSH3]** [NVIDIA · DGX B200 产品页 Specifications（8× Blackwell GPU、1,440 GB HBM3e / 64 TB/s、NVLink 14.4 TB/s、2 TB 系统内存、8× ConnectX-7 400 Gb/s、8× 3.84 TB NVMe U.2，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- **[DSH4]** [NVIDIA · GB200 NVL72 产品页 Specifications（72 GPU / 36 Grace、13.4 TB HBM3E / 576 TB/s；每超级芯片 372 GB HBM3E / 16 TB/s、最高 480 GB LPDDR5X，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- **[DSH5]** [NVIDIA · H100 产品页（Interconnect：PCIe Gen5 128 GB/s；NVLink-C2C 900 GB/s，为 PCIe Gen5 的 7 倍，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/h100/)
- **[DSH6]** [NVIDIA · NVLink and NVLink Switch 产品页（第五代 NVLink 每 GPU 1,800 GB/s，NVLink 域 8 | 72 GPU，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/nvlink/)
- **[HWR1]** [NVIDIA · GB300 NVL72 产品页 Specifications（72 Blackwell Ultra GPU / 36 Grace；GPU 内存 20 TB / 最高 576 TB/s；NVLink 130 TB/s；17 TB LPDDR5X / 14 TB/s，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)
- **[HWR2]** [AMD · Instinct MI325X 产品页 Specifications（256 GB HBM3E、6 TB/s、PCIe 5.0 x16、8 条 Infinity Fabric 链路 / 每条 128 GB/s，采集于 2026-11-20）](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- **[HWR3]** [AMD · Instinct MI355X 产品页 Specifications（288 GB HBM3E、8 TB/s、PCIe 5.0 x16、7 条 Infinity Fabric 链路 / 每条 144 GB/s，采集于 2026-11-20）](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
- **[HWR4]** [AWS Neuron 文档 · Trainium2 Architecture（每芯片 96 GiB 设备内存 / 2.9 TB/s；NeuronLink-v3 每芯片 1.28 TB/s；DMA 3.5 TB/s，Neuron 2.33.1 文档，采集于 2026-11-20）](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html)
- **[HWR5]** [华为昇腾社区 · 昇腾 NPU 处理器页 Ascend 950PR 技术规格（Memory 128/112/84 GB、1.6/1.4 TB/s；Unified Bus 18 端口 2016 GB/s 双向；PCIe 5.0 x16 128 GB/s 双向，采集于 2026-11-20）](https://www.hiascend.com/hardware/processor)
- **[HWR6]** [NVIDIA · Grace Hopper Superchip 产品页（NVLink-C2C 900 GB/s 一致性接口，为 PCIe Gen5 的 7 倍，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)
- **[HWR7]** [NVIDIA 文档 · ConnectX-7 Adapter Cards User Manual：Introduction / System Requirements（单口 400/200 Gb/s OSFP 或双口 200/100 Gb/s QSFP112；PCIe Gen 5.0 32 GT/s x16，采集于 2026-11-20）](https://docs.nvidia.com/networking/display/connectx7vpi/introduction)
- **[HWR8]** [NVIDIA 文档 · ConnectX-8 SuperNIC User Manual：Introduction / System Requirements（InfiniBand 与以太网最高 800 Gb/s；PCIe Gen6 64 GT/s x16；Socket Direct 在 PCIe Gen5/Gen4 服务器上提供 800/400 Gb/s，采集于 2026-11-20）](https://docs.nvidia.com/networking/display/connectx8supernic/introduction)
- **[HWR9]** [NVIDIA 文档 · BlueField-3 DPU User Manual：Introduction / System Requirements（以太网与 InfiniBand 最高 400 Gb/s；PCIe Gen 5.0 32 GT/s x16，采集于 2026-11-20）](https://docs.nvidia.com/networking/display/bf3dpu/introduction)
- **[HWR10]** [Broadcom · BCM57608 400G NIC Adapters Product Brief 957608-PB1（Thor 2；单/双/四端口 400 Gb/s，25–400 Gb/s；PCIe Gen 5.0 x16；RoCEv2，采集于 2026-11-20）](https://docs.broadcom.com/doc/957608-PB1)
- **[HWR11]** [CXL Consortium · About CXL（CXL 4.0 将带宽从 64 GT/s 提升到 128 GT/s、沿用 CXL 3.x 的 256B Flit、引入原生 x2 宽度与 bundled port，采集于 2026-11-20）](https://computeexpresslink.org/about-cxl/)
- **[HWR12]** [KIOXIA · CM7-R Series 企业级 SSD 产品页（PCIe 5.0 / NVMe 2.0；单端口 1x4 模式持续 128 KiB 顺序读最高 14,000 MB/s；4 KiB 随机读最高 2,700K IOPS、随机写最高 310K IOPS，采集于 2026-11-22）](https://americas.kioxia.com/en-us/business/ssd/enterprise-ssd/cm7-r.html)
- **[HWR13]** [Solidigm · D5-P5336 产品页（容量最高 122.88 TB；4 KB 随机读最高 1005K IOPS，采集于 2026-11-20）](https://www.solidigm.com/products/data-center/d5/p5336.html)
- **[KI1]** [Seagate · Exos 数据手册 DS2046.1-2512（12–32 TB 型号规格：随机读写 4K QD16、OD 持续传输、平均延迟，2025-12，采集于 2026-11-19）](https://www.seagate.com/content/dam/seagate/en/content-fragments/products/datasheets/exos-portfolio-refresh/exos-ds2046.1-2512-en_us_V2_en_AS.pdf)
- **[KI5]** [Western Digital · Ultrastar DC HC690 (SATA) Hard Disk Drive Specification（28–32 TB，持续传输、功耗表中随机读 IO/s、550 TB/年负载、Storage Element Depopulation，采集于 2026-11-19）](https://documents.westerndigital.com/content/dam/doc-library/en_us/assets/public/western-digital/product/data-center-drives/ultrastar-dc-hc600-series/product-manual-ultrastar-dc-hc690-sata.pdf)
- **[KT2]** [Toshiba Electronic Devices & Storage · MG11 Series Cloud-Scale Capacity HDD Product Overview（Rev.00，2024-08；14–24 TB 读写与主动空闲功耗）](https://toshiba.semicon-storage.com/content/dam/toshiba-ss-v3/master/en/storage/product/data-center-enterprise/MG11-Product-Overview_rev.0.pdf)
- **[PLAWS3]** [AWS · Amazon EC2 Instance Types：Specifications for accelerated computing instances（p5 / g5 / g6 / g6e 规格，采集于 2026-11-19）](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- **[SGH23]** [NVIDIA · H200 产品页 Specifications（141 GB HBM3e、4.8 TB/s；NVLink 900 GB/s；PCIe Gen5 128 GB/s，采集于 2026-11-20）](https://www.nvidia.com/en-us/data-center/h200/)
- **[SGH24]** [Solidigm · D7-PS1010 产品页（PCIe 5.0；128 KB 顺序读最高 14,500 MB/s、写 10,500 MB/s；4 KB 随机读最高 3.3M IOPS，采集于 2026-11-20）](https://www.solidigm.com/products/data-center/d7/ps1010.html)
- **[VLH16]** [AMD · Instinct MI300X 产品页 Specifications（192 GB HBM3、5.3 TB/s、PCIe 5.0 x16、8 条 Infinity Fabric 链路 / 每条 128 GB/s，采集于 2026-11-20）](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
