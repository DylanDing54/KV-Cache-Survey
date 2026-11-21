# Knowledge Base

Markdown edition of every chapter in the report (two books, four parts), plus the figures and a consolidated source index, for search, reuse and citation. Chapter text is in Chinese; each chapter ends with its source list.

# Book I · AI

## Part 1 · CSP and vendor survey (`vendors/`)

Every vendor chapter includes a 「放置策略：哪些 KV 放到哪一层」 section (which KV goes to which tier, and how that choice is made) and a 「硬件配置信息」 table of the hardware configurations mentioned in its sources.

| # | Chapter | Scope |
|---|---|---|
| 01 | [AWS / Amazon Bedrock](vendors/01-aws.md) | Bedrock prompt caching, SageMaker HyperPod tiered KV cache, FSx/S3 offload |
| 02 | [Tensormesh / LMCache 生态](vendors/02-lmcache.md) | Open-source KV cache layer: architecture, storage backends, P2P sharing, PD disaggregation |
| 03 | [Microsoft / Azure](vendors/03-microsoft.md) | Azure OpenAI / Foundry prompt caching, retention, pricing; Splitwise; Blob offload |
| 04 | [Google Cloud / Gemini](vendors/04-google.md) | Vertex AI / Gemini context caching, GKE Inference Gateway, llm-d tiered KV, Managed Lustre |
| 05 | [阿里云 / 百炼与 DashInfer](vendors/05-alibaba.md) | Bailian context cache, DashInfer, Tair KVCache, SGLang HiCache in production |
| 06 | [腾讯云 / TACO FlexKV](vendors/06-tencent.md) | TACO FlexKV multi-tier cache (GPU/CPU/SSD/remote), TACO-LLM prefix offload |
| 07 | [百度 / PaddlePaddle FastDeploy](vendors/07-baidu.md) | FastDeploy global cache pooling, AttentionStore, Qianfan prefix cache |
| 08 | [CoreWeave / Cohere](vendors/08-coreweave.md) | Cohere x CoreWeave x LMCache case on AI Object Storage / LOTA |
| 09 | [Moonshot AI / Kimi](vendors/09-moonshot.md) | Mooncake: KVCache-centric disaggregated serving, Transfer Engine, Store, SSD offload |
| 10 | [NVIDIA / Dynamo](vendors/10-nvidia.md) | Dynamo / KVBM / NIXL in the Seagate-SK hynix hybrid storage design |
| 11 | [Seagate / SK hynix](vendors/11-seagate.md) | SSD + HDD hybrid KV cache white paper: topology, test method, results |
| 12 | [Western Digital / Solidigm](vendors/12-wd.md) | Controlled benchmark: DRAM buffer in front of HDD / QLC capacity tiers |
| 13 | [AMD / MEXT](vendors/13-amd.md) | MEXT Predictive Memory: AI-driven DRAM / NVMe page tiering (official sources) |
| 14 | [华为 / KVU](vendors/14-huawei.md) | UCM, KVU compute-storage module, OceanStor, CloudMatrix384 EMS |
| 15 | [Meta](vendors/15-meta.md) | Transparent Memory Offloading (PSI + Senpai); LLM inference parallelism |
| 16 | [KV Cache 放置策略横向对比](vendors/16-placement-compare.md) | Cross-vendor comparison of KV placement policy (tier chain, admission, write timing, eviction, promotion, routing, decision-maker), design patterns, implications for an HDD tier, and a consolidated hardware-configuration table |

## Part 2 · AI inference engines (`inference-engines/`)

| # | Section | Chapter | Scope |
|---|---|---|---|
| 01 | 2.1 SGLang | [SGLang：整体架构与 KV 缓存管理](inference-engines/01-sglang-impl.md) | SGLang KV management implementation: RadixCache, memory pools, request lifecycle, HiCache tiers (with block diagrams) |
| 02 | 2.1 SGLang | [SGLang：HDD 接入方案与改动范围](inference-engines/02-sglang.md) | HiCache file backend, tiering options and required changes |
| 03 | 2.1 SGLang | [可行性验证与判定标准](inference-engines/03-validation.md) | Validation plan and pass criteria |
| 04 | 2.1 SGLang | [SGLang：与具体硬件的对应](inference-engines/04-sglang-hw.md) | Supported hardware matrix, HiCache tiers mapped to HBM / host DRAM / NVMe / RDMA / HDD, KV sizing and fetch-time worked example, LMSYS reference deployments, HDD-tier sizing |
| 05 | 2.2 vLLM | [vLLM：整体架构与 KV 缓存管理](inference-engines/05-vllm.md) | V1 process architecture, request scheduling, GPU execution, prefix cache and KV connectors |
| 06 | 2.2 vLLM | [vLLM：HDD 接入方案与改动范围](inference-engines/06-vllm-hdd.md) | Filesystem prototype, chunk layout, writeback and recovery |
| 07 | 2.2 vLLM | [vLLM：可行性验证与判定标准](inference-engines/07-vllm-validation.md) | Correctness, TTFT and restart tests |
| 08 | 2.2 vLLM | [vLLM：与具体硬件的对应](inference-engines/08-vllm-hw.md) | Backend/plugin hardware matrix, KV tiers and connectors mapped to hardware, Llama-3.1-70B / DeepSeek-V3 worked example, cloud reference deployments, HDD-tier sizing |
| 09 | 2.3 DeepInfer (DeepSeek) | [DeepInfer：整体架构与 KV 缓存管理](inference-engines/09-deepseek.md) | Online service topology, stage-level routing, offline engine loop and KV reuse |
| 10 | 2.3 DeepInfer (DeepSeek) | [DeepInfer（DeepSeek）：HDD 接入方案与改动范围](inference-engines/10-deepseek-hdd.md) | 3FS baseline, proposed HDD cold tier and changes to the read/write paths |
| 11 | 2.3 DeepInfer (DeepSeek) | [DeepInfer：可行性验证与判定标准](inference-engines/11-deepseek-validation.md) | Reuse intervals, read deadlines and fault injection |
| 12 | 2.3 DeepInfer (DeepSeek) | [DeepInfer：与具体硬件的对应](inference-engines/12-deepseek-hw.md) | SM100 / Ascend 950 support, V3–V4.1 deployment shapes, 3FS cluster hardware, KV sizing and retention capacity, HDD-tier sizing |
| 13 | 2.4 Enterprise research | [NVIDIA：完整 KV 是否值得长期保存](inference-engines/13-context-feasibility.md) | Value of retaining full long-context KV |
| 14 | 2.5 硬件协同 | [推理引擎与硬件协同：计算、互连、网络与存储](inference-engines/14-engine-hardware.md) | Compute and memory, GPU/NIC topology, RDMA fabrics, storage paths, bandwidth budgets and validation |
| 15 | 2.5 硬件协同 | [硬件参数速查：加速器、互连、网卡、SSD 与 HDD](inference-engines/15-hw-reference.md) | Datasheet tables for accelerators, interconnects, NICs / cloud instances, SSDs and HDDs; theoretical time to fetch 1 GiB over each link |

# Book II · Big data

## Part 3 · Big-data I/O optimization (`bigdata-io/`)

| # | Chapter | Scope |
|---|---|---|
| 01 | [AWS / Amazon S3](bigdata-io/01-bd-aws.md) | S3 heat management, ShardStore append-only storage node, request-spreading guidance; large drives at multi-tenant scale |
| 02 | [Google / Colossus](bigdata-io/02-bd-google.md) | Colossus placement, flash caching (CacheSack, L4), "Disks for Data Centers" asks of HDD makers |
| 03 | [Meta / Tectonic](bigdata-io/03-bd-meta.md) | Tectonic multi-tenant disk-time sharing, f4 warm storage, Tectonic-Shift flash tier for ML training IO |
| 04 | [Microsoft / Azure Storage](bigdata-io/04-bd-azure.md) | WAS append-only stream layer, IO scheduling, LRC erasure coding, Pelican rack-scale cold storage |
| 05 | [阿里云 / 盘古与 OSS](bigdata-io/05-bd-alibaba.md) | Pangu evolution, OSS SSD-fronted HDD path, SMRSTORE on host-managed SMR |
| 06 | [国内厂商与存力政策](bigdata-io/06-bd-china.md) | Huawei OceanStor Pacific, Tencent, ByteDance HDD stacks; China storage-capacity (存力) policy |
| 07 | [开源大数据存储：HDFS / Ozone / Ceph](bigdata-io/07-bd-opensource.md) | EC, storage policies, recovery throttles, Ceph DB/WAL sizing; Parquet / ORC / Iceberg IO patterns |
| 08 | [Dropbox Magic Pocket 与 Backblaze](bigdata-io/08-bd-dropbox.md) | SMR at scale, hardware generations (racks and power per EB), Backblaze Drive Stats by capacity |
| 09 | [HDD 厂商：密度与接口技术](bigdata-io/09-bd-hddtech.md) | HAMR / Mozaic, MACH.2 dual actuator, ePMR / UltraSMR, MAMR, CDL, depopulation, NVMe HDD |

## Part 4 · HDD density and capacity (`bigdata-capacity/`)

| # | Section | Chapter | Scope |
|---|---|---|---|
| 01 | 4.1 Constraints | [访问密度约束：IOPS/TB 与尾延迟](bigdata-capacity/01-cap-iops.md) | IOPS/TB and bandwidth/TB by capacity tier; which workloads are IOPS-bound; relief knobs |
| 02 | 4.1 Constraints | [重建、故障域与数据可靠性](bigdata-capacity/02-cap-reliability.md) | Rebuild time and repair traffic by capacity; RAID vs replication vs EC / LRC; AFR by capacity; mitigations |
| 03 | 4.1 Constraints | [写入模型约束：SMR / 分区存储与软件改造](bigdata-capacity/03-cap-write.md) | CMR vs DM-/HM-SMR, zone semantics, Linux zoned support, software changes and adopters |
| 04 | 4.1 Constraints | [成本、功耗与机架密度：大容量盘的 TCO](bigdata-capacity/04-cap-tco.md) | $/usable TB model with IOPS-limited fill, W/TB, rack power limits, embodied carbon claims |
| 05 | 4.2 Chinese deployments | [国内大数据场景的容量偏好：成因分析](bigdata-capacity/05-cap-china.md) | Sourced vs inferred reasons for preferring mid-capacity HDDs, three-way split of constraints, requirements-survey questions |
| 06 | 4.3 Actions | [让高密度大容量成为真实需求：行动建议](bigdata-capacity/06-cap-actions.md) | Device, system / software and commercial levers mapped to the constraints; roadmap; where the largest capacity is not the right choice |

## Topic notes (`topics/`)

| Note | Scope |
|---|---|
| [common-techniques.md](topics/common-techniques.md) | Techniques shared across CSPs, inference engines and storage vendors |
| [wd-tiering-rule.md](topics/wd-tiering-rule.md) | WD / Solidigm tiering rule, buffer sizing, and what "no performance impact" actually means |
| [nvidia-long-context.md](topics/nvidia-long-context.md) | NVIDIA *You don't need long context* and its implications for long-term KV retention |

## Other files

- [sources.md](sources.md) — all 610 cited sources with keys and URLs
- [figures/](figures/) — figures extracted from the report (official diagrams, document screenshots, PDF page renders, block diagrams drawn from source code)
