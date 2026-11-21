# vLLM：可行性验证与判定标准

> 推理引擎 · HDD 可行性分析

### 可行性判断与资料边界

| 判断层次 | 目前证据 | 尚缺验证 |
|---|---|---|
| 能否接入 | 原生 fs 二级层已在官方文档中给出 `root_dir` 配置；源码使用普通 POSIX I/O，并会探测 O_DIRECT 是否可用、不可用时回退；LMCache 提供 local_disk 和 FS L2 两条路径 | 在实际 HDD 挂载点、目标模型、TP 配置和混合注意力模型上完成读写，以及跨实例共享时哈希种子的一致性 |
| 能否比重算更合算 | CPU 层的官方测试中 TTFT 降低 2–22 倍（H100，Llama-3.1-8B）；WD / Solidigm 报告显示有 DRAM 前置层时 HDD 与 QLC 在测试点上无差异 | HDD 全冷恢复的耗时、有效命中率、P99 TTFT；promotion 等待没有超时，需要验证“放弃加载、改为重算”的阈值策略 |
| 能否长期运行 | tmp + rename 防止读到半写文件；短读自动删除并重算；二级层数据跨 reset 和重启保留；可选背压可在压力下丢弃写入 | 没有 fsync 时的掉电语义；fs 层没有淘汰机制时的容量治理；按 HDD 重新标定背压水位；多实例共享目录、长时间压力测试与故障注入 |

表中 CPU 层的数据来自 vLLM 官方博客 [VL5]，HDD 对照来自 WD / Solidigm 报告 [VL18]。本章没有在 GPU + HDD 环境中运行任何配置，所有“可行”的判断都停留在接口和源码层面。vLLM 官方没有发布原生 fs 层在 HDD 上的测试结果。LMCache 各后端是否调用 fsync，本次没有在其固定版本源码中核实，只引用了官方文档的描述。树外 spec / tier 的加载接口在源码中标注为 experimental，`max_load_tokens` 和 `max_offload_tokens` 在文档中也标为 experimental，参数和行为可能随版本变化，复现时应重新锁定提交。

### 验证顺序与通过条件

| 试验 | 方法与判定 |
|---|---|
| 命中与回退 | 固定模型、token 序列与解码参数。先写入前缀，再驱逐 GPU、CPU 副本，确认第二次请求触发 fs 读取；删除或截断目标文件，确认引擎重算且输出与基线一致。 |
| 延迟与读放大 | 用同一请求轨迹比较纯重算、CPU 主层、CPU + SSD、CPU + HDD。分别统计命中和未命中的 P50/P99 TTFT、排队时间、实际读取字节及避免的 prefill 时间。在目标并发下，HDD 命中须比重算更快，整体 P99 须满足业务预算。 |
| 容量与恢复 | 限制磁盘容量，持续注入新前缀直到触及水位，再模拟进程重启。记录积压、丢写、残留文件和恢复时间。只有空间能够回收、缓存错误可回退且前台服务满足既定预算，才进入长稳测试。 |

## 本章资料

- **[VL5]** [vLLM 官方博客 · Inside vLLM's New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput（Or Ozeri、Danny Harnik，IBM Research），2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
- **[VL18]** [WD × Solidigm · KV Cache Offload Across Storage Tiers，September 2026 / Revision 02](https://documents.westerndigital.com/content/dam/doc-library/en_us/assets/public/western-digital/collateral/tech-brief/technical-brief-kv-cache-offload-across-storage-tiers-wd.pdf)
