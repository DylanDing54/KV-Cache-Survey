# vLLM：HDD 接入方案与改动范围

> 推理引擎 · HDD 可行性分析

### 哪些路径能把 KV 放到本地文件系统 / HDD

按固定提交的源码和官方文档核对，能把 KV 写到普通文件路径的有四条，持久化语义各不相同：

| 路径 | 落盘形态与 I/O | 同步 / 异步与持久性 | 容量与淘汰 |
|---|---|---|---|
| OffloadingConnector + TieringOffloadingSpec，二级层 `type: "fs"`（vLLM 原生） | `root_dir/<model>_<digest>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin`，每个 chunk、每个 rank 一个文件；写入走 `open(tmp, O_CREAT\|O_EXCL[\|O_DIRECT])` → `write` → `os.replace`，读取走 `os.readv` | 由 `DualQueueThreadPool` 异步执行（读写各默认 16 线程）；临时文件 + rename 保证读不到半写的块；Python 与 C++（`csrc/fs_io.cpp`）两条路径都没有 fsync / fdatasync | fs 层没有容量上限，也不淘汰；配置变化会生成新的 digest 目录，旧目录需人工清理 |
| SimpleCPUOffloadConnector，`kv_offload_backend="disk"` | 每个 rank 一个预分配大文件 `disk_path.rank_N`，按 slot 偏移用 `pwritev` / `preadv` 读写，pinned 双缓冲 | 读、写各有一个独立线程；默认 O_DIRECT；启动和关闭时都会 `unlink` 该文件 | `disk_capacity_bytes`（默认 100 GiB）；文件不跨进程保留，只用于扩大单进程的卸载容量 |
| LMCacheConnectorV1（LMCache 进程内模式）`local_disk` / `gds_path` | 每个 chunk 一个文件（文档架构图中为 .pt），默认 `chunk_size` 256 token | put 异步、get 阻塞，可 prefetch 到 CPU；`use_odirect` 可选 | `max_local_disk_size`，按 LRU 等策略淘汰；LMCache 文档已把进程内模式标为 deprecated |
| LMCacheMPConnector + LMCache server 的 FS L2 adapter | aiofiles 写 `.data` 文件，文件名编码完整 ObjectKey | 可配临时目录，写完后原子 rename；`use_odirect` 默认 false | L1 / L2 的淘汰策略由 LMCache MP 模式管理 |

来源：fs 层见 [VL3][VL9]，disk 模式见 [VL12][VL3]，LMCache 两条路径见 [VL15][LMC24][VL16]。另外，`ExampleConnector` 会对每个请求的前缀 token 做哈希、建一个目录，在 `save_kv_layer()` 中把每一层同步写成一个 `safetensors` 文件。它的定位是调试示例，不适合当作 HDD 原型 [VL13]。

~~~python
# 源码出处（节选） https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/fs/io.py#L103-L109
# Check if block already exists to avoid redundant writes
if os.path.exists(dest_path):
    return False

tmp_path = dest_path + _get_tmp_suffix()
# Ensure parent directories exist
_ensure_dirs(dest_path)

# 源码出处（节选） https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/fs/io.py#L113-L123
view_slice = buffer.cast("B")[offset : offset + block_size]
o_direct = O_DIRECT if use_o_direct else 0
try:
    fd = os.open(
        tmp_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC | o_direct,
        0o644,
    )
    try:
        written = os.write(fd, view_slice)
        if written < len(view_slice):

# 源码出处（节选） https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/fs/io.py#L131-L134
finally:
    os.close(fd)
try:
    os.replace(tmp_path, dest_path)
~~~

原生 fs 层的数据流比文件格式更值得注意。`TieringOffloadingManager` 的设计注释写明了几条原则：任何 chunk 一旦存入 CPU 主层，就会**级联写到所有二级层**；二级层不能直接访问 GPU，读写都要经 CPU 主层中转；二级层的命中必须先“提升”（promotion）回 CPU 主层，GPU 才能读取 [VL10]。GPU 与 CPU 之间由 `CPUOffloadingWorker` 用 `swap_blocks_batch` 发起 DMA 异步拷贝 [VL5][VL10]。查找时，`OffloadingConnectorScheduler.get_num_new_matched_tokens()` 逐个 chunk 调用 `TieringOffloadingManager.lookup()`：先查 CPU 主层；未命中再由 `FsAsyncLookupManager.batch_lookup()` 在后台线程中批量检查文件是否存在（C 扩展用 faccessat）；在 fs 层命中后调用 `_initiate_promotion()` 并返回 `HIT_PENDING`。只要有一个 chunk 处于 `RETRY` 或 `HIT_PENDING`，整个查找就返回 `None`，请求留到下一步再调度。这个等待在源码里**没有超时上限** [VL10][VL9]。CPU 主层满了、无法接纳 promotion 时按 MISS 处理。批量读在第一个坏块处停止，之前读成功的块保留，失败块及其后面的块标记为 miss，交给 GPU 重算；短读的文件会被删除 [VL9]。所以二级层读失败通常会退化为局部重算，不会触发上一节的 `kv_load_failure_policy`，后者处理的是 worker 侧加载到 GPU 时的错误。

![KV 下沉到文件系统 / HDD 的三条官方路径及共同风险点：A 为 vLLM 原生 OffloadingConnector + fs 二级层（含写、读、查找与回退），B 为 LMCache 的两种 connector，C 为 SimpleCPUOffloadConnector 的 disk 模式（本文根据 vLLM 固定提交 0c80c3a6 源码 tiering/manager.py、tiering/fs/、offloading/scheduler.py、simple_kv_offload/disk_backend.py 及 LMCache 官方文档绘制）](../figures/vllm-hdd-1.svg)

*KV 下沉到文件系统 / HDD 的三条官方路径及共同风险点：A 为 vLLM 原生 OffloadingConnector + fs 二级层（含写、读、查找与回退），B 为 LMCache 的两种 connector，C 为 SimpleCPUOffloadConnector 的 disk 模式（本文根据 vLLM 固定提交 0c80c3a6 源码 tiering/manager.py、tiering/fs/、offloading/scheduler.py、simple_kv_offload/disk_backend.py 及 LMCache 官方文档绘制）* — [原始来源](https://github.com/vllm-project/vllm/tree/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload)

写入侧有可选的背压机制。在 fs 层配置 `backpressure` 后，`EMABackpressureDetector` 按“秒 / MiB”统计 store 延迟的指数移动平均，压力过高时由 `DropStorePolicy` 或 `ThrottledDropPolicy` 丢弃部分写入；连续 3 次写失败（如 ENOSPC）也会进入压力状态。fs 层默认使用 LOCAL 水位（高水位 0.010 s/MiB，低水位 0.002 s/MiB），源码注释说明这组默认值是在 NVMe 上用 fio 标定的 [VL11]。HDD 的随机写延迟与 NVMe 不在同一量级，沿用这组默认值可能让 HDD 层长期处于“丢写”状态，水位需要按实际阵列重新标定。

![LMCache 官方文档 Local storage 页的 LocalDiskBackend 架构图：元数据字典、可配置淘汰策略、带 PREFETCH / DELETE / PUT 优先级队列的 LocalDiskWorker（默认 4 个 I/O 线程），每个 chunk 一个文件](../figures/vllm-hdd-2.png)

*LMCache 官方文档 Local storage 页的 LocalDiskBackend 架构图：元数据字典、可配置淘汰策略、带 PREFETCH / DELETE / PUT 优先级队列的 LocalDiskWorker（默认 4 个 I/O 线程），每个 chunk 一个文件* — [原始来源](https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html)

### 最小原型配置（未执行示例）

下面两组配置依据固定提交的官方文档和 LMCache 文档整理，**均未执行**。目的是在 HDD 上建立“写入 → 冷恢复 → 失败回退到重算”的正确性基线，不能直接用于生产。示例 A 走 vLLM 原生 fs 层，在 HDD 前面放 30 GiB CPU 主层，对应 WD / Solidigm 报告中 DRAM 前置层的规模；示例 B 是任务要求的 LMCache `local_disk` 写法 [VL3][VL11][VL15]。

~~~bash
# 出处（据此整理） [VL3] https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
# 出处（据此整理） [VL11] https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/backpressure.py
# 示例 A：vLLM 原生 OffloadingConnector + fs 二级层（未执行）
# /mnt/hdd/vllm_kv 必须实际位于 HDD 阵列，不能是系统盘 SSD 或 tmpfs。
# n_read_threads / n_write_threads / blocks_per_chunk 都是待扫描的参数，不是推荐值；
# backpressure 的水位需要按 HDD 实测重新设定（默认值按 NVMe 标定）。
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 32212254720,
      "blocks_per_chunk": 4,
      "eviction_policy": "lru",
      "secondary_tiers": [{
        "type": "fs",
        "root_dir": "/mnt/hdd/vllm_kv",
        "n_read_threads": 8,
        "n_write_threads": 4,
        "locality": "LOCAL",
        "backpressure": {"backpressure_cls": "EMABackpressureDetector",
                         "policy_cls": "ThrottledDropPolicy"}
      }]
    }
  }'

# 出处（据此整理） [VL15] https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html
# 示例 B：LMCache 进程内模式 local_disk（未执行；LMCache 文档已将该模式标为 deprecated）
LMCACHE_CHUNK_SIZE=256 \
LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=30 \
LMCACHE_LOCAL_DISK="file:///mnt/hdd/lmcache/" LMCACHE_MAX_LOCAL_DISK_SIZE=2000 \
LMCACHE_EXTRA_CONFIG='{"use_odirect": true, "disk_io_threads": 4}' \
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
~~~

LMCache 的 MP 模式使用独立的存储服务。先启动服务，再在另一个终端启动 vLLM；以下配置尚未执行。

~~~bash
# 出处（据此整理） https://docs.lmcache.ai/mp/l2_storage/fs.html
lmcache server --l1-size-gb 30 --l2-adapter '{"type":"fs","base_path":"/mnt/hdd/lmcache_l2"}'

# 出处（据此整理） https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
vllm serve "$MODEL" --kv-offloading-backend lmcache --kv-offloading-size 30
~~~

注意 `kv_offloading_size` 在 lmcache 后端下只起开关作用，不会传给 LMCache [LMC24][VL14]。验证时，可以在请求的 `kv_transfer_params` 中设 `max_load_tokens: 0` 得到“只用 GPU 前缀缓存”的对照组，用 `kv_load_tiers` 选择是否查询二级层 [VL3]。

### 接入路线与改动范围

### HDD 特有瓶颈：对象数、随机读与写入吸收

**chunk 大小与对象数。**fs 层每个文件的大小等于单个 rank 的一个 chunk。vLLM 博客给出的 16-token 物理块在新内存布局下约为 0.5–2 MB [VL5]。以 WD / Solidigm 报告中 70B 模型每 token 320 KiB KV 估算：一个 64-token 块约 20 MiB，TP=8 时平均每个 rank 的文件约 2.5 MiB（本文估算，按 KV 在各 rank 间均分） [VL18]。TP 越大，单个文件越小，文件数越多；`canonical_layout` 可以让 MLA 等副本相同的 KV 只存一份 [VL3]。调大 `blocks_per_chunk` 可以减少对象数和元数据操作，代价是查找粒度变粗 [VL3]。每个 chunk 落盘时都要经过“检查是否存在、建目录、创建临时文件、写入、rename”这一串元数据操作，对 HDD 来说是 create-heavy 的写流。WD / Solidigm 报告测得：该阵列上一次持久化的 1 MiB 写入约需 13 ms，理想提交粒度下 create-heavy 持久写的上限约为 692 MB/s，提交粒度不同，持久写吞吐可相差约 9 倍 [VL18]。

**随机读与等待。**文件按哈希前缀分散到 `<hhh>/<hh>_g<group>` 目录。一条前缀在逻辑上是连续的，落到盘上却是一组相互独立的小文件，冷恢复因此表现为随机读。promotion 的数据要先读进 CPU 主层，再由 DMA 搬到 GPU。等待期间请求持续拿到 `None`，没有超时机制把它切换回重算。HDD 读得慢，代价会直接体现在该请求的 TTFT 上。这与 SGLang HiCache 提供 timeout 预取策略不同，需要在 vLLM 之上自行实现“预计读取比 prefill 慢就放弃加载”的判断（可以借助 `max_load_tokens`）。

**写入吸收。**“存入主层即级联到所有二级层”意味着 HDD 要承接全部卸载写流。`offload_prompt_only` 默认只卸载 prompt 块，能减少一部分写入 [VL3][VL10]。fs 层只有在 store 任务成功完成后才把键标记为存在，查找时也以文件是否存在为准，因此还没写完的 chunk 一律算 MISS [VL9]。WD / Solidigm 报告（vLLM 0.22.0 + LMCache 0.4.6，NFS 远端 HDD）观察到的正是这种现象：没有 DRAM 前置层时，同步（持久）导出的 HDD 在并发 32 下外部命中率只有 19.0%，上限为 71.7%，decode 为 34.7 tokens/s。报告的解释是“chunk 在被再次查询时尚未落盘”，并强调这不是读延迟造成的。前置 30 GiB DRAM 后，各容量层在这些测试点上的结果无法区分 [VL18]。需要说明，该报告用的是 LMCache 路径，不是本章所述的 vLLM 原生 fs 层，只能作为机制上的对照。

**持久化与重启。**vLLM 原生 fs 层的写路径没有 fsync；O_DIRECT 只是绕过页缓存，不等于掉电持久。掉电后可能留下短文件，读到短文件时实现会删除该文件并按 miss 重算，这为“重算兜底”提供了一定的自愈能力 [VL9]。另一方面，`reset_cache()` 刻意不清空 FS / 网络这类持久二级层，所以 HDD 上的 KV 能跨重启保留并复用 [VL10]。代价是 fs 层没有淘汰机制：`SecondaryTierManager` 的接口约定由层自身负责“按需淘汰”，fs 实现却没有这部分逻辑，长期运行必须依赖外部的容量监控和清理 [VL10][VL9]。此外，二级层目前不支持 worker 跨多节点的引擎 [VL3]。

### 从挂载点原型到受控写回

第一步保留原生 fs 层，固定块布局，只切换 SSD 和 HDD 挂载点。第二步再改变每个 chunk 的块数，记录减少文件操作换来的收益，以及短前缀命中时多读的字节数。如果压力来自写入，先限制待写队列并观察丢写后的重算成本。只有这些手段仍无法满足尾延迟目标时，再评估合并小文件、维护偏移索引和独立容量回收的工程成本。

## 本章资料

- **[LMC24]** [LMCache 官方文档 · FileSystem L2 adapter（采集于 2026-11-18）](https://docs.lmcache.ai/mp/l2_storage/fs.html)
- **[VL3]** [vLLM 官方文档 · KV Offloading Usage Guide（OffloadingConnector、TieringOffloadingSpec、fs / obj / p2p 二级层、On-Disk Layout、max_load_tokens，采集于 2026-11-18）](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- **[VL5]** [vLLM 官方博客 · Inside vLLM's New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput（Or Ozeri、Danny Harnik，IBM Research），2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
- **[VL9]** [vLLM 固定提交源码 · vllm/v1/kv_offload/tiering/fs/manager.py 与 io.py（FileSystemTierManager、batch_store_block / batch_load_block、probe_o_direct）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/fs/manager.py)
- **[VL10]** [vLLM 固定提交源码 · vllm/v1/kv_offload/tiering/manager.py（TieringOffloadingManager：级联、promotion、lookup、reset_cache）与 kv_connector/v1/offloading/scheduler.py](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/manager.py)
- **[VL11]** [vLLM 固定提交源码 · vllm/v1/kv_offload/tiering/backpressure.py（EMABackpressureDetector、DropStorePolicy、ThrottledDropPolicy 及默认水位）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/backpressure.py)
- **[VL12]** [vLLM 固定提交源码 · vllm/v1/simple_kv_offload/disk_backend.py 与 kv_connector/v1/simple_cpu_offload_connector.py（disk 模式）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/simple_kv_offload/disk_backend.py)
- **[VL13]** [vLLM 固定提交源码 · vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py（ExampleConnector，safetensors 按层落盘示例）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py)
- **[VL14]** [vLLM 固定提交源码 · vllm/config/cache.py（DEFAULT_BLOCK_SIZE、prefix_match_unit、kv_offloading_size / kv_offloading_backend）、config/vllm.py（_post_init_kv_transfer_config）、config/kv_transfer.py（kv_load_failure_policy）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/config/cache.py)
- **[VL15]** [LMCache 官方文档 · Legacy (In-Process Mode) / Local storage（local_disk、max_local_disk_size、LocalDiskBackend 架构，采集于 2026-11-18）](https://docs.lmcache.ai/kv_cache/storage_backends/local_storage.html)
- **[VL16]** [LMCache 官方文档 · Legacy (In-Process Mode) / GDS Backend（gds_path、disk_io_threads，采集于 2026-11-18）](https://docs.lmcache.ai/kv_cache/storage_backends/gds.html)
- **[VL18]** [WD × Solidigm · KV Cache Offload Across Storage Tiers，September 2026 / Revision 02](https://documents.westerndigital.com/content/dam/doc-library/en_us/assets/public/western-digital/collateral/tech-brief/technical-brief-kv-cache-offload-across-storage-tiers-wd.pdf)
