# vLLM：整体架构与 KV 缓存管理

> 推理引擎 · vLLM 官方文档与固定提交源码核查

### 整体架构：谁接收请求，谁调度，谁执行

以 `vllm serve` 部署的 V1 在线文本服务为例，一条请求跨过三个边界：API Server 把 HTTP 输入转换成引擎请求，Engine Core 决定本轮为哪些序列计算多少 token，GPU Worker 按计划执行模型。返回方向上，引擎把各请求的新 token 交回前端，由前端解码并向客户端流式输出。下图是单副本、四卡张量并行的实例，四张卡共同执行同一模型副本。 [VL19][VL20]

![vLLM 官方 V1 进程图（TP=4、DP=1）。一个 API Server 对接一个 Engine Core；同一轮调度驱动四个 GPU Worker。](../figures/vllm-1.jpg)

*vLLM 官方 V1 进程图（TP=4、DP=1）。一个 API Server 对接一个 Engine Core；同一轮调度驱动四个 GPU Worker。* — [原始来源](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/design/arch_overview.md)

| 组件 | 接收什么，决定什么 | 与下游的接口 |
|---|---|---|
| API Server / AsyncLLM | 接收提示词与采样参数，处理分词、多模态输入及请求标识，维护客户端输出流。 | 通过 ZMQ 把请求交给 Engine Core；根据请求标识将增量结果返回相应连接。 |
| Engine Core / Scheduler | 每个数据并行副本有一个核心，持有等待与运行队列。依据 token 预算、已计算长度和可用 KV 块组批。 | 把各请求本步的 token 数、块分配等调度结果交给模型执行器；根据执行结果更新进度。 |
| Executor / GPU Worker | 执行器组织同一副本的多个 Worker。Worker 持有相应模型权重和设备上的 KV 张量，执行前向计算与采样流程。 | 返回模型执行结果。张量并行时多个 rank 协作完成一次前向，各自保存相应的权重与 KV 分片。 |
| DP Coordinator（多副本部署） | 在多个数据并行 rank 之间协调负载，以及 MoE 模型所需的同步前向。 | 每个副本仍独立组批。增加 DP 副本扩大可接收的并发，增加 TP 则改变单个副本的执行拓扑。 |

例如 TP=2、DP=4、PP=1 时，八张 GPU 组成四个双卡副本，对应四个 Engine Core。复用某个副本的本地前缀要求请求到达持有该前缀的副本；如需跨副本或跨存储层取回 KV，还需要相应的传输组件。 [VL19][VL4]

### 硬件对接：模型并行、跨实例 KV 与文件卸载各走哪条路

| 谁发起 | 数据经过哪里 | 如何核实 |
|---|---|---|
| 一个副本内的多个 GPU Worker | TP 将一层的运算分给多个 rank，并交换中间结果。节点内可用 NVLink / NVSwitch，或经 PCIe；跨机时，集合通信库根据拓扑选用网络传输。PP 则在连续层段之间交换激活。DP 副本各有自己的调度与 KV 状态。 | 先核对各 rank 与物理卡的映射，再查看 NCCL 日志实际采用的链路。同样的 TP 数，跨节点的同步等待可能让逐 token 延迟明显不同。 [VL19][VL21] |
| P/D 实例间的 NixlConnector | 常规拉取模式中，P 完成 prefill 并保留 KV，D 根据交换的块与传输元数据从 P 读取，再继续生成。侧通道负责握手，批量数据由 NIXL 后端传送。默认 UCX 可以按设备和平台选用不同传输。 | 对照 D 端传输完成事件、P 端缓冲区保留时间、UCX 使用的网卡和 NIC 字节数。NCCL 的网卡配置只说明集合通信的配置；NIXL/UCX 有自己的设备选择。 [VL22][VL7][VL8] |
| OffloadingConnector + fs 层 | GPU 块和 CPU 主层之间有设备搬运，文件二级层从主机缓冲区按 chunk 读写文件。目录落在本地 NVMe、HDD 或共享挂载点，会得到不同的物理路径。普通 fs 后端的直接 I/O 选项涉及页缓存；GPU 直读存储另有驱动和 API 要求。 | 同时核查模型重算量、CPU 主层命中、文件读入字节和设备读队列。仅从文件系统看到吞吐，无法判断 GPU 等待了多久。 [VL3][VL9][VL10] |

### 请求生命周期：长提示词如何与正在生成的请求同批运行

1. **入队。**前端把编码后的请求送入引擎。调度器观察请求尚需计算的 token 数；前缀命中可缩短待处理区间。
2. **组批。**假设请求 A 有较长提示词，而请求 B 正在生成下一个 token。V1 在同一轮的 token 预算内为运行和等待请求分配工作。启用分块 prefill 后，A 的未命中后缀可分多轮处理，B 也可在其间继续生成。若 KV 槽位不足，调度策略可抢占请求、释放块，稍后重新入队。
3. **执行与反馈。**`EngineCore.step()` 先调用 `scheduler.schedule()`，再由执行器运行本轮模型计算，按需采样；`scheduler.update_from_output()` 归并结果。异步执行可以重叠相邻轮次的调度与计算。
4. **返回或继续。**新生成的 token 经前端输出。尚未结束的请求进入后续轮次，结束后释放对块的占用；已索引的完整前缀块仍可能被后续请求复用，直到被回收。

对存储的影响集中在组批之前：本地命中可以直接跳过相应的 prefill；远端命中还要等数据抵达可供 Worker 使用的槽位。评价慢层收益时，应比较取回 KV 新增的等待与省下的 prefill 时间。 [VL19][VL20][VL8][VL2]

vLLM 的 KV 管理可以拆成两层来看。第一层在 GPU 内部：调度器通过 `KVCacheManager` 和 `BlockPool` 管理分页 KV 块，用块哈希做自动前缀缓存（APC），空闲块按 LRU 回收。第二层在 GPU 之外：所有跨进程、跨层级的 KV 搬运都要经过 KV Connector API（`KVConnectorBase_V1`），CPU 卸载、LMCache、NIXL 的 P/D 传输和文件系统下沉都挂在这个接口上。本章依据 vLLM GitHub main 分支固定提交 `0c80c3a6aa0201d1b4f71915f90c03fd64837cec`（提交时间 2026-11-18 08:26 UTC）阅读源码，并对照 docs.vllm.ai、vLLM 官方博客和 LMCache 官方文档，采集于 2026-11-18 [VL1][VL2][VL3]。

### GPU 内的前缀缓存：块哈希、BlockPool 与 LRU 空闲队列

vLLM V1 的 APC 采用哈希方案：每个 KV 块的键由**父块哈希、本块 token、额外键**共同算出。额外键包括 LoRA ID、多模态输入哈希和 `cache_salt`。父块哈希参与计算，所以一个块哈希隐含了它前面的整条前缀。只有写满的块才会被缓存。自 v0.11 起默认哈希算法为 `sha256`，`--prefix-caching-hash-algo` 还可以选 `sha256_cbor`、`xxhash`、`xxhash_cbor`。官方文档提醒，非密码学哈希在多租户环境下会增加碰撞和信息泄露风险。`cache_salt` 只注入第一个块的哈希，用来把缓存复用限制在同一信任组内 [VL2]。源码中由 `hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys)` 计算单块哈希。链首的 `NONE_HASH` 默认由固定种子导出；选择 xxhash 类算法且未设置 `PYTHONHASHSEED` 时，种子改为每个进程随机生成，不同实例算出的哈希就对不上。后文的文件系统层正是靠这一点实现跨实例共享 [VL6][VL3]。

数据结构方面，`BlockPool` 在初始化时一次性创建 `num_gpu_blocks` 个 `KVCacheBlock`。每个块记录 `block_id`、`ref_cnt`、`block_hash`，以及 `prev_free_block` / `next_free_block` 两个指针。空闲队列 `FreeKVCacheBlockQueue` 直接用这两个指针串成双向链表：从中间摘除、在头尾插入都是 O(1)，也不用另建 Python 对象。哈希表 `cached_block_hash_to_block`（类型为 `BlockHashToBlockMap`）把带 KV 组号的哈希映射到具体的块 [VL2][VL6]。

![vLLM 官方设计文档 Automatic Prefix Caching 的组件总览图：Block Pool（KVCacheBlock 列表）、Free Block Queue（只保存头尾指针的双向链表）、Cache Blocks（哈希→块）与 Request Blocks（请求→块）](../figures/vllm-2.png)

*vLLM 官方设计文档 Automatic Prefix Caching 的组件总览图：Block Pool（KVCacheBlock 列表）、Free Block Queue（只保存头尾指针的双向链表）、Cache Blocks（哈希→块）与 Request Blocks（请求→块）* — [原始来源](https://docs.vllm.ai/en/latest/design/prefix_caching/)

控制流分三步，见下图。（1）新请求到来时，调度器先调用 `KVCacheManager.get_computed_blocks()`，经 `KVCacheCoordinator.find_longest_cache_hit()` 沿 `request.block_hashes` 逐块查哈希表，得到最长的全命中前缀。如果整个 prompt 都命中，最后一个 token 仍要重算以产生 logits。（2）`allocate_slots()` 先对命中块执行 `BlockPool.touch()`：`ref_cnt` 加 1，原来为 0 的块从空闲队列中摘除，避免被驱逐。随后 `get_new_blocks()` 用 `popleft_n()` 从队首取新块；取到的块如果还带着哈希，就在此刻经 `_maybe_evict_cached_block()` 从哈希表中删除，这就是“驱逐”。新写满的块由 `cache_full_blocks()` 登记进哈希表。（3）请求结束或被抢占时，`SingleTypeKVCacheManager.free()` 以**逆序**调用 `BlockPool.free_blocks()`。`ref_cnt` 归零后，没有哈希的块用 `prepend_n()` 放回队首优先复用；带哈希的块用 `append_n()` 放到队尾。因为是逆序放回，请求最末尾的块（覆盖的前缀最长、最难被复用）排在前面，最先被淘汰 [VL2][VL6]。V1 的块表只允许追加，同一内容可能短暂存在重复块，等请求释放后才消除 [VL2]。

![vLLM V1 GPU 前缀缓存的数据结构与控制流：请求哈希 → get_computed_blocks / allocate_slots / free → BlockPool 哈希表与 LRU 空闲队列（本文根据 vLLM 固定提交 0c80c3a6 源码 block_pool.py、kv_cache_utils.py、kv_cache_manager.py 绘制）](../figures/vllm-3.svg)

*vLLM V1 GPU 前缀缓存的数据结构与控制流：请求哈希 → get_computed_blocks / allocate_slots / free → BlockPool 哈希表与 LRU 空闲队列（本文根据 vLLM 固定提交 0c80c3a6 源码 block_pool.py、kv_cache_utils.py、kv_cache_manager.py 绘制）* — [原始来源](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/core/block_pool.py)

块大小方面，固定提交中 `CacheConfig.DEFAULT_BLOCK_SIZE = 64`。用户没有指定 `--block-size` 时，平台层会在各注意力后端都支持的取值里另选一个首选值。另有 `prefix_match_unit`（即 `hash_block_size`），可以让前缀命中落在比物理块更细的边界上 [VL14]。vLLM 博客在 2026-01-08 写的是“默认每块 16 token”，说明默认值在此后已经变化，做容量估算时应以实际运行日志为准 [VL5]。

### 内存系统管理：显存预算、块所有权与跨设备借用

vLLM 在三个时刻回答不同的问题：启动时给 KV 留多少 HBM；一次请求结束后，其块能否被读、被写或被重新分配；跨实例读取尚未完成时，源端要把显存保留到什么时候。只看 GPU 上的已分配字节，无法区分这些状态。

#### 启动时如何划出 KV 池

V1 GPU Worker 的自动定容路径先取得设备内存快照，通过模型执行器做一次 profile，计入模型权重以及非 KV 的峰值消耗；启用相应估算时，还要扣掉 CUDA Graph 的内存。所得余额才作为 KV 容量返回。实现对 profile 期间临时张量的分配也做了处理：MoE 工作区逐级扩张时，小分配可能占住已释放的大段，使整段显存无法退回并被误算为常驻开销；源码临时限制分配器拆分大块，测量结束后恢复设置。显式指定 KV 字节数则走单独分支。由此，同型号 GPU 装载同一模型，执行图、工作区和启动配置不同，也可能得到不同的池大小。 [VL23]

#### 释放、可命中、可覆盖是三个不同的时刻

| BlockPool 中的状态 | 下一次可做的操作 | 源码里的约束 |
|---|---|---|
| 活动请求独占的新块 | 继续写入本请求的 KV。 | 可写判定要求块有效、引用计数为一、尚无已发布的块哈希。独占一个已发布的前缀仍不足以原地改写。 |
| 完整前缀最后一个引用释放 | 块进入空闲队列；复用相同前缀仍可能命中原内容。 | 命中时 touch 将零引用块从空闲队列摘除并增加引用。这里的“空闲”表示可以重新分配，统计上可能与“仍可命中”重叠。 |
| 分配器把旧 ID 交给新请求 | 从空闲队列取块，撤销旧缓存映射，再取得本次引用。 | 对于已解除固定且登记了复用回调的块，重新分配前先通知原读者。保存一个 block ID 本身不能延长其内容的有效期。 |

这三个状态直接来自固定提交的 BlockPool。它们解释了为什么监控“前缀命中率”与“还能分配多少块”必须使用两套计数。 [VL24]

#### 逻辑页大小不变，物理搬运粒度为何会变化

vLLM 团队公开对比了 0.11.0 与 0.12.0 的卸载布局。在其 Llama-3.1-8B-Instruct 配置中，同为 16-token 逻辑块，旧布局按层分散，表中单次连续片段为 32 KB；跨层合并布局对应 2 MB 的连续物理块。同一个前缀从主机 DRAM 回填时，拷贝引擎处理的片段数量因此改变。团队还对比了设备 DMA 和占用 SM 的拷贝内核：该 8B 模型下，DMA 在 TTFT 相当时，吞吐最高比自定义内核高 32%。实验条件为单张 H100 80 GB HBM3、500 GB 主机 DRAM、Xeon Sapphire Rapids（容器限 8 核）、FlashAttention，关闭 GPU 前缀缓存以隔离 CPU 命中；这些数字属于 CPU–GPU 实验。 [VL5]

#### 远端读者如何影响本地显存的回收

NIXL 的 P/D 拉取路径中，Prefill 完成后 P 仍持有源 GPU 块，因为 D 可能还在排队。官方设计将所有权延伸到传输完成：P 按请求授予租约，D 从进入调度器时就开始跟踪并续约；收到读完成通知，P 可以立即释放；D 崩溃并停止续约，P 才在租约到期后回收。文档默认初始租期 30 秒，心跳随前向循环处理。这样既避免 D 排队时被提前覆盖，也限制故障请求长期占住 HBM。反向保留 Decode 块等待下一轮对话采用另一套固定 TTL，不能套用这项按请求续约的语义。 [VL25]

### KV Connector API：调度侧与 Worker 侧

同一个 connector 类会被实例化两次：调度进程里一份（`KVConnectorRole.SCHEDULER`），每个 GPU worker 里各一份（`KVConnectorRole.WORKER`）。两边通过 `SchedulerOutput.kv_connector_metadata` 和回传的 `KVConnectorOutput` 交换信息 [VL4][VL7]。调度侧的核心钩子是 `get_num_new_matched_tokens(request, num_computed_tokens)`。它返回两个值：外部缓存还能在本地命中之外补上多少 token，以及是否异步加载。文档要求这个方法可以被重复调用、没有副作用，而且只能把当下确实可加载的前缀算进去。返回 `None` 表示 connector 还没查完，调度器会对该请求执行 `skip_request()`，下一步再问 [VL7][VL8]。之后的顺序是：`allocate_slots(..., num_external_computed_tokens)` 分配块，`update_state_after_alloc()` 记录本次加载计划；如果是异步加载，请求状态置为 `WAITING_FOR_REMOTE_KVS`；最后由 `build_connector_meta()` 打包本步的 load/save 元数据。Worker 侧依次执行 `bind_connector_metadata()`、`handle_preemptions()`、`start_load_kv()`，并在逐层前向中调用 `wait_for_layer_load()` 和 `save_kv_layer()`；前向结束后，`wait_for_save()`、`get_finished()` / `get_transfer_results()`、`get_block_ids_with_load_errors()` 把完成情况和失败的块回报给调度器 [VL7]。

![vLLM 官方文档 Disaggregated Prefilling 一节的 connector 高层设计图：调度进程中的 connector 通过 SchedulerOutput 的 connector_metadata 字段驱动各 worker 进程中的 connector，再经数据传输层与其他实例交换 KV](../figures/vllm-4.png)

*vLLM 官方文档 Disaggregated Prefilling 一节的 connector 高层设计图：调度进程中的 connector 通过 SchedulerOutput 的 connector_metadata 字段驱动各 worker 进程中的 connector，再经数据传输层与其他实例交换 KV* — [原始来源](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

失败处理方面，`KVTransferConfig.kv_load_failure_policy` 的默认值是 `"fail"`，即加载失败的请求直接以错误结束。设为 `"recompute"` 后，调度器在 `_handle_invalid_blocks()` 中把受影响的请求重新排队、重算失败的块 [VL14][VL8]。块级的 `invalid_block_ids` 只适用于单 KV 组模型；有多个 KV 组时，connector 必须改用请求级的 `failed_recving` 上报，否则会直接抛错 [VL8]。请求结束时调度器调用 `request_finished(request, block_ids)`。如果它返回 `True`，表示 connector 还在异步发送或落盘，GPU 块要等 `finished_sending` 回报后才会释放。异步保存不阻塞推理，代价是 GPU 块被占用得更久 [VL7][VL8]。

![KVConnectorBase_V1 在一次 engine step 中的调用顺序：调度侧 get_num_new_matched_tokens → allocate_slots / update_state_after_alloc → build_connector_meta，worker 侧 start_load_kv / save_kv_layer / wait_for_save / get_finished，回到调度侧处理完成、失败与 request_finished（本文根据 vLLM 固定提交 0c80c3a6 源码 kv_connector/v1/base.py 与 v1/core/sched/scheduler.py 绘制）](../figures/vllm-5.svg)

*KVConnectorBase_V1 在一次 engine step 中的调用顺序：调度侧 get_num_new_matched_tokens → allocate_slots / update_state_after_alloc → build_connector_meta，worker 侧 start_load_kv / save_kv_layer / wait_for_save / get_finished，回到调度侧处理完成、失败与 request_finished（本文根据 vLLM 固定提交 0c80c3a6 源码 kv_connector/v1/base.py 与 v1/core/sched/scheduler.py 绘制）* — [原始来源](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/distributed/kv_transfer/kv_connector/v1/base.py)

固定提交的 `KVConnectorFactory` 注册了以下 connector：`ExampleConnector`（固定提交中已无 SharedStorageConnector 这一注册名，同类示例改由它承担，参数仍叫 `shared_storage_path`）、`LMCacheConnectorV1`、`LMCacheMPConnector`、`NixlConnector`（`NixlPullConnector` 的别名）/ `NixlPushConnector`、`MultiConnector`、`OffloadingConnector`、`SimpleCPUOffloadConnector`、`MooncakeConnector`、`MooncakeStoreConnector`、`FlexKVConnectorV1`、`HF3FSKVConnector`、`MoRIIOConnector`、`UMBPStoreConnector`、`HiSparseConnector`、`DecodeBenchConnector` 等 [VL1]。它们统一通过 `--kv-transfer-config` 的 JSON 配置（`kv_connector`、`kv_role`、`kv_connector_extra_config`）。另有两个快捷参数：`--kv-offloading-size`（单位 GiB，TP>1 时为各 rank 合计）和 `--kv-offloading-backend`。后者取 `native` 时映射为 `OffloadingConnector`（设置环境变量 `VLLM_USE_SIMPLE_KV_OFFLOAD` 时改为 `SimpleCPUOffloadConnector`），并令 `cpu_bytes_to_use = size × 2^30`；取 `lmcache` 时映射为 `LMCacheMPConnector`，容量交给独立的 LMCache server 管理 [VL14]。`NixlConnector` 用 NIXL 库在 P/D 实例之间做全异步的 GPU KV 收发，默认传输后端是 UCX，也可以在 `backends` 中指定 GDS 等插件。它的定位是实例间传输，不是持久化的容量层 [VL17][VL4]。

## 本章资料

- **[VL1]** [vLLM 官方 GitHub 仓库 · main 分支固定提交 0c80c3a6aa0201d1b4f71915f90c03fd64837cec（2026-11-18；含 kv_connector/factory.py 注册表，采集于 2026-11-18）](https://github.com/vllm-project/vllm/tree/0c80c3a6aa0201d1b4f71915f90c03fd64837cec)
- **[VL2]** [vLLM 官方文档 · Design: Automatic Prefix Caching（块哈希、KVCacheBlock、Free Block Queue、分配 / 释放 / LRU 淘汰，采集于 2026-11-18）](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- **[VL3]** [vLLM 官方文档 · KV Offloading Usage Guide（OffloadingConnector、TieringOffloadingSpec、fs / obj / p2p 二级层、On-Disk Layout、max_load_tokens，采集于 2026-11-18）](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
- **[VL4]** [vLLM 官方文档 · Disaggregated Prefilling (experimental)（各 connector 用法与高层设计图，采集于 2026-11-18）](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- **[VL5]** [vLLM 官方博客 · Inside vLLM's New KV Offloading Connector: Smarter Memory Transfer for Maximizing Inference Throughput（Or Ozeri、Danny Harnik，IBM Research），2026-01-08](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
- **[VL6]** [vLLM 固定提交源码 · vllm/v1/core/block_pool.py（BlockPool、get_new_blocks、touch、free_blocks）与 kv_cache_utils.py（KVCacheBlock、FreeKVCacheBlockQueue、hash_block_tokens、NONE_HASH）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/core/block_pool.py)
- **[VL7]** [vLLM 固定提交源码 · vllm/distributed/kv_transfer/kv_connector/v1/base.py（KVConnectorBase_V1 调度侧 / worker 侧方法定义）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
- **[VL8]** [vLLM 固定提交源码 · vllm/v1/core/sched/scheduler.py（get_num_new_matched_tokens 调用、WAITING_FOR_REMOTE_KVS、_update_from_kv_xfer_finished、_handle_invalid_blocks）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/core/sched/scheduler.py)
- **[VL9]** [vLLM 固定提交源码 · vllm/v1/kv_offload/tiering/fs/manager.py 与 io.py（FileSystemTierManager、batch_store_block / batch_load_block、probe_o_direct）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/fs/manager.py)
- **[VL10]** [vLLM 固定提交源码 · vllm/v1/kv_offload/tiering/manager.py（TieringOffloadingManager：级联、promotion、lookup、reset_cache）与 kv_connector/v1/offloading/scheduler.py](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/kv_offload/tiering/manager.py)
- **[VL14]** [vLLM 固定提交源码 · vllm/config/cache.py（DEFAULT_BLOCK_SIZE、prefix_match_unit、kv_offloading_size / kv_offloading_backend）、config/vllm.py（_post_init_kv_transfer_config）、config/kv_transfer.py（kv_load_failure_policy）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/config/cache.py)
- **[VL17]** [vLLM 官方文档 · NixlConnector Usage Guide（NIXL 传输后端选择，采集于 2026-11-18）](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/)
- **[VL19]** [vLLM V1 · Architecture Overview：在线服务进程、并行拓扑与调度预算（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/design/arch_overview.md)
- **[VL20]** [vLLM V1 · EngineCore.step 与 step_with_batch_queue（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/engine/core.py)
- **[VL21]** [vLLM · 并行扩展、多节点通信与 GPUDirect RDMA 验证（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/serving/parallelism_scaling.md)
- **[VL22]** [vLLM · NixlConnector：侧通道、UCX、设备选择及 P/D 读取方向（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/features/nixl_connector_usage.md)
- **[VL23]** [vLLM V1 GPU Worker · determine_available_memory 与显存 profile（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/worker/gpu_worker.py#L716-L826)
- **[VL24]** [vLLM V1 BlockPool · 重新分配、解除固定、可写判定与释放（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/vllm/v1/core/block_pool.py#L681-L824)
- **[VL25]** [vLLM · NIXL KV Cache Lease：跨实例块固定、续约、完成与超时回收（固定提交）](https://github.com/vllm-project/vllm/blob/0c80c3a6aa0201d1b4f71915f90c03fd64837cec/docs/design/nixl_kv_cache_lease.md)
