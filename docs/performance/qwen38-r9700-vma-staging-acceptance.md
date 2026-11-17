# 方向4：失效VMA定位与固定暂存区验收

日期：2026-11-06。方向2继续冻结；原11文件、FP8候选2文件、模型及expert选择/fetch策略均冻结。本轮只定位内存失效并验收暂存区，不改驱动、模块参数、sysctl、NUMA/THP全局设置，也不启用内核追踪。

证据前提见[独立HIP复现](qwen38-r9700-rocm-vs-freetoken.md)：合法register与hostmalloc均复现本进程KFD驱逐型停顿，但干净控制与全机规整活动停止同时出现，不能认定已修复。

## 顺序与门槛

1. **定位准备**：给独立HIP程序的288块bank、握手区、9MiB固定暂存区建立地址清单；只记录用户态KFD注册/注销及madvise/munmap/mprotect入口，不改变调用参数、返回结果或已安装库。通过私有KFD事件读取者与只读pagemap/kpageflags采样，对齐本PID队列暂停、VMA寿命与物理页变化。
2. **失效归属**：需要自然复现，PFN变化覆盖相应VMA且与驱逐时间相符；严格区分VMA仍有效但物理页变化、页面暂时缺失，以及主动释放。事件不带地址时，时间接近只能作为相关证据，不能凭一个PFN变化就把它当成唯一的MMU回调地址。
3. **暂存区控制**：相同63.63GiB CPU bank、同样50组握手与CPU活动，比较全量GPU可见bank与固定9MiB暂存区。交错独立启动；必须记录自然内存规整活动，静默阶段通过不算故障条件验收。
4. **整模型采用**：只有控制成立才进入FreeToken的固定暂存池适配与整模型正反向A/B。输出/路由、驱逐计数、最长token间隔及吞吐均须验收；独立HIP的float检查不能替代模型精度。

## 只读范围与观测边界

- 只读取本实验进程的`/proc/PID/pagemap`、`maps`、`numa_maps`和线程统计，以及`/proc/kpageflags`、全机vmstat与KFD主机计数。root仅用于这些读取和现有SMI事件订阅；订阅掩码只属于读取者自身。
- 未清除soft-dirty位、未写clear_refs，未强制迁移、回收、丢页或改变页面保护；不人为制造全机内存压力。
- 用户态拦截仅覆盖动态符号入口；不把“没有日志”视为所有库内部及直接syscall都没发生操作的证明。
- PFN扫描有时间成本，按每段扫描前后时间给出区间，先量化观察者开销；观察运行不用于正式吞吐。运行时附加范围按当前RW VMA与已记录注册区求交，覆盖上限和后续新增分配盲区明确记录。
- PCI83当前按共享定性诊断使用，不停止外部进程。ROCm10.1/HIP7.16锁定镜像与foundation报告保存在实验目录。

## 逐步结果

实验目录：`.freetoken-zihao/rdna4-fp8-vma-staging-20261106`。

### 第1步：准备及预检

已完成只读前置配置快照、共享GPU预约检查和ROCm10.1 foundation验证。独立程序保留原有合法内存寿命、Graph和握手，仅增加CPU侧地址清单及诊断启动门；未修改FreeToken代码。

全量扫描约1670万页需约1.05秒，初版会连续占用观察者CPU，因此已停止重复全量扫描，首轮只用于方法校准。新版先建立全量基线，再每16MiB采16个连续页及区间尾页；逐轮约38ms，每约750ms执行一次，范围包括显式bank/握手/暂存区与实际注册的运行时VMA。稀疏覆盖不能排除未采样页面变化。发现候选变化后保存该VMA完整pagemap快照，与初始逐页基线比对，原始PFN仅留在仓库外实验目录。

初版API记录按完整ioctl编码匹配，漏掉了部分变长SVM请求；已按类型/编号匹配并核对固定ABI，实际捕获到800MiB bank的SVM注册。初版的运行时范围覆盖不作为完整证据；显式bank地址清单不受该缺口影响。

### 第2步：自然复现与VMA归属

- 校准轮`diag-register-a1`：180秒、1782个Graph，数值/89600次握手通过；最大1.126ms，运行期驱逐0。全量采样停止后的部分时段无PFN观察，不作为无失效验收。
- 稀疏轮`diag-register-a2-sparse`：180秒、1758个Graph，逐token变化的非零bank数据及88400次握手通过；最大1.171ms，驱逐0，采样页未见PFN/驻留变化，后台规整未活动。故尚不能锁定失效VMA。
- `diag-hostmalloc-a3-sparse`使用同样的非零动态数据和180秒时长，保留完整初始PFN供候选细查，结果见下一节。

冻结整模型调用链的CPU地址/API/PFN诊断启动器已经准备，**本轮未运行整模型**。完整规模所需128GiB可用RAM门槛未满足，未降低门槛；以下结果均明确区分独立HIP控制与仓库已有小型功能测试。

### 第2步新增：已定位参与失效的具体VMA

`diag-hostmalloc-a3-sparse`自然复现：180秒、1740个Graph，所有非零动态数据及87500次握手通过，计时阶段5次USERPTR驱逐、累计1899ms。采样捕获到`bank-L7-gate_up`的16个物理页改变；针对该bank的完整前后快照确认128页、0.5MiB的PFN改变。

- CPU bank范围：`0x790543600000–0x790575600000`，800MiB，模拟模型第7层gate/up packed bank。
- KFD分配记录的CPU地址和800MiB长度完全对应，flags包含USERPTR。
- 基线及5次驱逐快照中该VMA均保持相同的`rw-s /dev/zero (deleted)`及inode，没有被应用注销/释放；未发现计时阶段覆盖该bank的munmap/madvise/mprotect操作。
- 变动页的旧、新PFN都属于node7，证明是节点内物理页移动，并非把这些样本从node7搬到其他NUMA节点。对应窗口有后台规整扫描增长；仍不把它称为唯一发起者或把所有5次暂停归于这一块bank。

这确认该bank是参与失效的VMA之一，进一步落实了“合法存活的GPU可见主机内存也会失效”。独立程序的层号用于匹配模型几何维度，不冒充原整模型进程中的同一虚拟地址。[逐页与注册证据](../../../.freetoken-zihao/rdna4-fp8-vma-staging-20261106/runs/diag-hostmalloc-a3-sparse/localized-bank-evidence.json)、[PFN所属节点与同时段vmstat](../../../.freetoken-zihao/rdna4-fp8-vma-staging-20261106/runs/diag-hostmalloc-a3-sparse/movement-node-context.json)

**观测影响修正**：这轮逐事件读取numa_maps、变化后立即全量读取pagemap，会增加CPU和映射锁开销，可能放大恢复耗时。638ms窗口只能作为诊断运行的观察值，不能拿来计算生产收益。后续改为先抓轻量线程栈，计时期间仅稀疏采样，完整变化快照移到GPU同步后的阶段外。

还发现实际放置不同：register对照的bank为`interleave:0-7`，hostmalloc对照为`prefer=static:7`，整块800MiB初始均在node7，PCI83也位于node7。不能把两组差异只归因于API名字，也不能按全机MemAvailable就排除局部节点压力。

### 第3步：保留同一CPU bank的交错对照

63.63GiB完整规模的A/B/A/B程序已准备，但启动前MemAvailable降至约100GiB，未达到128GiB门槛，未执行该规模。旧实验进程与KFD条目均已退出；新启动的其他任务占用了大量RAM，没有停止它们。

因此先验收单个已定位尺寸的800MiB bank，容器上限4GiB，完整规模门槛保留。该归因对照在首次触页前，使用进程内mbind复制参考HIP分配器的`prefer=static:7`放置，flags=0，不迁移现有页面、不改全局策略；生产NUMA配置不因此改变。A/B各阶段保留同一块mmap及数据，CPU准备与32KiB暂存复制工作相同；A注册大bank并直接读它，B注销大bank后只让GPU读固定9MiB暂存区。逐阶段查询确认HIP指针类型确实在Host与Unregistered之间切换。

| 阶段（各90秒） | Graph数 | 最大回读 ms | 累计驱逐增量 ms | 数值/握手 |
|---|---:|---:|---:|---|
| A1 direct | 891 | 35.007 | 89 | 通过 |
| B1 stage | 890 | 55.000 | 61 | 通过 |
| A2 direct | 889 | 202.289 | 214 | 通过 |
| B2 stage | 891 | 1.078 | 0 | 通过 |

四阶段都有后台规整扫描，分别增加4525492、3446983、5364224、6001306页，不再是此前“干净组正好没有规整”的对照。全部3561个Graph及180050次握手正确；阶段外注销/注册和完整快照不计入表中。

**仅取消大bank的GPU映射仍不足以通过稳定性验收**：B1还有驱逐，且B1最大延迟没有优于A1。不能用B2一个干净阶段认定修复。[阶段记录](../../../.freetoken-zihao/rdna4-fp8-vma-staging-20261106/runs/single-vma-phases-abab/phase-results.json)

### 第3步新增：隐式注册的CPU拷贝缓冲

A1还观察到约1MiB的运行时SVM区域发生PFN变化：`0x5edefe4b3000`起、1052672字节。它不是800MiB bank。只读目标实验进程内存，与A2完整262144-float验算数组逐字节匹配，偏移3888字节，对应数据指针`104311842094896`；指向程序用于H2D初始化及完整D2H验算的pageable CPU数组。

这提供第二个必须覆盖的来源：**HIP拷贝可隐式注册普通CPU缓冲，专家bank改用暂存区并不自动清除其他GPU可见主机区。** 完整数据匹配及注册记录保留，补测直接记录输入指针，并与预分配HIP host缓冲比较；不把它直接归属到FreeToken某一调用。[缓冲区归属证据](../../../.freetoken-zihao/rdna4-fp8-vma-staging-20261106/runs/single-vma-phases-abab/runtime-buffer-attribution.json)

### 第3步功能门槛：现有FreeToken暂存链

已在同一ROCm10.1/HIP7.16、PCI83上运行仓库现有`test_locked_hybrid_layer_fetches_through_staging_in_a_graph`，memops与GPU自旋等待两种模式均通过（2 passed）。测试覆盖真实LockedFetch原生CPU线程、GPU gather、唯一expert ID路由、18轮Graph重放中的源/目标数据一致性及cache rebuild后重新捕获。

首轮因遗漏TVM_FFI_CACHE_DIR而停在不可写的默认缓存目录，未进入待验收kernel；补齐既有专用缓存目录后重跑通过，没有改HOME、仓库代码、驱动或全局配置。这是暂存搬运/Graph功能证据，不能代替完整模型logits、路由、吞吐和长停顿验收。

### 第3步补充：固定小输入/验算缓冲

`single-vma-fixed-io-ab`将原1MiB pageable数组改为构造时分配的1MiB HIP host数组，其他bank、9MiB暂存区、节点放置、数据与50组握手不变。两个阶段各90秒：

| 阶段 | Graph数 | 最大回读 ms | 累计驱逐增量 ms | 后台规整扫描增量（页） |
|---|---:|---:|---:|---:|
| A1 direct | 891 | 1.090 | 0 | 5649817 |
| B1 stage | 891 | 1.738 | 0 | 1928974 |

全部1782个Graph及90100次握手通过，采样范围未见PFN/驻留变化。原来约1MiB的隐式SVM注册请求消失；仍有约29MiB的运行时SVM请求，不能声称GPU可见范围只剩专家暂存区。**直接读取组也未复现，不能把这轮零驱逐唯一归因于固定暂存区或输入缓冲。** [分阶段证据](../../../.freetoken-zihao/rdna4-fp8-vma-staging-20261106/runs/single-vma-fixed-io-ab/phase-analysis.json)

随后按顺序独立重启，均保留800MiB CPU bank及9MiB专家暂存区，每组90秒：

| 重启配置 | Graph数 | 最大回读 ms | 累计驱逐增量 ms | 后台规整扫描增量（页） |
|---|---:|---:|---:|---:|
| 反向：旧pageable输入 | 891 | 1.166 | 0 | 1854290 |
| 复核：固定HIP host输入 | 891 | 1.560 | 0 | 2457471 |

两组数值及各45050次握手通过。反向组直接记录的输入指针`106865536212032`被1052672字节SVM SET范围覆盖，确认隐式注册确实属于这个数组；固定输入组对应请求消失。但是旧配置也未复现，**反向稳定性门槛未通过，不能把“固定输入 + 专家暂存”标成已修复**。后台扫描总量只说明存在活动，不能保证相同页受到相同规整/保护操作。[反向注册证据](../../../.freetoken-zihao/rdna4-fp8-vma-staging-20261106/runs/pageable-stage-reverse/pageable-input-registration.json)、[全部交错与重启结果](../../../.freetoken-zihao/rdna4-fp8-vma-staging-20261106/phase-results.json)

### 第4步：整模型采用门槛与本轮结论

| 门槛 | 结论 |
|---|---|
| 独立复现的具体VMA/PFN归属 | 已取得专家bank及运行时数组的证据；未取得唯一MMU notifier回调地址/发起者 |
| 固定暂存链搬运、握手与Graph功能 | 独立控制通过；FreeToken现有原生测试2项通过 |
| 固定小池稳定性修复 | **未通过**：单独专家暂存仍会驱逐；固定输入的干净结果未被反向故障对照确认 |
| 63.63GiB全规模控制与整模型精度/性能 | **待验收**：可用RAM未达128GiB门槛，本轮未运行；未宣称模型tok/s收益或ready |

本轮7次独立HIP启动共12405个计时Graph、625750次握手，数值和握手检查均通过；这些是float与通信检查，不是NVFP4/FP8 MoE精度验收。最后可用RAM约118.9GiB，未降低128GiB门槛。FP8候选保持默认关闭，方向2不解冻。

继续方向4时，采用对象应为**完整主机传输生命周期的固定小池**，而不是仅将decode的GPU源指针换成暂存区。已有`LockedLayerFetch`可作为decode搬运基础，不必另写握手协议；`OffloadMoeCache.copy_missing`的unpinned prefill分支仍通过`view.copy_(per_layer[layer_id])`整层pageable H2D，可能再次引入大范围隐式注册，需与加载/初始化拷贝一起核验。运行时仍存在约29MiB SVM区域，也需要保留覆盖；HIP指针变为Unregistered不能替代完整驱动映射生命周期证据。

下一次只补全规模的同CPU bank交错/独立重启与整模型采用门槛：保持expert ID顺序、CPU/GPU分工、max-fetch1和cache容量不变，先确认从加载、prefill到decode不再隐式注册大bank，再比较输出/路由、驱逐、最长token间隔及吞吐。若旧配置未在条件可比窗口自然复现，继续记为稳定性证据不足，不按零故障时间推算修复收益。

全部自有程序、KFD条目与私有读取会话已退出，GPU4预约已释放。原11文件和候选2文件SHA256不变，驱动及所记录全局参数前后相同；未改底层、未提交或推送。只新增/更新验收文档及仓库外诊断文件。[结束与冻结校验](../../../.freetoken-zihao/rdna4-fp8-vma-staging-20261106/completion.json)
