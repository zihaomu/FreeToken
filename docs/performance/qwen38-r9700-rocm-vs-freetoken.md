# R9700 秒级停顿：ROCm 与 FreeToken 的责任边界

日期：2026-11-06。继续方向4的故障归因，方向2冻结。原模型、FP8候选和原11个文件冻结。本轮只审计源代码、读取现有系统状态，并在仓库外运行独立用户态HIP诊断；不改驱动、模块参数、sysctl、NUMA/THP全局设置或内核追踪配置。

## 目前判断

**已在完全不加载FreeToken/PyTorch的合法HIP程序中复现同类秒级停顿，并与它自己的KFD驱逐对齐。FreeToken不是触发这种故障机制的必要条件；当前证据指向ROCm/KFD/HMM与主机内存管理的交互，尚不能进一步定性为某个已确认的ROCm代码缺陷。** 未找到FreeToken提前释放GPU可见内存的直接证据。后续已取得一个800MiB专家bank及约1MiB拷贝缓冲的具体VMA/PFN变化证据，但尚无唯一MMU回调地址或发起者证明，见[失效VMA与暂存验收](qwen38-r9700-vma-staging-acceptance.md)。

此前两次无额外GPU标记的整模型启动共11个>=1秒回读等待，均被同PID驱逐窗口覆盖97.05%–99.38%，独立累计计数同时增长。这里“GPU队列暂停”已经由事件确认，“哪个地址先失效、为什么失效”仍未知。[前轮逐点证据](qwen38-r9700-stall-attribution.md)

## 1. 将长停顿分成驱逐与恢复阶段

重新分析前轮原始事件，而非仅比较总耗时：

| CPU回读等待 ms | 同PID驱逐窗口 ms | 首次恢复重试距驱逐 ms | 恢复重试数 |
|---:|---:|---:|---:|
| 3239.359 | 3209.675 | 5.584 | 120 |
| 2531.387 | 2500.279 | 13.648 | 22 |
| 4600.367 | 4571.635 | 41.395 | 24 |
| 4347.810 | 4318.350 | 21.145 | 23 |

恢复worker在驱逐后的数十毫秒内已经执行并报告重试，因此这些样本不能解释成初次驱逐先等待9秒抢占超时。队列随后持续暂停，存在秒级重试间隔及正常恢复前的长间隔；间隔本身不能区分worker调度、锁、页检查、页表更新或fence等待。

已安装驱动的`svm_range_restore_work`和`amdgpu_amdkfd_restore_userptr_worker`都在页检查失败或并发失效时重新排队；只有恢复成功才重启队列。事件`a/R`不能当成恢复完成。`queue_preemption_timeout_ms=9000`仅作为只读配置记录，未修改。

原始相位数据：[b1](../../../.freetoken-zihao/rdna4-fp8-stall-attribution-20261106/unmarked-gate-b1-restore-phases.json)、[b2](../../../.freetoken-zihao/rdna4-fp8-stall-attribution-20261106/unmarked-gate-b2-restore-phases.json)。部分驱逐窗口早于回读或晚于回读，故窗口总长与前轮“回读内覆盖长度”略有不同。

## 2. FreeToken 的内存寿命和地址审计

- `HostBank`默认使用匿名共享`mmap`，填充后执行portable+mapped注册；`_LIVE_BUFFERS`保留底层映射。静态调用链未发现decode中逐token注销或释放专家bank。
- `unpin()`仅见于启动期pin探测/回退；`release()`的bank调用仅见checkpoint converter，且注册bank直接返回。不能据源码里存在`MADV_DONTNEED`就判定decode主动丢页。
- GPU gather通过`device_ptr()`解析设备别名；CPU MoE信号区也取得设备别名，未发现这两条链直接把可能不同的主机地址当设备地址的明确错误。
- PLE gate路径逐token执行`.to("cpu").to(int64).tolist()`，有CPU临时分配，但它在读取前等待D2H完成。这是停顿的观察位置，尚无证据说明这处临时分配就是失效的VMA。
- PLE staging、CPU MoE IO及握手区均在初始化时分配并长期持有。静态审计不是运行期寿命证明，也没有覆盖PyTorch/HSA内部所有分配。

FreeToken将注册成功记为`PINNED`，并在`lock()`注释中假定注册已锁页。HIP7.16的API文档提醒部分系统上注册内存不一定实际锁页；其统一内存表又将register和hostmalloc都列为host/zero-copy。因此只能说应用的物理固定假设需要验证，不能直接把`hipHostRegister`称为错误API或把`hipHostMalloc`称为已证明的修复。[HIP注册语义](https://rocmdocs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html)、[统一内存分配表](https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html)

`smaps Locked=0`本身也不能证明没有驱动持页；需要结合实际后端及内核路径，避免把mlock统计等同于所有pin机制。

## 3. 独立 HIP 对照（已完成13组）

实验目录：`.freetoken-zihao/rdna4-fp8-rocm-vs-freetoken-20261106`。程序不加载FreeToken或PyTorch，只用同一ROCm10.1/HIP7.16应用镜像、gfx1201、同一PCI83，沿用进程自身NUMA interleave策略。使用2GiB稳定主机bank、约2MiB显式设备缓冲区、HIP Graph与4字节D2H；普通CPU活动只映射/填充/释放另一个未注册、从未给GPU使用的128MiB临时区。

所有GPU可见内存都在最终同步后才注销/释放；不强制页面迁移、回收或丢弃。增加一组纯HIP的50次CPU/GPU WRITE/WAIT握手对照，检查Graph握手在小规模下是否独立触发问题。

PCI83目前有约12.5GiB其他服务显存占用，故只做小占用的定性诊断，不启动整模型，不停止外部服务，不据此比较吞吐。当前主机可用内存显著多于此前失败运行，也不等同于整模型的大规模映射条件；小程序阴性不能排除ROCm在原条件下的问题。

基础6组已完成：纯设备Graph、2GiB register静态、register+CPU活动R1、hostmalloc+CPU活动M1/M2、register+CPU活动R2，均无同PID驱逐、累计驱逐计数0，最长回读2.041ms，全部输出检查通过。2GiB的50次握手控制也完成：register组297次Graph、15350次握手（含10次预热），最长4.066ms；hostmalloc组298次Graph、15400次握手，最长1.063ms。两组均无自有驱逐，计数和数值完全通过。

继续增加与模型一致的主机bank维度：48层×6个bank，共288块、68325212160字节（63.6328125GiB）；仍只有小型HIP计算/回读与50次握手，没有FreeToken、PyTorch、模型权重或专家计算。这只能缩小注册规模差异，并未还原实际CPU专家访存和共享主机内存压力。

命令、源代码与每组日志保留在实验目录；[持续更新的汇总](../../../.freetoken-zihao/rdna4-fp8-rocm-vs-freetoken-20261106/probe-results.json)。

### 大规模注册已独立复现

`modelbanks-register`（PID1605728）在45秒计时区间出现3次>100ms等待，累计驱逐计数增长2726ms；全部输出和21500次握手（含预热）检查通过。GPU可见bank从初始化到最终同步均保持有效，没有注销、释放或强制迁移。

| Graph序号 | 回读等待 ms | 同PID驱逐覆盖 ms | 覆盖比例 |
|---:|---:|---:|---:|
| 14 | 811.691 | 810.521 | 99.86% |
| 79 | 734.257 | 733.120 | 99.85% |
| 82 | 1091.267 | 1090.235 | 99.91% |

计时区间有3条SVM驱逐、3条USERPTR驱逐和19条恢复重试，原始事件与累计计数相互支持。启动阶段也发生过驱逐，表中的2726ms是进入计时后的增量，未混入初始化耗时。时钟对齐采用外部夹读校准，BOOTTIME与MONOTONIC偏差中位数10ns，最大半采样区间3150ns；没有把程序非同时打印的begin时钟差当成真实偏差。

这组实验提供了前轮欠缺的反事实：即使没有FreeToken的模型调度、Triton kernel、PLE磁盘读取、专家计算，也能发生同PID映射恢复型停顿。它不能证明历史9.7秒样本的具体地址，也不能排除FreeToken的映射规模放大了当前平台的恢复成本。

`modelbanks-hostmalloc`（PID1624841）也复现：45秒、424个Graph，3次>100ms，最长860.873ms，计时期累计驱逐计数增长2323ms；有3条USERPTR驱逐与16条恢复重试。21700次握手及数值检查通过。**所以换成`hipHostMalloc`仍会触发，不能把`FREETOKEN_BANK_CUDA_ALLOC=1`当成已验证的修复。**

### 去握手与小暂存区控制

| 控制 | CPU大bank | GPU可见主机数据 | Graph等待握手 | 计时区间 | 最长回读 ms | 驱逐增量 ms |
|---|---|---|---|---:|---:|---:|
| register | 63.63GiB / 288块 | 全量 | 50组 | 45秒 | 1091.267 | 2726 |
| hostmalloc | 63.63GiB / 288块 | 全量 | 50组 | 45秒 | 860.873 | 2323 |
| register-nowait | 同上 | 全量 | 无 | 45秒 | 8.981 | 0 |
| staging | 同上，未注册CPU bank | 固定9MiB暂存区 | 50组 | 45秒 | 93.990 | 0 |
| register-repeat | 同上 | 全量，恢复原配置 | 50组 | 45秒 | 1.257 | 0 |

五组均有同样的CPU-only mmap活动，数值与握手计数均通过。nowait去掉的是GPU WRITE/WAIT及对应CPU协调线程，不能据单次阴性认定只换掉一个等待API就能修复。staging在上一轮GPU完成后复制每个CPU bank的样本窗口到固定暂存区，之后GPU读暂存区；未改变系统内存策略。该控制保留大RAM footprint但改变了GPU可见范围、地址布局及CPU拷贝活动，也需要正反向复核。staging仍有约94ms等待，因此不声称所有延迟都被消除。

恢复原register+握手配置的第三次独立启动也没有复现，446个Graph、22800次握手通过。因此目前证明的是该机制可独立出现且有条件性，不能把staging/nowait的干净结果认定为修复。小规模阴性、大规模阳性不能进一步唯一拆分总注册字节数、BO个数、物理页面状态和调度时序。

### 主机内存活动与条件性

| 运行 | 全机后台规整迁移扫描页增量 | kcompactd唤醒增量 | 本进程计时期驱逐 ms |
|---|---:|---:|---:|
| register | 9158550 | 638 | 2726 |
| hostmalloc | 4981609 | 63 | 2323 |
| register-nowait | 0 | 0 | 0 |
| staging | 0 | 0 | 0 |
| register-repeat | 0 | 0 | 0 |

这些是每组启动、计时、清理整个区间的全机`/proc/vmstat`增量，不是每个停顿精确区间，也不是本进程物理页归属证据。干净组仍有全机NUMA页迁移，因此不能把`numa_balancing=1`本身认定为原因。结合前轮后台规整线程栈，**kcompactd页面移动触发本进程MMU失效**成为优先验证的上游假设；目前尚未采到对应VMA/PFN，不能宣称已证实。

后几组内存条件也变化了；这构成staging与nowait控制的实质混杂因素。下一轮必须在自然复现的内存活动下对齐具体页，不能强制调系统内存策略来制造或掩盖结果。[主机上下文](../../../.freetoken-zihao/rdna4-fp8-rocm-vs-freetoken-20261106/host-memory-context.json)

## 4. 驱动恢复范围的核对

已安装驱动的USERPTR恢复不是只更新触发失效的单页：`update_invalid_user_pages`对每个仍失效的BO调用`amdgpu_ttm_tt_get_user_pages`；后者按该BO的`ttm->num_pages`重新取页，`amdgpu_hmm_range_get_pages`分段遍历，单段最大2GiB。SVM恢复也按失效range处理，而非恢复每个token实际访问的字节数。

模型NVFP4布局由H=2560、I=640、E=512、L=48决定，单层最大的gate/up packed bank为800MiB，即204800个4KiB页。若这样的bank是失效BO，恢复工作可能显著大于一次decode真正取用的数据；并发失效又可能引发重试。这是源代码支持的成本放大机制，**尚未确认当前失效地址恰好属于专家bank**。

前轮等待期间，全机worker栈采到过HMM遍历以及`amdgpu_vm_cpu_update`的PTE更新/fence迭代。`vm_update_mode`只读值为-1；驱动在large BAR compute条件下默认可选CPU更新。worker栈不含对应process_info参数，不能把所有全机栈唯一归属本FreeToken PID，也不能据此声称某个fence已证实卡了几秒。没有更改该参数。

## 5. 与官方已知问题比对

AMD开发者的[ROCm PR6676](https://github.com/ROCm/rocm-systems/pull/6676)描述过pageable拷贝的临时注册、allocator回收与进程级队列暂停的交互，提出强制内部staging。该PR为closed而非merged；本应用镜像的`libamdhip64.so.7`也未找到其`ROC_FORCE_STAGED_D2H`标志字符串，因此不把它当成当前可用的修复开关。它提示需审计运行时隐式映射，但不是本次具体失效地址的证据。

另有[TheRock issue9257](https://github.com/ROCm/TheRock/issues/9257)报告过HMM过期notifier序号导致按秒忙重试。本机安装的`amdgpu_hmm_range_get_pages`在`hmm_range_fault`失败时退出并向上返回EAGAIN，没有该报告所述的内层超时重试循环；不按“都是约1秒”就认定同一缺陷。本轮没有切换内核或ROCm版本。

参考ROCm源码checkout为`2efc5ad0e55a31dc25653d6101d699f2fb967f39`，其中pageable-copy pin路径还受传输大小阈值控制，不能把大拷贝的临时注册行为直接套到PLE的4字节回读。该checkout不等于已安装二进制的完整源身份；二进制字符串检查仅证明没有找到字面标志，不替代运行期分支验证。

## 唯一后续验收

只继续方向4：用现有自有PID事件，在自然复现窗口只读比对GPU可见bank的页面映射，定位失效VMA及上游页面移动，再对固定小暂存区做条件可比的正反向控制。当前staging结果只是候选依据，没有通过采用验收；不再重复验证仅换hostmalloc的方案。

保持模型、FP8候选和expert选择/fetch策略不变，仅验证主机内存可见范围/暂存生命周期的AMD适配；不恢复方向2的fetch策略研究。

### 后续VMA与暂存区验收进展

在独立HIP复现中，存活的800MiB `bank-L7-gate_up`保留相同VMA/inode，完整前后pagemap确认128页的PFN改变，采样旧/新PFN都在node7；用户态记录未见应用在ROI内释放该bank。另一个约1MiB运行时SVM区域归属于pageable H2D/D2H输入验算数组，提示不能只处理专家bank。

保留同一800MiB CPU bank的A/B/A/B中，仅改为9MiB专家暂存区仍有61ms累计驱逐，未通过稳定性验收。固定小输入缓冲的补测需要结合反向重启判定；FreeToken已有LockedFetch两种等待模式的Graph功能测试通过，但本轮未运行整模型。63.63GiB控制与整模型仍受128GiB可用RAM门槛限制，不降低门槛、不宣称ready。完整逐阶段结果、观察者影响与剩余门槛统一记录在[验收文档](qwen38-r9700-vma-staging-acceptance.md)。

整模型验收需要足够空闲显存及可比的共享内存条件。判据为相同输出、同PID运行期驱逐/重试、`evicted_ms`、最长token间隔和吞吐；独立HIP控制的无故障不能替代整模型稳定性验收。当前只定位机制，不声明FreeToken已修复或候选ready。

## 本轮收尾

13个独立HIP进程，共14773个计时Graph，所有数值与握手计数检查通过。两种大规模内存API均复现自有驱逐型停顿；未加载模型、FreeToken或PyTorch，也未执行FP8/GEMV/MoE计算。动态依赖仅HIP及标准C/C++库，源代码和二进制SHA见实验目录。

原11个文件和候选2个文件指纹一致；记录的驱动参数、srcversion、NUMA/THP状态前后完全一致。所有自有GPU进程、私有读取者已退出，预约已按共享锁释放，未停止外部进程，未提交或推送代码。[完成记录](../../../.freetoken-zihao/rdna4-fp8-rocm-vs-freetoken-20261106/completion.json)
