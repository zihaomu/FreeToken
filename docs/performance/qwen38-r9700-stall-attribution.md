# Qwen3.8 / R9700：停顿的进程级归因

日期：2026-11-06。仅继续方向4的长停顿归因；FP8候选和模型配置冻结，方向2继续冻结。本轮不改驱动、模块参数、NUMA/THP/sysctl、全局队列配置或kernel代码。

固定环境：R9700物理GPU4 / PCI83，ROCm10.1 / HIP7.16，应用镜像`6cf7ba2162685f0282ebbe3b3f98dfbb9c06af0663809bef5d72b508ef26c962`。沿用Qwen3.8-Flash-Next NVFP4 revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`的selective dense FP8派生权重；batch1/TP1，BF16 activation/KV，85个dense矩阵native packed FP8，hybrid 7610槽/31个CPU worker/max-fetch1/shared stream/PLE disk。五轮source patch SHA完全一致。完整命令见[无标记复现启动记录](../../../.freetoken-zihao/rdna4-fp8-stall-attribution-20261106/sessions/unmarked-gate-b1/command.json)。

## 当前已经确认到哪一层

无GPU标记的原始`gate`回读路径再次出现秒级停顿。对同一FreeToken PID1377947、PCI83 / KFD节点23051，已同时捕获队列驱逐、恢复重试、累计驱逐耗时和CPU等待。**本次复现的长等待主要由进程自己的GPU队列在主机内存映射恢复期间暂停导致。** 这已超出上轮“全机userptr告警时间相关”的证据层次。

原9.7秒案例没有同时采到这些事件，因此不能给历史单个样本补造精确因果；新的同类复现把机制落实到了自己的进程。已继续只读捕获发起端线程和内存管理后台线程的栈，尚未唯一确定哪块内存、哪个调用触发失效；已确认的队列暂停机制与未确认的上游触发源分开记录。

## 直接对齐证据

| 请求 | CPU回读等待 ms | 同PID驱逐窗口覆盖 ms | 覆盖比例 |
|---|---:|---:|---:|
| 2 | 3239.359 | 3209.675 | 99.08% |
| 7 | 2531.387 | 2498.815 | 98.71% |
| 3 | 1957.337 | 1921.137 | 98.15% |
| 2 | 1406.057 | 1379.971 | 98.14% |
| 5 | 636.350 | 610.273 | 95.90% |

最长等待线程CPU时间3231.547ms，表明CPU在持续等待GPU进度。该窗口对应的累计驱逐计数也出现增长，详见[对齐原始数据](../../../.freetoken-zihao/rdna4-fp8-stall-attribution-20261106/unmarked-gate-b1-causal-alignment.json)。本轮有SVM触发0和USERPTR触发1，存在嵌套驱逐以及`R`恢复重试。

队列事件在驱逐操作前、恢复操作前发出。分析按源代码的驱逐引用计数配对：9增加深度、正常a/0减少深度、a/R只是重试，不能当成恢复完成；并用`evicted_ms`累计增量交叉验证。事件时间使用`ktime_get_boottime_ns`对应主机CLOCK_BOOTTIME，记录中同时保留perf/boottime用于核对。

## 本轮方法与只读范围

读取驱动现有KFD事件FD。事件掩码只写入该读取者自己的`client->events`，源码已核对不会更改硬件寄存器、模块参数、其他进程策略或全局tracing开关；关闭FD即移除读取会话。没有启用tracefs、perf/BPF或CPU内核追踪。事件流补足普通kernel warning缺失的PID和驱逐原因。

同时读取KFD主机atomic计数和`/proc/PID/task/*/schedstat`；CPU等待状态通过实验目录内CPU共享页发布，不增加GPU操作。等待已超过0.5秒时，只读采样后台worker栈，已见`svm_range_restore_work`、`amdgpu_amdkfd_restore_userptr_worker`、`hmm_range_fault`与GPU页表更新路径。worker扫描耗时约68–105ms，全部属于诊断运行，不计入性能验收。

## 反证与观测影响

- 首轮`attribution-b1`：11请求、2805 Graph，最大96.910ms，运行期自有驱逐事件为0，计数0。与此同时外部容器有28302条恢复重试。故不能仅凭全机恢复告警认定本进程被暂停。
- 读取者能抓到该FreeToken进程在最后请求结束约37ms后的USERPTR驱逐，验证了PID筛选；退出清理事件与解码期间事件分开计算。
- 容器内PID1会在不同容器重复，不能把外部所有PID1事件指向同一个宿主PID。我们的模型使用host PID namespace，自有归属没有该歧义。
- KFD对进程的所有GPU节点发事件，即使某节点没有活跃队列；外部节点23051事件不等于外部任务正在GPU4计算。
- 有GPU标记的`gate-attribution-b1`未复现秒级事件；去掉额外GPU标记与请求末尾GPU回读后，`unmarked-gate-b1`复现。尚不足以把差异唯一归因于标记，但后续触发来源采样继续使用无标记路径。

## GPU派发记录

另做应用层ROCProfiler活动记录，只观察预热后的请求5；没有开启PMC/SPM/ATT。小型校准的3次Graph、96条dispatch全部通过。整模型SQL输出在退出阶段未完整写出，原始ring-buffer数据保留；已根据应用镜像中SDK的公开ABI解码680974条连续dispatch，ID1–680974。

255个decode Graph的255个begin和159885个中间标记全部完整；GPU时钟与硬件dispatch时长最大差约0.001005ms。该正常运行没有秒级故障，最长decode dispatch61.605ms；暂未恢复kernel名字，不给数字ID编造函数名。Profiler将应用的device-memory队列拦截为system-memory队列，因此这轮不能作为原调度等价的性能证据或修复证据。[活动记录核对](../../../.freetoken-zihao/rdna4-fp8-stall-attribution-20261106/activity-b1-marker-crosscheck.json)

## 第二次独立启动复核

`unmarked-gate-b2`仍使用同一冻结源代码及11条固定历史，确认再次复现：

| 请求 | 回读等待 ms | 驱逐窗口覆盖 ms | 覆盖比例 | 独立累计计数增量 ms |
|---|---:|---:|---:|---:|
| 8 | 4600.367 | 4571.635 | 99.38% | 4573 |
| 1 | 4347.810 | 4318.350 | 99.32% | 4320 |
| 6 | 3383.079 | 3352.976 | 99.11% | 3356 |
| 3 | 3104.841 | 3075.089 | 99.04% | 3076 |

两次启动共11个>=1秒的回读等待，全部由同PID队列驱逐窗口覆盖97.05%–99.38%。计时请求阶段的累计驱逐计数分别增长13145ms、22541ms；native PLE flush最大分别仅10.041ms、13.942ms。该结果确认这些复现的主要等待来源，并不表示每个100ms级延迟都有相同原因：第二轮仍有两个较短样本没有驱逐窗口覆盖。[第二轮对齐数据](../../../.freetoken-zihao/rdna4-fp8-stall-attribution-20261106/unmarked-gate-b2-causal-alignment.json)

### 发起端只读采样的结果与边界

在自有驱逐事件到来时，采集本进程线程stack/syscall及NUMA统计，共21次快照，单次约8–12ms。没有捕获到直接执行`svm_range_evict`/`mprotect`/`munmap`的发起线程；采到的`kfd_ioctl_wait_events`是HSA事件等待线程，不能称为失效发起者。所读线程的NUMA迁移统计均为0，也不能据系统`numa_balancing=1`就认定自动NUMA迁移导致故障。

另对kcompactd、kswapd、khugepaged做只读栈采样，观察到后台内存规整的`migrate_pages -> migrate_folio_unmap -> try_to_migrate`，但没有把相应物理页或MMU回调唯一归属到本PID。因此“后台规整触发”仍是候选，不作为已经确认的上游根因。

## HIP等待节点的补充核对

对同一HIP7.16应用镜像做最小调用校准：signal预先已就绪，不注入长等待；直接`hipStreamWaitValue64`记录为`__amd_rocclr_streamOpsWait.kd`，捕获后重放记录为`__amd_rocclr_batchMemOp.kd`，均为1线程dispatch。模型原始活动记录中编号13、每token50次的1线程runtime记录与该内建batchMemOp匹配；因模型symbol表未完整落盘，这部分函数名对应是基于内建模块序号、形状及调用次数的一致性推断。

因此不能一概用“HIP Graph wait不占计算单元的前端memop”来解释AMD利用率，也不能把这里的61.6ms等待时间计为FP8算术耗时。校准只确认当前实现路径，不构成替换等待实现就能解决队列驱逐的证据。[等待调用校准](../../../.freetoken-zihao/rdna4-fp8-stall-attribution-20261106/wait-probe-validation.json)

## 本轮决定

**已确认新复现的秒级停顿机制：本进程SVM/USERPTR失效，GPU队列被暂停，页映射/页表恢复反复重试，CPU回读忙等。** 这能解释不同kernel区间出现异常墙钟时间，也说明此类样本不适合拿来归因FP8算术性能。

尚未唯一确定失效的具体VMA和发起者，不声称原历史9.7秒样本已逐点验证，也不声称已修复。后续唯一剩余追踪点是长期GPU可见主机内存的具体失效区域与来源；方向2不恢复，不先调整驱动或全局内存策略。

本轮5次整模型诊断共55请求、14025个decode Graph，其中两个无GPU标记启动重复验证了自有驱逐型停顿；固定history验证全部通过，不新增自然生成精度认证，不更新正式吞吐收益。FP8候选仍默认关闭。所有自有服务和读取会话均结束，预约释放与源码/配置冻结校验见[完成记录](../../../.freetoken-zihao/rdna4-fp8-stall-attribution-20261106/completion.json)。

后续独立HIP定位已取得800MiB存活bank内128页物理映射改变的证据，还识别了pageable拷贝数组的隐式SVM注册；这些是独立进程中的具体VMA，不能冒充上述整模型PID的同一地址。仅缩小专家暂存区尚未通过稳定性验收。所有新增对照与剩余整模型门槛见[失效VMA与固定暂存区验收](qwen38-r9700-vma-staging-acceptance.md)，方向2及原FP8候选继续冻结，驱动和全局参数未改。

后续已补上独立HIP反事实：不加载FreeToken/PyTorch，同模型维度的主机映射在两种分配API下均复现自有驱逐型停顿。13组实验及背景规整混杂因素见[责任边界与独立复现](qwen38-r9700-rocm-vs-freetoken.md)。因此已能确定FreeToken不是此类机制的必要条件，仍未唯一定位具体失效地址，也未证明某个特定ROCm缺陷或完成修复。
