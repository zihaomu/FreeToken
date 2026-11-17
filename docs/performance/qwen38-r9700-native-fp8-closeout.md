# Qwen3.8 / R9700：方向4稳定性收尾

更新：2026-11-06。唯一执行任务是冻结原生packed FP8候选、定位长停顿、复核独立重启正反向A/B。[方向2已冻结](../../../doc/qwen38_direction2_frozen_20261106.md)，不做新增研究或组合。推理kernel保持现有候选，不扩大shape、改量化或调tile。

## 冻结输入

- 原生候选工作树：`worktrees/qwen38-native-fp8-20261106`，分支`perf/qwen38-native-fp8-20261106`。
- `kernel/triton/fp8_pertensor_linear.py` SHA256：`dbabe20d7bfbc1e96b36a954a2b1fde9b26ba3f34e681acd2cd99800c39c1573`。
- `layers/quantization/linear/fp8_tensor.py` SHA256：`7b6c44f294ab98ffd8e812d69f7448ee052b53e8c52ccc3040b6f1be92c7a3ce`。
- 对照为原冻结baseline工作树；原11个修改在主工作树、对照和候选中逐文件SHA一致。
- 相同模型revision与dense-FP8派生权重，BF16 activation/KV、7610 slots、8256 KV tokens、fetch1、31 CPU workers、shared stream、MTP off、TP1、PLE disk、host embedding。`GPU_MAX_HW_QUEUES`保持unset。
- 应用镜像`sha256:6cf7ba2162685f0282ebbe3b3f98dfbb9c06af0663809bef5d72b508ef26c962`；ROCm10.1 / HIP7.16，Torch2.14 / Triton3.8。
- 本轮选择物理GPU4 / PCI `0000:83:00.0` / UUID `GPU-0ef5d77c2f19f98d`。PCI03初检空闲但申请时已被另一任务预约，未在其上启动模型测量。PCI83已单独预约，foundation核验通过；不将旧PCI03的数值混入本轮统计。

全部新证据位于仓库外[收尾目录](../../../.freetoken-zihao/rdna4-fp8-closeout-20261106/)，包括`frozen-inputs.json`、`foundation-gpu4.log`、每轮命令/patch/监控/原始请求。

## 验收方案

首轮正式对照采用A1→B1→B2→A2→A3→B3，六个独立模型进程、三组启动对。每轮沿用同一11请求固定history：4 warmup、每输入长度3个计时请求、1个独立stats请求；每请求256输出。每配置每长度共9个计时样本。512/2048分别报告，不合并成一个有利均值。

正式测速期间监控只读取主机`/proc`和KFD PID目录，不读取GPU VRAM、hwmon、busy或进程显存文件。启动前/服务退出后检查实际GPU占用；运行期间记录新KFD进程、CPU/IO/memory pressure。这个方式不能完全排除已存在外部进程瞬时切到本卡，局限保留，不把弱监控写成严格独占证明。

完整样本均进入性能统计。保留既定门槛：吞吐改善至少1%、配对95%区间下界>0、启动对收益一致为正，TTFT和p95无超过3%回退。请求级区间与独立重启对结果分列。诊断轮单列，不混入正式样本。

## 长停顿：已有证据的再定位

对旧固定history ABBA，使用已有`finish_wall_time`与最后token monotonic stamp对齐；>100ms仅为诊断阈值，不改变性能样本或验收门槛。

| 旧运行 | >100ms间隔数 | 最大间隔 | 与GPU监控读取直接重叠 | 新KFD触发GPU探针次数 |
|---|---:|---:|---:|---:|
| A1 | 20 | 500.2ms | 0 | 3 |
| B1 | 23 | 2700.8ms | 0 | 4 |
| B2 | 0 | — | 0 | 41 |
| A2 | 0 | — | 0 | 0 |

据此不能将旧异常直接归为监控造成；新KFD进程出现也不必然触发停顿。基线和候选均有异常，因此尚无FP8独有错误的证据。时间对齐存在小量误差，不能据“未重叠”排除间接驱动影响。[完整审计](../../../.freetoken-zihao/rdna4-fp8-closeout-20261106/archived-stall-audit.json)

本轮若复现长停顿，诊断优先区分host graph replay、PLE填充/磁盘flush、CPU任务等待和最终copy_done等待；沿用原调用，不通过删依赖或更换kernel消除现象。

## 执行记录

1. 候选、原11文件、环境和模型配置冻结完成。
2. 方向2冻结文档与当前范围更新完成。
3. 旧异常时间线复核完成；监控直接读取不能解释全部旧停顿。
4. 六次独立启动的同卡固定history对照进行中，首轮A1已开始。新性能与诊断结果将在本节追加，不预先宣布ready。

首个A/B启动对已完成，全样本保留：

| 输入 | A1均值tok/s | B1均值tok/s | A1/B1平均TTFT | A1/B1平均p95间隔 |
|---|---:|---:|---|---|
| 512 | 38.2433 | 42.6680 | 1.3950 / 1.5028 s | 30.555 / 27.088 ms |
| 2048 | 40.7264 | 41.9587 | 1.4128 / 1.4133 s | 30.303 / 25.101 ms |

A1正式样本复现1006ms间隔，B1复现698ms间隔；两轮均无GPU监控读取。因此首对不能按正常约43tok/s的片段宣布稳定通过。后续B2/A2/A3/B3继续；以三个独立启动对的区间为主，请求级区间另列。

测量脚本修正记录：退出后的自有engine PID短暂仍列在KFD显存计数中，首次被误判外部占用；已按该轮source-identity中的PID校正，原summary另存。随后进程条目消失时VRAM计数仍短暂偏高，使B2首次preflight拒绝启动；该次没有模型请求，保留为`fixed-b2-preflight-only`。现在仅在模型退出后等待显存回收，并在下一轮启动前确认连续两次空闲。推理候选和正式请求未修改，没有丢弃慢样本。

### 六次独立启动：全量正式结果

三组启动对已完成。每配置每长度9个计时请求，统计以3个独立启动对的平均log吞吐比计算Student-t 95%区间（df=2）；9个请求的区间另列在JSON中，仅作描述，不把请求数当成9次独立重启。

| 输入 | A全量均值tok/s | B全量均值tok/s | 配对几何收益 | 启动对95%区间 | 三组启动对收益 |
|---|---:|---:|---:|---|---|
| 512 | 39.6903 | 39.5089 | -1.2604% | [-34.4600%, +48.7567%] | +12.1239%、-18.0547%、+4.7735% |
| 2048 | 38.7946 | 42.4556 | +10.6798% | [-11.9632%, +39.1465%] | +2.9153%、+22.7627%、+7.3146% |

512平均TTFT回退3.041%，p95间隔改善6.50%；2048 TTFT+0.015%，p95改善11.58%。全量两档CI跨零，且存在未解释秒级间隔，**本轮正式稳定性门槛未通过**。2048的表观+10.68%受基线慢样本影响，不能宣布为FP8的稳定收益；也不能因512均值基本持平就认定转换kernel无效。

[全量结果与逐请求停顿](../../../.freetoken-zihao/rdna4-fp8-closeout-20261106/fixed-summary.json)。6轮均完整校验11条固定history、检测到的常驻外部本卡进程为0，运行期GPU监控读取为0；记录到其他任务的进程启动，保留共享主机/瞬态干扰限制。A2没有新增KFD进程，仍复现显著停顿，故新进程启动不是必要条件。

接下来仅执行单独的B/A host-span诊断，不扩大正式测速样本以追求通过。诊断不新增GPU event或同步，记录原有Graph replay、Event同步、PLE native stage/flush和调度阶段的wall/线程CPU时间；全部排除出性能验收。

### 诊断B：停顿已定位到PLE token回读事件等待

`diagnostic-b`完成11个固定history请求。85个生产矩阵全部核实为packed mode2；候选源码不变。捕获8次>100ms token间隔，8次均有至少90%的间隔与`Event.synchronize@_fill_decode:222`重叠。

| 原有调用 | 最长wall时间 | 线程CPU时间情况 | >100ms次数 |
|---|---:|---|---:|
| PLE `_readback_event.synchronize()` | 5642.697ms | 最大约5639.733ms，接近持续忙等 | 8 |
| PLE native `flush` | 14.030ms | 最大5.102ms | 0 |
| GraphRunner host replay | 3.773ms | 最大0.459ms | 0 |
| Scheduler `copy_done.synchronize()` | 23.265ms | 同量级 | 0 |

主线程通过`DiskRowTable._await_pending()`等待PLE worker；秒级时间暴露在GPU token回读完成事件上，而非这些样本的磁盘flush或host Graph提交。这里定位的是阻塞位置，不据此声称已找到HIP内部唯一根因，GPU前序工作仍可能是其等待对象。[对齐分析](../../../.freetoken-zihao/rdna4-fp8-closeout-20261106/host-span-analysis.json)

基线A诊断进行中。随后只用已有`FREETOKEN_PLE_SYNC=gate`做一次明确标注的反事实诊断：PLE填充完成后再提交Graph，避开当前异步wait-sync路径。该控制保持FP8候选文件、权重、tile与专家分工不变，不计入上方正式A/B，也不预先作为采用配置。

### 基线与launch-gating反事实：简单切换模式不是修复

基线`diagnostic-a`完成：85个矩阵核实为mode0；捕获1次>100ms间隔（106.218ms），同样主要与PLE回读事件等待重叠，native flush最长9.733ms。候选/基线的诊断发生在不同启动，不拿8次对1次推算故障概率或归因FP8。

`diagnostic-gate-b`完整完成11个请求，确认日志为launch-gating且85个矩阵仍为mode2。结果为49次>100ms间隔，最长**9701.794ms**；第6个请求吞吐降到约5.07tok/s。旧的异步worker回读等待已不存在，但同样的长时间暴露在`DiskRowTable.host_fill_batch`的同步token回读阶段（`Tensor.to("cpu")`所在路径），最长9700.109ms，其中线程CPU时间9392.502ms。native flush最长10.046ms、fill最长约10ms，没有承载这次秒级停顿。

因此不能把“使用gate”作为修复，也不能把异步PLE worker或磁盘吞吐直接当唯一根因。当前证据收敛到**host读取GPU token的完成等待，以及其前序GPU执行/依赖进度**。需要区分GPU上前序工作本身变慢，还是主机完成等待通知延迟。

最后补一次`gpu-timeline-b`诊断：仅在已有Graph replay两侧增加GPU timestamp事件，记录前后Graph的设备时间与间隔；无额外逐token synchronize，结束时读取已完成事件。该轮有新增GPU标记，可能改变队列调度，因此与正式性能样本分离，不将“不再复现”当修复。候选仍冻结；本轮不再扩展其他环境或kernel变体。

### GPU时间线：至少一个停顿来自前一Graph设备进度

`gpu-timeline-b`完成11个请求，获得2805段decode设备区间，中位22.831ms，最大188.831ms。捕获到的一次187.255ms回读事件等待，与前一个Graph的188.831ms设备区间对应；随后当前Graph为21.903ms，两个Graph之间设备间隔仅0.132ms。这支持该案例中**前一Graph设备进度延迟**，不能只归为主机Event通知晚。

该轮没有复现秒级停顿；新增标记会影响队列行为，且环境在变化，因此不能把这个约187ms案例无条件外推为所有9.7秒停顿的唯一根因。Graph时间包含kernel、搬运和图内等待，不代表某个FP8算子执行了188ms。[时间线证据](../../../.freetoken-zihao/rdna4-fp8-closeout-20261106/gpu-timeline-analysis.json)

## 本轮决定与唯一剩余阻塞

**方向4仍未通过默认采用的稳定性门槛，冻结候选继续默认关闭。** 本轮完成6次正式独立启动及4次单独诊断；正式全量结果未过，不能用诊断轮正常片段或旧自然生成+2.85%替代。既有转换/GEMV/真实activation/416个logits位置/4352个自然token的通过证据保留，但本轮的固定history输出是强制重放，不新增精度认证。

长停顿已从“可能是监控、磁盘或FP8”收敛到GPU token回读完成等待及其前序decode Graph进度；已排除“简单改用gate即可解决”的方案。**仍未定位Graph内部的具体慢节点，尚不能宣称HIP唯一根因或问题已修复。** 后续如继续方向4，唯一工作项是把前序Graph分段定位，保持当前FP8候选、专家分工和精度冻结。

方向2继续冻结，不自动恢复；未启动其他kernel、量化或TP路线。候选两个文件SHA与冻结记录一致，原11文件未改变。本次服务均已退出，物理GPU4 / PCI83预约已释放；其他任务预约未改动。[收尾机器记录](../../../.freetoken-zihao/rdna4-fp8-closeout-20261106/completion.json)


## 后续停顿定位（2026-11-06）

[继续定位报告](qwen38-r9700-stall-localization.md)已捕获8904.302ms设备Graph，并与同一token的8906.705ms间隔和8903.080ms CPU回读Event等待对齐。延迟分散在不同计算/等待区间，最终触发原因尚未确定；后续空标记、纯Graph和带KFD主机计数的运行没有复现秒级事件，不构成修复或稳定性通过。候选默认关闭、方向2冻结的决定保持不变。


sudo只读日志后续：原9.7秒所在连续异常窗口出现`restore_userptr_worker` CPU密集告警，安装源码证实其恢复主机注册内存前会暂停所属进程GPU队列。这是候选机制，告警缺PID/PCI，尚未确认FreeToken归属；8.9秒精确窗口没有记录到fault/reset/MES超时。未修改底层，详情与限制均在[定位报告](qwen38-r9700-stall-localization.md)。


进一步进程归因：两个无GPU标记gate启动共捕获11次>=1秒回读等待，同PID队列驱逐窗口覆盖97.05%–99.38%，并由`evicted_ms`增量复核。已确认这些复现主要来自自有SVM/USERPTR队列暂停/恢复；具体失效VMA与发起者尚未唯一锁定。FP8候选及正式稳定性结论不变，详见[进程归因报告](qwen38-r9700-stall-attribution.md)。
