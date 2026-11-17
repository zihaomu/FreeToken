# Qwen3.8 / R9700：七项适配研究执行计划

创建：2026-11-05；首轮研究更新至2026-11-06。目标：降低单请求 decode 的每 token 成本，向 100 tok/s 推进。背景与历史结果见 [核心记录](qwen38-r9700-100tps.md)。本轮按该文第 11 节的前七项依次研究，不与此前七类 kernel 工作树混淆。

当前范围（2026-11-06用户最新决定）：仅执行[方向4稳定性收尾](qwen38-r9700-native-fp8-closeout.md)。方向2停止投入，见仓库外[冻结文档](../../../doc/qwen38_direction2_frozen_20261106.md)。MTP/投机解码暂不单独推进；下文保留首轮研究记录。

## 执行约定与冻结基线

- 顺序执行 1→7；每项先写假设和实验，再在本节追加原始证据、结果、局限和决定。允许得到无收益或入口不支持的结论，不把未完成验收写成 ready。
- 源码基于 `cf32a5364303d71ece7979a6c69857c475555eeb`，保留原有 11 个修改文件；实验代码在独立工作树/仓库外目录中，候选默认不启用。
- 模型：`nvidia/Qwen3.8-Flash-Next-NVFP4`，revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`。现有 selective dense FP8 派生权重只读；激活与 KV 为 BF16。
- ROCm 10.1 / HIP 7.16，Torch 2.14.0+rocm10.1.0 / Triton 3.8.0；应用镜像 `sha256:6cf7ba2162685f0282ebbe3b3f98dfbb9c06af0663809bef5d72b508ef26c962`。每轮记录 GPU PCI、CPU affinity、源码和缓存。检查空卡与预约，出现外部竞争则停止自己的实验并保留无效记录。
- 单 stream、batch 1、TP 1、最多 8192 context、fetch1、31 CPU workers 是初始对照。此前约 42 tok/s 只作历史参考，新收益与本轮相同环境的对照比较。
- 算术候选：独立参考局部数值验收；真实模型 32-token 初筛及六组 ×64 token 验收，top-1 ≥99%、KL ≤0.001。纯调度变更固定路由下优先要求 bitwise。缓存分工改变 CPU W4A8 / GPU W4A16 路径时，单独报告质量差异。
- 微测需独立重复、A/A、热权重与至少 256 MiB 轮换权重；性能结论必须覆盖整个调用链。整模型正式采用需 A→B / B→A、每项至少六个样本、收益 ≥1% 且 95% CI 下界 >0，TTFT/p95 无 >3% 回退。探索筛选不等于正式采用。
- 原始文件根目录：`.freetoken-zihao/rdna4-seven-step-20261105/`（记为 `S/`）。运行中持续更新本文，失败、资源竞争和负结果也保留。

## 顺序与状态

| 顺序 | 任务 | 首轮交付 | 状态 | 采用状态 |
|---|---|---|---|---|
| 1 | 显存预算重新分配 | KV 上限、专家 slots 扫描与真实模型对照 | 首轮完成；四档整模型筛选 | 未采用 |
| 2 | 少量 miss 的 hybrid 成本模型 | 实际调用链与分工成本、最小实验 | 首轮完成；成本表已测 | 未采用 |
| 3 | CPU 提前 publish，重叠 shared expert | 生命周期审计与最小调度候选 | 首轮完成；收益有回退 | 未采用 |
| 4 | RDNA4 packed vector 算术 | BF16 DOT2 / FP8 成对转换的 ISA 与数值证据 | 首轮完成；指令语义已测 | 未采用 |
| 5 | 按真实形状调 tile / split / warps | FP8 三类形状和 QSA 的完整链筛选 | 首轮完成；42+16 案例 | 未采用 |
| 6 | MTP 多 token 复用 | 该 NVFP4 checkpoint 的实际加载与执行约束 | 首轮完成；实际 draft 可运行 | 未采用 |
| 7 | 按数据用途区分缓存策略 | 可用 counters、hint 映射与对照 | 首轮完成；hint 微测及工具限制 | 未采用 |

## 1. 显存预算重新分配

假设：dense FP8 释放约 2.70 GiB，但当前剩余预算进入远超 8192 请求长度的 KV pool。限制无用 KV 后增加 expert slots，可能降低 CPU miss。

实验顺序：

1. 固定旧 7610 slots 运行本轮对照，记录实际 KV、显存与路由统计。
2. 复用 `--num-tokens` / cache-auto 入口，核实 QSA pool 的容量、对齐与完整 8K 需求。
3. 保持其余配置，扫描 7866 / 8122 / 8378 slots；记录 prefill workspace、miss、CPU-active、decode、TTFT。
4. 仅有稳定收益的配置再做反向重启和数值验收；显存容量增加本身不算提速。

### 1.1 实施记录

- 已建立实验工作树 `worktrees/qwen38-rdna4-sequential-20261105`，分支 `perf/qwen38-rdna4-sequential-20261105`；原有代码文件指纹归档于 `S/initial-source.json`。
- 物理 GPU5/6 在申请前被其他任务预约，未覆盖预约；改用空闲物理 GPU0 / PCI `0000:03:00.0` / HIP index 2。ROCm 10 foundation 检查通过，见 `S/foundation-gpu0.log`。CPU affinity 固定 `0-31,64-95`，内存跨 NUMA 0–7 interleave。
- QSA 最终 page_size=64，`--num-tokens` 在后端调整 page_size 后校验并转换为 pages；使用 8256 tokens（8192 上限加一页余量）筛选，不改变逻辑长度。
- `s1-baseline-a` 启动失败：新工作树缺少安装时编译的 `_pinned_tensor`。已复制同一锁定应用环境验证过的三个 native 扩展，记录 SHA256 于 `S/native-extensions.json`；失败记录保留，不计测速。重试 `s1-baseline-a2`。
- 先做了自然生成筛选，再补固定 history 对照。固定的是模型输入 token 历史，不是中间 expert ID；分工改变仍可能影响 hidden 和后续路由，因此还需独立质量验收。

本轮实测初始化：7610 slots 时 KV = **124736 tokens / 2.94 GiB**，启动空闲显存 31.75 GiB，初始化后 3.17 GiB。已复现“多余 KV 吸收量化释放容量”的事实。

筛选协议更新：额外的四个配置全部重启服务，并重放已冻结的 11 个请求、每请求 256 个输出 token 的 history；7610/7866/8122/8378 均固定 KV=8256。自然生成初始基线不直接与固定 history 宣称速度收益。路径：`S/frozen-history.json`、`S/s1_sweep.py`。

### 1.2 缓存容量与 kernel 分支边界

新发现：`offload_kernels.py` 的 `_HYBRID_EVICT_CHUNK=8192`。`num_slots <= 8192` 时一次读入 usage 并在寄存器中找 victim；更大时每个 victim 都按 8192 一块扫描。因此 8378 不是“纯粹更多容量”的对照，也改变淘汰路径。已准备 7610/7866/8122/8192/8193/8378 的固定单 miss graph 探针，单独解释这个边界，不能把 metadata 探针冒充完整 MoE 时延。

自然生成基线 `s1-baseline-a2`：512-input 三轮 **42.5932 / 42.3747 / 25.9926 tok/s**；2048-input **42.1801 / 35.1995 / 42.4197**。慢轮次全部保留，GPU 无外部竞争；尚不能认定主机环境稳定。独立 stats 请求 CPU experts/layer=0.552369，miss/layer=1.077288。固定 history 对照结果与中断记录见下文。


### 1.3 资源冲突与重启

`s1-fixed-7866` 在 warmup 时出现外部 GPU 进程，监控停止了自己的容器，退出 143，`performance_eligible=false`；没有完整样本，不用于比较。GPU0 既有完成结果保留。

切换为空闲物理 GPU1 / PCI `0000:23:00.0`。为避免不同工具编号混淆，使用 `ROCR_VISIBLE_DEVICES=GPU-98103ef4c22a3ccc` + 进程内 HIP0，并实际断言 `pci_bus_id=35` 和 device_count=1。禁用的物理 GPU3 为 PCI63，不在可见集合。UUID 隔离依据 [AMD GPU isolation](https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html)。四个配置在 GPU1 重新开始，不能把 GPU0 和 GPU1 数据当配对 A/B。

<!-- S1-LIVE-START -->
| Run | history | 512 input 三轮 tok/s | 2048 input 三轮 tok/s | miss/layer | CPU/layer |
|---|---|---|---|---:|---:|
| s1-baseline-a2 | real_generation | 42.593 / 42.375 / 25.993 | 42.180 / 35.200 / 42.420 | 1.077288 | 0.552369 |
| s1-fixed-7610 | fixed_token_history | 42.504 / 31.792 / 42.775 | 42.479 / 42.573 / 42.574 | 1.077288 | 0.552369 |
| s1-gpu5-fixed-7610 | fixed_token_history | 43.028 / 42.967 / 42.949 | 42.809 / 42.817 / 42.725 | 1.077288 | 0.552369 |
| s1-gpu5-fixed-7866 | fixed_token_history | 43.225 / 43.142 / 43.184 | 42.991 / 42.944 / 42.995 | 1.029575 | 0.525899 |
| s1-gpu5-fixed-8122 | fixed_token_history | 43.321 / 43.200 / 43.256 | 43.037 / 43.022 / 43.026 | 0.955719 | 0.478350 |
| s1-gpu5-fixed-8378 | fixed_token_history | 43.476 / 43.361 / 43.408 | 43.200 / 43.148 / 43.182 | 0.907843 | 0.453922 |
<!-- S1-LIVE-END -->

### 1.4 首次中断时的结论与未完成项

第二次尝试 `s1-gpu1-fixed-7610` 又被外部 `sglang::scheduler_TP3` 占卡打断，退出 143；同一环境连续两次失去 GPU 独占，未擅自停止任何外部进程。完整记录见对应 `sessions/*/{summary.json,resources.jsonl}`。

当时结论：**显存预算浪费已实测确认；增加专家容量是否提速尚未完成验收。** 截至这次中断，只能证明 KV=8256、slots=7866 可以启动，不能据此推荐替换原 7610 配置。8378 跨过淘汰算法边界，应与 8122 分开判断。

这次中断后待补：同一空卡四配置固定 history 全部完成、单 miss metadata 探针、完整 8K 请求、优胜配置反向重启及质量。已保留 `s1_sweep_gpu1.py` 和 `s1_ensure_probe.py`，没有把失败轮次算作通过。先按顺序研究第 2 项；后续补测见1.5及下文。

### 1.5 8192 slots 边界实测（2026-11-06 补录）

`S/s1-ensure-probe.json`，物理 GPU5 / PCIA3、无 GPU 竞争；7 个案例的单一 miss/fetch1 映射检查全部通过。测量 reset+ensure metadata 图，不含专家拷贝或计算；每配置5组×1000次。7610首尾 A/A 为8.993/9.005 µs。

| slots | 7610 | 7866 | 8122 | 8192 | 8193 | 8378 |
|---|---:|---:|---:|---:|---:|---:|
| reset+ensure 中位数 µs | 8.993 | 8.959 | 9.016 | 8.971 | 10.538 | 10.450 |

8192→8193 约增加1.57 µs（17.5%）；8378也比8192约慢1.48 µs。这证实源码分支确有成本，**不代表8378整模型一定更慢**，额外命中收益仍需完整模型对照。部分配置首个计时偏高，原始样本全部保留。

### 1.6 同卡四档完整模型筛选结果（2026-11-06）

`s1-gpu5-fixed-{7610,7866,8122,8378}` 均完整结束，监控未发现外部 GPU 竞争。每档四个 warmup、六个测量请求（512/2048输入各三轮）及一个 stats 请求；每请求256个输出 token，全部11个请求通过 frozen-history 精确重放检查。结果为每个长度的三轮平均，单位 tok/s。

| slots | 初始化后空闲 GiB | 512 input | 较7610 | 2048 input | 较7610 | miss/layer | CPU/layer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7610 | 5.93 | 42.981 | +0.00% | 42.784 | +0.00% | 1.077288 | 0.552369 |
| 7866 | 5.27 | 43.184 | +0.47% | 42.977 | +0.45% | 1.029575 | 0.525899 |
| 8122 | 4.61 | 43.259 | +0.65% | 43.028 | +0.57% | 0.955719 | 0.478350 |
| 8378 | 3.96 | 43.415 | +1.01% | 43.177 | +0.92% | 0.907843 | 0.453922 |

所有配置实际 KV=8256 / 0.19 GiB。8378相对7610，miss/layer减少15.7%、CPU/layer减少17.8%；吞吐仅增加约0.92%–1.01%。512/2048平均TTFT分别变化+0.014%/-0.142%，平均p95 token间隔下降1.40%/0.98%。这说明减小miss有收益，但当前不能依靠这条路径获得显著的整体提速。

**决定：保留8378作为约1%的配置候选，尚未采用。** 这是同卡正向筛选，没有反向重启或模型质量门槛；2048场景还低于1%门槛。不能把这组值标为已确认的生产收益。8122/8378的个别warmup分别出现约29.1/30.8 tok/s，仍保留原始记录，测量样本没有删选。

完整8K请求、候选反向A/B及真实logits仍未做。后续若叠加MTP，需先重新分配draft显存，不能直接延续8378 slots。原始汇总 `S/s1-gpu5-summary.json`，逐请求日志与重放证明见 `S/runs/s1-gpu5-fixed-*/`。

## 2. 少量 miss 的 hybrid 成本模型

假设：top-10 饱和带宽推算不能代表 miss=1/2 时的固定发布、CPU 收尾及 GPU 临界路径。

实验顺序：追踪现有 `benchbw`、`fetch_fraction` 与 min-wait 的实际 dispatch；构造唯一 expert ID 的 miss=0/1/2/4/10、fetch=0…miss 的完整链，固定 worker/NUMA，纳入 GPU expert、拷贝与 wait；将结果与当前 fetch1 对照。决策需同时保留后续缓存复用价值。

验收：记录 CPU/GPU 分工、实际 miss/fetch、整链 graph 时延、数值、稳定性。只有完整链数据支持时才实现查表/成本公式替换，禁止用两个独立带宽数字直接宣布收益。

### 2.1 校准语义审计

实际执行源码中的 `_usable_profile`：相同 GPU name、不同 UUID 的显式 profile **仍被接受**，见 `S/s2-profile-audit.json`。当前 reader 校验 version/name，未按 UUID、CPU core 列表、NUMA 或工具链拒绝失配。行键虽然含 worker 数，但相同线程数不保证相同绑核。此问题影响自动 fetch；本轮固定 fetch1 不受这条自动策略影响。

现存 overlap 校准得 frac=0.512835；按实际 Q16 min-wait 公式，miss=0/1/2/4/10 时选 fetch=0/1/1/2/5。源码的完整链仍比公式多出 ensure、publish、GPU experts、merge 及固定开销。

### 2.2 完整 hybrid 单层探针

已运行生产 `OffloadMoELayer._decode_hybrid`：H=2560、I=640、top10、7610 slots、31 workers，唯一 IDs，500 专家轮换；包含 reset、ensure、publish、gather、GPU NVFP4 experts、CPU wait/combine。权重为有限随机合成数据，**不含 router/shared，数值只检查有限性和 miss/fetch 计数，尚非真实模型质量验收**。探针 `S/s2_hybrid_probe.py`，顺序轮输出 `S/s2-hybrid-probe-a.json`；19 秒完成，无外部 GPU 竞争。反向独立进程复测已完成，见下表。

初步结果显示少量 miss 的最优 fetch 需按完整链测量，不能直接使用饱和带宽比例；两次独立进程结果见下表。


### 2.3 首轮结果与决定

两轮均无 GPU 竞争，每个格子各 3×100 次 replay，顺序 A 与逆序 B 的独立进程结果如下（µs）：

| miss | 最低耗时 fetch（A/B） | 该格子耗时 A/B | fixed fetch1 耗时 A/B |
|---:|---:|---:|---:|
| 0 | 0 / 0 | 86.09 / 80.77 | 86.09 / 80.77 |
| 1 | 0 / 0 | 101.71 / 94.73 | 162.64 / 160.41 |
| 2 | 0 / 0 | 185.70 / 185.90 | 229.38 / 220.28 |
| 4 | 1 / 1 | 350.71 / 340.57 | 350.71 / 340.57 |
| 10 | 9 / 8 | 827.53 / 813.76 | 870.95 / 854.63 |

完整 22 格数据保留于 `S/s2-hybrid-probe-{a,b}.json`。这里是 500 个专家轮换、每步重置 metadata 的合成完整 routed 链；不是最终 workload 加权结果，不含 router/shared，未做真实模型数值验收。

重要结论：miss=1/2 的一次性处理，fetch0 比固定 fetch1 快；miss4 的最低值为 fetch1，而现有比例选 fetch2。miss10 的最优格子在 8/9 间变化，不能硬编码。**带宽比例没有代表当前完整临界路径。**

同样重要：探针每步重置 miss，不能衡量缓存未来复用。以 miss1 的本次开销差约 63 µs、后续从 CPU1 到全命中每次约节省 15 µs 粗算，需要约 5 次未来命中才可能回本；这是该合成探针的成本示意，不是生产阈值。

最小后续实现：校准记录完整 `T(m,f)`，GPU 决策查表；再结合专家近期复用/驻留时间估算 fetch 的未来收益与 victim 代价。给 profile 加 UUID/CPU ISA/core list/NUMA/compiler/source fingerprint，失配时保留现有默认策略。**本轮不直接把默认 fetch1 改为 fetch0，也没有宣称整模型加速。**

## 3. CPU 提前发布，重叠 shared expert

假设：当前 shared/gate 在 routed 内部 publish 之前完成，CPU 无法与这段独立工作重叠。

最小方案：保持单 GPU stream，拆分 routed prepare/finish，在 publish 后、可能覆盖 hidden_states 的 routed GPU 运算前执行 shared/gate。审计量化输入、唯一 expert ID、异步缓冲所有权、graph replay 和 CPU 无任务快路。

验收：固定路由/权重下输出 bitwise；CPU=0/1/2/4 路径、graph 更新、重复调用无死锁；整层时延与完整模型收益分别记录。理论收益只来自 CPU 暴露的等待窗口，不把 shared 全部耗时当可消除量。

### 3.1 生命周期审计与最小实现边界

`decode_submit` 在 kernel IO 路径通过 `flag_publish` 发布 activation/ids/weights，随后 `decode_sync` 才 wait/combine。shared/gate 只读原始 hidden，不使用 CPU output，可以放在两者之间；但必须早于可能覆盖 hidden 的 `_expert_gemm`。

已有一个 executor 的固定 IO/flag 缓冲，**不可连续 publish 两个 layer 而不 wait**。图内遵循“一次 publish → 独立 shared → fetch/GPU → 一次 wait/combine”；不增加第二 stream，不改变 top-k/分工/舍入，也不移动 shared 到 routed GPU 之后。

最小接入建议：为 hybrid decode 提供一个 capture 时调用的 independent-work callback，Qwen4 的 shared/gate 在此回调求值；标准 CPU-only、prefill、非 hybrid 路径仍在原位置执行，all-reduce 保持一次。不暴露可被跨层误用的 pending 对象。先以仓库外等价调用链探针验证，未经模型数值门槛不加默认开关。

已运行 `S/s3_publish_probe.py`：同一真实 routed 调用链，加 BF16 F.linear/SwiGLU/shared-gate；单 stream，对 CPU=0/1/2/4、输入更新、graph replay 做 bitwise 检查及交替顺序计时。

### 3.2 首轮结果与决定

无 GPU 竞争的一次进程内 A/B、B/A 交替测量，每个状态各 6×100 replay；5 个路由状态 ×12 次输入/ID 更新的 **60 个 bitwise 案例通过**。单位 µs：

| miss | CPU experts | 原顺序 | 提前 publish | 时延下降 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 107.88 | 108.04 | -0.14% |
| 1 | 0 | 188.79 | 188.73 | +0.03% |
| 2 | 1 | 248.42 | 231.15 | +6.95% |
| 3 | 2 | 284.21 | 294.62 | -3.67% |
| 5 | 4 | 465.03 | 447.80 | +3.70% |

原始数据 `S/s3-publish-probe.json`；CPU0 路径基本无差异，CPU1/4 局部约省 17 µs，CPU2 反而约慢 10 µs。它说明“扩大重叠窗口必然更快”不成立；共享内存/PCIe 竞争与 CPU worker 时序尚未分离。反向新进程 `s3-publish-b` 的 preflight 遇到约 28 GiB 的外部 SGLang 进程，不能完成独立复测。

决定：保留原型和生命周期设计，**不修改生产默认顺序**。尚需独立重复、真实 shared 权重/输入、CPU2 回退原因及整模型 logits/速度验收；不把局部 17 µs×48 当整模型收益。

## 4. RDNA4 packed vector 算术

假设：本地 BF16 GEMV 尚未用 DOT2，native FP8 upcast 仍可见标量转换。分别尝试 BF16 DOT2→FP32 和保持 W8A16 的成对 FP8 upcast。

实验顺序：匹配真实 shape/JIT；确认已有 128-bit 权重读取，避免重复“宽加载”优化；构造最小指令候选、检查 ISA/VGPR/LDS/scratch、独立参考和完整 GEMV。改变累加顺序需重新过真实模型质量门槛。将 BF16 激活量化成 FP8 不属于透明优化。

### 4.1 AsmEvo audit 模式与实测语义

本项使用 [asmevo/SKILL.md](/home/zihaomu/.codex/skills/asmevo/SKILL.md) 的 `audit_only`，冻结原始源码并运行 capability/preflight；**没有执行优化搜索、没有对新指令测速、没有 controller-backed speedup 声明**。第一次 capability 参数 role=clang 被工具拒绝，修正为 compiler 后通过，失败日志保留。

ROCm 10 HIP 编译器针对 gfx1201 实际生成 `v_cvt_pk_f32_fp8` 和 `v_dot2_f32_bf16`；小核 VGPR 分别 5/8，LDS/scratch 均 0。这些是指令语义小核资源，不能推断完整 GEMV 资源。原型及独立 CPU 参考：`S/s4_packed_probe.cpp`；工具记录 `s4-{capabilities,preflight}.json`、ISA `s4_packed_probe.s`、设备结果 `s4-numerics.json`。

在已核实的物理 GPU6 / PCI C3（其预约已登记）上：

- FP8：65536 个字节对 / 131072 个分量，含 1024 个 NaN 分量；全部有限值逐 bit 匹配（包括 signed zero/subnormal），NaN 分类匹配，**0 mismatches**。
- BF16 DOT2：32768 个有限输入案例，对顺序两个 FP32 FMA 有 **6671 个 bit 差异**；对 double 中间参考转回 FP32 有 5600 个差异，最大相对误差约 `2.4637e-4`。256 个特设 subnormal 案例无差异。这说明该累加指令不能按 bitwise 透明替换现有归约。

新的边界发现：现有软件 `e4m3_u8_to_f32` 有意把 0x7f/0xff 解成 ±480，硬件按 NaN 解码。因此即使有限权重路径可一致，也不能未经契约检查全局替换 helper。当前 `e4m3_native_cx` 只接受 CUDA capability；应分开“AMD 有原生 decode 转换”与“所有 FP8 prefill/torch 算子可用”，避免一处布尔开关开启未验收路径。[Clang builtin 原型](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html)、[AMD RDNA4 ISA](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture)。

决定：优先把成对 FP8 转换接到有限值 W8A16 GEMV 的独立候选；DOT2 另走模型精度门槛。完整 kernel/模型收益仍未验收，均未采用。


## 5. 按形状调整 tile / split / warps

FP8 形状 `[N,K]`：`[16384,2560]` ×36、`[2560,6144]` ×48、`[248320,2560]` ×1。先审计固定 1536 CTA 目标；扫描独立 split-K / tile / warps，计主核与归约完整时间。QSA 后续联合扫描主核与 merge，覆盖短/长上下文，不能只测 merge。

验收：每形状正确性、真实轮换权重、独立重复、85 个矩阵加权总成本；记录资源和编译失败；筛选结果不能直接相加到历史 tok/s。

### 5.1 FP8 三类形状实测

首轮末尾有外部进程，`s5-tiles` 整轮性能作废；重新运行 `s5-tiles-repeat` 完成，监控无 GPU 竞争。42 个配置案例均通过独立 FP32 `torch.mv` 参考门槛。每配置分别测 hot 与至少 256 MiB 轮换权重，每组 3×100 graph replay；记录主核与归约完整链。结果单位 µs；配置为 `[BLOCK_N,BLOCK_K,split,warps]`：

| [N,K] | 基线 / 末尾 A/A | 本轮最低配置 | 最低耗时 | 相对首个基线下降 |
|---|---:|---|---:|---:|
| [16384, 2560] | 75.952 / 77.793 | [32, 256, 1, 2] | 72.528 | 4.51% |
| [2560, 6144] | 31.544 / 31.537 | [16, 128, 4, 1] | 31.309 | 0.75% |
| [248320, 2560] | 1091.758 / 1091.904 | [16, 256, 1, 2] | 1026.917 | 5.94% |

85 个矩阵按实际次数加权：**5.3402 → 5.1408 ms，减少 0.1994 ms（3.73%）**。这是 synthetic GEMV 链的加权成本，不是整模型收益。GDN baseline A/A 漂移约 2.4%；output projection 最低值仅约 0.75% 变化，不值得单凭这轮换默认。候选均 scratch=0；更高 VGPR 的候选也可以更快，不能只按寄存器数量挑选。

原始 `S/s5-tile-probe.json`、`s5-summary.json`；失败轮为 `s5-tile-probe-collision.json`。本轮未接入生产、未验收真实模型 logits，不发布新的最优默认参数。

### 5.2 QSA 主核 + merge 联合筛选

使用真实 Q24/KV2/D256/GQA12，context512/7936，包含 -1 padding，轮换 12 层独立 KV；8 组 tile/split/warps（含 A/A），16 个案例全部通过独立 FP32 softmax attention 参考。`S/s5-qsa-probe.json`，无 GPU 竞争。

短上下文最低 `block_n=32,target_splits=16,warps=4`：9.286→9.167 µs/层，约 1.28%，12 层绝对量仅约 1.4 µs/token。长上下文最低仍是原 `16/64/4`，其首尾 A/A 为14.627/14.134 µs；**这轮没有比原配置更好的长上下文候选**。

决定：保留 FP8 的 GDN/lm_head 配置供独立复测；output/QSA 不改默认。按此次覆盖的维度，分块调优的绝对收益有限，不能靠它解释或实现 100 tok/s。


## 6. MTP 多 token 复用

假设：draft 权重和实现存在，但 NVFP4 格式支持、额外显存和 target verify 效率尚未验收。

实验顺序：核验 3101 个 `mtp.*` tensors 的实际 dtype/scale 与 loader；最小加载并运行 draft；原权重下验证 GDN reject/rollback、QSA replay、CPU experts M>1，再试 proposal 1/2/4；最终叠加 dense FP8。若 loader 拒绝该格式，记录准确失败入口和最小适配范围，不能绕过成错误权重。

验收：统计最终 committed tokens/s、接受率、draft/verify/reject 时间、额外显存和 TTFT。proposal 数不是吞吐；未验证 state 回滚不进入性能采用。

### 6.1 实际 checkpoint 与加载结果

纠正原先可能的误判：3101 个 `mtp.*` tensor 中，1536 个是 F8_E4M3 专家权重，其余1565 个是 BF16 dense/scale。`config_groups.group_1` 独立指定 MTP 为 128×128 block-FP8，主模型的 NVFP4 group 不覆盖它。**这份 NVFP4 模型的 draft 并不需要补 NVFP4 专家 reader。** 精确 header 归档 `S/s6-mtp-headers.json`。

在原始 checkpoint 上调用真实 `Qwen4ExpMTPDraft.load`，加载成功，选择 `Fp8BlockMoEMethod / TritonFp8BlockMoEKernel`，resident expert decode 输出 `[1,2560]` 有限。实测加载分配 **2,768,138,752 bytes（约2.578 GiB）**，约4.04秒。原始权重本体专家2.344 GiB + dense0.169 GiB；差额还含运行时/rope 等。目标 embedding 仅作为占位共享引用，本探针没有加载 target，也没有 logits/接受率声明。见 `S/s6-mtp-load.json`。

首个 harness 漏设 `set_rope_device`，已修正；这是实验初始化错误，不是产品 reader 不支持。GPU6 的重复加载曾因外部任务在 preflight 被拒绝，成功加载发生在另一次核实的空闲物理 GPU4 / PCI83。

### 6.2 状态基础验收与剩余边界

原有 `test_gdn_verify` 小形状、Qwen4/QSA 的 graph on/off target verify contract 共 **3 passed / 5 deselected**，ROCm10、无 GPU 竞争，见 `S/sessions/s6-verify/output.log`。verify contract 使用 tiny FP16 模型和 oracle stub；测试允许经过检查的近 tie，不能把它升级成真实 BF16 NVFP4 target + MTP 的正确性证明。

真实 draft 的完整 QSA+HC+MoE **四步 forward 已通过有限值 smoke**，hidden `[1,10240]`、head input `[1,2560]`，使用真实权重、合成 embedding/relay；它不证明 proposal 的语言质量。另直接复用既有 GDN oracle 验证真实头形状 `layers=2,bs=1,Hk=16,Hv=48,D=128`，五种提交长度的 stash/replay bitwise 通过。见 `S/s6-mtp-full-layer.json`，物理 GPU5 / PCIA3，无竞争。harness 的 mRoPE positions 初次只给一维已改为文本三轴；错误记录保留。

**首轮决定：MTP 格式与执行入口可用，值得进入整模型阶段，暂无净吞吐结论。** 完整模型 proposal1/2/4 的接受率、committed tokens/s、真实 BF16 target logits 与完整 reject/replay 仍未验收；这一步需要保持模型运行窗口和联合显存预算，不能以 tiny/stub 测试替代。

预算约束已明确：draft 额外约2.58 GiB，几乎消耗 dense FP8 释放的2.70 GiB。先给 MTP 留出预算，再决定 expert slots；不能叠加“+768 slots”与“完整 resident draft”而沿用原预算。纯 BF16 dense target 要明显减少 expert cache；其 miss 增加必须计入净吞吐。

### 6.3 verify 的实际路径与后续实验

源码确认 `phase=verify` 不是 prefill：target 的 OffloadMoELayer 仍走 `_decode_hybrid`，不会在每次 verify 时走整层专家 materialize。CPU executor 的 `max_tokens` 乘以 `decode_rows_per_req`，已有多行 IO 容量；这些仅为入口审计，不替代多行 CPU 精度测试。

同时有两处不能沿用 M=1 的性能结论：

- dense FP8 在 M=2/3/5 时从 GEMV 改走 W8A16 GEMM，第5项的 M=1 tile 最低值不适用于 verify。
- NVFP4 routed 路径仍为 `fused_experts_decode_nvfp4_marlin`，CTA 按 route 与 N tile 分配；多个 token 命中同一专家时，并没有在一个 CTA 内对多行共同复用权重的显式 GEMM。可能有缓存复用，但不能假设专家读流量按 proposal 数成倍下降。

对应源码：[MoE dispatch](../../python/freetoken/layers/moe.py)、[NVFP4 方法](../../python/freetoken/layers/quantization/moe/nvfp4.py)、[route kernel](../../python/freetoken/kernel/triton/nvfp4_fused_moe.py)、[FP8 dispatch](../../python/freetoken/kernel/triton/fp8_pertensor_linear.py)、[CPU IO 容量](../../python/freetoken/engine/engine.py)。

完整模型阶段固定同一 slots/KV/权重，先无 MTP 对照，再分别 `--speculative-algorithm mtp --speculative-num-steps 1/2/4`。先做32-token精度初筛，再记录真实 accepted/committed 数、draft/post-verify/target-verify 耗时及自然生成吞吐；原有每步单 token 的 forced-history harness 不能直接套在 MTP 上。接受率与 net tokens/s 达标之后，才研究 verify 的多行专家复用。


## 7. 按数据用途区分缓存策略

假设：dense 流式权重、刚更新的专家 slots、GDN state/activation 的复用距离不同，统一默认 hint 可能留下空间。

实验顺序：枚举锁定 ROCm/gfx1201 实际可用的 counters；核验 LLVM/Triton hint 到 GFX12 ISA 的映射；只改一类数据 hint，保持算术和加载布局，分别测热/轮换权重以及组合调用。gfx1201 的 thread trace 不使用不受支持的 ATT perfcounters。

验收：ISA 确认 hint 生效、输出 bitwise、无竞争的配对时间；DRAM/cache 结论必须有适用硬件证据，逻辑 GB/s 不替代物理流量。NT/HT 是 hint，不能承诺驻留。

### 7.1 生成 ISA 与时间对照

在物理 GPU5 / PCIA3、无 GPU 竞争条件下，只改变 FP8 dense 权重的 `tl.load` hint，算术/布局不变。两类 shape ×7个配置（含 A/A），全部与原默认输出 bitwise 相同。hot 与至少256 MiB轮换权重分别测3×100 replay；表中为轮换权重，主核+归约完整时间。

| hint | gfx1201 实际 weight load | [2560,6144] µs | [248320,2560] µs |
|---|---|---:|---:|
| 默认 | `buffer_load_b128`，无额外 modifier | 31.468 | 1093.726 |
| `.ca` | 与默认相同 | 31.470 | 1093.753 |
| `.cg` | `scope:SCOPE_DEV` | 31.472 | 1093.647 |
| `.cv` | `th:TH_LOAD_BYPASS scope:SCOPE_SYS` | 30.606 | 1083.946 |
| `evict_first` | 与默认相同 | 31.509 | 1093.878 |
| `evict_last` | 与默认相同 | 31.478 | 1093.585 |
| 默认 A/A | 与默认相同 | 31.480 | 1093.662 |

本工具链下，`evict_first/evict_last` **没有改变规范化后的任何指令**，不能称为实现了流式/保留缓存策略。`.cv`确有 ISA 改变，两个 shape 首轮约减少2.74%/0.89%；独立复测与整模型尚未验收，也未测专家 slot/state 的组合影响。不能把 NVIDIA 风格 cache modifier 名称直接解释为 GFX12 的相同驻留行为。[LLVM GFX12 modifier 说明](https://llvm.org/docs/AMDGPUModifierSyntax.html)。

证据 `S/s7-cache-probe.json`、`s7-*.s`、`s7-isa-audit.json`。原 probe 的指令摘要正则漏了 buffer/global 类指令，已用后者的完整 ISA 审计纠正；勿使用前者的 instruction hash 判断 load 是否一致。

### 7.2 计数器能力与工具限制

锁定 ROCm10 的 `rocprofv3 --list-avail` 实际列出 gfx1201 的 `GL2C_HIT/MISS`、`GL2C_EA_RDREQ_{32B,64B,128B,256B}`、`TCP_REQ/MISS`、`SQ_WAIT_*` 等 counters，另列出 host-trap PC sampling 和 SPM 类型。**枚举可用不等于采集成功。** gfx1201 的 ATT trace 与 ATT perfcounter 支持也不能混淆。[AMD thread trace 支持表](https://rocmdocs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html)。

在同一镜像中给 PyTorch/Triton 加 `rocprofv3 --pmc`，进程在实际 GEMV 前退出134：`Option 'spirv-expand-step' registered more than once / LLVM ERROR`。此轮没有 counter 样本，不能据此发布 DRAM 带宽或 cache hit-rate 结论。枚举 `S/s7-counters.txt`，失败 `sessions/s7-pmc-cache/output.log`。独立 HIP 流式读写探针已运行，PCI A3 身份和输出值检查通过，CSV 中有4次实际 dispatch；但 `GL2C_HIT_sum / GL2C_MISS_sum / GL2C_EA_RDREQ_sum` **12个值全部为0**。探针每次明确读256 MiB、写256 MiB，这组计数无法反映实际流量，判为**计数有效性未通过**，不能解释成全命中、零访存或硬件不支持。原始 `S/s7-pmc-standalone/counters_counter_collection.csv`。另两次 harness 失败为 PATH 误选 Python SDK hipcc（改用 `/opt/rocm/bin/hipcc`）、profiler 工作目录只读（改为 `/experiment`），均保留日志；两者不属于 GPU 性能问题。

随后使用同一HIP探针验证基础计数器：4次 `SQ_WAVES_sum` 均为2,097,152，恰好等于67,108,864线程÷wave32；`GRBM_GUI_ACTIVE` 也为非零。证据 `S/s7-pmc-basic/counters_counter_collection.csv`。因此不能归纳为“整个硬件计数器不可用”：基础wave计数已通过 sanity check，GL2这组计数仍未通过，Python进程另有LLVM装载冲突。

再用不带 `_sum` 的 `GL2C_HIT / GL2C_MISS` 复测，8个返回值仍全部为0，见 `S/s7-pmc-raw-gl2/counters_counter_collection.csv`；不能只归咎于 `_sum` 派生表达式。其根因仍待定位，本轮不更换已冻结的运行时来制造一组不可比数据。

决定：`.cv`保留为单独候选，默认不变；先打通实际 workload 的 profiler，再按 dense/expert/state 三类做可解释的组合试验。没有证据支持“一次统一改 hint 就能吃满带宽”。

## 执行日志

- 2026-11-05：创建计划，冻结顺序与采用门槛；开始第 1 项。原 11 个修改文件保持不变。

## 首轮后续补测

- 2026-11-06：七项已依次完成首轮研究。第1项在物理 GPU5 / PCI A3 补齐四档整模型筛选与8192边界探针，结果见1.5/1.6；第7项补齐hint ISA、负的GL2计数证据与有效的基础wave计数。跨卡/中断样本不混为配对收益。

## 首轮结论与后续验收入口

七项均已有实测或源码证据，**本轮没有新增通过全部采用门槛的生产默认方案**。每项后续边界如下，不把“研究过”标成“整模型 ready”：

| 顺序 | 本轮结论 | 下一项决定性验收 |
|---|---|---|
| 1 | 多余KV容量存在；8378 slots正向整模型筛选约+0.92%–1.01% | 反向重启、真实logits、完整8K；与MTP重新分配显存 |
| 2 | miss1/2完整单层链更适合即时CPU处理，原带宽比例不代表完整临界路径 | 加入未来专家复用价值，按真实路由分布验证成本表 |
| 3 | CPU1/4可隐藏部分shared时间，CPU2本轮回退 | 复测CPU2原因，再做生产接入与整模型精度/速度 |
| 4 | 原生成对FP8转换语义可用；BF16 DOT2存在舍入差异 | 有限值契约下接入完整GEMV；DOT2单独走模型质量门槛 |
| 5 | FP8三类矩阵加权局部成本-3.73%；QSA收益很小 | GDN/lm_head独立复测及真实模型质量，不改output/QSA默认 |
| 6 | 实际block-FP8 draft可加载并运行；状态基础检查通过 | proposal1/2/4完整target验收、接受率、committed tokens/s |
| 7 | evict hints本工具链未改变ISA；`.cv`局部约0.9%–2.7% | 修复Python profiler冲突、验证GL2计数，再做组合cache策略 |

本轮只更新本执行计划与核心文档入口；原有11个修改文件的SHA256保持一致，生产默认没有改动。实验源码、命令、原始样本、失败日志和数值证据都保留在 `S/`；索引为 `session-index.json`，关键文件指纹为 `research-evidence-sha256.json`。性能监控采用进程/显存/主机pressure采样，未锁定GPU频率；局部最低值仍需独立复测。结束后释放本轮GPU预约，不保留后台推理服务。
