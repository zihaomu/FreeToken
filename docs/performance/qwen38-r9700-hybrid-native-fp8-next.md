# Qwen3.8 / R9700：hybrid 分工与原生 FP8 转换下一阶段计划

更新：2026-11-06。本文保存[七项研究](qwen38-r9700-seven-step-plan.md)中方向2/4的首轮计划与结果；当前仅执行[方向4稳定性收尾](qwen38-r9700-native-fp8-closeout.md)，方向2已冻结。MTP/投机解码不作为独立任务推进。100 tok/s仍是总目标，本阶段不预先承诺能达到。

## 当前执行范围（2026-11-06收敛更新）

用户最新决定：**当前只收尾方向4；方向2冻结，停止投入，不自动恢复。** 方向2的实现、证据与阻塞已归档到仓库外[冻结文档](../../../doc/qwen38_direction2_frozen_20261106.md)。前一轮方向2、4的实验完成不代表已通过默认采用的全部门槛。

- **当前唯一主线：方向4。** 冻结packed FP8候选、85个矩阵、原有精度/tile/分工/缓存配置，复核独立重启的正反向整模型A/B并定位长停顿。转换、完整GEMV、实际activation、graph、416个logits/有序路由比较和4352个自然输出token的现有通过证据保留；稳定性仍未通过。
- **方向2冻结。** 原有候选默认关闭，不新增实现、实验或组合；下方方向2各节保留为历史记录。仅用户重新授权后恢复。
- 方向4须满足既定完整性能门槛（收益至少1%、95%区间下界大于0、TTFT/p95无超过3%的退化），或得到明确否决当前候选的结果；不靠删慢样本或放宽门槛收尾。
- 系统研究中的新增dense量化、TP=2、QSA FP8 KV及其他kernel路线仅为后续备选，不在本阶段启动。100 tok/s的物理预算作为目标约束，不据此同时扩展任务范围。

本轮方向4收尾已完成6次正式独立启动与4次诊断，稳定性仍未过，默认关闭。停顿暴露于GPU token回读等待；一个GPU时间线案例对应前序decode Graph进度延迟，Graph内部具体节点尚未定位。`launch-gating`控制也有秒级停顿，不能作为修复。完整结果见[方向4收尾文档](qwen38-r9700-native-fp8-closeout.md)。方向2保持冻结。

## 范围与执行顺序

首轮已按 **冻结基线 → 方向2 → 方向4** 完成探索；当前后续顺序以上方收敛更新为准。每个方向独立候选、独立开关、独立证据，只有单项合格后才进入组合验收。

- 方向2先覆盖 batch1、top10、miss=1/2的fetch决策。沿用现有单stream、CPU publish/wait顺序、专家选择的recency排序与LRU淘汰。
- 方向4先覆盖gfx1201上M=1的三类dense W8A16 GEMV。保持BF16激活、FP32累加、原tile/split/归约。BF16 DOT2留作该方向后续子项，先完成原生FP8转换的验收。
- 第一版不叠加8378 slots、提前publish、tile最优值、cache hints或其他量化。M>1预填充、CPU专家格式、NVFP4专家kernel及其他GPU架构走原路径。只计算末token的lm_head预填充也是M=1，属于此dispatch范围，需一同通过质量/TTFT验收。

| 顺序 | 交付 | 完成依据 | 当前状态 |
|---|---|---|---|
| 0 | 冻结本阶段源码、模型、环境及同卡对照 | 可重放，原有修改已登记 | 已冻结并完成同卡质量/性能对照 |
| 2A | 真实路由复用记录与完整成本表 | 校准/验收数据分离，含连续缓存演化 | 已完成首轮校准；收益预测较小 |
| 2B | 小miss分工最小候选 | 决策设备端执行，分工正确，精度通过 | 已实现；生效区间精度有风险，保持关闭 |
| 2C | 方向2整模型A/B | 收益、尾延迟、稳定性达标或明确淘汰 | 完成；有小幅收益，但质量/稳定性不足以采用 |
| 4A | packed FP8转换完整语义门槛 | 高/低半字、布局、特殊值与回退检查通过 | 已通过转换穷举和dispatch门槛 |
| 4B | 三类完整GEMV候选 | ISA、资源、独立参考与真实权重检查通过 | 通过；真实权重/activation与graph重放均已验收 |
| 4C | 方向4整模型A/B | 转换收益能体现在模型上或明确淘汰 | 本轮完成：模型精度通过，自然生成约+2.85%；全量ABBA门槛未过 |
| J | 两个合格候选组合验收 | 实测组合收益，不相加局部数字 | 未组合；hybrid不合格，两个独立候选与证据已保留 |

## 0. 固定对照与证据

复用前一阶段冻结状态：源码`cf32a5364303d71ece7979a6c69857c475555eeb`及已登记的11项本地修改；实际基线以新实验启动时的完整patch和文件SHA256为准。实现时从同一状态创建两个独立实验工作树，不在一个候选里混入另一方向的修改。

模型为revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`的Qwen3.8-Flash-Next-NVFP4及现有selective dense FP8派生权重；BF16激活/BF16 KV、7610 slots、KV=8256、fetch1、31 CPU workers、单stream、batch1、TP1、MTP关闭。锁定ROCm10.1/HIP7.16、Torch/Triton/native扩展与镜像，沿用CPU affinity和NUMA策略。`GPU_MAX_HW_QUEUES`保持unset。

选择同一空闲GPU进行成对实验，记录PCI/UUID、显存和外部进程；竞争轮作废并保留日志。每个候选先有相邻A/A对照，不能直接拿历史约43 tok/s作新候选的分母。profile绑定GPU UUID、CPU型号/ISA/core list、NUMA、worker数、量化格式/shape、缓存配置、工具链和相关MoE实现哈希。

新证据放在仓库外`.freetoken-zihao/rdna4-hybrid-native-fp8-20261106/`，按`baseline/`、`hybrid/`、`fp8/`、`combined/`区分；原研究证据继续只读保留。

实施记录（2026-11-06）：已创建`worktrees/qwen38-hybrid-20261106`与`worktrees/qwen38-native-fp8-20261106`，对应分支`perf/qwen38-hybrid-20261106` / `perf/qwen38-native-fp8-20261106`。主工作树原11项修改指纹一致，候选从同一冻结patch开始。选择并预约空闲物理GPU0 / PCI03 / UUID `GPU-29dcb03ca15afb63`；foundation检查通过，进程内PCI核实为3，ROCm10.1 / HIP7.16。

## 2A. 用真实路由补齐分工决策依据

上一轮的约95 µs（miss1/fetch0）与160 µs（miss1/fetch1）只证明即时完整routed链的成本差；探针每步重置缓存，未计未来命中收益，不能据此直接把生产默认改为fetch0。

1. **采样实际模型路由。** 在expert ID被改写为slot ID之前，记录layer、请求/步序号、唯一expert IDs、命中/CPU/fetch分工；保存trace起点的cache映射、usage和recency。使用有界GPU缓冲，在请求结束后读取，避免每层D2H同步。记录原始路由与部分真实activation用于正确性探针。
2. **区分校准与验收。** 使用不同prompt覆盖不同路由分布；复用间隔和驻留时间仅由校准集估计，验收集固定。当前策略只能使用过去状态；利用未来trace得到的oracle仅作离线上限，不作为生产决策。
3. **补齐成本表。** 在同一设备/绑核下测完整`T(m,f)`：优先细测m=1/2的f=0/1，同时覆盖m=0及3–10的原fallback路径。记录真实NVFP4权重、CPU量化、GPU compute、copy、wait/combine与新增策略开销。保持shared/router顺序；分别报告routed链与完整层/模型耗时。
4. **连续回放缓存演化。** 使用同一初始cache与路由trace，比较fetch1、诊断性“小miss fetch0”和复用感知策略。连续更新缓存，不在每步重置miss。记录未来miss、CPU任务量、fetch字节、被淘汰专家的再次访问、单步尾延迟与累计时间。固定路由回放用于解释机制，真实模型仍需重新生成路由验收。

交付：`routing-reuse-report.md`、trace及起点快照、带指纹的`hybrid-cost-profile.json`、连续回放对照表。若即时收益被后续miss完全抵消，明确否决该策略，不进入复杂预测器开发。

结果记录：真实模型采集已完成（`sessions/routing-a`）。校准集为代码/中文说明，验收集为数学/系统设计，四个不同prompt各256输出。记录每层原始top10 IDs、权重、输入activation、前后分工及初始/最终cache状态；有界GPU缓冲，请求结束后落盘。新增trace kernel与回读开销使该轮明确不具性能资格。

### 2A实测结果（2026-11-06）

四个真实请求各256输出，记录255 decode步骤×48层，共48,960次layer调用。CPU回放逐次匹配原始slots、recency、missing/fetch数、CPU/GPU分工，并完整匹配每个请求最终mapping/usage/recency/step。`hybrid-replay-results.json`保留证明；不使用带trace的模型tok/s作为测速。

提取真实layer0 NVFP4专家银行和真实activation，测完整reset+routed调用链。16个实际routing样本偏热，额外做500个expert ID轮换、独立顺序/逆序两轮；冷轮换的m1/f0约93 µs、m1/f1约163 µs，m2/f0约178 µs、m2/f1约219 µs。冷热工作集对CPU成本影响显著，热微测不能直接作为整模型成本。与全GPU结果相比，混合路径的局部rRMSE最高约0.97%，这里只作为量化路径差异记录，未宣布模型精度通过。

连续回放中，“m1/2一律fetch0”会明显增加后续miss/CPU量；用冷工作集成本加权，两个校准prompt的routed总成本均变差。保留cold-start fetch1的recency阈值扫描，仅使用校准集选择，两个cold成本表取较差结果：最佳已测阈值为128次该层访问，预测校准routed成本约-0.58%～-0.62%；留出prompt分别约-1.98%～-2.07%、-0.47%～-0.59%。这是成本表估计，不是GPU连续回放实测或整模型加速。

下一步仅做该最小候选的实际kernel/模型筛选；目前信号偏小，不展开复杂预测器。阈值、选择过程、输入指纹已冻结在`hybrid-calibration.json`；没有依据留出集重新选阈值。

## 2B. 最小分工实现

只在`miss∈{1,2}`时比较f=0和f=1；m=0及m≥3保持当前行为。复用现有`expert_recency`，对齐其实际计数单位与每层访问间隔，避免把全局cache step直接当token数。保留现有“选哪个miss去fetch”的排序，第一版只改变fetch数量。

选择依据使用相同时间单位：

```text
score(f) = 完整链成本 T(m,f)
           - 预计后续命中节省
           + 预计victim再次访问代价
```

预计项由校准集的少量复用分桶得到，验收时冻结；不照搬上一轮“约5次未来命中回本”的合成示例。victim代价先用trace测得的保守统计值，第一版不为预测额外扫描全cache。只有差值超过实测噪声裕量时才改变fetch1；冷启动、低置信度、profile缺失/失配、未覆盖shape或其他batch回退现有策略。若已有recency无法提供可泛化的预测信号，应停留在负结果，不增加大量新状态。

实施入口：[bench_profile.py](../../python/freetoken/moe/bench_profile.py)、[offload_cache.py](../../python/freetoken/moe/offload_cache.py)、[offload_kernels.py](../../python/freetoken/moe/offload_kernels.py)。成本表在加载时进入固定设备缓冲，决策融合到现有ensure kernel；不增加每层host callback、同步或CPU轮询。每次路由仍恰好由CPU/GPU一方计算，fetch数量遵守现有staging/cap与slot保护规则。

校验重点：CPU=0/1/2路径，graph重复replay及输入更新、cache reset/冷启动、profile失配回退；不改变同一token的唯一expert ID和top-k语义。CPU W4A8与GPU W4A16分工变化会改变数值，必须过真实模型质量门槛，不能仅凭分工计数正确或输出有限判定通过。

实施记录：独立hybrid工作树已加入默认关闭的`FREETOKEN_HYBRID_COST_PROFILE`。加载期核对GPU/CPU/NUMA/线程/native扩展及三个MoE源文件指纹，设备端ensure仅在top10、固定fetch1、miss=1/2且最近missing expert距上次访问超过6144个全局layer step（128次该层访问）时改为fetch0；cold-start仍fetch1。没有新增host callback、publish/wait或cache扫描。

`sessions/policy-metadata`通过16个miss/age边界案例，包括6144/6145、未见expert、graph内更新和重复replay；CPU/GPU分工互斥、recency更新、profile失配及不支持shape回退通过。这里仅证明元数据决策，模型质量另测。`runs/hybrid-quality-reference`已保存32-token A/A（top1=100%、KL=0）及六组64-token参考；服务重启打断了外层launcher收尾，内部complete.json成功，故只作为精度参考，不作性能数据。长样本候选质量与采用决定见下文。

## 2C. 方向2采用条件

先做32-token质量初筛；通过后做六组×64 token的独立验收，使用相同输入history比较logits，top-1一致率≥99%、KL≤0.001，检查有限值。性能测量关闭trace记录，保持生产开销；另保留带统计的诊断轮解释miss/CPU/fetch变化。

整模型512/2048输入、每请求256输出，固定warmup；A→B、B→A成对重启，每配置每个长度至少六个有效样本。分别报告固定history对照和自然生成结果。采用要求：预先声明的目标负载收益≥1%、95% CI下界>0，另一负载无显著回退；TTFT/p95不恶化超过3%，无死锁、越界或非有限输出。新增决策开销计入整链，不从结果中扣除。

无净收益或质量失败则回退方向2，仍继续方向4。交付完整A/B、质量报告、采用/不采用决定。

验收补充：初轮六组64-token结果为bitwise一致、top1=100%、KL=0，但CPU/GPU分工计数也完全相同。源码确认请求间reset会清空recency，因此64-token样本无法覆盖128次访问阈值，不能把它当作该策略生效后的精度证明。已延长为六组256-token固定history，并要求观察分工变化。另修正外部质量driver，使基线/候选都执行同一条screen A/A请求序列；最初不匹配序列的轮次保留且不作采用依据。

`hybrid-quality-matched`重测仍全部bitwise一致，确认短样本没有覆盖策略。第一轮`hybrid-quality-long-reference`因服务在hook更新前已import旧版32/64步骤断言而失败；这是外部采样脚本的长度限制，已保存interpretation与traceback。`hybrid-quality-long-reference-v2`使用支持256步骤的固定hook重新开始。

### 2B/2C长样本质量结果

`hybrid-quality-long-reference-v2`与`hybrid-quality-long-candidate`完成同卡、同请求序列的六组256-token验收（各1536 token），均无外部GPU竞争。候选汇总top1=99.6094%、平均KL=0.00090809，输出有限；原冻结比较器的汇总门槛通过。各prompt top1最低99.2188%。但数学样本平均KL=0.00305667，高于0.001的单样本参考线，不能称每组质量均通过，更不能据此宣布可默认采用。

候选确实生效：每个prompt CPU专家数增加210～346，fetch减少135～260；不再是短样本的无变化通过。保留该质量风险，继续关闭quality hook的A→B/B→A测速，用于判断实际收益是否值得进一步修正；采用决定仍需处理这一单样本偏差。

逐步诊断（`hybrid-quality-diagnosis.json`）：所有prompt前128 token的KL均为0；第一次分工变化与logit变化同时出现在第132～141个零基token索引。数学样本后128 token平均KL=0.00611333，第184个索引KL峰值0.283414。这将差异定位到策略实际改变CPU/GPU量化计算路径之后，不能用未触发策略的前半段稀释风险。

测速补充：首轮512输入均值42.883→43.641 tok/s（+1.77%），只是第一对数据。2048基线有1998.7 ms的单token停顿，同期资源snapshot阻塞1.23 s，CPU/内存PSI无明显压力；尚不能确定监控读取与停顿的因果关系。原ABBA全量保留，额外加入同history的quiet A/B：运行期只读host状态，发现新KFD进程时才读取GPU占用，启动前/结束后完整检查。该监控模式对已有外部进程临时更换GPU的检测弱于持续轮询，因此同时依赖PCI03预约与前后检查，单独标注证据，不静默替换原始结果。

### 2C完整ABBA结果

`hybrid-perf-{a1,b1,b2,a2}`全部完成，每配置每个输入长度六个有效请求、每请求256输出；四次启动均校验11条固定history，未发现外部GPU占用。全部样本原样进入`hybrid-perf-summary.json`，没有删慢样本。

| 输入 | 基线均值 tok/s | 候选均值 tok/s | 配对几何收益 | 请求配对95%区间 | 两次启动对收益 |
|---|---:|---:|---:|---|---|
| 512 | 42.8895 | 43.6705 | +1.8209% | [+1.6207%, +2.0216%] | +1.7685%、+1.8735% |
| 2048 | 40.9572 | 43.4461 | +6.6538% | [-5.7114%, +20.6406%] | +11.9087%、+1.6456% |

512的TTFT均值+0.015%、p95 token间隔-0.935%，速度门槛通过。2048均值受上述2秒停顿影响，区间跨零，不能宣称+6.65%为稳定收益；无异常的反向启动对为+1.65%，仅作诊断。区间按六个请求配对计算，只有两组独立重启，不能推广成所有负载的统计保证。quiet对照和最终决定记录如下；数学样本的精度风险仍在，候选继续默认关闭。

quiet补测（`hybrid-quiet-{a,b}`，各长度各3样本）：512均值42.7354→43.1642 tok/s（+1.00%），2048为42.0648→42.4836（+1.00%）。2048 p95均值25.511→27.217 ms（+6.69%），未达到尾延迟门槛。候选第三个warmup仍有2695 ms停顿；本轮有新外部KFD进程触发的事件式GPU读取，某次snapshot阻塞1.64 s，因此仍不能把外部进程创建/驱动停顿与监控读取分离，更不能宣布因果已定位。

**方向2本轮决定：保留默认关闭的研究分支，不采用默认配置，不进入组合。** 已证实小miss保守分工在512负载有约1%～1.8%的小收益；但实际生效区间的数学样本偏差和长上下文尾延迟风险尚未解决。没有继续用相同验收prompt调整阈值，也未用自然生成的不同路由掩盖质量差异。自然生成采用验收暂不执行；先完成方向4。

## 4A. 先补齐转换契约

上一轮`__builtin_amdgcn_cvt_pk_f32_fp8(i, false)`只验证了低半字的65536种字节对；本轮补齐高半字选择、一个32-bit word的两次成对转换、重排后的K顺序、对齐及N尾部mask。

现有软件helper把0x7f/0xff解成±480，而原生转换将其视为NaN。第一版限定经过一次加载期检查的有限E4M3FN权重；出现这些码值、未覆盖dtype/stride/alignment时整个矩阵走原kernel。验收回退确实生效，不静默改变特殊值契约。检查只做一次并记录启动开销，不在每个token扫描权重。

增加单独的gfx1201 decode转换能力判定，不把全局`e4m3_native()`或`e4m3_native_cx()`改成AMD通用true，避免同时改变prefill、activation量化和torch FP8 dispatch。

准备记录：独立FP8工作树已加入默认关闭的`FREETOKEN_RDNA4_NATIVE_FP8=scalar|packed`，仅Triton W8A16、gfx1201、BF16 M=1、三类已声明shape可启用。加载期检查dtype/连续性/4-byte对齐及0x7f/0xff；不改全局`e4m3_native()`，不改M>1 prefill、W8A8和tile。离线gfx1201编译已生成高/低半字的`v_cvt_pk_f32_fp8`；两个输出使用early-clobber约束，避免第一条指令覆盖第二条仍要读取的packed输入。GPU语义与完整kernel验收结果记录如下。

接口依据：[Triton inline_asm_elementwise文档](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html)。本机编译与后续穷举结果作为实际工具链依据，不以网上main分支代码替代本机验证。

`sessions/fp8-conversion`通过524,288个分量：520,192个有限值与独立CPU PyTorch E4M3FN参考bitwise一致（含符号零/subnormal），4096个NaN分类一致，两个半字均穷举且使用不同配对码检查重排。`fp8-dispatch`通过默认关闭、scalar/packed选择、两个特殊码回退、CPU/shape/stride/对齐/dtype回退和N/K尾部检查。尾部[17,132]的packed FP32部分和最大差9.54e-7，生产shape白名单不会选中该尾部shape；三类生产shape的完整结果见下文。40 MiB有限码扫描一次约0.616 ms。

## 4B. 接入完整dense GEMV

入口：[fp8_pertensor_linear.py](../../python/freetoken/kernel/triton/fp8_pertensor_linear.py)的`_gemv_splitk_kernel`及其dispatch；原[e4m3_compat.py](../../python/freetoken/kernel/triton/e4m3_compat.py)保留既有调用契约。先只支持M=1、连续且满足packed加载对齐的权重，其余走原路径。

覆盖真实`[N,K]`及调用次数：

| shape | 次数/token | 用途 |
|---|---:|---|
| [16384,2560] | 36 | GDN输入投影 |
| [2560,6144] | 48 | attention输出投影 |
| [248320,2560] | 1 | lm_head |

保持原tile/split、BF16激活、FP32乘法/累加和归约顺序，仅替换权重解码。先比较软件转换与成对原生转换；必要时增加标量原生转换作为诊断对照，区分收益来自“使用硬件转换”还是“成对处理”。不把BF16 DOT2或第5项的最优tile混入第一次A/B。

必须核验：生成`v_cvt_pk_f32_fp8`；原有宽加载未退化；没有意外LDS交换或scratch spill；记录VGPR、LDS、scratch和转换指令数。Triton布局重排可能吞掉转换收益，ISA中出现目标指令本身不算成功。

数值顺序：字节转换穷举 → 真实权重/activation的完整GEMV → 模型质量。转换的有限值要求bitwise；完整GEMV以bitwise为首要目标，出现差异先定位归约/FMA/布局变化，并同时检查独立FP32参考（rRMSE≤0.005、max-normalized error≤0.02）。不因达到宽松局部容差而跳过模型门槛。

测速包括主核+归约，分别测hot与至少256 MiB轮换权重，A/A及独立反向重复；报告每shape和按36/48/1加权成本。局部无稳定收益时停止该布局候选；定位转换是否仍在临界路径后才决定是否需要HIP专用实现。

### 4B首轮结果

`fp8-gemv-{a,b}`使用真实checkpoint权重、BF16随机activation、独立FP32 matvec参考，正序/反序两个独立进程；每个shape有前后A/A，计主核+归约，hot与至少256 MiB地址轮换分别保存。以下是轮换工作集结果：

| shape | 软件基线 µs | packed µs | 耗时下降（两轮） |
|---|---:|---:|---:|
| [16384,2560] | 76.28～76.37 | 72.68～72.75 | 4.63%～4.84% |
| [2560,6144] | 31.56～31.60 | 31.01～31.05 | 1.72% |
| [248320,2560] | 1091.73～1091.79 | 1029.62～1029.78 | 5.67%～5.69% |

按36/48/1调用次数加权：约5.352～5.358 ms→5.137 ms，下降4.02%～4.12%，每token局部约节省0.215～0.221 ms。这不是整模型加速；直接按约23 ms decode步折算约1%，但缓存状态和混合负载不同，不能据此设严格上限。标量原生对照的加权下降仅约1.70%～1.76%，因此本轮选择packed进入模型验收。

hot诊断中GDN下降18.63%～19.40%，out下降11.51%～18.21%，lm_head约5.69%；out的hot轮波动较大，且小矩阵可留在GPU缓存中。hot不能代替全模型权重轮换，完整原始timing数组保留，主要选型依据仍为轮换结果和后续模型A/B。

三类shape的BF16输出与FP32部分和均对软件基线bitwise一致；独立FP32参考rRMSE最大2.45e-5。基线/packed的LDS均512 B，VGPR为92/99，无scratch spill；已有LDS重排未增加。权重宽加载保持128 bit（编译器把buffer_load变为global_load），packed静态转换32条，对照scalar为64条；不把整体收益全归因于一条指令。tile/split、dtype、scale和归约保持不变。真实模型activation与整模型质量见后续验收记录。

补充`fp8-real-inputs`：从完整模型实际decode取三类投影各8个activation（核对模型内权重字节/scale与checkpoint重建一致），同一graph更新输入后replay，共24例。所有生产shape的输出和FP32部分和仍与基线bitwise一致，独立FP32参考门槛通过。模型质量参考`fp8-quality-reference`已完成A/A及六组64-token；packed模型候选验收结果见4C。

## 4C. 方向4采用条件

使用与方向2相同的32-token初筛、六组×64 token质量及整模型A/B门槛；方向2策略保持关闭。覆盖fallback、graph replay、三类shape和实际模型。优先验收纯转换的净收益，不能用新tile收益掩盖转换本身无效。

若部分shape回退，允许仅对已经独立证明有收益的shape dispatch；必须保留并报告全部shape结果。完整kernel收益未转化成整模型收益时，候选不升级为默认。

质量结果：`fp8-quality-{reference,candidate}`通过32-token初筛与六组64-token，共416个比较位置，logits和有序expert IDs均bitwise一致，top1=100%、KL=0。`fp8-model-verification.json`核实基线85个矩阵全为mode0，候选36个GDN/48个out/1个lm_head全为mode2；没有因dispatch失效得到假通过。固定history ABBA已完成，hybrid候选始终关闭。

### 4C固定history ABBA结果

`fp8-perf-{a1,b1,b2,a2}`每配置每长度六样本，均完整验证11条history、无持续占用PCI03的外部GPU进程被检测到。使用host/event监控模式。第一对有多秒停顿和明显吞吐波动，第二对较稳定；全部样本均进入统计。

| 输入 | 全量基线均值 tok/s | 全量packed均值 tok/s | 配对几何收益 | 请求配对95%区间 | 第二对收益 |
|---|---:|---:|---:|---|---:|
| 512 | 41.9054 | 37.9288 | -11.6694% | [-32.2650%, +15.1886%] | +2.1744% |
| 2048 | 38.4713 | 39.5305 | +3.6118% | [-3.8771%, +11.6842%] | +2.1169% |

512全量p95均值26.325→32.935 ms（+25.11%）；两个长度的区间均跨零，正式速度门槛未通过。第二对512基线约42.9、packed约43.8 tok/s，提示存在小幅整模型收益，但不能单独取这一对宣布采用。尚未分清共享运行环境、监控/驱动停顿与候选执行之间的因果，不能直接把慢样本全部归为外因。计划内自然生成的独立结果如下，不替换固定history结果。

### 4C自然生成结果与决定

`fp8-natural-{a,b}`使用正常采样，不强制reference token。每个配置包含4个固定warmup、每个输入长度6个测量请求和1个统计请求，共17个请求、4352个输出token。逐请求比较确认4352个输出token ID全部一致；两轮均正常结束，无已检测到的外部PCI03占用。

| 输入 | 基线均值 tok/s | packed均值 tok/s | 配对收益 | 请求配对95%区间 | TTFT变化 | p95 token间隔变化 |
|---|---:|---:|---:|---|---:|---:|
| 512 | 42.8791 | 44.1013 | +2.8504% | [+2.7995%, +2.9013%] | -0.043% | -2.626% |
| 2048 | 42.7224 | 43.9422 | +2.8553% | [+2.8233%, +2.8874%] | -0.069% | -2.588% |

这是一个独立新启动A/B对，每长度有六个请求，区间只描述该对内的请求波动。它与前面稳定的反向对照都支持约2%～3%的小幅收益，但不能覆盖或删除第一对的慢样本。

**方向4本轮决定：保留packed实现作为最优先收口的独立实验候选，默认仍关闭。** 转换、完整GEMV、实际activation、graph更新、模型logits及自然输出均通过精度/功能验收；自然生成有实测收益。全量固定history ABBA未通过原定稳定性门槛，因此尚不声明默认ready。下一轮应针对这个冻结候选，在稳定环境下复核相邻重启对照及长停顿来源；暂不扩大shape或改变精度。当前约44 tok/s，100 tok/s目标未达到。

## J. 组合与交付

分别保留A（现有基线）、B（hybrid候选）、C（FP8候选）、BC。只有单项合格才做对应组合；组合使用相同显存/缓存预算，重新检查质量、完整模型性能及必要的成本profile匹配。不能把B/C局部节省相加作为BC收益。

最终交付：两个可独立review/revert的候选及其证据、原始运行索引、精度报告、采用配置和未通过项。只将通过完整门槛的配置更新进[100 tok/s核心记录](qwen38-r9700-100tps.md)；其他候选明确保留为未采用。每个里程碑完成或失败后直接回填本文对应的“结果记录”。

结果记录：本轮未执行BC组合，因为hybrid候选仍有生效区间的质量风险。两个独立工作树和开关都已保留；FP8更值得继续收口，hybrid应先解决CPU/GPU数值路径差异，再讨论策略。主工作树原11项修改的SHA256一致，候选实现未混入主分支代码。

原始运行入口为`launch.py`，每轮`command.json`/`source.patch`/`resources.jsonl`和`summary.json`保留复现信息。关键汇总：`hybrid-replay-results.json`、`hybrid-calibration.json`、`hybrid-quality-diagnosis.json`、`hybrid-perf-summary.json`、`hybrid-quiet-summary.json`、`fp8-conversion.json`、`fp8-dispatch.json`、`fp8-local-summary.json`、`fp8-real-inputs.json`、`fp8-model-verification.json`、`fp8-perf-summary.json`、`fp8-natural-summary.json`。目录内`README.md`和`session-index.json`提供索引。

收尾：31个session目录均已归档分类，包含旧hook长度断言失败及外层中断恢复记录。候选源码语法、三个工作树diff检查、文档本地链接和原11文件指纹检查通过。实验服务已退出，物理GPU0 / PCI03的预约已释放；`completion.json`记录最终状态。
