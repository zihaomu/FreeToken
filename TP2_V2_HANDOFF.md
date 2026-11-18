# TP2 v2 工作入口

状态：**P0–P8 完成，两个独立 TP2 正式性能 cohort 全部通过。没有待运行 GPU 作业。**

先读[完整设计、执行记录与最终验收](docs/design/qwen38-tp2-v2.md)，第16节为当前结论，第15节保留逐步实现及失败/修复证据。不要把历史 checkpoint 的等待状态当成现在状态，也不要重新跑已通过的实验来挑样。

## 代码与归属

- base：206a2060ac6257cbf5a530143df95bb1df19906a，v1 工作树仍干净。
- v2 分支：feat/qwen38-r9700-tp2-v2-20261117，当前代码在本 worktree，未自动 commit/push。
- FreeToken 模型规格：models/qwen4_exp/composition.py；共享调度：scheduler/core.py 与 batch_executor.py；逻辑状态/prefix：kvcache/storage.py；专家预算/策略：moe/residency.py 与 native_executor/ExpertPlacement。
- 原生源：python/freetoken/kernel/csrc/native_executor；通用 DeviceRuntime 负责物理执行，复用既有 kernel/AQL/存储/Mover/通信机制。
- 生产 worker 不加载原架构 plugin，rad::Engine 与 rad::Scheduler 符号均为0；实际图/批次/状态/位置与运行证据共同验证归属，不能只靠符号名称判断。

## 正式结果

同一物理GPU1/7、R9700/gfx1201、锁定ROCm10.1、batch/concurrency1、FP8 KV/exact、host expert总12288MiB、32000MiB/卡、prefix关闭、256最终输出。每MTP独立3A/A通过2%门后做A/B/B/A/B/A/A/B四对，全部原始样本保留。

| 模式 | 2K decode tok/s | 32K decode tok/s | 2K/32K配对保留率 |
|---|---:|---:|---|
| MTP0 | 123.02 | 120.45 | 99.22% / 99.11% |
| MTP3 | 210.97 | 197.94 | 100.42% / 99.82% |

所有512/2K/8K/32K case通过95%保速、TTFT/wall≤1.10及2K/32K decode≥100。22次独立启动、528 HTTP请求含A/A及预热，配对文本集合全部相同。32K完整请求吞吐含prefill，约40–45tok/s，不能把decode值当完整wall吞吐。

## 构建与运行身份

- artifact：/home/zihaomu/zihaomu_workspace/freetoken_workspace/.freetoken-zihao/qwen38-tp2-v2-20261117。
- 最新build13/stage08/image15：sha256:3d9473c27363d64f7bb4f9e90b99fef66cab0901798e2270cc664c95a25faedc。
- worker SHA256：240b6f721ae5ddfad84984d213d2c036c71dbe6b002ae753f0bc395a34ce2b42。
- 独立runtime-v2 stage，13项包清单；无原架构plugin。操作入口见[构建与启动](docs/radiance-native.md#build-and-run-the-v2-components)，显式选择native-kernels。
- 1629个非Markdown源码与正式镜像一致。修改产品源后必须重新build/stage/manifest/image及做适用资格，不能继承旧镜像的全部证据。
- 模型只读复用/models/qwen3.8-next-flash-iqr-v4.rad，SHA256 25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0；它是BF16-lineage IQR-V4，不是原NVFP4 bank无损封装。原NVFP4目录仅作匹配tokenizer。
- 不得对运行prefix原地使用objcopy导出section；ELF审计用只读mmap/readelf。

## 功能与交付证据

- CPU共享回归971 passed/1 skipped；取消defer原竞态old/new Controller各2000序列复现/闭合，原GPU失败场景修复通过。
- image15 TP2 MTP0/3 W1/W2各20生成逐ID；32K冷热host prefix、carry/truncate、首chunk取消后2048复用、flush、MTP3 context34816末端全部通过。
- MTP0双rank208 op groups/99787权重行；MTP3为229/101871，全部图/TP span/arena/scratch counts与v1一致。MTP0离线审计01的missing-drafter工具适配失败保留，02通过，未重跑GPU实验。
- 两模式默认包内worker、隐藏开发目录、各full21/API与rank1定向SIGKILL后503/监督退出通过。仅资格fatal signal，不冒称可恢复HIP错误。
- final manifest：p8-final-manifest-01/manifest.json，1018项证据SHA，objective_complete及performance_adopted为true。
- cleanup-proof.json：85个记录的自有GPU PID/starttime身份全部退出，最后容器为空、租约释放，GPU1/7各约58MiB。未停止外部任务。
- 知识库MG098–MG109归档本轮，累计109条/504项证据断言通过（kb-verify-09.json）。

范围是文本TP2B1、FP8/exact、MTP0/3及完整独立prefix/image；未声称构建wheel，旧CPU expert hybrid、多模态、TP4/并发与可恢复HIP故障另行处理。用户明确授权GPU1/HIP3与7仅适用于本任务；未来GPU工作须重新发现PCI/UUID/占用并使用Runtime双锁/foundation/observer/own-only cleanup，物理GPU3继续排除。不覆盖v1、模型或共享驱动配置。
