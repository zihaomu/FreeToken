# Qwen3.8 Flash Next / R9700 TP2 专用工作树

创建日期：2026-11-06。用户要求将当前工作拆出独立worktree，交给另一个session实现TP=2。**本次只准备分支、继承当前状态和整理交接，没有实现TP2或启动GPU验收。**

## 位置与继承状态

| 项目 | 值 |
|---|---|
| 本工作树 | `/home/zihaomu/zihaomu_workspace/freetoken_workspace/worktrees/qwen38-tp2-20261106` |
| 独立分支 | `feat/qwen38-r9700-tp2-20261106` |
| 起点commit | `cf32a5364303d71ece7979a6c69857c475555eeb` |
| 来源工作树 | `/home/zihaomu/zihaomu_workspace/freetoken_workspace/freetoken-qwen38-flash-next-4bit-100tps-20261105` |
| 来源分支 | `perf/qwen38-flash-next-4bit-100tps-20261105` |
| 独立实验目录 | `/home/zihaomu/zihaomu_workspace/freetoken_workspace/.freetoken-zihao/qwen38-tp2-20261106` |
| 拆分记录 | `/home/zihaomu/zihaomu_workspace/freetoken_workspace/.freetoken-zihao/qwen38-tp2-split-20261106` |

继承了来源工作树的12个已跟踪文件未提交修改及8份未跟踪研究文档，逐文件SHA256与patch核对一致。来源工作树没有移动、清理或回滚。本分支因此有未提交基线，**不是干净upstream**；继承清单和`inherited.patch`保存在拆分记录目录，供区分原有改动与后续TP2改动。

未自动合入方向2或原生FP8候选工作树的独立改动，也未复制ignored模型、虚拟环境、编译产物和JIT缓存。Git对象和refs仍按worktree机制共享，各工作树有独立文件和index；后续只在本目录修改。

`docs/performance`继承文档中的“只沿方向4/TP2后续备选”描述的是原session的执行范围。**本工作树的新任务是独立实现TP2**；原工作树继续保留其冻结范围。继承文档部分外部相对路径依赖原目录布局，查历史证据优先使用本交接的绝对路径。

## 任务与此前阻塞

目标：同一个Qwen3.8-Flash-Next请求在两张R9700上做真正tensor parallel，先实现正确执行，再对相同条件TP1测量batch1 decode。两实例并发、简单逐层放到两张卡不等于该目标。

模型：`nvidia/Qwen3.8-Flash-Next-NVFP4`，revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`。48层，36 GDN + 12 QSA；H=2560，512 experts/top-10，expert intermediate=640，HC=4、low-rank=320。

2026-11-05已经实际尝试双卡：

- RCCL两rank基础all-reduce通过：1024个float32元素相加，两rank均检查为3.0；报告双向peer access可用。它不代表通信带宽已验收。
- 真实服务启动在NVFP4 kernel选择时报错：`TP > 1 is not supported for this expert format`，没有生成token，没有TP2 tok/s结果。
- 独立加载器复现：`qwen4_exp weight loading supports TP=1 only`。

历史证据已复制到实验目录的`prior-dual-gpu/`，先读其中`result.json`、`loader-repro.json`、`startup-result.json`、`command.json`与`probe.py`。这是原样历史记录，旧设备、路径及run脚本不能直接当新实验配置。

本分支代码入口：

- [weight.py](python/freetoken/models/qwen4_exp/weight.py)：`iter_weights`显式TP=1限制，需定义正确的rank-local切片与融合布局。
- [nvfp4.py](python/freetoken/layers/quantization/moe/nvfp4.py)：`TritonNvfp4MoEKernel.unusable_reason`的`tp_ok=False`；不能只删除检查，还需处理packed权重/scale、expert bank、局部计算和归约语义。
- [qwen4_exp](python/freetoken/models/qwen4_exp/)：核对QSA/GDN head/state、HC、shared expert、PLE、embedding/lm_head及采样语义。
- [distributed](python/freetoken/distributed/)：已有collective抽象；历史尝试通过`--disable-pynccl`选择PyTorch/RCCL路径。
- [offload_cache.py](python/freetoken/moe/offload_cache.py)：cache、CPU/GPU专家分工及bank布局必须与rank-local计算一致。

## 建议实施与验收顺序

1. 先写清各权重、量化scale、GDN/QSA状态和expert bank的分片/复制规则，以及每处collective的含义。
2. 实现loader与NVFP4 rank-local布局，验证切片、scale、gate/up融合顺序和跨rank归约；保留TP1路径。
3. 跑小规模数值与eager正确性，再验收prefill、decode、Graph重放及hybrid/cache。固定token/history比较logits、路由和状态，记录归约顺序带来的容差；不以启动成功代替数值通过。
4. 正确性通过后，双卡测相同模型/输入输出/精度的TP1与TP2，分别记录吞吐、最长token间隔、显存、通信和CPU/GPU分工。100 tok/s是目标，当前没有达成证据。

基线研究总账在[100 tok/s文档](docs/performance/qwen38-r9700-100tps.md)。原始模型位于工作区`models/Qwen3.8-Flash-Next-NVFP4-fc694b54`；选择性dense FP8派生权重在工作区`.freetoken-zihao/qwen38-flash-next-4bit-100tps-20261105/execution-5steps/derived-dense-fp8`。它们没有复制进worktree，使用时只读挂载并记录是哪一种权重。

## 环境、缓存与并行会话边界

先读仓库`AGENTS.md`、`CONTRIBUTING.md`与父工作区`/home/zihaomu/zihaomu_workspace/AGENTS.md`。新AMD实验必须用锁定ROCm10.1/HIP7.16环境，并从项目目录运行`/home/zihaomu/zihaomu_workspace/rocm10-launch.sh`核对foundation。

历史应用镜像：`sha256:6cf7ba2162685f0282ebbe3b3f98dfbb9c06af0663809bef5d72b508ef26c962`；foundation：`sha256:a79c374ab2bc93c0e4358da6c7bda5cb84cf1308e8aff74fc99fc95c511cbd03`。身份为Python3.13.13、Torch2.14.0+rocm10.1.0、Triton3.8、HIP7.16.26385，gfx1201。更换环境后重新验收，原native extension未复制，应在匹配环境中构建。

为TP2单独设置`TRITON_CACHE_DIR`、`TORCH_EXTENSIONS_DIR`、`FREETOKEN_CACHE_DIR`、`XDG_CACHE_HOME`及`TVM_FFI_CACHE_DIR`，使用本任务实验目录的`caches/`各子目录；不修改HOME，不覆盖另一session缓存。日志、产物写本任务`logs/`、`results/`。

历史双卡是物理GPU4/5、PCI83/a3，当时容器序号不同；**不是当前空闲或预约证明**。新运行前检查全部`deepinfer_workspace/artifacts/amd/rocm10/gpu-reservation*.json`及占用，创建独立owner预约，按既有双锁协议管理。物理GPU3禁止计算，不停止其他任务。

底层映射停顿研究已独立整理到`/home/zihaomu/zihaomu_workspace/freetoken_workspace/staging_bug/README.md`。TP2若出现异常等待，保留同PID/同GPU事件与计数，将其和算子/通信性能区分；不要在此分支同时展开驱动修复。用户仍限定底层只读，不改驱动、内核、模块/sysfs/sysctl、全局NUMA/THP或全局队列参数，不开启内核追踪。

本次没有占用GPU、实现TP2、提交或推送。后续实现只改此worktree，不修改原冻结工作树或`staging_bug/archive`。

## 新session可以直接使用的任务

```text
请在 /home/zihaomu/zihaomu_workspace/freetoken_workspace/worktrees/qwen38-tp2-20261106
读取TP2_HANDOFF.md，继续实现Qwen3.8-Flash-Next NVFP4在双R9700上的真正TP=2。
此分支已继承当前研究基线的未提交改动，清单和patch见交接文档；原工作树保持不变。
先设计并记录分片与归约语义，再解决loader和NVFP4 bank/kernel两个已有阻塞，
完成数值、prefill/decode/Graph及hybrid/cache验收后再测TP1/TP2性能。
新实验使用独立日志、缓存和GPU预约，遵守ROCm10.1及底层只读限制。
不要把本次worktree准备当作TP2已经实现或测试通过。
```
