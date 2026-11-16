# 历史dense FP8配置的32K速度复测

两启动共6次实测decode中位数40.053tok/s，server口径40.044tok/s，TTFT中位数31.248秒；启动稳定性门通过=True。

单张R9700 GPU5/TP1/batch1，实际输入32768、输出256，未启用DFlash/MTP。专家NVFP4、121选定dense checkpoint矩阵row-wise E4M3（加载融合为85个FP8矩阵、2900623360权重字节），BF16激活/BF16 KV，hybrid31workers/fetch1/7610slots，shared调度流、host embedding、disk PLE、Graph、2048prefill chunk、OMP/MKL/OpenBLAS1。使用保留旧工作树，11个历史实验源码hash一致，工作树原有dirty状态保持。

ROCm10.1锁定foundation，Torch2.14/HIP7.16/Triton3.8；原生三扩展导入和设备PCI核对通过。原checkpoint只读使用，不下载；原转换脚本重建121矩阵，代表性M1/M16独立参考检查通过、转换清单与历史算法/shape/节省字节数一致。没有把这些检查当作新完整任务质量验收。

序列上限从历史8192提高到34816，显式KV35072以容纳32K+256；专家slots保持7610。历史NUMA interleave0–7未复现，当前策略保持默认；CPU affinity32–63,96–127。共享host未证明CPU/带宽独占。32K使用当前受控合成archive输入，路由分布依赖内容。因此这是同历史核心配置的32K新测，不是严格历史512/32K因果A/B。

| 启动 | 三次client decode tok/s | client中位 | server中位 | 三次TTFT秒 |
|---|---|---:|---:|---|
| dense32k_boot1 | 40.611, 39.375, 40.950 | 40.611 | 40.603 | 31.283, 31.597, 30.774 |
| dense32k_boot2 | 40.026, 40.081, 27.907 | 40.026 | 40.020 | 30.889, 31.213, 55.937 |

client decode=255/(首个非空文本至terminal finish秒数)。warmup单列保留、不计入中位数；所有慢样本保留，不扣KFD累计量。完整SSE/usage/command/runtime均保留。decode/TTFT启动中位漂移比={'decode_tok_s_median': 1.0146150506077245, 'ttft_s_median': 1.002269066760398}，两项≤1.10才通过本次有限稳定性门；未过则上述数字仅作本批描述。输出文本全部同hash=False。

历史42.03tok/s来自512输入/256输出，原记录没有这套配置的32K结果；不把本轮速度与历史42.03直接解释为context导致的纯回退。所有生产源码、模型原件和默认参数保持；自有GPU容器/预约释放。summary.json保存身份和全部样本，evidence-manifest.json保存证据哈希。

测试后按此前仅保留原始模型的约定，已移除本次新建派生模型的safetensors权重文件（含本目录对原件的硬链接别名），释放约70.72GiB；原始模型未改。转换脚本、转换/index/config清单和所有测速证据保留。`measurement-verification.json`是移除前的校验结果，`payload-cleanup.json`记录清理范围；再次运行模型须在新目录重建。
