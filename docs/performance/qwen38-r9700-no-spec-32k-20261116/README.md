# Qwen3.8 Flash Next: 40.05 tok/s without speculative decoding

This branch preserves the experimental source used for the November 16, 2026
32K measurement. It starts at `cf32a5364303d71ece7979a6c69857c475555eeb` and
captures the existing local changes. All 872 tracked files in the measurement's
source manifest match this snapshot. The original worktree was left untouched.
This is a historical experiment snapshot, with no clean upstream-main A/B result.

## Results and configuration

Single Radeon AI PRO R9700, TP1, batch=1, exactly 32,768 input tokens and 256
generated tokens. DFlash2 and MTP were disabled. Each fresh server received one
warmup followed by three measured requests; all measured samples are included.

| Startup | Client decode samples, tok/s | Median, tok/s | Median TTFT, s |
|---|---|---:|---:|
| 1 | 40.610790, 39.375006, 40.949914 | 40.610790 | 31.283462 |
| 2 | 40.025811, 40.080609, 27.907383 | 40.025811 | 31.212639 |
| All six samples | | **40.053210** | **31.248051** |

The slowest request also had a 55.936890-second TTFT. Startup medians passed the
predeclared 10% drift screen, which is a limited check, not proof of sustained
stability. Output text hashes differ across requests. No new full-model quality
qualification was performed. Client decode speed is
`255 / (terminal finish time - first nonempty text time)`; it excludes prefill.
The pooled server-reported decode median was 40.043881 tok/s.

| Item | Measured setting |
|---|---|
| Original checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Routed experts | Original NVFP4 |
| Selected dense weights | 121 checkpoint matrices, per-output-row E4M3 absmax/448 with FP32 scales; 85 fused runtime matrices, 2,900,623,360 weight bytes |
| Dense selection | `lm_head`, linear-attention `in_proj_qkv`, `in_proj_z`, `out_proj`, and self-attention `o_proj` |
| Activations / KV | BF16 / BF16 |
| MoE | hybrid, 31 CPU workers, fetch=1, 7610 expert slots |
| Scheduling / embedding / PLE | shared scheduler/engine stream, host embedding, disk PLE |
| Graph / prefill chunk | enabled / 2048 tokens |
| Sequence limit / KV capacity | 34816 / 35072 tokens, actual KV page size 64 |
| Runtime | ROCm 10.1.0, Torch `2.14.0+rocm10.1.0`, HIP `7.16.26385`, Triton `3.8.0` |
| Image | `sha256:a79c374ab2bc93c0e4358da6c7bda5cb84cf1308e8aff74fc99fc95c511cbd03` |
| Tested device | physical GPU5, HIP-visible index 7, PCI `0000:a3:00.0`, gfx1201 |
| CPU | affinity 32-63,96-127; OMP/MKL/OpenBLAS threads=1; default NUMA policy |

The historical 512-token experiment used NUMA interleave across nodes 0-7;
this 32K measurement used the default policy. The shared host was not proven
exclusive for CPU or bandwidth. The prompt is a synthetic archive; routing and
performance depend on its contents. These results do not isolate context length
as the cause of any difference from the earlier 512-token measurement.

## Preserved local changes

- `--scheduler-stream shared`: use the engine stream for scheduler metadata,
  preserving the host result pipeline. The default remains `separate`.
- NVFP4 Triton decode: skip inactive GPU slots belonging to CPU routes and write
  zero results, with the backend advertising inactive-slot support.
- Pinned PLE prefetch controls and their existing tests: preserved as part of
  the source snapshot. They were inactive in this run, which used disk PLE.
- Existing CLI documentation, tests, and local research documents are included
  without changing their historical conclusions or enabling later candidates.

Dense FP8 uses the existing W8A16 backend and a converted checkpoint. This
branch does not include the later native FP8 conversion candidate from another
worktree. Model weights and native extension binaries are not committed.

## Recreate the local dense-FP8 checkpoint

Use the pinned runtime above, select an available GPU, and build this branch's
native extensions for that runtime. The original extension hashes are retained
in [source-manifest.json](source-manifest.json). Rebuild native extensions after
changing Python, Torch, ROCm or the GPU architecture.

The original checkpoint must already be available locally. The portable copy
of the converter accepts paths and creates its own unchanged-file hardlinks:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/quantize_qwen38_dense_fp8.py \
  --source /path/to/original-nvfp4-checkpoint \
  --output /path/to/new-dense-fp8-checkpoint
```

The output must be a fresh directory outside the source directory and on the
same filesystem. The script rewrites affected shards and metadata; unchanged
files are hardlinked and must be treated as read-only. Allow about 71 GiB of
additional storage for rewritten weights. The original conversion arithmetic,
M=1/M=16 reference checks and representative microbenchmarks are preserved.
The portable path handling was checked on CPU during packaging; GPU conversion
and inference were not rerun for this branch publication.

The measured temporary derivative was removed after testing. Recreate it before
launching; the original NVFP4 checkpoint is unchanged. The exact measured
converter is also archived as
[historical-quantize-dense.py](historical-quantize-dense.py).

## Serve the measured configuration

Run in the qualified runtime with the converted checkpoint. Select the GPU
explicitly; the historical HIP index is machine-specific. This is the measured
server command with the checkpoint path, executable and port made portable:

```bash
taskset -c 32-63,96-127 python -m freetoken.cli serve \
  --model /path/to/new-dense-fp8-checkpoint \
  --served-model-name qwen38-nvfp4 \
  --host 127.0.0.1 --port 58861 \
  --dtype bfloat16 --attention-backend qsa_sparse \
  --moe-strategy hybrid --moe-cache-size 7610 --moe-cpu-threads 31 \
  --moe-hybrid-max-fetch 1 --memory-ratio 0.9 \
  --max-running-requests 1 --cuda-graph-max-bs 1 \
  --max-seq-len-override 34816 --max-prefill-length 2048 \
  --cache-type naive --num-tokens 35072 \
  --ple-backend disk --embed-weights host --scheduler-stream shared \
  --text-model-only --enable-cache-report --enable-timings-report \
  --moe-collect-stats
```

The exact synthetic prompt and request are included as
[prompt-32768.txt](prompt-32768.txt) and [request.json](request.json). For example,
from this evidence directory:

```bash
curl --no-buffer http://127.0.0.1:58861/v1/completions \
  -H 'Content-Type: application/json' --data-binary @request.json
```

For a timing comparison, restart twice, run one warmup and three measured
requests per startup, verify usage is 32768/256 with zero cached prompt tokens,
and retain every slow sample. The original client and process-control code is
archived as [historical-bench.py](historical-bench.py); it references the original
local plan, launcher and inventory hook and is not a standalone portable runner.

## Evidence

- [Original report](report.md), [all six samples and runtime summary](summary.json).
- [Measurement verification](measurement-verification.json),
  [checkpoint conversion check](conversion-check.json),
  [conversion runtime](conversion-runtime.json).
- [Original plan](plan.json) and exact commands for
  [startup 1](dense32k_boot1-command.json) /
  [startup 2](dense32k_boot2-command.json).
- [Original source hashes](source-manifest.json),
  [snapshot verification](snapshot-verification.json),
  [packaging changes](packaging.json),
  [temporary-weight cleanup](payload-cleanup.json).

Historical JSON paths identify the original local measurement. This branch
packages its source and compact evidence; it does not upload the original model,
full server logs, native binaries or JIT caches. The complete local run remains
under `.freetoken-zihao/qwen38-dense-fp8-32k-20261116-0703/`.
