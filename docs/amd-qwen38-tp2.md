# Qwen3.8 Flash Next TP2 on Radeon AI PRO R9700

Status: implementation and local partition/operator checks are complete; full-model
numerical and Graph qualification is pending. Do not treat this branch as ready
or use the earlier native-backend performance results as its throughput.

## Source and checkpoint

- Base: PR #952, `67456830c59e045ece14c233c28f9c59d378818b`.
- Local branch: `feat/qwen38-r9700-tp2-pr952-20261126`.
- Checkpoint: original Qwen3.8-Flash-Next-NVFP4 safetensors,
  revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`.
- Initial scope: text-only, TP2, expert offload, original NVFP4 routed experts,
  BF16 dense projections and KV, without speculative decoding.
- Runtime: locked ROCm 10.1 image `a79c374ab2bc`, Torch 2.14.0,
  HIP 7.16, Triton 3.8.0, two `gfx1201` R9700 GPUs.

The original PR does not register a Qwen parallel plan. Its CUDA-only PyNCCL
wrapper also fails to compile in this ROCm environment. This branch registers a
Qwen `ParallelPlan`, attaches its `ExpertGroup` to routed experts, and selects
the existing Torch/RCCL implementation automatically on ROCm TP.

The production request fan-out fix from PR #951 is included: rank 0 waits for
every named rank subscription before the server reports ready. The PR #952 base
does not include that prerequisite.

## Partitioning

The checkpoint has 48 layers: 36 GDN and 12 QSA, with hidden size 2560,
512 routed experts, top-10 routing and expert intermediate size 640.

| Component | TP2 layout |
| --- | --- |
| GDN | 8 key / 24 value heads per rank; split q, k, v, z, b, a and conv segments separately |
| GDN state | Local conv channels and recurrent heads, matching the existing state pool's TP layout |
| QSA | 12 query / 1 KV heads per rank; preserve each head's interleaved q/gate rows |
| Attention output | Local input columns, followed by an output sum across ranks |
| Shared expert | Half of gate/up intermediate rows and down input columns, followed by an output sum |
| Routed experts | PR #952 placement and quant-kernel sharding; `[0,320)` and `[320,640)` when equally placed |
| Embedding / LM head | Vocabulary rows split across ranks; existing embedding reduction and logits gather |
| HC, router, QSA indexer, PLE | Replicated |

HC operations and routing receive identical inputs on both ranks after each
attention output reduction. `FREETOKEN_CHECK_SPLIT_EXPERTS=1` checks actual
cross-rank route agreement during eager qualification.

Historical branch `feat/qwen38-r9700-radiance-native-tp2-20261117` at
`206a2060ac6257cbf5a530143df95bb1df19906a` retains the earlier Torch TP implementation.
Its head/conv/vocabulary partitions are useful references. It also records that
HC can amplify BF16 rounding differences: its conservative dense path performs
whole projections with additional gathers. This branch instead implements
local projections and must pass the numerical gate below before acceptance.
Later native TP implementations use IQR weights and a different execution ABI;
they are separate workloads.

## Verification record

All attempts, including failures, are retained outside the source tree under
`.freetoken-zihao/pr952-qwen38-tp2-port-20261126/` in the parent workspace.

| Check | Result |
| --- | --- |
| CPU model loading, partition reconstruction and engine configuration regressions | 190 passed, 1 skipped |
| Existing expert-kernel shard reference matrix, including NVFP4 I640 -> 320+320 prefill/decode | 8 passed |
| Shared bank mapping and cross-process loading, including libc memfd fallback | 4 passed |
| PR #951 delayed-subscriber and timeout tests, artifact-only | 3 passed |
| Original model dense weights loaded on both TP2 ranks | Passed |
| Complete model startup and real requests | Pending |
| Fresh TP1 versus TP2 logits | Pending |
| Uninstrumented Graph requests | Pending |
| End-to-end throughput | Unqualified |

Full-model startup exposed two integration issues addressed in this branch:
the expert-placement meta model constructs RoPE before normal model creation,
so the RoPE device is set earlier; this runtime's portable Python omits
`os.memfd_create`, so shared expert banks fall back to the same libc operation.
`_pinned_tensor` and `_row_store` were rebuilt from this source with the locked
runtime; historical binaries were not copied.

The last full-model attempt reached shared bank creation before the memfd fix.
The subsequent startup was rejected by the resource preflight: external jobs
occupied approximately 30 GiB on every allowed GPU. No model process started in
that attempt. No driver or global memory/queue setting was changed.

## Offline prerequisites completed

The no-GPU preparation was completed in `runs/offline-prep-04`. Its container
exposes no GPU devices, mounts the source and checkpoint read-only, and checks
that Torch has zero visible GPUs and never initializes a GPU context.

| Prerequisite | Result |
| --- | --- |
| Original checkpoint index/header consistency | 299,545 tensors across 11 shards |
| Real checkpoint fusion/partition versus the full local meta model | 794 state tensors per rank, both ranks match exactly in shape |
| GDN state geometry | 5120 conv channels, 8 key / 24 value heads per rank |
| Original routed expert storage | 294,912 tensors checked against H2560/I640 NVFP4 shape/dtype expectations |
| Real FreeToken tokenizer path | Four prompts: 23, 25, 31 and 987 tokens; the last spans two 512-token chunks |
| Pending TP2 eager / TP1 eager / TP2 Graph commands | All three parse with the intended dtype, memory, context and mode settings |
| Numerical acceptance tool | 27 synthetic scenarios and 3 CLI checks pass; missing real model outputs are rejected |

The meta-model parameters occupy a logical 5.334 GiB per rank. This is parameter
storage, not measured peak VRAM: it excludes expert pools, KV, runtime buffers
and allocator/Graph overhead. Header/meta checks do not evaluate checkpoint
payload values or model numerics.

Each future model run now records a contract covering source and native-library
hashes, model metadata, runtime, workload and harness. Checkpoint payload identity
is limited to metadata hashes and shard size/mtime; it is not a full payload hash.
The comparator rejects mixed contracts, verifies every prompt chunk against the
actual tokenizer IDs, and checks the complete vocabulary. It rejects missing
chunks/ranks, shifted inputs/positions, non-finite logits and numeric failures.
The original top-1, rank-equality and KL <= 0.02 gates are unchanged. Synthetic
tool checks are not model accuracy results.

The first offline preparation attempt failed in its draft token-count check.
The preparation now uses FreeToken's actual `TokenizeManager` path; subsequent
layout/prompt checks and the complete final preparation passed. Earlier failed
artifacts are retained.

To repeat the CPU preparation, select a new output phase name:

```bash
cd /home/zihaomu/zihaomu_workspace/freetoken_workspace
python3 .freetoken-zihao/pr952-qwen38-tp2-port-20261126/offline.py offline-prep-05
```

## Remaining acceptance

Use the same original checkpoint, BF16 KV, 4 GiB expert cache per GPU, fixed
8192 KV slots, 512-token prefill chunks, max sequence length 4096 and one active
request. Run arithmetic, one-word factual, code and multi-chunk prompts.

1. Complete TP2 eager startup and all four real requests, checking rank routes.
2. Start a fresh TP1 eager oracle with matching settings. For each final prefill
   position, match exact input IDs and positions; require finite logits,
   identical top-1, and `KL(TP1 || TP2) <= 0.02` nats. Both TP2 ranks must have
   identical gathered logits. These gates were declared before model outputs.
3. Run the same requests using uninstrumented TP2 Graph capture and replay.
   Confirm actual capture and decode replay; the Graph CLI flag alone is insufficient.
4. Measure speed only after correctness, on an exclusive card pair with fresh
   controls. Eager diagnostic hooks copy logits to CPU and are not speed data.

The artifact controller stores commands, PCI/UUID identities, runtime hashes,
resource snapshots, outputs and cleanup per attempt. Existing artifacts are
never overwritten; restart with a new phase name. Under sufficient resources:

```bash
cd /home/zihaomu/zihaomu_workspace/freetoken_workspace
python3 .freetoken-zihao/pr952-qwen38-tp2-port-20261126/run.py eager-06
python3 .freetoken-zihao/pr952-qwen38-tp2-port-20261126/run.py tp1-01
python3 .freetoken-zihao/pr952-qwen38-tp2-port-20261126/offline.py offline-numeric-01 --compare tp1-01 eager-06
# Continue after the numerical, route and generated-text gates pass.
python3 .freetoken-zihao/pr952-qwen38-tp2-port-20261126/run.py graph-01
```

FP8 KV, quantized dense releases, GGUF, vision and speculative decoding are not
qualified by these checks. This document does not change the status of earlier
TP1 or native-backend experiments.
