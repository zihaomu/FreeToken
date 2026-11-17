# Qwen3.8 Flash Next TP2 with row-FP8 dense weights

This optional checkpoint path targets Radeon AI PRO R9700 (gfx1201), ROCm 10.1,
batch 1 and GPU expert offload. It keeps the original NVFP4 expert encoding and
uses BF16 activations. Selecting a derived checkpoint explicitly enables its
FP8 dense weights; the original checkpoint continues to use BF16 dense weights.

The local derivative converts 121 matrices: 36 GDN qkv, 36 GDN z, 36 GDN output,
12 QSA output projections and one LM head. Weights use E4M3 and one FP32 scale
per output row (`max(absmax/448, 1e-12)`). The GDN b/a gates stay BF16.
This weight-only format uses the existing Triton W8A16 kernel.

The first implementation retains the R0w communication and full-K dense
products. GDN qkvz outputs are sliced separately by Q/K/V/Z head group; b/a
outputs follow their own head groups. LM-head weights and scales share the same
vocabulary partition and own compact allocations. K-partial projections,
experimental MoE layouts and Radiance collectives are separate candidates.

## Loading and storage

A derivative can hardlink the original checkpoint shards, write only new dense
FP8 shards and redirect the selected weight/scale keys in
`model.safetensors.index.json`. The reader obeys that index for each tensor,
including when an old BF16 tensor with the same name remains in a shared shard.
Config, index and new shards must be separate writable files; never overwrite a
hardlinked original. The local 121-matrix derivative adds about 2.705 GiB of new
payload without copying the expert or PLE shards.

TP2 accepts row-E4M3 weight-only schemes on the selected projection families.
Missing, incorrectly sized or nonpositive row scales fail loading. Other dense
quantization schemes and combinations with experimental TP options are rejected.

## Serving

Use an exclusively reserved GPU pair and the qualified ROCm environment. In the
container, mount the complete derived checkpoint at `/model` and the source at
`/workspace`, then run:

```bash
python -m freetoken.cli serve \
  --model /model --served-model-name qwen38 \
  --tensor-parallel-size 2 --disable-pynccl --dtype bfloat16 \
  --attention-backend qsa_sparse \
  --quant-backend linear.fp8_tensor=triton,moe.nvfp4=triton \
  --moe-strategy offload --expert-load parallel --moe-cache-size 6144 \
  --ple-backend disk --memory-ratio 0.9 \
  --max-running-requests 1 --cuda-graph-max-bs 1 \
  --num-tokens 33024 --max-seq-len-override 33024 \
  --max-prefill-length 2048 --cache-type naive --text-model-only
```

Choose GPUs by the current PCI/UUID mapping; historical GPU numbers are not a
reservation. The local experiment runner excludes physical GPU3 and unfiltered
HIP ordinal3. The derivative must retain every shard link named by its index,
the newly written dense files, its config and tokenizer files while serving.

## Validation scope

Local qualification on the locked ROCm 10.1 image completed the following checks:

| Check | Result |
|---|---|
| Minimal-source CPU loading/layout/scale regression | 44 passed, 1 skipped, 9 deselected |
| Real-weight GPU operators and dynamic Graph inputs | 64 checks across two ranks passed |
| Minimal-source TP2 / TP1 generation | 31 / 10 requests passed finite, usage and applicable rank checks |
| Main-source normal EOS chat, original BF16 versus derived FP8 | 14 tasks per checkpoint passed manual answer review and rank checks, including 32,858-token context |
| Actual main-source FP8 smoke | Both requests matched the qualified version's complete logits/token hashes |
| Actual main-source original BF16 TP1 fallback | 6 requests passed, including 5 usable normal EOS chat tasks and 32K context |

The normal chat set covers Chinese, English, Python, arithmetic, JSON,
translation and long-context retrieval. Generated Python syntax and matrix
indexing were reviewed; generated code was not executed. Historical repeated raw
prompts also repeat under BF16 and are retained as continuous-generation
regressions, not counted as usable answers. These checks are a bounded quality
screen, not a general accuracy guarantee. Quality-only logits hashes synchronize
with the CPU and cannot provide normal-path performance numbers.

With the same actual 6,144-slot expert bank (8,540,651,520 bytes per rank),
post-Engine initialization Torch allocated memory was 16.665 GiB for BF16 dense
and 14.262 GiB for row-FP8 dense, a reduction of about 2.402 GiB per rank.
The engine resets Torch peak counters internally; this observation does not
measure whole-startup or whole-request peak memory.

Three normal-path BF16 TP2 control boots, 48 requests in total, failed the
predeclared 5% stability gate. Boot-median drift was 17.80% / 11.85% / 5.36% /
32.75% for short Chinese / 2K / 8K / 32K input. Candidate ABBA and formal balanced
pairs were therefore skipped. Normal-path FP8 throughput and a net speedup are
not established, and this remains an explicitly selected checkpoint option.

The qualified scope uses BF16 KV, offload, text-only, MTP=0 and batch 1. TP2
hybrid remains unsupported; the FP8 KV combination has not been qualified.
100 tok/s has not been demonstrated. The implementation uses the existing
FreeToken Triton kernel and adds no package or native-build dependency.
