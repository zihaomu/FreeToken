# Qwen Flash Next on R9700

The `radiance-native` execution backend runs the bundled Radiance Qwen graph, IQR experts,
memory placement, prefetch, chained dispatch and tensor parallel collectives in one native
worker. FreeToken owns the API, chat formatting, tokenization, streaming and supervision.
Python processes do not allocate the model or hold GPU contexts on this path.

The v2 `native-kernels` backend transfers graph, batch, state and expert-policy ownership into
FreeToken. Its [design and live qualification record](design/qwen38-tp2-v2.md) track the current
implementation. The measurements below belong to `radiance-native` v1.

This backend targets Linux x86_64, Radeon AI PRO R9700 (`gfx1201`), ROCm 10.1, TP1/TP2 and one
active text generation. It loads a Qwen4Exp `.rad` file. The original NVFP4 expert bank cannot
be passed to the IQR kernels. Keep the tokenizer directory matched to the `.rad` vocabulary.
TP2 hybrid CPU expert execution, multimodal inputs and online cache resizing are unavailable.
The existing FreeToken backend remains the default.

## Build

Use a ROCm 10.1 development environment with CMake, Ninja and `g++-14`:

```bash
python -m freetoken.native.build --build-dir /path/to/native-build --jobs 12
```

This builds the source bundled under `freetoken/kernel/csrc/radiance_native`; a separate
Radiance checkout is unnecessary. It stages the worker, five required plugins, licenses and
a source/binary SHA manifest under `freetoken/native/runtime`. An explicit `--prefix` can stage
elsewhere; pass that prefix's `bin/freetoken-native-worker` and `share/radiance` using
`--native-worker` and `--native-home`. `--stage-only` reuses an existing complete build.
Model weights are never copied into the runtime. The staged runtime is excluded from Git;
package data includes it when preparing a deployment distribution after the build.

The tested container configuration passes `/dev/kfd` and `/dev/dri`, the host's render/video
groups and `--security-opt seccomp=unconfined` for native io_uring. Native GPU libraries resolve
from `/opt/rocm/lib`. No host driver or system settings need changing.

## Serve

Choose and reserve currently available cards first. The UUID and PCI vectors must have matching
order and exactly as many entries as the tensor parallel size. The worker checks the actual
visible PCI addresses before loading the model.

```bash
HIP_VISIBLE_DEVICES="$RESERVED_GPU_UUIDS" \
FREETOKEN_NATIVE_EXPECTED_PCIS="$RESERVED_GPU_PCIS_JSON" \
python -m freetoken.cli serve \
  --execution-backend radiance-native \
  --model /models/qwen3.8-next-flash-iqr-v4.rad \
  --tokenizer-path /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --tensor-parallel-size 2 \
  --native-mtp-depth 3 \
  --native-kv-cache-dtype fp8 \
  --native-wire exact \
  --enable-timings-report --enable-cache-report
```

MTP depths `0` and `3`, KV types `i8`, `bf16` and `fp8`, and wire types `exact` and `wht6` are
separate settings. `wht6` compresses communication and can alter outputs. A speed result must
name all three settings. The `flashnext-speed` CLI preset defaults to MTP3, FP8 KV and exact
wire. Use explicit MTP0/INT8 flags for those controls. Performance qualification is reported
separately for each combination.

The defaults allow a 34,816-token context, a 2,048-token prefill chunk and a total 12,288 MiB host
pool across TP2. TP1 control runs for this large container used `--native-host-pool-mib 43000`.
`--native-gpu-memory-mib` fixes a per-card byte budget for comparisons; zero uses auto sizing
with `--native-headroom-mib`. B1 remains the admission limit even when two cards are selected.

Prefix reuse is off by default. `--native-prefix-cache` enables native snapshots and
`--native-prefix-cache-mib N` adds a bounded host tier. Exact hybrid reuse keys pages by the
complete cold prefill chunk: short or changed chunks are recomputed, and the checkpoint and
pages always use the same key chain. Stop strings, EOS, aborts and max-token
limits are handled through the native token protocol and FreeToken's existing detokenizer.

## Inspect and diagnose

`/v1/models` identifies the canonical `.rad` root. `/v1/cache/status` exposes the native engine
configuration and the last completed request's native memory/mover snapshot. These are native
owned allocations; they are not Torch reserved memory or GPU driver VRAM. The owner lifetime
high-water includes device ABI allocations and elastic committed backing across frees/rebuilds;
direct plugin allocations are excluded. Driver VRAM is sampled separately. Expert tier units/bytes
use the atomic published tier mirror; routing hits and logical streamed bytes are cumulative weighted
histogram observations. Prefetch copied bytes are counted; useful/wasted attribution is unknown.
Online legacy cache rebuild/flush requests are refused.

`--native-work-dir /path/to/run` preserves the resolved config, readiness, stderr and exit record.
Orderly shutdown additionally exports the per-rank operator, weight-layout, collective and
memory ledgers. The detailed ledgers can total about 100 MB per boot; use a run directory and
retain it only as long as needed. An abrupt worker death cannot export its final ledgers.
The supervisor fails waiting requests and terminates the service if the native worker dies.

Omitting `--execution-backend radiance-native` selects the existing FreeToken backend and its
original checkpoint/quantization options. Loading a `.rad` file through that backend is not a
fallback. Current measurements, limitations and acceptance decisions are recorded in the
workspace execution plan and its immutable experiment artifacts.

## Optional round accounting

Set `FREETOKEN_NATIVE_ROUND_LEDGER=1` for a diagnostic run to retain producer timestamp
groups and completed speculative verification callbacks in the last request's native snapshot.
It is off by default. Timestamps reuse the scheduler's existing host commit clock; group intervals
include draft, verify, scheduling and waits. They are not GPU kernel durations or SSE intervals.
The ledger separates sampler acceptance, native-produced tokens and tokens sent through IPC;
stop/EOS display trimming can reduce the final API count further. Issued-ahead passes whose
results the scheduler discards are outside the completed-verification counter. Capture the
flag and qualify instrumentation overhead before treating a diagnostic run as performance data.

## Measured delivery

See the [full-stack validation report](performance/qwen38_tp2_radiance_full_stack_speed_report_20261117.md)
and [implementation history](performance/qwen38_tp2_radiance_full_stack_speed_plan_20261117.md).
The final trace-off TP2/B1 control with FP8 KV, MTP3 and exact wire measured 210.07 tok/s
at 2K and 198.84 tok/s at 32K input. This is decode throughput; the 32K full-request
rate including prefill is about 45-46 tok/s. These use the authorized IQR-V4 derivative,
not the original NVFP4 expert bank. The main repeated qualification and final-source
single-boot control are reported separately.

## Build and run the v2 components

V2 stages into its own `freetoken/native/runtime-v2` prefix. Build it from this worktree in the
qualified ROCm environment:

```bash
python -m freetoken.native.build \
  --backend native-kernels --build-dir /path/to/v2-native-build --jobs 12
```

It packages `freetoken-batch-worker`, four kernel/quantizer plugins, licenses, a link map and
source/binary identities. It does not package the original architecture plugin. Keep the stage
with the matching Python package; an older staged worker cannot represent a newer source build.

After reserving a matching GPU UUID/PCI pair, launch with:

```bash
HIP_VISIBLE_DEVICES="$RESERVED_GPU_UUIDS" \
FREETOKEN_NATIVE_EXPECTED_PCIS="$RESERVED_GPU_PCIS_JSON" \
python -m freetoken.cli serve \
  --execution-backend native-kernels \
  --model /models/qwen3.8-next-flash-iqr-v4.rad \
  --tokenizer-path /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --tensor-parallel-size 2 \
  --native-mtp-depth 3 --native-kv-cache-dtype fp8 --native-wire exact \
  --native-gpu-memory-mib 32000 --native-host-pool-mib 12288 \
  --max-seq-len-override 34816 --max-extend-tokens 2048 \
  --native-work-dir /path/to/v2-run \
  --enable-timings-report --enable-cache-report
```

This uses the installed v2 worker by default. The qualified workload is text, batch1, FP8 KV,
exact wire and MTP0/3. TP1 functional controls of this large model used 49152 MiB host memory;
they do not qualify the TP2 budget. `--native-prefix-cache --native-prefix-cache-mib 1024`
enables FreeToken's radix policy and the worker's bounded host snapshot provider. It is off
for the initial speed comparison. Cache flush is supported while idle; online rebuild is
unsupported. No CPU expert hybrid or multimodal execution has been added in v2.

The terminal native snapshot distinguishes authorized, retired and committed steps, proposed/
verified/accepted drafts and final committed tokens. `completion_fence_drained` and
`as_of=final_state_release` identify the completed state lifetime; cumulative routing counters
remain estimates of logical routing, separate from mover copy bytes. Fatal worker loss fails
waiting requests and the service. Internal-rank fatal-signal and recoverable device-error
qualification must be reported separately.

Consult the live design record for per-build evidence. Final image15 TP2 batch1 qualification
passes both independent A/A and balanced v1/v2 cohorts. Decode medians at 2K/32K are
123.02/120.45 tok/s for MTP0 and 210.97/197.94 for MTP3. Paired retention is
99.22%/99.11% and 100.42%/99.82%, respectively. All four input lengths pass the
95% retention and TTFT/wall guards. These are decode rates; full request wall time includes
prefill. Long-prefix, cancellation, context-end, installed-prefix/API and rank1 fatal-signal
gates also pass. The installed prefix/image is qualified; an actual wheel build and
recoverable HIP fault injection are separate work.
