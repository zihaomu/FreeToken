# libavx

The conventional op vocabulary in hand-written x86 SIMD, dispatched at run time by CPU feature.

libavx is a kernel plugin beside `libref` and `libr4d`. libref is the only other host
implementation, and it is a slow reference. libavx implements all 161 of libref's ops with the
same schemas and the same answers to a tolerance, the image generator's included (below).

    cmake -S libavx -B build-avx -DCMAKE_PREFIX_PATH=<radiance prefix> && cmake --build build-avx -j4
    K=build-avx/radiance_home/kernels; R=<radiance prefix>/share/radiance/kernels
    $K/rad-avx-check --ref $R/libref.so --avx $K/libavx.so
    $K/rad-avx-bench --ref $R/libref.so --avx $K/libavx.so --all

It builds on its own as a third party's plugin does (`find_package(radiance)` and the same
`rad_add_plugin`) against an installed radiance, with no ROCm and no core build. A full-tree build
on x86-64 always builds it, because it supplies the engine's host kernels: libref is the oracle and
the engine refuses to serve on it, so an op placed on the host (the n-gram embedding's gather,
every step) needs this plugin.

## The embedding gather

`embed_lookup_q` over an E4M3 table into bf16 is the one host op a GPU deployment runs, wherever
`ngram-placement` leaves the n-gram table for the host gather: read from the container file
(`disk`), or a locked copy in RAM when no device kernel for the op is loaded (`ram`). Where libr4d
is loaded, `ram` is a pinned copy its kernel reads over the link, and this one does not run.

The rows come through `avx_ngram.cpp`, compiled baseline so the process has one of each:

- a row cache shared by every thread (2^19 rows, eight ways, CLOCK replacement);
- an O_DIRECT read through io_uring for each miss, with the ranks dividing a call's misses rather
  than duplicating them;
- a reader thread that fetches the next prefill chunk's rows while the current one runs.

The level's kernel decodes the rows: at AVX-512 the call's 256-entry bf16 table sits in eight
registers and four `vpermt2w` look up 32 codes; below it, one code at a time. The row's layout hook
only checks the table's encoding (E4M3 with one bf16 scale, or the checkpoint's plain bf16) and asks
for no re-layout, because the gather reads rows from the container file as stored.

`rad-avx-check` runs the op over a file-backed table with two ranks and the reader racing, and
compares every row with the file's bytes: 1000 calls at the highest level the CPU executes and 50 at
each other level. The reads, the cache and the reader thread are compiled once, at the baseline, so
the other levels differ only in the decode. Each table is written once per run under a name of its
own.

## Dispatch

Every kernel source is compiled four times into four object libraries (scalar, AVX, AVX2+FMA,
AVX-512) with `AVX_LEVEL` defined and the matching `-m` flags. `avx_vec.h` puts each level in its
own inline namespace so the four symbol sets cannot collide; otherwise the linker picks one
definition of `avx::row_to_f32` for all four tables and the scalar path executes AVX-512.

`avx_dispatch.cpp` and `avx_registry.cpp` decide whether the others may run, so they are compiled
baseline, with no `-m` flags. That is a property of the flags, not the source: a compiler given
`-mavx512f` may auto-vectorise anything. `-march=native` appears nowhere.

Detection uses `__builtin_cpu_supports`, not raw CPUID, because the OS must also enable the state:
a CPU may enumerate AVX-512 while the kernel leaves the ZMM state disabled, and using it then
faults. AVX-512 requires F + BW + DQ + VL, all of which the kernels use.

`avx_isa_force()` is an exported symbol, not an environment variable, so the checker can sweep all
four levels and the engine never reaches it.

## Coverage

Every op `libref` implements, `libavx` implements, so the selector resolves no op of a CPU-only
deployment to libref: Flash-Next's QSA indexer, the DFlash2 drafters' convolution and the image
and audio generators included. `rad-avx-check` fails on any missing op, and reports a row that returns
`RAD_E_UNSUPPORTED` separately from one that is absent: a declining row is worse than none, because
the selector resolves to it and the step fails at issue rather than at declare.

Every `RAD_E_UNSUPPORTED` a libavx kernel returns, beside libref's answer to the same geometry:

| where | libavx refuses | libref |
|---|---|---|
| `attn_paged`, `attn_dense` | head_dim over 2048 | the same (`REF_HD_MAX`) |
| `attn_local_rel` | a right context; head_dim or left context + 1 over 2048 | the same |
| `rope`, `qk_norm_rope`, `qsa_block_key` | an unknown `mode`; for `rope`, an unknown `ladder` | the same |
| `rope` | YaRN with `axial` or a position scale | the same |
| `rope` | Llama 3 scaling beside YaRN, with `axial`, or over multi-component positions | the same |
| `rope` | a rotary half-width over 1024 | the same |
| `had_quant_act_i8`, `had_quant_act_i8g`, `rmsnorm_had_quant_i8`, `gated_had_quant_i8`, `hc_read` | a Hadamard group over 4096 | the same (`REF_HAD_MAX`) |
| `gdn_conv_prep`, `gdn_conv_update` | a convolution history over 16 | the same |
| `gdn_recurrent_update` | a `state_form` other than `per_candidate` or `anchor` | the same |
| `ple_conv`, `causal_conv1d` | a history over 64 | the same |
| `ngram_ids` | `ngram` over 8 | the same |
| `moe_gemm`, `moe_gemm_q`, `moe_gemm_q_mod4` | an `a_order` other than `token` or `sorted` | the same |
| `gemm_nt_q` | parts (`pb1`/`pb2`) over a weight other than MXFP4 and the ParoQuant grids, or on the bias form | the same |
| `gemm_nt_q` | parts at `w2a8` / `w4a8_asym`; a zero-point grid's bias form; `a_sum` at any other grid; `b_ref` beside a weight other than e2m1 | the same |
| `attn_prefix_self` | head_dim over 1024; the int8 score form with descales | the same |
| `all_reduce`, `all_gather` | world size over 1 | serves up to 8 ranks; deliberate, below |
| `embed_lookup_q` | a file without direct I/O; a bf16 gather in a process without io_uring | reads the table as memory; deliberate, see the gather above |

Everything but the last two rows refuses exactly where libref does, so the selector never
resolves a geometry to libavx that libref would have served. The per-entry image ops
(`lane_gather`, `lane_store`, `randn_lanes`, `flow_euler`) and the audio generator's
(`frame_mix_store`, `end_status`, `frame_store`, `win_input`, `win_prep`, `win_finish`, and
`cfg_topk_sample` with a lane history or a suppressed id range) check every entry before writing
any, where libref writes the entries before a bad one; they return libref's code, and refuse (with
`RAD_E_SHAPE`) the entries libref would read or write past an operand for.

`moe_gemm_q` and `moe_gemm_q_mod4` carry layout hooks for Qwen3.8-Flash-Next's codebook experts
(`w4nla8h`, `w4nl64a8h`, `w4nl32a8h`, `w5nl64a8h`; `docs/MOE-W4.md`): a view of several planes
(`codes,table`, `scale,scale.1`) is stored in the joined form of `abi/rad_planes.h`, the bytes
libref's hooks store, and a selection the `dtype` does not name is refused at declare with
`RAD_E_DTYPE`, as libref refuses it. At launch both libraries refuse with `RAD_E_DTYPE` a joined
form of other planes, plain and codebook experts in one call, and a joined scale beside plain codes.

Two things are deliberately narrower than libref and say so at the call site:

- **`all_reduce` / `all_gather` serve world size 1** and refuse anything wider by name. libref's
  implementation is a rendezvous over shared state with a barrier and a generation counter, one per
  process; two plugins each carrying one would make two rendezvous groups for one world, and which
  one a rank joined would depend on selection. libref sits below this plugin in the hierarchy, so a
  tensor-parallel host deployment resolves the collectives there and everything else here.
- **The rows carry no `opd_shape`.** libref publishes one for the whole conventional vocabulary,
  and the ABI makes an op in `docs/OPS.md` describable even when the kernel under test is not. A
  second copy would be a second set of judgement calls about the same ops.

## Correctness

`rad-avx-check` opens both plugins with `dlopen` and links neither, so it checks the `.so` the
engine would load. It checks, in order: schema agreement (a copy of the loader's `schema_same`, run
before any arithmetic, because the loader refuses the whole plugin on one disagreement), coverage,
the numbers, and every ISA level. Each level prints:

    924 checked, 924 passed, 0 failed, 0 declined, 0 skipped, 0 missing.  worst rel_l2 = 3.556e-04
    worst case: rmsnorm M=3 n=645 eps=1e-05 dtype=bf16 wadd=1
    924 rerun on 4 threads, byte for byte
    hand-built: 449 checked, 0 failed
    zero rows: 188 ops checked, 0 failed, 2 not reducible to zero rows

against `rad-kbench`'s tolerance table, ids (i32, u32, i64 outputs) exactly; the worst case differs
per level. Every conversion is also checked exhaustively: all 256 or 65536 bit patterns of each
narrow format, in both directions, bit-identical to libref including NaN, subnormals and negative
zero. Normal draws reach none of those patterns, so a conversion can be wrong by octaves at the
bottom of its range while every geometry passes.

Each case also checks what a value comparison cannot see:

- **Guard bytes.** Every operand and scratch buffer carries 64 bytes of a known pattern on each
  side, checked after the launch, for both plugins. A write past a row's end is otherwise invisible
  until it lands on something the engine reads.
- **Inputs unchanged.** Every input and weight must hold its drawn bytes after the launch; a kernel
  writing into one would corrupt a weight or a shared buffer in the engine.
- **Something was computed.** A case whose libref output equals what was drawn (all-zero weights,
  a sampler stage its parameters switch off) fails as vacuous.
- **Threads.** Every check geometry is below the kernels' OpenMP thresholds, so each case runs a
  second time with every parallel region forced onto four threads through `avx_par_force` (an
  exported symbol, like `avx_isa_force`) and must reproduce the serial run byte for byte, as the
  threading rule in `avx_common.h` promises. `--threads N` changes the team, 0 skips the rerun.
- **Zero rows.** Each op's first case is described at its row count and at twice it; the extents
  that change are set to zero and the op launched with its row parameter at 0. It must return
  `RAD_E_SHAPE` and write nothing, as docs/OPS.md requires and libref does.

A case libref cannot describe or run fails rather than being skipped, because the table is then
wrong. Two forms libref cannot describe have no table case for that reason: `sample_merge_topk`'s
packed (id, logit) words and the bias form of `gemm_nt_q` at `fp8a16` (it requires `a_scale`); the
first is a hand-built case.

The harness draws what libref's descriptions cannot state: integer and 4-bit codes across their
grid (a weight-scale draw truncates them to zero), weight tables (a MoE layer's experts) like any
weight, an operand flagged `RAD_OPD_F_IDX_PREFIX` (a routing histogram's offsets) as a
non-decreasing prefix array, and an E8M0 exponent plane as the power of two nearest each draw.
Sampler rows are its own too: libref describes them as random requests (`RAD_FILL_SAMPLE_PARAMS`,
over a `RAD_OPD_F_IDX_HISTORY` history), which rad-kbench draws; the harness writes each row as a
request with every stage switched on, over a history that repeats itself.

A cold description cannot reach every form of every op, and the hand-built cases cover the rest:
operand views and index patterns a description does not draw, and the forms an optional operand
selects. For example, a description draws `dflash_select`'s codebooks as weights at 0.02, where the
pairwise term is a thousandth of the unary score and never decides a pick; so the walk is checked
cold and the form it computes is checked by hand, with unit-scale codebooks, once with `hp` and
once without. Others cover formats libref does not describe: MXFP4 weights with e8m0 exponents
(one of them 0xFF, NaN), ParoQuant's multi-part GEMM, an fp8 KV cache read with per-head descales,
4-bit experts, activation scale grids finer than the weight's, packed-code stores, and
`gdn_recurrent_update`'s anchor state form, both against libref and against itself: over four steps
the anchor replay must give the per-candidate form's `o` bit for bit, which is what lets greedy
speculation emit the same text as greedy decode.

The codebook experts reach both libraries in the joined form, which no description draws. Their
cases draw canonical planes (codes over the whole code range, the shipped w4nl and w5nl tables,
E4M3 or bf16 scales, the fixed 2^-13 level), store them through each library's own layout hooks,
and run each library on its own stored bytes, every form at four geometries that reach each decode
at each level: AVX-512's lines, the chunk decode, the generic one; the fused and the panel forms;
narrower classes, a dropped slot, a partial row block and odd widths. Besides the checks above,
both libraries must store the same bytes and each unrelayout give the planes back; libref over the
stored form must equal libref over the logical form rad-kbench and the oracle pass, bit for bit;
and both must agree with the core's own decoder (`core/format/encoding.h`, compiled into the
checker in the source tree): the weight decoded with its Hadamard undone against the activation
rotated back, summed in double. The hooks of both libraries must refuse the same mismatched
selections with the same code.

The QSA indexer's hand-built cases are batches a description cannot draw: `cu` not starting at 0
and rows no sequence holds, -1 pages and a sequence whose blocks are all dead, the score plane as a
column slice of its declared width, block keys in the KV cache's rank, ring wrap in the tail store
and the work list, and libr4d's eight few-live selection rows, held to the selection rule itself as
well as to libref. Their outputs start as a sentinel, so an element only one side writes shows.

### Weights of several planes

A weight joined over several planes (docs/OPS.md, Weights of several planes; abi/rad_planes.h) has
no description, so its cases are hand-built: canonical planes with codes across their range, real
scales, zeros at both ends of the code range; both libraries' layout hooks must join them into the
same bytes and invert them back; both launch over the joined form. Each GEMM case is also held to
an answer neither library computes: in radiance's own tree the checker compiles in the core's
affine decoder (core/format/encoding.cpp), dequantises the planes and the activation, and takes a
double product. The cases:

- `logits_gemm` over `fp8_e4m3*bf16[128x128]` (N = 200, K = 320: a partial block row and column)
  and `i8*bf16[1x128]` heads, M 1, 5 and 9, bf16 and f32 activations; `logit_rerank` over both,
  candidates across the shard and both block rows.
- `hc_read` (inject and the final mixer) and `mtp_enter` over E4M3 rows with f32 scales per 128
  columns and per quarter row, bf16 and f32.
- `gemm_nt_q` at `w2a8`, `w4a8_asym`, `w4a16_asym`, `w5a16`, `w4a8i8_asym` and `w5a8i8`, M 1,
  13 and 17 (the small-M paths' blocks and the decoded driver), N not a multiple of 6 or 16, the
  ParoQuant grids in two and three parts; a plain `b_scale` at a grid is refused by both.
- The hooks' refusals, pinned for both libraries: a u4 head, a transformed head, a grid's scale
  and zero over the wrong codes, at the wrong group, out of order, or at a grid without a zero.

At every level the heads' f32 logits agree with libref to 4e-7 and the grids' bf16 outputs to 5e-9
(most of them byte for byte); against the core decoder the logits are within 4e-7 and the
grids within 2e-3, their bf16 rounding.

## Checking another host library

`rad-avx-check` checks any host kernel library against libref, without a GPU and without the
engine:

    rad-avx-check --ref <prefix>/share/radiance/kernels/libref.so --avx path/to/mylib.so

A library not named `libavx` is checked as a host library rather than held to libavx's contract:

- Schemas must agree with libref's wherever both declare an op, since the loader refuses the
  plugin otherwise. Ops libref does not declare have no oracle and are not checked.
- Coverage is reported: the libref ops the library has no host row for are listed, and every table
  and hand-built case for them is left out. A row that returns `RAD_E_UNSUPPORTED` still fails, for
  the reason above.
- Each case runs on the row the selector would choose: the host row whose constraints the
  geometry meets, highest priority first. A case no row admits is counted, not failed. Device rows
  are never launched; the checker passes host pointers.
- The ISA sweep needs `avx_isa_force`, and the threaded rerun `avx_par_force`. Without them the
  library is checked once, at the level it picks for itself, on the thread count it picks.
- The file-backed gather and its init are libavx's own and do not run.

Everything else applies unchanged: the tolerances, guard bytes, unchanged inputs, exact ids, the
vacuity check and the exhaustive conversions (when the library has `cast`).

## Measurement platform

The numbers below are one thread on a Ryzen 9 9950X3D (Zen 5, 16 cores, 128 MiB L3), measured with
`rad-avx-bench --peak`. All four ISA levels execute natively on that part. Re-run the commands above
for another CPU; the ratios are what the rest of this document reasons about.

    FMA scalar     29.3 GFLOP/s      triad L1     129 GB/s   (48 KiB)
    FMA avx       135.0 GFLOP/s      triad L2     157 GB/s   (1.5 MiB)
    FMA avx2      175.0 GFLOP/s      triad DRAM    42 GB/s   (768 MiB)
    FMA avx512    351.4 GFLOP/s

The avx2 and avx512 figures are 2 FMA/cycle at 8 and 16 lanes; avx (no FMA) reaches 4 vector FP
ops/cycle on Zen 5's four FP pipes. The triad counts three arrays (two read, one written) and not a
read-for-ownership the compiler may have elided.

`--peak --threads N` measures the memory ceilings on N threads, and adds a read-only stream, the
ceiling of an op that only reads (the QSA indexer's keys). Each eight-core complex of this part
reaches memory through its own link, narrower for writes than for reads, so the triad hardly moves
with threads while the read stream does:

    read DRAM     39 GB/s  1 thread     53-57 GB/s  8 threads (cores 0-7)     64-66 GB/s  16 threads

The ranges are run-to-run spread on a machine shared with other work.

## Conversion, MB/s of source bytes

`rad-avx-bench --cvt`, through the plugin's own `cast`, same platform.

| pair | scalar | avx | avx2 | avx512 | libref |
|---|---|---|---|---|---|
| f32 → bf16 | 17885 | 31739 | 40436 | 78707 | 1725 |
| bf16 → f32 | 28598 | 23382 | 34531 | 40154 | 990 |
| f32 → f16 | 1245 | 1246 | 138027 | 159337 | 670 |
| f16 → f32 | 2706 | 2706 | 34122 | 39493 | 717 |
| f32 → fp8_e4m3 | 1102 | 1165 | 20619 | 26463 | 623 |
| fp8_e4m3 → f32 | 4512 | 4486 | 8294 | 12545 | 360 |
| f32 → i8 | 4247 | 10699 | 99319 | 164889 | 1071 |
| i8 → f32 | 12088 | 11182 | 17324 | 20526 | 384 |

f16 is software below AVX2 because F16C is a separate CPUID bit that arrived a generation after
AVX; a level called `avx` that required it would fault on the machines it exists for.

## Per op, at model shapes

`rad-avx-bench`, microseconds per launch, median of 7 reps, one thread, cache-warm, same platform.
The geometry is a ~1.5B dense model: n_embd 2048, n_ff 5632, n_vocab 32000, 16 heads of 128. M = 1
is decode, 64 a speculative verify window, 512 a prefill chunk.

| op | geometry | libref | scalar | avx | avx2 | avx512 | rate (avx512) |
|---|---|---|---|---|---|---|---|
| gemm_nt | M=1 N=2048 K=2048 | 1589 | 537 | 345 | 282 | **241** | 34.8 GFLOP/s |
| gemm_nt | M=64 N=2048 K=2048 | 101770 | 30355 | 6958 | 3846 | **2860** | 187.7 GFLOP/s |
| gemm_nt | M=512 N=2048 K=2048 | 816412 | 242612 | 55778 | 30745 | **22870** | 187.8 GFLOP/s |
| gemm_nt | M=512 N=5632 K=2048 | 2244318 | 672900 | 156165 | 87061 | **65176** | 181.2 GFLOP/s |
| logits_gemm | M=1 32000×2048 | 25055 | 8861 | 6808 | 6866 | **5525** | 23.7 GB/s |
| rmsnorm | M=512 n=2048 | 2351 | 433 | 248 | 200 | **87** | 48.2 GB/s |
| silu_mul | M=512 n=5632 | 9971 | 5426 | 1874 | 1281 | **740** | 23.4 GB/s |
| softmax | M=512 n=2048 | 3236 | 2618 | 475 | 306 | **204** | 41.0 GB/s |
| quant_act_fp8 | M=512 n=2048 g=128 | 7745 | 5254 | 4767 | 344 | **251** | 12.5 GB/s |
| had_quant_act_i8 | M=512 n=2048 g=128 | 7362 | 3633 | 731 | 367 | **261** | 12.1 GB/s |
| sample_argmax | M=1 n_vocab=32000 | 23.7 | 14.3 | 2.02 | 2.02 | **1.15** | 111 GB/s |
| sample_topk | M=1 n_vocab=32000 | 2287 | 1067 | 174 | 139 | **77** | |
| attn_paged | q_len=1 hd=128 gqa=4 ctx=2048 | 6.73 | 1.53 | 1.18 | 1.11 | **0.93** | |

The libref column is about libref, a scalar loop with a per-element dtype switch; the ratio only
shows an op is wired. The rate column against the ceilings above is what describes libavx:

- `gemm_nt` at prefill reaches 188 GFLOP/s of a 351 GFLOP/s ceiling (54%).
- `sample_argmax` reaches 111 GB/s of a 129 GB/s L1 ceiling (86%): a single pass over the logits,
  close to the memory system's limit.
- `softmax` at 41 GB/s and `logits_gemm` at 24 GB/s are DRAM-bound. `logits_gemm`'s weight is
  128 MiB in bf16 and fits no cache, so it is a memory benchmark, and 24 against a 42 GB/s
  single-thread triad is the comparison that applies.
- `silu_mul` at 23 GB/s and `rmsnorm` at 48 GB/s have room left.

All numbers are single-thread. The plugin parallelises over the outer dimension with OpenMP above a
work threshold; the bench pins one thread because that ranks a kernel change, while a 32-thread
number measures the memory controller.

### The codebook experts

`rad-avx-bench --moe [--moe-dtype LIST] [--threads N]` runs `moe_gemm_q` over Qwen3.8-Flash-Next's
experts at the model's geometry on one rank: 512 experts in two parity tables at top-k 10, gate_up
`[1280, 2560]` and down `[2560, 640]`, 1.3 GB stored through libavx's own layout hooks. Each call
draws a new routing, so a decode step reads its experts from DRAM as a served one does. GB/s counts
the codes and scales of the experts a call routes to; GFLOP/s is `2 M top_k N K`. One thread on a
core that measured 347 GFLOP/s (FMA, avx512) and a 39.6 GB/s DRAM triad in the same session;
microseconds per call:

| dtype | M | proj | libref | scalar | avx | avx2 | avx512 | rate (avx512) |
|---|---|---|---|---|---|---|---|---|
| w4nl64a8h | 1 | gate_up | 52215 | 17166 | 14371 | 1706 | **591** | 28.6 GB/s |
| w4nl64a8h | 1 | down | 25558 | 8655 | 7351 | 869 | **392** | 21.5 GB/s |
| w4nl64a8h | 4 | gate_up | 209824 | 67939 | 56762 | 6738 | **2427** | 27.1 GB/s |
| w4nl64a8h | 4 | down | 100874 | 33854 | 28751 | 3426 | **1523** | 21.6 GB/s |
| w4nl64a8h | 64 | gate_up | | | | 73779 | **32050** | 19.4 GB/s |
| w4nl64a8h | 64 | down | | | | 37662 | **18215** | 17.0 GB/s |
| w4nl64a8h | 2048 | gate_up | | | | 1109980 | **665963** | 201.5 GFLOP/s |
| w4nl64a8h | 2048 | down | | | | 557618 | **293857** | 228.4 GFLOP/s |
| w4nla8h | 1 | gate_up | 49967 | 18048 | 15348 | 1629 | **538** | 31.4 GB/s |
| w4nla8h | 2048 | gate_up | | | | 1100388 | **662523** | 202.6 GFLOP/s |
| w4nl32a8h | 1 | gate_up | 54194 | 17841 | 15042 | 1840 | **696** | 25.0 GB/s |
| w4nl32a8h | 2048 | gate_up | | | | 1196490 | **668847** | 200.7 GFLOP/s |
| w5nl64a8h | 1 | gate_up | 51501 | 18977 | 16711 | 3895 | **1202** | 17.5 GB/s |
| w5nl64a8h | 2048 | gate_up | | | | 1149541 | **694854** | 193.2 GFLOP/s |

On eight threads (eight cores, an 8-thread DRAM triad of 45.8 GB/s) the served form reads 54.4 /
53.6 GB/s at M = 1 (gate_up / down) and 55.1 / 53.4 at M = 64, and computes 1529 / 1607 GFLOP/s at
M = 2048, 55-58% of eight cores' FMA peak. Decode is then at the memory system's limit: the triad
also writes, and the read stream is faster than it.

- **Decode on one thread** reaches 72% of the triad on gate_up, and the decode is what limits it
  (eight vpermps and seven shifts per 128 codes): cache-warm, the same call reads about 30 GB/s.
  down's rows are 320 bytes, so the per-row work (scales, horizontal sums, stores) weighs more. At
  M = 64 most experts take one or two rows, and each 64-row item is mostly overhead around a short
  decode.
- **u5 decodes at half u4's rate**: a code straddles bytes, so each sixteen take a broadcast, a
  vpshufb, a variable shift and a two-table vpermt2ps. w5nl64a8h is not the served form.
- **AVX2** decodes through two 8-entry vpermps and a blend per eight codes; it is 2-3x the AVX-512
  time on one thread and the same on eight, where both wait on DRAM.
- **Prefill** reaches 58-66% of the FMA peak, gemm_nt's share: the 3 x 8 register block over an
  L2-resident panel decoded once per 64 rows, one horizontal sum per output.
- The scalar and AVX levels decode a row through the table and take the generic dot; they are the
  correctness baselines.

### The QSA indexer and DFlash2's convolution

Qwen3.8-Flash-Next's indexer (`arch/qwen4exp_fp8`: 4 heads of 128, ratio 4, topk 512, a 7-row
ring) over 8K, 64K and 200K tokens of context, with every block complete and every query past
them; and the 27B DFlash2 drafter's convolution (`arch/common/rad_block_dflash2.h`: n_embd 5120,
2 taps, groups of 16, a 16-row block). One thread, cache-warm, as above; `blocks` is context / 4.

| op | geometry | libref | scalar | avx | avx2 | avx512 | rate (avx512) |
|---|---|---|---|---|---|---|---|
| qsa_score | M=1, 8K (2048 blocks) | 896 | 269 | 41.6 | 34.0 | **31.4** | 17.2 GB/s |
| qsa_score | M=1, 64K | 7938 | 2159 | 341 | 296 | **225** | 19.2 GB/s |
| qsa_score | M=1, 200K | 24222 | 6662 | 1061 | 944 | **724** | 18.2 GB/s |
| qsa_score | M=4, 200K | 88009 | 23875 | 2277 | 2100 | **1400** | 146 GFLOP/s |
| qsa_score | M=2048, 64K | 14.9 s | 3.92 s | 313 ms | 322 ms | **184 ms** | 187 GFLOP/s |
| qsa_select | M=1, 8K | 6.19 | 6.33 | 5.99 | 1.69 | **1.64** | 10.0 GB/s |
| qsa_select | M=1, 200K | 339 | 135 | 135 | 23.0 | **21.0** | 19.1 GB/s |
| qsa_select | M=2048, 64K | 241 ms | 130 ms | 111 ms | 36.4 ms | **29.9 ms** | 9.0 GB/s |
| qsa_tail_store | 4 sequences, decode | 1.03 | 0.21 | 0.15 | 0.12 | **0.10** | 20.0 GB/s |
| qsa_work | 4 sequences, decode | 4.05 | 0.83 | 0.62 | 0.49 | **0.41** | 20.1 GB/s |
| qsa_work | 2048 rows, 512 blocks | 467 | 93.0 | 67.5 | 51.8 | **40.8** | 25.7 GB/s |
| dflash_conv | T=16 | 538 | 41.6 | 27.1 | 13.4 | **8.28** | 44.5 GB/s |
| dflash_conv | T=64 | 2156 | 166 | 109 | 55.0 | **37.0** | 38.2 GB/s |

The indexer is the only decode op that grows with context, and at decode its keys (256 bytes a
block, 12.8 MB at 200K) fit this part's L3 though no real step leaves them there. So decode is also
measured the way it runs: on a team (`--threads`), with the operands flushed from every cache
before each launch (`--cold`), against the read-only ceiling of the same team. Eight threads on
cores 0-7:

| op | geometry | avx512 | rate | ceiling |
|---|---|---|---|---|
| qsa_score | M=1, 8K | 14.0 | 38.5 GB/s | 53-57 GB/s read |
| qsa_score | M=1, 64K | 84.7 | 51.1 GB/s | 53-57 GB/s read |
| qsa_score | M=1, 200K | 213 | 62.0 GB/s | 53-57 GB/s read |
| qsa_score | M=4, 200K | 278 | 49.7 GB/s | 53-57 GB/s read |
| qsa_score | M=2048, 64K | 25.3 ms | 1360 GFLOP/s | 2680 GFLOP/s FMA |
| qsa_select | M=4, 200K | 47.8 | 33.5 GB/s | |
| qsa_select | M=2048, 64K | 4.94 ms | 54.4 GB/s | |

On all 16 cores (0-15), with another job on the machine, decode over 200K tokens takes 219 us at
M=1 (60 GB/s of a 64-69 GB/s read ceiling) and 277 us at M=4, no faster than on one complex: the
stream is at the memory's limit either way. The prefill chunk takes 13.7 ms (2.5 TFLOP/s).

- `qsa_score` at decode streams the keys at or near the read ceiling from 64K up, with the software
  prefetch 32 blocks ahead (a block table need not be in page order, so the hardware prefetcher
  cannot follow it). At M=4 a block's 16 dot rows are compute the stream does not fully hide. At 8K
  a launch moves 0.5 MB, too little for the stream to reach its rate. One thread reaches 44% of the
  FMA ceiling at M=4 and 56% at prefill: each (row, block group) adds a 45-instruction lane
  reduction and the accumulators' setup to its 128 FMAs.
- `qsa_select` is linear in the blocks, under 1 ns a block on one core, a row a thread; at decode a
  row is one thread's work, a tenth of the scoring before it.
- `qsa_tail_store` and `qsa_work` copy at most a few hundred key rows a step through f32, as libref
  does, so a key lands with the bits libref's would.
- `dflash_conv` is bound by instructions, not by L2 (152 GB/s): the bf16 store rounds in software,
  RTNE with the ABI's NaN rule, about a third of each vector's instructions.

### Weights of several planes, at model size

The decode step's largest reads are an lm head and, in the 2-bit draft heads and ParoQuant's int4
and int5 linears, zero-point grids, all stored joined (docs/OPS.md). At M = 1 to 8 they are memory
benchmarks, so `rad-avx-bench --joined --threads 16` times them on every core (pinned to cores
0-15), each weight over a working set past the 128 MiB L3 (copies of a weight under 1 GiB), beside
a read-only stream of 1 GiB on the same threads, measured before and after the kernels and the
better reading kept: 70.5 GB/s in the run below (the triad: 50). Best of 7, milliseconds, from the
quietest of several runs on a machine shared with other work.

| op | geometry | M = 1 | M = 4 | M = 8 | M = 64 |
|---|---|---|---|---|---|
| logits_gemm | fp8_e4m3*bf16[128x128] 248320x5120 (1.27 GB) | 19.2 (66 GB/s) | 21.2 | | 70 (2.3 TFLOP/s) |
| logits_gemm | i8*bf16[1x128] 248320x5120 (1.29 GB) | 17.9 (72 GB/s) | 19.0 | | 83 (2.0 TFLOP/s) |
| logits_gemm | i8*bf16[1x128] 248320x2560 (0.65 GB) | 10.2 (63 GB/s) | 9.7 | | 34 (2.4 TFLOP/s) |
| gemm_nt_q | w2a8 248320x5120 (0.35 GB) | 5.2 (68 GB/s) | 5.7 | 6.1 | |
| gemm_nt_q | w4a16_asym 34816x5120, 2 parts (93 MB) | 1.8 (51 GB/s) | 1.5 | 2.7 | |
| gemm_nt_q | w4a16_asym 5120x17408 (47 MB) | 1.0 (46 GB/s) | 1.3 | 1.4 | |
| gemm_nt_q | w5a16 34816x5120, 2 parts (115 MB) | 2.6 (44 GB/s) | 4.6 | 5.5 | |
| gemm_nt_q | w5a16 5120x17408 (58 MB) | 1.3 (45 GB/s) | 1.5 | 2.6 | |
| gemm_nt_q | w4a8i8_asym / w5a8i8 34816x5120, 2 parts, M = 256 | | | | 53 / 47 (1.7 / 1.9 TFLOP/s) |

Against the read-only ceiling:

- **The lm heads run at 85-102% of it at M = 1 and 4.** A joined head goes through the GEMM's
  register-blocked driver with a row decoder in its packing step (E4M3 by avx_vec.h's bit
  reconstruction, int8 by widening, then one multiply by the covering scale), so the codes are
  widened six rows at a time into L2 and never in memory. At M = 64 it is compute-bound at 2.0-2.4
  TFLOP/s, 36-43% of the 5.6 TFLOP/s 16-core FMA peak: a decoded weight's M block is the whole of
  M, so the head is decoded once, and the 1.3 MiB activation panel is read again for every six head
  rows.
- **The 2-bit head runs at 96% of it at M = 1 and 81% at M = 8** on the integer path: code planes
  in registers against the activation in plane order, one vpdpbusd a plane where AVX512-VNNI is
  present (vpmaddubsw, vpmaddwd and an add where it is not).
- **ParoQuant int4 and int5 run at 62-72% of it at M = 1** (int4 reached 92-99% in a quieter run),
  and fall to 30-50% by M = 8: every 16 codes decode in registers (int5 by pdep) and meet M
  activation rows with one load and one FMA each, two weight rows sharing each load at AVX-512, so
  from M = 4 the activation loads outnumber the rest of the loop.
- **The prefill grids are compute-bound** at 1.7-1.9 TFLOP/s on the decoded driver, a third of the
  FMA peak: the codes are widened to f32 rather than multiplied as int8.

On a busy machine each figure moves by up to 2x between runs; repeat before comparing two.

## Image generation

Qwen-Image 2.1 (`arch/qwenimage21`) generates on the host backend with `server.kernels: libavx` alone,
no reference library loaded: engine_e2e_test's `qwen_image_generates_on_libavx_alone_as_on_libref`
serves a tiny synthetic pipeline converted to a bf16 and a block-fp8 container, with and without
guidance. A seed repeats its PNG byte for byte, and the picture agrees with libref's at 51-53 dB
PSNR (another seed: 15 dB). The core needed no change for it.

The generator's 18 ops (`avx_gen.cpp`, `avx_gen_attn.cpp`) round where libref rounds, which is
where the upstream bf16 model stores a tensor. The sampler (`randn_lanes`, `flow_euler`), the gated
residual, the lane copies, the shortcuts, `chan_affine`, `timestep_embed` and `to_rgba8` are libref's
bit for bit, held to zero error by the hand-built cases: a seed's noise and every later sampler
state are the same on both host libraries. `flow_euler` keeps libref's two roundings where it
multiplies and adds (`keep`: this plugin is built with `-ffp-contract=fast`). The norms, the
convolution and the attention reduce in other orders and are held to the tolerance table.

- **`conv2d_nhwc`** is GEMMs over bands of patch rows. The weight is re-laid once a launch as
  `[cout][kh][kw][cin]`, so a pixel's patch row is k * k channel runs copied back to back; a band
  of output rows (at least 8 Mi patch elements and eight times the weight) is one `gemm_nt_into`
  call, the bf16 outer-product path where the CPU has AVX512_BF16. A 1 x 1 layer reads x in place;
  a layer narrower than one 48-column panel (the decoder's last, 4 channels) computes y^T so the
  threads split the pixels.
- **`attn_prefix_self`** is flash attention over each (entry, kv head)'s keys, packed once a launch
  into 64-key tiles (the prefix from the paged cache, then the own rows), for kv heads in groups of
  at most 64 MiB of tiles. A task is 32 query positions of one kv head; S = Q K^T and O += P V are
  `vdpbf16ps` broadcasts against the tile (P rounded to bf16, as attn_dense's bf16 form and
  PyTorch's CPU flash attention do) on AVX512_BF16 parts and f32 FMAs elsewhere. The int8 score
  form computes libref's exact integer dot products (codes of at most 127 are exact in bf16).
- `qk_norm_rope` gained libref's `gptj_blocks` (the DiT's adjacent pairs, in whole vectors) and
  `imrope` (the text encoder's) modes, which Qwen-Image needs.

One thread, microseconds per launch, the model's widths (`rad-avx-bench --op <op>`; libref is not
timed past 2e10 FLOP). The bf16 dot-product ceiling is twice the f32 FMA one, about 700 GFLOP/s a
core at AVX-512:

| op | geometry | libref | scalar | avx | avx2 | avx512 | rate (avx512) |
|---|---|---|---|---|---|---|---|
| conv2d_nhwc | 3x3 1152->1152, 1024 px | - | 1530639 | 401811 | 242491 | **48957** | 500 GFLOP/s |
| conv2d_nhwc | 3x3 576->576, 4096 px | - | 1470642 | 350498 | 205767 | **39526** | 619 GFLOP/s |
| conv2d_nhwc | 3x3 up 2 288->288, 16384 px | - | 1465933 | 343962 | 201592 | **40163** | 609 GFLOP/s |
| conv2d_nhwc | 3x3 144->144, 65536 px | - | 1506985 | 349133 | 194474 | **41631** | 588 GFLOP/s |
| conv2d_nhwc | 3x3 144->4, 262144 px | 1066406 | 192806 | 68558 | 45259 | **35668** | 76 GFLOP/s |
| conv2d_nhwc | 1x1 1152->3456, 1024 px | 3259307 | 497989 | 113977 | 63326 | **13314** | 612 GFLOP/s |
| attn_prefix_self | 1024 rows + 256 prefix, 32 x 128, bf16 | - | 736629 | 171322 | 169915 | **48335** | 444 GFLOP/s |
| attn_prefix_self | the same, int8 scores | - | 1024020 | 224256 | 198774 | **66715** | 322 GFLOP/s |
| attn_prefix_self | 4096 rows + 256, 4 x 128, bf16 | - | 1243625 | 287250 | 282482 | **80237** | 455 GFLOP/s |
| ln_mod | 1024 x 4096 | 53696 | 3490 | 1909 | 1347 | **879** | 19.1 GB/s |
| gate_add | 1024 x 4096 | 48096 | 2974 | 2105 | 1444 | **991** | 25.4 GB/s |
| qk_norm_rope | gptj_blocks, 1024 x 64 heads of 128 | 83241 | 9851 | 4963 | 3851 | **3078** | 10.9 GB/s |
| l2norm_chan | 262144 px x 144, silu | 534042 | 100513 | 50344 | 34219 | **23405** | 6.5 GB/s |
| d2s_add | 576 -> 288, 512 x 512 out | 174944 | 47387 | 42418 | 34430 | **29654** | 12.7 GB/s |

- The convolutions reach 70-88% of the bf16 dot-product ceiling at AVX-512 and 58-70% of the f32
  FMA one at AVX2. The 4-channel output layer is f32 dot products over 1296-long patch rows (24% of
  the FMA ceiling); it is 0.06% of the decoder's FLOPs.
- The attention reaches 64% of the bf16 ceiling: the softmax between the two products costs about
  a fifth of a tile at head width 128. The int8 rows above draw their key scales as unit normals,
  so the running maximum moves in most tiles; a model's moves rarely.
- The row ops are not bandwidth-bound: each rounds its row through bf16 two or three times in L1
  (the stores libref models), and l2norm_chan divides and takes an exponential per element.

16 threads (`--threads 16`, the 16 cores pinned; other work was running on the machine,
so these vary by about 15%):

| op | geometry | avx512 | avx2 |
|---|---|---|---|
| conv2d_nhwc | 3x3 at 576, 1152, 288 (up 2) | 3.7-4.1 TFLOP/s | 0.48-0.55 TFLOP/s |
| conv2d_nhwc | 3x3 144->144 at the picture's resolution | 3.1-3.9 TFLOP/s | 0.39 TFLOP/s |
| conv2d_nhwc | 1x1 1152->3456 | 6.3 TFLOP/s | 0.71 TFLOP/s |
| attn_prefix_self | bf16 scores, 1024 and 4096 rows | 4.4-5.9 TFLOP/s | 1.5-1.8 TFLOP/s |
| attn_prefix_self | int8 scores | 3.7-4.4 TFLOP/s | 1.4-1.5 TFLOP/s |
| gemm_nt | the DiT's gate_up, bf16, M 1024 | 5.2 TFLOP/s | 0.64 TFLOP/s |
| gemm_nt_q | the same, fp8a8 (block fp8) | 0.31-0.42 TFLOP/s | 0.26 TFLOP/s |
| ln_mod, gate_add | 1024 x 4096 | 92-101 GB/s | |
| qk_norm_rope | 1024 x 4096 | 33-35 GB/s | |

At AVX2 the convolution and the bf16 GEMM run on the generic driver, which at large M forks one
parallel region per 6- to 16-row block of A: it is a decode-shaped loop, and why their 16-thread
rate there is a fraction of one thread's times sixteen.

**Seconds per denoising step and per image, this machine.** A DiT step is 32 blocks of linears
(13.96 GFLOP a latent row), the attention over the conditioning and the image's rows, and the row
ops (about 2.5 ms a block at 1024 rows). The decoder is about 4.3 TFLOP at 512 x 512 and 17 TFLOP
at 1024 x 1024, plus a few GB of row passes. Estimated from the 16-thread rates above with a 256-key
conditioning and no guidance (guidance doubles the steps' cost):

| picture | container | linears | attention | step | 40 steps + condition + decode |
|---|---|---|---|---|---|
| 512 x 512 (1024 rows) | bf16 | 14.3 TFLOP, 2.8 s | 0.7 TFLOP, 0.15 s | **3.0 s** | **about 2 minutes** |
| 1024 x 1024 (4096 rows) | bf16 | 57 TFLOP, 11 s | 9.3 TFLOP, 1.9 s | **13 s** | **about 9 minutes** |
| 512 x 512 | block fp8 | 14.3 TFLOP, 40 s | 0.2 s | **40 s** | 27 minutes; Turbo's 8 steps 5.5 |
| 1024 x 1024 | block fp8 | 57 TFLOP, 160 s | 2.3 s | **162 s** | 1.8 hours; Turbo's 8 steps 22 |

The fp8 rows are `gemm_nt_q`'s fp8a8 rate, a twelfth to a seventeenth of the bf16 GEMM's here:
its generic driver widens each fp8 weight row to f32 again for every 16-row block of the
activation. With a bf16-pair path for block fp8 the fp8 container would cost about what the bf16
one does. The decoder takes about 1.3 s at 512 x 512 and 5 s at 1024 x 1024, the condition (the
8B text encoder and the DiT over about 100 prompt tokens) about 0.6 s in bf16 and 8 s in fp8.

## Audio generation

MiniMax-Music3 (`arch/minimaxmusic3`) adds 16 ops (`avx_audio.cpp`), and a speech model's frame
loop and causal codec one more (`frame_store`) and these forms: a causal `conv1d`, the causal
`conv_t1d` (`pad` 0, odd strides, a kernel of one stride), SnakeBeta (`snake` with `beta`, both as
logarithms with `log`), and a sampler that suppresses reserved ids, takes each lane's guidance and
penalises a lane's earlier picks (`hist`, cleared at frame 0). The sampler (`cfg_topk_sample`),
the window noise and its overlap binding (`win_prep`), the frame embeddings (`rvq_embed` in both
roundings, `frame_mix_store`, `fourier_embed`), the code history (`frame_store`), the window copies
and the index helpers are libref's bit for bit, held to zero error by the hand-built cases: a seed
gives the same codes, the same noise and the same piece on both host libraries. With a history the
sampler takes its lanes in order, as libref does, since two entries may share a lane's row.
`cfg_topk_sample` and `frame_mix_store` keep libref's two roundings where they multiply and add
(`keep`). The vocoder's three ops reduce in other orders and are held to the tolerance table:

- **`conv1d`** is GEMMs over bands of patch rows, as `conv2d_nhwc`: the weight re-laid once a launch
  as `[cout][k][cin]`, an output row's patch its k input rows' channel runs (zeros outside its own
  sequence), each band one `gemm_nt_into` into f32 rows, then the residual and the tanh or the
  clamp, rounded once. A 1-wide kernel over a dense input reads x in place.
- **`conv_t1d`** is one GEMM per phase: output row q s + r - pad reads input rows q, q - 1, ..
  against taps r, r + s, .. below k (phases of one kernel may differ by a tap: kernel 9 at stride 5
  takes two taps in phases 0 to 3 and one in phase 4), so a phase's rows are a GEMM over
  `[x[q] | x[q - 1] | ..]` patch rows and the weight re-laid per phase as `[cout][taps][cin]`.
- **`snake`** squares a sine reduced to [-pi/2, pi/2] (the sign does not matter once squared) and
  evaluated as an odd Taylor polynomial through r^13; a row with an argument past 8192, where the
  three-part pi stops being exact, is computed again with the libm sine. SnakeBeta's multiplier
  1 / (beta + 1e-9) is taken once a channel, as libref takes it.
- **`snake_aa`** (BigVGAN's alias-free SnakeBeta, MiniMax-H3's vocoder) works on bands of 64 output
  rows, channels innermost: a band computes the upsampled rows its filter reads into scratch (a
  row a function of its index, so neighbouring bands agree where they overlap), with `snake`'s
  sine, and filters them down; both sums run in libref's order.

At the vocoder's third stage (384 channels), bf16, this machine (`rad-avx-bench --op <op>`):

| op | geometry | libref, 1 thread | avx512, 1 thread | avx512, 16 threads |
|---|---|---|---|---|
| conv1d | 7 taps, dilation 3, 384 -> 384, 8192 rows | 10.9 s | 56.8 ms, 298 GFLOP/s | 6.5 ms, 2.6 TFLOP/s |
| conv1d | 1 x 1, 384 -> 384, 8192 rows | 1.57 s | 9.7 ms, 250 GFLOP/s | 1.4 ms, 1.7 TFLOP/s |
| conv_t1d | stride 4, 768 -> 384, 2048 -> 8192 rows | 6.33 s | 34.7 ms, 279 GFLOP/s | 3.8 ms, 2.5 TFLOP/s |
| snake | 8192 x 384 | 20.3 ms | 1.2 ms, 10.5 GB/s | 0.32 ms, 39 GB/s |

A window's vocode (two streams, 689 latents to 705536 samples each, four upsampling stages of
three residual units) is about 2.2 TFLOP, of the order of a second on 16 cores at these rates; the
96-channel last stage was not timed. On one thread the convolutions reach 36-43% of the bf16
dot-product ceiling, against the image decoder's 70-88%; they have not been tuned.

## Video generation

MiniMax-H3 adds eight ops (`avx_gen.cpp`, `avx_video.cpp`, and `snake_aa` in `avx_audio.cpp`)
and these forms: `qk_norm_rope`
`axial` over f32 positions (its positions a view of a wider plane), with absent gains and as the
norm alone; `rope` over f32 positions; `gate_add` without tanh; `add` with `div`; `chan_affine` in
f32; `conv_t1d` at any kernel of a stride or more; `conv1d` with its clamp; and `snake_aa` above.
The sampler step (`flow_x0_step`), the patch unfold, the decoder's tiles (`vtile_gather`), their
stitch into bytes (`vtile_stitch`) and their positions (`grid_pos3`) are libref's bit for bit, held
to zero error by the hand-built cases: a seed gives the same video bytes on both host libraries.
The stitch blends a line of a tile at a time, each product and sum a pass of the line through x's
dtype; the frames that refill the overlap run after the frames that read it, so no row is read
after another task rewrote it. `rms_mod` and its quantised form reduce in another order and are
held to the tolerance table.

Its step-distilled form's video sparse attention adds two (`avx_vsa.cpp`). `vsa_select`'s
selection is libref's bit for bit, since one tile more or less changes what a whole query tile
sees: the tile means sum rows in libref's order with the vector lanes across the head, and each
score's dot runs over the head in libref's order with the vector lanes across tiles and every
product rounded before its add. Its coarse branch is held to the tolerance. `attn_vsa` is one task
per (head, query tile), 64 keys of a visible tile packed at a time for broadcast products and an
online softmax, as `attn_prefix_self`'s f32 form; a prefix query tile sees every tile, so the tasks
are scheduled dynamically.

Its latent upscaler adds six (`avx_video.cpp`). `conv3d_ndhwc` is GEMMs over bands of patch rows, as
`conv2d_nhwc`: the weight re-laid once a launch as `[cout][dt][kh][kw][cin]`, an output voxel's
patch its 27 input voxels' channel runs (zeros outside the volume), each band one `gemm_nt_into`
into f32 rows; a row is then rounded once, or with `res` rounded, its residual added and rounded
again, the residual read before the row is written so it may be y. `group_norm` sums its
statistics in f64 over fixed 64-row blocks combined in block order, so its bytes do not depend on
the thread count, and keeps those sums in its own memory: only `stats[0, 2 groups)` is written.
Both are held to the tolerance table. `tconv_dw` (each tap's product rounded before its add, in
libref's order), `resize_bilinear` (the source rows and weights taken once a launch in libref's
f32 steps), `patch_fold` and `flow_renoise` (`rad_philox_normal_s` a Philox block at a time) are
libref's bit for bit and held to zero error by the hand-built cases. The convolution has not been
tuned: MiniMax-H3 is not served on the CPU.

## Design choices

Each choice below was measured on the platform above. The variants that did not pay are recorded
beside the code.

**Register block** (`gemm_nt` bf16, M=512 N=2048 K=2048; `avx_gemm.cpp`):

    AVX-512, 32 ZMM   4x4 23.2 ms   4x6 22.1 (chosen)   6x4 24.2   8x4 23.8   8x2 35.2   12x2 34.1
    AVX2/AVX, 16 YMM  3x3 34.4 ms   2x6 30.7 (chosen)   2x5 31.9   2x4 34.8   3x4 58.4   4x3 58.1

3×4 and 4×3 at AVX2 fall to 58 ms against 31-34 around them: 12 accumulators plus 3-4 A vectors
plus one B is 16-17 live registers in a 16-register file, and the spill lands inside the k loop.
Wider in N beats wider in M at equal arithmetic intensity: 2×6 and 3×3 both load 8 vectors, for 12
and 9 FMAs (intensity 1.5 either way), and 2×6 is 11% faster.

**Cache-blocking the A panel gains about 6% on this part**, not the several-fold the traffic
argument predicts, and at M=64 it is slightly worse (2.91 ms against 2.85). The 128 MiB L3 serves
the traffic blocking removes. It is kept because it is never much worse and is the real fix on a
part with 8 MiB of L3.

**fp8 decode is branchless exponent reconstruction, not a table gather.** Every fp8 byte is one of
256 floats, but `vpgatherdd` measured 3268 MB/s at AVX2 and 3815 at AVX-512 against 4512 scalar,
because a gather issues one cache line per lane however small the table. The reconstruction
reaches 8294 and 12545 MB/s.

**The fp8 encode finishes in vector registers.** Computing sixteen lanes with vector integer
arithmetic and then spilling them to a stack array for a scalar loop that tests one mask bit each
costs about twice the vector form: on a normalised activation the mask is all ones every time, so
that loop always runs.

**Every conversion is vectorised at AVX2 as well as AVX-512.** A level that falls back to the ABI's
scalar converter costs an order of magnitude more than the vector form, not the 2× its width
suggests, and AVX2 is the level most machines run.

**The scalar top-R selection is single-pass.** The two-pass form (vector max, then find the index)
pays only when the first pass is vector work; at the scalar level it is two passes where one would
do and is slower than libref's. The confirmation pass is vectorised at every level for the same
reason: scalar, it reads n/2 elements on average and is most of what `sample_argmax` costs.

**Peak kernels are compiled per level.** A peak loop compiled baseline measures SSE2 whatever level
it is nominally for, and every kernel then looks excellent against a ceiling an order of magnitude
too low. So the peak kernels live in the plugin, compiled per level like every other kernel. The
accumulator count matters as much: sixteen accumulators plus two constants is eighteen live vectors
and spills a 16-register file, costing more than the FMA saves, so the AVX and AVX2 peaks use ten.

## Layout

    avx_isa.h          the ISA levels and AVX_OP_LIST -- the only place the op set is written
    avx_vec.h          the vector abstraction and the dtype conversions; compiled 4x
    avx_common.h       libref's addressing contract, the scratch arena, the threading rule
    avx_dispatch.cpp   BASELINE: CPUID, the level tables, the thunks, the fp8 table
    avx_registry.cpp   BASELINE: the schemas (libref's, slot for slot) and the kernel rows
    avx_planes.h       weights joined over several planes: the hooks (libref's rules) and readers
    avx_ngram.cpp      BASELINE: the embedding gather's row cache, direct reads and reader
    avx_rows.h         the gather's two halves: the row source and the per-level decode
    avx_table.cpp      one level's table of entry points, expanded from AVX_OP_LIST; compiled 4x
    avx_support.cpp    the generic row, the reductions, the top-R step; compiled 4x
    avx_planes.cpp     a joined row's decode: scaled E4M3 / int8, zero-point u2 / u4 / u5; compiled 4x
    avx_had.h          the Hadamard rotation, the fp8 group, the fp8 fusions' shared epilogue
    avx_{elementwise,norm,hc,quant,gemm,attn,qsa,moe,gdn,sample,misc}.cpp   the kernels; compiled 4x
    avx_gen.cpp        the image generator's norms, sampler, lanes and decoder ops; compiled 4x
    avx_gen_attn.cpp   attn_prefix_self, the generator's attention; compiled 4x
    avx_video.cpp      the video decoder's patch unfold, tiles, stitch and positions, and the
                       latent upscaler's convolutions, norm, resize, fold and re-noise; compiled 4x
    avx_vsa.cpp        video sparse attention: the tile selection and the attention; compiled 4x
    avx_audio.cpp      the audio generator's vocoder, sampler, embeddings and window ops; compiled 4x
    avx_harness.h      plugin loading, row selection, the operand draw, guarded buffers, the
                       geometry table -- shared by both tools
    avx_cases.cpp      the geometries; one table so the check and the bench cannot disagree
    avx_check.h        what the checker's files share: the comparison, the threaded rerun
    avx_check.cpp      rad-avx-check: schemas, coverage, the table, the conversions, the levels
    avx_check_hand.cpp rad-avx-check's hand-built cases
    avx_check_rows.cpp rad-avx-check's file-backed embed_lookup_q case
    avx_bench.cpp      rad-avx-bench
