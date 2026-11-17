/* rad_block_moe_fp8.h -- the routed feed-forward block with block-scaled FP8 experts.
 *
 *   rmsnorm -> quant_act_fp8 -> router gemm -> router_topk -> moe_scatter
 *                            -> gate_up moe_gemm_q -> silu_mul + quant -> down moe_gemm_q
 *                            -> moe_gather
 *                            -> [shared expert: gate_up -> silu_mul + quant -> down -> gate]
 *                            -> add -> [all_reduce] -> residual add
 *
 * rad_block_mlp_fp8.h with the feed-forward routed; that header covers the gate/up fusion, the
 * fused norm/residual/collective and the bf16 activations. This one covers what routing changes.
 *
 * ============================== THE SHARED EXPERT ==============================
 *
 * Qwen3.5-MoE runs a dense expert beside the routed ones on every token, gated by a one-output
 * projection: `y = routed(x) + sigmoid(shared_gate(x)) * shared(x)` (`mlp.shared_expert.*` and
 * `mlp.shared_expert_gate.weight`). `moe_gather` applies the gate to the scalar and multiplies as
 * it sums the routed slots (docs/OPS.md's `scale_rows` says why `mul` cannot). A family without
 * one passes `Src::shared` null and the arm, buffers included, is not declared.
 *
 * ============================== ROW SLICING ==============================
 *
 * The expert intermediates are `top_k` times as tall as the step: at an 8192-token chunk and
 * top_k 8 the down projection's input is 65536 rows, 256 MiB at n_embd 2048. So the block runs in
 * passes of at most `Config::rows` tokens and declares those buffers at that height. Routing is
 * per pass (`expert_offset` and `sorted_tok` describe one pass), which is exact because a token's
 * routing depends only on that token.
 *
 * The default is 2048, the prefill chunk: core/sched/geometry.cpp clamps the chunk to the
 * linear-state checkpoint interval (2048 on this family), so the shipped configuration is one
 * pass. Slicing serves deployments with a larger chunk.
 *
 * ============================== EXPERT WEIGHT RUNS ==============================
 *
 * `moe_gemm_q` takes weights as tables (RAD_OPD_WTAB): a run of consecutively declared handles,
 * `RAD_WTAB(first, n)`. Each projection's experts are two tables by parity, because all entries of
 * a table share one geometry and a rank holding an uneven slice of each expert has two (see
 * Config::ff_lo). So the arrays are declared in separate loops (gate_up codes, gate_up scales,
 * down codes, down scales, each even then odd), never grouped per expert, which would give strided
 * runs the operand cannot express. When four ranks split every expert there are four classes by
 * number mod 4 (`moe_gemm_q_mod4`, Config::ff_lo4), declared the same way class by class.
 *
 * A projection's codes run and scale run are two views of one stored weight
 * (RadWeightDecl::source). Each expert holds two logical weights, gate_up and down, and is its own
 * movement unit (`grp_expert_slot`, slots 0 and 1) with RAD_ACCESS_CONDITIONAL, so the placement
 * planner can tier experts and the heat engine can promote hot ones.
 */
#ifndef RAD_BLOCK_MOE_FP8_H
#define RAD_BLOCK_MOE_FP8_H

#include "rad_fp8.h"
#include "rad_block_glm_router.h"
#include "rad_device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {
namespace arch {

/* Most routing slots the sort serves in one workgroup, bounding `router_topk_scatter` (libr4d's
 * `kScatterSmall`). Fixed here so every rank decides identically at declare whether the op
 * exists. */
static const int64_t kMoeSortRows = 1024;

/* The skinny bf16 GEMM's decode band, within which the fused router op reproduces that GEMM's
 * arithmetic; above it the router uses the tiled kernel and the unfused pair. */
static const int64_t kMoeRouterFusedRows = 64;

/* ---- RADIANCE_DEBUG_ROUTING: prints token 0's routing weights once per layer.
 *
 * The core's histogram (core/runtime/ctx.cpp) shows which experts were chosen and dbg_resid shows
 * each arm's magnitude; neither shows the weights, so a top-k that sums to 0.4 instead of 1 would
 * go unnoticed. Synchronises the stream. Off unless set: a layer number, or negative for all.
 * `f32`: the weights are f32 rather than bf16 (the sigmoid router). */
inline void moe_debug_weights(RadCtx* c_, int layer, int64_t top_k, rad_buf ids, rad_buf ew,
                              bool f32 = false) {
    static const char* dbg = std::getenv("RADIANCE_DEBUG_ROUTING");
    if (!dbg || !ids || !ew) return;
    static const int want = (int)std::strtol(dbg, nullptr, 10);
    if ((want >= 0 && layer != want) || !dbg_first(layer, "moe")) return;

    void* pi = rad_buf_ptr(c_, ids);
    void* pw = rad_buf_ptr(c_, ew);
    if (!pi || !pw) return;
    RadStream s = rad_stream(c_);
    std::vector<int32_t>  vi((size_t)top_k);
    std::vector<uint16_t> vw((size_t)top_k);
    std::vector<float> vf((size_t)top_k);
    rad_memcpy_async(vi.data(), pi, top_k * 4, s);
    rad_memcpy_async(f32?(void*)vf.data():(void*)vw.data(), pw, top_k * (f32?4:2), s);
    rad_stream_sync(s);

    std::string line;
    double sum = 0.0;
    for (int64_t r = 0; r < top_k; ++r) {
        const uint32_t u = (uint32_t)vw[(size_t)r] << 16;
        float f;
        std::memcpy(&f, &u, 4);
        if (f32) f = vf[(size_t)r];
        sum += f;
        char b[48];
        std::snprintf(b, sizeof b, "%s%d:%.4f", r ? " " : "", vi[(size_t)r], f);
        line += b;
    }
    std::fprintf(stderr, "D moe layer %d token 0 top_k = %s  (sum %.4f)\n",
                 layer, line.c_str(), sum);
}

/* Most experts a layer protects on one rank (MoeFP8::prot): libr4d's sort takes at most eight in
 * its `protect` list. */
constexpr int64_t kMaxProtected = 8;

struct MoeFP8 {
    struct Config {
        /* Routed experts this rank owns. Equal to `n_expert_all` unless experts are split across
         * ranks, which only a calibration run does (a Hessian spans an expert's whole K); otherwise
         * each rank holds a slice of every expert (see ff_lo). */
        int64_t n_expert     = 0;
        /* The model's expert count: the router's width and what top_k picks from. 0 means
         * n_expert.
         *
         * The router stays replicated under expert parallelism: every rank scores every expert from
         * the same residual stream and picks the same top_k, so ranks agree on routing without a
         * collective. Each sorts only the slots in its own range and the end-of-block all-reduce
         * sums the partials. */
        int64_t n_expert_all = 0;
        /* This rank's first expert id in the model's numbering. moe_scatter subtracts it so the
         * offsets, counts and weight table are local. */
        int64_t expert_base  = 0;
        int64_t top_k        = 0;
        int64_t n_ff_exp     = 0;   /* one expert's intermediate width, per rank */
        /* This rank's slice of each routed expert when the width does not halve into whole fp8
         * scale blocks: expert e's intermediate columns [ff_lo[e % 2], ff_hi[e % 2]) in model
         * numbering, by parity. hi 0 is the even split (or none). `n_ff_exp` is then the wider
         * slice, the width of the buffers between the two grouped GEMMs.
         *
         * This family's expert is 640 columns (five 128-blocks), so two ranks split each expert
         * three blocks and two. Alternating by parity balances the ranks on average; the imbalance
         * is a fifth of an expert per unit of even/odd difference, versus whole experts under
         * expert parallelism, which the all-reduce turns into waiting (about 1 ms a step at one
         * sequence).
         *
         * Each rank computes its gate/up rows and its K slice of down for every routed expert; the
         * routing weight scales each partial and the all-reduce sums them. Rounding differs (two
         * partial sums), so this is checked by agreement, not bytes. */
        int64_t ff_lo[2] = { 0, 0 }, ff_hi[2] = { 0, 0 };
        /* Or by number mod 4 when four ranks split each expert: expert e's columns
         * [ff_lo4[e % 4], ff_hi4[e % 4]), served by `moe_gemm_q_mod4`, taking precedence over the
         * parity slices. hi 0 offers none.
         *
         * By parity at four ranks, two ranks would carry three blocks of every even/odd pair and
         * two would carry two. With four classes each rank takes two blocks of one class and one of
         * each other: five per four experts. Any expert count works; with 510 quantised experts
         * (two kept bf16) the first two classes hold one more (the first tables of a table op take
         * the remainder). `n_ff_exp` is the widest slice; "e" is the expert's place in the sort. */
        int64_t ff_lo4[4] = { 0, 0, 0, 0 }, ff_hi4[4] = { 0, 0, 0, 0 };
        int64_t n_ff_shared  = 0;   /* the shared expert's, per rank; 0 = no shared expert */
        /* The shared arm's slice on this rank when its width does not halve into whole fp8 scale
         * blocks: columns [shared_lo, shared_hi) in model numbering, `n_ff_shared` its width. hi 0
         * is the even split (or none at world 1); an empty slice declares no arm on this rank.
         *
         * The 640-column shared expert is split three blocks and two rather than run on one rank,
         * which would leave the other waiting at the all-reduce (about 20 us a layer at one
         * sequence). Each rank computes its gate/up columns and K slice of down, scales the partial
         * by the replicated per-token gate, and the all-reduce sums them:
         * `sgate * (down_0 + down_1)`. Checked by agreement, not bytes.
         *
         * The arm runs behind the routed stages on the same queue (see pass()), filling compute
         * units a routed GEMM's tail leaves idle. */
        int64_t shared_lo = 0, shared_hi = 0;
        int     norm_topk    = 1;   /* renormalise the kept routing weights to sum to one */
        /* GLM routing (GlmRouter): sigmoid scores with a correction bias, top-k, `route_scale` on
         * the kept weights, which are then f32 in `ew`. */
        int     sigmoid_router = 0;
        /* With `sigmoid_router`: the selection, the scatter and an ungated shared arm's gate fill in
         * one launch (`router_sigmoid_topk_scatter`) wherever the pass fits the sort's one
         * workgroup; the separate ops otherwise. Byte-identical either way. */
        int     router_fuse = 0;
        /* Block-fp8 experts whose scale plane is f32 (format 9, dtype fp8a8f32). */
        int     expert_f32_scale = 0;
        double  route_scale = 1.0;
        /* With `sigmoid_router`: group-limited selection over `n_group` groups, `topk_group` kept
         * (router_sigmoid_group_topk; Ling-3.0). 1 is GLM's single group. The fused selection and
         * sort is single-group, so a grouped router runs the separate pair. */
        int64_t n_group = 1, topk_group = 1;
        /* Clamp of the SwiGLU product (swiglu_clamped); 0 is none. */
        double  swiglu_limit = 0.0;
        /* The shared arm's clamp where it differs from the routed one (Ling-3.0 clamps its shared
         * expert at 5 or 7 and its routed ones at 4, in different layers); negative: the routed
         * arm's. */
        double  shared_swiglu_limit = -1.0;
        /* Clip after the SiLU (`swiglu_step`) rather than before it (`swiglu_clamped`), for both
         * arms. */
        int     swiglu_step = 0;
        /* With `swiglu_limit`: the clamped product and its quantiser as one launch
         * (GateQuantFP8::fuse_clamped), for the routed and the shared arm. */
        int     swiglu_quant = 0;
        /* A shared arm without its sigmoid gate: the gate plane is filled with 1. */
        int     shared_ungated = 0;
        /* At TP1 the caller may fill the ungated arm's ones plane once for the whole step,
         * rather than each layer/pass filling the same plane again. */
        int     shared_one_external = 0;
        /* Add the already-rounded routed and shared BF16 outputs in FP32, then round to BF16.
         * At TP1 the ungated gather preserves both roundings; TP>1 reduces each arm separately
         * before combining, so its gather cannot fold the shared output. */
        int     shared_fp32_combine = 0;
        /* Experts without a gate (Nemotron-H): one projection up (`up_proj`), relu(x)^2, and down,
         * so the first GEMM is n_ff_exp wide rather than 2 * n_ff_exp and the quantiser between
         * the GEMMs squares rather than gates (GateQuantFP8's "relu2"). The routed arm only: such a
         * family's shared expert is the caller's. */
        int     ungated = 0;
        /* The router's input width when it is not the experts' (Nemotron-H's latent experts: the
         * router scores the full hidden state, the experts read a narrower projection of it, and
         * this block's Geom is the experts'). 0 means n_embd. With it the router reads
         * Wire::route. The sigmoid router only. */
        int64_t router_k = 0;
        /* The caller reduces the block's delta across ranks, because it applies more linear work
         * to it first (Nemotron-H's latent up-projection); no all-reduce is declared. */
        int     no_reduce = 0;
        /* 4-bit experts (`i4*bf16[1x128]`) rather than E4M3 (`fp8_e4m3*bf16[128x128]`); for
         * Qwen3.8-Flash-Next this decides whether the model fits two cards (see the `moe_gemm_w4a8`
         * registry row). It is the stored encoding: probe() reads it from the first expert's
         * encoding (container, checkpoint or rad-convert recipe) and declare() refuses an expert
         * whose encoding differs.
         *
         * A packed 4-bit row is K/2 bytes, so a width shard must fall on a scale group; declare()
         * checks this per rank. */
        int     expert_w4    = 0;
        /* OCP E2M1 codes with E8M0 scales per 32 of K; kept canonical for host-tier reads. */
        int     expert_mx    = 0;
        /* Rotated: a 128-wide Walsh-Hadamard along K baked into the stored expert weight and the
         * activation meeting it (the encoding's `fwht128` transform, dtype `w4a8h`; only with
         * `expert_w4`).
         *
         * Runtime cost: the rotation rides in the quantiser's registers. The down projection's
         * input comes from this block's own gated quantiser, which simply becomes the rotated one.
         * The gate_up input is the normed activation, which the shared expert also reads and which
         * must stay unrotated, so it costs a second (q, scale) pair and one `had_quant_act_fp8`
         * launch per layer.
         *
         * Alone, rotation is worse for round-to-nearest (2x for RTN int4 on the 27B; group-128
         * absmax already handles range). It is the basis GPTQ's error feedback quantises in. */
        int     expert_rot   = 0;
        /* Codebook: the rotated codes index a sixteen-entry `table` plane (libquant's
         * `table=w4nl`, non-uniform levels) instead of -8..7 (dtype `w4nla8h`). Only with
         * `expert_rot`. The codes view selects the table beside the codes so the kernel's relayout
         * can check it. */
        int     expert_nl    = 0;
        /* Scales per 64 of K: one E4M3 byte under a fixed 2^-13 (dtype `w4nl64a8h`, libr4d's
         * kW4G64Scale), the same bytes as one bf16 per 128. Only with `expert_nl`. The scale views
         * select the fixed second level, which the relayout checks against the kernel's constant.
         * The value is scale groups per 128 of K: 2 for 64, 4 for 32 (`w4nl32a8h`). */
        int     expert_g64   = 0;
        /* 5-bit codes: libquant's 32-level `table=w5nl` at w4nl64a8h's scales (dtype
         * `w5nl64a8h`), sign in the high bit. Only with `expert_g64` 2. This is the probed
         * layer's; the 64-group codebook forms may differ by layer and projection, and declare()
         * resolves each layer's (MoeFP8::fmt). */
        int     expert_w5    = 0;
        /* Int8 experts (`i8*bf16[1x128]/fwht128`): codes -127..127 with a bf16 scale per (row, 128
         * of K), each 128 of K rotated as `expert_rot` describes, read against int8 activations
         * (dtype `i8a8h`). The eight-bit expert of an int8 model, where block fp8 is that of an
         * E4M3 one. Set with `expert_rot`. */
        int     expert_i8    = 0;
        /* Experts as IQR weights: the rotated computed trellis (libquant's tcq_code=mad) under one
         * f32 scale a row, two to six bits a code, each expert decoded where the grouped GEMM
         * reads it against the rotated int8 activations (dtype `i8a8h`, libr4d's moe_gemm_iqr).
         * Set with `expert_rot`. */
        int     expert_iqr   = 0;
        /* Plain bf16 experts (no quantising recipe). Both GEMMs are `moe_gemm` over one table of
         * whole experts with `silu_mul` between, and the shared arm follows the trunk
         * (Geom::w_bf16). Not sliced by parity: each expert is served whole on its rank, which
         * also keeps it one container run for the file tier. */
        int     expert_bf16  = 0;
        /* The layer and global expert whose gate_up probe() read the format from (and so mapped);
         * declare() maps the rest. Layer -1: none. */
        int64_t probed_layer = -1;
        int64_t probed_expert = 0;
        /* Experts from `probed_expert` on that probe() mapped: it skips experts a quantised layer
         * keeps bf16 (see `prot`) to reach the first in the layer's format. */
        int64_t probed_count = 1;
        /* The GPTQ calibration tap (rad_arch.h). On, the block declares an f32 [K, K] PERSIST
         * accumulator and a `gram_accum` for each expert projection, runs until
         * RADIANCE_CALIB_TOKENS tokens, writes them to RADIANCE_CALIB_DIR and switches off.
         *
         * A separate run, not a serving mode: the accumulators are 1.34 GB over 48 layers at this
         * model's widths and the dump synchronises the stream. rad-convert reads the files.
         *
         * It taps the quantiser's output, the operand the expert GEMM contracts; under
         * `expert_rot` that is rotated, the same basis the packer quantises in. */
        int     calib        = 0;
        int64_t rows         = 0;   /* tokens per pass; 0 means the whole step */
    };

    /* Only the four planes with one row per (token, slot) are a pass tall; everything else has one
     * row per token, is sized at max_tok and sliced by the pass offset. This is required, not an
     * optimisation: LinearFP8 and GateQuantFP8 slice input and output with the same `r0`, so a
     * pass-tall shared-expert output would be overrun on the second pass. */
    struct Wire {
        rad_buf x       = 0;   /* [max_tok, n_embd]  the residual stream */
        ActFP8  h{};           /* [max_tok, n_embd]  normed input, then the block's delta */
        /* The router's input when Config::router_k is set; `h` otherwise. */
        rad_buf route   = 0;   /* [max_tok, router_k]  bf16 */
        /* Rotated codes of the same input (only `q` and `s`; `x` is `h`'s). Declared only with
         * `expert_rot`: the routed gate_up reads them, the shared expert reads `h`'s. */
        ActFP8  hr{};          /* [max_tok, n_embd]  q + scale only */
        /* the routing, one row per token */
        rad_buf logits  = 0;   /* [max_tok, n_expert_all] bf16, the model's expert count */
        rad_buf ids     = 0;   /* [max_tok, top_k]      i32 */
        rad_buf ew      = 0;   /* [max_tok, top_k]      F32 for sigmoid_router, BF16 otherwise */
        rad_buf eoff    = 0;   /* [n_expert + 1]        i32 */
        rad_buf ecnt    = 0;   /* [n_expert]            i32 */
        /* Above 0, `ecnt` is the model's routing table (rad_route_table), [layers, ecnt_pitch],
         * and this block's counts are its row `layer`: the report then costs no copy and every
         * layer keeps its own counts. 0: `ecnt` is one row, which the layers sharing the wire
         * overwrite in turn. */
        int64_t ecnt_pitch = 0;
        /* the expert planes, one row per (token, slot), a pass tall */
        rad_buf sorted  = 0;   /* [rows * top_k]               i32, pass-relative slot indices */
        /* Each slot's row in `sorted`, -1 when dropped; with it the sort writes the gather's slot
         * table (moe_scatter_inv, moe_gather_inv). Optional. */
        rad_buf inv     = 0;   /* [rows * top_k]               i32 */
        rad_buf egu     = 0;   /* [rows * top_k, 2 * n_ff_exp] bf16 */
        ActFP8  eff{};         /* [rows * top_k, n_ff_exp] */
        rad_buf edn     = 0;   /* [rows * top_k, n_embd]       bf16, in sorted order */
        /* back in token order, and the shared expert, one row per token */
        rad_buf moe     = 0;   /* [max_tok, n_embd]           bf16; separate routed output if needed */
        rad_buf sgu     = 0;   /* [max_tok, 2 * n_ff_shared]  bf16 */
        ActFP8  sff{};         /* [max_tok, n_ff_shared] */
        rad_buf sout    = 0;   /* [max_tok, n_embd]           bf16 */
        rad_buf sgate   = 0;   /* [max_tok, 1]                bf16 */
    };

    /* Checkpoint names. `experts` is a prefix ("....mlp.experts") from which per-expert names are
     * built; `shared` likewise for the dense expert, null when the family has none (then
     * `shared_gate` must be null too).
     *
     * The config travels here, not as a declare() argument, so declare() has the same signature
     * as MlpFP8's and MtpBlockFP8 can be a template over both. */
    struct Src {
        Config      cfg{};
        const char* norm        = nullptr;
        const char* router      = nullptr;   /* mlp.gate.weight, [n_expert, n_embd] bf16 */
        const char* router_bias = nullptr;   /* GLM noaux_tc correction bias */
        const char* experts     = nullptr;   /* prefix; ".<e>.{gate,up,down}_proj.weight" */
        const char* shared      = nullptr;   /* prefix; ".{gate,up,down}_proj.weight" */
        /* Stacked checkpoints (recent releases) store all experts in two tensors:
         * `mlp.experts.gate_up_proj` [n_expert, 2*n_ff, n_embd] and `mlp.experts.down_proj`
         * [n_expert, n_embd, n_ff]. Set these instead of `experts`; each expert maps to a dim-0
         * slice (rad_arch.h: map_slice; core/format/checkpoint.cpp takes it as a view, with a
         * block-fp8 checkpoint's stacked `_scale_inv` at the same index). The stacked gate_up is
         * already fused gate rows first, as this block wants. */
        const char* experts_gate_up = nullptr;
        const char* experts_down    = nullptr;
        /* Some GGUF writers retain gate and up as separate stacked tensors. These three fields
         * describe that layout; each expert's gate/up slices are fused at conversion. */
        const char* experts_gate = nullptr;
        const char* experts_up   = nullptr;
        const char* shared_mlp_gate = nullptr;
        const char* shared_mlp_up   = nullptr;
        const char* shared_mlp_down = nullptr;
        const char* shared_gate = nullptr;   /* mlp.shared_expert_gate.weight, [1, n_embd] */
    };

    Geom   g{};
    Config c{};
    Wire   w{};
    int    layer = 0;

    rad_weight w_norm = 0, w_router = 0, w_sgate = 0;
    GlmRouter glm_router{};
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    /* The weight runs. Only the first handle and the count reach an issue (RAD_WTAB); the rest are
     * kept so a declaration failure can name its expert. */
    std::vector<rad_weight> w_gu, w_gus, w_dn, w_dns;

    NormQuantFP8 nq_h{};
    GateQuantFP8 gq_exp{}, gq_sh{};
    /* The shared arm's gate_up and its quantiser as one launch (decl_gated_fold_iqr), and the rows
     * it serves; 0 where the pair runs. */
    rad_op     op_sh_gated = 0;
    int64_t    sh_gated_rows = 0;
    /* Whether the routed gate_up is a trellis weight (moe_gemm_iqr), and the routed rows from
     * which the shared arm's first stage then runs beside the router instead (pass()). */
    bool       gu_trellis = false;
    static constexpr int64_t kSharedFirstRows = 100;
    LinearFP8    sh_gate_up{}, sh_down{};
    rad_op op_router = 0, op_topk = 0, op_scatter = 0;
    /* `op_scatter` is moe_scatter_inv and `op_gather` moe_gather_inv: the sort writes `w.inv` and
     * the gather reads it (Wire::inv). */
    bool   scatter_inv = false, gather_inv = false;
    /* `router_topk_scatter`, if it resolves: topk and scatter in one launch. The pair is declared
     * either way, since the fused form is bounded by the sort's single workgroup and a prefill
     * chunk exceeds it. */
    rad_op  op_topk_scatter = 0;
    int64_t topk_scatter_rows = 0;
    /* `router_gemm_topk_scatter`, if it resolves: the router GEMM fused in front, up to
     * `router_fused_rows` tokens, plus the shared expert's gate (`op_sgate`) when
     * `router_fused_gate`. Otherwise, or for larger passes, the separate ops run. */
    rad_op  op_router_fused = 0;
    int64_t router_fused_rows = 0;
    bool    router_fused_gate = false;
    /* `router_sigmoid_topk_scatter` (Config::router_fuse), up to `sigmoid_scatter_rows` tokens,
     * writing the shared gate's fill too when `sigmoid_scatter_gate`. */
    rad_op  op_sigmoid_scatter = 0;
    int64_t sigmoid_scatter_rows = 0;
    bool    sigmoid_scatter_gate = false;
    /* The rotated re-quantisation of the block input, declared only when `expert_rot` is set and
     * the caller does not already write `w.hr`. */
    rad_op op_hq = 0;
    /* The caller's read writes the rotated pair (HyperConn::Config::rotate), so this block declares
     * no quantiser for it. Set before declare. */
    bool   hr_in = false;
    rad_op op_gu = 0, op_dn = 0, op_gather = 0;
    /* The routed arm's activations are int8: rotated codes with an f32 scale per (row, group),
     * which the grouped GEMMs meet with the iu8 WMMA (the `a8i8` and `i8a8h` dtypes). The
     * architecture asks for it by feeding `Wire::eff` int8 (ActFP8::q8_fed with q8_rot); int8
     * experts take nothing else, and the rotated four-bit forms take either. Resolved by
     * declare(). */
    bool    a8 = false;
    /* The codes gate_up reads and the K one of their scales covers: `h`'s own pair where the
     * trunk feeds rotated int8 (at its group, 128 or IQ8R's 256), else `hr`'s. */
    rad_buf gin_q = 0, gin_s = 0;
    int64_t gin_g = RAD_FP8_BLOCK;
    /* Protected experts: local experts a quantised layer keeps bf16, the few whose output dwarfs
     * the rest (docs/MOE-W4.md). The scatter sorts them last (`protect`, `prot_list`), the
     * quantised GEMMs serve the other `n_reg`, and a bf16 grouped GEMM pair serves these over the
     * trailing offsets: gate_up into `b_pgu`, the gate into `pff`, down into the `edn` rows the
     * quantised down left zero. */
    std::vector<int64_t>    prot;
    const char*             prot_list = nullptr;
    /* Next-layer expert prefetch (rad_expert_prefetch). `xp_next` is the next routed layer, which
     * this one predicts for from its router input; the architecture links them after declare.
     * `xp` describes this layer to its predecessor: its router and, per model expert, the four
     * handles its table entries hold and its place in the routing counts. Quantised experts only;
     * a layer without a description gets no prediction. */
    const MoeFP8*           xp_next = nullptr;
    RadPrefetchNext         xp{};
    std::vector<rad_weight> xp_parts;
    std::vector<int32_t>    xp_count_at;
    int64_t                 n_reg = 0;
    /* How many weight-table classes the quantised GEMMs take: 2 by parity (`moe_gemm_q`), or 4 by
     * number mod 4 (`moe_gemm_q_mod4`, Config::ff_lo4). */
    int                     ncls = 2;
    /* This layer's gate_up and down formats (format_of's numbers), resolved by declare(): the
     * probed layer's, except that among the forms reading one input (declare()'s `family`: the
     * codebook forms at E4M3 scales, or under an int8 routed arm every form with an int8 row) each
     * projection of each layer takes its own experts'. */
    int                     fmt[2] = { 0, 0 };
    /* The encoding of the layer's first quantised expert, per projection: a trellis's code width
     * is the views' dtype. */
    RadEncoding             fenc[2]{};
    std::vector<rad_weight> w_pgu, w_pdn;
    rad_buf                 b_pgu = 0;
    ActFP8                  pff{};
    GateQuantFP8            gq_prot{};
    rad_op                  op_pgu = 0, op_pdn = 0;
    /* The gather folds in the shared arm when there is one, reading `sout` and `sgate` and writing
     * the block's delta straight into `h.x`; `w.moe` is then unused. */
    bool   gfold = false;
    /* The following write performs the gather up to this many tokens, per wire (HyperConn's
     * gather_rows / gather_rows6, set by the architecture after that write is declared; 0 where
     * not). Where gather_taken() agrees, the block leaves its routed rows unsummed. */
    int64_t gather_out_rows = 0;
    int64_t gather_out_rows6 = 0;
    /* The calibration tap's accumulators and ops; zero unless `Config::calib`. `calib_rows` and
     * `calib_done` are mutable because `pass` is const. The count is per block, so every block
     * reaches its target on the same pass. */
    rad_buf b_gram_gu = 0, b_gram_dn = 0;
    rad_op  op_gram_gu = 0, op_gram_dn = 0;
    /* The domain the tap sums in, recorded in its files (calib_write): the rotation width when the
     * GEMMs read rotated codes, else 0. */
    uint32_t gram_domain = 0;
    /* Files are named after the weight they will pack, minus the expert index. That is the whole
     * protocol with the GPTQ quantiser: the recipe gives the directory (`calib=`), rad-convert the
     * weight's logical name, and libquant strips `.<e>.weight` to find the file. */
    const char* gram_gu_stem = nullptr;
    const char* gram_dn_stem = nullptr;
    mutable long long calib_rows = 0;
    mutable bool      calib_done = false;
    /* Per-expert down-projection Hessians (rad_arch.h: CalibExperts), beside the pooled one: a
     * layer-sized scratch of `ne` [n_ff_exp, n_ff_exp] squares, a `gram_accum` into each over that
     * expert's rows, summed on the host. */
    rad_buf b_gram_dne = 0;
    rad_op  op_gram_dne = 0;
    mutable CalibExperts calib_dne;
    mutable bool         calib_dne_failed = false;
    void calib_experts_pass(RadCtx* c_, int64_t slots) const;
    rad_op op_sgate = 0;
    rad_op op_combine = 0; /* BF16 inputs, FP32 addition, one BF16 rounding */
    rad_op op_moe_ar = 0, op_shared_ar = 0;
    rad_op op_ar = 0, op_add = 0;
    bool   ar_out = false;
    /* Which wires the consumer of this block's all-reduce takes. Set before declare; see
     * AttnGatedFP8::ar_out_take. */
    int    ar_out_take = kArOutNorm;

    /* Reads the stored expert format from layer `l`'s first expert's gate_up encoding into `cfg`
     * (expert_w4, expert_rot, ...). That name map is declared here and skipped by declare(). Called
     * before any layer is declared, because the architecture sizes the rotated activation and
     * tells the preceding connection to write it. An unknown encoding (no model behind the
     * builder) means block fp8. */
    static int probe(RadBuilder* b, Names& nm, int l, const Src& src, Config* cfg);
    /* The format `e` names: 0 block fp8, 1 four-bit, 2 four-bit rotated, 3 plain bf16, 4 the
     * rotated w4nl codebook, 5 and 6 that codebook at E4M3 scales per 64 and 32 of K, 7 its
     * five-bit form per 64, 8 OCP MXFP4, 9 block fp8 at f32 scales, 10 rotated int8, 11 the rotated
     * computed trellis under one f32 scale a row (an IQR weight an expert); RAD_E_DTYPE
     * otherwise. */
    static int format_of(const RadEncoding& e);
    /* format_of's numbers as the grouped GEMMs' dtype strings against an E4M3 activation, and
     * against an int8 one (null: the format has no such row). Int8 experts read int8 only. */
    static constexpr const char* kServed[12] = { "fp8a8", "w4a8", "w4a8h", "bf16", "w4nla8h",
                                                "w4nl64a8h", "w4nl32a8h", "w5nl64a8h", "mxfp4a8",
                                                "fp8a8f32", nullptr, nullptr };
    static constexpr const char* kServed8[12] = { nullptr, nullptr, "w4a8i8h", nullptr, nullptr,
                                                 "w4nl64a8i8h", "w4nl32a8i8h", nullptr, nullptr,
                                                 nullptr, "i8a8h", "i8a8h" };

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                 const Wire& wire, const Src& src, bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
    /* The block over the first `T` rows of its buffers, for passes that need fewer rows than the
     * batch (the MTP head's history pass reads its output only at drafting rows). */
    void step_rows(RadCtx* c, int64_t T) const;
    /* One pass of at most `c.rows` tokens. `own_ar` false: the caller issues the pass's
     * all-reduce itself (issue_ar), as GdnFP8::step describes. */
    void pass(RadCtx* c, int64_t T, int64_t r0, int64_t rows, bool own_ar = true) const;
    /* The heat engine's handoff, after a step's last pass (step() says what it reports). */
    void route_report(RadCtx* c) const;
    bool issues_ar(int64_t T, int64_t rows) const {
        return op_ar && !ar_taken(g, ar_out_take != kArOutNorm ? T : rows, ar_out, ar_out_take);
    }
    /* `beside` false: the collective runs on its own (a pass that owns it), so it takes the
     * size-chosen block count and not the overlap cap, which only pays beside other work. */
    void issue_ar(RadCtx* c, int64_t r0, int64_t rows, bool beside = true) const {
        RAD_ISSUE_N(c, beside && op_ar_ov ? op_ar_ov : op_ar, rows * g.n_embd,
                    brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    }
    /* Set before declare: the `overlap` issue_ar's collective is declared at, 0 for none. */
    int    ar_overlap = 0;
    rad_op op_ar_ov = 0;
    void shared_up(RadCtx* c, int64_t T, int64_t r0, int64_t rows, bool gate_done = false) const;
};

inline int MoeFP8::format_of(const RadEncoding& e) {
    if (linear_is_mxfp4(e)) return 8;
    if (rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16) return 3;
    /* The computed trellis under one f32 scale a row, rotated (IQR-V4's feed-forward form). */
    if (rad_enc_is(&e, "trellis"))
        return std::strcmp(e.transform, "fwht128") == 0 && linear_w8h_mad(e) &&
               linear_w8h_rowf(e) ? 11 : RAD_E_DTYPE;
    if (!rad_enc_is(&e, "affine")) return RAD_E_DTYPE;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    /* The codebook at E4M3 scales per 64 or 32 of K under one fixed f32 (w4nl64a8h, w4nl32a8h),
     * and its five-bit form per 64 (w5nl64a8h). */
    if (e.n_planes == 4 && sc && sc->dtype == RAD_F8E4M3) {
        const RadEncPlane* s2 = rad_enc_plane(&e, "scale.1");
        const RadEncPlane* t  = rad_enc_plane(&e, "table");
        if (!(s2 && s2->dtype == RAD_F32 && s2->block[0] == 0 && s2->block[1] == 0 && t &&
              t->kind == RAD_PLANE_TABLE && t->dtype == RAD_F32 && t->extent[0] == 1 &&
              sc->block[0] == 1 && std::strcmp(e.transform, "fwht128") == 0))
            return RAD_E_DTYPE;
        if (t->extent[1] == 16 && e.plane[0].dtype == RAD_U4 &&
            (sc->block[1] == RAD_FP8_BLOCK / 2 || sc->block[1] == RAD_FP8_BLOCK / 4))
            return sc->block[1] == RAD_FP8_BLOCK / 2 ? 5 : 6;
        if (t->extent[1] == 32 && e.plane[0].dtype == RAD_U5 && sc->block[1] == RAD_FP8_BLOCK / 2)
            return 7;
        return RAD_E_DTYPE;
    }
    if (!sc) return RAD_E_DTYPE;
    if (e.n_planes == 2 && e.plane[0].dtype == RAD_F8E4M3 && sc->dtype == RAD_F32 &&
        !e.transform[0] && sc->block[0] == 128 && sc->block[1] == 128)
        return 9;
    if (sc->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (e.n_planes == 3) {
        const RadEncPlane* t = rad_enc_plane(&e, "table");
        if (t && t->kind == RAD_PLANE_TABLE && t->dtype == RAD_F32 && t->extent[0] == 1 &&
            t->extent[1] == 16 && e.plane[0].dtype == RAD_U4 && sc->block[0] == 1 &&
            sc->block[1] == RAD_FP8_BLOCK && std::strcmp(e.transform, "fwht128") == 0)
            return 4;
        return RAD_E_DTYPE;
    }
    if (e.n_planes != 2) return RAD_E_DTYPE;
    if (e.plane[0].dtype == RAD_F8E4M3 && !e.transform[0] && sc->block[0] == RAD_FP8_BLOCK &&
        sc->block[1] == RAD_FP8_BLOCK)
        return 0;
    if (e.plane[0].dtype == RAD_I4 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK) {
        if (!e.transform[0]) return 1;
        if (std::strcmp(e.transform, "fwht128") == 0) return 2;
    }
    if (e.plane[0].dtype == RAD_I8 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK &&
        std::strcmp(e.transform, "fwht128") == 0)
        return 10;
    return RAD_E_DTYPE;
}

/* Fills `src` with whether the checkpoint stores a layer's experts stacked
 * (`<prefix>.gate_up_proj`, `<prefix>.down_proj`; recent releases) or one by one
 * (`<prefix>.<e>.{gate,up,down}_proj`; e.g. AutoRound exports), by probing a name mapped to the
 * per-expert form's first expert. Defaults to stacked: a container is read by declared name, the
 * same under both maps. `tag` keeps different callers' probe names apart. */
inline int moe_expert_sources(RadBuilder* b, Names& nm, const char* tag, const char* prefix,
                              MoeFP8::Src* src) {
    const char* gu = nm.ckpt("%s.gate_up_proj", prefix);
    const char* probe = nm.f("probe.%s.%s.0", tag, prefix);
    RadEncoding e{};
    RAD_ARCH_TRY(map_concat(b, probe, { nm.ckpt("%s.0.gate_proj.weight", prefix),
                                        nm.ckpt("%s.0.up_proj.weight", prefix) }));
    if (weight_enc(b, probe, &e)) {
        src->experts = nm.ckpt("%s", prefix);
        src->experts_gate_up = src->experts_down = nullptr;
    } else {
        src->experts = nullptr;
        src->experts_gate_up = gu;
        src->experts_down    = nm.ckpt("%s.down_proj", prefix);
    }
    return RAD_OK;
}

/* The gate_up and down name maps of one expert: a dim-0 slice of the stacked tensors, or the
 * checkpoint's per-expert gate and up joined along rows. `gate_up` false maps down alone. */
inline int moe_map_gate_up(RadBuilder* b, Names& nm, const MoeFP8::Src& src, const char* guw,
                           long long ge, bool ungated = false) {
    if (src.experts_gate_up) return map_slice(b, guw, src.experts_gate_up, (int)ge);
    if (src.experts_gate && src.experts_up)
        return map_slice_concat(b, guw, {src.experts_gate, src.experts_up}, (int)ge);
    /* Ungated experts (Nemotron-H) have no gate_proj: the first GEMM is up_proj alone. */
    if (ungated) return map_copy(b, guw, nm.ckpt("%s.%lld.up_proj.weight", src.experts, ge));
    return map_concat(b, guw, {nm.ckpt("%s.%lld.gate_proj.weight", src.experts, ge),
                                nm.ckpt("%s.%lld.up_proj.weight", src.experts, ge)});
}

inline int moe_map_expert(RadBuilder* b, Names& nm, const MoeFP8::Src& src, const char* guw,
                          const char* dnw, long long ge, bool gate_up) {
    if (gate_up) RAD_ARCH_TRY(moe_map_gate_up(b, nm, src, guw, ge, src.cfg.ungated));
    if (src.experts_gate_up || src.experts_gate)
        return map_slice(b, dnw, src.experts_down, (int)ge);
    return map_copy(b, dnw, nm.ckpt("%s.%lld.down_proj.weight", src.experts, ge));
}

inline int MoeFP8::probe(RadBuilder* b, Names& nm, int l, const Src& src, Config* cfg) {
    if (!cfg || (!src.experts && !src.experts_gate_up && !src.experts_gate)) return RAD_E_INVAL;
    const bool split_stack = src.experts_gate || src.experts_up;
    if ((src.experts_gate_up && split_stack) ||
        (split_stack && (!src.experts_gate || !src.experts_up || !src.experts_down ||
                         src.experts)) ||
        (src.experts_gate_up && (!src.experts_down || src.experts))) return RAD_E_INVAL;
    /* The caller names the expert in model numbering (the first this rank owns under expert
     * parallelism); declare() skips the ones mapped here.
     *
     * A bf16 expert does not settle the format: a quantised layer keeps up to kMaxProtected
     * experts bf16, so the probe reads on to the first expert in another format, and only a longer
     * run of bf16 ones means the model is bf16. */
    const long long g0 = (long long)cfg->probed_expert;
    const long long all = cfg->n_expert_all > 0 ? (long long)cfg->n_expert_all
                                                : (long long)cfg->n_expert;
    const long long gn = all > g0 ? all : g0 + 1;
    cfg->probed_layer = l;
    int f = 0;
    for (long long ge = g0; ge < gn && ge <= g0 + kMaxProtected; ++ge) {
        const char* guw = nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge);
        RAD_ARCH_TRY(moe_map_gate_up(b, nm, src, guw, ge, cfg->ungated));
        cfg->probed_count = ge - g0 + 1;
        RadEncoding e{};
        f = 0;
        const bool known = weight_enc(b, guw, &e);
        if (known && (f = format_of(e)) < 0) {
            char en[256];
            rad_enc_format(&e, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, and the routed experts are served as block fp8 "
                            "(fp8_e4m3*bf16[128x128]), four-bit (i4*bf16[1x128], optionally "
                            "/fwht128; or u4 codes into a 16-entry f32 table, /fwht128) or plain "
                            "bf16; or the codebook at E4M3 scales a 64 or 32 of K under a fixed "
                            "f32, or its 32-entry five-bit form at a 64, or OCP MXFP4 "
                            "(fp4_e2m1*e8m0[1x32]), or rotated int8 (i8*bf16[1x128]/fwht128), or "
                            "the rotated computed trellis under one f32 scale a row. Convert it "
                            "with a recipe that makes one of them.\n", guw, en);
            return RAD_E_DTYPE;
        }
        if (known && (f == 0 || f == 8 || f == 9))
            return linear_refuse_retired(guw, f == 8 ? "MXFP4" : "block fp8");
        if (f != 3) break;
    }
    cfg->expert_w4   = (f == 1 || f == 2 || f == 4 || f == 5 || f == 6 || f == 7) ? 1 : 0;
    cfg->expert_rot  = (f == 2 || f == 4 || f == 5 || f == 6 || f == 7 || f == 10 || f == 11) ? 1 : 0;
    cfg->expert_i8   = f == 10 ? 1 : 0;
    cfg->expert_iqr  = f == 11 ? 1 : 0;
    cfg->expert_nl   = (f == 4 || f == 5 || f == 6 || f == 7) ? 1 : 0;
    cfg->expert_g64  = (f == 5 || f == 7) ? 2 : f == 6 ? 4 : 0;
    cfg->expert_w5   = f == 7 ? 1 : 0;
    cfg->expert_bf16 = f == 3 ? 1 : 0;
    cfg->expert_mx   = f == 8 ? 1 : 0;
    cfg->expert_f32_scale = f == 9 ? 1 : 0;
    return RAD_OK;
}

inline int MoeFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                           const Wire& wire, const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; c = src.cfg; w = wire; layer = l;
    ar_out = ar_out_;
    if (c.n_group > 1) c.router_fuse = 0;
    const double sh_limit = c.shared_swiglu_limit < 0 ? c.swiglu_limit : c.shared_swiglu_limit;
    /* 0 means not expert-parallel: this rank owns every expert. */
    if (c.n_expert_all <= 0) c.n_expert_all = c.n_expert;
    if (c.n_expert <= 0 || c.top_k <= 0 || c.top_k > c.n_expert_all || c.n_ff_exp <= 0)
        return RAD_E_INVAL;
    /* The expert slice must lie inside the model, or tokens routed past it are silently dropped. */
    if (c.expert_base < 0 || c.expert_base + c.n_expert > c.n_expert_all) return RAD_E_INVAL;
    if (c.rows <= 0) c.rows = g.max_tok;
    /* A rank whose slice of the shared arm is empty (see Config) declares no arm, and the
     * all-reduce carries the others' share. */
    const bool split_shared = src.shared_mlp_gate || src.shared_mlp_up || src.shared_mlp_down;
    if (split_shared && (!src.shared_mlp_gate || !src.shared_mlp_up || !src.shared_mlp_down ||
                         src.shared)) return RAD_E_INVAL;
    const bool shared = c.n_ff_shared > 0 && (src.shared || src.shared_mlp_gate);
    if (shared && c.shared_ungated && src.shared_gate) return RAD_E_INVAL;
    if (c.shared_one_external && (!shared || !c.shared_ungated || !w.sgate || g.world != 1))
        return RAD_E_INVAL;
    if (c.shared_fp32_combine && (!shared || !w.moe || std::strcmp(g.dtype, "bf16"))) {
        fprintf(stderr, "radiance: layer %d requests an FP32 shared/routed combine without its "
                        "BF16 shared arm and routed output\n", l);
        return RAD_E_INVAL;
    }
    /* The first GEMM's rows per intermediate column (gate and up, or up alone) and the activation
     * between the GEMMs. */
    const int64_t gu = c.ungated ? 1 : 2;
    const char* const gact = c.ungated ? "relu2" : "silu";
    if (c.ungated && (shared || src.experts_gate_up || src.experts_gate ||
                      c.swiglu_limit > 0)) {
        fprintf(stderr, "radiance: ungated experts are served one by one (`up_proj`), unclamped and "
                        "without a shared arm in this block\n");
        return RAD_E_INVAL;
    }
    if (c.router_k && (!c.sigmoid_router || !w.route)) {
        fprintf(stderr, "radiance: a router over another input than the experts' is the sigmoid "
                        "router reading Wire::route\n");
        return RAD_E_INVAL;
    }
    /* Whether `w.sgate` is excluded from buffer sharing; set with the shared arm and required by
     * the fused router's gate write (see there). */
    bool sgate_concurrent = false;

    /* `src.norm == nullptr`: the architecture prepares the input (rad_block_hc.h's `hc_read`), so
     * `wire.h` is already normed and quantised and no norm weight is declared. `ar_in` is then
     * refused, since the fused all-reduce lives inside the norm. See rad_block_attn_gated_fp8.h. */
    ext_in = (src.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.cq() || !w.h.cs())) || ar_in != kArNone)
            return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }

    /* ---- The experts' name maps (one per logical weight, gate_up and down; scale views have
     * none; the probed experts' gate_up is already mapped), and from their encodings the experts
     * this layer protects. One table takes one encoding, but a quantised layer may keep a few
     * whole experts bf16: those whose output dwarfs the rest, where the same relative rounding
     * error is large in absolute terms (docs/MOE-W4.md). They are sorted last and served by their
     * own bf16 GEMM pair, so they must be known before the sort is declared. */
    const int64_t ne = c.n_expert;
    for (int64_t e = 0; e < ne; ++e) {
        const long long ge = (long long)(c.expert_base + e);   /* the checkpoint's numbering */
        const bool probed = c.probed_layer == l && ge >= (long long)c.probed_expert &&
                            ge < (long long)(c.probed_expert + c.probed_count);
        RAD_ARCH_TRY(moe_map_expert(b, nm, src, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge),
                                    nm.f("blk.%d.ffn_down_exps.%lld.weight", l, ge), ge, !probed));
    }
    /* The routed arm's activations (see `a8`), known before the layer's formats are resolved:
     * which forms may share a model depends on the codes they read. */
    a8 = w.eff.q8_fed && !c.expert_bf16;
    prot.clear();
    {
        const int want = c.expert_iqr ? 11 : c.expert_i8 ? 10 : c.expert_f32_scale ? 9 : c.expert_mx ? 8 : c.expert_bf16 ? 3 : c.expert_w5 ? 7 : c.expert_g64 == 4 ? 6
                       : c.expert_g64 ? 5 : c.expert_nl ? 4
                       : c.expert_w4 ? (c.expert_rot ? 2 : 1) : 0;
        auto fmt_of = [&](int64_t e, RadEncoding (&enc)[2], int (&fe)[2], const char* (&nmv)[2]) {
            const long long ge = (long long)(c.expert_base + e);
            nmv[0] = nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l, ge);
            nmv[1] = nm.f("blk.%d.ffn_down_exps.%lld.weight", l, ge);
            for (int i = 0; i < 2; ++i) {
                fe[i] = want;
                if (weight_enc(b, nmv[i], &enc[i])) fe[i] = format_of(enc[i]);
            }
        };
        /* The forms that read one input may differ by layer and projection: against E4M3 codes the
         * codebook forms at E4M3 scales (same rotated input, scale views, fixed second level,
         * accumulators; only the GEMM's dtype differs), and against the rotated int8 pair every
         * form with a row for it (four-bit, the codebook at a 64 or 32 of K, int8), whose views
         * efmt_of states per projection. The layer's first non-bf16 expert decides both
         * projections; every other must match. Outside its family a layer uses the probed
         * layer's form. */
        const bool a8f = a8;
        auto family = [a8f](int f) {
            return a8f ? (f == 2 || f == 5 || f == 6 || f == 10 || f == 11)
                       : (f == 5 || f == 6 || f == 7);
        };
        fmt[0] = fmt[1] = want;
        fenc[0] = fenc[1] = RadEncoding{};
        for (int64_t e = 0; e < ne; ++e) {
            RadEncoding enc[2]{};
            int fe[2];
            const char* nmv[2];
            fmt_of(e, enc, fe, nmv);
            if (fe[0] == 3 && fe[1] == 3) continue;
            for (int i = 0; i < 2; ++i) {
                if (family(want) && family(fe[i])) fmt[i] = fe[i];
                fenc[i] = enc[i];
            }
            break;
        }
        for (int64_t e = 0; e < ne; ++e) {
            RadEncoding enc[2]{};
            int fe[2];
            const char* names[2];
            fmt_of(e, enc, fe, names);
            /* A trellis layer is one code width a projection: the views are declared at it. */
            const bool same = fe[0] == fmt[0] && fe[1] == fmt[1] &&
                              (fmt[0] != 11 || enc[0].plane[0].dtype == fenc[0].plane[0].dtype) &&
                              (fmt[1] != 11 || enc[1].plane[0].dtype == fenc[1].plane[0].dtype);
            if (same) continue;
            if (want != 3 && fe[0] == 3 && fe[1] == 3) { prot.push_back(e); continue; }
            const int i = fe[0] != fmt[0] ? 0 : 1;
            char en[256];
            rad_enc_format(&enc[i], en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, and this layer's experts are served as %s -- one "
                            "table of experts takes one encoding, and only a whole expert kept "
                            "plain bf16 is served beside it.\n", names[i], en,
                    kServed[fmt[i]] ? kServed[fmt[i]] : kServed8[fmt[i]]);
            return RAD_E_DTYPE;
        }
    }
    n_reg = ne - (int64_t)prot.size();
    if (!prot.empty()) {
        if ((int64_t)prot.size() > kMaxProtected) {
            fprintf(stderr, "radiance: layer %d keeps %zu experts plain bf16 on this rank; the sort "
                            "places at most %lld last\n", l, prot.size(), (long long)kMaxProtected);
            return RAD_E_INVAL;
        }
        /* The quantised GEMM takes two equal tables by parity; with four classes
         * (Config::ff_lo4), four tables with the first n_reg % 4 one longer, so any count works. */
        if (!(c.ff_hi[0] > 0 && c.ff_hi4[0] > 0) && n_reg % 2) {
            fprintf(stderr, "radiance: layer %d keeps %zu experts plain bf16 on this rank, which "
                            "leaves %lld quantised; they come in pairs on a rank so the rest split "
                            "into two equal tables by parity\n", l, prot.size(), (long long)n_reg);
            return RAD_E_INVAL;
        }
        /* The tap's per-expert files are named by expert, but this sort numbers by place. */
        if (c.calib) {
            fprintf(stderr, "radiance: calibrate the bf16 container: layer %d of this one keeps "
                            "experts plain bf16 and sorts them out of order\n", l);
            return RAD_E_INVAL;
        }
        std::string lst;
        for (int64_t e : prot) lst += (lst.empty() ? "" : ",") + std::to_string(e);
        prot_list = nm.ckpt("%s", lst.c_str());
    }

    /* The router is replicated (see Config::n_expert_all): every rank computes the same routing
     * from the full residual stream, so `sorted_tok` and `expert_offset` agree across ranks
     * without a collective. The sigmoid router (GlmRouter) declares its own projection. */
    if (!c.sigmoid_router) {
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_gate_inp.weight", l), src.router));
    w_router = decl_w(b, nm.f("blk.%d.ffn_gate_inp.weight", l), RAD_BF16,
                      {c.n_expert_all, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }

    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));

    if (c.sigmoid_router) {
        if (!src.router_bias) return RAD_E_INVAL;
        GlmRouter::Config rc;
        rc.experts = c.n_expert_all;
        rc.top_k = c.top_k;
        rc.rows = c.rows;
        rc.norm = c.norm_topk;
        rc.route_scale = c.route_scale;
        rc.groups = c.n_group;
        rc.topk_groups = c.topk_group;
        const GlmRouter::Wire wire{ c.router_k ? w.route : w.h.x, w.logits, w.ids, w.ew };
        const GlmRouter::Src rs{ src.router, src.router_bias };
        Geom rg = g;
        if (c.router_k) rg.n_embd = c.router_k;
        RAD_ARCH_TRY(glm_router.declare(b, nm, rg, l, nm.f("blk.%d.glm_router", l), rc, wire, rs));
        w_router = glm_router.projection;
        op_router = glm_router.op_project;
        op_topk = glm_router.op_select;
    } else {
    op_router = rw(b, RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", c.n_expert_all),
                                 RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_router)),
                  {w.h.x}, {w.logits});
    op_topk = rw(b, RAD_OP(b, "router_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert_all),
                               RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm_topk),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.logits}, {w.ids, w.ew});
    }
    /* Protected experts are sorted last (`protect`); layers without them declare no protect
     * list, so it is the last parameter and cut off. */
    const int n_prot = prot_list ? 1 : 0;
    {
        const RadParam ps[] = { RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.n_expert),
                                RAD_INT("top_k", c.top_k), RAD_INT("expert_base", c.expert_base),
                                RAD_STR("protect", prot_list) };
        /* With `w.inv` the sort also writes the gather's slot table. It writes everything
         * moe_scatter does, so a moe_gather_inv that does not resolve leaves moe_gather to serve. */
        if (w.inv && rad_op_declared(b, "moe_scatter_inv") && rad_op_declared(b, "moe_gather_inv"))
            op_scatter = rw(b, rad_decl_op(b, "moe_scatter_inv", ps, 4 + n_prot, RAD_NOWEIGHTS),
                            {w.ids}, {w.sorted, w.eoff, w.ecnt, w.inv});
        scatter_inv = op_scatter != 0;
        if (!op_scatter)
            op_scatter = rw(b, rad_decl_op(b, "moe_scatter", ps, 4 + n_prot, RAD_NOWEIGHTS),
                            {w.ids}, {w.sorted, w.eoff, w.ecnt});
    }

    /* topk and scatter in one launch. At decode each is a single 256-thread workgroup ranking
     * about a kilobyte, so each is a dispatch with almost no work, and the sort reads only what
     * the selection writes.
     *
     * Bounded by the sort's single-workgroup form: `moe_scatter` is one launch at
     * M*top_k <= 1024 and three above (for a stable sort without atomics), and only the first is
     * fusible. That covers decode (eight sequences at depth 3 and top_k 10 is 320); prefill
     * chunks exceed it. The pair is declared either way, so the graph and buffer plan do not
     * depend on it. The sigmoid router selects in its own op and takes no fused pair. */
    topk_scatter_rows = (int64_t)c.rows * c.top_k <= kMoeSortRows ? c.rows
                                                                   : kMoeSortRows / c.top_k;
    if (c.sigmoid_router || scatter_inv) topk_scatter_rows = 0;
    /* Asked for only where a loaded library declares it: a CPU-only engine would refuse the
     * declare, and the two separate ops serve without it. */
    if (topk_scatter_rows > 0 && rad_op_declared(b, "router_topk_scatter")) {
        const RadParam ps[] = { RAD_RANGE("M", 1, topk_scatter_rows),
                                RAD_INT("n_expert", c.n_expert_all), RAD_INT("n_local", c.n_expert),
                                RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm_topk),
                                RAD_STR("dtype", g.dtype), RAD_INT("expert_base", c.expert_base),
                                RAD_STR("protect", prot_list) };
        op_topk_scatter = rw(b, rad_decl_op(b, "router_topk_scatter", ps, 7 + n_prot,
                                            RAD_NOWEIGHTS),
                             {w.logits}, {w.ids, w.ew, w.sorted, w.eoff, w.ecnt});
    }

    /* Exactly one of a per-expert prefix, fused stacked gate_up, or separate stacked gate/up. */
    const bool split_stack = src.experts_gate || src.experts_up;
    const bool stacked = src.experts_gate_up || split_stack;
    if ((src.experts_gate_up && split_stack) ||
        (split_stack && (!src.experts_gate || !src.experts_up)) ||
        stacked != (src.experts_down != nullptr) || (stacked == (src.experts != nullptr))) {
        fprintf(stderr, "radiance: a routed feed-forward needs a per-expert prefix or complete "
                        "stacked expert tensors; prefix %s, gate_up %s, gate %s, up %s, down %s\n",
                src.experts ? "yes" : "no", src.experts_gate_up ? "yes" : "no",
                src.experts_gate ? "yes" : "no", src.experts_up ? "yes" : "no",
                src.experts_down ? "yes" : "no");
        return RAD_E_INVAL;
    }

    /* ---- the expert weights, as runs: gate_up codes, gate_up scales, down codes, down scales,
     * each per class (even then odd; entry e / 2 of class e % 2). A weight table is a run of
     * consecutive handles (see the header), so these loops must not be merged. */
    if (ne % 2) {
        fprintf(stderr, "radiance: %lld routed experts on this rank; the grouped GEMM takes them as "
                        "two equal tables by parity, so the count must be even\n", (long long)ne);
        return RAD_E_INVAL;
    }
    /* The quantised experts by place in the sort (protected ones excluded). Each is declared in
     * its place's slot, so routing counts written in place order credit the right unit. */
    std::vector<int64_t> reg;
    reg.reserve((size_t)n_reg);
    for (int64_t e = 0; e < ne; ++e)
        if (std::find(prot.begin(), prot.end(), e) == prot.end()) reg.push_back(e);
    w_gu.resize((size_t)ne); w_gus.resize((size_t)ne);
    w_dn.resize((size_t)ne); w_dns.resize((size_t)ne);
    /* Each class's width on this rank: the Config's slice when split unevenly, else `n_ff_exp`.
     * The buffers between the GEMMs take the widest. Four classes when Config::ff_lo4 is set, else
     * the parity classes. */
    const bool split = c.ff_hi[0] > 0 || c.ff_hi4[0] > 0;
    ncls = c.ff_hi4[0] > 0 ? 4 : 2;
    if (c.expert_mx && ncls == 4) {
        fprintf(stderr, "radiance: MXFP4 experts use two parity tables; use equal TP width slices\n");
        return RAD_E_UNSUPPORTED;
    }
    if (ncls == 4 && n_reg < 4) {
        fprintf(stderr, "radiance: layer %d serves %lld quantised experts on this rank; four "
                        "classes need at least one each\n", l, (long long)n_reg);
        return RAD_E_INVAL;
    }
    int64_t qlo[4] = { 0, 0, 0, 0 }, qhi[4] = { 0, 0, 0, 0 };
    int64_t wq[4] = { c.n_ff_exp, c.n_ff_exp, c.n_ff_exp, c.n_ff_exp };
    if (split)
        for (int q = 0; q < ncls; ++q) {
            qlo[q] = ncls == 4 ? c.ff_lo4[q] : c.ff_lo[q];
            qhi[q] = ncls == 4 ? c.ff_hi4[q] : c.ff_hi[q];
            wq[q]  = qhi[q] - qlo[q];
        }
    if (split) {
        int64_t widest = 0;
        bool    bad = false;
        for (int q = 0; q < ncls; ++q) {
            bad = bad || wq[q] <= 0;
            widest = wq[q] > widest ? wq[q] : widest;
        }
        if (bad || c.n_ff_exp != widest) {
            fprintf(stderr, "radiance: the routed experts' %d class slices, the first [%lld, %lld) "
                            "and [%lld, %lld), do not describe a width of %lld\n", ncls,
                    (long long)qlo[0], (long long)qhi[0], (long long)qlo[1], (long long)qhi[1],
                    (long long)c.n_ff_exp);
            return RAD_E_INVAL;
        }
    }
    for (int q = 0; q < ncls; ++q)
        if (wq[q] % RAD_FP8_BLOCK || g.n_embd % RAD_FP8_BLOCK ||
            (split && qlo[q] % RAD_FP8_BLOCK)) {
            fprintf(stderr, "radiance: an expert is %lldx%lld per rank, which is not a whole number "
                            "of %dx%d scale blocks\n",
                    (long long)wq[q], (long long)g.n_embd, RAD_FP8_BLOCK, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
    /* A format's views: the codes view's dtype and planes (a codebook's table is selected with its
     * codes so the relayout sees it), and the scale view's -- bf16 per
     * (row, 128 of K) or per 128x128 tile for E4M3, or for the 64-group forms E4M3 per (row, 64 of
     * K) selected with the fixed second level; f32 per 128x128 tile (format 9) or E8M0 per (row, 32
     * of K) for MXFP4. `sgk` is the K one declared scale column covers: a view of two planes is a
     * view of the logical weight, like the codes view (rad_builder.cpp), declared at the weight's
     * extents and sliced in its columns, with the stored [rows, K/64] plane left to the layout
     * hook. */
    struct EFmt { uint32_t cdt; const char* cplanes; uint32_t sdt; int64_t sgk; const char* splanes; };
    auto efmt_of = [](int f, const RadEncoding& fe) -> EFmt {
        switch (f) {
        /* An IQR weight: the codes at their stored width with the code's planes, and the row and
         * block scales as one view at the weight's extents (LinearFP8's views of the same form). */
        case 11: return { fe.plane[0].dtype, "codes,trellis,trellis.mad", RAD_F32, 1,
                          "scale,scale.col" };
        case 10: return { RAD_I8, "codes", RAD_BF16, RAD_FP8_BLOCK, "scale" };
        case 9:  return { RAD_F8E4M3, "codes", RAD_F32, RAD_FP8_BLOCK, "scale" };
        case 8:  return { RAD_FP4E2M1, "codes", RAD_E8M0, RAD_MX_GROUP, "scale" };
        case 7:  return { RAD_U5, "codes,table", RAD_F8E4M3, 1, "scale,scale.1" };
        case 5: case 6: return { RAD_U4, "codes,table", RAD_F8E4M3, 1, "scale,scale.1" };
        case 4:  return { RAD_U4, "codes,table", RAD_BF16, RAD_FP8_BLOCK, "scale" };
        case 1: case 2: return { RAD_I4, "codes", RAD_BF16, RAD_FP8_BLOCK, "scale" };
        default: return { RAD_F8E4M3, "codes", RAD_BF16, RAD_FP8_BLOCK, "scale" };
        }
    };
    const EFmt ef[2] = { efmt_of(fmt[0], fenc[0]), efmt_of(fmt[1], fenc[1]) };
    /* Under expert parallelism each owned expert is whole, so its weights are not sharded;
     * RAD_SHARD_ROW would make the loader slice it again (and refuse). */
    const bool ep = c.n_expert != c.n_expert_all;
    if (ep && split) {
        fprintf(stderr, "radiance: a rank either owns whole experts or a slice of every one; this "
                        "configuration asks for both\n");
        return RAD_E_INVAL;
    }
    /* A packed 4-bit row (K/2 bytes, eight codes per dword) shards along K only on 128-groups,
     * which the check above requires; row shards are whole rows. The 128-wide rotation lies within
     * a group, so neither cuts it.
     *
     * Rotation exists only for the 4-bit plane (the fp8 layout hook has no rotated form), and the
     * codebook only with rotation; other combinations are refused. */
    if ((c.expert_rot && !c.expert_w4 && !c.expert_i8 && !c.expert_iqr) ||
        (c.expert_nl && !c.expert_rot) || ((c.expert_i8 || c.expert_iqr) && !c.expert_rot)) {
        fprintf(stderr, "radiance: rotated experts (expert_rot) require the 4-bit or the int8 "
                        "expert format, and a codebook (expert_nl) or int8 experts the rotated "
                        "one; there is no rotated E4M3 expert plane, no unrotated codebook and no "
                        "unrotated int8 expert\n");
        return RAD_E_INVAL;
    }
    /* ---- The routed arm's activations (see `a8`). */
    if ((c.expert_i8 || c.expert_iqr || fmt[0] >= 10 || fmt[1] >= 10) && !a8) {
        fprintf(stderr, "radiance: layer %d's experts read int8 activations (int8, or an IQR weight); the "
                        "architecture declared the routed arm's codes E4M3\n", l);
        return RAD_E_UNSUPPORTED;
    }
    if (a8) {
        const bool pair = w.eff.q8 && w.eff.s8 && w.eff.q8_rot && w.eff.g8 == RAD_FP8_BLOCK;
        if (!pair || !c.expert_rot || !kServed8[fmt[0]] || !kServed8[fmt[1]] || c.calib ||
            c.swiglu_limit > 0) {
            fprintf(stderr, "radiance: layer %d: int8 activations meet rotated experts (int8, "
                            "four-bit, or the codebook at E4M3 scales a 64 or 32 of K) through "
                            "rotated codes at one scale a 128 of K, unclamped, outside "
                            "calibration; these are %s / %s%s\n", l,
                    kServed[fmt[0]] ? kServed[fmt[0]] : kServed8[fmt[0]],
                    kServed[fmt[1]] ? kServed[fmt[1]] : kServed8[fmt[1]],
                    pair ? "" : ", and the gated product's int8 pair is not that");
            return RAD_E_UNSUPPORTED;
        }
    }
    const int  sh_row = ep ? RAD_SHARD_NONE : RAD_SHARD_ROW;
    const int  sh_col = ep ? RAD_SHARD_NONE : RAD_SHARD_COL;
    const int  n_parts = ep ? 0 : (int)gu;
    /* The scale grid: one per 128x128 tile for E4M3, one per row per 128 of K for 4-bit (per 32
     * for MXFP4). `sdiv` is how many weight rows or columns one scale covers along the split
     * dimension. */
    const bool rowscale = c.expert_w4 || c.expert_mx || c.expert_i8 || c.expert_iqr;
    const int64_t sdiv = rowscale ? 1 : RAD_FP8_BLOCK;
    /* The dtype string per projection selects the grouped GEMM; its layout hook then accepts the
     * experts' encoding or the kernel is not a candidate. */
    gu_trellis = fmt[0] == 11;
    const char* const edtype[2] = { a8 ? kServed8[fmt[0]] : kServed[fmt[0]],
                                    a8 ? kServed8[fmt[1]] : kServed[fmt[1]] };
    /* This class's slice of each expert: gate and up rows [lo, hi), down columns [lo, hi), scale
     * planes the same in their own units. */
    auto span = [&](rad_weight h, int q, int64_t unit) -> int {
        if (!split || !h) return RAD_OK;
        return rad_weight_shard_span(b, h, qlo[q] / unit, qhi[q] / unit);
    };
    const bool rot = c.expert_rot != 0;
    if (c.expert_bf16) {
        /* ---- Plain bf16 experts: one table per projection of whole experts in expert order
         * (`moe_gemm`, expert e at entry e), so the two loops must not be merged. Each expert is
         * its own movement unit, and its container bytes are what the kernel reads, so the file
         * tier can serve it directly. */
        if (split || rot || c.expert_w4) {
            fprintf(stderr, "radiance: bf16 experts are whole on their rank (expert parallel, or "
                            "one rank) and unrotated; this configuration %s\n",
                    split ? "slices every expert" : "rotates or packs them");
            return RAD_E_INVAL;
        }
        for (int64_t e = 0; e < ne; ++e) {
            const int64_t parts[2] = { c.n_ff_exp, c.n_ff_exp };   /* gate, up (up alone: one) */
            w_gu[(size_t)e] = decl_w_n(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                                               (long long)(c.expert_base + e)),
                                       RAD_BF16, {gu * c.n_ff_exp, g.n_embd}, RAD_ACCESS_CONDITIONAL,
                                       sh_row, grp_expert_slot(l, (int)e, 0), 0, parts, n_parts);
            if (!w_gu[(size_t)e]) return RAD_E_INVAL;
        }
        for (int64_t e = 0; e < ne; ++e) {
            w_dn[(size_t)e] = decl_w(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                                             (long long)(c.expert_base + e)),
                                     RAD_BF16, {g.n_embd, c.n_ff_exp}, RAD_ACCESS_CONDITIONAL,
                                     sh_col, grp_expert_slot(l, (int)e, 1));
            if (!w_dn[(size_t)e]) return RAD_E_INVAL;
        }
        /* Row orders as for the quantised GEMMs (see `a_order` below): gate_up gathers token rows,
         * down reads sorted rows. */
        op_gu = rw(b, rad_decl_op(b, "moe_gemm",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", gu * c.n_ff_exp),
                                     RAD_INT("K", g.n_embd), RAD_INT("n_expert", ne),
                                     RAD_INT("top_k", c.top_k), RAD_STR("a_order", "token"),
                                     RAD_STR("dtype", "bf16")),
                          w_gu.data(), (int)ne),
                   {w.h.x, w.sorted, w.eoff}, {w.egu});
        /* silu(gate) * up into the bf16 plane the down GEMM reads, regardless of the trunk's format
         * (rad_fp8.h: GateQuantFP8). */
        Geom gx = g;
        gx.w_bf16 = true;
        RAD_ARCH_TRY(gq_exp.declare(b, gx, w.egu, w.eff, c.n_ff_exp, gact, c.rows * c.top_k,
                                    true, false, c.swiglu_limit, c.swiglu_step != 0));
        op_dn = rw(b, rad_decl_op(b, "moe_gemm",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                     RAD_INT("K", c.n_ff_exp), RAD_INT("n_expert", ne),
                                     RAD_INT("top_k", c.top_k), RAD_STR("a_order", "sorted"),
                                     RAD_STR("dtype", "bf16")),
                          w_dn.data(), (int)ne),
                   {w.eff.x, w.sorted, w.eoff}, {w.edn});
        if (!op_gu || !op_dn) return RAD_E_INVAL;
    } else {
    /* ---- the runs (eight, or sixteen at four classes), declared as views class by class. */
    for (int q = 0; q < ncls; ++q)
        for (int64_t s = q; s < n_reg; s += ncls) {
            const int64_t e = reg[(size_t)s];
            const int64_t parts[2] = { wq[q], wq[q] };
            rad_weight& h = w_gu[(size_t)s];
            h = decl_view(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                                  (long long)(c.expert_base + e)),
                          nullptr, ef[0].cplanes, ef[0].cdt, {gu * wq[q], g.n_embd},
                          RAD_ACCESS_CONDITIONAL, sh_row, grp_expert_slot(l, (int)s, 0),
                          parts, n_parts);
            if (!h) return RAD_E_INVAL;
            RAD_ARCH_TRY(span(h, q, 1));
        }
    for (int q = 0; q < ncls; ++q)
        for (int64_t s = q; s < n_reg; s += ncls) {
            const int64_t e = reg[(size_t)s];
            const int64_t parts[2] = { wq[q] / sdiv, wq[q] / sdiv };
            rad_weight& h = w_gus[(size_t)s];
            h = decl_view(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.scale", l,
                                  (long long)(c.expert_base + e)),
                          nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                               (long long)(c.expert_base + e)),
                          ef[0].splanes, ef[0].sdt,
                          {rowscale ? gu * wq[q] : fp8_blocks(gu * wq[q]),
                           rowscale ? g.n_embd / ef[0].sgk : fp8_blocks(g.n_embd)},
                          RAD_ACCESS_CONDITIONAL, sh_row, grp_expert_slot(l, (int)s, 0),
                          parts, n_parts);
            if (!h) return RAD_E_INVAL;
            RAD_ARCH_TRY(span(h, q, sdiv));
        }
    for (int q = 0; q < ncls; ++q)
        for (int64_t s = q; s < n_reg; s += ncls) {
            const int64_t e = reg[(size_t)s];
            rad_weight& h = w_dn[(size_t)s];
            h = decl_view(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                                  (long long)(c.expert_base + e)),
                          nullptr, ef[1].cplanes, ef[1].cdt, {g.n_embd, wq[q]},
                          RAD_ACCESS_CONDITIONAL, sh_col, grp_expert_slot(l, (int)s, 1));
            if (!h) return RAD_E_INVAL;
            RAD_ARCH_TRY(span(h, q, 1));
        }
    for (int q = 0; q < ncls; ++q)
        for (int64_t s = q; s < n_reg; s += ncls) {
            const int64_t e = reg[(size_t)s];
            rad_weight& h = w_dns[(size_t)s];
            h = decl_view(b, nm.f("blk.%d.ffn_down_exps.%lld.scale", l,
                                  (long long)(c.expert_base + e)),
                          nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                               (long long)(c.expert_base + e)),
                          ef[1].splanes, ef[1].sdt,
                          {rowscale ? g.n_embd : fp8_blocks(g.n_embd), wq[q] / ef[1].sgk},
                          RAD_ACCESS_CONDITIONAL, sh_col, grp_expert_slot(l, (int)s, 1));
            if (!h) return RAD_E_INVAL;
            RAD_ARCH_TRY(span(h, q, ef[1].sgk));
        }

    /* ---- the two grouped GEMMs. `M` is a pass's token count, not the sorted row count: kernels
     * read the row count from `y`, and the scratch hook and operand descriptions size from `M`.
     *
     * `a_order` differs between them. gate_up reads the normed input, one row per token, so sorted
     * row i gathers `a[sorted_tok[i] / top_k]`. down reads the gated quantiser's output, already
     * one row per (token, slot) in sorted order, so row i is row i. Gathering there too would give
     * every slot the first slot's intermediate (at decode, all top_k experts read slot 0), and a
     * reference that made the same mistake would agree; the parameter states which order is
     * meant. */
    /* The routed arm's rotated copy of the input. Not in place: the shared expert also reads
     * `w.h.q` with an unrotated weight (see Config::expert_rot). Declared just before its one
     * consumer so graph and issue order agree. */
    gin_q = w.h.q;
    gin_s = w.h.s;
    gin_g = RAD_FP8_BLOCK;
    if (a8 && w.h.q8_fed && w.h.q8_rot) {
        /* The trunk's own rotated int8 codes, at its group: no second quantiser. */
        gin_q = w.h.q8;
        gin_s = w.h.s8;
        gin_g = w.h.g8;
    } else if (rot) {
        if (a8 ? !(w.hr.q8 && w.hr.s8 && w.hr.q8_fed && w.hr.q8_rot) : !(w.hr.q && w.hr.s)) {
            fprintf(stderr, "radiance: rotated experts need the `hr` (q, scale) pair on the wire%s; "
                            "the architecture declares the buffers\n",
                    a8 ? ", int8 and fed" : "");
            return RAD_E_INVAL;
        }
        gin_q = w.hr.cq();
        gin_s = w.hr.cs();
        gin_g = w.hr.cg();
        if (!hr_in)
            op_hq = rw(b, RAD_OP(b, a8 ? "had_quant_act_i8g" : "had_quant_act_fp8",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                     RAD_INT("group", gin_g), RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.h.x}, {gin_q, gin_s});
    }
    if (!gin_q || !gin_s || g.n_embd % gin_g) return RAD_E_INVAL;
    /* A table op's weights in operand order: even experts' codes and scales, then odd experts'
     * (`w`, `w_scale`, `w_odd`, `w_odd_scale`), or at four classes classes 0 to 3. */
    auto tables = [&](const std::vector<rad_weight>& wc, const std::vector<rad_weight>& ws) {
        std::vector<rad_weight> tab;
        tab.reserve((size_t)n_reg * 2);
        for (int q = 0; q < ncls; ++q) {
            for (int64_t s = q; s < n_reg; s += ncls) tab.push_back(wc[(size_t)s]);
            for (int64_t s = q; s < n_reg; s += ncls) tab.push_back(ws[(size_t)s]);
        }
        return tab;
    };
    /* The gate_up output is two parts (gate, up) of the widest class's width. A narrower class
     * fills each part's leading columns and zeros the rest, so the gated quantiser reads one layout
     * and the zeros past the width are never read by that class's down projection. */
    {
        const std::vector<rad_weight> tab = tables(w_gu, w_gus);
        /* `a_group` is stated to the int8 rows alone: last, and cut off otherwise, so an E4M3
         * graph is the one it was. */
        const RadParam p4[] = { RAD_RANGE("M", 1, c.rows), RAD_INT("N", gu * wq[0]),
                                RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                                RAD_INT("top_k", c.top_k), RAD_INT("group", RAD_FP8_BLOCK),
                                RAD_STR("a_order", "token"), RAD_STR("dtype", edtype[0]),
                                RAD_INT("N_1", gu * wq[1]), RAD_INT("K_1", g.n_embd),
                                RAD_INT("N_2", gu * wq[2]), RAD_INT("K_2", g.n_embd),
                                RAD_INT("N_3", gu * wq[3]), RAD_INT("K_3", g.n_embd),
                                RAD_INT("parts", gu), RAD_INT("a_group", gin_g) };
        const RadParam p2[] = { RAD_RANGE("M", 1, c.rows), RAD_INT("N", gu * wq[0]),
                                RAD_INT("K", g.n_embd), RAD_INT("n_expert", n_reg),
                                RAD_INT("top_k", c.top_k), RAD_INT("group", RAD_FP8_BLOCK),
                                RAD_STR("a_order", "token"), RAD_STR("dtype", edtype[0]),
                                RAD_INT("N_odd", gu * wq[1]), RAD_INT("K_odd", g.n_embd),
                                RAD_INT("parts", gu), RAD_INT("a_group", gin_g) };
        const int cut = a8 ? 0 : 1;
        rad_op h = ncls == 4
            ? rad_decl_op(b, "moe_gemm_q_mod4", p4, (int)(sizeof p4 / sizeof *p4) - cut,
                          tab.data(), (int)tab.size())
            : rad_decl_op(b, "moe_gemm_q", p2, (int)(sizeof p2 / sizeof *p2) - cut,
                          tab.data(), (int)tab.size());
        op_gu = rw(b, h, {gin_q, gin_s, w.sorted, w.eoff}, {w.egu});
    }
    /* The down projection's input can be rotated in place: `w.eff` is read only by the down GEMM,
     * which takes the codes alone (no bf16 product is written), so the gated quantiser resolves to
     * the rotated op at no extra launch. */
    gq_exp.fuse_clamped = c.swiglu_quant != 0;
    /* Fed int8, `eff` names the rotating int8 quantiser itself (ActFP8::q8_rot). */
    RAD_ARCH_TRY(gq_exp.declare(b, g, w.egu, w.eff, c.n_ff_exp, gact, c.rows * c.top_k,
                                false, rot && !a8, c.swiglu_limit, c.swiglu_step != 0));
    {
        const std::vector<rad_weight> tab = tables(w_dn, w_dns);
        const RadParam p4[] = { RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                RAD_INT("K", wq[0]), RAD_INT("n_expert", n_reg),
                                RAD_INT("top_k", c.top_k), RAD_INT("group", RAD_FP8_BLOCK),
                                RAD_STR("a_order", "sorted"), RAD_STR("dtype", edtype[1]),
                                RAD_INT("N_1", g.n_embd), RAD_INT("K_1", wq[1]),
                                RAD_INT("N_2", g.n_embd), RAD_INT("K_2", wq[2]),
                                RAD_INT("N_3", g.n_embd), RAD_INT("K_3", wq[3]),
                                RAD_INT("parts", 1), RAD_INT("a_group", RAD_FP8_BLOCK) };
        const RadParam p2[] = { RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                RAD_INT("K", wq[0]), RAD_INT("n_expert", n_reg),
                                RAD_INT("top_k", c.top_k), RAD_INT("group", RAD_FP8_BLOCK),
                                RAD_STR("a_order", "sorted"), RAD_STR("dtype", edtype[1]),
                                RAD_INT("N_odd", g.n_embd), RAD_INT("K_odd", wq[1]),
                                RAD_INT("parts", 1), RAD_INT("a_group", RAD_FP8_BLOCK) };
        const int cut = a8 ? 0 : 1;
        rad_op h = ncls == 4
            ? rad_decl_op(b, "moe_gemm_q_mod4", p4, (int)(sizeof p4 / sizeof *p4) - cut,
                          tab.data(), (int)tab.size())
            : rad_decl_op(b, "moe_gemm_q", p2, (int)(sizeof p2 / sizeof *p2) - cut,
                          tab.data(), (int)tab.size());
        op_dn = rw(b, h, {w.eff.cq(), w.eff.cs(), w.sorted, w.eoff}, {w.edn});
    }
    /* ---- The protected experts' own GEMM pair. Their weights belong to the layer group, not to
     * expert slots: each container expert unit has a fixed size and these are four times larger, so
     * they are static and resident with the router. One run per projection, a table of
     * `prot.size()`; the GEMMs take the trailing offsets, the rows the quantised down GEMM left
     * zero.
     *
     * When every rank holds a slice of every expert, it holds a slice of these too. A bf16 expert
     * has no scale blocks, so the slice is even where possible (rank r takes intermediate units
     * [r*w, (r+1)*w)), and the all-reduce sums the partials as for the other experts. */
    if (!prot.empty()) {
        const int64_t np = (int64_t)prot.size();
        int64_t wp = c.n_ff_exp;
        int     psh_row = RAD_SHARD_NONE, psh_col = RAD_SHARD_NONE, pparts = 0;
        /* This rank's range within one plain expert, in model numbering; `phi` 0 is the even split,
         * which the loader divides itself. */
        int64_t plo = 0, phi = 0;
        if (split) {
            /* The expert's whole width: `n_ff_all` (the max of `ff_hi` is the whole width only at
             * two ranks). A plain expert has no scale blocks, so the unit is the GEMM's 64-wide
             * output tile. */
            const int64_t full = g.n_ff_all > 0 ? g.n_ff_all
                                                : (c.ff_hi[0] > c.ff_hi[1] ? c.ff_hi[0]
                                                                          : c.ff_hi[1]);
            /* One slice per rank, possibly uneven in whole 64-wide tiles: 640 columns are ten
             * tiles, which four ranks take as 3, 3, 2, 2 (lower ranks take the extra). Each tile
             * has one owner and the all-reduce sums the partials. An evenly divisible width uses
             * the plain even split with no span. */
            psh_row = RAD_SHARD_ROW;
            psh_col = RAD_SHARD_COL;
            pparts  = (int)gu;
            if (full % g.world == 0 && (full / g.world) % 64 == 0) {
                wp = full / g.world;
            } else if (full % 64 || full / 64 < g.world) {
                fprintf(stderr, "radiance: layer %d's plain experts are %lld wide, which %d ranks "
                                "cannot share in 64-wide tiles\n", l, (long long)full, g.world);
                return RAD_E_INVAL;
            } else {
                const int64_t nt = full / 64, base = nt / g.world, extra = nt % g.world;
                const int64_t t0 = (int64_t)g.rank * base + std::min<int64_t>(g.rank, extra);
                const int64_t t1 = t0 + base + ((int64_t)g.rank < extra ? 1 : 0);
                plo = t0 * 64;
                phi = t1 * 64;
                wp  = phi - plo;
            }
        }
        w_pgu.assign((size_t)np, 0);
        w_pdn.assign((size_t)np, 0);
        for (int64_t j = 0; j < np; ++j) {
            const int64_t parts[2] = { wp, wp };
            w_pgu[(size_t)j] = decl_w_n(b, nm.f("blk.%d.ffn_gate_up_exps.%lld.weight", l,
                                                (long long)(c.expert_base + prot[(size_t)j])),
                                        RAD_BF16, {gu * wp, g.n_embd}, RAD_ACCESS_PER_TOKEN,
                                        psh_row, grp_layer(l), 0, parts, pparts);
            if (!w_pgu[(size_t)j]) return RAD_E_INVAL;
            if (phi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w_pgu[(size_t)j], plo, phi));
        }
        for (int64_t j = 0; j < np; ++j) {
            w_pdn[(size_t)j] = decl_w(b, nm.f("blk.%d.ffn_down_exps.%lld.weight", l,
                                              (long long)(c.expert_base + prot[(size_t)j])),
                                      RAD_BF16, {g.n_embd, wp}, RAD_ACCESS_PER_TOKEN, psh_col,
                                      grp_layer(l));
            if (!w_pdn[(size_t)j]) return RAD_E_INVAL;
            if (phi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w_pdn[(size_t)j], plo, phi));
        }
        /* The gate_up and gate outputs, bf16, one sorted row each; regular experts' rows are
         * unused. */
        b_pgu = decl_b(b, nm.f("blk.%d.moe_prot_gu", l), RAD_BF16, {c.rows * c.top_k, gu * wp});
        pff.x = decl_b(b, nm.f("blk.%d.moe_prot_ff", l), RAD_BF16, {c.rows * c.top_k, wp});
        if (!b_pgu || !pff.x) return RAD_E_INVAL;
        op_pgu = rw(b, rad_decl_op(b, "moe_gemm",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", gu * wp),
                                      RAD_INT("K", g.n_embd), RAD_INT("n_expert", np),
                                      RAD_INT("top_k", c.top_k), RAD_STR("a_order", "token"),
                                      RAD_STR("dtype", "bf16")),
                           w_pgu.data(), (int)np),
                    {w.h.x, w.sorted, w.eoff}, {b_pgu});
        Geom gx = g;
        gx.w_bf16 = true;
        RAD_ARCH_TRY(gq_prot.declare(b, gx, b_pgu, pff, wp, gact, c.rows * c.top_k, true,
                                     false, c.swiglu_limit, c.swiglu_step != 0));
        op_pdn = rw(b, rad_decl_op(b, "moe_gemm",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", g.n_embd),
                                      RAD_INT("K", wp), RAD_INT("n_expert", np),
                                      RAD_INT("top_k", c.top_k), RAD_STR("a_order", "sorted"),
                                      RAD_STR("dtype", "bf16")),
                           w_pdn.data(), (int)np),
                    {pff.x, w.sorted, w.eoff}, {w.edn});
        if (!op_pgu || !op_pdn) return RAD_E_INVAL;
    }
    }
    /* ---- the calibration tap, declared after the routed arm. Its PERSIST buffers would not be
     * aliased anyway, but declaration order is the buffer plan, so keep it last. */
    if (c.calib) {
        /* A rank holding a slice of each expert sees a slice of K, and the packer needs the full
         * Hessian, so calibration runs without expert slicing. */
        if (split) {
            fprintf(stderr, "radiance: calibration runs at one rank; this rank holds a slice of "
                            "every expert's width, and its Hessians would be of the slice\n");
            return RAD_E_INVAL;
        }
        if (!calib_dir()) {
            fprintf(stderr, "radiance: calibration is configured but RADIANCE_CALIB_DIR is not "
                            "set; there is nowhere to write the Hessians\n");
            return RAD_E_INVAL;
        }
        /* Both K must be multiples of 128 (as `expert_rot` also requires). gram_accum needs only a
         * multiple of four, but 128 keeps each 64-wide output tile inside one scale group. */
        if (!c.expert_bf16 && ((g.n_embd % RAD_FP8_BLOCK) || (c.n_ff_exp % RAD_FP8_BLOCK)))
            return RAD_E_SHAPE;
        gram_gu_stem = nm.f("blk.%d.ffn_gate_up_exps", l);
        gram_dn_stem = nm.f("blk.%d.ffn_down_exps", l);
        b_gram_gu = decl_b(b, nm.f("blk.%d.gram_gu", l), RAD_F32, {g.n_embd, g.n_embd},
                           RAD_BUF_PERSIST);
        b_gram_dn = decl_b(b, nm.f("blk.%d.gram_dn", l), RAD_F32, {c.n_ff_exp, c.n_ff_exp},
                           RAD_BUF_PERSIST);
        /* bf16 experts read the model's activation and the tap sums that; quantised ones read
         * codes, rotated when the plane is; the files record which. */
        gram_domain = rot ? (uint32_t)RAD_FP8_BLOCK : 0u;
        if (c.expert_bf16) {
            op_gram_gu = rw(b, RAD_OP(b, "gram_accum",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                          RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                           {w.h.x, b_gram_gu}, {b_gram_gu});
            op_gram_dn = rw(b, RAD_OP(b, "gram_accum",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                          RAD_INT("n", c.n_ff_exp), RAD_INT("group", RAD_FP8_BLOCK),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                           {w.eff.x, b_gram_dn}, {b_gram_dn});
        } else {
        op_gram_gu = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                      RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                           RAD_NOWEIGHTS),
                       {gin_q, gin_s, b_gram_gu}, {b_gram_gu});
        /* The down tap's M is the sorted height `rows * top_k` (one row per (token, slot)).
         * gate_up's is the token height: every token goes to exactly top_k experts, so the gathered
         * distribution is the token distribution scaled by top_k, and H is used only up to scale. */
        op_gram_dn = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                      RAD_INT("n", c.n_ff_exp),
                                      RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                           RAD_NOWEIGHTS),
                       {w.eff.q, w.eff.s, b_gram_dn}, {b_gram_dn});
        }
        /* One launch per expert over its run of sorted rows, into its square of the scratch.
         * Transient: it is copied to the host before the next op is issued. */
        b_gram_dne = decl_b(b, nm.f("blk.%d.gram_dn_experts", l), RAD_F32,
                            {c.n_expert * c.n_ff_exp, c.n_ff_exp});
        if (c.expert_bf16)
            op_gram_dne = rw(b, RAD_OP(b, "gram_accum",
                                RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                           RAD_INT("n", c.n_ff_exp),
                                           RAD_INT("group", RAD_FP8_BLOCK),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                             {w.eff.x, b_gram_dne}, {b_gram_dne});
        else
            op_gram_dne = rw(b, RAD_OP(b, "gram_accum",
                                RAD_PARAMS(RAD_RANGE("M", 1, c.rows * c.top_k),
                                           RAD_INT("n", c.n_ff_exp),
                                           RAD_INT("group", RAD_FP8_BLOCK),
                                           RAD_STR("dtype", "fp8a8")),
                                RAD_NOWEIGHTS),
                             {w.eff.q, w.eff.s, b_gram_dne}, {b_gram_dne});
        if (!b_gram_dne || !op_gram_dne) return RAD_E_INVAL;
    }

    /* The gather rounds the routed sum to BF16 before adding the rounded shared product.
     * At TP1 a unity gate therefore preserves HYV3's FP32 combine exactly. TP>1 must first
     * reduce each arm independently, and cannot combine them inside the local gather. */
    const bool separate_fp32_combine = shared && c.shared_fp32_combine &&
                                       (g.world > 1 || !c.shared_ungated);
    gfold = shared && !separate_fp32_combine;
    for (int form = scatter_inv ? 0 : 1; form < 2 && !op_gather; ++form) {
        rad_buf grd[5] = { w.edn, w.ew, form ? w.sorted : w.inv, w.sout, w.sgate };
        rad_buf gwr[1] = { separate_fp32_combine ? w.moe : w.h.x };
        op_gather = RAD_OP(b, form ? "moe_gather" : "moe_gather_inv",
                           RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", g.n_embd),
                                      RAD_INT("top_k", c.top_k), RAD_STR("dtype", g.dtype),
                                      RAD_STR("act", gfold && !c.shared_ungated ? "sigmoid" : "none")),
                           RAD_NOWEIGHTS);
        if (!op_gather) continue;
        gather_inv = !form;
        rad_op_reads (b, op_gather, grd, gfold ? 5 : 3);
        rad_op_writes(b, op_gather, gwr, 1);
    }
    if (!op_gather) return RAD_E_INVAL;

    /* ---- the shared expert. Ordinary dense linears over the same normed input. */
    if (shared) {
        /* The shared arm runs concurrently with the routed GEMMs, so its buffers are excluded from
         * sharing (rad_buf_concurrent). Otherwise the linear order would allow `sgu` over `egu`,
         * two streams writing the same bytes. Costs a few MiB of arena. */
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sgu));
        /* Probe the down projection first: what it reads (codes, or the bf16 plane for a plain
         * weight) decides what the gate writes. */
        const char* down_base = nm.f("blk.%d.ffn_down_shexp", l);
        const char* shared_down = src.shared_mlp_down
                                      ? src.shared_mlp_down
                                      : nm.ckpt("%s.down_proj.weight", src.shared);
        const char* shared_gate = src.shared_mlp_gate
                                      ? src.shared_mlp_gate
                                      : nm.ckpt("%s.gate_proj.weight", src.shared);
        const char* shared_up = src.shared_mlp_up
                                    ? src.shared_mlp_up
                                    : nm.ckpt("%s.up_proj.weight", src.shared);
        /* An architecture that sized the gate's code pair from this weight has probed it. */
        if (!sh_down.mapped) RAD_ARCH_TRY(sh_down.probe(b, nm, g, down_base, { shared_down }));
        /* The code pair the gate writes, E4M3 or int8 (ActFP8::q8_fed); an int8 down projection
         * that quantises for itself also writes the int8 pair on this lane. */
        if (w.sff.cq()) {
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.cq()));
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.cs()));
        }
        if (w.sff.q8 && !w.sff.q8_fed && sh_down.probed_i8) {
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.q8));
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.s8));
        }
        if (!w.sff.q || sh_limit>0 || (sh_down.reads_x && !(w.sff.q8_fed && sh_down.probed_i8)))
            RAD_ARCH_TRY(rad_buf_concurrent(b, w.sff.x));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sout));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.sgate));
        sgate_concurrent = true;
        /* gate/up rows and down columns of this rank's slice (even, or Config's uneven split); the
         * gate/up row parts stay, as a property of the stacked checkpoint tensor. */
        RAD_ARCH_TRY(sh_gate_up.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up_shexp", l),
                                        2 * c.n_ff_shared, g.n_embd, RAD_SHARD_ROW, grp_layer(l),
                                        { shared_gate, shared_up },
                                        w.h, w.sgu, c.rows,
                                        { c.n_ff_shared, c.n_ff_shared },
                                        c.shared_lo, c.shared_hi));
        /* An int8 down projection fed the int8 pair reads no bf16 plane, so none is written. */
        gq_sh.fuse_clamped = c.swiglu_quant != 0;
        RAD_ARCH_TRY(gq_sh.declare(b, g, w.sgu, w.sff, c.n_ff_shared, "silu", c.rows,
                                   sh_down.reads_x && !(w.sff.q8_fed && sh_down.probed_i8), false,
                                   sh_limit, c.swiglu_step != 0));
        /* An IQR gate_up between the rotated int8 trunk's fed pairs folds SwiGLU and the rotating
         * quantiser into its epilogue, to 16 rows: one launch a layer less than the pair, which
         * stays declared for longer steps, other encodings and a clamped product. */
        if (sh_limit <= 0 && sh_gate_up.iqr && w.h.q8_fed && w.h.q8_rot && w.sff.q8_fed &&
            w.sff.q8_rot)
            op_sh_gated = decl_gated_fold_iqr(
                b, c.rows, sh_gate_up.N, sh_gate_up.K, sh_gate_up.w, sh_gate_up.ws,
                { w.h.q8, w.h.s8 },
                gq_sh.want_x ? std::initializer_list<rad_buf>{ w.sff.q8, w.sff.s8, w.sff.x }
                             : std::initializer_list<rad_buf>{ w.sff.q8, w.sff.s8 },
                &sh_gated_rows);
        RAD_ARCH_TRY(sh_down.declare(b, nm, g, down_base,
                                     g.n_embd, c.n_ff_shared, RAD_SHARD_COL, grp_layer(l),
                                     { shared_down },
                                     w.sff, w.sout, c.rows, {},
                                     c.shared_lo, c.shared_hi));
        /* The shared gate, replicated like the router so every rank computes the same gate. One
         * output column: the skinny GEMM kernel clamps its N tile rather than masking, so N = 1 is
         * legal. An ungated arm (shared_ungated) fills the gate with 1 instead. */
        if (c.shared_ungated) {
            op_sgate = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n", 1),
                                        RAD_STR("dtype", g.dtype), RAD_F64("value", 1.0)),
                             RAD_NOWEIGHTS),
                         {}, {w.sgate});
        } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_gate_shexp.weight", l), src.shared_gate));
        w_sgate = decl_w(b, nm.f("blk.%d.ffn_gate_shexp.weight", l), RAD_BF16, {1, g.n_embd},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        op_sgate = rw(b, RAD_OP(b, "gemm_nt",
                         RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", 1),
                                    RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_sgate)),
                     {w.h.x}, {w.sgate});
        }
    }

    if (separate_fp32_combine) {
        /* BF16 add widens both inputs to FP32, adds, then rounds once to BF16: the same bytes as
         * cast + cast + F32 add + cast. The routed gather's BF16 rounding and the separate TP
         * reductions stay; a local gather cannot fold those reductions. */
        op_combine = rw(b, RAD_OP(b, "add",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                       {w.moe, w.sout}, {w.h.x});
        if (!op_combine) return RAD_E_INVAL;
    }

    /* ---- The router GEMM and `router_topk_scatter` as one op, plus the shared gate when the
     * layer has one; up to the pair's row bound and the skinny GEMM's decode band. Unresolved, the
     * separate ops run. Declared here, after the gate weight, because it names both weights;
     * declaring late only lengthens its buffers' planned lifetimes, since each is also declared by
     * the op it replaces.
     *
     * Exception: it writes `w.sgate` before `op_sgate`'s declared index, so the planner could lend
     * those bytes to a buffer live in between. That is safe only because the shared arm excluded
     * `w.sgate` from sharing (rad_buf_concurrent); otherwise the fused gate is refused.
     *
     * The parameter list, weights and write set are conditional, so each is an array cut to
     * length. */
    if (op_topk_scatter && !c.shared_ungated && rad_op_declared(b, "router_gemm_topk_scatter")) {
        router_fused_rows = topk_scatter_rows < kMoeRouterFusedRows ? topk_scatter_rows
                                                                    : kMoeRouterFusedRows;
        const bool gate = op_sgate != 0;
        if (gate && !sgate_concurrent) {
            fprintf(stderr, "radiance: layer %d: the fused router writes the shared gate ahead of "
                            "its declared writer, which needs `sgate` out of the sharing pool, "
                            "and it is not\n", l);
            return RAD_E_STATE;
        }
        const RadParam fp[] = {
            RAD_RANGE("M", 1, router_fused_rows), RAD_INT("K", g.n_embd),
            RAD_INT("n_expert", c.n_expert_all), RAD_INT("n_local", c.n_expert),
            RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm_topk),
            RAD_STR("dtype", g.dtype), RAD_INT("expert_base", c.expert_base),
            RAD_STR("protect", prot_list),
        };
        const int n_fp = (int)(sizeof(fp) / sizeof(fp[0])) - (prot_list ? 0 : 1);
        const rad_weight fw[2] = { w_router, w_sgate };
        op_router_fused = rad_decl_op(b, "router_gemm_topk_scatter", fp, n_fp, fw, gate ? 2 : 1);
        if (op_router_fused) {
            const rad_buf rd[1] = { w.h.x };
            const rad_buf wr[7] = { w.logits, w.ids, w.ew, w.sorted, w.eoff, w.ecnt, w.sgate };
            rad_op_reads (b, op_router_fused, rd, 1);
            rad_op_writes(b, op_router_fused, wr, gate ? 7 : 6);
        }
        router_fused_gate = op_router_fused && gate;
    }

    /* GLM's selection and the sort as one op, plus the ungated shared arm's gate fill. Declared
     * here for the reason the fused router above is: it writes `w.sgate` ahead of `op_sgate`'s
     * declared index, which is safe only with `sgate` out of the sharing pool. Its other outputs
     * and its input are also declared by the ops it replaces, so their planned lifetimes cover it.
     * Undeclared or unresolved, the separate ops serve. */
    if (c.sigmoid_router && c.router_fuse) {
        if (!rad_op_declared(b, "router_sigmoid_topk_scatter")) {
            static bool told = false;
            if (!told) rad_note(b, "no loaded library declares router_sigmoid_topk_scatter; the "
                                   "selection and the sort stay separate launches");
            told = true;
        } else {
            sigmoid_scatter_rows = (int64_t)c.rows * c.top_k <= kMoeSortRows
                                       ? c.rows : kMoeSortRows / c.top_k;
            const bool gate = op_sgate != 0 && c.shared_ungated && !c.shared_fp32_combine;
            if (gate && !sgate_concurrent) {
                fprintf(stderr, "radiance: layer %d: the fused selection writes the shared gate "
                                "ahead of its declared writer, which needs `sgate` out of the "
                                "sharing pool, and it is not\n", l);
                return RAD_E_STATE;
            }
            const RadParam sp[] = {
                RAD_RANGE("M", 1, sigmoid_scatter_rows), RAD_INT("n_expert", c.n_expert_all),
                RAD_INT("n_local", c.n_expert), RAD_INT("top_k", c.top_k),
                RAD_INT("norm", c.norm_topk), RAD_F64("route_scale", c.route_scale),
                RAD_STR("dtype", "f32"), RAD_INT("expert_base", c.expert_base),
                RAD_STR("protect", prot_list),
            };
            const int n_sp = (int)(sizeof(sp) / sizeof(sp[0])) - (prot_list ? 0 : 1);
            op_sigmoid_scatter = rad_decl_op(b, "router_sigmoid_topk_scatter", sp, n_sp,
                                             &glm_router.correction, 1);
            if (op_sigmoid_scatter) {
                const rad_buf rd[1] = { w.logits };
                rad_buf wr[7] = { w.ids, w.ew, w.sorted, w.eoff, w.ecnt };
                int n_wr = 5;
                if (gate) wr[n_wr++] = w.sgate;
                if (scatter_inv) wr[n_wr++] = w.inv;
                rad_op_reads (b, op_sigmoid_scatter, rd, 1);
                rad_op_writes(b, op_sigmoid_scatter, wr, n_wr);
            }
            sigmoid_scatter_gate = op_sigmoid_scatter && gate;
        }
    }

    if (g.world > 1 && !c.no_reduce) {
        const auto all_reduce = [&](const char* dtype, rad_buf buffer) {
            return rw(b, RAD_OP(b, "all_reduce",
                         RAD_PARAMS(RAD_INT("world_size", g.world),
                                    RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                    RAD_STR("dtype", dtype), RAD_INT("exact", g.wire_exact),
                                    RAD_INT("min_bytes", g.wire_min_bytes)),
                         RAD_NOWEIGHTS),
                     {buffer}, {buffer});
        };
        if (separate_fp32_combine) {
            /* Each arm is a tensor-parallel partial. Reduce the BF16 arm outputs independently,
             * then perform HYV3's shared+routed addition in F32 before the final BF16 cast. */
            op_moe_ar = all_reduce(g.dtype, w.moe);
            op_shared_ar = all_reduce(g.dtype, w.sout);
        } else {
            op_ar = all_reduce(g.dtype, w.h.x);
        }
    }
    /* The same collective for a caller that runs it beside other work (issue_ar, a pipelined
     * prefill): at `overlap` 1 it gets the rows that hold fewer CUs while waiting on the link. */
    if (g.world > 1 && !c.no_reduce && ar_overlap > 0)
        op_ar_ov = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes),
                                     RAD_INT("overlap", ar_overlap)),
                          RAD_NOWEIGHTS),
                      {w.h.x}, {w.h.x});

    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h.x}, {w.x});

    /* The layer as the prefetch sees it, indexed by model expert number (what the router ranks). */
    if (!c.sigmoid_router && !c.expert_bf16 && c.n_expert_all > 0 && n_reg > 0) {
        const int64_t na = c.n_expert_all;
        xp_parts.assign((size_t)(na * 4), 0);
        xp_count_at.assign((size_t)na, -1);
        for (int64_t s = 0; s < n_reg; ++s) {
            const int64_t e = c.expert_base + reg[(size_t)s];
            if (e < 0 || e >= na) continue;
            xp_parts[(size_t)(e * 4 + 0)] = w_gu[(size_t)s];
            xp_parts[(size_t)(e * 4 + 1)] = w_gus[(size_t)s];
            xp_parts[(size_t)(e * 4 + 2)] = w_dn[(size_t)s];
            xp_parts[(size_t)(e * 4 + 3)] = w_dns[(size_t)s];
            xp_count_at[(size_t)e] = (int32_t)s;
        }
        xp.layer = layer;
        xp.router = w_router;
        xp.n_expert = na;
        xp.top_k = c.top_k;
        xp.n_parts = 4;
        xp.parts = xp_parts.data();
        xp.count_at = xp_count_at.data();
    }
    return RAD_OK;
}

/* The shared arm's first stage: its one-column gate and its gate_up over the normed input. Its
 * buffers are step-tall, so the pass offset applies as in an MlpFP8 slice. */
inline void MoeFP8::shared_up(RadCtx* c_, int64_t T, int64_t r0, int64_t rows,
                              bool gate_done) const {
    /* `gate_done`: the fused router already wrote this pass's gate. A separate FP32 combine
     * reads no ones plane. At TP1 an ungated folded gather does, unless the caller prepared the
     * whole step's ones plane once (shared_one_external). */
    if (!gate_done && c.shared_ungated) {
        if (gfold && !c.shared_one_external)
            RAD_ISSUE_N(c_, op_sgate, rows, brow_slice(w.sgate, r0, rows, 1));
    } else if (!gate_done) {
        RAD_ISSUE_N(c_, op_sgate, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_W(w_sgate),
                    brow_slice(w.sgate, r0, rows, 1));
    }
    if (op_sh_gated && rows <= sh_gated_rows)
        RAD_ISSUE_N(c_, op_sh_gated, rows, brow_slice(w.h.cq(), r0, rows, g.n_embd),
                    brow_slice(w.h.cs(), r0, rows, g.n_embd / RAD_FP8_BLOCK), RAD_W(sh_gate_up.w),
                    RAD_W(sh_gate_up.ws), brow_slice(w.sff.cq(), r0, rows, c.n_ff_shared),
                    brow_slice(w.sff.cs(), r0, rows, c.n_ff_shared / RAD_FP8_BLOCK),
                    gq_sh.want_x ? brow_slice(w.sff.x, r0, rows, c.n_ff_shared) : RAD_NONE);
    else
        sh_gate_up.step(c_, w.h, w.sgu, T, r0, rows);
}

/* One pass of at most `c.rows` tokens. Every op is per row (norm and quantiser within a row,
 * router and top-k per row, per-sorted-row GEMM accumulation, the gather per token), so a pass
 * computes exactly its rows.
 *
 * The expert buffers are addressed from zero, not `r0`: they are a pass tall and `sorted_tok` is
 * pass-relative. Only step-tall operands (residual stream, normed input) carry the offset. */
inline void MoeFP8::pass(RadCtx* c_, int64_t T, int64_t r0, int64_t rows, bool own_ar) const {
    /* `ne` is this rank's expert count, `ne_all` the model's: logits and top-k are model-wide,
     * everything after the scatter is this rank's slice. */
    const int64_t ne = c.n_expert, ne_all = c.n_expert_all, k = c.top_k;
    const bool shared = op_sgate != 0;

    if (!ext_in) nq_h.step(c_, T, r0, rows);

    /* The shared arm is interleaved with the routed arm on the same queue. It reads the same
     * input and writes nothing the routed arm touches before the gather, so each of its stages
     * (gate + gate_up, quantiser, down) is issued right behind the matching routed stage (its
     * first stage behind the router instead where `sh_first` below says so). A recorded pass
     * leaves each such pair unordered (Ctx::haz_mark), so they overlap without a second queue's
     * fork/join cost. The arm's buffers are excluded from sharing (see declare), so
     * issuing them ahead of their declared position is safe. */

    /* The routed arm. */
    /* Predict the next routed layer's experts from this router's input, on the compute stream
     * (rad_expert_prefetch). A failure only loses the prediction. */
    if (xp_next && xp_next->xp.n_parts > 0)
        (void)rad_expert_prefetch(c_, &xp_next->xp, w.h.x, r0, rows, g.n_embd);
    /* Bind the block's own ecnt buffer, never the core's raw routing row: a raw-pointer operand
     * does not survive the step tape's replay (the decode path re-issues ops from the tape), so
     * the histogram the kernel writes there never lands, and a consumer reading the reported
     * counts -- the file stager, at the layer's trigger -- sees zeros and skips every expert's
     * weights. The declared buffer is stable across record and replay, and route_report hands
     * the same buffer on, so the stager's stream-ordered read sees this pass's counts. With a
     * routing table (Wire::ecnt_pitch) the buffer is the core's rows too, and this is the
     * layer's. */
    const RadOperand ecnt = w.ecnt_pitch > 0 ? brow_slice(w.ecnt, layer, 1, w.ecnt_pitch)
                                             : brows(w.ecnt, ne);
    /* The fused router when it resolved and the pass fits: router, topk + scatter and (with
     * `router_fused_gate`) the shared gate in one launch. Otherwise the separate ops. */
    const bool rfused = op_router_fused && rows <= router_fused_rows;
    const bool sfused = op_sigmoid_scatter && rows <= sigmoid_scatter_rows;
    if (sfused) {
        glm_router.project(c_, r0, rows, r0);
        RAD_ISSUE_N(c_, op_sigmoid_scatter, rows, brow_slice(w.logits, r0, rows, ne_all),
                    RAD_W(glm_router.correction), brow_slice(w.ids, r0, rows, k),
                    brow_slice(w.ew, r0, rows, k), brows(w.sorted, rows * k),
                    brows(w.eoff, ne + 1), ecnt,
                    sigmoid_scatter_gate ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    scatter_inv ? brows(w.inv, rows * k) : RAD_NONE);
    } else if (c.sigmoid_router) {
        glm_router.step(c_, r0, rows, r0);
    } else if (rfused)
        RAD_ISSUE_N(c_, op_router_fused, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_W(w_router), router_fused_gate ? RAD_W(w_sgate) : RAD_NONE,
                    brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt,
                    router_fused_gate ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE);
    else
        RAD_ISSUE_N(c_, op_router, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_W(w_router),
                    brow_slice(w.logits, r0, rows, ne_all));
    /* topk + scatter fused, or the pair. In the fused op `w.ids` is an output. */
    if (rfused || sfused) {
        /* the fused router wrote all five */
    } else if (op_topk_scatter && rows <= topk_scatter_rows) {
        RAD_ISSUE_N(c_, op_topk_scatter, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    } else {
        if (!c.sigmoid_router)
        RAD_ISSUE_N(c_, op_topk, rows, brow_slice(w.logits, r0, rows, ne_all),
                    brow_slice(w.ids, r0, rows, k), brow_slice(w.ew, r0, rows, k));
        if (scatter_inv)
            RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt,
                        brows(w.inv, rows * k));
        else
            RAD_ISSUE_N(c_, op_scatter, rows, brow_slice(w.ids, r0, rows, k),
                        brows(w.sorted, rows * k), brows(w.eoff, ne + 1), ecnt);
    }

    /* The rotated quantisation of the input, when the experts are rotated and the caller did not
     * write it. Unrotated, the GEMM reads `h`'s pair. */
    const bool rot = c.expert_rot != 0;
    if (op_hq)
        RAD_ISSUE_N(c_, op_hq, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(gin_q, r0, rows, g.n_embd),
                    brow_slice(gin_s, r0, rows, g.n_embd / gin_g));

    /* The grouped GEMMs. Each table is the first handle of its run and its length; runs are
     * contiguous (see declare), so `w_gu[q] + i` is expert 2 * i + q, or 4 * i + q at four
     * classes, where class q holds ceil((n_reg - q) / 4). */
    const int64_t n4[4] = { (n_reg + 3) / 4, (n_reg + 2) / 4, (n_reg + 1) / 4, n_reg / 4 };
    /* Behind a trellis gate_up from kSharedFirstRows routed rows, the shared arm's first stage is
     * issued here, so it overlaps the router and the routed gate_up runs alone: that GEMM is
     * compute-bound at such heights, and with the arm beside it the pair ends later than the two
     * in turn. Flash-Next's 16- and 32-row verify passes on two R9700: 22.1 -> 21.9 ms and
     * 31.6 -> 30.95 ms. Below, the arm fits inside the routed GEMM's time and stays beside it. */
    const bool sh_first = shared && gu_trellis && rows * k >= kSharedFirstRows;
    if (sh_first)
        shared_up(c_, T, r0, rows,
                  (rfused && router_fused_gate) || (sfused && sigmoid_scatter_gate));
    /* Apply the previous layer's prefetch prediction: substitute the copied experts into the
     * tables. */
    (void)rad_expert_prefetch_step(c_, layer);
    if (c.expert_bf16)
        RAD_ISSUE_N(c_, op_gu, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_WTAB(w_gu[0], ne), brows(w.sorted, rows * k), brows(w.eoff, ne + 1),
                    brows(w.egu, rows * k));
    else if (ncls == 4)
        RAD_ISSUE_N(c_, op_gu, rows,
                    brow_slice(gin_q, r0, rows, g.n_embd),
                    brow_slice(gin_s, r0, rows, g.n_embd / gin_g),
                    RAD_WTAB(w_gu[0], n4[0]), RAD_WTAB(w_gus[0], n4[0]),
                    brows(w.sorted, rows * k), brows(w.eoff, n_reg + 1),
                    RAD_WTAB(w_gu[1], n4[1]), RAD_WTAB(w_gus[1], n4[1]),
                    RAD_WTAB(w_gu[2], n4[2]), RAD_WTAB(w_gus[2], n4[2]),
                    RAD_WTAB(w_gu[3], n4[3]), RAD_WTAB(w_gus[3], n4[3]),
                    brows(w.egu, rows * k));
    else
        RAD_ISSUE_N(c_, op_gu, rows,
                    brow_slice(gin_q, r0, rows, g.n_embd),
                    brow_slice(gin_s, r0, rows, g.n_embd / gin_g),
                    RAD_WTAB(w_gu[0], n_reg / 2), RAD_WTAB(w_gus[0], n_reg / 2),
                    brows(w.sorted, rows * k), brows(w.eoff, n_reg + 1),
                    RAD_WTAB(w_gu[1], n_reg / 2), RAD_WTAB(w_gus[1], n_reg / 2),
                    brows(w.egu, rows * k));
    if (shared && !sh_first)
        shared_up(c_, T, r0, rows,
                  (rfused && router_fused_gate) || (sfused && sigmoid_scatter_gate));
    gq_exp.step(c_, w.egu, rows * k, 0, rows * k);
    if (shared && !(op_sh_gated && rows <= sh_gated_rows)) gq_sh.step(c_, w.sgu, T, r0, rows);
    if (c.expert_bf16)
        RAD_ISSUE_N(c_, op_dn, rows, brows(w.eff.x, rows * k), RAD_WTAB(w_dn[0], ne),
                    brows(w.sorted, rows * k), brows(w.eoff, ne + 1), brows(w.edn, rows * k));
    else if (ncls == 4)
        RAD_ISSUE_N(c_, op_dn, rows,
                    brows(w.eff.cq(), rows * k), brows(w.eff.cs(), rows * k),
                    RAD_WTAB(w_dn[0], n4[0]), RAD_WTAB(w_dns[0], n4[0]),
                    brows(w.sorted, rows * k), brows(w.eoff, n_reg + 1),
                    RAD_WTAB(w_dn[1], n4[1]), RAD_WTAB(w_dns[1], n4[1]),
                    RAD_WTAB(w_dn[2], n4[2]), RAD_WTAB(w_dns[2], n4[2]),
                    RAD_WTAB(w_dn[3], n4[3]), RAD_WTAB(w_dns[3], n4[3]),
                    brows(w.edn, rows * k));
    else
        RAD_ISSUE_N(c_, op_dn, rows,
                    brows(w.eff.cq(), rows * k), brows(w.eff.cs(), rows * k),
                    RAD_WTAB(w_dn[0], n_reg / 2), RAD_WTAB(w_dns[0], n_reg / 2),
                    brows(w.sorted, rows * k), brows(w.eoff, n_reg + 1),
                    RAD_WTAB(w_dn[1], n_reg / 2), RAD_WTAB(w_dns[1], n_reg / 2),
                    brows(w.edn, rows * k));
    /* This layer's streamed experts are consumed; the next layer's copies may start. */
    (void)rad_expert_prefetch_mark(c_);
    /* The protected experts, over the trailing offsets: the quantised down GEMM zeroed their
     * sorted rows as past its live count, and this down GEMM writes them afterwards (ordered by
     * the shared `edn`). */
    if (op_pgu) {
        const int64_t np = (int64_t)prot.size();
        const RadOperand poff = brow_slice(w.eoff, n_reg, np + 1, 1);
        RAD_ISSUE_N(c_, op_pgu, rows, brow_slice(w.h.x, r0, rows, g.n_embd),
                    RAD_WTAB(w_pgu[0], np), brows(w.sorted, rows * k), poff,
                    brows(b_pgu, rows * k));
        gq_prot.step(c_, b_pgu, rows * k, 0, rows * k);
        RAD_ISSUE_N(c_, op_pdn, rows, brows(pff.x, rows * k), RAD_WTAB(w_pdn[0], np),
                    brows(w.sorted, rows * k), poff, brows(w.edn, rows * k));
    }
    if (shared) sh_down.step(c_, w.sff, w.sout, T, r0, rows);
    /* The calibration tap reads the GEMMs' operands and writes nothing they touch, so it does not
     * change the model's output. */
    if (op_gram_gu && !calib_done) {
        if (calib_rows == 0) {
            /* The accumulators are arena memory, so zero them first. */
            rad_memset_async(rad_buf_ptr(c_, b_gram_gu), 0,
                             g.n_embd * g.n_embd * 4, rad_stream(c_));
            rad_memset_async(rad_buf_ptr(c_, b_gram_dn), 0,
                             c.n_ff_exp * c.n_ff_exp * 4, rad_stream(c_));
        }
        if (c.expert_bf16) {
            RAD_ISSUE_N(c_, op_gram_gu, rows, brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE,
                        brows(b_gram_gu, g.n_embd));
            RAD_ISSUE_N(c_, op_gram_dn, rows * k, brows(w.eff.x, rows * k), RAD_NONE,
                        brows(b_gram_dn, c.n_ff_exp));
        } else {
        RAD_ISSUE_N(c_, op_gram_gu, rows,
                    brow_slice(rot ? w.hr.q : w.h.q, r0, rows, g.n_embd),
                    brow_slice(rot ? w.hr.s : w.h.s, r0, rows,
                               g.n_embd / RAD_FP8_BLOCK),
                    brows(b_gram_gu, g.n_embd));
        RAD_ISSUE_N(c_, op_gram_dn, rows * k,
                    brows(w.eff.q, rows * k), brows(w.eff.s, rows * k),
                    brows(b_gram_dn, c.n_ff_exp));
        }
        if (op_gram_dne) calib_experts_pass(c_, rows * k);
        calib_rows += rows;
        if (calib_rows >= calib_target()) {
            const bool a = calib_write(c_, b_gram_gu, g.n_embd, calib_rows, gram_gu_stem,
                                       gram_domain);
            const bool bb = calib_write(c_, b_gram_dn, c.n_ff_exp, calib_rows * k,
                                        gram_dn_stem, gram_domain);
            const bool ce = !op_gram_dne ||
                            (!calib_dne_failed &&
                             calib_dne.write(gram_dn_stem, c.expert_base, gram_domain,
                                             rad_rank(c_)));
            /* Written: the host sums are done with (a layer's can be gigabytes). */
            calib_dne.acc = {};
            calib_dne.rows = {};
            calib_done = true;
            if (layer == 0)
                std::fprintf(stderr, "radiance: calibration wrote layer Hessians after %lld "
                                     "tokens to %s%s\n", calib_rows, calib_dir(),
                             (a && bb && ce) ? "" : ", with errors");
        }
    }

    /* The gather last, since with the fold it reads the shared arm (`sout`, `sgate`); it writes
     * the block's delta into `h.x`. */
    if (op_combine || !gather_taken(g, T, gather_out_rows, gather_out_rows6))
        RAD_ISSUE_N(c_, op_gather, rows, brows(w.edn, rows * k), brow_slice(w.ew, r0, rows, k),
                    brows(gather_inv ? w.inv : w.sorted, rows * k),
                    gfold ? brow_slice(w.sout, r0, rows, g.n_embd) : RAD_NONE,
                    gfold ? brow_slice(w.sgate, r0, rows, 1) : RAD_NONE,
                    brow_slice(op_combine ? w.moe : w.h.x, r0, rows, g.n_embd));
    if (op_moe_ar)
        RAD_ISSUE_N(c_, op_moe_ar, rows * g.n_embd,
                    brow_slice(w.moe, r0, rows, g.n_embd), RAD_NONE);
    if (op_shared_ar)
        RAD_ISSUE_N(c_, op_shared_ar, rows * g.n_embd,
                    brow_slice(w.sout, r0, rows, g.n_embd), RAD_NONE);
    if (op_combine)
        RAD_ISSUE_N(c_, op_combine, rows, brow_slice(w.moe, r0, rows, g.n_embd),
                    brow_slice(w.sout, r0, rows, g.n_embd),
                    brow_slice(w.h.x, r0, rows, g.n_embd));

    /* A write that takes the all-reduce does it once for the whole step, deciding on T, so this
     * pass must ask with T, not its own rows. */
    if (own_ar && issues_ar(T, rows)) issue_ar(c_, r0, rows, false);
    if (op_add)
        RAD_ISSUE_N(c_, op_add, rows, brow_slice(w.x, r0, rows, g.n_embd),
                    brow_slice(w.h.x, r0, rows, g.n_embd),
                    brow_slice(w.x, r0, rows, g.n_embd));
}

/* One pass of the per-expert tap. The scatter's expert offsets are copied back first (they decide
 * the launches); then the scratch is zeroed, each expert's run of the `slots` sorted rows is summed
 * into its square, and the scratch is copied back and added on the host. Both syncs are on the
 * block's stream with nothing issued after the tap, so the transient scratch never overlaps a live
 * buffer's use. */
inline void MoeFP8::calib_experts_pass(RadCtx* c_, int64_t slots) const {
    if (calib_dne_failed) return;
    const int64_t ne = c.n_expert, K = c.n_ff_exp;
    const RadStream s = rad_stream(c_);
    if (calib_dne.ne == 0) calib_dne.init(ne, K);

    std::vector<int32_t> off((size_t)(ne + 1));
    rad_memcpy_async(off.data(), rad_buf_ptr(c_, w.eoff), (ne + 1) * 4, s);
    rad_stream_sync(s);
    if (off[0] != 0 || off[(size_t)ne] < 0 || off[(size_t)ne] > slots) {
        std::fprintf(stderr, "radiance: layer %d's expert offsets end at %d of %lld sorted rows; "
                             "the per-expert calibration stops\n",
                     layer, off[(size_t)ne], (long long)slots);
        calib_dne_failed = true;
        return;
    }

    const int64_t bytes = ne * K * K * 4;
    float* staging = calib_staging(c_, bytes);
    if (!staging) { calib_dne_failed = true; return; }
    void* scratch = rad_buf_ptr(c_, b_gram_dne);
    rad_memset_async(scratch, 0, bytes, s);
    for (int64_t e = 0; e < ne; ++e) {
        const int64_t r0 = off[(size_t)e], n = off[(size_t)e + 1] - r0;
        if (n <= 0) continue;
        if (c.expert_bf16)
            RAD_ISSUE_N(c_, op_gram_dne, n, brow_slice(w.eff.x, r0, n, K), RAD_NONE,
                        brow_slice(b_gram_dne, e * K, K, K));
        else
            RAD_ISSUE_N(c_, op_gram_dne, n, brow_slice(w.eff.q, r0, n, K),
                        brow_slice(w.eff.s, r0, n, K / RAD_FP8_BLOCK),
                        brow_slice(b_gram_dne, e * K, K, K));
        calib_dne.rows[(size_t)e] += n;
    }
    rad_memcpy_async(staging, scratch, bytes, s);
    rad_stream_sync(s);
    calib_dne.add(staging);
}

inline void MoeFP8::step(RadCtx* c_, const RadBatch* batch) const {
    step_rows(c_, batch->n_tok);
}

inline void MoeFP8::step_rows(RadCtx* c_, int64_t T) const {
    for (int64_t r0 = 0; r0 < T; r0 += c.rows) {
        const int64_t n = (T - r0 < c.rows) ? (T - r0) : c.rows;
        pass(c_, T, r0, n);
    }
    route_report(c_);
}

inline void MoeFP8::route_report(RadCtx* c_) const {
    /* ---- Hand the routing to the heat engine. `rad_route_report` copies the histogram back
     * asynchronously and the engine consumes it next step, so the compute stream is not blocked
     * (spec §5.5). Without it, expert-tiered placement has nothing to promote on.
     *
     * It reports the last pass, which is the whole step by default (`c.rows` is the prefill
     * chunk). With more passes it is a sample, adequate for a popularity estimate.
     *
     * `expert_w` stays null: RadRouting declares it `const float*` but these weights are bf16
     * (f32 under the sigmoid router). Nothing in the core reads it (the heat engine uses
     * `expert_count`). */
    /* The router kernel writes the block's own ecnt buffer (w.ecnt) every pass, so report THAT
     * pointer: a consumer reading it stream-ordered (the file stager's trigger runs after this
     * layer's router on the same stream) sees this pass's counts. Reporting rad_route_counts'
     * row instead names a buffer the router never writes; the pointer then equals the report's
     * destination row, the copy here is skipped, and the row stays zero -- which the file stager
     * read as "no expert routed" and skipped every layer's weights (NaN logits, empty text).
     * Copying w.ecnt into the row (the pointers now differ) also feeds the heat engine real
     * histograms instead of the zeros it saw before. With a routing table (Wire::ecnt_pitch) the
     * router did write the core's row, as a declared operand, and the report copies nothing. */
    moe_debug_weights(c_, layer, c.top_k, w.ids, w.ew, c.sigmoid_router != 0);
    RadRouting r{};
    r.expert_ids    = (const int32_t*)rad_buf_ptr(c_, w.ids);
    r.expert_w      = nullptr;
    r.expert_count  = (const int32_t*)rad_buf_ptr(c_, w.ecnt) + (int64_t)layer * w.ecnt_pitch;
    r.sorted_tok    = (const int32_t*)rad_buf_ptr(c_, w.sorted);
    r.expert_offset = (const int32_t*)rad_buf_ptr(c_, w.eoff);
    r.top_k         = c.top_k;
    r.n_expert      = c.n_expert;
    rad_route_report(c_, layer, &r);
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MOE_FP8_H */
