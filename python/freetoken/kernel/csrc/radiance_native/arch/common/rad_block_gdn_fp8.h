/* rad_block_gdn_fp8.h -- the gated-delta-net block with block-scaled FP8 projections.
 *
 * The architecture (A_log convention, in-kernel L2 norm of q and k, the two declared paths,
 * `num_accepted`) is documented in rad_block_gdn.h and does not change with the weight format.
 *
 * ============================== a AND b SPLIT OUT; z FUSED ==============================
 *
 * rad_block_gdn.h fuses in_proj_qkv, in_proj_a and in_proj_b into one weight. Here the
 * checkpoint prevents it:
 *
 *   in_proj_qkv.weight   F8_E4M3 [10240, 5120]   with weight_scale_inv [80, 40]
 *   in_proj_z.weight     F8_E4M3 [ 6144, 5120]   with weight_scale_inv [48, 40]
 *   in_proj_a.weight     BF16    [48, 5120]      -- no scale plane
 *   in_proj_b.weight     BF16    [48, 5120]      -- no scale plane
 *
 * a and b have one row per value head (48, not a whole 128-row block), so the quantiser left them
 * bf16 and they cannot be concatenated with E4M3 tensors. They become a separate small bf16
 * `gemm_nt` ([96, 5120], 0.5% of the layer's projection bytes). gdn_conv_prep and
 * gdn_recurrent_update take a and b as separate operands in both blocks (column views of the
 * fused row there, of their own buffer here); libref sizes its `x` read as
 * `conv_dim + (a present ? 0 : 2*H)` for this case.
 *
 * The output gate is fused instead, into `ssm_inz`: in_proj_z is E4M3 with whole scale blocks and
 * reads the same normed activation, so [q|k|v|z] is one weight and one launch. `w.in.x` is then
 * conv_dim + v_dim wide, and the convolution input and the gate are column views of it; the
 * consuming kernels all take a row pitch.
 *
 * ============================== THE OUT-PROJECTION'S INPUT IS RANK 3 ==============================
 *
 * `o` is [tokens, value heads, head width] because libr4d's chunk-scan and recurrent kernels need
 * rank 3. The quantiser reads dim 0 as rows and the rest as the row (as r4d_row_pitch does and
 * the GEMM's K requires); one row per head would put scales on a grid the GEMM does not read.
 */
#ifndef RAD_BLOCK_GDN_FP8_H
#define RAD_BLOCK_GDN_FP8_H

#include "rad_fp8.h"

#include <cmath>
#include <cstring>

namespace rad {
namespace arch {

struct GdnFP8 {
    /* The bf16 block's geometry plus the gate activation, uneven head splits, and `ab_dim`
     * (2 * n_head_v), the width of the separate a|b projection. */
    struct Config {
        int64_t n_head_k   = 0;
        int64_t head_k     = 0;
        int64_t n_head_v   = 0;
        int64_t head_v     = 0;
        int64_t conv_width = 4;
        int64_t chunk      = 64;
        /* The output gate's activation, a per-checkpoint property: `RMSNormGated` defaults to
         * silu (every Qwen3.5/3.6 release), while Qwen4-Exp sets `output_gate_type: "sigmoid"`.
         * libr4d's r4d_gdn_gated_rmsnorm (`gn_act`) and `gdn_recurrent_update` implement both.
         * "silu" or "sigmoid"; anything else is refused. */
        const char* act    = "silu";
        /* Uneven head split (rad_weight_shard_span_parts). n_head_k / n_head_v are this rank's;
         * k_first / v_first are its first heads within the container's n_head_k_all /
         * n_head_v_all. k_first < 0 means the plain even row shard. Three ranks over 16 K heads
         * take 6, 5 and 5 (18, 15, 15 of 48 V heads), with no padding. */
        int64_t k_first = -1, v_first = -1;
        int64_t n_head_k_all = 0, n_head_v_all = 0;
        bool    uneven() const { return k_first >= 0; }

        int64_t k_dim()    const { return n_head_k * head_k; }
        int64_t v_dim()    const { return n_head_v * head_v; }
        int64_t conv_dim() const { return 2 * k_dim() + v_dim(); }
        int64_t ab_dim()   const { return 2 * n_head_v; }
    };

    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]           the residual stream */
        ActFP8  h{};        /* [max_tok, n_embd]           normed input, then the out projection */
        /* [q|k|v] and the output gate from one GEMM (ssm_inz): the convolution reads the left
         * `conv_dim` columns and the gate is the right `v_dim`, both as `bcol` views. */
        ActFP8  in{};       /* [max_tok, conv_dim+v_dim]   [q|k|v] then the output gate */
        rad_buf ab   = 0;   /* [max_tok, 2*n_head_v]       the decay and beta inputs, bf16 */
        rad_buf q    = 0;   /* [max_tok, n_head_k, head_k] post-convolution, l2-normed */
        rad_buf k    = 0;   /* [max_tok, n_head_k, head_k] */
        rad_buf v    = 0;   /* [max_tok, n_head_v, head_v] */
        rad_buf gdec = 0;   /* [max_tok, n_head_v] */
        rad_buf beta = 0;   /* [max_tok, n_head_v] */
        rad_buf kkt  = 0;   /* [max_tok, n_head_v, chunk] */
        ActFP8  o{};        /* [max_tok, n_head_v, head_v] rank 3, see the header */
    };

    struct Src {
        const char* norm     = nullptr;
        const char* in_qkv   = nullptr;   /* fp8 */
        const char* in_a     = nullptr;   /* bf16 */
        const char* in_b     = nullptr;   /* bf16 */
        const char* in_z     = nullptr;   /* fp8 */
        const char* conv1d   = nullptr;
        const char* a_log    = nullptr;
        const char* dt_bias  = nullptr;
        const char* out_norm = nullptr;
        const char* out      = nullptr;   /* fp8 */
    };

    Geom        g{};
    Config      cfg{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv_state = 0, kv_conv = 0;

    rad_weight w_norm = 0, w_ab = 0, w_conv = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    /* Set by the architecture before declare: the a|b projection runs on the second lane beside the
     * in-projection at decode and verify sizes. Not where lane 1 carries collectives, which the
     * projection would queue behind. Its concurrent buffers cost memory (at Qwen3.8-27B TP2: +70
     * MiB arena and +33 MiB queues a card) for about +1 % at one stream. */
    bool       ab_lane = false;
    rad_weight w_a_log = 0, w_dt_bias = 0, w_out_norm = 0;
    LinearFP8  in{}, out{};
    NormQuantFP8 nq_h{};
    QuantFP8   q_o{};

    rad_op op_ab = 0;
    rad_op op_conv_prep = 0, op_kkt = 0, op_scan = 0, op_gnorm = 0;   /* prefill */
    rad_op op_conv_update = 0, op_recur = 0;                          /* decode */
    /* `gdn_conv_recurrent_update`, if it resolves: the pair above in one launch, with the
     * convolution in the recurrent update's prologue. The pair is declared either way; cr_rows is
     * the widest decode window the fold serves. */
    rad_op  op_cr = 0;
    /* Whether op_cr writes the out projection's codes; otherwise q_o follows it. */
    bool    cr_codes = false;
    int64_t cr_rows = 0;
    /* Set by the architecture after declare: a `prefetch` of the out-projection's leading bytes
     * (decl_prefetch, rad_arch.h), issued beside the decode recurrent update on steps of up to
     * pf_out_rows rows and pf_out_seqs sequences. 0 where none is declared. */
    rad_op  op_pf_out = 0;
    int64_t pf_out_rows = 0, pf_out_seqs = 0;
    rad_op op_ar = 0, op_add = 0;
    /* The next block absorbs this block's closing all-reduce (see rad_block_attn_gated_fp8.h). */
    bool ar_out = false;
    /* Which wires that consumer takes (ArOut, rad_fp8.h); set before declare. */
    int  ar_out_take = kArOutNorm;

    /* `fold`: the previous block left its delta in `w.h.x` for this norm to add. `add_out`: the
     * next block will not add, so this one issues the trailing add. See rad_fp8.h. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                 rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    /* `own_ar` false: the caller issues this step's all-reduce itself, a row slice at a time
     * (issue_ar), to overlap it with other work (a pipelined prefill). Only on a block without a
     * trailing add, which would read the delta before the reduction. */
    void step(RadCtx* c, const RadBatch* batch, bool own_ar = true,
              ArPieces* pieces = nullptr) const;
    /* Whether step() with `own_ar` false runs the whole block a piece at a time over `pieces`
     * (its piece path), asked by a caller that must know before it commits to a schedule. */
    bool takes_pieces(const RadBatch* batch, const ArPieces* pieces) const {
        return pieces && pieces->cu && pieces->cuts(batch->n_tok) &&
               batch->n_tok <= 2 * pieces->rows && (ext_in || pieces->in_ready) && !op_add &&
               batch->n_seq == 1 && batch->phase == RAD_PHASE_PREFILL;
    }
    /* Whether a step of T rows issues the all-reduce itself, asked by a caller before taking it
     * over; and that all-reduce over rows [r0, r0 + rows) of the output. */
    bool issues_ar(int64_t T) const { return op_ar && !ar_taken(g, T, ar_out, ar_out_take); }
    void issue_ar(RadCtx* c, int64_t r0, int64_t rows) const {
        RAD_ISSUE_N(c, op_ar_ov ? op_ar_ov : op_ar, rows * g.n_embd,
                    brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    }
    /* Set before declare: the `overlap` issue_ar's collective is declared at, 0 for none. */
    int    ar_overlap = 0;
    rad_op op_ar_ov = 0;
};

inline int GdnFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                           rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s,
                           bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom;
    ar_out = ar_out_; cfg = c; w = wire; layer = l; kv_state = state; kv_conv = conv;
    if (cfg.n_head_k <= 0 || cfg.head_k <= 0 || cfg.n_head_v <= 0 || cfg.head_v <= 0 ||
        cfg.conv_width <= 0 || cfg.chunk <= 0)
        return RAD_E_INVAL;
    if (cfg.n_head_v % cfg.n_head_k != 0) {
        fprintf(stderr, "radiance: %lld value heads is not a multiple of %lld key heads\n",
                (long long)cfg.n_head_v, (long long)cfg.n_head_k);
        return RAD_E_INVAL;
    }

    /* The query scale is required; rad_block_gdn.h explains why the gated norm does not cancel
     * it. */
    const double q_scale = 1.0 / std::sqrt((double)cfg.head_k);

    if (!cfg.act || (std::strcmp(cfg.act, "silu") != 0 && std::strcmp(cfg.act, "sigmoid") != 0)) {
        fprintf(stderr, "radiance: the delta net's output gate is '%s'; libr4d's gated norm "
                        "implements silu and sigmoid and nothing else\n", cfg.act ? cfg.act : "(null)");
        return RAD_E_INVAL;
    }
    const int64_t v_dim    = cfg.v_dim();
    const int64_t conv_dim = cfg.conv_dim();
    const int64_t ab_dim   = cfg.ab_dim();

    /* `s.norm == nullptr`: the architecture prepares the input (a gated-residual model's
     * rad_block_hc.h `hc_read`), so `wire.h` is already normed and quantised and no norm weight is
     * declared. `ar_in` is then refused, since the fused all-reduce lives inside the norm. See
     * rad_block_attn_gated_fp8.h. */
    ext_in = (s.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), s.norm));
        w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }
    /* Maps before weights: weight encodings are looked up through the map. */
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ssm_ab.weight", l),     { s.in_a, s.in_b }, 0));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_conv1d.weight", l), s.conv1d));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_a_log", l),         s.a_log));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_dt.bias", l),       s.dt_bias));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_norm.weight", l),   s.out_norm));
    /* The decay and beta projections, bf16 as in the checkpoint. Row-sharded with their value
     * heads: the two-part split gives rank r rows [r*n_head_v, (r+1)*n_head_v) of each of a and
     * b, which a contiguous slice of the concatenation would not. */
    w_ab       = decl_w(b, nm.f("blk.%d.ssm_ab.weight", l),     RAD_BF16, {ab_dim, g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l), 0,
                        {cfg.n_head_v, cfg.n_head_v});
    /* The depthwise convolution shards with its channels, [q|k|v] (three parts). */
    w_conv     = decl_w(b, nm.f("blk.%d.ssm_conv1d.weight", l), RAD_BF16,
                        {conv_dim, cfg.conv_width},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l), 0,
                        {cfg.k_dim(), cfg.k_dim(), cfg.v_dim()});
    w_a_log    = decl_w(b, nm.f("blk.%d.ssm_a_log", l),         RAD_F32, {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    w_dt_bias  = decl_w(b, nm.f("blk.%d.ssm_dt.bias", l),       RAD_F32, {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    w_out_norm = decl_w(b, nm.f("blk.%d.ssm_norm.weight", l),   RAD_F32, {cfg.head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* Norm and quantise in one op (rad_fp8.h: NormQuantFP8); the qkv|z and a|b projections read
     * it. */
    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));
    /* Two checkpoint tensors (in_proj_qkv = [q|k|v], in_proj_z = the gate), four projections, one
     * GEMM. At decode two calls at 5120 and 3072 rows cost more than one at 8192, since each pays
     * its own ramp and tail. Bit-identical: row arithmetic is unchanged and both extents are past
     * the split-K threshold.
     *
     * The four row parts are required so rank r owns its slice of each of q, k, v and z; one
     * contiguous cut would give rank 0 all of q and k. */
    RAD_ARCH_TRY(in.declare(b, nm, g, nm.f("blk.%d.ssm_inz", l), conv_dim + v_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { s.in_qkv, s.in_z }, w.h, w.in.x, 0,
                            { cfg.k_dim(), cfg.k_dim(), cfg.v_dim(), cfg.v_dim() }, 0, 0,
                            /* a rotated weight was rotated per checkpoint tensor: q|k|v, then z */
                            { 2 * cfg.k_dim() + cfg.v_dim(), cfg.v_dim() }));
    if (cfg.uneven()) {
        /* This rank's heads of each part, in container rows: q and k by K head, v and z by V head.
         * The scale plane of a row shard has the same rows (its blocks run along K). */
        const int64_t kl = cfg.k_first * cfg.head_k, vl = cfg.v_first * cfg.head_v;
        const int64_t ka = cfg.n_head_k_all * cfg.head_k, va = cfg.n_head_v_all * cfg.head_v;
        const int64_t lo[4] = { kl, kl, vl, vl };
        const int64_t hi[4] = { kl + cfg.k_dim(), kl + cfg.k_dim(), vl + cfg.v_dim(), vl + cfg.v_dim() };
        const int64_t tot[4] = { ka, ka, va, va };
        RAD_ARCH_TRY(in.span_parts(b, lo, hi, tot, 4));
        const int64_t clo[3] = { kl, kl, vl }, chi[3] = { kl + cfg.k_dim(), kl + cfg.k_dim(), vl + cfg.v_dim() };
        const int64_t ctot[3] = { ka, ka, va };
        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w_conv, clo, chi, ctot, 3));
        const int64_t alo[2] = { cfg.v_first, cfg.v_first };
        const int64_t ahi[2] = { cfg.v_first + cfg.n_head_v, cfg.v_first + cfg.n_head_v };
        const int64_t atot[2] = { cfg.n_head_v_all, cfg.n_head_v_all };
        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w_ab, alo, ahi, atot, 2));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_a_log,   cfg.v_first, cfg.v_first + cfg.n_head_v));
        RAD_ARCH_TRY(rad_weight_shard_span(b, w_dt_bias, cfg.v_first, cfg.v_first + cfg.n_head_v));
    }
    /* One launch over all rows: libr4d's bf16 kernel for this 96-column shape is general in M
     * (MT row tiles per block, grid.y covers the rest). A kernel capped at M = 64 would cap every
     * op in the step, and so the prefill chunk, at 64 tokens (see brow_slice in rad_arch.h).
     *
     * At decode the cost is launch count: `gemm_nt` is the most-dispatched decode kernel, nearly
     * all fixed cost, across three small per-layer projections of the same activation (the MoE
     * router, the shared expert's gate on rank 0 only, and this a|b). Merging them would need
     * contiguous weights (a re-convert) or a grouped gemm_nt over a weight-pointer table, as
     * moe_gemm_q does. */
    op_ab = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", ab_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_ab)), {w.h.x}, {w.ab});
    if (ab_lane) {
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.h.x));
        RAD_ARCH_TRY(rad_buf_concurrent(b, w.ab));
    }

    /* ---- the chunked path. As in the bf16 block, except `a` and `b_gate` are views of `ab`. */
    op_conv_prep = rw(b, RAD_OP(b, "gdn_conv_prep",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                     RAD_INT("head_v", cfg.head_v), RAD_INT("chunk", cfg.chunk),
                                     RAD_INT("conv_width", cfg.conv_width)),
                          RAD_WEIGHTS(w_conv, w_a_log, w_dt_bias)),
                      {w.in.x, w.ab}, {w.q, w.k, w.v, w.gdec, w.beta});
    op_kkt  = rw(b, RAD_OP(b, "gdn_kkt_solve",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k)),
                     RAD_NOWEIGHTS),
                 {w.k, w.beta, w.gdec}, {w.kkt});
    op_scan = rw(b, RAD_OP(b, "gdn_chunk_scan",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                RAD_INT("head_v", cfg.head_v), RAD_F64("scale", q_scale)),
                     RAD_NOWEIGHTS),
                 {w.q, w.k, w.v, w.kkt, w.gdec, w.beta}, {w.o.x});
    op_gnorm = rw(b, RAD_OP(b, "gdn_gated_rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("channels", cfg.head_v),
                                 RAD_F64("eps", g.eps), RAD_STR("act", cfg.act)),
                      RAD_WEIGHTS(w_out_norm)),
                  {w.o.x, w.in.x}, {w.o.x});

    /* ---- the recurrent path. */
    op_conv_update = rw(b, RAD_OP(b, "gdn_conv_update",
                            RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                       RAD_INT("head_k", cfg.head_k),
                                       RAD_INT("head_v", cfg.head_v),
                                       RAD_INT("conv_width", cfg.conv_width)),
                            RAD_WEIGHTS(w_conv)),
                        {w.in.x}, {w.q, w.k, w.v});
    /* `state_form`: anchor when speculating, else per_candidate (rad_block_gdn.h explains). */
    op_recur = rw(b, RAD_OP(b, "gdn_recurrent_update",
                      RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                 RAD_INT("head_v", cfg.head_v),
                                 RAD_INT("n_head_v", cfg.n_head_v),
                                 RAD_F64("scale", q_scale),
                                 RAD_F64("eps", g.eps), RAD_STR("act", cfg.act),
                                 RAD_STR("state_form",
                                         g.max_spec > 0 ? "anchor" : "per_candidate")),
                      RAD_WEIGHTS(w_a_log, w_dt_bias, w_out_norm)),
                  {w.q, w.k, w.v, w.ab, w.in.x}, {w.o.x});
    /* The epilogue's codes for an fp8 or int8 out-projection (int8 when ActFP8::q8_fed; the codes
     * dtype picks the quantiser). A bf16 out-projection reads only `o.x` and these stay absent.
     * Rotated int8 codes (ActFP8::q8_rot) only the conv fold writes (gdn_conv_recurrent_update_had);
     * rows this update serves are quantised by q_o. */
    if (op_recur && w.o.cq() && !w.o.q8_rot) {
        const rad_buf oq[2] = { w.o.cq(), w.o.cs() };
        RAD_ARCH_TRY(rad_op_writes(b, op_recur, oq, 2));
    }
    /* The conv + recurrent fold. Its window is a register preload, so the band stops at eight
     * tokens per sequence. Optional (no RAD_ARCH_TRY): without it the pair runs. */
    {
        const int64_t hi = g.max_tok < 8 ? g.max_tok : 8;
        /* The fold's rotated epilogue writes one scale per (row, 128), or per 256 for an IQ8R
         * trunk's codes (ActFP8::g8; `group`, stated to that form alone: last, and cut off
         * otherwise), where this rank's value heads pair up. */
        const bool cr_wide = w.o.q8_rot && w.o.g8 == RAD_I8_GROUP_WIDE && cfg.n_head_v % 2 == 0;
        cr_codes = w.o.cq() && (!w.o.q8_rot || w.o.g8 == RAD_FP8_BLOCK || cr_wide);
        const RadParam cp[] = { RAD_RANGE("q_len", 1, hi), RAD_INT("head_k", cfg.head_k),
                                RAD_INT("head_v", cfg.head_v),
                                RAD_INT("conv_width", cfg.conv_width),
                                RAD_INT("n_head_v", cfg.n_head_v), RAD_F64("scale", q_scale),
                                RAD_F64("eps", g.eps), RAD_STR("act", cfg.act),
                                RAD_STR("state_form", g.max_spec > 0 ? "anchor" : "per_candidate"),
                                RAD_INT("group", RAD_I8_GROUP_WIDE) };
        const rad_weight cw[] = { w_conv, w_a_log, w_dt_bias, w_out_norm };
        op_cr = rw(b, rad_decl_op(b, w.o.q8_rot && cr_codes ? "gdn_conv_recurrent_update_had"
                                                            : "gdn_conv_recurrent_update",
                                  cp, (int)(sizeof cp / sizeof *cp) - (cr_wide && cr_codes ? 0 : 1),
                                  cw, 4),
                   {w.in.x, w.ab}, {w.o.x});
        if (op_cr && cr_codes) {
            const rad_buf oq[2] = { w.o.cq(), w.o.cs() };
            RAD_ARCH_TRY(rad_op_writes(b, op_cr, oq, 2));
        }
        if (op_cr) cr_rows = hi;
    }

    RAD_ARCH_TRY(q_o.declare(b, g, w.o, v_dim));
    RAD_ARCH_TRY(out.declare(b, nm, g, nm.f("blk.%d.ssm_out", l), g.n_embd, v_dim,
                             RAD_SHARD_COL, grp_layer(l), { s.out }, w.o, w.h.x, 0, {},
                             cfg.uneven() ? cfg.v_first * cfg.head_v : 0,
                             cfg.uneven() ? cfg.v_first * cfg.head_v + v_dim : 0));

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});
    /* The same collective for a caller that runs it beside other work (issue_ar, a pipelined
     * prefill): at `overlap` 1 it gets the rows that hold fewer CUs while waiting on the link. */
    if (g.world > 1 && ar_overlap > 0)
        op_ar_ov = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes),
                                     RAD_INT("overlap", ar_overlap)),
                          RAD_NOWEIGHTS),
                      {w.h.x}, {w.h.x});
    /* Only when the next block's norm will not add (rad_fp8.h: NormQuantFP8). */
    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h.x}, {w.x});
    return RAD_OK;
}

inline void GdnFP8::step(RadCtx* c, const RadBatch* batch, bool own_ar,
                         ArPieces* pieces) const {
    const RadKVGroupBatch* st = kv_batch(batch, kv_state);
    const RadKVGroupBatch* cv = kv_batch(batch, kv_conv);
    const int32_t* st_idx = st ? st->state_index : nullptr;
    /* Slots per state_index row: the state, plus scratch when a kernel serving speculation asks
     * for it (RadKVGroupBatch). Passed through unread. */
    const int64_t st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;

    const int64_t T = batch->n_tok;
    const int64_t S = batch->n_seq;
    const int64_t H = cfg.n_head_v;
    /* The two parts of `w.in.x`: the convolution's [q|k|v], then the output gate. */
    const int64_t conv_dim = cfg.conv_dim();
    const int64_t v_dim    = cfg.v_dim();

    /* One prefill sequence in two pieces (ArPieces::cu): the whole block a piece at a time, each
     * piece the chunk a span of 1 makes (the projection over its rows; the convolution, solve and
     * scan over its bounds, with the state handed on through the cache; the gated norm, quantiser
     * and out projection), so the first piece's reduction runs beside all of the second piece's
     * work. The second piece reads the history the first left in the convolution slot, so each
     * piece passes its own context length, its first row's position: the step's would tell the
     * second piece of a prompt's first chunk that it has none. */
    if (!own_ar && takes_pieces(batch, pieces)) {
        const int64_t in_w = conv_dim + v_dim;
        for (int64_t p = 0; p < 2; ++p) {
            const int64_t r0 = p * pieces->rows;
            const int64_t n  = p == 0 ? pieces->rows : T - pieces->rows;
            const int32_t* cu = pieces->cu + p;
            in.step(c, w.h, w.in.x, T, r0, n);
            RAD_ISSUE_N(c, op_ab, n, brow_slice(w.h.x, r0, n, g.n_embd), RAD_W(w_ab),
                        brow_slice(w.ab, r0, n, 2 * H));
            RAD_ISSUE_N(c, op_conv_prep, n,
                        bcol(w.in.x, 0, conv_dim, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log),
                        RAD_W(w_dt_bias), kv_cache(kv_conv, layer), praw(cu, RAD_I32, 2),
                        bcol(w.ab, 0, H, T), bcol(w.ab, H, H, T),
                        praw(cv_idx, RAD_I32, 1), praw(batch->positions + r0, RAD_I32, 1),
                        brows(w.q, T), brows(w.k, T), brows(w.v, T),
                        brows(w.gdec, T), brows(w.beta, T));
            RAD_ISSUE_N(c, op_kkt, n,
                        brows(w.k, T), brows(w.beta, T), brows(w.gdec, T),
                        praw(cu, RAD_I32, 2), brows(w.kkt, T));
            RAD_ISSUE_N(c, op_scan, n,
                        brows(w.q, T), brows(w.k, T), brows(w.v, T), brows(w.kkt, T),
                        brows(w.gdec, T), brows(w.beta, T),
                        kv_cache(kv_state, layer), praw(cu, RAD_I32, 2),
                        brows(w.o.x, T), kv_cache(kv_state, layer),
                        praw2(st_idx, RAD_I32, 1, st_w));
            RAD_ISSUE_N(c, op_gnorm, n,
                        brow_slice(w.o.x, r0, n, v_dim),
                        bcol_at(w.in.x, r0, in_w, conv_dim, v_dim, n),
                        RAD_W(w_out_norm), brow_slice(w.o.x, r0, n, v_dim));
            q_o.step(c, T, r0, n);
            out.step(c, w.o, w.h.x, T, r0, n);
            if (p == 0) pieces->hand(r0, n);
        }
        return;
    }

    if (!ext_in) nq_h.step(c, T);
    /* At decode and verify sizes the a|b projection is nearly all launch cost and reads only the
     * norm's output, as the in-projection does, so it runs beside it on lane 1 and is joined before
     * the convolution reads `ab`. A prefill chunk keeps one lane, where the two would share
     * bandwidth. Where launches chain it stays on lane 0: its chained form starts beside the
     * in-projection without the two lane joins, each of which holds a launch back until the host
     * has seen the other lane finish (Qwen3.8-27B on two cards: the projection started 55 us late
     * and the update behind it another 20 to 40 us). */
    if (ab_lane && !rad_chain_on() && T <= 64 && rad_lane_join(c, 0, 1) == RAD_OK &&
        rad_lane(c, 1) == RAD_OK) {
        RAD_ISSUE_N(c, op_ab, T, brows(w.h.x, T), RAD_W(w_ab), brows(w.ab, T));
        rad_lane(c, 0);
        in.step(c, w.h, w.in.x, T);
        rad_lane_join(c, 1, 0);
    } else {
        in.step(c, w.h, w.in.x, T);
        RAD_ISSUE_N(c, op_ab, T, brows(w.h.x, T), RAD_W(w_ab), brows(w.ab, T));
    }

    /* Each path runs over its own run of sequences. The step is sorted decode-first: decode rows
     * are sequences [0, D) with tokens [0, DT), prefill chunks are [D, S) with the rest. A pure
     * step issues one path.
     *
     * Per-sequence arrays are offset, token operands are not. Every kernel here derives its grid
     * and rows from cu_seqlens (N = len(cu) - 1, a sequence's first row is the absolute cu[n]), so
     * the prefill half gets `cu + D` with q, k, v at their base; offsetting those too would count
     * the displacement twice. The decode half is the prefix: cu ends at DT and its token operands
     * are cut to DT rows. The gated norm and the standalone quantiser take no cu and are per
     * token, so they take row slices at DT.
     *
     * This keeps decode-only sequences in a mixed step off the chunked path, which commits state
     * directly and has no num_accepted, so drafts verified there could not be rolled back. */
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    /* The decode half's band. In a mixed step max_q_len is the prefill chunk's; in a pure decode
     * step it is already right, so a batch leaving max_q_len_decode zero still works there. */
    const int32_t qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode : batch->max_q_len;
    const int64_t P  = S - D;
    const int64_t PT = T - DT;
    const int64_t in_w = conv_dim + v_dim;      /* w.in.x's declared row width */

    /* The recurrent update at decode heights runs one workgroup per (sequence, head) and leaves the
     * card's memory mostly idle, so the out-projection's weight is read into the on-die cache
     * beside it: in front of the update on lane 0 where launches chain (the read's chained form
     * holds nothing behind it), else on the second lane behind a join from lane 0 (as
     * prefetch_beside). */
    if (op_pf_out && D > 0 && D <= pf_out_seqs && P == 0 && T <= pf_out_rows) {
        if (prefetch_lane0()) {
            RAD_ISSUE(c, op_pf_out, RAD_W(out.w));
        } else {
            if (rad_lane_join(c, 0, 1) == RAD_OK && rad_lane(c, 1) == RAD_OK)
                RAD_ISSUE(c, op_pf_out, RAD_W(out.w));
            rad_lane(c, 0);
        }
    }
    if (D > 0 && op_cr && qd <= cr_rows) {
        RAD_ISSUE_N(c, op_cr, qd,
                    bcol(w.in.x, 0, conv_dim, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D),
                    bcol(w.ab, 0, H, DT), bcol(w.ab, H, H, DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    bcol(w.in.x, conv_dim, v_dim, DT), RAD_W(w_out_norm), brows(w.o.x, DT),
                    cr_codes ? brows(w.o.cq(), DT) : RAD_NONE,
                    cr_codes ? brows(w.o.cs(), DT) : RAD_NONE);
        if (w.o.cq() && !cr_codes) q_o.step(c, T, 0, DT);
    } else if (D > 0) {
        RAD_ISSUE_N(c, op_conv_update, qd,
                    bcol(w.in.x, 0, conv_dim, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT));
        RAD_ISSUE_N(c, op_recur, qd,
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT),
                    bcol(w.ab, 0, H, DT), bcol(w.ab, H, H, DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    bcol(w.in.x, conv_dim, v_dim, DT), RAD_W(w_out_norm), brows(w.o.x, DT),
                    w.o.cq() && !w.o.q8_rot ? brows(w.o.cq(), DT) : RAD_NONE,
                    w.o.cq() && !w.o.q8_rot ? brows(w.o.cs(), DT) : RAD_NONE);
        /* Rotated codes come only from the fold; here the decode rows take the quantiser. */
        if (w.o.q8_rot) q_o.step(c, T, 0, DT);
    }

    /* The output a piece at a time where the caller takes the all-reduce that way (ArPieces): the
     * gated norm, quantiser and out projection are per row, so a pure prefill step can finish them
     * a piece at a time and hand off each piece's reduction behind it. */
    const bool cut = !own_ar && pieces && pieces->cuts(T) && D == 0 && P > 0 && !op_add;
    if (P > 0) {
        RAD_ISSUE(c, op_conv_prep,
                  bcol(w.in.x, 0, conv_dim, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log), RAD_W(w_dt_bias),
                  kv_cache(kv_conv, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  bcol(w.ab, 0, H, T), bcol(w.ab, H, H, T),
                  praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
                  praw(batch->ctx_lens + D, RAD_I32, P),
                  brows(w.q, T), brows(w.k, T), brows(w.v, T),
                  brows(w.gdec, T), brows(w.beta, T));
        RAD_ISSUE(c, op_kkt,
                  brows(w.k, T), brows(w.beta, T), brows(w.gdec, T),
                  praw(batch->cu_seqlens + D, RAD_I32, P + 1), brows(w.kkt, T));
        RAD_ISSUE(c, op_scan,
                  brows(w.q, T), brows(w.k, T), brows(w.v, T), brows(w.kkt, T),
                  brows(w.gdec, T), brows(w.beta, T),
                  kv_cache(kv_state, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  brows(w.o.x, T), kv_cache(kv_state, layer),
                  praw2(st_idx ? st_idx + D * st_w : nullptr, RAD_I32, P, st_w));
        if (cut) {
            for (int64_t r0 = 0; r0 < T; r0 += pieces->rows) {
                const int64_t n = T - r0 < pieces->rows ? T - r0 : pieces->rows;
                RAD_ISSUE_N(c, op_gnorm, n,
                            brow_slice(w.o.x, r0, n, v_dim),
                            bcol_at(w.in.x, r0, in_w, conv_dim, v_dim, n),
                            RAD_W(w_out_norm), brow_slice(w.o.x, r0, n, v_dim));
                q_o.step(c, T, r0, n);
                out.step(c, w.o, w.h.x, T, r0, n);
                if (r0 + n < T) pieces->hand(r0, n);
            }
            return;
        }
        /* Per token, so a row slice over the prefill rows only; the recurrent kernel already
         * gated the decode rows. */
        RAD_ISSUE_N(c, op_gnorm, PT,
                    brow_slice(w.o.x, DT, PT, v_dim),
                    bcol_at(w.in.x, DT, in_w, conv_dim, v_dim, PT),
                    RAD_W(w_out_norm), brow_slice(w.o.x, DT, PT, v_dim));
        /* Only the prefill rows are quantised here; the decode kernel quantises in its epilogue.
         * The chunked scan cannot: its workgroup is a chunk, not a row, so it cannot reduce a
         * row group's amax. */
        q_o.step(c, T, DT, PT);
    }

    out.step(c, w.o, w.h.x, T);
    if (own_ar && op_ar && !ar_taken(g, T, ar_out, ar_out_take))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_GDN_FP8_H */
