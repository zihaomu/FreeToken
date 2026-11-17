/* rad_fp8.h -- the pieces every block-scaled FP8 block is built from: a linear whose weight is E4M3
 * with a block scale plane, the activation quantisers that feed it, and the fused norm/residual/
 * all-reduce front op.
 *
 * ============================== BLOCK-SCALED FP8 ==============================
 *
 * The checkpoint's format. Qwen3.8-27B-FP8 and other
 * `quantization_config: {quant_method: fp8, weight_block_size: [128,128]}` releases store each
 * linear as
 *
 *   <linear>.weight             F8_E4M3, row-major [N][K]
 *   <linear>.weight_scale_inv   BF16,    row-major [ceil(N/128)][ceil(K/128)]
 *
 * and libr4d's fp8 GEMMs read those bytes (grid verified in libr4d/r4d_fp8_layout.cpp), so
 * rad-convert copies them unchanged. Widening a 27B FP8 checkpoint to bf16 would double the
 * weight bytes.
 *
 * ============================== THE SCALE PLANE IS A WEIGHT ==============================
 *
 * `gemm_nt_q` takes `a`, `a_scale`, `b`(w), `b_scale`(w) -> `y`, so the scale plane is its own
 * rad_weight rather than an `aux` stream: RadArgs cannot deliver an aux stream to a launch
 * (libr4d/r4d_fp8_layout.cpp), and a weight can be placed, moved and load-time sharded on its own
 * (a row-sharded [136,40] plane is sliced like its [17408,5120] weight). The cost is one extra
 * weight per linear (2.1 KiB of scales against 89 MiB at gate_proj's shape) and a name-map entry,
 * `<src>_scale_inv`, derived here.
 *
 * ============================== FP8 ACTIVATIONS ==============================
 *
 * Every block-fp8 linear is `fp8a8`, whose two kernels cover every M (`nt_m16` to 16, `tiled`
 * above), at the cost of quantising each activation that feeds a GEMM. libr4d's `fp8a16` matvec,
 * which reads bf16 directly, is not used: it assumes 16 consecutive k of a row per lane, which the
 * stored WMMA fragment order (libr4d/r4d_fp8_frag.h) does not provide.
 *
 * Quantisation is per activation, not per linear: the three attention projections, or the GDN's
 * in and z projections, share one quantised input. Hence QuantFP8 is separate from LinearFP8.
 */
#ifndef RAD_FP8_H
#define RAD_FP8_H

#include "rad_arch.h"

#include <cstring>
#include <vector>

namespace rad {
namespace arch {

/* The scale block in both dimensions, fixed by the checkpoint (`weight_block_size`). libr4d
 * compiles it in and its rows carry C_EQ("group", 128), so other widths resolve to no kernel. */
enum { RAD_FP8_BLOCK = 128 };
/* IQ8R's scale group: an int8 weight rotated 128 wide with one bf16 scale per row per 256 of K,
 * met by an int8 activation with one f32 scale per (token, 256). */
enum { RAD_I8_GROUP_WIDE = 256 };

/* Rows of the scale plane for an N-row weight, and columns for a K-wide one. */
inline int64_t fp8_blocks(int64_t n) { return (n + RAD_FP8_BLOCK - 1) / RAD_FP8_BLOCK; }

/* An activation and its quantised codes. The bf16 buffer is what the rest of the graph uses; the
 * codes exist only between the quantiser and the GEMMs that consume them. */
struct ActFP8 {
    rad_buf x = 0;   /* [max_tok, n]                bf16 */
    rad_buf q = 0;   /* [max_tok, n]                e4m3 */
    rad_buf s = 0;   /* [max_tok, ceil(n/block)]    f32 */
    /* Int8 codes of the same activation, for an int8 linear (LinearFP8::i8): [max_tok, n] i8 and
     * an f32 scale per (row, block). Declared only for an int8 trunk (declare_qs's `i8`); written
     * by the linear's own quant_act_i8g from `x`, or by the producer when `q8_fed`. */
    rad_buf q8 = 0;
    rad_buf s8 = 0;
    /* The int8 pair's scale group: RAD_FP8_BLOCK, or RAD_I8_GROUP_WIDE for an IQ8R trunk, whose
     * producers and quantisers write one scale per (row, 256). s8 has ceil(n / g8) columns. */
    int64_t g8 = RAD_FP8_BLOCK;
    /* The producer writes q8/s8 instead of the E4M3 pair (a connection read with
     * HyperConn::Config::codes_i8, a gated quantiser, the attention gate or delta-net output
     * epilogue), so an int8 linear reading it declares no quantiser. Set by the model when it
     * declares the pair, before the wires are copied. */
    bool    q8_fed = false;
    /* With q8_fed: the producer's int8 codes are rotated, had_quant_act_i8g's (a `_had_quant_i8g`
     * producer), so only a weight stored rotated (LinearFP8 at i8a8h) reads them, and it declares
     * no quantiser of its own. Set by the model with q8_fed. */
    bool    q8_rot = false;

    /* The code pair the producer writes: int8 when the model fed it, E4M3 otherwise. */
    rad_buf cq() const { return q8_fed ? q8 : q; }
    rad_buf cs() const { return q8_fed ? s8 : s; }
    /* The scale group of cs(). */
    int64_t cg() const { return q8_fed ? g8 : (int64_t)RAD_FP8_BLOCK; }

    /* Declares the codes: the E4M3 pair with `fp8_pair`, the int8 pair with `i8` (an int8 model's
     * activations need no E4M3 plane taking arena); `x` is set by the caller.
     *
     * `rows` defaults to g.max_tok and must be given for any buffer that is not one row per token
     * (a MoE expert plane is `rows * top_k`, an encoder tower's its patch count); too small a value
     * overruns the arena (core/runtime/issue.cpp refuses such an issue). */
    int declare_qs(RadBuilder* b, Names& nm, const Geom& g, const char* name, int64_t n,
                   int64_t rows = 0, bool i8 = false, bool fp8_pair = true,
                   int64_t i8_group = RAD_FP8_BLOCK) {
        g8 = i8_group;
        /* bf16 linears read only `x` (Geom::w_bf16), so declare no codes; an unused plane would
         * still take arena space. */
        if (g.w_bf16) { q = s = q8 = s8 = 0; return RAD_OK; }
        const int64_t m = rows > 0 ? rows : g.max_tok;
        if (fp8_pair) {
            q = decl_b(b, nm.f("%s.q", name), RAD_F8E4M3, { m, n });
            s = decl_b(b, nm.f("%s.s", name), RAD_F32,    { m, fp8_blocks(n) });
            if (!q || !s) return RAD_E_INVAL;
        }
        if (i8) {
            q8 = decl_b(b, nm.f("%s.q8", name), RAD_I8,  { m, n });
            s8 = decl_b(b, nm.f("%s.s8", name), RAD_F32, { m, (n + g8 - 1) / g8 });
            if (!q8 || !s8) return RAD_E_INVAL;
        }
        return (fp8_pair || i8) ? RAD_OK : RAD_E_INVAL;
    }

    /* The code pair of a model that learned from a probe whether its linears are rotated int8
     * (`i8`, at `i8_group` columns a scale): declare_qs's E4M3 pair for a block-fp8 model, and for
     * an int8 one int8 codes. Those are written by the producer's own rotating quantiser `fed_op`
     * (had_quant_act_i8g, or gated_had_quant_i8g behind a gated product) where the kernels serve
     * it, and the linears reading them declare none; where they do not (the CPU kernels) the
     * producer keeps its E4M3 pair and each linear quantises its own input from the bf16 plane.
     * `x` is set by the caller. */
    int declare_pair(RadBuilder* b, Names& nm, const Geom& g, const char* name, int64_t n,
                     int64_t rows, bool i8, int64_t i8_group, const char* fed_op) {
        if (i8 && (i8_group <= 0 || n % i8_group)) {
            fprintf(stderr, "radiance: the int8 linears keep one scale a %lld of K, which does not "
                            "divide this rank's %lld-wide '%s'\n",
                    (long long)i8_group, (long long)n, name);
            return RAD_E_UNSUPPORTED;
        }
        if (i8 && rad_op_declared(b, fed_op)) q8_fed = q8_rot = true;
        return declare_qs(b, nm, g, name, n, rows, i8, !q8_fed, i8 ? i8_group : RAD_FP8_BLOCK);
    }
};

/* The fused attention prologue's row band, shared by the block issuing it, the rope table fill,
 * the QSA indexer reading that table, and the step deciding whether to fill it.
 *
 * The fusion is bit-exact against the unfused path (r4d_selftest asserts byte equality over 1.8M
 * elements at both gain widths; it reproduces `r4d_rmsnorm_bf16`'s sum-of-squares order). It stops
 * at the 64-row decode band, like the narrow GEMM and the gated fold: at a 2048-token chunk the
 * four separate kernels already fill the device and the fused form, which recomputes each rotary
 * partner's normed value, is slightly slower. */
inline int64_t qk_fuse_rows(const Geom& g) {
    return g.max_tok < 64 ? g.max_tok : 64;
}

/* The rope table's height. `rope_table` writes each position at its own row, so the table needs a
 * row for every position a step can name, and speculative steps (verify and draft rounds) name up
 * to `max_spec` positions past `max_ctx`; one more row is margin. */
inline int64_t rope_table_rows(const Geom& g) {
    return g.max_ctx + (g.max_spec > 0 ? (int64_t)g.max_spec : 0) + 1;
}

/* ================================================================== the activation quantiser */
struct QuantFP8 {
    ActFP8  a{};
    int64_t n  = 0;
    rad_op  op = 0;

    int  declare(RadBuilder* b, const Geom& g, const ActFP8& act, int64_t width);
    /* `r0`/`rows` narrow it to a row range: in a mixed step only the prefill rows need the
     * standalone quantiser, since the recurrent path quantises in its epilogue. */
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int QuantFP8::declare(RadBuilder* b, const Geom& g, const ActFP8& act, int64_t width) {
    a = act;
    n = width;
    /* Nothing to quantise for a bf16 model: no op, and step() issues nothing. */
    if (g.w_bf16) { op = 0; return a.x ? RAD_OK : RAD_E_INVAL; }
    if (!a.x || !a.cq() || !a.cs() || n <= 0) return RAD_E_INVAL;
    /* `dtype` is the input's (bf16); the op name gives the output format. An input fed at int8
     * codes is quantised by quant_act_i8g into its int8 pair, or by had_quant_act_i8g where they
     * are rotated (ActFP8::q8_rot). */
    op = rw(b, RAD_OP(b, a.q8_fed ? (a.q8_rot ? "had_quant_act_i8g" : "quant_act_i8g") : "quant_act_fp8",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                          RAD_INT("group", a.cg()), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
           { a.x }, { a.cq(), a.cs() });
    return RAD_OK;
}

inline void QuantFP8::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (!op) return;
    if (rows <= 0) rows = T - r0;
    RAD_ISSUE(c, op, brow_slice(a.x, r0, rows, n), brow_slice(a.cq(), r0, rows, n),
              brow_slice(a.cs(), r0, rows, n / a.cg()));
}

/* ================================== the quantiser folded into the op that produces the row
 *
 * A standalone quantiser is mostly launch cost (an 82 KiB row), and most have a producer in the
 * same block:
 *
 *     rmsnorm -> quant     the input of every attention, GDN and MLP linear
 *     silu_mul -> quant    down_proj's input
 *
 * libr4d's `rmsnorm_quant_fp8` and `gated_quant_fp8` quantise in the producer's pass. They still
 * write the bf16 output and take the codes from the bf16-rounded value, so the bytes match the
 * unfused pair exactly.
 *
 * The residual add folds in too. Every block ends with out-projection into the shared `h`,
 * all-reduce, `x += h`, and begins with a norm of `x`, so the add moves to the next block's norm,
 * which already reads both buffers; `fold` says a block has a predecessor to add. The kernel rounds
 * the sum to bf16 and stores it before normalising (unlike the int8 sibling, which keeps the more
 * accurate f32 sum), reproducing the unfused pair byte for byte. */
/* Whether the two-rank all-reduce at `T` rows uses the exact wire. Must match the plugin's
 * `all_reduce` row, which decides from the same three facts: lossy wire not accepted, a message
 * the rotation's group of 64 does not tile, or a message under the byte floor. Every rank
 * evaluates it on the same T.
 *
 * The 6-bit wire carries 2.6x fewer bytes and the fused kernel carries both payloads, so this
 * chooses the fused op's wire; `ar_fused_ok` says whether a fused form exists. The floor exists
 * because small messages (one sequence, T around 8) lose more to the rotation's arithmetic than
 * they save in bytes, while from two sequences up the compressed wire wins increasingly; 128 KiB
 * sits between the regimes.
 */
inline bool ar_wire_is_exact(const Geom& g, int64_t T) {
    if (g.world != 2) return false;
    const int64_t numel = T * g.n_embd;
    if (numel <= 0) return false;
    if (g.wire_exact) return true;
    if (numel % 64) return true;
    return numel * 2 < g.wire_min_bytes;   /* bf16 on the wire */
}

/* The compressed fused form's packing chunk, in elements. The kernel gives each block whole rows
 * (the sum of squares closes over one row), so a chunk must not straddle a row. Fixed here so
 * every rank decides identically at declare whether the op exists. */
static const int64_t kArWire6Chunk = 512;

/* Whether the all-reduce at `T` rows can be folded into the following norm: always on the exact
 * wire, on the compressed one when a chunk tiles the row. The preceding block asks the same
 * question before skipping its own collective. */
inline bool ar_fused_ok(const Geom& g, int64_t T) {
    if (g.world != 2 || T * g.n_embd <= 0) return false;
    return ar_wire_is_exact(g, T) || (g.n_embd % kArWire6Chunk) == 0;
}

/* The four-rank lossy wire's tile: tiered int8 in groups of 32 that may not straddle a shard, so a
 * message whose element count is not a multiple of 128 goes out exact. */
static const int64_t kArWideWireTile = 128;

/* The most rows a decode or verify step carries: 1 + max_spec per sequence, at most max_tok. The
 * four-rank fused write and gather are declared up to this, and ar_wide_write_exact takes no
 * larger step. */
inline int64_t ar_wide_write_rows(const Geom& g) {
    const int64_t r = g.max_seqs * (1 + (g.max_spec > 0 ? g.max_spec : 0));
    return r < g.max_tok ? r : g.max_tok;
}

/* Rows the exact fused write is declared for: every step at two ranks, decode and verify steps at
 * four (ar_wide_write_rows), none otherwise. */
inline int64_t ar_exact_write_rows(const Geom& g) {
    if (g.world == 2) return g.max_tok;
    if (g.world == 4) return ar_wide_write_rows(g);
    return 0;
}

/* Whether a four-rank step takes the fused write (ar_hc_write, exact one-shot only). Both must hold:
 *
 *   rows      T <= ar_wide_write_rows (decode and verify steps). Larger steps keep the separate
 *             `all_reduce`, whose size bands choose one-shot or two-shot. This is a cost bound,
 *             not a correctness one.
 *   exact     --tp-wire exact, or a message the four-rank lossy row sends exact anyway
 *             (not a multiple of kArWideWireTile, or under `min_bytes`).
 *
 * ar_wire_is_exact stays false off two ranks: the embedding's collective uses it to ask whether
 * layer 0's fused norm absorbs the reduction, and at four ranks none does. */
inline bool ar_wide_write_exact(const Geom& g, int64_t T) {
    if (g.world != 4 || T < 1 || T > ar_wide_write_rows(g)) return false;
    const int64_t numel = T * g.n_embd;
    if (numel <= 0) return false;
    if (g.wire_exact) return true;
    if (numel % kArWideWireTile) return true;
    return numel * 2 < g.wire_min_bytes;   /* bf16 on the wire */
}

/* Whether a gated residual's exact fused write takes the all-reduce at T rows: at two ranks when
 * the wire is exact, at four per ar_wide_write_exact. Asked by both the write (HyperConn::ar_write)
 * and the preceding block (ar_taken with kArOutExact). */
inline bool ar_exact_write_takes(const Geom& g, int64_t T) {
    return g.world == 2 ? ar_wire_is_exact(g, T) : ar_wide_write_exact(g, T);
}

/* Which wires the consumer of a block's all-reduce takes; a block is declared without knowing what
 * follows, so the architecture tells it:
 *
 *   kArOutNorm   the next block's norm, which takes either wire it has a fused form for
 *                (ar_fused_ok);
 *   kArOutExact  a gated residual's write with the exact fused form only, so a message that would
 *                go out compressed stays with the block (ar_exact_write_takes -- at four ranks
 *                also one of a step larger than a decode or verify step);
 *   kArOutBoth   a gated residual's write with a fused form on each wire, which takes every
 *                message (HyperConn::ar_take). */
enum ArOut : int {
    kArOutNorm  = 0,
    kArOutExact = 1,
    kArOutBoth  = 2,
};

/* Whether a block skips its own all-reduce because its consumer takes it. The consumer asks the
 * same question of the same T, so exactly one of them issues it. */
inline bool ar_taken(const Geom& g, int64_t T, bool out, int take) {
    if (!out) return false;
    if (take == kArOutBoth)  return true;
    if (take == kArOutExact) return ar_exact_write_takes(g, T);
    return ar_fused_ok(g, T);
}

/* A caller that takes a block's all-reduce a piece at a time (qwen4exp_fp8's pipelined prefill).
 * A block whose last stages are per row finishes its output `rows` rows at a time and calls `done`
 * behind every piece but the last, so the caller can start that piece's reduction while the block
 * works on the next; `handed` counts the calls. A block that cannot cut its output calls nothing
 * and the caller issues every piece itself. The last piece is always the caller's: it waits for the
 * first pieces, and a wait on the second lane is a wait on everything issued there. */
struct ArPieces {
    int64_t rows = 0;
    void  (*done)(void* u, int64_t r0, int64_t n) = nullptr;
    void*   u = nullptr;
    int64_t handed = 0;
    /* Optional, on the card: {0, rows, T} for a step of one prefill sequence in two pieces, so a
     * block whose recurrence runs over cu_seqlens can run it a piece at a time (each piece the
     * chunk a span of 1 makes, the state carried through the cache) and hand off the first
     * piece's reduction before the second piece's projection. Null for any other step. */
    const int32_t* cu = nullptr;
    /* The caller normed each piece's input itself (qwen35_fp8's pipelined prefill, on the second
     * lane behind the previous block's piece), so a block with its own norm takes the piece path
     * as one with an external input does and skips that norm. */
    bool in_ready = false;
    bool cuts(int64_t T) const { return done && rows > 0 && T > rows; }
    void hand(int64_t r0, int64_t n) { done(u, r0, n); ++handed; }
};

/* Whether a MoE block's gather is folded into the following reduction (ar_gather_hc_write).
 * `rows` and `rows6` are the fused op's declared bounds on the exact and rotated wires (0 if not
 * declared), at most one MoE pass, so a served step is a single pass. The step's wire picks the
 * bound: `rows` wherever ar_exact_write_takes (at four ranks `rows6` is always 0). The block and
 * the write ask this with the same T and bounds, so they agree. */
inline bool gather_taken(const Geom& g, int64_t T, int64_t rows, int64_t rows6) {
    const int64_t r = ar_exact_write_takes(g, T) ? rows : rows6;
    return r > 0 && T <= r;
}

/* Which preceding collective a norm may absorb.
 *
 * `kArAny`: a layer boundary, where the predecessor reduced partial sums and the compressed wire is
 * acceptable.
 *
 * `kArExact`: the embedding. Its collective sums disjoint rows (one rank has the token, the others
 * zeros), so a lossy wire would only degrade an exact value at the model's input, and would differ
 * from the drafter's exact embedding reduction, hurting acceptance. A prompt of a few dozen tokens
 * crosses --tp-wire-min-kb, so allowing it changes greedy text. Exact-only keeps the decode fold
 * and leaves prefill on the standalone collective. */
enum ArFold : int {
    kArNone  = 0,
    kArAny   = 1,
    kArExact = 2,
};

/* Where trunk layer `l` of `n` does its residual add, for a trunk of fp8 blocks whose feed-forward
 * always folds its mixer's delta into its own norm (NormQuantFP8), so a mixer never adds.
 *
 * The mixer folds its predecessor's delta unless it is layer 0 or follows a layer that kept its
 * add. A feed-forward keeps its add where something reads `x` between blocks: a drafter tap (in
 * `taps`), which must see the layer's real output, and the final norm, a plain `rmsnorm` with no
 * residual operand.
 *
 * The all-reduce moves with the add, at two ranks only (tp1 has no collective and there is no
 * fused kernel at four): a block's norm absorbs the preceding collective where it folds. Layer 0's
 * mixer does not fold but absorbs the vocab-sharded embedding's reduction (`embed_ar`), exact
 * only (kArExact). */
struct LayerFold {
    bool mix_fold   = false;
    int  mix_ar_in  = kArNone;
    bool mix_ar_out = false;   /* the mixer leaves its collective to the feed-forward's norm */
    bool mlp_add    = false;
    int  mlp_ar_in  = kArNone;
    bool mlp_ar_out = false;
};

inline LayerFold layer_fold(const Geom& g, int64_t l, int64_t n, const int* taps, int n_taps,
                            bool embed_ar) {
    auto tapped = [&](int64_t i) { return std::find(taps, taps + n_taps, (int)i) != taps + n_taps; };
    const bool tp2 = g.world == 2;
    LayerFold f;
    f.mix_fold   = l > 0 && !tapped(l - 1);
    f.mix_ar_in  = !tp2 ? kArNone : f.mix_fold ? kArAny : (l == 0 && embed_ar) ? kArExact : kArNone;
    f.mix_ar_out = tp2;
    f.mlp_add    = tapped(l) || l + 1 >= n;
    f.mlp_ar_in  = tp2 ? kArAny : kArNone;
    f.mlp_ar_out = tp2 && !f.mlp_add;
    return f;
}

struct NormQuantFP8 {
    ActFP8     a{};
    rad_weight w = 0;
    /* The norm's input: the residual stream, or with a fold the previous block's delta, which the
     * op first adds into the stream (`res`). */
    rad_buf    src = 0;
    rad_buf    res = 0;
    rad_op     op = 0;
    /* The same op with the preceding all-reduce folded in, declared when the caller passes
     * `fuse_ar` (this norm is the collective's sole consumer). Both forms are declared and step()
     * chooses, since the wire depends on T; the predecessor asks the same question and skips its
     * own all_reduce on the same answer. */
    rad_op     op_arn = 0;
    /* The fused form on the rotated 6-bit wire, taken from the second concurrent sequence up
     * (the T * n_embd * 2-byte message then crosses --tp-wire-min-kb). The wire is a declare-time
     * parameter, hence a separate op. */
    rad_op     op_arn6 = 0;
    /* The ops write had_quant_act_i8g's rotated int8 codes into the int8 pair (ActFP8::q8_rot). */
    bool       hi8 = false;
    Geom       gm{};
    /* The row width, for turning a row slice into element offsets. */
    int64_t    n = 0;

    int  declare(RadBuilder* b, const Geom& g, rad_weight gain, rad_buf x, const ActFP8& act,
                 int64_t n, rad_buf residual = 0, int64_t max_rows = 0, int fuse_ar = kArNone);
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
    /* The op this step runs: the fused form for the step's wire if declared, else the plain op. */
    rad_op pick(int64_t T) const {
        if (op_arn && ar_wire_is_exact(gm, T)) return op_arn;
        if (op_arn6 && ar_fused_ok(gm, T)) return op_arn6;
        return op;
    }
    /* Asked by the preceding block before issuing its own collective. */
    bool takes_ar(int64_t T) const { return pick(T) != op; }
    /* The plain op whatever the wire, for a caller that issued the preceding all-reduce itself
     * (qwen35_fp8's pipelined prefill, which runs it narrow beside compute). */
    void step_plain(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int NormQuantFP8::declare(RadBuilder* b, const Geom& g, rad_weight gain, rad_buf x,
                                 const ActFP8& act, int64_t n, rad_buf residual,
                                 int64_t max_rows, int fuse_ar) {
    a = act;
    w = gain;
    src = x;
    res = residual;
    gm = g;
    this->n = n;                    /* the parameter shadows the member; the member is the pitch */
    /* bf16 models of this family take block inputs from hc_read and declare no block norm; a
     * pre-norm bf16 model uses the bf16 blocks. */
    if (g.w_bf16) {
        fprintf(stderr, "radiance: a fused norm and quantiser was declared on a model whose "
                        "linears are bf16; its blocks take a prepared input instead\n");
        return RAD_E_UNSUPPORTED;
    }
    /* Rotated int8 codes (ActFP8::q8_rot) come from the `_had_quant_i8g` twins, into the int8
     * pair; the model checked they are declared before feeding. Otherwise E4M3, as always. */
    hi8 = a.q8_fed && a.q8_rot;
    const rad_buf cq = hi8 ? a.q8 : a.q, cs = hi8 ? a.s8 : a.s;
    if (!src || !a.x || !cq || !cs || !w || n <= 0) return RAD_E_INVAL;
    op = RAD_OP(b, hi8 ? "rmsnorm_had_quant_i8g" : "rmsnorm_quant_fp8",
                RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                           RAD_INT("n", n), RAD_F64("eps", g.eps),
                           RAD_INT("group", a.cg()), RAD_STR("dtype", g.dtype),
                           RAD_F64("wadd", g.wadd)),
                RAD_WEIGHTS(w));
    if (!op) return RAD_E_INVAL;
    /* Arrays rather than rw(): the sets are conditional, and a braced list in a ternary is not
     * lifetime-extended. `residual` is INOUT, so it is in both sets; otherwise the planner could
     * reorder a reader around it. */
    rad_buf rd[2] = { src, res };
    rad_buf wr[4] = { cq, cs, a.x, res };
    rad_op_reads (b, op, rd, res ? 2 : 1);
    rad_op_writes(b, op, wr, res ? 4 : 3);

    /* The fused form also writes the reduced message into `src`. With a fold, `src` is `a.x`,
     * already in the write set; without one it is the residual stream, so it is added (as `res`'s
     * slot) to keep liveness correct.
     *
     * `res` is optional: the embedding's partial (vocab-sharded) lookup also needs this reduction,
     * with `x` the stream and no residual. The kernel supports it (r4d_fused_quant_fp8.hip) and the
     * schema marks `residual` optional. */
    if (fuse_ar && g.world == 2) {
        op_arn = RAD_OP(b, hi8 ? "ar_rmsnorm_had_quant_i8g" : "ar_rmsnorm_quant_fp8",
                        RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                                   RAD_INT("n", n), RAD_F64("eps", g.eps),
                                   RAD_INT("group", a.cg()),
                                   RAD_INT("world_size", g.world),
                                   RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                        RAD_WEIGHTS(w));
        if (!op_arn) return RAD_E_INVAL;
        rad_buf wr_ar[4] = { cq, cs, a.x, res ? res : src };
        rad_op_reads (b, op_arn, rd, res ? 2 : 1);
        rad_op_writes(b, op_arn, wr_ar, 4);

        /* Only where the compressed wire is reachable and servable: a lossy payload accepted, and
         * the width tiled by the chunk (otherwise a fatal step-time shape refusal). */
        if (fuse_ar != kArExact && !g.wire_exact && (n % kArWire6Chunk) == 0) {
            op_arn6 = RAD_OP(b, hi8 ? "ar_rmsnorm_had_quant_i8g" : "ar_rmsnorm_quant_fp8",
                             RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                                        RAD_INT("n", n), RAD_F64("eps", g.eps),
                                        RAD_INT("group", a.cg()),
                                        RAD_INT("world_size", g.world),
                                        RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd),
                                        RAD_INT("wire", 6)),
                             RAD_WEIGHTS(w));
            if (!op_arn6) return RAD_E_INVAL;
            rad_op_reads (b, op_arn6, rd, res ? 2 : 1);
            rad_op_writes(b, op_arn6, wr_ar, 4);
        }
    }
    return RAD_OK;
}

/* step() below with the plain op, whatever the wire. */
inline void NormQuantFP8::step_plain(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    RAD_ISSUE_N(c, op, rows, brow_slice(src, r0, rows, n),
                res ? brow_slice(res, r0, rows, n) : RAD_NONE, RAD_W(w),
                brow_slice(a.q, r0, rows, n),
                brow_slice(a.s, r0, rows, n / RAD_FP8_BLOCK),
                brow_slice(a.x, r0, rows, n));
}

/* `r0`/`rows` select a row slice, defaulting to the whole step (`brow_slice` at offset 0 equals
 * `brows`). Slicing lets a block run in passes (the routed feed-forward). Each slice must land in
 * the same bucket-table band as the whole (core/runtime/ctx.h), or a different kernel with a
 * different summation order would run. */
inline void NormQuantFP8::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    RAD_ISSUE_N(c, pick(rows), rows, brow_slice(src, r0, rows, n),
                res ? brow_slice(res, r0, rows, n) : RAD_NONE, RAD_W(w),
                brow_slice(hi8 ? a.q8 : a.q, r0, rows, n),
                brow_slice(hi8 ? a.s8 : a.s, r0, rows, n / (hi8 ? a.g8 : (int64_t)RAD_FP8_BLOCK)),
                brow_slice(a.x, r0, rows, n));
}

/* Gate activation then quantise, over a [gate|up] input. `act` names the gate: "silu" for
 * down_proj's input, "sigmoid" for a gated attention o_proj's. */
struct GateQuantFP8 {
    ActFP8 a{};
    rad_op op = 0;
    /* The swiglu_clamped path (`limit` > 0): `op` writes the clamped bf16 product and `op_quant`
     * quantises it into the codes; with `clamped_fused`, `op` is `swiglu_clamped_quant_fp8` doing
     * both and `op_quant` is unset. */
    rad_op op_quant = 0;
    bool clipped = false;
    /* Set before declare: serve the clamped path as one `swiglu_clamped_quant_fp8` where a loaded
     * library declares it and it resolves (E4M3 codes only). Byte-identical to the pair. */
    bool fuse_clamped = false;
    bool clamped_fused = false;
    /* Whether the bf16 product is written. A consumer of the bf16 value must request it
     * (docs/OPS.md), but a block-fp8 down_proj reads only codes and scales; only one that reads
     * the bf16 plane (LinearFP8::reads_x) needs it. Skipping it saves [T, n_ff] per layer (18.2 GB
     * over a 16k prefill). Left out of the write list, not just passed absent, so the plan is
     * accurate. */
    bool   want_x = true;
    /* The output width; `gate_up` is 2n wide. */
    int64_t n = 0;
    /* `silu_mul` into the bf16 plane alone, for a model whose linears are bf16 (Geom::w_bf16). */
    bool   bf16 = false;
    /* The rotated form's chain, where no loaded library declares `gated_had_quant_fp8` or it does
     * not resolve: `silu_mul` into the bf16 plane (`op`), then `had_quant_act_fp8` over it into the
     * codes (`op_q`). */
    rad_op op_q = 0;
    /* act "relu2" (Nemotron-H's ungated experts): `gate_up` is one part, n wide, and `op` is
     * act_mul's relu over it twice, relu(x)^2 into the bf16 plane, which `op_q` quantises (rotated
     * for a rotated weight). No fused form yet: the plane is written either way. */
    bool   ungated = false;

    /* Set before declare: the product in bf16 alone, as for bf16 linears, for a consumer that
     * quantises its input itself (an int8 linear's op_q). */
    bool silu_only = false;

    /* `step`: with `limit`, clip after the SiLU (`swiglu_step`, Ling-3.0) rather than before it
     * (`swiglu_clamped`, GLM-5.3). It has no fused quantiser, so the pair runs. */
    int  declare(RadBuilder* b, const Geom& g, rad_buf gate_up, const ActFP8& act, int64_t n,
                 const char* act_name = "silu", int64_t max_rows = 0, bool bf16_out = true,
                 bool rot = false, double limit = 0, bool step = false);
    void step(RadCtx* c, rad_buf gate_up, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

/* `rot` selects a different op (`gated_had_quant_fp8`), whose rotated codes only a weight stored
 * with the matching rotation can read; used for the routed MoE down projection at four bits. The
 * bf16 output is unrotated either way. */
inline int GateQuantFP8::declare(RadBuilder* b, const Geom& g, rad_buf gate_up, const ActFP8& act,
                                 int64_t n, const char* act_name, int64_t max_rows,
                                 bool bf16_out, bool rot, double limit, bool step) {
    a = act;
    want_x = bf16_out;
    op_quant = 0;
    this->n = n;
    /* bf16 linears and `silu_only` read the product itself: `silu_mul` into the bf16 plane only,
     * and no rotation. */
    bf16 = g.w_bf16 || silu_only;
    clipped = limit > 0;
    clamped_fused = false;
    if (!std::isfinite(limit) || limit < 0) return RAD_E_INVAL;
    ungated = std::strcmp(act_name, "relu2") == 0;
    if (ungated) {
        if (!gate_up || !a.x || n <= 0 || clipped || (!bf16 && (!a.cq() || !a.cs())) ||
            (rot && a.q8_fed))
            return RAD_E_INVAL;
        const int64_t rows = max_rows > 0 ? max_rows : g.max_tok;
        want_x = true;
        op = rw(b, RAD_OP(b, "act_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("act", "relu"),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               { gate_up }, { a.x });
        if (!op) return RAD_E_INVAL;
        if (bf16) return RAD_OK;
        const char* qn = rot ? "had_quant_act_fp8" : a.q8_fed ? "quant_act_i8g" : "quant_act_fp8";
        op_q = rw(b, RAD_OP(b, qn,
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                 { a.x }, { a.cq(), a.cs() });
        return op_q ? RAD_OK : RAD_E_INVAL;
    }
    if (clipped) {
        /* No rotated int8 form: its quantiser would be quant_act_i8g, unrotated; and no clamp
         * with `silu_only`. */
        if (!gate_up || !a.x || n <= 0 || rot || silu_only || (a.q8_fed && a.q8_rot) ||
            std::strcmp(act_name, "silu"))
            return RAD_E_INVAL;
        const int64_t rows = max_rows > 0 ? max_rows : g.max_tok;
        if (fuse_clamped && !step && !bf16 && !a.q8_fed && a.cq() && a.cs() &&
            rad_op_declared(b, "swiglu_clamped_quant_fp8")) {
            op =
                RAD_OP(b, "swiglu_clamped_quant_fp8",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("limit", limit),
                                  RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS);
            if (op) {
                const rad_buf wr[3] = { a.cq(), a.cs(), a.x };
                rad_op_reads(b, op, &gate_up, 1);
                rad_op_writes(b, op, wr, want_x ? 3 : 2);
                clamped_fused = true;
                return RAD_OK;
            }
        }
        want_x = true;
        op = rw(b,
                RAD_OP(b, step ? "swiglu_step" : "swiglu_clamped",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("limit", limit),
                                  RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                { gate_up }, { a.x });
        if (!bf16) {
            if (!a.cq() || !a.cs()) return RAD_E_INVAL;
            op_quant =
                rw(b,
                   RAD_OP(b, a.q8_fed ? "quant_act_i8g" : "quant_act_fp8",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                     RAD_INT("group", a.cg()), RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                   { a.x }, { a.cq(), a.cs() });
            if (!op_quant) return RAD_E_INVAL;
        }
        return op?RAD_OK:RAD_E_INVAL;
    }
    if (bf16) {
        if (!gate_up || !a.x || n <= 0 || rot || std::strcmp(act_name, "silu") != 0) {
            fprintf(stderr, "radiance: a bf16 gate is silu over [gate|up] into a bf16 plane; "
                            "asked for '%s'%s\n", act_name, rot ? " rotated" : "");
            return RAD_E_INVAL;
        }
        want_x = true;
        op = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                              RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               { gate_up }, { a.x });
        return op ? RAD_OK : RAD_E_INVAL;
    }
    if (!gate_up || !a.x || !a.cq() || !a.cs() || n <= 0) return RAD_E_INVAL;
    /* Rotated codes feed a four-bit expert, which reads E4M3 only. */
    if (rot && a.q8_fed) return RAD_E_INVAL;
    /* Rotated int8 codes (ActFP8::q8_rot) feed a down projection stored at i8a8h:
     * gated_had_quant_i8g, which the model checked is declared before feeding. */
    const bool hi8 = a.q8_fed && a.q8_rot;
    const int64_t rows = max_rows > 0 ? max_rows : g.max_tok;
    const char* gq = hi8 ? "gated_had_quant_i8g" : rot ? "gated_had_quant_fp8" : "gated_quant_fp8";
    if (!rot || rad_op_declared(b, "gated_had_quant_fp8"))
        op = RAD_OP(b, gq,
                    RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                               RAD_INT("group", a.cg()), RAD_STR("dtype", g.dtype),
                               RAD_STR("act", act_name)),
                    RAD_NOWEIGHTS);
    /* The rotated op's chain, which writes the bf16 plane whether or not a consumer asked. */
    if (!op && rot && std::strcmp(act_name, "silu") == 0) {
        want_x = true;
        op = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               { gate_up }, { a.x });
        op_q = rw(b, RAD_OP(b, "had_quant_act_fp8",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                 { a.x }, { a.cq(), a.cs() });
        return op && op_q ? RAD_OK : RAD_E_INVAL;
    }
    if (!op) return RAD_E_INVAL;
    const rad_buf wr[3] = { a.cq(), a.cs(), a.x };
    rad_op_reads (b, op, &gate_up, 1);
    rad_op_writes(b, op, wr, want_x ? 3 : 2);
    return RAD_OK;
}

inline void GateQuantFP8::step(RadCtx* c, rad_buf gate_up, int64_t T, int64_t r0,
                               int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    if (ungated) {
        RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, n), brow_slice(gate_up, r0, rows, n),
                    brow_slice(a.x, r0, rows, n));
        if (op_q)
            RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, n),
                        brow_slice(a.cq(), r0, rows, n),
                        brow_slice(a.cs(), r0, rows, n / RAD_FP8_BLOCK));
        return;
    }
    if (clamped_fused) {
        RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n),
                    brow_slice(a.cq(), r0, rows, n),
                    brow_slice(a.cs(), r0, rows, n / a.cg()),
                    want_x ? brow_slice(a.x, r0, rows, n) : RAD_NONE);
        return;
    }
    if (clipped) {
        RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n),
                    brow_slice(a.x, r0, rows, n));
        if (op_quant)
            RAD_ISSUE_N(c, op_quant, rows, brow_slice(a.x, r0, rows, n),
                        brow_slice(a.cq(), r0, rows, n),
                        brow_slice(a.cs(), r0, rows, n / a.cg()));
        return;
    }
    if (bf16 || op_q) {
        RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n), brow_slice(a.x, r0, rows, n));
        if (op_q)
            RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, n),
                        brow_slice(a.cq(), r0, rows, n),
                        brow_slice(a.cs(), r0, rows, n / a.cg()));
        return;
    }
    RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n),
                brow_slice(a.cq(), r0, rows, n),
                brow_slice(a.cs(), r0, rows, n / a.cg()),
                want_x ? brow_slice(a.x, r0, rows, n) : RAD_NONE);
}

/* A linear's stacked row parts (RadWeightDecl::row_parts) in weight rows (`w`) and in the rows of
 * its scale plane (`s`, `div` times fewer). */
struct RowParts {
    int64_t w[RAD_MAX_ROW_PARTS] = {0}, s[RAD_MAX_ROW_PARTS] = {0};
    int     n = 0;

    /* Refuses more than RAD_MAX_ROW_PARTS parts, or a part that is not whole `unit`-row tiles
     * (`tile` names them), since a row shard cuts the weight at part boundaries. */
    int set(std::initializer_list<int64_t> parts, int64_t unit, int64_t div, const char* base,
            const char* tile) {
        n = 0;
        for (int64_t p : parts) {
            if (n >= RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
            if (p % unit) {
                fprintf(stderr, "radiance: '%s' stacks a %lld-row part, which is not a whole "
                                "number of %lld-row %s; it cannot be row-sharded\n",
                        base, (long long)p, (long long)unit, tile);
                return RAD_E_INVAL;
            }
            w[n] = p;
            s[n] = p / div;
            ++n;
        }
        return RAD_OK;
    }
};

/* ================================================================== one block-scaled linear */
struct LinearFP8 {
    rad_weight w = 0, ws = 0;
    rad_op     op = 0;
    int64_t    N = 0, K = 0;
    /* Weight rows per scale-plane row: RAD_FP8_BLOCK for block fp8, 1 otherwise. Row spans on `ws`
     * are converted with it (declare, span_parts). */
    int64_t    ws_row_div = 1;
    /* Plain bf16 weight read by `gemm_nt` over the bf16 activation (Geom::w_bf16): no scale plane,
     * no quantiser. */
    bool       bf16 = false;
    /* Block-fp8 weight whose scale plane is f32 rather than bf16 (GLM's checkpoints), read by
     * `gemm_nt_q` at dtype fp8a8f32. Read from the weight's encoding. */
    bool       f32_scale = false;
    /* Int8 weight read by `gemm_nt_q` at dtype i8a8: i8 codes with a bf16 scale per row per 128 of
     * K (libquant `codes=i8 group=128 scale=bf16`), against an i8 activation with an f32 scale per
     * (row, 128). A quarter of E4M3's error at the same bytes, and iu8 WMMA runs at the fp8 rate.
     * `op_q` (quant_act_i8g) writes the activation codes from the bf16 plane when the producer
     * writes E4M3. Read from the weight's encoding. */
    bool       i8 = false;
    rad_op     op_q = 0;
    /* An int8 weight stored rotated (i8a8h, or a rotated four- or five-bit one): its activation
     * codes are in the 128-wide Hadamard domain, whoever wrote them (had_quant_act_i8g or a
     * rotating producer). */
    bool       i8_rot = false;
    /* With i8_rot: an IQR weight (libr4d's r4d_iqr.h), whose codes decode where they are read: a
     * u6 code under a zero, a table, a grid or a trellis. Read from the weight's encoding. */
    bool       iqr = false;
    /* OCP-MXFP4 weight read by `gemm_nt_q` at dtype mxfp4a8 against the usual E4M3 activation pair:
     * e2m1 codes with one E8M0 exponent per 32 of K (libquant `codes=fp4_e2m1 group=32
     * scale=e8m0`). libr4d folds each block's exponent into the e4m3 operand against a per-row
     * reference, so the scale plane is passed twice, as `b_scale` and as `b_ref` (`wref`, a second
     * view whose layout hook reduces it). Read from the weight's encoding. */
    bool       mx = false;
    rad_weight wref = 0;
    /* Or M2XFP4 (libquant `m2xfp4`): the MXFP4 views, the meta plane riding the weight operand,
     * read by `gemm_nt_q` at dtype m2xfp4. */
    bool       m2x = false;
    /* Rotated (ParoQuant) variant of `mx`: the same GEMM over an activation that
     * `givens_dense_quant` makes from the bf16 plane (channel scale, dense per-group rotation and
     * E4M3 quantiser in one pass), once per independently rotated part of a fused projection
     * (gate|up, qkv|z), into this linear's own code and scale planes side by side. The tables are
     * views of the weight's planes. The GEMM takes the parts' row bounds as `pb1`/`pb2` and meets
     * copy p with part p's rows. */
    bool       paro = false;
    /* The 5-bit sibling: the same rotation to f16 (`rot_q` is the f16 plane, no `rot_s`), then
     * gemm_nt_q at w5a16 over u5 codes with an f16 scale and a zero point. */
    bool       paro5 = false;
    /* A 4-bit weight with a zero point and no rotation (linear_is_w4): the int4 W4A16 releases.
     * At or below `w4_split` rows the w4a16_asym GEMM `op` reads the input's bf16 plane and
     * narrows it on load; above it `op_q` quantises it to int8 with a scale per (row, 128)
     * (`a8q`, `a8s`) for the w4a8i8_asym GEMM `op2`. */
    bool       w4 = false;
    int64_t    w4_split = 0;
    rad_buf    a8q = 0, a8s = 0;
    /* Prefill band for z-lab's int4 and the int5: above `rot_split` rows the f16 skinny kernel is
     * bound by the slow f16 matrix instruction, so a second chain runs: the same rotation to int8
     * with a scale per (row, 128) into `rot_q2`/`rot_s2`, then the int8 tiled GEMM (`w4a8i8_asym` /
     * `w5a8i8`) over the same weight views. At or below the split the f16 chain runs. */
    int64_t    rot_split = 0;
    rad_buf    rot_q2 = 0, rot_s2 = 0;
    rad_op     op_rot2 = 0, op2 = 0;
    rad_weight rot_pairs = 0, rot_angle = 0, rot_scale = 0, rot_dense = 0;
    rad_buf    rot_q = 0, rot_s = 0;
    rad_op     op_rot = 0;
    int64_t    rot_parts = 1, rot_stages = 0;
    /* probe() found an int8 weight; lets a model learn its trunk's format from one linear before
     * declaring activation buffers (ActFP8::declare_qs's `i8`). */
    bool       probed_i8 = false;
    /* With probed_i8: the weight's scale group along K (linear_w8h_group), which the model's int8
     * activation pairs must share (ActFP8::g8). */
    int64_t    probed_g8 = RAD_FP8_BLOCK;
    /* Whether this linear's block-fp8 GEMM takes the short-prompt split rule
     * (`gemm_nt_q_short`, narrow_ksplit's `short_unsplit`): at N <= 512, K <= 5120 and 41..64 rows
     * the split's per-slice floors dominate, so the row runs unsplit. Opt-in, so every other
     * caller keeps the split and its bits. Set before declare(). */
    bool       short_split = false;
    /* Whether this linear reads the input's bf16 plane, known from probe() before the producer is
     * declared. A producer writes the bf16 plane only when asked, so the block must ask for it when
     * this is set; otherwise the linear reads unwritten arena bytes. */
    bool       reads_x = false;
    bool       mapped  = false;   /* probe() made the name map; declare() does not make it again */
    /* The stored weight this linear's views take when it is not `<base>.weight`: a second linear
     * over one container entry, e.g. a whole projection beside its tensor-parallel slice. Set
     * before declare(); no name map is made, and only block fp8 is served this way. */
    const char* stored = nullptr;
    /* The weight's access class (rad_builder.h): read every pass by default; a linear run once a
     * request (an encoder's) says so, which lets placement keep it off the card. Set before
     * declare(). */
    int        access  = RAD_ACCESS_PER_TOKEN;

    /* Declares the name map and reads the weight's format (reads_x, probed_i8). `base` and `src`
     * are the same as for declare(). */
    int  probe(RadBuilder* b, Names& nm, const Geom& g, const char* base,
               std::initializer_list<const char*> src);

    /* `base` is the declared name without suffix ("blk.7.ffn_gate_up"); the logical weight is
     * `<base>.weight` with views `.weight` (codes) and `.scale`. `src` is one checkpoint name, or
     * several concatenated along dim 0; block-fp8 scale planes concatenate with them, so each part
     * must be a whole number of 128-row blocks ([17408,5120] x 2 -> [34816,5120] with scales
     * [136,40] x 2 -> [272,40]).
     *
     * `span_lo`/`span_hi` (when hi is set): this rank's uneven slice of the split dimension in
     * model units (rows of each row part for RAD_SHARD_ROW, columns for RAD_SHARD_COL), whole scale
     * blocks; n_out / n_in is already its width (rad_weight_shard_span).
     *
     * `rot_part_rows`: rows of each independently rotated part of a rotated weight. Equal to
     * `row_parts` when the cuts coincide (gate|up), not otherwise (the delta net's qkv|z has four
     * shard parts and two rotated ones). */
    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 std::initializer_list<const char*> src,
                 const ActFP8& a, rad_buf y, int64_t max_rows = 0,
                 std::initializer_list<int64_t> row_parts = {},
                 int64_t span_lo = 0, int64_t span_hi = 0,
                 std::initializer_list<int64_t> rot_part_rows = {});
    /* Declares a rotated weight's rotation: parts and stages (`rot_parts`, `rot_stages`) and the
     * channel scale as a view of the weight's plane. `cut` is each rotated part's rows; the GEMM's
     * part bounds are returned in pb1 and pb2. */
    int  declare_rotation(RadBuilder* b, Names& nm, const char* base, const char* wn,
                          const RadEncoding& enc, int64_t N, int64_t K, RadWeightGroup grp,
                          const std::vector<int64_t>& cut, int64_t* pb1, int64_t* pb2);
    /* The sequential rotation network (cheap at decode's few rows): pairs and angles as views, and
     * givens_rotate over them to f16 for M in [lo, hi]. */
    int  decl_seq_tables(RadBuilder* b, Names& nm, const char* base, const char* wn, RadWeightGroup grp);
    rad_op decl_rot_op(RadBuilder* b, const Geom& g, const ActFP8& a, int64_t lo, int64_t hi,
                       std::initializer_list<rad_buf> outs) const;
    /* The dense rotation (rot_dense, decl_dense): givens_dense_quant to `out` ("fp8" or "i8") codes
     * and a scale, for M in [lo, hi]. */
    rad_op decl_dense_op(RadBuilder* b, const Geom& g, const ActFP8& a, const char* out, int64_t lo,
                         int64_t hi, std::initializer_list<rad_buf> outs) const;
    /* gemm_nt_q over this weight, M in [lo, hi]. `pb1`/`pb2` are passed only for rotated parts,
     * so a one-part op keeps the plain geometry (the tune cache key). */
    rad_op decl_gemm_q(RadBuilder* b, int64_t lo, int64_t hi, int64_t group, const char* dtype,
                       int64_t pb1, int64_t pb2, std::initializer_list<rad_weight> wts,
                       std::initializer_list<rad_buf> reads, rad_buf y) const;
    rad_weight decl_dense(RadBuilder* b, Names& nm, const char* base, const char* wn, const RadEncoding& enc,
                          int64_t K,
                          RadWeightGroup grp);
    /* Rows [r0, r0 + rows) of the input into the same rows of `y`, or from row `y0` of `y` when
     * that is set: a rank's run of a step's rows written to a buffer sized for the run. */
    void step(RadCtx* c, const ActFP8& a, rad_buf y, int64_t T, int64_t r0 = 0,
              int64_t rows = 0, int64_t y0 = -1) const;
    /* A span per stacked row part (rad_weight_shard_span_parts), in weight rows; the scale plane
     * gets the same spans in its rows. Call after declare(). With a 1/128 plane every span and
     * total must be whole scale blocks. */
    int  span_parts(RadBuilder* b, const int64_t* lo, const int64_t* hi, const int64_t* total,
                    int n) const {
        RAD_ARCH_TRY(rad_weight_shard_span_parts(b, w, lo, hi, total, n));
        if (!ws) return RAD_OK;
        if (ws_row_div == 1) return rad_weight_shard_span_parts(b, ws, lo, hi, total, n);
        int64_t slo[RAD_MAX_ROW_PARTS], shi[RAD_MAX_ROW_PARTS], stot[RAD_MAX_ROW_PARTS];
        for (int g = 0; g < n && g < RAD_MAX_ROW_PARTS; ++g) {
            if (lo[g] % ws_row_div || hi[g] % ws_row_div || total[g] % ws_row_div) {
                fprintf(stderr, "radiance: a part span [%lld, %lld) of %lld rows is not whole "
                                "%lld-row scale blocks\n", (long long)lo[g], (long long)hi[g],
                        (long long)total[g], (long long)ws_row_div);
                return RAD_E_INVAL;
            }
            slo[g] = lo[g] / ws_row_div;
            shi[g] = hi[g] / ws_row_div;
            stot[g] = total[g] / ws_row_div;
        }
        return rad_weight_shard_span_parts(b, ws, slo, shi, stot, n);
    }
};

/* A linear's name map, declared before its weight because the weight's encoding is looked up
 * through it (rad_weight_encoding). One source is copied; several are concatenated along dim 0
 * (with block-fp8 scale planes, so each part must be whole 128-row blocks). */
inline int map_linear(RadBuilder* b, const char* wn, std::initializer_list<const char*> src) {
    if (src.size() == 1) return map_copy(b, wn, *src.begin());
    RadNameMap mw = map_base(wn, RAD_MAP_CONCAT);
    int i = 0;
    for (const char* q : src) {
        if (i >= 8) return RAD_E_INVAL;
        mw.src[i++] = q;
    }
    mw.n_src = i;
    mw.concat_dim = 0;
    return rad_decl_name_map(b, &mw);
}

/* The int8 weight LinearFP8 serves at i8a8: i8 codes, a bf16 scale per [1 x 128], no transform or
 * other plane (libr4d's i8 block layout hook). */
inline bool linear_is_w8(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 2 || e.transform[0]) return false;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return e.plane[0].dtype == RAD_I8 && e.plane[0].block[1] == 1 && sc && sc->dtype == RAD_BF16 &&
           sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK;
}

/* The same planes rotated by the unnormalised 128-wide Hadamard along K, the 1/128 in the scale
 * (libquant's `transform=fwht128`), served at i8a8h behind had_quant_act_i8g. */
/* How a weight served at i8a8h stores its codes on the same grid (a bf16 scale per [1 x 128],
 * /fwht128): int8, or a narrower form kept at its codes' width in VRAM and decoded by the GEMM that
 * reads it (libr4d's gemm_iqr rows) -- signed 2..7-bit codes; unsigned codes less a zero point per
 * [1 x 128]; unsigned codes indexing an int8 codebook; the codes a bitshift trellis shifts in down
 * each column ("trellis", whose scale may also be an E4M3 one per [1 x 128] under an f32 or bf16
 * second level, the bf16 of their product). Each decodes to the int8 grid's values exactly. */
enum class W8hForm { None, I8, Signed, Zero, Table, Grid, Trellis };
/* A computed trellis (libquant's tcq_code=mad) carries its code's constants as one more plane,
 * "trellis.mad", which the GEMM reads instead of the table. */
inline int linear_w8h_mad(const RadEncoding& e) { return rad_enc_plane(&e, "trellis.mad") ? 1 : 0; }
/* A trellis under one f32 scale a row times one f32 a 128-K block for all rows (libquant's
 * tcq_colf): "scale" [1 x whole] and "scale.col" [whole x 128], one view. */
inline bool linear_w8h_rowf(const RadEncoding& e) {
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    const RadEncPlane* cf = rad_enc_plane(&e, "scale.col");
    return rad_enc_is(&e, "trellis") && sc && cf && sc->kind == RAD_PLANE_TILED &&
           sc->dtype == RAD_F32 && sc->block[0] == 1 && sc->block[1] == 0 &&
           cf->kind == RAD_PLANE_TILED && cf->dtype == RAD_F32 && cf->block[0] == 0 &&
           cf->block[1] == RAD_FP8_BLOCK && !rad_enc_plane(&e, "scale.1");
}
/* A trellis weight's scale in two levels: E4M3 per [1 x 128] under "scale.1" for the tensor or
 * the row. */
inline bool linear_w8h_scale2(const RadEncoding& e) {
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    const RadEncPlane* s2 = rad_enc_plane(&e, "scale.1");
    return rad_enc_is(&e, "trellis") && e.n_planes - linear_w8h_mad(e) == 4 && sc && s2 &&
           sc->dtype == RAD_F8E4M3 &&
           s2->kind == RAD_PLANE_TILED && (s2->dtype == RAD_F32 || s2->dtype == RAD_BF16) &&
           (s2->block[0] == 0 || s2->block[0] == 1) && s2->block[1] == 0;
}
/* IQ8R: int8 codes rotated 128 wide with a u8 scale per [1 x 256] under one bf16 scale a row
 * (libquant `codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128`), served as the
 * I8 form at twice the scale group. */
inline bool linear_is_iq8r(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 3 ||
        std::strncmp(e.transform, "fwht128", RAD_ENC_STR) != 0)
        return false;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    const RadEncPlane* s2 = rad_enc_plane(&e, "scale.1");
    return e.plane[0].dtype == RAD_I8 && e.plane[0].block[1] == 1 && sc && sc->dtype == RAD_U8 &&
           sc->kind == RAD_PLANE_TILED && sc->block[0] == 1 && sc->block[1] == RAD_I8_GROUP_WIDE &&
           s2 && s2->dtype == RAD_BF16 && s2->kind == RAD_PLANE_TILED && s2->block[0] == 1 &&
           s2->block[1] == 0;
}
/* The scale group along K of a weight served at i8a8 / i8a8h. */
inline int64_t linear_w8h_group(const RadEncoding& e) {
    return linear_is_iq8r(e) ? (int64_t)RAD_I8_GROUP_WIDE : (int64_t)RAD_FP8_BLOCK;
}
inline W8hForm linear_w8h_form(const RadEncoding& e) {
    if (linear_is_iq8r(e)) return W8hForm::I8;
    const bool trellis = rad_enc_is(&e, "trellis");
    if ((!rad_enc_is(&e, "affine") && !trellis) ||
        std::strncmp(e.transform, "fwht128", RAD_ENC_STR) != 0)
        return W8hForm::None;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    const bool two = linear_w8h_scale2(e), rowf = linear_w8h_rowf(e);
    if (!sc || (sc->dtype != RAD_BF16 && !two && !rowf) || sc->block[0] != 1 ||
        sc->block[1] != (rowf ? 0 : RAD_FP8_BLOCK))
        return W8hForm::None;
    const uint32_t cd = e.plane[0].dtype;
    /* A vector grid: one code a [1 x V] run of columns, an i8 grid {E x V}. */
    if (e.n_planes == 3 && e.plane[0].block[1] > 1) {
        const RadEncPlane* gr = rad_enc_plane(&e, "grid");
        return gr && gr->kind == RAD_PLANE_TABLE && gr->dtype == RAD_I8 &&
                       gr->extent[1] == e.plane[0].block[1] && RAD_FP8_BLOCK % gr->extent[1] == 0
                   ? W8hForm::Grid : W8hForm::None;
    }
    const bool sgn = cd == RAD_I2 || cd == RAD_I3 || cd == RAD_I4 || cd == RAD_I5 || cd == RAD_I6 ||
                     cd == RAD_I7;
    const bool uns = cd == RAD_U1 || cd == RAD_U2 || cd == RAD_U3 || cd == RAD_U4 || cd == RAD_U5 ||
                     cd == RAD_U6 || cd == RAD_U7 || cd == RAD_U8;
    if (e.plane[0].block[1] != 1) return W8hForm::None;
    if (e.n_planes == 2) return cd == RAD_I8 ? W8hForm::I8 : sgn ? W8hForm::Signed : W8hForm::None;
    if (e.n_planes - (trellis ? linear_w8h_mad(e) + (rowf ? 1 : 0) : 0) != (two ? 4 : 3) || !uns)
        return W8hForm::None;
    /* A trellis: the codes a bitshift trellis shifts in down each column, an i8 value a state. */
    if (trellis) {
        const RadEncPlane* tp = rad_enc_plane(&e, "trellis");
        if (!tp) return W8hForm::None;
        const int64_t S = tp->extent[0];
        return tp->kind == RAD_PLANE_TABLE && tp->dtype == RAD_I8 && tp->extent[1] == 1 &&
                       tp->block[0] > 0 && S > ((int64_t)1 << rad_dtype_bits(cd)) && S <= 65536 &&
                       (S & (S - 1)) == 0
                   ? W8hForm::Trellis : W8hForm::None;
    }
    const RadEncPlane* zp = rad_enc_plane(&e, "zero");
    if (zp && cd != RAD_U1 && cd != RAD_U8 && (zp->dtype == cd || zp->dtype == RAD_U8) &&
        zp->block[0] == 1 && zp->block[1] == RAD_FP8_BLOCK)
        return W8hForm::Zero;
    const RadEncPlane* tb = rad_enc_plane(&e, "table");
    if (tb && tb->kind == RAD_PLANE_TABLE && tb->dtype == RAD_I8 && tb->extent[0] == 1 &&
        tb->extent[1] >= 2 && tb->extent[1] <= ((int64_t)1 << rad_dtype_bits(cd)))
        return W8hForm::Table;
    return W8hForm::None;
}
inline bool linear_is_w8h(const RadEncoding& e) { return linear_w8h_form(e) != W8hForm::None; }

/* A rotated asymmetric weight at four or five bits: u4 or u5 codes, an f16 scale and a u8 (or
 * code-width) zero point per 128 of K under the 128-wide Hadamard (libquant `codes=u5 zero=u8
 * group=128 scale=f16 transform=fwht128`). Read by gemm_nt_q at w4a8i8_asym / w5a8i8 over the
 * rotated int8 codes an i8a8h trunk's producers write, so it mixes with i8a8h linears in one trunk.
 * The scale carries the transform's 1/128, as an i8a8h weight's does. */
inline bool linear_is_wnh(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 3 ||
        std::strncmp(e.transform, "fwht128", RAD_ENC_STR) != 0)
        return false;
    const RadEncPlane *sc = rad_enc_plane(&e, "scale"), *zp = rad_enc_plane(&e, "zero");
    const uint32_t cd = e.plane[0].dtype;
    return (cd == RAD_U4 || cd == RAD_U5) && e.plane[0].block[1] == 1 && sc && zp &&
           sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK &&
           (zp->dtype == RAD_U8 || zp->dtype == cd) && zp->block[0] == 1 &&
           zp->block[1] == RAD_FP8_BLOCK;
}

/* The OCP-MXFP4 weight LinearFP8 serves at mxfp4a8: e2m1 codes and an E8M0 exponent per [1 x 32],
 * no transform (libr4d's MXFP4 layout hook). */
enum { RAD_MX_GROUP = 32 };
inline bool linear_is_mxfp4(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 2 || e.transform[0]) return false;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return e.plane[0].dtype == RAD_FP4E2M1 && e.plane[0].block[1] == 1 && sc &&
           sc->dtype == RAD_E8M0 && sc->block[0] == 1 && sc->block[1] == RAD_MX_GROUP;
}

/* M2XFP4 (arXiv 2601.19213, weight-side Sg-EM): the MXFP4 weight plus one metadata byte per group
 * of 32, four 2-bit subgroup extra mantissas that refine the scale to (1 + k/4) . 2^E. The
 * exponent byte carries the adaptive bias; the GEMM applies nothing beyond the planes. */
inline bool linear_is_m2xfp4(const RadEncoding& e) {
    if (!rad_enc_is(&e, "m2xfp4") || e.n_planes != 3 || e.transform[0]) return false;
    const RadEncPlane *sc = rad_enc_plane(&e, "scale"), *mt = rad_enc_plane(&e, "meta");
    return e.plane[0].dtype == RAD_FP4E2M1 && e.plane[0].block[0] == 1 &&
           e.plane[0].block[1] == 1 && sc && sc->dtype == RAD_E8M0 && sc->block[0] == 1 &&
           sc->block[1] == RAD_MX_GROUP && mt && mt->dtype == RAD_U8 && mt->block[0] == 1 &&
           mt->block[1] == RAD_MX_GROUP;
}

/* A rotated weight (ParoQuant): an affine encoding in the basis of a "givens" transform, with the
 * rotation tables as planes (int32 column pairs, f32 angles, f32 channel scales; one set per part
 * of a fused projection). The GEMM reads the codes as stored; the activation is rotated first. */
inline bool linear_is_givens(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || std::strncmp(e.transform, "givens", RAD_ENC_STR) != 0) return false;
    if (!rad_enc_valid(&e)) return false;
    const RadEncPlane *pr = rad_enc_plane(&e, "t.pairs"), *an = rad_enc_plane(&e, "t.angle"),
                      *rs = rad_enc_plane(&e, "t.scale");
    return pr && an && rs && pr->dtype == RAD_I32 && an->dtype == RAD_F32 && rs->dtype == RAD_F32;
}

/* A rotated weight with OCP-MXFP4 codes and exponents, read like the plain MXFP4 weight. */
inline bool linear_is_paro_mxfp4(const RadEncoding& e) {
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return linear_is_givens(e) && e.plane[0].dtype == RAD_FP4E2M1 && e.plane[0].block[1] == 1 && sc &&
           sc->dtype == RAD_E8M0 && sc->block[0] == 1 && sc->block[1] == RAD_MX_GROUP;
}

/* A rotated weight with M2XFP4 planes (ParoQuant's `m2xfp4` grid): the scheme's three planes in
 * the basis of a "givens" transform, whose tables are views of the weight's planes. Read like the
 * plain M2XFP4 weight -- the meta plane rides the weight operand's stored blob -- over an
 * activation the rotation makes, as linear_is_paro_mxfp4's is. */
inline bool linear_is_paro_m2xfp4(const RadEncoding& e) {
    if (!rad_enc_is(&e, "m2xfp4") || std::strncmp(e.transform, "givens", RAD_ENC_STR) != 0 ||
        !rad_enc_valid(&e) || e.n_planes < 6)
        return false;
    const RadEncPlane *sc = rad_enc_plane(&e, "scale"), *mt = rad_enc_plane(&e, "meta"),
                      *pr = rad_enc_plane(&e, "t.pairs"), *an = rad_enc_plane(&e, "t.angle"),
                      *rs = rad_enc_plane(&e, "t.scale");
    return pr && an && rs && pr->dtype == RAD_I32 && an->dtype == RAD_F32 && rs->dtype == RAD_F32 &&
           e.plane[0].dtype == RAD_FP4E2M1 && e.plane[0].block[0] == 1 &&
           e.plane[0].block[1] == 1 && sc && sc->dtype == RAD_E8M0 && sc->block[0] == 1 &&
           sc->block[1] == RAD_MX_GROUP && mt && mt->dtype == RAD_U8 && mt->block[0] == 1 &&
           mt->block[1] == RAD_MX_GROUP;
}

/* The weight formats rad-convert reads and the engine does not serve: block fp8 (E4M3
 * codes under a scale per 128 x 128), OCP MXFP4, M2XFP4 and the ParoQuant-rotated forms of the
 * last two. The format's name, or nullptr for one that is served. */
inline const char* linear_retired(const RadEncoding& e) {
    if (linear_is_paro_m2xfp4(e)) return "ParoQuant M2XFP4";
    if (linear_is_paro_mxfp4(e))  return "ParoQuant MXFP4";
    if (linear_is_givens(e))      return "ParoQuant";
    if (linear_is_m2xfp4(e))      return "M2XFP4";
    if (linear_is_mxfp4(e))       return "MXFP4";
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    if (rad_enc_is(&e, "affine") && e.n_planes == 2 && !e.transform[0] &&
        e.plane[0].dtype == RAD_F8E4M3 && sc && (sc->dtype == RAD_BF16 || sc->dtype == RAD_F32) &&
        sc->block[0] == RAD_FP8_BLOCK && sc->block[1] == RAD_FP8_BLOCK)
        return "block fp8";
    return nullptr;
}

/* The refusal of a weight stored in one of them, by name, with what serves the model instead. */
inline int linear_refuse_retired(const char* weight, const char* format) {
    fprintf(stderr, "radiance: '%s' is stored as %s, which this build does not serve. Convert "
                    "the checkpoint again with an int8 recipe (the IQ8R ones in data/recipes); if "
                    "the weight belongs to the container's drafter, `num-speculative-tokens: 0` "
                    "serves the model without it\n", weight, format);
    return RAD_E_UNSUPPORTED;
}

/* A 4-bit weight with a zero point and no transform: u4 codes, an f16 scale and a u4 or u8 zero
 * point per 128 of K (libquant `codes=u4 zero=u4 group=128 scale=f16`, or rad-convert's import rule
 * over an int4 release). At decode w4a16_asym narrows the BF16 input to F16 on load; larger
 * batches use w4a8i8_asym over an int8 activation, the rotated int4's GEMMs with no rotation. */
inline bool linear_is_w4(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 3 || e.transform[0]) return false;
    const RadEncPlane *sc = rad_enc_plane(&e, "scale"), *zp = rad_enc_plane(&e, "zero");
    return e.plane[0].dtype == RAD_U4 && e.plane[0].block[1] == 1 && sc && zp &&
           sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK &&
           (zp->dtype == RAD_U4 || zp->dtype == RAD_U8) && zp->block[0] == 1 &&
           zp->block[1] == RAD_FP8_BLOCK;
}

/* A rotated weight with ParoQuant int5 (u5) or z-lab int4 (u4) asymmetric codes: an f16 scale and
 * u8 zero point per 128 of K. Read by gemm_nt_q at w5a16 / w4a16_asym over an f16 activation, or
 * w5a8i8 / w4a8i8_asym over an int8 one. */
inline bool linear_is_paro_w5(const RadEncoding& e) {
    const RadEncPlane *sc = rad_enc_plane(&e, "scale"), *zp = rad_enc_plane(&e, "zero");
    return linear_is_givens(e) && (e.plane[0].dtype == RAD_U5 || e.plane[0].dtype == RAD_U4) && sc &&
           zp && sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK &&
           zp->dtype == RAD_U8 && zp->block[0] == 1 && zp->block[1] == RAD_FP8_BLOCK;
}

inline int LinearFP8::probe(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                            std::initializer_list<const char*> src) {
    if (src.size() == 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    RAD_ARCH_TRY(map_linear(b, wn, src));
    mapped = true;
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    const bool plain = known && rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                       enc.plane[0].dtype == RAD_BF16;
    probed_i8 = known && (linear_is_w8(enc) || linear_is_w8h(enc) || linear_is_wnh(enc));
    probed_g8 = known ? linear_w8h_group(enc) : (int64_t)RAD_FP8_BLOCK;
    /* A rotated weight rotates and quantises the bf16 plane itself; a 4-bit one casts or
     * quantises it. */
    const bool rotated = known && (linear_is_paro_mxfp4(enc) || linear_is_paro_w5(enc) ||
                                   linear_is_paro_m2xfp4(enc));
    reads_x = g.w_bf16 || plain || probed_i8 || rotated || (known && linear_is_w4(enc));
    return RAD_OK;
}

inline rad_op LinearFP8::decl_gemm_q(RadBuilder* b, int64_t lo, int64_t hi, int64_t group,
                                     const char* dtype, int64_t pb1, int64_t pb2,
                                     std::initializer_list<rad_weight> wts,
                                     std::initializer_list<rad_buf> reads, rad_buf y) const {
    const RadParam ps[] = { RAD_RANGE("M", lo, hi), RAD_INT("N", N), RAD_INT("K", K),
                            RAD_INT("group", group), RAD_STR("dtype", dtype),
                            RAD_INT("pb1", pb1), RAD_INT("pb2", pb2) };
    const int n_ps = (int)(sizeof ps / sizeof ps[0]) - (rot_parts > 1 ? 0 : 2);
    return rw(b, rad_decl_op(b, "gemm_nt_q", ps, n_ps, wts.begin(), (int)wts.size()), reads, { y });
}

inline int LinearFP8::declare_rotation(RadBuilder* b, Names& nm, const char* base, const char* wn,
                                       const RadEncoding& enc, int64_t N, int64_t K,
                                       RadWeightGroup grp, const std::vector<int64_t>& cut_in,
                                       int64_t* pb1, int64_t* pb2) {
    /* The container records the number of rotated parts but not their row bounds, so the block
     * supplies them (`rot_part_rows`, or its row parts when those are the cut). */
    const int64_t P = rad_enc_givens_parts(&enc);
    const RadEncPlane* tp = rad_enc_plane(&enc, "t.pairs");
    if (P < 1 || !tp || tp->extent[0] % P || tp->extent[1] != K) {
        fprintf(stderr, "radiance: '%s' has rotation tables that do not fit %lld columns\n",
                wn, (long long)K);
        return RAD_E_SHAPE;
    }
    const int64_t S = tp->extent[0] / P;
    std::vector<int64_t> cut = cut_in;
    if (cut.empty() && P == 1) cut.push_back(N);
    int64_t sum = 0;
    for (int64_t p : cut) sum += p;
    if ((int64_t)cut.size() != P || sum != N || P > 3) {
        fprintf(stderr, "radiance: '%s' was rotated as %lld part(s) and is declared as %zu "
                        "rotated part(s) of %lld rows; the GEMM meets up to three copies "
                        "of the activation\n", wn, (long long)P, cut.size(), (long long)sum);
        return RAD_E_SHAPE;
    }
    for (int64_t i = 0; i + 1 < P; ++i) if (cut[(size_t)i] % 128) {
        fprintf(stderr, "radiance: '%s' rotation part of %lld rows is not whole 128s\n",
                wn, (long long)cut[(size_t)i]);
        return RAD_E_SHAPE;
    }
    rot_parts = P;
    rot_stages = S;
    *pb1 = P > 1 ? cut[0] : 0;
    *pb2 = P > 2 ? cut[0] + cut[1] : 0;
    rot_scale = decl_view(b, nm.f("%s.rot_scale", base), wn, "t.scale", RAD_F32,
                          { P, K }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    return rot_scale ? RAD_OK : RAD_E_INVAL;
}

inline int LinearFP8::decl_seq_tables(RadBuilder* b, Names& nm, const char* base, const char* wn,
                                      RadWeightGroup grp) {
    const int64_t PS = rot_parts * rot_stages;
    rot_pairs = decl_view(b, nm.f("%s.rot_pairs", base), wn, "t.pairs", RAD_I32,
                          { PS, K }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    rot_angle = decl_view(b, nm.f("%s.rot_angle", base), wn, "t.angle", RAD_F32,
                          { PS, K / 2 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    return rot_pairs && rot_angle ? RAD_OK : RAD_E_INVAL;
}

inline rad_op LinearFP8::decl_rot_op(RadBuilder* b, const Geom& g, const ActFP8& a, int64_t lo,
                                     int64_t hi, std::initializer_list<rad_buf> outs) const {
    return rw(b, RAD_OP(b, "givens_rotate",
                  RAD_PARAMS(RAD_RANGE("M", lo, hi), RAD_INT("n", K), RAD_INT("group", RAD_FP8_BLOCK),
                             RAD_STR("dtype", g.dtype), RAD_INT("stages", rot_stages),
                             RAD_INT("parts", rot_parts)),
                  RAD_WEIGHTS(rot_pairs, rot_angle, rot_scale)),
              { a.x }, outs);
}

inline rad_op LinearFP8::decl_dense_op(RadBuilder* b, const Geom& g, const ActFP8& a, const char* out,
                                       int64_t lo, int64_t hi, std::initializer_list<rad_buf> outs) const {
    return rw(b, RAD_OP(b, "givens_dense_quant",
                  RAD_PARAMS(RAD_RANGE("M", lo, hi), RAD_INT("n", K), RAD_INT("group", RAD_FP8_BLOCK),
                             RAD_STR("dtype", g.dtype), RAD_INT("parts", rot_parts),
                             RAD_STR("out", out)),
                  RAD_WEIGHTS(rot_dense, rot_scale)),
              { a.x }, outs);
}

/* The dense rotation matrices of a rotated weight: a view of its "t.dense" table, f16
 * [P * K/128, 128*128], which rad-convert composes from the givens tables (abi/rad_encoding.h). A
 * container with only pairs and angles is refused with a message to reconvert. */
inline rad_weight LinearFP8::decl_dense(RadBuilder* b, Names& nm, const char* base, const char* wn,
                                        const RadEncoding& enc, int64_t K, RadWeightGroup grp) {
    if (!rad_enc_plane(&enc, "t.dense")) {
        fprintf(stderr, "radiance: '%s' carries its rotation as pairs and angles only and the dense "
                        "quantiser reads a composed \"t.dense\" table; convert it again (--reuse "
                        "<this container> copies every other weight)\n", wn);
        return 0;
    }
    return decl_view(b, nm.f("%s.rot_dense", base), wn, "t.dense", RAD_F16,
                     { rot_parts * (K / RAD_FP8_BLOCK), RAD_FP8_BLOCK * RAD_FP8_BLOCK },
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
}

inline int LinearFP8::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                              int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                              std::initializer_list<const char*> src,
                              const ActFP8& a, rad_buf y, int64_t max_rows,
                              std::initializer_list<int64_t> row_parts,
                              int64_t span_lo, int64_t span_hi,
                              std::initializer_list<int64_t> rot_part_rows) {
    N = n_out;
    K = n_in;
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;

    /* Name map first: the weight's encoding, looked up through it, picks the branch below. */
    const char* wn = nm.f("%s.weight", base);
    if (!mapped && !stored) RAD_ARCH_TRY(map_linear(b, wn, src));
    const char* sw = stored ? stored : wn;
    RadEncoding enc{};
    const bool known = weight_enc(b, sw, &enc);
    const bool plain = known && rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                       enc.plane[0].dtype == RAD_BF16;
    if (known && !plain)
        if (const char* gone = linear_retired(enc)) return linear_refuse_retired(sw, gone);

    /* ---- bf16 weight, `gemm_nt` over `a.x`: every linear of a bf16 model, or one a quantised
     * container keeps at checkpoint precision (e.g. the QSA indexer's projection, whose top-k is a
     * hard choice; 20 MB to keep exact). Row parts and uneven slices apply without block
     * constraints. */
    bf16 = g.w_bf16 || plain;
    if (bf16) {
        if (stored) {
            fprintf(stderr, "radiance: '%s' views '%s', which only a block-fp8 linear "
                            "may\n", wn, stored);
            return RAD_E_INVAL;
        }
        if (known && !plain) {
            char en[256];
            rad_enc_format(&enc, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, but this linear runs as a plain bf16 GEMM "
                            "(a bf16 model, or a block that keeps its linears bf16); the "
                            "container must store it bf16\n", wn, en);
            return RAD_E_DTYPE;
        }
        /* A bf16 weight needs the input's bf16 plane, which a quantised model's producer may not
         * write (a gated quantiser writes only codes). */
        if (!a.x) {
            fprintf(stderr, "radiance: '%s' is plain bf16 and its input has no bf16 plane in this "
                            "model; quantise it with its neighbours\n", wn);
            return RAD_E_DTYPE;
        }
        RowParts rp;
        RAD_ARCH_TRY(rp.set(row_parts, 1, 1, base, "rows"));
        w = decl_w_n(b, wn, RAD_BF16, { N, K }, access, shard, grp, 0, rp.w, rp.n);
        if (!w) return RAD_E_INVAL;
        if (span_hi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
        op = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                              RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w)),
               { a.x }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* ---- int8 weight, `gemm_nt_q` at i8a8 over the input's `a.q8` / `a.s8`. The scale plane is
     * one bf16 per row per 128 of K: row parts and row slices apply as is, column slices in groups
     * of 128. The codes come from the producer (ActFP8::q8_fed) or from op_q over `a.x` (probe()
     * set reads_x). The model must have declared the pair (ActFP8::declare_qs's `i8`).
     *
     * A rotated weight (i8a8h) takes had_quant_act_i8g instead. It reads a producer's codes only
     * when they are rotated too (ActFP8::q8_rot, a `_had_quant_i8g` producer); a plain int8 weight
     * only when they are not. A rotated four- or five-bit asymmetric weight (linear_is_wnh) reads
     * the same rotated codes at w4a8i8_asym / w5a8i8, through two views: codes, and (scale, zero)
     * as `b_scale` at the logical extents. */
    const bool asym = known && linear_is_wnh(enc);
    const bool rot8 = known && (linear_is_w8h(enc) || asym);
    i8 = known && (linear_is_w8(enc) || rot8);
    i8_rot = i8 && rot8;
    if (i8) {
        if (stored) {
            fprintf(stderr, "radiance: '%s' views '%s', which only a block-fp8 linear "
                            "may\n", wn, stored);
            return RAD_E_INVAL;
        }
        if (a.q8_fed && rot8 != a.q8_rot) {
            fprintf(stderr, "radiance: '%s' is a%s int8 weight and its input is fed codes its "
                            "producer wrote %s\n", wn, rot8 ? " rotated" : " plain",
                    rot8 ? "unrotated" : "rotated");
            return RAD_E_DTYPE;
        }
        /* The bf16 plane only when this linear quantises it itself. */
        if (!a.q8 || !a.s8 || (!a.q8_fed && !a.x)) {
            fprintf(stderr, "radiance: '%s' is an int8 weight and its input lacks the int8 code "
                            "pair%s (the model declared its input before learning the trunk is "
                            "int8)\n", wn, a.q8_fed ? "" : " or the bf16 plane it quantises");
            return RAD_E_DTYPE;
        }
        /* The scale group: 128, or 256 for IQ8R, which the input's int8 pair must share. */
        const int64_t wg = linear_w8h_group(enc);
        if (wg != a.g8) {
            fprintf(stderr, "radiance: '%s' has a scale per %lld columns and its input's int8 codes "
                            "one per %lld; a trunk mixes IQ8R and 128-group int8 weights\n",
                    wn, (long long)wg, (long long)a.g8);
            return RAD_E_DTYPE;
        }
        if (N % 16 || K % wg) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an int8 weight takes rows in "
                            "16s and K in whole %lld-column scale groups\n",
                    base, (long long)N, (long long)K, (long long)wg);
            return RAD_E_INVAL;
        }
        RowParts rp;
        RAD_ARCH_TRY(rp.set(row_parts, 1, 1, base, "rows"));
        const char* sn = nm.f("%s.scale", base);
        /* A narrower i8a8h weight's codes view takes the plane that decodes it too. */
        const W8hForm form = linear_w8h_form(enc);
        iqr = rot8 && !asym && (form == W8hForm::Zero || form == W8hForm::Table ||
                                form == W8hForm::Grid || form == W8hForm::Trellis);
        const RadEncPlane* tpl = rad_enc_plane(&enc, "trellis");
        const bool trb = form == W8hForm::Trellis && tpl && tpl->block[1] > 1;
        const char* cv = form == W8hForm::Zero    ? "codes,zero"
                       : form == W8hForm::Table ? "codes,table"
                       : form == W8hForm::Grid  ? "codes,grid"
                       : form == W8hForm::Trellis ? (linear_w8h_mad(enc) ? "codes,trellis,trellis.mad"
                                                                          : "codes,trellis")
                                                : "codes";
        /* Declared at the stored codes' dtype; the kernel's layout keeps them at that width
         * (libr4d's r4d_iqr_layout.cpp). */
        w  = decl_view(b, wn, nullptr, cv, enc.plane[0].dtype, { N, K }, access, shard, grp,
                       rp.w, rp.n);
        /* A view of two planes is declared at the weight's logical extents and sliced in its
         * columns, the stored planes left to the layout hook. */
        const bool iq8r = linear_is_iq8r(enc);
        const bool two = linear_w8h_scale2(enc) || linear_w8h_rowf(enc) || iq8r;
        ws = asym ? decl_view(b, sn, wn, "scale,zero", RAD_F16, { N, K }, access, shard, grp,
                              rp.w, rp.n)
             : linear_w8h_rowf(enc) ? decl_view(b, sn, wn, "scale,scale.col", RAD_F32, { N, K },
                                                access, shard, grp, rp.w, rp.n)
             : two ? decl_view(b, sn, wn, "scale,scale.1", iq8r ? RAD_U8 : RAD_F8E4M3, { N, K },
                               access, shard, grp, rp.w, rp.n)
                   : decl_view(b, sn, wn, "scale", RAD_BF16, { N, K / wg }, access,
                               shard, grp, rp.w, rp.n);
        if (!w || !ws) return RAD_E_INVAL;
        if (span_hi > 0) {
            /* The (scale, zero) and (scale, scale.1) views are at the logical extents, sliced as
             * the codes are. */
            const bool col = shard == RAD_SHARD_COL && !asym && !two;
            if (col && (span_lo % wg || span_hi % wg)) {
                fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                                "%lld-column scale group\n", base, (long long)span_lo,
                        (long long)span_hi, (long long)wg);
                return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
            if ((asym || two) && shard == RAD_SHARD_COL && (span_lo % wg || span_hi % wg)) {
                fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                                "%lld-column scale group\n", base, (long long)span_lo,
                        (long long)span_hi, (long long)wg);
                return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_weight_shard_span(b, ws, col ? span_lo / wg : span_lo,
                                               col ? span_hi / wg : span_hi));
        }
        const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
        if (!a.q8_fed)
            op_q = rw(b, RAD_OP(b, rot8 ? "had_quant_act_i8g" : "quant_act_i8g",
                     RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("n", K),
                                RAD_INT("group", wg), RAD_STR("dtype", "bf16")),
                     RAD_NOWEIGHTS),
                  { a.x }, { a.q8, a.s8 });
        const char* qdt = asym ? (enc.plane[0].dtype == RAD_U5 ? "w5a8i8" : "w4a8i8_asym")
                               : rot8 ? "i8a8h" : "i8a8";
        /* A trellis on an int8 grid of several rows a scale (libquant's tcq_rb) says how many
         * (`w_rb`), which a GEMM's fold may read as one scale a block of rows. */
        op = trb ? rw(b, RAD_OP(b, "gemm_nt_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", qdt),
                                     RAD_INT("w_rb", tpl->block[1])),
                          RAD_WEIGHTS(w, ws)),
                      { a.q8, a.s8 }, { y })
                 : rw(b, RAD_OP(b, "gemm_nt_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_INT("group", wg), RAD_STR("dtype", qdt)),
                          RAD_WEIGHTS(w, ws)),
                      { a.q8, a.s8 }, { y });
        return (op_q || a.q8_fed) && op ? RAD_OK : RAD_E_INVAL;
    }

    /* ---- Rotated 5-bit or asymmetric 4-bit weight (ParoQuant int5, z-lab int4): the input's bf16
     * plane rotated into this linear's own f16 plane, then the w5a16 / w4a16_asym GEMM. Two views:
     * codes, and (scale, zero) as `b_scale`. */
    paro5 = known && linear_is_paro_w5(enc);
    if (paro5) {
        if (stored) {
            fprintf(stderr, "radiance: '%s' views '%s', which only a block-fp8 linear "
                            "may\n", wn, stored);
            return RAD_E_INVAL;
        }
        const int bits = enc.plane[0].dtype == RAD_U4 ? 4 : 5;
        if (!a.x) {
            fprintf(stderr, "radiance: '%s' is a rotated %d-bit weight, rotated from its input's "
                            "bf16 plane, and its input has none\n", wn, bits);
            return RAD_E_DTYPE;
        }
        if (g.world > 1 || span_hi > 0) {
            fprintf(stderr, "radiance: '%s' is a rotated %d-bit weight; its rotation is not cut by "
                            "a tensor-parallel shard (world %d)\n", wn, bits, g.world);
            return RAD_E_UNSUPPORTED;
        }
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld; a %d-bit weight takes rows in 16s and K "
                            "in whole 128s\n", base, (long long)N, (long long)K, bits);
            return RAD_E_INVAL;
        }
        std::vector<int64_t> cut;
        for (int64_t p : rot_part_rows) cut.push_back(p);
        if (cut.empty()) for (int64_t p : row_parts) cut.push_back(p);
        int64_t pb1 = 0, pb2 = 0;
        const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
        const bool bits5 = enc.plane[0].dtype == RAD_U5;
        const int64_t kSplit = 64;                      /* the skinny kernel's M ceiling */
        const bool band = hi > kSplit;
        const int64_t f16_hi = band ? kSplit : hi;      /* the f16 chain's rows */
        RAD_ARCH_TRY(declare_rotation(b, nm, base, wn, enc, N, K, grp, cut, &pb1, &pb2));
        RAD_ARCH_TRY(decl_seq_tables(b, nm, base, wn, grp));
        const int64_t P = rot_parts;
        rot_q = decl_b(b, nm.f("%s.rot_y", base), RAD_F16, { f16_hi, P * K });
        if (!rot_q) return RAD_E_INVAL;
        op_rot = decl_rot_op(b, g, a, 1, f16_hi, { rot_q });
        if (!op_rot) return RAD_E_INVAL;
        w  = decl_view(b, wn, nullptr, "codes", enc.plane[0].dtype, { N, K }, access,
                       shard, grp);
        ws = decl_view(b, nm.f("%s.scale", base), wn, "scale,zero", RAD_F16, { N, K },
                       access, shard, grp);
        if (!w || !ws) return RAD_E_INVAL;
        op = decl_gemm_q(b, 1, f16_hi, RAD_FP8_BLOCK, bits5 ? "w5a16" : "w4a16_asym",
                         pb1, pb2, { w, ws }, { rot_q }, y);
        if (!op) return RAD_E_INVAL;
        if (band) {
            /* Prefill band: rows above the skinny kernel's ceiling rotate to int8 and use the int8
             * tiled GEMM, over the same tables and weight views. */
            rot_split = kSplit;
            rot_q2 = decl_b(b, nm.f("%s.rot_q2", base), RAD_I8, { hi, P * K });
            rot_s2 = decl_b(b, nm.f("%s.rot_s2", base), RAD_F32, { hi, P * (K / RAD_FP8_BLOCK) });
            if (!rot_q2 || !rot_s2) return RAD_E_INVAL;
            rot_dense = decl_dense(b, nm, base, wn, enc, K, grp);
            if (!rot_dense) return RAD_E_INVAL;
            op_rot2 = decl_dense_op(b, g, a, "i8", kSplit + 1, hi, { rot_q2, rot_s2 });
            op2 = decl_gemm_q(b, kSplit + 1, hi, RAD_FP8_BLOCK, bits5 ? "w5a8i8" : "w4a8i8_asym",
                              pb1, pb2, { w, ws }, { rot_q2, rot_s2 }, y);
            if (!op_rot2 || !op2) return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* ---- 4-bit weight with a zero point and no rotation: the rotated int4's two GEMMs over this
     * linear's BF16 input. At or below kSplit rows (the skinny F16 kernel's ceiling) the GEMM
     * narrows that input on load; above it, where that kernel is bound by the F16 matrix
     * instruction, the input is quantised to int8 with an F32 scale per
     * (row, 128) for w4a8i8_asym. Two views: codes, and (scale, zero) as `b_scale`, both at the
     * logical extents, so row parts and spans apply to both as they are; a column slice must be
     * whole 128s, the scale group. */
    w4 = known && linear_is_w4(enc);
    if (w4) {
        if (stored) {
            fprintf(stderr, "radiance: '%s' views '%s', which only a block-fp8 linear "
                            "may\n", wn, stored);
            return RAD_E_INVAL;
        }
        if (!a.x) {
            fprintf(stderr, "radiance: '%s' is a 4-bit weight, read from its input's bf16 plane, "
                            "and its input has none\n", wn);
            return RAD_E_DTYPE;
        }
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; a 4-bit weight takes rows in "
                            "16s and K in whole %d-column groups\n",
                    base, (long long)N, (long long)K, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        RowParts rp;
        RAD_ARCH_TRY(rp.set(row_parts, 1, 1, base, "rows"));
        w  = decl_view(b, wn, nullptr, "codes", RAD_U4, { N, K }, access, shard, grp, rp.w, rp.n);
        ws = decl_view(b, nm.f("%s.scale", base), wn, "scale,zero", RAD_F16, { N, K }, access,
                       shard, grp, rp.w, rp.n);
        if (!w || !ws) return RAD_E_INVAL;
        if (span_hi > 0) {
            if (shard == RAD_SHARD_COL && (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK)) {
                fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                                "%d-column scale group\n", base, (long long)span_lo,
                        (long long)span_hi, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
            RAD_ARCH_TRY(rad_weight_shard_span(b, ws, span_lo, span_hi));
        }
        const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
        const int64_t kSplit = 64;
        const int64_t f16_hi = hi > kSplit ? kSplit : hi;
        op = decl_gemm_q(b, 1, f16_hi, RAD_FP8_BLOCK, "w4a16_asym", 0, 0, { w, ws }, { a.x }, y);
        if (!op) return RAD_E_INVAL;
        if (hi > kSplit) {
            w4_split = kSplit;
            a8q = decl_b(b, nm.f("%s.a8q", base), RAD_I8, { hi, K });
            a8s = decl_b(b, nm.f("%s.a8s", base), RAD_F32, { hi, K / RAD_FP8_BLOCK });
            if (!a8q || !a8s) return RAD_E_INVAL;
            op_q = rw(b, RAD_OP(b, "quant_act_i8g",
                         RAD_PARAMS(RAD_RANGE("M", kSplit + 1, hi), RAD_INT("n", K),
                                    RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                      { a.x }, { a8q, a8s });
            op2 = decl_gemm_q(b, kSplit + 1, hi, RAD_FP8_BLOCK, "w4a8i8_asym", 0, 0, { w, ws },
                              { a8q, a8s }, y);
            if (!op_q || !op2) return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* The remaining formats read the E4M3 pair, which an int8-fed input lacks (a trunk mixing
     * formats that the model's probe did not detect). */
    if (a.q8_fed) {
        fprintf(stderr, "radiance: '%s' is not an int8 weight, and its input is fed int8 codes "
                        "because this model's trunk is int8; quantise it with its neighbours\n", wn);
        return RAD_E_DTYPE;
    }

    /* ---- OCP-MXFP4 or M2XFP4 weight over an E4M3 pair, the producer's or, for a rotated
     * (ParoQuant) weight, its own. Three views: codes (M2XFP4's with its meta plane, which rides
     * the weight operand), exponents as `b_scale`, and the same plane as `b_ref`. Row parts apply
     * as is (the exponent grid is one row tall); column slices must be whole 128s, the
     * activation's scale block. */
    const bool paro_mx = known && linear_is_paro_mxfp4(enc);
    const bool paro_m2 = known && linear_is_paro_m2xfp4(enc);
    paro = paro_mx || paro_m2;
    mx = known && (linear_is_mxfp4(enc) || paro_mx);
    m2x = known && (linear_is_m2xfp4(enc) || paro_m2);
    if (mx || m2x) {
        const char* fmt = m2x ? "M2XFP4" : "MXFP4";
        if (stored) {
            fprintf(stderr, "radiance: '%s' views '%s', which only a block-fp8 linear "
                            "may\n", wn, stored);
            return RAD_E_INVAL;
        }
        if (paro) {
            /* The rotation is applied to the activation, so this linear needs the input's bf16
             * plane (probe() set reads_x) and quantises it itself. */
            if (!a.x) {
                fprintf(stderr, "radiance: '%s' is a rotated %s weight, rotated and quantised "
                                "from its input's bf16 plane, and its input has none\n", wn, fmt);
                return RAD_E_DTYPE;
            }
            if (g.world > 1 || span_hi > 0) {
                fprintf(stderr, "radiance: '%s' is a rotated %s weight; its rotation is per "
                                "projection and is not cut by a tensor-parallel shard (world %d)\n",
                        wn, fmt, g.world);
                return RAD_E_UNSUPPORTED;
            }
        } else if (!a.q || !a.s) {
            fprintf(stderr, "radiance: '%s' is an %s weight read against its input's E4M3 "
                            "pair, and its input has none\n", wn, fmt);
            return RAD_E_DTYPE;
        }
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an %s weight takes rows in "
                            "16s and K in whole %d-column activation scale blocks\n",
                    base, (long long)N, (long long)K, fmt, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        RowParts rp;
        RAD_ARCH_TRY(rp.set(row_parts, 16, 1, base, m2x ? "M2XFP4 tiles" : "MXFP4 tiles"));
        int64_t pb1 = 0, pb2 = 0;
        if (paro) {
            std::vector<int64_t> cut;
            for (int64_t p : rot_part_rows) cut.push_back(p);
            if (cut.empty()) for (int64_t p : row_parts) cut.push_back(p);
            RAD_ARCH_TRY(declare_rotation(b, nm, base, wn, enc, N, K, grp, cut, &pb1, &pb2));
            const int64_t rows = max_rows > 0 ? max_rows : g.max_tok;
            rot_q = decl_b(b, nm.f("%s.rot_q", base), RAD_F8E4M3, { rows, rot_parts * K });
            rot_s = decl_b(b, nm.f("%s.rot_s", base), RAD_F32, { rows, rot_parts * (K / RAD_FP8_BLOCK) });
            if (!rot_q || !rot_s) return RAD_E_INVAL;
            rot_dense = decl_dense(b, nm, base, wn, enc, K, grp);
            if (!rot_dense) return RAD_E_INVAL;
            op_rot = decl_dense_op(b, g, a, "fp8", 1, rows, { rot_q, rot_s });
            if (!op_rot) return RAD_E_INVAL;
        }
        const char* sn = nm.f("%s.scale", base);
        const char* rn = nm.f("%s.ref", base);
        w    = decl_view(b, wn, nullptr, m2x ? "codes,meta" : "codes", RAD_FP4E2M1, { N, K },
                         access, shard, grp, rp.w, rp.n);
        ws   = decl_view(b, sn, wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                         access, shard, grp, rp.w, rp.n);
        wref = decl_view(b, rn, wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                         access, shard, grp, rp.w, rp.n);
        if (!w || !ws || !wref) return RAD_E_INVAL;
        if (span_hi > 0) {
            const bool col = shard == RAD_SHARD_COL;
            if (col && (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK)) {
                fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                                "%d-column activation scale block\n", base, (long long)span_lo,
                        (long long)span_hi, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            const int64_t lo = col ? span_lo / RAD_MX_GROUP : span_lo;
            const int64_t hi = col ? span_hi / RAD_MX_GROUP : span_hi;
            RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
            RAD_ARCH_TRY(rad_weight_shard_span(b, ws, lo, hi));
            RAD_ARCH_TRY(rad_weight_shard_span(b, wref, lo, hi));
        }
        /* One GEMM: a rotated weight reads its own rotated pair, a plain one the producer's. */
        op = decl_gemm_q(b, 1, max_rows > 0 ? max_rows : g.max_tok, RAD_MX_GROUP,
                         m2x ? "m2xfp4" : "mxfp4a8", pb1, pb2,
                         { w, ws, wref }, { paro ? rot_q : a.q, paro ? rot_s : a.s }, y);
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* Block fp8 is the last form; any other encoding is refused by name with the ones read. */
    if (known) {
        const RadEncPlane* sc = rad_enc_plane(&enc, "scale");
        if (!rad_enc_is(&enc, "affine") || enc.n_planes != 2 || enc.transform[0] ||
            enc.plane[0].dtype != RAD_F8E4M3 || !sc ||
            (sc->dtype != RAD_BF16 && sc->dtype != RAD_F32) ||
            sc->block[0] != RAD_FP8_BLOCK || sc->block[1] != RAD_FP8_BLOCK) {
            char en[256];
            rad_enc_format(&enc, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, which this model's linears do not read; they "
                            "read bf16, fp8_e4m3*bf16[128x128] or *f32[128x128], i8*bf16[1x128], "
                            "fp4_e2m1*e8m0[1x32], u4*f16[1x128]-u4[1x128] (or a u8 zero) and the "
                            "ParoQuant forms. A recipe rule stores the weight in one of them "
                            "(docs/TOOLS.md, the import rule for a shipped int4 grid)\n", wn, en);
            return RAD_E_DTYPE;
        }
    }

    /* Block fp8: per-rank extents must be whole scale blocks, checked here to name the weight (this
     * also catches a tensor-parallel width that cuts a block). */
    if (N % RAD_FP8_BLOCK || K % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: '%s' is %lldx%lld per rank, which is not a whole number of "
                        "%dx%d scale blocks. A block-scaled fp8 weight cannot be split inside a "
                        "block; use a bf16 linear here or a world_size that divides cleanly\n",
                base, (long long)N, (long long)K, RAD_FP8_BLOCK, RAD_FP8_BLOCK);
        return RAD_E_INVAL;
    }

    /* The scale plane is split where the weight is, in RAD_FP8_BLOCK rows: fused [q|k|v] parts of
     * 1024/1024/3072 rows are 8/8/24 scale rows. Without this the scale would be sliced
     * contiguously while the weight was not, passing shape checks with wrong values. Parts must be
     * whole blocks. */
    RowParts rp;
    RAD_ARCH_TRY(rp.set(row_parts, RAD_FP8_BLOCK, RAD_FP8_BLOCK, base, "scale blocks"));

    const char* sn = nm.f("%s.scale",  base);
    const auto* fp8_scale = known ? rad_enc_plane(&enc, "scale") : nullptr;
    f32_scale = fp8_scale && fp8_scale->dtype == RAD_F32;
    /* One stored weight, two operands: the codes and scale plane of `<base>.weight`, block fp8 in
     * the checkpoint or made so by rad-convert's recipe. */
    w  = decl_view(b, wn, stored, "codes", RAD_F8E4M3, { N, K },
                   access, shard, grp, rp.w, rp.n);
    ws_row_div = RAD_FP8_BLOCK;
    ws = decl_view(b, sn, sw, "scale", f32_scale ? RAD_F32 : RAD_BF16,
                   { fp8_blocks(N), fp8_blocks(K) }, access, shard, grp, rp.s, rp.n);
    if (!w || !ws) return RAD_E_INVAL;
    /* An uneven slice; the scale plane takes the same slice in blocks. */
    if (span_hi > 0) {
        if (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a %d-"
                            "wide scale block\n", base, (long long)span_lo, (long long)span_hi,
                    RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
        RAD_ARCH_TRY(rad_weight_shard_span(b, ws, span_lo / RAD_FP8_BLOCK,
                                           span_hi / RAD_FP8_BLOCK));
    }

    /* `M` ranges over the token count and selects libr4d's narrow (M <= 16) or tiled kernel. A
     * caller with a tighter bound (the lm_head, over sampled rows only) passes `max_rows`, so the
     * narrow kernel's split-K scratch is not sized for max_tok. */
    const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
    op = rw(b, RAD_OP(b, short_split && !f32_scale ? "gemm_nt_q_short" : "gemm_nt_q",
               RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                          RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", f32_scale?"fp8a8f32":"fp8a8")),
               RAD_WEIGHTS(w, ws)),
           { a.q, a.s }, { y });
    return RAD_OK;
}

/* `gemm_nt_q` has seven positional operands; the optional `a_sum` and `b_ref` after `y` (for
 * asymmetric integer and mxfp4 grids) are passed as RAD_NONE when unused, since the runtime checks
 * the operand count against the schema (core/runtime/issue.cpp). */
inline void LinearFP8::step(RadCtx* c, const ActFP8& a, rad_buf y, int64_t T, int64_t r0,
                            int64_t rows, int64_t y0) const {
    if (rows <= 0) rows = T - r0;
    const int64_t yr = y0 < 0 ? r0 : y0;
    if (bf16) {
        RAD_ISSUE_N(c, op, rows, brow_slice(a.x, r0, rows, K), RAD_W(w), brow_slice(y, yr, rows, N));
        return;
    }
    if (i8) {
        if (op_q)
            RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, K), brow_slice(a.q8, r0, rows, K),
                        brow_slice(a.s8, r0, rows, K / a.g8));
        RAD_ISSUE_N(c, op, rows, brow_slice(a.q8, r0, rows, K),
                    brow_slice(a.s8, r0, rows, K / a.g8),
                    RAD_W(w), RAD_W(ws), brow_slice(y, yr, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    /* The 4-bit weight's copies of the input start at their buffers' first row: they live only
     * between the two ops issued here. */
    if (w4 && w4_split && rows > w4_split) {
        RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, K), brow_slice(a8q, 0, rows, K),
                    brow_slice(a8s, 0, rows, K / RAD_FP8_BLOCK));
        RAD_ISSUE_N(c, op2, rows, brow_slice(a8q, 0, rows, K),
                    brow_slice(a8s, 0, rows, K / RAD_FP8_BLOCK), RAD_W(w), RAD_W(ws),
                    brow_slice(y, yr, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (w4) {
        RAD_ISSUE_N(c, op, rows, brow_slice(a.x, r0, rows, K), RAD_NONE, RAD_W(w), RAD_W(ws),
                    brow_slice(y, yr, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (paro5 && rot_split && rows > rot_split) {
        RAD_ISSUE_N(c, op_rot2, rows, brow_slice(a.x, r0, rows, K), RAD_W(rot_dense),
                    RAD_W(rot_scale), brow_slice(rot_q2, r0, rows, rot_parts * K),
                    brow_slice(rot_s2, r0, rows, rot_parts * (K / RAD_FP8_BLOCK)));
        RAD_ISSUE_N(c, op2, rows, brow_slice(rot_q2, r0, rows, rot_parts * K),
                    brow_slice(rot_s2, r0, rows, rot_parts * (K / RAD_FP8_BLOCK)),
                    RAD_W(w), RAD_W(ws), brow_slice(y, yr, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (paro5) {
        RAD_ISSUE_N(c, op_rot, rows, brow_slice(a.x, r0, rows, K), RAD_W(rot_pairs),
                    RAD_W(rot_angle), RAD_W(rot_scale), brow_slice(rot_q, r0, rows, rot_parts * K));
        RAD_ISSUE_N(c, op, rows, brow_slice(rot_q, r0, rows, rot_parts * K), RAD_NONE,
                    RAD_W(w), RAD_W(ws), brow_slice(y, yr, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (mx || m2x) {
        /* A rotated weight (MXFP4 or M2XFP4) first rotates its input into its own codes,
         * `rot_parts` copies side by side; a plain weight reads the producer's. */
        const rad_buf aq = paro ? rot_q : a.q, as = paro ? rot_s : a.s;
        const int64_t ak = rot_parts * K;
        if (paro)
            RAD_ISSUE_N(c, op_rot, rows, brow_slice(a.x, r0, rows, K), RAD_W(rot_dense),
                        RAD_W(rot_scale), brow_slice(aq, r0, rows, ak),
                        brow_slice(as, r0, rows, ak / RAD_FP8_BLOCK));
        RAD_ISSUE_N(c, op, rows, brow_slice(aq, r0, rows, ak),
                    brow_slice(as, r0, rows, ak / RAD_FP8_BLOCK),
                    RAD_W(w), RAD_W(ws), brow_slice(y, yr, rows, N), RAD_NONE, RAD_W(wref));
        return;
    }
    RAD_ISSUE_N(c, op, rows, brow_slice(a.q, r0, rows, K),
                brow_slice(a.s, r0, rows, K / RAD_FP8_BLOCK),
                RAD_W(w), RAD_W(ws), brow_slice(y, yr, rows, N), RAD_NONE, RAD_NONE);
}

/* ================================================== an fp8 weight with no op of its own
 *
 * LinearFP8's views without the GEMM, for blocks that declare their own op over the weight: the
 * DFlash2 and DSpark drafters' projections, shipped bf16 and made block fp8 by the recipe (read
 * whole every decode step, and drafts are verified).
 *
 * Views are declared at what they select ([N][K] E4M3 codes, [N/128][K/128] BF16 scales), the
 * shapes libr4d checks before running. Row parts are rows for the weight and 128-row blocks for
 * the plane, as in LinearFP8; parts must be whole blocks.
 */
struct DenseFP8 {
    rad_weight w = 0, s = 0;
    /* Or OCP-MXFP4 when the container holds that (`codes=fp4_e2m1 group=32 scale=e8m0`): e2m1
     * codes, the E8M0 plane, and that plane again as the per-row reference (LinearFP8's mx branch).
     * Blocks declare the GEMM with `gemm()` and pass `bref()` as its last operand, so either format
     * works against the same E4M3 activation pair. */
    rad_weight ref = 0;
    bool       mx = false;
    /* Or M2XFP4 (libquant `m2xfp4`): the MXFP4 views, the meta plane riding the weight operand. */
    bool       m2x = false;
    /* Or a weight served at i8a8h (linear_w8h_form): IQ8R, the rotated int8 grid at 128, or a
     * narrower form on it (signed codes, a zero point, a codebook, a trellis), its views as
     * LinearFP8 declares them. It reads an INT8 activation pair with one scale per (row, `g8`) under
     * the same rotation (had_quant_act_i8g at that group); the block declares that pair where it
     * would the E4M3 one. `w_rb`: a trellis on a grid of several rows a scale says how many. */
    bool       w8h = false;
    int64_t    g8 = RAD_FP8_BLOCK;
    int64_t    w_rb = 0;
    /* Set before declare by a block that declares the int8 activation pair such a weight reads;
     * any other block's activations are E4M3, so declare refuses the weight there. */
    bool       w8h_ok = false;

    /* `mapped`: the caller made the name map already, to ask the encoding before choosing this
     * struct over another grid's -- a COPY map declared twice is refused (rad_builder.cpp). */
    int declare(RadBuilder* b, Names& nm, const char* base, int64_t N, int64_t K,
                int access, int shard, RadWeightGroup grp,
                std::initializer_list<const char*> src,
                std::initializer_list<int64_t> row_parts = {}, bool mapped = false);
    /* Rows [lo, hi) of the stored block-fp8 weight `src_wn` (codes and scale plane; mapped by
     * whoever declared it), read as an [hi - lo, K] weight named `<base>`: a row slice that need
     * not be a rank's even share, and may be the same rows on every rank. Whole 128-row blocks,
     * so each slice's codes and scales are the stored ones. fp8 only: an MXFP4 linear's GEMM
     * splits K by its own rule. A slice issued with `gemm(..., split_n)` set to the stored width
     * computes the stored linear's outputs to the bit. */
    int declare_slice(RadBuilder* b, Names& nm, const char* base, const char* src_wn, int64_t lo,
                      int64_t hi, int64_t K, int access, RadWeightGroup grp) {
        if (lo < 0 || hi <= lo || lo % RAD_FP8_BLOCK || hi % RAD_FP8_BLOCK || K % RAD_FP8_BLOCK)
            return RAD_E_INVAL;
        mx  = false;
        m2x = false;
        ref = 0;
        const char* wn = nm.f("%s.weight", base);
        const char* sn = nm.f("%s.scale", base);
        const bool own = !std::strcmp(wn, src_wn);   /* the stored weight's own name */
        w = decl_view(b, wn, own ? nullptr : src_wn, "codes", RAD_F8E4M3, { hi - lo, K }, access,
                      RAD_SHARD_ROW, grp);
        s = decl_view(b, sn, src_wn, "scale", RAD_BF16, { fp8_blocks(hi - lo), fp8_blocks(K) },
                      access, RAD_SHARD_ROW, grp);
        if (!w || !s) return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_weight_shard_span(b, w, lo, hi));
        return rad_weight_shard_span(b, s, lo / RAD_FP8_BLOCK, hi / RAD_FP8_BLOCK);
    }
    /* This rank's rows [lo, hi) of a RAD_SHARD_ROW weight where the even shard is the wrong cut: a
     * KV head replicated over ranks (Geom::kv_off), the same rows on every rank that shares it.
     * Call after declare(), whose N is already hi - lo. The fp8 scale plane takes the span in its
     * 128-row blocks; the MXFP4 and M2XFP4 planes are one row a weight row, as in LinearFP8. */
    int span(RadBuilder* b, int64_t lo, int64_t hi) const {
        RAD_ARCH_TRY(rad_weight_shard_span(b, w, lo, hi));
        if (mx || m2x) {
            RAD_ARCH_TRY(rad_weight_shard_span(b, s, lo, hi));
            return rad_weight_shard_span(b, ref, lo, hi);
        }
        if (lo % RAD_FP8_BLOCK || hi % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: a row span [%lld, %lld) is not whole %d-row scale blocks\n",
                    (long long)lo, (long long)hi, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        return rad_weight_shard_span(b, s, lo / RAD_FP8_BLOCK, hi / RAD_FP8_BLOCK);
    }
    /* `gemm_nt_q` over this weight, `fp8a8`, `mxfp4a8` or `m2xfp4` as it was declared, ranged
     * M 1..hi. `split_n`, when set, is the stored width a row slice takes its K split from
     * (docs/OPS.md). */
    rad_op gemm(RadBuilder* b, int64_t hi, int64_t N, int64_t K,
                std::initializer_list<rad_buf> reads, std::initializer_list<rad_buf> writes,
                int64_t split_n = 0) const {
        if (w8h && w_rb > 0)
            return rw(b, RAD_OP(b, "gemm_nt_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_INT("group", g8), RAD_STR("dtype", "i8a8h"),
                                     RAD_INT("w_rb", w_rb)),
                          RAD_WEIGHTS(w, s)),
                      reads, writes);
        if (w8h)
            return rw(b, RAD_OP(b, "gemm_nt_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_INT("group", g8), RAD_STR("dtype", "i8a8h")),
                          RAD_WEIGHTS(w, s)),
                      reads, writes);
        if (mx || m2x)
            return rw(b, RAD_OP(b, "gemm_nt_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_INT("group", RAD_MX_GROUP),
                                     RAD_STR("dtype", m2x ? "m2xfp4" : "mxfp4a8")),
                          RAD_WEIGHTS(w, s, ref)),
                      reads, writes);
        if (split_n > 0)
            return rw(b, RAD_OP(b, "gemm_nt_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8"),
                                     RAD_INT("split_n", split_n)),
                          RAD_WEIGHTS(w, s)),
                      reads, writes);
        return rw(b, RAD_OP(b, "gemm_nt_q",
                      RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                      RAD_WEIGHTS(w, s)),
                  reads, writes);
    }
    /* The op's trailing `b_ref` operand: the reference view for MXFP4 and M2XFP4, absent for
     * fp8. */
    RadOperand bref() const { return mx || m2x ? RAD_W(ref) : RAD_NONE; }
};

inline int DenseFP8::declare(RadBuilder* b, Names& nm, const char* base, int64_t N, int64_t K,
                             int access, int shard, RadWeightGroup grp,
                             std::initializer_list<const char*> src,
                             std::initializer_list<int64_t> row_parts, bool mapped) {
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;

    /* ---- OCP-MXFP4 or M2XFP4, from the weight's encoding. The exponent grid is one row tall, so
     * parts need only whole 16-row tiles; K takes whole 128s for the activation's per-(row, 128)
     * scale. */
    {
        const char* wn = nm.f("%s.weight", base);
        if (!mapped) RAD_ARCH_TRY(map_linear(b, wn, src));
        RadEncoding enc{};
        const bool known = weight_enc(b, wn, &enc);
        mx = known && linear_is_mxfp4(enc);
        m2x = known && linear_is_m2xfp4(enc);
        /* Plain MXFP4 is still read here: the published IQR-V4 containers carry their DFlash2
         * drafter in it. */
        if (known && !mx)
            if (const char* gone = linear_retired(enc)) return linear_refuse_retired(wn, gone);
        const W8hForm form = known ? linear_w8h_form(enc) : W8hForm::None;
        w8h = form != W8hForm::None;
        if (w8h && !w8h_ok) {
            fprintf(stderr, "radiance: '%s' is a rotated int8 weight, which reads an int8 "
                            "activation; this block quantises its activations to E4M3 and takes "
                            "block fp8 or MXFP4 weights\n", base);
            return RAD_E_DTYPE;
        }
        if (w8h) {
            g8 = linear_w8h_group(enc);
            if (N % 16 || K % g8) {
                fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; a rotated int8 weight takes "
                                "rows in 16s and K in whole %lld-column scale groups\n",
                        base, (long long)N, (long long)K, (long long)g8);
                return RAD_E_INVAL;
            }
            RowParts rp;
            RAD_ARCH_TRY(rp.set(row_parts, 16, 1, base, "int8 tiles"));
            /* The views LinearFP8's i8a8h branch declares: a narrower form's codes take the plane
             * that decodes them, and a scale of two planes is one view at the weight's logical
             * extents, which the kernel's layout combines. */
            const RadEncPlane* tpl = rad_enc_plane(&enc, "trellis");
            w_rb = form == W8hForm::Trellis && tpl && tpl->block[1] > 1 ? tpl->block[1] : 0;
            const char* cv = form == W8hForm::Zero    ? "codes,zero"
                           : form == W8hForm::Table   ? "codes,table"
                           : form == W8hForm::Grid    ? "codes,grid"
                           : form == W8hForm::Trellis ? (linear_w8h_mad(enc) ? "codes,trellis,trellis.mad"
                                                                              : "codes,trellis")
                                                      : "codes";
            const char* sn = nm.f("%s.scale", base);
            const bool iq8r = linear_is_iq8r(enc);
            w = decl_view(b, wn, nullptr, cv, enc.plane[0].dtype, { N, K }, access, shard, grp,
                          rp.w, rp.n);
            s = linear_w8h_rowf(enc)
                    ? decl_view(b, sn, wn, "scale,scale.col", RAD_F32, { N, K }, access, shard, grp,
                                rp.w, rp.n)
              : (linear_w8h_scale2(enc) || iq8r)
                    ? decl_view(b, sn, wn, "scale,scale.1", iq8r ? RAD_U8 : RAD_F8E4M3, { N, K },
                                access, shard, grp, rp.w, rp.n)
                    : decl_view(b, sn, wn, "scale", RAD_BF16, { N, K / g8 }, access, shard, grp,
                                rp.w, rp.n);
            return w && s ? RAD_OK : RAD_E_INVAL;
        }
        if (mx || m2x) {
            if (N % 16 || K % RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an %s weight takes rows "
                                "in 16s and K in whole %d-column activation scale blocks\n",
                        base, (long long)N, (long long)K, m2x ? "M2XFP4" : "MXFP4", RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            RowParts rp;
            RAD_ARCH_TRY(rp.set(row_parts, 16, 1, base, m2x ? "M2XFP4 tiles" : "MXFP4 tiles"));
            w   = decl_view(b, wn, nullptr, m2x ? "codes,meta" : "codes", RAD_FP4E2M1, { N, K },
                            access, shard, grp, rp.w, rp.n);
            s   = decl_view(b, nm.f("%s.scale", base), wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                            access, shard, grp, rp.w, rp.n);
            ref = decl_view(b, nm.f("%s.ref", base), wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                            access, shard, grp, rp.w, rp.n);
            return w && s && ref ? RAD_OK : RAD_E_INVAL;
        }
    }
    if (N % RAD_FP8_BLOCK || K % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: '%s' is %lldx%lld per rank, which is not a whole number of "
                        "%dx%d scale blocks. A block-scaled fp8 weight cannot be split inside a "
                        "block; use a bf16 linear here or a world_size that divides cleanly\n",
                base, (long long)N, (long long)K, RAD_FP8_BLOCK, RAD_FP8_BLOCK);
        return RAD_E_INVAL;
    }

    RowParts rp;
    RAD_ARCH_TRY(rp.set(row_parts, RAD_FP8_BLOCK, RAD_FP8_BLOCK, base, "scale blocks"));

    const char* wn = nm.f("%s.weight", base);
    const char* sn = nm.f("%s.scale",  base);   /* the name map was made above */
    w = decl_view(b, wn, nullptr, "codes", RAD_F8E4M3, { N, K }, access, shard, grp, rp.w, rp.n);
    s = decl_view(b, sn, wn, "scale", RAD_BF16, { fp8_blocks(N), fp8_blocks(K) }, access, shard,
                  grp, rp.s, rp.n);
    return w && s ? RAD_OK : RAD_E_INVAL;
}

/* gate_up, silu_mul and the E4M3 quantiser in one GEMM epilogue (`gemm_nt_q_gated`, an fp8a8
 * tile over a block-fp8 [2n, K] weight), declared for M up to min(rows, 64): the kernel refuses M
 * past that band, unless the caller asks for the wide row (`wide`), whose gated prefill tile
 * serves every M. Optional: when it does not resolve the caller's unfused ops run, and the issue
 * site checks `*gated_rows`, set to the band only when the op exists. */
inline rad_op decl_gated_fold(RadBuilder* b, int64_t rows, int64_t N, int64_t K, rad_weight w,
                              rad_weight ws, std::initializer_list<rad_buf> reads,
                              std::initializer_list<rad_buf> writes, int64_t* gated_rows,
                              bool wide = false) {
    const int64_t hi = wide ? rows : (rows < 64 ? rows : 64);
    if (hi <= 0) return 0;
    const rad_op op = rw(b, RAD_OP(b, wide ? "gemm_nt_q_gated_wide" : "gemm_nt_q_gated",
                             RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                        RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8"),
                                        RAD_STR("act", "silu")),
                             RAD_WEIGHTS(w, ws)),
                         reads, writes);
    if (op) *gated_rows = hi;
    return op;
}

/* The same fold over an OCP-MXFP4 [2n, K] gate_up weight (`gemm_nt_q_gated_mx`: codes, the E8M0
 * plane and that plane again as the per-row reference), declared for M up to min(rows, 16):
 * libr4d's gated_gemv serves 1..8 rows and gated_gemv_wide 9..64, each byte-identical to its band's
 * GEMM row (gemv, gemv_wide) followed by gated_quant_fp8. 16, not the wide row's 64: Qwen3.8-27B
 * at TP2 on 2x R9700 (spec 7, ms a decode step, three interleaved rounds against the 1..8 band)
 * gains at 16 rows (two sequences: 1k 20.96 -> 20.78, 16k 21.96 -> 21.70) but loses at 32 (four
 * sequences at 1k 23.97 -> 24.10, eight in the short block 24.99 -> 25.14), and 64 lost more
 * (16k x16 +2.1 %). The wide row refuses a short K (decode_shortk's corner), so where 1..16 does
 * not resolve the band drops to 1..8 rather than lose the fold. Optional like decl_gated_fold, and
 * asked for only when a loaded library declares the op: libref does not, so a CPU-only engine takes
 * the pair without ever naming it. */
inline rad_op decl_gated_fold_mx(RadBuilder* b, int64_t rows, int64_t N, int64_t K, rad_weight w,
                                 rad_weight ws, rad_weight wref,
                                 std::initializer_list<rad_buf> reads,
                                 std::initializer_list<rad_buf> writes, int64_t* gated_rows) {
    if (rows <= 0 || N % 256 || K % RAD_FP8_BLOCK) return 0;
    if (!rad_op_declared(b, "gemm_nt_q_gated_mx")) return 0;
    for (const int64_t band : { (int64_t)16, (int64_t)8 }) {
        const int64_t hi = rows < band ? rows : band;
        const rad_op op = rw(b, RAD_OP(b, "gemm_nt_q_gated_mx",
                                 RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                            RAD_INT("group", RAD_MX_GROUP),
                                            RAD_STR("dtype", "mxfp4a8"), RAD_STR("act", "silu")),
                                 RAD_WEIGHTS(w, ws, wref)),
                             reads, writes);
        if (op) {
            *gated_rows = hi;
            return op;
        }
        if (hi <= 8) break;
    }
    return 0;
}

/* The same fold over an IQR [2n, K] gate_up weight under the rotated int8 trunk (`gemm_nt_q_gated`
 * at i8a8h: the codes and scale views gemm_nt_q reads, the input's and the output's int8 pairs, both
 * in the Hadamard domain), declared for M up to min(rows, 16): libr4d's gemm_iqr_gated serves its
 * GEMV and 16-token tile, each byte-identical to the GEMM row of its band followed by
 * gated_had_quant_i8g. Optional like decl_gated_fold, and asked for only when a loaded library
 * declares the op at that dtype: a CPU-only engine takes the pair. */
inline rad_op decl_gated_fold_iqr(RadBuilder* b, int64_t rows, int64_t N, int64_t K, rad_weight w,
                                  rad_weight ws, std::initializer_list<rad_buf> reads,
                                  std::initializer_list<rad_buf> writes, int64_t* gated_rows) {
    if (rows <= 0 || N % 256 || K % 256) return 0;
    const int64_t hi = rows < 16 ? rows : 16;
    const rad_op op = rw(b, RAD_OP(b, "gemm_nt_q_gated",
                             RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                        RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "i8a8h"),
                                        RAD_STR("act", "silu")),
                             RAD_WEIGHTS(w, ws)),
                         reads, writes);
    if (op) *gated_rows = hi;
    return op;
}

/* ================================================== a calibration tap */
/* One tap of the GPTQ calibration (calib_dir): the Gram of a linear's bf16 input, summed over the
 * corpus in a PERSIST [K, K] f32 buffer that calib_write() saves. */
struct CalibTap {
    rad_buf h = 0;
    rad_op  op = 0;
    int64_t K = 0;

    int declare(RadBuilder* b, Names& nm, int64_t max_tok, const char* name, rad_buf a, int64_t k) {
        K = k;
        h = decl_b(b, nm.f("%s.gram", name), RAD_F32, { k, k }, RAD_BUF_PERSIST);
        if (!h) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "gram_accum",
                    RAD_PARAMS(RAD_RANGE("M", 1, max_tok), RAD_INT("n", k),
                               RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                    RAD_NOWEIGHTS),
                { a, h }, { h });
        return op ? RAD_OK : RAD_E_INVAL;
    }
    /* gram_accum takes a, a_scale?, h: no scale for a bf16 input. */
    void step(RadCtx* c, rad_buf a, int64_t T, bool live) const {
        if (op && live) RAD_ISSUE_N(c, op, T, brows(a, T), RAD_NONE, brows(h, K));
    }
};

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_FP8_H */
