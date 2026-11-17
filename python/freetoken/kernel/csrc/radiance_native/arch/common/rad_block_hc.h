/* rad_block_hc.h -- the gated residual (hyper-connection) Qwen4-Exp uses instead of a pre-norm
 * and a residual add.
 *
 * ============================== THE RESIDUAL STREAM IS `hc` TIMES AS WIDE ==============================
 *
 * hidden_size 2560 at hc_count 4 gives a 10240-wide stream, entered as four identical copies of
 * the embedding, carried through all 48 layers, and collapsed at the end by a connection with no
 * write half. Each of the 96 blocks (attention or delta net, then feed-forward) reads a 2560 mix
 * of the four streams and writes its output back into all four; both halves are data-dependent:
 *
 *     read    N      = grouped_rmsnorm(h)    one rsqrt per 2560 stream, one 10240 gain, `1 + w`
 *             g      = sigmoid(mix_up @ silu(mix_down @ N / hc))
 *             x      = mean over the four streams of g * N            -> the block's input
 *             inj[s] = 2 * sigmoid(inject_w[s] . N / hc)              -> four scalars in (0, 2)
 *     write   h[s]  += inj[s] * y                                     for each stream s
 *
 * docs/OPS.md's `hc_read` / `hc_write` rows state the same and name what a second implementation
 * must match: the mix is a mean, each gate argument has its own `/ hc`, and the write adds into
 * the unnormed stream.
 *
 * One struct declares both halves because they share weights and buffers and must pair: each
 * write consumes the `inj` of its own read.
 *
 * Cost: the arithmetic is small (2 x 10240 x 320 + 4 x 10240 = 6.6M MACs per token, about one
 * expert pair); the cost is launches, 96 connections per step. So the read emits the fp8 form of
 * `x` itself instead of a following `quant_act_fp8`, and the write could later be folded into the
 * producing block's epilogue.
 *
 * Nothing is sharded: the residual stream is replicated (spec §9), and so are these weights
 * (6.7 MB per connection at bf16, 1.3 GB over the model). Sharding the low-rank gate would add an
 * all-reduce between `mix_down` and `mix_up`, 96 more small collectives per step.
 */
#ifndef RAD_BLOCK_HC_H
#define RAD_BLOCK_HC_H

#include "rad_fp8.h"

namespace rad {
namespace arch {

/* Initialises the wide stream with hc identical copies of the embedding, one launch per step.
 * Separate from the embedding lookup, whose output width is n_embd. The copies diverge at the
 * first write, which uses a different gain per stream. */
struct HcEnter {
    Geom    g{};
    int64_t hc = 0;
    rad_buf x = 0, h = 0;
    rad_op  op = 0;

    int declare(RadBuilder* b, const Geom& geom, int64_t hc_, rad_buf x_, rad_buf h_) {
        g = geom; hc = hc_; x = x_; h = h_;
        if (hc <= 1 || !x || !h) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "hc_enter",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                              RAD_INT("hc", hc), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {x}, {h});
        return op ? RAD_OK : RAD_E_INVAL;
    }

    void step(RadCtx* c_, int64_t T) const {
        RAD_ISSUE_N(c_, op, T, brows(x, T), brows(h, T));
    }
};

struct HyperConn {
    struct Config {
        int64_t hc      = 0;    /* streams; the reference refuses hc <= 1 and so does this */
        int64_t lowrank = 0;    /* the read gate's waist */
        /* Whether the connection has a write half. The final mixer does not: it collapses the
         * stream into the hidden state the lm_head reads. */
        int     inject  = 1;
        /* Whether the read also emits the block input's fp8 codes. Not for the final mixer, whose
         * consumer (the lm_head) reads bf16. */
        int     quant   = 1;
        /* Whether the write also performs the all-reduce of the block's output; the block then
         * skips its own on every step takes_ar() accepts. The block's partial sum is the write's
         * input buffer, so this saves a launch. */
        int     reduce  = 0;
        /* Whether the read also emits Hadamard-rotated codes of the block input, for a consumer
         * whose weight carries the rotation (MoeFP8's rotated expert plane). Produced in the
         * plain codes' finisher, so requires `quant`; the consumer declares no quantiser. */
        int     rotate  = 0;
        /* Whether the plain codes are int8 (q8/s8) rather than E4M3 (q/s), for int8 trunk linears
         * (LinearFP8::i8). The rotated codes take the same dtype (had_quant_act_i8g's beside int8,
         * had_quant_act_fp8's beside E4M3), so `Wire::rq` is declared to match. */
        int     codes_i8 = 0;
        /* The most rows one read is declared for, 0 for max_tok; a wider step reads in pieces
         * (read_pieces). The read's scratch is sized for its band (some 22 KiB a row at
         * Flash-Next's shape, on every lane with a scratch region), so a model that reads a wider
         * step in pieces anyway declares the band no wider than a piece. */
        int64_t read_rows = 0;
    };

    struct Wire {
        rad_buf h   = 0;    /* [max_tok, hc * n_embd]  the wide stream, updated in place */
        ActFP8  x{};        /* [max_tok, n_embd]       the block input; q/s unused if !quant */
        rad_buf inj = 0;    /* [max_tok, hc]           the per-branch write gains */
        rad_buf rq  = 0;    /* [max_tok, n_embd]       the rotated codes, when `rotate` */
        rad_buf rs  = 0;    /* [max_tok, n_embd/128]   and their scales */
    };

    struct Src {
        const char* norm      = nullptr;   /* hc_norm.weight,                [hc*n_embd] */
        const char* mix_down  = nullptr;   /* input_mix_weight_down.weight,  [lowrank, hc*n_embd] */
        const char* mix_up    = nullptr;   /* input_mix_weight_up.weight,    [hc*n_embd, lowrank] */
        const char* inject_w  = nullptr;   /* block_inject_weight.weight,    [hc, hc*n_embd] */
    };

    Geom   g{};
    Config c{};
    Wire   w{};

    rad_weight w_norm = 0, w_down = 0, w_up = 0, w_inj = 0;
    /* Whether the mixing matrices are E4M3 rows with a per-group scale rather than bf16 (OPS.md's
     * `mix`), read from their encoding at declare. */
    bool       mix_e4m3 = false;
    rad_op     op_read = 0, op_write = 0;
    /* The block input's own int8 pair is the rotated one (ActFP8::q8_rot, a trunk stored rotated):
     * the read writes its rotated codes there and its plain ones into `pq` / `ps`, a pair of this
     * connection's that nothing reads. */
    bool       xrot = false;
    rad_buf    pq = 0, ps = 0;
    /* The write with the all-reduce folded in; 0 when not requested or not served.
     * `op_ar_write6` is the same on the rotated 6-bit wire, declared only when a lossy payload is
     * accepted and issued on the steps ar_wire_is_exact rejects (every trunk all-reduce at a
     * prefill chunk); its payload is 2.6x smaller. */
    rad_op     op_ar_write = 0;
    rad_op     op_ar_write6 = 0;
    /* The same with the MoE gather folded in front: the block leaves its routed experts' rows
     * unsummed, and the reduction's message is their sum. Declared by declare_gather() after the
     * block, whose buffers it reads; `gather_rows` is its bound, 0 when not declared or not
     * served (see gather_taken). */
    struct GatherSrc {
        rad_buf ye = 0, ew = 0, sorted = 0, sh = 0, sg = 0;   /* sh and sg: the shared arm, or 0 */
        int64_t top_k = 0;
        int64_t rows = 0;                                      /* one pass of the block, at most */
    };
    GatherSrc  gsrc{};
    rad_op     op_ar_gather_write = 0, op_ar_gather_write6 = 0;
    int64_t    gather_rows = 0, gather_rows6 = 0;

    int64_t hn() const { return c.hc * g.n_embd; }
    /* The reducing write for this step's wire, or 0 when that wire has no fused form (then the
     * block's own collective runs before the plain write). At four ranks there is no rotated
     * form, so steps ar_exact_write_takes declines get 0. */
    rad_op ar_write(int64_t T) const {
        return ar_exact_write_takes(g, T) ? op_ar_write : op_ar_write6;
    }
    /* Whether this step's write reduces the block's output. The block asks the same of the same T
     * (ar_taken with ar_take()) before skipping its own collective. */
    bool takes_ar(int64_t T) const { return ar_write(T) != 0; }
    /* Which wires this write takes, for the block. The rotated form exists only beside the exact
     * one, so together they take every message. */
    int ar_take() const { return op_ar_write6 ? kArOutBoth : kArOutExact; }
    bool takes_gather(int64_t T) const {
        return takes_ar(T) && gather_taken(g, T, gather_rows, gather_rows6);
    }

    /* `name` is the declared prefix -- "blk.7.attn_hc" -- and `src` the checkpoint side. */
    int declare(RadBuilder* b, Names& nm, const Geom& geom, const char* name,
                const Config& cfg, const Wire& wire, const Src& src);

    int  declare_gather(RadBuilder* b, const GatherSrc& src);

    void read (RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const;
    void write(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const;
    /* The read over rows [0, T), in pieces of Config::read_rows where it is set; one read
     * otherwise. The read's down product splits K by how many row tiles it is handed
     * (hc_down_kchunk in libr4d), so one read over twice the rows sums in a different order than
     * two reads over half of them; reading a wide step in pieces computes what the narrower steps
     * did. */
    void read_pieces(RadCtx* c_, int64_t T) const {
        const int64_t piece = c.read_rows;
        if (piece <= 0 || T <= piece) { read(c_, T, 0, T); return; }
        for (int64_t r0 = 0; r0 < T; r0 += piece) read(c_, T, r0, T - r0 < piece ? T - r0 : piece);
    }
};

inline int HyperConn::declare(RadBuilder* b, Names& nm, const Geom& geom, const char* name,
                              const Config& cfg, const Wire& wire, const Src& src) {
    g = geom; c = cfg; w = wire;
    if (c.hc <= 1 || c.lowrank <= 0 || g.n_embd <= 0) {
        fprintf(stderr, "radiance: %s: hc = %lld, lowrank = %lld -- a gated residual needs hc > 1 "
                        "and a positive waist\n",
                name, (long long)c.hc, (long long)c.lowrank);
        return RAD_E_INVAL;
    }
    if (!src.norm || !src.mix_down || !src.mix_up) return RAD_E_INVAL;
    /* The inject weight and the `inj` buffer must both be present or both absent. */
    if (c.inject && (!src.inject_w || !w.inj)) {
        fprintf(stderr, "radiance: %s: inject is on but %s is missing\n", name,
                src.inject_w ? "the inj buffer" : "block_inject_weight");
        return RAD_E_INVAL;
    }
    if (!c.inject && (src.inject_w || w.inj)) {
        fprintf(stderr, "radiance: %s: inject is off, so neither block_inject_weight nor an inj "
                        "buffer may be passed\n", name);
        return RAD_E_INVAL;
    }
    if (c.quant && (c.codes_i8 ? (!w.x.q8 || !w.x.s8) : (!w.x.q || !w.x.s))) return RAD_E_INVAL;
    /* A trunk whose int8 linears are stored rotated reads the rotated codes alone: `x`'s int8 pair
     * is then that pair (ActFP8::q8_rot), and every reader of the block input -- the trunk's
     * projections and the rotated experts -- takes it. The read derives the rotated codes in its
     * plain codes' finisher, so the plain ones still have somewhere to go: a pair of this
     * connection's own that nothing reads. */
    xrot = c.quant && c.codes_i8 && w.x.q8_rot;
    if (xrot && c.rotate) {
        fprintf(stderr, "radiance: %s: the block input's own codes are the rotated ones "
                        "(ActFP8::q8_rot), so there is no second rotated pair to ask for\n", name);
        return RAD_E_INVAL;
    }
    if (c.rotate && (!c.quant || !w.rq || !w.rs)) {
        fprintf(stderr, "radiance: %s: the rotated codes need the plain ones (quant) and the "
                        "rq/rs pair on the wire\n", name);
        return RAD_E_INVAL;
    }
    if (!w.h || !w.x.x) return RAD_E_INVAL;

    const int64_t HN = hn();

    /* Maps first: weight encodings are looked up through them. */
    const char* nn = nm.f("%s_norm.weight", name);
    const char* dn = nm.f("%s_down.weight", name);
    const char* un = nm.f("%s_up.weight", name);
    const char* in = nm.f("%s_inject.weight", name);
    RAD_ARCH_TRY(map_copy(b, nn, src.norm));
    RAD_ARCH_TRY(map_copy(b, dn, src.mix_down));
    RAD_ARCH_TRY(map_copy(b, un, src.mix_up));
    if (c.inject) RAD_ARCH_TRY(map_copy(b, in, src.inject_w));

    /* The mixing matrices are bf16 or E4M3 rows with a per-group scale (lowrank 320 is 2.5 scale
     * blocks, so 128x128 tiles do not fit). Each is declared over all its planes at the codes'
     * dtype; both must share one format for the read kernel. */
    const uint32_t mdt = weight_codes_dtype(b, dn, RAD_BF16);
    if (weight_codes_dtype(b, un, RAD_BF16) != mdt || (mdt != RAD_BF16 && mdt != RAD_F8E4M3)) {
        fprintf(stderr, "radiance: %s: the mixing matrices are %s and %s; a connection reads both "
                        "as bf16 or both as E4M3 rows\n", name,
                rad_dtype_name(mdt), rad_dtype_name(weight_codes_dtype(b, un, RAD_BF16)));
        return RAD_E_DTYPE;
    }
    mix_e4m3 = mdt == RAD_F8E4M3;

    /* F32 like every norm gain in this tree. */
    w_norm = decl_w(b, nn, RAD_F32, {HN}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_down = decl_w(b, dn, mdt, {c.lowrank, HN}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                    grp_model());
    w_up   = decl_w(b, un, mdt, {HN, c.lowrank}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                    grp_model());
    if (c.inject)
        w_inj = decl_w(b, in, RAD_BF16, {c.hc, HN}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                       grp_model());
    if (!w_norm || !w_down || !w_up || (c.inject && !w_inj)) return RAD_E_INVAL;

    /* `inject` and `group` are declared as 0 when off, not omitted: kernel rows constrain them, and
     * an absent key fails every constraint.
     *
     * RAD_WEIGHTS expands to two arguments, so the weight list is built as an array. A missing
     * inject weight shortens the list; a zero weight handle would be a declaration failure. */
    const rad_weight wts[4] = { w_norm, w_down, w_up, w_inj };
    const int n_wts = c.inject ? 4 : 3;

    /* Outputs are x, inj?, codes?, scales?, rotated pair?. The write set is positional with
     * optional outputs at the tail, so an absent one shortens the list. */
    rad_buf wr[6] = { w.x.x, 0, 0, 0, 0, 0 };
    int n_wr = 1;
    if (c.inject) wr[n_wr++] = w.inj;
    if (xrot) {
        pq = decl_b(b, nm.f("%s.plain.q8", name), RAD_I8, { g.max_tok, g.n_embd });
        ps = decl_b(b, nm.f("%s.plain.s8", name), RAD_F32, { g.max_tok, fp8_blocks(g.n_embd) });
        if (!pq || !ps || w.x.g8 != RAD_FP8_BLOCK) return RAD_E_INVAL;
        wr[n_wr++] = pq;     wr[n_wr++] = ps;
        wr[n_wr++] = w.x.q8; wr[n_wr++] = w.x.s8;
    } else {
        if (c.quant)  { wr[n_wr++] = c.codes_i8 ? w.x.q8 : w.x.q; wr[n_wr++] = c.codes_i8 ? w.x.s8 : w.x.s; }
        if (c.rotate) { wr[n_wr++] = w.rq;  wr[n_wr++] = w.rs; }
    }

    op_read = RAD_OP(b, "hc_read",
                  RAD_PARAMS(RAD_RANGE("M", 1, c.read_rows > 0 && c.read_rows < g.max_tok
                                                   ? c.read_rows : g.max_tok),
                             RAD_INT("n", g.n_embd),
                             RAD_INT("hc", c.hc), RAD_INT("lowrank", c.lowrank),
                             RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype),
                             RAD_F64("wadd", g.wadd), RAD_INT("inject", c.inject ? 1 : 0),
                             RAD_INT("group", c.quant ? RAD_FP8_BLOCK : 0),
                             RAD_INT("rotate", (c.rotate || xrot) ? 1 : 0),
                             RAD_STR("mix", mix_e4m3 ? "e4m3" : "bf16")),
                  wts, n_wts);
    if (!op_read) return RAD_E_INVAL;
    {
        const rad_buf rd[1] = { w.h };
        rad_op_reads (b, op_read, rd, 1);
        rad_op_writes(b, op_read, wr, n_wr);
    }

    if (c.inject) {
        op_write = rw(b, RAD_OP(b, "hc_write",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                    RAD_INT("hc", c.hc), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {w.x.x, w.inj, w.h}, {w.h});
        if (!op_write) return RAD_E_INVAL;
        /* An unresolved handle is 0, not an error: without the fused kernel the pair runs and
         * takes_ar() always answers no. At four ranks only the exact form exists, declared over the
         * decode and verify rows. `wire` 0 is stated, not omitted, because rows constrain it. */
        const int64_t ar_rows = ar_exact_write_rows(g);
        if (c.reduce && ar_rows >= 1)
            op_ar_write = rw(b, RAD_OP(b, "ar_hc_write",
                                RAD_PARAMS(RAD_RANGE("M", 1, ar_rows), RAD_INT("n", g.n_embd),
                                           RAD_INT("hc", c.hc), RAD_INT("world_size", g.world),
                                           RAD_STR("dtype", g.dtype), RAD_INT("wire", 0)),
                                RAD_NOWEIGHTS),
                             {w.x.x, w.inj, w.h}, {w.x.x, w.h});
        /* The rotated form at two ranks only: there is no four-rank form on that wire
         * (docs/OPS.md), and an unresolvable op would be reported at startup as an unservable
         * band. */
        if (op_ar_write && !g.wire_exact && g.world == 2)
            op_ar_write6 = rw(b, RAD_OP(b, "ar_hc_write",
                                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                            RAD_INT("hc", c.hc), RAD_INT("world_size", g.world),
                                            RAD_STR("dtype", g.dtype), RAD_INT("wire", 6)),
                                 RAD_NOWEIGHTS),
                              {w.x.x, w.inj, w.h}, {w.x.x, w.h});
    }
    return RAD_OK;
}

inline int HyperConn::declare_gather(RadBuilder* b, const GatherSrc& src) {
    if (!op_ar_write || src.top_k <= 0 || !src.ye || !src.ew || !src.sorted) return RAD_OK;
    /* Two or four ranks. At four only the exact form exists (op_ar_write6 is 0). Other widths keep
     * the block's own gather and both bounds stay 0. */
    if (g.world != 2 && g.world != 4) return RAD_OK;
    const bool shared = src.sh && src.sg;
    gsrc = src;
    /* Up to one pass's rows, capped at the write's own bound (ar_exact_write_rows): prefill chunks
     * at two ranks, decode and verify steps at four. The kernel reads its slots from an LDS table
     * at decode and from an inverse table at a chunk. */
    int64_t rows = src.rows;
    const int64_t ar_rows = ar_exact_write_rows(g);
    if (rows > ar_rows) rows = ar_rows;
    if (rows < 1) return RAD_OK;
    const rad_buf rd[8] = { src.ye, src.ew, src.sorted, w.x.x, w.inj, w.h, src.sh, src.sg };
    const rad_buf wr[2] = { w.x.x, w.h };
    /* One op per wire, the rotated form only beside the rotated write. An unresolved handle is 0:
     * the block keeps its gather and that wire's bound stays 0. */
    op_ar_gather_write = RAD_OP(b, "ar_gather_hc_write",
                                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                           RAD_INT("hc", c.hc), RAD_INT("top_k", src.top_k),
                                           RAD_INT("world_size", g.world),
                                           RAD_STR("dtype", g.dtype),
                                           RAD_STR("act", shared ? "sigmoid" : "none"),
                                           RAD_INT("wire", 0)),
                                RAD_NOWEIGHTS);
    if (op_ar_gather_write) {
        RAD_ARCH_TRY(rad_op_reads (b, op_ar_gather_write, rd, shared ? 8 : 6));
        RAD_ARCH_TRY(rad_op_writes(b, op_ar_gather_write, wr, 2));
        gather_rows = rows;
    }
    if (op_ar_write6) {
        op_ar_gather_write6 = RAD_OP(b, "ar_gather_hc_write",
                                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                                RAD_INT("hc", c.hc), RAD_INT("top_k", src.top_k),
                                                RAD_INT("world_size", g.world),
                                                RAD_STR("dtype", g.dtype),
                                                RAD_STR("act", shared ? "sigmoid" : "none"),
                                                RAD_INT("wire", 6)),
                                     RAD_NOWEIGHTS);
        if (op_ar_gather_write6) {
            RAD_ARCH_TRY(rad_op_reads (b, op_ar_gather_write6, rd, shared ? 8 : 6));
            RAD_ARCH_TRY(rad_op_writes(b, op_ar_gather_write6, wr, 2));
            gather_rows6 = rows;
        }
    }
    return RAD_OK;
}

inline void HyperConn::read(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const {
    (void)T;
    const int64_t HN = hn();
    RAD_ISSUE_N(c_, op_read, rows,
                brow_slice(w.h, r0, rows, HN), RAD_W(w_norm), RAD_W(w_down), RAD_W(w_up),
                c.inject ? RAD_W(w_inj) : RAD_NONE,
                brow_slice(w.x.x, r0, rows, g.n_embd),
                c.inject ? brow_slice(w.inj, r0, rows, c.hc) : RAD_NONE,
                c.quant  ? brow_slice(xrot ? pq : c.codes_i8 ? w.x.q8 : w.x.q, r0, rows,
                                      g.n_embd) : RAD_NONE,
                c.quant  ? brow_slice(xrot ? ps : c.codes_i8 ? w.x.s8 : w.x.s, r0, rows,
                                      fp8_blocks(g.n_embd)) : RAD_NONE,
                (c.rotate || xrot) ? brow_slice(xrot ? w.x.q8 : w.rq, r0, rows, g.n_embd)
                                   : RAD_NONE,
                (c.rotate || xrot) ? brow_slice(xrot ? w.x.s8 : w.rs, r0, rows,
                                                fp8_blocks(g.n_embd)) : RAD_NONE);
}

/* ================================================================== the control-vector projection
 *
 * A control vector is one unit direction per layer (a `controlvector` GGUF, core/cvec.h). For a
 * request that takes it, a steered layer's wide stream loses its component along that layer's
 * direction right after the feed-forward's write:
 *
 *     h[s] -= scale * (h[s] . v) * v        for each of the hc streams s
 *
 * This is llama.cpp's `--cvec-mode project`. The dot is per stream, not on the stream mean, as
 * llama.cpp's `mul_mat(v, cur)` over the [n_embd, hc, tokens] view reduces dim 0 alone.
 *
 * The switch is per request: `on` is a derived buffer of one int32 per row
 * (core/sched/derive.cpp, `projection_on_rows`), so a step that mixes a projected conversation
 * with a stock one projects only the rows that asked. The kernel neither reads nor writes a zero
 * row, and a pass with no row on issues nothing (RadBatch::control_vector). `scale` is the norm
 * the file stored for the layer times the package's own scale.
 *
 * One instance per layer; `declare` emits an op only where the vector steers, so a deployment
 * without a vector declares no op and no buffer. */
struct HcProject {
    rad_op       op = 0;          /* 0: this layer is not steered (or there is no vector) */
    const float* dir = nullptr;    /* this layer's row of the device direction table */
    rad_buf      on = 0;           /* [rows] int32: 1 = project this row */
    rad_buf      h = 0;            /* the model's wide stream */
    int64_t      n = 0, hc = 0;

    bool steered() const { return op != 0; }

    /* The per-token switch, declared once per model as a derived buffer. 0 when the deployment has
     * no vector (or one of another width), which leaves every layer unsteered. */
    static rad_buf declare_mask(RadBuilder* b, const char* name, const Geom& g,
                                const RadControlVector* cv) {
        if (!b || g.n_embd <= 0 || g.max_tok <= 0) return 0;
        if (!cv || !cv->dir || cv->n_layers <= 0) return 0;
        if (cv->n_embd != g.n_embd) {
            fprintf(stderr, "radiance: %s: the control vector is %lld wide and this model's "
                            "residual streams are %lld; not steering it\n",
                    name, (long long) cv->n_embd, (long long) g.n_embd);
            return 0;
        }
        RadBufDecl d{};
        d.dtype    = RAD_I32;
        d.rank     = 1;
        d.shape[0] = g.max_tok;
        d.kind     = RAD_BUF_DERIVED;
        d.domain   = RAD_DOMAIN_DEVICE;
        d.derive   = "projection_on_rows";
        return rad_decl_buffer(b, name, &d);
    }

    /* `layer`'s projection, when the vector steers it. Called for every layer; the unsteered ones
     * (outside the file, outside the range, or no vector at all) declare nothing. */
    int declare(RadBuilder* b, const char* name, const Geom& g, int64_t layer,
                const RadControlVector* cv, rad_buf mask, rad_buf stream, int64_t hc_streams) {
        if (!b || !mask || !stream || !cv || !cv->dir || hc_streams <= 1) return RAD_OK;
        if (layer <= 0 || layer >= cv->n_layers || layer >= g.n_layer) return RAD_OK;
        const float scale = cv->scales ? cv->scales[(size_t) layer] : 0.0f;
        if (scale == 0.0f) return RAD_OK;

        n   = g.n_embd;
        hc  = hc_streams;
        on  = mask;
        h   = stream;
        dir = cv->dir + (size_t) layer * (size_t) n;

        op = rw(b, RAD_OP(b, "hc_project",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n), RAD_INT("hc", hc),
                              RAD_F64("scale", (double) scale), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {h, mask}, {h});
        if (!op) {
            fprintf(stderr, "radiance: %s: hc_project resolves to no kernel at n %lld, hc %lld -- "
                            "this rank cannot serve the control vector\n",
                    name, (long long) n, (long long) hc);
            return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    /* Rows [r0, r0 + rows) of the stream. The whole step is r0 = 0; the pipelined step issues one
     * call per piece, in the same place in the graph. Nothing is issued on a pass where no row
     * takes the vector (RadBatch::control_vector): every row would be skipped. */
    void step(RadCtx* c_, const RadBatch* batch, int64_t r0, int64_t rows) const {
        if (!op || !batch || !batch->control_vector) return;
        RAD_ISSUE_N(c_, op, rows,
                    brow_slice(h, r0, rows, hc * n),
                    praw(dir, RAD_F32, n),
                    praw((const int32_t*) rad_buf_ptr(c_, on) + r0, RAD_I32, rows));
    }
};

/* The block's output `y` is read from `w.x.x`, the buffer the read wrote the block's input to:
 * every block leaves its delta in the activation its norm produced, and the input is dead by
 * then. */
inline void HyperConn::write(RadCtx* c_, int64_t T, int64_t r0, int64_t rows) const {
    if (!op_write) return;
    if (takes_gather(T)) {
        const int64_t k = gsrc.top_k;
        const bool shared = gsrc.sh && gsrc.sg;
        const rad_op op = ar_exact_write_takes(g, T) ? op_ar_gather_write : op_ar_gather_write6;
        RAD_ISSUE_N(c_, op, rows,
                    brows(gsrc.ye, rows * k), brow_slice(gsrc.ew, r0, rows, k),
                    brows(gsrc.sorted, rows * k),
                    shared ? brow_slice(gsrc.sh, r0, rows, g.n_embd) : RAD_NONE,
                    shared ? brow_slice(gsrc.sg, r0, rows, 1) : RAD_NONE,
                    brow_slice(w.x.x, r0, rows, g.n_embd),
                    brow_slice(w.inj, r0, rows, c.hc),
                    brow_slice(w.h, r0, rows, hn()));
        return;
    }
    RAD_ISSUE_N(c_, takes_ar(T) ? ar_write(T) : op_write, rows,
                brow_slice(w.x.x, r0, rows, g.n_embd),
                brow_slice(w.inj, r0, rows, c.hc),
                brow_slice(w.h, r0, rows, hn()));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_HC_H */
