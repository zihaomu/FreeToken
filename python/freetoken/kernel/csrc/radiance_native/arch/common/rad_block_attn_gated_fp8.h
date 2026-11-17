/* rad_block_attn_gated_fp8.h -- gated GQA attention with block-scaled FP8 projections.
 *
 * The architecture (interleaved q|gate projection, partial M-RoPE, separate k and v, sigmoid
 * gate) is documented in rad_block_attn_gated.h. What differs here:
 *
 *   - q|gate, k, v and o are LinearFP8: an E4M3 weight with a bf16 [N/128][K/128] scale plane,
 *     through `gemm_nt_q` at dtype fp8a8.
 *   - One input quantise feeds the three input projections (they all read the post-norm hidden
 *     state); the gate pass quantises the out-projection's input.
 *   - Per-head norms, rope, kv_store and attn_paged stay bf16.
 *   - The KV cache dtype is `g.kv_dtype`, not `g.dtype`: it is a deployment choice that selects
 *     the attention family's cache rows. The attention op and kv_store both take it, so they agree.
 *   - Optional fused prologue, QSA sparse attention and an attention-zero mode (see declare).
 *
 * The q|gate projection is [2*n_head*head_dim, n_embd] = [12288, 5120] at 27B, scale plane
 * [96, 40]. Row-sharding by head keeps whole 128-row scale blocks only because 2*head_dim = 512 is
 * a multiple of 128; LinearFP8 checks this.
 */
#ifndef RAD_BLOCK_ATTN_GATED_FP8_H
#define RAD_BLOCK_ATTN_GATED_FP8_H

#include "rad_fp8.h"

#include <cstring>

namespace rad {
namespace arch {

struct AttnGatedFP8 {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]              the residual stream */
        ActFP8  h{};        /* [max_tok, n_embd]              normed input, then the o output */
        rad_buf qg   = 0;   /* [max_tok, n_head, 2*head_dim]  query and gate, interleaved */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf q    = 0;   /* [max_tok, q_dim]   de-interleaved, normed, rotated */
        /* [rope_table_rows, rot] f32 cos/sin table, shared by every layer, filled once a step. Zero
         * if the plugin builds none; the fused prologue is then not requested. */
        rad_buf cos_sin = 0;
        ActFP8  attn{};     /* [max_tok, q_dim]   attention output, then gated in place */
        /* [max_tok, q_dim] de-interleaved sigmoid gate, used only with bf16 linears
         * (Geom::w_bf16), where the gate is `sigmoid` then `mul`; the fused gate quantiser always
         * writes codes. */
        rad_buf gate = 0;
    };

    struct Src {
        const char* norm   = nullptr;
        const char* qg     = nullptr;   /* self_attn.q_proj.weight -- query AND gate */
        const char* k      = nullptr;
        const char* v      = nullptr;
        const char* q_norm = nullptr;
        const char* k_norm = nullptr;
        const char* o      = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    int64_t     n_rot = 0;
    /* The KV page in tokens; 0 means RAD_KV_BLOCK (16). Qwen4-Exp's QSA selects 4-token blocks,
     * and a page of 4 turns the selected set into a paged block table, so no new attention kernel
     * is needed (docs/QSA.md). The attention kernel's rows must carry a matching
     * C_EQ("block_size", ...) or nothing resolves. */
    int64_t     kv_block = 0;
    /* The QSA selection, when this layer has an indexer (produced by arch/common/rad_qsa.h; see
     * docs/QSA.md). On a sparse step the attention is issued with these in place of each row's
     * block table and context length: same op, same kernel, different list. `sel` is `topk + 1`
     * wide and the kernel takes its block-table pitch from the operand. */
    rad_buf     qsa_sel  = 0;
    rad_buf     qsa_nsel = 0;   /* [max_tok] i32, the valid entries of each `qsa_sel` row */
    rad_buf     qsa_sequ = 0;
    /* Context length up to which dense attention gives the same result, `budget + ratio - 1`.
     * Below it every complete block is selected, the paths agree to the bit, and dense is cheaper
     * at prefill (it reads each key once per query tile, sparse reads a table per query). Zero
     * means always sparse. */
    int64_t     qsa_exact_to = 0;
    /* `sel`'s row pitch is `topk + 1`; the extra slot is the partial page
     * (r4d_qsa_select_bf16.hip). */
    int64_t     qsa_topk = 0;
    /* Query rows per sparse attention launch, a scratch budget. The decode kernel's split-KV
     * partials are sized at declare from `max_seqs` and `q_len`, so issuing more rows than
     * `max_seqs` means declaring a larger value, which grows the scratch arena (the maximum over
     * ops) by `rows * q_len_band * n_head * (head_dim + 2) * 4` bytes: about 32 MiB at 256 rows on
     * this model, 63 MiB at 512, 253 MiB at 2048. With max_seqs alone, `--max-num-seqs 4` would
     * split a 2048-token prefill chunk into 512 launches per layer; 256 makes it 8, past which
     * launch cost no longer matters. */
    static constexpr int64_t kQsaRows = 256;
    int64_t qsa_rows() const { return g.max_seqs > kQsaRows ? g.max_seqs : kQsaRows; }
    /* The gated form takes the whole step in one launch: its band is q_len 1, so partials are only
     * `rows * n_head * (2 * head_dim + 8)` bytes (12.8 MiB for 2048 rows on this model). One
     * 2048-row launch is about 2.16 ms per layer versus 3.0 ms for eight 256-row launches, each
     * with its own tail. */
    int64_t qsa_gq_rows() const { return g.max_tok > qsa_rows() ? g.max_tok : qsa_rows(); }
    /* The rotation's pairing. Halves (NeoX, HF's rotate_half over the partial rotary width) and
     * adjacent pairs (GPT-J) are both self-consistent and pass every per-op check, so it must be
     * stated. "mrope" over one position component per token is NeoX (rad_block_attn_gated.h).
     *
     * The q|gate order is fixed as well: q is the first half of each head's pair, the gate the
     * second (as vLLM chunks q_gate). The unfused q norm reads column 0 of each head, the gate
     * quantiser column head_dim, and the fused prologue splits the same way. A swap would pass
     * silently, since q_norm and the sigmoid accept either half. */
    const char* mode  = "mrope";

    rad_weight w_norm = 0, w_q_norm = 0, w_k_norm = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    LinearFP8  qg{}, kp{}, vp{}, o{};
    /* Set by the architecture after declare: a `prefetch` of the out projection's leading bytes
     * (decl_prefetch, rad_arch.h), issued behind the q, k and v projections on decode steps
     * (prefetch_beside). 0 where none is declared. */
    rad_op     op_pf_o = 0;
    NormQuantFP8 nq_h{};

    rad_op     op_fused = 0;                                 /* qk_norm_rope_gate, if it resolves */
    rad_op     op_fused_kv = 0;                              /* the same with kv_store folded in */
    /* The same with the streamed store folded in, issued instead when the batch hands this group a
     * stream (its mirror and map are the batch's). */
    rad_op     op_fused_stream = 0;
    int64_t    fused_rows = 0;                               /* widest T the fused ops take */
    rad_op     op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op     op_attn = 0, op_kv_store = 0, op_gq = 0;
    rad_op     op_attn_gq = 0;          /* attn_paged + gate_quant_fp8 or gate_had_quant_i8g */
    /* kv_store_streamed: the host mirror always, the ring slot when `res` says the block is resident
     * (docs/OPS.md). Zero when no row serves it; a streamed group then fails the step. */
    rad_op     op_kv_stream = 0;
    /* kv_resolve: this layer's selection made resident and rewritten as ring slots, issued before
     * each sparse launch. */
    rad_op     op_resolve = 0;
    /* The output gate as `sigmoid` then `mul`: the bf16-linear form (Geom::w_bf16), and with fp8
     * linears the chain gate_quant_fp8 fuses, where it is not served. `gate` is the sigmoid's
     * output (the wire's with bf16 linears, the block's own otherwise) and `q_gate` the chain's
     * quantiser (no op with bf16 linears). */
    rad_op     op_sig = 0, op_mul = 0;
    rad_buf    gate = 0;
    QuantFP8   q_gate{};
    rad_op     op_ar = 0, op_add = 0;
    rad_op     op_fill = 0;                                  /* the attention-zero rank: see declare */
    /* The next block absorbs this block's closing all-reduce, so this one must not issue it.
     * Counterpart of `ar_in` (this block's norm absorbs the predecessor's); separate because a
     * block is declared without knowing what follows. Both sides ask ar_fused_ok(g, T), so they
     * agree on whether the collective was issued. */
    bool ar_out = false;
    /* Which wires that consumer takes (ArOut, rad_fp8.h): the norm's by default, or a gated
     * residual's own (HyperConn::ar_take). Set before declare; both sides then ask
     * ar_taken(g, T, ar_out, ar_out_take). */
    int  ar_out_take = kArOutNorm;

    /* `fold`: the previous block left its delta in `w.h.x` for this norm to add. `add_out`: the
     * next block will not add, so this one issues the trailing add. See rad_fp8.h. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    /* Stores the pass's K and V exactly as step() does, with no query, attention or output. Used
     * by passes that only leave layer state behind, such as the MTP head's history pass over a
     * prompt chunk with no drafting row. */
    void store_kv(RadCtx* c, const RadBatch* batch) const;
    /* The layer's cache store, before its resolve and attention: `kv_store`, or for a streamed
     * group the stage wait and `kv_store_streamed`. */
    void store_cache(RadCtx* c, const RadKVGroupBatch* kvb, const int32_t* slot, int64_t T) const;
    /* A staged pass reads the pool at block-id identity, while the streamed store writes only
     * where the residency map names a ring slot (none before the group's first decode). This
     * stores the same bytes at identity, after the mirror writer. */
    void store_stage_identity(RadCtx* c, const int32_t* slot, int64_t T) const;

    /* The block's tail, declared last by every rank: the all-reduce and the trailing add. */
    int  declare_out(RadBuilder* b, bool add_out);
    /* `own_ar` false: the caller issues the all-reduce itself (GdnFP8::step says when). */
    void step(RadCtx* c, const RadBatch* batch, bool own_ar = true,
              ArPieces* pieces = nullptr) const;
    bool issues_ar(int64_t T) const { return op_ar && !ar_taken(g, T, ar_out, ar_out_take); }
    void issue_ar(RadCtx* c, int64_t r0, int64_t rows) const {
        RAD_ISSUE_N(c, op_ar_ov ? op_ar_ov : op_ar, rows * g.n_embd,
                    brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    }
    /* Set before declare: the `overlap` issue_ar's collective is declared at, 0 for none. */
    int    ar_overlap = 0;
    rad_op op_ar_ov = 0;

    /* Whether step() with `own_ar` false runs the dense path a piece at a time over a caller's
     * normed pieces (ArPieces::in_ready): one prefill sequence in two pieces, each the chunk a
     * span of 1 makes, on the unfused prologue with one rotary component per token (every token's
     * components agree, so plane 0 of an M-RoPE batch's positions is each token's position) and
     * the gate quantised by gate_quant_fp8. A streamed group never takes it: the pieces path's
     * plain store never writes the host mirror. */
    bool takes_pieces(const RadBatch* batch, const ArPieces* pieces) const {
        const int64_t T = batch->n_tok;
        const RadKVGroupBatch* kvb = kv_batch(batch, kv);
        return pieces && pieces->in_ready && pieces->cu && pieces->cuts(T) &&
               T <= 2 * pieces->rows && !op_add && !op_fill && op_gq && !op_sig && !qsa_sel &&
               batch->n_seq == 1 && batch->phase == RAD_PHASE_PREFILL &&
               !(T <= fused_rows) && batch->positions && !rope_mixed(batch) &&
               kvb && !kvb->stream && kvb->slot_mapping && kvb->block_table && kvb->seqused;
    }
};

inline int AttnGatedFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                 rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                 const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();
    const int64_t qg_dim = 2 * q_dim;

    /* Row spans, in the container's unsharded rows, used where the even shard is the wrong cut.
     *
     * Above the KV head count geom_from replicates KV heads (rad_arch.h): with 2 KV heads at
     * head_dim 256 and 4 ranks, the k/v entry is 512 rows and ranks 0-1 take [0, 256), ranks 2-3
     * [256, 512), where the even shard would hand rank 2 half a head. The scale plane takes the
     * same span in 128-blocks; kv_off is a multiple of head_dim, so it never cuts a block. At
     * world <= n_head_kv_all the span stays 0/0 and the even shard applies.
     *
     * Under a smaller attention world (Geom::attn_world) q|gate and the output also take this
     * rank's heads [attn_rank * n_head, +n_head) as a span. An attention-zero rank declares none
     * of them (see the `fill` below). */
    const bool    a_spanned  = g.attn_world != g.world;
    const bool    kv_spanned = g.world > g.n_head_kv_all || a_spanned;
    const int64_t kv_span_lo = kv_spanned ? g.kv_off : 0;
    const int64_t kv_span_hi = kv_spanned ? g.kv_off + kv_dim : 0;
    const int64_t qg_span_lo = a_spanned ? g.attn_rank * qg_dim : 0;
    const int64_t qg_span_hi = a_spanned ? (g.attn_rank + 1) * qg_dim : 0;
    const int64_t o_span_lo  = a_spanned ? g.attn_rank * q_dim : 0;
    const int64_t o_span_hi  = a_spanned ? (g.attn_rank + 1) * q_dim : 0;

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* `src.norm == nullptr`: the architecture prepares the input. A pre-norm block owns its norm
     * (rad_fp8.h's NormQuantFP8: add the previous delta, normalise, quantise). In a gated-residual
     * model, rad_block_hc.h's `hc_read` produces the block input from a 4x wider stream, so this
     * block takes `wire.h` as already normed and quantised and declares no norm weight.
     *
     * `ar_in` is refused in that mode: the fused all-reduce lives inside the norm, so it would be
     * silently dropped. The caller emits a standalone collective instead (`add_out`/`ar_out`). */
    ext_in = (src.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
        if (g.w_bf16 && !w.gate) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* One norm+quantise op feeds all three input projections (rad_fp8.h: NormQuantFP8). */
    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));

    /* An attention-zero rank holds no heads, so its output projection contributes zero to the
     * reduce. Its output wire is zeroed by one `fill` instead of running projections on zero
     * weights (and it keeps no KV pool, rad_kv_group_zero). The all-reduce and residual add run as
     * usual, and the norm above stays because a pre-norm block's `fold` and `ar_in` live in it. */
    if (g.attn_zero) {
        op_fill = rw(b, RAD_OP(b, "fill",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {}, {w.h.x});
        return declare_out(b, add_out);
    }
    RAD_ARCH_TRY(qg.declare(b, nm, g, nm.f("blk.%d.attn_qg", l), qg_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.qg }, w.h, w.qg,
                            0, {}, qg_span_lo, qg_span_hi));
    RAD_ARCH_TRY(kp.declare(b, nm, g, nm.f("blk.%d.attn_k", l), kv_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.k }, w.h, w.k,
                            0, {}, kv_span_lo, kv_span_hi));
    RAD_ARCH_TRY(vp.declare(b, nm, g, nm.f("blk.%d.attn_v", l), kv_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.v }, w.h, w.v,
                            0, {}, kv_span_lo, kv_span_hi));

    /* The fused prologue (qk norms + rope + q|gate split) is a probe; the four unfused ops below
     * are declared either way and run when it does not resolve. Its schema takes a precomputed
     * cos/sin table instead of `theta`: `rope_table` fills `w.cos_sin` once a step for every
     * layer, and without a table the fusion is not requested.
     *
     * The kernel rotates NeoX pairs and has no `mode`. M-RoPE qualifies because the table is
     * filled at one component per token, which is exact when the components agree; step() takes
     * the fused path only then.
     *
     * No `gate_out` is passed: `gate_quant_fp8` reads the gate in place in the interleaved
     * projection.
     *
     * Bit-identical to the unfused ops, asserted in r4d_selftest: the kernel rounds the norm output
     * to bf16 before rotating and reproduces `r4d_rmsnorm_bf16`'s sum-of-squares order element by
     * element. A ULP-level difference would shift the engine's transcript within a few dozen
     * steps. */
    const bool rope_neox = !std::strcmp(mode, "neox") || !std::strcmp(mode, "mrope") || g.rope_mc;
    fused_rows = qk_fuse_rows(g);
    if (w.cos_sin && rope_neox && fused_rows > 0 && rad_op_declared(b, "qk_norm_rope_gate"))
        op_fused = rw(b, RAD_OP(b, "qk_norm_rope_gate",
                         RAD_PARAMS(RAD_RANGE("M", 1, fused_rows), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head", g.n_head),
                                    RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                    RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1)),
                         RAD_WEIGHTS(w_q_norm, w_k_norm)),
                     {w.qg, w.k, w.cos_sin}, {w.q, w.k});

    op_q_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_q_norm)),
                   {w.qg}, {w.q});
    op_k_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_k_norm)),
                   {w.k}, {w.k});
    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, mode),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, mode),
                   {w.k}, {w.k});

    /* Declared before kv_store, which reads back the resolved `block_size`
     * (rad_block_attn_gated.h). */
    op_attn = rad_decl_op(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK),
                                /* The sparse path issues one row per query, so the split-KV
                                 * scratch is sized for the widest launch (kQsaRows). */
                                RAD_INT("max_seqs", (qsa_sel && qsa_sequ) ? qsa_rows()
                                                                          : g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS);
    /* The QSA selection buffers must be declared as reads of this op. Otherwise their last use is
     * `qsa_select`, the arena may overlay another buffer, and the attention reads garbage page
     * ids: an aperture violation outside the cache. rad_op_reads rather than rw() because the list
     * depends on whether the layer has an indexer. */
    if (op_attn) {
        const rad_buf rd[4] = { w.q, qsa_sel, qsa_sequ, qsa_nsel };
        rad_op_reads(b, op_attn, rd,
                     (qsa_sel && qsa_sequ) ? (qsa_nsel ? 4 : 3) : 1);
        rad_op_writes(b, op_attn, &w.attn.x, 1);
    }

    /* The resolve belongs here rather than in the indexer: it must follow this layer's store,
     * since a block copied in from the mirror has to carry this step's tokens. Its band is the
     * step's widest launch and its row `qsa_topk + 1` wide, as the attention declares. */
    if (qsa_sel && qsa_nsel)
        op_resolve = kv_resolve_declare(b, g, kv, qsa_sel, qsa_nsel, qsa_topk + 1);

    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* The streamed store: kv_store's operands and parameters plus the residency map and the layer's
     * mirror row, which the batch carries (store_cache). `block_size` does not bind the group;
     * `attn_paged` already has. */
    if (rad_op_declared(b, "kv_store_streamed"))
        op_kv_stream = rw(b, RAD_OP(b, "kv_store_streamed",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                        RAD_INT("head_dim", g.head_dim),
                                        RAD_INT("n_head_kv", g.n_head_kv),
                                        RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                        RAD_STR("kv_dtype", g.kv_dtype)),
                             RAD_NOWEIGHTS),
                         {w.k, w.v}, {});

    /* The fused prologue with the cache store folded in: a k head's block already holds its
     * token's rotated k, so it also writes k and v into the cache, byte-identical to kv_store.
     * Declared here because the cache block size is read back from the attention op above.
     * Resolves for either cache. */
    if (op_fused && rad_op_declared(b, "qk_norm_rope_gate_kv_store"))
        op_fused_kv = rw(b, RAD_OP(b, "qk_norm_rope_gate_kv_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, fused_rows),
                                       RAD_INT("head_dim", g.head_dim), RAD_INT("n_head", g.n_head),
                                       RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                       RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                       RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_WEIGHTS(w_q_norm, w_k_norm)),
                         {w.qg, w.k, w.cos_sin, w.v}, {w.q, w.k});

    /* The fold for a streamed group: also writes the mirror and consults the residency map, whose
     * operands are added at issue. */
    if (op_fused && rad_op_declared(b, "qk_norm_rope_gate_kv_store_streamed"))
        op_fused_stream = rw(b, RAD_OP(b, "qk_norm_rope_gate_kv_store_streamed",
                            RAD_PARAMS(RAD_RANGE("M", 1, fused_rows),
                                       RAD_INT("head_dim", g.head_dim), RAD_INT("n_head", g.n_head),
                                       RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                       RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                       RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_WEIGHTS(w_q_norm, w_k_norm)),
                         {w.qg, w.k, w.cos_sin, w.v}, {w.q, w.k});

    /* Output gate, multiply and quantiser in one pass (`gate_quant_fp8`); at decode the dispatch
     * gaps between three small ops are comparable to the work. Each row is a (token, head) pair:
     * the gate is a fixed stride inside [q|gate], the attention output is contiguous, and the
     * [T * n_head, head_dim / 128] scale plane is byte-identical to the [T, q_dim / 128] the out
     * projection reads.
     *
     * The bf16 product is still written to `w.attn.x` and the codes are taken over it
     * (r4d_fused_quant_fp8.hip says why). The codes operand's dtype picks the quantiser, int8 when
     * the out projection is fed int8 (ActFP8::q8_fed). Serves prefill and decode: there is no
     * cross-row reduction.
     *
     * With bf16 linears (Geom::w_bf16) the gate is the plain `sigmoid` + `mul` pair instead.
     *
     * Where no loaded library declares gate_quant_fp8, or it does not resolve, the fp8 block runs
     * the chain it fuses: the gate into a buffer of its own, the multiply, then the activation
     * quantiser, which write what the fused op writes. A head of whole 128-column blocks makes
     * the quantiser's [T, q_dim / 128] scale plane the fused op's [T * n_head, head_dim / 128]. */
    auto gate_pair = [&](rad_buf out) {
        gate = out;
        op_sig = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg}, {gate});
        op_mul = rw(b, RAD_OP(b, "mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.attn.x, gate}, {w.attn.x});
        return op_sig && op_mul ? RAD_OK : RAD_E_INVAL;
    };
    if (g.w_bf16) {
        RAD_ARCH_TRY(gate_pair(w.gate));
    } else {
        /* Rotated int8 codes (ActFP8::q8_rot): gate_had_quant_i8g, had_quant_act_i8g's codes. */
        const char* gq = w.attn.q8_rot ? "gate_had_quant_i8g" : "gate_quant_fp8";
        if (rad_op_declared(b, gq))
            op_gq = rw(b, RAD_OP(b, gq,
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                      RAD_INT("n", g.head_dim), RAD_INT("group", w.attn.cg()),
                                      RAD_STR("dtype", g.dtype), RAD_STR("act", "sigmoid")),
                           RAD_NOWEIGHTS),
                       {w.qg, w.attn.x}, {w.attn.cq(), w.attn.cs(), w.attn.x});
        if (!op_gq && g.head_dim % RAD_FP8_BLOCK == 0) {
            RAD_ARCH_TRY(gate_pair(decl_b(b, nm.f("blk.%d.attn_gate", l), g.act_dtype,
                                          {g.max_tok, q_dim})));
            RAD_ARCH_TRY(q_gate.declare(b, g, w.attn, q_dim));
            if (!q_gate.op) return RAD_E_INVAL;
        }
    }

    /* On the sparse path the gate is applied inside the attention's split-KV merge: each launch
     * row is one (token, head) at q_len 1, and the merge already holds the head_dim values the gate
     * and quantiser read, so one op merges, gates, rounds and quantises (byte-identical to the
     * pair). Declared only with a selection: the dense path also uses the prefill kernel, which
     * has no merge. Rotated int8 codes (ActFP8::q8_rot) take attn_paged_gate_had_quant, the merge
     * with gate_had_quant_i8g's quantiser. */
    const char* attn_gq = w.attn.q8_rot ? "attn_paged_gate_had_quant" : "attn_paged_gate_quant";
    if (op_attn && qsa_sel && qsa_sequ && !g.w_bf16 && w.attn.cg() == RAD_FP8_BLOCK &&
        rad_op_declared(b, attn_gq)) {
        op_attn_gq = RAD_OP(b, attn_gq,
                            RAD_PARAMS(RAD_INT("q_len", 1), RAD_INT("head_dim", g.head_dim),
                                       RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                       RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_INT("max_seqs", qsa_gq_rows()),
                                       RAD_INT("max_ctx", g.max_ctx),
                                       RAD_STR("q_dtype", g.dtype),
                                       RAD_STR("kv_dtype", g.kv_dtype),
                                       RAD_INT("group", RAD_FP8_BLOCK),
                                       RAD_STR("act", "sigmoid")),
                            RAD_NOWEIGHTS);
        if (op_attn_gq) {
            const rad_buf rd[4] = { w.q, qsa_sel, qsa_sequ, w.qg };
            const rad_buf wr[3] = { w.attn.x, w.attn.cq(), w.attn.cs() };
            RAD_ARCH_TRY(rad_op_reads (b, op_attn_gq, rd, 4));
            RAD_ARCH_TRY(rad_op_writes(b, op_attn_gq, wr, 3));
        }
    }

    RAD_ARCH_TRY(o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                           RAD_SHARD_COL, grp_layer(l), { src.o }, w.attn, w.h.x,
                           0, {}, o_span_lo, o_span_hi));
    return declare_out(b, add_out);
}

inline int AttnGatedFP8::declare_out(RadBuilder* b, bool add_out) {
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

inline void AttnGatedFP8::store_stage_identity(RadCtx* c, const int32_t* slot, int64_t T) const {
    RAD_ISSUE_N(c, op_kv_store, T, brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T),
                kv_cache(kv, layer));
}

inline void AttnGatedFP8::store_cache(RadCtx* c, const RadKVGroupBatch* kvb,
                                      const int32_t* slot, int64_t T) const {
    if (!kvb || !kvb->stream) {
        RAD_ISSUE(c, op_kv_store,
                  brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
        return;
    }
    /* A streamed group writes its host mirror, and the ring slot only where the block is resident
     * (docs/OPS.md "kv_store_streamed"). kv_store alone would treat a block id as a ring slot, so
     * a build without the streamed store fails the step. */
    const bool staged = kv_stage_wait(c, kv, kvb, layer);
    const KVStreamLayer L = kv_stream_layer(c, kv, layer, kvb);
    if (!op_kv_stream || !L.host || L.blocks <= 0) {
        rad_step_fail(c, "kv streaming: this layer's KV group streams but the build has no "
                         "kv_store_streamed row to write its host mirror with");
        return;
    }
    RAD_ISSUE(c, op_kv_stream,
              brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer),
              praw(L.res, RAD_I32, L.blocks), praw2(L.host, RAD_U8, L.blocks, L.bbytes));
    if (staged) store_stage_identity(c, slot, T);
}

inline void AttnGatedFP8::store_kv(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (!ext_in) nq_h.step(c, T);
    if (op_fill) return;                /* the attention-zero rank holds no heads to store */
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;
    kp.step(c, w.h, w.k, T);
    vp.step(c, w.h, w.v, T);
    RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv, brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
    RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
    store_cache(c, kvb, slot, T);
}

inline void AttnGatedFP8::step(RadCtx* c, const RadBatch* batch, bool own_ar,
                               ArPieces* pieces) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;

    /* One prefill sequence in two pieces over the caller's normed input (takes_pieces): the
     * projections, prologue, cache store, attention, gate and out projection of the first piece,
     * then the second's, so the first piece's reduction runs beside all of the second piece's
     * work. Each piece is the chunk a span of 1 makes: its rows, its positions, its slots, and an
     * attention over its own query bounds (ArPieces::cu) that sees the cache up to its own end --
     * the first piece's position after its last row, the step's seqused for the second, which
     * reads the first piece's keys from the cache as a later chunk would. */
    if (!own_ar && takes_pieces(batch, pieces)) {
        const int64_t R = pieces->rows, hd = g.head_dim, qw = g.q_dim();
        const int64_t qgw = g.n_head * 2 * hd, kvw = g.n_head_kv * hd;
        const int64_t tpitch = kvb->block_table_pitch;
        for (int64_t p = 0; p < 2; ++p) {
            const int64_t r0 = p == 0 ? 0 : R;
            const int64_t n  = p == 0 ? R : T - R;
            const int32_t* pos = (batch->rope_pos ? batch->rope_pos : batch->positions) + r0;
            qg.step(c, w.h, w.qg, T, r0, n);
            kp.step(c, w.h, w.k,  T, r0, n);
            vp.step(c, w.h, w.v,  T, r0, n);
            RAD_ISSUE_N(c, op_q_norm, n * g.n_head,
                        bcol_at(w.qg, r0, qgw, 0, hd, n), RAD_W(w_q_norm),
                        brow_slice(w.q, r0, n, qw));
            RAD_ISSUE_N(c, op_k_norm, n * g.n_head_kv,
                        brow_slice(w.k, r0, n, kvw), RAD_W(w_k_norm),
                        brow_slice(w.k, r0, n, kvw));
            RAD_ISSUE_N(c, op_rope_q, n, brow_slice(w.q, r0, n, qw), praw(pos, RAD_I32, n));
            RAD_ISSUE_N(c, op_rope_k, n, brow_slice(w.k, r0, n, kvw), praw(pos, RAD_I32, n));
            RAD_ISSUE_N(c, op_kv_store, n,
                        brow_slice(w.k, r0, n, kvw), brow_slice(w.v, r0, n, kvw),
                        praw(slot + r0, RAD_I32, n), kv_cache(kv, layer));
            RAD_ISSUE_N(c, op_attn, n,
                        brows(w.q, T), kv_cache(kv, layer), praw2(table, RAD_I32, 1, tpitch),
                        praw(p == 0 ? batch->positions + R : used, RAD_I32, 1),
                        RAD_NONE, RAD_NONE, praw(pieces->cu + p, RAD_I32, 2),
                        brows(w.attn.x, T));
            RAD_ISSUE_N(c, op_gq, n * g.n_head,
                        bcol_at(w.qg, r0, qgw, hd, hd, n), brow_slice(w.attn.x, r0, n, qw),
                        brow_slice(w.attn.cq(), r0, n, qw),
                        brow_slice(w.attn.cs(), r0, n, qw / w.attn.cg()),
                        brow_slice(w.attn.x, r0, n, qw));
            o.step(c, w.attn, w.h.x, T, r0, n);
            if (p == 0) pieces->hand(r0, n);
        }
        return;
    }

    if (!ext_in) nq_h.step(c, T);
    if (op_fill) {
        /* Attention-zero rank: zero output, then the reduce and the residual. */
        RAD_ISSUE_N(c, op_fill, T, brows(w.h.x, T));
        if (own_ar && op_ar && !ar_taken(g, T, ar_out, ar_out_take))
            RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
        if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
        return;
    }
    qg.step(c, w.h, w.qg, T);
    kp.step(c, w.h, w.k,  T);
    vp.step(c, w.h, w.v,  T);
    /* The out projection's weight, read beside the prologue and the attention (op_pf_o). */
    prefetch_beside(c, op_pf_o, o.w, T);

    /* Fused or unfused prologue. The fused kernel takes the whole [q|gate] projection and splits it
     * itself, and rotates k in place like the unfused ops. It reads one rotary component per
     * token, so a pass with media tokens (differing components) takes the unfused path with the
     * full [3, T] positions. */
    const bool fuse_ok = T <= fused_rows && !rope_mixed(batch);
    /* A streamed group folds the streamed store into the prologue instead (op_fused_stream); one
     * whose layer has no mirror row leaves the store to store_cache, which fails the step. */
    const bool streamed = kvb && kvb->stream;
    const bool fused_kv = op_fused_kv && fuse_ok && !streamed;
    const KVStreamLayer L = streamed && op_fused_stream && fuse_ok
                                ? kv_stream_layer(c, kv, layer, kvb) : KVStreamLayer{};
    const bool folded_stream = L.host && L.blocks > 0;
    if (folded_stream) {
        /* The prologue is the store, so the stage copy must land before it (kv_stage_wait). */
        const bool staged = kv_stage_wait(c, kv, kvb, layer);
        RAD_ISSUE_N(c, op_fused_stream, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE,
                    brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer),
                    praw(L.res, RAD_I32, L.blocks), praw2(L.host, RAD_U8, L.blocks, L.bbytes));
        if (staged) store_stage_identity(c, slot, T);
    } else if (fused_kv) {
        RAD_ISSUE_N(c, op_fused_kv, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE,
                    brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
    } else if (op_fused && fuse_ok) {
        RAD_ISSUE_N(c, op_fused, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE);
    } else {
        RAD_ISSUE_N(c, op_q_norm, T * g.n_head,
                    bcol(w.qg, 0, g.head_dim, T), RAD_W(w_q_norm), brows(w.q, T));
        RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv,
                    brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
        RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
    }

    if (!fused_kv && !folded_stream) store_cache(c, kvb, slot, T);

    /* Sparse attention issues one row per query. `attn_paged` indexes its block table and length
     * by row, and a row need not be a sequence, so T queries issued as T rows of q_len 1, each with
     * its own selection and `seqused`, is per-query sparse attention; at decode T is the sequence
     * count.
     *
     * Below `qsa_exact_to` the dense path gives the same result more cheaply (see the field);
     * `max_ctx_len` is a host-side bound the batch already carries.
     *
     * Launches are chunked because the split-KV scratch is sized at declare from `max_seqs` and
     * `q_len` (rows = max_seqs * q_len_max * heads); more rows are refused as RAD_E_SCRATCH at
     * issue. Rows of q_len 1 up to the declared max_seqs always fit. */
    const int64_t reach  = (int64_t)batch->max_ctx_len + batch->max_q_len;
    /* A pass on a streamed group's ring (a ring smaller than the id space, and not staged) must
     * take the sparse path: on the ring a block id is not a pool address, so the dense path would
     * read ring slots as pages. `T == n_seq` alone misses a verify pass, whose T is
     * n_seq x (1 + n_spec), on a context under qsa_exact_to. A ring that covers the context keeps
     * the dense path, where ids and slots coincide. */
    const RadKVStream* st = kvb ? kvb->stream : nullptr;
    const int64_t st_blocks =
        (st && st->block_bytes > 0) ? st->host_layer_stride / st->block_bytes : 0;
    const bool ring_addressed = st && st->slots > 0 && st_blocks > 0 && st->slots < st_blocks;
    const bool ring_bound = ring_addressed && !kv_stage_layer(c, kv, kvb, layer);
    const bool    sparse = qsa_sel && qsa_sequ &&
                           (ring_bound || T == batch->n_seq || qsa_exact_to <= 0 ||
                            reach > qsa_exact_to);
    const bool gated = sparse && op_attn_gq;
    /* A streamed group's selection, per launch: make the rows' blocks resident and rewrite `sel`
     * as ring slots, after this layer's store (a block copied in from the mirror must carry this
     * step's tokens) and before the attention. A pass on the stage view leaves `sel` as block ids
     * and never resolves, so a prefill cannot overflow the ring. */
    const KVStreamLayer res_l = op_resolve ? kv_stream_layer(c, kv, layer, kvb) : KVStreamLayer{};
    auto resolve_sel = [&](int64_t off, int64_t rows) {
        if (kv_stage_layer(c, kv, kvb, layer)) return;
        if (!res_l.host || res_l.blocks <= 0 || res_l.slots <= 0 || rows <= 0) return;
        kv_resolve_issue(c, op_resolve, res_l, brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                         brow_slice(qsa_nsel, off, rows, 1), kv, layer, rows);
    };
    /* A piece at a time where the caller takes the all-reduce that way (ArPieces). On the gated
     * sparse path every query row is its own (q_len 1) and the gate rides in the merge, so the
     * attention and out projection can finish a piece's rows and hand off its reduction while the
     * next piece's attention runs. Each piece is launched as a step of the piece's width would
     * launch it. */
    if (gated && !own_ar && pieces && pieces->cuts(T) && !op_add) {
        const int64_t chunk = qsa_gq_rows();
        const int64_t qw = g.q_dim(), hd = g.head_dim;
        for (int64_t p0 = 0; p0 < T; p0 += pieces->rows) {
            const int64_t pn = T - p0 < pieces->rows ? T - p0 : pieces->rows;
            for (int64_t off = p0; off < p0 + pn; off += chunk) {
                const int64_t rows = (p0 + pn - off) < chunk ? (p0 + pn - off) : chunk;
                resolve_sel(off, rows);
                RAD_ISSUE_N(c, op_attn_gq, 1,
                            brow_slice(w.q, off, rows, qw),
                            kv_cache(kv, layer),
                            brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                            brow_slice(qsa_sequ, off, rows, 1),
                            RAD_NONE, RAD_NONE, RAD_NONE,
                            brow_slice(w.attn.x, off, rows, qw),
                            bcol_at(w.qg, off, g.n_head * 2 * hd, hd, hd, rows),
                            brow_slice(w.attn.cq(), off, rows, qw),
                            brow_slice(w.attn.cs(), off, rows, qw / w.attn.cg()));
            }
            o.step(c, w.attn, w.h.x, T, p0, pn);
            if (p0 + pn < T) pieces->hand(p0, pn);
        }
        /* Every attention launch of this layer is issued: start the next layer's stage copy. */
        if (kvb && kvb->stream) kv_stage_next(c, kv, layer);
        return;
    }
    if (gated) {
        const int64_t chunk = qsa_gq_rows();
        const int64_t qw = g.q_dim(), hd = g.head_dim;
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            resolve_sel(off, rows);
            RAD_ISSUE_N(c, op_attn_gq, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                        brow_slice(qsa_sequ, off, rows, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE,
                        brow_slice(w.attn.x, off, rows, qw),
                        bcol_at(w.qg, off, g.n_head * 2 * hd, hd, hd, rows),
                        brow_slice(w.attn.cq(), off, rows, qw),
                        brow_slice(w.attn.cs(), off, rows, qw / w.attn.cg()));
        }
    } else if (sparse) {
        const int64_t chunk = qsa_rows();
        const int64_t qw = g.q_dim();
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            resolve_sel(off, rows);
            RAD_ISSUE_N(c, op_attn, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                        brow_slice(qsa_sequ, off, rows, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE,
                        brow_slice(w.attn.x, off, rows, qw));
        }
    } else {
        /* Below qsa_exact_to the dense path reads block ids straight from the table and never
         * resolves, so a staged pass stages this layer here (a no-op without a stage view). A
         * ring-bound pass with no selection to resolve cannot be served at all. */
        if (ring_bound) {
            rad_step_fail(c, "kv streaming: a pass on the streamed group's ring has no selection "
                             "to resolve, so its block ids are not pool addresses");
        } else if (kvb && kvb->stream) {
            (void)kv_stage_layer(c, kv, kvb, layer);
        }
        /* A mixed step issues its attention once per band, as the GDN blocks do: one launch at
         * the chunk's band would walk every decode sequence's whole context with one query-tiled
         * workgroup a KV head, the step's tail beside a long chunk. The decode half (sequences
         * [0, D), rows [0, DT)) is a prefix at its own band. The prefill half takes its
         * per-sequence arrays past the decode sequences and keeps q, out and cu at their absolute
         * rows, as the kernels index q and out by cu.
         *
         * The prefill half is issued at the op's top band, not at its own query length: a launch
         * over part of the step must be served by the query-tiled row, while a short chunk's band
         * (10 rows or fewer at gqa 6) is the split-KV decode row's, whose partials are sized at
         * declare for that band and would overflow the scratch. Kernels read exact lengths from
         * cu, so the loose band costs nothing. A chunk no longer than the verify blocks beside it
         * keeps one launch at their band: the whole step is decode-shaped then. */
        int64_t D = 0, DT = 0;
        batch_split(batch, &D, &DT);
        const int64_t pitch = kvb ? kvb->block_table_pitch : 0;
        const int32_t qp = batch->max_q_len_prefill > 0 ? batch->max_q_len_prefill
                                                        : batch->max_q_len;
        if (batch->phase == RAD_PHASE_MIXED && D > 0 && D < batch->n_seq &&
            batch->max_q_len_decode > 0 && qp > batch->max_q_len_decode) {
            const int64_t P = batch->n_seq - D;
            RAD_ISSUE_N(c, op_attn, batch->max_q_len_decode,
                        brows(w.q, DT), kv_cache(kv, layer),
                        praw2(table, RAD_I32, D, pitch), praw(used, RAD_I32, D),
                        RAD_NONE, RAD_NONE, praw(batch->cu_seqlens, RAD_I32, D + 1),
                        brows(w.attn.x, DT));
            RAD_ISSUE_N(c, op_attn, g.max_tok,
                        brows(w.q, T), kv_cache(kv, layer),
                        praw2(table ? table + D * pitch : nullptr, RAD_I32, P, pitch),
                        praw(used ? used + D : nullptr, RAD_I32, P),
                        RAD_NONE, RAD_NONE, praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                        brows(w.attn.x, T));
        } else {
            RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                        brows(w.q, T), kv_cache(kv, layer),
                        praw2(table, RAD_I32, batch->n_seq, pitch),
                        praw(used, RAD_I32, batch->n_seq),
                        RAD_NONE, RAD_NONE,
                        /* cu_seqlens: verify rows may be of different lengths. */
                        praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                        brows(w.attn.x, T));
        }
    }

    /* Every attention launch of this layer is issued: start the next layer's stage copy, which
     * crosses the link under the out projection and the next layer's projections. */
    if (kvb && kvb->stream) kv_stage_next(c, kv, layer);
    if (gated) {
        /* the sparse attention gated and quantised in its merge */
    } else if (op_sig) {
        /* The gate is the strided second half of each head's [q|gate] pair. */
        RAD_ISSUE_N(c, op_sig, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(gate, T));
        RAD_ISSUE(c, op_mul, brows(w.attn.x, T), brows(gate, T), brows(w.attn.x, T));
        q_gate.step(c, T);
    } else {
        RAD_ISSUE_N(c, op_gq, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.attn.x, T),
                    brows(w.attn.cq(), T), brows(w.attn.cs(), T), brows(w.attn.x, T));
    }

    o.step(c, w.attn, w.h.x, T);
    if (own_ar && op_ar && !ar_taken(g, T, ar_out, ar_out_take))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_GATED_FP8_H */
