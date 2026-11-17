/* rad_block_attn_fp8.h -- plain GQA attention with block-scaled FP8 projections.
 *
 *   rmsnorm -> quant_act_fp8 -> fused q|k|v gemm_nt_q -> rope -> kv_store -> attn_paged
 *   -> quant_act_fp8 -> o gemm_nt_q -> [all_reduce] -> residual add
 *
 * rad_block_attn_gated_fp8.h without its two family-specific features:
 *
 *   - No per-head q/k norm. Qwen3.5 norms each head before rotating; LLaMA and MiniCPM5 rotate
 *     the GEMM output directly.
 *   - No output gate. The attention output goes straight into the out-projection.
 *
 * `rotary_dim` 0 is attention without positions (NoPE, Nemotron-H's attention layers): no `rope`
 * is declared, and position reaches the model through its recurrent layers alone.
 *
 * That allows a fused q|k|v, which the gated block cannot have (its query projection is [q|gate]
 * interleaved per head). The three weights concatenate along dim 0 into one
 * [q_dim + 2*kv_dim, n_embd], and `rope` takes that row as is: q heads, then k heads, v untouched
 * (docs/OPS.md). One GEMM, one rope and one quantise.
 *
 * `row_parts` is required: a tensor-parallel rank owns a slice of each of the three projections,
 * not a contiguous slice of the stack. LinearFP8 checks that each part is a whole number of
 * 128-row scale blocks.
 *
 * The fused front op, the KV dtype and the fold flags are documented in rad_fp8.h and
 * rad_block_attn_gated_fp8.h.
 */
#ifndef RAD_BLOCK_ATTN_FP8_H
#define RAD_BLOCK_ATTN_FP8_H

#include "rad_fp8.h"

namespace rad {
namespace arch {

struct AttnFP8 {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]                   the residual stream */
        ActFP8  h{};        /* [max_tok, n_embd]                   normed input, then the o output */
        rad_buf qkv  = 0;   /* [max_tok, q_dim + 2*kv_dim]         the fused projection */
        rad_buf q    = 0;   /* [max_tok, q_dim]                    the query, CONTIGUOUS */
        ActFP8  attn{};     /* [max_tok, q_dim]                    attention output */
    };

    /* Checkpoint tensor names; the declaration is the same whether they hold block fp8 or a dense
     * tensor the recipe quantises. */
    struct Src {
        const char* norm  = nullptr;
        const char* q     = nullptr;
        const char* k     = nullptr;
        const char* v     = nullptr;
        const char* o     = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    int64_t     n_rot = 0;
    /* The rotation pairs halves (NeoX, HF's rotate_half), as LLaMA derivatives ship. Stated
     * explicitly: GPT-J's adjacent pairs would pass every per-op check too. */
    const char* mode  = "neox";

    rad_weight w_norm = 0;
    LinearFP8  qkv{}, o{};
    NormQuantFP8 nq_h{};
    QuantFP8     q_attn{};
    rad_op     op_rope = 0, op_qlift = 0, op_attn = 0, op_kv_store = 0;
    rad_op     op_ar = 0, op_add = 0;
    bool       ar_out = false;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                            int64_t rotary_dim, const Wire& wire, const Src& src,
                            bool fold, bool add_out, int ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();

    if (n_rot < 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* Above the KV head count geom_from replicates KV heads (rad_arch.h): each of the fused row's
     * three parts takes its own span (as gemma4's q|k|v does), q this rank's heads and k and v the
     * shared head at `kv_off`, where the even shard would hand a rank part of a head. A smaller
     * attention world (Geom::attn_world) is not served here. */
    const bool kv_repl = g.world > g.n_head_kv_all;
    if (kv_repl && g.attn_world != g.world) {
        fprintf(stderr, "radiance: %s replicates KV heads but declares no attention-zero ranks; "
                        "world_size = %d must be a multiple of n_head_kv = %lld.\n", __func__,
                g.world, (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* Norm and quantise in one op (rad_fp8.h: NormQuantFP8). */
    RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                              fold ? w.x : 0, 0, ar_in));
    RAD_ARCH_TRY(qkv.declare(b, nm, g, nm.f("blk.%d.attn_qkv", l), q_dim + 2 * kv_dim, g.n_embd,
                             RAD_SHARD_ROW, grp_layer(l), { src.q, src.k, src.v }, w.h, w.qkv,
                             0, { q_dim, kv_dim, kv_dim }));
    if (kv_repl) {
        const int64_t lo[3] = { g.rank * q_dim, g.kv_off, g.kv_off };
        const int64_t hi[3] = { lo[0] + q_dim, lo[1] + kv_dim, lo[2] + kv_dim };
        const int64_t tot[3] = { g.n_head_all * g.head_dim, g.n_head_kv_all * g.head_dim,
                                 g.n_head_kv_all * g.head_dim };
        RAD_ARCH_TRY(qkv.span_parts(b, lo, hi, tot, 3));
    }

    if (n_rot > 0)
        op_rope = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, g.n_head_kv, n_rot, mode),
                     {w.qkv}, {w.qkv});

    /* Copies the query out of the fused row: libr4d's paged attention reads `q` as a dense
     * [total_q, n_head, head_dim] block with no pitch argument (r4d_attn_args.h, fill_paged), so a
     * column slice with pitch q_dim + 2*kv_dim is refused at issue. One copy per layer is cheaper
     * at decode than splitting the projection (one more GEMM and one more rope). Remove it if the
     * kernels gain a q pitch. `cast` with from == to is this tree's row copy. */
    op_qlift = rw(b, RAD_OP(b, "cast",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                 RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.qkv}, {w.q});

    /* Declared before kv_store, which reads back the resolved `block_size` (rad_block_attn_gated.h
     * explains declare vs issue order). */
    op_attn = rw(b, rad_decl_op(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.attn.x});

    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.qkv}, {});

    /* The out-projection's input quantiser; a plain `quant_act_fp8`, as there is no gate pass to
     * fold it into. */
    RAD_ARCH_TRY(q_attn.declare(b, g, w.attn, q_dim));
    RAD_ARCH_TRY(o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                           RAD_SHARD_COL, grp_layer(l), { src.o }, w.attn, w.h.x));

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});

    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h.x}, {w.x});
    return RAD_OK;
}

inline void AttnFP8::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();

    nq_h.step(c, T);
    qkv.step(c, w.h, w.qkv, T);
    /* In place over the fused row: rotates n_head query heads, then n_head_kv key heads, and
     * leaves the value heads alone. */
    if (op_rope) RAD_ISSUE(c, op_rope, brows(w.qkv, T), rope_posmc(g, batch, T));

    /* k and v are column slices of the fused row, at q_dim and q_dim + kv_dim. */
    RAD_ISSUE(c, op_kv_store,
              bcol(w.qkv, q_dim, kv_dim, T), bcol(w.qkv, q_dim + kv_dim, kv_dim, T),
              praw(slot, RAD_I32, T), kv_cache(kv, layer));

    RAD_ISSUE(c, op_qlift, bcol(w.qkv, 0, q_dim, T), brows(w.q, T));

    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn.x, T));

    q_attn.step(c, T);
    o.step(c, w.attn, w.h.x, T);
    if (op_ar && !(ar_out && ar_fused_ok(g, T)))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_FP8_H */
