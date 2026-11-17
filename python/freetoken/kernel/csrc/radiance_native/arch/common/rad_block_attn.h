/* rad_block_attn.h -- one grouped-query attention block with per-head q/k RMS norms, bf16
 * operands, paged KV: Qwen3's attention.
 *
 *   rmsnorm -> q gemm, k gemm, v gemm -> q norm, k norm (per head) -> rope (NeoX)
 *   -> kv_store -> paged attention -> o gemm -> [all_reduce] -> residual
 *
 * rad_block_attn_gated.h without the output gate, and more than a flag apart: Qwen3's q_proj is
 * [n_head * head_dim, n_embd] (queries only, no interleaved gate), the rotary covers every head
 * dimension as plain NeoX over the token's position, and the norms are plain RMS norms,
 * `x_hat * w` (Qwen3RMSNorm), not Gemma's `1 + w`. Qwen3-1.7B is Qwen3-ASR's decoder
 * (arch/qwen3asr_bf16).
 *
 * q, k and v are three projections into three contiguous buffers: paged attention reads `q`
 * contiguous (libr4d's fill_paged refuses a pitched query), so a fused [q|k|v] row would need a
 * copy before attention. Each per-head norm is an ordinary `rmsnorm` with M = tokens x heads and
 * n = head_dim, and the rotary is `rope` over q and over k, which every kernel library serves.
 *
 * Positions are RadBatch::positions, one component a token. A media span in the prompt (an audio
 * clip's placeholders) counts up like text, as the reference's position ids do; this block never
 * reads the M-RoPE planes.
 */
#ifndef RAD_BLOCK_ATTN_H
#define RAD_BLOCK_ATTN_H

#include "rad_arch.h"

namespace rad {
namespace arch {

struct AttnBF16 {
    /* The activation buffers, declared once by the plugin and shared by every layer: layers that
     * share a geometry share the buffer (spec §8). */
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]  the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd]  normed input, then the o output */
        rad_buf q    = 0;   /* [max_tok, q_dim]   projected, normed, rotated */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf attn = 0;   /* [max_tok, q_dim]   attention output */
    };

    /* The checkpoint-side names for one layer; the plugin owns the spelling (spec §2.4). */
    struct Src {
        const char* norm   = nullptr;   /* input_layernorm.weight */
        const char* q      = nullptr;
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

    rad_weight w_norm = 0, w_q = 0, w_k = 0, w_v = 0, w_q_norm = 0, w_k_norm = 0, w_o = 0;
    rad_op op_norm = 0, op_q = 0, op_k = 0, op_v = 0, op_q_norm = 0, op_k_norm = 0;
    rad_op op_rope_q = 0, op_rope_k = 0, op_kv_store = 0, op_attn = 0, op_o = 0, op_ar = 0;
    rad_op op_add = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 const Wire& wire, const Src& src);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                             const Wire& wire, const Src& src) {
    g = geom; w = wire; layer = l; kv = group;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();

    /* KV-head replication is not plumbed here, for the reason rad_block_attn_gated.h gives: the
     * plain row-shard of k and v would hand a rank part of one KV head. Refused by name. */
    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: %s declares no KV row span and cannot serve world_size = %d "
                        "with n_head_kv = %lld\n", __func__, g.world, (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }

    /* ---- weights. [out, in], so nothing is transposed at load. q, k and v row-shard by head;
     * o column-shards, so each rank holds a partial sum of the residual update. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),   src.norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q.weight", l),      src.q));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k.weight", l),      src.k));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_v.weight", l),      src.v));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_output.weight", l), src.o));

    w_norm   = decl_w(b, nm.f("blk.%d.attn_norm.weight", l),   RAD_F32,  {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_q      = decl_w(b, nm.f("blk.%d.attn_q.weight", l),      RAD_BF16, {q_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_k      = decl_w(b, nm.f("blk.%d.attn_k.weight", l),      RAD_BF16, {kv_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_v      = decl_w(b, nm.f("blk.%d.attn_v.weight", l),      RAD_BF16, {kv_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    /* One gain shared by every head, replicated across ranks though the heads are not. */
    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32,  {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32,  {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_o      = decl_w(b, nm.f("blk.%d.attn_output.weight", l), RAD_BF16, {g.n_embd, q_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    auto gemm = [&](int64_t N, int64_t K, rad_weight wt) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(wt));
    };
    auto head_norm = [&](int64_t heads, rad_weight wt) {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * heads), RAD_INT("n", g.head_dim),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(wt));
    };

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});
    op_q = rw(b, gemm(q_dim,  g.n_embd, w_q), {w.h}, {w.q});
    op_k = rw(b, gemm(kv_dim, g.n_embd, w_k), {w.h}, {w.k});
    op_v = rw(b, gemm(kv_dim, g.n_embd, w_v), {w.h}, {w.v});
    /* Each norm runs in place: a row's scale is computed before the row is written. */
    op_q_norm = rw(b, head_norm(g.n_head,    w_q_norm), {w.q}, {w.q});
    op_k_norm = rw(b, head_norm(g.n_head_kv, w_k_norm), {w.k}, {w.k});
    /* Two rope calls because q and k are two buffers: a `rope` that names only q heads is a fused
     * row with an empty k and v. The full head dim rotates. */
    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, g.head_dim, "neox"),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, g.head_dim, "neox"),
                   {w.k}, {w.k});

    /* attn_paged before kv_store: block_size is RAD_DERIVED, read off the resolved attention
     * kernel and back into kv_store with rad_kv_block_size() (rad_block_attn_gated.h says why). */
    op_attn = rw(b, rad_decl_op(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.attn});
    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    op_o = rw(b, gemm(g.n_embd, q_dim, w_o), {w.attn}, {w.h});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h}, {w.x});
    return RAD_OK;
}

inline void AttnBF16::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;
    const int64_t T = batch->n_tok;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);

    RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
    RAD_ISSUE(c, op_q,    brows(w.h, T), RAD_W(w_q),    brows(w.q, T));
    RAD_ISSUE(c, op_k,    brows(w.h, T), RAD_W(w_k),    brows(w.k, T));
    RAD_ISSUE(c, op_v,    brows(w.h, T), RAD_W(w_v),    brows(w.v, T));
    RAD_ISSUE_N(c, op_q_norm, T * g.n_head,    brows(w.q, T), RAD_W(w_q_norm), brows(w.q, T));
    RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv, brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
    RAD_ISSUE(c, op_rope_q, brows(w.q, T), pos);
    RAD_ISSUE(c, op_rope_k, brows(w.k, T), pos);
    RAD_ISSUE(c, op_kv_store,
              brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
    /* The actual query length selects the band; cu_seqlens carries a ragged step. */
    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn, T));
    RAD_ISSUE(c, op_o, brows(w.attn, T), RAD_W(w_o), brows(w.h, T));
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_H */
