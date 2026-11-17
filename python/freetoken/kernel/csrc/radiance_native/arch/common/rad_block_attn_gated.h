/* rad_block_attn_gated.h -- gated grouped-query attention, bf16 operands, paged KV.
 *
 *   rmsnorm -> q|gate gemm, k gemm, v gemm -> qk-norm + rope -> kv_store -> paged attention
 *   -> attn * sigmoid(gate) -> o gemm -> [all_reduce] -> residual
 *
 * Qwen3.5's attention, which differs from Qwen3's in three ways:
 *
 *   * `q_proj` emits the query and a per-head output gate interleaved at a pitch of 2*head_dim:
 *     head h's query at h*2*head_dim, its gate at h*2*head_dim + head_dim. Both are strided views
 *     of one buffer. libr4d's `qk_norm_rope_gate` takes the stride from the `qg` operand's row
 *     pitch (there is no `qg_pitch` parameter; pQkNormRopeGate in libr4d/r4d_rows.cpp), so the
 *     interleaved layout is declared as is.
 *   * The gate multiplies the attention output, `out = attn * sigmoid(gate)`. The config's `output_gate_type: "swish"` refers to the
 *     GDN block's gate; this one is sigmoid, as in llama.cpp (src/models/qwen35.cpp,
 *     build_layer_attn).
 *   * RoPE is partial (`partial_rotary_factor` 0.25 rotates the first 64 of 256 head dims) and is
 *     M-RoPE; see `mode` in declare().
 *
 * k and v are separate projections (two GEMMs, two movement units) so every consumer gets a
 * contiguous operand. With a fused [k|v] row the per-head k norm would be a column slice whose
 * last dimension is kv_dim, not head_dim, and a kernel reading extents from its operands (libref
 * does) would normalise across the k/v seam. libr4d's kernel takes a `k_pitch` for the fused
 * form, so a quantised sibling can fuse without changing this block's shape.
 */
#ifndef RAD_BLOCK_ATTN_GATED_H
#define RAD_BLOCK_ATTN_GATED_H

#include "rad_arch.h"

namespace rad {
namespace arch {

struct AttnGatedBF16 {
    /* Activation buffers, declared once by the plugin and shared by every layer of this kind
     * (spec §3.1); liveness over the declared read/write sets makes the sharing safe. */
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]              the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd]              normed input, then the o output */
        rad_buf qg   = 0;   /* [max_tok, n_head, 2*head_dim]  query and gate, interleaved */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf q    = 0;   /* [max_tok, q_dim]   de-interleaved, normed, rotated */
        rad_buf gate = 0;   /* [max_tok, q_dim]   sigmoid of the de-interleaved gate */
        rad_buf attn = 0;   /* [max_tok, q_dim]   attention output, then gated in place */
    };

    /* Checkpoint-side names for one layer, supplied by the plugin, which owns the map (spec §2.4):
     * `model.language_model.layers.N.*` from safetensors, `blk.N.*` from GGUF. */
    struct Src {
        const char* norm   = nullptr;   /* input_layernorm.weight */
        const char* qg     = nullptr;   /* self_attn.q_proj.weight -- query AND gate */
        const char* k      = nullptr;
        const char* v      = nullptr;
        const char* q_norm = nullptr;
        const char* k_norm = nullptr;
        const char* o      = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer   = 0;
    rad_kvgroup kv      = 0;
    int64_t     n_rot   = 0;          /* partial rotary: the leading n_rot of head_dim */
    const char* mode    = "mrope";

    rad_weight w_norm = 0, w_qg = 0, w_k = 0, w_v = 0, w_q_norm = 0, w_k_norm = 0, w_o = 0;

    rad_op op_norm = 0, op_qg = 0, op_kp = 0, op_vp = 0;
    rad_op op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op op_attn = 0, op_kv_store = 0, op_sigmoid = 0, op_gate = 0;
    rad_op op_o = 0, op_ar = 0, op_add = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnGatedBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                  rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                  const Src& src) {
    g = geom; w = wire; layer = l; kv = group; n_rot = rotary_dim;

    const int64_t q_dim   = g.q_dim();
    const int64_t kv_dim  = g.kv_dim();
    const int64_t qg_dim  = 2 * q_dim;          /* query and gate, one per head, interleaved */

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* KV-head replication is not supported here. Above the KV head count geom_from gives each rank
     * one KV head and a `kv_off` row span, but this block declares k and v with the plain even
     * row shard, which would hand a rank part of one head and compute wrong values. The span is
     * passed in rad_block_attn_gated_fp8.h. */
    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: %s declares no KV row span and cannot serve world_size = %d "
                        "with n_head_kv = %lld; the KV-head replication in geom_from needs the "
                        "span passed to its k/v weights.\n", __func__, g.world,
                (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }

    /* ---- weights. Shapes are [out, in]: gemm_nt multiplies A[M,K] by B[N,K] transposed, the
     * layout checkpoints already use.
     *
     * The q|gate projection row-shards by head because each head's query and gate are adjacent:
     * rows [h*2*head_dim, (h+1)*2*head_dim) are one head's pair, so a rank's heads form a
     * contiguous row range. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),   src.norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_qg.weight", l),     src.qg));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k.weight", l),      src.k));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_v.weight", l),      src.v));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_output.weight", l), src.o));

    w_norm   = decl_w(b, nm.f("blk.%d.attn_norm.weight", l),   RAD_F32,  {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_qg     = decl_w(b, nm.f("blk.%d.attn_qg.weight", l),     RAD_BF16, {qg_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_k      = decl_w(b, nm.f("blk.%d.attn_k.weight", l),      RAD_BF16, {kv_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    w_v      = decl_w(b, nm.f("blk.%d.attn_v.weight", l),      RAD_BF16, {kv_dim, g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l));
    /* One gain vector shared by every head, so replicated across ranks. */
    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32,  {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32,  {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_o      = decl_w(b, nm.f("blk.%d.attn_output.weight", l), RAD_BF16, {g.n_embd, q_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    /* ---- ops, in run order except attn_paged (see there). Each states its read and write sets,
     * from which the buffer planner computes liveness. */
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    op_qg = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", qg_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_qg)),
               {w.h}, {w.qg});
    op_kp = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_k)),
               {w.h}, {w.k});
    op_vp = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_v)),
               {w.h}, {w.v});

    /* The fused qk_norm_rope_gate is not declared: libr4d's version (libr4d/r4d_rows.cpp,
     * pQkNormRopeGate) takes a precomputed cos/sin table operand, and this block has no PERSIST
     * [max_ctx, rot] table to give it. The unfused sequence below runs everywhere, including
     * without libr4d.
     *
     * `mode` is "mrope" (mrope_section [11,11,10], interleaved). Each rotary dimension has its own
     * frequency and the section only chooses which position component feeds it, so with equal
     * components (text) mrope is bit-identical to neox.
     *
     * Per-head RMS norm is a rmsnorm with M = tokens x heads, n = head_dim. The q norm reads the
     * strided view of the interleaved projection and writes the contiguous q buffer, so it
     * de-interleaves in the same pass. The k norm runs in place, which is safe because a row's
     * scale is computed before the row is written. */
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
    /* Two rope calls because q and k are separate buffers. `rope` describes a fused [q|k|v] row by
     * n_head and n_head_kv; n_head_kv 0 makes it a q-only (or k-only) row. */
    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, mode),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, mode),
                   {w.k}, {w.k});

    /* attn_paged is declared before kv_store although it runs after it. `block_size` is
     * RAD_DERIVED, a property of the resolved attention kernel (spec §7.2): the selector writes the
     * winner's value into the resolved geometry and kv_store reads it with rad_kv_block_size().
     * The ops touch disjoint buffers, so the order mismatch does not affect buffer planning.
     *
     * The ranged parameter is q_len, not M: 1 at decode, up to n_spec+1 at a verify, thousands in
     * a prefill chunk, with a libr4d kernel per band. `scale` defaults to 1/sqrt(head_dim). */
    op_attn = rw(b, rad_decl_op(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                /* Bounds for the kernel's scratch hook: libr4d sizes a split-KV
                                 * partial buffer of num_seqs * q_len * q_heads * splits rows. */
                                RAD_INT("n_head", g.n_head), RAD_INT("block_size", RAD_KV_BLOCK), RAD_INT("max_seqs", g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.attn});

    /* kv_store writes the group's pool, which is not a declared buffer, so its write set is
     * empty. */
    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* The gate, as two ops: the op vocabulary has no gated multiply with a bf16 output
     * (`gated_had_quant_i8` is the same arithmetic with an int8 epilogue). The sigmoid reads the
     * strided gate view of the interleaved projection into the contiguous gate buffer. */
    op_sigmoid = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg}, {w.gate});

    op_gate = rw(b, RAD_OP(b, "mul",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.attn, w.gate}, {w.attn});

    op_o = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                             RAD_INT("K", q_dim), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_o)),
              {w.attn}, {w.h});

    /* The column-sharded output projection leaves a partial sum on each rank. The plugin emits the
     * collective itself, so --debug-graph shows it here (spec §9). `exact` is always stated, so a
     * deployment that has not accepted a lossy wire gets the exact kernel. `hops` (one-shot vs
     * two-shot) is left to the kernel library, by message size. */
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

inline void AttnGatedBF16::step(RadCtx* c, const RadBatch* batch) const {
    /* Issue statuses are ignored: rad_issue records a failure and the core aborts the step and
     * fails its requests, naming the kernel and geometry (spec §17). */
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;

    RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));
    RAD_ISSUE(c, op_qg,   brows(w.h, T), RAD_W(w_qg),   brows(w.qg, T));
    RAD_ISSUE(c, op_kp,   brows(w.h, T), RAD_W(w_k),    brows(w.k, T));
    RAD_ISSUE(c, op_vp,   brows(w.h, T), RAD_W(w_v),    brows(w.v, T));

    /* The q view of the interleaved projection: head_dim columns of every 2*head_dim. */
    RAD_ISSUE_N(c, op_q_norm, T * g.n_head,
                bcol(w.qg, 0, g.head_dim, T), RAD_W(w_q_norm), brows(w.q, T));
    RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv,
                brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
    RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
    RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));

    RAD_ISSUE(c, op_kv_store,
              brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));

    /* The longest query length, not the token count, selects the attention band: a verify step and
     * a prefill chunk can have the same token count at very different q_len. */
    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                /* cu_seqlens: the batch may be ragged (a prefill chunk beside decode rows). */
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn, T));

    /* out = attn * sigmoid(gate), the gate read out of the interleaved projection. */
    RAD_ISSUE_N(c, op_sigmoid, T * g.n_head,
                bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.gate, T));
    RAD_ISSUE(c, op_gate, brows(w.attn, T), brows(w.gate, T), brows(w.attn, T));

    RAD_ISSUE(c, op_o, brows(w.attn, T), RAD_W(w_o), brows(w.h, T));
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_GATED_H */
