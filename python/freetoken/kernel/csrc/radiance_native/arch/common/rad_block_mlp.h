/* rad_block_mlp.h -- one SwiGLU feed-forward block, bf16 operands.
 *
 *   rmsnorm -> gate_up gemm -> silu_mul -> down gemm -> [all_reduce] -> residual add
 *
 * gate and up are fused into one weight at convert time (spec §2.4): the map concatenates the
 * checkpoint's gate and up along dim 0 and silu_mul splits [M, 2n] back into gate * up. One GEMM
 * launch and one movement unit instead of two.
 */
#ifndef RAD_BLOCK_MLP_H
#define RAD_BLOCK_MLP_H

#include "rad_arch.h"

namespace rad {
namespace arch {

struct MlpBF16 {
    struct Wire {
        rad_buf x       = 0;   /* [max_tok, n_embd] the residual stream */
        rad_buf h       = 0;   /* [max_tok, n_embd] normed input, then the down-projection output */
        rad_buf gate_up = 0;   /* [max_tok, 2 * n_ff] */
        rad_buf ffn     = 0;   /* [max_tok, n_ff] */
    };

    /* Checkpoint-side names for one layer. The plugin supplies them because only it knows both
     * sides of the map (spec §2.4); spellings differ per checkpoint (`...layers.N.mlp.*`,
     * `blk.N.ffn_*`). */
    struct Src {
        const char* norm = nullptr;   /* post_attention_layernorm.weight */
        const char* gate = nullptr;
        const char* up   = nullptr;
        const char* down = nullptr;
    };

    Geom g{};
    Wire w{};
    int  layer = 0;

    rad_weight w_norm = 0, w_gate_up = 0, w_down = 0;
    rad_op op_norm = 0, op_gate_up = 0, op_act = 0, op_down = 0, op_ar = 0, op_add = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                 const Src& src);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int MlpBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                            const Src& src) {
    g = geom; w = wire; layer = l;

    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ffn_norm.weight", l),    src.norm));
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ffn_gate_up.weight", l), { src.gate, src.up }, 0));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ffn_down.weight", l),    src.down));

    w_norm    = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l),    RAD_F32,  {g.n_embd},
                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_gate_up = decl_w(b, nm.f("blk.%d.ffn_gate_up.weight", l), RAD_BF16, {2 * g.n_ff, g.n_embd},
                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,  grp_layer(l), 0, {g.n_ff, g.n_ff});
    w_down    = decl_w(b, nm.f("blk.%d.ffn_down.weight", l),    RAD_BF16, {g.n_embd, g.n_ff},
                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL,  grp_layer(l));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    op_gate_up = rw(b, RAD_OP(b, "gemm_nt",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 2 * g.n_ff),
                                   RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(w_gate_up)),
                    {w.h}, {w.gate_up});

    op_act = rw(b, RAD_OP(b, "silu_mul",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.gate_up}, {w.ffn});

    op_down = rw(b, RAD_OP(b, "gemm_nt",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                RAD_INT("K", g.n_ff), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_down)),
                 {w.ffn}, {w.h});

    /* Column-sharded down projection leaves a partial sum on every rank. The plugin emits the
     * reduction so the graph dump shows it (spec §9). */
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

inline void MlpBF16::step(RadCtx* c, const RadBatch* batch) const {
    /* Operands carry the step's row count, not the buffer's: kernels read extents from the
     * operands because a ranged parameter is collapsed to a band before issue (rad_arch.h). */
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, op_norm,    brows(w.x, T),       RAD_W(w_norm),    brows(w.h, T));
    RAD_ISSUE(c, op_gate_up, brows(w.h, T),       RAD_W(w_gate_up), brows(w.gate_up, T));
    RAD_ISSUE(c, op_act,     brows(w.gate_up, T), brows(w.ffn, T));
    RAD_ISSUE(c, op_down,    brows(w.ffn, T),     RAD_W(w_down),    brows(w.h, T));
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    RAD_ISSUE(c, op_add,     brows(w.x, T),       brows(w.h, T),    brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MLP_H */
