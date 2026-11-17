/* GLM-5.3's draft head: checkpoint layer 45 as a serial drafter (RAD_DRAFT_SERIAL), around the
 * one MLA + indexer + MoE layer the plugin declares and steps itself.
 *
 *   h = gather(the trunk's final hidden states, at the rows this pass continues from)
 *   e = embed(the token one position on), the encoder's rows over its placeholders
 *   x = eh_proj( concat( enorm(e), hnorm(h) ) )            the normalized embedding first
 *   x += attention(norm(x)); the proposal rows: x += moe(norm(x))        <- the plugin's layer
 *   h' = shared_head.norm(x);  logits = the trunk's lm_head over h'
 *
 * h' is both the logits input and the next round's `h`. No mHC: the layer's residual is a plain
 * add. By the sign of RadBatch::draft_pass a history pass (< 0) covers every row the trunk step
 * had, fills the draft layer's latent, index and tail for all of them and runs the MoE and the
 * head on the proposal rows only (draft_out_ids); a draft round (> 1) is one row a sequence.
 *
 * The block is declared and stepped in three parts around the layer, in this order: `inputs`
 * before the layer's attention, `join` (the residual add and the proposal rows' selection) between
 * its attention and its feed-forward, `head` after it. The embedding and lm_head are the trunk's.
 * Its names live under the `mtp.` scope (Names), its weights in layer 45's group. eh_proj
 * (4096 x 8192 bf16, 64 MiB) is replicated and read whole in every draft pass.
 *
 * The same head serves another DeepSeek-style drafter by its checkpoint layout, set before
 * declare_inputs: the residual width, the layer's checkpoint prefix and index, and the name of the
 * head's final norm (Ling-3.0: 2560, `model.layers.42`, 42, `final_layernorm.weight`). */
#ifndef RAD_BLOCK_GLM_MTP_H
#define RAD_BLOCK_GLM_MTP_H
#include "rad_glm_tp.h"
#include "rad_block_vit.h"

namespace rad {
namespace arch {
struct GlmMtp {
    /* The trunk's: its final hidden states, its logits, its embedding and head. */
    struct Wire {
        rad_buf final_hidden = 0, logits = 0;
        rad_weight token = 0, head = 0;
    };
    static constexpr int kDepth = 3;   /* --num-speculative-tokens auto */
    int64_t     width = 4096;
    const char* prefix = "model.language_model.layers.45";
    int         layer = 45;
    const char* norm_src = "shared_head.norm.weight";
    Names nm{ "mtp" };
    Geom g{};
    Wire w{};
    rad_buf x = 0, hidden = 0, embedding = 0, concat = 0, out = 0;
    rad_weight en = 0, hn = 0, eh = 0, norm = 0;
    rad_op op_prev = 0, op_embed = 0, op_embed_ar = 0, op_en = 0, op_hn = 0, op_eh = 0;
    rad_op op_residual = 0, op_select = 0, op_back = 0;
    rad_op op_norm = 0, op_copy = 0, op_head = 0;
    MediaRows mrows{};

    /* The buffers, the weights, and the ops up to the layer's input. `media`: the encoder's rows
     * replace the placeholders' embeddings, as the target's do (SGLang feeds its draft model the
     * target's image rows); without them the drafts read the placeholder's embedding and
     * acceptance falls on a prompt with a picture. */
    int declare_inputs(RadBuilder* b, const Geom& geom, const Wire& wire, bool media) {
        g = geom;
        w = wire;
        if (!b || !w.final_hidden || !w.logits || !w.token || !w.head || g.max_spec < 1)
            return RAD_E_INVAL;
        const int64_t E = width;
        x = decl_b(b, nm.f("x"), RAD_BF16, { g.max_tok, E });
        hidden = decl_b(b, nm.f("hidden"), RAD_BF16, { g.max_tok, E }, RAD_BUF_PERSIST);
        embedding = decl_b(b, nm.f("embedding"), RAD_BF16, { g.max_tok, E });
        concat = decl_b(b, nm.f("concat"), RAD_BF16, { g.max_tok, 2 * E });
        out = decl_b(b, nm.f("final"), RAD_BF16, { g.max_tok, E });
        const char* pre = prefix;
        auto gain = [&](const char* n, const char* src) {
            const char* name = nm.f("%s", n);
            if (map_copy(b, name, nm.ckpt("%s.%s", pre, src)) < 0) return (rad_weight)0;
            return decl_w(b, name, RAD_F32, { E }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                          grp_layer(layer));
        };
        en = gain("enorm.weight", "enorm.weight");
        hn = gain("hnorm.weight", "hnorm.weight");
        norm = gain("output_norm.weight", norm_src);
        RAD_ARCH_TRY(map_copy(b, nm.f("eh_proj.weight"), nm.ckpt("%s.eh_proj.weight", pre)));
        eh = decl_w(b, nm.f("eh_proj.weight"), RAD_BF16, { E, 2 * E }, RAD_ACCESS_PER_TOKEN,
                    RAD_SHARD_NONE, grp_layer(layer));
        if (!x || !hidden || !embedding || !concat || !out || !en || !hn || !norm || !eh)
            return RAD_E_INVAL;
        op_prev = rw(b,
                     RAD_OP(b, "gather_rows",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", E),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                     { w.final_hidden }, { hidden });
        op_embed = rw(b,
                      RAD_OP(b, "embed_lookup",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", E),
                                        RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                        RAD_INT("vocab_offset", g.vocab_off)),
                             RAD_WEIGHTS(w.token)),
                      {}, { embedding });
        if (media) RAD_ARCH_TRY(mrows.declare(b, g.max_tok, E, embedding));
        op_embed_ar = GlmTp::reduce(b, g, embedding, true);
        op_en = rmsnorm(b, en, embedding, concat);
        op_hn = rmsnorm(b, hn, hidden, concat);
        op_eh = rw(b,
                   RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", E),
                                     RAD_INT("K", 2 * E), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(eh)),
                   { concat }, { x });
        return op_prev && op_embed && op_en && op_hn && op_eh && (g.world == 1 || op_embed_ar)
                   ? RAD_OK
                   : RAD_E_INVAL;
    }
    /* Between the layer's attention and its feed-forward: the residual add of the attention's
     * output `y`, and on a history pass the proposal rows gathered to the head of `x`. */
    int declare_join(RadBuilder* b, rad_buf y) {
        op_residual = rw(b,
                         RAD_OP(b, "add",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", width),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                         { x, y }, { x });
        // All history rows fill the draft cache; only proposal rows need its FFN.
        op_select = rw(b,
                       RAD_OP(b, "gather_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", width),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                       { x }, { hidden });
        op_back = cast(b, hidden, x);
        return op_residual && op_select && op_back ? RAD_OK : RAD_E_INVAL;
    }
    /* After the layer: the head's norm, its output kept as the next round's hidden, the logits,
     * and the drafter's declaration. */
    int declare_head(RadBuilder* b) {
        op_norm = rmsnorm(b, norm, x, out);
        op_copy = cast(b, out, hidden);
        op_head = rw(b,
                     RAD_OP(b, "logits_gemm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                       RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", width),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w.head)),
                     { hidden }, { w.logits });
        if (!op_norm || !op_copy || !op_head) return RAD_E_INVAL;
        RadDrafterDecl dd{};
        dd.name = "mtp";
        dd.kind = RAD_DRAFT_SERIAL;
        dd.depth = g.max_spec;
        dd.mask_token = -1;
        return rad_declare_drafter(b, &dd);
    }

    /* The rows a pass runs its feed-forward and head on: the proposal rows of a history pass,
     * every row of a draft round. */
    static int64_t head_rows(const RadBatch* batch) {
        return batch->draft_pass < 0 ? batch->n_draft_out : batch->n_tok;
    }
    void step_inputs(RadCtx* c, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        if (batch->draft_pass < 0 && batch->n_out > 0)
            RAD_ISSUE_N(c, op_prev, batch->n_out, RAD_B(w.final_hidden),
                        praw(batch->out_ids, RAD_I32, batch->n_out), brows(hidden, batch->n_out));
        RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w.token),
                    brows(embedding, T));
        mrows.step(c, batch, embedding, T);
        if (op_embed_ar) RAD_ISSUE_N(c, op_embed_ar, T * width, brows(embedding, T), RAD_NONE);
        RAD_ISSUE_N(c, op_en, T, brows(embedding, T), RAD_W(en), bcol(concat, 0, width, T));
        RAD_ISSUE_N(c, op_hn, T, brows(hidden, T), RAD_W(hn), bcol(concat, width, width, T));
        RAD_ISSUE_N(c, op_eh, T, brows(concat, T), RAD_W(eh), brows(x, T));
    }
    void step_join(RadCtx* c, const RadBatch* batch, rad_buf y) const {
        const int64_t T = batch->n_tok, L = head_rows(batch);
        RAD_ISSUE_N(c, op_residual, T, brows(x, T), brows(y, T), brows(x, T));
        if (batch->draft_pass < 0) {
            RAD_ISSUE_N(c, op_select, L, RAD_B(x), praw(batch->draft_out_ids, RAD_I32, L),
                        brows(hidden, L));
            RAD_ISSUE_N(c, op_back, L, brows(hidden, L), brows(x, L));
        }
    }
    void step_head(RadCtx* c, const RadBatch* batch) const {
        const int64_t L = head_rows(batch);
        RAD_ISSUE_N(c, op_norm, L, brows(x, L), RAD_W(norm), brows(out, L));
        RAD_ISSUE_N(c, op_copy, L, brows(out, L), brows(hidden, L));
        RAD_ISSUE_N(c, op_head, L, brows(hidden, L), RAD_W(w.head), brows(w.logits, L));
    }

private:
    rad_op rmsnorm(RadBuilder* b, rad_weight gain, rad_buf in, rad_buf out) const {
        return rw(b,
                  RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", width),
                                    RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                         RAD_WEIGHTS(gain)),
                  { in }, { out });
    }
    rad_op cast(RadBuilder* b, rad_buf in, rad_buf out) const {
        return rw(b,
                  RAD_OP(b, "cast",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", width),
                                    RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                         RAD_NOWEIGHTS),
                  { in }, { out });
    }
};
}
} // namespace rad::arch
#endif
