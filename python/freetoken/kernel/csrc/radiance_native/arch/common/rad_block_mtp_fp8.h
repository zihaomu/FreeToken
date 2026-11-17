/* rad_block_mtp_fp8.h -- the multi-token-prediction head, with block-scaled FP8 projections.
 *
 * ============================== WHAT IT IS ==============================
 *
 * Qwen3.5 / Qwen3-Next ship one extra transformer layer under an `mtp.` prefix that predicts
 * token i+2 from the trunk's hidden state at i and the embedding of token i+1. It is the
 * speculative-decoding drafter; the embedding table and lm_head are the trunk's
 * (`mtp_use_dedicated_embeddings: false`, no `mtp.lm_head`).
 *
 * Indexing:
 *
 *     head index i    reads  hidden(i) and embed(token at i + 1)
 *                     writes its K and V at KV position i, with rope position i
 *                     predicts the token at i + 2
 *
 * This is DeepSeek-V3's MTP layout: the head runs over the trunk's position indices and the shift
 * is only in which token is embedded. An off-by-one shows only as collapsed acceptance.
 *
 * The forward:
 *
 *     h = gather(the trunk's final hidden states, at the rows this pass continues from)
 *     e = embed(the token one position on)
 *     x = fc( concat( pre_fc_norm_embedding(e), pre_fc_norm_hidden(h) ) )
 *     x = one gated-GQA layer + one SwiGLU feed-forward, over x            <- AttnGatedFP8, MlpFP8
 *     h'= mtp.norm(x)
 *     logits = the trunk's lm_head over h'
 *
 * h' is both the logits input and the next round's `h`, as in the reference (a Qwen3-Next MTP
 * forward returns one tensor; an EAGLE head returns a (normed, unnormed) pair).
 *
 * ============================== THE TWO KINDS OF PASS ==============================
 *
 * By the sign of RadBatch::draft_pass:
 *
 *   < 0   History pass, one row per token, gathering the trunk's hidden states. The head's
 *         attention reads its own K and V for every earlier position, which the trunk's prefill
 *         does not write; without this pass the head attends to stale pool contents and proposes
 *         noise. The scheduler runs it over the positions the last step added, verified ones
 *         included.
 *
 *         It is also draft round 1 for every drafting sequence: its last history row is the
 *         committed position plus the token the trunk chose after it, so the pass ends in the
 *         lm_head over those rows (RadBatch::draft_out_ids) and leaves the head's output there
 *         for round 2.
 *
 *   > 1   Draft round, one row per sequence, reading the previous round's output from the same
 *         buffer and ending in the lm_head. There is no separate round-1 pass.
 *
 * ============================== PRECISION ==============================
 *
 * As stored in mtp.safetensors:
 *
 *   F8_E4M3 + scale   layers.0.self_attn.{q,k,v,o}_proj, layers.0.mlp.{gate,up,down}_proj
 *   BF16, no scale    fc, norm, pre_fc_norm_embedding, pre_fc_norm_hidden, and every norm inside
 *                     the layer
 *
 * `fc` ([5120, 10240], bf16) is a plain `gemm_nt` issued once over the whole chunk: libr4d's bf16
 * row is C_LE("M", 65536) and K = 10240 is a multiple of 16. The history pass has one row per
 * token of a prefill chunk, so slicing at 64 would cost dozens of launches.
 *
 * ============================== THE CONCAT ORDER ==============================
 *
 * `fc` reads the normed embedding and the normed hidden state concatenated. The order is not in
 * the checkpoint (both are one [5120, 10240] matrix); this follows DeepSeek-V3's MTP:
 * `enorm(embedding)` then `hnorm(hidden)`. Per-op checks cannot tell the orders apart; the draft
 * acceptance rate does (the wrong order accepts almost nothing).
 *
 * ============================== THE NORM GAINS ==============================
 *
 * GemmaRMSNorm's `1 + w` (Geom::wadd) applies: pre_fc_norm_hidden is -0.06 .. -0.24 and
 * pre_fc_norm_embedding -0.27 .. -0.61, negative gains that only `1 + w` explains. `mtp.norm`
 * (0.59 .. 1.31) is ambiguous, so it takes the trunk output norm's `final_wadd`.
 *
 * ============================== THE KV CACHE ==============================
 *
 * The head's attention is bound to the trunk's full-attention group as one more layer, at
 * absolute index n_main:
 *
 *   - It shares the trunk's slot mapping and block table, writing K and V at the slot of the
 *     trunk's token for that position. A rejected draft's entry is overwritten by the next step's
 *     correct token at the same address, and each round writes its position before attending to
 *     it, so no rollback is needed (as in vLLM's EAGLE/MTP proposer).
 *   - A second group would need its own page geometry, free list and eviction for a cache one
 *     seventeenth the size.
 *
 * The group's page grows from 16 layers to 17 (6% more attention KV), only when speculation is
 * declared.
 */
#ifndef RAD_BLOCK_MTP_FP8_H
#define RAD_BLOCK_MTP_FP8_H

#include "rad_block_attn_gated_fp8.h"
#include "rad_block_mlp_fp8.h"
#include "rad_block_vit.h"

namespace rad {
namespace arch {

/* Declares the name map of the 2-bit draft head `head` (the checkpoint lm_head `lm` under its own
 * name, quantised by a recipe to u2 codes with an f16 scale and u8 zero per group of 128) and
 * returns whether the model holds it at two bits. If not (no such recipe), draft rounds read the
 * bf16 head.
 *
 * Called once by the plugin, before it declares the head's buffers; the block infers the answer
 * from whether those buffers exist. */
inline bool mtp_draft_head_q2(RadBuilder* b, const char* head, const char* lm) {
    if (!head || !lm) return false;
    if (map_copy(b, head, lm) < 0) return false;
    RadEncoding e{};
    return weight_enc(b, head, &e) && e.n_planes > 0 && e.plane[0].dtype == RAD_U2;
}

/* Templated on the feed-forward, since the head mirrors a trunk layer, which is dense or routed
 * by model (Qwen3.6-35B-A3B's `mtp.layers.0` has `mlp.experts.*`, `mlp.gate` and a shared expert).
 * `Ffn` is MlpFP8 or MoeFP8; both have the same declare() signature over their own Wire and Src
 * (MoeFP8's Config travels inside its Src for this reason). */
template <class Ffn>
struct MtpBlockFP8 {
    struct Wire {
        /* [max_tok, n_embd] the trunk's final hidden states for the step just run (the plugin's
         * `h`). A pass names the rows it wants through out_ids. */
        rad_buf src    = 0;
        /* [max_tok, n_embd] in: the hidden state this pass predicts from; out: the head's own,
         * the next round's input. Sized for a prefill chunk (history passes are one row per
         * token). */
        rad_buf hout   = 0;
        rad_buf e      = 0;   /* [max_tok, n_embd]      the input token's embedding */
        rad_buf c      = 0;   /* [max_tok, 2 * n_embd]  the two normed halves, fc's input */
        rad_buf x      = 0;   /* [max_tok, n_embd]      the head's own residual stream */
        rad_buf logits = 0;   /* [max_logit_rows, n_vocab]  the trunk's, shared */
        /* The 2-bit draft head's working set (see op_dlogits), sized at max_logit_rows: only
         * drafting rows reach it. */
        rad_buf dq   = 0;   /* [max_logit_rows, n_embd]   int8 codes */
        rad_buf ds   = 0;   /* [max_logit_rows, 1]        f32, one scale a row */
        rad_buf dsum = 0;   /* [n_embd/group, mpad]       f32, group-major (quant_act_i8's asum) */
        rad_buf dlog = 0;   /* [max_logit_rows, n_vocab]  bf16, the coarse logits */
        rad_buf didx = 0;   /* [max_logit_rows, 1]        i32, the proposed token */
        rad_buf dval = 0;   /* [max_logit_rows, 1]        f32, its logit */
        /* The layer reuses the trunk's working set, which is dead by the time the head runs (a
         * second set would be 150 MiB). */
        AttnGatedFP8::Wire attn{};
        typename Ffn::Wire mlp{};
    };

    struct Src {
        const char* fc                 = nullptr;
        const char* norm               = nullptr;
        const char* pre_norm_hidden    = nullptr;
        const char* pre_norm_embedding = nullptr;
        /* The checkpoint's lm_head. The trunk maps it as `output.weight`; the draft head is a
         * second logical weight over the same tensor, quantised to 2 bits (mtp_draft_head_q2). */
        const char* lm                 = nullptr;
        AttnGatedFP8::Src attn{};
        typename Ffn::Src mlp{};
    };

    Geom       g{};
    Wire       w{};
    int        layer   = 0;    /* absolute, for the KV group's layer_slot lookup */
    int64_t    rows    = 0;    /* the widest pass: one row per token of a prefill chunk */
    int64_t    off_e = 0, off_h = 0;   /* the two halves' column offsets inside `c` */
    rad_weight w_tok = 0, w_lm = 0, w_fc = 0, w_norm = 0, w_pre_h = 0, w_pre_e = 0;
    rad_weight w_dh = 0, w_dhs = 0;    /* the 2-bit lm_head and its (scale, zero) plane */
    bool       q2 = false;             /* is the draft round reading it? */
    AttnGatedFP8 attn{};
    Ffn          mlp{};
    rad_op     op_gather = 0, op_embed = 0, op_embed_ar = 0, op_pre_h = 0, op_pre_e = 0, op_fc = 0;
    /* Media rows over the head's embedding: the history pass embeds the prompt too (one position
     * on), and an image token's embedding comes from the encoder. Set `media` before declare when
     * the program has an encoder. */
    bool       media = false;
    MediaRows  mrows{};
    rad_op     op_norm = 0, op_dsel = 0, op_dback = 0, op_logits = 0;
    rad_op     op_dquant = 0, op_dlogits = 0, op_dtopk = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup kv,
                 int64_t rotary_dim, rad_weight tok_embd, rad_weight lm_head, float out_wadd,
                 const Wire& wire, const Src& src);
    void step(RadCtx* c, const RadBatch* batch) const;
};

template <class Ffn>
inline int MtpBlockFP8<Ffn>::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                     rad_kvgroup kv, int64_t rotary_dim, rad_weight tok_embd,
                                     rad_weight lm_head, float out_wadd, const Wire& wire,
                                     const Src& src) {
    g = geom; w = wire; layer = l; w_tok = tok_embd; w_lm = lm_head;
    rows = g.max_tok;
    if (rows <= 0 || !w_tok || !w_lm) return RAD_E_INVAL;

    /* The normed embedding first, then the normed hidden state (see the header). */
    off_e = 0;
    off_h = g.n_embd;

    /* The head's own weights, all replicated: three norms over the replicated stream, and `fc`,
     * which column-sharded would save 52 MiB per rank at the cost of an all-reduce per round. */
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.pre_fc_norm_hidden.weight"),    src.pre_norm_hidden));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.pre_fc_norm_embedding.weight"), src.pre_norm_embedding));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.norm.weight"),                  src.norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.fc.weight"),                    src.fc));

    w_pre_h = decl_w(b, nm.f("mtp.pre_fc_norm_hidden.weight"),    RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_pre_e = decl_w(b, nm.f("mtp.pre_fc_norm_embedding.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_norm  = decl_w(b, nm.f("mtp.norm.weight"),                  RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_fc    = decl_w(b, nm.f("mtp.fc.weight"), RAD_BF16, {g.n_embd, 2 * g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* Separate gather_rows and embed_lookup ops over the trunk's buffers and weights, because
     * they write different buffers and liveness comes from each op's read/write sets. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                  RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                   {w.src}, {w.hout});
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_embd", g.n_embd),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                RAD_INT("vocab_offset", g.vocab_off)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.e});
    if (media) RAD_ARCH_TRY(mrows.declare(b, rows, g.n_embd, w.e));
    /* The trunk's table is vocab-sharded, so the lookup is partial and needs an exact
     * all-reduce. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, rows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.e}, {w.e});

    op_pre_h = rw(b, RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_pre_h)),
                  {w.hout}, {w.c});
    op_pre_e = rw(b, RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_pre_e)),
                  {w.e}, {w.c});

    op_fc = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", g.n_embd),
                              RAD_INT("K", 2 * g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_fc)),
               {w.c}, {w.x});

    RAD_ARCH_TRY(attn.declare(b, nm, g, l, kv, rotary_dim, w.attn, src.attn));

    /* Moves round 1's rows of a history pass (RadBatch::draft_out_ids, one per drafting sequence)
     * to the front of `x` before the feed-forward, which then runs on those rows only. Goes through
     * `hout` (dead between fc's read and the norm's write), since an in-place gather could
     * overwrite a row not yet read. */
    op_dsel = rw(b, RAD_OP(b, "gather_rows",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x}, {w.hout});
    op_dback = rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.hout}, {w.x});
    if (!op_dsel || !op_dback) return RAD_E_INVAL;

    RAD_ARCH_TRY(mlp.declare(b, nm, g, l, w.mlp, src.mlp));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", out_wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.hout});

    /* The trunk's lm_head over the head's hidden state, banded at the logits buffer's extent: it
     * runs over one row per drafting sequence. */
    op_logits = rw(b, RAD_OP(b, "logits_gemm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                 RAD_INT("n_vocab", g.n_vocab),
                                 RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_lm)),
                  {w.hout}, {w.logits});

    /* ---- The 2-bit draft head.
     *
     * `logits_gemm` is at the memory roofline for the bf16 lm_head's 2.54 GB, and a speculative
     * step runs the projection once to verify plus once per draft round. Drafts are verified by the
     * exact head, so a coarser draft head costs only acceptance, never an emitted token.
     *
     * The draft rounds read a 2-bit copy: 318 MB of codes and 40 MB of (scale, zero) dwords, an
     * eighth of the traffic. libr4d's `gemm_w2a8_nt` takes codes and the scale/zero planes as two
     * operands, laid out as `r4d.w2a8.frag.g128` at load (libr4d/r4d_layout.cpp). The grid is
     * asymmetric (four levels are too few for a symmetric one), so the activation side uses
     * `quant_act_i8`'s three-output form, whose per-group row sums subtract the stored zero.
     *
     * The draft head bypasses the sampler, which reads f32 logits: `row_topk` at R=1 takes the
     * argmax on the GEMM's bf16 output and the engine reads the id from `didx`. The trunk's logits
     * plane is untouched.
     */
    enum { MTP_W2_GROUP = 128 };      /* libr4d's R4D_GEMM_W2A8_GROUP, and the layout's g128 tag */
    /* The plugin declares the working set only when mtp_draft_head_q2 found a 2-bit head (and
     * declared its name map). */
    q2 = src.lm && w.dq && w.ds && w.dsum && w.dlog && w.didx && w.dval;
    if (q2) {
        /* Replicated over the whole vocabulary. Without the sampler's cross-rank merge, a sharded
         * plane would let each rank propose only its own vocabulary slice. Replication costs 341
         * MiB per card at 2 bits and keeps the draft path single-rank. (The stored bytes are also
         * tile-major, 16 output rows per packed block, so a shard would have to be whole tiles.) */
        const char* dhn = nm.f("mtp.draft_head.weight");
        w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_U2, {g.n_vocab_all, g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        w_dhs = decl_view(b, nm.f("mtp.draft_head.scale"), dhn, "scale,zero", RAD_F16,
                          {g.n_vocab_all, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                          grp_layer(l));
        if (!w_dh || !w_dhs) return RAD_E_INVAL;

        op_dquant = rw(b, RAD_OP(b, "quant_act_i8",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                     RAD_INT("n", g.n_embd), RAD_INT("group", MTP_W2_GROUP),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.hout}, {w.dq, w.ds, w.dsum});

        op_dlogits = rw(b, RAD_OP(b, "gemm_nt_q",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                      RAD_INT("N", g.n_vocab_all), RAD_INT("K", g.n_embd),
                                      RAD_INT("group", MTP_W2_GROUP), RAD_STR("dtype", "w2a8")),
                           RAD_WEIGHTS(w_dh, w_dhs)),
                       {w.dq, w.ds, w.dsum}, {w.dlog});

        op_dtopk = rw(b, RAD_OP(b, "row_topk",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("N", g.n_vocab_all), RAD_INT("R", 1),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {w.dlog}, {w.didx, w.dval});
    }

    /* Keep `hout` live through the head's last op: it is the next round's input, and if only the
     * lm_head's first op read it, the planner could give its bytes to later ops (silently lowering
     * acceptance). */
    {
        const rad_op last = q2 ? op_dtopk : op_logits;
        if (rad_op_reads(b, last, &w.hout, 1) < 0) return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* One pass of the head over the scheduler's batch for it: n_tok rows, `token_ids` already shifted
 * one position on (batch.cpp, token_at), `positions` the head's indices, and the KV group batch
 * carrying each row's slot, so the attention layer needs no special case. */
template <class Ffn>
inline void MtpBlockFP8<Ffn>::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int r = batch->draft_pass;

    /* The history pass gathers; later rounds read the previous pass's output from the same
     * buffer. */
    if (r == 1) {
        rad_step_fail(c, "the MTP head was handed a draft-round-1 pass: that round is carried by "
                         "the history pass (RadBatch::draft_out_ids) and has no pass of its own");
        return;
    }
    if (r < 0 && batch->n_out > 0)
        RAD_ISSUE_N(c, op_gather, batch->n_out,
                    RAD_B(w.src), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(w.hout, batch->n_out));

    RAD_ISSUE(c, op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.e, T));
    mrows.step(c, batch, w.e, T);
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.e, T), RAD_NONE);
    RAD_ISSUE(c, op_pre_h, brows(w.hout, T), RAD_W(w_pre_h), bcol(w.c, off_h, g.n_embd, T));
    RAD_ISSUE(c, op_pre_e, brows(w.e, T),    RAD_W(w_pre_e), bcol(w.c, off_e, g.n_embd, T));

    RAD_ISSUE(c, op_fc, brows(w.c, T), RAD_W(w_fc), brows(w.x, T));

    /* A history pass with no drafting rows (every prefill chunk but a prompt's last) only needs
     * K and V. `n_draft_out` is the same on every rank, so skipped collectives match. */
    if (r < 0 && batch->n_draft_out <= 0) {
        attn.store_kv(c, batch);
        return;
    }
    attn.step(c, batch);

    /* A history pass with drafting rows continues on those rows alone (round 1): they are moved
     * to the front of `x` and the rest of the head runs over them. */
    int64_t L = T;                      /* the rows the stream goes on with */
    if (r < 0) {
        L = batch->n_draft_out;
        RAD_ISSUE_N(c, op_dsel, L, RAD_B(w.x), praw(batch->draft_out_ids, RAD_I32, L),
                    brows(w.hout, L));
        RAD_ISSUE_N(c, op_dback, L, brows(w.hout, L), brows(w.x, L));
    }

    mlp.step_rows(c, L);

    RAD_ISSUE(c, op_norm, brows(w.x, L), RAD_W(w_norm), brows(w.hout, L));
    if (q2) {
        /* `dsum` is group-major [n_embd/group][ceil(L/16)*16], so it has no row slice; pass it
         * whole. */
        RAD_ISSUE_N(c, op_dquant, L, brows(w.hout, L), brows(w.dq, L), brows(w.ds, L),
                    RAD_B(w.dsum));
        /* The trailing `b_ref` is mxfp4's and absent; operands are positional, so pass RAD_NONE
         * (rad_fp8.h). */
        RAD_ISSUE_N(c, op_dlogits, L, brows(w.dq, L), brows(w.ds, L), RAD_W(w_dh), RAD_W(w_dhs),
                    brows(w.dlog, L), RAD_B(w.dsum), RAD_NONE);
        RAD_ISSUE_N(c, op_dtopk, L, brows(w.dlog, L), brows(w.didx, L), brows(w.dval, L),
                    RAD_NONE);   /* the `pairs` slot, absent: operands are positional */
    } else {
        RAD_ISSUE_N(c, op_logits, L, brows(w.hout, L), RAD_W(w_lm), brows(w.logits, L));
    }
}

/* The dense head (SwiGLU feed-forward). The routed one is MtpBlockFP8<MoeFP8>, named by the
 * plugin that needs it; this header does not include rad_block_moe_fp8.h. */
using MtpFP8 = MtpBlockFP8<MlpFP8>;

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MTP_FP8_H */
