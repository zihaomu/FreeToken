/* rad_block_dflash2.h -- the DFlash2 block-diffusion drafter, bf16.
 *
 * A small transformer that proposes a whole block of tokens in one forward pass. Compared with
 * the MTP head (rad_block_mtp_fp8.h):
 *
 *     MTP        depth d costs d chained forwards; round i+1 embeds round i's token.
 *     DFlash2    depth d costs one forward over d+1 positions: the anchor token plus d mask
 *                tokens are denoised together.
 *
 * So depth is nearly free here, and there is no `draft_pass`. Published acceptance is 4.10-5.46
 * versus MTP's 3.74-5.02 at 7 drafts.
 *
 * Since positions are proposed together, position l+1 does not know what position l chose. The
 * candidate selector (`dflash_select`) restores that dependence: each position keeps the draft
 * head's top K, a low-rank bilinear form scores edges between consecutive positions, and a greedy
 * walk picks one path.
 *
 * ============================== THE TWO PASSES ==============================
 *
 * Context pass, one row per committed token. The drafter does not run its layers over the
 * context; it turns the target's hidden states at five intermediate layers into its own K and V:
 *
 *     c  = fc(concat of the trunk's residual stream at layers 5, 19, 33, 47, 61)   [n_taps*n_embd]
 *     c  = hidden_norm(c)
 *     for each draft layer:  k|v = kv_proj(c);  k = k_norm(k) then rope;  kv_store(k, v)
 *
 * The taps and the `dflash_src` buffer they land in are the trunk's (qwen35_fp8.cpp).
 *
 * Query pass, one row per draft position, `block` = 1 + n_spec rows per sequence:
 *
 *     x = embed(anchor, mask, mask, ...)
 *     for each layer:
 *         h  = attn_norm(x);   coef = attn_conv_proj(h)
 *         h  = dflash_conv(h, coef[0:taps*NG], attn_conv_base[0])          <- prepare
 *         h  = o_proj(attention(q_proj(h), kv_proj(h)))                    <- non-causal, windowed
 *         h  = dflash_conv(h, coef[taps*NG:], attn_conv_base[1]);  x += h  <- finish
 *         h  = ffn_norm(x);    coef = ffn_conv_proj(h)
 *         h  = dflash_conv(h, coef[0:taps*NG], ffn_conv_base[0])
 *         h  = down(silu_mul(gate_up(h)))
 *         h  = dflash_conv(h, coef[taps*NG:], ffn_conv_base[1]);   x += h
 *     x = norm(x)
 *
 * then, over the mask rows of each block (offsets 1..block-1; the anchor at offset 0 is already
 * chosen and only conditions the rest), each predicting the token at its own position:
 *
 *     cand, unary = top_k(draft_head(x))          the candidates and their logits
 *     hp          = hidden_projection(x)          the selector's query, rank 256
 *     tokens      = dflash_select(cand, unary, hp, anchor, predecessor, successor)
 *
 * ============================== LILICORR ==============================
 *
 * Config::lili is NVIDIA's LiLiCorr backbone (Nemotron-3.5-Bolt's drafter): the same layers with a
 * learned sink logit a query head in the attention (attn_paged_sink, `attention_sink_bias`), and no
 * selector or draft head here. The query pass ends with the gather into `sel_h`; the plugin then
 * runs rad_block_lilicorr.h's correlator over those rows, which reads the target's own head.
 *
 * ============================== WHAT IS SHARED WITH THE TARGET ==============================
 *
 * The embedding table and lm_head. The checkpoint (81 tensors) ships neither, and
 * `mask_token_id` is an ordinary id in the target's vocabulary (248070 here), so there is no
 * separate mask embedding.
 *
 * The head reads the 2-bit copy of the lm_head, as the MTP head does (rad_block_mtp_fp8.h,
 * op_dlogits): the target verifies every draft with its own head, so a coarser draft costs only
 * acceptance, never an emitted token. Candidates are rescored against the target's head when it
 * is fp8 or int8 (op_drerank), so coarseness affects only which candidates are proposed.
 *
 * ============================== PLAIN NORMS ==============================
 *
 * The norms use `w`, not the trunk's `1 + w`: vLLM uses plain RMSNorm here, and the gains are all
 * positive and centred near one (hidden_norm 0.11..1.24, input_layernorm 0.05..1.70, k_norm
 * 0.32..2.53, final norm 2.45..5.25). So this geometry's `wadd` and `wadd_qk` are zero.
 *
 * ============================== SHARDING ==============================
 *
 * q/kv and gate_up shard by row, o and down by column with an all-reduce: ten collectives per
 * draft pass, each 5120 * block * 2 bytes (80 KiB at block 8), below the lossy wire's floor.
 * Fewer KV heads than ranks are replicated (drafter_geom's `kv_repl_ok`): each rank reads its
 * head's k/v rows at `kv_off` and keeps that head in its cache.
 *
 * fc              [n_embd, n_taps*n_embd], 262 MiB in fp8 at Qwen3.8's width, read whole by the
 *                   context pass every step. Split by rows over the ranks and all-gathered, and
 *                   kept bit for bit. Each rank hands over an equal share of c's columns
 *                   (n_embd / world) and computes the fewest whole 128-row scale blocks that cover
 *                   it (the 31B at tp 4: 1344 columns a rank inside 11 blocks, 1408 rows, against
 *                   42 blocks whole), a row span of the one stored weight
 *                   (DenseFP8::declare_slice), so no reconvert. The GEMM takes its K split from
 *                   the whole width (OPS.md's `split_n`), so each output is summed as the unsplit
 *                   fc sums it, and the all-gather reads each rank's window out of its wider rows
 *                   (an input pitch) and writes c whole. A GEMM cannot write a column slice (the
 *                   narrow fp8 kernel's store is tight), which is why the windows are the
 *                   gather's to cut. One rank, or an fc off block fp8, keeps it whole.
 *
 * Replicated:
 *
 *   conv weights    projection and base kernel, 26 MiB a layer. The convolution spans the whole
 *                   replicated hidden row, so every rank needs every coefficient.
 *   codebooks       [n_vocab, 256], 127 MiB each, gathered by global token id. Only 128 rows are
 *                   read per pass; the cost is residency.
 *
 * Cost: the context pass is the expensive half (`fc` plus five KV projections read 3.4 GB over a
 * 2048-token prefill chunk). The attention window is 2048 tokens, so context rows further back
 * are never read and the scheduler need not compute them.
 */
#ifndef RAD_BLOCK_DFLASH2_H
#define RAD_BLOCK_DFLASH2_H

#include "rad_arch.h"
#include "rad_fp8.h"
#include "rad_spechub.h"   /* RAD_MTP_COUPLE_*: the coupling wire (--draft-coupling) */

namespace rad {
namespace arch {

/* brow_slice() for a weight. The conv base kernel is one [2, taps, n_embd] tensor holding the
 * prepare and finish tap sets; the op takes one side at a time. */
inline RadOperand wrow_slice(rad_weight h, int64_t first, int64_t n, int64_t pitch) {
    RadOperand o = RAD_W(h);
    o.offset = first * pitch;
    o.rows   = n;
    return o;
}

struct Dflash2 {
    enum { MAX_LAYERS = 8 };

    /* The drafter config beyond its geometry, read by the plugin from `draft.*` metadata keys. */
    struct Config {
        int64_t n_taps   = 0;     /* how many trunk layers `fc` concatenates */
        int64_t block    = 0;     /* query rows a sequence: the anchor plus n_spec mask tokens */
        int64_t taps     = 2;     /* the convolution's */
        int64_t group    = 16;
        int64_t sel_rank = 256;   /* the codebooks' width */
        int64_t sel_topk = 16;    /* candidates kept per position */
        /* Propose a tree of `steps()` nodes instead of the selector's one path (`dflash_tree`,
         * rad_drafter_tree): best-first over the candidate lattice by path probability, each
         * edge's probability the softmax of its scores over the K candidates divided by
         * `tree_temp`. False: the greedy walk. */
        bool    tree      = false;
        double  tree_temp = 1.0;
        /* A tree's draft nodes a step, 0 for `steps()`. More than the block drafts is a wider tree
         * over the same lattice (its depth is still at most `steps()`); the proposal buffers, the
         * drafter's declared depth and the verify are this wide. */
        int64_t tree_nodes = 0;
        int64_t window   = 0;     /* sliding attention width, 0 for none */
        int     causal   = 0;
        /* The target's embedding scale, applied to the query block as the drafter was trained
         * (dflash_config.input_embedding_scale): 1 for Qwen, sqrt(hidden) in bf16 for Gemma. */
        double  escale   = 1.0;
        /* Bytes of the next GEMM's weight read into the cache beside each all-reduce of the query
         * pass (prefetch_beside, rad_arch.h); 0 reads none. */
        int64_t prefetch = 0;

        /* Also declare the short block's query pass (kShortBlock rows, or `short_rows`) for
         * --draft-short-from; an architecture that does not call rad_drafter_short_block leaves
         * it off and its graph as it was. A tree's selector at that block grows a tree of its own
         * over the shorter pass, `short_rows - 1` nodes or fewer. */
        int     short_block = 0;
        /* --draft-coupling: the temperature of the draft law Q the coupled walk samples from
         * (dflash_couple_propose's `q_temp`). Any value keeps the output exact; it only moves
         * acceptance, sum min(P, Q). */
        double  q_temp = 1.0;
        int64_t short_rows  = 0;
        /* The drafter's linears are served at i8a8h (DenseFP8::w8h: IQ8R, or a rotated int8 or
         * narrower form at group 128), learned by the plugin from `probe_i8` before it declares the
         * activation pairs: they are then int8 codes with one f32 scale per (row, act_g8), written
         * by had_quant_act_i8g. Every linear must agree. `fc_mapped`: probe_i8 made
         * `dflash.fc.weight`'s name map. */
        bool    act_i8    = false;
        int64_t act_g8    = RAD_FP8_BLOCK;
        bool    fc_mapped = false;
        /* LiLiCorr's backbone (the LILICORR section above): sinks, and the head left to the
         * plugin. sel_rank and sel_topk are then unused. */
        int     lili = 0;
        int64_t steps() const { return block - 1; }
        int64_t nodes() const { return tree && tree_nodes > 0 ? tree_nodes : steps(); }
        int64_t ng(int64_t n_embd) const { return group ? n_embd / group : 0; }
        /* The kernel projection emits [2, taps, NG] per token (prepare's set, then finish's);
         * `dflash_conv` reads a slice of it at the full row pitch. */
        int64_t coef_dim(int64_t n_embd) const { return 2 * taps * ng(n_embd); }
    };

    struct Wire {
        /* the context pass */
        rad_buf src  = 0;   /* [max_tok, n_taps * n_embd]  the trunk's taps, read only */
        rad_buf c    = 0;   /* [max_tok, n_embd]           fc's output, then normed in place */
        /* [max_tok, fc_rows]  the block-aligned rows of a split fc this rank computes, whose
         * fc_share-wide window at fc_off is gathered into c. The block declares it (only when it
         * splits fc); leave it 0. */
        rad_buf cpart = 0;
        /* K and V are separate buffers because of the per-head k norm: `rmsnorm` takes one row
         * pitch, and K heads inside a fused [K|V] row are head_dim apart within a token but
         * 2*kv_dim apart between tokens. libr4d refuses such an operand (libr4d/r4d_model.cpp,
         * rows_pitch). Same reason as in rad_block_attn_gated.h. */
        rad_buf ck   = 0;   /* [max_tok, kv_dim]           ONE layer's K, reused across layers */
        rad_buf cv   = 0;   /* [max_tok, kv_dim]           ONE layer's V, reused across layers */
        /* the query pass */
        rad_buf x    = 0;   /* [qrows, n_embd]  the drafter's residual stream */
        rad_buf h    = 0;   /* [qrows, n_embd]  normed input, o/down output, and the final norm */
        rad_buf h2   = 0;   /* [qrows, n_embd]  the convolution's output; must not alias x */
        rad_buf coef = 0;   /* [qrows, coef_dim] */
        rad_buf q    = 0;   /* [qrows, q_dim] */
        rad_buf k    = 0;   /* [qrows, kv_dim]  separate from v, see `ck` */
        rad_buf v    = 0;   /* [qrows, kv_dim] */
        rad_buf attn = 0;   /* [qrows, q_dim] */
        rad_buf gate_up = 0;/* [qrows, 2 * n_ff] */
        rad_buf ffn  = 0;   /* [qrows, n_ff] */
        /* FP8 codes and scales of the activations the linears read, written by `quant_act_fp8`
         * and read only by the `gemm_nt_q` linears (int8 codes with a scale per Config::act_g8,
         * written by `had_quant_act_i8g`, when the drafter's linears are rotated int8:
         * Config::act_i8). One pair per activation, not per linear: q, k
         * and v all read h2. h2 is still quantised twice a layer, because the convolution
         * rewrites it between the attention and feed-forward halves. */
        rad_buf srcq = 0, srcs = 0;  /* [max_tok, n_taps*n_embd] e4m3 | [.., /128] f32 */
        rad_buf cq   = 0, cs   = 0;  /* [max_tok, n_embd] */
        rad_buf h2q  = 0, h2s  = 0;  /* [qrows,   n_embd] */
        rad_buf atq  = 0, ats  = 0;  /* [qrows,   q_dim] */
        rad_buf ffq  = 0, ffs  = 0;  /* [qrows,   n_ff] */
        /* the head and the selector, one row per draft position (block - 1 per sequence) */
        rad_buf sel_h = 0;  /* [srows, n_embd]     the gathered final hidden states */
        rad_buf hp    = 0;  /* [srows, sel_rank]   hidden_projection of them */
        rad_buf dq    = 0;  /* [srows, n_embd]  i8   the 2-bit head's quantised activation */
        rad_buf ds    = 0;  /* [srows, 1]       f32  one scale a row */
        rad_buf dsum  = 0;  /* [n_embd/128, pad] f32 GROUP-major, quant_act_i8's asum plane */
        rad_buf dlog  = 0;  /* [srows, head_rows(g)] bf16  the coarse logits */
        rad_buf cand  = 0;  /* [srows, sel_topk] i32  candidate token ids, global */
        rad_buf unary = 0;  /* [srows, sel_topk] f32  and their logits */
        /* Vocab-parallel pairs, declared only when head_sharded(g): (int32 id, f32 value) pairs
         * stored as f32, this rank's top-R and the all-gathered world_size * R. */
        rad_buf dpair = 0;  /* [srows, sel_topk, 2]         f32 */
        rad_buf dgath = 0;  /* [srows, world * sel_topk, 2] f32 */
        rad_buf tokens = 0; /* [max_seqs, steps] i32  the draft; the engine reads it by name */
        /* --draft-coupling: [max_seqs, DRAW_WORDS] draws the engine stages, and [steps, max_seqs,
         * STATE_WORDS] states it reads back (abi/rad_spechub.h). Both or neither. */
        rad_buf couple_draws = 0, couple_state = 0;
        /* [max_seqs, steps] i32  a tree's parents (Config::tree, rad_drafter_tree); 0 otherwise */
        rad_buf parents = 0;
    };

    /* Whether the 2-bit draft head is vocab-sharded (rad_arch.h: vocab_head_sharded), so whoever
     * declares the buffers can ask the block. Sharding halves the draft path's widest read (341 MiB
     * of codes and scales at 248320 rows) for one all_gather of R pairs a row. */
    static bool    head_sharded(const Geom& g) { return vocab_head_sharded(g); }
    /* Rows of the draft head on this rank, and so the width of `dlog`. */
    static int64_t head_rows(const Geom& g)    { return vocab_head_rows(g); }

    Geom    g{};      /* the drafter's geometry, not the trunk's */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;   /* absolute, for the KV group's layer_slot lookup */
    int64_t qrows = 0, crows = 0, srows = 0;
    int64_t gated_rows = 0;    /* the band op_gug was declared over; 0 when there is no fold */
    rad_kvgroup kv = 0;

    /* model-scope weights.
     *
     * The linears are block fp8 or OCP-MXFP4 (rad_fp8.h: DenseFP8, per the recipe) although the
     * checkpoint is bf16: the drafter is read whole every decode step. On Qwen3.8-27B at depth 7,
     * MXFP4 accepts ~1% fewer tokens per step than fp8 and saves ~1.5 ms of a ~34 ms step.
     * Quantisation can only change acceptance, since every draft is verified. */
    DenseFP8   w_fc{};
    /* fc split by rows (see SHARDING): w_fc is this rank's fc_rows block-aligned rows, holding its
     * fc_share output columns at row fc_off. fc_rows 0: w_fc is the whole fc, replicated. */
    int64_t    fc_share = 0, fc_rows = 0, fc_off = 0;
    rad_weight w_hnorm = 0, w_norm = 0;
    rad_weight w_selp = 0, w_pred = 0, w_succ = 0;
    rad_weight w_tok = 0;                    /* the target's embedding table */
    rad_weight w_dh = 0, w_dhs = 0;          /* the 2-bit (or 8-bit) copy of the target's lm_head */
    /* The copy is int8 with an f16 scale per (row, 128 of K) where the recipe made it so (the
     * 31B's: the head read at w8a8, so the selector sees the candidates an 8-bit head ranks), and
     * the 2-bit grid otherwise. Set by declare from the copy's encoding. */
    bool       head_i8 = false;

    /* per-layer weights */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    /* K and V are separate weights: a block-scaled operand's layout follows the op's N, so a fused
     * [2*kv_dim, n_embd] weight issued one half at a time would hand the layout hook a
     * [2*kv_dim/128][K/128] plane for a [kv_dim, K] op, which it refuses. */
    DenseFP8   w_q[MAX_LAYERS]{},  w_k[MAX_LAYERS]{}, w_v[MAX_LAYERS]{}, w_o[MAX_LAYERS]{};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {};
    DenseFP8   w_gu[MAX_LAYERS]{}, w_dn[MAX_LAYERS]{};
    rad_weight w_acp[MAX_LAYERS] = {}, w_acb[MAX_LAYERS] = {};
    rad_weight w_fcp[MAX_LAYERS] = {}, w_fcb[MAX_LAYERS] = {};

    /* context-pass ops */
    rad_op op_cfc = 0, op_cnorm = 0, op_crope = 0, op_cstore = 0;
    rad_op op_cgath = 0;   /* the split fc's all-gather */
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};
    /* The k norm, the k rotary and the cache store as one launch a layer (qkv_norm_rope_kv_store
     * with n_head 0 and v_norm 0), issued in place of those three where it resolves. */
    rad_op op_cqkr[MAX_LAYERS] = {};

    /* Activation quantisers, one per activation that feeds a linear. Weightless, so one declaration
     * serves every layer; ops with weights are per layer because a weight's prefetch interval
     * comes from the op that declares it. */
    rad_op op_qsrc = 0, op_qc = 0;                  /* context pass */
    rad_op op_qh2 = 0, op_qat = 0, op_qff = 0;      /* query pass */

    /* query-pass ops. Weightless ones are declared once and issued per layer; their read and write
     * sets are the same every layer. */
    rad_op op_embed = 0, op_embed_ar = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {};
    rad_op op_acoef[MAX_LAYERS] = {}, op_fcoef[MAX_LAYERS] = {};
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {};
    rad_op op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {};
    /* The q and k norms, both rotaries and the cache store as one launch a layer
     * (qkv_norm_rope_kv_store with v_norm 0), issued in place of those five where it resolves. */
    rad_op op_qkr[MAX_LAYERS] = {};
    rad_op op_gu[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_gug[MAX_LAYERS] = {};   /* the gate_up|silu|quant fold; 0 when nothing serves it */
    rad_op op_aconv[MAX_LAYERS] = {}, op_fconv[MAX_LAYERS] = {};
    /* The short block's (block_lo rows a sequence): the convolutions mask at the block's first row
     * and the selector strides by it, so each has an op of its own; every other op is per row. */
    rad_op op_aconv_lo[MAX_LAYERS] = {}, op_fconv_lo[MAX_LAYERS] = {}, op_sel_lo = 0;
    int64_t block_lo = 0;
    int64_t sel_lo_nodes = 0;   /* a tree's nodes at the short block (op_sel_lo) */
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn = 0;
    /* LiLiCorr: the attention with each layer's sinks, a weight, so one op a layer. */
    rad_weight w_sink[MAX_LAYERS] = {};
    rad_op     op_attn_sink[MAX_LAYERS] = {};
    rad_op op_act = 0, op_ar = 0, op_add = 0, op_final = 0;
    /* The residual add folded into the norm after it (add_rmsnorm), issued in place of op_add and
     * that norm where it resolves: op_afn[i] after layer i's attention (its feed-forward norm),
     * op_aan[i] after layer i-1's feed-forward (layer i's attention norm, i >= 1), op_afinal after
     * the last layer's (the final norm). All or none. */
    rad_op op_afn[MAX_LAYERS] = {}, op_aan[MAX_LAYERS] = {}, op_afinal = 0;
    /* The reads beside the query pass's all-reduces (cfg.prefetch), each the GEMMs that follow
     * in their order: beside the attention's, the feed-forward's coefficient projection, then
     * gate_up; beside the feed-forward's, the next layer's attention coefficient projection, then
     * its q -- after the last layer, the draft head; and o's beside the attention itself. 0 where
     * none is declared. */
    rad_op     op_pfa[MAX_LAYERS][2] = {}, op_pff[MAX_LAYERS][2] = {}, op_pfo[MAX_LAYERS] = {};
    rad_weight pfa_w[MAX_LAYERS][2] = {}, pff_w[MAX_LAYERS][2] = {}, pfo_w[MAX_LAYERS] = {};

    /* head and selector */
    rad_op op_gather = 0, op_dquant = 0, op_dlogits = 0, op_dtopk = 0;
    rad_op op_dgather = 0, op_dmerge = 0;   /* the vocab-parallel pair; 0 when the head is whole */
    rad_op op_drerank = 0;                  /* the candidates rescored against the target's head */
    /* The target's lm_head, set by the plugin before declare when logit_rerank can read it (fp8 or
     * int8 rows); 0 keeps the coarse head's scores. */
    rad_weight w_lm = 0;
    /* Candidates the coarse head proposes for the rerank (logit_rerank's and row_topk's ceiling).
     * Rescoring is exact, so a wider set can only find more of the exact top sel_topk. */
    static constexpr int64_t kRerankR = 32;
    int64_t    top_w = 0;                   /* the coarse top-K's width: kRerankR or sel_topk */
    rad_op op_hp = 0, op_sel = 0;
    /* --draft-coupling: the sampled walk after the greedy one (dflash_couple_propose), for the
     * drafter's block and its short one. 0 when the coupling is off. */
    rad_op op_couple = 0, op_couple_lo = 0;
    /* The short block: kShortBlock rows (depth 3, the window that pays at many sequences), when
     * the drafter's own block is longer (rad_drafter_short_block). */
    static constexpr int64_t kShortBlock = 4;

    /* Whether the container's drafter is served at i8a8h and at which scale group, from its `fc`,
     * whose name map this makes (set Config::fc_mapped). Asked before the activation pairs are
     * declared, since it decides their dtype and scale grid. */
    static int probe_i8(RadBuilder* b, Names& nm, const char* ck, bool* i8, int64_t* g8) {
        const char* fwn = nm.f("dflash.fc.weight");
        RAD_ARCH_TRY(map_linear(b, fwn, { nm.ckpt("%s.fc.weight", ck) }));
        RadEncoding fe{};
        *i8 = weight_enc(b, fwn, &fe) && linear_is_w8h(fe);
        *g8 = *i8 ? linear_w8h_group(fe) : (int64_t)RAD_FP8_BLOCK;
        return RAD_OK;
    }

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck_prefix,
                 const char* ck_lm, const Wire& wire);
    /* The context pass, one row per committed token; `batch` carries those rows and the draft KV
     * group's slot mapping. */
    void context(RadCtx* c, const RadBatch* batch) const;
    /* The query pass and the selector, `block` rows per sequence. */
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }
};

/* ------------------------------------------------------------------------------ declare */

inline int Dflash2::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                            int base_layer, rad_kvgroup group, rad_weight tok_embd,
                            const char* ck, const char* ck_lm, const Wire& wire) {
    g = geom; cfg = c; w = wire; first_layer = base_layer; kv = group; w_tok = tok_embd;
    if (!ck || !ck_lm || !w_tok) return RAD_E_INVAL;
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: dflash2 has %lld layers; this block carries at most %d\n",
                (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.block < 2 || cfg.n_taps <= 0 || (!cfg.lili && (cfg.sel_rank <= 0 || cfg.sel_topk <= 0)))
        return RAD_E_INVAL;

    const int64_t NG   = cfg.ng(g.n_embd);
    const int64_t CD   = cfg.coef_dim(g.n_embd);
    const int64_t qd   = q_dim(), kvd = kv_dim();

    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    srows = g.max_seqs * cfg.steps();
    if (qrows <= 0 || srows <= 0) return RAD_E_INVAL;

    /* The linears read the activation pairs the plugin declared: int8 ones take rotated int8
     * weights (cfg.act_i8), E4M3 ones block fp8 or MXFP4. */
    w_fc.w8h_ok = cfg.act_i8;
    for (int i = 0; i < MAX_LAYERS; ++i)
        w_q[i].w8h_ok = w_k[i].w8h_ok = w_v[i].w8h_ok = w_o[i].w8h_ok = w_gu[i].w8h_ok =
            w_dn[i].w8h_ok = cfg.act_i8;

    /* ---------------------------------------------------------------- model-scope weights */
    {
        const int64_t K   = cfg.n_taps * g.n_embd;
        const char*   fwn = nm.f("dflash.fc.weight");
        if (!cfg.fc_mapped) RAD_ARCH_TRY(map_linear(b, fwn, { nm.ckpt("%s.fc.weight", ck) }));
        /* Split only block fp8 (an MXFP4 GEMM's K split is its own), into equal shares of whole
         * 16-byte words (the gather's unit). */
        RadEncoding  fe{};
        const RadEncPlane* fsc = nullptr;
        const bool fp8 = weight_enc(b, fwn, &fe) && rad_enc_is(&fe, "affine") && fe.n_planes == 2 &&
                         !fe.transform[0] && fe.plane[0].dtype == RAD_F8E4M3 &&
                         (fsc = rad_enc_plane(&fe, "scale")) && fsc->block[0] == RAD_FP8_BLOCK &&
                         fsc->block[1] == RAD_FP8_BLOCK;
        const int64_t share = (g.world > 1 && g.n_embd % RAD_FP8_BLOCK == 0 &&
                               g.n_embd % g.world == 0 && (g.n_embd / g.world) % 8 == 0)
                                  ? g.n_embd / g.world : 0;
        if (fp8 && share > 0) {
            const int64_t B  = RAD_FP8_BLOCK;
            const int64_t lo = share * g.rank / B * B;
            const int64_t hi = (share * (g.rank + 1) + B - 1) / B * B;
            fc_share = share;
            fc_rows  = hi - lo;
            fc_off   = share * g.rank - lo;
            RAD_ARCH_TRY(w_fc.declare_slice(b, nm, nm.f("dflash.fc"), fwn, lo, hi, K,
                                            RAD_ACCESS_PER_TOKEN, grp_model()));
            if (!w.cpart)
                w.cpart = decl_b(b, nm.f("df_cpart"), g.act_dtype, { g.max_tok, fc_rows });
        } else {
            RAD_ARCH_TRY(w_fc.declare(b, nm, nm.f("dflash.fc"), g.n_embd, K, RAD_ACCESS_PER_TOKEN,
                                      RAD_SHARD_NONE, grp_model(), { nm.ckpt("%s.fc.weight", ck) },
                                      {}, true));
        }
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.hidden_norm.weight"), nm.ckpt("%s.hidden_norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.norm.weight"),        nm.ckpt("%s.norm.weight", ck)));
    w_hnorm = decl_w(b, nm.f("dflash.hidden_norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_norm  = decl_w(b, nm.f("dflash.norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* The 2-bit draft head, declared as the MTP head's is (rad_block_mtp_fp8.h):
     * `dflash.draft_head.weight` is the target's lm_head under its own name, quantised by the
     * recipe to u2 codes with an f16 scale and u8 zero per group of 128. One checkpoint tensor,
     * two views: codes, and the scale|zero planes the GEMM's relayout interleaves.
     *
     * When sharded, each rank's `row_topk` reports global ids, an all_gather moves R pairs a row,
     * and `row_topk_merge` reduces them to exactly the unsharded global top-R, ties included
     * (docs/OPS.md). This runs before the selector, so every rank holds the same candidates for
     * the replicated codebooks. The all_gather is `world * R` pairs a row (2 KiB at R = 16).
     *
     * LiLiCorr has neither the selector nor this head (its correlator reads the target's). */
    enum { DF_W2_GROUP = 128 };
    const bool dh_shard = head_sharded(g);
    const int  dh_split = dh_shard ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    if (!cfg.lili) {
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_proj.weight"),
                              nm.ckpt("%s.candidate_selector.hidden_projection.weight", ck)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_pred"),
                              nm.ckpt("%s.candidate_selector.predecessor_codebook", ck)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_succ"),
                              nm.ckpt("%s.candidate_selector.successor_codebook", ck)));
        w_selp  = decl_w(b, nm.f("dflash.sel_proj.weight"), RAD_BF16, {cfg.sel_rank, g.n_embd},
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        w_pred  = decl_w(b, nm.f("dflash.sel_pred"), RAD_BF16, {g.n_vocab_all, cfg.sel_rank},
                         RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        w_succ  = decl_w(b, nm.f("dflash.sel_succ"), RAD_BF16, {g.n_vocab_all, cfg.sel_rank},
                         RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());

        const char* dhn = nm.f("dflash.draft_head.weight");
        RAD_ARCH_TRY(map_copy(b, dhn, ck_lm));
        /* The 8-bit form: int8 codes, one f16 scale per (row, 128 of K), symmetric, which
         * gemm_nt_q reads at w8a8 against the same quant_act_i8 activation (without its asum). */
        {
            RadEncoding e{};
            const RadEncPlane* sc = nullptr;
            head_i8 = weight_enc(b, dhn, &e) && rad_enc_is(&e, "affine") && e.n_planes == 2 &&
                      !e.transform[0] && e.plane[0].dtype == RAD_I8 && e.plane[0].block[1] == 1 &&
                      (sc = rad_enc_plane(&e, "scale")) && sc->dtype == RAD_F16 &&
                      sc->block[0] == 1 && sc->block[1] == DF_W2_GROUP;
        }
        if (head_i8) {
            w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_I8, {head_rows(g), g.n_embd},
                              RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
            w_dhs = decl_view(b, nm.f("dflash.draft_head.scale"), dhn, "scale", RAD_F16,
                              {head_rows(g), g.n_embd / DF_W2_GROUP}, RAD_ACCESS_PER_TOKEN,
                              dh_split, grp_model());
        } else {
            w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_U2, {head_rows(g), g.n_embd},
                              RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
            w_dhs = decl_view(b, nm.f("dflash.draft_head.scale"), dhn, "scale,zero", RAD_F16,
                              {head_rows(g), g.n_embd}, RAD_ACCESS_PER_TOKEN, dh_split,
                              grp_model());
        }
        if (!w_dh || !w_dhs) return RAD_E_INVAL;
    }

    /* ---------------------------------------------------------------- per-layer weights */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.input_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.post_attention_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.q_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.k_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_conv_proj.weight", i),
                              nm.ckpt("%s.layers.%d.attention_conv.kernel_projection.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_conv_base", i),
                              nm.ckpt("%s.layers.%d.attention_conv.base_kernel", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_conv_proj.weight", i),
                              nm.ckpt("%s.layers.%d.mlp_conv.kernel_projection.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_conv_base", i),
                              nm.ckpt("%s.layers.%d.mlp_conv.base_kernel", ck, i)));

        w_anorm[i] = decl_w(b, nm.f("dflash.blk.%d.attn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        w_fnorm[i] = decl_w(b, nm.f("dflash.blk.%d.ffn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        RAD_ARCH_TRY(w_q[i].declare(b, nm, nm.f("dflash.blk.%d.attn_q", i), qd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                    { nm.ckpt("%s.layers.%d.self_attn.q_proj.weight", ck, i) }));
        /* q, k and v are separate weights. The context pass needs only k and v, so a fused qkv
         * would compute and discard q for every context token. k and v: see w_q's declaration. */
        RAD_ARCH_TRY(w_k[i].declare(b, nm, nm.f("dflash.blk.%d.attn_k", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                    { nm.ckpt("%s.layers.%d.self_attn.k_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_v[i].declare(b, nm, nm.f("dflash.blk.%d.attn_v", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                    { nm.ckpt("%s.layers.%d.self_attn.v_proj.weight", ck, i) }));
        /* Fewer KV heads than ranks (drafter_geom's `kv_repl_ok`): k and v are this rank's
         * replicated head, rows [kv_off, kv_off + kv_dim) of the unsharded projection, where the
         * even shard would hand a rank part of a head. q stays an even share of the heads. */
        if (g.world > g.n_head_kv_all) {
            RAD_ARCH_TRY(w_k[i].span(b, g.kv_off, g.kv_off + kvd));
            RAD_ARCH_TRY(w_v[i].span(b, g.kv_off, g.kv_off + kvd));
        }
        w_qn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        w_kn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        RAD_ARCH_TRY(w_o[i].declare(b, nm, nm.f("dflash.blk.%d.attn_output", i), g.n_embd, qd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(first_layer + i),
                                    { nm.ckpt("%s.layers.%d.self_attn.o_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_gu[i].declare(b, nm, nm.f("dflash.blk.%d.ffn_gate_up", i),
                                     2 * g.n_ff, g.n_embd,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                     { nm.ckpt("%s.layers.%d.mlp.gate_proj.weight", ck, i),
                                       nm.ckpt("%s.layers.%d.mlp.up_proj.weight", ck, i) },
                                     { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(w_dn[i].declare(b, nm, nm.f("dflash.blk.%d.ffn_down", i), g.n_embd, g.n_ff,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(first_layer + i),
                                     { nm.ckpt("%s.layers.%d.mlp.down_proj.weight", ck, i) }));
        w_acp[i]   = decl_w(b, nm.f("dflash.blk.%d.attn_conv_proj.weight", i), RAD_BF16,
                            {CD, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_acb[i]   = decl_w(b, nm.f("dflash.blk.%d.attn_conv_base", i), RAD_BF16,
                            {2, cfg.taps, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_fcp[i]   = decl_w(b, nm.f("dflash.blk.%d.ffn_conv_proj.weight", i), RAD_BF16,
                            {CD, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_fcb[i]   = decl_w(b, nm.f("dflash.blk.%d.ffn_conv_base", i), RAD_BF16,
                            {2, cfg.taps, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        /* LiLiCorr's sink logit a query head, split with the heads. */
        if (cfg.lili) {
            RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_sinks", i),
                                  nm.ckpt("%s.layers.%d.self_attn.attention_sink_bias", ck, i)));
            w_sink[i] = decl_w(b, nm.f("dflash.blk.%d.attn_sinks", i), RAD_F32, {g.n_head},
                               RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(first_layer + i));
            if (!w_sink[i]) return RAD_E_INVAL;
        }

    }

    /* Non-causal and windowed: the block is denoised as a whole, so each query row sees its whole
     * block plus the context (`causal` comes from the config's `is_causal: false`). The window is
     * the checkpoint's `sliding_window`.
     *
     * Declared before both kv_stores, which read the resolved `block_size` back with
     * rad_kv_block_size(); before this declaration it returns zero. */
    if (!cfg.lili) {
        op_attn = rw(b, RAD_OP(b, "attn_paged",
                        RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                                   RAD_INT("gqa", g.gqa()), RAD_INT("causal", cfg.causal),
                                   RAD_INT("window", cfg.window), RAD_INT("n_head", g.n_head),
                                   RAD_INT("block_size", RAD_KV_BLOCK),
                                   RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                   RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                        RAD_NOWEIGHTS),
                    {w.q}, {w.attn});
    } else {
        for (int64_t l = 0; l < g.n_layer; ++l) {
            const int i = (int)l;
            op_attn_sink[i] = rw(b, RAD_OP(b, "attn_paged_sink",
                                    RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block),
                                               RAD_INT("head_dim", g.head_dim),
                                               RAD_INT("gqa", g.gqa()), RAD_INT("causal", cfg.causal),
                                               RAD_INT("window", cfg.window),
                                               RAD_INT("n_head", g.n_head),
                                               RAD_INT("block_size", RAD_KV_BLOCK),
                                               RAD_INT("max_seqs", g.max_seqs),
                                               RAD_INT("max_ctx", g.max_ctx),
                                               RAD_STR("q_dtype", g.dtype),
                                               RAD_STR("kv_dtype", g.kv_dtype)),
                                    RAD_WEIGHTS(w_sink[i])),
                                {w.q}, {w.attn});
            if (!op_attn_sink[i]) return RAD_E_INVAL;
        }
    }

    /* ---------------------------------------------------------------- the context pass */
    /* Every linear is `fp8a8` (or `mxfp4a8` for a 4-bit recipe; DenseFP8 reads which from the
     * container and declares the GEMM), reading an E4M3 activation pair from a `quant_act_fp8`.
     * There is no `fp8a16` band: that kernel is M <= 1, and neither pass runs at one row.
     *
     * Nothing is row-sliced: the fp8a8 pair covers every M, and libr4d's bf16 GEMM row (used by
     * the conv coefficient projections and the selector's hidden projection, K = n_embd) is
     * C_LE("M", 65536). */
    /* A rotated int8 drafter (cfg.act_i8): the same five quantisers as had_quant_act_i8g at its
     * scale group, and no fp8a8 fold. Its linears read one activation pair, so they are all
     * i8a8h forms at one group or none is. */
    const char* const qop = cfg.act_i8 ? "had_quant_act_i8g" : "quant_act_fp8";
    const int64_t     qgr = cfg.act_i8 ? cfg.act_g8 : (int64_t)RAD_FP8_BLOCK;
    {
        auto fits = [&](const DenseFP8& d) {
            return d.w8h == cfg.act_i8 && (!d.w8h || d.g8 == cfg.act_g8);
        };
        bool same = fits(w_fc);
        for (int64_t l = 0; l < g.n_layer; ++l)
            same = same && fits(w_q[l]) && fits(w_k[l]) && fits(w_v[l]) && fits(w_o[l]) &&
                   fits(w_gu[l]) && fits(w_dn[l]);
        if (!same) {
            fprintf(stderr, "radiance: the drafter's linears are not all rotated int8 at one scale "
                            "group or all block fp8 / MXFP4; they read one activation pair, so the "
                            "recipe must give them one format\n");
            return RAD_E_DTYPE;
        }
    }
    op_qsrc = rw(b, RAD_OP(b, qop,
                    RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", cfg.n_taps * g.n_embd),
                               RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.src}, {w.srcq, w.srcs});
    if (fc_rows) {
        /* The split (SHARDING above): this rank's block-aligned rows into `cpart`, its window of
         * them gathered into `c`. The GEMM takes its K split from the whole width. The gather
         * rides the trunk's all-reduce instance (the same numel bound, so the same instance).
         * The message is a range, as the all-reduce's: a fixed numel follows max_tok, and the
         * arena's sizing declare (RadBuildCtx::shape_probe) refuses a fixed parameter that
         * changes with the step size. */
        op_cfc = w_fc.gemm(b, crows, fc_rows, cfg.n_taps * g.n_embd, {w.srcq, w.srcs}, {w.cpart},
                           g.n_embd);
        op_cgath = rw(b, RAD_OP(b, "all_gather",
                         RAD_PARAMS(RAD_INT("world_size", g.world),
                                    RAD_RANGE("numel", g.n_embd, crows * g.n_embd),
                                    RAD_STR("dtype", g.dtype), RAD_INT("row", fc_share)),
                         RAD_NOWEIGHTS),
                     {w.cpart}, {w.c});
    } else {
        op_cfc = w_fc.gemm(b, crows, g.n_embd, cfg.n_taps * g.n_embd, {w.srcq, w.srcs}, {w.c});
    }
    /* In place, like the per-head q/k norms below: a row's scale is computed before the row is
     * written. */
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_hnorm)),
                 {w.c}, {w.c});
    /* One quantise for all layers: every layer's k and v read the same `c`. */
    op_qc = rw(b, RAD_OP(b, qop,
                  RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                             RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS),
              {w.c}, {w.cq, w.cs});
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        /* One op per weight: an op's declared weight list gives each weight its prefetch interval
         * and residency class. */
        op_ck[i] = w_k[i].gemm(b, crows, kvd, g.n_embd, {w.cq, w.cs}, {w.ck});
        op_cv[i] = w_v[i].gemm(b, crows, kvd, g.n_embd, {w.cq, w.cs}, {w.cv});
        op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                        RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                        RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(w_kn[i])),
                         {w.ck}, {w.ck});
    }
    /* A whole-head NeoX rope over a q-only or k-only tensor of `heads` heads: `rope` rotates the
     * first n_head + n_head_kv heads, so n_head_kv is 0. Weightless, so one op serves every layer. */
    auto rope = [&](int64_t rows, int64_t heads, rad_buf x) {
        return rw(b, decl_rotary(b, g, "rope",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                                    RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                    RAD_STR("mode", "neox")),
                         RAD_NOWEIGHTS),
                  {x}, {x});
    };
    op_crope = rope(crows, g.n_head_kv, w.ck);
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS),
                  {w.ck, w.cv}, {});

    /* ---------------------------------------------------------------- the query pass */
    /* An unscaled lookup is declared exactly as before the scale existed, so a Qwen container
     * resolves the same op. */
    if (cfg.escale != 1.0)
        op_embed = rw(b, RAD_OP(b, "embed_lookup",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                                    RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                    RAD_INT("vocab_offset", g.vocab_off), RAD_F64("mult", cfg.escale)),
                         RAD_WEIGHTS(w_tok)),
                     {}, {w.x});
    else
        op_embed = rw(b, RAD_OP(b, "embed_lookup",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                                    RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                    RAD_INT("vocab_offset", g.vocab_off)),
                         RAD_WEIGHTS(w_tok)),
                     {}, {w.x});
    /* The trunk's embedding is vocab-sharded, so the lookup is partial and needs an exact
     * all-reduce. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.x}, {w.x});
    /* The query pass's quantisers, issued per layer: h2 twice (the convolution rewrites it between
     * the halves), attn and ffn once each. */
    op_qh2 = rw(b, RAD_OP(b, qop,
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                              RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.h2}, {w.h2q, w.h2s});
    op_qat = rw(b, RAD_OP(b, qop,
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", qd),
                              RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.attn}, {w.atq, w.ats});
    op_qff = rw(b, RAD_OP(b, qop,
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_INT("group", qgr), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.ffn}, {w.ffq, w.ffs});

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_anorm[i])),
                        {w.x}, {w.h});
        op_acoef[i] = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", CD),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_acp[i])),
                        {w.h}, {w.coef});
        op_qp[i] = w_q[i].gemm(b, qrows, qd, g.n_embd, {w.h2q, w.h2s}, {w.q});
        op_kp[i] = w_k[i].gemm(b, qrows, kvd, g.n_embd, {w.h2q, w.h2s}, {w.k});
        op_vp[i] = w_v[i].gemm(b, qrows, kvd, g.n_embd, {w.h2q, w.h2s}, {w.v});
        op_qn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_qn[i])),
                     {w.q}, {w.q});
        op_kn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head_kv), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_kn[i])),
                     {w.k}, {w.k});
        op_op[i] = w_o[i].gemm(b, qrows, g.n_embd, qd, {w.atq, w.ats}, {w.h});
        op_fnorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fnorm[i])),
                        {w.x}, {w.h});
        op_fcoef[i] = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", CD),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fcp[i])),
                        {w.h}, {w.coef});
        op_gu[i] = w_gu[i].gemm(b, qrows, 2 * g.n_ff, g.n_embd, {w.h2q, w.h2s}, {w.gate_up});
        /* The fold saves two launches and the bf16 round trip of the [T, n_ff] product; it is an
         * fp8a8 tile, so an MXFP4 or M2XFP4 gate_up takes the pair. */
        if (!w_gu[i].mx && !w_gu[i].m2x && !w_gu[i].w8h)
            op_gug[i] = decl_gated_fold(b, qrows, 2 * g.n_ff, g.n_embd, w_gu[i].w, w_gu[i].s,
                                        {w.h2q, w.h2s}, {w.ffq, w.ffs}, &gated_rows);
        op_dn[i] = w_dn[i].gemm(b, qrows, g.n_embd, g.n_ff, {w.ffq, w.ffs}, {w.h});
    }

    /* The convolutions. `block_size` is the query block: the shifted tap is masked at each block's
     * first row so one sequence's block does not convolve with the previous one's.
     *
     * One op per (layer, convolution) because the base kernel is a weight (see op_ck). Each op is
     * issued twice, prepare and finish, against the two halves of its [2, taps, n_embd] tensor. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_aconv[i] = rw(b, RAD_OP(b, "dflash_conv",
                            RAD_PARAMS(RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
                                       RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group),
                                       RAD_INT("NG", NG), RAD_INT("block_size", cfg.block),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_acb[i])),
                        {w.h, w.coef}, {w.h2});
        op_fconv[i] = rw(b, RAD_OP(b, "dflash_conv",
                            RAD_PARAMS(RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
                                       RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group),
                                       RAD_INT("NG", NG), RAD_INT("block_size", cfg.block),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fcb[i])),
                        {w.h, w.coef}, {w.h2});
    }
    /* The attention output is written and read inside every layer, between the two halves of the
     * attention convolution, but its two ops (op_attn, op_qat) are declared once ahead of the
     * layers. Planned from those alone, its range would end before the coefficients' begins and
     * the plan could lay one over the other. Read by the last convolution as well, it is live
     * wherever the coefficients are. */
    {
        const int64_t rows = cfg.short_rows > 0 ? cfg.short_rows : kShortBlock;
        block_lo = cfg.short_block && rows >= 2 && cfg.block > rows ? rows : 0;
    }
    for (int64_t l = 0; block_lo && l < g.n_layer; ++l) {
        const int i = (int)l;
        op_aconv_lo[i] = rw(b, RAD_OP(b, "dflash_conv",
                               RAD_PARAMS(RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
                                          RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group),
                                          RAD_INT("NG", NG), RAD_INT("block_size", block_lo),
                                          RAD_STR("dtype", g.dtype)),
                               RAD_WEIGHTS(w_acb[i])),
                           {w.h, w.coef}, {w.h2});
        op_fconv_lo[i] = rw(b, RAD_OP(b, "dflash_conv",
                               RAD_PARAMS(RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
                                          RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group),
                                          RAD_INT("NG", NG), RAD_INT("block_size", block_lo),
                                          RAD_STR("dtype", g.dtype)),
                               RAD_WEIGHTS(w_fcb[i])),
                           {w.h, w.coef}, {w.h2});
    }
    if (rad_op_reads(b, op_fconv[g.n_layer - 1], &w.attn, 1) < 0) return RAD_E_INVAL;
    if (block_lo && rad_op_reads(b, op_fconv_lo[g.n_layer - 1], &w.attn, 1) < 0) return RAD_E_INVAL;
    op_qrope = rope(qrows, g.n_head, w.q);
    op_krope = rope(qrows, g.n_head_kv, w.k);
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.k, w.v}, {});
    /* The query pass's q/k norms, rotaries and store as one launch a layer where a library
     * serves it: qkv_norm_rope_kv_store with v_norm 0 (v stored as it is), byte-identical to the
     * two rmsnorms, the two ropes and kv_store (docs/OPS.md), which stay declared beside it for
     * a library that does not. No YaRN (the fold takes none). */
    {
        /* Only for a shape a library serves: an op declared for one that resolves to no device
         * kernel is a hole that refuses startup (Program::device_holes), whether or not it is
         * issued, so the block asks within libr4d's span for the fold (head_dim 256 to 512 in
         * steps of 256, bf16 rows, either cache). Qwen3.8's DFlash2 drafter (head_dim 128) keeps
         * the chain. */
        const bool fold_shape = g.head_dim >= 256 && g.head_dim <= 512 && g.head_dim % 256 == 0 &&
                                !std::strcmp(g.dtype, "bf16");
        if (fold_shape && g.yarn.factor == 0.0f &&
            rad_op_declared(b, "qkv_norm_rope_kv_store")) {
            const int64_t bsz = rad_kv_block_size(b, kv);
            for (int64_t l = 0; l < g.n_layer; ++l) {
                const int i = (int)l;
                op_qkr[i] = RAD_OP(b, "qkv_norm_rope_kv_store",
                               RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                          RAD_INT("n_head", g.n_head),
                                          RAD_INT("n_head_kv", g.n_head_kv),
                                          RAD_F64("theta", g.theta), RAD_F64("eps", g.eps),
                                          RAD_F64("scale", g.rope_scale), RAD_STR("mode", "neox"),
                                          RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype),
                                          RAD_INT("v_norm", 0), RAD_INT("block_size", bsz),
                                          RAD_STR("kv_dtype", g.kv_dtype)),
                               RAD_WEIGHTS(w_qn[i], w_kn[i]));
                if (!op_qkr[i] || !rad_op_resolved(b, op_qkr[i])) {
                    for (int j = 0; j <= i; ++j) op_qkr[j] = 0;
                    break;
                }
                op_qkr[i] = rw(b, op_qkr[i], {w.q, w.k, w.v}, {w.q, w.k, w.v});
            }
            /* The context pass's keys the same way: no q (n_head 0, `q` passed as k's buffer and
             * `q_w` as k's norm, neither read), the context rows' slots. */
            for (int64_t l = 0; op_qkr[0] && l < g.n_layer; ++l) {
                const int i = (int)l;
                op_cqkr[i] = RAD_OP(b, "qkv_norm_rope_kv_store",
                                RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                           RAD_INT("n_head", 0), RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_F64("theta", g.theta), RAD_F64("eps", g.eps),
                                           RAD_F64("scale", g.rope_scale), RAD_STR("mode", "neox"),
                                           RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype),
                                           RAD_INT("v_norm", 0), RAD_INT("block_size", bsz),
                                           RAD_STR("kv_dtype", g.kv_dtype)),
                                RAD_WEIGHTS(w_kn[i], w_kn[i]));
                if (!op_cqkr[i] || !rad_op_resolved(b, op_cqkr[i])) {
                    for (int j = 0; j <= i; ++j) op_cqkr[j] = 0;
                    break;
                }
                op_cqkr[i] = rw(b, op_cqkr[i], {w.ck, w.cv}, {w.ck});
            }
        }
    }
    op_act = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.gate_up}, {w.ffn});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                      RAD_PARAMS(RAD_INT("world_size", g.world),
                                 RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                 RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                 RAD_INT("min_bytes", g.wire_min_bytes)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.x, w.h2}, {w.x});
    op_final = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});
    /* Each residual add and the norm after it as one launch where a library serves add_rmsnorm
     * (byte-identical to the pair: the norm is taken over the bf16 sum it stores), taken only
     * where every handle resolves (rad_op_resolved), all or none; the pairs declared beside it run
     * otherwise. */
    if (rad_op_declared(b, "add_rmsnorm")) {
        auto an = [&](rad_weight wt) -> rad_op {
            const rad_op h = RAD_OP(b, "add_rmsnorm",
                                    RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                               RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                               RAD_STR("dtype", g.dtype)),
                                    RAD_WEIGHTS(wt));
            return h && rad_op_resolved(b, h) ? rw(b, h, {w.x, w.h2}, {w.x, w.h}) : (rad_op)0;
        };
        bool all = true;
        for (int64_t l = 0; all && l < g.n_layer; ++l) {
            const int i = (int)l;
            op_afn[i] = an(w_fnorm[i]);
            if (l > 0) op_aan[i] = an(w_anorm[i]);
            all = op_afn[i] && (l == 0 || op_aan[i]);
        }
        op_afinal = all ? an(w_norm) : 0;
        if (!op_afinal)
            for (int i = 0; i < MAX_LAYERS; ++i) op_afn[i] = op_aan[i] = 0;
    }

    /* ---------------------------------------------------------------- the head and the selector
     *
     * The head runs over the mask rows, offsets 1..block-1 of each block: `steps` rows per
     * sequence, each predicting the token at its own position. The anchor row predicts nothing.
     * An off-by-one here only shows as collapsed acceptance; the output stays correct.
     *
     * The gather makes those rows contiguous, numbered s * steps + l as `dflash_select` reads
     * them. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.sel_h});
    /* LiLiCorr's head is the plugin's (rad_block_lilicorr.h), over sel_h. */
    if (cfg.lili) return RAD_OK;
    op_dquant = rw(b, RAD_OP(b, "quant_act_i8",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_INT("group", DF_W2_GROUP), RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.sel_h}, {w.dq, w.ds, w.dsum});
    op_dlogits = head_i8
        ? rw(b, RAD_OP(b, "gemm_nt_q",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("K", g.n_embd), RAD_INT("group", DF_W2_GROUP),
                           RAD_STR("dtype", "w8a8")),
                RAD_WEIGHTS(w_dh, w_dhs)),
             {w.dq, w.ds}, {w.dlog})
        : rw(b, RAD_OP(b, "gemm_nt_q",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("K", g.n_embd), RAD_INT("group", DF_W2_GROUP),
                           RAD_STR("dtype", "w2a8")),
                RAD_WEIGHTS(w_dh, w_dhs)),
             {w.dq, w.ds, w.dsum}, {w.dlog});
    /* Sharded, the top-R runs over this rank's plane, reports global ids and also writes the pair
     * plane the gather moves. Replicated, `vocab_off` is omitted: an optional parameter present
     * with value 0 is a different declaration from an absent one. */
    const bool rerank = w_lm && (dh_shard || g.world <= 1);   /* see op_drerank below */
    if (!rerank) top_w = cfg.sel_topk;
    op_dtopk = dh_shard
        ? rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", top_w), RAD_INT("vocab_off", g.vocab_off),
                           RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary, w.dpair})
        : rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", top_w), RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary});
    /* Exact rescoring when the target's head can be read row by row: the coarse head proposes its
     * top kRerankR, `logit_rerank` scores them against the target's head and re-sorts, and the
     * selector reads the best sel_topk, giving the candidates and unary logits the drafter was
     * trained against. On Qwen3.8-27B the 2-bit head's top-1 matches the exact head on 76% of
     * draft rows, the rescored one on all; this matches the 4-bit head's speed at half its bytes.
     *
     * Sharded, each rank rescores only ids it owns and the merge below ranks exact values. A
     * replicated head above one rank proposes ids other ranks own, so it keeps its own scores. */
    if (rerank) {
        op_drerank = RAD_OP(b, "logit_rerank",
                        RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("R", top_w),
                                   RAD_INT("n_vocab", head_rows(g)), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("vocab_off", g.vocab_off), RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(w_lm));
        if (!op_drerank) return RAD_E_INVAL;
        rad_buf rd[2] = { w.sel_h, w.cand };
        rad_buf wr[3] = { w.cand, w.unary, w.dpair };
        rad_op_reads (b, op_drerank, rd, 2);
        rad_op_writes(b, op_drerank, wr, dh_shard ? 3 : 2);
    }
    if (dh_shard) {
        /* Row-interleaved, so one draft position's pairs from every rank are adjacent, as
         * `row_topk_merge` reads them. */
        op_dgather = rw(b, RAD_OP(b, "all_gather",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_INT("numel", srows * top_w * 2),
                                      RAD_STR("dtype", "f32"),
                                      RAD_INT("row", top_w * 2)),
                           RAD_NOWEIGHTS),
                       {w.dpair}, {w.dgath});
        op_dmerge = rw(b, RAD_OP(b, "row_topk_merge",
                          RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("R", top_w),
                                     RAD_INT("world_size", g.world),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.dgath}, {w.cand, w.unary});
    }
    op_hp = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", cfg.sel_rank),
                             RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_selp)),
              {w.sel_h}, {w.hp});
    /* `anchor_stride` is `block`: the operand is the step's token_ids, `block` per sequence with
     * the anchor first.
     *
     * A tree (Config::tree): `dflash_tree` grows `steps` nodes best-first over the same lattice
     * and the same edge scores, writing each node's parent beside its token. Where no library
     * serves it the block falls back to the chain and says so; the caller reads `cfg.tree` after
     * declare to decide whether to tell the core (rad_drafter_tree). --draft-coupling samples one
     * path per sequence, so a coupled drafter proposes the chain. */
    if (cfg.tree && (w.couple_draws || w.couple_state)) {
        rad_note(b, "DFlash2: --draft-coupling samples a chain; the drafter proposes no tree");
        cfg.tree = false;
    }
    if (cfg.tree) {
        op_sel = RAD_OP(b, "dflash_tree",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", cfg.steps()),
                              RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                              RAD_INT("n_vocab", g.n_vocab_all),
                              RAD_INT("anchor_stride", cfg.block),
                              RAD_INT("nodes", cfg.nodes()), RAD_F64("temp", cfg.tree_temp),
                              RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_pred, w_succ));
        if (op_sel && rad_op_resolved(b, op_sel)) {
            rw(b, op_sel, {w.cand, w.unary, w.hp}, {w.tokens, w.parents});
        } else {
            rad_note(b, "DFlash2: no library serves dflash_tree here; the drafter proposes its "
                        "selector's chain");
            cfg.tree = false;
            op_sel = 0;
        }
    }
    if (!cfg.tree) op_sel = rw(b, RAD_OP(b, "dflash_select",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", cfg.steps()),
                              RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                              RAD_INT("n_vocab", g.n_vocab_all),
                              RAD_INT("anchor_stride", cfg.block), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_pred, w_succ)),
               {w.cand, w.unary, w.hp}, {w.tokens});
    if (block_lo && cfg.tree) {
        /* A tree over the short block's own lattice (the pass attends across its block, so it is
         * not the full pass's prefix): as many nodes as the block drafts, or the tree's own count
         * where that is fewer. */
        sel_lo_nodes = std::min<int64_t>(cfg.nodes(), block_lo - 1);
        op_sel_lo = RAD_OP(b, "dflash_tree",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", block_lo - 1),
                                 RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                                 RAD_INT("n_vocab", g.n_vocab_all),
                                 RAD_INT("anchor_stride", block_lo),
                                 RAD_INT("nodes", sel_lo_nodes), RAD_F64("temp", cfg.tree_temp),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_pred, w_succ));
        if (op_sel_lo && rad_op_resolved(b, op_sel_lo)) {
            rw(b, op_sel_lo, {w.cand, w.unary, w.hp}, {w.tokens, w.parents});
        } else {
            rad_note(b, "DFlash2: no library serves dflash_tree at the short block; the tree keeps "
                        "its own block under load");
            op_sel_lo = 0;
            block_lo = 0;
            sel_lo_nodes = 0;
        }
    } else if (block_lo)
        op_sel_lo = rw(b, RAD_OP(b, "dflash_select",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", block_lo - 1),
                                     RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                                     RAD_INT("n_vocab", g.n_vocab_all),
                                     RAD_INT("anchor_stride", block_lo), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(w_pred, w_succ)),
                      {w.cand, w.unary, w.hp}, {w.tokens});

    /* --draft-coupling: the walk again, sampled (abi/rad_spechub.h, rad_dflash_couple_step), over
     * the same lattice right after the greedy walk. A greedy request's row keeps the selector's
     * tokens; a sampled one gets x ~ Q at every position and the Q it was drawn from, which the
     * engine reads back and binds to the verify. The tokens stay global ids, so the engine's take
     * (ADV_BTAKE) is unchanged. */
    if (w.couple_draws || w.couple_state) {
        if (!w.couple_draws || !w.couple_state || cfg.steps() > RAD_MTP_COUPLE_MAX_ROUNDS ||
            cfg.sel_topk > RAD_DFLASH_COUPLE_MAX_K)
            return RAD_E_UNSUPPORTED;
        auto couple = [&](int64_t steps, int64_t stride) {
            return rw(b, RAD_OP(b, "dflash_couple_propose",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", steps),
                                    RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                                    RAD_INT("n_vocab", g.n_vocab_all),
                                    RAD_INT("anchor_stride", stride),
                                    RAD_INT("max_rounds", cfg.steps()),
                                    RAD_F64("q_temp", cfg.q_temp)),
                         RAD_WEIGHTS(w_pred, w_succ)),
                      {w.cand, w.unary, w.hp, w.couple_draws, w.tokens},
                      {w.tokens, w.couple_state});
        };
        op_couple = couple(cfg.steps(), cfg.block);
        if (!op_couple) return RAD_E_INVAL;
        if (block_lo) {
            op_couple_lo = couple(block_lo - 1, block_lo);
            if (!op_couple_lo) return RAD_E_INVAL;
        }
    }

    /* The reads beside the all-reduces, after every weight they read is declared. */
    if (op_ar && cfg.prefetch > 0)
        for (int64_t l = 0; l < g.n_layer; ++l) {
            const int i = (int)l;
            pfa_w[i][0] = w_fcp[i];
            pfa_w[i][1] = w_gu[i].w;
            pff_w[i][0] = l + 1 < g.n_layer ? w_acp[i + 1] : w_dh;
            pff_w[i][1] = l + 1 < g.n_layer ? w_q[i + 1].w : 0;
            for (int j = 0; j < 2; ++j) {
                op_pfa[i][j] = decl_prefetch(b, pfa_w[i][j], cfg.prefetch);
                op_pff[i][j] = decl_prefetch(b, pff_w[i][j], cfg.prefetch);
            }
            pfo_w[i]  = w_o[i].w;
            op_pfo[i] = decl_prefetch(b, pfo_w[i], cfg.prefetch);
        }

    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void Dflash2::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;

    /* As wide as the prefill chunk, one launch per op (the fp8a8 pair covers every M). */
    RAD_ISSUE_N(c, op_qsrc, T, brows(w.src, T), brows(w.srcq, T), brows(w.srcs, T));
    if (fc_rows) {
        RAD_ISSUE_N(c, op_cfc, T, brows(w.srcq, T), brows(w.srcs, T), RAD_W(w_fc.w), RAD_W(w_fc.s),
                    brows(w.cpart, T), RAD_NONE, w_fc.bref());
        RAD_ISSUE_N(c, op_cgath, T * g.n_embd, bcol(w.cpart, fc_off, fc_share, T),
                    brows(w.c, T));
    } else {
        RAD_ISSUE_N(c, op_cfc, T, brows(w.srcq, T), brows(w.srcs, T), RAD_W(w_fc.w), RAD_W(w_fc.s),
                    brows(w.c, T), RAD_NONE, w_fc.bref());
    }
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.c, T));
    RAD_ISSUE_N(c, op_qc, T, brows(w.c, T), brows(w.cq, T), brows(w.cs, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ISSUE_N(c, op_ck[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.ck, T), RAD_NONE, w_k[i].bref());
        RAD_ISSUE_N(c, op_cv[i], T, brows(w.cq, T), brows(w.cs, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.cv, T), RAD_NONE, w_v[i].bref());
        if (op_cqkr[i]) {
            RAD_ISSUE_N(c, op_cqkr[i], T, brows(w.ck, T), brows(w.ck, T), brows(w.cv, T),
                        praw(batch->positions, RAD_I32, T), RAD_W(w_kn[i]), RAD_W(w_kn[i]),
                        praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
            continue;
        }
        RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv,
                    brows(w.ck, T), RAD_W(w_kn[i]), brows(w.ck, T));
        RAD_ISSUE_N(c, op_crope, T, brows(w.ck, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_cstore, T, brows(w.ck, T), brows(w.cv, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void Dflash2::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t NG = cfg.ng(g.n_embd);
    const int64_t half = cfg.taps * NG;      /* prepare's coefficients, then finish's */

    /* The block this pass was staged at: every drafting sequence carries the same number of rows,
     * the drafter's own block or its short one (rad_drafter_short_block). */
    const int64_t blk = batch->n_seq > 0 && T % batch->n_seq == 0 ? T / batch->n_seq : 0;
    const bool lo = block_lo > 0 && blk == block_lo;
    if (!lo && blk != cfg.block) {
        rad_step_fail(c, "dflash2: a query pass staged at a block this drafter has no ops for");
        return;
    }
    const rad_op* aconv = lo ? op_aconv_lo : op_aconv;
    const rad_op* fconv = lo ? op_fconv_lo : op_fconv;

    RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.x, T));
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.x, T), RAD_NONE);

    /* x += h2 waits for the norm after it, which takes it in (add_rmsnorm) where declared */
    bool add_pending = false;
    auto norm = [&](rad_op plain, rad_op folded, rad_weight wt) {
        if (add_pending && folded) {
            RAD_ISSUE_N(c, folded, T, brows(w.x, T), brows(w.h2, T), RAD_W(wt), brows(w.h, T));
        } else {
            if (add_pending) RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h2, T), brows(w.x, T));
            RAD_ISSUE_N(c, plain, T, brows(w.x, T), RAD_W(wt), brows(w.h, T));
        }
        add_pending = false;
    };

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;

        /* ---- attention, wrapped in the two halves of one convolution */
        norm(op_anorm[i], op_aan[i], w_anorm[i]);
        RAD_ISSUE_N(c, op_acoef[i], T, brows(w.h, T), RAD_W(w_acp[i]), brows(w.coef, T));
        RAD_ISSUE_N(c, aconv[i], T, brows(w.h, T), bcol(w.coef, 0, half, T),
                    wrow_slice(w_acb[i], 0, 1, cfg.taps * g.n_embd), brows(w.h2, T));

        RAD_ISSUE_N(c, op_qh2, T, brows(w.h2, T), brows(w.h2q, T), brows(w.h2s, T));
        RAD_ISSUE_N(c, op_qp[i], T, brows(w.h2q, T), brows(w.h2s, T),
                    RAD_W(w_q[i].w), RAD_W(w_q[i].s), brows(w.q, T), RAD_NONE, w_q[i].bref());
        RAD_ISSUE_N(c, op_kp[i], T, brows(w.h2q, T), brows(w.h2s, T),
                    RAD_W(w_k[i].w), RAD_W(w_k[i].s), brows(w.k, T), RAD_NONE, w_k[i].bref());
        RAD_ISSUE_N(c, op_vp[i], T, brows(w.h2q, T), brows(w.h2s, T),
                    RAD_W(w_v[i].w), RAD_W(w_v[i].s), brows(w.v, T), RAD_NONE, w_v[i].bref());
        prefetch_beside(c, op_pfo[i], pfo_w[i], T);
        if (op_qkr[i]) {
            RAD_ISSUE_N(c, op_qkr[i], T, brows(w.q, T), brows(w.k, T), brows(w.v, T),
                        praw(batch->positions, RAD_I32, T), RAD_W(w_qn[i]), RAD_W(w_kn[i]),
                        praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
        } else {
            RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.q, T));
            RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv,
                        brows(w.k, T), RAD_W(w_kn[i]), brows(w.k, T));
            RAD_ISSUE_N(c, op_qrope, T, brows(w.q, T), praw(batch->positions, RAD_I32, T));
            RAD_ISSUE_N(c, op_krope, T, brows(w.k, T), praw(batch->positions, RAD_I32, T));
            RAD_ISSUE_N(c, op_store, T, brows(w.k, T), brows(w.v, T),
                        praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
        }
        if (cfg.lili)
            RAD_ISSUE_N(c, op_attn_sink[i], batch->max_q_len,
                        brows(w.q, T), kv_cache(kv, first_layer + i),
                        praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                        praw(used, RAD_I32, batch->n_seq),
                        RAD_NONE, RAD_NONE,
                        praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                        RAD_W(w_sink[i]), brows(w.attn, T));
        else
            RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                        brows(w.q, T), kv_cache(kv, first_layer + i),
                        praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                        praw(used, RAD_I32, batch->n_seq),
                        RAD_NONE, RAD_NONE,
                        praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                        brows(w.attn, T));
        RAD_ISSUE_N(c, op_qat, T, brows(w.attn, T), brows(w.atq, T), brows(w.ats, T));
        RAD_ISSUE_N(c, op_op[i], T, brows(w.atq, T), brows(w.ats, T),
                    RAD_W(w_o[i].w), RAD_W(w_o[i].s), brows(w.h, T), RAD_NONE, w_o[i].bref());
        for (int j = 0; j < 2; ++j) prefetch_beside(c, op_pfa[i][j], pfa_w[i][j], T);
        /* Reduce before the closing convolution. It is linear with coefficients from the
         * replicated norm, so either order gives the same result; reducing first keeps x's
         * update agreed across ranks. */
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        RAD_ISSUE_N(c, aconv[i], T, brows(w.h, T), bcol(w.coef, half, half, T),
                    wrow_slice(w_acb[i], 1, 1, cfg.taps * g.n_embd), brows(w.h2, T));
        add_pending = true;

        /* ---- the feed-forward, wrapped in the other convolution */
        norm(op_fnorm[i], op_afn[i], w_fnorm[i]);
        RAD_ISSUE_N(c, op_fcoef[i], T, brows(w.h, T), RAD_W(w_fcp[i]), brows(w.coef, T));
        RAD_ISSUE_N(c, fconv[i], T, brows(w.h, T), bcol(w.coef, 0, half, T),
                    wrow_slice(w_fcb[i], 0, 1, cfg.taps * g.n_embd), brows(w.h2, T));
        RAD_ISSUE_N(c, op_qh2, T, brows(w.h2, T), brows(w.h2q, T), brows(w.h2s, T));
        if (op_gug[i] && T <= gated_rows) {
            RAD_ISSUE_N(c, op_gug[i], T, brows(w.h2q, T), brows(w.h2s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.ffq, T), brows(w.ffs, T),
                        RAD_NONE);
        } else {
            RAD_ISSUE_N(c, op_gu[i], T, brows(w.h2q, T), brows(w.h2s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.gate_up, T), RAD_NONE, w_gu[i].bref());
            RAD_ISSUE_N(c, op_act, T, brows(w.gate_up, T), brows(w.ffn, T));
            RAD_ISSUE_N(c, op_qff, T, brows(w.ffn, T), brows(w.ffq, T), brows(w.ffs, T));
        }
        RAD_ISSUE_N(c, op_dn[i], T, brows(w.ffq, T), brows(w.ffs, T),
                    RAD_W(w_dn[i].w), RAD_W(w_dn[i].s), brows(w.h, T), RAD_NONE, w_dn[i].bref());
        for (int j = 0; j < 2; ++j) prefetch_beside(c, op_pff[i][j], pff_w[i][j], T);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        RAD_ISSUE_N(c, fconv[i], T, brows(w.h, T), bcol(w.coef, half, half, T),
                    wrow_slice(w_fcb[i], 1, 1, cfg.taps * g.n_embd), brows(w.h2, T));
        add_pending = true;
    }

    norm(op_final, op_afinal, w_norm);

    /* The predicting rows, `steps` per sequence, named by the scheduler through out_ids. */
    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_gather, S, RAD_B(w.h), praw(batch->out_ids, RAD_I32, S), brows(w.sel_h, S));
    if (cfg.lili) return;   /* the plugin's correlator takes sel_h from here */

    /* `dsum` is group-major, [n_embd/group][ceil(M/16)*16], so it has no row slice; pass it
     * whole. */
    RAD_ISSUE_N(c, op_dquant, S, brows(w.sel_h, S), brows(w.dq, S), brows(w.ds, S), RAD_B(w.dsum));
    /* The trailing operand is mxfp4's optional reference, absent here. Operands are positional,
     * so an absent optional one is RAD_NONE, not a shorter list. The 8-bit head is symmetric and
     * takes no asum. */
    RAD_ISSUE_N(c, op_dlogits, S, brows(w.dq, S), brows(w.ds, S), RAD_W(w_dh), RAD_W(w_dhs),
                brows(w.dlog, S), head_i8 ? RAD_NONE : RAD_B(w.dsum), RAD_NONE);
    /* RAD_NONE fills the unused `pairs` slot. Sharded, this rank's top-R are only candidates and
     * the merge overwrites `cand`/`unary`. */
    if (op_dmerge) {
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    brows(w.dpair, S));
        if (op_drerank)
            RAD_ISSUE_N(c, op_drerank, S, brows(w.sel_h, S), RAD_W(w_lm), brows(w.cand, S),
                        brows(w.cand, S), brows(w.unary, S), brows(w.dpair, S));
        RAD_ISSUE(c, op_dgather, brows(w.dpair, S), brows(w.dgath, S));
        RAD_ISSUE_N(c, op_dmerge, S, brows(w.dgath, S), brows(w.cand, S), brows(w.unary, S));
    } else {
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    RAD_NONE);
        if (op_drerank)
            RAD_ISSUE_N(c, op_drerank, S, brows(w.sel_h, S), RAD_W(w_lm), brows(w.cand, S),
                        brows(w.cand, S), brows(w.unary, S), RAD_NONE);
    }

    RAD_ISSUE_N(c, op_hp, S, brows(w.sel_h, S), RAD_W(w_selp), brows(w.hp, S));
    /* A tree's outputs as views exactly its nodes wide, at the buffers' row pitch: at the short
     * block it grows fewer nodes than a row holds, and the proposals are read at the full pitch. */
    const int64_t nn = lo ? sel_lo_nodes : cfg.nodes();
    if (cfg.tree)
        RAD_ISSUE_N(c, lo ? op_sel_lo : op_sel, batch->n_seq, bcol(w.cand, 0, cfg.sel_topk, S),
                    bcol(w.unary, 0, cfg.sel_topk, S), brows(w.hp, S),
                    praw(batch->token_ids, RAD_I32, T), RAD_W(w_pred), RAD_W(w_succ),
                    bcol(w.tokens, 0, nn, batch->n_seq), bcol(w.parents, 0, nn, batch->n_seq));
    else
        RAD_ISSUE_N(c, lo ? op_sel_lo : op_sel, batch->n_seq, bcol(w.cand, 0, cfg.sel_topk, S),
                    bcol(w.unary, 0, cfg.sel_topk, S), brows(w.hp, S),
                    praw(batch->token_ids, RAD_I32, T), RAD_W(w_pred), RAD_W(w_succ),
                    brows(w.tokens, batch->n_seq));
    if (op_couple)
        RAD_ISSUE_N(c, lo ? op_couple_lo : op_couple, batch->n_seq,
                    bcol(w.cand, 0, cfg.sel_topk, S), bcol(w.unary, 0, cfg.sel_topk, S),
                    brows(w.hp, S), praw(batch->token_ids, RAD_I32, T), RAD_W(w_pred),
                    RAD_W(w_succ), praw(batch->seq_ids, RAD_I32, batch->n_seq),
                    brows(w.couple_draws, batch->n_seq), brows(w.tokens, batch->n_seq),
                    RAD_B(w.couple_state));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_DFLASH2_H */
