/* rad_gemma4_dflash.h -- the DFlash (v1) block-diffusion drafter, as z-lab trains it for Gemma 4,
 * and what both Gemma 4 plugins need to carry it: arch/gemma4 (the 26B-A4B and the dense sizes,
 * from bf16) and arch/gemma4_bf16 (the dense 31B at w4a8 or w8a8).
 *
 * ============================== WHAT IT IS ==============================
 *
 * rad_block_dflash2.h without DFlash2's additions (the per-layer dynamic convolutions around
 * attention and the feed-forward, and the candidate selector). The original DFlash is a small
 * Qwen3-shaped transformer that denoises one anchor token plus `block - 1` mask tokens in one
 * forward and takes every position's argmax from the target's lm_head.
 *
 *     context pass, one row a committed token:
 *         c = hidden_norm(fc(concat of the trunk's residual stream at target_layer_ids)))
 *         for each layer:  k = k_norm(k_proj(c)) then rope;  v = v_proj(c);  kv_store(k, v)
 *
 *     query pass, `block` rows a sequence:
 *         x = embed(anchor, mask, ...) * sqrt(hidden)         <- Gemma's embedding scale, see below
 *         for each layer:
 *             h = input_layernorm(x)
 *             q, k, v = q_proj(h), k_proj(h), v_proj(h);  q_norm, k_norm;  rope;  kv_store
 *             x += o_proj(attention(q, cache))        sliding layers causal over a 2048 window,
 *                                                     the full layer non-causal over everything
 *             x += down(silu(gate(h')) * up(h')),  h' = post_attention_layernorm(x)
 *         x = norm(x)
 *         over the mask rows: logits = lm_head(x) [the target's];  draft = argmax
 *
 * z-lab publishes one for gemma-4-26B-A4B-it (2816 wide, 32 heads, taps 1 6 11 17 22 27 of 30) and
 * one for gemma-4-31B-it (5376 wide, 64 heads / 8 kv x 128, ffn 10752, taps 1 12 23 35 46 57 of
 * 60), both five layers, four sliding and one full, at block 16. Nothing below is per size.
 *
 * ============================== THE THREE GEMMA-SPECIFIC FACTS ==============================
 *
 * From vLLM PR #41703. Getting any of them wrong costs only acceptance, since every draft is
 * verified against the trunk.
 *
 *   The embedding is scaled. Gemma multiplies its token embedding by sqrt(hidden_size), stored as
 *   bf16 (53.0 at 2816, 73.5 at 5376), and the drafter was trained through the target's embedding
 *   module, which applies it. `embed_lookup`'s `mult` carries it; embed_scale() below is the value.
 *
 *   The sliding layers are causal and the full layer is not. The checkpoint's `layer_types` names
 *   four sliding layers and one full one, and the PR's test pins `causal True` on the sliding
 *   layers and `False` on the full one, so this block declares two attention ops and issues the one
 *   each layer's type names. A config that states `dflash_config.causal` (NVIDIA's E4B drafter:
 *   false, all five layers sliding) sets the sliding layers' causality.
 *
 *   The logits are soft-capped, which is deliberately ignored: the draft is an argmax, and
 *   cap * tanh(x / cap) is strictly increasing.
 *
 * ============================== THE HEAD ==============================
 *
 * The target's own head, read in bf16 by an ordinary `gemm_nt` over the gathered mask rows. Gemma
 * ties it to the embedding, so it is the weight the trunk's `embed_lookup` reads and costs no extra
 * bytes. It is vocab-sharded with the trunk: the top-1 runs over this rank's slice and the ranks'
 * candidates merge exactly (`row_topk` at R = 2 -> `all_gather` -> `pair_argmax`; R = 2 because the
 * gather moves 16-byte rows, see kTopR).
 *
 * The bf16 head (262144 x 2816 x 2 bytes) is the widest read on the draft path, about 566 us of a
 * 26B-A4B decode step at tp 4 on 4x R9700. So where the container holds an 8-bit copy of the table
 * under its own name, `dflash.draft_head.weight` (one checkpoint tensor, two logical weights, as in
 * rad_block_dflash2.h), made by one of the recipe lines
 *
 *     dflash.draft_head.weight  rtn codes=fp8_e4m3 block=128x128 scale=bf16     (block fp8)
 *     dflash.draft_head.weight  rtn codes=i8 group=128 scale=f16 clamp=sym      (int8, w8a8)
 *
 * the head is `gemm_nt_q` over that copy, half the bytes, with the argmax over its coarser logits.
 * Drafts are still verified, so the copy can change proposals but never accepted text. Without the
 * copy, or in another encoding, the bf16 table is read. The copy costs half the table's bytes again
 * (176 MiB per rank at tp 4 on the 26B, 352 MiB on the 31B). There is no rerank against the bf16
 * table: DFlash2's rerank needs the target's head in fp8 or int8, which a tied Gemma head is not.
 *
 * ============================== THE LINEARS ARE 8-BIT ==============================
 *
 * As in DFlash2, since the drafter is read whole on every decode step: the container must be
 * converted with a recipe that makes the seven linears (fc, and six a layer) 8-bit, on one of two
 * activation grids, which the container's choice for `fc` decides:
 *
 *   E4M3   block fp8 (`codes=fp8_e4m3 block=128x128 scale=bf16`) or OCP-MXFP4, read at fp8a8 /
 *          mxfp4a8 against an E4M3 activation with an f32 scale per (row, 128): quant_act_fp8, or
 *          rmsnorm_quant_fp8 where the input is a norm.
 *   int8   int8 codes with an f16 scale per (row, 128 columns) (`codes=i8 group=128 scale=f16
 *          clamp=sym`), the Q8 grid gemma4_bf16's trunk takes at w8a8, read by gemm_nt_q at w8a8
 *          against quant_act_i8's activation: int8 in the GEMM's fragment order and one f32 scale
 *          per row, or rmsnorm_quant_i8 where the input is a norm.
 *
 * The two cannot mix within the drafter: one quantised activation feeds several linears (q, k and v
 * read one pair), so every linear must read the grid its producer writes. declare() refuses a bf16
 * linear, and a linear on the other grid, by name rather than letting the loader fail on a view.
 */
#ifndef RAD_GEMMA4_DFLASH_H
#define RAD_GEMMA4_DFLASH_H

#include "rad_arch.h"
#include "rad_fp8.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gemma4 {

using namespace rad::arch;

/* ============================================================== what both plugins share */

/* The embedding scale as the reference applies it: Gemma4TextScaledWordEmbedding multiplies the
 * looked-up rows by `embed_scale.to(weight.dtype)`, sqrt(hidden) rounded to bf16 (53.0 at 2816, not
 * 53.066; 73.5 at 5376, not 73.32). The trunk, the assistant head and the drafter all read the
 * embedding through that module, so all three take this value. */
inline double embed_scale(int64_t hidden) {
    return rad_bf16_to_f32(rad_f32_to_bf16((float)std::sqrt((double)hidden)));
}

/* prefetch where a library serves it: the 31B trunk reading the next GEMM's leading weight bytes
 * into the on-die cache beside the all-reduce in front of that GEMM, at decode heights (on lane 0
 * as a chained read, or on the second lane: rad_arch.h prefetch_lane0), and the DFlash2 drafter's
 * reads likewise. It writes nothing the model reads. Each read takes a weight's first 20 MB: on 4x
 * R9700 with the reads on lane 0, 16 and 24 MB decoded 1-2% slower and 12 MB about 6%. */
constexpr int64_t kPrefetchBytes = (int64_t)20 << 20;

/* down_proj's read, the trunk GEMM no other read covers: its first 9 MB, beside the attention, in
 * place of a read of o_proj there (about 3.5% faster on the 31B at tp 4 on 4x R9700; 10-20 MB of
 * down_proj were slower, as were two reads beside one reduction). The reads beside a reduction
 * are one queue, and a longer one beside the attention pushes gate_up's read past the reduction
 * it should hide in. */
constexpr int64_t kPrefetchDownBytes = (int64_t)9 << 20;

/* Block fp8 as libr4d's fp8a8 GEMMs read it: fp8_e4m3 codes, a bf16 scale per 128x128 block, no
 * transform (a recipe's `rtn codes=fp8_e4m3 block=128x128 scale=bf16`). */
inline bool block_fp8_128(const RadEncoding& e) {
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return rad_enc_is(&e, "affine") && e.n_planes == 2 && !e.transform[0] &&
           e.plane[0].dtype == RAD_F8E4M3 && sc && sc->dtype == RAD_BF16 &&
           sc->block[0] == RAD_FP8_BLOCK && sc->block[1] == RAD_FP8_BLOCK;
}

/* Int8 on the w8a8 grid: i8 codes and an f16 scale per [1 x 128], no transform (a recipe's `rtn
 * codes=i8 group=128 scale=f16 clamp=sym`), what gemm_nt_q reads at w8a8. Not rad_fp8.h's
 * linear_is_w8, which is the i8a8 grid's bf16 scale over a per-(row, 128) activation. */
inline bool linear_w8a8(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 2 || e.transform[0]) return false;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return e.plane[0].dtype == RAD_I8 && e.plane[0].block[1] == 1 && sc && sc->dtype == RAD_F16 &&
           sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK;
}

/* Which drafter the container carries under `draft.`: 1 for DFlash (v1), 2 for DFlash2 (which is
 * rad_block_dflash2.h's, not this block), 0 for none or a draft that is not a DFlash. */
inline int dflash_kind(const RadModelMeta* meta) {
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (!da || !std::strstr(da, "DFlash")) return 0;
    return std::strstr(da, "DFlash2") ? 2 : 1;
}

/* The block a DFlash checkpoint was trained at, 0 where it states none. v1 states it at the top
 * level; a SpecForge config nests it under dflash_config. */
inline long long dflash_block(const RadModelMeta* meta) {
    long long bs = rad_meta_geti(meta, "draft.block_size", 0);
    if (bs <= 0) bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    return bs > 0 ? bs : 0;
}

/* DFlash proposes its whole block in one pass, so its depth is the block it was trained at less
 * the anchor, and nothing else: 15 for z-lab's Gemma 4 drafters (block_size 16). True when the
 * container carries one and `out` names its depth. Chained launches pay with it (the 26B-A4B's
 * speculative step is 1% faster on 4x R9700), so the probe asks for them
 * (RadArchProbe::chain_pays). */
inline bool dflash_probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (dflash_kind(meta) != 1) return false;
    const long long bs = dflash_block(meta);
    if (bs <= 1) return false;
    out->draft_depth = (int)(bs - 1);
    out->draft_depth_fixed = 1;
    out->chain_pays = 1;
    return true;
}

/* The same for a DFlash2 drafter, for a plugin that serves rad_block_dflash2.h (the 31B's): its
 * anchor predicts nothing either, so the depth is the trained block less one, 7 for RedHatAI's
 * gemma-4-31B-it-speculator.dflash2 (block_size 8). Chained launches pay with it (the 31B's
 * speculative step is 13% faster on 4x R9700), so the probe asks for them. */
inline bool dflash2_probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (dflash_kind(meta) != 2) return false;
    const long long bs = dflash_block(meta);
    if (bs <= 1) return false;
    out->draft_depth = (int)(bs - 1);
    out->draft_depth_fixed = 1;
    out->chain_pays = 1;
    /* Its proposals verified as a tree of the same node count (dflash_tree, rad_drafter_tree): on
     * 4x R9700 +7% single-stream decode and +6-9% aggregate at 1-16 users over the selector's
     * chain (BetterBench, thinking off; on two cards +4% and +6-12%). */
    out->tree_pays = 1;
    return true;
}

/* A DFlash2 drafter's trunk taps, as the layers whose output the trunk copies into `dflash_src`.
 *
 * vLLM's speculators format names them `aux_hidden_state_layer_ids`, counted as vLLM captures
 * them: id i is the residual stream entering layer i, so the output of layer i - 1 (id n_layer,
 * which `--include-last-layer` emits, is the last layer's output). z-lab's `target_layer_ids` count
 * the layer whose output is taken. RedHatAI's 1 17 29 47 59 are this trunk's layers 0 16 28 46
 * 58. Id 0 would be the embedding, which no layer outputs, and is refused. */
inline int dflash2_taps(const RadModelMeta* meta, const RadBuildCtx* ctx, int64_t n_layer, int* out,
                        int max) {
    if (ctx->max_spec <= 0 || dflash_kind(meta) != 2) return 0;
    const char* aux = rad_meta_gets(meta, "draft.aux_hidden_state_layer_ids", nullptr);
    int n = 0;
    if (aux) {
        int raw[16];
        const int m = parse_layer_ids(aux, n_layer + 1, raw, max < 16 ? max : 16);
        for (int i = 0; i < m; ++i) {
            if (raw[i] < 1) {
                fprintf(stderr, "radiance: gemma4: the DFlash2 drafter taps aux hidden state 0, the "
                                "embedding, which no trunk layer outputs\n");
                return RAD_E_UNSUPPORTED;
            }
            out[n++] = raw[i] - 1;
        }
    } else {
        const char* ids = rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr);
        if (!ids) ids = rad_meta_gets(meta, "draft.target_layer_ids", nullptr);
        n = parse_layer_ids(ids, n_layer, out, max);
    }
    if (n <= 0) {
        fprintf(stderr, "radiance: gemma4: the DFlash2 drafter names no trunk layer this model has; "
                        "it is paired with a different target\n");
        return RAD_E_UNSUPPORTED;
    }
    return n;
}

/* ============================================================== the block */

struct Dflash1 {
    enum { MAX_LAYERS = 8, MAX_TAPS = 8 };

    struct Config {
        int64_t n_taps = 0;      /* how many trunk layers `fc` concatenates */
        int64_t block  = 0;      /* query rows per sequence: the anchor plus block - 1 masks */
        int64_t window = 0;      /* the sliding layers' window; 0 for none */
        int     causal = 1;      /* the sliding layers: causal (z-lab) or not (dflash_config.causal) */
        int     full[MAX_LAYERS] = {};   /* 1: a full-attention (non-causal, unwindowed) layer */
        double  escale = 1.0;    /* the target's embedding scale, rounded as the target holds it */
        int64_t kv_rep = 1;      /* how many times the cache holds each kv head (kv_rep_for) */
        bool    qrow = false;    /* attention at the drafter's own gqa via attn_paged_qrow */
        int64_t steps() const { return block - 1; }
    };

    /* The attention's queries per kv head. libr4d serves the drafter's head-128 attention, causal
     * over a window and non-causal alike, on one kernel that takes 4 queries per kv head and
     * q_len * gqa <= 64, so a block of 16 at 4. The 26B's drafter is gqa 4. The 31B's is 64 heads
     * over 8 kv heads, gqa 8 at every rank count, which no attn_paged kernel serves at a block of
     * 16: unless attn_paged_qrow does (qrow_serves), its cache holds each kv head twice, and
     * attention sees 4 queries per copy. That is exact (query head q still reads kv head q / 8),
     * and costs a copy of k and v per head and layer and a cache twice the size, a few MiB at the
     * drafter's width. */
    static constexpr int64_t kAttnGqa = 4;
    static int64_t kv_rep_for(const Geom& g) {
        const int64_t gqa = g.gqa();
        return gqa > kAttnGqa && gqa % kAttnGqa == 0 ? gqa / kAttnGqa : 1;
    }
    /* libr4d also serves gqa 8 at head_dim 128 under attn_paged_qrow: that kernel with each kv head
     * read by two query groups, so the cache holds each head once and nothing is copied. No host
     * library declares attn_paged_qrow, so it resolves only to a card's kernel; asked first, it
     * decides the cache's head count (declare_drafter). The asking ops take no buffers and are not
     * issued; the attention the drafter issues is declared after its KV group, which takes its
     * block size from it. */
    static bool qrow_serves(RadBuilder* b, const Geom& g, const Config& c) {
        bool any_full = false, any_swa = false;
        for (int64_t l = 0; l < g.n_layer; ++l) (c.full[l] ? any_full : any_swa) = true;
        auto ask = [&](int causal, int64_t window) {
            const rad_op h = RAD_OP(b, "attn_paged_qrow",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, c.block), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", causal),
                                RAD_INT("window", window), RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS);
            return h != RAD_NULL_HANDLE && rad_op_resolved(b, h);
        };
        return (!any_swa || ask(1, c.window)) && (!any_full || ask(0, 0));
    }

    /* What the trunk hands the drafter (declare_drafter). */
    struct Trunk {
        int64_t     n_taps = 0;
        double      escale = 1.0;      /* embed_scale(hidden) */
        rad_buf     src = 0;           /* [max_tok, n_taps * n_embd] bf16: the taps, in tap order */
        rad_weight  w_tok = 0;         /* the trunk's tied table, row-sharded with the vocabulary */
        const char* tok_src = nullptr; /* its checkpoint name, which the head's 8-bit copy maps */
        int         base_layer = 0;    /* the drafter's first absolute layer, past the trunk's */
    };

    struct Wire {
        /* the context pass */
        rad_buf src = 0, srcq = 0, srcs = 0;   /* [max_tok, n_taps*n_embd] bf16 | codes | f32 */
        rad_buf c = 0, cq = 0, cs = 0;         /* [max_tok, n_embd] */
        rad_buf ck = 0, cv = 0;                /* [max_tok, kv_dim] -- see rad_block_dflash2.h */
        rad_buf ck4 = 0, cv4 = 0;              /* [max_tok, kv_dim * kv_rep]: the cache's copies */
        /* the query pass */
        rad_buf x = 0, h = 0, hq = 0, hs = 0;  /* [qrows, n_embd] */
        rad_buf q = 0, k = 0, v = 0;           /* [qrows, q_dim | kv_dim | kv_dim] */
        rad_buf k4 = 0, v4 = 0;                /* [qrows, kv_dim * kv_rep] */
        rad_buf attn = 0, atq = 0, ats = 0;    /* [qrows, q_dim] */
        rad_buf gate_up = 0;                   /* [qrows, 2 * n_ff] */
        rad_buf ffn = 0, ffq = 0, ffs = 0;     /* [qrows, n_ff] */
        /* the head, one row per draft position */
        rad_buf sel_h = 0;                     /* [srows, n_embd] */
        rad_buf selq = 0, sels = 0;            /* [srows, n_embd] codes | f32, the 8-bit head's */
        rad_buf dlog = 0;                      /* [srows, n_vocab on this rank] bf16 */
        rad_buf cand = 0, unary = 0;           /* [srows, kTopR] i32 | f32 -- this rank's best */
        rad_buf dpair = 0, dgath = 0;          /* [srows, kTopR, 2] | [srows, world * kTopR, 2] f32 */
        rad_buf tokens = 0;                    /* [srows, 1] i32: the draft, pitch `steps` */
    };

    /* One of the drafter's linears, in the grid the container holds it (see the file header): block
     * fp8 or MXFP4 through DenseFP8, or int8 at w8a8 -- codes and an f16 scale per [1 x 128] as two
     * views of one stored weight, the w4a8 grid's layout at twice the radix. `w` and `s` are the
     * GEMM's two weight operands either way, and `bref()` its trailing one. */
    struct Lin {
        rad_weight w = 0, s = 0;
        bool       i8 = false;
        DenseFP8   f{};

        int declare(RadBuilder* b, Names& nm, const char* base, int64_t N, int64_t K, int access,
                    int shard, RadWeightGroup grp, std::initializer_list<const char*> src,
                    std::initializer_list<int64_t> row_parts = {});
        rad_op gemm(RadBuilder* b, int64_t hi, int64_t N, int64_t K,
                    std::initializer_list<rad_buf> reads,
                    std::initializer_list<rad_buf> writes) const {
            if (!i8) return f.gemm(b, hi, N, K, reads, writes);
            return rw(b, RAD_OP(b, "gemm_nt_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "w8a8")),
                          RAD_WEIGHTS(w, s)),
                      reads, writes);
        }
        RadOperand bref() const { return i8 ? RAD_NONE : f.bref(); }
    };

    Geom    g{};          /* the drafter's geometry */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;
    int64_t qrows = 0, crows = 0, srows = 0, gated_rows = 0;
    rad_kvgroup kv = 0;
    bool    sharded = false;
    /* The activation grid: false is E4M3 with a scale per (row, 128), true int8 with a scale per
     * row. Read off `fc`'s encoding; every other linear has to agree (see the file header). */
    bool    i8 = false;

    Lin        w_fc{};
    rad_weight w_hnorm = 0, w_norm = 0;
    rad_weight w_tok = 0;                     /* the target's table: embedding and head */
    Lin        w_dh{};                        /* its 8-bit copy, the head where it is stored */
    bool       q_head = false;                /* the head reads w_dh */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    Lin        w_q[MAX_LAYERS]{}, w_k[MAX_LAYERS]{}, w_v[MAX_LAYERS]{}, w_o[MAX_LAYERS]{};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {};
    Lin        w_gu[MAX_LAYERS]{}, w_dn[MAX_LAYERS]{};

    rad_op op_qsrc = 0, op_cfc = 0, op_cnorm = 0, op_qc = 0, op_crope = 0, op_cstore = 0;
    /* the kv heads' copies into the cache's layout (kv_rep > 1): k and v, context and query pass */
    rad_op op_rep_ck = 0, op_rep_cv = 0, op_rep_k = 0, op_rep_v = 0;
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};

    rad_op op_embed = 0, op_embed_ar = 0, op_qh = 0, op_qat = 0, op_qff = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {};
    /* The residual add, norm and quantiser in one launch where `rmsnorm_quant_fp8` (on the int8
     * grid `rmsnorm_quant_i8`) is served: op_anq[i] for the attention input (layer 0's without the
     * add) and op_fnq[i] for the feed-forward input. Zero where not served; the `add` / `rmsnorm` /
     * quantiser chain is issued instead. */
    rad_op op_anq[MAX_LAYERS] = {}, op_fnq[MAX_LAYERS] = {};
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {}, op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {};
    rad_op op_gu[MAX_LAYERS] = {}, op_gug[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn_swa = 0, op_attn_full = 0;
    rad_op op_act = 0, op_ar = 0, op_add = 0, op_final = 0;
    rad_op op_gather = 0, op_qsel = 0, op_head = 0, op_topk = 0, op_dgather = 0, op_pick = 0;

    /* Candidates each rank contributes under a sharded head. The draft needs one, but the gather
     * moves whole 16-byte words per row and an (int32 id, f32 value) pair is 8 bytes, so R = 1 is
     * refused at launch (as rad_block_mtp_hc.h states). Two pairs make one word; `pair_argmax`
     * reduces the world * 2 of them to the global top-1. */
    static constexpr int64_t kTopR = 2;

    static bool head_sharded(const Geom& g) { return vocab_head_sharded(g); }

    /* The taps the container's drafter reads, when it carries a DFlash (v1) drafter and the serve
     * speculates: the trunk layers into `out` and their count (0 when there is nothing to tap), or a
     * refusal -- a DFlash2 drafter, which this block is not, or one naming no layer of an
     * `n_layer`-layer trunk, which belongs to another target. */
    static int taps(const RadModelMeta* meta, const RadBuildCtx* ctx, int64_t n_layer, int* out,
                    int max);

    /* The whole drafter, declared after the trunk (declaration order is the buffer planner's
     * liveness): its config read off `draft.`, its KV group bound from `t.base_layer`, the block,
     * and the RadDrafterDecl the engine drives it by. `trunk` is the trunk's geometry, which the
     * drafter's copies for what they share: the ranks, the vocabulary shard, the step bounds. */
    static int declare_drafter(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                               Names& nm, const Geom& trunk, const Trunk& t, Dflash1* df,
                               rad_kvgroup* kv_out);

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck, rad_buf src,
                 const char* tok_src);
    void context(RadCtx* c, const RadBatch* batch) const;
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }

    /* The pair a quantiser writes for an n-wide activation of `rows`: the codes, and the f32 scale
     * plane -- one per (row, 128) on the E4M3 grid, one per row on the int8 one. */
    void decl_pair(RadBuilder* b, Names& nm, const char* q, const char* s, int64_t rows, int64_t n,
                   rad_buf* qb, rad_buf* sb) const {
        *qb = decl_b(b, nm.f("%s", q), i8 ? RAD_I8 : RAD_F8E4M3, {rows, n});
        *sb = decl_b(b, nm.f("%s", s), RAD_F32, {rows, i8 ? 1 : fp8_blocks(n)});
    }
    /* The standalone quantiser of an n-wide activation, in the grid: `quant_act_fp8` with a scale
     * per 128 columns, or `quant_act_i8` with one per row, whose `group` only sizes the optional
     * code sums this symmetric grid does not take (gemma4_bf16.cpp's G4Linear). */
    rad_op decl_quant(RadBuilder* b, int64_t rows, int64_t n, rad_buf x, rad_buf q,
                      rad_buf s) const {
        return rw(b, RAD_OP(b, i8 ? "quant_act_i8" : "quant_act_fp8",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                    RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                  {x}, {q, s});
    }
    /* ...and its issue. quant_act_i8 has a third output, the row sums an asymmetric grid reads,
     * absent on this symmetric one and passed as a null operand, since operands are positional. */
    void quant(RadCtx* c, rad_op op, int64_t T, rad_buf x, rad_buf q, rad_buf s) const {
        if (i8) RAD_ISSUE_N(c, op, T, brows(x, T), brows(q, T), brows(s, T), RAD_NONE);
        else    RAD_ISSUE_N(c, op, T, brows(x, T), brows(q, T), brows(s, T));
    }
    /* A fused norm and quantiser's issue: x (the delta where it adds), the residual or none, the
     * gain, the pair, no bf16 copy, and on the int8 grid rmsnorm_quant_i8's two junction gains
     * (post-norm and scalar), which a drafter layer has neither of. */
    void norm_quant(RadCtx* c, rad_op op, int64_t T, RadOperand x, RadOperand res,
                    rad_weight gain) const {
        if (i8)
            RAD_ISSUE_N(c, op, T, x, res, RAD_W(gain), brows(w.hq, T), brows(w.hs, T), RAD_NONE,
                        RAD_NONE, RAD_NONE);
        else
            RAD_ISSUE_N(c, op, T, x, res, RAD_W(gain), brows(w.hq, T), brows(w.hs, T), RAD_NONE);
    }
    /* The kv heads into the cache's layout: head h to copies h * kv_rep .. h * kv_rep + kv_rep - 1,
     * so the copies of one head sit side by side and query head q of a gqa-4 attention, which reads
     * cache head q / 4, reads head q / (4 * kv_rep), the projection's own. One head-wide strided row
     * copy (`cast`, bf16 to bf16) a copy. */
    void rep_kv(RadCtx* c, rad_op op, int64_t T, rad_buf src, rad_buf dst) const {
        const int64_t hd = g.head_dim, r = cfg.kv_rep;
        for (int64_t h = 0; h < g.n_head_kv; ++h)
            for (int64_t i = 0; i < r; ++i)
                RAD_ISSUE(c, op, bcol(src, h * hd, hd, T), bcol(dst, (h * r + i) * hd, hd, T));
    }
    /* A linear's GEMM over its input's pair. */
    void gemm(RadCtx* c, rad_op op, int64_t T, rad_buf q, rad_buf s, const Lin& l,
              rad_buf y) const {
        RAD_ISSUE_N(c, op, T, brows(q, T), brows(s, T), RAD_W(l.w), RAD_W(l.s), brows(y, T),
                    RAD_NONE, l.bref());
    }
    /* The linears must be quantised by the recipe, in the grid `fc` set. A bf16 one would be read
     * through a view of a dtype it is not stored in, and the loader's error would name the view,
     * not the missing recipe; one on the other grid would read a pair its producer does not write. */
    int need_quantised(RadBuilder* b, const char* wn, const Lin& l) const {
        RadEncoding e{};
        if (!weight_enc(b, wn, &e) || e.n_planes <= 0) return RAD_OK;   /* nothing to ask yet */
        const uint32_t dt = e.plane[0].dtype;
        if (dt != RAD_F8E4M3 && dt != RAD_FP4E2M1 && dt != RAD_I8) {
            fprintf(stderr, "radiance: gemma4: the DFlash drafter's '%s' is stored %s. Its linears "
                            "are served 8-bit: convert with a recipe that names them, e.g. "
                            "`dflash.blk.*.attn_q.weight rtn codes=fp8_e4m3 block=128x128 "
                            "scale=bf16` (or `codes=i8 group=128 scale=f16 clamp=sym`) and the "
                            "same for fc, attn_k/v/output, ffn_gate_up and ffn_down.\n",
                    wn, rad_dtype_name(dt));
            return RAD_E_DTYPE;
        }
        if (l.i8 != i8) {
            fprintf(stderr, "radiance: gemma4: the DFlash drafter's '%s' is %s and its fc is %s; "
                            "one quantised activation feeds several linears, so the drafter's "
                            "linears take one grid and the recipe's rules for them must agree\n",
                    wn, l.i8 ? "int8 (w8a8)" : "on the E4M3 grid (fp8 or MXFP4)",
                    i8 ? "int8 (w8a8)" : "on the E4M3 grid (fp8 or MXFP4)");
            return RAD_E_DTYPE;
        }
        return RAD_OK;
    }
};

/* ------------------------------------------------------------------------------ one linear */

inline int Dflash1::Lin::declare(RadBuilder* b, Names& nm, const char* base, int64_t N, int64_t K,
                                 int access, int shard, RadWeightGroup grp,
                                 std::initializer_list<const char*> src,
                                 std::initializer_list<int64_t> row_parts) {
    /* The name map once, before the question it answers: a COPY map declared twice is refused and
     * a CONCAT one extended (rad_builder.cpp), so DenseFP8 is told the map is made. */
    const char* wn = nm.f("%s.weight", base);
    RAD_ARCH_TRY(map_linear(b, wn, src));
    RadEncoding enc{};
    i8 = weight_enc(b, wn, &enc) && linear_w8a8(enc);
    if (!i8) {
        RAD_ARCH_TRY(f.declare(b, nm, base, N, K, access, shard, grp, src, row_parts, true));
        w = f.w;
        s = f.s;
        return RAD_OK;
    }
    /* w8a8: rows in 16s (the fragment's tile) and K in whole 128-column scale groups. The scale
     * plane has the weight's rows, so it takes the weight's row parts as they are. */
    if (N % 16 || K % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an int8 weight takes rows in 16s "
                        "and K in whole %d-column scale groups.\n",
                base, (long long)N, (long long)K, RAD_FP8_BLOCK);
        return RAD_E_INVAL;
    }
    int64_t p[RAD_MAX_ROW_PARTS] = {0};
    int     np = 0;
    for (int64_t q : row_parts) {
        if (np >= RAD_MAX_ROW_PARTS || q % 16) {
            fprintf(stderr, "radiance: '%s' has a %lld-row part per rank; an int8 weight takes at "
                            "most %d parts, each a multiple of 16 rows.\n", base, (long long)q,
                    (int)RAD_MAX_ROW_PARTS);
            return RAD_E_INVAL;
        }
        p[np++] = q;
    }
    w = decl_view(b, wn, nullptr, "codes", RAD_I8, { N, K }, access, shard, grp, p, np);
    s = decl_view(b, nm.f("%s.scale", base), wn, "scale", RAD_F16, { N, K / RAD_FP8_BLOCK },
                  access, shard, grp, p, np);
    return w && s ? RAD_OK : RAD_E_INVAL;
}

/* ------------------------------------------------------------------------------ the taps */

inline int Dflash1::taps(const RadModelMeta* meta, const RadBuildCtx* ctx, int64_t n_layer,
                         int* out, int max) {
    if (ctx->max_spec <= 0) return 0;
    const int kind = dflash_kind(meta);
    if (kind == 0) return 0;
    if (kind == 2) {
        fprintf(stderr, "radiance: gemma4: the container carries a DFlash2 drafter; this plugin "
                        "serves the DFlash (v1) block z-lab publishes for Gemma 4\n");
        return RAD_E_UNSUPPORTED;
    }
    const char* ids = rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr);
    if (!ids) ids = rad_meta_gets(meta, "draft.target_layer_ids", nullptr);
    const int n = parse_layer_ids(ids, n_layer, out, max);
    if (n <= 0) {
        fprintf(stderr, "radiance: gemma4: the drafter names no trunk layer this model has; it is "
                        "paired with a different target\n");
        return RAD_E_UNSUPPORTED;
    }
    return n;
}

/* ------------------------------------------------------------------------------ the drafter */

inline int Dflash1::declare_drafter(RadBuilder* b, const RadModelMeta* meta,
                                    const RadBuildCtx* ctx, Names& nm, const Geom& trunk,
                                    const Trunk& t, Dflash1* df, rad_kvgroup* kv_out) {
    Geom dg{};
    RAD_ARCH_TRY(drafter_geom(dg, trunk, meta, "gemma4 DFlash"));
    if (rad_meta_geti(meta, "draft.hidden_size", dg.n_embd) != dg.n_embd) {
        fprintf(stderr, "radiance: gemma4: the drafter is %lld wide and the target %lld; it reads "
                        "the target's embedding and head, so it is not this target's drafter\n",
                (long long)rad_meta_geti(meta, "draft.hidden_size", 0), (long long)dg.n_embd);
        return RAD_E_UNSUPPORTED;
    }

    Config dc{};
    dc.n_taps = t.n_taps;
    dc.escale = t.escale;
    const long long trained = dflash_block(meta);
    if (trained < 2) {
        fprintf(stderr, "radiance: gemma4: the DFlash checkpoint states no block_size\n");
        return RAD_E_FORMAT;
    }
    if ((long long)ctx->max_spec + 1 != trained) {
        fprintf(stderr, "radiance: gemma4: the DFlash drafter was trained at block_size %lld, "
                        "which is %lld draft tokens, and this serve asks for %d. A block-diffusion "
                        "drafter denoises a fixed number of mask positions; run it at "
                        "--num-speculative-tokens %lld (or `auto`).\n",
                trained, trained - 1, ctx->max_spec, trained - 1);
        return RAD_E_UNSUPPORTED;
    }
    dc.block = trained;
    if (dg.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: gemma4: the DFlash drafter has %lld layers; this block carries "
                        "at most %d\n", (long long)dg.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    {
        std::vector<int> full;
        const char* lt = rad_meta_gets(meta, "draft.layer_types", nullptr);
        if (lt) {
            RAD_ARCH_TRY(parse_layer_kinds("gemma4", lt, {"full_attention"}, "sliding_attention",
                                           &full));
            if ((int64_t)full.size() != dg.n_layer) {
                fprintf(stderr, "radiance: gemma4: the drafter's layer_types names %zu layers and "
                                "it has %lld\n", full.size(), (long long)dg.n_layer);
                return RAD_E_INVAL;
            }
        } else {
            full.assign((size_t)dg.n_layer, 1);
        }
        bool any_swa = false;
        for (int64_t l = 0; l < dg.n_layer; ++l) {
            dc.full[l] = full[(size_t)l];
            any_swa = any_swa || !full[(size_t)l];
        }
        if (any_swa) {
            dc.window = rad_meta_geti(meta, "draft.dflash_config.swa_window_size",
                                      rad_meta_geti(meta, "draft.sliding_window", 0));
            if (!rad_meta_geti(meta, "draft.use_sliding_window", 1) || dc.window <= 0) {
                fprintf(stderr, "radiance: gemma4: the drafter names sliding layers and no "
                                "window\n");
                return RAD_E_INVAL;
            }
            /* z-lab's configs state no causality and train the sliding layers causal; NVIDIA's
             * (Nemotron-3.5-Bolt-E4B's DFlash) states `dflash_config.causal: false`, every layer
             * sliding and non-causal over its window. */
            dc.causal = rad_meta_geti(meta, "draft.dflash_config.causal",
                                      rad_meta_geti(meta, "draft.is_causal", 1)) != 0 ? 1 : 0;
        }
    }
    if (!head_sharded(dg) && dg.world > 1) {
        fprintf(stderr, "radiance: gemma4: the vocabulary (%lld) does not split evenly into whole "
                        "16-row blocks over %d ranks, which the drafter's sharded head needs\n",
                (long long)dg.n_vocab_all, dg.world);
        return RAD_E_UNSUPPORTED;
    }

    /* One unbounded cache for all the drafter's layers, because its full layer reads every
     * position. The sliding layers bound themselves with their attention op's `window`, so the
     * declared drafter window is 0 and the core stages the whole context. A separate window group
     * for the sliding layers would save memory. */
    /* The cache holds kv_rep copies of each of this rank's kv heads (kv_rep_for), or each once where
     * a card serves the drafter's own gqa (qrow_serves). */
    dc.kv_rep = kv_rep_for(dg);
    if (dc.kv_rep > 1 && rad_op_declared(b, "attn_paged_qrow")) {
        Geom pg = dg;
        pg.kv_dtype = rad_dtype_name(rad_kv_cache_dtype(ctx, RAD_BF16));   /* as the group takes it */
        if (qrow_serves(b, pg, dc)) {
            dc.kv_rep = 1;
            dc.qrow = true;
        }
    }
    {
        Geom cg = dg;
        cg.n_head_kv     *= dc.kv_rep;
        cg.n_head_kv_all *= dc.kv_rep;
        RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, cg, 0, t.base_layer, kv_out));
        dg.kv_dtype = cg.kv_dtype;
    }

    RAD_ARCH_TRY(df->declare(b, nm, dg, dc, t.base_layer, *kv_out, t.w_tok, "draft", t.src,
                             t.tok_src));

    RadDrafterDecl drf{};
    drf.name           = "dflash";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;          /* the anchor predicts nothing: block = depth + 1 */
    drf.proposal       = df->w.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = 0;                 /* the full layer reads everything; see the group */
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id",
                                                rad_meta_geti(meta, "draft.mask_token_id", -1));
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: gemma4: the DFlash checkpoint declares no mask_token_id\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));

    int nfull = 0;
    for (int64_t l = 0; l < dg.n_layer; ++l) nfull += dc.full[l];
    rad_note(b, "the DFlash drafter is declared: %lld layers at absolute %d..%d (%d full, %lld "
                "sliding over %lld, %s), %lld heads / %lld kv x %lld, ffn %lld, reading the "
                "trunk at %lld tap(s); linears %s. One forward over %lld rows proposes %lld "
                "tokens; the head is %s, argmax %s",
             (long long)dg.n_layer, t.base_layer, t.base_layer + (int)dg.n_layer - 1, nfull,
             (long long)(dg.n_layer - nfull), (long long)dc.window,
             dc.causal ? "causal" : "non-causal", (long long)dg.n_head,
             (long long)dg.n_head_kv, (long long)dg.head_dim, (long long)dg.n_ff,
             (long long)t.n_taps,
             df->i8 ? "int8 (w8a8, an int8 activation per row)" : "on the E4M3 grid (fp8a8 / mxfp4a8)",
             (long long)dc.block, (long long)dc.steps(),
             !df->q_head ? "the target's tied bf16 table"
             : df->w_dh.i8 ? "the container's int8 copy of the target's table (dflash.draft_head)"
                           : "the container's block-fp8 copy of the target's table (dflash.draft_head)",
             head_sharded(dg) ? "per vocab shard, merged" : "over the whole vocabulary");
    if (dc.kv_rep > 1)
        rad_note(b, "the DFlash drafter's cache holds each of its %lld kv head(s) a rank %lld times, "
                    "so its attention runs at %lld queries per kv head, the width libr4d's head-%lld "
                    "kernel serves at a block of %lld (gqa %lld otherwise)",
                 (long long)dg.n_head_kv, (long long)dc.kv_rep, (long long)kAttnGqa,
                 (long long)dg.head_dim, (long long)dc.block, (long long)dg.gqa());
    else if (dc.qrow)
        rad_note(b, "the DFlash drafter attends at %lld queries per kv head through attn_paged_qrow, "
                    "its cache holding each of its %lld kv head(s) a rank once",
                 (long long)dg.gqa(), (long long)dg.n_head_kv);
    return RAD_OK;
}

/* ------------------------------------------------------------------------------ declare */

inline int Dflash1::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                            int base_layer, rad_kvgroup group, rad_weight tok_embd,
                            const char* ck, rad_buf src, const char* tok_src) {
    g = geom; cfg = c; first_layer = base_layer; kv = group; w_tok = tok_embd;
    w = Wire{};
    w.src = src;
    if (!ck || !w_tok || !tok_src || !src) return RAD_E_INVAL;
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: gemma4: the drafter has %lld layers; this block carries at most "
                        "%d\n", (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.block < 2 || cfg.n_taps <= 0) return RAD_E_INVAL;
    if (cfg.kv_rep < 1 || g.gqa() % cfg.kv_rep) return RAD_E_INVAL;

    const int64_t qd = q_dim(), kvd = kv_dim(), rep = cfg.kv_rep;
    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    srows = g.max_seqs * cfg.steps();
    if (qrows <= 0 || srows <= 0) return RAD_E_INVAL;
    sharded = head_sharded(g);
    const int64_t tw = cfg.n_taps * g.n_embd;

    /* ---------------------------------------------------------------- model-scope weights
     * `fc` first: its encoding is the drafter's activation grid, which the buffers below take. */
    RAD_ARCH_TRY(w_fc.declare(b, nm, nm.f("dflash.fc"), g.n_embd, tw, RAD_ACCESS_PER_TOKEN,
                              RAD_SHARD_NONE, grp_model(), { nm.ckpt("%s.fc.weight", ck) }));
    i8 = w_fc.i8;
    RAD_ARCH_TRY(need_quantised(b, nm.f("dflash.fc.weight"), w_fc));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.hidden_norm.weight"), nm.ckpt("%s.hidden_norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.norm.weight"),        nm.ckpt("%s.norm.weight", ck)));
    w_hnorm = decl_w(b, nm.f("dflash.hidden_norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_norm  = decl_w(b, nm.f("dflash.norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* ---------------------------------------------------------------- the buffers */
    decl_pair(b, nm, "df_srcq", "df_srcs", g.max_tok, tw, &w.srcq, &w.srcs);
    w.c  = decl_b(b, nm.f("df_c"), g.act_dtype, {g.max_tok, g.n_embd});
    decl_pair(b, nm, "df_cq", "df_cs", g.max_tok, g.n_embd, &w.cq, &w.cs);
    w.ck = decl_b(b, nm.f("df_ck"), g.act_dtype, {g.max_tok, kvd});
    w.cv = decl_b(b, nm.f("df_cv"), g.act_dtype, {g.max_tok, kvd});
    if (rep > 1) {
        w.ck4 = decl_b(b, nm.f("df_ck4"), g.act_dtype, {g.max_tok, kvd * rep});
        w.cv4 = decl_b(b, nm.f("df_cv4"), g.act_dtype, {g.max_tok, kvd * rep});
    }
    w.x  = decl_b(b, nm.f("df_x"),  g.act_dtype, {qrows, g.n_embd});
    w.h  = decl_b(b, nm.f("df_h"),  g.act_dtype, {qrows, g.n_embd});
    decl_pair(b, nm, "df_hq", "df_hs", qrows, g.n_embd, &w.hq, &w.hs);
    w.q    = decl_b(b, nm.f("df_q"),    g.act_dtype, {qrows, qd});
    w.k    = decl_b(b, nm.f("df_k"),    g.act_dtype, {qrows, kvd});
    w.v    = decl_b(b, nm.f("df_v"),    g.act_dtype, {qrows, kvd});
    if (rep > 1) {
        w.k4 = decl_b(b, nm.f("df_k4"), g.act_dtype, {qrows, kvd * rep});
        w.v4 = decl_b(b, nm.f("df_v4"), g.act_dtype, {qrows, kvd * rep});
    }
    w.attn = decl_b(b, nm.f("df_attn"), g.act_dtype, {qrows, qd});
    decl_pair(b, nm, "df_atq", "df_ats", qrows, qd, &w.atq, &w.ats);
    w.gate_up = decl_b(b, nm.f("df_gate_up"), g.act_dtype, {qrows, 2 * g.n_ff});
    w.ffn  = decl_b(b, nm.f("df_ffn"), g.act_dtype, {qrows, g.n_ff});
    decl_pair(b, nm, "df_ffq", "df_ffs", qrows, g.n_ff, &w.ffq, &w.ffs);
    w.sel_h = decl_b(b, nm.f("df_selh"), g.act_dtype, {srows, g.n_embd});
    w.dlog  = decl_b(b, nm.f("df_dlog"), g.act_dtype, {srows, g.n_vocab});
    if (sharded) {
        w.cand  = decl_b(b, nm.f("df_cand"),  RAD_I32, {srows, kTopR});
        w.unary = decl_b(b, nm.f("df_unary"), RAD_F32, {srows, kTopR});
        w.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {srows, kTopR, 2});
        w.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {srows, (int64_t)g.world * kTopR, 2});
    } else {
        w.unary = decl_b(b, nm.f("df_unary"), RAD_F32, {srows, 1});
    }
    /* The draft, one row per draft position: [n_seq * steps, 1], which is [n_seq, steps] with a
     * pitch of `steps`, read by the engine through the drafter's declaration. */
    w.tokens = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {srows, 1});

    /* ---------------------------------------------------------------- per-layer weights */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const RadWeightGroup gl = grp_layer(first_layer + i);
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.input_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.post_attention_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.q_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.k_norm.weight", ck, i)));
        w_anorm[i] = decl_w(b, nm.f("dflash.blk.%d.attn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, gl);
        w_fnorm[i] = decl_w(b, nm.f("dflash.blk.%d.ffn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, gl);
        w_qn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, gl);
        w_kn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, gl);
        /* The declared names match rad_block_dflash2.h's, so one recipe quantises both
         * drafters. */
        RAD_ARCH_TRY(w_q[i].declare(b, nm, nm.f("dflash.blk.%d.attn_q", i), qd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, gl,
                                    { nm.ckpt("%s.layers.%d.self_attn.q_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_k[i].declare(b, nm, nm.f("dflash.blk.%d.attn_k", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, gl,
                                    { nm.ckpt("%s.layers.%d.self_attn.k_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_v[i].declare(b, nm, nm.f("dflash.blk.%d.attn_v", i), kvd, g.n_embd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, gl,
                                    { nm.ckpt("%s.layers.%d.self_attn.v_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_o[i].declare(b, nm, nm.f("dflash.blk.%d.attn_output", i), g.n_embd, qd,
                                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, gl,
                                    { nm.ckpt("%s.layers.%d.self_attn.o_proj.weight", ck, i) }));
        RAD_ARCH_TRY(w_gu[i].declare(b, nm, nm.f("dflash.blk.%d.ffn_gate_up", i), 2 * g.n_ff,
                                     g.n_embd, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, gl,
                                     { nm.ckpt("%s.layers.%d.mlp.gate_proj.weight", ck, i),
                                       nm.ckpt("%s.layers.%d.mlp.up_proj.weight", ck, i) },
                                     { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(w_dn[i].declare(b, nm, nm.f("dflash.blk.%d.ffn_down", i), g.n_embd, g.n_ff,
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, gl,
                                     { nm.ckpt("%s.layers.%d.mlp.down_proj.weight", ck, i) }));
        const Lin* lins[6] = { &w_q[i], &w_k[i], &w_v[i], &w_o[i], &w_gu[i], &w_dn[i] };
        const char* names[6] = { "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_up",
                                 "ffn_down" };
        for (int k = 0; k < 6; ++k)
            RAD_ARCH_TRY(need_quantised(b, nm.f("dflash.blk.%d.%s.weight", i, names[k]), *lins[k]));
    }
    /* The int8 grid's prefill band: above 64 rows libr4d's w8a8 GEMM is the tiled one, which takes
     * N in 256s. The context pass runs at prefill heights (fc, k and v over every committed row) and
     * the query pass at qrows, so a linear whose rows on this rank are not whole 256s has no kernel
     * past 64, and is refused here by name rather than as an op that does not resolve. */
    if (i8) {
        const int64_t ns[4] = { g.n_embd, kvd, qd, 2 * g.n_ff };
        const int64_t hi[4] = { crows, crows, qrows, qrows };
        for (int k = 0; k < 4; ++k)
            if (hi[k] > 64 && ns[k] % 256) {
                fprintf(stderr, "radiance: gemma4: one of the DFlash drafter's int8 linears is "
                                "%lld rows per rank, and above 64 rows the w8a8 GEMM takes them in "
                                "256s; serve this drafter at fewer ranks or on the E4M3 grid\n",
                        (long long)ns[k]);
                return RAD_E_UNSUPPORTED;
            }
    }

    /* ---------------------------------------------------------------- attention: two kinds
     * Declared before either kv_store, which reads `block_size` back from the resolved kernel.
     * Both read the one KV group, so whichever resolves first answers for it. */
    auto attn = [&](int causal, int64_t window) {
        return rw(b, RAD_OP(b, cfg.qrow ? "attn_paged_qrow" : "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa() / rep), RAD_INT("causal", causal),
                                RAD_INT("window", window), RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                  {w.q}, {w.attn});
    };
    bool any_full = false, any_swa = false;
    for (int64_t l = 0; l < g.n_layer; ++l) (cfg.full[l] ? any_full : any_swa) = true;
    if (any_swa)  op_attn_swa  = attn(cfg.causal, cfg.window);
    if (any_full) op_attn_full = attn(0, 0);

    /* The copies into the cache's layout (rep_kv), a head-wide strided row copy each. Each pass's
     * pair is declared in its first layer, where it is first issued: declaration order is the
     * buffer planner's liveness, and declared after the layers the copies' buffers would look born
     * after the last layer's input and free to share its bytes. */
    auto rep_op = [&](int64_t rows, rad_buf src, rad_buf dst) {
        return rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.head_dim),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                  {src}, {dst});
    };

    /* ---------------------------------------------------------------- the context pass */
    op_qsrc = decl_quant(b, crows, tw, w.src, w.srcq, w.srcs);
    op_cfc = w_fc.gemm(b, crows, g.n_embd, tw, {w.srcq, w.srcs}, {w.c});
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_hnorm)),
                 {w.c}, {w.c});
    op_qc = decl_quant(b, crows, g.n_embd, w.c, w.cq, w.cs);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_ck[i] = w_k[i].gemm(b, crows, kvd, g.n_embd, {w.cq, w.cs}, {w.ck});
        op_cv[i] = w_v[i].gemm(b, crows, kvd, g.n_embd, {w.cq, w.cs}, {w.cv});
        if (rep > 1 && i == 0) {
            op_rep_ck = rep_op(crows, w.ck, w.ck4);
            op_rep_cv = rep_op(crows, w.cv, w.cv4);
        }
        op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                        RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                        RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(w_kn[i])),
                         {w.ck}, {w.ck});
    }
    op_crope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.ck}, {w.ck});
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head_kv", g.n_head_kv * rep),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS),
                  {rep > 1 ? w.ck4 : w.ck, rep > 1 ? w.cv4 : w.cv}, {});

    /* ---------------------------------------------------------------- the query pass */
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                RAD_INT("vocab_offset", g.vocab_off), RAD_F64("mult", cfg.escale)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.x});
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.x}, {w.x});
    op_qh  = decl_quant(b, qrows, g.n_embd, w.h, w.hq, w.hs);
    op_qat = decl_quant(b, qrows, qd, w.attn, w.atq, w.ats);
    op_qff = decl_quant(b, qrows, g.n_ff, w.ffn, w.ffq, w.ffs);

    /* x's norm under `gain`, quantised into hq|hs, in one launch; with `add`, the
     * fused_add_rms_norm form (x += h first). On the E4M3 grid it computes exactly the chain's
     * result, the norm of the bf16-rounded sum and the codes of the bf16-rounded norm
     * (libr4d/r4d_rows.cpp); the int8 one keeps the sum in f32 (docs/OPS.md). It is asked about
     * first, since declaring an op no loaded library declares is an error. */
    const bool has_nq = rad_op_declared(b, i8 ? "rmsnorm_quant_i8" : "rmsnorm_quant_fp8") != 0;
    auto norm_quant_op = [&](rad_weight gain, bool add) -> rad_op {
        if (!has_nq) return (rad_op)0;
        const rad_op h = i8
            ? RAD_OP(b, "rmsnorm_quant_i8",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype),
                                RAD_F64("wadd", 0.0)),
                     RAD_WEIGHTS(gain))
            : RAD_OP(b, "rmsnorm_quant_fp8",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_INT("group", RAD_FP8_BLOCK),
                                RAD_STR("dtype", g.dtype), RAD_F64("wadd", 0.0)),
                     RAD_WEIGHTS(gain));
        return add ? rw(b, h, {w.h, w.x}, {w.x, w.hq, w.hs}) : rw(b, h, {w.x}, {w.hq, w.hs});
    };

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_anorm[i])),
                        {w.x}, {w.h});
        op_anq[i] = norm_quant_op(w_anorm[i], l > 0);
        op_qp[i] = w_q[i].gemm(b, qrows, qd, g.n_embd, {w.hq, w.hs}, {w.q});
        op_kp[i] = w_k[i].gemm(b, qrows, kvd, g.n_embd, {w.hq, w.hs}, {w.k});
        op_vp[i] = w_v[i].gemm(b, qrows, kvd, g.n_embd, {w.hq, w.hs}, {w.v});
        if (rep > 1 && i == 0) {
            op_rep_k = rep_op(qrows, w.k, w.k4);
            op_rep_v = rep_op(qrows, w.v, w.v4);
        }
        op_qn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_qn[i])),
                     {w.q}, {w.q});
        op_kn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head_kv),
                                    RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_kn[i])),
                     {w.k}, {w.k});
        op_op[i] = w_o[i].gemm(b, qrows, g.n_embd, qd, {w.atq, w.ats}, {w.h});
        op_fnorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fnorm[i])),
                        {w.x}, {w.h});
        op_fnq[i] = norm_quant_op(w_fnorm[i], true);
        op_gu[i] = w_gu[i].gemm(b, qrows, 2 * g.n_ff, g.n_embd, {w.hq, w.hs}, {w.gate_up});
        /* The gate_up|silu|quant fold where it resolves, else the pair (rad_block_dflash2.h's
         * rule, band included). E4M3 grid only: libr4d's fold is fp8a8. */
        {
            const int64_t hi = qrows < 64 ? qrows : 64;
            if (hi > 0 && !i8 && !w_gu[i].f.mx) {
                op_gug[i] = rw(b, RAD_OP(b, "gemm_nt_q_gated",
                                   RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", 2 * g.n_ff),
                                              RAD_INT("K", g.n_embd),
                                              RAD_INT("group", RAD_FP8_BLOCK),
                                              RAD_STR("dtype", "fp8a8"), RAD_STR("act", "silu")),
                                   RAD_WEIGHTS(w_gu[i].w, w_gu[i].s)),
                               {w.hq, w.hs}, {w.ffq, w.ffs});
                if (op_gug[i]) gated_rows = hi;
            }
        }
        op_dn[i] = w_dn[i].gemm(b, qrows, g.n_embd, g.n_ff, {w.ffq, w.ffs}, {w.h});
    }
    op_qrope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.q});
    op_krope = rw(b, decl_rotary(b, g, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.k}, {w.k});
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head_kv", g.n_head_kv * rep),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {rep > 1 ? w.k4 : w.k, rep > 1 ? w.v4 : w.v}, {});
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
               {w.x, w.h}, {w.x});
    op_final = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    /* ---------------------------------------------------------------- the head
     * The mask rows (offsets 1..block-1 of each block, named by out_ids) are gathered contiguous,
     * projected onto this rank's slice of the target's vocabulary, and the argmax taken. Sharded,
     * each rank reports its best kTopR as global ids with logits; these are all-gathered and
     * reduced to the global top-1 by `pair_argmax` (ties to the lower id), which is exact since the
     * global winner is its own shard's first. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.sel_h});
    /* The 8-bit copy, where the container holds one (see the file header). The name map comes
     * first since it is how the encoding is looked up; rad-convert writes only declared weights.
     * The copy is row-sharded like the table, so this rank's rows are the same vocabulary slice and
     * `vocab_off` holds. Its grid is its own, not the linears': it reads its own quantised copy of
     * the gathered rows. */
    const char* dhn = nm.f("dflash.draft_head.weight");
    RAD_ARCH_TRY(map_copy(b, dhn, tok_src));
    RadEncoding he{};
    const bool known = weight_enc(b, dhn, &he);
    const bool head_fp8 = known && block_fp8_128(he);
    const bool head_i8  = known && linear_w8a8(he);
    q_head = head_fp8 || head_i8;
    if (q_head) {
        if (g.n_vocab % RAD_FP8_BLOCK || g.n_embd % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: gemma4: the 8-bit draft head is %lldx%lld a rank, which "
                            "is not whole %d-row scale blocks\n",
                    (long long)g.n_vocab, (long long)g.n_embd, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        const int shard = g.world > 1 ? RAD_SHARD_ROW : RAD_SHARD_NONE;
        w_dh.i8 = head_i8;
        w_dh.w = decl_view(b, dhn, nullptr, "codes", head_i8 ? RAD_I8 : RAD_F8E4M3,
                           { g.n_vocab, g.n_embd }, RAD_ACCESS_PER_TOKEN, shard, grp_model());
        w_dh.s = head_i8
            ? decl_view(b, nm.f("dflash.draft_head.scale"), dhn, "scale", RAD_F16,
                        { g.n_vocab, g.n_embd / RAD_FP8_BLOCK }, RAD_ACCESS_PER_TOKEN, shard,
                        grp_model())
            : decl_view(b, nm.f("dflash.draft_head.scale"), dhn, "scale", RAD_BF16,
                        { fp8_blocks(g.n_vocab), fp8_blocks(g.n_embd) }, RAD_ACCESS_PER_TOKEN,
                        shard, grp_model());
        w_dh.f.w = w_dh.w;
        w_dh.f.s = w_dh.s;
        if (!w_dh.w || !w_dh.s) return RAD_E_INVAL;
        w.selq = decl_b(b, nm.f("df_selq"), head_i8 ? RAD_I8 : RAD_F8E4M3, {srows, g.n_embd});
        w.sels = decl_b(b, nm.f("df_sels"), RAD_F32,
                        {srows, head_i8 ? 1 : fp8_blocks(g.n_embd)});
        op_qsel = rw(b, RAD_OP(b, head_i8 ? "quant_act_i8" : "quant_act_fp8",
                         RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                    RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {w.sel_h}, {w.selq, w.sels});
        op_head = w_dh.gemm(b, srows, g.n_vocab, g.n_embd, {w.selq, w.sels}, {w.dlog});
    } else {
        op_head = rw(b, RAD_OP(b, "gemm_nt",
                        RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab),
                                   RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(w_tok)),
                    {w.sel_h}, {w.dlog});
    }
    if (sharded) {
        op_topk = rw(b, RAD_OP(b, "row_topk",
                        RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab),
                                   RAD_INT("R", kTopR), RAD_INT("vocab_off", g.vocab_off),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.dlog}, {w.cand, w.unary, w.dpair});
        /* Row-interleaved, so one draft position's pairs from every rank land side by side. */
        op_dgather = rw(b, RAD_OP(b, "all_gather",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_INT("numel", srows * kTopR * 2), RAD_STR("dtype", "f32"),
                                      RAD_INT("row", kTopR * 2)),
                           RAD_NOWEIGHTS),
                       {w.dpair}, {w.dgath});
        op_pick = rw(b, RAD_OP(b, "pair_argmax",
                        RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.world * kTopR),
                                   RAD_STR("dtype", "f32")),
                        RAD_NOWEIGHTS),
                    {w.dgath}, {w.tokens});
        if (!op_pick) {
            fprintf(stderr, "radiance: gemma4: the sharded draft head needs `pair_argmax` on the "
                            "device, and no loaded kernel library serves it there\n");
            return RAD_E_UNSUPPORTED;
        }
    } else {
        op_topk = rw(b, RAD_OP(b, "row_topk",
                        RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", g.n_vocab),
                                   RAD_INT("R", 1), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.dlog}, {w.tokens, w.unary});
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void Dflash1::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;

    quant(c, op_qsrc, T, w.src, w.srcq, w.srcs);
    gemm(c, op_cfc, T, w.srcq, w.srcs, w_fc, w.c);
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.c, T));
    quant(c, op_qc, T, w.c, w.cq, w.cs);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        gemm(c, op_ck[i], T, w.cq, w.cs, w_k[i], w.ck);
        gemm(c, op_cv[i], T, w.cq, w.cs, w_v[i], w.cv);
        RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv, brows(w.ck, T), RAD_W(w_kn[i]),
                    brows(w.ck, T));
        RAD_ISSUE_N(c, op_crope, T, brows(w.ck, T), praw(batch->positions, RAD_I32, T));
        const bool rep = cfg.kv_rep > 1;
        if (rep) {
            rep_kv(c, op_rep_ck, T, w.ck, w.ck4);
            rep_kv(c, op_rep_cv, T, w.cv, w.cv4);
        }
        RAD_ISSUE_N(c, op_cstore, T, brows(rep ? w.ck4 : w.ck, T), brows(rep ? w.cv4 : w.cv, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void Dflash1::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.x, T));
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.x, T), RAD_NONE);

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        const rad_op op_attn = cfg.full[i] ? op_attn_full : op_attn_swa;

        /* The attention input, with the last layer's residual add where one launch takes both. */
        if (op_anq[i]) {
            norm_quant(c, op_anq[i], T, l > 0 ? brows(w.h, T) : brows(w.x, T),
                       l > 0 ? brows(w.x, T) : RAD_NONE, w_anorm[i]);
        } else {
            RAD_ISSUE_N(c, op_anorm[i], T, brows(w.x, T), RAD_W(w_anorm[i]), brows(w.h, T));
            quant(c, op_qh, T, w.h, w.hq, w.hs);
        }
        gemm(c, op_qp[i], T, w.hq, w.hs, w_q[i], w.q);
        gemm(c, op_kp[i], T, w.hq, w.hs, w_k[i], w.k);
        gemm(c, op_vp[i], T, w.hq, w.hs, w_v[i], w.v);
        RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.q, T));
        RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv, brows(w.k, T), RAD_W(w_kn[i]), brows(w.k, T));
        RAD_ISSUE_N(c, op_qrope, T, brows(w.q, T), praw(batch->positions, RAD_I32, T));
        RAD_ISSUE_N(c, op_krope, T, brows(w.k, T), praw(batch->positions, RAD_I32, T));
        const bool rep = cfg.kv_rep > 1;
        if (rep) {
            rep_kv(c, op_rep_k, T, w.k, w.k4);
            rep_kv(c, op_rep_v, T, w.v, w.v4);
        }
        RAD_ISSUE_N(c, op_store, T, brows(rep ? w.k4 : w.k, T), brows(rep ? w.v4 : w.v, T),
                    praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, first_layer + i),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq), RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1), brows(w.attn, T));
        quant(c, op_qat, T, w.attn, w.atq, w.ats);
        gemm(c, op_op[i], T, w.atq, w.ats, w_o[i], w.h);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        if (op_fnq[i]) {
            norm_quant(c, op_fnq[i], T, brows(w.h, T), brows(w.x, T), w_fnorm[i]);
        } else {
            RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h, T), brows(w.x, T));
            RAD_ISSUE_N(c, op_fnorm[i], T, brows(w.x, T), RAD_W(w_fnorm[i]), brows(w.h, T));
            quant(c, op_qh, T, w.h, w.hq, w.hs);
        }
        if (op_gug[i] && T <= gated_rows) {
            RAD_ISSUE_N(c, op_gug[i], T, brows(w.hq, T), brows(w.hs, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.ffq, T), brows(w.ffs, T),
                        RAD_NONE);
        } else {
            gemm(c, op_gu[i], T, w.hq, w.hs, w_gu[i], w.gate_up);
            RAD_ISSUE_N(c, op_act, T, brows(w.gate_up, T), brows(w.ffn, T));
            quant(c, op_qff, T, w.ffn, w.ffq, w.ffs);
        }
        gemm(c, op_dn[i], T, w.ffq, w.ffs, w_dn[i], w.h);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        /* the residual add, unless the next layer's input norm takes it */
        if (l + 1 == g.n_layer || !op_anq[i + 1])
            RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h, T), brows(w.x, T));
    }
    RAD_ISSUE_N(c, op_final, T, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));

    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_gather, S, RAD_B(w.h), praw(batch->out_ids, RAD_I32, S), brows(w.sel_h, S));
    if (q_head) {
        if (w_dh.i8)
            RAD_ISSUE_N(c, op_qsel, S, brows(w.sel_h, S), brows(w.selq, S), brows(w.sels, S),
                        RAD_NONE);
        else
            RAD_ISSUE_N(c, op_qsel, S, brows(w.sel_h, S), brows(w.selq, S), brows(w.sels, S));
        RAD_ISSUE_N(c, op_head, S, brows(w.selq, S), brows(w.sels, S), RAD_W(w_dh.w),
                    RAD_W(w_dh.s), brows(w.dlog, S), RAD_NONE, w_dh.bref());
    } else {
        RAD_ISSUE_N(c, op_head, S, brows(w.sel_h, S), RAD_W(w_tok), brows(w.dlog, S));
    }
    if (sharded) {
        RAD_ISSUE_N(c, op_topk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    brows(w.dpair, S));
        RAD_ISSUE(c, op_dgather, brows(w.dpair, S), brows(w.dgath, S));
        RAD_ISSUE_N(c, op_pick, S, brows(w.dgath, S), brows(w.tokens, S), RAD_NONE);
    } else {
        RAD_ISSUE_N(c, op_topk, S, brows(w.dlog, S), brows(w.tokens, S), brows(w.unary, S),
                    RAD_NONE);
    }
}

}  /* namespace gemma4 */

#endif /* RAD_GEMMA4_DFLASH_H */
