/* rad_block_ple.h -- Qwen4-Exp's Per-Layer Embedding, declared and issued.
 *
 * ============================== WHAT IT IS ==============================
 *
 * One layer has a second embedding table addressed by a hash of the last three tokens.
 * Qwen3.8-Flash-Next's is 320,001,536 rows of 160 (51 of the checkpoint's 125 billion
 * parameters), read sixteen rows per token and mixed into the residual stream by a dot product
 * against the stream:
 *
 *   ids  = ngram_ids(tokens)                     [T, 16]   a hash of the last 3 ids, per head
 *   e    = embed_lookup_q(ids, table, scale)     [T, 2560]  16 x 160, flattened
 *   k    = key_proj(e)    [T, hc*n]              v = value_proj(e)    [T, n]
 *   gv,
 *   gvn  = ple_gate(k, h, v, ...)                the dot, the signed sqrt, the broadcast
 *   out  = ple_conv(gvn, resid = gv, ...)        dilated depthwise conv, silu, and the add
 *   h   += out
 *
 * This runs before the attention hyper-connection reads the stream, so `h` is the wide, un-normed
 * residual.
 *
 * ============================== THE TABLE ==============================
 *
 * The 51.2 GB table cannot be staged onto a card, so `embed_lookup_q` reads it where
 * --ngram-placement put it: a host gather over the container's mmap or a locked RAM copy
 * (libavx/avx_ngram.cpp), a device kernel reading a pinned copy over the link, or a card's own
 * copy. The result reaches the card through the host arena, 5 KB per token at decode. Hence:
 *
 *   The table is a weight operand: the planner places an op on the host only when all its weight
 *   operands are Site::Host, judged from the table's placement.
 *
 *   `ids` and `e` are host-domain buffers, written and read by the host gather; a device gather
 *   accesses them over the link. Everything downstream of `e` is device work.
 *
 *   It is 128 checkpoint tensors, mapped in pieces by map_concat_shards(); the order is the
 *   concatenation, and a permuted table silently looks up wrong rows.
 *
 * ============================== THE TWO ROLLING WINDOWS ==============================
 *
 * The n-gram hash needs the two committed ids before the step, and the convolution nine timesteps
 * of its input. Both are RAD_KV_CONV windows read at `num_accepted - 1`. The token window's cold
 * value is EOS, the convolution's zero (zero is a real token id); `has_init` marks cold slots.
 *
 * What a step leaves in the window depends on the step, signalled by whether `num_accepted` is
 * passed (libref/ref_ple.cpp). A decode row may have drafts rejected, so its window uses
 * gdn_conv_update's shifted layout, where the next read offset selects the history ending at the
 * last kept token. A prefill chunk commits everything and leaves its tail at offset zero, read by
 * the next step's `num_accepted` of 1. One layout cannot serve both, so step() issues both ops
 * once per half of the batch, as rad_block_gdn_fp8.h does.
 */
#ifndef RAD_BLOCK_PLE_H
#define RAD_BLOCK_PLE_H

#include "rad_arch.h"
#include "rad_fp8.h"

namespace rad {
namespace arch {

struct PleLayer {
    struct Config {
        int64_t hc         = 0;    /* hyper-connection streams */
        int64_t embed_dim  = 0;    /* ple_embed_dim: heads * head_dim */
        int64_t heads      = 0;    /* (ngram - 1) * heads_per_ngram */
        int64_t ngram      = 0;
        int64_t eos        = -1;
        int64_t rows       = 0;    /* the table's row count, padded */
        int64_t shards     = 0;    /* how many checkpoint tensors it is stored as */
        int64_t conv_width = 0;
        int64_t dilation   = 0;    /* == ngram, in this model */
        int64_t head_dim() const { return heads > 0 ? embed_dim / heads : 0; }
        int64_t hist()     const { return (conv_width - 1) * dilation; }
    };

    struct Wire {
        rad_buf h = 0;   /* [max_tok, hc * n_embd] the wide stream, read and added into */
    };

    struct Src {
        /* The table's shards are named `<prefix><i><suffix>`; everything else is one tensor. */
        const char* shard_prefix = nullptr;
        const char* shard_suffix = nullptr;
        const char* mult         = nullptr;
        const char* vocab_sizes  = nullptr;
        const char* offsets      = nullptr;
        const char* key_proj     = nullptr;
        const char* value_proj   = nullptr;
        const char* norm_key     = nullptr;
        const char* norm_query   = nullptr;
        const char* norm_conv    = nullptr;
        const char* conv1d       = nullptr;
    };

    Geom   g{};
    Config c{};
    Wire   w{};
    int    layer_ = 0;

    rad_weight w_table = 0, w_scale = 0, w_mult = 0, w_vsz = 0, w_off = 0;
    /* The table is plain bf16 as shipped (no scale weight) rather than E4M3. */
    bool       table_bf16 = false;
    rad_weight w_nk = 0, w_nq = 0, w_nc = 0, w_conv = 0;
    /* The two projections of the gathered embedding, each in the encoding its container stores:
     * plain bf16 as the checkpoint ships them, or rotated int8 (or a narrower rotated form), which
     * halves the bytes a pass waits for once the rows are on the card. `a_embd` is their input and
     * `a_fed` the same planes for the second projection, whose codes the first one's quantiser
     * wrote. */
    LinearFP8  key{}, value{};
    ActFP8     a_embd{}, a_fed{};

    rad_kvgroup kv_tok = 0, kv_conv = 0;
    rad_buf b_ids = 0, b_emb = 0, b_embd = 0, b_k = 0, b_v = 0, b_gv = 0, b_gvn = 0, b_out = 0;
    rad_op  op_ids = 0, op_gather = 0, op_embd = 0;
    rad_op  op_gate = 0, op_conv = 0, op_add = 0;
    /* The draft passes' hint (hint_ids, hint_fetch), where a drafter and `ngram_hint` exist: the
     * per-slot tails the hash continues from, the hinted ids (host domain, as `ple_ids`) and the
     * rows fetched for them, which nothing reads. */
    rad_buf b_tail = 0, b_hint_ids = 0, b_hint_emb = 0;
    rad_op  op_hint_hist = 0, op_hint_round = 0, op_hint_gather = 0;

    int64_t hn() const { return c.hc * g.n_embd; }
    /* The read-ahead queue's depth in steps: batch.cpp's k_ahead_chunks steps of the following
     * prompt, whose rows the gather's reader thread prefetches. Clamped to what `b_ids` holds, so
     * a larger offer only shortens the queue. */
    static constexpr int64_t kAheadChunks = 2;
    /* The prompt that follows the step, bounded by what `b_ids` holds after the step's rows. */
    int64_t ahead_rows(const RadBatch* batch) const {
        const int64_t a = batch->n_ahead;
        if (a <= 0) return 0;
        const int64_t cap = kAheadChunks * g.max_tok;
        return a < cap ? a : cap;
    }

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& cfg, int layer,
                 const Wire& wire, const Src& src);
    /* The hash, issued at the top of the step; the rest at the layer (see ids()). */
    void ids(RadCtx* c_, const RadBatch* batch) const;
    void step(RadCtx* c_, const RadBatch* batch) const;
    /* On a draft head's pass: the ids of each sequence's newest token, first in the pass, and
     * the fetch of their rows, last in it (see hint_ids()). */
    void hint_ids(RadCtx* c_, const RadBatch* batch) const;
    void hint_fetch(RadCtx* c_, const RadBatch* batch) const;
    bool hints(const RadBatch* batch) const {
        return op_hint_gather != 0 &&
               (batch->draft_pass > 1 || (batch->draft_pass < 0 && batch->n_draft_out > 0));
    }
};

inline int PleLayer::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& cfg,
                             int layer, const Wire& wire, const Src& src) {
    g = geom; c = cfg; w = wire;

    if (c.hc <= 1 || c.heads <= 0 || c.ngram < 2 || c.eos < 0 || c.rows <= 0 ||
        c.shards <= 0 || c.conv_width <= 1 || c.dilation <= 0 || c.embed_dim <= 0) {
        fprintf(stderr, "radiance: ple: geometry refused -- hc %lld heads %lld ngram %lld eos "
                        "%lld rows %lld shards %lld width %lld dilation %lld embed %lld\n",
                (long long)c.hc, (long long)c.heads, (long long)c.ngram, (long long)c.eos,
                (long long)c.rows, (long long)c.shards, (long long)c.conv_width,
                (long long)c.dilation, (long long)c.embed_dim);
        return RAD_E_INVAL;
    }
    /* `heads` is `ngram - 1` equal blocks, one per n-gram order, and `embed_dim` splits evenly
     * across heads (the reference's constructor checks). */
    if (c.heads % (c.ngram - 1) || c.embed_dim % c.heads) {
        fprintf(stderr, "radiance: ple: %lld heads over %lld n-gram orders, %lld embed dims -- "
                        "neither divides\n", (long long)c.heads, (long long)(c.ngram - 1),
                (long long)c.embed_dim);
        return RAD_E_INVAL;
    }
    if (!w.h) return RAD_E_INVAL;
    if (!src.shard_prefix || !src.shard_suffix || !src.mult || !src.vocab_sizes ||
        !src.offsets || !src.key_proj || !src.value_proj || !src.norm_key || !src.norm_query ||
        !src.norm_conv || !src.conv1d) return RAD_E_INVAL;
    /* ============================== REPLICATED UNDER TP ==============================
     *
     * The whole layer is replicated and has no collective. The table is addressed by a token hash
     * every rank computes identically, so a vocabulary shard is useless; and it is Tier::Mapped,
     * so all ranks gather through one shared row cache over the same file.
     *
     * The rest is 63 MiB per rank (key_proj 50, value_proj 12.5, three norm gains, a conv weight).
     * Column-sharding the projections would shard `gv` and `gvn` across the hyper-connection
     * streams and need an all-gather per step to save 31 MiB per card. Every rank adds the same
     * `out` into an identical copy of the stream, so the result stays bit-identical across ranks.
     *
     * The duplicated gather is `heads` rows per token: 16 at decode, 32768 at a 2048-token chunk,
     * about 70 ms per rank through the mapping (7.6% of a prefill step). That cost is page-table
     * population, not I/O: a prefill's distinct rows land about one per 4 KiB page (of 25), so the
     * mapping pays a fault per row. At decode half of each step's rows are new and the page cache
     * makes the waiting thread start each read. So the kernel keeps its own row cache and reads
     * misses straight from the file (libavx/avx_ngram.cpp). Ranks share that cache: each claims
     * and reads the misses it reaches first and waits only for rows another rank has claimed, so
     * each miss is read once without a cross-rank barrier.
     */

    const int64_t HN = hn();
    const int64_t HD = c.head_dim();

    /* ---- the table, its scale, and the three hash constants -------------------------------- */
    /* E4M3 with one scale for all 320 million rows (Qwen3.8-Flash-Next-FP8's choice; E4M3 has its
     * own exponent, a row's dynamic range is 2.3x, and the scale sits on a three-octave plateau).
     * One stored weight, `fp8_e4m3*bf16[*x*]`: codes plus a one-element scale plane, made by the
     * recipe (data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe) or read as that release ships it (its shards under
     * one `weight_scale`); the gather takes them as two operands. The name map
     * joins the shards along rows and is declared first, since the encoding is looked up through
     * it. RAD_ACCESS_VOCAB: it is a vocabulary table, just not the token one. */
    const char* tn = nm.f("blk.%d.ple_ngram.weight", layer);
    RAD_ARCH_TRY(map_concat_shards(b, tn, src.shard_prefix, src.shard_suffix, (int)c.shards, 0));
    /* Or the checkpoint's own bf16 table (102 GB) when the container holds it as shipped: the same
     * op with no scale (an absent scale means 1), still a host gather over the mapping. */
    RadEncoding te{};
    table_bf16 = weight_enc(b, tn, &te) && rad_enc_is(&te, "plain") && te.n_planes == 1 &&
                 te.plane[0].dtype == RAD_BF16;
    if (table_bf16) {
        w_table = decl_w(b, tn, RAD_BF16, {c.rows, HD}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                         grp_model());
    } else {
        w_table = decl_view(b, tn, nullptr, "codes", RAD_F8E4M3, {c.rows, HD},
                            RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        w_scale = decl_view(b, nm.f("blk.%d.ple_ngram.scale", layer), tn, "scale", RAD_BF16, {1},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    }

    /* I64: the multipliers are ~2.4e13 (a token id times one fills 63 bits), so narrowing them
     * would change every id. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_mult", layer), src.mult));
    w_mult = decl_w(b, nm.f("blk.%d.ple_mult", layer), RAD_I64, {c.ngram},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_vocab_sizes", layer), src.vocab_sizes));
    w_vsz = decl_w(b, nm.f("blk.%d.ple_vocab_sizes", layer), RAD_I64, {c.heads},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_offsets", layer), src.offsets));
    w_off = decl_w(b, nm.f("blk.%d.ple_offsets", layer), RAD_I64, {c.heads},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* ---- the projections and the three gains ------------------------------------------------ */
    /* bf16 as shipped (the FP8 checkpoint's `modules_to_not_convert` names both projections); a
     * recipe may store them rotated int8. The name maps come first: the encodings, looked up
     * through them, say whether the input needs its int8 pair. They are declared below, with the
     * buffers they read and write. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_key.weight", layer), src.key_proj));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_value.weight", layer), src.value_proj));

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_norm_key.weight", layer), src.norm_key));
    w_nk = decl_w(b, nm.f("blk.%d.ple_norm_key.weight", layer), RAD_F32, {HN},
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_norm_query.weight", layer), src.norm_query));
    w_nq = decl_w(b, nm.f("blk.%d.ple_norm_query.weight", layer), RAD_F32, {HN},
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_norm_conv.weight", layer), src.norm_conv));
    w_nc = decl_w(b, nm.f("blk.%d.ple_norm_conv.weight", layer), RAD_F32, {HN},
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* The checkpoint stores it [hc*n, 1, width]; the op wants [hc*n, width], which is the same
     * bytes with the depthwise group axis dropped. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_conv.weight", layer), src.conv1d));
    w_conv = decl_w(b, nm.f("blk.%d.ple_conv.weight", layer), RAD_BF16, {HN, c.conv_width},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    if (!w_table || (!table_bf16 && !w_scale) || !w_mult || !w_vsz || !w_off ||
        !w_nk || !w_nq || !w_nc || !w_conv) return RAD_E_INVAL;

    /* ---- the two rolling windows ------------------------------------------------------------ */
    /* RAD_KV_CONV is a per-sequence rolling window of width-1 + num_spec entries. The token
     * window's `conv_width` is ngram, so its window is ngram - 1 + n_spec.
     *
     * The channel count is n_head_kv * head_dim, not `state_dim` (RAD_KV_LINEAR's per-head
     * geometry, ignored by the CONV kind). Set via state_dim, both groups would size to zero and
     * the KV manager would refuse the model; rad-kbench would not catch it, since it synthesises
     * the state operand from the op's shape function. */
    {
        RadKVGroupDecl td{};
        td.kind = RAD_KV_CONV;
        td.dtype = RAD_I32;
        td.conv_width = c.ngram;
        td.n_head_kv = 1;
        td.head_dim = 1;            /* one token id a slot */
        kv_tok = rad_decl_kv_group(b, nm.f("kv_ple_tok"), &td);

        RadKVGroupDecl cd{};
        cd.kind = RAD_KV_CONV;
        cd.dtype = g.act_dtype;
        /* The conv's history is (width-1)*dilation deep, so the group's width (window =
         * conv_width - 1 + n_spec) is that plus one; the raw conv_width would size an undilated
         * window and the deepest tap would read unwritten data. */
        cd.conv_width = c.hist() + 1;
        cd.n_head_kv = 1;
        cd.head_dim = HN;           /* [n_states, hc*n_embd, hist + n_spec] */
        kv_conv = rad_decl_kv_group(b, nm.f("kv_ple_conv"), &cd);
    }
    if (!kv_tok || !kv_conv) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_bind_layer_kv(b, layer, kv_tok));
    RAD_ARCH_TRY(rad_bind_layer_kv(b, layer, kv_conv));

    /* ---- buffers ---------------------------------------------------------------------------- */
    /* The two buffers the gather touches are host domain. `ids` is written by ngram_ids and read by
     * the gather; `e` is written by the gather and copied onto the card (`ple_emb_dev`). The host
     * arena is mapped, so the device reads it over the link (core/runtime/ctx.cpp), 5 KB per token
     * at decode. A device gather (--ngram-placement ram or vram, with libr4d) reads `ids` and
     * writes `e` over the link too; the domain is fixed here, before placement decides who gathers.
     *
     * `ids` is PERSIST: it is written at the top of the step and read at this layer, but liveness
     * follows declared order, where the two ops are adjacent, so a transient could be overwritten
     * in between. */
    {
        /* The step's rows, then kAheadChunks further steps of the following prompt
         * (RadBatch::n_ahead) for the reader to prefetch. */
        RadBufDecl d{};
        d.dtype = RAD_I32; d.rank = 2;
        d.shape[0] = (1 + kAheadChunks) * g.max_tok; d.shape[1] = c.heads;
        d.kind = RAD_BUF_PERSIST; d.domain = RAD_DOMAIN_HOST;
        b_ids = rad_decl_buffer(b, nm.f("ple_ids"), &d);

        /* [max_tok, embed_dim], not [max_tok * heads, head_dim]: one shape must serve the gather
         * (`heads` rows per token) and the projections (one `embed_dim` row per token). The gather
         * derives its row count as numel / width (r4d_rows_pitch), so [T, 2560] read at 160 is
         * T*16 rows. The GEMM takes M and K from the operand's shape (gemm_ops), so the tall shape
         * would give it M = T*16, K = 160 and the step would be refused. */
        RadBufDecl e{};
        e.dtype = g.act_dtype; e.rank = 2;
        e.shape[0] = g.max_tok; e.shape[1] = c.embed_dim;
        e.kind = RAD_BUF_TRANSIENT; e.domain = RAD_DOMAIN_HOST;
        b_emb = rad_decl_buffer(b, nm.f("ple_emb"), &e);
    }
    /* The device copy the projections read. A tiled GEMM reads each A row band once per column
     * block (80 times for the key projection), so reading `e` from the host arena repeats link
     * traffic: at a 2048-token chunk the projections take about 3.0 and 0.9 ms from the host arena
     * versus 1.4 and 0.3 ms from VRAM. The copy is 10.5 MB per chunk, 5 KB per token at decode. */
    b_embd = decl_b(b, nm.f("ple_emb_dev"), g.act_dtype, {g.max_tok, c.embed_dim});
    b_k   = decl_b(b, nm.f("ple_k"),   g.act_dtype, {g.max_tok, HN});
    b_v   = decl_b(b, nm.f("ple_v"),   g.act_dtype, {g.max_tok, g.n_embd});
    b_gv  = decl_b(b, nm.f("ple_gv"),  g.act_dtype, {g.max_tok, HN});
    b_gvn = decl_b(b, nm.f("ple_gvn"), g.act_dtype, {g.max_tok, HN});
    b_out = decl_b(b, nm.f("ple_out"), g.act_dtype, {g.max_tok, HN});
    if (!b_ids || !b_emb || !b_embd || !b_k || !b_v || !b_gv || !b_gvn || !b_out)
        return RAD_E_INVAL;

    /* State slots come from RadKVGroupBatch::state_index, filled per group by the step batch, as in
     * rad_block_gdn_fp8.h; the core's `conv_state_index` is not used.
     *
     * ---- the ops, in the order step() issues them ------------------------------------------- */
    layer_ = layer;

    op_ids = rw(b, RAD_OP(b, "ngram_ids",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", c.heads),
                              RAD_INT("ngram", c.ngram), RAD_INT("eos", c.eos)),
                   RAD_WEIGHTS(w_mult, w_vsz, w_off)),
               {}, {b_ids});

    /* M is T * heads: the gather fetches one table row per head per token, and the band resolves
     * on that count. */
    {
        const rad_weight wt[2] = { w_table, w_scale };   /* a bf16 table has no scale plane */
        op_gather = rw(b, RAD_OP(b, "embed_lookup_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * c.heads), RAD_INT("n_embd", HD),
                                     RAD_INT("n_vocab", c.rows), RAD_STR("dtype", g.dtype)),
                          wt, table_bf16 ? 1 : 2),
                      {b_ids}, {b_emb});
    }

    /* ---- the draft passes' hint. A verify step's rows are the token the last verify chose and
     * the drafter's proposals, each known a draft pass or more before the verify hashes it,
     * except the last. Each draft pass hashes its sequences' newest token (`ngram_hint`) and
     * gathers those rows through the same row source, which for a table on a drive reads them
     * into its row cache; the verify step's gather then waits for the last proposal's rows
     * only. All three ops or none: without a drafter, or where no library hashes the hint,
     * nothing is declared and the verify step reads every row itself, as before. */
    if (g.max_spec > 0 && rad_op_declared(b, "ngram_hint")) {
        const int64_t ctx = c.ngram - 1;
        b_tail = decl_b(b, nm.f("ple_hint_tail"), RAD_I32, {g.max_seqs, 2 * ctx}, RAD_BUF_PERSIST);
        RadBufDecl hi{};
        hi.dtype = RAD_I32; hi.rank = 2;
        hi.shape[0] = g.max_seqs; hi.shape[1] = c.heads;
        hi.kind = RAD_BUF_PERSIST; hi.domain = RAD_DOMAIN_HOST;
        b_hint_ids = rad_decl_buffer(b, nm.f("ple_hint_ids"), &hi);
        /* PERSIST although nothing reads it: a transient's bytes are shared by declared op
         * order, and the fetch runs on the host while the draft pass's other buffers are live. */
        RadBufDecl he{};
        he.dtype = g.act_dtype; he.rank = 2;
        he.shape[0] = g.max_seqs; he.shape[1] = c.embed_dim;
        he.kind = RAD_BUF_PERSIST; he.domain = RAD_DOMAIN_HOST;
        b_hint_emb = rad_decl_buffer(b, nm.f("ple_hint_emb"), &he);
        if (!b_tail || !b_hint_ids || !b_hint_emb) return RAD_E_INVAL;
        for (const int hist : { 1, 0 })
            (hist ? op_hint_hist : op_hint_round) =
                rw(b, RAD_OP(b, "ngram_hint",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("heads", c.heads),
                                  RAD_INT("ngram", c.ngram), RAD_INT("eos", c.eos),
                                  RAD_INT("hist", hist)),
                       RAD_WEIGHTS(w_mult, w_vsz, w_off)),
                   {b_tail}, {b_tail, b_hint_ids});
        const rad_weight wt[2] = { w_table, w_scale };
        op_hint_gather = rw(b, RAD_OP(b, "embed_lookup_q",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs * c.heads),
                                          RAD_INT("n_embd", HD), RAD_INT("n_vocab", c.rows),
                                          RAD_STR("dtype", g.dtype)),
                               wt, table_bf16 ? 1 : 2),
                           {b_hint_ids}, {b_hint_emb});
        if (!op_hint_hist || !op_hint_round || !op_hint_gather)
            op_hint_hist = op_hint_round = op_hint_gather = 0;
    }

    op_embd = rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.embed_dim),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                 {b_emb}, {b_embd});

    /* The projections. An int8 one reads the embedding's int8 codes at its own scale group; the
     * first such quantises the bf16 plane and a second reads those codes, so two int8 projections
     * must agree in rotation and group (LinearFP8 refuses the pair otherwise). */
    {
        const char* kb = nm.f("blk.%d.ple_key", layer);
        const char* vb = nm.f("blk.%d.ple_value", layer);
        RadEncoding ke{}, ve{};
        const bool kk = weight_enc(b, nm.f("%s.weight", kb), &ke);
        const bool vk = weight_enc(b, nm.f("%s.weight", vb), &ve);
        const bool k8 = !g.w_bf16 && kk && (linear_is_w8(ke) || linear_is_w8h(ke));
        const bool v8 = !g.w_bf16 && vk && (linear_is_w8(ve) || linear_is_w8h(ve));
        a_embd.x = b_embd;
        if (k8 || v8)
            RAD_ARCH_TRY(a_embd.declare_qs(b, nm, g, "ple_emb_dev", c.embed_dim, 0, true, false,
                                           linear_w8h_group(k8 ? ke : ve)));
        a_fed = a_embd;
        if (k8 && v8) {
            a_fed.q8_fed = true;
            a_fed.q8_rot = linear_is_w8h(ke);
        }
        key.mapped = value.mapped = true;
        /* Anything but an int8 form is the plain bf16 GEMM, in a quantised model too. */
        Geom gb = g;
        gb.w_bf16 = true;
        RAD_ARCH_TRY(key.declare(b, nm, k8 ? g : gb, kb, HN, c.embed_dim, RAD_SHARD_NONE,
                                 grp_model(), { src.key_proj }, a_embd, b_k));
        RAD_ARCH_TRY(value.declare(b, nm, v8 ? g : gb, vb, g.n_embd, c.embed_dim, RAD_SHARD_NONE,
                                   grp_model(), { src.value_proj }, a_fed, b_v));
    }

    /* `q` is the incoming stream, un-normed; the op applies all three norms because the third is
     * over bytes it writes. */
    op_gate = rw(b, RAD_OP(b, "ple_gate",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_INT("hc", c.hc), RAD_F64("eps", g.eps),
                               RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                    RAD_WEIGHTS(w_nk, w_nq, w_nc)),
                {b_k, w.h, b_v}, {b_gv, b_gvn});

    op_conv = rw(b, RAD_OP(b, "ple_conv",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", HN),
                               RAD_INT("width", c.conv_width), RAD_INT("dilation", c.dilation),
                               RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w_conv)),
                {b_gvn, b_gv}, {b_out});

    /* The add into the stream: `hidden_states + ple(...)`, where ple_conv's residual operand
     * already produced `gv + silu(conv)`. */
    op_add = rw(b, RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", HN),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.h, b_out}, {w.h});

    if (!op_ids || !op_gather || !op_embd || !key.op || !value.op || !op_gate || !op_conv ||
        !op_add)
        return RAD_E_INVAL;
    return RAD_OK;
}

/* Slots come from the group batch, and `has_init` is `ctx_lens` (nonzero means a history), as in
 * rad_block_gdn_fp8.h.
 *
 * The two windowed ops are issued once per half of the step (batch_split: decode sequences
 * [0, D) with tokens [0, DT) first, then prefill). The decode half passes `num_accepted`, so the
 * op leaves its window readable after a partial rejection; the prefill half passes none and leaves
 * each chunk's tail at offset zero. The prefill half gets per-sequence arrays at `+ D` with token
 * operands at their base (kernels use the absolute offsets in `cu`); the decode half narrows to DT
 * rows.
 *
 * ids() runs first in the step, before every layer: it reads only the token ids and its own
 * window. The host gather that consumes it then waits for one small kernel, and the layers issued
 * in between keep the device busy during the gather. */
inline void PleLayer::ids(RadCtx* c_, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int64_t S = batch->n_seq;

    const RadKVGroupBatch* tk = kv_batch(batch, kv_tok);
    const int32_t* tk_idx = tk ? tk->state_index : nullptr;

    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t P = S - D;

    if (D > 0)
        RAD_ISSUE_N(c_, op_ids, DT,
                    praw(batch->token_ids, RAD_I32, DT), kv_cache(kv_tok, layer_),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    RAD_W(w_mult), RAD_W(w_vsz), RAD_W(w_off),
                    praw(tk_idx, RAD_I32, D),
                    praw(batch->ctx_lens, RAD_I32, D),
                    praw(batch->num_accepted, RAD_I32, D),
                    brows(b_ids, DT), RAD_NONE);
    /* The prompt following the last prefill chunk is hashed in the same launch into the rows
     * after the step's (`tok` reaches A tokens past the step); the window left behind is the
     * step's alone. */
    const int64_t A = P > 0 ? ahead_rows(batch) : 0;
    if (P > 0)
        RAD_ISSUE_N(c_, op_ids, T,
                    praw(batch->token_ids, RAD_I32, T + A), kv_cache(kv_tok, layer_),
                    praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                    RAD_W(w_mult), RAD_W(w_vsz), RAD_W(w_off),
                    praw(tk_idx ? tk_idx + D : nullptr, RAD_I32, P),
                    praw(batch->ctx_lens + D, RAD_I32, P),
                    RAD_NONE,
                    brows(b_ids, T),
                    A > 0 ? brow_slice(b_ids, T, A, c.heads) : RAD_NONE);
}

/* The hint of a draft head's pass. The history pass ends at the token the verify chose and each
 * later round reads the proposal before it (RadBatch::draft_pass), so the pass's newest token per
 * sequence is one the next verify step runs. hint_ids() hashes it first in the pass, so the ids
 * are in host memory as the pass starts; hint_fetch() goes last, where the host has issued the
 * whole pass and only waits for that one kernel, and the rows it reads arrive while the card
 * drafts. A history pass that drafts nothing gets neither. */
inline void PleLayer::hint_ids(RadCtx* c_, const RadBatch* batch) const {
    if (!hints(batch)) return;
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const RadKVGroupBatch* tk = kv_batch(batch, kv_tok);
    const int32_t* tk_idx = tk ? tk->state_index : nullptr;
    RAD_ISSUE_N(c_, batch->draft_pass < 0 ? op_hint_hist : op_hint_round, S,
                praw(batch->token_ids, RAD_I32, T), praw(batch->positions, RAD_I32, T),
                praw(batch->cu_seqlens, RAD_I32, S + 1),
                RAD_W(w_mult), RAD_W(w_vsz), RAD_W(w_off),
                tk_idx ? praw(tk_idx, RAD_I32, S) : RAD_NONE,
                RAD_B(b_tail), brows(b_hint_ids, S));
}

inline void PleLayer::hint_fetch(RadCtx* c_, const RadBatch* batch) const {
    if (!hints(batch)) return;
    const int64_t S = batch->n_seq;
    RAD_ISSUE_N(c_, op_hint_gather, S * c.heads,
                brows(b_hint_ids, S), RAD_W(w_table), table_bf16 ? RAD_NONE : RAD_W(w_scale),
                brows(b_hint_emb, S), RAD_NONE);
}

inline void PleLayer::step(RadCtx* c_, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int64_t S = batch->n_seq;

    const RadKVGroupBatch* cv = kv_batch(batch, kv_conv);
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;

    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t P = S - D;

    /* Operands are T rows (the gather derives its per-head rows from numel); the op's M band is
     * T * heads. The trailing operand is the following prompt's ids, which the gather's reader
     * prefetches for the next step. */
    const int64_t A = P > 0 ? ahead_rows(batch) : 0;
    RAD_ISSUE_N(c_, op_gather, T * c.heads,
                brows(b_ids, T), RAD_W(w_table), table_bf16 ? RAD_NONE : RAD_W(w_scale),
                brows(b_emb, T),
                A > 0 ? brow_slice(b_ids, T, A, c.heads) : RAD_NONE);

    RAD_ISSUE_N(c_, op_embd,  T, brows(b_emb, T), brows(b_embd, T));
    key.step(c_, a_embd, b_k, T, 0, T);
    value.step(c_, a_fed, b_v, T, 0, T);

    RAD_ISSUE_N(c_, op_gate, T,
                brows(b_k, T), brows(w.h, T), brows(b_v, T),
                RAD_W(w_nk), RAD_W(w_nq), RAD_W(w_nc),
                brows(b_gv, T), brows(b_gvn, T));

    if (D > 0)
        RAD_ISSUE_N(c_, op_conv, DT,
                    brows(b_gvn, DT), RAD_W(w_conv), kv_cache(kv_conv, layer_),
                    praw(batch->cu_seqlens, RAD_I32, D + 1), brows(b_gv, DT),
                    praw(cv_idx, RAD_I32, D),
                    praw(batch->ctx_lens, RAD_I32, D),
                    praw(batch->num_accepted, RAD_I32, D),
                    brows(b_out, DT));
    if (P > 0)
        RAD_ISSUE_N(c_, op_conv, T,
                    brows(b_gvn, T), RAD_W(w_conv), kv_cache(kv_conv, layer_),
                    praw(batch->cu_seqlens + D, RAD_I32, P + 1), brows(b_gv, T),
                    praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
                    praw(batch->ctx_lens + D, RAD_I32, P),
                    RAD_NONE,
                    brows(b_out, T));

    /* M is T rows of `n`, not the element count, matching the declared band 1..max_tok. */
    RAD_ISSUE_N(c_, op_add, T, brows(w.h, T), brows(b_out, T), brows(w.h, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_PLE_H */
