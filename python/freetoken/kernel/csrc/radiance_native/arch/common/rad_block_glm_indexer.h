/* GLM-5.3's DSA indexer: the stateful compressed-index core over its paged key cache (DsaCache),
 * which scores every complete block of a sequence's history against each query and selects the
 * 512 the MLA attends to, and the front end around it (GlmIndexer), whose casts bridge ordinary
 * linear rows to the flattened per-head quantiser and join normalized keys with learned pool
 * gates. No RoPE; the only rotation is the Hadamard inside dsa_query. Selected pages belong to
 * the MLA.
 *
 * Under tensor parallelism the key side stays replicated (every rank pools into the whole index
 * cache and scores against it), but from GlmTp::kSplitRows rows, or a smaller step whose rows x
 * live blocks reach GlmTp::kSplitWork, each rank scores and selects for its own run of query rows
 * (GlmTp::run_of) and two all_gathers join the runs' page tables and lengths: every row's
 * selection is computed by the same kernels from the same operands as on one rank, so the joined
 * table is the replicated one's bits. The query side (projection, head gates, quantiser) then runs
 * on this rank's run of rows only, since no other rank reads them. */
#ifndef RAD_BLOCK_GLM_INDEXER_H
#define RAD_BLOCK_GLM_INDEXER_H
#include "rad_fp8.h"
#include "rad_glm_tp.h"

namespace rad {
namespace arch {
struct DsaCache {
    struct Config {
        int64_t blocks = 0, work = 0, ring = 0, topk = 512;
    };
    struct Wire {
        rad_buf query = 0, codes = 0, scales = 0, gate = 0, kg = 0;
        rad_buf stage = 0, page = 0, bpos = 0, nc = 0, pooled = 0, pool_scale = 0;
        rad_buf score = 0, selected = 0, nsel = 0, length = 0;
        /* A taking-all step's zero scores, [max_tok, topk + 1]: such a step does not split, and
         * the score plane holds only a run of rows when the step may. */
        rad_buf unscored = 0;
        /* TP only: a rank's rebased bounds, page tables and lengths before the gathers. */
        rad_buf cu_part = 0, sel_part = 0, len_part = 0;
    };
    /* The narrower of the two gathered rows, one i32 length a query row (the page table's 513 a
     * row is a multiple of it), so a run is whole 16-byte words of both: a multiple of four rows.
     * The query and output planes hold world times this. */
    static constexpr int64_t kGatherRowBytes = 4;
    static int64_t part_rows(int64_t rows, int world) {
        return GlmTp::run_rows(rows, world, kGatherRowBytes);
    }
    Geom g{};
    Config c{};
    Wire w{};
    int layer = 0;
    rad_kvgroup index = 0, tail = 0, attention = 0;
    rad_weight ape = 0;
    rad_op op_query = 0, op_work = 0, op_pool = 0, op_tail = 0, op_score = 0, op_select = 0;
    rad_op op_unscored = 0, op_select_all = 0;
    rad_op op_score_part = 0, op_select_part = 0, op_gather_sel = 0, op_gather_len = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Config& cfg,
                const Wire& wire, rad_kvgroup attn, rad_kvgroup tail_group,
                rad_kvgroup* index_group, const char* ape_source) {
        g = geom;
        c = cfg;
        w = wire;
        layer = l;
        attention = attn;
        tail = tail_group;
        if (!b || !index_group || !ape_source || !attention || !tail || c.blocks < 1 ||
            c.work < (g.max_tok + 3) / 4 + g.max_seqs || c.ring < 4 + g.max_spec || c.topk != 512 ||
            !w.query || !w.codes || !w.scales || !w.gate || !w.kg || !w.stage || !w.page ||
            !w.bpos || !w.nc || !w.pooled || !w.pool_scale || !w.score || !w.selected || !w.nsel ||
            !w.length)
            return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, tail));
        const char* name = nm.f("blk.%d.index_ape", l);
        RAD_ARCH_TRY(map_copy(b, name, ape_source));
        ape = decl_w(b, name, RAD_F32, { 4, 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                     grp_layer(l));
        if (!ape) return RAD_E_INVAL;
        // Query buffers are [tokens*32,128]; scoring uses their rank-three view.
        op_query = rw(b,
                      RAD_OP(b, "dsa_query",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * 32), RAD_INT("n", 128),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                      { w.query }, { w.codes, w.scales });
        op_work = rw(
            b,
            RAD_OP(b, "qsa_work",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 256), RAD_INT("ratio", 4),
                              RAD_INT("ring", c.ring), RAD_INT("seqs", g.max_seqs),
                              RAD_INT("work", c.work), RAD_STR("dtype", "bf16")),
                   RAD_NOWEIGHTS),
            { w.kg }, { w.stage, w.page, w.bpos, w.nc });
        // Declare immediately before the cache consumer: page size comes from that op.
        if (!*index_group) {
            RadKVGroupDecl d{};
            d.kind = RAD_KV_FULL;
            d.dtype = RAD_U8;
            d.n_head_kv = 1;
            d.head_dim = 17;
            *index_group = rad_decl_kv_group(b, nm.f("kv_dsa_index"), &d);
            /* dsa_score reads this group over the whole history every step, so it stays resident and
             * holds every sequence's context while a sibling streams (kvstream/carve.cpp). */
            RAD_ARCH_TRY(rad_kv_group_dense_read(b, *index_group));
        }
        index = *index_group;
        if (!index) return RAD_E_INVAL;
        op_pool =
            rw(b,
               RAD_OP(b, "dsa_pool",
                      RAD_PARAMS(RAD_RANGE("M", 1, c.work), RAD_INT("n", 128), RAD_INT("ratio", 4),
                                 RAD_INT("block_size", 4), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(ape)),
               { w.stage, w.page }, { w.pooled, w.pool_scale });
        if (!op_pool || rad_kv_block_size(b, index) != 4) return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, index));
        op_tail = rw(b,
                     RAD_OP(b, "qsa_tail_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 256),
                                       RAD_INT("ratio", 4), RAD_INT("ring", c.ring),
                                       RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                     { w.kg }, {});
        /* A step whose every row reaches topk or fewer blocks selects them all (takes_all), and
         * qsa_select takes any finite scores in block order then: zeros serve as well as the
         * scorer's. The fill spans topk + 1 columns, at least the live reach of such a step. */
        op_unscored = rw(b,
                         RAD_OP(b, "fill",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                           RAD_INT("n", unscored_cols()), RAD_STR("dtype", "f32")),
                                RAD_NOWEIGHTS),
                         {}, { w.unscored });
        op_score = rw(b,
                      RAD_OP(b, "dsa_score",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                        RAD_INT("heads", 32), RAD_INT("blocks", c.blocks),
                                        RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", "fp8_e4m3")),
                             RAD_NOWEIGHTS),
                      { w.codes, w.scales, w.gate, w.nc }, { w.score });
        const bool split = g.world > 1;
        const int64_t part = split ? part_rows(g.max_tok, g.world) : 0;
        if (split && (!w.cu_part || !w.sel_part || !w.len_part)) return RAD_E_INVAL;
        if (split)
            op_score_part =
                rw(b,
                   RAD_OP(b, "dsa_score",
                          RAD_PARAMS(RAD_RANGE("M", 1, part), RAD_INT("n", 128),
                                     RAD_INT("heads", 32), RAD_INT("blocks", c.blocks),
                                     RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", "fp8_e4m3"),
                                     RAD_INT("row_part", g.rank)),
                          RAD_NOWEIGHTS),
                   { w.codes, w.scales, w.gate, w.nc }, { w.score, w.cu_part });
        op_select = rw(b,
                       RAD_OP(b, "qsa_select",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("blocks", c.blocks),
                                         RAD_INT("topk", 512), RAD_INT("ratio", 4),
                                         RAD_INT("seqs", g.max_seqs), RAD_INT("signed", 1)),
                              RAD_NOWEIGHTS),
                       { w.score, w.nc }, { w.selected, w.nsel, w.length });
        op_select_all = rw(b,
                           RAD_OP(b, "qsa_select",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                             RAD_INT("blocks", unscored_cols()),
                                             RAD_INT("topk", 512), RAD_INT("ratio", 4),
                                             RAD_INT("seqs", g.max_seqs), RAD_INT("signed", 1)),
                                  RAD_NOWEIGHTS),
                           { w.unscored, w.nc }, { w.selected, w.nsel, w.length });
        if (split) {
            op_select_part = rw(
                b,
                RAD_OP(b, "qsa_select",
                       RAD_PARAMS(RAD_RANGE("M", 1, part), RAD_INT("blocks", c.blocks),
                                  RAD_INT("topk", 512), RAD_INT("ratio", 4),
                                  RAD_INT("seqs", g.max_seqs), RAD_INT("signed", 1)),
                       RAD_NOWEIGHTS),
                { w.score, w.nc, w.cu_part }, { w.sel_part, w.nsel, w.len_part });
            op_gather_sel = rw(b,
                               RAD_OP(b, "all_gather",
                                      RAD_PARAMS(RAD_INT("world_size", g.world),
                                                 RAD_RANGE("numel", 4 * 513, part * 513),
                                                 RAD_STR("dtype", "f32")),
                                      RAD_NOWEIGHTS),
                               { w.sel_part }, { w.selected });
            op_gather_len = rw(b,
                               RAD_OP(b, "all_gather",
                                      RAD_PARAMS(RAD_INT("world_size", g.world),
                                                 RAD_RANGE("numel", 4, part),
                                                 RAD_STR("dtype", "f32")),
                                      RAD_NOWEIGHTS),
                               { w.len_part }, { w.length });
            if (!op_score_part || !op_select_part || !op_gather_sel || !op_gather_len)
                return RAD_E_INVAL;
        }
        return op_query && op_work && op_tail && op_unscored && op_score && op_select &&
                       op_select_all
                   ? RAD_OK
                                                                                      : RAD_E_INVAL;
    }
    int64_t unscored_cols() const { return c.topk + 1 < c.blocks ? c.topk + 1 : c.blocks; }
    /* The columns a step scores: the live reach the batch bounds on the host. */
    int64_t live_blocks(const RadBatch* batch) const {
        const auto* ib = kv_batch(batch, index);
        int64_t live = ((int64_t)batch->max_ctx_len + batch->n_tok + batch->n_spec + 4) / 4 + 1;
        if (live > c.blocks) live = c.blocks;
        // T is the total across sequences, so the conservative bound can exceed the
        // per-sequence page table. Columns without a table entry cannot be scored.
        if (ib && live > ib->block_table_pitch) live = ib->block_table_pitch;
        return live < 1 ? 1 : live;
    }
    /* The decision reads the same host bound as the score plane's width (max_ctx_len, rounded by
     * the core), so every rank takes the same path and a recorded decode pass changes it only when
     * that bound moves. A step that takes all its blocks never splits: with nothing to score, every
     * rank selects the whole step itself, and the gathers have nothing to save. */
    bool splits(const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        return op_score_part && live_blocks(batch) > c.topk &&
               (GlmTp::splits(T, g.world) ||
                (T >= GlmTp::kSplitWorkRows && T * live_blocks(batch) >= GlmTp::kSplitWork));
    }
    /* Every row's history fits the selection, so the query side (projection, head gates,
     * quantiser, scorer) has nothing to decide: the selected table is every live block in order,
     * bit for bit what scoring would select. Read from the same host bound as splits(), so every
     * rank agrees. */
    bool takes_all(const RadBatch* batch) const { return live_blocks(batch) <= c.topk; }
    /* The query rows this rank runs: the whole step, or its run of a split step (GlmTp::run_of).
     * A rank whose run lies wholly past a small step runs the step's first R rows instead: no
     * sequence holds its run, so its scores are -inf, and its answers land past T. The caller's
     * projections into `query` and `gate` cover the same rows. */
    void query_rows(const RadBatch* batch, int64_t* r0, int64_t* rows) const {
        const int64_t T = batch->n_tok;
        *r0 = 0;
        *rows = T;
        if (!splits(batch)) return;
        const GlmTp::Run run = GlmTp::run_of(T, g.world, g.rank, kGatherRowBytes);
        *r0 = run.first;
        *rows = run.rows;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const auto* ib = kv_batch(batch, index);
        const auto* tb = kv_batch(batch, tail);
        const auto* ab = kv_batch(batch, attention);
        if (!ib || !tb || !ab || !ib->block_table || !ab->block_table || !tb->state_index) {
            rad_step_fail(ctx, "DSA requires index/attention block tables and tail state indices");
            return;
        }
        const int64_t T = batch->n_tok, S = batch->n_seq;
        const int64_t work = (T + 3) / 4 + S,
                      pitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
        const bool all = takes_all(batch);
        int64_t q0 = 0, qn = T;
        query_rows(batch, &q0, &qn);
        if (!all)
            RAD_ISSUE_N(ctx, op_query, qn * 32, brow_slice(w.query, q0 * 32, qn * 32, 128),
                        brow_slice(w.codes, q0 * 32, qn * 32, 128),
                        brow_slice(w.scales, q0 * 32, qn * 32, 1));
        RAD_ISSUE_N(ctx, op_work, T, brows(w.kg, T), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), kv_cache(tail, layer),
                    praw2(tb->state_index, RAD_I32, S, pitch),
                    praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), RAD_NONE,
                    brows(w.stage, work * 4), brows(w.page, work), brows(w.bpos, work),
                    brows(w.nc, S));
        RAD_ISSUE_N(ctx, op_pool, work, brows(w.stage, work * 4), RAD_W(ape), brows(w.pooled, work),
                    brows(w.pool_scale, work), brows(w.page, work), kv_cache(index, layer));
        // Gather before overwriting the tail; the wider ring preserves rejected-token rollback.
        RAD_ISSUE_N(ctx, op_tail, T, brows(w.kg, T), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), kv_cache(tail, layer),
                    praw2(tb->state_index, RAD_I32, S, pitch));
        const int64_t live = live_blocks(batch);
        if (splits(batch)) {
            step_split(ctx, batch, live);
            return;
        }
        if (all) {
            const int64_t U = unscored_cols();
            RAD_ISSUE_N(ctx, op_unscored, T, brows(w.unscored, T));
            RAD_ISSUE_N(ctx, op_select_all, T, bcol_at(w.unscored, 0, U, 0, live, T),
                        praw2(ab->block_table, RAD_I32, S, ab->block_table_pitch),
                        brows(w.nc, S), praw(batch->positions, RAD_I32, T),
                        praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.selected, T),
                        brows(w.nsel, T), brows(w.length, T));
            return;
        }
        // dsa_score declares ten operands, cu_part the optional tenth; the engine counts them.
        RAD_ISSUE_N(ctx, op_score, T, brows(w.codes, T * 32), brows(w.scales, T * 32),
                    brows(w.gate, T), kv_cache(index, layer), RAD_NONE,
                    praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S),
                    bcol_at(w.score, 0, c.blocks, 0, live, T), RAD_NONE);
        RAD_ISSUE_N(ctx, op_select, T, bcol_at(w.score, 0, c.blocks, 0, live, T),
                    praw2(ab->block_table, RAD_I32, S, ab->block_table_pitch), brows(w.nc, S),
                    praw(batch->positions, RAD_I32, T), praw(batch->cu_seqlens, RAD_I32, S + 1),
                    brows(w.selected, T), brows(w.nsel, T), brows(w.length, T));
    }
    /* Rank r scores rows [r R, r R + R) (rows past the step score -inf) and selects for those
     * that exist (query_rows), against the bounds the scorer rebased to its first row. The
     * gathers lay the runs end to end, so rows past the step land past T, where nothing reads
     * them. */
    void step_split(RadCtx* ctx, const RadBatch* batch, int64_t live) const {
        const auto* ib = kv_batch(batch, index);
        const auto* ab = kv_batch(batch, attention);
        const int64_t T = batch->n_tok, S = batch->n_seq, W = g.world;
        const int64_t R = part_rows(T, g.world);
        int64_t p0 = 0, rows = 0;
        query_rows(batch, &p0, &rows);
        RAD_ISSUE_N(ctx, op_score_part, R, brows(w.codes, W * R * 32),
                    brows(w.scales, W * R * 32), brows(w.gate, W * R), kv_cache(index, layer),
                    RAD_NONE, praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S),
                    bcol_at(w.score, 0, c.blocks, 0, live, R), brows(w.cu_part, S + 1));
        RAD_ISSUE_N(ctx, op_select_part, rows, bcol_at(w.score, 0, c.blocks, 0, live, rows),
                    praw2(ab->block_table, RAD_I32, S, ab->block_table_pitch), brows(w.nc, S),
                    praw(batch->positions + p0, RAD_I32, rows), brows(w.cu_part, S + 1),
                    brows(w.sel_part, rows), brows(w.nsel, rows), brows(w.len_part, rows));
        RAD_ISSUE_N(ctx, op_gather_sel, R * 513, brows(w.sel_part, R), brows(w.selected, W * R));
        RAD_ISSUE_N(ctx, op_gather_len, R, brows(w.len_part, R), brows(w.length, W * R));
    }
};

struct GlmIndexer {
    struct Wire {
        rad_buf input = 0, q_low = 0, q_wide = 0, kg_raw = 0;
        DsaCache::Wire cache{};
    };
    Geom g{};
    Wire w{};
    LinearFP8 query{}, key_gate{};
    DsaCache cache{};
    rad_weight gain = 0, bias = 0, head_weight = 0;
    rad_op op_norm = 0, op_query_copy = 0, op_gate_copy = 0, op_head = 0;
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                const char* prefix, const DsaCache::Config& cfg, rad_kvgroup attention,
                rad_kvgroup tail, rad_kvgroup* index) {
        g = geom;
        w = wire;
        if (!b || !prefix || g.n_embd != 4096 || !w.input || !w.q_low || !w.q_wide || !w.kg_raw)
            return RAD_E_INVAL;
        Geom bg = g;
        bg.w_bf16 = true;
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        RAD_ARCH_TRY(query.declare(b, nm, bg, nm.f("blk.%d.index_query", l), 4096, 1536,
                                   RAD_SHARD_NONE, grp_layer(l),
                                   { nm.ckpt("%s.wq_b.weight", prefix) }, act(w.q_low), w.q_wide));
        RAD_ARCH_TRY(key_gate.declare(
            b, nm, bg, nm.f("blk.%d.index_key_gate", l), 256, 4096, RAD_SHARD_NONE, grp_layer(l),
            { nm.ckpt("%s.wk.weight", prefix), nm.ckpt("%s.index_kpool_compress_gate", prefix) },
            act(w.input), w.kg_raw, 0, { 128, 128 }));
        const char *gn = nm.f("blk.%d.index_norm.weight", l),
                   *bn = nm.f("blk.%d.index_norm.bias", l),
                   *hn = nm.f("blk.%d.index_head.weight", l);
        RAD_ARCH_TRY(map_copy(b, gn, nm.ckpt("%s.k_norm.weight", prefix)));
        RAD_ARCH_TRY(map_copy(b, bn, nm.ckpt("%s.k_norm.bias", prefix)));
        RAD_ARCH_TRY(map_copy(b, hn, nm.ckpt("%s.weights_proj.weight", prefix)));
        gain = decl_w(b, gn, RAD_F32, { 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        bias = decl_w(b, bn, RAD_F32, { 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        head_weight = decl_w(b, hn, RAD_F32, { 32, 4096 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                             grp_layer(l));
        if (!gain || !bias || !head_weight) return RAD_E_INVAL;
        op_norm = rw(b,
                     RAD_OP(b, "layernorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                       RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(gain, bias)),
                     { w.kg_raw }, { w.cache.kg });
        op_query_copy = rw(b,
                           RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * 32), RAD_INT("n", 128),
                                             RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                  RAD_NOWEIGHTS),
                           { w.q_wide }, { w.cache.query });
        op_gate_copy = rw(b,
                          RAD_OP(b, "cast",
                                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                            RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                 RAD_NOWEIGHTS),
                          { w.kg_raw }, { w.cache.kg });
        op_head = rw(b,
                     RAD_OP(b, "gemm_nt_f32",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 32),
                                       RAD_INT("K", 4096), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(head_weight)),
                     { w.input }, { w.cache.gate });
        RAD_ARCH_TRY(cache.declare(b, nm, g, l, cfg, w.cache, attention, tail, index,
                                   nm.ckpt("%s.index_kpool_compress_ape", prefix)));
        return op_norm && op_query_copy && op_gate_copy && op_head ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        int64_t r0 = 0, rows = T;
        cache.query_rows(batch, &r0, &rows);
        /* The key side runs on every step, since the index cache must stay current; the query
         * side only where the selection is a choice (DsaCache::takes_all). */
        const bool all = cache.takes_all(batch);
        /* Every output row of these ops is independent of the others and of how many there are,
         * except the bf16 projection's at 64 rows and under, a band whose split count a tuned
         * configuration may set: it stays on the step's rows unless every run is wider. */
        const bool runs = cache.splits(batch);
        if (!all) {
            if (runs && T - (g.world - 1) * DsaCache::part_rows(T, g.world) > 64)
                query.step(ctx, act(w.q_low), w.q_wide, T, r0, rows);
            else
                query.step(ctx, act(w.q_low), w.q_wide, T);
            RAD_ISSUE_N(ctx, op_query_copy, rows * 32, brow_slice(w.q_wide, r0, rows, 4096),
                        brow_slice(w.cache.query, r0 * 32, rows * 32, 128));
        }
        key_gate.step(ctx, act(w.input), w.kg_raw, T);
        RAD_ISSUE_N(ctx, op_norm, T, bcol(w.kg_raw, 0, 128, T), RAD_W(gain), RAD_W(bias),
                    bcol(w.cache.kg, 0, 128, T));
        RAD_ISSUE_N(ctx, op_gate_copy, T, bcol(w.kg_raw, 128, 128, T),
                    bcol(w.cache.kg, 128, 128, T));
        if (!all)
            RAD_ISSUE_N(ctx, op_head, rows, brow_slice(w.input, r0, rows, 4096),
                        RAD_W(head_weight), brow_slice(w.cache.gate, r0, rows, 32));
        cache.step(ctx, batch);
    }
};
}
} // namespace rad::arch
#endif
