/* GLM-5.3's manifold-constrained residual connections (mHC): four residual streams in one
 * 16384-wide row. read() mixes them down to the block's 4096-wide input and saves the mixing
 * coefficients; write() adds the block's output back into every stream. The mixed residual must
 * have a separate destination: each output stream reads every original input stream. */
#ifndef RAD_BLOCK_GLM_MHC_H
#define RAD_BLOCK_GLM_MHC_H
#include "rad_glm_tp.h"
#include <cmath>

namespace rad {
namespace arch {
struct ManifoldConn {
    struct Config {
        int64_t hc = 4, iterations = 20;
        double eps = 1e-6, hc_eps = 1e-6;
    };
    struct Src {
        const char *fn = nullptr, *scale = nullptr, *base = nullptr;
    };
    struct Wire {
        rad_buf h = 0, x = 0, y = 0, out = 0;
        // Saved by read(), consumed by write() after the block produces y.
        rad_buf post = 0, comb = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    rad_weight w_fn = 0, w_scale = 0, w_base = 0;
    rad_op op_pre = 0, op_post = 0;
    /* Above the split threshold under TP, GLM's projection is split by residual chunks: each rank
     * computes the partials of its chunks with mhc_pre's own kernels, an all_gather joins them,
     * and mhc_pre_finish completes the read on every rank (rad_glm_tp.h). */
    static constexpr int64_t kChunk = 2048, kPartStride = 28;
    rad_buf part = 0, parts = 0;
    rad_op op_partial = 0, op_gather = 0, op_finish = 0;
    int64_t split_rows = 0, part_row = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int layer, const char* name,
                const Config& cfg, const Wire& wire, const Src& src, bool defer_post = false) {
        g = geom;
        c = cfg;
        w = wire;
        if (!b || !name || !src.fn || !src.scale || !src.base || g.n_embd < 1 || g.max_tok < 1 ||
            c.hc < 2 || c.hc > 8 || c.iterations < 1 || c.iterations > 256 ||
            !std::isfinite(c.eps) || !std::isfinite(c.hc_eps) || c.eps <= 0 || c.hc_eps < 0 ||
            !w.h || !w.x || !w.y || !w.out || !w.post || !w.comb || w.h == w.out || w.h == w.x ||
            w.h == w.y || w.x == w.out || w.y == w.out || w.post == w.comb)
            return RAD_E_INVAL;
        const int64_t width = c.hc * g.n_embd, mix = c.hc * (c.hc + 2);
        const char *fn = nm.f("%s.fn", name), *scale = nm.f("%s.scale", name),
                   *base = nm.f("%s.base", name);
        RAD_ARCH_TRY(map_copy(b, fn, src.fn));
        RAD_ARCH_TRY(map_copy(b, scale, src.scale));
        RAD_ARCH_TRY(map_copy(b, base, src.base));
        // Mixing coefficients are F32 in the checkpoint; keep them replicated under TP.
        const rad_weight wf = w_fn = decl_w(b, fn, RAD_F32, { mix, width }, RAD_ACCESS_PER_TOKEN,
                                            RAD_SHARD_NONE, grp_layer(layer));
        const rad_weight ws = w_scale = decl_w(b, scale, RAD_F32, { 3 }, RAD_ACCESS_PER_TOKEN,
                                               RAD_SHARD_NONE, grp_layer(layer));
        const rad_weight wb = w_base = decl_w(b, base, RAD_F32, { mix }, RAD_ACCESS_PER_TOKEN,
                                              RAD_SHARD_NONE, grp_layer(layer));
        if (!wf || !ws || !wb) return RAD_E_INVAL;
        op_pre = rw(b,
                    RAD_OP(b, "mhc_pre",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                      RAD_INT("hc", c.hc), RAD_INT("iterations", c.iterations),
                                      RAD_F64("eps", c.eps), RAD_F64("hc_eps", c.hc_eps),
                                      RAD_STR("dtype", g.dtype)),
                           RAD_WEIGHTS(wf, ws, wb)),
                    { w.h }, { w.x, w.post, w.comb });
        if (!op_pre) return RAD_E_INVAL;
        RAD_ARCH_TRY(declare_split(b, nm, name));
        return defer_post ? RAD_OK : declare_post(b);
    }
    int declare_split(RadBuilder* b, Names& nm, const char* name) {
        const int64_t total = c.hc * g.n_embd / kChunk;
        split_rows = GlmTp::kSplitRows;
        if (g.world < 2 || g.n_embd != 4096 || c.hc != 4 || total % g.world) {
            split_rows = 0;
            return RAD_OK;
        }
        const int64_t mine = total / g.world, T = g.max_tok;
        part_row = mine * kPartStride;
        part = decl_b(b, nm.f("%s.part", name), RAD_F32, { T, part_row });
        parts = decl_b(b, nm.f("%s.parts", name), RAD_F32, { T, total * kPartStride });
        op_partial =
            rw(b,
               RAD_OP(b, "mhc_pre_partial",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd), RAD_INT("hc", c.hc),
                                 RAD_INT("chunk", kChunk), RAD_INT("first", g.rank * mine),
                                 RAD_INT("chunks", mine), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_fn)),
               { w.h }, { part });
        op_gather = rw(b,
                       RAD_OP(b, "all_gather",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", part_row, T * part_row),
                                         RAD_STR("dtype", "f32"), RAD_INT("row", part_row)),
                              RAD_NOWEIGHTS),
                       { part }, { parts });
        op_finish = rw(b,
                       RAD_OP(b, "mhc_pre_finish",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                         RAD_INT("hc", c.hc), RAD_INT("iterations", c.iterations),
                                         RAD_F64("eps", c.eps), RAD_F64("hc_eps", c.hc_eps),
                                         RAD_INT("chunk", kChunk), RAD_STR("dtype", g.dtype)),
                              RAD_WEIGHTS(w_scale, w_base)),
                       { parts, w.h }, { w.x, w.post, w.comb });
        return part && parts && op_partial && op_gather && op_finish ? RAD_OK : RAD_E_INVAL;
    }
    int declare_post(RadBuilder* b) {
        op_post = rw(b,
                     RAD_OP(b, "mhc_post",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                       RAD_INT("hc", c.hc), RAD_STR("dtype", g.dtype)),
                            RAD_NOWEIGHTS),
                     { w.y, w.h, w.post, w.comb }, { w.out });
        return op_post ? RAD_OK : RAD_E_INVAL;
    }
    void read(RadCtx* ctx, int64_t rows) const {
        if (split_rows && rows >= split_rows) {
            RAD_ISSUE_N(ctx, op_partial, rows, brows(w.h, rows), RAD_W(w_fn), brows(part, rows));
            RAD_ISSUE_N(ctx, op_gather, rows * part_row, brows(part, rows), brows(parts, rows));
            RAD_ISSUE_N(ctx, op_finish, rows, brows(parts, rows), brows(w.h, rows), RAD_W(w_scale),
                        RAD_W(w_base), brows(w.x, rows), brows(w.post, rows), brows(w.comb, rows));
            return;
        }
        RAD_ISSUE_N(ctx, op_pre, rows, brows(w.h, rows), RAD_W(w_fn), RAD_W(w_scale), RAD_W(w_base),
                    brows(w.x, rows), brows(w.post, rows), brows(w.comb, rows));
    }
    void write(RadCtx* ctx, int64_t rows) const {
        RAD_ISSUE_N(ctx, op_post, rows, brows(w.y, rows), brows(w.h, rows), brows(w.post, rows),
                    brows(w.comb, rows), brows(w.out, rows));
    }
};
}
} // namespace rad::arch
#endif
