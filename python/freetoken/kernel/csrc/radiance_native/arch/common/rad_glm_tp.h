/* GLM-5.3's tensor parallelism in one place: what a rank's shard of the model is, from which step
 * size a replicated or head-split piece of work splits over the ranks instead, how a step's rows
 * fall into one run a rank, and the column split of a replicated F32 projection.
 *
 * The residual stream is replicated (spec.md §9). Three kinds of split share the ranks' work:
 *
 *   heads     the MLA and KDA heads, with them the rows of their projections, the KDA state and
 *             the convolution history; the experts' and the dense feed-forward's width
 *             (`heads`, `kda_width`, `expert_width`, `dense_width`); joined by an all_reduce
 *   rows      a prefill-sized step's rows, one run a rank, each rank computing its run with the
 *             unsplit kernels (`run_rows`, `run_of`); joined by an all_gather of whole runs
 *   columns   a replicated F32 projection's output columns (SplitF32Proj); joined by a
 *             row-interleaved all_gather, bit for bit the replicated projection
 *
 * The one rule for a run: ceil(T / world) rows, rounded up to whole 16-byte words of the gathered
 * row, because an all_gather moves whole words. The MLA gathers 4096-wide bf16 rows, so its runs
 * are ceil(T / world); the indexer gathers one i32 length a row beside its page tables, so its
 * runs are multiples of four rows. The last rank's run may be short, or lie wholly past a small
 * step (`run_of`). */
#ifndef RAD_GLM_TP_H
#define RAD_GLM_TP_H
#include "rad_arch.h"
#include <algorithm>
#include <numeric>

namespace rad {
namespace arch {
struct GlmTp {
    /* The model's widths a rank's shard divides: 64 heads of 128 (KDA) and 256 (MLA), experts of
     * 2048 and a dense feed-forward of 12288. */
    static constexpr int64_t kHeads = 64, kKdaDim = 128, kMlaDim = 256, kExpertFF = 2048,
                             kDenseFF = 12288;
    static bool world_ok(int world) { return world == 1 || world == 2 || world == 4 || world == 8; }
    static int64_t heads(int world) { return kHeads / world; }
    static int64_t kda_width(int world) { return heads(world) * kKdaDim; }
    static int64_t mla_width(int world) { return heads(world) * kMlaDim; }
    static int64_t expert_width(int world) { return kExpertFF / world; }
    static int64_t dense_width(int world) { return kDenseFF / world; }

    /* Rows from which a step splits by rows or columns over the ranks. Below it, as in decode, the
     * collective's latency outweighs the saved arithmetic and every rank runs the replicated or
     * head-split form. */
    static constexpr int64_t kSplitRows = 256;
    /* Rows x live blocks from which the indexer splits a smaller step as well: a decode step at
     * long context, where scoring every query row against every block of its sequence outweighs
     * the two gathers. On one R9700 48 rows over 8000 blocks (32k of context) score and select in
     * about 250 us and a run of 8 in 50, against roughly 50 us for the two eight-rank gathers;
     * over 4000 blocks the saving is about the gathers' cost, and six rows never pay. From 16
     * rows, so that at least four ranks share the work and every run fits in the step. */
    static constexpr int64_t kSplitWork = 1 << 18, kSplitWorkRows = 16;
    static bool splits(int64_t rows, int world) { return world > 1 && rows >= kSplitRows; }

    /* A rank's run of a step of `rows`, gathered as rows of `row_bytes`: ceil(rows / world) rounded
     * up to whole 16-byte words of the gathered row. Every rank but possibly the last takes a whole
     * run; the planes a split writes hold world runs, which can reach past the step. */
    static int64_t run_rows(int64_t rows, int world, int64_t row_bytes) {
        const int64_t word = 16 / std::gcd<int64_t>(16, row_bytes);
        return word * ((rows + word * world - 1) / (word * world));
    }
    struct Run {
        int64_t run = 0, first = 0, rows = 0;
    };
    /* Rank `rank`'s rows [first, first + rows) of a step of `rows`. A rank whose run lies wholly
     * past a small step takes the step's first run instead, so every plane it computes from and
     * gathers is written; what it computes there is dead rows past the step, and the gather lays
     * them past T, where nothing reads them. */
    static Run run_of(int64_t rows, int world, int rank, int64_t row_bytes) {
        Run r;
        r.run = run_rows(rows, world, row_bytes);
        r.first = rank * r.run;
        r.rows = std::min(r.run, rows - r.first);
        if (r.rows <= 0) {
            r.first = 0;
            r.rows = r.run;
        }
        return r;
    }

    /* The all_reduce of a replicated 4096-wide bf16 row over the ranks: the embedding's (exact)
     * and each head-split block's (the deployment's wire). */
    static rad_op reduce(RadBuilder* b, const Geom& g, rad_buf x, bool exact = false) {
        if (g.world == 1) return 0;
        return rw(b,
                  RAD_OP(b, "all_reduce",
                         RAD_PARAMS(RAD_INT("world_size", g.world),
                                    RAD_RANGE("numel", 4096, g.max_tok * 4096),
                                    RAD_STR("dtype", "bf16"),
                                    RAD_INT("exact", exact ? 1 : g.wire_exact),
                                    RAD_INT("min_bytes", g.wire_min_bytes)),
                         RAD_NOWEIGHTS),
                  { x }, { x });
    }
};

/* The split's ops are declared whatever the step size, and each gather's message as a range of
 * rows: the engine's arena levels declare again at a smaller max_tok and must get the same graph
 * with the same fixed parameters.
 *
 * A replicated [n, k] F32 projection computed column-split: rank r runs the same gemm_nt_f32 on
 * rows [r*n/world, (r+1)*n/world) of the weight and a row-interleaved all_gather lays the slices
 * out as the full [M, n]. The kernel computes every output column independently of n and the
 * gather moves bytes, so each value is bit-identical to the replicated projection's. */
struct SplitF32Proj {
    rad_buf part = 0;
    rad_op op_part = 0, op_gather = 0;
    int64_t cols = 0, k = 0, min_rows = 0, first_col = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& g, const char* name, int64_t rows, int64_t n,
                int64_t width, rad_weight weight, rad_buf input, rad_buf output) {
        min_rows = GlmTp::kSplitRows;
        if (g.world < 2 || n % g.world || (n / g.world * 4) % 16) {
            min_rows = 0;
            return RAD_OK;
        }
        cols = n / g.world;
        k = width;
        first_col = g.rank * cols;
        part = decl_b(b, nm.f("%s.part", name), RAD_F32, { rows, cols });
        op_part = rw(b,
                     RAD_OP(b, "gemm_nt_f32",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", cols), RAD_INT("K", k),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(weight)),
                     { input }, { part });
        op_gather = rw(
            b,
            RAD_OP(b, "all_gather",
                   RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", cols, rows * cols),
                              RAD_STR("dtype", "f32"), RAD_INT("row", cols)),
                   RAD_NOWEIGHTS),
            { part }, { output });
        return part && op_part && op_gather ? RAD_OK : RAD_E_INVAL;
    }
    bool takes(int64_t rows) const { return min_rows && rows >= min_rows; }
    void step(RadCtx* ctx, int64_t rows, RadOperand input, rad_weight weight,
              RadOperand output) const {
        RadOperand slice = RAD_W(weight);
        slice.offset = first_col * k;
        slice.rows = cols;
        RAD_ISSUE_N(ctx, op_part, rows, input, slice, brows(part, rows));
        RAD_ISSUE_N(ctx, op_gather, rows * cols, brows(part, rows), output);
    }
};
}
} // namespace rad::arch
#endif
