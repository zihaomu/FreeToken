/* lq_gptq.h -- error-feedback quantisation on any grid, against a calibration Hessian.
 *
 * GPTQ picks the codes one column at a time and pushes each column's rounding error into the
 * columns not yet reached, weighted by the inverse Hessian of the layer's real input. Plain
 * rounding treats every column as independent, and a 4-bit grid lacks the levels to absorb that.
 *
 * This file holds the Hessian and the loop, not the grid: codes are chosen on whatever grid the
 * rule names (lq_grid.h), against scales fixed before the loop from the (transformed) weight. This
 * "static groups" form keeps a scale plane a function of the weight alone.
 *
 * The Hessian is the expert's own where the tap recorded one, else the layer's (lq_hess.h,
 * calib_resolve). A layer's pooled Hessian is found from the weight's logical name with the expert
 * index stripped, so its experts share one factor; an expert's own is factored for its one call.
 * The engine's calibration tap writes both (arch/common/rad_arch.h, RADIANCE_CALIB_DIR);
 * docs/MOE-W4.md describes the bootstrap.
 */
#pragma once
#include "lq_grid.h"
#include "lq_hess.h"

#include <string>

namespace lq {

/* Upper Cholesky factor U of the damped inverse Hessian, U^T U = (H + lambda I)^-1, as a K x K
 * row-major f32 plane, for weight `name`. Null when `dir` has no calibration for it (not an error:
 * the weight is then rounded to nearest). Also null, with a message, when a file exists but is
 * unusable, so a mistyped directory is not a silent quality loss. With `perm` (gptq_act_order's)
 * it factors the permuted Hessian P H P^T. `domain` is the grid's transform width, 0 without one
 * (calib_sum). A layer's factor is cached for the process by (dir, stem, K, domain, damp, act
 * order); an expert's own is not. */
Gram gptq_factor(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                 const int32_t* perm = nullptr);

/* gptq_factor in act order (gptq_act_order's), the order into `order`: one read of the calibration
 * for both, cached with the factor. */
Gram gptq_factor_ordered(const char* dir, const char* name, int64_t K, int64_t domain, double damp,
                         std::vector<int32_t>* order);

/* Act order: columns by descending Hessian diagonal (the input energy each carries), ties in
 * column order. The identity when `dir` holds nothing for the weight. */
std::vector<int32_t> gptq_act_order(const char* dir, const char* name, int64_t K, int64_t domain);

/* Quantise `w` [N, K] in place on grid `g` against `scales` (grid_scales'), writing codes to
 * `codes` [N, K]. On return `w` holds the decoded values. With `u` null each value is rounded to
 * nearest on its own, which is byte-identical to grid_rtn. A trellis grid runs on the cards when
 * gpu_set_devices named any (lq_gpu.h), with the same result; RAD_OK, or the card's failure with
 * the reason in *why.
 *
 * `dynamic`: each row's scale for a 128-column group is chosen again when the loop reaches the
 * group, from the values the earlier columns' error has moved it to (GPTQ's own grouping), rather
 * than from the weight as it stood; `scales` is updated. Needs a one-row, 128-column scale grid
 * with one level (gptq_dynamic_ok). */
int gptq_quantise(const Grid& g, float* w, int32_t* codes, Scales& scales, int64_t N,
                  int64_t K, const float* u, bool dynamic = false, std::string* why = nullptr);
/* tcq_colf: the blocks' shared factors into scales.colf and each row's scale against them, from
 * the weight `w` [N, K] as it stands (Grid::tcq_colf). */
void gptq_tcq_colf(const float* w, Scales& scales, int64_t N, int64_t K);

bool gptq_dynamic_ok(const Grid& g, const Scales& s);

/* The trellis loop's two scale steps, shared by the host loop and the card's (lq_tcq_gpu.hip).
 * gptq_tcq_dyn_scales: under dyn=1, rows [r0, r1)'s scales for the block [b0, b1), chosen from
 * `w` as it stands. gptq_tcq_col_scales: for the n rows from n0 (a multiple of tcq_rows), the
 * scale each row's values take in column `col`, and under tcq_rb the row's own scale over its row
 * block's into `rt` (Grid::tcq_rb). */
void gptq_tcq_dyn_scales(const Grid& g, const float* w, int64_t K, Scales& scales, int64_t r0,
                         int64_t r1, int64_t b0, int64_t b1);
void gptq_tcq_col_scales(const Grid& g, const Scales& scales, int64_t n0, int64_t n, int64_t col,
                         float* s, float* rt);

/* Coordinate descent after the GPTQ loop: `passes` sweeps over each row's columns, re-choosing each
 * code at its conditional optimum (the value minimising the row's e (H + lambda I) e^T,
 * e = decoded - x, other columns held) against the same damped Hessian. GPTQ fixes each column
 * before later ones are known; a sweep revisits it with all in place and never raises the
 * objective. `x` [N, K] is the (transformed) weight GPTQ was given, `w` its decoded result,
 * updated in place with `codes`. `h` is the undamped Gram (calib_gram's); null means no-op. */
void gptq_refine(const Grid& g, const float* x, float* w, int32_t* codes, const Scales& scales,
                 int64_t N, int64_t K, const float* h, double damp, int passes);

/* Each row's scale refitted after the loop, the codes held: the row's decoded values w scaled by
 * a = (w H x^T) / (w H w^T), the factor minimising (a w - x) H (a w - x)^T. GPTQ chooses a row's
 * codes against a scale fixed beforehand from the weight alone; the refit is the least-squares
 * scale for the codes it chose, under the same (undamped) Hessian. Rows with one static scale
 * (Scales with one block a row, f32 so the stored value moves exactly); `x`, `w` and `h` as for
 * gptq_refine, `w` scaled in place. A factor that is not positive leaves its row alone. */
void gptq_rescale_rows(const float* x, float* w, Scales& scales, int64_t N, int64_t K,
                       const float* h);


}  /* namespace lq */
