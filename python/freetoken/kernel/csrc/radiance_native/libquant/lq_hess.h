/* lq_hess.h -- the calibration Hessians the calibrated quantisers read, and the product they spend
 * their time in.
 *
 * The engine's calibration tap (arch/common/rad_arch.h, calib_write) sums x x^T over the real
 * input of a layer's projection, per rank, into `gram.r<rank>.<stem>.bin`. The input is the one
 * the kernel multiplies, so each file records its domain: the model's own activation (domain 0),
 * or that activation rotated block by block by the unnormalised Hadamard of width N (domain N),
 * which is what a container storing rotated weights multiplies. A reader asks for its grid's
 * transform domain and each file is converted before the sum: rotating is H -> F H F, and back is
 * the same divided by N^2 (F is its own inverse up to N). So calibration from a bf16 model and from
 * a rotated 4-bit container both serve a rotated recipe.
 *
 * A layer-local quantiser's error is the quadratic form
 *
 *     L = sum over rows r of (w_hat_r - w_r) H (w_hat_r - w_r)^T
 *
 * which decomposes over rows: rows share H and nothing else. GPTQ, AWQ's scale search, AutoRound's
 * rounding descent and ParoQuant's rotation fit all minimise it; the last three spend their time
 * in D . H (mat_hess).
 *
 * The most specific Hessian available is used: a routed expert's own when the tap recorded one (it
 * does for `down`, whose input is the expert's own intermediate units), else the layer's, pooled
 * over its experts and found from the name with the expert index stripped (calib_stem). A layer's
 * experts share that one and the process reads it once.
 */
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lq {

/* A Hessian or a factor of one. Shared, so an uncached one (an expert's own) lives exactly as long
 * as the call that read it. */
using Gram = std::shared_ptr<const std::vector<float>>;

/* `blk.7.ffn_gate_up_exps.123.weight` -> `blk.7.ffn_gate_up_exps`: the stem a layer's pooled
 * Hessian file is named by. A vocabulary-weighted head's width class, `output.k4.weight`, pools
 * to `output`. */
std::string calib_stem(const char* name);

/* The stem `name`'s Hessian is read from in `dir`: the weight's own (the name less `.weight`,
 * `blk.7.ffn_down_exps.123`) when its rank files together hold at least kOwnRowsPerK * K rows,
 * else calib_stem's, noted on stderr. A GPTQ fit to a Gram of few rows overfits them: on
 * Qwen3.8-Flash-Next's down projection an expert's own Gram cuts held-out output error by 20-55%
 * versus the pooled one, except for experts routed under ~2K rows, where it is worse than
 * round-to-nearest. Those carry a fraction of a percent of the routing. */
constexpr int64_t kOwnRowsPerK = 4;
std::string calib_resolve(const char* dir, const char* name, int64_t K);

/* Every rank's file for `stem` in `dir`, each converted into `domain` (0, or the Hadamard width a
 * rotated grid quantises in), summed into `acc` (K x K, double, row-major). Returns the number of
 * files read, 0 when there are none. A file that exists but is unusable (wrong magic, wrong K,
 * short, or a domain that cannot be converted) is reported on stderr and skipped.
 *
 * With no whole file, the parts: under tensor parallelism a column-split linear (an attention
 * output or down projection) sees only its rank's K / world input columns, so each rank's tap
 * writes the Gram of those as `<stem>.part` (gram.r<rank>.<stem>.part.bin). They are placed on
 * the diagonal in rank order, the cross-rank products, never formed on one card, left zero. The
 * parts must cover K exactly and be in domain 0 (a rotation of K does not keep the blocks); the
 * result is then converted into `domain` like a whole file. */
int calib_sum(const char* dir, const std::string& stem, int64_t K, int64_t domain,
              std::vector<double>* acc);

/* The summed Gram of `name`'s input in `domain` as f32 K x K, from calib_resolve's stem, or null
 * when `dir` holds none (noted once on stderr). The last few layers' are cached by (dir, stem, K,
 * domain), for a MoE layer's experts; an expert's own is read per call, since each is used once
 * and a model's worth would not fit. */
Gram calib_gram(const char* dir, const char* name, int64_t K, int64_t domain, const char* who);

/* G [R x K] = D [R x K] . H [K x K], H symmetric. Each element of G sums over k in ascending order
 * regardless of thread count and vector width, so the result depends on the inputs alone: two
 * converts of the same weight must write the same file. Threads over column panels of H and row
 * blocks of D. */
void mat_hess(const float* D, int64_t R, int64_t K, const float* H, float* G);

/* out[r] = D_r . G_r, the quadratic form of each row, summed in double. */
void row_dots(const float* D, const float* G, int64_t R, int64_t K, double* out);

}  /* namespace lq */
