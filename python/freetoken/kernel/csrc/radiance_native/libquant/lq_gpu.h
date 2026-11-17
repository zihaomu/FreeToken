/* lq_gpu.h -- GPTQ over a trellis grid on the card.
 *
 * The trellis's Viterbi search is nearly all of a trellis recipe's conversion time: every value
 * of every column takes a pass over the 2^L states. Each run of tcq_rows rows is independent of
 * every other through the whole loop (the error feedback stays inside a row, the search inside a
 * run), so a run is one workgroup and a weight is as many workgroups as it has runs.
 *
 * The card performs the host loop's operations in the host's order: the same products, sums and
 * comparisons, in f32 without contraction (-ffp-contract=off on both sides), ties to the first
 * state, IEEE division and rounding. The codes and the decoded weight are therefore the host's,
 * bit for bit, which quant_tcq_gpu_test checks. The scales a dyn=1 block chooses are computed on
 * the host by the host's own code, from the block's columns as the card left them.
 *
 * Only built where radiance has a device build (RAD_HIP_FOUND); elsewhere gpu_set_devices
 * answers RAD_E_UNSUPPORTED and the host loop is the only one.
 */
#pragma once
#include "lq_grid.h"

#include <string>

namespace lq {

/* The cards trellis GPTQ runs on (rad_quant.h, RadQuantDevicesFn); n = 0 is the host alone.
 * RAD_E_UNSUPPORTED without a device build, RAD_E_DEVICE for a card that is not there. */
int gpu_set_devices(const int* ids, int n);

/* Whether gpu_set_devices named any card. */
bool gpu_enabled();

/* gptq_quantise over a trellis grid (lq_gptq.cpp's gptq_quantise_tcq) on the cards: the same
 * codes, the same decoded `w` and the same `scales`. RAD_OK, or a failure with the reason in
 * *why; nothing is written to `codes` then. */
int gptq_tcq_gpu(const Grid& g, float* w, int32_t* codes, Scales& scales, int64_t N, int64_t K,
                 const float* u, bool dynamic, std::string* why);

}  /* namespace lq */
