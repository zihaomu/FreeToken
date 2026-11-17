/* lq_quant.h -- shared by every libquant quantiser: the grid options a recipe spells, reading a
 * grid off a rule, and the table each algorithm file exports its quantiser through.
 *
 * lq_registry.cpp holds the plugin's exports and the short quantisers (rtn, gptq, cast). An
 * algorithm with its own loop (AWQ, AutoRound, ParoQuant) has its own file and returns its
 * RadQuantizerInfo from the function named for it below.
 */
#pragma once
#include "lq_grid.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

/* The grid options, shared by every quantiser over a grid. */
#define LQ_GRID_OPTIONS                                                                          \
    { "codes", RAD_P_STR, nullptr,                                                               \
      "code dtype: i2..i8, i16, i32 (signed, symmetric), u1..u8, u16 with zero= (asymmetric), "  \
      "fp8_e4m3, fp8_e5m2, fp6_e2m3, fp6_e3m2, fp4_e2m1" },                                      \
    { "table", RAD_P_STR, nullptr,                                                               \
      "a codebook the codes index: nf4, iq4nl, w4nl, w5nl, binary, ternary, or values "          \
      "separated by ;" },                                                                        \
    { "vq", RAD_P_STR, nullptr,                                                                \
      "a vector codebook file: one codeword a line, its integers (-127..127) separated by "     \
      "spaces; one code a codeword's width of columns" },                                       \
    { "tcq", RAD_P_INT, nullptr,                                                               \
      "a trellis of 2^L states down each column over codes=uK, one code a row (gptq)" },         \
    { "tcq_rows", RAD_P_INT, "256", "the rows one tail-biting trellis runs over" },             \
    { "tcq_gain", RAD_P_F64, "1", "a trellis block's rms over the table's, its scale's divisor" }, \
    { "tcq_peak", RAD_P_INT, "0",                                                              \
      "1: a trellis block's scale never puts its largest value past the table's" },              \
    { "tcq_rb", RAD_P_INT, "0",                                                                \
      "R: a trellis's values on the int8 grid of one scale per R rows (the block's largest)" },  \
    { "tcq_code", RAD_P_STR, nullptr,                                                          \
      "mad: the states' values computed, bytesum((st*A+B) & 0x3F3F3F3F) - 126 (no tcq_lut=); "  \
      "mad16: bytesum(st*A+B) - 510, wider than int8, for the 16-bit path (gptqsim only)" },     \
    { "tcq_a", RAD_P_INT, "11872635", "tcq_code=mad's multiplier A (24 bits)" },               \
    { "tcq_b", RAD_P_INT, "625341585", "tcq_code=mad's addend B" },                             \
    { "tcq_colf", RAD_P_INT, "0",                                                              \
      "1: one scale a row times a factor a 128-column block shared by all rows (block=1x*)" },   \
    { "tcq_lut", RAD_P_STR, nullptr,                                                           \
      "the trellis's 2^L state values, integers in -127..127 (default: normal quantiles)" },     \
    { "group", RAD_P_INT, nullptr, "one scale per this many columns of a row" },               \
    { "block", RAD_P_STR, "1x*", "the scale block RxC, * for a whole extent: 128x128, *x*" },    \
    { "scale", RAD_P_STR, "bf16",                                                              \
      "scale dtype: f32, bf16, f16, fp8_e4m3, e8m0; under scale2= also u4..u8, i6, i8" },        \
    { "scale2", RAD_P_STR, nullptr,                                                            \
      "a second scale level the first is relative to: f32, bf16, f16, fp8_e4m3, e8m0" },        \
    { "group2", RAD_P_INT, nullptr, "the second level's block, this many columns of a row" },  \
    { "block2", RAD_P_STR, nullptr, "the second level's block RxC (default *x*, the tensor)" }, \
    { "scale2_value", RAD_P_F64, nullptr,                                                      \
      "the second level fixed at this value, exact in scale2's dtype, for every block of it" },   \
    { "zero", RAD_P_STR, nullptr, "an integer zero point a block (asymmetric): u8, u4, ..." },  \
    { "transform", RAD_P_STR, nullptr, "fwhtN: rotate each row by the unnormalised Hadamard" }, \
    { "rule", RAD_P_STR, "absmax",                                                             \
      "absmax | mx (OCP shared exponent) | fixed | search (least squared error, imatrix-"        \
      "weighted when the converter has one)" },                                                 \
    { "scale_value", RAD_P_F64, nullptr, "the scale every block takes under rule=fixed" },     \
    { "clamp", RAD_P_STR, "full", "a signed code's range: full or sym" }

/* The calibration option every calibrated quantiser takes. */
#define LQ_CALIB_OPTION(WITHOUT)                                                                 \
    { "calib", RAD_P_STR, nullptr,                                                               \
      "the directory an engine calibration run wrote (RADIANCE_CALIB_DIR); without one, or for "  \
      "a weight it holds nothing for, " WITHOUT }

namespace lq {

/* The grid a rule's options describe, checked against the weight; a message on stderr, with the
 * weight's name, when it is refused. */
int grid_of(const RadParam* o, int n_o, const RadQuantWeight* w, Grid* g, bool vq_ok = false);

/* Rank-1 weights (norms, biases) are declined rather than quantised, so a recipe pattern need not
 * exclude every vector it happens to match. */
inline bool declines(const RadQuantWeight* w) { return !w || w->rank < 2; }

/* Say why a rule is refused, with the weight's name, and return RAD_E_INVAL. */
int refuse(const RadQuantWeight* w, const char* why);

/* A 64-bit hash of a string, for a seed that is the same on every machine and every run. */
uint64_t name_seed(const char* s);

/* The rows of `w` [N, K] in the listed row units of `unit` rows each (the last unit may be short),
 * packed in list order into `out`. Returns the row count. */
inline int64_t gather_units(const float* w, int64_t N, int64_t K, int64_t unit,
                            const std::vector<int64_t>& units, std::vector<float>* out) {
    int64_t rows = 0;
    for (int64_t u : units) rows += std::min(unit, N - u * unit);
    out->resize((size_t)(rows * K));
    int64_t at = 0;
    for (int64_t u : units) {
        const int64_t n = std::min(unit, N - u * unit);
        std::memcpy(out->data() + at * K, w + u * unit * K, (size_t)(n * K) * sizeof(float));
        at += n;
    }
    return rows;
}

/* The algorithm files' quantisers. */
const RadQuantizerInfo& awq_info();
const RadQuantizerInfo& autoround_info();
const RadQuantizerInfo& paroquant_info();
const RadQuantizerInfo& m2xfp_info();

}  /* namespace lq */
