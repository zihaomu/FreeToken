/* quantise.h -- one weight through a quantiser plugin, as rad-convert writes it.
 *
 * The loop asks the quantiser for its rows per call, hands it bands of a multiple of that near
 * `band_bytes` of f32 (so a table of tens of GB streams and a GPTQ weight, whose row block is 0, is
 * passed whole), writes each row-banded plane per band, and holds whole-tensor planes (a codebook, a
 * per-tensor scale) until the last call. rad-convert writes a container through it, and the
 * quantiser tests (tests/quant_fixture.h) run the same loop into memory, so the tests exercise the
 * path every container is made by.
 */
#pragma once
#include "../rad_internal.h"
#include "rad_encoding.h"
#include "rad_quant.h"

#include <functional>
#include <string>

namespace rad {

/* The band rad-convert quantises at: 64 MiB of f32 rows. */
constexpr int64_t kQuantBandBytes = (int64_t)64 << 20;

struct QuantIo {
    /* Rows [r0, r0 + n) of the source weight as f32 [n, cols]. RAD_OK, or a failure with *why. */
    std::function<int(int64_t r0, int64_t n, float* out, std::string* why)> read;
    /* `bytes` of plane `k`, at byte `off` within that plane. RAD_OK, or a failure. */
    std::function<int(int k, int64_t off, const void* data, int64_t bytes)> write;
    /* Optional, after each quantise call: the band's source rows as read, before the quantiser
     * changed them in place, and the planes that call wrote (a row-banded plane's rows for this
     * band, a whole plane from its start), as a decoder takes them over `n` rows. */
    std::function<void(int64_t r0, int64_t n, const float* src, const void* const* planes)> band;
};

/* Quantise `qw` under quantiser `q` and its `params` into `enc`, the encoding the quantiser answered
 * when the weight was planned. Returns RAD_OK, or with the reason in *why:
 *   RAD_E_STATE    given `qw`'s movement group and importance, the quantiser answers another
 *                  encoding than the one planned;
 *   RAD_E_SHAPE    its row block splits a block of its own encoding's scales;
 *   RAD_E_FORMAT   the read failed;
 *   RAD_E_IO       a write failed;
 *   otherwise      the quantiser's own refusal. */
int quantise_weight(const RadQuantizerInfo& q, const RadParam* params, int n_params,
                    const RadQuantWeight& qw, const RadEncoding& enc, int64_t band_bytes,
                    const QuantIo& io, std::string* why);

}  /* namespace rad */
