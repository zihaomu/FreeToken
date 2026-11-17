/* quantise.cpp -- see quantise.h. */
#include "quantise.h"

#include <algorithm>
#include <vector>

namespace rad {

int quantise_weight(const RadQuantizerInfo& q, const RadParam* params, int n_params,
                    const RadQuantWeight& qw, const RadEncoding& enc, int64_t band_bytes,
                    const QuantIo& io, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    /* The movement group and the importance are known now; the encoding chosen without them must
     * not change. */
    {
        RadEncoding again{};
        if (q.encoding(params, n_params, &qw, &again) != RAD_OK || !rad_enc_equal(&again, &enc))
            return refuse(RAD_E_STATE,
                          std::string(q.name) + " answers with another encoding once it is told "
                          "the weight's movement group, and declare could not tell it that. An "
                          "encoding may depend on the weight's name and shape, not on where it is "
                          "placed.");
    }
    const int64_t R = qw.rows, C = qw.cols;
    int64_t rb = q.row_block ? q.row_block(params, n_params, &qw) : 0;
    if (rb <= 0 || rb > R) rb = R;
    int64_t blk = rb;
    if (rb < R) {
        const int64_t want = std::max<int64_t>(1, band_bytes / std::max<int64_t>(C * 4, 1));
        blk = std::max<int64_t>(rb, want - want % rb);
        if (blk > R) blk = R;
    }
    const int np = enc.n_planes;
    std::vector<int64_t> prow_bytes((size_t)np), pband((size_t)np);
    std::vector<std::vector<uint8_t>> whole((size_t)np), band((size_t)np);
    bool misaligned = false;
    for (int k = 0; k < np; ++k) {
        const RadEncPlane& p = enc.plane[k];
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&p, R, C, &pr, &pc);
        prow_bytes[(size_t)k] = rad_enc_row_bytes(p.dtype, pc);
        const bool banded = p.kind == RAD_PLANE_TILED && p.block[0] > 0;
        pband[(size_t)k] = banded ? p.block[0] : 0;
        if (banded && blk < R && blk % p.block[0]) misaligned = true;
        if (!banded) whole[(size_t)k].assign((size_t)(pr * prow_bytes[(size_t)k]), 0);
    }
    if (misaligned)
        return refuse(RAD_E_SHAPE, fmt("%s takes %lld rows a call, which splits a block of its "
                                       "own encoding's scales", q.name, (long long)rb));

    std::vector<float> f, ref;
    for (int64_t r0 = 0; r0 < R; r0 += blk) {
        const int64_t n = std::min(blk, R - r0);
        f.resize((size_t)(n * C));
        std::string w1;
        if (io.read(r0, n, f.data(), &w1) != RAD_OK) return refuse(RAD_E_FORMAT, w1);
        if (io.band) ref = f;                      /* a quantiser may rotate its input in place */
        std::vector<void*> ptr((size_t)np);
        for (int k = 0; k < np; ++k) {
            if (!pband[(size_t)k]) { ptr[(size_t)k] = whole[(size_t)k].data(); continue; }
            const int64_t rows_k = (n + pband[(size_t)k] - 1) / pband[(size_t)k];
            band[(size_t)k].assign((size_t)(rows_k * prow_bytes[(size_t)k]), 0);
            ptr[(size_t)k] = band[(size_t)k].data();
        }
        const int st = q.quantize(params, n_params, &qw, f.data(), r0, n, ptr.data());
        if (st != RAD_OK)
            return refuse(st, fmt("%s could not quantise rows [%lld, %lld): %s", q.name,
                                  (long long)r0, (long long)(r0 + n), rad_strerror(st)));
        for (int k = 0; k < np; ++k) {
            if (!pband[(size_t)k]) continue;
            if (io.write(k, (r0 / pband[(size_t)k]) * prow_bytes[(size_t)k],
                         band[(size_t)k].data(), (int64_t)band[(size_t)k].size()) != RAD_OK)
                return refuse(RAD_E_IO, fmt("writing plane %d of rows [%lld, %lld) failed", k,
                                            (long long)r0, (long long)(r0 + n)));
        }
        if (io.band) io.band(r0, n, ref.data(), (const void* const*)ptr.data());
    }
    for (int k = 0; k < np; ++k)
        if (!pband[(size_t)k] &&
            io.write(k, 0, whole[(size_t)k].data(), (int64_t)whole[(size_t)k].size()) != RAD_OK)
            return refuse(RAD_E_IO, fmt("writing plane %d failed", k));
    return RAD_OK;
}

}  /* namespace rad */
