/* avx_sam_attn.cpp -- Unlimited-OCR's DeepEncoder (SAM ViT-B/16) attention with decomposed relative
 * position (arch/common/rad_block_sam.h; libref/ref_sam_attn.cpp is the oracle).
 *
 * The logits of a query row i against a key j of the same window tile are
 *
 *     scale * q.k  +  q.Rh[i_h, j_h]  +  q.Rw[i_w, j_w]
 *
 * over the head's whole head_dim, the two tables indexed by the grid offset inside the tile. With
 * `window` > 0 the grid is padded with ZERO rows to a multiple of the window and cut into
 * window x window tiles (deepencoder's window_partition, after norm1), so the pad rows are keys and
 * values of their tile and take part in the softmax; `window` 0 is one tile over the segment.
 *
 * This is the same structure as libref's: one thread owns one output row and one head, the tile's
 * keys are walked jh-major so the q.Rh dot is computed once a tile row and the q.Rw dots are
 * hoisted per query row (the hip kernel's hoist, bit-identical reuse), the softmax subtracts the
 * tile maximum, and the output is rounded once. The dots stay scalar f32 loops over head_dim on
 * purpose: the x86 leg is the no-GPU/harness path, and a SIMD reduction would reassociate the f32
 * accumulation and break bit-parity with the libref oracle. The tables are widened to f32 once a
 * launch and resampled along dim 0 when a geometry's axes need it (get_rel_pos's linear
 * interpolation).
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace avx;

namespace {

/* A rel_pos table as f32 rows of `hd`, resampled to `want` rows when its own length differs:
 * F.interpolate(..., size=want, mode="linear") with align_corners False. */
static std::vector<float> table_rows(const RadTensor* t, int64_t hd, int64_t want) {
    const int64_t rows = hd > 0 ? numel(t) / hd : 0;
    std::vector<float> raw((size_t)(rows * hd), 0.0f);
    const int64_t st = laststride(t);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t base = rowoff(t, r, hd);
        for (int64_t c = 0; c < hd; ++c) raw[(size_t)(r * hd + c)] = ldt(t, base + c * st);
    }
    if (rows == want || rows <= 0 || want <= 0) return raw;
    std::vector<float> out((size_t)(want * hd), 0.0f);
    const double span = (double)rows / (double)want;
    for (int64_t d = 0; d < want; ++d) {
        double src = ((double)d + 0.5) * span - 0.5;
        if (src < 0.0) src = 0.0;
        if (src > (double)(rows - 1)) src = (double)(rows - 1);
        const int64_t i0 = (int64_t)src;
        const int64_t i1 = i0 + 1 < rows ? i0 + 1 : i0;
        const float f = (float)(src - (double)i0);
        for (int64_t c = 0; c < hd; ++c) {
            const float a = raw[(size_t)(i0 * hd + c)], b = raw[(size_t)(i1 * hd + c)];
            out[(size_t)(d * hd + c)] = a + (b - a) * f;
        }
    }
    return out;
}

}  /* namespace */

AVX_KERNEL(sam_attn) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *cu = &a->t[3];
    const RadTensor *o = &a->t[4], *th_tab = &a->t[5], *tw_tab = &a->t[6];

    const int64_t hd  = p_int(a, "head_dim", 0);
    const int64_t nh  = p_int(a, "n_head", 0);
    const int64_t gh  = p_int(a, "grid_h", 0);
    const int64_t gw  = p_int(a, "grid_w", 0);
    const int64_t win = p_int(a, "window", 0);
    if (hd <= 0 || nh <= 0 || gh <= 0 || gw <= 0 || win < 0) return RAD_E_SHAPE;
    const int64_t W = nh * hd, R = gh * gw;
    if (!q->data || !k->data || !v->data || !o->data || !th_tab->data || !tw_tab->data)
        return RAD_E_INVAL;
    for (const RadTensor* t : { q, k, v, o })
        if (t->rank < 2 || t->shape[t->rank - 1] != W) return RAD_E_SHAPE;
    const int64_t M = numel(q) / W;
    if (M <= 0) return RAD_E_SHAPE;
    if (numel(k) != M * W || numel(v) != M * W || numel(o) < M * W) return RAD_E_SHAPE;

    int64_t nseq = 1;
    if (cu->data) {
        if (cu->rank == 0 || numel(cu) < 2) return RAD_E_SHAPE;
        nseq = numel(cu) - 1;
        if (ldi(cu, offlin(cu, 0)) != 0 || ldi(cu, offlin(cu, nseq)) != M) return RAD_E_SHAPE;
        for (int64_t s = 0; s < nseq; ++s)
            if (ldi(cu, offlin(cu, s + 1)) - ldi(cu, offlin(cu, s)) != R) return RAD_E_SHAPE;
    } else if (M != R) {
        return RAD_E_SHAPE;
    }

    const float scale = p_f32(a, "scale", (float)(1.0 / std::sqrt((double)hd)));
    const int64_t th = win > 0 ? win : gh, tw = win > 0 ? win : gw;
    const int64_t keys = th * tw;
    if (numel(th_tab) < hd || numel(tw_tab) < hd) return RAD_E_SHAPE;
    const std::vector<float> Rh = table_rows(th_tab, hd, 2 * th - 1);
    const std::vector<float> Rw = table_rows(tw_tab, hd, 2 * tw - 1);

    AVX_PARALLEL_FOR_IF(nseq * R * nh * keys * hd >= AVX_PAR_MIN)
    for (int64_t work = 0; work < nseq * R; ++work) {
        const int64_t s  = work / R, r = work % R;
        const int64_t ih = r / gw, iw = r % gw;
        const int64_t ty = win > 0 ? ih / win : 0, tx = win > 0 ? iw / win : 0;
        const int64_t lh = ih - ty * th, lw = iw - tx * tw;
        const int64_t base = cu->data ? ldi(cu, offlin(cu, s)) : 0;
        const int64_t out_row = base + r;

        float* logits = scratch_f32((size_t)keys);
        float* acc = scratch_f32_b((size_t)(2 * hd));
        float* qt = acc + hd;

        for (int64_t h = 0; h < nh; ++h) {
            const int64_t qbase = rowoff(q, out_row, W) + h * hd;
            const int64_t qst = laststride(q);
            for (int64_t c = 0; c < hd; ++c) qt[c] = ldt(q, qbase + c * qst);

            /* The w rel-pos dot depends on (lw, jw) alone, not on jh: dot it once per query
             * row and reuse it through the tile rows. The same scalar loop in the same order,
             * so every reused value is bit-identical to the per-(jh, jw) recompute. */
            std::vector<float> wr((size_t)tw);
            for (int64_t jw = 0; jw < tw; ++jw) {
                const float* rwrow = &Rw[(size_t)((lw - jw + tw - 1) * hd)];
                float wv = 0.0f;
                for (int64_t c = 0; c < hd; ++c) wv += qt[c] * rwrow[c];
                wr[(size_t)jw] = wv;
            }

            for (int64_t jh = 0; jh < th; ++jh) {
                const float* rhrow = &Rh[(size_t)((lh - jh + th - 1) * hd)];
                float hr = 0.0f;
                for (int64_t c = 0; c < hd; ++c) hr += qt[c] * rhrow[c];
                for (int64_t jw = 0; jw < tw; ++jw) {
                    float lgt = hr + wr[(size_t)jw];
                    const int64_t kh = ty * th + jh, kw = tx * tw + jw;
                    if (kh < gh && kw < gw) {
                        const int64_t krow = base + kh * gw + kw;
                        const int64_t kb = rowoff(k, krow, W) + h * hd;
                        const int64_t ks = laststride(k);
                        float s_ = 0.0f;
                        for (int64_t c = 0; c < hd; ++c) s_ += qt[c] * ldt(k, kb + c * ks);
                        lgt += scale * s_;
                    }
                    logits[jh * tw + jw] = lgt;
                }
            }

            float m = -INFINITY;
            for (int64_t j = 0; j < keys; ++j) m = logits[j] > m ? logits[j] : m;
            float sum = 0.0f;
            for (int64_t j = 0; j < keys; ++j) {
                logits[j] = sc_exp(logits[j] - m);
                sum += logits[j];
            }
            for (int64_t c = 0; c < hd; ++c) acc[c] = 0.0f;
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int64_t jh = 0; jh < th; ++jh) {
                const int64_t kh = ty * th + jh;
                if (kh >= gh) continue;
                for (int64_t jw = 0; jw < tw; ++jw) {
                    const int64_t kw = tx * tw + jw;
                    if (kw >= gw) continue;
                    const float p = logits[jh * tw + jw] * inv;
                    if (p == 0.0f) continue;
                    const int64_t krow = base + kh * gw + kw;
                    const int64_t vb = rowoff(v, krow, W) + h * hd;
                    const int64_t vs = laststride(v);
                    for (int64_t c = 0; c < hd; ++c) acc[c] += p * ldt(v, vb + c * vs);
                }
            }

            const int64_t ob = rowoff(o, out_row, W) + h * hd;
            const int64_t os = laststride(o);
            for (int64_t c = 0; c < hd; ++c) stt(o, ob + c * os, acc[c]);
        }
    }
    return RAD_OK;
}
