/* ref_sam_attn.cpp -- Unlimited-OCR's DeepEncoder (SAM ViT-B/16) attention with decomposed relative
 * position (arch/common/rad_block_sam.h).
 *
 * The reference is the checkpoint's own deepencoder.py (Attention.forward and
 * add_decomposed_rel_pos), whose logits are
 *
 *     scale * q.k  +  q.Rh[i_h, j_h]  +  q.Rw[i_w, j_w]
 *
 * where Rh = get_rel_pos(rel_pos_h, q_h, k_h) indexes the table at (i_h - j_h + len - 1) and the dot
 * runs over the whole head_dim of the head's own q (deepencoder computes r_q of shape
 * (B*heads, q_h, q_w, head_dim) and einsums "bhwc,hkc->bhwk"). So the per-pair term folds into
 * neither q, k nor v: no plain attention op can carry it, which is why this op exists.
 *
 * ============================== Windows ==============================
 *
 * Each cu segment is grid_h * grid_w rows in row-major grid order. With `window` > 0 the segment is
 * padded to multiples of `window` with ZERO rows and cut into window x window tiles, exactly as
 * window_partition does after norm1: the pad rows are keys AND values of their tile, so their
 * logits (which carry the rel terms even though q.k is 0) take part in the softmax, and their v is
 * zero. `window` 0 is one tile over the whole segment (the reference's global blocks).
 *
 * ============================== Table resizing ==============================
 *
 * get_rel_pos linearly interpolates the table along dim 0 when its length is not
 * 2*max(q,k) - 1 (F.interpolate, mode linear, align_corners False). Every geometry the tower
 * declares matches exactly, but a non-square global grid has axes of different extents, so the
 * resampling is implemented here rather than assumed away.
 *
 * ============================== Numerics ==============================
 *
 * f32 throughout: q, k, v and the tables are widened once, every dot and softmax accumulates in
 * f32 with the max subtracted, and the output is rounded once to the operand's dtype. Summation
 * order is fixed (increasing channel, then key), so two runs agree bit for bit at any thread count.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace ref;

namespace {

/* A rel_pos table as f32 rows of `hd`, resampled to `want` rows when its own length differs.
 * F.interpolate(..., size=want, mode="linear") with align_corners False: output d samples the input
 * at (d + 0.5) * rows / want - 0.5, clamped to the ends, linearly between the neighbours. */
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

/* See the file header for the semantics. `cu_seqlens` absent is one segment of every row: the
 * reference's single-view form. */
int ref_sam_attn(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
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

    /* Segments: cu[s]..cu[s+1], each exactly one grid. */
    int64_t nseq = 1;
    if (cu->data) {
        if (cu->rank == 0 || numel(cu) < 2) return RAD_E_SHAPE;
        nseq = numel(cu) - 1;
        if (cu_at(cu, 0) != 0 || cu_at(cu, nseq) != M) return RAD_E_SHAPE;
        for (int64_t s = 0; s < nseq; ++s)
            if (cu_at(cu, s + 1) - cu_at(cu, s) != R) return RAD_E_SHAPE;
    } else if (M != R) {
        return RAD_E_SHAPE;
    }

    const float scale = p_f32(a, "scale", (float)(1.0 / std::sqrt((double)hd)));

    /* The tile geometry: window x window, or the whole grid. Table lengths follow the tile axes. */
    const int64_t th = win > 0 ? win : gh, tw = win > 0 ? win : gw;
    const int64_t keys = th * tw;
    if (numel(th_tab) < hd || numel(tw_tab) < hd) return RAD_E_SHAPE;
    std::vector<float> Rh = table_rows(th_tab, hd, 2 * th - 1);
    std::vector<float> Rw = table_rows(tw_tab, hd, 2 * tw - 1);

    /* Every (channel-ordered) dot is written out in one place so the order is explicit: q[c] times
     * the row's element c, increasing c. */
    auto dot = [hd](const float* x, const float* y) {
        float s = 0.0f;
        for (int64_t c = 0; c < hd; ++c) s += x[c] * y[c];
        return s;
    };

    /* One thread owns one output row; the work unit is about keys * head_dim per head. */
    REF_PAR_FOR(nseq * R * nh * keys * hd)
    for (int64_t work = 0; work < nseq * R; ++work) {
        const int64_t s  = work / R, r = work % R;
        const int64_t ih = r / gw, iw = r % gw;
        const int64_t ty = win > 0 ? ih / win : 0, tx = win > 0 ? iw / win : 0;
        const int64_t lh = ih - ty * th, lw = iw - tx * tw;
        const int64_t base = cu->data ? cu_at(cu, s) : 0;
        const int64_t out_row = base + r;

        /* logits[keys], qt[hd], acc[hd]. */
        float* buf = ws_get(keys + 2 * hd);
        float* logits = buf;
        float* qt = buf + keys;
        float* acc = qt + hd;

        for (int64_t h = 0; h < nh; ++h) {
            const int64_t qbase = rowoff(q, out_row, W) + h * hd;
            const int64_t qst = laststride(q);
            for (int64_t c = 0; c < hd; ++c) qt[c] = ldt(q, qbase + c * qst);

            for (int64_t jh = 0; jh < th; ++jh) {
                const float hr = dot(qt, &Rh[(size_t)((lh - jh + th - 1) * hd)]);
                for (int64_t jw = 0; jw < tw; ++jw) {
                    float lgt = hr + dot(qt, &Rw[(size_t)((lw - jw + tw - 1) * hd)]);
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

            /* Softmax in f32 with the max subtracted, then the weighted sum of v. */
            float m = -INFINITY;
            for (int64_t j = 0; j < keys; ++j) m = logits[j] > m ? logits[j] : m;
            float sum = 0.0f;
            for (int64_t j = 0; j < keys; ++j) {
                logits[j] = ref_exp(logits[j] - m);
                sum += logits[j];
            }
            for (int64_t c = 0; c < hd; ++c) acc[c] = 0.0f;
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int64_t jh = 0; jh < th; ++jh) {
                const int64_t kh = ty * th + jh;
                if (kh >= gh) continue;                      /* pad rows carry v = 0 */
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
