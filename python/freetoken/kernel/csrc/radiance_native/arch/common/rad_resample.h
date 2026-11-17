/* rad_resample.h -- fp32 bicubic resampling reproducing torch's
 * F.interpolate(x, size=(n, m), mode="bicubic", antialias=True) on the CPU float32 kernel,
 * operation for operation, for the Unlimited-OCR DeepEncoder's position tables.
 *
 * The reference resamples two fixed tables (deepencoder.py):
 * get_abs_pos_sam() permutes SAM's pos_embed [1,64,64,768] to (0,3,1,2), casts fp32 and
 * interpolates to 40x40; get_abs_pos() takes CLIP position_embedding rows 1: as a 16x16 grid,
 * interpolates to 10x10, and cats row 0 (cls) in front. Here the table is row-major [h, w, C]
 * with C innermost and each of the C channels is interpolated over the (h, w) grid
 * independently; the cls split/catenation is the caller's job.
 *
 * The float32 antialias path (upsample_bicubic2d_aa_kernel_impl ->
 * upsample_separable_Nd_kernel_impl -> _compute_index_ranges_weights with antialias=true,
 * aten/src/ATen/native/cpu/UpSampleKernel.cpp, torch v2.14.0) is mirrored operation for
 * operation, scalar_t = float:
 *   per axis scale = (float)(in / out); support = scale >= 1 ? (interp_size/2) * scale =
 *     2 * scale : 2; invscale = scale >= 1 ? 1 / scale : 1;
 *   the output-pixel centre center = scale * (i + 0.5f);
 *   the taps run over xmin = max((int64)(center - support + 0.5), 0) ..
 *     min((int64)(center + support + 0.5), in) - 1 -- truncating casts, so a negative
 *     start truncates toward zero before the clamp to 0; border taps are simply absent,
 *     the row normalization below absorbs the lost mass;
 *   each tap weight is aa_filter((j + xmin - center + 0.5) * invscale), the Keys cubic
 *     with a = -0.5 (torch's antialias filter is the Pillow-compatible -0.5 kernel; the
 *     -0.75 OpenCV-compatible one belongs to the non-antialias path);
 *   the axis weights are normalized by their float running sum total_w, exactly as
 *     _compute_indices_min_size_weights_aa divides wt_ptr[j] /= total_w;
 *   the resize is separable: the width pass over every input row first, then the height
 *     pass (the order upsample_separable_Nd_kernel_impl walks the dims), each tap
 *     accumulating acc = (float)(acc + w * v) in float.
 * Golden-checked by tests/resample_golden.py (geometries 64->40, 24->20, 8->5, 16->10,
 * 16x20->10x13, and both checkpoint tables).
 */
#pragma once
#ifndef RAD_RESAMPLE_H
#define RAD_RESAMPLE_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

namespace resample_detail {

/* aa_filter, the Keys cubic with a = -0.5 that torch's antialias path uses (the -0.75
 * OpenCV-compatible kernel belongs to the non-antialias path), in float. */
inline float cubic_aa(float x) {
    const float ax = std::fabs(x);
    if (ax < 1.0f) return ((1.5f * ax - 2.5f) * ax) * ax + 1.0f;
    if (ax < 2.0f) return (((-0.5f * ax + 2.5f) * ax - 4.0f) * ax) + 2.0f;
    return 0.0f;
}

struct Axis {
    int n = 0;                     /* tap slots per output pixel (max_interp_size) */
    std::vector<int32_t> size;     /* [out] taps this output pixel reads */
    std::vector<int32_t> idx;      /* [out*n] input index per tap (xmin + j) */
    std::vector<float> w;          /* [out*n] weights, normalized by the tap sum */
};

/* One axis of _compute_indices_min_size_weights_aa, float32 expression for expression. */
inline Axis make_axis(int in, int out) {
    Axis a;
    const float scale = (float)((double)in / (double)out);
    const float support = scale >= 1.0f ? (float)(2.0 * (double)scale) : 2.0f;
    const float invscale = scale >= 1.0f ? (float)(1.0 / (double)scale) : 1.0f;
    a.n = (int)std::ceil((double)support) * 2 + 1;  /* max_interp_size */
    a.size.resize((size_t)out, 0);
    a.idx.assign((size_t)out * (size_t)a.n, 0);
    a.w.assign((size_t)out * (size_t)a.n, 0.0f);
    for (int i = 0; i < out; ++i) {
        const float c = ((float)i + 0.5f) * scale;
        int xmin = (int)(c - support + 0.5f);
        if (xmin < 0) xmin = 0;
        int xsize = (int)(c + support + 0.5f);
        if (xsize > in) xsize = in;
        xsize -= xmin;
        if (xsize < 0) xsize = 0;
        /* ATen's own `xsize = min(xsize, n)`, kept for parity. It cannot bind here: with
         * xmin > 0 the window spans 2*support + 1 taps, at most a.n = 2*ceil(support) + 1
         * (the bound ATen sized the slots from), and with xmin == 0 it spans fewer. So the
         * clamp is a no-op safety net, never a silent tap truncation. */
        if (xsize > a.n) xsize = a.n;
        int* idx = &a.idx[(size_t)i * (size_t)a.n];
        float* w = &a.w[(size_t)i * (size_t)a.n];
        float total = 0.0f;
        for (int j = 0; j < xsize; ++j) {
            w[j] = cubic_aa(((float)(xmin + j) - c + 0.5f) * invscale);
            idx[j] = xmin + j;
            total = (float)(total + w[j]);
        }
        a.size[i] = xsize;
        if (total != 0.0f)
            for (int j = 0; j < xsize; ++j) w[j] = (float)(w[j] / total);
    }
    return a;
}

/* One 1D resize pass. `rows` scans travel `src_row_stride` floats apart, taps step
 * `tap_stride` floats along the resized axis, and outputs land at
 * dst + r * dst_row_stride + i * out_stride + channel; tap order and float accumulation
 * follow ATen's separable 1D resize. */
inline void resize_axis(const float* src, int64_t src_row_stride, int64_t tap_stride,
                        float* dst, int64_t dst_row_stride, int64_t out_stride, int rows,
                        const Axis& ax, int out_n, int channels) {
    for (int r = 0; r < rows; ++r) {
        const float* srow = src + (size_t)r * (size_t)src_row_stride;
        float* drow = dst + (size_t)r * (size_t)dst_row_stride;
        for (int i = 0; i < out_n; ++i) {
            const int n = ax.size[i];
            const int32_t* idx = &ax.idx[(size_t)i * (size_t)ax.n];
            const float* w = &ax.w[(size_t)i * (size_t)ax.n];
            for (int c = 0; c < channels; ++c) {
                if (n == 0) { drow[(size_t)i * (size_t)out_stride + (size_t)c] = 0.0f; continue; }
                float acc = (float)(srow[(size_t)idx[0] * (size_t)tap_stride + (size_t)c] * w[0]);
                for (int j = 1; j < n; ++j)
                    acc = (float)(acc +
                                  srow[(size_t)idx[j] * (size_t)tap_stride + (size_t)c] * w[j]);
                drow[(size_t)i * (size_t)out_stride + (size_t)c] = acc;
            }
        }
    }
}

}  // namespace resample_detail

/* Resample an [in_h, in_w, channels] fp32 table (channels innermost) to
 * [out_h, out_w, channels] with torch's bicubic + antialias interpolation: the width pass
 * over every input row, then the height pass. Runs once at load. */
inline void resample_bicubic_aa(const float* src, int in_h, int in_w, int out_h, int out_w,
                                int channels, float* dst) {
    if (!src || !dst || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0 || channels <= 0)
        return;
    if (in_h == out_h && in_w == out_w) {
        std::memcpy(dst, src, (size_t)in_h * (size_t)in_w * (size_t)channels * sizeof(float));
        return;
    }
    const resample_detail::Axis ay = resample_detail::make_axis(in_h, out_h);
    const resample_detail::Axis ax = resample_detail::make_axis(in_w, out_w);
    const float* work = src;
    int work_w = in_w;
    std::vector<float> wide;
    if (in_w != out_w) {  /* width pass first, the order upsample_separable_Nd walks the dims */
        wide.assign((size_t)in_h * (size_t)out_w * (size_t)channels, 0.0f);
        resample_detail::resize_axis(src, (int64_t)in_w * channels, channels, wide.data(),
                                     (int64_t)out_w * channels, channels, in_h, ax, out_w,
                                     channels);
        work = wide.data();
        work_w = out_w;
    }
    if (in_h == out_h) {
        if (work != dst)
            std::memcpy(dst, work, (size_t)out_h * (size_t)out_w * (size_t)channels * sizeof(float));
        return;
    }
    /* height pass over the widened rows: one scan per output-width column */
    resample_detail::resize_axis(work, channels, (int64_t)work_w * channels, dst, channels,
                                 (int64_t)out_w * channels, work_w, ay, out_h, channels);
}

}  // namespace arch
}  // namespace rad

#endif  // RAD_RESAMPLE_H
