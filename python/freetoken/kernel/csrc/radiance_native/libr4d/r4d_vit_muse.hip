// r4d_vit_muse.hip -- Muse Glimmer's vision-tower coordinate ops: the zero-padded position-table
// resample (grid_embed_pad) and the window-attention permutation (window_order).
//
// Arithmetic matches libref (ref_elementwise.cpp): grid_embed_pad is grid_embed with the
// half-pixel-centre resample (align_corners=False) and grid_sample's padding_mode="zeros" -- a tap
// whose UNCLAMPED index falls outside [0, side) contributes nothing rather than reading an edge --
// the four taps summed in f32 in libref's order, rounded to bf16, added and rounded again.
// window_order derives the whole permutation on the device from the coordinate planes: one
// workgroup walks the segments' windows, writes the cumulative boundaries, then scatters the
// gather index, its inverse and the swapped one-offset rope positions.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"   // bf2f / f32_to_bf16

#include <cstdint>

namespace {

constexpr int kThr = 256;

__device__ __forceinline__ int imin2(int a, int b) { return a < b ? a : b; }

// ldw: a table channel as f32, the weight stored bf16 or f32 (as grid_embed's kernel reads it).
template <bool F32>
__device__ __forceinline__ float ldw(const void* p, uint32_t i) {
    return F32 ? ((const float*)p)[i] : bf2f(((const unsigned short*)p)[i]);
}

// ---------------------------------------------------------------------------- grid_embed_pad
// ref_grid_embed_pad: grid_embed's operands, half-pixel-centre taps, off-table taps weighted zero.
// `crd` is [4, planes]: row, column, segment height, segment width.
template <bool TF32>
__global__ __launch_bounds__(kThr) void grid_embed_pad_kernel(
    unsigned short* __restrict__ x, const void* __restrict__ tab, const int32_t* __restrict__ crd,
    uint32_t n, int side, uint32_t x_ld, uint32_t plane) {
    const uint32_t m = blockIdx.x;
    const int g[4] = { crd[m], crd[plane + m], crd[2 * plane + m], crd[3 * plane + m] };
    int th[2], tw[2];
    float wh[2], ww[2];
    auto axis = [&](int idx, int size, int* t, float* wt) {
        if (size < 1) size = 1;
        const float src = (float)side / (float)size * ((float)idx + 0.5f) - 0.5f;
        const float fl = floorf(src);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int raw = (int)fl + k;
            t[k] = raw < 0 ? 0 : (raw > side - 1 ? side - 1 : raw);
            const float d = fabsf(src - fl - (float)k);
            const float u = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
            wt[k] = raw >= 0 && raw <= side - 1 ? u : 0.0f;
        }
    };
    axis(g[0], g[2], th, wh);
    axis(g[1], g[3], tw, ww);
    unsigned short* __restrict__ xr = x + (size_t)m * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float acc = 0.0f;
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
#pragma unroll
            for (int kj = 0; kj < 2; ++kj)
                acc += ldw<TF32>(tab, (uint32_t)(th[ki] * side + tw[kj]) * n + i) *
                       (wh[ki] * ww[kj]);
        const float pe = bf2f(f32_to_bf16(acc));
        xr[i] = f32_to_bf16(bf2f(xr[i]) + pe);
    }
}

// ---------------------------------------------------------------------------- window_order
// ref_window_order: HF's get_vision_window_index as a gather-index computation, entirely from the
// coordinate planes. One workgroup for the pass, three phases over global memory (P is a tower
// pass's patches, at most a few thousand):
//   1. every thread lays down the identity over [0, P);
//   2. thread 0 walks the segments' windows in permuted order, writing each window's base slot as
//      cuw's boundary -- the prefix the caller's windowed attention and the scatter both read;
//   3. the threads re-derive each patch's window from the boundaries and scatter idx, inv and pos.
// `window` <= 0 is the identity permutation with cuw = cu.
__global__ __launch_bounds__(kThr) void window_order_kernel(
    const int32_t* __restrict__ crd, const int32_t* __restrict__ cu, int32_t* __restrict__ idx,
    int32_t* __restrict__ inv, int32_t* __restrict__ pos, int32_t* __restrict__ cuw,
    uint32_t P, uint32_t M, uint32_t plane, uint32_t pos_plane, uint32_t cu_len,
    uint32_t cuw_len, int window) {
    // 1. the identity everywhere, so slots the pass does not cover are defined; the covered slots
    //    already get their rope positions (col + 1, row + 1), identity or not.
    for (uint32_t p = threadIdx.x; p < P; p += kThr) {
        idx[p] = (int32_t)p;
        inv[p] = (int32_t)p;
        if (p < M) {
            pos[p] = crd[plane + p] + 1;
            pos[pos_plane + p] = crd[p] + 1;
        } else {
            pos[p] = 1;
            pos[pos_plane + p] = 1;
        }
    }
    if (window <= 0) {
        if (threadIdx.x == 0) {
            /* The caller narrows cuw to the pass's window count + 1, so every write below is
             * bounded by its length rather than by P. */
            const uint32_t k = cu_len < cuw_len ? cu_len : cuw_len;
            for (uint32_t i = 0; i < k; ++i) cuw[i] = cu[i];
            const int32_t last = k ? cuw[k - 1] : 0;
            for (uint32_t i = k; i < cuw_len; ++i) cuw[i] = last;
        }
        return;
    }
    __syncthreads();
    // 2. the windows of each segment in permuted order; each boundary is a window's base slot.
    if (threadIdx.x == 0) {
        int64_t slot = 0;
        uint32_t nwin = 0;
        for (uint32_t s = 0; s + 1 < cu_len && cu[s] < (int32_t)M; ++s) {
            const int start = cu[s] < 0 ? 0 : cu[s];
            int end = cu[s + 1];
            if (end > (int32_t)M) end = (int32_t)M;
            const int gh = start < end ? crd[2 * plane + start] : 1;
            const int gw = start < end ? crd[3 * plane + start] : 1;
            const int nh = (gh + window - 1) / window, nw = (gw + window - 1) / window;
            for (int wr = 0; wr < nh; ++wr)
                for (int wc = 0; wc < nw; ++wc) {
                    const int rows_in = imin2(window, gh - wr * window);
                    const int cols_in = imin2(window, gw - wc * window);
                    if (nwin < cuw_len) cuw[nwin] = (int32_t)slot;
                    ++nwin;
                    slot += (int64_t)rows_in * cols_in;
                }
        }
        if (nwin < cuw_len) cuw[nwin] = (int32_t)slot;
        for (uint32_t i = nwin + 1; i < cuw_len; ++i) cuw[i] = (int32_t)slot;
    }
    __syncthreads();
    // 3. each patch's slot: its segment, window id and rank in the window, off the boundaries.
    for (uint32_t p = threadIdx.x; p < M; p += kThr) {
        const int r = crd[p], c = crd[plane + p], gw = crd[3 * plane + p];
        int wr = r / window, wc = c / window;
        if (wr < 0) wr = 0;
        if (wc < 0) wc = 0;
        const int cols_in = imin2(window, gw - wc * window);
        // The patch's segment: the last boundary at or before p; the windows before it, replayed.
        uint32_t s = 0;
        int64_t wbefore = 0;
        for (; s + 1 < cu_len && cu[s + 1] <= (int32_t)p; ++s) {
            const int tstart = cu[s] < 0 ? 0 : cu[s];
            int tend = cu[s + 1];
            if (tend > (int32_t)M) tend = (int32_t)M;
            const int th = tstart < tend ? crd[2 * plane + tstart] : 1;
            const int tw_ = tstart < tend ? crd[3 * plane + tstart] : 1;
            wbefore += (int64_t)((th + window - 1) / window) * ((tw_ + window - 1) / window);
        }
        const int start = cu[s] < 0 ? 0 : cu[s];
        const int sgw = start < (int)M ? crd[3 * plane + start] : 1;
        const int nw = (sgw + window - 1) / window;
        const int64_t wid = wbefore + (int64_t)wr * nw + wc;
        const int64_t at = cuw[wid] + (int64_t)(r - wr * window) * cols_in + (c - wc * window);
        idx[at] = (int32_t)p;
        inv[p] = (int32_t)at;
        pos[at] = c + 1;
        pos[pos_plane + at] = r + 1;
    }
}

}  // namespace

/* ------------------------------------------------------------------ grid_embed_pad */
enum { GEP_X = 0, GEP_TAB, GEP_CRD };

extern "C" int r4d_rad_grid_embed_pad(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = r4d_opt(a, GEP_X);
    const RadTensor* tab = r4d_opt(a, GEP_TAB);
    const RadTensor* crd = r4d_opt(a, GEP_CRD);
    if (!x || !tab || !crd) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || crd->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (tab->dtype != RAD_BF16 && tab->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tab)) return R4D_NO(RAD_E_STRIDE);
    const long long n = r4d_geti_or(a, "n", r4d_dim(tab, (int)tab->rank - 1));
    const long long side = r4d_geti_or(a, "side", 0);
    if (n <= 0 || side <= 0 || r4d_numel(tab) < side * side * n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M || crd->stride[1] != 1)
        return R4D_NO(RAD_E_SHAPE);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    /* crd is a [4, pass] view of the full-pass coord buffer: the plane pitch is its leading
     * stride when that exceeds the shape. */
    const uint32_t plane = (uint32_t)(crd->stride[0] >= crd->shape[1] ? crd->stride[0]
                                                                     : crd->shape[1]);
    if (tab->dtype == RAD_F32)
        return R4D_EMIT(stream, grid_embed_pad_kernel<true>, dim3((uint32_t)M), dim3(kThr), 0,
                        (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                        (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
    return R4D_EMIT(stream, grid_embed_pad_kernel<false>, dim3((uint32_t)M), dim3(kThr), 0,
                    (unsigned short*)r4d_wp(x), r4d_p(tab), (const int32_t*)r4d_p(crd),
                    (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
}

/* ------------------------------------------------------------------ window_order */
enum { WO_CRD = 0, WO_CU, WO_IDX, WO_INV, WO_POS, WO_CUW };

extern "C" int r4d_rad_window_order(const RadArgs* a, RadStream stream) {
    const RadTensor* crd = r4d_opt(a, WO_CRD);
    const RadTensor* cu  = r4d_opt(a, WO_CU);
    const RadTensor* idx = r4d_opt(a, WO_IDX);
    const RadTensor* inv = r4d_opt(a, WO_INV);
    const RadTensor* pos = r4d_opt(a, WO_POS);
    const RadTensor* cuw = r4d_opt(a, WO_CUW);
    if (!crd || !cu || !idx || !inv || !pos || !cuw) return R4D_NO(RAD_E_INVAL);
    if (crd->dtype != RAD_I32 || cu->dtype != RAD_I32 || idx->dtype != RAD_I32 ||
        inv->dtype != RAD_I32 || pos->dtype != RAD_I32 || cuw->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(cu) || !r4d_contig(idx) || !r4d_contig(inv) || !r4d_contig(cuw))
        return R4D_NO(RAD_E_STRIDE);
    /* `M` is a ranged parameter: the params carry the band's bound (16384), while this issue's
     * patches are the coord view's columns. cuw is narrowed to the pass's windows + 1. */
    const long long M = crd ? crd->shape[crd->rank - 1] : 0;
    const long long window = r4d_geti_or(a, "window", 0);
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    /* crd and pos are plane views of full-pass buffers ([4, P] and [2, P] declared) read at this
     * pass's M rows; their plane pitch is the leading stride when it exceeds the shape. */
    if (crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M || crd->stride[1] != 1)
        return R4D_NO(RAD_E_SHAPE);
    /* `pos` is the two rope planes either as [2, M] (or a pitched [2, P] view) or as one
     * [2 * M] row: both describe the same bytes, so the pitch is the row length in the first form
     * and M in the second. */
    long long pplane = 0;
    if (pos->rank == 1) {
        if (pos->stride[0] != 1 || r4d_numel(pos) < 2 * M) return R4D_NO(RAD_E_SHAPE);
        pplane = M;
    } else if (pos->rank == 2) {
        if (pos->shape[0] < 2 || pos->shape[1] < M || pos->stride[1] != 1)
            return R4D_NO(RAD_E_SHAPE);
        pplane = pos->stride[0] >= pos->shape[1] ? pos->stride[0] : pos->shape[1];
    } else {
        return R4D_NO(RAD_E_SHAPE);
    }
    /* P is THIS pass's patches (M), not the coord buffer's declared width: the caller narrows
     * idx/inv/pos/cuw to the pass, so a slot lies in [0, M). */
    const long long P = M;
    const long long cplane = crd->stride[0] >= crd->shape[1] ? crd->stride[0] : crd->shape[1];
    if (r4d_numel(cu) < 2 || r4d_numel(idx) < M || r4d_numel(inv) < M || r4d_numel(pos) < 2 * M ||
        r4d_numel(cuw) < 2)
        return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, window_order_kernel, dim3(1), dim3(kThr), 0,
                    (const int32_t*)r4d_p(crd), (const int32_t*)r4d_p(cu),
                    (int32_t*)r4d_wp(idx), (int32_t*)r4d_wp(inv), (int32_t*)r4d_wp(pos),
                    (int32_t*)r4d_wp(cuw), (uint32_t)P, (uint32_t)M, (uint32_t)cplane,
                    (uint32_t)pplane, (uint32_t)r4d_numel(cu), (uint32_t)r4d_numel(cuw),
                    (int)window);
}
