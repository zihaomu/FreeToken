// r4d_sam_attn_bf16 -- Unlimited-OCR's DeepEncoder (SAM ViT-B/16) attention with decomposed relative
// position (`sam_attn`; arch/common/rad_block_sam.h, libref/ref_sam_attn.cpp is the oracle).
//
// The logits of a query row i against a key j of the same window tile are
//
//     scale * q.k  +  q.Rh[i_h, j_h]  +  q.Rw[i_w, j_w]
//
// over the head's whole head_dim, the tables indexed by the offset inside the tile. With `window`
// > 0 the grid is padded with ZERO rows to a multiple of the window and cut into window x window
// tiles (deepencoder's window_partition, after norm1): the pad rows are keys and values of their
// tile and take part in the softmax, their v being zero. `window` 0 is one tile over the segment.
//
// Shape. One block per (segment, tile, head), so the tile's keys are shared by the block's query
// rows and the k/v rows a wave reads at each step are one address (an L1/L2 hit, broadcast across
// the wave). Each thread owns one query row: q and the output accumulator are HD floats in
// registers, the softmax is the online form (running max and sum), and the rel_pos dot of a tile
// row is computed once per (query row, tile row) rather than once per key.
//
// Head widths. HD is a template parameter and only 64 is instantiated (SAM ViT-B's 12 x 64); the
// shim refuses any other width rather than reading past the registers. A table whose row count is
// not 2*tile-1 is resampled on read (get_rel_pos's F.interpolate(size=want, mode="linear",
// align_corners=False) in f32): the 127-row global tables give 79 rows at a 40 grid, and every
// other geometry the tower declares already has exactly the rows it wants.
//
// Exactness. f32 throughout, one bf16 rounding of the output, exp2 (the hardware transcendental)
// with the log2(e) factor; the selftest holds it to the house tolerance against a host port of the
// reference, not to bits.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

#include <cmath>

namespace {

constexpr uint32_t kThreads = 128;
constexpr float    kLog2e   = 1.4426950408889634f;

/* Row loads stay scalar uint16: the operand contract allows any row pitch (the tower's q/k/v are
 * column slices of one fused projection, and a tower may pad rows), so a 16-byte bf16x8 read
 * would be misaligned for a pitch that is not a multiple of 8 elements. The register diet is the
 * w-dot hoist below, not the load width. */

/* q . row(d) of a rel_pos table as the reference resized it to `want` rows: linear interpolation
 * between the two source rows at align_corners=False's coordinate, clamped at the edges. With
 * rows == want the source is row d itself (f == 0), so the direct read is unchanged bit for bit. */
template <int HD>
__device__ inline float tab_dot(const uint16_t* __restrict__ tab, int rows, int want, int d,
                                const float* qf) {
    if (rows == want) {
        const uint16_t* rp = tab + (size_t)d * HD;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HD; ++c) s += qf[c] * bf16_to_f32(rp[c]);
        return s;
    }
    double src = ((double)d + 0.5) * (double)rows / (double)want - 0.5;
    if (src < 0.0) src = 0.0;
    if (src > (double)(rows - 1)) src = (double)(rows - 1);
    const int i0 = (int)src, i1 = i0 + 1 < rows ? i0 + 1 : i0;
    const float f = (float)(src - (double)i0);
    const uint16_t* a = tab + (size_t)i0 * HD;
    const uint16_t* b = tab + (size_t)i1 * HD;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HD; ++c) {
        const float av = bf16_to_f32(a[c]);
        s += qf[c] * (av + (bf16_to_f32(b[c]) - av) * f);
    }
    return s;
}

/* Eight bf16 of a q/k/v/out row as one 16-byte uint4. VEC is only instantiated when the caller
 * proved every involved pointer is 16-byte aligned and every row pitch a multiple of 8 elements
 * (launch_sam_attn); otherwise the scalar uint16 path below runs, because the operand contract
 * allows any pitch (the tower's q/k/v are column slices and may carry a filler column). The
 * element order is the scalar order, so both paths produce bit-identical dots. */
__device__ inline void bf16x8_to_f32(uint4 u, float* f) {
    const unsigned* w = (const unsigned*)&u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i]     = bf16_to_f32((uint16_t)(w[i] & 0xffffu));
        f[2 * i + 1] = bf16_to_f32((uint16_t)(w[i] >> 16));
    }
}

template <bool VEC, int HD>
__device__ inline void ld_row(const uint16_t* p, float* f) {
    if (VEC) {
        const uint4* u = (const uint4*)p;
#pragma unroll
        for (int c8 = 0; c8 < HD / 8; ++c8) bf16x8_to_f32(u[c8], f + c8 * 8);
    } else {
#pragma unroll
        for (int c = 0; c < HD; ++c) f[c] = bf16_to_f32(p[c]);
    }
}

template <bool VEC, int HD>
__device__ inline void st_row(uint16_t* p, const float* f) {
    if (VEC) {
        uint4 u[HD / 8];
        uint16_t* b = (uint16_t*)u;
#pragma unroll
        for (int c = 0; c < HD; ++c) b[c] = f32_to_bf16(f[c]);
        uint4* o = (uint4*)p;
#pragma unroll
        for (int c8 = 0; c8 < HD / 8; ++c8) o[c8] = u[c8];
    } else {
#pragma unroll
        for (int c = 0; c < HD; ++c) p[c] = f32_to_bf16(f[c]);
    }
}

template <bool VEC, int HD>
__device__ inline float dot_row(const uint16_t* p, const float* qf) {
    float s = 0.0f;
    if (VEC) {
        const uint4* u = (const uint4*)p;
#pragma unroll
        for (int c8 = 0; c8 < HD / 8; ++c8) {
            const unsigned* w = (const unsigned*)&u[c8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s += qf[c8 * 8 + 2 * i] * bf16_to_f32((uint16_t)(w[i] & 0xffffu));
                s += qf[c8 * 8 + 2 * i + 1] * bf16_to_f32((uint16_t)(w[i] >> 16));
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < HD; ++c) s += qf[c] * bf16_to_f32(p[c]);
    }
    return s;
}

template <bool VEC, int HD>
__device__ inline void axpy_row(float* acc, const uint16_t* p, float pv) {
    if (VEC) {
        const uint4* u = (const uint4*)p;
#pragma unroll
        for (int c8 = 0; c8 < HD / 8; ++c8) {
            float f8[8];
            bf16x8_to_f32(u[c8], f8);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[c8 * 8 + i] += pv * f8[i];
        }
    } else {
#pragma unroll
        for (int c = 0; c < HD; ++c) acc[c] += pv * bf16_to_f32(p[c]);
    }
}

template <int HD, bool VEC>
__device__ inline void sam_attn_body(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const int* __restrict__ cu, const uint16_t* __restrict__ rh,
    const uint16_t* __restrict__ rw, int rh_rows, int rw_rows,
    int q_ld, int k_ld, int v_ld, int o_ld, int nhead, int gh, int gw, int th, int tw,
    int seq, int head, int ty, int tx, int row_lo, int row_hi, float scale) {
    // The caller supplies the (segment, tile, head) block coordinates: the windowed kernel reads
    // them from blockIdx.x's window-tile layout, the global one from its query-tile layout.
    const int s0 = cu[seq];
    if (cu[seq + 1] - s0 != gh * gw) {
        // A segment that is not one grid is a tower bug, not a shape: the shim pre-checks the
        // row count against the segment count, and this is the device-side guard. Write NaN over
        // the segment's rows (the vit kernel's poison convention) so the corruption reaches the
        // consumer -- a silent return would pass a garbage plane on as success.
        const int n = cu[seq + 1] - s0;
        for (int r = (int)threadIdx.x; r < n; r += (int)kThreads) {
            uint16_t* op = go + (size_t)(s0 + r) * (size_t)o_ld + (size_t)head * HD;
            for (int c = 0; c < HD; ++c) op[c] = (uint16_t)0x7fc0u;   // bf16 NaN, as vit poisons
        }
        return;
    }

    for (int r = row_lo + (int)threadIdx.x; r < row_hi; r += (int)kThreads) {
        const int lh = r / tw, lw = r % tw;
        const int ih = ty * th + lh, iw = tx * tw + lw;
        if (ih >= gh || iw >= gw) continue;    // a pad row: no output

        const size_t qrow = (size_t)(s0 + ih * gw + iw);
        const uint16_t* qp = gq + qrow * (size_t)q_ld + (size_t)head * HD;
        float qf[HD];
        ld_row<VEC, HD>(qp, qf);

        float acc[HD];
#pragma unroll
        for (int c = 0; c < HD; ++c) acc[c] = 0.0f;
        float m = -INFINITY, l = 0.0f;

        /* The w rel-pos dots depend on (lw, jw) only: computed once for the query row when the
         * tile is narrow (the windowed tiles are 14 wide), recomputed per key otherwise. For a
         * whole-grid tile that is gw dots a query row instead of gw per key row, i.e. the same
         * count as the q.k dots rather than gh times it; the array is spilled to local memory
         * when the compiler runs out of registers, which is still far cheaper than the dot. */
        float wr[16];
        float wr_all[64];
        const bool wr_hoist = tw <= 16;
        const bool wr_wide  = !wr_hoist && tw <= 64;
        if (wr_hoist)
            for (int jw = 0; jw < tw; ++jw)
                wr[jw] = tab_dot<HD>(rw, rw_rows, 2 * tw - 1, lw - jw + tw - 1, qf);
        else if (wr_wide)
            for (int jw = 0; jw < tw; ++jw)
                wr_all[jw] = tab_dot<HD>(rw, rw_rows, 2 * tw - 1, lw - jw + tw - 1, qf);

        for (int jh = 0; jh < th; ++jh) {
            const float hr = tab_dot<HD>(rh, rh_rows, 2 * th - 1, lh - jh + th - 1, qf);

            const int kh = ty * th + jh;
            const bool hreal = kh < gh;
            for (int jw = 0; jw < tw; ++jw) {
                float s = hr + (wr_hoist ? wr[jw]
                                         : (wr_wide ? wr_all[jw]
                                                    : tab_dot<HD>(rw, rw_rows, 2 * tw - 1,
                                                                  lw - jw + tw - 1, qf)));

                const int kw = tx * tw + jw;
                const bool real = hreal && kw < gw;
                const uint16_t* vp = nullptr;
                if (real) {
                    const size_t krow = (size_t)(s0 + kh * gw + kw);
                    const uint16_t* kp = gk + krow * (size_t)k_ld + (size_t)head * HD;
                    const float d = dot_row<VEC, HD>(kp, qf);
                    s += scale * d;
                    vp = gv + krow * (size_t)v_ld + (size_t)head * HD;
                }

                // Online softmax: a new maximum rescales what has been accumulated.
                if (s > m) {
                    const float corr = ex2((m - s) * kLog2e);
#pragma unroll
                    for (int c = 0; c < HD; ++c) acc[c] *= corr;
                    l *= corr;
                    m = s;
                }
                const float p = ex2((s - m) * kLog2e);
                l += p;
                if (real) axpy_row<VEC, HD>(acc, vp, p);
            }
        }

        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        uint16_t* op = go + qrow * (size_t)o_ld + (size_t)head * HD;
#pragma unroll
        for (int c = 0; c < HD; ++c) acc[c] *= inv;
        st_row<VEC, HD>(op, acc);
    }
}

template <int HD>
__global__ __launch_bounds__(kThreads) void sam_attn_kernel(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const int* __restrict__ cu, const uint16_t* __restrict__ rh,
    const uint16_t* __restrict__ rw, int rh_rows, int rw_rows,
    int q_ld, int k_ld, int v_ld, int o_ld, int nhead, int gh, int gw, int th, int tw,
    int nth, int ntw, int vec_rows, float scale) {
    // blockIdx.x = ((seq * nth + ty) * ntw + tx) * nhead + head: heads adjacent, so the tile's k/v
    // rows stay hot across them.
    int b = blockIdx.x;
    const int head = b % nhead; b /= nhead;
    const int tx   = b % ntw;   b /= ntw;
    const int ty   = b % nth;   b /= nth;
    const int seq  = b;
    if (vec_rows)
        sam_attn_body<HD, true>(gq, gk, gv, go, cu, rh, rw, rh_rows, rw_rows, q_ld, k_ld, v_ld,
                                o_ld, nhead, gh, gw, th, tw, seq, head, ty, tx, 0, th * tw, scale);
    else
        sam_attn_body<HD, false>(gq, gk, gv, go, cu, rh, rw, rh_rows, rw_rows, q_ld, k_ld, v_ld,
                                 o_ld, nhead, gh, gw, th, tw, seq, head, ty, tx, 0, th * tw, scale);
}

/* The global path (window 0) is one tile over the whole segment, so the kernel above would launch
 * nseq*nhead blocks with each thread walking rows/N-threads query rows in series (12-32 on a 40 or
 * 64 grid): 60 blocks for a 5-tile page, far too few to fill the card. This variant keeps the same
 * per-row math and key order but tiles the query rows, one block per (segment, query tile, head).
 * Grid positions are the segment's row-major cells, so a tile is a contiguous row range. */
template <int HD>
__global__ __launch_bounds__(kThreads) void sam_attn_qtile_kernel(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const int* __restrict__ cu, const uint16_t* __restrict__ rh,
    const uint16_t* __restrict__ rw, int rh_rows, int rw_rows,
    int q_ld, int k_ld, int v_ld, int o_ld, int nhead, int gh, int gw,
    int nqb, int qrows, int vec_rows, float scale) {
    int b = blockIdx.x;
    const int head = b % nhead; b /= nhead;
    const int qblk = b % nqb;   b /= nqb;
    const int seq  = b;
    const int keys = gh * gw;
    const int lo   = qblk * qrows;
    const int hi   = keys < lo + qrows ? keys : lo + qrows;
    if (vec_rows)
        sam_attn_body<HD, true>(gq, gk, gv, go, cu, rh, rw, rh_rows, rw_rows, q_ld, k_ld, v_ld,
                                o_ld, nhead, gh, gw, gh, gw, seq, head, 0, 0, lo, hi, scale);
    else
        sam_attn_body<HD, false>(gq, gk, gv, go, cu, rh, rw, rh_rows, rw_rows, q_ld, k_ld, v_ld,
                                 o_ld, nhead, gh, gw, gh, gw, seq, head, 0, 0, lo, hi, scale);
}

int launch_sam_attn(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* o,
                    const int* cu, const uint16_t* rh, const uint16_t* rw,
                    int rh_rows, int rw_rows,
                    int q_ld, int k_ld, int v_ld, int o_ld, int nseq, int nhead, int gh, int gw,
                    int win, float scale, long stream) {
    const int th = win > 0 ? win : gh, tw = win > 0 ? win : gw;
    const int nth = (gh + th - 1) / th, ntw = (gw + tw - 1) / tw;
    /* The 16-byte row loads are legal only when every q/k/v/out row starts on a 16-byte
     * boundary: all four bases aligned, every row pitch a multiple of 8 bf16. The selftest's
     * padded pitch and a slice with a filler column both fail this and take the scalar path. */
    const uintptr_t addr_bits = (uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)o;
    const bool vec_rows = (addr_bits & 15u) == 0 &&
                          ((q_ld | k_ld | v_ld | o_ld) & 7) == 0;
    if (win == 0) {
        /* One block per (segment, 128-row query tile, head): the whole-segment tile above would
         * give each thread 12-32 query rows in series and launch only nseq*nhead blocks. */
        const int qrows = 128;
        const int keys  = gh * gw;
        const int nqb   = (keys + qrows - 1) / qrows;
        const size_t qblocks = (size_t)nseq * nqb * nhead;
        if (qblocks == 0 || qblocks > 0x7fffffff) return R4D_NO(RAD_E_SHAPE);
        hipLaunchKernelGGL((sam_attn_qtile_kernel<64>), dim3((uint32_t)qblocks),
                           dim3(kThreads), 0, (hipStream_t)stream, q, k, v, o, cu, rh, rw,
                           rh_rows, rw_rows, q_ld, k_ld, v_ld, o_ld, nhead, gh, gw, nqb, qrows,
                           vec_rows ? 1 : 0, scale);
        return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
    }
    const size_t blocks = (size_t)nseq * nth * ntw * nhead;
    if (blocks == 0 || blocks > 0x7fffffff) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL((sam_attn_kernel<64>), dim3((uint32_t)blocks), dim3(kThreads), 0,
                       (hipStream_t)stream, q, k, v, o, cu, rh, rw, rh_rows, rw_rows,
                       q_ld, k_ld, v_ld, o_ld, nhead, gh, gw, th, tw, nth, ntw,
                       vec_rows ? 1 : 0, scale);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

}  // namespace

// ================================================================================ entry point
//
// The shim checks what the kernel assumes: bf16 operands, rank 2 with dense columns and any row
// pitch (the tower's q/k/v are the three column slices of one fused projection), an i32 cu with one
// entry a segment plus the terminator, tables of exactly 2*tile-1 rows, head_dim 64. The shim
// checks the row count against the segment count host-side; per-segment length is the kernel's to
// check, and a malformed segment writes NaN over its rows (the vit kernel's poison convention)
// so the corruption reaches the consumer instead of passing as success.

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

static inline int bf16_rows(const RadTensor* t, int64_t cols, int64_t* rows, int64_t* ld) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (t->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (t->rank != 2) return RAD_E_SHAPE;
    if (t->shape[1] != cols || t->stride[1] != 1) return RAD_E_SHAPE;
    *rows = t->shape[0];
    *ld   = t->stride[0];
    return RAD_OK;
}

extern "C" int r4d_rad_sam_attn(const RadArgs* a, RadStream s) {
    enum { V_Q = 0, V_K, V_V, V_CU, V_OUT, V_RH, V_RW };
    const RadTensor* q  = r4d_opt(a, V_Q);
    const RadTensor* k  = r4d_opt(a, V_K);
    const RadTensor* v  = r4d_opt(a, V_V);
    const RadTensor* cu = r4d_opt(a, V_CU);
    const RadTensor* o  = r4d_opt(a, V_OUT);
    const RadTensor* rh = r4d_opt(a, V_RH);
    const RadTensor* rw = r4d_opt(a, V_RW);
    if (!q || !k || !v || !cu || !o || !rh || !rw) return R4D_NO(RAD_E_INVAL);

    const long long hd  = r4d_geti_or(a, "head_dim", 0);
    const long long nh  = r4d_geti_or(a, "n_head", 0);
    const long long gh  = r4d_geti_or(a, "grid_h", 0);
    const long long gw  = r4d_geti_or(a, "grid_w", 0);
    const long long win = r4d_geti_or(a, "window", 0);
    if (hd != 64 || nh <= 0 || gh <= 0 || gw <= 0 || win < 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long W = nh * hd;

    int64_t qrows = 0, qld = 0, krows = 0, kld = 0, vrows = 0, vld = 0, orows = 0, old = 0;
    R4D_NEED(bf16_rows(q, W, &qrows, &qld));
    R4D_NEED(bf16_rows(k, W, &krows, &kld));
    R4D_NEED(bf16_rows(v, W, &vrows, &vld));
    R4D_NEED(bf16_rows(o, W, &orows, &old));
    if (qrows <= 0 || krows < qrows || vrows < qrows || orows < qrows) return R4D_NO(RAD_E_SHAPE);

    if (!cu->data) return R4D_NO(RAD_E_INVAL);
    if (cu->dtype != RAD_I32 || !r4d_contig(cu)) return R4D_NO(RAD_E_DTYPE);
    const long long nseq = r4d_numel(cu) - 1;
    if (nseq < 1) return R4D_NO(RAD_E_SHAPE);
    /* The tower declares cu as one segment per grid, so the host-side row count pins the
     * segment count. A mismatch is a malformed segment and fails at issue, not on the device
     * (the kernel's per-segment guard poisons the visible rows for what this cannot see). */
    if (qrows != nseq * gh * gw) return R4D_NO(RAD_E_SHAPE);

    // The tables: [2*tile-1, head_dim] when the geometry has its own, else the checkpoint's rows
    // (the 127-row global table at a 40 grid), resampled on read.
    int64_t rhrows = 0, rhld = 0, rwrows = 0, rwld = 0;
    R4D_NEED(bf16_rows(rh, hd, &rhrows, &rhld));
    R4D_NEED(bf16_rows(rw, hd, &rwrows, &rwld));
    if (rhrows < 1 || rwrows < 1) return R4D_NO(RAD_E_UNSUPPORTED);

    const float scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)hd));

    return launch_sam_attn((const uint16_t*)r4d_addr(q), (const uint16_t*)r4d_addr(k),
                           (const uint16_t*)r4d_addr(v), (uint16_t*)r4d_addr(o),
                           (const int*)r4d_addr(cu), (const uint16_t*)r4d_addr(rh),
                           (const uint16_t*)r4d_addr(rw), (int)rhrows, (int)rwrows,
                           (int)qld, (int)kld, (int)vld, (int)old,
                           (int)nseq, (int)nh, (int)gh, (int)gw, (int)win, scale,
                           (long)(uintptr_t)s);
}
