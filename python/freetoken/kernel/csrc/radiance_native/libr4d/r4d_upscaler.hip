// R4D -- a video latent upscaler's ops (docs/OPS.md, "The latent upscaler"): MiniMax-H3's 3D-conv
// latent resizer, a stack of 3x3x3 convolutions over 512-channel voxel rows with group norms,
// depthwise temporal convolutions and one spatial resize between two resolutions, plus the patch
// fold back to the latent's own rows and the re-noising of a clean latent.
//
// Activations are voxel rows: row (t, y, x) row-major over (frames, rows, columns), channels
// innermost. A request's volume is a device operand (`geom`, i32), never a parameter, so every
// kernel reads it on the device; one the operands cannot hold writes nothing, and rows past what
// the volume and the output hold are left alone.
//
// Each kernel rounds where its libref function does (libref/ref_video.cpp), under the library's
// -ffp-contract=off: the convolutions accumulate in f32 in their own order and round once (twice
// with the residual), the group norm's statistics are f32 merges in a fixed order, and the resize,
// the fold and the re-noise are the reference's arithmetic in its order (the noise through the
// device's log, sin and cos).
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include "rad_philox.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kT = 256;

// Eight consecutive values of a bf16 or f32 row as floats, and a store rounded to its dtype.
__device__ __forceinline__ void ld8(const uint16_t* p, float* f) {
    const uint4 u = *(const uint4*)p;
    f[0] = bf16w_lo(u.x); f[1] = bf16w_hi(u.x); f[2] = bf16w_lo(u.y); f[3] = bf16w_hi(u.y);
    f[4] = bf16w_lo(u.z); f[5] = bf16w_hi(u.z); f[6] = bf16w_lo(u.w); f[7] = bf16w_hi(u.w);
}
__device__ __forceinline__ void ld8(const float* p, float* f) {
    const float4 a = *(const float4*)p, b = *(const float4*)(p + 4);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
__device__ __forceinline__ void st8(uint16_t* p, const float* f) {
    uint4 u;
    u.x = bf16x2_rne(f[0], f[1]);
    u.y = bf16x2_rne(f[2], f[3]);
    u.z = bf16x2_rne(f[4], f[5]);
    u.w = bf16x2_rne(f[6], f[7]);
    *(uint4*)p = u;
}
__device__ __forceinline__ void st8(float* p, const float* f) {
    *(float4*)p = make_float4(f[0], f[1], f[2], f[3]);
    *(float4*)(p + 4) = make_float4(f[4], f[5], f[6], f[7]);
}
// Four- and two-channel forms of the above, and a chunk of CW (8, 4 or 2) channels through them.
__device__ __forceinline__ void ld4(const uint16_t* p, float* f) {
    const uint2 u = *(const uint2*)p;
    f[0] = bf16w_lo(u.x); f[1] = bf16w_hi(u.x); f[2] = bf16w_lo(u.y); f[3] = bf16w_hi(u.y);
}
__device__ __forceinline__ void ld4(const float* p, float* f) {
    const float4 a = *(const float4*)p;
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
}
__device__ __forceinline__ void st4(uint16_t* p, const float* f) {
    *(uint2*)p = make_uint2(bf16x2_rne(f[0], f[1]), bf16x2_rne(f[2], f[3]));
}
__device__ __forceinline__ void st4(float* p, const float* f) {
    *(float4*)p = make_float4(f[0], f[1], f[2], f[3]);
}
__device__ __forceinline__ void ld2(const uint16_t* p, float* f) {
    const uint32_t u = *(const uint32_t*)p;
    f[0] = bf16w_lo(u); f[1] = bf16w_hi(u);
}
__device__ __forceinline__ void ld2(const float* p, float* f) {
    const float2 a = *(const float2*)p;
    f[0] = a.x; f[1] = a.y;
}
__device__ __forceinline__ void st2(uint16_t* p, const float* f) {
    *(uint32_t*)p = bf16x2_rne(f[0], f[1]);
}
__device__ __forceinline__ void st2(float* p, const float* f) {
    *(float2*)p = make_float2(f[0], f[1]);
}
template <int CW, typename T> __device__ __forceinline__ void ldc(const T* p, float* f) {
    if constexpr (CW == 8) ld8(p, f);
    else if constexpr (CW == 4) ld4(p, f);
    else ld2(p, f);
}
template <int CW, typename T> __device__ __forceinline__ void stc(T* p, const float* f) {
    if constexpr (CW == 8) st8(p, f);
    else if constexpr (CW == 4) st4(p, f);
    else st2(p, f);
}
// CW per-channel values of an f32 or bf16 vector, its dtype known only at run time.
template <int CW> __device__ __forceinline__ void ldcw(const void* p, bool f32, int c, float* f) {
    if (f32) ldc<CW>((const float*)p + c, f);
    else     ldc<CW>((const uint16_t*)p + c, f);
}
// The volume a request runs at: (T, H, W) and its row count, or false for a geometry the operands
// cannot hold (`rows` rows of the input).
__device__ __forceinline__ bool volume_of(const int* geom, long long rows, int* T, int* H, int* W,
                                          long long* vol) {
    *T = geom[0];
    *H = geom[1];
    *W = geom[2];
    if (*T <= 0 || *H <= 0 || *W <= 0) return false;
    *vol = (long long)*T * *H * *W;
    return *vol <= rows;
}

// ---------------------------------------------------------------------------- conv3d_ndhwc
// An implicit GEMM over voxel rows: y[voxel][cout] = sum over (tap, cin) of x[voxel's tap
// neighbour][cin] * w[cout][tap][cin], the weight stored tap-major (r4d_layout.cpp relayouts
// PyTorch's [cout][cin][3][3][3] at load), so K = 27 cin runs over taps then channels and a 64-wide
// K tile of a voxel is eight 16-byte reads of one neighbour row. No im2col buffer.
//
// The block is gemm_nt_bias's large-grid tile (vit_gemm_wa_kernel, r4d_vit_bf16.hip): 128 output
// channels x 128 voxels, eight waves as 4 (weight) x 2 (voxels), a wave 2 x 4 WMMA tiles. The
// weight is the WMMA's A operand, read from global straight into registers; the voxels are its B
// operand, staged through LDS (double-buffered, one barrier a tile) and read by the four waves that
// share their rows. Staging both operands through LDS would take all of gfx1201's LDS bandwidth
// (64 bytes a clock per CU) at one WMMA every 16 clocks per SIMD; this costs 256 bytes of fragment
// reads and 64 of stores per WMMA, the weight's bytes coming through the vector cache. A lane's
// accumulator is eight consecutive output channels of one voxel, so the bias, the residual and the
// store are 16-byte accesses.
//
// K is relabelled inside a staged tile as gemm_nt_bias does (lane half h, k step s and element e
// hold k = 32h + 8s + e of the tile, in both operands), so a lane's weight fragments for a tile are
// one 64-byte run of its row and its voxel fragment one 16-byte LDS read; the summation order is
// that permutation's.
//
// The gather reads x as a raw buffer, whose reads at or past its size return zeros, so a neighbour
// outside the volume is the offset kOob and is staged with no select. A thread stages one 16-byte
// chunk column of four voxel rows, so it decomposes its four voxels into (t, y, x) once. With cin a
// multiple of 64 (C64) a K tile lies inside one tap: the four neighbours' byte offsets are
// recomputed only when the tap changes, every cin / 64 tiles, from a block-uniform (dt, dy, dx),
// and the tile's channel offset rides in the load's scalar offset. Otherwise (the 24-channel input)
// each chunk's tap follows its own K, every tile. The weight is a raw buffer too, the tile's K in
// the load's scalar offset and a chunk's in its immediate (the 24-channel input reads it through a
// guarded load instead, its last tile running past K), so neither operand's addresses cost vector
// arithmetic per tile.
//
// `tpad` pads the frames other than with zeros (LTX-2's VAE): 1 replicate, the frame clamped into
// the volume; 2 causal, output frame t's tap dt reading frame max(t + dt - 1 - tpre, 0), so the
// output has tpre more frames than the input. A neighbour's frame is resolved when its offset is,
// once a tap.
//
// Measured on the R9700 at 300 W, warm, against gemm_nt_bias at bf16 at the same M, N and K = 27
// cin: 512 -> 512 at 84 x 48 x 37 148 TFLOP/s (the GEMM 146), at 42 x 24 x 37 149 (145). A
// select-and-pointer form of the gather runs 136 and 134: its loop issues twice the GEMM's vector
// instructions. The weight through global loads at a uniform base is 0.2-1% slower;
// raster groups of 4 or 16 bands are within noise of 8. 512 -> 24 (the output convolution) runs at
// 30 with 13/16 of its 128-wide channel tile idle, a few milliseconds of a whole upscale.
constexpr uint32_t kCTW = 2, kCTA = 4, kCWW = 4, kCWA = 2;
constexpr uint32_t kCBN = kCWW * kCTW * 16, kCBM = kCWA * kCTA * 16;   // channels x voxels
constexpr uint32_t kCThreads = kCWW * kCWA * 32;
constexpr uint32_t kCFK = 64;                                          // K a staged tile
constexpr uint32_t kCPB = kCFK * 2 + 16;                               // LDS row pitch, bytes
constexpr uint32_t kCCPR = kCFK * 2 / 16;                              // 16-byte chunks a row
constexpr uint32_t kCRS = kCThreads / kCCPR;                           // rows one staging pass
constexpr uint32_t kCSPT = kCBM / kCRS;                                // rows a thread stages
constexpr uint32_t kCWC = kCFK / 16;                                   // a lane's weight chunks
constexpr uint32_t kCGroup = 8;                                        // voxel bands a raster group
constexpr int kTaps = 27;
static_assert(kCBM % kCRS == 0, "whole rows a thread");
constexpr uint32_t kOob = 0xFFFF0000u;   // past every x the shim accepts, after any added offset
// gfx12's raw-buffer descriptor word 3 (composable_kernel's for gfx11 and gfx12): offsets checked
// against the size, reads out of range returning zero.
constexpr int kBufWord3 = 0x31004000;

template <bool RES>
__device__ __forceinline__ unsigned short conv_finish(float acc, float bv, unsigned short r) {
    unsigned short t = f32_to_bf16(acc + bv);
    if constexpr (RES) t = f32_to_bf16(bf2f(r) + bf2f(t));
    return t;
}

/* Floor division by b > 0 of a possibly negative a. */
__device__ __forceinline__ int idiv_dev(int a, int b) {
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

/* `vec` (host-checked): y, res and the bias take 16-byte accesses at every eight-column boundary.
 * `xrows` bounds the volume the device geometry may name, `xbytes` is x's extent (at most kOob)
 * and `Mcap` bounds the rows written; `tpad` and `tpre` as above.
 *
 * `GEN`, for any stride, `causal`, `down` or reflect (the encoder's forms): a staged voxel is one
 * of the (T2, H2, W2) output, its base source coordinate the stride less the pad, and with `refl`
 * a tap is mirrored into the volume (torch's reflect, the volume checked by the launcher). */
template <bool C64, bool RES, bool BF32, bool GEN>
__global__ __launch_bounds__(kCThreads) R4D_WAVES(6) void conv3d_kernel(
    const unsigned short* __restrict__ X, const unsigned short* __restrict__ Wt,
    const void* __restrict__ bias, const unsigned short* R, unsigned short* C,
    const int* __restrict__ geom, unsigned ldx, unsigned ldr, unsigned ldc, int Mcap, int xrows,
    unsigned xbytes, int cin, int N, unsigned vec, int st, int sp, int ft, int bk, int pt,
    int pb, int pl, int pr, int refl, int tpad, int tpre) {
    typedef r4d_u32x4 U4;
    __shared__ __attribute__((aligned(16))) unsigned char s_a[2][kCBM * kCPB];

    int gT, gH, gW;
    long long vol;
    if (!volume_of(geom, xrows, &gT, &gH, &gW, &vol)) return;
    const int HW = gH * gW;
    int H2 = gH, W2 = gW;
    long long Mv64 = vol + (long long)tpre * HW;
    if constexpr (GEN) {
        const int T2 = idiv_dev(gT + ft + bk - 3, st) + 1;
        H2 = idiv_dev(gH + pt + pb - 3, sp) + 1;
        W2 = idiv_dev(gW + pl + pr - 3, sp) + 1;
        if (T2 <= 0 || H2 <= 0 || W2 <= 0) return;
        Mv64 = (long long)T2 * H2 * W2;
    }
    const int OHW = H2 * W2;
    const int Mv = Mv64 < (long long)Mcap ? (int)Mv64 : Mcap;

    const uint32_t tid = threadIdx.x, lane = tid & 31u;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const uint32_t ww = wave % kCWW, wa = wave / kCWW;

    // Grouped raster, as gemm_nt_bias's: kCGroup voxel bands sweep the channels together, so the
    // bands' shared halo rows and a channel slice's weight stay in L2 between them.
    uint32_t bm = blockIdx.x, bn = blockIdx.y;
    if (gridDim.x > kCGroup) {
        const uint32_t per = kCGroup * gridDim.y;
        const uint32_t lin = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t first = (lin / per) * kCGroup, r = lin % per;
        const uint32_t gm = min(gridDim.x - first, kCGroup);
        bm = first + r % gm;
        bn = r / gm;
    }
    const uint32_t m0 = bm * kCBM, n0 = bn * kCBN;
    if (m0 >= (uint32_t)Mv) return;

    // This thread's staged rows and their voxels; a row past the volume sits outside every frame.
    // The voxel's row, or with GEN its base source coordinate.
    const uint32_t ch = tid % kCCPR;
    int vm[kCSPT], vt[kCSPT], vy[kCSPT], vx[kCSPT];
    uint32_t adst[kCSPT];
#pragma unroll
    for (uint32_t i = 0; i < kCSPT; ++i) {
        const uint32_t r = tid / kCCPR + kCRS * i;
        const int m = (int)(m0 + r);
        adst[i] = r * kCPB + ch * 16u;
        if constexpr (GEN) {
            vm[i] = -4;   // frame: fails every tap's bounds
            vt[i] = 0;    // row
            vy[i] = 0;    // column
            if (m < Mv) {
                const int t = m / OHW, rem = m - t * OHW, yy = rem / W2;
                vm[i] = t * st - ft;
                vt[i] = yy * sp - pt;
                vy[i] = (rem - yy * W2) * sp - pl;
            }
        } else {
            vm[i] = m;
            vt[i] = -4;
            vy[i] = 0;
            vx[i] = 0;
            if (m < Mv) {
                vt[i] = m / HW;
                const int rem = m - vt[i] * HW;
                vy[i] = rem / gW;
                vx[i] = rem - vy[i] * gW;
            }
        }
    }
    // The four neighbours' byte offsets at a tap, kOob outside the volume; with C64 they include
    // the thread's chunk column.
    const __amdgpu_buffer_rsrc_t xr =
        __builtin_amdgcn_make_buffer_rsrc(const_cast<unsigned short*>(X), 0, (int)xbytes, kBufWord3);
    uint32_t off[kCSPT];
    auto at_tap = [&](int tap) {
        if constexpr (GEN) {
            const int dt = tap / 9, dy = (tap / 3) % 3, dx = tap % 3;   // the base has the pads
#pragma unroll
            for (uint32_t i = 0; i < kCSPT; ++i) {
                int tz = vm[i] + dt;
                int sy = vt[i] + dy, sx = vy[i] + dx;
                if (tpad) tz = min(max(tz, 0), gT - 1);
                if (refl) {   // a mirror with no repeated edge
                    if (sy < 0) sy = -sy; else if (sy >= gH) sy = 2 * gH - 2 - sy;
                    if (sx < 0) sx = -sx; else if (sx >= gW) sx = 2 * gW - 2 - sx;
                }
                const bool ok = tap < kTaps && (unsigned)tz < (unsigned)gT &&
                                (unsigned)sy < (unsigned)gH && (unsigned)sx < (unsigned)gW;
                off[i] = ok
                    ? (uint32_t)(((long long)tz * gH + sy) * gW + sx) * ldx * 2u +
                          (C64 ? ch * 16u : 0u)
                    : kOob;
            }
        } else {
            const int dt = tap / 9 - 1, dy = (tap / 3) % 3 - 1, dx = tap % 3 - 1;
            const int delta = dt * HW + dy * gW + dx;
#pragma unroll
            for (uint32_t i = 0; i < kCSPT; ++i) {
                int ft = vt[i] + dt;
                if (tpad == 1) ft = min(max(ft, 0), gT - 1);
                else if (tpad == 2) ft = max(ft - 1 - tpre, 0);
                const bool ok = tap < kTaps && vt[i] >= 0 && (unsigned)ft < (unsigned)gT &&
                                (unsigned)(vy[i] + dy) < (unsigned)gH &&
                                (unsigned)(vx[i] + dx) < (unsigned)gW;
                const int src = tpad ? (ft * gH + vy[i] + dy) * gW + vx[i] + dx : vm[i] + delta;
                off[i] = ok ? (uint32_t)src * ldx * 2u + (C64 ? ch * 16u : 0u) : kOob;
            }
        }
    };

    // Weight: lane row (l & 15) of each tile, elements [32h, 32h + 32) of the staged tile, as byte
    // offsets to which a tile adds its K. Rows past N re-read row N-1; nothing past N is stored.
    const uint32_t K = (uint32_t)kTaps * (uint32_t)cin;
    const __amdgpu_buffer_rsrc_t wr = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<unsigned short*>(Wt), 0, (int)((uint32_t)N * K * 2u), kBufWord3);
    uint32_t woff[kCTW];
#pragma unroll
    for (uint32_t i = 0; i < kCTW; ++i)
        woff[i] = min(n0 + (ww * kCTW + i) * 16u + (lane & 15u), (uint32_t)N - 1u) * K * 2u +
                  (lane >> 4) * kCFK;
    const uint32_t frow = (wa * kCTA * 16u + (lane & 15u)) * kCPB + (lane >> 4) * kCFK;

    v8f acc[kCTW][kCTA];
#pragma unroll
    for (uint32_t i = 0; i < kCTW; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kCTA; ++j) acc[i][j] = (v8f)(0.f);

    auto mma_tile = [&](const unsigned char* sa, const U4 (&cur)[kCTW][kCWC]) {
#pragma unroll
        for (uint32_t s = 0; s < kCWC; ++s) {
            v8bf bfr_[kCTA];
#pragma unroll
            for (uint32_t j = 0; j < kCTA; ++j)
                bfr_[j] = __builtin_bit_cast(v8bf, *(const U4*)(sa + frow + j * 16u * kCPB + s * 16u));
#pragma unroll
            for (uint32_t i = 0; i < kCTW; ++i) {
                const v8bf af = __builtin_bit_cast(v8bf, cur[i][s]);
#pragma unroll
                for (uint32_t j = 0; j < kCTA; ++j) acc[i][j] = mma(af, bfr_[j], acc[i][j]);
            }
        }
    };

    // The K walk: C64 counts a tap's channel tiles (uniform); otherwise each thread follows its
    // chunk's (tap, channel) and the last tile may run past K, its weight chunks zeroed there.
    const uint32_t nk = (K + kCFK - 1u) / kCFK;
    const uint32_t ctiles = (uint32_t)cin / kCFK;
    int tap = 0;
    uint32_t cc = 0, ci = 0;
    if constexpr (C64) {
        at_tap(0);
    } else {
        tap = (int)(ch * 8u / (uint32_t)cin);
        ci = ch * 8u - (uint32_t)tap * (uint32_t)cin;
        at_tap(tap);
    }
    auto advance = [&]() {
        if constexpr (C64) {
            if (++cc == ctiles) {
                cc = 0;
                at_tap(++tap);
            }
        } else {
            ci += kCFK;
            while (ci >= (uint32_t)cin) { ci -= (uint32_t)cin; ++tap; }
            at_tap(tap);
        }
    };

    U4 pa[kCSPT], w0[kCTW][kCWC], w1[kCTW][kCWC];
    const U4 zero = (U4)(0u);
    /* Every load is issued unconditionally (the last tile's prefetch re-reads the last tile), so
     * the loop's waits count exactly. */
    auto load = [&](uint32_t k0, U4 (&w)[kCTW][kCWC]) {
#pragma unroll
        for (uint32_t i = 0; i < kCSPT; ++i) {
            if constexpr (C64)
                pa[i] = __builtin_amdgcn_raw_buffer_load_b128(xr, (int)off[i], (int)(cc * kCFK * 2u), 0);
            else
                pa[i] = __builtin_amdgcn_raw_buffer_load_b128(xr, (int)(off[i] + ci * 2u), 0, 0);
        }
        const unsigned char* const wb = (const unsigned char*)Wt + (size_t)k0 * 2u;
#pragma unroll
        for (uint32_t i = 0; i < kCTW; ++i)
#pragma unroll
            for (uint32_t c = 0; c < kCWC; ++c) {
                if constexpr (C64) {
                    w[i][c] = __builtin_amdgcn_raw_buffer_load_b128(wr, (int)(woff[i] + c * 16u),
                                                                    (int)(k0 * 2u), 0);
                } else {
                    const bool in = k0 + (lane >> 4) * (kCFK / 2u) + c * 8u < K;
                    w[i][c] = in ? *(const U4*)(wb + (in ? woff[i] + c * 16u : 0u)) : zero;
                }
            }
    };
    auto stage = [&](uint32_t buf) {
        unsigned char* const sa = s_a[buf];
#pragma unroll
        for (uint32_t i = 0; i < kCSPT; ++i) *(U4*)(sa + adst[i]) = pa[i];
    };
    auto tile = [&](uint32_t t, uint32_t buf, U4 (&cur)[kCTW][kCWC], U4 (&nxt)[kCTW][kCWC]) {
        stage(buf);
        lds_barrier();
        if (t + 1u < nk) advance();
        load(min(t + 1u, nk - 1u) * kCFK, nxt);
        mma_tile(s_a[buf], cur);
    };
    load(0, w0);
    uint32_t t = 0;
    for (; t + 1u < nk; t += 2u) {
        tile(t, 0, w0, w1);
        tile(t + 1u, 1, w1, w0);
    }
    if (t < nk) tile(t, 0, w0, w1);

    // D is [channel][voxel]: eight consecutive channels of one output row a lane a tile. The
    // residual is read by the thread that stores the same element, so it may alias y.
#pragma unroll
    for (uint32_t i = 0; i < kCTW; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kCTA; ++j) {
            const uint32_t m = m0 + (wa * kCTA + j) * 16u + (lane & 15u);
            const uint32_t n = n0 + (ww * kCTW + i) * 16u + (lane >> 4) * 8u;
            if (m >= (uint32_t)Mv) continue;
            const v8f v = acc[i][j];
            if (vec && n + 8u <= (uint32_t)N) {
                float bv[8];
                if constexpr (BF32) {
                    const float4 b0 = *(const float4*)((const float*)bias + n);
                    const float4 b1 = *(const float4*)((const float*)bias + n + 4);
                    bv[0] = b0.x; bv[1] = b0.y; bv[2] = b0.z; bv[3] = b0.w;
                    bv[4] = b1.x; bv[5] = b1.y; bv[6] = b1.z; bv[7] = b1.w;
                } else {
                    const uint4 b = *(const uint4*)((const unsigned short*)bias + n);
                    const uint32_t d[4] = { b.x, b.y, b.z, b.w };
#pragma unroll
                    for (uint32_t e = 0; e < 8; ++e) bv[e] = bf2f((unsigned short)(d[e / 2] >> (16 * (e % 2))));
                }
                uint32_t rr[4] = { 0u, 0u, 0u, 0u };
                if constexpr (RES) {
                    const uint4 r4 = *(const uint4*)(R + (size_t)m * ldr + n);
                    rr[0] = r4.x; rr[1] = r4.y; rr[2] = r4.z; rr[3] = r4.w;
                }
                uint32_t o[4];
#pragma unroll
                for (uint32_t e = 0; e < 8; e += 2)
                    o[e / 2] = (uint32_t)conv_finish<RES>(v[e], bv[e], (unsigned short)rr[e / 2]) |
                               ((uint32_t)conv_finish<RES>(v[e + 1], bv[e + 1],
                                                           (unsigned short)(rr[e / 2] >> 16)) << 16);
                *(uint4*)(C + (size_t)m * ldc + n) = make_uint4(o[0], o[1], o[2], o[3]);
            } else {
#pragma unroll
                for (uint32_t e = 0; e < 8; ++e) {
                    if (n + e >= (uint32_t)N) continue;
                    const float bv = BF32 ? ((const float*)bias)[n + e]
                                          : bf2f(((const unsigned short*)bias)[n + e]);
                    const unsigned short r = RES ? R[(size_t)m * ldr + n + e] : (unsigned short)0;
                    C[(size_t)m * ldc + n + e] = conv_finish<RES>(v[e], bv, r);
                }
            }
        }
}

// ---------------------------------------------------------------------------- group_norm
// nn.GroupNorm over all of y's rows in three launches, every reduction in a fixed order so two runs
// give the same bytes:
//
//   1. parts: block p takes rows [p rpp, (p + 1) rpp) and writes each group's (count, mean, M2)
//      of them to the scratch. A row is `tpr` threads, each owning `cpt` consecutive CW-channel
//      chunks of one group, and the block holds `slots` = 256 / tpr rows at a time (the threads
//      past slots * tpr idle). A thread keeps its chunks over every `slots`-th row, summing
//      deviations from its first value (the shift keeps the f32 sums of squares from cancelling
//      against a large mean); the block merges its threads' triples per group in thread order
//      (Chan's parallel update).
//   2. final: block g merges group g's parts, 256 strided runs then a fixed tree, and writes
//      stats[2g] = mean and stats[2g + 1] = rstd = 1 / sqrt(M2 / count + eps), the count the
//      group's exact M n / groups.
//   3. apply: a thread a CW-channel chunk, v = (x - mean) rstd w + b, the modulation and SiLU in
//      f32, one rounding.
//
// The scratch beyond the 2 groups leading values holds part p of group g at 3 (g kGnParts + p).
constexpr int kGnParts = 1024;

__device__ __forceinline__ void chan_merge(float& n, float& mean, float& m2, float nb, float mb,
                                           float m2b) {
    if (nb == 0.0f) return;
    if (n == 0.0f) { n = nb; mean = mb; m2 = m2b; return; }
    const float nab = n + nb;
    const float f = nb / nab;
    const float d = mb - mean;
    mean = mean + d * f;
    m2 = m2 + m2b + d * d * n * f;
    n = nab;
}

/* A chunk is CW channels (8, 4 or 2: the widest dividing a group). `tpr` threads a row, each over
 * `cpt` chunks; `tpg` threads a group (its chunks / cpt). */
template <typename TX, int CW>
__global__ __launch_bounds__(kT) void gn_part_kernel(const TX* __restrict__ x,
                                                     float* __restrict__ stats, int M, int groups,
                                                     int tpr, int cpt, int tpg, int ldx, int rpp) {
    __shared__ float sh[3][kT];
    const int tid = threadIdx.x;
    const int slots = kT / tpr;
    const int c = tid % tpr, slot = tid / tpr;
    const int p = blockIdx.x;
    const int r0 = p * rpp, r1 = min(M, r0 + rpp);
    float cnt = 0.0f, s1 = 0.0f, s2 = 0.0f, shift = 0.0f;
    if (slot < slots)
        for (int r = r0 + slot; r < r1; r += slots)
            for (int q = 0; q < cpt; ++q) {
                float v[CW];
                ldc<CW>(x + (size_t)r * ldx + (c * cpt + q) * CW, v);
                if (cnt == 0.0f) shift = v[0];
#pragma unroll
                for (int j = 0; j < CW; ++j) {
                    const float d = v[j] - shift;
                    s1 += d;
                    s2 += d * d;
                }
                cnt += (float)CW;
            }
    float mean = 0.0f, m2 = 0.0f;
    if (cnt > 0.0f) {
        const float q = s1 / cnt;
        mean = shift + q;
        m2 = fmaxf(s2 - s1 * q, 0.0f);
    }
    sh[0][tid] = cnt;
    sh[1][tid] = mean;
    sh[2][tid] = m2;
    __syncthreads();
    for (int g = tid; g < groups; g += kT) {
        float N = 0.0f, Mn = 0.0f, Q = 0.0f;
        for (int s = 0; s < slots; ++s)
            for (int k = 0; k < tpg; ++k) {
                const int t = s * tpr + g * tpg + k;
                chan_merge(N, Mn, Q, sh[0][t], sh[1][t], sh[2][t]);
            }
        float* o = stats + 2 * groups + ((size_t)g * kGnParts + p) * 3;
        o[0] = N;
        o[1] = Mn;
        o[2] = Q;
    }
}

__global__ __launch_bounds__(kT) void gn_final_kernel(float* __restrict__ stats, int groups,
                                                      int parts, float count, float eps) {
    __shared__ float sh[3][kT];
    const int g = blockIdx.x, tid = threadIdx.x;
    const float* P = stats + 2 * groups + (size_t)g * kGnParts * 3;
    float N = 0.0f, Mn = 0.0f, Q = 0.0f;
    for (int p = tid; p < parts; p += kT) chan_merge(N, Mn, Q, P[3 * p], P[3 * p + 1], P[3 * p + 2]);
    sh[0][tid] = N;
    sh[1][tid] = Mn;
    sh[2][tid] = Q;
    __syncthreads();
    for (int s = kT / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = sh[0][tid], b = sh[1][tid], q = sh[2][tid];
            chan_merge(a, b, q, sh[0][tid + s], sh[1][tid + s], sh[2][tid + s]);
            sh[0][tid] = a;
            sh[1][tid] = b;
            sh[2][tid] = q;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float var = sh[2][0] / count;
        stats[2 * g] = sh[1][0];
        stats[2 * g + 1] = 1.0f / sqrtf(var + eps);
    }
}

/* `wf`, `bf`: the gain's and bias's dtype is f32; `modk` 0 no modulation, 1 a bf16 row, 2 an f32
 * one. A thread is chunk `i % cpr` of row `i / cpr`, `gs8` chunks a group. x and y may be the same
 * memory: a thread reads its chunk before it writes it. */
template <typename TX, typename TY, int CW>
__global__ __launch_bounds__(kT) void gn_apply_kernel(
    const TX* x, TY* y, const void* __restrict__ w, const void* __restrict__ b,
    const void* __restrict__ mod, const float* __restrict__ stats, int M, int n, int cpr,
    int gs8, int ldx, int ldy, int wf, int bf, int modk, int silu) {
    const long long i = (long long)blockIdx.x * kT + threadIdx.x;
    const int r = (int)(i / cpr);
    if (r >= M) return;
    const int ck = (int)(i - (long long)r * cpr), c = ck * CW;
    const int g = ck / gs8;
    const float mean = stats[2 * g], rstd = stats[2 * g + 1];
    float v[CW], wv[CW], bv[CW];
    ldc<CW>(x + (size_t)r * ldx + c, v);
    ldcw<CW>(w, wf != 0, c, wv);
    ldcw<CW>(b, bf != 0, c, bv);
#pragma unroll
    for (int j = 0; j < CW; ++j) v[j] = (v[j] - mean) * rstd * wv[j] + bv[j];
    if (modk) {
        float sc[CW], sf[CW];
        ldcw<CW>(mod, modk == 2, c, sc);
        ldcw<CW>(mod, modk == 2, n + c, sf);
#pragma unroll
        for (int j = 0; j < CW; ++j) v[j] = v[j] * (1.0f + sc[j]) + sf[j];
    }
    if (silu) {
#pragma unroll
        for (int j = 0; j < CW; ++j) v[j] = v[j] / (1.0f + expf(-v[j]));
    }
    stc<CW>(y + (size_t)r * ldy + c, v);
}

// ---------------------------------------------------------------------------- tconv_dw
// A depthwise convolution along the frames: a thread one 8-channel chunk of one frame position,
// walking every frame with the k-frame window in registers, so each input row is read once.
// kTdFB frames a step: their loads are issued together, then the step's outputs, each the taps in
// ascending order in f32 and one rounding after the bias. The grid strides over (position, chunk).
constexpr int kTdK = 5;
constexpr int kTdFB = 4;

__global__ __launch_bounds__(kT) void tconv_dw_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const uint16_t* __restrict__ b,
    uint16_t* __restrict__ y, const int* __restrict__ geom, int Mcap, int xrows, int n, int ldx,
    int ldy) {
    int gT, gH, gW;
    long long vol;
    if (!volume_of(geom, xrows, &gT, &gH, &gW, &vol)) return;
    const int Mv = vol < (long long)Mcap ? (int)vol : Mcap;
    const int P = gH * gW, cpr = n / 8;
    const long long pairs = (long long)P * cpr;
    constexpr int H = kTdK / 2, WIN = kTdFB + kTdK - 1;
    for (long long q = (long long)blockIdx.x * kT + threadIdx.x; q < pairs;
         q += (long long)gridDim.x * kT) {
        const int p = (int)(q / cpr), c = (int)(q % cpr) * 8;
        float wv[8][kTdK], bv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#pragma unroll
            for (int j = 0; j < kTdK; ++j) wv[e][j] = bf2f(w[(size_t)(c + e) * kTdK + j]);
        }
        ld8(b + c, bv);
        // win[i] holds frame t0 - H + i; frames outside [0, T) stay zero and add nothing.
        uint4 win[WIN];
#pragma unroll
        for (int i = 0; i < WIN; ++i) win[i] = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int i = H; i < kTdK - 1; ++i)
            if (i - H < gT) win[i] = *(const uint4*)(x + ((size_t)(i - H) * P + p) * ldx + c);
        for (int t0 = 0; t0 < gT; t0 += kTdFB) {
#pragma unroll
            for (int i = kTdK - 1; i < WIN; ++i) {
                const int f = t0 - H + i;
                win[i] = f < gT ? *(const uint4*)(x + ((size_t)f * P + p) * ldx + c)
                                : make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (int s = 0; s < kTdFB; ++s) {
                const int t = t0 + s;
                const long long row = (long long)t * P + p;
                if (t >= gT || row >= Mv) break;
                float acc[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[e] = 0.0f;
#pragma unroll
                for (int j = 0; j < kTdK; ++j) {
                    const int f = t + j - H;
                    if (f < 0 || f >= gT) continue;
                    float xv[8];
                    const uint4 u = win[s + j];
                    xv[0] = bf16w_lo(u.x); xv[1] = bf16w_hi(u.x); xv[2] = bf16w_lo(u.y);
                    xv[3] = bf16w_hi(u.y); xv[4] = bf16w_lo(u.z); xv[5] = bf16w_hi(u.z);
                    xv[6] = bf16w_lo(u.w); xv[7] = bf16w_hi(u.w);
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[e] = acc[e] + wv[e][j] * xv[e];
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[e] = acc[e] + bv[e];
                st8(y + (size_t)row * ldy + c, acc);
            }
#pragma unroll
            for (int i = 0; i < kTdK - 1; ++i) win[i] = win[i + kTdFB];
        }
    }
}

// ---------------------------------------------------------------------------- resize_bilinear
// torch's bilinear resize of each frame, align_corners=False at a given size (upsample_bilinear2d's
// source index and lambdas, in f32): a thread one 8-channel chunk of one output voxel, its two
// rows' and two columns' taps recomputed from its indices.
__device__ __forceinline__ void bil_axis(int o, int in, int out, int* i0, int* i1, float* l0,
                                         float* l1) {
    const float s = (float)in / (float)out;
    float src = s * ((float)o + 0.5f) - 0.5f;
    src = src < 0.0f ? 0.0f : src;
    *i0 = (int)src;
    *i1 = *i0 + (*i0 < in - 1 ? 1 : 0);
    *l1 = src - (float)*i0;
    *l0 = 1.0f - *l1;
}

template <typename TV>
__global__ __launch_bounds__(kT) void resize_bilinear_kernel(
    const TV* __restrict__ x, TV* __restrict__ y, const int* __restrict__ geom, long long cap,
    long long xrows, int cpr, int ldx, int ldy) {
    const int T = geom[0], h = geom[1], w = geom[2], H = geom[3], W = geom[4];
    if (T <= 0 || h <= 0 || w <= 0 || H <= 0 || W <= 0) return;
    if ((long long)T * h * w > xrows) return;
    const long long rows = (long long)T * H * W;
    const long long id = (long long)blockIdx.x * kT + threadIdx.x;
    const long long r = id / cpr;
    if (r >= rows || r >= cap) return;
    const int c = (int)(id % cpr) * 8;
    const int t = (int)(r / ((long long)H * W));
    const int rem = (int)(r - (long long)t * H * W);
    const int oi = rem / W, oj = rem - oi * W;
    int i0, i1, j0, j1;
    float ly0, ly1, lx0, lx1;
    bil_axis(oi, h, H, &i0, &i1, &ly0, &ly1);
    bil_axis(oj, w, W, &j0, &j1, &lx0, &lx1);
    const size_t f0 = (size_t)t * h * w;
    float a[8], b[8], d[8], e[8], o[8];
    ld8(x + (f0 + (size_t)i0 * w + j0) * ldx + c, a);
    ld8(x + (f0 + (size_t)i0 * w + j1) * ldx + c, b);
    ld8(x + (f0 + (size_t)i1 * w + j0) * ldx + c, d);
    ld8(x + (f0 + (size_t)i1 * w + j1) * ldx + c, e);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        o[k] = ly0 * (lx0 * a[k] + lx1 * b[k]) + ly1 * (lx0 * d[k] + lx1 * e[k]);
    st8(y + (size_t)r * ldy + c, o);
}

// ---------------------------------------------------------------------------- patch_fold
// patch_unfold's inverse: voxel rows to patch rows, f32, a thread an output value. The two lists
// are f32 kernel arguments.
constexpr int kMaxChan = 64;
struct ChanList { float v[kMaxChan]; };

template <typename TX>
__global__ __launch_bounds__(kT) void patch_fold_kernel(
    const TX* __restrict__ x, const int* __restrict__ geom, float* __restrict__ y, ChanList mul,
    ChanList add, int C, int pt, int ph, int pw, int ldx, int ldy, long long cap, long long xrows) {
    const int F = geom[0], Hp = geom[1], Wp = geom[2];
    if (F <= 0 || Hp <= 0 || Wp <= 0) return;
    const long long Hv = (long long)Hp * ph, Wv = (long long)Wp * pw;
    if ((long long)F * pt * Hv * Wv > xrows) return;
    const int K = C * pt * ph * pw;
    const long long rows = (long long)F * Hp * Wp;
    const long long id = (long long)blockIdx.x * kT + threadIdx.x;
    const long long r = id / K;
    if (r >= rows || r >= cap) return;
    const int col = (int)(id % K);
    const int c = col / (pt * ph * pw), sub = col % (pt * ph * pw);
    const int a = sub / (ph * pw), bb = (sub / pw) % ph, d = sub % pw;
    const long long f = r / ((long long)Hp * Wp), i = (r / Wp) % Hp, j = r % Wp;
    const long long vox = ((f * pt + a) * Hv + i * ph + bb) * Wv + j * pw + d;
    float v;
    if constexpr (sizeof(TX) == 2) v = bf2f(x[vox * ldx + c]);
    else                           v = x[vox * ldx + c];
    y[r * ldy + col] = v * mul.v[c] + add.v[c];
}

// ---------------------------------------------------------------------------- flow_renoise
// x = (1 - s) x + s z, z element k n + c of the (seed, stream) normal stream: a thread a value,
// each product and the sum one f32 rounding.
__global__ __launch_bounds__(kT) void flow_renoise_kernel(
    const float* __restrict__ sigma, const uint32_t* __restrict__ seed, float* __restrict__ x,
    int n, int ldx, long long total, unsigned long long stream) {
    const long long id = (long long)blockIdx.x * kT + threadIdx.x;
    if (id >= total) return;
    const long long k = id / n;
    const int c = (int)(id % n);
    const float s = sigma[0];
    const uint64_t sd = (uint64_t)seed[0] | ((uint64_t)seed[1] << 32);
    const float z = rad_philox_normal_s(sd, stream, (uint64_t)id);
    float* p = x + k * ldx + c;
    const float a = (1.0f - s) * *p;
    const float bz = s * z;
    *p = a + bz;
}

// ---------------------------------------------------------------------------- cond_aug
// x = a x + oma z, a the aug parameter, oma the f32 of the f64 1 - a, z element k n + c of the
// (seed, stream) normal stream: a thread a value, each product and the sum one f32 rounding.
__global__ __launch_bounds__(kT) void cond_aug_kernel(
    const uint32_t* __restrict__ seed, float* __restrict__ x, float a, float oma,
    int n, int ldx, long long total, unsigned long long stream) {
    const long long id = (long long)blockIdx.x * kT + threadIdx.x;
    if (id >= total) return;
    const long long k = id / n;
    const int c = (int)(id % n);
    const uint64_t sd = (uint64_t)seed[0] | ((uint64_t)seed[1] << 32);
    const float z = rad_philox_normal_s(sd, stream, (uint64_t)id);
    float* p = x + k * ldx + c;
    const float av = a * *p;
    const float bz = oma * z;
    *p = av + bz;
}

inline bool is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
inline bool is_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }
inline bool is_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }

// A row-major operand's row count and pitch, the pitch a multiple of eight elements and the base
// 16-byte aligned, so a row's eight-element chunks are 16- or 32-byte accesses.
inline bool rows8(const RadTensor* t, long long n, long long* rows, long long* ld) {
    if (!t || n <= 0 || (n & 7)) return false;
    *ld = r4d_rows_pitch(t, n, rows);
    return *ld >= 0 && !(*ld & 7) && !((uintptr_t)t->data & 15);
}

// A space-separated list of floats kept in f32, as libref's parse_f32_list.
int parse_f32_list(const char* s, float* out, int max) {
    int n = 0;
    while (s && *s) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) break;
        if (n == max) return max + 1;
        out[n++] = (float)v;
        s = end;
    }
    return n;
}

}  // namespace

// ============================================================== the plugin ABI entry points

/* ------------------------------------------------------------------ conv3d_ndhwc */
enum { C3_X = 0, C3_W, C3_B, C3_GEOM, C3_RES, C3_Y };

extern "C" int r4d_rad_conv3d_ndhwc(const RadArgs* a, RadStream stream) {
    const RadTensor *x = r4d_opt(a, C3_X), *w = r4d_opt(a, C3_W), *b = r4d_opt(a, C3_B);
    const RadTensor *geom = r4d_opt(a, C3_GEOM), *res = r4d_opt(a, C3_RES), *y = r4d_opt(a, C3_Y);
    if (!x || !w || !b || !geom || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(w) || !is_bf16(y) || (!is_bf16(b) && !is_f32(b)) ||
        !is_i32(geom) || (res && !is_bf16(res)))
        return R4D_NO(RAD_E_DTYPE);
    const long long cin = r4d_geti_or(a, "cin", 0), cout = r4d_geti_or(a, "cout", 0);
    /* 3 x 3 x 3 only: the tap decode is the kernel's constant. */
    if (r4d_geti_or(a, "k", 3) != 3 || r4d_geti_or(a, "kt", 3) != 3) return R4D_NO(RAD_E_SHAPE);
    if (cin <= 0 || cout <= 0 || (cin & 7)) return R4D_NO(RAD_E_SHAPE);
    /* The encoder's CausalConv3d forms (GEN above); the pads are k / 2 = 1, or kt - 1 = 2 in
     * front with `causal`. */
    const long long sp = r4d_geti_or(a, "stride", 1), st = r4d_geti_or(a, "stride_t", 1);
    const long long causal = r4d_geti_or(a, "causal", 0), reflect = r4d_geti_or(a, "reflect", 0);
    const char* pads = r4d_gets(a, "pad");
    if (pads && std::strcmp(pads, "same") && std::strcmp(pads, "down")) return R4D_NO(RAD_E_SHAPE);
    const bool down = pads && !std::strcmp(pads, "down");
    if (sp <= 0 || st <= 0 || sp > (1ll << 30) || st > (1ll << 30) || (causal && causal != 1) ||
        (reflect && reflect != 1))
        return R4D_NO(RAD_E_SHAPE);
    const bool gen = sp != 1 || st != 1 || causal || down || reflect;
    const int pt = down ? 0 : 1, pl = down ? 0 : 1, pb = 1, pr = 1;
    const long long K = (long long)kTaps * cin;
    if (r4d_numel(w) < cout * K || r4d_numel(b) < cout || r4d_numel(geom) < 3)
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(w) || !r4d_contig(b) || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    const char* tp = r4d_gets(a, "tpad");
    int tpad = 0;
    if (tp && *tp) {
        if (!std::strcmp(tp, "replicate")) tpad = 1;
        else if (!std::strcmp(tp, "causal")) tpad = 2;
        else if (std::strcmp(tp, "zero")) return R4D_NO(RAD_E_INVAL);
    }
    const long long tpre = r4d_geti_or(a, "tpre", 0);
    if (tpre < 0 || tpre > 64 || (tpre && tpad != 2) || (causal && tpad))
        return R4D_NO(RAD_E_INVAL);
    /* GEN's time pads: `causal` and `tpad` causal pad the front only, `tpre` frames more. */
    const int ft = causal ? 2 : tpad == 2 ? 2 + (int)tpre : 1;
    const int bk = causal || tpad == 2 ? 0 : 1;
    long long rx = 0, ldx = 0, ry = 0, rr = 0;
    if (!rows8(x, cin, &rx, &ldx) || !r4d_aligned(r4d_p(w), 16)) return R4D_NO(RAD_E_STRIDE);
    const long long ldy = r4d_rows_pitch(y, cout, &ry);
    const long long ldr = res ? r4d_rows_pitch(res, cout, &rr) : 0;
    if (ldy < 0 || ldr < 0) return R4D_NO(RAD_E_STRIDE);
    /* The rows written are the device volume's, as libref's; the outputs bound them (and the
     * grid), M is not read. */
    long long Mcap = ry;
    if (res && Mcap > rr) Mcap = rr;
    /* Byte offsets into x and the weight are 32-bit; x's extent (its last row ends at cin) is
     * the gather's buffer size, below the out-of-range offset. */
    const long long xbytes = ((rx - 1) * ldx + cin) * 2;
    if (Mcap <= 0 || rx <= 0 || xbytes > (long long)kOob || cout * K * 2 >= (1ll << 32) ||
        Mcap >= (1ll << 31) || cin > 16384)
        return R4D_NO(RAD_E_SHAPE);
    auto Xp = (const unsigned short*)r4d_p(x);
    auto Wp = (const unsigned short*)r4d_p(w);
    auto Rp = (const unsigned short*)r4d_p(res);
    auto Cp = (unsigned short*)r4d_wp(y);
    const void* Bp = r4d_p(b);
    const unsigned vec = (ldy % 8) == 0 && r4d_aligned(Cp, 16) && r4d_aligned(Bp, 16) &&
                         (!res || ((ldr % 8) == 0 && r4d_aligned(Rp, 16)));
    const dim3 grid((unsigned)((Mcap + kCBM - 1) / kCBM), (unsigned)((cout + kCBN - 1) / kCBN));
    const bool c64 = cin % kCFK == 0, bf32 = is_f32(b);
#define C3_L(G, C6, RS, BF)                                                                       \
    return R4D_EMIT(stream, (conv3d_kernel<C6, RS, BF, G>), grid, dim3(kCThreads), 0, Xp, Wp,     \
                    Bp, Rp, Cp, (const int*)r4d_p(geom), (unsigned)ldx, (unsigned)ldr,            \
                    (unsigned)ldy, (int)Mcap, (int)rx, (unsigned)xbytes, (int)cin, (int)cout,     \
                    vec, (int)st, (int)sp, ft, bk, pt, pb, pl, pr, (int)reflect, tpad,            \
                    (int)tpre)
#define C3_RES(G, C6, BF) do { if (res) C3_L(G, C6, true, BF); C3_L(G, C6, false, BF); } while (0)
#define C3_BF(G, C6) do { if (bf32) C3_RES(G, C6, true); C3_RES(G, C6, false); } while (0)
#define C3_C6(G) do { if (c64) C3_BF(G, true); C3_BF(G, false); } while (0)
    if (gen) C3_C6(true);
    C3_C6(false);
#undef C3_C6
#undef C3_BF
#undef C3_RES
#undef C3_L
}

/* ------------------------------------------------------------------ group_norm */
enum { GN_X = 0, GN_W, GN_B, GN_MOD, GN_STATS, GN_Y };

extern "C" int r4d_rad_group_norm(const RadArgs* a, RadStream stream) {
    const RadTensor *x = r4d_opt(a, GN_X), *w = r4d_opt(a, GN_W), *b = r4d_opt(a, GN_B);
    const RadTensor *mod = r4d_opt(a, GN_MOD), *st = r4d_opt(a, GN_STATS), *y = r4d_opt(a, GN_Y);
    if (!x || !w || !b || !st || !y) return R4D_NO(RAD_E_INVAL);
    if ((!is_bf16(x) && !is_f32(x)) || (!is_bf16(y) && !is_f32(y)) || (!is_bf16(w) && !is_f32(w)) ||
        (!is_bf16(b) && !is_f32(b)) || (mod && !is_bf16(mod) && !is_f32(mod)) || !is_f32(st))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), groups = r4d_geti_or(a, "groups", 0);
    const char* act = r4d_gets(a, "act");
    int silu = 0;
    if (act && *act) {
        if (!std::strcmp(act, "silu")) silu = 1;
        else if (std::strcmp(act, "none")) return R4D_NO(RAD_E_INVAL);
    }
    /* A chunk is CW channels inside one group: 8, 4 or 2, the widest dividing the group (the
     * FLUX.2 VAE's 128 channels in 32 groups take 4, SDXL's 320 in 32 take 2). A row's chunks are
     * split over at most 256 threads, each taking `cpt` consecutive chunks of one group, `cpt`
     * the smallest divisor of the group's chunks that fits the row in the block. */
    if (groups <= 0 || groups > kT || n <= 0 || n % groups || (n / groups) % 2)
        return R4D_NO(RAD_E_SHAPE);
    const long long cg = n / groups;
    const int CW = cg % 8 == 0 ? 8 : cg % 4 == 0 ? 4 : 2;
    const int cpr = (int)(n / CW), gs8 = (int)(cg / CW);
    int cpt = 1;
    while (cpr / cpt > kT || gs8 % cpt) ++cpt;
    const int tpr = cpr / cpt, tpg = gs8 / cpt;
    if (r4d_numel(w) < n || r4d_numel(b) < n || (mod && r4d_numel(mod) < 2 * n) ||
        r4d_numel(st) < groups * (2 + 3 * kGnParts))
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(w) || !r4d_contig(b) || (mod && !r4d_contig(mod)) || !r4d_contig(st) ||
        !r4d_aligned(r4d_p(w), 16) || !r4d_aligned(r4d_p(b), 16) || !r4d_aligned(r4d_p(mod), 16))
        return R4D_NO(RAD_E_STRIDE);
    long long rx = 0, ldx = 0, ry = 0, ldy = 0;
    if (!rows8(x, n, &rx, &ldx) || !rows8(y, n, &ry, &ldy)) return R4D_NO(RAD_E_STRIDE);
    /* Over y's rows, as libref's (x may hold more); M is not read. A part's count is exact in f32
     * (rpp cg <= 2^24); the merged counts only weigh the merges, and the variance divides by the
     * group's exact count. */
    const long long M = ry;
    if (M <= 0 || M > rx || M > ry || M * cg >= (1ll << 31)) return R4D_NO(RAD_E_SHAPE);
    const int slots = kT / tpr;
    /* Whole passes of the block a part, as few parts as fill the scratch. */
    long long rpp = (M + kGnParts - 1) / kGnParts;
    rpp = (rpp + slots - 1) / slots * slots;
    const int parts = (int)((M + rpp - 1) / rpp);
    float* S = (float*)r4d_wp(st);
    const float eps = (float)r4d_getf(a, "eps", 1e-5);
    int rc;
#define GN_P(TX, C4)                                                                              \
    rc = R4D_EMIT(stream, (gn_part_kernel<TX, C4>), dim3((unsigned)parts), dim3(kT), 0,           \
                  (const TX*)r4d_p(x), S, (int)M, (int)groups, tpr, cpt, tpg, (int)ldx, (int)rpp)
    if (is_f32(x)) {
        if (CW == 8)      GN_P(float, 8);
        else if (CW == 4) GN_P(float, 4);
        else              GN_P(float, 2);
    } else {
        if (CW == 8)      GN_P(uint16_t, 8);
        else if (CW == 4) GN_P(uint16_t, 4);
        else              GN_P(uint16_t, 2);
    }
#undef GN_P
    if (rc != RAD_OK) return R4D_NO(rc);
    rc = R4D_EMIT(stream, gn_final_kernel, dim3((unsigned)groups), dim3(kT), 0, S, (int)groups,
                  parts, (float)(M * cg), eps);
    if (rc != RAD_OK) return R4D_NO(rc);
    const int modk = !mod ? 0 : is_f32(mod) ? 2 : 1;
    const dim3 grid((unsigned)((M * cpr + kT - 1) / kT));
#define GN_A2(TX, TY, C4)                                                                         \
    return R4D_EMIT(stream, (gn_apply_kernel<TX, TY, C4>), grid, dim3(kT), 0, (const TX*)r4d_p(x),\
                    (TY*)r4d_wp(y), r4d_p(w), r4d_p(b), r4d_p(mod), (const float*)S, (int)M,      \
                    (int)n, cpr, gs8, (int)ldx, (int)ldy, (int)is_f32(w), (int)is_f32(b),         \
                    modk, silu)
#define GN_A(TX, TY)                                                                              \
    do { if (CW == 8) GN_A2(TX, TY, 8); if (CW == 4) GN_A2(TX, TY, 4); GN_A2(TX, TY, 2); } while (0)
    if (is_f32(x)) {
        if (is_f32(y)) GN_A(float, float);
        GN_A(float, uint16_t);
    }
    if (is_f32(y)) GN_A(uint16_t, float);
    GN_A(uint16_t, uint16_t);
#undef GN_A
#undef GN_A2
}

/* ------------------------------------------------------------------ tconv_dw */
enum { TD_X = 0, TD_W, TD_B, TD_GEOM, TD_Y };

extern "C" int r4d_rad_tconv_dw(const RadArgs* a, RadStream stream) {
    const RadTensor *x = r4d_opt(a, TD_X), *w = r4d_opt(a, TD_W), *b = r4d_opt(a, TD_B);
    const RadTensor *geom = r4d_opt(a, TD_GEOM), *y = r4d_opt(a, TD_Y);
    if (!x || !w || !b || !geom || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(w) || !is_bf16(b) || !is_bf16(y) || !is_i32(geom))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), k = r4d_geti_or(a, "k", 0);
    if (n <= 0 || (n & 7) || k != kTdK || r4d_numel(w) < n * k || r4d_numel(b) < n ||
        r4d_numel(geom) < 3)
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(w) || !r4d_contig(b) || !r4d_contig(geom) || !r4d_aligned(r4d_p(b), 16))
        return R4D_NO(RAD_E_STRIDE);
    long long rx = 0, ldx = 0, ry = 0, ldy = 0;
    if (!rows8(x, n, &rx, &ldx) || !rows8(y, n, &ry, &ldy)) return R4D_NO(RAD_E_STRIDE);
    /* The device volume's rows, at most y's; M is not read. */
    const long long M = ry;
    if (M <= 0 || rx * ldx >= (1ll << 40)) return R4D_NO(RAD_E_SHAPE);
    /* One frame position's chunks a thread, at most the rows' worth; the kernel strides over the
     * device geometry's (position, chunk) pairs. */
    const long long pairs = M * (n / 8);
    const unsigned blocks = (unsigned)std::min<long long>((pairs + kT - 1) / kT, 2048);
    return R4D_EMIT(stream, tconv_dw_kernel, dim3(blocks), dim3(kT), 0,
                    (const uint16_t*)r4d_p(x), (const uint16_t*)r4d_p(w),
                    (const uint16_t*)r4d_p(b), (uint16_t*)r4d_wp(y), (const int*)r4d_p(geom),
                    (int)M, (int)std::min<long long>(rx, (1ll << 31) - 1), (int)n, (int)ldx,
                    (int)ldy);
}

/* ------------------------------------------------------------------ resize_bilinear */
enum { RB_X = 0, RB_GEOM, RB_Y };

extern "C" int r4d_rad_resize_bilinear(const RadArgs* a, RadStream stream) {
    const RadTensor *x = r4d_opt(a, RB_X), *geom = r4d_opt(a, RB_GEOM), *y = r4d_opt(a, RB_Y);
    if (!x || !geom || !y) return R4D_NO(RAD_E_INVAL);
    if ((!is_bf16(x) && !is_f32(x)) || x->dtype != y->dtype || !is_i32(geom))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0 || (n & 7) || r4d_numel(geom) < 5) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    long long rx = 0, ldx = 0, ry = 0, ldy = 0;
    if (!rows8(x, n, &rx, &ldx) || !rows8(y, n, &ry, &ldy)) return R4D_NO(RAD_E_STRIDE);
    /* The device volume's rows, at most y's; M is not read. */
    const long long M = ry;
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long total = M * (n / 8);
    const dim3 grid((unsigned)((total + kT - 1) / kT));
    if (is_f32(x))
        return R4D_EMIT(stream, resize_bilinear_kernel<float>, grid, dim3(kT), 0,
                        (const float*)r4d_p(x), (float*)r4d_wp(y), (const int*)r4d_p(geom), M, rx,
                        (int)(n / 8), (int)ldx, (int)ldy);
    return R4D_EMIT(stream, resize_bilinear_kernel<uint16_t>, grid, dim3(kT), 0,
                    (const uint16_t*)r4d_p(x), (uint16_t*)r4d_wp(y), (const int*)r4d_p(geom), M,
                    rx, (int)(n / 8), (int)ldx, (int)ldy);
}

/* ------------------------------------------------------------------ patch_fold */
enum { PF_X = 0, PF_GEOM, PF_Y };

extern "C" int r4d_rad_patch_fold(const RadArgs* a, RadStream stream) {
    const RadTensor *x = r4d_opt(a, PF_X), *geom = r4d_opt(a, PF_GEOM), *y = r4d_opt(a, PF_Y);
    if (!x || !geom || !y) return R4D_NO(RAD_E_INVAL);
    if ((!is_f32(x) && !is_bf16(x)) || !is_f32(y) || !is_i32(geom)) return R4D_NO(RAD_E_DTYPE);
    const long long C = r4d_geti_or(a, "C", 0), pt = r4d_geti_or(a, "pt", 0);
    const long long ph = r4d_geti_or(a, "ph", 0), pw = r4d_geti_or(a, "pw", 0);
    ChanList mul{}, add{};
    if (C <= 0 || C > kMaxChan || pt <= 0 || ph <= 0 || pw <= 0 ||
        parse_f32_list(r4d_gets(a, "mul"), mul.v, kMaxChan) != C ||
        parse_f32_list(r4d_gets(a, "add"), add.v, kMaxChan) != C || r4d_numel(geom) < 3)
        return R4D_NO(RAD_E_SHAPE);
    const long long K = C * pt * ph * pw;
    long long rx = 0, ry = 0;
    const long long ldx = r4d_rows_pitch(x, C, &rx), ldy = r4d_rows_pitch(y, K, &ry);
    if (ldx < 0 || ldy < 0 || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    /* The device geometry's patches, at most y's rows; M is not read. */
    const long long M = ry;
    if (M <= 0 || rx <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long total = M * K;
    const dim3 grid((unsigned)((total + kT - 1) / kT));
    if (is_f32(x))
        return R4D_EMIT(stream, patch_fold_kernel<float>, grid, dim3(kT), 0,
                        (const float*)r4d_p(x), (const int*)r4d_p(geom), (float*)r4d_wp(y), mul,
                        add, (int)C, (int)pt, (int)ph, (int)pw, (int)ldx, (int)ldy, M, rx);
    return R4D_EMIT(stream, patch_fold_kernel<uint16_t>, grid, dim3(kT), 0,
                    (const uint16_t*)r4d_p(x), (const int*)r4d_p(geom), (float*)r4d_wp(y), mul,
                    add, (int)C, (int)pt, (int)ph, (int)pw, (int)ldx, (int)ldy, M, rx);
}

/* ------------------------------------------------------------------ flow_renoise */
enum { FR_SIGMA = 0, FR_SEED, FR_X };

extern "C" int r4d_rad_flow_renoise(const RadArgs* a, RadStream stream) {
    const RadTensor *sg = r4d_opt(a, FR_SIGMA), *seed = r4d_opt(a, FR_SEED), *x = r4d_opt(a, FR_X);
    if (!sg || !seed || !x) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(sg) || !is_f32(x) || (seed->dtype != RAD_U32 && seed->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), sm = r4d_geti_or(a, "stream", 0);
    if (n <= 0 || sm < 0 || r4d_numel(sg) < 1 || r4d_numel(seed) < 2) return R4D_NO(RAD_E_SHAPE);
    long long rx = 0;
    const long long ldx = r4d_rows_pitch(x, n, &rx);
    if (ldx < 0 || !r4d_contig(seed)) return R4D_NO(RAD_E_STRIDE);
    /* Every row of x, as libref's; M is not read. */
    const long long M = rx;
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long total = M * n;
    return R4D_EMIT(stream, flow_renoise_kernel, dim3((unsigned)((total + kT - 1) / kT)), dim3(kT),
                    0, (const float*)r4d_p(sg), (const uint32_t*)r4d_p(seed), (float*)r4d_wp(x),
                    (int)n, (int)ldx, total, (unsigned long long)sm);
}

/* ------------------------------------------------------------------ cond_aug */
enum { CA_SEED = 0, CA_X };

extern "C" int r4d_rad_cond_aug(const RadArgs* a, RadStream stream) {
    const RadTensor *seed = r4d_opt(a, CA_SEED), *x = r4d_opt(a, CA_X);
    if (!seed || !x) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(x) || (seed->dtype != RAD_U32 && seed->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), sm = r4d_geti_or(a, "stream", 0);
    const double aug = r4d_getf(a, "aug", -1.0);
    if (n <= 0 || sm < 0 || !(aug >= 0.0 && aug < 1.0) || r4d_numel(seed) < 2)
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0;
    const long long ldx = r4d_rows_pitch(x, n, &rx);
    if (ldx < 0 || !r4d_contig(seed)) return R4D_NO(RAD_E_STRIDE);
    /* Every row of x, as libref's; M is not read. */
    const long long M = rx;
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long total = M * n;
    const float av = (float)aug, omv = (float)(1.0 - aug);
    return R4D_EMIT(stream, cond_aug_kernel, dim3((unsigned)((total + kT - 1) / kT)), dim3(kT),
                    0, (const uint32_t*)r4d_p(seed), (float*)r4d_wp(x), av, omv,
                    (int)n, (int)ldx, total, (unsigned long long)sm);
}
