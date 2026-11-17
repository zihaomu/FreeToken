// R4D -- the video generator's ops (docs/OPS.md, "Video generation"): a diffusion transformer's
// modulated RMS norm and its sampler step, the video decoder's patch unfold, the tiles it runs
// over, their stitch back into frames and their rotary positions; the video encoder's plan, cut
// and blend for a picture bigger than one volume; and its soundtrack decoder's alias-free
// SnakeBeta (docs/OPS.md, "Audio generation").
//
// Each kernel rounds where its libref function does (libref/ref_gen.cpp, libref/ref_video.cpp,
// libref/ref_audio.cpp), under the library's -ffp-contract=off. The norm's sum of squares is a
// block sum here and a sequential one there, so a norm output can land one bf16 step apart, and
// sin and exp are the device's; every other value is the reference's arithmetic in its order.
//
// A request's latent geometry is a device operand (`geom`), never a parameter, so the decoder's
// kernels read it on the device and recompute an axis's tile layout from its length there. Rows
// past what the geometry writes are left alone.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include "r4d_q8.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kT = 256;

__device__ __forceinline__ float bfr(float v) { return bf2f(f32_to_bf16(v)); }

__device__ __forceinline__ void ld8(const uint16_t* p, float* f) {
    const uint4 u = *(const uint4*)p;
    f[0] = bf16w_lo(u.x); f[1] = bf16w_hi(u.x); f[2] = bf16w_lo(u.y); f[3] = bf16w_hi(u.y);
    f[4] = bf16w_lo(u.z); f[5] = bf16w_hi(u.z); f[6] = bf16w_lo(u.w); f[7] = bf16w_hi(u.w);
}
__device__ __forceinline__ void st8(uint16_t* p, const float* f) {
    uint4 u;
    u.x = (uint32_t)f32_to_bf16(f[0]) | ((uint32_t)f32_to_bf16(f[1]) << 16);
    u.y = (uint32_t)f32_to_bf16(f[2]) | ((uint32_t)f32_to_bf16(f[3]) << 16);
    u.z = (uint32_t)f32_to_bf16(f[4]) | ((uint32_t)f32_to_bf16(f[5]) << 16);
    u.w = (uint32_t)f32_to_bf16(f[6]) | ((uint32_t)f32_to_bf16(f[7]) << 16);
    *(uint4*)p = u;
}

// The entry that owns row r of a cu_seqlens partition: linear, because a step has three
// segments and a vocoder pass two channels.
__device__ __forceinline__ int seq_of(const int* cu, int n_seq, int r) {
    int s = 0;
    while (s + 1 < n_seq && cu[s + 1] <= r) ++s;
    return s;
}

// A sum over the 256 threads, every thread receiving it. `sh` is 8 floats of LDS.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = r4d_wave_sum(v);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    #pragma unroll
    for (int i = 0; i < kT / 32; ++i) t += sh[i];
    return t;
}

// A value of a bf16 or f32 operand, and a store rounded to it.
__device__ __forceinline__ float ldv(const uint16_t* p, int64_t i) { return bf2f(p[i]); }
__device__ __forceinline__ float ldv(const float* p, int64_t i) { return p[i]; }
__device__ __forceinline__ void stv(uint16_t* p, int64_t i, float v) { p[i] = f32_to_bf16(v); }
__device__ __forceinline__ void stv(float* p, int64_t i, float v) { p[i] = v; }
__device__ __forceinline__ float rdv(const uint16_t*, float v) { return bfr(v); }
__device__ __forceinline__ float rdv(const float*, float v) { return v; }

// ---------------------------------------------------------------------------- rms_mod
// The gated residual of the branch before, then the RMS norm with a gain and the entry's shift and
// scale, a workgroup a row: libref's rms_mod_row with the residual and the chain rounded to bf16.
// With QUANT the chain feeds quant_act_fp8 as ln_mod_quant_fp8 does (r4d_gen.hip): a 128-column
// group is sixteen consecutive lanes, so its amax is a half-wave reduction, and the codes are
// r4d_q8.h's. `s` is a column slice of the modulation table, [shift | scale] in each entry's row.
constexpr int kRmChunks = 4;   // 8-wide chunks a thread keeps: n <= kT * 8 * 4 = 8192

// QMODE: 0 writes the bf16 chain to `z`, 1 also quantises it to E4M3 (q, scale), 2 to int8, the
// two differing only in r4d_q8.h's `q8_top` and `q8_pack4`. 3 is rms_mod_had_quant_i8g,
// had_quant_act_i8g's quantiser: the bf16-rounded chain rotated by the 128-wide Hadamard where it
// lies, eight columns a lane and sixteen lanes a block (strides 1, 2, 4 in the lane, then 8 to 64
// across lanes, r4d_fwht_wave's order, so the sums are the standalone kernel's), then int8 codes
// under one scale per (row, 128 * SUB): SUB 2 is IQ8R's 256-wide group, a whole wave. `z` receives
// the unrotated chain.
template <bool GATED, int QMODE, int SUB = 1>
__global__ __launch_bounds__(kT) void rms_mod_kernel(
    const uint16_t* __restrict__ yd, const uint16_t* __restrict__ g, const uint16_t* __restrict__ s,
    const uint16_t* __restrict__ w, const int* __restrict__ cu, uint16_t* __restrict__ x,
    uint8_t* __restrict__ q, float* __restrict__ qs, uint16_t* __restrict__ z, int n, int n_seq,
    float eps, int y_ld, int g_ld, int s_ld, int x_ld, int q_ld, int qs_ld, int z_ld, float wadd,
    int once, int layer, int none, const int* __restrict__ colmap) {
    __shared__ float sh[kT / 32];
    const int r = blockIdx.x;
    int e = seq_of(cu, n_seq, r);
    // Optional: `cu` may bound more groups than the modulation table has rows (groups that share
    // a cell share a row), so the map turns a group into its row; without it the group is the row.
    if (colmap) e = colmap[e];
    uint16_t* xr = x + (size_t)r * x_ld;
    float v[kRmChunks][8];
    float ss = 0.0f;
    #pragma unroll
    for (int k = 0; k < kRmChunks; ++k) {
        const int i = ((int)threadIdx.x + k * kT) * 8;
        if (i < n) {
            float f[8];
            ld8(xr + i, f);
            if (GATED) {
                float a[8], b[8];
                ld8(yd + (size_t)r * y_ld + i, a);
                ld8(g + (size_t)e * g_ld + i, b);
                #pragma unroll
                for (int j = 0; j < 8; ++j) f[j] = f[j] + bfr(b[j] * a[j]);
                st8(xr + i, f);
                #pragma unroll
                for (int j = 0; j < 8; ++j) f[j] = bfr(f[j]);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) { v[k][j] = f[j]; ss += f[j] * f[j]; }
        }
    }
    /* A LayerNorm centres first: the mean's block sum, then the centred squares'. */
    float mu = 0.0f;
    if (layer) {
        float sm = 0.0f;
        #pragma unroll
        for (int k = 0; k < kRmChunks; ++k) {
            const int i = ((int)threadIdx.x + k * kT) * 8;
            if (i < n) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) sm += v[k][j];
            }
        }
        mu = block_sum(sm, sh) / (float)n;
        __syncthreads();
        ss = 0.0f;
        #pragma unroll
        for (int k = 0; k < kRmChunks; ++k) {
            const int i = ((int)threadIdx.x + k * kT) * 8;
            if (i < n) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) { v[k][j] -= mu; ss += v[k][j] * v[k][j]; }
            }
        }
    }
    /* `none`: no norm (libref's n2 = x); the flag is the same for the whole workgroup. */
    const float rs = none ? 1.0f : 1.0f / sqrtf(block_sum(ss, sh) / (float)n + eps);
    const uint16_t* sr = s + (size_t)e * s_ld;
    #pragma unroll
    for (int k = 0; k < kRmChunks; ++k) {
        const int i = ((int)threadIdx.x + k * kT) * 8;
        if (i >= n) break;
        float shf[8], scf[8], wf[8], o[8];
        ld8(sr + i, shf);
        ld8(sr + n + i, scf);
        if (w) ld8(w + i, wf);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float n1 = once ? v[k][j] * rs : bfr(v[k][j] * rs);
            const float n2 = w ? bfr(n1 * (wf[j] + wadd)) : bfr(v[k][j] * rs);
            o[j] = bfr(bfr(n2 * bfr(1.0f + scf[j])) + shf[j]);
        }
        if (z) st8(z + (size_t)r * z_ld + i, o);
        if (QMODE) {
            if constexpr (QMODE == 3) {
                #pragma unroll
                for (int h = 1; h < 8; h <<= 1) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        if ((j & h) == 0) {
                            const float a = o[j], b = o[j + h];
                            o[j] = a + b;
                            o[j + h] = a - b;
                        }
                }
                const int lane = (int)threadIdx.x & 31;
                r4d_fwht_xstage<8, 1>(o, lane);
                r4d_fwht_xstage<8, 2>(o, lane);
                r4d_fwht_xstage<8, 4>(o, lane);
                r4d_fwht_xstage<8, 8>(o, lane);
            }
            float m = 0.0f;
            #pragma unroll
            for (int j = 0; j < 8; ++j) m = fmaxf(m, fabsf(o[j]));
            m = fmaxf(m, r4d_xor_lane<8>(m));
            m = fmaxf(m, r4d_xor_lane<4>(m));
            m = fmaxf(m, r4d_xor_lane<2>(m));
            m = fmaxf(m, r4d_xor_lane<1>(m));
            if constexpr (SUB == 2) m = fmaxf(m, r4d_xor_lane<16>(m));
            const float sc = m > 0.0f ? m * (1.0f / q8_top<QMODE >= 2>()) : 1.0f;
            const float inv = 1.0f / sc;
            if ((threadIdx.x & (16 * SUB - 1)) == 0) qs[(size_t)r * qs_ld + i / (128 * SUB)] = sc;
            *(uint2*)(q + (size_t)r * q_ld + i) =
                make_uint2(q8_pack4<QMODE >= 2>(o[0], o[1], o[2], o[3], inv),
                           q8_pack4<QMODE >= 2>(o[4], o[5], o[6], o[7], inv));
        }
    }
}

// ---------------------------------------------------------------------------- flow_x0_step
// One sampler step a lane value: the clean estimate d = x + (1 - t) v, then x' = r x + (1 - r) d
// with r = sigma_next / sigma, every product and sum one f32 rounding.
template <typename TV>
__global__ __launch_bounds__(kT) void flow_x0_kernel(
    const TV* __restrict__ v, const float* __restrict__ t, const float* __restrict__ sg,
    const float* __restrict__ sgn, float* __restrict__ x, int n, int v_ld, int x_ld, int64_t total) {
    const int64_t id = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (id >= total) return;
    const int64_t r = id / n;
    const int c = (int)(id % n);
    const float st = 1.0f - t[0];
    const float ra = sgn[0] / sg[0];
    const float omr = 1.0f - ra;
    float* xp = x + r * x_ld + c;
    const float xv = *xp;
    const float d = xv + st * ldv(v, r * v_ld + c);
    *xp = ra * xv + omr * d;
}

// ---------------------------------------------------------------------------- flow_res_multistep
// The second-order step on the clean estimate d = x + sigma v: x' = r x + hb1 d + hb2 x0, x0 the
// last step's estimate, which d then replaces. With order < 2 it is Euler, x' = r x + (1 - r) d.
template <typename TV>
__global__ __launch_bounds__(kT) void flow_rm_kernel(
    const TV* __restrict__ v, float* __restrict__ x0, const float* __restrict__ sg,
    const float* __restrict__ rr, const float* __restrict__ hb1, const float* __restrict__ hb2,
    const float* __restrict__ ord, float* __restrict__ x, int n, int v_ld, int x_ld, int x0_ld,
    int64_t total) {
    const int64_t id = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (id >= total) return;
    const int64_t r = id / n;
    const int c = (int)(id % n);
    const float sgv = sg[0];
    const float ra = rr[0];
    const float b1 = hb1[0];
    const float b2 = hb2[0];
    const bool two = ord[0] >= 1.5f;
    const float omr = 1.0f - ra;
    float* xp = x + r * x_ld + c;
    float* op = x0 + r * x0_ld + c;
    const float xv = *xp;
    const float d = xv + sgv * ldv(v, r * v_ld + c);
    *xp = two ? ra * xv + b1 * d + b2 * (*op) : ra * xv + omr * d;
    *op = d;
}

// ---------------------------------------------------------------------------- patch_unfold
// The two parameter lists of the latent's statistics, f32 kernel arguments.
constexpr int kMaxChan = 64;
struct ChanList { float v[kMaxChan]; };

// A thread a voxel channel: y[(t, yy, xx), c] = f32(f32(x[patch, (c, sub)] * mul[c]) + add[c]).
__global__ __launch_bounds__(kT) void patch_unfold_kernel(
    const float* __restrict__ x, const int* __restrict__ geom, float* __restrict__ y, ChanList mul,
    ChanList add, int C, int pt, int ph, int pw, int x_ld, int y_ld, int64_t cap, int64_t x_rows) {
    const int64_t id = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int F = geom[0], Hp = geom[1], Wp = geom[2];
    const int64_t Hv = (int64_t)Hp * ph, Wv = (int64_t)Wp * pw;
    const int64_t rows = (int64_t)F * pt * Hv * Wv;
    if (id >= cap || id >= rows * C || (int64_t)F * Hp * Wp > x_rows) return;
    const int64_t r = id / C;
    const int c = (int)(id % C);
    const int64_t tt = r / (Hv * Wv), yy = (r / Wv) % Hv, xx = r % Wv;
    const int64_t patch = ((tt / pt) * Hp + yy / ph) * Wp + xx / pw;
    const int64_t sub = ((tt % pt) * ph + yy % ph) * pw + xx % pw;
    y[r * y_ld + c] = x[patch * x_ld + c * pt * ph * pw + sub] * mul.v[c] + add.v[c];
}

// ---------------------------------------------------------------------------- the tile layout
// diffusers' _split_tiles in latent units (libref's split_tiles), in closed form: the tile count
// is the fewest whose overlaps are all at least `overlap`; the slack tile n - overlap (n - 1) - len
// goes to the overlaps one unit at a time from the first, so overlap i is overlap + q + (i < rr)
// with q, rr the slack's quotient and remainder by n - 1. A tile at least as long as the axis is
// one tile, a layout only when it is exactly as long.
struct Tiles {
    int n = 0, q = 0, rr = 0, tile = 0, overlap = 0;
    __device__ int ov(int i) const { return overlap + q + (i < rr ? 1 : 0); }
    __device__ int start(int i) const { return i * (tile - overlap - q) - min(i, rr); }
};

__device__ __forceinline__ Tiles split_tiles(int len, int tile, int overlap) {
    Tiles t;
    t.tile = tile;
    t.overlap = overlap;
    if (len <= 0 || tile <= 0 || overlap < 0 || overlap >= tile) return t;
    if (tile >= len) {
        t.n = len == tile ? 1 : 0;
        return t;
    }
    int n = (len + tile - 1) / tile;
    while (tile * n - overlap * (n - 1) < len) ++n;
    const int rem = tile * n - overlap * (n - 1) - len;
    t.n = n;
    t.q = rem / (n - 1);
    t.rr = rem % (n - 1);
    return t;
}

// The encoder's split over one axis (libref's split_tiles_axis): the decoder's split above, except
// that an axis no longer than a tile is one tile, short if need be. A lone tile starts at 0 with no
// overlap, which the closed form already gives at n = 1.
__device__ __forceinline__ Tiles split_tiles_axis(int len, int tile, int overlap) {
    if (tile >= len && len > 0 && tile > 0 && overlap >= 0 && overlap < tile) {
        Tiles t;
        t.n = 1;
        t.tile = tile;
        t.overlap = overlap;
        return t;
    }
    return split_tiles(len, tile, overlap);
}

// ---------------------------------------------------------------------------- vtile_gather
// A workgroup a tile row: its voxel from x, a register token from `reg`, or zeros.
template <typename TX, typename TR, typename TY>
__global__ __launch_bounds__(kT) void vtile_gather_kernel(
    const TX* __restrict__ x, const TR* __restrict__ reg, const int* __restrict__ geom,
    TY* __restrict__ y, int n, int tile, int overlap, int frames, int period, int n_reg, int x_ld,
    int reg_ld, int y_ld, int64_t x_rows) {
    const int row = blockIdx.x;
    const int H = geom[1], W = geom[2];
    if (geom[0] != frames || (int64_t)frames * H * W > x_rows) return;
    const Tiles ty = split_tiles(H, tile, overlap), tx = split_tiles(W, tile, overlap);
    if (ty.n == 0 || tx.n == 0 || row >= ty.n * tx.n * period) return;
    const int k = row / period, p = row % period;
    const int vox = frames * tile * tile;
    TY* yr = y + (size_t)row * y_ld;
    if (p < vox) {
        const int t = p / (tile * tile), iy = (p / tile) % tile, ix = p % tile;
        const int64_t src = ((int64_t)t * H + ty.start(k / tx.n) + iy) * W + tx.start(k % tx.n) + ix;
        const TX* xr = x + src * x_ld;
        for (int c = threadIdx.x; c < n; c += kT) stv(yr, c, ldv(xr, c));
    } else if (p < vox + n_reg) {
        const TR* rr = reg + (size_t)(p - vox) * reg_ld;
        for (int c = threadIdx.x; c < n; c += kT) stv(yr, c, ldv(rr, c));
    } else {
        for (int c = threadIdx.x; c < n; c += kT) stv(yr, c, 0.0f);
    }
}

// ---------------------------------------------------------------------------- vtile_stitch
// A thread a (pixel, colour) over every frame the chunk writes, so a pixel's reads of `prev`
// precede its writes of it in one thread: the kept frames, blended with the previous chunk's
// tail unless first, then the tail handed on (and written when last). The blends round to x's
// dtype and the tail is stored at prev's, as libref's ref_vtile_stitch.
struct PixList { float mean[3], stdv[3]; };

template <typename TX, typename TP>
__global__ __launch_bounds__(kT) void vtile_stitch_kernel(
    const TX* __restrict__ x, const int* __restrict__ geom, TP* __restrict__ prev,
    uint8_t* __restrict__ y, PixList pl, int tile, int overlap, int S, int pt, int skip, int keep,
    int blend, int period, int x_ld, int p_ld, int y_ld, int64_t cap, int64_t x_rows,
    int64_t p_rows) {
    const int frames = geom[0], H = geom[1], W = geom[2];
    const bool first = geom[3] != 0, last = geom[4] != 0;
    const int64_t Hp = (int64_t)H * S, Wp = (int64_t)W * S, npx = Hp * Wp;
    const int64_t id = (int64_t)blockIdx.x * kT + threadIdx.x;
    if (id >= npx * 3 || 2 * skip + keep + blend > frames * pt || period < frames * tile * tile)
        return;
    const int64_t fout = keep + (last ? blend : 0);
    if (fout * npx > cap) return;
    const Tiles ty = split_tiles(H, tile, overlap), tx = split_tiles(W, tile, overlap);
    if (ty.n == 0 || tx.n == 0 || (int64_t)ty.n * tx.n * period > x_rows ||
        (int64_t)blend * npx > p_rows)
        return;
    const int c = (int)(id / npx);
    const int64_t px = id % npx;
    const int Y = (int)(px / Wp), X = (int)(px % Wp);
    // The owner tile of the pixel's row and column, and its offset in it.
    int i = 0, j = 0;
    while (i + 1 < ty.n && Y >= ty.start(i + 1) * S) ++i;
    while (j + 1 < tx.n && X >= tx.start(j + 1) * S) ++j;
    const int ly = Y - ty.start(i) * S, lx = X - tx.start(j) * S;
    const int ev = i > 0 ? ty.ov(i - 1) * S : 0, eh = j > 0 ? tx.ov(j - 1) * S : 0;
    auto rd = [&](float v) { return rdv(x, v); };
    auto wb = [&](int p, int e) { return rd((float)p / (float)e); };
    auto wa = [&](int p, int e) { return rd(1.0f - rd((float)p / (float)e)); };
    auto T = [&](int ti, int tj, int ay, int ax, int fc) {
        const int64_t row = (int64_t)(ti * tx.n + tj) * period +
                            ((fc / pt) * tile + ay / S) * tile + ax / S;
        const int col = ((c * pt + fc % pt) * S + ay % S) * S + ax % S;
        return ldv(x, row * x_ld + col);
    };
    auto stitch = [&](int fc) {
        float v = T(i, j, ly, lx, fc);
        if (ly < ev) {
            const float up = T(i - 1, j, ly + tile * S - ev, lx, fc);
            v = rd(rd(up * wa(ly, ev)) + rd(v * wb(ly, ev)));
        }
        if (lx < eh) {
            const float left = T(i, j - 1, ly, lx + tile * S - eh, fc);
            v = rd(rd(left * wa(lx, eh)) + rd(v * wb(lx, eh)));
        }
        return v;
    };
    auto byte = [&](int f, float v) {
        float p = v * pl.stdv[c] + pl.mean[c];
        p = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        p = p * 255.0f;
        float r = p != p ? 0.0f : rintf(p);
        r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);
        y[(f * npx + px) * y_ld + c] = (uint8_t)r;
    };
    for (int k = 0; k < keep; ++k) {
        float v = stitch(skip + k);
        if (!first && k < blend) {
            const float pv = ldv(prev, (k * npx + px) * p_ld + c);
            v = rd(rd(pv * wa(k, blend)) + rd(v * wb(k, blend)));
        }
        byte(k, v);
    }
    for (int k = 0; k < blend; ++k) {
        const float v = stitch(2 * skip + keep + k);
        if (last) byte(keep + k, v);
        stv(prev, (k * npx + px) * p_ld + c, v);
    }
}

// ---------------------------------------------------------------------------- the tiled encoder
// A one-frame media larger than a tile is encoded as vae.py's `tiled_encode` does: the plan, the
// cut and the blend (libref's ref_vtile_plan/crop/join); the encoder is the per-level ops, issued
// per tile.

// A thread a (tile, entry): the tile's first pixel, its pixel extent and the six level volumes,
// as ref_vtile_plan's row of 22 i32.
__global__ __launch_bounds__(kT) void vtile_plan_kernel(
    const int* __restrict__ geom, int* __restrict__ y, int tile, int overlap, int ratio, int y_ld,
    int64_t M, int64_t y_rows) {
    const int64_t id = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int H = geom[1], W = geom[2];
    if (id >= M * 22 || id >= y_rows * 22 || H <= 0 || W <= 0 || H % ratio || W % ratio) return;
    const Tiles ty = split_tiles_axis(H / ratio, tile, overlap);
    const Tiles tx = split_tiles_axis(W / ratio, tile, overlap);
    if (ty.n == 0 || tx.n == 0) return;
    const int64_t t = id / 22;
    const int c = (int)(id % 22);
    const int nh = ty.n, nw = tx.n;
    if (t >= (int64_t)nh * nw) return;
    const int i = (int)(t / nw), j = (int)(t % nw);
    const int th = min(tile, H / ratio - ty.start(i)) * ratio;
    const int tw = min(tile, W / ratio - tx.start(j)) * ratio;
    int v;
    if (c < 4) {
        v = c == 0 ? ty.start(i) * ratio : c == 1 ? tx.start(j) * ratio : c == 2 ? th : tw;
    } else {
        const int L = (c - 4) / 3, k = (c - 4) % 3, d = 1 << min(L, 4);
        v = k == 0 ? 1 : k == 1 ? th / d : tw / d;
    }
    y[t * y_ld + c] = v;
}

// A thread a tile pixel row: its plane row copied. The tile is found by walking the tiles' sizes,
// at most a hundred of them (libref's upper_bound over the same prefix sums).
template <typename T>
__global__ __launch_bounds__(kT) void vtile_crop_kernel(
    const T* __restrict__ x, const int* __restrict__ geom, T* __restrict__ y, int tile,
    int overlap, int ratio, int x_ld, int y_ld, int64_t M, int64_t x_rows, int64_t y_rows) {
    const int64_t r = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int H = geom[1], W = geom[2];
    if (r >= M || r >= y_rows || H <= 0 || W <= 0 || H % ratio || W % ratio) return;
    const Tiles ty = split_tiles_axis(H / ratio, tile, overlap);
    const Tiles tx = split_tiles_axis(W / ratio, tile, overlap);
    if (ty.n == 0 || tx.n == 0 || (int64_t)H * W > x_rows) return;
    const int nh = ty.n, nw = tx.n;
    int t = -1;
    int64_t acc = 0, k = 0;
    for (int tt = 0; tt < nh * nw; ++tt) {
        const int64_t sz = (int64_t)min(tile, H / ratio - ty.start(tt / nw)) * ratio *
                           min(tile, W / ratio - tx.start(tt % nw)) * ratio;
        if (r < acc + sz) { t = tt; k = r - acc; break; }
        acc += sz;
    }
    if (t < 0) return;  // r past the tiles' sum: M was not it
    const int tw = min(tile, W / ratio - tx.start(t % nw)) * ratio;
    const int64_t src = ((int64_t)ty.start(t / nw) * ratio + k / tw) * W +
                        (int64_t)tx.start(t % nw) * ratio + k % tw;
    for (int c = 0; c < 3; ++c) stv(y + r * (int64_t)y_ld, c, ldv(x + src * (int64_t)x_ld, c));
}

// A thread a canvas row: its tile's moments blended with the unblended top, then left neighbour,
// every product and sum rounded to x's dtype (ref_vtile_join). A thread a (row, channel) would
// repeat the tile walk n times.
template <typename T>
__global__ __launch_bounds__(kT) void vtile_join_kernel(
    const T* __restrict__ x, const int* __restrict__ geom, T* __restrict__ y, int tile,
    int overlap, int n, int x_ld, int y_ld, int64_t M, int64_t x_rows, int64_t y_rows) {
    const int64_t r = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int H = geom[1], W = geom[2];
    if (r >= M || r >= y_rows || r >= (int64_t)H * W || H <= 0 || W <= 0) return;
    const Tiles ty = split_tiles_axis(H, tile, overlap), tx = split_tiles_axis(W, tile, overlap);
    if (ty.n == 0 || tx.n == 0) return;
    const int nh = ty.n, nw = tx.n;
    // The tiles' row extents and their prefix sums, in tile-major order.
    const int64_t ext_y = nh == 1 ? H : (int64_t)tile * nh, ext_x = nw == 1 ? W : (int64_t)tile * nw;
    if (ext_y * ext_x > x_rows) return;
    auto sz = [&](int ii, int jj) {
        return (int64_t)min(tile, H - ty.start(ii)) * min(tile, W - tx.start(jj));
    };
    auto off = [&](int ii, int jj) {
        int64_t a = 0;
        for (int t = 0; t < ii * nw + jj; ++t) a += sz(t / nw, t % nw);
        return a;
    };
    const int Y = (int)(r / W), X = (int)(r % W);
    int i = 0;
    while (i + 1 < nh && Y >= ty.start(i + 1)) ++i;
    int j = 0;
    while (j + 1 < nw && X >= tx.start(j + 1)) ++j;
    const int ly = Y - ty.start(i), lx = X - tx.start(j);
    const int ev = i > 0 ? ty.ov(i - 1) : 0, eh = j > 0 ? tx.ov(j - 1) : 0;
    const int64_t own = off(i, j), up = i > 0 ? off(i - 1, j) : 0, left = j > 0 ? off(i, j - 1) : 0;
    // A tile's width: this tile and the one above share a column (`own_w`, short only in the last
    // column); the left neighbour's column is never the last, so it is `tile` wide.
    const int own_w = min(tile, W - tx.start(j));
    const auto fetch = [&](int64_t base, int w, int ay, int ax, int c) {
        return ldv(x, (base + (int64_t)ay * w + ax) * x_ld + c);
    };
    auto rd = [&](float v) { return rdv(x, v); };
    auto wb = [&](int p, int e) { return rd((float)p / (float)e); };
    auto wa = [&](int p, int e) { return rd(1.0f - rd((float)p / (float)e)); };
    for (int c = 0; c < n; ++c) {
        float v = fetch(own, own_w, ly, lx, c);
        if (ly < ev)
            v = rd(rd(fetch(up, own_w, ly + tile - ev, lx, c) * wa(ly, ev)) + rd(v * wb(ly, ev)));
        if (lx < eh)
            v = rd(rd(fetch(left, tile, ly, lx + tile - eh, c) * wa(lx, eh)) + rd(v * wb(lx, eh)));
        stv(y + r * (int64_t)y_ld, c, v);
    }
}

// ---------------------------------------------------------------------------- grid_pos3
// A thread a row: its token's three centred, scaled coordinates, or zeros past the tile's tokens.
__global__ __launch_bounds__(kT) void grid_pos3_kernel(
    float* __restrict__ y, int T, int H, int W, int period, float scale, int M, int rs, int cs) {
    const int m = blockIdx.x * kT + threadIdx.x;
    if (m >= M) return;
    const int k = m % period;
    float p[3] = { 0.0f, 0.0f, 0.0f };
    if (k < T * H * W) {
        const int cc[3] = { k / (H * W), (k / W) % H, k % W }, nn[3] = { T, H, W };
        #pragma unroll
        for (int a = 0; a < 3; ++a) {
            const float qq = ((float)cc[a] + 0.5f) / (float)nn[a];
            p[a] = scale * (2.0f * qq - 1.0f);
        }
    }
    #pragma unroll
    for (int a = 0; a < 3; ++a) y[(size_t)a * rs + (size_t)m * cs] = p[a];
}

// ---------------------------------------------------------------------------- snake_aa
// A thread an output (row, channel) of a sequence: the `k` activations its low-pass filter reads,
// each the upsampled value at that position through SnakeBeta, recomputed per output rather than
// staged (an activation is read by k / ratio outputs). Indices and orders are libref's
// ref_snake_aa: the replicate pad P = k / ratio - 1, the upsampler's crop, the low-pass's left pad
// k / 2 (one less for an even k), each sum in ascending tap.
constexpr int kMaxTaps = 32;

__global__ __launch_bounds__(kT) void snake_aa_kernel(
    const float* __restrict__ x, const float* __restrict__ al, const float* __restrict__ be,
    const float* __restrict__ fu, const float* __restrict__ fd, const int* __restrict__ cu,
    float* __restrict__ y, int n, int R, int k, int lg, int n_seq, int x_ld, int y_ld,
    int64_t cap) {
    __shared__ float sfu[kMaxTaps], sfd[kMaxTaps];
    if ((int)threadIdx.x < k) {
        sfu[threadIdx.x] = fu[threadIdx.x];
        sfd[threadIdx.x] = fd[threadIdx.x];
    }
    __syncthreads();
    const int64_t id = (int64_t)blockIdx.x * kT + threadIdx.x;
    const int r0 = cu[0], r1 = cu[n_seq];
    if (id >= cap || id >= (int64_t)(r1 - r0) * n) return;
    const int row = r0 + (int)(id / n), c = (int)(id % n);
    const int e = seq_of(cu, n_seq, row);
    const int s0 = cu[e], L = cu[e + 1] - s0, o = row - s0;
    const int P = k / R - 1, crop = P * R + (k - R) / 2, pad = k / 2 - (k % 2 == 0 ? 1 : 0);
    const float A = lg ? expf(al[c]) : al[c];
    const float B = lg ? expf(be[c]) : be[c];
    const float ib = 1.0f / (B + 1e-9f);
    const int LR = L * R;
    float acc = 0.0f;
    for (int jt = 0; jt < k; ++jt) {
        const int m = min(max(o * R + jt - pad, 0), LR - 1);
        const int oo = crop + m;
        const int i_lo = oo < k ? 0 : (oo - k) / R + 1;
        const int i_hi = min(oo / R, L + 2 * P - 1);
        float u = 0.0f;
        for (int ii = i_lo; ii <= i_hi; ++ii) {
            const int qrow = min(max(ii - P, 0), L - 1);
            u += sfu[oo - ii * R] * x[(size_t)(s0 + qrow) * x_ld + c];
        }
        u = (float)R * u;
        const float sn = sinf(u * A);
        acc += sfd[jt] * (u + ib * (sn * sn));
    }
    y[(size_t)row * y_ld + c] = acc;
}

inline int st_of(hipError_t e) { return e == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE); }

inline unsigned blocks_for(int64_t n) { return (unsigned)((n + kT - 1) / kT); }

inline bool is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
inline bool is_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }
inline bool is_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }

// A row-major operand's row count and pitch, with the pitch a multiple of eight elements so a row
// is read 16 bytes at a time.
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

// rms_mod (QUANT false: y, g, s, w, cu, x, z) and rms_mod_quant_fp8 (y, g, s, w, cu, x, q, scale,
// z), one shim. qmode 2 is rms_mod_quant_i8 and 3 rms_mod_had_quant_i8g, whose `group` is 128 or
// 256 with n whole groups.
static int rms_mod_launch(const RadArgs* a, RadStream st, int qmode) {
    const RadTensor *y = r4d_opt(a, 0), *g = r4d_opt(a, 1), *sc = r4d_opt(a, 2);
    const RadTensor *w = r4d_opt(a, 3), *cu = r4d_opt(a, 4), *x = r4d_opt(a, 5);
    const RadTensor* q = qmode ? r4d_opt(a, 6) : nullptr;
    const RadTensor* qs = qmode ? r4d_opt(a, 7) : nullptr;
    const RadTensor* z = r4d_opt(a, qmode ? 8 : 6);
    // The optional group->row map, after the last operand of either form (slot 9 quantising,
    // slot 7 plain).
    const RadTensor* cm = r4d_opt(a, qmode ? 9 : 7);
    if (!sc || !cu || !x || (qmode ? (!q || !qs) : !z)) return R4D_NO(RAD_E_INVAL);
    const bool gated = r4d_geti_or(a, "gated", 0) != 0;
    if (gated && (!y || !g)) return R4D_NO(RAD_E_INVAL);
    if (!is_bf16(x) || !is_bf16(sc) || (w && !is_bf16(w)) || !is_i32(cu) ||
        (gated && (!is_bf16(y) || !is_bf16(g))) || (z && !is_bf16(z)) ||
        (qmode && (q->dtype != (qmode >= 2 ? RAD_I8 : RAD_F8E4M3) || qs->dtype != RAD_F32)))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    const long long grp = qmode ? r4d_geti_or(a, "group", 0) : 128;
    if (n <= 0 || (n & 7) || n > (long long)kT * 8 * kRmChunks ||
        (qmode && (n % grp || (grp != 128 && !(qmode == 3 && grp == 256)))))
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, xl = 0, ry = 0, yl = 0, rg = 0, gl = 0, rs = 0, sl = 0, rz = 0, zl = 0;
    long long rq = 0, rqs = 0, ql = 0, qsl = 0;
    if (!rows8(x, n, &rx, &xl) || !rows8(sc, 2 * n, &rs, &sl) || !r4d_contig(cu) ||
        (w && (!r4d_contig(w) || ((uintptr_t)w->data & 15))) ||
        (gated && (!rows8(y, n, &ry, &yl) || !rows8(g, n, &rg, &gl))) ||
        (z && !rows8(z, n, &rz, &zl)))
        return R4D_NO(RAD_E_STRIDE);
    if (qmode) {
        ql = r4d_rows_pitch(q, n, &rq);
        qsl = r4d_rows_pitch(qs, n / grp, &rqs);
        if (ql < 0 || (ql & 7) || ((uintptr_t)q->data & 7) || qsl < 0) return R4D_NO(RAD_E_STRIDE);
    }
    const long long n_seq = r4d_numel(cu) - 1, rows = qmode ? rq : rz;
    if (cm && (!is_i32(cm) || !r4d_contig(cm) || r4d_numel(cm) < n_seq))
        return R4D_NO(RAD_E_SHAPE);
    // With the map the table may hold fewer rows than groups; the issuer keeps every mapped row
    // inside it.
    if (rows <= 0 || n_seq <= 0 || rx < rows || (!cm && rs < n_seq) || (w && r4d_numel(w) < n) ||
        (qmode && rqs < rows) || (gated && (ry < rows || (!cm && rg < n_seq))) || (z && rz < rows))
        return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);
    const int once = (int)r4d_geti_or(a, "norm_once", 0);
    const char* nk = r4d_gets(a, "norm");
    const int layer = nk && !std::strcmp(nk, "layer");
    const int none = nk && !std::strcmp(nk, "none");
    if ((nk && !layer && !none && std::strcmp(nk, "rms")) || (none && w)) return R4D_NO(RAD_E_INVAL);
#define R4D_RMQ(G, Q, ...)                                                                        \
    hipLaunchKernelGGL((rms_mod_kernel<G, Q, ##__VA_ARGS__>), dim3((unsigned)rows), dim3(kT), 0,  \
                       (hipStream_t)st, G ? (const uint16_t*)y->data : nullptr,                   \
                       G ? (const uint16_t*)g->data : nullptr, (const uint16_t*)sc->data,         \
                       w ? (const uint16_t*)w->data : nullptr, (const int*)cu->data,              \
                       (uint16_t*)x->data,                                                        \
                       Q ? (uint8_t*)q->data : nullptr, Q ? (float*)qs->data : nullptr,           \
                       z ? (uint16_t*)z->data : nullptr, (int)n, (int)n_seq, eps, (int)yl,        \
                       (int)gl, (int)sl, (int)xl, (int)ql, (int)qsl, (int)zl, wadd, once, layer, \
                       none, cm ? (const int*)cm->data : nullptr)
    if (qmode == 3 && grp == 256) {
        if (gated) R4D_RMQ(true, 3, 2);
        else       R4D_RMQ(false, 3, 2);
    } else if (qmode == 3) {
        if (gated) R4D_RMQ(true, 3);
        else       R4D_RMQ(false, 3);
    } else if (qmode == 2) {
        if (gated) R4D_RMQ(true, 2);
        else       R4D_RMQ(false, 2);
    } else if (qmode == 1) {
        if (gated) R4D_RMQ(true, 1);
        else       R4D_RMQ(false, 1);
    } else {
        if (gated) R4D_RMQ(true, 0);
        else       R4D_RMQ(false, 0);
    }
#undef R4D_RMQ
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_rms_mod(const RadArgs* a, RadStream s) { return rms_mod_launch(a, s, 0); }

extern "C" int r4d_rad_rms_mod_quant_i8(const RadArgs* a, RadStream s) {
    return rms_mod_launch(a, s, 2);
}

extern "C" int r4d_rad_rms_mod_had_quant_i8g(const RadArgs* a, RadStream s) {
    return rms_mod_launch(a, s, 3);
}

extern "C" int r4d_rad_flow_x0_step(const RadArgs* a, RadStream s) {
    const RadTensor *v = r4d_opt(a, 0), *t = r4d_opt(a, 1), *sg = r4d_opt(a, 2);
    const RadTensor *sgn = r4d_opt(a, 3), *x = r4d_opt(a, 4);
    if (!v || !t || !sg || !sgn || !x) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(x) || !is_f32(t) || !is_f32(sg) || !is_f32(sgn) || (!is_f32(v) && !is_bf16(v)))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rx = 0, rv = 0;
    const long long xl = n > 0 ? r4d_rows_pitch(x, n, &rx) : -1;
    const long long vl = n > 0 ? r4d_rows_pitch(v, n, &rv) : -1;
    if (xl < 0 || vl < 0) return R4D_NO(RAD_E_STRIDE);
    if (rx <= 0 || rv < rx || r4d_numel(t) < 1 || r4d_numel(sg) < 1 || r4d_numel(sgn) < 1)
        return R4D_NO(RAD_E_SHAPE);
    const int64_t total = (int64_t)rx * n;
#define R4D_FX0(TV)                                                                               \
    hipLaunchKernelGGL(flow_x0_kernel<TV>, dim3(blocks_for(total)), dim3(kT), 0, (hipStream_t)s,  \
                       (const TV*)v->data, (const float*)t->data, (const float*)sg->data,         \
                       (const float*)sgn->data, (float*)x->data, (int)n, (int)vl, (int)xl, total)
    if (is_f32(v)) R4D_FX0(float);
    else           R4D_FX0(uint16_t);
#undef R4D_FX0
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_flow_res_multistep(const RadArgs* a, RadStream s) {
    const RadTensor *v = r4d_opt(a, 0), *x0 = r4d_opt(a, 1), *sg = r4d_opt(a, 2);
    const RadTensor *rr = r4d_opt(a, 3), *hb1 = r4d_opt(a, 4), *hb2 = r4d_opt(a, 5);
    const RadTensor *ord = r4d_opt(a, 6), *x = r4d_opt(a, 7);
    if (!v || !x0 || !sg || !rr || !hb1 || !hb2 || !ord || !x) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(x) || !is_f32(x0) || !is_f32(sg) || !is_f32(rr) || !is_f32(hb1) || !is_f32(hb2) ||
        !is_f32(ord) || (!is_f32(v) && !is_bf16(v)))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    long long rx = 0, rv = 0, rx0 = 0;
    const long long xl = n > 0 ? r4d_rows_pitch(x, n, &rx) : -1;
    const long long vl = n > 0 ? r4d_rows_pitch(v, n, &rv) : -1;
    const long long x0l = n > 0 ? r4d_rows_pitch(x0, n, &rx0) : -1;
    if (xl < 0 || vl < 0 || x0l < 0) return R4D_NO(RAD_E_STRIDE);
    if (rx <= 0 || rv < rx || rx0 < rx || r4d_numel(sg) < 1 || r4d_numel(rr) < 1 ||
        r4d_numel(hb1) < 1 || r4d_numel(hb2) < 1 || r4d_numel(ord) < 1)
        return R4D_NO(RAD_E_SHAPE);
    const int64_t total = (int64_t)rx * n;
#define R4D_FRM(TV)                                                                               \
    hipLaunchKernelGGL(flow_rm_kernel<TV>, dim3(blocks_for(total)), dim3(kT), 0, (hipStream_t)s,  \
                       (const TV*)v->data, (float*)x0->data, (const float*)sg->data,              \
                       (const float*)rr->data, (const float*)hb1->data, (const float*)hb2->data,  \
                       (const float*)ord->data, (float*)x->data, (int)n, (int)vl, (int)xl,        \
                       (int)x0l, total)
    if (is_f32(v)) R4D_FRM(float);
    else           R4D_FRM(uint16_t);
#undef R4D_FRM
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_patch_unfold(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *geom = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !geom || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(x) || !is_f32(y) || !is_i32(geom)) return R4D_NO(RAD_E_DTYPE);
    const long long C = r4d_geti_or(a, "C", 0), pt = r4d_geti_or(a, "pt", 0);
    const long long ph = r4d_geti_or(a, "ph", 0), pw = r4d_geti_or(a, "pw", 0);
    ChanList mul{}, add{};
    if (C <= 0 || C > kMaxChan || pt <= 0 || ph <= 0 || pw <= 0 ||
        parse_f32_list(r4d_gets(a, "mul"), mul.v, kMaxChan) != C ||
        parse_f32_list(r4d_gets(a, "add"), add.v, kMaxChan) != C || r4d_numel(geom) < 3)
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, C * pt * ph * pw, &rx), yl = r4d_rows_pitch(y, C, &ry);
    if (xl < 0 || yl < 0 || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx <= 0) return R4D_NO(RAD_E_SHAPE);
    const int64_t cap = (int64_t)ry * C;
    hipLaunchKernelGGL(patch_unfold_kernel, dim3(blocks_for(cap)), dim3(kT), 0, (hipStream_t)s,
                       (const float*)x->data, (const int*)geom->data, (float*)y->data, mul, add,
                       (int)C, (int)pt, (int)ph, (int)pw, (int)xl, (int)yl, cap, (int64_t)rx);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_vtile_gather(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *reg = r4d_opt(a, 1), *geom = r4d_opt(a, 2);
    const RadTensor* y = r4d_opt(a, 3);
    if (!x || !reg || !geom || !y) return R4D_NO(RAD_E_INVAL);
    if ((!is_f32(x) && !is_bf16(x)) || (!is_f32(reg) && !is_bf16(reg)) ||
        (!is_f32(y) && !is_bf16(y)) || !is_i32(geom))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), tile = r4d_geti_or(a, "tile", 0);
    const long long overlap = r4d_geti_or(a, "overlap", 0), frames = r4d_geti_or(a, "frames", 0);
    const long long period = r4d_geti_or(a, "period", 0), n_reg = r4d_geti_or(a, "n_reg", 0);
    if (n <= 0 || tile <= 0 || frames <= 0 || n_reg < 0 || overlap < 0 || overlap >= tile ||
        period < frames * tile * tile + n_reg || r4d_numel(geom) < 3)
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, rr = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, n, &rx), rl = r4d_rows_pitch(reg, n, &rr);
    const long long yl = r4d_rows_pitch(y, n, &ry);
    if (xl < 0 || rl < 0 || yl < 0 || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rr < n_reg) return R4D_NO(RAD_E_SHAPE);
#define R4D_VTG(TX, TR, TY)                                                                       \
    hipLaunchKernelGGL((vtile_gather_kernel<TX, TR, TY>), dim3((unsigned)ry), dim3(kT), 0,        \
                       (hipStream_t)s, (const TX*)x->data, (const TR*)reg->data,                  \
                       (const int*)geom->data, (TY*)y->data, (int)n, (int)tile, (int)overlap,     \
                       (int)frames, (int)period, (int)n_reg, (int)xl, (int)rl, (int)yl,     \
                       (int64_t)rx)
#define R4D_VTG_Y(TX, TR)                                                                         \
    do { if (is_f32(y)) R4D_VTG(TX, TR, float); else R4D_VTG(TX, TR, uint16_t); } while (0)
#define R4D_VTG_R(TX)                                                                             \
    do { if (is_f32(reg)) R4D_VTG_Y(TX, float); else R4D_VTG_Y(TX, uint16_t); } while (0)
    if (is_f32(x)) R4D_VTG_R(float);
    else           R4D_VTG_R(uint16_t);
#undef R4D_VTG_R
#undef R4D_VTG_Y
#undef R4D_VTG
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_vtile_stitch(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *geom = r4d_opt(a, 1), *prev = r4d_opt(a, 2);
    const RadTensor* y = r4d_opt(a, 3);
    if (!x || !geom || !prev || !y) return R4D_NO(RAD_E_INVAL);
    if ((!is_f32(x) && !is_bf16(x)) || (!is_f32(prev) && !is_bf16(prev)) || y->dtype != RAD_U8 ||
        !is_i32(geom))
        return R4D_NO(RAD_E_DTYPE);
    const long long tile = r4d_geti_or(a, "tile", 0), overlap = r4d_geti_or(a, "overlap", 0);
    const long long S = r4d_geti_or(a, "scale", 0), pt = r4d_geti_or(a, "pt", 0);
    const long long skip = r4d_geti_or(a, "skip", 0), keep = r4d_geti_or(a, "keep", 0);
    const long long blend = r4d_geti_or(a, "blend", 0), period = r4d_geti_or(a, "period", 0);
    PixList pl{};
    if (tile <= 0 || overlap < 0 || overlap >= tile || S <= 0 || pt <= 0 || skip < 0 ||
        keep <= 0 || blend < 0 || blend > keep || period <= 0 ||
        parse_f32_list(r4d_gets(a, "mean"), pl.mean, 3) != 3 ||
        parse_f32_list(r4d_gets(a, "std"), pl.stdv, 3) != 3 || r4d_numel(geom) < 5)
        return R4D_NO(RAD_E_SHAPE);
    const long long cols = 3 * pt * S * S;
    long long rx = 0, rp = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, cols, &rx), pl_ld = r4d_rows_pitch(prev, 3, &rp);
    const long long yl = r4d_rows_pitch(y, 3, &ry);
    if (xl < 0 || pl_ld < 0 || yl < 0 || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || rx < period) return R4D_NO(RAD_E_SHAPE);
    // A thread a (pixel, colour) of the largest frame the output holds; the kernel bounds the rest
    // by the device geometry.
    const int64_t npx_cap = ry / keep, total = npx_cap * 3;
#define R4D_VTS(TX, TP)                                                                           \
    hipLaunchKernelGGL((vtile_stitch_kernel<TX, TP>), dim3(blocks_for(total)), dim3(kT), 0,       \
                       (hipStream_t)s, (const TX*)x->data, (const int*)geom->data,                \
                       (TP*)prev->data, (uint8_t*)y->data, pl, (int)tile, (int)overlap, (int)S,   \
                       (int)pt, (int)skip, (int)keep, (int)blend, (int)period, (int)xl,           \
                       (int)pl_ld, (int)yl, (int64_t)ry, (int64_t)rx, (int64_t)rp)
#define R4D_VTS_P(TX)                                                                             \
    do { if (is_f32(prev)) R4D_VTS(TX, float); else R4D_VTS(TX, uint16_t); } while (0)
    if (is_f32(x)) R4D_VTS_P(float);
    else           R4D_VTS_P(uint16_t);
#undef R4D_VTS_P
#undef R4D_VTS
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_vtile_plan(const RadArgs* a, RadStream s) {
    const RadTensor *geom = r4d_opt(a, 0), *y = r4d_opt(a, 1);
    if (!geom || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_i32(geom) || !is_i32(y)) return R4D_NO(RAD_E_DTYPE);
    const long long M = r4d_geti_or(a, "M", 0), tile = r4d_geti_or(a, "tile", 0);
    const long long overlap = r4d_geti_or(a, "overlap", 0), ratio = r4d_geti_or(a, "ratio", 0);
    if (M <= 0 || tile <= 0 || overlap < 0 || overlap >= tile || ratio <= 0 ||
        r4d_numel(geom) < 3)
        return R4D_NO(RAD_E_SHAPE);
    /* The tile count comes from y; the ranged `M` is the band's bound, a ceiling only
     * (ref_common.h, band_rows). */
    long long ry = 0;
    const long long yl = r4d_rows_pitch(y, 22, &ry);
    if (yl < 0 || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || (M > 0 && ry > M)) return R4D_NO(RAD_E_SHAPE);
    const int64_t total = ry * 22;
    hipLaunchKernelGGL(vtile_plan_kernel, dim3(blocks_for(total)), dim3(kT), 0, (hipStream_t)s,
                       (const int*)geom->data, (int*)y->data, (int)tile, (int)overlap, (int)ratio,
                       (int)yl, ry, ry);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_vtile_crop(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *geom = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !geom || !y) return R4D_NO(RAD_E_INVAL);
    if ((!is_f32(x) && !is_bf16(x)) || x->dtype != y->dtype || !is_i32(geom))
        return R4D_NO(RAD_E_DTYPE);
    const long long M = r4d_geti_or(a, "M", 0), tile = r4d_geti_or(a, "tile", 0);
    const long long overlap = r4d_geti_or(a, "overlap", 0), ratio = r4d_geti_or(a, "ratio", 0);
    if (M <= 0 || tile <= 0 || overlap < 0 || overlap >= tile || ratio <= 0 ||
        r4d_numel(geom) < 3)
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0;
    /* y's rows are the tiles' pixel rows; the ranged `M` is a ceiling only (band_rows). Rows past
     * the tiles' sum write nothing. */
    const long long xl = r4d_rows_pitch(x, 3, &rx), yl = r4d_rows_pitch(y, 3, &ry);
    if (xl < 0 || yl < 0 || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || (M > 0 && ry > M)) return R4D_NO(RAD_E_SHAPE);
#define R4D_VTC(T)                                                                                \
    hipLaunchKernelGGL((vtile_crop_kernel<T>), dim3(blocks_for(ry)), dim3(kT), 0, (hipStream_t)s, \
                       (const T*)x->data, (const int*)geom->data, (T*)y->data, (int)tile,         \
                       (int)overlap, (int)ratio, (int)xl, (int)yl, ry, rx, ry)
    if (is_f32(x)) R4D_VTC(float);
    else           R4D_VTC(uint16_t);
#undef R4D_VTC
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_vtile_join(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *geom = r4d_opt(a, 1), *y = r4d_opt(a, 2);
    if (!x || !geom || !y) return R4D_NO(RAD_E_INVAL);
    if ((!is_f32(x) && !is_bf16(x)) || x->dtype != y->dtype || !is_i32(geom))
        return R4D_NO(RAD_E_DTYPE);
    const long long M = r4d_geti_or(a, "M", 0), tile = r4d_geti_or(a, "tile", 0);
    const long long overlap = r4d_geti_or(a, "overlap", 0), n = r4d_geti_or(a, "n", 0);
    if (M <= 0 || tile <= 0 || overlap < 0 || overlap >= tile || n <= 0 || r4d_numel(geom) < 3)
        return R4D_NO(RAD_E_SHAPE);
    long long rx = 0, ry = 0;
    /* A thread a row of the canvas y; the ranged `M` is a ceiling only (band_rows). */
    const long long xl = r4d_rows_pitch(x, n, &rx), yl = r4d_rows_pitch(y, n, &ry);
    if (xl < 0 || yl < 0 || !r4d_contig(geom)) return R4D_NO(RAD_E_STRIDE);
    if (ry <= 0 || (M > 0 && ry > M)) return R4D_NO(RAD_E_SHAPE);
#define R4D_VTJ(T)                                                                                \
    hipLaunchKernelGGL((vtile_join_kernel<T>), dim3(blocks_for(ry)), dim3(kT), 0, (hipStream_t)s, \
                       (const T*)x->data, (const int*)geom->data, (T*)y->data, (int)tile,         \
                       (int)overlap, (int)n, (int)xl, (int)yl, ry, rx, ry)
    if (is_f32(x)) R4D_VTJ(float);
    else           R4D_VTJ(uint16_t);
#undef R4D_VTJ
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_grid_pos3(const RadArgs* a, RadStream s) {
    const RadTensor* y = r4d_opt(a, 0);
    if (!y) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(y)) return R4D_NO(RAD_E_DTYPE);
    const long long T = r4d_geti_or(a, "T", 0), H = r4d_geti_or(a, "H", 0);
    const long long W = r4d_geti_or(a, "W", 0), period = r4d_geti_or(a, "period", 0);
    if (T <= 0 || H <= 0 || W <= 0 || period < T * H * W || y->rank != 2 || y->shape[0] < 3)
        return R4D_NO(RAD_E_SHAPE);
    const long long M = y->shape[1];
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL(grid_pos3_kernel, dim3(blocks_for(M)), dim3(kT), 0, (hipStream_t)s,
                       (float*)y->data, (int)T, (int)H, (int)W, (int)period,
                       (float)r4d_getf(a, "scale", 1.0), (int)M, (int)y->stride[0],
                       (int)y->stride[1]);
    return st_of(hipGetLastError());
}

extern "C" int r4d_rad_snake_aa(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *al = r4d_opt(a, 1), *be = r4d_opt(a, 2);
    const RadTensor *fu = r4d_opt(a, 3), *fd = r4d_opt(a, 4), *cu = r4d_opt(a, 5);
    const RadTensor* y = r4d_opt(a, 6);
    if (!x || !al || !be || !fu || !fd || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (!is_f32(x) || !is_f32(y) || !is_f32(al) || !is_f32(be) || !is_f32(fu) || !is_f32(fd) ||
        !is_i32(cu))
        return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0), R = r4d_geti_or(a, "ratio", 0);
    const long long k = r4d_geti_or(a, "k", 0), lg = r4d_geti_or(a, "log", 0);
    if (n <= 0 || R <= 0 || k < R || k > kMaxTaps || lg < 0 || lg > 1 || r4d_numel(al) < n ||
        r4d_numel(be) < n || r4d_numel(fu) < k || r4d_numel(fd) < k)
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(al) || !r4d_contig(be) || !r4d_contig(fu) || !r4d_contig(fd) || !r4d_contig(cu))
        return R4D_NO(RAD_E_STRIDE);
    long long rx = 0, ry = 0;
    const long long xl = r4d_rows_pitch(x, n, &rx), yl = r4d_rows_pitch(y, n, &ry);
    if (xl < 0 || yl < 0) return R4D_NO(RAD_E_STRIDE);
    const long long n_seq = r4d_numel(cu) - 1;
    if (n_seq <= 0 || ry <= 0 || rx < ry) return R4D_NO(RAD_E_SHAPE);
    const int64_t cap = (int64_t)ry * n;
    hipLaunchKernelGGL(snake_aa_kernel, dim3(blocks_for(cap)), dim3(kT), 0, (hipStream_t)s,
                       (const float*)x->data, (const float*)al->data, (const float*)be->data,
                       (const float*)fu->data, (const float*)fd->data, (const int*)cu->data,
                       (float*)y->data, (int)n, (int)R, (int)k, (int)lg, (int)n_seq, (int)xl,
                       (int)yl, cap);
    return st_of(hipGetLastError());
}
