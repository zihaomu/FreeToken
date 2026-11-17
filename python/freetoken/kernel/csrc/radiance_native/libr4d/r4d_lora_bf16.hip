/* r4d_lora_bf16.hip -- runtime LoRA's two halves, lora_shrink and lora_expand, per-row slot.
 *
 *     lora_shrink   t[m, j]  = sum_k x(m, k) * A[s, j, k]                    s = slot[m]
 *     lora_expand   y[m, n]  = y[m, n] + scale[s, n] * sum_j t[m, j] * B[s, n, j]
 *
 * The core issues the pair right after a targeted trunk GEMM, in the GEMM's own row slice, with
 * `a`/`a_scale` the GEMM's activation operands and `y` its output (docs/OPS.md). A row whose slot
 * is outside [0, S) is a base row and both kernels leave it before touching anything: shrink
 * writes no `t`, expand neither reads nor writes `y` (`y + 0` would turn -0 into +0), so a base
 * row stays bit for bit what the GEMM wrote.
 *
 * Invariants every row here keeps:
 *   - No split that moves with M: a row's arithmetic does not depend on the batch it arrives in,
 *     so a request alone and the same request in a mixed batch get the same bytes.
 *   - The grid is M and the declared parameters. Which adapters are active is read on the device
 *     from `slot` and the pools, so a recorded pass replays the same packets whatever they hold.
 *   - x(m, k) is gemm_nt_q's activation value: a code times its group's scale (one f32 product),
 *     or a bf16/f16/f32 element as it stands, the scale stream's grid read as libref's make_scales
 *     reads it. The decodes below are the ABI's conversions (rad_plugin.h) restated for the device
 *     bit for bit; gfx1201 has no bf16 convert, so the narrowing is the ABI's RTNE with its
 *     quiet-NaN rule. Every multiply-add rounds twice (-ffp-contract=off, RadAddPlugin.cmake).
 *
 * The scalar rows (lora_shrink_scalar, lora_expand_scalar): shrink is one block a row, one wave a
 * column j, lane l summing the k that are l mod 32 in ascending order, then a fixed xor butterfly;
 * expand is one thread an (m, n), j ascending, so its bytes are libref's. No scratch and no LDS. They
 * serve every dtype, and are the only rows for f16 and f32 activations.
 *
 * The tiled rows (lora_shrink_wmma, lora_shrink_wmma_q, lora_expand_tiled), preferred:
 *   - The grid's unit is a row tile (16 rows shrink, 32 expand). Every wave reads the tile's slots
 *     and a ballot decides; a tile of base rows leaves before any barrier or operand load. A tile
 *     holding rows of several adapters runs one pass per distinct slot, ascending, each with its
 *     own slot's pools; a row's value comes from its own slot's pass alone. The passes may be
 *     spread over blockIdx.z when the grid is small, which decides which block runs a pass, never
 *     its arithmetic.
 *   - Shrink is WMMA bf16 (16 x 16 x 16) with the pool fragment as the A operand and x as B, so a
 *     lane's accumulator is one row. Its order is fixed by K and R alone: K in 32-wide steps, a
 *     block's steps split over its 4 waves (wave w takes every 4th), the waves summed 0..3 in LDS,
 *     and for a long K the steps split over KS blocks whose f32 partials the last to arrive sums
 *     0..KS-1 (sh_plan). fp8 and i8 codes are exact in bf16 and feed the WMMA as codes; each step's
 *     partial is scaled by its group's f32 scale, which needs the group to be whole steps (a
 *     multiple of 32, or one group across K). Every rounding is a sum or product of the contract's
 *     terms, so the result is inside the scalar row's (K + 2) 2^-24 sum|terms| bound.
 *   - Expand is the scalar arithmetic in a tile: B[s]'s [128 columns, 32 j] slab is staged in LDS
 *     as f32 (exact) and reused by the tile's 32 rows; each output keeps the scalar sequence (acc =
 *     0, j ascending acc = acc + t*B, d = scale*acc, one rounding into y), so its bytes are
 *     libref's.
 *
 * Kernels are written against r4d_launch.h so tests/r4d_emu runs this source on the host.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#include "r4d_keyed.h"
#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_lora {

/* What `a` holds; a template argument so the decode is chosen once per launch, not per element. */
enum { kBF16 = 0, kF16 = 1, kF32 = 2, kE4M3 = 3, kE5M2 = 4, kI8 = 5 };

struct ShrinkArgs {
    const void*     a;    long long a_rs, a_cs;        /* element strides of `a` */
    const float*    as;   long long s_rs, s_cs;        /* null: no scale stream */
    const int32_t*  slot; long long slot_st;
    const uint16_t* la;   long long la_s0, la_s1, la_s2;
    float*          t;    long long t_rs, t_cs;
    int             K, group, R, S;                  /* K < 2^31: 32-bit k and k / group */
};

struct ExpandArgs {
    const float*    t;    long long t_rs, t_cs;
    const int32_t*  slot; long long slot_st;
    const uint16_t* lb;   long long lb_s0, lb_s1, lb_s2;
    const float*    sc;   long long sc_rs, sc_cs;
    void*           y;    long long y_rs, y_cs;
    long long       N, ntiles;
    int             R, S;
};

/* ------------------------------------------------------- the ABI's conversions, on the device */
__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((unsigned)h << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {           /* rad_f32_to_bf16 */
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ inline float f16_to_f32(uint16_t h) {             /* rad_f16_to_f32 */
    const unsigned s = (unsigned)(h >> 15) << 31;
    const unsigned e = ((unsigned)h >> 10) & 0x1fu;
    unsigned m = (unsigned)h & 0x3ffu, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            u = s | ((unsigned)(127 - 15 - sh + 1) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    return __uint_as_float(u);
}

__device__ inline uint16_t f32_to_f16(float f) {             /* rad_f32_to_f16 */
    const unsigned u = __float_as_uint(f);
    const unsigned s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return (uint16_t)(s | 0x7c00u | (a > 0x7f800000u ? ((a >> 13) & 0x3ffu) | 0x200u : 0u));
    if (a >= 0x47800000u) return (uint16_t)(s | 0x7c00u);
    if (a < 0x33000000u) return (uint16_t)s;
    const int e = (int)(a >> 23) - 127;
    unsigned m = a & 0x7fffffu;
    if (e < -14) {
        const int sh = -14 - e + 13;
        m |= 0x800000u;
        const unsigned lo = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        unsigned q = m >> sh;
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint16_t)(s | q);
    }
    unsigned q = m >> 13, he = (unsigned)(e + 15);
    const unsigned lo = m & 0x1fffu;
    if (lo > 0x1000u || (lo == 0x1000u && (q & 1u))) ++q;
    if (q == 0x400u) { q = 0; ++he; }
    if (he >= 0x1fu) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | (he << 10) | q);
}

__device__ inline float e4m3_to_f32(uint8_t b) {             /* rad_fp8e4m3_to_f32 */
    const unsigned s = (unsigned)(b >> 7) << 31;
    const unsigned e = ((unsigned)b >> 3) & 0xfu;
    unsigned m = (unsigned)b & 0x7u, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((unsigned)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

template <int ADT>
__device__ __forceinline__ float ld_act(const void* p, long long i) {
    if (ADT == kBF16) return bf2f(((const uint16_t*)p)[i]);
    if (ADT == kF16)  return f16_to_f32(((const uint16_t*)p)[i]);
    if (ADT == kF32)  return ((const float*)p)[i];
    if (ADT == kE4M3) return e4m3_to_f32(((const uint8_t*)p)[i]);
    if (ADT == kE5M2) return f16_to_f32((uint16_t)((unsigned)((const uint8_t*)p)[i] << 8));
    return (float)((const int8_t*)p)[i];
}

/* The sum of a wave's 32 values, the same on every lane and in a fixed order: an xor butterfly over
 * ds_bpermute. Each step adds the same two values on both partners (a + b == b + a), so the lanes
 * never drift apart and no lane's answer depends on the schedule. */
__device__ __forceinline__ float wave_sum(float v, int lane) {
    for (int off = 16; off >= 1; off >>= 1) {
        const int o = __builtin_amdgcn_ds_bpermute((lane ^ off) << 2, __float_as_int(v));
        v = v + __int_as_float(o);
    }
    return v;
}

/* ---------------------------------------------------------------- the kernels */

/* One block a row, eight waves; wave w owns columns j = w, w + 8, ... The x row is re-read per
 * column from cache rather than staged: K reaches 17408, past what LDS holds as f32. */
template <int ADT>
__global__ __launch_bounds__(256) void lora_shrink_kernel(ShrinkArgs A) {
    const long long m = blockIdx.x;
    const int s = A.slot[m * A.slot_st];
    if (s < 0 || s >= A.S) return;                   /* a base row: no t written */
    const int lane = (int)(threadIdx.x & 31u), wave = (int)(threadIdx.x >> 5);
    const long long x0 = m * A.a_rs, s0 = m * A.s_rs;
    for (int j = wave; j < A.R; j += 8) {
        const uint16_t* arow = A.la + (long long)s * A.la_s0 + (long long)j * A.la_s1;
        float acc = 0.0f;
        for (int k = lane; k < A.K; k += 32) {
            float x = ld_act<ADT>(A.a, x0 + (long long)k * A.a_cs);
            if (A.as) x = x * A.as[s0 + (long long)(k / A.group) * A.s_cs];
            acc += x * bf2f(arow[(long long)k * A.la_s2]);
        }
        acc = wave_sum(acc, lane);
        if (lane == 0) A.t[m * A.t_rs + (long long)j * A.t_cs] = acc;
    }
}

/* One thread an output element over a flat grid of M row-strips of 256 columns. */
template <int YDT>
__global__ __launch_bounds__(256) void lora_expand_kernel(ExpandArgs A) {
    const long long b = blockIdx.x;
    const long long m = b / A.ntiles;
    const long long n = (b - m * A.ntiles) * 256 + threadIdx.x;
    if (n >= A.N) return;
    const int s = A.slot[m * A.slot_st];
    if (s < 0 || s >= A.S) return;                   /* a base row: y not read, not written */
    const float* trow = A.t + m * A.t_rs;
    const uint16_t* brow = A.lb + (long long)s * A.lb_s0 + n * A.lb_s1;
    float acc = 0.0f;
    for (int j = 0; j < A.R; ++j)
        acc += trow[(long long)j * A.t_cs] * bf2f(brow[(long long)j * A.lb_s2]);
    const float d = A.sc[(long long)s * A.sc_rs + n * A.sc_cs] * acc;
    const long long o = m * A.y_rs + n * A.y_cs;
    if (YDT == kBF16) {
        uint16_t* y = (uint16_t*)A.y + o;
        *y = f2bf(bf2f(*y) + d);
    } else if (YDT == kF16) {
        uint16_t* y = (uint16_t*)A.y + o;
        *y = f32_to_f16(f16_to_f32(*y) + d);
    } else {
        float* y = (float*)A.y + o;
        *y = *y + d;
    }
}

/* ================================================================ the tiled rows */

typedef short    lr_v8s __attribute__((ext_vector_type(8)));    /* 8 bf16, as the WMMA takes them */
typedef float    lr_v8f __attribute__((ext_vector_type(8)));
typedef unsigned lr_v4u __attribute__((ext_vector_type(4)));    /* 16 codes */

enum {
    kShMT = 16,            /* shrink: rows a tile -- the WMMA's N side */
    kShJT = 16,            /* shrink: columns j a tile -- the WMMA's M side */
    kShStep = 32,          /* k a step: two WMMAs, 16 contiguous k a lane */
    kShWaves = 4,
    kShThreads = kShWaves * 32,
    kShU = 4,              /* steps a wave keeps in flight */
    kShSplitMin = 16,      /* no K split leaves a block fewer than 16 steps (512 k) */
    kShFill = 24,          /* the blocks one row tile is spread over, j tiles x K splits */
    kExMT = 32,            /* expand: rows a tile, four a wave */
    kExNT = 128,           /* expand: columns a tile, four a lane */
    kExJC = 32,            /* expand: j a staged slab */
    kExThreads = 256,
    kZMax = 16,            /* the passes of one tile spread over at most this many blocks */
    kZGrid = 256,          /* ... and only while the grid is smaller than this */
    kMmMT = 128,           /* expand_mm: rows a tile, 32 a wave */
    kMmNT = 256,           /* expand_mm: columns a tile, 128 a wave */
    kMmJC = 32,            /* expand_mm: j a staged step */
    kMmWaves = 8,          /* 4 (rows) x 2 (columns) over the tile */
    kMmThreads = kMmWaves * 32,
};

struct ShrinkTArgs {
    ShrinkArgs b;
    float*     part;       /* [ks][M][R] split partials, the op's scratch; null when ks == 1 */
    unsigned*  ctr;        /* arrival counters, one a (row tile, j tile, z); null when ks == 1 */
    int        M, jt, ks, spb, nsteps, Z;
};

struct ExpandTArgs {
    ExpandArgs b;
    int        M, Z;
};

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

/* The tile's distinct adapter slots, ascending; -1 entries are base rows. One thread, sixteen or
 * thirty-two values: an insertion sort is the whole cost. */
__device__ inline int distinct_slots(const int* s, int* out, int n) {
    int k = 0;
    for (int i = 0; i < n; ++i) {
        const int v = s[i];
        if (v < 0) continue;
        int p = 0;
        while (p < k && out[p] < v) ++p;
        if (p < k && out[p] == v) continue;
        for (int q = k; q > p; --q) out[q] = out[q - 1];
        out[p] = v;
        ++k;
    }
    return k;
}

/* A code as the bf16 that holds it exactly: E4M3 (3 mantissa bits, exponents 2^-9..2^8), E5M2 (2,
 * 2^-16..2^15) and i8 (|v| <= 128, eight significant bits) all fit bf16's 8-bit significand and
 * exponent. Bit for bit the upper half of the ABI's f32 decode above (e4m3_to_f32, f16_to_f32 of
 * the E5M2 byte, the i8 value), NaN payloads included; tests/r4d_emu walks every code. */
template <int ADT>
__device__ __forceinline__ uint16_t code_bf16(unsigned b) {
    if (ADT == kE4M3) {
        const unsigned s = (b & 0x80u) << 8, e = (b >> 3) & 0xfu, m = b & 7u;
        if (e == 0u) return (uint16_t)(s | (__float_as_uint((float)m * 0.001953125f) >> 16));
        if (e == 0xfu && m == 7u) return (uint16_t)(s | 0x7fc0u);
        return (uint16_t)(s | ((e + 120u) << 7) | (m << 4));
    }
    if (ADT == kE5M2) {
        const unsigned s = (b & 0x80u) << 8, e = (b >> 2) & 0x1fu, m = b & 3u;
        if (e == 0u) return (uint16_t)(s | (__float_as_uint((float)m * 1.52587890625e-05f) >> 16));
        if (e == 0x1fu) return (uint16_t)(s | 0x7f80u | (m << 5));
        return (uint16_t)(s | ((e + 112u) << 7) | (m << 5));
    }
    return (uint16_t)(__float_as_uint((float)(int8_t)(uint8_t)b) >> 16);
}

/* Sixteen bf16 of a pool row from k on, as two WMMA fragments; zeros past K and for a lane that is
 * off (j >= R). Zeros and not whatever follows: the next row's values times a zero x could be a
 * NaN times zero. V: the row is dense and 16-byte aligned, so two 16-byte loads. */
template <bool V>
__device__ __forceinline__ void ld_pool16(const uint16_t* p, int k, int K, long long cs, bool on,
                                          lr_v8s* f) {
    f[0] = (lr_v8s)(short)0;
    f[1] = (lr_v8s)(short)0;
    if (!on) return;
    if (V && k + 16 <= K) {
        f[0] = *(const lr_v8s*)(p + k);
        f[1] = *(const lr_v8s*)(p + k + 8);
        return;
    }
    for (int e = 0; e < 16; ++e)
        if (k + e < K) f[e >> 3][e & 7] = (short)p[(long long)(k + e) * cs];
}

/* Sixteen activation elements of a row from k on, as bf16 fragments: bf16 as it is, a code as the
 * bf16 that holds it. Off (the row is another slot's, or past M): zeros, and nothing is read. */
template <int ADT, bool V>
__device__ __forceinline__ void ld_act16(const void* a, long long row, int k, int K, long long cs,
                                         bool on, lr_v8s* f) {
    f[0] = (lr_v8s)(short)0;
    f[1] = (lr_v8s)(short)0;
    if (!on) return;
    if (ADT == kBF16) {
        ld_pool16<V>((const uint16_t*)a + row, k, K, cs, true, f);
        return;
    }
    const uint8_t* p = (const uint8_t*)a + row;
    if (V && k + 16 <= K) {
        const lr_v4u q = *(const lr_v4u*)(p + k);
        for (int e = 0; e < 16; ++e)
            f[e >> 3][e & 7] = (short)code_bf16<ADT>((q[e >> 2] >> (8 * (e & 3))) & 0xffu);
        return;
    }
    for (int e = 0; e < 16; ++e)
        if (k + e < K) f[e >> 3][e & 7] = (short)code_bf16<ADT>(p[(long long)(k + e) * cs]);
}

__device__ __forceinline__ lr_v8f wmma_bf16(lr_v8s a, lr_v8s b, lr_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

/* Shrink, one block a (j tile x K split, row tile, z), 4 waves. The pool fragment is the WMMA's A
 * operand (16 j x 16 k) and the x fragment its B (16 k x 16 rows), so lane l's eight outputs are
 * row l % 16 at j = 8 (l / 16) .. +7: one row, one group scale a lane. A lane holds 16 contiguous
 * k of its row in both operands, the first 8 in the step's first WMMA and the next 8 in its
 * second; any assignment of k to WMMA positions is a sum of the same products, and this one is
 * fixed. Per element the order is, all of it a function of K, R and the group alone:
 *
 *   wave w:  acc = 0; for its steps i ascending (i = w, w + 4, ... of the block's run)
 *              bf16:  acc = wmma(A1, X1, wmma(A0, X0, acc))
 *              coded: acc = acc + scale(m, group of i) * wmma(A1, X1, wmma(A0, X0, 0))
 *   block:   v = ((w0 + w1) + w2) + w3
 *   t:       v of the single block, or ((p0 + p1) + ...) + p[ks-1] over the K splits
 *
 * x of a row that is not in the pass is a zero fragment and its column of the WMMA is never
 * stored; a WMMA's output column depends on that column of B alone, so it cannot reach the rows
 * that are stored. */
template <int ADT, bool VX, bool VA>
__global__ __launch_bounds__(kShThreads) void lora_shrink_wmma_kernel(ShrinkTArgs T) {
    __shared__ float red[kShWaves][kShJT * kShMT];      /* [wave][j][m] */
    __shared__ int sh_slot[kShMT], sh_list[kShMT];
    __shared__ int sh_n;
    __shared__ unsigned sh_arr;
    const ShrinkArgs& A = T.b;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int jt = (int)blockIdx.x / T.ks, ks = (int)blockIdx.x - jt * T.ks;
    const int m0 = (int)blockIdx.y * kShMT, z = (int)blockIdx.z;

    /* The tile's slots, read by every wave, so a tile of base rows leaves here: no barrier, no
     * operand load. */
    int my = -1;
    if (lane < kShMT && m0 + lane < T.M) {
        const int s = A.slot[(long long)(m0 + lane) * A.slot_st];
        if (s >= 0 && s < A.S) my = s;
    }
    if (__builtin_amdgcn_ballot_w32(my >= 0) == 0u) return;
    if (wave == 0 && lane < kShMT) sh_slot[lane] = my;
    __syncthreads();
    if (tid == 0) sh_n = distinct_slots(sh_slot, sh_list, kShMT);
    __syncthreads();
    const int npass = sh_n;
    if (z >= npass) return;

    const int st0 = ks * T.spb, st1 = imin(st0 + T.spb, T.nsteps);
    const int j = jt * kShJT + col;                    /* this lane's pool row */
    const bool aon = j < A.R;
    const long long xrow = (long long)(m0 + col) * A.a_rs;
    const long long srow = (long long)(m0 + col) * A.s_rs;
    for (int p = z; p < npass; p += T.Z) {
        const int s = sh_list[p];
        const bool xon = sh_slot[col] == s;            /* this lane's row is in the pass */
        const uint16_t* ap = A.la + (long long)s * A.la_s0 + (long long)j * A.la_s1;
        lr_v8f acc = (lr_v8f)(0.0f);
        for (int st = st0 + wave; st < st1; st += kShWaves * kShU) {
            lr_v8s af[kShU][2], xf[kShU][2];
            float sc[kShU];
#pragma unroll
            for (int u = 0; u < kShU; ++u) {
                const int su = st + u * kShWaves;
                const bool in = su < st1;
                const int k = su * kShStep + half * 16;
                ld_pool16<VA>(ap, k, A.K, A.la_s2, aon && in, af[u]);
                ld_act16<ADT, VX>(A.a, xrow, k, A.K, A.a_cs, xon && in, xf[u]);
                sc[u] = 0.0f;
                if (ADT != kBF16 && xon && in)
                    sc[u] = A.as[srow + (long long)((su * kShStep) / A.group) * A.s_cs];
            }
#pragma unroll
            for (int u = 0; u < kShU; ++u) {
                if (st + u * kShWaves >= st1) break;   /* wave-uniform */
                if (ADT == kBF16) {
                    acc = wmma_bf16(af[u][0], xf[u][0], acc);
                    acc = wmma_bf16(af[u][1], xf[u][1], acc);
                } else {
                    lr_v8f d = wmma_bf16(af[u][0], xf[u][0], (lr_v8f)(0.0f));
                    d = wmma_bf16(af[u][1], xf[u][1], d);
                    acc = acc + d * sc[u];
                }
            }
        }
        for (int e = 0; e < 8; ++e) red[wave][(half * 8 + e) * kShMT + col] = acc[e];
        __syncthreads();
        for (int i = tid; i < kShJT * kShMT; i += kShThreads) {
            const int jj = i / kShMT, mm = i - jj * kShMT, jg = jt * kShJT + jj;
            float v = red[0][i];
            for (int w = 1; w < kShWaves; ++w) v = v + red[w][i];
            if (sh_slot[mm] != s || jg >= A.R) continue;
            if (T.ks == 1)
                A.t[(long long)(m0 + mm) * A.t_rs + (long long)jg * A.t_cs] = v;
            else
                T.part[((long long)ks * T.M + m0 + mm) * A.R + jg] = v;
        }
        __syncthreads();
    }
    if (T.ks == 1) return;

    /* The K split's sum, in the last block to arrive. Every thread's partial stores are fenced
     * before the arrival, and the counter is left zero for the next launch (launches are ordered:
     * the lora rows are not concurrent, and each lane's `t` keys its own counter block). */
    __threadfence();
    __syncthreads();
    const int cidx = ((int)blockIdx.y * T.jt + jt) * kZMax + z;
    if (tid == 0)
        sh_arr = __hip_atomic_fetch_add(&T.ctr[cidx], 1u, __ATOMIC_ACQ_REL,
                                        __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (sh_arr != (unsigned)(T.ks - 1)) return;
    if (tid == 0) T.ctr[cidx] = 0u;
    __threadfence();
    for (int i = tid; i < kShJT * kShMT; i += kShThreads) {
        const int jj = i / kShMT, mm = i - jj * kShMT, jg = jt * kShJT + jj;
        const int s = sh_slot[mm];
        if (s < 0 || jg >= A.R) continue;
        int r = 0;
        while (sh_list[r] != s) ++r;
        if (r % T.Z != z) continue;                    /* another block's pass */
        const long long o = (long long)(m0 + mm) * A.R + jg, pl = (long long)T.M * A.R;
        float v = T.part[o];
        for (int q = 1; q < T.ks; ++q) v = v + T.part[(long long)q * pl + o];
        A.t[(long long)(m0 + mm) * A.t_rs + (long long)jg * A.t_cs] = v;
    }
}

/* Expand, one block a (128 columns, 32 rows, z), 8 waves: wave w owns rows 4w..4w+3 and lane l
 * columns 4l..4l+3, sixteen outputs a thread. B[s]'s slab [128 columns][32 j] is staged as f32
 * (a bf16 widens exactly) and t's [32 rows][32 j] beside it; every output runs the scalar row's
 * sequence over the slab and the next, so the bytes are libref's. A pass's accumulators start at
 * zero and only its own rows are written back: a row of another slot accumulates against zeros it
 * never stores. V: B is dense, 16-byte aligned and R a multiple of 8, so 16-byte loads. */
template <int YDT, bool V>
__global__ __launch_bounds__(kExThreads) void lora_expand_tiled_kernel(ExpandTArgs T) {
    __shared__ float bs[kExJC][kExNT];
    __shared__ float ts[kExJC][kExMT];
    __shared__ int ex_slot[kExMT], ex_list[kExMT];
    __shared__ int ex_n;
    const ExpandArgs& A = T.b;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const long long n0 = (long long)blockIdx.x * kExNT;
    const int m0 = (int)blockIdx.y * kExMT, z = (int)blockIdx.z;

    int my = -1;
    if (m0 + lane < T.M) {
        const int s = A.slot[(long long)(m0 + lane) * A.slot_st];
        if (s >= 0 && s < A.S) my = s;
    }
    if (__builtin_amdgcn_ballot_w32(my >= 0) == 0u) return;     /* a tile of base rows */
    if (wave == 0) ex_slot[lane] = my;
    __syncthreads();
    if (tid == 0) ex_n = distinct_slots(ex_slot, ex_list, kExMT);
    __syncthreads();
    const int npass = ex_n;
    if (z >= npass) return;

    for (int p = z; p < npass; p += T.Z) {
        const int s = ex_list[p];
        const uint16_t* bp = A.lb + (long long)s * A.lb_s0;
        /* A wave none of whose four rows is in the pass stages with the others and computes
         * nothing: at decode that is seven of the eight. Wave-uniform, outside every barrier. */
        const bool live = ex_slot[4 * wave] == s || ex_slot[4 * wave + 1] == s ||
                          ex_slot[4 * wave + 2] == s || ex_slot[4 * wave + 3] == s;
        float acc[4][4];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;
        for (int j0 = 0; j0 < A.R; j0 += kExJC) {
            const int jc = imin(kExJC, A.R - j0);
            __syncthreads();                           /* the last slab's readers are done */
            if (V) {
                const int nv = jc >> 3;                /* four lanes cover a column's 32 j */
                for (int v = tid; v < kExNT * nv; v += kExThreads) {
                    const int nl = v / nv, jv = v - nl * nv;
                    const long long n = n0 + nl;
                    lr_v8s q = (lr_v8s)(short)0;
                    if (n < A.N) q = *(const lr_v8s*)(bp + n * A.lb_s1 + j0 + jv * 8);
                    for (int e = 0; e < 8; ++e) bs[jv * 8 + e][nl] = bf2f((uint16_t)q[e]);
                }
            } else {
                for (int v = tid; v < kExNT * jc; v += kExThreads) {
                    const int jj = v / kExNT, nl = v - jj * kExNT;
                    const long long n = n0 + nl;
                    bs[jj][nl] = n < A.N ? bf2f(bp[n * A.lb_s1 + (long long)(j0 + jj) * A.lb_s2])
                                         : 0.0f;
                }
            }
            for (int v = tid; v < kExMT * jc; v += kExThreads) {
                const int jj = v / kExMT, r = v - jj * kExMT;
                ts[jj][r] = ex_slot[r] == s
                                ? A.t[(long long)(m0 + r) * A.t_rs + (long long)(j0 + jj) * A.t_cs]
                                : 0.0f;
            }
            __syncthreads();
            if (!live) continue;
            /* Four j a round, their LDS reads issued together: one round's wait instead of four.
             * The adds stay in j order for every output. */
            int jj = 0;
            for (; jj + 4 <= jc; jj += 4) {
                float tv[4][4], bv[4][4];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    for (int r = 0; r < 4; ++r) tv[q][r] = ts[jj + q][4 * wave + r];
                    for (int c = 0; c < 4; ++c) bv[q][c] = bs[jj + q][4 * lane + c];
                }
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c)
                            acc[r][c] = acc[r][c] + tv[q][r] * bv[q][c];
            }
            for (; jj < jc; ++jj) {
                float tv[4], bv[4];
                for (int r = 0; r < 4; ++r) tv[r] = ts[jj][4 * wave + r];
                for (int c = 0; c < 4; ++c) bv[c] = bs[jj][4 * lane + c];
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) acc[r][c] = acc[r][c] + tv[r] * bv[c];
            }
        }
        for (int r = 0; r < 4; ++r) {
            const int row = 4 * wave + r;
            if (ex_slot[row] != s) continue;           /* another pass's row, or a base row */
            const long long m = m0 + row;
            for (int c = 0; c < 4; ++c) {
                const long long n = n0 + 4 * lane + c;
                if (n >= A.N) continue;
                const float d = A.sc[(long long)s * A.sc_rs + n * A.sc_cs] * acc[r][c];
                const long long o = m * A.y_rs + n * A.y_cs;
                if (YDT == kBF16) {
                    uint16_t* y = (uint16_t*)A.y + o;
                    *y = f2bf(bf2f(*y) + d);
                } else if (YDT == kF16) {
                    uint16_t* y = (uint16_t*)A.y + o;
                    *y = f32_to_f16(f16_to_f32(*y) + d);
                } else {
                    float* y = (float*)A.y + o;
                    *y = *y + d;
                }
            }
        }
    }
}

/* lora_expand_mm: the same delta on the WMMA, t rounded to bf16, for the passes whose rows are
 * many and mostly of one slot (an image or video generator's). One block a (256 columns, 128 rows),
 * eight waves, wave w the 32 rows at 32 (w >> 1) and the 128 columns at 128 (w & 1): 2 x 8 WMMA
 * tiles. Each 32-wide j step stages t's [128 rows][32 j] (rounded to bf16) and B[s]'s [256 columns]
 * [32 j] in LDS, read by every wave; B's fragment is the WMMA's A operand (16 n x 16 j) and t's its
 * B (16 j x 16 m), so lane l's eight outputs are row l % 16 of its m tile at columns 8 (l / 16) ..
 * +7 -- eight neighbours in y. Rows of a slot not in the pass are staged as zeros and not stored. */
template <int YDT, bool V>
__global__ __launch_bounds__(kMmThreads) void lora_expand_mm_kernel(ExpandTArgs T) {
    __shared__ uint16_t ts[kMmMT][kMmJC + 8];          /* +8: rows land on different banks */
    __shared__ uint16_t bs[kMmNT][kMmJC + 8];
    __shared__ int mm_slot[kMmMT], mm_list[kMmMT];
    __shared__ int mm_n;
    const ExpandArgs& A = T.b;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4;
    const long long nb = (long long)blockIdx.x * kMmNT;
    const int mb = (int)blockIdx.y * kMmMT;
    const int wm = 32 * (wave >> 1), wn = 128 * (wave & 1);

    for (int i = tid; i < kMmMT; i += kMmThreads) {
        int s = -1;
        if (mb + i < T.M) {
            s = A.slot[(long long)(mb + i) * A.slot_st];
            if (s < 0 || s >= A.S) s = -1;
        }
        mm_slot[i] = s;
    }
    __syncthreads();
    if (tid == 0) mm_n = distinct_slots(mm_slot, mm_list, kMmMT);
    __syncthreads();
    const int npass = mm_n;

    for (int p = 0; p < npass; ++p) {
        const int s = mm_list[p];
        const uint16_t* bp = A.lb + (long long)s * A.lb_s0;
        lr_v8f acc[2][8];
        for (int b = 0; b < 2; ++b)
            for (int a = 0; a < 8; ++a) acc[b][a] = (lr_v8f)(0.0f);
        for (int j0 = 0; j0 < A.R; j0 += kMmJC) {
            __syncthreads();                           /* the last step's readers are done */
            /* t: 128 rows x 32 j, one row's 32 a pair of lanes */
            for (int i = tid; i < kMmMT * 2; i += kMmThreads) {
                const int r = i >> 1, h = (i & 1) * 16;
                const bool on = mm_slot[r] == s;
                const long long row = (long long)(mb + r) * A.t_rs;
                for (int e = 0; e < 16; ++e) {
                    const int j = j0 + h + e;
                    ts[r][h + e] = on && j < A.R ? f2bf(A.t[row + (long long)j * A.t_cs]) : (uint16_t)0;
                }
            }
            /* B[s]: 256 columns x 32 j, one column's 32 a pair of lanes */
            for (int i = tid; i < kMmNT * 2; i += kMmThreads) {
                const int c = i >> 1, h = (i & 1) * 16;
                const long long n = nb + c;
                lr_v8s f[2];
                ld_pool16<V>(bp + (n < A.N ? n : 0) * A.lb_s1, j0 + h, A.R, A.lb_s2, n < A.N, f);
                for (int e = 0; e < 16; ++e) bs[c][h + e] = (uint16_t)f[e >> 3][e & 7];
            }
            __syncthreads();
            lr_v8s tf[2][2];
            for (int b = 0; b < 2; ++b)
                for (int q = 0; q < 2; ++q)
                    for (int e = 0; e < 8; ++e)
                        tf[b][q][e] = (short)ts[wm + 16 * b + col][16 * half + 8 * q + e];
            for (int a = 0; a < 8; ++a) {
                lr_v8s bf[2];
                for (int q = 0; q < 2; ++q)
                    for (int e = 0; e < 8; ++e)
                        bf[q][e] = (short)bs[wn + 16 * a + col][16 * half + 8 * q + e];
                for (int b = 0; b < 2; ++b) {
                    acc[b][a] = wmma_bf16(bf[0], tf[b][0], acc[b][a]);
                    acc[b][a] = wmma_bf16(bf[1], tf[b][1], acc[b][a]);
                }
            }
        }
        for (int b = 0; b < 2; ++b) {
            const int ml = wm + 16 * b + col;
            if (mm_slot[ml] != s) continue;
            const long long m = mb + ml;
            for (int a = 0; a < 8; ++a) {
                const long long n0 = nb + wn + 16 * a + 8 * half;
                const long long o = m * A.y_rs + n0 * A.y_cs;
                const float* scp = A.sc + (long long)s * A.sc_rs;
                if (YDT == kBF16 && A.y_cs == 1 && n0 + 8 <= A.N && (((uintptr_t)A.y + 2 * o) & 15) == 0) {
                    lr_v8s* yp = (lr_v8s*)((uint16_t*)A.y + o);
                    lr_v8s yv = *yp;
                    for (int e = 0; e < 8; ++e)
                        yv[e] = (short)f2bf(bf2f((uint16_t)yv[e]) + scp[(n0 + e) * A.sc_cs] * acc[b][a][e]);
                    *yp = yv;
                    continue;
                }
                for (int e = 0; e < 8; ++e) {
                    const long long n = n0 + e;
                    if (n >= A.N) continue;
                    const float d = scp[n * A.sc_cs] * acc[b][a][e];
                    const long long oe = m * A.y_rs + n * A.y_cs;
                    if (YDT == kBF16) {
                        uint16_t* y = (uint16_t*)A.y + oe;
                        *y = f2bf(bf2f(*y) + d);
                    } else if (YDT == kF16) {
                        uint16_t* y = (uint16_t*)A.y + oe;
                        *y = f32_to_f16(f16_to_f32(*y) + d);
                    } else {
                        float* y = (float*)A.y + oe;
                        *y = *y + d;
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------- operand reading (host) */

/* `t` read as [rows, cols]: the trailing axes that span `cols` must be one dense run, the leading
 * ones fold to rows, and the row stride is the axis just outside the run -- libref/ref_gemm.h's
 * row_stride, so a rank-3 [tokens, heads, width] activation and a column slice of a fused plane are
 * read the way the oracle reads them. False when no single (rows, row stride, column stride)
 * describes the tensor. */
static bool as_rows(const RadTensor* t, int64_t cols, int64_t* rows, int64_t* rs, int64_t* cs) {
    if (!t || t->rank < 1 || cols <= 0) return false;
    const int r = (int)t->rank;
    int i = r - 1;
    int64_t span = t->shape[i];
    while (span < cols && i > 0) {
        if (t->stride[i - 1] != t->stride[i] * t->shape[i]) return false;
        --i;
        span *= t->shape[i];
    }
    if (span != cols) return false;
    int64_t n = 1;
    for (int d = 0; d < i; ++d) n *= t->shape[d];
    for (int d = 0; d + 1 < i; ++d)
        if (t->shape[d + 1] > 1 && t->stride[d] != t->stride[d + 1] * t->shape[d + 1]) return false;
    *rows = n;
    *rs = i > 0 ? t->stride[i - 1] : cols;
    *cs = t->stride[r - 1];
    return true;
}

/* The per-row slot array: [M], or [M, 1] -- its length and element stride. */
static bool as_vec(const RadTensor* t, int64_t* n, int64_t* st) {
    if (!t) return false;
    if (t->rank == 1) { *n = t->shape[0]; *st = t->stride[0]; return true; }
    int64_t cs = 0;
    return as_rows(t, 1, n, st, &cs);
}

/* A pool [S, rows, cols]: its three strides; anything but rank 3 is read dense. */
static void pool3(const RadTensor* t, int64_t rows, int64_t cols, long long* s0, long long* s1,
                  long long* s2) {
    if (t->rank == 3) { *s0 = t->stride[0]; *s1 = t->stride[1]; *s2 = t->stride[2]; return; }
    *s0 = rows * cols; *s1 = cols; *s2 = 1;
}

/* libref/ref_gemm.h's grid_block: the power of two whose ceiling division gives `blocks`. */
static int64_t grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

/* The dtype parameter is the tensor's own name (rad_dtype_name); a row resolved for one dtype and
 * handed another would compute the wrong decode, so a disagreement is refused, not followed. */
static bool dtype_agrees(const RadArgs* a, uint32_t dt) {
    const char* d = rad_args_gets(a, "dtype");
    return !d || rad_dtype_parse(d) == dt;
}

/* The scalar rows' launches, one per dtype instantiation: M blocks for shrink and M strips of 256
 * columns for expand, 256 threads each, whatever the slots hold. */
template <int ADT>
static void launch_shrink(const ShrinkArgs& P, int64_t blocks, RadStream stream) {
    (void)stream;                                 /* the host emulator's launch takes none */
    R4D_KLAUNCH(R4D_KNAME(lora_shrink_kernel<ADT>), dim3((unsigned)blocks), dim3(256),
                (hipStream_t)stream, P);
}
template <int YDT>
static void launch_expand(const ExpandArgs& P, int64_t blocks, RadStream stream) {
    (void)stream;
    R4D_KLAUNCH(R4D_KNAME(lora_expand_kernel<YDT>), dim3((unsigned)blocks), dim3(256),
                (hipStream_t)stream, P);
}

/* lora_shrink's operands, read and checked once for every row: on RAD_OK, `P` and `adt` describe
 * the launch over *M > 0 rows. Zero rows is refused, as every op refuses it (docs/OPS.md). */
static int shrink_operands(const RadArgs* a, ShrinkArgs* out, int* adt_out, int64_t* M_out) {
    *M_out = 0;
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *X = rad_arg_in(a, 0), *XS = rad_arg_in(a, 1), *SL = rad_arg_in(a, 2);
    const RadTensor *LA = rad_arg_in(a, 3), *T = rad_arg_in(a, 4);
    if (!X || !SL || !LA || !T) return RAD_E_INVAL;
    if (XS && XS->dtype == RAD_DT_INVALID) XS = nullptr;     /* absent, as libref */
    const int64_t K = rad_args_geti_or(a, "K", 0), R = rad_args_geti_or(a, "R", 0);
    const int64_t S = rad_args_geti_or(a, "S", 0);
    if (K <= 0 || K > 0x7fffffff || R <= 0 || S <= 0 || R > (1 << 16) || S > (1 << 20))
        return RAD_E_SHAPE;
    int adt;
    switch (X->dtype) {
        case RAD_BF16:   adt = kBF16; break;
        case RAD_F16:    adt = kF16;  break;
        case RAD_F32:    adt = kF32;  break;
        case RAD_F8E4M3: adt = kE4M3; break;
        case RAD_F8E5M2: adt = kE5M2; break;
        case RAD_I8:     adt = kI8;   break;
        default:         return RAD_E_DTYPE;
    }
    if (!dtype_agrees(a, X->dtype)) return RAD_E_DTYPE;
    if (adt >= kE4M3 && !XS) return RAD_E_INVAL;     /* a code without its scale has no value */
    if (LA->dtype != RAD_BF16 || T->dtype != RAD_F32 || SL->dtype != RAD_I32) return RAD_E_DTYPE;

    int64_t M = 0, t_rs = 0, t_cs = 0, xr = 0, x_rs = 0, x_cs = 0, sr = 0, sl_st = 0;
    if (!as_rows(T, R, &M, &t_rs, &t_cs)) return RAD_E_SHAPE;     /* the step's rows: t's view */
    if (M <= 0) return RAD_E_SHAPE;                               /* zero rows is not a call */
    const int64_t band = rad_args_geti_or(a, "M", 0);             /* libref's band_rows rule */
    if (band > 0 && M > band) return RAD_E_SHAPE;
    if (!as_rows(X, K, &xr, &x_rs, &x_cs) || xr < M) return RAD_E_SHAPE;
    if (!as_vec(SL, &sr, &sl_st) || sr < M) return RAD_E_SHAPE;
    if (rad_tensor_numel(LA) < S * R * K) return RAD_E_SHAPE;
    if (M > 0x7fffffff) return RAD_E_SHAPE;

    ShrinkArgs P{};
    P.a = X->data; P.a_rs = x_rs; P.a_cs = x_cs;
    if (XS) {
        /* gemm_nt_q's reading of the stream: [rows, groups] with its own strides, or a flat
         * [rows * groups]; the group is what the stream's extent implies. A grid with fewer rows
         * than the activation is a weight's block grid, never an activation's. */
        if (XS->dtype != RAD_F32) return RAD_E_DTYPE;
        int64_t groups = 0;
        if (XS->rank >= 2) {
            if (XS->shape[XS->rank - 2] < M) return RAD_E_SHAPE;
            groups = XS->shape[XS->rank - 1];
            P.s_rs = XS->stride[XS->rank - 2];
            P.s_cs = XS->stride[XS->rank - 1];
        } else {
            const int64_t nel = rad_tensor_numel(XS);
            if (nel < M) return RAD_E_SHAPE;
            groups = nel / M;
            P.s_rs = groups;
            P.s_cs = 1;
        }
        if (groups <= 0) return RAD_E_SHAPE;
        const int64_t group = grid_block(K, groups);
        const int64_t gp = rad_args_geti_or(a, "group", 0);         /* the grid, as libref */
        if (gp > 0 && (K + gp - 1) / gp != (K + group - 1) / group) return RAD_E_SHAPE;
        P.group = (int)group;
        P.as = (const float*)XS->data;
    }
    P.slot = (const int32_t*)SL->data; P.slot_st = sl_st;
    P.la = (const uint16_t*)LA->data;
    pool3(LA, R, K, &P.la_s0, &P.la_s1, &P.la_s2);
    P.t = (float*)T->data; P.t_rs = t_rs; P.t_cs = t_cs;
    P.K = (int)K; P.R = (int)R; P.S = (int)S;
    if (!P.as) P.group = (int)K;
    *out = P;
    *adt_out = adt;
    *M_out = M;
    return RAD_OK;
}

/* lora_expand's, the same way. */
static int expand_operands(const RadArgs* a, ExpandArgs* out, int* ydt_out, int64_t* M_out) {
    *M_out = 0;
    if (!a || a->n_t < 5 || !rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *T = rad_arg_in(a, 0), *SL = rad_arg_in(a, 1), *LB = rad_arg_in(a, 2);
    const RadTensor *SC = rad_arg_in(a, 3), *Y = rad_arg_in(a, 4);
    const int64_t N = rad_args_geti_or(a, "N", 0), R = rad_args_geti_or(a, "R", 0);
    const int64_t S = rad_args_geti_or(a, "S", 0);
    if (N <= 0 || R <= 0 || S <= 0 || R > (1 << 16) || S > (1 << 20)) return RAD_E_SHAPE;
    int ydt;
    switch (Y->dtype) {
        case RAD_BF16: ydt = kBF16; break;
        case RAD_F16:  ydt = kF16;  break;
        case RAD_F32:  ydt = kF32;  break;
        default:       return RAD_E_DTYPE;
    }
    if (!dtype_agrees(a, Y->dtype)) return RAD_E_DTYPE;
    if (T->dtype != RAD_F32 || SC->dtype != RAD_F32 || LB->dtype != RAD_BF16 ||
        SL->dtype != RAD_I32)
        return RAD_E_DTYPE;

    int64_t M = 0, y_rs = 0, y_cs = 0, tr = 0, t_rs = 0, t_cs = 0, sr = 0, sl_st = 0;
    if (!as_rows(Y, N, &M, &y_rs, &y_cs)) return RAD_E_SHAPE;     /* the step's rows: y's view */
    if (M <= 0) return RAD_E_SHAPE;                               /* zero rows is not a call */
    const int64_t band = rad_args_geti_or(a, "M", 0);             /* libref's band_rows rule */
    if (band > 0 && M > band) return RAD_E_SHAPE;
    if (!as_rows(T, R, &tr, &t_rs, &t_cs) || tr < M) return RAD_E_SHAPE;
    if (!as_vec(SL, &sr, &sl_st) || sr < M) return RAD_E_SHAPE;
    if (rad_tensor_numel(LB) < S * N * R || rad_tensor_numel(SC) < S * N) return RAD_E_SHAPE;
    if (M > 0x7fffffff) return RAD_E_SHAPE;

    ExpandArgs P{};
    P.t = (const float*)T->data; P.t_rs = t_rs; P.t_cs = t_cs;
    P.slot = (const int32_t*)SL->data; P.slot_st = sl_st;
    P.lb = (const uint16_t*)LB->data;
    pool3(LB, N, R, &P.lb_s0, &P.lb_s1, &P.lb_s2);
    P.sc = (const float*)SC->data;
    P.sc_rs = SC->rank >= 2 ? SC->stride[SC->rank - 2] : N;
    P.sc_cs = SC->rank >= 2 ? SC->stride[SC->rank - 1] : 1;
    P.y = Y->data; P.y_rs = y_rs; P.y_cs = y_cs;
    P.N = N; P.ntiles = (N + 255) / 256; P.R = (int)R; P.S = (int)S;
    *out = P;
    *ydt_out = ydt;
    *M_out = M;
    return RAD_OK;
}

static int shrink_scalar(const ShrinkArgs& P, int adt, int64_t M, RadStream stream) {
    switch (adt) {
        case kBF16: launch_shrink<kBF16>(P, M, stream); break;
        case kF16:  launch_shrink<kF16>(P, M, stream);  break;
        case kF32:  launch_shrink<kF32>(P, M, stream);  break;
        case kE4M3: launch_shrink<kE4M3>(P, M, stream); break;
        case kE5M2: launch_shrink<kE5M2>(P, M, stream); break;
        default:    launch_shrink<kI8>(P, M, stream);   break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ---------------------------------------------------------------- the tiled rows' plans (host) */

/* The shrink split, from K and R alone and never M, because it is the summation order. A row tile
 * covers ceil(R / 16) j tiles; at decode that tile is the whole grid, so K is split until j tiles
 * times splits reach kShFill blocks, but never below 16 steps (512 k) a block, past which the
 * partials' traffic and the arrival outweigh what the extra blocks read. The split is rounded so
 * no block is left without a step. E.g. K 2048 R 48 -> 3 x 4 blocks; K 6144 R 16 -> 1 x 12. */
struct ShPlan { int jt, ks, spb, nsteps; };
static ShPlan sh_plan(int64_t K, int64_t R) {
    ShPlan p;
    p.nsteps = (int)((K + kShStep - 1) / kShStep);
    p.jt = (int)((R + kShJT - 1) / kShJT);
    int ks = (kShFill + p.jt - 1) / p.jt;
    const int kmax = (p.nsteps + kShSplitMin - 1) / kShSplitMin;
    if (ks > kmax) ks = kmax;
    if (ks < 1) ks = 1;
    p.spb = (p.nsteps + ks - 1) / ks;
    p.ks = (p.nsteps + p.spb - 1) / p.spb;
    return p;
}
static int64_t sh_part_bytes(int64_t M, int64_t R, const ShPlan& p) {
    return p.ks > 1 ? (int64_t)p.ks * M * R * (int64_t)sizeof(float) : 0;
}

/* How many blocks one tile's passes spread over. Decode with several adapters is a handful of
 * blocks a pass; run one after another inside a block they would add up, spread over blockIdx.z
 * they run at once. Only while the grid is small (every z block of a tile with fewer passes leaves
 * after reading the slots, and a prefill grid would have thousands), and never more than the
 * passes a tile can have. Which block runs a pass is all this decides. */
static int z_fan(int64_t base, int64_t S, int64_t M, int64_t tile) {
    int64_t z = base >= kZGrid ? 1 : kZGrid / base;
    if (z > S) z = S;
    if (z > kZMax) z = kZMax;
    if (z > M) z = M;
    if (z > tile) z = tile;
    return z < 1 ? 1 : (int)z;
}

/* The split's arrival counters, one block a (device, `t`). They cannot live in the scratch: other
 * ops write the arena between two of ours and the counters must be zero at entry. Each lane's `t`
 * is its own buffer and the lora rows are ordered on their lane (not concurrent), so two launches
 * that could overlap never share a block; the reducer leaves every counter zero. A block too small
 * for a launch is replaced by a larger one and the old one is kept, since a recorded pass may
 * still name it. */
/* Every block made for a (device, t), the current one last; all of them go when the memory `t`
 * lives in is freed (r4d_keyed.h). */
struct ShrinkBlocks { std::vector<unsigned*> all; int64_t cap = 0; };
static std::mutex g_shrink_mu;
static std::map<std::pair<int, const char*>, ShrinkBlocks> g_shrink;

static void shrink_freed(const char* lo, const char* hi) {
    std::vector<std::pair<int, unsigned*>> gone;
    {
        std::lock_guard<std::mutex> lk(g_shrink_mu);
        for (auto it = g_shrink.begin(); it != g_shrink.end();) {
            if (it->first.second >= lo && it->first.second < hi) {
                for (unsigned* p : it->second.all) gone.push_back({ it->first.first, p });
                it = g_shrink.erase(it);
            } else {
                ++it;
            }
        }
    }
    int was = 0;
    (void)hipGetDevice(&was);
    for (auto& [dev, p] : gone) {
        (void)hipSetDevice(dev);
        (void)hipFree(p);
    }
    (void)hipSetDevice(was);
}
[[maybe_unused]] static const int g_shrink_reg = r4d::on_freed(shrink_freed);

static unsigned* shrink_ctr(const void* t, int64_t n, RadStream stream) {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return nullptr;
    std::lock_guard<std::mutex> lk(g_shrink_mu);
    ShrinkBlocks& b = g_shrink[std::make_pair(dev, static_cast<const char*>(t))];
    if (!b.all.empty() && b.cap >= n) return b.all.back();
    int64_t cap = !b.all.empty() ? 2 * b.cap : 4096;
    while (cap < n) cap *= 2;
    void* p = nullptr;
    if (hipMalloc(&p, (size_t)cap * sizeof(unsigned)) != hipSuccess) return nullptr;
    /* On the launch's own stream, so the zeros land before the kernel that counts on them. */
    if (hipMemsetAsync(p, 0, (size_t)cap * sizeof(unsigned), (hipStream_t)stream) != hipSuccess)
        return nullptr;
    b.all.push_back((unsigned*)p);
    b.cap = cap;
    return (unsigned*)p;
}

template <int ADT, bool VX, bool VA>
static void launch_shrink_wmma(const ShrinkTArgs& P, dim3 grid, RadStream stream) {
    (void)stream;
    R4D_KLAUNCH(R4D_KNAME(lora_shrink_wmma_kernel<ADT, VX, VA>), grid, dim3(kShThreads),
                (hipStream_t)stream, P);
}
template <int ADT>
static void launch_shrink_wmma_v(const ShrinkTArgs& P, dim3 grid, bool vx, bool va,
                                 RadStream stream) {
    if (vx && va)  launch_shrink_wmma<ADT, true, true>(P, grid, stream);
    else if (vx)   launch_shrink_wmma<ADT, true, false>(P, grid, stream);
    else if (va)   launch_shrink_wmma<ADT, false, true>(P, grid, stream);
    else           launch_shrink_wmma<ADT, false, false>(P, grid, stream);
}
template <int YDT, bool V>
static void launch_expand_tiled(const ExpandTArgs& P, dim3 grid, RadStream stream) {
    (void)stream;
    R4D_KLAUNCH(R4D_KNAME(lora_expand_tiled_kernel<YDT, V>), grid, dim3(kExThreads),
                (hipStream_t)stream, P);
}

template <int YDT, bool V>
static void launch_expand_mm(const ExpandTArgs& P, dim3 grid, RadStream stream) {
    (void)stream;
    R4D_KLAUNCH(R4D_KNAME(lora_expand_mm_kernel<YDT, V>), grid, dim3(kMmThreads),
                (hipStream_t)stream, P);
}

}  /* namespace r4d_lora */

extern "C" {

/* operands: a, a_scale?, slot, lora_a, t */
int r4d_rad_lora_shrink(const RadArgs* a, RadStream stream) {
    using namespace r4d_lora;
    ShrinkArgs P{};
    int adt = 0;
    int64_t M = 0;
    const int rc = shrink_operands(a, &P, &adt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    return shrink_scalar(P, adt, M, stream);
}

/* operands: t, slot, lora_b, scale, y (inout) */
int r4d_rad_lora_expand(const RadArgs* a, RadStream stream) {
    using namespace r4d_lora;
    ExpandArgs P{};
    int ydt = 0;
    int64_t M = 0;
    const int rc = expand_operands(a, &P, &ydt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    if (M * P.ntiles > 0x7fffffff) return RAD_E_SHAPE;
    switch (ydt) {
        case kBF16: launch_expand<kBF16>(P, M * P.ntiles, stream); break;
        case kF16:  launch_expand<kF16>(P, M * P.ntiles, stream);  break;
        default:    launch_expand<kF32>(P, M * P.ntiles, stream);  break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* The tiled shrink: lora_shrink_wmma (bf16 a) and lora_shrink_wmma_q (fp8 / i8 codes whose group
 * is whole 32-wide steps) share this entry point. What the WMMA cannot hold exactly (f16 and f32
 * activations, a scale group that ends inside a step) goes to the scalar kernel, decided by the
 * dtype and the stream's grid and never by M; the rows' bands do not send either here, and a
 * direct caller still gets the exact answer. */
int r4d_rad_lora_shrink_wmma(const RadArgs* a, RadStream stream) {
    using namespace r4d_lora;
    ShrinkArgs P{};
    int adt = 0;
    int64_t M = 0;
    const int rc = shrink_operands(a, &P, &adt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    if (adt == kF16 || adt == kF32) return shrink_scalar(P, adt, M, stream);
    if (adt != kBF16 && P.group % kShStep != 0 && P.group < P.K)
        return shrink_scalar(P, adt, M, stream);

    const ShPlan pl = sh_plan(P.K, P.R);
    const int64_t rt = (M + kShMT - 1) / kShMT;
    if (rt > 65535 || (int64_t)pl.jt * pl.ks > 0x7fffffff) return RAD_E_SHAPE;
    ShrinkTArgs T{};
    T.b = P;
    T.M = (int)M; T.jt = pl.jt; T.ks = pl.ks; T.spb = pl.spb; T.nsteps = pl.nsteps;
    T.Z = z_fan(rt * pl.jt * pl.ks, P.S, M, kShMT);
    if (pl.ks > 1) {
        const int64_t need = sh_part_bytes(M, P.R, pl);
        if (!a->scratch || a->scratch_bytes < need) return RAD_E_SCRATCH;
        const int64_t band = rad_args_geti_or(a, "M", 0);
        const int64_t rows = band > M ? band : M;
        T.part = (float*)a->scratch;
        T.ctr = shrink_ctr(P.t, ((rows + kShMT - 1) / kShMT) * pl.jt * kZMax, stream);
        if (!T.ctr) return RAD_E_DEVICE;
    }
    /* 16-byte loads where the rows allow them; the arithmetic is the same either way. */
    const uintptr_t xa = (uintptr_t)P.a, pa = (uintptr_t)P.la;
    const bool vx = P.a_cs == 1 && xa % 16 == 0 && P.a_rs % (adt == kBF16 ? 8 : 16) == 0;
    const bool va = P.la_s2 == 1 && pa % 16 == 0 && P.la_s1 % 8 == 0 && P.la_s0 % 8 == 0;
    const dim3 grid((unsigned)(pl.jt * pl.ks), (unsigned)rt, (unsigned)T.Z);
    switch (adt) {
        case kBF16: launch_shrink_wmma_v<kBF16>(T, grid, vx, va, stream); break;
        case kE4M3: launch_shrink_wmma_v<kE4M3>(T, grid, vx, va, stream); break;
        case kE5M2: launch_shrink_wmma_v<kE5M2>(T, grid, vx, va, stream); break;
        default:    launch_shrink_wmma_v<kI8>(T, grid, vx, va, stream);   break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* The K split's partials, sized at declare for the band's widest M (the core calls this with it).
 * f16 / f32 activations run the scalar kernel and need none. */
int64_t r4d_rad_lora_shrink_wmma_scratch(const RadArgs* a) {
    using namespace r4d_lora;
    const int64_t M = rad_args_geti_or(a, "M", 0), K = rad_args_geti_or(a, "K", 0);
    const int64_t R = rad_args_geti_or(a, "R", 0);
    if (M <= 0 || K <= 0 || R <= 0 || K > 0x7fffffff || R > (1 << 16)) return 0;
    const char* d = rad_args_gets(a, "dtype");
    if (d && (rad_dtype_parse(d) == RAD_F16 || rad_dtype_parse(d) == RAD_F32)) return 0;
    return sh_part_bytes(M, R, sh_plan(K, R));
}

/* The tiled expand: one block a (128 columns, 32 rows), the grid a function of M's tiles. */
int r4d_rad_lora_expand_tiled(const RadArgs* a, RadStream stream) {
    using namespace r4d_lora;
    ExpandArgs P{};
    int ydt = 0;
    int64_t M = 0;
    const int rc = expand_operands(a, &P, &ydt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    const int64_t rt = (M + kExMT - 1) / kExMT, nt = (P.N + kExNT - 1) / kExNT;
    if (rt > 65535 || nt > 0x7fffffff) return RAD_E_SHAPE;
    ExpandTArgs T{};
    T.b = P;
    T.M = (int)M;
    T.Z = z_fan(rt * nt, P.S, M, kExMT);
    const bool v = P.lb_s2 == 1 && (uintptr_t)P.lb % 16 == 0 && P.lb_s1 % 8 == 0 &&
                   P.lb_s0 % 8 == 0 && P.R % 8 == 0;
    const dim3 grid((unsigned)nt, (unsigned)rt, (unsigned)T.Z);
    switch (ydt) {
        case kBF16: v ? launch_expand_tiled<kBF16, true>(T, grid, stream)
                      : launch_expand_tiled<kBF16, false>(T, grid, stream); break;
        case kF16:  v ? launch_expand_tiled<kF16, true>(T, grid, stream)
                      : launch_expand_tiled<kF16, false>(T, grid, stream);  break;
        default:    v ? launch_expand_tiled<kF32, true>(T, grid, stream)
                      : launch_expand_tiled<kF32, false>(T, grid, stream);  break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* lora_expand_mm: one block a (256 columns, 128 rows); expand_operands reads the same operands. */
int r4d_rad_lora_expand_mm(const RadArgs* a, RadStream stream) {
    using namespace r4d_lora;
    ExpandArgs P{};
    int ydt = 0;
    int64_t M = 0;
    const int rc = expand_operands(a, &P, &ydt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    const int64_t rt = (M + kMmMT - 1) / kMmMT, nt = (P.N + kMmNT - 1) / kMmNT;
    if (rt > 65535 || nt > 0x7fffffff) return RAD_E_SHAPE;
    ExpandTArgs T{};
    T.b = P;
    T.M = (int)M;
    T.Z = 1;
    const bool v = P.lb_s2 == 1 && (uintptr_t)P.lb % 16 == 0 && P.lb_s1 % 8 == 0 &&
                   P.lb_s0 % 8 == 0;
    const dim3 grid((unsigned)nt, (unsigned)rt, 1u);
    switch (ydt) {
        case kBF16: v ? launch_expand_mm<kBF16, true>(T, grid, stream)
                      : launch_expand_mm<kBF16, false>(T, grid, stream); break;
        case kF16:  v ? launch_expand_mm<kF16, true>(T, grid, stream)
                      : launch_expand_mm<kF16, false>(T, grid, stream);  break;
        default:    v ? launch_expand_mm<kF32, true>(T, grid, stream)
                      : launch_expand_mm<kF32, false>(T, grid, stream);  break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* extern "C" */
