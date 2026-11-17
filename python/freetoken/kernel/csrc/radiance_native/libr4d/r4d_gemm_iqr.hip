// gemm_nt_q over the IQR weights (r4d_iqr.h), decoded from their codes where they are read; VRAM
// holds a B-bit weight in B bits. Two rows, as the int8 family's (r4d_gemm_fp8a8.hip):
//
//   gemm_iqr_nt_m64   M <= 64 (decode and speculative verify). At one token (a computed trellis;
//                     M <= 8 for the other forms) a GEMV: a block a 16-row tile, its waves across
//                     K, each lane's decoded values one sudot4 against the activation. To 16 tokens
//                     a WMMA tile of the same shape, each 128-K unit decoded and transposed into
//                     fragment order through the wave's LDS just before its WMMA steps. Above, the
//                     i8a8 narrow tile with the same per-unit decode.
//   gemm_iqr_tiled    any M (prefill). The i8a8 prefill tile over the int8 fragment plane in the
//                     caller's scratch, which the tile's blocks decode as they arrive, each value
//                     once.
//
// The narrow tile above 16 tokens and the prefill tile compute what the int8 narrow and prefill
// tiles compute over the same int8 values (their split, staging, accumulation order and fold), so
// a weight decoded here and the same weight's int8 values in those tiles give the same output
// bytes (the int8 rows' K-parallel tile, r4d_gemm_fp8a8.hip, sums in another order). The GEMV and
// the 16-token tile sum per wave and then across waves: the product within f32 rounding.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"
#include "r4d_iqr.h"
#include "r4d_iqr_dec.h"
#include "r4d_fwht.h"
#include "r4d_q8.h"

#include <algorithm>
#include <cstdio>

/* The i8a8 narrow tile's grid and split rules and its finisher counters (r4d_gemm_fp8a8.hip): a
 * split here sums each output as the int8 tile's does. */
uint32_t r4d_8a8_narrow_mw(uint32_t n, uint32_t m);
uint32_t r4d_8a8_narrow_ksplit(uint32_t N, uint32_t K, uint32_t M, uint32_t* ck, uint32_t target);
unsigned int* r4d_8a8_split_ctr(RadStream s, const void* out);


namespace {

using namespace r4d_fp8;
using namespace r4d_iqr;
using namespace r4d_iqr_dev;

constexpr uint32_t kWave = 32, kBlock = 128, kFK = 256;

/* The waves the narrow tile's K split aims for: four times the int8 tile's 128. A wave here decodes
 * each unit and transposes it through LDS before its WMMA steps, so a weight N = 5120 rows wide
 * (down, the attention and delta-net outputs: 320 waves) left unsplit keeps about one wave a SIMD
 * and stalls on its own decode. At 8 tokens, 2 / 4 / 6 bits: down (K = 8704) 74 -> 54 / 56 / 66 us,
 * the outputs (K = 3072) 29 -> 24 / 25 / 30 us; 1024 and more lose on the outputs. */
constexpr uint32_t kIqrSplitWaves = 512;

/* The forms are built in kIqrParts units, each this file under its own R4D_IQR_PART (the
 * r4d_gemm_iqr_p<k>.hip wrappers; this unit is part 0), so their kernels compile in parallel. A
 * form's part is its place in R4D_IQR_FORMS modulo the count, which spreads every family over all
 * parts. Part 0 alone defines the rows' entry points and hands each call to its form's part. */
#ifndef R4D_IQR_PART
#define R4D_IQR_PART 0
#endif
constexpr int kIqrParts = 4;

#define R4D_IQR_W(k, b, lv, l, lrb, sc) FormT<k, b, lv, l, lrb, sc>::word(),
constexpr uint32_t kIqrWords[] = { R4D_IQR_FORMS(R4D_IQR_W) };
#undef R4D_IQR_W

/* The part building the form with kernel word `kw`; 0 for a word that is no form, whose call part 0
 * then refuses as before. */
constexpr int iqr_part(uint32_t kw) {
    for (size_t i = 0; i < sizeof(kIqrWords) / sizeof(kIqrWords[0]); ++i)
        if (kIqrWords[i] == kw) return (int)(i % kIqrParts);
    return 0;
}
template <class F>
constexpr bool in_part() { return iqr_part(F::word()) == R4D_IQR_PART; }

#include "r4d_gemm_8a8_prefill.h"

/* ---- the gated fold (gemm_nt_q_gated at i8a8h) ----------------------------------------------
 *
 * gate_up's GEMM with gated_had_quant_i8g's pass in its epilogue. A block computes one 16-column
 * tile of the gated output: the gate rows' tile and the up rows' tile, each with the waves and K
 * order of the plain kernel at the full weight's N, so their bf16 values are the plain kernel's.
 * It writes silu(gate) * up, rounded as gated_quant_fp8_kernel rounds it, into the op's scratch
 * (and `out_bf16`), and the last of a 128-column group's eight blocks to arrive rotates and
 * quantises the group, a wave a token, as that kernel does: so the codes and scales are the
 * pair's bytes. */
struct GatedOut {
    uint16_t* H;            /* [M][n] the bf16 product, scratch */
    int8_t* Q;              /* [M][q_ld] int8 codes */
    float* S;               /* [M][s_ld] a scale per (token, 128 columns) */
    uint16_t* Yb;           /* [M][y_ld] the unrotated bf16 product, or null */
    unsigned int* cnt;      /* an arrival counter per 128-column group */
    uint32_t n, q_ld, s_ld, y_ld;
};

/* Token m's gated value at column `col` from the gate and up rows' f32 sums (row scale applied). */
__device__ __forceinline__ void gated_put(const GatedOut& go, uint32_t m, uint32_t col, float vg,
                                          float vu) {
    const float gv = bf16_to_f32(f32_to_bf16(vg));
    const float a = gv / (1.0f + expf(-gv));
    const uint16_t h = f32_to_bf16(a * bf16_to_f32(f32_to_bf16(vu)));
    go.H[(size_t)m * go.n + col] = h;
    if (go.Yb) go.Yb[(size_t)m * go.y_ld + col] = h;
}

/* After every thread's gated_put: the group's last arriver quantises it, `nw` waves. */
__device__ __forceinline__ void gated_tail(const GatedOut& go, uint32_t M, uint32_t nw) {
    __shared__ unsigned int s_last;
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const uint32_t g = blockIdx.x / 8u;
    __threadfence();
    __syncthreads();
    if (tid == 0)
        s_last = __hip_atomic_fetch_add(&go.cnt[g], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_last != 7u) return;
    if (tid == 0) __hip_atomic_store(&go.cnt[g], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();
    for (uint32_t m = wave; m < M; m += nw) {
        const uint32_t c0 = g * 128u + lane * 4u;
        float v[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) v[j] = bf16_to_f32(go.H[(size_t)m * go.n + c0 + j]);
        r4d_fwht_wave<128>(v, (int)lane);
        const float mx = r4d_wave_max(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
        const float sc = mx > 0.0f ? mx * (1.0f / q8_top<true>()) : 1.0f;
        if (lane == 0) go.S[(size_t)m * go.s_ld + g] = sc;
        *(uint32_t*)(go.Q + (size_t)m * go.q_ld + c0) = q8_pack4<true>(v[0], v[1], v[2], v[3], 1.0f / sc);
    }
}

/* ---- the GEMV: M <= 8 ----------------------------------------------------------------------
 *
 * A block a 16-row tile, its KW waves across K: wave w takes the tile's 128-K blocks w, w + KW, ..,
 * so at any time a block's waves read KW consecutive units of one tile. Within a wave lane l takes
 * columns 4l .. 4l + 3 of each block, so its decoded values of a row are one sudot4 against the
 * activation's four bytes there, and a row's sum is folded into f32 with the (row, block) and
 * (token, block) scales a block at a time, in each lane. At the end each wave sums its lanes and
 * the block sums its waves through LDS in ascending order. The weight stream is the codes' own
 * bytes, fetched a block ahead; the work a value is its decode, a quarter sudot4 a token and half
 * an f32 fold.
 *
 * A block lives for 1 / KW of a tile's K, so the card balances over N / 16 short blocks and no
 * weight needs a split across blocks. A wave a tile over all of K (the previous form) left a
 * partial last round of long waves on every shape: at 102 VGPRs a SIMD holds 15 waves, and N / 16
 * tiles (gate_up 2176, down split 7 ways 2240) never divide that.
 *
 * The fold order differs from the WMMA tile's (per lane, then across lanes and waves), so the
 * output is the product within f32 rounding rather than the int8 tile's bytes. */
template <class F, uint32_t MT, uint32_t KW, bool GATED, class CH>
__global__ __launch_bounds__((GATED ? 2u : 1u) * KW * kWave) R4D_WAVES(6)
void iqr_gemv_kernel(Ctx c0, const int8_t* __restrict__ X, const float* __restrict__ XS,
                     uint16_t* __restrict__ Y, uint32_t N, uint32_t x_ld, uint32_t xs_ld,
                     GatedOut go) {
    if (CH::on) r4d_chain_wait();
    using D = Dec<F, true>;
    constexpr uint32_t NW = (GATED ? 2u : 1u) * KW;
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    __shared__ float s_red[NW][16 * MT];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1);
    const uint32_t wave0 = __builtin_amdgcn_readfirstlane(tid / kWave);
    Ctx c = c0;
    c.tab = s_tab;
    D::stage(s_tab, c, tid, NW * kWave);
    __syncthreads();

    /* Gated, waves KW and up take the up rows' tile, n rows below the gate rows'. */
    const uint32_t half = GATED ? wave0 / KW : 0u, wave = GATED ? wave0 % KW : wave0;
    const uint32_t rt = blockIdx.x + half * (N / 32u);
    const uint32_t myrow = rt * 16u + (lane & 15u);

    float acc[16][MT];
#pragma unroll
    for (uint32_t q = 0; q < 16; ++q)
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) acc[q][m] = 0.0f;
    float fsum[MT];   /* the biased path's fold factors since its last flush (wave-uniform) */
#pragma unroll
    for (uint32_t m = 0; m < MT; ++m) fsum[m] = 0.0f;
    uint32_t nflush = 0;

    typename D::Raw cur = D::fetch(c, rt, wave, lane, wave < c.nkb), nxt;
    for (uint32_t kb = wave; kb < c.nkb; kb += KW) {
        if (kb + KW < c.nkb) nxt = D::fetch(c, rt, kb + KW, lane, true);
        uint32_t xv[MT];
        float as[MT];
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) {
            xv[m] = *(const uint32_t*)(X + (size_t)m * x_ld + kb * kBlock + 4u * lane);
            as[m] = XS[(size_t)m * xs_ld + kb];
        }
        /* Row (lane & 15)'s fold scale and ratio, read by every lane through readlane. */
        /* A bf16 scale a (row, 128): the tile's sixteen rows' scales of this block are 32
         * consecutive bytes at a wave-uniform address, read through the constant address space
         * into scalar registers, so no row's scale is broadcast by readlane. */
        constexpr bool SROW = F::SC == kBf16 && F::LRB == 0;
        [[maybe_unused]] uint32_t sw[8];
        if constexpr (SROW) {
            typedef const __attribute__((address_space(4))) uint32_t cu32;
            cu32* sp = (cu32*)((const uint16_t*)c.S + ((size_t)rt * c.nkb + kb) * 16u);
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) sw[i] = sp[i];
        }
        auto row_fold = [&](uint32_t q, float fs_) {
            if constexpr (SROW)
                return __uint_as_float(q & 1u ? sw[q / 2u] & 0xFFFF0000u : sw[q / 2u] << 16);
            else
                return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__float_as_int(fs_), q));
        };
        const float fs = SROW ? 0.0f : fold_f32<F>(c, myrow, kb);
        const float ratio = rb_ratio<F>(c, myrow, kb);
        uint32_t w[16];
        /* A computed trellis off the tcq_rb grid decodes each value as its byte sum, the value
         * plus 126: one sudot4 of the unsigned sums less 126 times the lane's activation sum. */
        constexpr bool UNS = D::MAD && F::LRB == 0;
        [[maybe_unused]] int xsum[MT];
        if constexpr (UNS) {
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m)
                xsum[m] = __builtin_amdgcn_sudot4(true, (int)xv[m], false, 0x7E7E7E7E, 0, false);
        }
        if constexpr (UNS && MT <= 4 && F::SC == kRowF32) {
            /* One token: each value's hash dotted with its activation byte in every byte lane
             * sums the value's bytes and multiplies in one instruction, nothing packed. Under one
             * f32 scale a row the block's fold is one factor for all rows, so a row's dot starts
             * at 0x4B400000 less the lane's bias sum: read as an f32 it is exactly 12582912 plus
             * the sum (|sum| < 2^22), one fma folds it, and 12582912 times the block's factors
             * leaves each row once at the end. */
            uint32_t h[16][4];
            D::hashes(cur, c, h);
#pragma unroll
            for (uint32_t m = 0; m < MT; ++m) {
                uint32_t xs[4];
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e) xs[e] = __builtin_amdgcn_perm(0u, xv[m], e * 0x01010101u);
                const float s = fs * as[m];
                fsum[m] += s;
                const int b0 = kRbBias - __builtin_amdgcn_sudot4(true, (int)xv[m], false, 0x7E7E7E7E, 0, false);
#pragma unroll
                for (uint32_t q = 0; q < 16; ++q) {
                    int d = b0;
#pragma unroll
                    for (uint32_t e = 0; e < 4; ++e)
                        d = __builtin_amdgcn_sudot4(false, (int)h[q][e], true, (int)xs[e], d, false);
                    acc[q][m] = fmaf(__int_as_float(d), s, acc[q][m]);
                }
            }
            /* the bias out every kRbFlush blocks, so a row's sum stays within a few bias terms
             * of its value (the prefill tile's bound) */
            if (++nflush == kRbFlush) {
#pragma unroll
                for (uint32_t m = 0; m < MT; ++m) {
#pragma unroll
                    for (uint32_t q = 0; q < 16; ++q) acc[q][m] = fmaf(-kRbBiasF, fsum[m], acc[q][m]);
                    fsum[m] = 0.0f;
                }
                nflush = 0;
            }
        } else if constexpr (UNS && MT == 1) {
            uint32_t h[16][4];
            D::hashes(cur, c, h);
            uint32_t xs[4];
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) xs[e] = __builtin_amdgcn_perm(0u, xv[0], e * 0x01010101u);
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                const float wsq = row_fold(q, fs);
                int d = -xsum[0];
#pragma unroll
                for (uint32_t e = 0; e < 4; ++e)
                    d = __builtin_amdgcn_sudot4(false, (int)h[q][e], true, (int)xs[e], d, false);
                acc[q][0] = fmaf((float)d, wsq * as[0], acc[q][0]);
            }
        } else {
            D::template rows<UNS>(cur, c, ratio, w);
            D::unzero(c, rt, kb, w);
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) {
                const float wsq = row_fold(q, fs);
#pragma unroll
                for (uint32_t m = 0; m < MT; ++m) {
                    int d = __builtin_amdgcn_sudot4(!UNS, (int)w[q], true, (int)xv[m], 0, false);
                    if constexpr (UNS) d -= xsum[m];
                    acc[q][m] = fmaf((float)d, wsq * as[m], acc[q][m]);
                }
            }
        }
        cur = nxt;
    }
    /* every lane's rows carry 12582912 times the factors once for each of its blocks; a token's
     * sixteen rows are summed over the wave together (red16), lane l left with row red16_row(l) */
    float rv[MT];
#pragma unroll
    for (uint32_t m = 0; m < MT; ++m) {
        float a16[16];
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) a16[q] = fmaf(-kRbBiasF, fsum[m], acc[q][m]);
        rv[m] = red16(a16, lane);
    }
    if (lane < 16u) {
#pragma unroll
        for (uint32_t m = 0; m < MT; ++m) s_red[wave0][lane * MT + m] = rv[m];
    }
    __syncthreads();
    if constexpr (GATED) {
        if (tid < 16u * MT) {
            const uint32_t q = red16_row(tid / MT), m = tid % MT;
            float vg = 0.0f, vu = 0.0f;
#pragma unroll
            for (uint32_t w = 0; w < KW; ++w) vg += s_red[w][tid];
#pragma unroll
            for (uint32_t w = 0; w < KW; ++w) vu += s_red[KW + w][tid];
            vg *= row_scale<F>(c, blockIdx.x * 16u + q);
            vu *= row_scale<F>(c, (blockIdx.x + N / 32u) * 16u + q);
            gated_put(go, m, blockIdx.x * 16u + q, vg, vu);
        }
        gated_tail(go, MT, NW);
    } else {
        if (tid < 16u * MT) {
            const uint32_t q = red16_row(tid / MT), m = tid % MT;
            float v = 0.0f;
#pragma unroll
            for (uint32_t w = 0; w < KW; ++w) v += s_red[w][tid];
            /* each row's own scale, after its blocks' sums */
            v *= row_scale<F>(c, rt * 16u + q);
            Y[(size_t)m * N + rt * 16u + q] = f32_to_bf16(v);
        }
    }
    if (CH::on) r4d_chain_done();
}

/* ---- the narrow tile at M <= 16 -------------------------------------------------------------
 *
 * One 16-token tile: a block a 16-row weight tile, its KW waves across K as the GEMV's (wave w
 * the tile's 128-K blocks w, w + KW, ..). A wave decodes each unit and transposes it into
 * fragment order through its own LDS tile (frag), then runs the unit's eight WMMA steps against
 * the activation's fragments read straight from global memory, where the M x K activation sits
 * in L2, so the loop has no block barrier and no shared staging. Rows of the token tile past M
 * read token M - 1; their outputs are not stored. NT token tiles a block (to 64 tokens): a unit
 * is decoded and transposed once and multiplied against each tile in turn, the further tiles'
 * fragments asked for a tile ahead. Measured cold against the narrow tile at the trunk's per-rank
 * shapes, 6 bits, us a launch: 6144 x 2560 at 32 tokens 38.8 -> 30.4 and at 64 53.8 -> 34.6;
 * 2560 x 3072 at 32 tokens 31.1 -> 20.4 and at 64 47.4 -> 24.6; 8192 x 2560 level (36.2 / 37.2
 * at 32, 42.0 / 42.5 at 64).
 *
 * Under one f32 scale a row each unit's int32 sums start at 0x4B400000, so a slot read as an f32
 * is 12582912 plus its sum and one fma folds it (the prefill tile's RB 32 fold), flushed every
 * kRbFlush units; other scale forms fold the converted sum with their (row, block) scales. The
 * waves' partials are summed through LDS in ascending wave order, so the output is the product
 * within f32 rounding, not the int8 tile's bytes. */
template <class F, uint32_t KW, bool GATED, uint32_t NT, class CH>
__global__ __launch_bounds__((GATED ? 2u : 1u) * KW * kWave) R4D_WAVES(6)
void iqr_m16_kernel(Ctx c0, const int8_t* __restrict__ X, const float* __restrict__ XS,
                    uint16_t* __restrict__ Y, uint32_t N, uint32_t M, uint32_t x_ld, uint32_t xs_ld,
                    GatedOut go) {
    if (CH::on) r4d_chain_wait();
    using D = Dec<F>;
    static_assert(!GATED || NT == 1u, "the gated form serves one token tile");
    constexpr bool BIAS = F::SC == kRowF32;
    constexpr uint32_t NW = (GATED ? 2u : 1u) * KW;
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    /* the waves' transpose tiles, then their partials (8 x 32 floats a wave): one token tile's
     * in the wave's own transpose tile, several tiles' in s_nt */
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[NW * kTrBytes];
    __shared__ float s_nt[NT > 1u ? NW : 1u][NT > 1u ? NT : 1u][NT > 1u ? 8u * kWave : 1u];
    static_assert(8u * kWave * 4u <= kTrBytes, "a wave's partials in its transpose tile");
    auto red = [&](uint32_t w, uint32_t j) {
        return NT > 1u ? &s_nt[w][j][0] : (float*)(s_tr + w * kTrBytes);
    };
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1);
    const uint32_t wave0 = __builtin_amdgcn_readfirstlane(tid / kWave);
    Ctx c = c0;
    c.tab = s_tab;
    uint8_t* const tr = s_tr + wave0 * kTrBytes;
    D::stage(s_tab, c, tid, NW * kWave);
    __syncthreads();

    /* Gated, waves KW and up take the up rows' tile, n rows below the gate rows'. */
    const uint32_t half = GATED ? wave0 / KW : 0u, wave = GATED ? wave0 % KW : wave0;
    const uint32_t rt = blockIdx.x + half * (N / 32u), row0 = rt * 16u;
    /* Token tile j is tokens 16 j .. 16 j + 15; a column past M reads token M - 1. */
    uint32_t tok[NT];
    const int8_t* xrow[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        tok[j] = min(j * 16u + frag_c(lane), M - 1u);
        xrow[j] = X + (size_t)tok[j] * x_ld + frag_k0(lane);
    }

    f8v out[NT];
    float corr[NT];   /* BIAS: the units' fold factors since the last flush */
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        out[j] = acc_zero();
        corr[j] = 0.0f;
    }
    uint32_t nflush = 0;
    auto bload = [&](uint32_t j, uint32_t kb, i2v (&b)[8], bool live) {
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks) {
            const uint2 v = live ? *(const uint2*)(xrow[j] + kb * kBlock + ks * kK)
                                 : make_uint2(0u, 0u);
            b[ks][0] = (int)v.x;
            b[ks][1] = (int)v.y;
        }
    };
    /* The weight and the first token tile are asked for a unit ahead of their WMMA steps, each
     * further tile's fragments a tile ahead (gemm_i8_kpar_kernel's order). */
    typename D::Raw cur = D::fetch(c, rt, wave, lane, wave < c.nkb), nxt;
    i2v bc[8], bn[8], bq[2][8];
    bload(0u, wave, bc, wave < c.nkb);
    for (uint32_t kb = wave; kb < c.nkb; kb += KW) {
        if (kb + KW < c.nkb) {
            nxt = D::fetch(c, rt, kb + KW, lane, true);
            bload(0u, kb + KW, bn, true);
        }
        uint32_t w[16];
        D::rows(cur, c, rb_ratio<F>(c, row0 + (lane & 15u), kb), w);
        D::unzero(c, rt, kb, w);
        Stage8 a8[2];
        frag(w, tr, lane, a8);
        float wsr[8];   /* the lane's eight rows' (row, block) scales, where the form has them */
        if constexpr (!BIAS) {
            if constexpr (F::SC == kBf16 && F::LRB == 0) {
                /* A bf16 scale a (row, 128): the lane's eight rows are consecutive in the tile's
                 * scale run (own_scale's order), 16 aligned bytes, one load. */
                const uint4 q4 = *(const uint4*)((const uint16_t*)c.S + ((size_t)rt * c.nkb + kb) * 16u +
                                                 acc_row(lane, 0));
                const uint32_t qd[4] = { q4.x, q4.y, q4.z, q4.w };
#pragma unroll
                for (uint32_t t = 0; t < 4; ++t) {
                    wsr[2 * t] = __uint_as_float(qd[t] << 16);
                    wsr[2 * t + 1] = __uint_as_float(qd[t] & 0xFFFF0000u);
                }
            } else {
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) wsr[t] = fold_f32<F>(c, row0 + acc_row(lane, t), kb);
            }
        }
        float sbw = 0.0f;   /* BIAS: the unit's fold factor before the token's scale */
        if constexpr (BIAS) sbw = fold_f32<F>(c, row0, kb);
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks) bq[0][ks] = bc[ks];
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            if (j + 1u < NT) bload(j + 1u, kb, bq[(j + 1u) & 1u], true);
            /* two chains of four steps, so a step does not wait on the one before it */
            f8v raw = rb_first<BIAS ? 32 : 0>();
#pragma unroll
            for (uint32_t ks = 0; ks < kBlock / kK; ++ks)
                raw = mma8<true>(frag_of(a8[ks / 4u], ks % 4u), bq[j & 1u][ks], raw);
            const float as = XS[(size_t)tok[j] * xs_ld + kb];
            if constexpr (BIAS) {
                const float sb = sbw * as;
                corr[j] += sb;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) out[j][t] = __builtin_fmaf(raw[t], sb, out[j][t]);
            } else {
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    out[j][t] = __builtin_fmaf(acc_val<true>(raw[t]), wsr[t] * as, out[j][t]);
            }
        }
        if constexpr (BIAS) {
            if (++nflush == kRbFlush) {
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) {
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        out[j][t] = __builtin_fmaf(-kRbBiasF, corr[j], out[j][t]);
                    corr[j] = 0.0f;
                }
                nflush = 0;
            }
        }
        cur = nxt;
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks) bc[ks] = bn[ks];
    }
    /* each wave's partials in its own tile, behind its last transpose's reads */
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t)
            red(wave0, j)[t * kWave + lane] =
                BIAS ? __builtin_fmaf(-kRbBiasF, corr[j], out[j][t]) : out[j][t];
    __syncthreads();
    if constexpr (GATED) {
        if (wave0 == 0) {
            const uint32_t col = frag_c(lane);
            const uint32_t r0 = acc_row(lane, 0);
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                float ag = 0.0f, au = 0.0f;
#pragma unroll
                for (uint32_t w = 0; w < KW; ++w) ag += red(w, 0u)[t * kWave + lane];
#pragma unroll
                for (uint32_t w = 0; w < KW; ++w) au += red(KW + w, 0u)[t * kWave + lane];
                ag *= row_scale<F>(c, blockIdx.x * 16u + r0 + t);
                au *= row_scale<F>(c, (blockIdx.x + N / 32u) * 16u + r0 + t);
                if (col < M) gated_put(go, col, blockIdx.x * 16u + r0 + t, ag, au);
            }
        }
        gated_tail(go, M, NW);
    } else if (wave == 0) {
        const uint32_t r0 = row0 + acc_row(lane, 0);
        float rs[8];
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t) rs[t] = row_scale<F>(c, r0 + t);
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const uint32_t col = j * 16u + frag_c(lane);
            float v[8];
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                float a = 0.0f;
#pragma unroll
                for (uint32_t w = 0; w < KW; ++w) a += red(w, j)[t * kWave + lane];
                v[t] = a * rs[t];
            }
            if (col < M)
                *(uint4*)(Y + (size_t)col * N + r0) =
                    make_uint4(bf16x2_rne(v[0], v[1]), bf16x2_rne(v[2], v[3]),
                               bf16x2_rne(v[4], v[5]), bf16x2_rne(v[6], v[7]));
        }
    }
    if (CH::on) r4d_chain_done();
}

/* ---- the narrow tile: 16 < M <= 64 -----------------------------------------------------------
 *
 * r4d_gemm_fp8a8.hip's gemm_fp8a8_body at I8, unfused: MW waves a block, a wave one 16-row weight
 * tile against NT 16-token tiles staged through LDS, split-K over the caller's scratch with the
 * last z block reducing. Only the weight operand's source differs: each 128-K block's unit is
 * decoded and transposed into fragment order through the wave's LDS tile (r4d_iqr_dec.h), its two
 * supertiles just before their eight WMMA steps. The int8 tile's arithmetic, so its bytes. */
#define R4D_IQR_M64_PARAMS                                                                        \
    Ctx c0, const uint8_t* __restrict__ Bp, const float* __restrict__ Sb, void* __restrict__ Y,   \
    uint16_t* __restrict__ C, unsigned int* __restrict__ Ctr, uint32_t rows, uint32_t cols,       \
    uint32_t nb, uint32_t b_ld, uint32_t sb_ld, uint32_t ck
template <class F, uint32_t NT, uint32_t MW>
__device__ __forceinline__ void iqr_m64_body(R4D_IQR_M64_PARAMS) {
    using D = Dec<F>;
    constexpr uint32_t FK = kFK, FN = NT * 16, FM = MW * 16, kU = FK / kBlock;
    __shared__ uint8_t s_b[FN * lds_pitch<FK>()];
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[MW * kTrBytes];

    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const uint32_t kbeg = ck ? blockIdx.z * ck : 0u;
    const uint32_t kend = ck ? min(cols, kbeg + ck) : cols;
    if (kbeg >= kend) return;

    Ctx c = c0;
    c.tab = s_tab;
    uint8_t* const tr = s_tr + wave * kTrBytes;
    D::stage(s_tab, c, tid, MW * kWave);

    const uint32_t m0 = blockIdx.x * FM, n0 = blockIdx.y * FN;
    const uint32_t wrow0 = m0 + wave * 16u;

    f8v raw[NT], out[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) { raw[j] = acc_zero(); out[j] = acc_zero(); }

    /* The activation's staging: the int8 narrow tile's (kTPR threads a row). */
    constexpr uint32_t kSeg = 32, kTPR = FK / kSeg, kRPP = (MW * kWave) / kTPR;
    constexpr uint32_t kBP = (FN + kRPP - 1) / kRPP;
    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;
    const uint32_t brow0 = n0 + frag_c(lane);
    const uint8_t* bsrc = Bp + (size_t)(n0 + sn) * b_ld + sk;
    const uint32_t arow = wrow0 / 16u;
    const bool alive = wrow0 < rows;
    const bool k_live = kbeg + sk + kSeg <= cols;

    Stage8 pb[kBP];
#pragma unroll
    for (uint32_t p = 0; p < kBP; ++p) {
        const uint32_t r = sn + p * kRPP;
        pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + kbeg, k_live && r < FN && n0 + r < nb);
    }
    typename D::Raw pf[kU], pfn[kU];
#pragma unroll
    for (uint32_t p = 0; p < kU; ++p) {
        const uint32_t kb = kbeg / kBlock + p;
        pf[p] = D::fetch(c, arow, kb, lane, alive && kb < c.nkb);
    }
    __syncthreads();   /* the table */

    for (uint32_t c0k = kbeg; c0k < kend; c0k += FK) {
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r < FN) stage8_store<FK>(s_b, r, sk, pb[p]);
        }
        __syncthreads();
        const uint32_t c1 = c0k + FK;
        if (c1 < kend) {
            const bool live = c1 + sk + kSeg <= cols;
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r < FN) pb[p] = stage8_load(bsrc + (size_t)p * kRPP * b_ld + c1, live && n0 + r < nb);
            }
#pragma unroll
            for (uint32_t p = 0; p < kU; ++p) {
                const uint32_t kb = c1 / kBlock + p;
                pfn[p] = D::fetch(c, arow, kb, lane, alive && kb < c.nkb);
            }
        }

#pragma unroll
        for (uint32_t p = 0; p < kU; ++p) {
            const uint32_t kbu = c0k / kBlock + p;
            uint32_t w[16];
            D::rows(pf[p], c, alive ? rb_ratio<F>(c, wrow0 + (lane & 15u), kbu) : 0.0f, w);
            D::unzero(c, arow, kbu, w);
            Stage8 a8[2];
            frag(w, tr, lane, a8);
#pragma unroll
            for (uint32_t kk = 0; kk < kBlock / kK; ++kk) {
                const uint32_t ks = p * kBlock + kk * kK;
                const i2v af = frag_of(a8[kk / 4u], kk % 4u);
                i2v bf[NT];
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) bf[j] = frag<FK>(s_b, j * 16, ks, lane);
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) raw[j] = mma8<true>(af, bf[j], raw[j]);
            }
            /* The unit's 128-K block is done: fold its sums. Units are 128-K aligned and a split's
             * slice is a whole number of FK chunks (narrow_ksplit), so every block ends here. */
            {
                const uint32_t kb = kbu;
                /* A lane's eight accumulator rows, consecutive rows of the wave's tile; under one
                 * f32 scale a row the block's factor is the same for all of them. */
                float wsr[8];
                if constexpr (F::SC == kRowF32) {
                    const float f = alive ? fold_f32<F>(c, wrow0, kb) : 0.0f;
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) wsr[t] = f;
                } else if constexpr (F::SC == kBf16 && F::LRB == 0) {
                    /* A bf16 scale a (row, 128): the lane's eight rows are consecutive in the
                     * tile's scale run (own_scale's order), 16 aligned bytes, one load. Every row
                     * of a live tile is live (rows is a multiple of 16). */
                    uint4 q = make_uint4(0u, 0u, 0u, 0u);
                    if (alive)
                        q = *(const uint4*)((const uint16_t*)c.S +
                                            ((size_t)(wrow0 / 16u) * c.nkb + kb) * 16u + acc_row(lane, 0));
                    const uint32_t qd[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
                    for (uint32_t t = 0; t < 4; ++t) {
                        wsr[2 * t] = __uint_as_float(qd[t] << 16);
                        wsr[2 * t + 1] = __uint_as_float(qd[t] & 0xFFFF0000u);
                    }
                } else {
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        wsr[t] = alive ? fold_f32<F>(c, wrow0 + acc_row(lane, t), kb) : 0.0f;
                }
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) {
                    const uint32_t tok = brow0 + j * 16;
                    const float s = tok < nb ? Sb[(size_t)tok * sb_ld + kb] : 0.0f;
                    if constexpr (F::SC == kRowF32) {
                        const float ws = wsr[0] * s;
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            out[j][t] += acc_val<true>(raw[j][t]) * ws;
                            raw[j][t] = 0.0f;
                        }
                    } else {
#pragma unroll
                        for (uint32_t t = 0; t < 8; ++t) {
                            out[j][t] += acc_val<true>(raw[j][t]) * (wsr[t] * s);
                            raw[j][t] = 0.0f;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (uint32_t p = 0; p < kU; ++p) pf[p] = pfn[p];
        __syncthreads();
    }

    /* The int8 narrow tile's store: eight consecutive weight rows of one token a lane a tile, one
     * 16-byte write (rows is a multiple of 16), into the split's partial plane or C. */
    float* const part = ck ? (float*)Y + (size_t)blockIdx.z * rows * nb : nullptr;
    const uint32_t cl = frag_c(lane);
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t col = n0 + j * 16 + cl;
        const uint32_t r0 = wrow0 + acc_row(lane, 0);
        if (col >= nb || r0 >= rows) continue;
        f8v ov = out[j];
        if constexpr (F::SC == kRowF32)
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) ov[t] *= row_scale<F>(c, r0 + t);
        const size_t o = (size_t)col * rows + r0;
        if (ck) {
            __attribute__((aligned(16))) float pv[8];
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) pv[t] = ov[t];
            *(uint4*)(part + o) = *(const uint4*)pv;
            *(uint4*)(part + o + 4) = *(const uint4*)(pv + 4);
        } else {
            *(uint4*)(C + o) = make_uint4(bf16x2_rne(ov[0], ov[1]), bf16x2_rne(ov[2], ov[3]),
                                          bf16x2_rne(ov[4], ov[5]), bf16x2_rne(ov[6], ov[7]));
        }
    }

    /* The int8 narrow tile's split finisher: the last z block of a column sums the partials in
     * ascending z, so the bytes are those of the separate reduce. */
    if (ck && Ctr) {
        __threadfence();
        __syncthreads();
        __shared__ unsigned int s_arrived;
        if (tid == 0)
            s_arrived = __hip_atomic_fetch_add(&Ctr[blockIdx.x], 1u, __ATOMIC_ACQ_REL,
                                               __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        if (s_arrived + 1u == gridDim.z) {
            if (tid == 0)
                __hip_atomic_store(&Ctr[blockIdx.x], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __threadfence();
            const uint32_t rlo = m0, rhi = min(m0 + FM, rows);
            const float* const w0 = (const float*)Y;
            for (uint32_t idx = tid; idx < (rhi - rlo) * nb; idx += MW * kWave) {
                const uint32_t r = rlo + idx / nb, col = idx % nb;
                const size_t o = (size_t)col * rows + r;
                float acc = 0.0f;
                for (uint32_t z = 0; z < gridDim.z; ++z) acc += w0[(size_t)z * rows * nb + o];
                C[o] = f32_to_bf16(acc);
            }
        }
    }
}

/* The kernel around the body, whose blocks past the split's K return at once: CH `r4dchain` is
 * the chained form (r4d_chain.h). */
template <class F, uint32_t NT, uint32_t MW, class CH>
__global__ __launch_bounds__(MW * kWave) R4D_WAVES(MW >= 8u ? 4u : 6u)
void iqr_m64_kernel(R4D_IQR_M64_PARAMS) {
    if (CH::on) r4d_chain_wait();
    iqr_m64_body<F, NT, MW>(c0, Bp, Sb, Y, C, Ctr, rows, cols, nb, b_ld, sb_ld, ck);
    if (CH::on) r4d_chain_done();
}
#undef R4D_IQR_M64_PARAMS

/* The split's separate reduce, for a split with no counter block (r4d_gemm_fp8a8.hip's). */
__global__ void iqr_splitk_reduce(const float* __restrict__ ws, uint16_t* __restrict__ y,
                                  uint32_t n_out, uint32_t ksplit) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    float acc = 0.0f;
    for (uint32_t z = 0; z < ksplit; ++z) acc += ws[(size_t)z * n_out + i];
    y[i] = f32_to_bf16(acc);
}

/* ---- the prefill: 64 < M ---------------------------------------------------------------------
 *
 * The int8 prefill tile (r4d_gemm_8a8_prefill.h) over the int8 fragment plane in the caller's
 * scratch, which the tile's own blocks decode as they arrive. The blocks of one 128-row N tile are
 * consecutive in the tile's raster order (its M bands within a group); each claims one of the N
 * tile's eight 16-row tiles at a time, decodes it into the plane (and its fold scales, where the
 * scale buffer is not already the int8 tile's), and once all eight are decoded every block of the
 * N tile runs the K loop over them. A block claims until none is left before it waits, so it never
 * waits on a block that has not started: no deadlock whatever the dispatch order. Later groups'
 * blocks find the N tile decoded and go straight to the K loop; the last block of the N tile
 * resets its three counters (claimed, decoded, finished), so they are zero for the next call.
 *
 * The decode is each value's once, overlapped with the other blocks' K loops, and the K loop is
 * the int8 tile's, so the output is its bytes. */
template <class F>
struct IqrPre {
    Ctx c;
    uint8_t* fragp;
    uint16_t* sc;          /* the fold scales the tile reads, or null where the scale buffer is */
    unsigned int* ctl;     /* three counters an N tile */
    uint32_t nbm;          /* the M bands: blocks an N tile */

    __device__ void begin(uint32_t bm, uint32_t bn, uint8_t* lds) const {
        (void)bm;
        using D = Dec<F>;
        static_assert(kPWaves * kTrBytes <= kPrefillLds, "a wave's transpose tile in the staging LDS");
        __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
        __shared__ uint32_t s_part;
        const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
        unsigned int* const claimed = ctl + 3u * bn;
        unsigned int* const decoded = claimed + 1;
        Ctx cc = c;
        cc.tab = s_tab;
        uint8_t* const tr = lds + wave * kTrBytes;
        const uint32_t ksuper = c.nkb * 2u;
        bool staged = false;
        for (;;) {
            if (tid == 0)
                s_part = __hip_atomic_load(decoded, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= 8u
                             ? 8u
                             : __hip_atomic_fetch_add(claimed, 1u, __ATOMIC_RELAXED,
                                                      __HIP_MEMORY_SCOPE_AGENT);
            __syncthreads();
            const uint32_t part = s_part;
            __syncthreads();
            if (part >= 8u) break;
            if (!staged) {
                D::stage(s_tab, cc, tid, kPWaves * kWave);
                __syncthreads();
                staged = true;
            }
            const uint32_t rt = bn * 8u + part;
            /* A unit's codes are fetched a unit ahead of its decode, so a wave's units do not
             * each wait out a memory round trip. */
            typename D::Raw cur = D::fetch(cc, rt, wave, lane, wave < c.nkb), nxt;
            for (uint32_t kb = wave; kb < c.nkb; kb += kPWaves) {
                if (kb + kPWaves < c.nkb) nxt = D::fetch(cc, rt, kb + kPWaves, lane, true);
                uint32_t w[16];
                D::rows(cur, cc, rb_ratio<F>(cc, rt * 16u + (lane & 15u), kb), w);
                cur = nxt;
                D::unzero(cc, rt, kb, w);
                Stage8 s8[2];
                frag(w, tr, lane, s8);
#pragma unroll
                for (uint32_t h = 0; h < 2; ++h) {
                    uint8_t* d = fragp + ((size_t)rt * ksuper + 2u * kb + h) * kTileBytes +
                                 lane * kLaneBytes;
                    *(uint4*)d = s8[h].v;
                    *(uint4*)(d + 16) = s8[h].w;
                }
            }
            if (sc)
                for (uint32_t i = tid; i < 16u * c.nkb; i += kPWaves * kWave) {
                    const uint32_t kb = i / 16u;
                    sc[((size_t)rt * c.nkb + kb) * 16u + i % 16u] =
                        (uint16_t)fold_scale<F>(cc, rt * 16u + i % 16u, kb);
                }
            __threadfence();
            __syncthreads();
            if (tid == 0)
                __hip_atomic_fetch_add(decoded, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        if (tid == 0)
            while (__hip_atomic_load(decoded, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < 8u)
                __builtin_amdgcn_s_sleep(2);
        __syncthreads();
    }

    __device__ void end(uint32_t bn) const {
        if (threadIdx.x != 0) return;
        unsigned int* const claimed = ctl + 3u * bn;
        if (__hip_atomic_fetch_add(claimed + 2, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u ==
            nbm) {
            __hip_atomic_store(claimed, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(claimed + 1, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(claimed + 2, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
};

template <class F, int RB>
__global__ __launch_bounds__(kPWaves * kWave) R4D_WAVES(6)
void iqr_prefill_kernel(IqrPre<F> pre, const uint8_t* __restrict__ A, const float* __restrict__ Sa,
                        const uint8_t* __restrict__ W, const uint16_t* __restrict__ Sw,
                        uint16_t* __restrict__ C, uint32_t M, uint32_t K, uint32_t N, uint32_t a_ld,
                        uint32_t sa_ld, uint32_t sw_ld) {
    if constexpr (F::SC == kRowF32) {
        constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
        const uint8_t* const sb = pre.c.S;
        gemm_8a8_prefill_body<true, 32, IqrPre<F>, true>(
            A, Sa, W, Sw, C, M, K, N, a_ld, sa_ld, sw_ld, pre, (const float*)sb,
            (const float*)(sb + cf_off(f, N, K)));
    } else {
        gemm_8a8_prefill_body<true, RB>(A, Sa, W, Sw, C, M, K, N, a_ld, sa_ld, sw_ld, pre);
    }
}

/* ---- the forms built: r4d_iqr.h's list ------------------------------------------------ */

#define R4D_IQR_T(k, b, lv, l, lrb, sc) FormT<k, b, lv, l, lrb, sc>

struct Operands {
    const RadTensor *a, *as, *b, *bs, *y;
    long long M, N, K, nkb, a_ld, sa_ld;
    uint32_t word;
    bool chain;   /* the row's `chain` parameter: the chained forms (r4d_chain.h) */
};

/* The forms whose GEMV and 16-token tile have chained instantiations: the computed trellises, and
 * the plain 4-bit table a DFlash2 drafter's linears are held in. Every other form's launch stays
 * plain, which keeps the instantiations to the forms a decode step meets. */
template <class F>
constexpr bool chained_form() {
    return F::KIND == kTrellisMad ||
           (F::KIND == kTable && F::B == 4 && F::LV == 0 && F::L == 0 && F::LRB == 0 && F::SC == 0);
}

int operands(const RadArgs* a, Operands* g) {
    g->a = r4d_opt(a, 0);
    g->as = r4d_opt(a, 1);
    g->b = r4d_opt(a, 2);
    g->bs = r4d_opt(a, 3);
    g->y = r4d_opt(a, 4);
    g->chain = r4d_geti_or(a, "chain", 0) != 0;
    if (!g->a || !g->as || !g->b || !g->bs || !g->y) return R4D_NO(RAD_E_INVAL);
    if (!is_weight_dtype(g->b->dtype) || !is_scale_dtype(g->bs->dtype)) return R4D_NO(RAD_E_DTYPE);
    g->word = g->b->dtype & kWordMask;
    if ((g->bs->dtype & kWordMask) != g->word) return R4D_NO(RAD_E_DTYPE);
    if (g->a->dtype != RAD_I8 || g->as->dtype != RAD_F32 || g->y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (g->b->rank != 2 || g->y->rank < 2 || g->a->rank < 2 || g->as->rank != 2)
        return R4D_NO(RAD_E_SHAPE);
    g->M = g->y->shape[0];
    if (g->M < 1 || r4d_numel(g->y) % g->M) return R4D_NO(RAD_E_SHAPE);
    g->N = r4d_numel(g->y) / g->M;
    g->K = g->b->shape[1];
    if (g->b->shape[0] != g->N || g->N % 16 || g->K % kFK) return R4D_NO(RAD_E_SHAPE);
    g->nkb = g->K / kBlock;
    if (g->a->shape[0] != g->M || r4d_numel(g->a) != g->M * g->K) return R4D_NO(RAD_E_SHAPE);
    if (g->as->shape[0] != g->M || g->as->shape[1] != g->nkb) return R4D_NO(RAD_E_SHAPE);
    g->a_ld = r4d_row_pitch(g->a);
    g->sa_ld = r4d_row_pitch(g->as);
    if (g->a_ld < 0 || g->sa_ld < 0 || r4d_row_pitch(g->y) != g->N) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_aligned(g->a->data, 16) || !r4d_aligned(g->b->data, 256)) return R4D_NO(RAD_E_ALIGN);
    const Form f = form(g->word);
    if (is_trellis(f.kind) && g->N % ((long long)1 << f.lrun)) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

Ctx ctx_of(const Operands& g) {
    Ctx c{};
    c.W = (const uint8_t*)g.b->data;
    c.S = (const uint8_t*)g.bs->data;
    c.N = (uint32_t)g.N;
    c.nkb = (uint32_t)g.nkb;
    const Form f = form(g.word);
    c.run_tiles = is_trellis(f.kind) ? (1u << f.lrun) / 16u : 1u;
    return c;
}

constexpr uint32_t kGemvMaxM = 8;

/* The GEMV, plain into `g.y`, or with `go` the gated fold (a block a gated tile, twice the waves). */
template <class F>
int gemv_launch(const Operands& g, RadStream stream, const GatedOut* go = nullptr) {
    const uint32_t N = (uint32_t)g.N, M = (uint32_t)g.M;
    const Ctx c = ctx_of(g);
    /* Four waves a tile where the tiles alone nearly fill the card, eight on narrow weights (M = 1,
     * 4 bits: gate_up 146.6 / 147.1 us, down 76.0 / 78.4, the 5120 x 6144 outputs 31.4 / 31.9,
     * attn_k 9.5 / 8.8). */
    const uint32_t kw = N >= 4096u ? 4u : 8u;
    const GatedOut gz = go ? *go : GatedOut{};
    uint16_t* const y = go ? nullptr : (uint16_t*)g.y->data;
    int rc = RAD_E_SHAPE;
#define R4D_IQR_GV_E(mt_, kw_, ch_)                                                              \
    rc = go ? R4D_EMIT(stream, (iqr_gemv_kernel<F, mt_, kw_, true, ch_>), dim3(N / 32u),         \
                       dim3(2u * kw_ * kWave), 0, c, (const int8_t*)g.a->data,                    \
                       (const float*)g.as->data, y, N, (uint32_t)g.a_ld, (uint32_t)g.sa_ld, gz)   \
            : R4D_EMIT(stream, (iqr_gemv_kernel<F, mt_, kw_, false, ch_>), dim3(N / 16u),        \
                       dim3(kw_ * kWave), 0, c, (const int8_t*)g.a->data,                         \
                       (const float*)g.as->data, y, N, (uint32_t)g.a_ld, (uint32_t)g.sa_ld, gz)
#define R4D_IQR_GV(mt_, kw_)                                                                     \
    do { if constexpr (chained_form<F>()) {                                                      \
             if (g.chain) R4D_IQR_GV_E(mt_, kw_, r4dchain);                                      \
             else         R4D_IQR_GV_E(mt_, kw_, R4dUnchained);                                  \
         } else {                                                                                \
             R4D_IQR_GV_E(mt_, kw_, R4dUnchained);                                               \
         } } while (0)
#define R4D_IQR_GVK(mt_) do { if (kw == 4u) R4D_IQR_GV(mt_, 4u); else R4D_IQR_GV(mt_, 8u); } while (0)
    switch (M) {
        case 1u: R4D_IQR_GVK(1u); break;
        case 2u: R4D_IQR_GVK(2u); break;
        case 3u: R4D_IQR_GVK(3u); break;
        case 4u: R4D_IQR_GVK(4u); break;
        case 5u: R4D_IQR_GVK(5u); break;
        case 6u: R4D_IQR_GVK(6u); break;
        case 7u: R4D_IQR_GVK(7u); break;
        case 8u: R4D_IQR_GVK(8u); break;
        default: return R4D_NO(RAD_E_SHAPE);
    }
#undef R4D_IQR_GVK
#undef R4D_IQR_GV
#undef R4D_IQR_GV_E
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

/* The forms whose 16-token tile also serves two to four token tiles a block: the computed
 * trellis under one f32 scale a row or a bf16 scale a (row, 128), the trunk's two. */
template <class F>
constexpr bool tiles_form() {
    return F::KIND == kTrellisMad && F::LRB == 0 && (F::SC == kRowF32 || F::SC == kBf16);
}

/* The 16-token tile, plain into `g.y` or gated as gemv_launch; plain, a tiles_form serves up to 64
 * tokens with it, a block multiplying its decoded weight tile against each token tile in turn. */
template <class F>
int m16_launch(const Operands& g, RadStream stream, const GatedOut* go = nullptr) {
    const uint32_t N = (uint32_t)g.N, M = (uint32_t)g.M;
    const Ctx c = ctx_of(g);
    const GatedOut gz = go ? *go : GatedOut{};
    uint16_t* const y = go ? nullptr : (uint16_t*)g.y->data;
    const uint32_t nt = (M + 15u) / 16u;
    if (nt > (go || !tiles_form<F>() ? 1u : 4u)) return R4D_NO(RAD_E_SHAPE);
    int rc = RAD_E_SHAPE;
#define R4D_IQR_M16_G(kw_, ch_)                                                                  \
    rc = R4D_EMIT(stream, (iqr_m16_kernel<F, kw_, true, 1u, ch_>), dim3(N / 32u),                \
                  dim3(2u * kw_ * kWave), 0, c, (const int8_t*)g.a->data,                         \
                  (const float*)g.as->data, y, N, M, (uint32_t)g.a_ld, (uint32_t)g.sa_ld, gz)
#define R4D_IQR_M16_P(kw_, nt_, ch_)                                                             \
    rc = R4D_EMIT(stream, (iqr_m16_kernel<F, kw_, false, nt_, ch_>), dim3(N / 16u),              \
                  dim3(kw_ * kWave), 0, c, (const int8_t*)g.a->data,                              \
                  (const float*)g.as->data, y, N, M, (uint32_t)g.a_ld, (uint32_t)g.sa_ld, gz)
#define R4D_IQR_M16_E(kw_, ch_)                                                                  \
    do { if (go) R4D_IQR_M16_G(kw_, ch_);                                                        \
         else if (nt == 1u) R4D_IQR_M16_P(kw_, 1u, ch_);                                         \
         else if constexpr (tiles_form<F>()) {                                                   \
             if (nt == 2u) R4D_IQR_M16_P(kw_, 2u, ch_);                                          \
             else if (nt == 3u) R4D_IQR_M16_P(kw_, 3u, ch_);                                     \
             else R4D_IQR_M16_P(kw_, 4u, ch_);                                                   \
         } } while (0)
#define R4D_IQR_M16(kw_)                                                                         \
    do { if constexpr (chained_form<F>()) {                                                      \
             if (g.chain) R4D_IQR_M16_E(kw_, r4dchain);                                          \
             else         R4D_IQR_M16_E(kw_, R4dUnchained);                                      \
         } else {                                                                                \
             R4D_IQR_M16_E(kw_, R4dUnchained);                                                   \
         } } while (0)
    if (N >= 4096u) R4D_IQR_M16(4u);
    else R4D_IQR_M16(8u);
#undef R4D_IQR_M16
#undef R4D_IQR_M16_E
#undef R4D_IQR_M16_P
#undef R4D_IQR_M16_G
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

/* A computed trellis's GEMV pays its decode in VALU work beside every token's dot and fold, the
 * tiles decode a unit once for all tokens. Measured on gate_up (34816 x 5120): at one token the
 * GEMV wins at every width (2 / 4 / 6 bits 92.6 / 145.2 / 214.6 us, the 16-token tile at 4 / 6 bits
 * 150.0 / 215.7); at two only on 2- and 3-bit codes under one f32 scale a row, its values dotted
 * straight from their hashes (128.4 / 132.4 against the tile's 137.7 / 138.5; 4 bits 152.8 against
 * 150.1); from three tokens the tile. */
template <class F>
constexpr uint32_t gemv_max() {
    return F::KIND != kTrellisMad ? kGemvMaxM : F::B <= 3u && F::SC == kRowF32 ? 2u : 1u;
}

template <class F>
int m64_launch(const Operands& g, void* scratch, long long scratch_bytes, int split_n,
               RadStream stream) {
    const uint32_t N = (uint32_t)g.N, K = (uint32_t)g.K, M = (uint32_t)g.M;
    if (M <= gemv_max<F>()) return gemv_launch<F>(g, stream);
    if (M <= (tiles_form<F>() ? 64u : 16u) && ((uintptr_t)g.y->data & 15u) == 0u)
        return m16_launch<F>(g, stream);
    /* At most eight waves a block: each has its own transpose tile in LDS. The wave count changes
     * no output bit (r4d_gemm_fp8a8.hip's narrow_mw). */
    const uint32_t mw = std::min(r4d_8a8_narrow_mw(N, M), 8u);
    const uint32_t gx = (N + mw * 16u - 1) / (mw * 16u);
    uint32_t ck = 0;
    uint32_t ksplit = r4d_8a8_narrow_ksplit(split_n > 0 ? (uint32_t)split_n : N, K, M, &ck,
                                            kIqrSplitWaves);
    const long long need = ksplit > 1 ? (long long)ksplit * M * N * 4 : 0;
    if (need && (!scratch || scratch_bytes < need)) { ksplit = 1; ck = 0; }
    const uint32_t nt = M > 48 ? 4u : M > 32 ? 3u : M > 16 ? 2u : 1u;
    unsigned int* const ctr =
        ksplit > 1 && gx <= 8192u ? r4d_8a8_split_ctr(stream, g.y->data) : nullptr;
    void* const dst = ksplit > 1 ? scratch : g.y->data;
    const Ctx c = ctx_of(g);
    const dim3 grid(gx, 1, ksplit);
    int rc = RAD_E_SHAPE;
#define R4D_IQR_M64_E(mw_, nt_, ch_)                                                             \
    rc = R4D_EMIT(stream, (iqr_m64_kernel<F, nt_, mw_, ch_>), grid, dim3(mw_ * kWave), 0, c,     \
                  (const uint8_t*)g.a->data, (const float*)g.as->data, dst, (uint16_t*)g.y->data, \
                  ctr, N, K, M, (uint32_t)g.a_ld, (uint32_t)g.sa_ld, ck)
#define R4D_IQR_M64(mw_, nt_)                                                                    \
    do { if constexpr (chained_form<F>()) {                                                      \
             if (g.chain) R4D_IQR_M64_E(mw_, nt_, r4dchain);                                     \
             else         R4D_IQR_M64_E(mw_, nt_, R4dUnchained);                                 \
         } else {                                                                                \
             R4D_IQR_M64_E(mw_, nt_, R4dUnchained);                                              \
         } } while (0)
#define R4D_IQR_M64_NT(mw_)                                                                      \
    do { if (nt == 4u) R4D_IQR_M64(mw_, 4u); else if (nt == 3u) R4D_IQR_M64(mw_, 3u);           \
         else if (nt == 2u) R4D_IQR_M64(mw_, 2u); else R4D_IQR_M64(mw_, 1u); } while (0)
    if (mw == 4u) R4D_IQR_M64_NT(4u);
    else if (mw == 8u) R4D_IQR_M64_NT(8u);
    else return R4D_NO(RAD_E_SHAPE);
#undef R4D_IQR_M64_NT
#undef R4D_IQR_M64
#undef R4D_IQR_M64_E
    if (rc != RAD_OK || hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    if (ksplit > 1 && !ctr) {
        const uint32_t n = M * N;
        if (R4D_EMIT(stream, iqr_splitk_reduce, dim3((n + 255) / 256), dim3(256), 0,
                     (const float*)scratch, (uint16_t*)g.y->data, n, ksplit) != RAD_OK)
            return R4D_NO(RAD_E_DEVICE);
    }
    return RAD_OK;
}

/* The prefill's scratch: the int8 fragment plane, then (unless the scale buffer already is the
 * int8 tile's) its bf16 tile-major scales. */
bool own_scales_serve(const Form& f) {
    return (f.scale == kBf16 && f.lrb == 0) || f.scale == kRowF32;   /* kRowF32: read in place */
}
long long tiled_scratch(const Form& f, long long N, long long K) {
    return align256(N * K) + (own_scales_serve(f) ? 0 : align256(N * (K / kBlock) * 2));
}

template <class F>
int tiled_launch(const Operands& g, void* scratch, long long scratch_bytes, RadStream stream) {
    const Form f = form(g.word);
    if (!scratch || scratch_bytes < tiled_scratch(f, g.N, g.K)) return R4D_NO(RAD_E_SHAPE);
    if (g.N % kPBN || (uintptr_t)g.y->data & 15u) return R4D_NO(RAD_E_SHAPE);
    unsigned int* const ctl = r4d_8a8_split_ctr(stream, g.y->data);
    if (!ctl || 3u * (uint32_t)(g.N / kPBN) > 8192u) return R4D_NO(RAD_E_DEVICE);
    IqrPre<F> pre;
    pre.c = ctx_of(g);
    pre.fragp = (uint8_t*)scratch;
    pre.sc = own_scales_serve(f) ? nullptr : (uint16_t*)(pre.fragp + align256(g.N * g.K));
    pre.ctl = ctl;
    const dim3 grid((uint32_t)((g.M + kPBM - 1) / kPBM), (uint32_t)(g.N / kPBN), 1);
    pre.nbm = grid.x;
    const uint16_t* const ws = pre.sc ? pre.sc : (const uint16_t*)g.bs->data;
    int rc;
#define R4D_IQR_PF(rb_)                                                                          \
    rc = R4D_EMIT(stream, (iqr_prefill_kernel<F, rb_>), grid, dim3(kPWaves * kWave), 0, pre,     \
                  (const uint8_t*)g.a->data, (const float*)g.as->data, (const uint8_t*)pre.fragp, \
                  ws, (uint16_t*)g.y->data, (uint32_t)g.M, (uint32_t)g.K, (uint32_t)g.N,          \
                  (uint32_t)g.a_ld, (uint32_t)g.sa_ld, (uint32_t)g.nkb)
    /* F's own lrb is the word's (kernel_word keeps it), so one tile form is built a form. */
    if constexpr (F::LRB >= 5) R4D_IQR_PF(32);
    else if constexpr (F::LRB >= 3) R4D_IQR_PF(8);
    else R4D_IQR_PF(0);
#undef R4D_IQR_PF
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

}  // namespace

#if R4D_IQR_PART == 0
extern "C" int64_t r4d_rad_gemm_iqr_m64_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), N = r4d_geti_or(a, "N", 0),
                    K = r4d_geti_or(a, "K", 0), sn = r4d_geti_or(a, "split_n", 0);
    if (M < 1 || N < 1 || K < 1 || M > 64) return 0;
    uint32_t ck = 0;
    const uint32_t k2 = r4d_8a8_narrow_ksplit((uint32_t)(sn > 0 ? sn : N), (uint32_t)K,
                                              (uint32_t)M, &ck, kIqrSplitWaves);
    return k2 > 1 ? (int64_t)k2 * M * N * 4 : 0;
}
#endif

#define R4D_IQR_CAT2(a, b) a##b
#define R4D_IQR_CAT(a, b) R4D_IQR_CAT2(a, b)

namespace {

/* A form's launch where this part builds it, else a refusal no call reaches: iqr_part sends each
 * word to the one part whose chain holds it. */
template <class F>
int m64_part(const Operands& g, void* scratch, long long scratch_bytes, int split_n, RadStream s) {
    if constexpr (in_part<F>()) return m64_launch<F>(g, scratch, scratch_bytes, split_n, s);
    else return R4D_NO(RAD_E_UNSUPPORTED);
}

}  // namespace

extern "C" int R4D_IQR_CAT(r4d_iqr_m64_p, R4D_IQR_PART)(const RadArgs* a, RadStream s) {
    Operands g;
    const int rc = operands(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.M > 64) return R4D_NO(RAD_E_SHAPE);
    const uint32_t kw = kernel_word(g.word);
    const int sn = (int)r4d_geti_or(a, "split_n", 0);
#define R4D_IQR_GO(...)                                                                          \
    if (in_part<R4D_IQR_T(__VA_ARGS__)>() && kw == R4D_IQR_T(__VA_ARGS__)::word())                \
        return m64_part<R4D_IQR_T(__VA_ARGS__)>(g, a->scratch, (long long)a->scratch_bytes, sn, s);
    R4D_IQR_FORMS(R4D_IQR_GO)
#undef R4D_IQR_GO
    return R4D_NO(RAD_E_UNSUPPORTED);
}

namespace {

/* gemm_nt_q_gated's operands at i8a8h: [0] a [1] a_scale [2] b [3] b_scale [4] q [5] scale
 * [6] out_bf16. The weight is one [N, K] IQR plane, gate rows then up rows. */
int gated_operands(const RadArgs* a, Operands* g, GatedOut* go) {
    g->a = r4d_opt(a, 0);
    g->as = r4d_opt(a, 1);
    g->b = r4d_opt(a, 2);
    g->bs = r4d_opt(a, 3);
    g->y = nullptr;
    const RadTensor* q = r4d_opt(a, 4);
    const RadTensor* sc = r4d_opt(a, 5);
    const RadTensor* yb = r4d_opt(a, 6);
    if (!g->a || !g->as || !g->b || !g->bs || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (!is_weight_dtype(g->b->dtype) || !is_scale_dtype(g->bs->dtype)) return R4D_NO(RAD_E_DTYPE);
    g->word = g->b->dtype & kWordMask;
    if ((g->bs->dtype & kWordMask) != g->word) return R4D_NO(RAD_E_DTYPE);
    if (g->a->dtype != RAD_I8 || g->as->dtype != RAD_F32 || q->dtype != RAD_I8 ||
        sc->dtype != RAD_F32 || (yb && yb->dtype != RAD_BF16))
        return R4D_NO(RAD_E_DTYPE);
    if (g->b->rank != 2 || g->a->rank < 2 || g->as->rank != 2) return R4D_NO(RAD_E_SHAPE);
    g->M = g->a->shape[0];
    g->N = g->b->shape[0];
    g->K = g->b->shape[1];
    const long long n = g->N / 2;
    if (g->M < 1 || g->M > 16 || g->N % 256 || g->K % kFK) return R4D_NO(RAD_E_SHAPE);
    g->nkb = g->K / kBlock;
    if (r4d_numel(g->a) != g->M * g->K || g->as->shape[0] != g->M || g->as->shape[1] != g->nkb)
        return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(q) != g->M * n || r4d_numel(sc) != g->M * (n / kBlock) ||
        (yb && r4d_numel(yb) != g->M * n))
        return R4D_NO(RAD_E_SHAPE);
    g->a_ld = r4d_row_pitch(g->a);
    g->sa_ld = r4d_row_pitch(g->as);
    const long long q_ld = r4d_row_pitch(q), s_ld = r4d_row_pitch(sc), y_ld = yb ? r4d_row_pitch(yb) : 0;
    if (g->a_ld < 0 || g->sa_ld < 0 || q_ld < 0 || s_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_aligned(g->a->data, 16) || !r4d_aligned(g->b->data, 256) || !r4d_aligned(q->data, 4) ||
        q_ld % 4)
        return R4D_NO(RAD_E_ALIGN);
    const Form f = form(g->word);
    if (is_trellis(f.kind) && g->N % ((long long)1 << f.lrun)) return R4D_NO(RAD_E_SHAPE);
    go->Q = (int8_t*)q->data;
    go->S = (float*)sc->data;
    go->Yb = yb ? (uint16_t*)yb->data : nullptr;
    go->n = (uint32_t)n;
    go->q_ld = (uint32_t)q_ld;
    go->s_ld = (uint32_t)s_ld;
    go->y_ld = (uint32_t)y_ld;
    return RAD_OK;
}

template <class F>
int gated_launch(const Operands& g, const GatedOut& go, RadStream stream) {
    if constexpr (in_part<F>()) {
        if (g.M <= gemv_max<F>()) return gemv_launch<F>(g, stream, &go);
        return m16_launch<F>(g, stream, &go);
    } else {
        return R4D_NO(RAD_E_UNSUPPORTED);
    }
}

template <class F>
int tiled_part(const Operands& g, void* scratch, long long scratch_bytes, RadStream s) {
    if constexpr (in_part<F>()) return tiled_launch<F>(g, scratch, scratch_bytes, s);
    else return R4D_NO(RAD_E_UNSUPPORTED);
}

}  // namespace

#if R4D_IQR_PART == 0
extern "C" int64_t r4d_rad_gemm_iqr_gated_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), N = r4d_geti_or(a, "N", 0);
    if (M < 1 || N < 2) return 0;
    return (int64_t)align256(M * (N / 2) * 2);
}
#endif

extern "C" int R4D_IQR_CAT(r4d_iqr_gated_p, R4D_IQR_PART)(const RadArgs* a, RadStream s) {
    Operands g;
    GatedOut go;
    const int rc = gated_operands(a, &g, &go);
    if (rc != RAD_OK) return rc;
    if (!a->scratch || (long long)a->scratch_bytes < align256(g.M * (g.N / 2) * 2))
        return R4D_NO(RAD_E_SCRATCH);
    go.H = (uint16_t*)a->scratch;
    go.cnt = r4d_8a8_split_ctr(s, go.Q);
    if (!go.cnt || go.n / 128u > 8192u) return R4D_NO(RAD_E_DEVICE);
    const uint32_t kw = kernel_word(g.word);
#define R4D_IQR_GO(...)                                                                          \
    if (in_part<R4D_IQR_T(__VA_ARGS__)>() && kw == R4D_IQR_T(__VA_ARGS__)::word())                \
        return gated_launch<R4D_IQR_T(__VA_ARGS__)>(g, go, s);
    R4D_IQR_FORMS(R4D_IQR_GO)
#undef R4D_IQR_GO
    return R4D_NO(RAD_E_UNSUPPORTED);
}

#if R4D_IQR_PART == 0
extern "C" int64_t r4d_rad_gemm_iqr_tiled_scratch(const RadArgs* a) {
    const long long N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
    if (N < 1 || K < 1) return 0;
    /* The scale part's size is the form's; the bound covers every form. */
    Form f;
    f.scale = kE4m3Tensor;
    return (int64_t)tiled_scratch(f, N, K);
}
#endif

extern "C" int R4D_IQR_CAT(r4d_iqr_tiled_p, R4D_IQR_PART)(const RadArgs* a, RadStream s) {
    Operands g;
    const int rc = operands(a, &g);
    if (rc != RAD_OK) return rc;
    if (g.N % kBlock) return R4D_NO(RAD_E_SHAPE);
    const uint32_t kw = kernel_word(g.word);
#define R4D_IQR_GO(...)                                                                          \
    if (in_part<R4D_IQR_T(__VA_ARGS__)>() && kw == R4D_IQR_T(__VA_ARGS__)::word())                \
        return tiled_part<R4D_IQR_T(__VA_ARGS__)>(g, a->scratch, (long long)a->scratch_bytes, s);
    R4D_IQR_FORMS(R4D_IQR_GO)
#undef R4D_IQR_GO
    return R4D_NO(RAD_E_UNSUPPORTED);
}

#if R4D_IQR_PART == 0
/* The rows' entry points: each call goes to the part that builds its weight's form, which parses
 * and checks the operands itself. */
using IqrPartFn = int (*)(const RadArgs*, RadStream);
#define R4D_IQR_PARTS_OF(row)                                                                    \
    extern "C" int row##1(const RadArgs*, RadStream);                                            \
    extern "C" int row##2(const RadArgs*, RadStream);                                            \
    extern "C" int row##3(const RadArgs*, RadStream);
R4D_IQR_PARTS_OF(r4d_iqr_m64_p)
R4D_IQR_PARTS_OF(r4d_iqr_gated_p)
R4D_IQR_PARTS_OF(r4d_iqr_tiled_p)
#undef R4D_IQR_PARTS_OF
static_assert(kIqrParts == 4, "the entry points below name every part");

static int iqr_by_part(const IqrPartFn (&fn)[kIqrParts], const RadArgs* a, RadStream s) {
    const RadTensor* b = r4d_opt(a, 2);
    return fn[b ? iqr_part(kernel_word(b->dtype & kWordMask)) : 0](a, s);
}

extern "C" int r4d_rad_gemm_iqr_m64(const RadArgs* a, RadStream s) {
    static constexpr IqrPartFn fn[] = { r4d_iqr_m64_p0, r4d_iqr_m64_p1, r4d_iqr_m64_p2,
                                        r4d_iqr_m64_p3 };
    return iqr_by_part(fn, a, s);
}

extern "C" int r4d_rad_gemm_iqr_gated(const RadArgs* a, RadStream s) {
    static constexpr IqrPartFn fn[] = { r4d_iqr_gated_p0, r4d_iqr_gated_p1, r4d_iqr_gated_p2,
                                        r4d_iqr_gated_p3 };
    return iqr_by_part(fn, a, s);
}

extern "C" int r4d_rad_gemm_iqr_tiled(const RadArgs* a, RadStream s) {
    static constexpr IqrPartFn fn[] = { r4d_iqr_tiled_p0, r4d_iqr_tiled_p1, r4d_iqr_tiled_p2,
                                        r4d_iqr_tiled_p3 };
    return iqr_by_part(fn, a, s);
}
#endif
