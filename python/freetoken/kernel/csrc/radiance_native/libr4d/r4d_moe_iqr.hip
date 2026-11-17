// moe_gemm_q over routed experts stored as IQR weights (r4d_iqr.h): every expert's codes in VRAM at
// their own width, decoded where they are read.
//
// One kernel, the dense 16-token narrow tile (r4d_gemm_iqr.hip, iqr_m16_kernel) made run-aware. A
// block serves one run of sorted rows of one expert against a group of 16-row tiles of that
// expert's weight, and a wave decodes each 128-K unit of its tile where it reads it. Decode is the
// cost (about 300 vector instructions a unit and lane, whatever the code width), so the kernel is
// shaped to decode each unit once and to spend as little as it can beside that:
//
//   a run of one or two rows   takes no tile at all (dot_run): each decoded value's masked hash is
//                 dotted with its column's activation byte, which sums the hash's bytes and
//                 multiplies in one instruction. Nothing is packed, transposed or multiplied for
//                 token columns no row owns. A decode step's runs are almost all of this kind.
//   a longer run  transposes the unit into fragment order through the wave's LDS tile and runs its
//                 eight WMMA steps against each 16-row token tile the run reaches, the activation's
//                 fragments read from global memory where it sits in L2.
//
// There is no shared staging and no block barrier in the K loop; where a tile's K is shared between
// waves their partials are summed through LDS in ascending wave order. The output is the product
// within f32 rounding.
//
// Two grids over x, and the weight's tile groups over y:
//   decode   inside r4d_moe.hip's decode band. x is (16-row tile of the sorted rows, run within
//            it), T slots in all since a tile of r rows holds at most r runs. The block's first
//            wave finds the run from the expert offsets in one round of loads and the others take
//            it from LDS, so a decode step issues no map. A wave takes a tile and a block four
//            tiles; under twenty routed rows four waves share a wide-K tile (gate_up), and through
//            the middle of the band a narrow-K wave (down) takes two tiles in turn.
//   map      above it (a prefill chunk). x is an entry of the run map moe_blockmap_kernel builds in
//            the op's scratch, (expert, first row) at 64 rows an entry: a block decodes its
//            expert's tiles once for up to four token tiles, a wave a weight tile down all of K.
//
// The forms built are the computed trellis under one f32 scale a row (TrellisMad, kRowF32), IQR-V4's
// form for a feed-forward weight: the fold is the block's factor times the activation's scale, one
// fma a slot, and a row's scale meets its sum once at the store.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"
#include "r4d_iqr.h"
#include "r4d_iqr_dec.h"
#include "r4d_moe_sort.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

/* r4d_moe.hip: the run map of a grouped GEMM at `fm` rows a block, built in `scratch` by one
 * launch; its entry count bound, and the bytes it takes. */
long long r4d_moe_blockmap_cap(long long T, long long n_expert, long long fm);
long long r4d_moe_blockmap_bytes(long long cap);
int r4d_moe_blockmap_emit(RadStream s, const int32_t* off, uint32_t n_expert, uint32_t T,
                          uint32_t fm, uint32_t cap, uint32_t* map);
/* Whether T sorted rows over n_expert experts are served by the decode grid. */
bool r4d_moe_in_decode_band(long long T, long long n_expert);

namespace {

using namespace r4d_fp8;
using namespace r4d_iqr;
using namespace r4d_iqr_dev;
using r4d_moesort::kMaxExpert;
using r4d_moesort::wave_incl_scan_i32;

constexpr uint32_t kWave = 32, kBlock = 128;
/* Sorted rows a block serves: one WMMA tile of tokens. */
constexpr uint32_t kRows = 16;
/* A run-map entry that owns no expert (r4d_moe.hip's kMapTail): dropped slots, zeroed. */
constexpr uint32_t kTail = 0xffffffffu;
/* The biased fold of the int8 tiles under one scale a block (r4d_gemm_8a8_prefill.h): a unit's
 * int32 sums start at 0x4B400000, so a slot read as an f32 is 12582912 plus its sum and one fma
 * folds it; 12582912 times the folded factors leaves every kRbFlush units and at the store. */
constexpr int kRbBias = 0x4B400000;
constexpr float kRbBiasF = 12582912.0f;
constexpr uint32_t kRbFlush = 8;

#define R4D_MOEIQR_CHK() do { hipError_t e_ = hipGetLastError();                    \
                              if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

/* One class of experts (r4d_moe.hip's MoeCls): expert e is entry e >> cshift of class
 * e & (2^cshift - 1), each entry the expert's weight buffer and scale buffer in r4d_iqr.h's
 * layout; `n` its rows and `k` its K on this rank. */
struct IqrCls {
    const void* const* w;
    const void* const* s;
    uint32_t n, k;
    uint32_t fc, pad_;   /* its rows a part: n / parts */
};

/* Routed slot f's token, f / top_k, by the launch's reciprocal `inv_k` (0: top_k is 1): the high
 * half of f * (2^32 / top_k + 1) is the quotient for every f below 2^32 / top_k, and a slot index
 * is far below that. One multiply where a division is thirty instructions a lane. */
__device__ __forceinline__ uint32_t slot_token(uint32_t f, uint32_t inv_k) {
    return inv_k ? __umulhi(f, inv_k) : f;
}

/* A wave-uniform pointer held in scalar registers: its loads take their base from there and a
 * lane's offset beside it, with no 64-bit address arithmetic in the lanes. */
template <class T>
__device__ __forceinline__ T* uniform(T* p) {
    const uint64_t u = (uint64_t)p;
    /* the builtin returns an int: through uint32_t, or the low half sign-extends over the high */
    const uint64_t lo = (uint32_t)__builtin_amdgcn_readfirstlane((uint32_t)u);
    const uint64_t hi = (uint32_t)__builtin_amdgcn_readfirstlane((uint32_t)(u >> 32));
    return (T*)(hi << 32 | lo);
}

/* A run of NR rows, NR at most kDotRows, without the WMMA tile: lane l holds columns 4l .. 4l + 3
 * of each unit, and a value's masked hash dotted with its column's activation byte, repeated in
 * each byte, sums the hash's bytes and multiplies in one instruction (the GEMV of
 * r4d_gemm_iqr.hip). Nothing is packed, transposed through LDS or multiplied for token columns no
 * row owns: a unit costs its decode and one dot a value and row. A row's dot starts at 0x4B400000
 * less 126 times the lane's activation sum, so read as an f32 it is 12582912 plus the lane's sum
 * and one fma folds it; 12582912 times the factors leaves at the end. Returns lane l's share: row
 * m's sum over the wave of tile row red16_row(l), in v[m]. */
constexpr uint32_t kDotRows = 2;
template <class F, uint32_t KW, uint32_t NR>
__device__ __forceinline__ void dot_run(const Ctx& c, uint32_t rt, uint32_t kph, uint32_t lane,
                                        const int8_t* const (&xrow)[kDotRows],
                                        const float* const (&xsr)[kDotRows], uint32_t ash,
                                        float (&v)[kDotRows]) {
    using D = Dec<F>;
    constexpr Form ff{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
    const uint32_t* ab = (const uint32_t*)(c.W + aux_off(ff, c.N, (int64_t)c.nkb * 128));
    const float* cf = (const float*)(c.S + cf_off(ff, c.N, (int64_t)c.nkb * 128));
    const uint32_t A = ab[0], Bc = ab[1];
    float acc[NR][16], fsum[NR];
#pragma unroll
    for (uint32_t m = 0; m < NR; ++m) {
        fsum[m] = 0.0f;
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) acc[m][q] = 0.0f;
    }
    uint32_t nflush = 0;
    typename D::Raw cur = D::fetch(c, rt, kph, lane, kph < c.nkb), nxt;
    uint32_t xc[NR], xn[NR];
#pragma unroll
    for (uint32_t m = 0; m < NR; ++m)
        xc[m] = kph < c.nkb ? *(const uint32_t*)(xrow[m] + kph * kBlock + 4u * lane) : 0u;
    for (uint32_t kb = kph; kb < c.nkb; kb += KW) {
        if (kb + KW < c.nkb) {
            nxt = D::fetch(c, rt, kb + KW, lane, true);
#pragma unroll
            for (uint32_t m = 0; m < NR; ++m)
                xn[m] = *(const uint32_t*)(xrow[m] + (kb + KW) * kBlock + 4u * lane);
        }
        uint32_t xs[NR][4];
        float s[NR];
        int b0[NR];
        const float fb = cf[kb];
#pragma unroll
        for (uint32_t m = 0; m < NR; ++m) {
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e)
                xs[m][e] = __builtin_amdgcn_perm(0u, xc[m], e * 0x01010101u);
            s[m] = fb * xsr[m][kb >> ash];
            fsum[m] += s[m];
            b0[m] = kRbBias -
                    __builtin_amdgcn_sudot4(true, (int)xc[m], false, 0x7E7E7E7E, 0, false);
        }
        uint32_t cw[4][D::NW], mw[4][D::NW];
        D::columns(cur, cw);
        D::mids(cw, mw);
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) {
            int d[NR];
#pragma unroll
            for (uint32_t m = 0; m < NR; ++m) d[m] = b0[m];
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t h = (__umul24(D::state(cw, mw, q, e), A) + Bc) & 0x3F3F3F3Fu;
#pragma unroll
                for (uint32_t m = 0; m < NR; ++m)
                    d[m] = __builtin_amdgcn_sudot4(false, (int)h, true, (int)xs[m][e], d[m], false);
            }
            /* the stream's position q is tile row 15 - q */
#pragma unroll
            for (uint32_t m = 0; m < NR; ++m)
                acc[m][15u - q] = __builtin_fmaf(__int_as_float(d[m]), s[m], acc[m][15u - q]);
        }
        if (++nflush == kRbFlush) {
#pragma unroll
            for (uint32_t m = 0; m < NR; ++m) {
#pragma unroll
                for (uint32_t q = 0; q < 16; ++q)
                    acc[m][q] = __builtin_fmaf(-kRbBiasF, fsum[m], acc[m][q]);
                fsum[m] = 0.0f;
            }
            nflush = 0;
        }
        cur = nxt;
#pragma unroll
        for (uint32_t m = 0; m < NR; ++m) xc[m] = xn[m];
    }
#pragma unroll
    for (uint32_t m = 0; m < NR; ++m) {
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) acc[m][q] = __builtin_fmaf(-kRbBiasF, fsum[m], acc[m][q]);
        v[m] = red16(acc[m], lane);
    }
#pragma unroll
    for (uint32_t m = NR; m < kDotRows; ++m) v[m] = 0.0f;
}

/* KW waves across a tile's K, RTB tiles a block: a wide-K projection (gate_up) takes four waves a
 * tile, a narrow one (down, two or three 128-K blocks a rank) one wave a tile and four tiles a
 * block. MAP: the run-map grid, whose block serves NT token tiles of its run. */
#define R4D_MOEIQR_PARAMS                                                                         \
    const int8_t* __restrict__ X, const float* __restrict__ XS, const IqrCls c_0,                 \
    const IqrCls c_1, const IqrCls c_2, const IqrCls c_3, const int32_t* __restrict__ sorted,     \
    const int32_t* __restrict__ off, uint16_t* __restrict__ Y, const uint32_t* __restrict__ map,  \
    uint32_t T, uint32_t fy, uint32_t n_expert, uint32_t inv_k, uint32_t x_ld, uint32_t xs_ld,    \
    uint32_t y_ld, uint32_t mode, uint32_t run_tiles, uint32_t inv_ft
template <class F, uint32_t KW, uint32_t RTB, bool MAP, uint32_t NT, uint32_t TPW>
__device__ __forceinline__ void moe_iqr_body(R4D_MOEIQR_PARAMS) {
    using D = Dec<F>;
    static_assert(D::MAD && F::SC == kRowF32 && F::LRB == 0,
                  "the routed kernel reads a computed trellis under one f32 scale a row");
    static_assert(NT == 1u || (MAP && KW == 1u),
                  "several token tiles a block: the run map's grid, a wave a weight tile");
    static_assert(TPW == 1u || KW == 1u, "several weight tiles a wave: a wave a tile's whole K");
    constexpr uint32_t NW = KW * RTB, NTHR = NW * kWave, kCols = RTB * TPW * 16u;
    /* `mode`: bit 0 `a_sorted`, bits 1-2 the class shift, bit 3 the activation's scale per 256 of
     * K, bits 8+ `parts` (r4d_moe.hip's). */
    const uint32_t a_sorted = mode & 1u, cshift = (mode >> 1) & 3u, ash = (mode >> 3) & 1u;
    const uint32_t parts = mode >> 8;
    /* the waves' transpose tiles, then their partials (8 x 32 floats a wave, in its own tile) */
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[NW * kTrBytes];
    static_assert(8u * kWave * 4u <= kTrBytes, "a wave's partials in its transpose tile");
    __shared__ uint32_t s_sel[4];
    auto red = [&](uint32_t w) { return (float*)(s_tr + w * kTrBytes); };
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1);
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const uint32_t ycol0 = blockIdx.y * kCols;
    /* Rows [ra, rb) of the block's columns, zeroed: dropped slots, which no run owns. */
    auto zero_rows = [&](uint32_t ra, uint32_t rb) {
        for (uint32_t x = tid; x < (rb - ra) * kCols; x += NTHR)
            Y[(size_t)(ra + x / kCols) * y_ld + ycol0 + x % kCols] = 0;
    };

    /* ---- the block's run: expert e, sorted rows [rs, re). */
    uint32_t e, rs, re;
    if constexpr (MAP) {
        if (blockIdx.x >= map[0]) return;
        e = map[1 + 2 * blockIdx.x];
        rs = map[2 + 2 * blockIdx.x];
        if (e == kTail) {
            zero_rows(rs, rs + kRows * NT < T ? rs + kRows * NT : T);
            return;
        }
        const uint32_t oe = (uint32_t)off[e + 1];
        re = oe < T ? oe : T;
        if (re > rs + kRows * NT) re = rs + kRows * NT;
    } else {
        /* The (tile, run) grid. The block's first wave finds the run and the others take it from
         * LDS after one barrier: each of its lanes holds kMaxExpert / 32 consecutive experts'
         * offsets clamped to the tile's rows, so an expert's run meets the tile where its clamped
         * bounds differ and those bounds are the run's rows there. The runs are numbered in expert
         * order (a lane's own, then a running count over the lanes before it), and the lane holding
         * run `jrun` hands over its expert and bounds. Offsets past the last expert read the last
         * one's end: an empty run. */
        constexpr uint32_t NPT = kMaxExpert / kWave;
        static_assert(kMaxExpert % kWave == 0 && NPT <= 16, "a lane's experts in one mask");
        const uint32_t r0 = (blockIdx.x / kRows) * kRows, jrun = blockIdx.x % kRows;
        if (r0 >= T) return;
        const uint32_t rN = r0 + kRows < T ? r0 + kRows : T;
        if (wave == 0u) {
            uint32_t cv[NPT + 1];
#pragma unroll
            for (uint32_t k = 0; k <= NPT; ++k) {
                const uint32_t x = lane * NPT + k;
                const uint32_t o = (uint32_t)off[x < n_expert ? x : n_expert];
                cv[k] = o < r0 ? r0 : o > rN ? rN : o;
            }
            uint32_t pm = 0;
#pragma unroll
            for (uint32_t k = 0; k < NPT; ++k) pm |= (cv[k] < cv[k + 1] ? 1u : 0u) << k;
            const uint32_t cnt = (uint32_t)__builtin_popcount(pm);
            const uint32_t inc = (uint32_t)wave_incl_scan_i32((int32_t)cnt, lane);
            if (lane == kWave - 1) s_sel[3] = inc;
            const uint32_t before = inc - cnt;
            if (jrun >= before && jrun < before + cnt) {
                /* the lane's (jrun - before)-th run: a search of its mask by halves */
                uint32_t rem = jrun - before, pos = 0, bits = pm;
#pragma unroll
                for (uint32_t h = NPT / 2u; h; h >>= 1) {
                    const uint32_t c = (uint32_t)__builtin_popcount(bits & ((1u << h) - 1u));
                    const bool up = rem >= c;
                    rem -= up ? c : 0u;
                    pos += up ? h : 0u;
                    bits = up ? bits >> h : bits;
                }
                const uint32_t ex = lane * NPT + pos;
                const uint32_t oa = (uint32_t)off[ex], ob = (uint32_t)off[ex + 1];
                s_sel[0] = ex;
                s_sel[1] = oa > r0 ? oa : r0;
                s_sel[2] = ob < rN ? ob : rN;
            }
        }
        lds_barrier();
        /* The tile's dropped slots, by one of its blocks. */
        const uint32_t live = (uint32_t)off[n_expert];
        if (jrun == 0 && live < rN) zero_rows(r0 > live ? r0 : live, rN);
        if (jrun >= s_sel[3]) return;
        e = s_sel[0];
        rs = s_sel[1];
        re = s_sel[2];
    }
    /* The run is the block's: in scalar registers, and with it the expert's buffers. */
    e = __builtin_amdgcn_readfirstlane(e);
    rs = __builtin_amdgcn_readfirstlane(rs);
    re = __builtin_amdgcn_readfirstlane(re);
    const uint32_t nrows = re - rs;

    /* ---- the expert's class and buffers, and this wave's tile of it. */
    const uint32_t cls = e & ((1u << cshift) - 1u);
    const IqrCls& C = cls == 0u ? c_0 : cls == 1u ? c_1 : cls == 2u ? c_2 : c_3;
    Ctx c{};
    c.W = uniform((const uint8_t*)C.w[e >> cshift]);
    c.S = uniform((const uint8_t*)C.s[e >> cshift]);
    c.tab = nullptr;
    c.N = C.n;
    c.nkb = C.k / kBlock;
    c.run_tiles = run_tiles;
    /* A part of the output is as wide as the widest class's; a narrower class fills each part's
     * leading columns and the rest are zero (r4d_moe.hip's `parts`). */
    const uint32_t rtb = wave / KW, kph = wave % KW;
    /* `fy` is a part's rows at the widest class and `inv_ft` the reciprocal of its tiles (0: one
     * part), the launch's, so the wave's part and tile cost a multiply, not two divisions. */
    const uint32_t fc = C.fc, ft = fy / 16u;

    /* ---- the run's activation rows, the same for every weight tile of the wave. A run of one or
     * two rows takes the dot path, whose rows are the wave's (drow); a longer one the tile path:
     * token tile j is the run's rows 16 j .. 16 j + 15, and a lane's token in it is its fragment
     * column's sorted row, the run's last row past the run (those columns are not stored).
     * gate_up gathers the token's row, down reads the sorted row. */
    const bool dot = nrows <= kDotRows;
    const uint32_t col = frag_c(lane);
    const int8_t* drow[kDotRows];
    const float* dsr[kDotRows];
    bool dok[kDotRows];
    const int8_t* xrow[NT];
    const float* xsr[NT];
    bool okr[NT];
    if (dot) {
#pragma unroll
        for (uint32_t m = 0; m < kDotRows; ++m) {
            const uint32_t sr = rs + (m < nrows ? m : nrows - 1u);
            const int32_t fr = (int32_t)__builtin_amdgcn_readfirstlane((uint32_t)sorted[sr]);
            const uint32_t tk = a_sorted ? sr : (fr < 0 ? 0u : slot_token((uint32_t)fr, inv_k));
            dok[m] = a_sorted || fr >= 0;
            drow[m] = X + (size_t)tk * x_ld;
            dsr[m] = XS + (size_t)tk * xs_ld;
        }
    } else {
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const uint32_t rj = j * kRows + col;
            const uint32_t srow = rs + (rj < nrows ? rj : nrows - 1u);
            const int32_t f = sorted[srow];
            const uint32_t tok = a_sorted ? srow : (f < 0 ? 0u : slot_token((uint32_t)f, inv_k));
            xrow[j] = X + (size_t)tok * x_ld + frag_k0(lane);
            xsr[j] = XS + (size_t)tok * xs_ld;
            okr[j] = a_sorted || f >= 0;
        }
    }

    /* ---- one weight tile of the wave: `ty` among the block's column tiles. */
    auto tile = [&](const uint32_t ty) {
    const uint32_t part = inv_ft ? __umulhi(ty, inv_ft) : 0u, tin = ty - part * ft;
    const bool pad = tin * 16u >= fc;
    const uint32_t rt = part * (fc / 16u) + tin, row0 = rt * 16u;

    /* ---- a run of one or two rows: the dot path, no tile. */
    if (dot) {
        float v[kDotRows];
#pragma unroll
        for (uint32_t m = 0; m < kDotRows; ++m) v[m] = 0.0f;
        if (!pad) {
            if (nrows == 1u) dot_run<F, KW, 1u>(c, rt, kph, lane, drow, dsr, ash, v);
            else             dot_run<F, KW, 2u>(c, rt, kph, lane, drow, dsr, ash, v);
        }
        /* Lanes 0-15 hold row 0's sums and 16-31 the same; the upper half takes row 1's. */
        float o = r4d_pick(lane < 16u, v[0], v[1]);
        if constexpr (KW > 1u) {
            red(wave)[lane] = o;
            __syncthreads();
            if (kph != 0u) return;
            o = 0.0f;
#pragma unroll
            for (uint32_t w = 0; w < KW; ++w) o += red(rtb * KW + w)[lane];
        }
        const uint32_t m = lane >> 4, q = red16_row(lane);
        if (m < nrows)
            Y[(size_t)(rs + m) * y_ld + part * fy + tin * 16u + q] =
                f32_to_bf16(!pad && dok[m] ? o * row_scale<F>(c, row0 + q) : 0.0f);
        return;
    }

    /* ---- the tile path. A unit is decoded and transposed once and then multiplied against each
     * token tile the run reaches, so a long run pays its decode once for NT * 16 rows. */

    f8v out[NT];
    float corr[NT];   /* the units' fold factors since the last flush */
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        out[j] = acc_zero();
        corr[j] = 0.0f;
    }
    if (!pad) {
        uint32_t nflush = 0;
        auto bload = [&](uint32_t j, uint32_t kb, i2v (&b)[8]) {
#pragma unroll
            for (uint32_t ks = 0; ks < 8; ++ks) {
                const uint2 v = *(const uint2*)(xrow[j] + kb * kBlock + ks * kK);
                b[ks][0] = (int)v.x;
                b[ks][1] = (int)v.y;
            }
        };
        /* The weight and the first tile's activation fragments are asked for a unit ahead (16-token
         * tile: 2-3 % behind without), each further tile's before the WMMA steps of the tile
         * before it. */
        typename D::Raw cur = D::fetch(c, rt, kph, lane, kph < c.nkb), nxt;
        i2v bq[2][8], bn[8];
        if (kph < c.nkb) bload(0u, kph, bq[0]);
        for (uint32_t kb = kph; kb < c.nkb; kb += KW) {
            if (kb + KW < c.nkb) {
                nxt = D::fetch(c, rt, kb + KW, lane, true);
                bload(0u, kb + KW, bn);
            }
            uint32_t w[16];
            D::rows(cur, c, 1.0f, w);
            Stage8 a8[2];
            frag(w, s_tr + wave * kTrBytes, lane, a8);
            const float fb = fold_f32<F>(c, row0, kb);
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                if (j * kRows < nrows) {
                    if (j + 1u < NT && (j + 1u) * kRows < nrows) bload(j + 1u, kb, bq[(j + 1u) & 1u]);
                    f8v raw = __builtin_bit_cast(f8v, (i8v)(kRbBias));
#pragma unroll
                    for (uint32_t ks = 0; ks < kBlock / kK; ++ks)
                        raw = mma8<true>(frag_of(a8[ks / 4u], ks % 4u), bq[j & 1u][ks], raw);
                    const float sb = fb * xsr[j][kb >> ash];
                    corr[j] += sb;
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t)
                        out[j][t] = __builtin_fmaf(raw[t], sb, out[j][t]);
                }
            }
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
            cur = nxt;
#pragma unroll
            for (uint32_t ks = 0; ks < 8; ++ks) bq[0][ks] = bn[ks];
        }
    }
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t) out[j][t] = __builtin_fmaf(-kRbBiasF, corr[j], out[j][t]);
    if constexpr (KW > 1u) {
        /* each wave's partials in its own tile, behind its last transpose's reads */
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t) red(wave)[t * kWave + lane] = out[0][t];
        __syncthreads();
        if (kph != 0u) return;
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {
            float a = 0.0f;
#pragma unroll
            for (uint32_t w = 0; w < KW; ++w) a += red(rtb * KW + w)[t * kWave + lane];
            out[0][t] = a;
        }
    }
    const uint32_t a0 = acc_row(lane, 0);
    float rsc[8];
#pragma unroll
    for (uint32_t t = 0; t < 8; ++t) rsc[t] = pad ? 0.0f : row_scale<F>(c, row0 + a0 + t);
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) {
        const uint32_t rj = j * kRows + col;
        if (rj < nrows) {
            float v[8];
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) v[t] = okr[j] ? out[j][t] * rsc[t] : 0.0f;
            *(uint4*)(Y + (size_t)(rs + rj) * y_ld + part * fy + tin * 16u + a0) =
                make_uint4(bf16x2_rne(v[0], v[1]), bf16x2_rne(v[2], v[3]),
                           bf16x2_rne(v[4], v[5]), bf16x2_rne(v[6], v[7]));
        }
    }
    };
    /* A wave whose tile is all of its K (KW 1) takes TPW tiles in turn: the run search and the
     * rows above are paid once for them, which a narrow-K projection (two or three units a tile)
     * otherwise pays as a third of its work. */
    for (uint32_t tw = 0; tw < TPW; ++tw) tile((blockIdx.y * RTB + rtb) * TPW + tw);
}

/* The kernel around the body, whose blocks return where their work ends: CH `r4dchain` is the
 * chained form (r4d_chain.h), every block waiting before its first read and adding once after
 * its last write. */
template <class F, uint32_t KW, uint32_t RTB, bool MAP, uint32_t NT, uint32_t TPW, class CH>
__global__ __launch_bounds__(KW * RTB * kWave) R4D_WAVES(8)
void moe_iqr_kernel(R4D_MOEIQR_PARAMS) {
    if (CH::on) r4d_chain_wait();
    moe_iqr_body<F, KW, RTB, MAP, NT, TPW>(X, XS, c_0, c_1, c_2, c_3, sorted, off, Y, map, T, fy,
                                           n_expert, inv_k, x_ld, xs_ld, y_ld, mode, run_tiles,
                                           inv_ft);
    if (CH::on) r4d_chain_done();
}
#undef R4D_MOEIQR_PARAMS

/* Token tiles a block of the run map's grid: a run of up to 64 rows decodes its expert once. */
constexpr uint32_t kMapTiles = 4;
/* The routed rows between which a narrow-K wave takes two weight tiles. Flash-Next's down
 * projection (2560 x 384:256, 512 experts, top 10), us a launch at one tile -> two: 10 rows 11.1 ->
 * 11.9 (too few waves to fill the card), 20 rows level, 30 rows 21.1 -> 20.3, 80 rows 48.0 -> 44.1,
 * 160 rows 88.1 -> 78.4, 240 rows 134.8 -> 129.2, and 320 to 480 rows 4-6 % slower again. */
constexpr uint32_t kTwoTileMinRows = 24, kTwoTileMaxRows = 256;
/* The routed rows from which a wide-K tile is one wave's instead of four waves' share: fewer waves
 * pay the run search and the partial sum through LDS, once enough runs fill the card. Flash-Next's
 * gate_up (768:512 x 2560), us a launch at four waves -> one: 10 rows 15.5 -> 16.3, 20 rows 24.8 ->
 * 24.2, 40 rows 46.3 -> 41.0, 80 rows 81.7 -> 69.7, 160 rows 188 -> 168, 320 rows 380 -> 332,
 * 640 rows 642 -> 522. */
constexpr uint32_t kOneWaveMinRows = 20;

/* The forms built: the computed trellis under one f32 scale a row at two to six bits. */
#define R4D_MOEIQR_FORMS(X)                                                                       \
    X(2, 15) X(2, 16) X(3, 16) X(4, 16) X(5, 16) X(6, 16)

struct Launch {
    const int8_t* x;
    const float* xs;
    IqrCls c[4];
    const int32_t *sorted, *off;
    uint16_t* y;
    uint32_t* map;
    uint32_t T, N, n_expert, inv_k, x_ld, xs_ld, y_ld, mode, run_tiles, inv_ft;
    uint32_t gx, nkb;
    bool chain;   /* the chained form (r4d_chain.h) */
};

template <class F>
int launch_form(const Launch& g, RadStream stream) {
    int rc;
#define R4D_MOEIQR_E(kw_, rtb_, map_, nt_, tpw_, ch_)                                             \
    rc = R4D_EMIT(stream, (moe_iqr_kernel<F, kw_, rtb_, map_, nt_, tpw_, ch_>),                   \
                  dim3(g.gx, g.N / (16u * rtb_ * tpw_)), dim3(kw_ * rtb_ * kWave), 0, g.x, g.xs,  \
                  g.c[0], g.c[1], g.c[2], g.c[3], g.sorted, g.off, g.y, (const uint32_t*)g.map,   \
                  g.T, g.N / (g.mode >> 8), g.n_expert, g.inv_k, g.x_ld, g.xs_ld, g.y_ld, g.mode, \
                  g.run_tiles, g.inv_ft)
#define R4D_MOEIQR_GO(kw_, rtb_, map_, nt_, tpw_)                                                 \
    do { if (g.chain) R4D_MOEIQR_E(kw_, rtb_, map_, nt_, tpw_, r4dchain);                         \
         else         R4D_MOEIQR_E(kw_, rtb_, map_, nt_, tpw_, R4dUnchained); } while (0)
    /* The run map's grid (prefill): a wave a weight tile down all of its K, four tiles a block,
     * kMapTiles token tiles a decode. The decode grid: a wave a tile and a block four, except that
     * a wide K (eight 128-K blocks and up) under kOneWaveMinRows routed rows is shared between
     * four waves a tile, and a narrow one between kTwoTileMinRows and kTwoTileMaxRows rows gives a
     * wave two tiles in turn and a block eight. */
    const bool wide_k = g.nkb >= 8u && (g.T < kOneWaveMinRows || (g.N / 16u) % 4u);
    const bool two = !wide_k && g.nkb < 8u && g.N % (16u * 8u) == 0u &&
                     g.T >= kTwoTileMinRows && g.T <= kTwoTileMaxRows;
    if (g.map)                     R4D_MOEIQR_GO(1u, 4u, true, kMapTiles, 1u);
    else if (wide_k)               R4D_MOEIQR_GO(4u, 1u, false, 1u, 1u);
    else if (!two)                 R4D_MOEIQR_GO(1u, 4u, false, 1u, 1u);
    else                           R4D_MOEIQR_GO(1u, 4u, false, 1u, 2u);
#undef R4D_MOEIQR_GO
#undef R4D_MOEIQR_E
    return rc;
}

/* moe_gemm_q's operands (r4d_moe.hip's moe_gemm_launch): [0] a [1] a_scale [2] w [3] w_scale
 * [4] sorted [5] off, then the further classes' tables and y. */
int moe_iqr_launch(const RadArgs* a, RadStream stream, int ncls) {
    if (ncls != 2 && ncls != 4) return R4D_NO(RAD_E_INVAL);
    const RadTensor* Aq     = r4d_opt(a, 0);
    const RadTensor* As     = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 4);
    const RadTensor* off    = r4d_opt(a, 5);
    const RadTensor* Wc[4]  = { r4d_opt(a, 2), r4d_opt(a, 6),
                                ncls == 4 ? r4d_opt(a, 8) : nullptr,
                                ncls == 4 ? r4d_opt(a, 10) : nullptr };
    const RadTensor* Wsc[4] = { r4d_opt(a, 3), r4d_opt(a, 7),
                                ncls == 4 ? r4d_opt(a, 9) : nullptr,
                                ncls == 4 ? r4d_opt(a, 11) : nullptr };
    const RadTensor* Y      = r4d_opt(a, ncls == 4 ? 12 : 8);
    if (!Aq || !As || !sorted || !off || !Y) return R4D_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data)
        return R4D_NO(RAD_E_INVAL);

    static const char* const kNk[4][2] = { { "N", "K" }, { "N_1", "K_1" }, { "N_2", "K_2" },
                                           { "N_3", "K_3" } };
    long long Nc[4] = { 0, 0, 0, 0 }, Kc[4] = { 0, 0, 0, 0 };
    Nc[0] = r4d_geti_or(a, "N", 0);
    Kc[0] = r4d_geti_or(a, "K", 0);
    for (int q = 1; q < ncls; ++q) {
        Nc[q] = r4d_geti_or(a, ncls == 4 ? kNk[q][0] : "N_odd", Nc[0]);
        Kc[q] = r4d_geti_or(a, ncls == 4 ? kNk[q][1] : "K_odd", Kc[0]);
    }
    const long long parts = r4d_geti_or(a, "parts", 1);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    if (parts <= 0 || parts > 255 || n_expert <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    if (n_expert > (long long)kMaxExpert) return R4D_NO(RAD_E_UNSUPPORTED);
    if (r4d_geti_or(a, "group", kBlock) != (long long)kBlock) return R4D_NO(RAD_E_SHAPE);
    const long long a_group = r4d_geti_or(a, "a_group", kBlock);
    if (a_group != (long long)kBlock && a_group != 2 * (long long)kBlock)
        return R4D_NO(RAD_E_UNSUPPORTED);

    /* Every class holds one form: the tables' dtype is the layout's word (r4d_iqr.h). */
    const uint32_t wdt = Wc[0] ? Wc[0]->dtype : 0u;
    if (!is_weight_dtype(wdt)) return R4D_NO(RAD_E_DTYPE);
    const uint32_t word_ = wdt & kWordMask;
    const Form fm = form(word_);
    if (fm.kind != kTrellisMad || fm.scale != kRowF32 || fm.lrb != 0 || fm.lrun < 4)
        return R4D_NO(RAD_E_DTYPE);
    const long long run = (long long)1 << fm.lrun;

    long long N = 0, K = 0;
    Launch g{};
    for (int q = 0; q < ncls; ++q) {
        const long long cnt = (n_expert + ncls - 1 - q) / ncls;
        if (!Wc[q] || !Wsc[q] || Nc[q] <= 0 || Kc[q] <= 0) return R4D_NO(RAD_E_INVAL);
        if (Wc[q]->dtype != wdt || Wsc[q]->dtype != (R4D_DT_IQR_S | word_))
            return R4D_NO(RAD_E_DTYPE);
        /* Pointer tables, one entry an expert (the core's weight_table_tensor). */
        if (Wc[q]->rank != 3 || Wsc[q]->rank != 3 || Wc[q]->stride[0] != 0 ||
            Wsc[q]->stride[0] != 0 || Wc[q]->shape[0] < cnt || Wsc[q]->shape[0] < cnt ||
            (cnt > 0 && (!Wc[q]->data || !Wsc[q]->data)))
            return R4D_NO(RAD_E_SHAPE);
        /* A part of a class is whole 16-row tiles and whole trellis runs; K whole 128s and whole
         * activation scale groups. */
        if (Kc[q] % a_group || Nc[q] % (16 * parts) || (Nc[q] / parts) % run)
            return R4D_NO(RAD_E_SHAPE);
        g.c[q].w = (const void* const*)Wc[q]->data;
        g.c[q].s = (const void* const*)Wsc[q]->data;
        g.c[q].n = (uint32_t)Nc[q];
        g.c[q].fc = (uint32_t)(Nc[q] / parts);
        g.c[q].pad_ = 0;
        g.c[q].k = (uint32_t)Kc[q];
        if (Nc[q] > N) N = Nc[q];
        if (Kc[q] > K) K = Kc[q];
    }
    for (int q = ncls; q < 4; ++q) g.c[q] = g.c[0];
    if (Aq->dtype != RAD_I8 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16 ||
        sorted->dtype != RAD_I32 || off->dtype != RAD_I32)
        return R4D_NO(RAD_E_DTYPE);

    long long T = 0, M = 0, Ms = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return R4D_NO(RAD_E_STRIDE);
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    if (a_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    const long long as_ld = r4d_rows_pitch(As, K / a_group, &Ms);
    if (as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);

    g.nkb = (uint32_t)(K / kBlock);
    const bool band = r4d_moe_in_decode_band(T, n_expert);
    /* a narrow-K projection and the run map's grid take four weight tiles a block */
    if (((!band || g.nkb < 8u) && (N / 16) % 4) || (N / parts) % 16) return R4D_NO(RAD_E_SHAPE);
    g.x = (const int8_t*)Aq->data;
    g.xs = (const float*)As->data;
    g.sorted = (const int32_t*)sorted->data;
    g.off = (const int32_t*)off->data;
    g.y = (uint16_t*)Y->data;
    g.T = (uint32_t)T;
    g.N = (uint32_t)N;
    g.n_expert = (uint32_t)n_expert;
    if (top_k * T > (long long)1 << 31) return R4D_NO(RAD_E_SHAPE);
    g.inv_k = top_k == 1 ? 0u : (uint32_t)(((uint64_t)1 << 32) / (uint64_t)top_k + 1u);
    g.x_ld = (uint32_t)a_ld;
    g.xs_ld = (uint32_t)as_ld;
    g.y_ld = (uint32_t)y_ld;
    g.run_tiles = (uint32_t)(run / 16);
    /* a part's tiles as a reciprocal (slot_token's bound: a tile index is far below 2^32 / tiles) */
    if (parts > 1 && N / parts < 32) return R4D_NO(RAD_E_SHAPE);
    g.inv_ft = parts > 1 ? (uint32_t)(((uint64_t)1 << 32) / (uint64_t)(N / parts / 16) + 1u) : 0u;
    const uint32_t cshift = ncls == 4 ? 2u : 1u;
    g.mode = (a_sorted ? 1u : 0u) | cshift << 1 | (a_group == 2 * (long long)kBlock ? 8u : 0u) |
             (uint32_t)parts << 8;

    if (band) {
        g.map = nullptr;
        /* A tile of r rows holds at most r runs, so T slots serve every tile: slot x is run
         * x % 16 of tile x / 16, and the last tile's slots end with its rows. */
        g.gx = (uint32_t)T;
        /* The band's one launch has a chained form; the run map's two launches stay plain. */
        g.chain = r4d_geti_or(a, "chain", 0) != 0;
    } else {
        const long long cap = r4d_moe_blockmap_cap(T, n_expert, kRows * kMapTiles);
        if (!a->scratch || a->scratch_bytes < r4d_moe_blockmap_bytes(cap))
            return R4D_NO(RAD_E_SCRATCH);
        g.map = (uint32_t*)a->scratch;
        const int rc = r4d_moe_blockmap_emit(stream, g.off, g.n_expert, g.T, kRows * kMapTiles,
                                             (uint32_t)cap, g.map);
        if (rc != RAD_OK) return rc;
        g.gx = (uint32_t)cap;
    }

    int rc = R4D_NO(RAD_E_DTYPE);
#define R4D_MOEIQR_CASE(b_, l_)                                                                   \
    if (fm.bits == b_ && fm.L == l_)                                                              \
        rc = launch_form<FormT<kTrellisMad, b_, 0, l_, 0, kRowF32>>(g, stream);
    R4D_MOEIQR_FORMS(R4D_MOEIQR_CASE)
#undef R4D_MOEIQR_CASE
    if (rc != RAD_OK) return rc;
    R4D_MOEIQR_CHK();
    return RAD_OK;
}

int64_t moe_iqr_scratch(const RadArgs* a) {
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long M        = r4d_geti_or(a, "M", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || M <= 0) return 0;
    const long long T = M * top_k;
    if (r4d_moe_in_decode_band(T, n_expert)) return 0;
    return r4d_moe_blockmap_bytes(r4d_moe_blockmap_cap(T, n_expert, kRows * kMapTiles));
}

}  // namespace

extern "C" int r4d_rad_moe_gemm_iqr(const RadArgs* a, RadStream s) {
    return moe_iqr_launch(a, s, 2);
}
extern "C" int r4d_rad_moe_gemm_iqr_mod4(const RadArgs* a, RadStream s) {
    return moe_iqr_launch(a, s, 4);
}
extern "C" int64_t r4d_rad_moe_gemm_iqr_scratch(const RadArgs* a) { return moe_iqr_scratch(a); }
