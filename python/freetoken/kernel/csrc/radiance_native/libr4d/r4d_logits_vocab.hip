// logits_gemm_vocab: the lm_head stored as width classes of the vocabulary's rows, each an IQR
// computed trellis (r4d_iqr.h: tcq_code=mad, one f32 scale a row under a factor a 128-K block,
// fwht128) decoded where it is read.
//
// KL weighs a head row by its token's probability, so a vocabulary-weighted head gives the likely
// tokens' rows more bits than the rare ones'. The rows are sorted into classes of one code width
// each (whole 256-row trellis runs), every class a weight of its own whose rows each carry their
// vocabulary id, and the logits come out in vocabulary order: nothing after the head knows the rows
// were permuted. Stored at its id from the tile, a logit in a random row order is a write to a line
// of its own (+120 us of 760 at 8 tokens, +970 of 2350 at 64). So within a class the rows are in
// ascending id (the converter's order; a class's rows share a width, so their order is free), and
// above 32 tokens the tile stores at the ids directly, the large narrow class's rows then nearly
// contiguous; up to 32 it writes a staging plane in row order and a gather through `vrow` (each
// id's class and row) puts it in vocabulary order, cheaper there by the narrow classes' scattered
// rows (626 / 646 us against 668 / 706 at 8 / 16 tokens; 1618 against 1559 at 64). A rank's rows of
// each class hold only ids of that rank's share (an id outside it is dropped), so the head shards by
// rows as any lm_head does.
//
// The activation is the head's bf16 input rotated by the unnormalised Hadamard per 128 of K (the
// rotation the codes were quantised under) and held at sixteen bits: one f32 scale a (token, 128)
// over integers q = 128 hi + lo, |q| <= 127 * 128, hi and lo int8. The int8 grid the trunk's GEMMs
// use costs the head 2e-5 KL on its own (half the weighted head's quantisation cost); sixteen bits
// are below bf16's rounding. Each 128-K block's dot is then two int8 WMMA passes over the same
// decoded weight fragments, hi then lo, the first scaled by 128 in the accumulator: an exact
// integer per (row, token, block), folded once into f32.
//
// One launch covers every class: a block finds its class among the classes' row ranges (the
// widest first, so the narrow classes' cheaper blocks fill the tail) and runs that width's
// decoder. A class of one width alone would leave most of the card idle (the widest is a few
// thousand rows).

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fwht.h"
#include "r4d_iqr.h"
#include "r4d_iqr_dec.h"

#include <algorithm>

namespace {

using namespace r4d_fp8;
using namespace r4d_iqr;
using namespace r4d_iqr_dev;

constexpr uint32_t kWave = 32, kBlock = 128, kFK = 256;
constexpr uint32_t kMaxClass = 4;
/* Tokens a pass of the tile: four 16-token tiles. More tokens are taken in passes, each reading
 * the head once (a prefill wants one row's logits; the all-position logits of a KL evaluation are
 * the case this serves). */
constexpr uint32_t kPassTok = 64;
/* |q| <= 127 * 128: hi = round(q / 128) stays in int8, lo = q - 128 hi in [-64, 64]. */
constexpr float kQMax = 16256.0f;

/* One width class, as the launch sees it. */
struct Cls {
    const uint8_t* W;       /* the IQR weight buffer (codes, then the computed code's A and B) */
    const uint8_t* S;       /* its scale buffer: the rows' f32 scales, then the blocks' factors */
    uint32_t N;             /* rows */
    uint32_t bits;          /* code width */
    const int32_t* ids;     /* the vocabulary id of each row, global, ascending */
    uint32_t blk0;          /* the first grid block of this class */
    uint32_t row0;          /* its first row in the classes' row order (the staging plane's) */
};
struct Head {
    Cls c[kMaxClass];       /* widest first */
    uint32_t n;             /* classes */
    uint32_t nkb;           /* K / 128 */
    int32_t off;            /* this rank's first vocabulary id */
    uint32_t n_vocab;       /* this rank's share */
};

/* ---- the activation: rotated, sixteen bits as two int8 planes ------------------------------
 *
 * A wave a (token, 128-K block): each lane's four values through r4d_fwht_wave, the block's
 * absolute maximum, then q = rint(x * kQMax / amax). An all-zero block stores zeros under scale 0. */
__global__ __launch_bounds__(kWave) void vocab_quant_kernel(const uint16_t* __restrict__ X,
                                                            uint32_t x_ld, int8_t* __restrict__ XH,
                                                            int8_t* __restrict__ XL,
                                                            float* __restrict__ XS, uint32_t K) {
    const uint32_t m = blockIdx.x, kb = blockIdx.y, lane = threadIdx.x;
    const uint32_t nkb = K / kBlock;
    const uint2 raw = *(const uint2*)(X + (size_t)m * x_ld + kb * kBlock + 4u * lane);
    float v[4] = { __uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xFFFF0000u),
                   __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xFFFF0000u) };
    r4d_fwht_wave<128>(v, (int)lane);
    float a = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) a = fmaxf(a, fabsf(v[i]));
    a = r4d_wave_max(a);
    const float inv = a > 0.0f ? kQMax / a : 0.0f;
    uint32_t hp = 0, lp = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const int q = (int)fminf(fmaxf(rintf(v[i] * inv), -kQMax), kQMax);
        const int hi = (q + 64) >> 7, lo = q - (hi << 7);
        hp |= ((uint32_t)hi & 0xFFu) << (8u * i);
        lp |= ((uint32_t)lo & 0xFFu) << (8u * i);
    }
    const size_t o = (size_t)m * K + kb * kBlock + 4u * lane;
    *(uint32_t*)(XH + o) = hp;
    *(uint32_t*)(XL + o) = lp;
    if (lane == 0) XS[(size_t)m * nkb + kb] = a / kQMax;
}

/* ---- vocabulary order through the staging plane ---------------------------------------
 *
 * `vrow` is the head's inverse map, one word a vocabulary id: its class's code width in the top four
 * bits and its row in the class's whole tensor below, of which this rank holds rows
 * [rank * rows, (rank + 1) * rows). A row's tokens are one contiguous read, consecutive ids
 * consecutive writes. A staging row is `mp` floats (16-byte aligned). */
__global__ void vocab_gather_kernel(Head h, const float* __restrict__ st,
                                    const int32_t* __restrict__ vrow, float* __restrict__ out,
                                    uint32_t c_ld, uint32_t mp, uint32_t nb) {
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= h.n_vocab) return;
    const uint32_t w = (uint32_t)vrow[v], b = w >> 28, rank = (uint32_t)h.off / h.n_vocab;
    uint32_t rc = 0, r0 = 0;
#pragma unroll
    for (uint32_t k = 0; k < kMaxClass; ++k)
        if (k < h.n && h.c[k].bits == b) { rc = h.c[k].N; r0 = h.c[k].row0; }
    const uint32_t j = (w & 0x0FFFFFFFu) - rank * rc;
    const bool ok = j < rc;
    const float4* const src = (const float4*)(st + (size_t)(ok ? r0 + j : 0u) * mp);
    for (uint32_t m4 = 0; m4 < nb; m4 += 4) {
        float4 q = src[m4 / 4];
        if (!ok) q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float qv[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
            if (m4 + i < nb) out[(size_t)(m4 + i) * c_ld + v] = qv[i];
    }
}

/* ---- the decoder ---------------------------------------------------------------------------
 *
 * r4d_iqr_dec.h's for the computed trellis under one scale a row, with the continuation read
 * compactly: a column's stream is 16 B bits, so each of a lane's four columns starts at bit 0 or
 * 16 of a dword, and the at most 15 bits a row's state takes from the unit above lie in that one
 * dword: four dword loads a unit, where Dec reads the unit above's whole segment again (fourteen
 * dwords at seven bits). */
template <uint32_t B>
struct HeadDec {
    static constexpr uint32_t L = 16, LANE_BITS = 64u * B, ND = LANE_BITS / 32u, CB = 16u * B;
    struct Raw { uint32_t d[ND]; uint32_t p[4]; };

    __device__ static Raw fetch(const uint8_t* W, uint32_t nkb, uint32_t rt, uint32_t kb,
                                uint32_t lane, bool live) {
        Raw r;
        constexpr size_t ub = 4u * LANE_BITS;
        load_dw<ND>(r.d, W + ((size_t)rt * nkb + kb) * ub + lane * (LANE_BITS / 8u), live);
        /* the unit above in the run of 256 rows (16 tiles), tail-biting at the run's first */
        const uint32_t rp = rt % 16u == 0u ? rt + 15u : rt - 1u;
        const uint32_t* up = (const uint32_t*)(W + ((size_t)rp * nkb + kb) * ub + lane * (LANE_BITS / 8u));
#pragma unroll
        for (uint32_t e = 0; e < 4; ++e) r.p[e] = live ? up[e * CB / 32u] : 0u;
        return r;
    }

    /* The lane's four values of each of the tile's 16 rows, packed: bytesum((st * A + Bc) &
     * 0x3F3F3F3F) - 126 a state. */
    __device__ static void rows(const Raw& r, uint32_t A, uint32_t Bc, uint32_t (&w)[16]) {
        /* a row's four values packed as soon as they are computed: the tile's NT accumulators
         * leave no room for all 64 at once */
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) {
            uint32_t t[4];
#pragma unroll
            for (uint32_t e = 0; e < 4; ++e) {
                const uint32_t pos = e * CB + q * B, own = CB - q * B;
                const uint32_t st =
                    L <= own ? field<ND>(r.d, pos, L)
                             : field<ND>(r.d, pos, own) |
                                   ((r.p[e] >> ((e * CB) & 31u)) & ((1u << (L - own)) - 1u)) << own;
                const uint32_t h = (__umul24(st, A) + Bc) & 0x3F3F3F3Fu;
                t[e] = (uint32_t)__builtin_amdgcn_sudot4(false, (int)h, false, 0x01010101, -126, false);
            }
            w[15u - q] = pack4(pack2(t[0], t[1]), pack2(t[2], t[3]));
        }
    }
};

/* ---- the tile ------------------------------------------------------------------------------
 *
 * gemm_iqr_nt_m64's narrow tile (r4d_gemm_iqr.hip) without the K split: MW waves a block, a wave
 * one 16-row tile of its class against NT 16-token tiles whose hi and lo planes are staged through
 * LDS a 256-K chunk at a time. Each unit's codes are fetched a chunk ahead, decoded to int8 and
 * transposed into fragment order through the wave's LDS tile, then used by both planes' WMMA
 * steps. */
template <uint32_t NT>
struct Lds {
    static constexpr uint32_t FN = NT * 16, kB = FN * lds_pitch<kFK>();
};

template <class F, uint32_t NT, uint32_t MW, bool STAGE>
__device__ __forceinline__ void vocab_tile(const Head& h, const Cls& cl, uint32_t bx,
                                           const int8_t* __restrict__ XH,
                                           const int8_t* __restrict__ XL,
                                           const float* __restrict__ XS, float* __restrict__ out,
                                           uint32_t c_ld, uint32_t nb, uint8_t* s_bh,
                                           uint8_t* s_bl, uint8_t* s_tr) {
    using D = HeadDec<F::B>;
    static_assert(F::KIND == kTrellisMad && F::LRB == 0 && F::SC == kRowF32 && F::L == 16,
                  "a computed trellis, one scale a row");
    constexpr uint32_t FK = kFK, FN = NT * 16, kU = FK / kBlock;
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const uint32_t K = h.nkb * kBlock;

    Ctx c{};
    c.W = cl.W;
    c.S = cl.S;
    c.N = cl.N;
    c.nkb = h.nkb;
    c.run_tiles = 256u / 16u;
    uint8_t* const tr = s_tr + wave * kTrBytes;

    const uint32_t m0 = bx * MW * 16u, n0 = blockIdx.y * FN;
    const uint32_t wrow0 = m0 + wave * 16u;
    const uint32_t arow = wrow0 / 16u;
    const bool alive = wrow0 < cl.N;

    f8v raw[NT], acc[NT];
#pragma unroll
    for (uint32_t j = 0; j < NT; ++j) { raw[j] = acc_zero(); acc[j] = acc_zero(); }

    /* Staging: kTPR threads a token row of a chunk, both planes. */
    constexpr uint32_t kSeg = 32, kTPR = FK / kSeg, kRPP = (MW * kWave) / kTPR;
    constexpr uint32_t kBP = (FN + kRPP - 1) / kRPP;
    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;
    const uint32_t brow0 = n0 + frag_c(lane);
    const uint8_t* const hsrc = (const uint8_t*)XH + (size_t)(n0 + sn) * K + sk;
    const uint8_t* const lsrc = (const uint8_t*)XL + (size_t)(n0 + sn) * K + sk;

    Stage8 ph[kBP], pl[kBP];
#pragma unroll
    for (uint32_t p = 0; p < kBP; ++p) {
        const uint32_t r = sn + p * kRPP;
        const bool live = r < FN && n0 + r < nb;
        ph[p] = stage8_load(hsrc + (size_t)p * kRPP * K, live);
        pl[p] = stage8_load(lsrc + (size_t)p * kRPP * K, live);
    }
    typename D::Raw pf[kU], pfn[kU];
#pragma unroll
    for (uint32_t p = 0; p < kU; ++p) pf[p] = D::fetch(c.W, c.nkb, arow, p, lane, alive);
    /* the computed code's A and B, after its 2^16 values (kept for the unrelayout) */
    constexpr Form fm{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
    const uint32_t* const ab = (const uint32_t*)(c.W + aux_off(fm, c.N, (int64_t)K));
    const uint32_t mA = ab[0], mB = ab[1];

    for (uint32_t k0 = 0; k0 < K; k0 += FK) {
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r < FN) {
                stage8_store<FK>(s_bh, r, sk, ph[p]);
                stage8_store<FK>(s_bl, r, sk, pl[p]);
            }
        }
        __syncthreads();
        const uint32_t k1 = k0 + FK;
        if (k1 < K) {
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                const bool live = r < FN && n0 + r < nb;
                if (r < FN) {
                    ph[p] = stage8_load(hsrc + (size_t)p * kRPP * K + k1, live);
                    pl[p] = stage8_load(lsrc + (size_t)p * kRPP * K + k1, live);
                }
            }
#pragma unroll
            for (uint32_t p = 0; p < kU; ++p) pfn[p] = D::fetch(c.W, c.nkb, arow, k1 / kBlock + p, lane, alive);
        }

#pragma unroll
        for (uint32_t p = 0; p < kU; ++p) {
            const uint32_t kb = k0 / kBlock + p;
            uint32_t w[16];
            D::rows(pf[p], mA, mB, w);
            Stage8 a8[2];
            frag(w, tr, lane, a8);
            /* hi: 128 times its sum, exactly, in the accumulator; then lo on top */
#pragma unroll
            for (uint32_t kk = 0; kk < kBlock / kK; ++kk) {
                const uint32_t ks = p * kBlock + kk * kK;
                const i2v af = frag_of(a8[kk / 4u], kk % 4u);
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) raw[j] = mma8<true>(af, frag<FK>(s_bh, j * 16, ks, lane), raw[j]);
            }
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    raw[j][t] = __int_as_float(__float_as_int(raw[j][t]) << 7);
#pragma unroll
            for (uint32_t kk = 0; kk < kBlock / kK; ++kk) {
                const uint32_t ks = p * kBlock + kk * kK;
                const i2v af = frag_of(a8[kk / 4u], kk % 4u);
#pragma unroll
                for (uint32_t j = 0; j < NT; ++j) raw[j] = mma8<true>(af, frag<FK>(s_bl, j * 16, ks, lane), raw[j]);
            }
            /* the block's factor (shared by every row) times each token's activation scale */
            const float f = alive ? fold_f32<F>(c, wrow0, kb) : 0.0f;
#pragma unroll
            for (uint32_t j = 0; j < NT; ++j) {
                const uint32_t tok = brow0 + j * 16;
                const float ws = tok < nb ? f * XS[(size_t)tok * h.nkb + kb] : 0.0f;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    acc[j][t] = fmaf(acc_val<true>(raw[j][t]), ws, acc[j][t]);
                    raw[j][t] = 0.0f;
                }
            }
        }
#pragma unroll
        for (uint32_t p = 0; p < kU; ++p) pf[p] = pfn[p];
        __syncthreads();
    }

    /* A lane's eight accumulator rows are consecutive rows of the wave's tile, of one token: their
     * row scales and ids are two 16-byte loads each. STAGE: the logits go to the staging plane in
     * the classes' row order, `c_ld` (the pass's tokens rounded up to four) a row, a wave's rows
     * then a few consecutive lines; else each goes to its id. */
    if (!alive) return;
    const uint32_t r0 = wrow0 + acc_row(lane, 0);
    const float4 sa = *(const float4*)((const float*)cl.S + r0), sb = *(const float4*)((const float*)cl.S + r0 + 4);
    const float rs[8] = { sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w };
    if constexpr (STAGE) {
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const uint32_t tok = brow0 + j * 16;
            if (tok >= nb) continue;
            float* const o = out + (size_t)(cl.row0 + r0) * c_ld + tok;
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) o[(size_t)t * c_ld] = acc[j][t] * rs[t];
        }
    } else {
        const int4 ia = *(const int4*)(cl.ids + r0), ib = *(const int4*)(cl.ids + r0 + 4);
        const int32_t id[8] = { ia.x, ia.y, ia.z, ia.w, ib.x, ib.y, ib.z, ib.w };
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j) {
            const uint32_t tok = brow0 + j * 16;
            if (tok >= nb) continue;
            float* const o = out + (size_t)tok * c_ld;
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                const uint32_t v = (uint32_t)(id[t] - h.off);
                if (v < h.n_vocab) o[v] = acc[j][t] * rs[t];
            }
        }
    }
}

template <uint32_t B>
using VF = FormT<kTrellisMad, B, 0, 16, 0, kRowF32>;

/* The widths the classes may have: the computed trellis's built row forms. */
#define R4D_VOCAB_BITS(X) X(2) X(3) X(4) X(5) X(6) X(7)

/* Six waves a SIMD: 4 and 8 measure within 2% of it at every width (M = 1 .. 64). */
template <uint32_t NT, uint32_t MW, bool STAGE>
__global__ __launch_bounds__(MW * kWave) R4D_WAVES(6)
void vocab_tile_kernel(Head h, const int8_t* __restrict__ XH, const int8_t* __restrict__ XL,
                       const float* __restrict__ XS, float* __restrict__ out, uint32_t c_ld,
                       uint32_t nb) {
    __shared__ __attribute__((aligned(16))) uint8_t s_bh[Lds<NT>::kB];
    __shared__ __attribute__((aligned(16))) uint8_t s_bl[Lds<NT>::kB];
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[MW * kTrBytes];
    /* the block's class: the last whose first block is at or below this one (uniform), copied
     * at constant indices so the argument block is never indexed at run time */
    Cls cl = h.c[0];
#pragma unroll
    for (uint32_t i = 1; i < kMaxClass; ++i)
        if (i < h.n && blockIdx.x >= h.c[i].blk0) cl = h.c[i];
    const uint32_t bx = blockIdx.x - cl.blk0;
    switch (cl.bits) {
#define R4D_VOCAB_CASE(b)                                                                         \
        case b: vocab_tile<VF<b>, NT, MW, STAGE>(h, cl, bx, XH, XL, XS, out, c_ld, nb, s_bh, s_bl, s_tr); \
            break;
        R4D_VOCAB_BITS(R4D_VOCAB_CASE)
#undef R4D_VOCAB_CASE
        default: break;
    }
}

constexpr uint32_t kMW = 8;

struct Operands {
    const RadTensor* x = nullptr;
    const RadTensor* y = nullptr;
    const RadTensor *w[kMaxClass] = {}, *s[kMaxClass] = {}, *r[kMaxClass] = {}, *vrow = nullptr;
    long long M = 0, K = 0, rows[kMaxClass] = {};
    uint32_t n = 0;
};

/* Operand order: x, then (w, s, ids) a class, the inverse map, the logits (r4d_rows.cpp's
 * oLogitsVocab). */
constexpr int kOpX = 0, kOpMap = 1 + 3 * (int)kMaxClass, kOpY = kOpMap + 1;

int operands(const RadArgs* a, Operands* g) {
    g->x = r4d_opt(a, kOpX);
    g->y = r4d_opt(a, kOpY);
    if (!g->x || !g->y) return R4D_NO(RAD_E_INVAL);
    if (g->x->dtype != RAD_BF16 || g->y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (g->x->rank < 2 || g->y->rank < 2) return R4D_NO(RAD_E_SHAPE);
    g->M = g->x->shape[0];
    g->K = r4d_geti_or(a, "n_embd", 0);
    if (g->M < 1 || g->K < kFK || g->K % kFK || r4d_numel(g->x) != g->M * g->K)
        return R4D_NO(RAD_E_SHAPE);
    if (g->y->shape[0] < g->M) return R4D_NO(RAD_E_SHAPE);
    static const char* kRows[kMaxClass] = { "rows0", "rows1", "rows2", "rows3" };
    for (uint32_t i = 0; i < kMaxClass; ++i) {
        const RadTensor* w = r4d_opt(a, 1 + 3 * (int)i);
        const RadTensor* s = r4d_opt(a, 2 + 3 * (int)i);
        const RadTensor* r = r4d_opt(a, 3 + 3 * (int)i);
        const long long n = r4d_geti_or(a, kRows[i], 0);
        if (!w && !s && !r && n == 0) break;
        if (!w || !s || !r || n < 256 || n % 256) return R4D_NO(RAD_E_SHAPE);
        if (!is_weight_dtype(w->dtype) || !is_scale_dtype(s->dtype) || r->dtype != RAD_I32)
            return R4D_NO(RAD_E_DTYPE);
        const uint32_t word = w->dtype & kWordMask;
        const Form f = form(word);
        if ((s->dtype & kWordMask) != word || f.kind != kTrellisMad || f.L != 16 || f.lrb != 0 ||
            f.scale != kRowF32 || f.lrun != 8 || f.bits < 2 || f.bits > 7)
            return R4D_NO(RAD_E_DTYPE);
        if (w->shape[0] != n || w->shape[1] != g->K || r4d_numel(r) != n) return R4D_NO(RAD_E_SHAPE);
        if (!r4d_aligned(w->data, 256) || !r4d_aligned(s->data, 16) || !r4d_aligned(r->data, 16))
            return R4D_NO(RAD_E_ALIGN);
        g->w[i] = w;
        g->s[i] = s;
        g->r[i] = r;
        g->rows[i] = n;
        g->n = i + 1;
    }
    if (g->n == 0) return R4D_NO(RAD_E_INVAL);
    g->vrow = r4d_opt(a, kOpMap);
    if (!g->vrow || g->vrow->dtype != RAD_I32 || r4d_numel(g->vrow) != r4d_geti_or(a, "n_vocab", -1))
        return R4D_NO(RAD_E_SHAPE);
    if (r4d_row_pitch(g->x) != g->K) return R4D_NO(RAD_E_STRIDE);
    return RAD_OK;
}

/* The staging plane's tokens: a pass of up to this many stages, a larger one stores directly
 * (staging 48 tokens measured 1175 us against the direct store's 1062 at 33). */
constexpr uint32_t kStageTok = 32;

/* Scratch: the two int8 planes and the scales for every token, then the staging plane. */
long long scratch_bytes(long long M, long long K, long long rows) {
    return 2 * align256(M * K) + align256(M * (K / kBlock) * 4) + align256(rows * kStageTok * 4);
}

}  // namespace

extern "C" int64_t r4d_rad_logits_gemm_vocab_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), K = r4d_geti_or(a, "n_embd", 0);
    static const char* kRows[kMaxClass] = { "rows0", "rows1", "rows2", "rows3" };
    long long rows = 0;
    for (const char* k : kRows) rows += r4d_geti_or(a, k, 0);
    if (M < 1 || K < 1) return 0;
    return (int64_t)scratch_bytes(M, K, rows);
}

extern "C" int r4d_rad_logits_gemm_vocab(const RadArgs* a, RadStream s) {
    Operands g;
    const int rc = operands(a, &g);
    if (rc != RAD_OK) return rc;
    const long long M = g.M, K = g.K, nkb = K / kBlock;
    long long rows = 0;
    for (uint32_t i = 0; i < g.n; ++i) rows += g.rows[i];
    if (!a->scratch || (long long)a->scratch_bytes < scratch_bytes(M, K, rows))
        return R4D_NO(RAD_E_SHAPE);
    int8_t* const xh = (int8_t*)a->scratch;
    int8_t* const xl = xh + align256(M * K);
    float* const xs = (float*)(xl + align256(M * K));
    float* const stg = (float*)((uint8_t*)xs + align256(M * (K / kBlock) * 4));

    Head h{};
    h.n = g.n;
    h.nkb = (uint32_t)nkb;
    h.off = (int32_t)r4d_geti_or(a, "vocab_offset", 0);
    h.n_vocab = (uint32_t)r4d_geti_or(a, "n_vocab", 0);
    const long long c_ld = r4d_row_pitch(g.y);
    if (h.n_vocab < 1 || c_ld < (long long)h.n_vocab) return R4D_NO(RAD_E_SHAPE);
    /* the widest class first: the narrow classes' cheaper blocks then fill the tail */
    uint32_t ord[kMaxClass];
    for (uint32_t i = 0; i < g.n; ++i) ord[i] = i;
    std::sort(ord, ord + g.n, [&](uint32_t x, uint32_t y) {
        return form(g.w[x]->dtype & kWordMask).bits > form(g.w[y]->dtype & kWordMask).bits;
    });
    uint32_t blocks = 0;
    for (uint32_t i = 0; i < g.n; ++i) {
        const uint32_t k = ord[i];
        Cls& c = h.c[i];
        c.W = (const uint8_t*)g.w[k]->data;
        c.S = (const uint8_t*)g.s[k]->data;
        c.ids = (const int32_t*)g.r[k]->data;
        c.N = (uint32_t)g.rows[k];
        c.bits = form(g.w[k]->dtype & kWordMask).bits;
        c.blk0 = blocks;
        c.row0 = blocks * kMW * 16u;
        blocks += c.N / (kMW * 16u);
    }

    if (R4D_EMIT(s, vocab_quant_kernel, dim3((uint32_t)M, (uint32_t)nkb), dim3(kWave), 0,
                 (const uint16_t*)g.x->data, (uint32_t)K, xh, xl, xs, (uint32_t)K) != RAD_OK)
        return R4D_NO(RAD_E_DEVICE);
    for (long long t0 = 0; t0 < M; t0 += kPassTok) {
        const uint32_t nb = (uint32_t)std::min<long long>(kPassTok, M - t0);
        const uint32_t nt = nb > 48 ? 4u : nb > 32 ? 3u : nb > 16 ? 2u : 1u;
        const dim3 grid(blocks, 1, 1);
        const int8_t* const ph = xh + t0 * K;
        const int8_t* const pl = xl + t0 * K;
        const float* const ps = xs + t0 * nkb;
        const bool stage = nb <= kStageTok;
        const uint32_t mp = (nb + 3u) / 4u * 4u;
        float* const po = stage ? stg : (float*)g.y->data + t0 * c_ld;
        const uint32_t ld = stage ? mp : (uint32_t)c_ld;
        int e;
#define R4D_VOCAB_GO(nt_, st_)                                                                   \
        e = R4D_EMIT(s, (vocab_tile_kernel<nt_, kMW, st_>), grid, dim3(kMW * kWave), 0, h, ph, pl, \
                     ps, po, ld, nb)
        if (nt == 4u) R4D_VOCAB_GO(4u, false);
        else if (nt == 3u) R4D_VOCAB_GO(3u, false);
        else if (nt == 2u) R4D_VOCAB_GO(2u, true);
        else R4D_VOCAB_GO(1u, true);
#undef R4D_VOCAB_GO
        if (e != RAD_OK) return R4D_NO(RAD_E_DEVICE);
        if (stage && R4D_EMIT(s, vocab_gather_kernel, dim3((h.n_vocab + 255u) / 256u), dim3(256), 0,
                              h, (const float*)stg, (const int32_t*)g.vrow->data,
                              (float*)g.y->data + t0 * c_ld, (uint32_t)c_ld, mp, nb) != RAD_OK)
            return R4D_NO(RAD_E_DEVICE);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
