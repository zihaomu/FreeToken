// r4d_fused_quant_fp8.hip - producers that already hold an activation row in registers, fused with
// r4d_quant_act_fp8's per-(row, 128) E4M3 quantisation.
//
// The fp8a8 GEMMs need a quantised activation before every linear (`activation_scheme: dynamic`,
// so no shipped scale). A decode step issues ~300 quantisers; each costs a launch, a grid fill and
// a second pass over a row another kernel just had in registers, not bandwidth. Fused producers:
//
//   rmsnorm -> quant   the input of every attention, GDN and MLP linear
//   silu_mul -> quant  down_proj's input
//
// A GEMM output feeding the next linear stays a separate launch. The int8 path has the same fusion
// (r4d_rmsnorm_had_quant_i8, r4d_gated_had_quant_i8), where the rotation is required to survive one
// scale per row; a 128-wide block scale does not need it.
//
// Bit-identical to the unfused pair it replaces (generated text is compared byte for byte across
// builds). The pair writes a bf16 norm output and the quantiser reads it back, so amax and codes
// are taken over bf16-rounded values; these kernels round to bf16 first and quantise the
// round-tripped float.
//
// The norm re-reads its row for the second pass, with or without the residual. Unlike
// r4d_rmsnorm_had_quant_i8 (which normalises the f32 sum), the sum is rounded to bf16 and stored
// first, matching `add` writing a bf16 residual that `rmsnorm` reads back; the re-read is L1-hot.
//
// The residual may alias `out_bf16`, and every caller here does so: a block reads the previous
// block's delta from the shared `h` buffer and writes its norm output back into it. Pass one reads
// every `x` element before the block_sum barrier and pass two writes `y` after it, so they never
// race.
//
// `wadd` is added to every gain in f32 (see r4d_rmsnorm_had_quant_i8): Qwen3.5's GemmaRMSNorm uses
// a zero-centred gamma, and forming `1 + w` in bf16 is a systematic 0.3% error. Pass 0 for a plain
// RMS norm.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_ar_seq.h"   // the all-reduce sequence step the fused all-reduce + norm takes
#include "r4d_fp8_args.h"
#include "r4d_emit.h"
#include "r4d_chain.h"    // the chained forms of the rotated int8 producers
#include "r4d_common.h"   // r4d_act_threads: the workgroup width, shared with the int8 path
#include "r4d_fwht.h"     // the in-register rotation the `_had_` forms put before the absmax
#include "r4d_q8.h"       // the 8-bit codes: E4M3, or int8 for an int8 trunk linear

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kGroup = 128;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

// Round-to-nearest-even, as every bf16 store in this library rounds.
__device__ __forceinline__ uint16_t f2bf(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Both reductions use DPP and permlanex16 (r4d_xor_lane, r4d_wave_sum) rather than `__shfl_xor`,
// which gfx1201 lowers to ds_bpermute_b32, an LDS round trip per stage. Same partners in the same
// order, so the same bits.
__device__ __forceinline__ float wave_max(float v) {
    v = fmaxf(v, r4d_xor_lane<16>(v));
    v = fmaxf(v, r4d_xor_lane<8>(v));
    v = fmaxf(v, r4d_xor_lane<4>(v));
    v = fmaxf(v, r4d_xor_lane<2>(v));
    v = fmaxf(v, r4d_xor_lane<1>(v));
    return v;
}

template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

// A row's sum of squares in the standalone norm's partition (kR4dSumLanes, r4d_common.h): lane
// v < 256 adds columns v, v + 256, ... of `row` in order and the lanes past it add nothing, so a
// row's bits do not depend on the block's width. When the block's threads stored `row`, the
// caller issues the barrier first, as add_rmsnorm does.
__device__ __forceinline__ float lanes_sumsq(const uint16_t* row, uint32_t n) {
    return r4d_lanes_sumsq(row, n, [](uint16_t h) { return bf2f(h); });
}

// Quantise one 128-column group from the four bf16-rounded values a lane owns. The clamp in
// q8_pack4 is required (see r4d_q8.h): v_cvt_pk_fp8_f32 returns NaN out of range, and
// `amax * (1/scale)` can land an ulp above 448.
//
// I8 writes int8 codes at amax / 127 instead (r4d_q8.h), the same codes quant_act_i8g produces.
template <bool I8 = false>
__device__ __forceinline__ void quant_group(const float v[4], float* __restrict__ s,
                                            uint8_t* __restrict__ q, size_t s_at, uint32_t c0,
                                            uint32_t n, uint32_t lane) {
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    m = wave_max(m);
    const float sc = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    if (lane == 0) s[s_at] = sc;
    const float inv = 1.0f / sc;
    const uint32_t pk = q8_pack4<I8>(v[0], v[1], v[2], v[3], inv);
    // A ragged last group is written zero-padded, as in r4d_quant_act_fp8; the caller owns `n`
    // rounded up to 4 bytes of `q`.
    if (c0 < n) *(uint32_t*)(q + c0) = pk;
}

// One scale group of SUB 128-column blocks (SUB 2: IQ8R's 256-wide group, int8 only): `v[h]` are
// the lane's four values of block h, already rotated where the form rotates (the rotation stays 128
// wide). One amax over the whole group, one scale, then each block's codes at its own `c0[h]`.
template <bool I8, uint32_t SUB>
__device__ __forceinline__ void quant_groups(const float (&v)[SUB][4], float* __restrict__ s,
                                             uint8_t* __restrict__ q, size_t s_at,
                                             const uint32_t (&c0)[SUB], uint32_t n, uint32_t lane) {
    static_assert(SUB == 1 || (SUB == 2 && I8), "a 256 group is the int8 quantiser's");
    if constexpr (SUB == 1) {
        quant_group<I8>(v[0], s, q, s_at, c0[0], n, lane);
    } else {
        float m = 0.0f;
#pragma unroll
        for (uint32_t h = 0; h < SUB; ++h)
            m = fmaxf(m, fmaxf(fmaxf(fabsf(v[h][0]), fabsf(v[h][1])),
                               fmaxf(fabsf(v[h][2]), fabsf(v[h][3]))));
        m = wave_max(m);
        const float sc = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
        if (lane == 0) s[s_at] = sc;
        const float inv = 1.0f / sc;
#pragma unroll
        for (uint32_t h = 0; h < SUB; ++h) {
            const uint32_t pk = q8_pack4<I8>(v[h][0], v[h][1], v[h][2], v[h][3], inv);
            if (c0[h] < n) *(uint32_t*)(q + c0[h]) = pk;
        }
    }
}

// Pass two of every norm kernel here: each wave takes a 128-column group and each lane four
// consecutive columns, so the group's amax is a wave reduction and the store a single dword. `nr`
// is the row the norm is taken of, already in memory (see the bit-identity note above). Shared so
// the fused all-reduce cannot drift from the standalone norm.
//
// HI8 is had_quant_act_i8g's form (the `_had_quant_i8g` ops): the width-128 Hadamard between the
// bf16-rounded norm output and the absmax, then int8 codes, as gated_quant_fp8_kernel's ROT and I8
// together. r4d_quant_act_fp8's kernel has the same mapping, rotation and rounding, so the codes
// are the norm + had_quant_act_i8g pair's bytes. `y` still receives the unrotated norm output.
// The launchers require n % 128 == 0, so no rotated group contains padding.
// SUB 2 (with HI8) is IQ8R's group: a wave takes both 128-column blocks of a 256-wide group and
// `nkb` counts groups.
template <bool WF32, uint32_t THR, bool HI8 = false, uint32_t SUB = 1>
__device__ __forceinline__ void norm_quant_row(
    const uint16_t* __restrict__ nr, const void* __restrict__ w, uint16_t* __restrict__ yr,
    uint8_t* __restrict__ qr, float* __restrict__ s, size_t s_row, uint32_t n, uint32_t nkb,
    float rs, float wadd) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    for (uint32_t kb = wave; kb < nkb; kb += THR / kWave) {
        float v[SUB][4];
        uint32_t c0[SUB];
#pragma unroll
        for (uint32_t b = 0; b < SUB; ++b) {
            c0[b] = (kb * SUB + b) * kGroup + lane * 4u;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[b][j] = 0.0f;
            // One branch for all four columns: every launcher requires n % 4 == 0, so a lane's
            // columns are all in or all out. Per-column guards would serialise the loads.
            if (c0[b] < n) {
                float g[4];
                uint16_t x[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    g[j] = WF32 ? ((const float*)w)[c0[b] + j]
                                : bf2f(((const uint16_t*)w)[c0[b] + j]);
                    x[j] = nr[c0[b] + j];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    // The bf16 round trip is deliberate: the unfused quantiser reads these bytes.
                    const uint16_t h = f2bf(bf2f(x[j]) * rs * (g[j] + wadd));
                    if (yr) yr[c0[b] + j] = h;
                    v[b][j] = bf2f(h);
                }
            }
            if constexpr (HI8) r4d_fwht_wave<(int)kGroup>(v[b], (int)lane);
        }
        quant_groups<HI8, SUB>(v, s, qr, s_row + kb, c0, n, lane);
    }
}

// ---------------------------------------------------------------------------- rmsnorm -> quant
// One block per row. Pass one is the block-wide sum of squares; pass two is norm_quant_row. The
// body of rmsnorm_quant_fp8_kernel and of its chained form r4dchain_rmsnorm_quant (r4d_chain.h); no
// block returns early.
#define R4D_RQ_PARAMS const uint16_t* __restrict__ x, uint16_t* __restrict__ res,                \
    const void* __restrict__ w, uint16_t* __restrict__ y, uint8_t* __restrict__ q,               \
    float* __restrict__ s, uint32_t n, uint32_t nkb, float eps, float wadd, uint32_t x_ld,        \
    uint32_t r_ld, uint32_t y_ld, uint32_t q_ld, uint32_t s_ld
#define R4D_RQ_ARGS x, res, w, y, q, s, n, nkb, eps, wadd, x_ld, r_ld, y_ld, q_ld, s_ld
template <bool WF32, bool RES, uint32_t THR, bool HI8 = false, uint32_t SUB = 1>
__device__ __forceinline__ void rmsnorm_quant_fp8_body(R4D_RQ_PARAMS) {
    __shared__ float lds[THR / kWave];
    const uint32_t row = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ rr = RES ? res + (size_t)row * r_ld : nullptr;
    // The norm is taken of the residual when there is one, else of the input.
    const uint16_t* __restrict__ nr = RES ? rr : xr;

    // Four strides are loaded before any is used, avoiding one round trip per stride. A stride past
    // the row reloads this thread's first column and is skipped below. At 256 threads the sum is
    // folded in that loop, whose order is kR4dSumLanes' partition; a wider block folds the stored
    // row in lanes_sumsq behind a barrier.
    constexpr uint32_t kU = 4;
    float ss = 0.0f;
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += kU * THR) {
        uint16_t xa[kU], ra[kU];
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
            const uint32_t i = i0 + u * THR < n ? i0 + u * THR : i0;
            xa[u] = xr[i];
            if (RES) ra[u] = rr[i];
        }
#pragma unroll
        for (uint32_t u = 0; u < kU; ++u) {
            const uint32_t i = i0 + u * THR;
            if (i < n) {
                float v;
                if (RES) {
                    const uint16_t h = f2bf(bf2f(ra[u]) + bf2f(xa[u]));
                    rr[i] = h;
                    v = bf2f(h);
                } else {
                    v = bf2f(xa[u]);
                }
                if constexpr (THR == kR4dSumLanes) ss += v * v;
                else (void)v;
            }
        }
    }
    if constexpr (THR != kR4dSumLanes) {
        if (RES) __syncthreads();   // the residual row, stored across the block's threads
        ss = lanes_sumsq(nr, n);
    }
    const float rs = rsqrtf(block_sum<THR / kWave>(ss, lds) / (float)n + eps);

    norm_quant_row<WF32, THR, HI8, SUB>(nr, w, y ? y + (size_t)row * y_ld : nullptr,
                                        q + (size_t)row * q_ld, s, (size_t)row * s_ld, n, nkb, rs,
                                        wadd);
}
template <bool WF32, bool RES, uint32_t THR, bool HI8 = false, uint32_t SUB = 1>
__global__ __launch_bounds__(THR) void rmsnorm_quant_fp8_kernel(R4D_RQ_PARAMS) {
    rmsnorm_quant_fp8_body<WF32, RES, THR, HI8, SUB>(R4D_RQ_ARGS);
}
template <bool WF32, bool RES, uint32_t THR, bool HI8 = false, uint32_t SUB = 1>
__global__ __launch_bounds__(THR) void r4dchain_rmsnorm_quant(R4D_RQ_PARAMS) {
    r4d_chain_wait();
    rmsnorm_quant_fp8_body<WF32, RES, THR, HI8, SUB>(R4D_RQ_ARGS);
    r4d_chain_done();
}
#undef R4D_RQ_ARGS
#undef R4D_RQ_PARAMS

// ---------------------------------------------------------------------------- gate -> quant
// silu(gate) * up or sigmoid(gate) * up, then the same quantiser. No row reduction, so no first
// pass and no barrier: one wave per group.
//
// ROT inserts the width-128 Hadamard between the product and the absmax, in registers, since this
// mapping (one wave per group, four values per lane) is r4d_fwht_wave<128>'s. `y` still receives
// the unrotated product: it is for a bf16 consumer, and the rotation belongs to the quantised pair
// only.
//
// The launcher sizes the workgroup to the row's groups (capped at r4d_act_threads' width) and the
// kernel reads the width from blockDim, so a narrow row does not leave most waves idle.
// SUB 2 (rotated int8 only) is IQ8R's 256-wide group, as in norm_quant_row.
// The body of gated_quant_fp8_kernel and of its chained form r4dchain_gated_quant; no block returns
// early.
#define R4D_GQ_PARAMS const uint16_t* __restrict__ gu, uint16_t* __restrict__ y,                 \
    uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n, uint32_t nkb, uint32_t gu_ld,     \
    uint32_t y_ld, uint32_t q_ld, uint32_t s_ld
template <int ACT, bool ROT = false, bool I8 = false, uint32_t SUB = 1>
__device__ __forceinline__ void gated_quant_fp8_body(R4D_GQ_PARAMS) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint16_t* __restrict__ gr = gu + (size_t)row * gu_ld;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
    uint8_t*  __restrict__ qr = q + (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < nkb; kb += blockDim.x / kWave) {
        float v[SUB][4];
        uint32_t c0[SUB];
#pragma unroll
        for (uint32_t b = 0; b < SUB; ++b) {
            c0[b] = (kb * SUB + b) * kGroup + lane * 4u;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[b][j] = 0.0f;
            // One branch for all four columns; see norm_quant_row.
            if (c0[b] < n) {
                uint16_t gb[4], ub[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    gb[j] = gr[c0[b] + j];
                    ub[j] = gr[n + c0[b] + j];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float g = bf2f(gb[j]);
                    const float a = ACT == 0 ? g / (1.0f + expf(-g)) : 1.0f / (1.0f + expf(-g));
                    const uint16_t h = f2bf(a * bf2f(ub[j]));
                    if (yr) yr[c0[b] + j] = h;
                    v[b][j] = bf2f(h);
                }
            }
            // The rotated launcher requires n % 128 == 0, so no rotated group contains padding.
            if constexpr (ROT) r4d_fwht_wave<(int)kGroup>(v[b], (int)lane);
        }
        quant_groups<I8, SUB>(v, s, qr, (size_t)row * s_ld + kb, c0, n, lane);
    }
}
template <int ACT, bool ROT = false, bool I8 = false, uint32_t SUB = 1>
__global__ __launch_bounds__(1024) void gated_quant_fp8_kernel(R4D_GQ_PARAMS) {
    gated_quant_fp8_body<ACT, ROT, I8, SUB>(gu, y, q, s, n, nkb, gu_ld, y_ld, q_ld, s_ld);
}
template <int ACT, bool ROT = false, bool I8 = false, uint32_t SUB = 1>
__global__ __launch_bounds__(1024) void r4dchain_gated_quant(R4D_GQ_PARAMS) {
    r4d_chain_wait();
    gated_quant_fp8_body<ACT, ROT, I8, SUB>(gu, y, q, s, n, nkb, gu_ld, y_ld, q_ld, s_ld);
    r4d_chain_done();
}
#undef R4D_GQ_PARAMS

// -------------------------------------------------- gate . x -> quant, two source buffers
//
// The same epilogue with the gate in a separate buffer. For gated attention, `x` is the attention
// output [T, n_head * head_dim] and the gate is a column slice of the [q|gate] projection,
// [T * n_head, 2 * head_dim] with the gate half at `g_off`. A row is a (token, head) pair: `x` is
// then contiguous, the gate a fixed stride apart, and the scale plane [T * n_head, head_dim / 128]
// is byte-identical to the [T, n_head * head_dim / 128] the out projection reads. Replaces three
// launches (sigmoid, multiply, quantise) per attention layer.
//
// ACT 2 is the product with no activation: the unfused pair is `mul` then the quantiser, so the
// gate is not rounded on its own and the product is rounded once, as mul_bf16 does.
//
// ROT (with I8: gate_had_quant_i8g) puts the width-128 Hadamard between the bf16-rounded product
// and the absmax, as gated_quant_fp8_kernel's ROT, for an out projection stored rotated: the codes
// are had_quant_act_i8g's over `y`.
// SUB 2 (rotated int8 only) is IQ8R's 256-wide group, as in norm_quant_row; a row (one head) is
// then whole 256s.
// The body of gate_quant_fp8_kernel and of its chained form r4dchain_gate_quant; no block returns
// early.
#define R4D_GTQ_PARAMS const uint16_t* __restrict__ g, const uint16_t* __restrict__ x,           \
    uint16_t* __restrict__ y, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n,         \
    uint32_t nkb, uint32_t g_ld, uint32_t x_ld, uint32_t y_ld, uint32_t q_ld, uint32_t s_ld
template <int ACT, bool I8 = false, bool ROT = false, uint32_t SUB = 1>
__device__ __forceinline__ void gate_quant_fp8_body(R4D_GTQ_PARAMS) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint16_t* __restrict__ gr = g + (size_t)row * g_ld;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
    uint8_t*  __restrict__ qr = q + (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < nkb; kb += blockDim.x / kWave) {
        float v[SUB][4];
        uint32_t c0[SUB];
#pragma unroll
        for (uint32_t b = 0; b < SUB; ++b) {
            c0[b] = (kb * SUB + b) * kGroup + lane * 4u;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[b][j] = 0.0f;
            // One branch for all four columns; see norm_quant_row.
            if (c0[b] < n) {
                uint16_t gb[4], xb[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    gb[j] = gr[c0[b] + j];
                    xb[j] = xr[c0[b] + j];
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint16_t h;
                    if (ACT == 2) {
                        h = f2bf(bf2f(gb[j]) * bf2f(xb[j]));
                    } else {
                        const float gv = bf2f(gb[j]);
                        // The activation is rounded to bf16 before the multiply, unlike
                        // gated_quant_fp8: the unfused reference here is `sigmoid` then `mul` with
                        // a bf16 buffer between them (silu_mul_bf16 keeps f32). Skipping the round
                        // trip changes the output.
                        const uint16_t ab =
                            f2bf(ACT == 0 ? gv / (1.0f + expf(-gv)) : 1.0f / (1.0f + expf(-gv)));
                        h = f2bf(bf2f(ab) * bf2f(xb[j]));
                    }
                    if (yr) yr[c0[b] + j] = h;
                    v[b][j] = bf2f(h);
                }
            }
            // The rotated launcher requires n % 128 == 0, so no rotated group contains padding.
            if constexpr (ROT) r4d_fwht_wave<(int)kGroup>(v[b], (int)lane);
        }
        quant_groups<I8, SUB>(v, s, qr, (size_t)row * s_ld + kb, c0, n, lane);
    }
}
template <int ACT, bool I8 = false, bool ROT = false, uint32_t SUB = 1>
__global__ __launch_bounds__(1024) void gate_quant_fp8_kernel(R4D_GTQ_PARAMS) {
    gate_quant_fp8_body<ACT, I8, ROT, SUB>(g, x, y, q, s, n, nkb, g_ld, x_ld, y_ld, q_ld, s_ld);
}
template <int ACT, bool I8 = false, bool ROT = false, uint32_t SUB = 1>
__global__ __launch_bounds__(1024) void r4dchain_gate_quant(R4D_GTQ_PARAMS) {
    r4d_chain_wait();
    gate_quant_fp8_body<ACT, I8, ROT, SUB>(g, x, y, q, s, n, nkb, g_ld, x_ld, y_ld, q_ld, s_ld);
    r4d_chain_done();
}
#undef R4D_GTQ_PARAMS

// -------------------------------------------------------- all-reduce -> rmsnorm -> quant
//
// The same norm with the tensor-parallel all-reduce that always precedes it folded in (o_proj,
// out_proj and down_proj outputs are all-reduced, then normalised). Saves a launch, a dispatch gap
// and a round trip through memory. The int8 path has the same fusion (r4d_ar_ln_had_quant_i8).
//
// Partitioned by row: r4d_ar_oneshot_2rank_exact gives block b a contiguous range of 16B words with
// its own flag and no grid barrier, but a sum of squares needs whole rows. Rows are contiguous, so a
// row range is a word range; both ranks derive it from (M, gridDim.x), keeping the handshake
// per block.
//
// Bit-identical to the pair it replaces. Pass two is norm_quant_row. Pass one reproduces the
// all-reduce's arithmetic (an f32 sum stored as bf16, as in r4d_ar_oneshot_2rank_exact's vadd) and
// stores it back into `x`, as the replaced kernel did, so other readers of `x` see the same bytes.
// Both sum the squares in the standalone norm's partition (lanes_sumsq).
//
// Both wires are served. Above --tp-wire-min-kb the message is sent 6-bit rotated; a step carries
// n_seq * (1 + n_spec) rows, so it crosses that floor from the second concurrent sequence on, and
// an exact-wire-only fusion would rarely apply.
//
// The compressed wire is r4d_ar_oneshot_2rank_wht6's format (orthonormal Walsh-Hadamard over 64,
// amax-scaled, 6 uniform bits plus a bf16 scale) packed in chunks of 512 elements instead of 2048.
// Rotation and amax close over a 64-element group, so the bytes are identical; the chunk size only
// sets ownership. A chunk may not straddle a row (both ranks must agree on a block's bytes), and a
// wave owns a whole chunk, so a smaller chunk employs more waves (see r4d_ar_ln_had_quant_i8.hip).
// A row of whole 64-element groups that is not whole chunks (TranslateGemma's 5376 = 10.5 chunks)
// ends in a partial chunk: its missing groups go out as zeros and are not stored back. Every real
// group is the same 64 elements as in the flat message, so its bytes, and the sum, are the
// standalone wire's; with whole chunks the mapping is the flat one.
//
// The reduced row goes back to global `x`, not LDS: unlike the int8 twin this norm needs no
// in-place rotation, and pass two re-reads the row L1-hot as the standalone norm does.
//
// This rank's own packed half also goes to global, in the standalone kernel's side buffer
// (`loc_pack`). LDS (64 KiB) holds only ~15 rows of ~4 KiB packed halves at n = 5120, while one op
// serves M = 8 through a 2048-row prefill chunk; a refused launch would abort the step. The side
// buffer costs 0.75 bytes per element, written and read back from L2.
#define R4D_ARQ_SPIN_MAX 4000000000ULL

#define ARQ_GROUP 64
#define ARQ_BITS 6
#define ARQ_EPL 8
#define ARQ_LPG (ARQ_GROUP / ARQ_EPL)                    /* 8 lanes span one group */
#define ARQ_WV 32
#define ARQ_GPW (ARQ_WV * ARQ_EPL / ARQ_GROUP)           /* 4 groups a wave-iteration */
#define ARQ_CG 8                                         /* groups a chunk */
#define ARQ_CE (ARQ_CG * ARQ_GROUP)                      /* 512 elements a chunk */
#define ARQ_ITERS (ARQ_CE / (ARQ_WV * ARQ_EPL))          /* 2 iterations to cover a chunk */
#define ARQ_NWORD (ARQ_ITERS * ARQ_EPL * ARQ_BITS / 32)  /* 3 dwords a lane holds for a chunk */
#define ARQ_CBYTES (ARQ_CE * ARQ_BITS / 8)               /* 384 wire bytes a chunk */
#define ARQ_QMAX ((1 << (ARQ_BITS - 1)) - 1)
#define ARQ_QMIN (-(1 << (ARQ_BITS - 1)))
#define ARQ_MASK ((1u << ARQ_BITS) - 1u)

// `lo ? (v + o) : (o - v)` written as `o + (lo ? v : -v)`: the sign flip is one XOR with a mask
// uniform over the stage, removing the select. Bit-identical, since IEEE negation is exact.
template <int M>
__device__ __forceinline__ void arq_xstage(float* v, int lig) {
    const unsigned int sm = (lig & M) ? 0x80000000u : 0u;
#pragma unroll
    for (int i = 0; i < ARQ_EPL; ++i) {
        const float o = r4d_xor_lane<M>(v[i]);
        v[i] = __int_as_float(__float_as_int(v[i]) ^ sm) + o;
    }
}

// Orthonormal Walsh-Hadamard over one group; its own inverse, so this also undoes it.
__device__ __forceinline__ void arq_wht(float* v, int lig) {
    static_assert(ARQ_LPG == 8, "the cross-lane stages are unrolled for a group of 8 lanes");
#pragma unroll
    for (int h = 1; h < ARQ_EPL; h <<= 1)
#pragma unroll
        for (int i = 0; i < ARQ_EPL; ++i)
            if ((i & h) == 0) { const float a = v[i], c = v[i ^ h]; v[i] = a + c; v[i ^ h] = a - c; }
    arq_xstage<1>(v, lig);
    arq_xstage<2>(v, lig);
    arq_xstage<4>(v, lig);
    const float nrm = rsqrtf((float)ARQ_GROUP);
#pragma unroll
    for (int i = 0; i < ARQ_EPL; ++i) v[i] *= nrm;
}

__device__ __forceinline__ void arq_store_rel(unsigned int* p, unsigned int v) {
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int arq_load_acq(const unsigned int* p) {
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// POST: a sandwich-norm junction (Gemma 3): the reduced sum is post-normed against `wp` before the
// residual add, `v = bf16(sum * s1 * (wp + wadd))` with s1 the sum's own rms, as rmsnorm_quant_i8's
// POST computes it. `x` still receives the reduced sum. Either wire (the compressed one has already
// stored the sum into `x`); a residual only.
//
// HI8: ar_rmsnorm_had_quant_i8g, pass two in norm_quant_row's rotated int8 form. The reduction and
// the norm are untouched, so `x`, the residual and `y` are ar_rmsnorm_quant_fp8's bytes. SUB 2
// (with HI8) is IQ8R's 256-wide scale group, as norm_quant_row's; `nkb` then counts groups.
// PAD (wire 6 only): the row is whole 64-element groups but not whole chunks, so its last chunk is
// padded and a chunk's elements are found by (row, chunk in row). Without it, for a row of whole
// chunks, the flat chunk index gives the element offset.
// CH `r4dchain`: the chained form (r4d_chain.h); no block returns early.
template <bool WF32, bool RES, uint32_t THR, int WIRE, bool POST = false, bool HI8 = false,
          bool PAD = false, uint32_t SUB = 1, class CH = R4dUnchained>
__global__ __launch_bounds__(THR) void ar_rmsnorm_quant_fp8_kernel(
    uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    const void* __restrict__ wp,
    uint16_t* __restrict__ y, uint8_t* __restrict__ q, float* __restrict__ s,
    int4* peer_base, const int4* my_base, unsigned int* peer_flags, unsigned int* my_flags,
    unsigned int* seq_ctrs, char* loc_pack, long loc_scale_off, int M, uint32_t n, uint32_t nkb,
    float eps, float wadd, uint32_t r_ld, uint32_t y_ld, uint32_t q_ld, uint32_t s_ld,
    int slot_stride16, int drain, int acq, unsigned int nbw, long stage) {
    if (CH::on) r4d_chain_wait();
    __shared__ float lds[THR / kWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, nb = (int)gridDim.x;
    const uint32_t tid = threadIdx.x;
    /* Integer arithmetic only, so both ranks land on the same split. */
    const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);
    /* Chunks per row and this block's window into the global chunk numbering, derived identically
     * on both ranks from (M, nb, n). */
    const int cpr = WIRE ? (int)(PAD ? (n + ARQ_CE - 1) / ARQ_CE : n / ARQ_CE) : 0;
    const int gpr = (int)(n / ARQ_GROUP);   /* the real groups a row has; the rest of its last chunk pads */
    (void)gpr;
    const int c00 = r0 * cpr, cpb = (r1 - r0) * cpr;
    const uint32_t lane = tid & (ARQ_WV - 1), wave = tid / ARQ_WV;
    const int lig = (int)(lane & (ARQ_LPG - 1)), gw = (int)(lane / ARQ_LPG);

    r4d_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   /* device-resident, replay-safe */
    const unsigned int sq = s_seq;
    int4* const ps = peer_base + (size_t)s_slot * slot_stride16;
    const int4* ms = my_base + (size_t)s_slot * slot_stride16;
    const long md = r4d_archk_md(stage, my_base);   // the host mailbox's VRAM mirror (r4d_ar_seq.h)
    /* The slot is split as in the standalone compressed all-reduce: packed symbols from the base,
     * bf16 group scales from half way up. Group indices are global over the [M][n] message. */
    char* const p_pay = (char*)ps;
    const char* m_pay = (const char*)ms;
    unsigned short* const p_sc =
        (unsigned short*)((char*)ps + (size_t)(slot_stride16 >> 1) * 16);
    const unsigned short* m_sc =
        (const unsigned short*)((const char*)ms + (size_t)(slot_stride16 >> 1) * 16);
    /* This rank's own packed half, laid out like the peer slot (symbols from the base, scales from
     * `loc_scale_off`) and indexed by global chunk and group, so blocks never overlap. */
    unsigned int* const Lpk = (unsigned int*)loc_pack;
    float* const Lsc = (float*)(loc_pack + loc_scale_off);

    unsigned int hck = 0u;   // the push check (r4d_ar_seq.h)
    if constexpr (WIRE == 0) {
        /* Push my rows into the peer's slot. `x` is a tight [M][n] of bf16 (the launcher refuses a
         * pitched one), so a row range is a 16B-word range. */
        const int wpr = (int)(n >> 3);
        const int wend = r1 * wpr;
        const int4* const in4 = (const int4*)x;
        for (int k = r0 * wpr + (int)tid; k < wend; k += (int)THR) {
            const int4 v = in4[k];
            ps[k] = v;
            hck += r4d_archk_u4(v, 4u * (unsigned)k);
        }
    } else {
        /* Rotate, amax, quantise and pack straight into the peer's slot. A wave owns whole chunks,
         * so every lane emits ARQ_NWORD dwords per chunk and the flush is lane-uniform. No group is
         * partial (n is whole groups, a block owns whole rows), so every cross-lane stage is fully
         * active; a group past the row's end (a partial last chunk) packs zeros. */
        for (int ci = (int)wave; ci < cpb; ci += (int)(THR / ARQ_WV)) {
            const int c = c00 + ci;
            unsigned int acc[ARQ_NWORD];
#pragma unroll
            for (int wi = 0; wi < ARQ_NWORD; ++wi) acc[wi] = 0u;
#pragma unroll
            for (int t = 0; t < ARQ_ITERS; ++t) {
                const int g = c * ARQ_CG + t * ARQ_GPW + gw;
                float v[ARQ_EPL];
                if constexpr (PAD) {
                    const int cr = c / cpr, gr = (c - cr * cpr) * ARQ_CG + t * ARQ_GPW + gw;
                    const bool live = gr < gpr;
                    const long off = (long)cr * n + (long)gr * ARQ_GROUP + (long)lig * ARQ_EPL;
#pragma unroll
                    for (int i = 0; i < ARQ_EPL; ++i) v[i] = live ? bf2f(x[off + i]) : 0.0f;
                } else {
                    const long off = (long)g * ARQ_GROUP + (long)lig * ARQ_EPL;
#pragma unroll
                    for (int i = 0; i < ARQ_EPL; ++i) v[i] = bf2f(x[off + i]);
                }
                arq_wht(v, lig);
                float a = 0.0f;
#pragma unroll
                for (int i = 0; i < ARQ_EPL; ++i) a = fmaxf(a, fabsf(v[i]));
                a = fmaxf(a, r4d_xor_lane<1>(a));
                a = fmaxf(a, r4d_xor_lane<2>(a));
                a = fmaxf(a, r4d_xor_lane<4>(a));
                float sc = (a > 0.0f) ? bf2f(f2bf(a * (1.0f / (float)ARQ_QMAX))) : 1.0f;
                if (!(sc > 0.0f)) sc = 1.0f;
                if (lig == 0) {
                    const unsigned short sc16 = (unsigned short)(__float_as_uint(sc) >> 16);
                    p_sc[g] = sc16;
                    Lsc[g] = sc;
                    hck += r4d_archk_mix(sc16, R4D_ARCHK_U16 + (unsigned)g);
                }
                const float inv = 1.0f / sc;
#pragma unroll
                for (int i = 0; i < ARQ_EPL; ++i) {
                    const unsigned int sym = (unsigned int)(
                        min(max((int)rintf(v[i] * inv), ARQ_QMIN), ARQ_QMAX) - ARQ_QMIN);
                    const int bit = (t * ARQ_EPL + i) * ARQ_BITS, wi = bit >> 5, bo = bit & 31;
                    acc[wi] |= sym << bo;
                    if (bo + ARQ_BITS > 32) acc[wi + 1] |= sym >> (32 - bo);
                }
            }
            /* Flush lane-contiguously, widest region first, so each is one coalesced store. */
            char* const pb = p_pay + (size_t)c * ARQ_CBYTES;
            const unsigned int cd = (unsigned)c * (ARQ_CBYTES / 4);   // the chunk's first dword
#pragma unroll
            for (int w4 = 0; w4 + 4 <= ARQ_NWORD; w4 += 4) {
                *(uint4*)(pb + w4 * 4 * ARQ_WV + lane * 16) =
                    make_uint4(acc[w4], acc[w4 + 1], acc[w4 + 2], acc[w4 + 3]);
                hck += r4d_archk_u4(make_uint4(acc[w4], acc[w4 + 1], acc[w4 + 2], acc[w4 + 3]),
                                    cd + (unsigned)(w4 * ARQ_WV) + lane * 4u);
            }
            if constexpr (ARQ_NWORD % 4 >= 2) {
                *(uint2*)(pb + (ARQ_NWORD & ~3) * 4 * ARQ_WV + lane * 8) =
                    make_uint2(acc[ARQ_NWORD & ~3], acc[(ARQ_NWORD & ~3) + 1]);
                const unsigned int e = cd + (unsigned)((ARQ_NWORD & ~3) * ARQ_WV) + lane * 2u;
                hck += r4d_archk_mix(acc[ARQ_NWORD & ~3], e) +
                       r4d_archk_mix(acc[(ARQ_NWORD & ~3) + 1], e + 1u);
            }
            if constexpr (ARQ_NWORD % 2 == 1) {
                *(unsigned int*)(pb + (ARQ_NWORD - 1) * 4 * ARQ_WV + lane * 4) =
                    acc[ARQ_NWORD - 1];
                hck += r4d_archk_mix(acc[ARQ_NWORD - 1],
                                     cd + (unsigned)((ARQ_NWORD - 1) * ARQ_WV) + lane);
            }
#pragma unroll
            for (int wi = 0; wi < ARQ_NWORD; ++wi)
                Lpk[((size_t)c * ARQ_NWORD + wi) * ARQ_WV + lane] = acc[wi];
        }
    }
    r4d_archk_put(hck, peer_flags, nbw, s_slot, (unsigned)b, sq);
    /* The drain modes are r4d_ar_oneshot_2rank_exact's, selected by the same variant table. */
    if (drain == 1) __threadfence_system();
    else if (drain == 3) asm volatile("s_wait_storecnt 0x0" ::: "memory");
    __syncthreads();
    if (tid == 0) {
        if (drain == 2) __threadfence_system();
        arq_store_rel(&peer_flags[b], sq);
        unsigned long long z = 0;
        /* A handshake that cannot complete is a lockstep violation; trap rather than sum a slot the
         * peer never filled. */
        while (r4d_ar_seq_behind(arq_load_acq(&my_flags[b]), sq)) {
            if (++z > R4D_ARQ_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
    if (acq) __threadfence_system();
    r4d_archk_wait(my_flags, nbw, s_slot, (unsigned)b, sq, [&](unsigned int& h) {
        if constexpr (WIRE == 0) {
            const int wpr = (int)(n >> 3);
            const int4* const m4 = (const int4*)ms;
            for (int k = r0 * wpr + (int)tid; k < r1 * wpr; k += (int)THR)
                h += r4d_archk_u4(r4d_archk_keep(&m4[k], md), 4u * (unsigned)k);
        } else {
            for (int ci = (int)wave; ci < cpb; ci += (int)(THR / ARQ_WV)) {
                const int c = c00 + ci;
                const char* const pb = m_pay + (size_t)c * ARQ_CBYTES;
                const unsigned int cd = (unsigned)c * (ARQ_CBYTES / 4);
#pragma unroll
                for (int w4 = 0; w4 + 4 <= ARQ_NWORD; w4 += 4)
                    h += r4d_archk_u4(
                        r4d_archk_keep((const uint4*)(pb + w4 * 4 * ARQ_WV + lane * 16), md),
                        cd + (unsigned)(w4 * ARQ_WV) + lane * 4u);
                if constexpr (ARQ_NWORD % 4 >= 2) {
                    const uint2 u = r4d_archk_keep(
                        (const uint2*)(pb + (ARQ_NWORD & ~3) * 4 * ARQ_WV + lane * 8), md);
                    const unsigned int e = cd + (unsigned)((ARQ_NWORD & ~3) * ARQ_WV) + lane * 2u;
                    h += r4d_archk_mix(u.x, e) + r4d_archk_mix(u.y, e + 1u);
                }
                if constexpr (ARQ_NWORD % 2 == 1)
                    h += r4d_archk_mix(
                        r4d_archk_keep((const unsigned int*)(pb + (ARQ_NWORD - 1) * 4 * ARQ_WV +
                                                             lane * 4), md),
                        cd + (unsigned)((ARQ_NWORD - 1) * ARQ_WV) + lane);
#pragma unroll
                for (int t = 0; t < ARQ_ITERS; ++t) {
                    const int g = c * ARQ_CG + t * ARQ_GPW + gw;
                    if (lig == 0)
                        h += r4d_archk_mix(r4d_archk_keep(&m_sc[g], md),
                                           R4D_ARCHK_U16 + (unsigned)g);
                }
            }
        }
    });
    ms = r4d_archk_moved(ms, md);
    m_pay = r4d_archk_moved(m_pay, md);
    m_sc = r4d_archk_moved(m_sc, md);

    if constexpr (WIRE == 6) {
        /* Reduce: unpack both halves of every chunk this block owns, add in fp32, inverse-rotate,
         * and store the bf16 sum into `x`, as the standalone compressed all-reduce does. */
        for (int ci = (int)wave; ci < cpb; ci += (int)(THR / ARQ_WV)) {
            const int c = c00 + ci;
            const char* const pb = m_pay + (size_t)c * ARQ_CBYTES;
            unsigned int pac[ARQ_NWORD], lac[ARQ_NWORD];
#pragma unroll
            for (int w4 = 0; w4 + 4 <= ARQ_NWORD; w4 += 4) {
                const uint4 pq = *(const uint4*)(pb + w4 * 4 * ARQ_WV + lane * 16);
                pac[w4] = pq.x; pac[w4 + 1] = pq.y; pac[w4 + 2] = pq.z; pac[w4 + 3] = pq.w;
            }
            if constexpr (ARQ_NWORD % 4 >= 2) {
                const uint2 pu = *(const uint2*)(pb + (ARQ_NWORD & ~3) * 4 * ARQ_WV + lane * 8);
                pac[ARQ_NWORD & ~3] = pu.x; pac[(ARQ_NWORD & ~3) + 1] = pu.y;
            }
            if constexpr (ARQ_NWORD % 2 == 1)
                pac[ARQ_NWORD - 1] =
                    *(const unsigned int*)(pb + (ARQ_NWORD - 1) * 4 * ARQ_WV + lane * 4);
#pragma unroll
            for (int wi = 0; wi < ARQ_NWORD; ++wi)
                lac[wi] = Lpk[((size_t)c * ARQ_NWORD + wi) * ARQ_WV + lane];
#pragma unroll
            for (int t = 0; t < ARQ_ITERS; ++t) {
                const int gl = t * ARQ_GPW + gw;
                const int g = c * ARQ_CG + gl;
                const float psc = __uint_as_float((unsigned int)m_sc[g] << 16);
                const float lsc = Lsc[g];
                float v[ARQ_EPL];
#pragma unroll
                for (int i = 0; i < ARQ_EPL; ++i) {
                    const int bit = (t * ARQ_EPL + i) * ARQ_BITS, wi = bit >> 5, bo = bit & 31;
                    unsigned int psy = pac[wi] >> bo, lsy = lac[wi] >> bo;
                    if (bo + ARQ_BITS > 32) {
                        psy |= pac[wi + 1] << (32 - bo);
                        lsy |= lac[wi + 1] << (32 - bo);
                    }
                    /* The two ranks add the pair in opposite order; fp32 add commutes exactly, so
                     * they agree bit for bit as long as neither product is contracted into an FMA
                     * (libr4d/CMakeLists.txt passes -ffp-contract=off for this). */
                    v[i] = (float)((int)(psy & ARQ_MASK) + ARQ_QMIN) * psc
                         + (float)((int)(lsy & ARQ_MASK) + ARQ_QMIN) * lsc;
                }
                arq_wht(v, lig);
                if constexpr (PAD) {
                    const int cr = c / cpr, gr = (c - cr * cpr) * ARQ_CG + gl;
                    if (gr < gpr) {
                        const long off = (long)cr * n + (long)gr * ARQ_GROUP + (long)lig * ARQ_EPL;
#pragma unroll
                        for (int i = 0; i < ARQ_EPL; ++i) x[off + i] = f2bf(v[i]);
                    }
                } else {
                    const long off = (long)g * ARQ_GROUP + (long)lig * ARQ_EPL;
#pragma unroll
                    for (int i = 0; i < ARQ_EPL; ++i) x[off + i] = f2bf(v[i]);
                }
            }
        }
        /* Pass two reads all of `x`'s row; each wave wrote only its own chunks. */
        __syncthreads();
    }

    /* WIRE == 0: the peer's rows are now in this rank's scratch, in x's [M][n] layout. */
    const uint16_t* __restrict__ P = (const uint16_t*)ms;

    for (int row = r0; row < r1; ++row) {
        uint16_t* __restrict__ xr = x + (size_t)row * n;
        const uint16_t* __restrict__ pr = P + (size_t)row * n;
        uint16_t* __restrict__ rr = RES ? res + (size_t)row * r_ld : nullptr;
        const uint16_t* __restrict__ nr = RES ? rr : xr;

        (void)pr;
        float ss = 0.0f;
        if constexpr (POST && RES) {
            /* Pass one reduces and takes the sum's sum of squares; pass two post-norms it, adds it
             * into the stream and takes the stream's. Each thread reads back only its own columns. */
            float s1s = 0.0f;
            for (uint32_t i = tid; i < n; i += THR) {
                uint16_t a;
                if constexpr (WIRE == 0) { a = f2bf(bf2f(xr[i]) + bf2f(pr[i])); xr[i] = a; }
                else                     { a = xr[i]; }
                if constexpr (THR == kR4dSumLanes) { const float av = bf2f(a); s1s += av * av; }
            }
            if constexpr (THR != kR4dSumLanes) { __syncthreads(); s1s = lanes_sumsq(xr, n); }
            const float s1 = rsqrtf(block_sum<THR / kWave>(s1s, lds) / (float)n + eps);
            __syncthreads();   // lds is reused by the second sum
            for (uint32_t i = tid; i < n; i += THR) {
                const float g = WF32 ? ((const float*)wp)[i] : bf2f(((const uint16_t*)wp)[i]);
                const uint16_t pv = f2bf(bf2f(xr[i]) * s1 * (g + wadd));
                const uint16_t h = f2bf(bf2f(rr[i]) + bf2f(pv));
                rr[i] = h;
                if constexpr (THR == kR4dSumLanes) { const float v = bf2f(h); ss += v * v; }
            }
        } else
        for (uint32_t i = tid; i < n; i += THR) {
            /* The exact wire reduces here, in the norm's first pass; the compressed one already
             * reduced into `x`. */
            uint16_t a;
            if constexpr (WIRE == 0) { a = f2bf(bf2f(xr[i]) + bf2f(pr[i])); xr[i] = a; }
            else                     { a = xr[i]; }
            float v;
            if (RES) { const uint16_t h = f2bf(bf2f(rr[i]) + bf2f(a)); rr[i] = h; v = bf2f(h); }
            else     { v = bf2f(a); }
            if constexpr (THR == kR4dSumLanes) ss += v * v;
            else (void)v;
        }
        if constexpr (THR != kR4dSumLanes) { __syncthreads(); ss = lanes_sumsq(nr, n); }
        /* block_sum's barrier is also what publishes pass one's stores to pass two's reads. */
        const float rs = rsqrtf(block_sum<THR / kWave>(ss, lds) / (float)n + eps);
        norm_quant_row<WF32, THR, HI8, SUB>(nr, w, y ? y + (size_t)row * y_ld : nullptr,
                                            q + (size_t)row * q_ld, s, (size_t)row * s_ld, n, nkb,
                                            rs, wadd);
        /* lds is reused by this block's next row; without the barrier a thread still reading it
         * would race the next row's write. */
        __syncthreads();
    }
    if (CH::on) r4d_chain_done();
}

}  // namespace

// `w` is the norm gain, f32 when `w_f32` and bf16 otherwise (see r4d_model_bf16's rmsnorm). `y`
// may be null when nothing reads the bf16 post-norm tensor; every current caller has a reader (the
// fp8a16 matvec at M = 1).
//
// `res`, when given, is the fused_add_rms_norm form: `res += x` in f32, rounded to bf16 and stored,
// and the norm is taken of the stored value, bit-identical to `add` + `rmsnorm`. Pass 0 for the
// plain form. It may alias `y`; see the note at the top of the file.
//
//   x    [M][n]              bf16, `x_ld` elements a row (0 = n)
//   res  [M][n]              bf16, `r_ld` elements a row (0 = n), optional, read and written
//   w    [n]                 f32 or bf16
//   y    [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q    [M][n]              E4M3 (rotated int8 when `hi8`), `q_ld` bytes a row (0 = n), rounded
//                            up to 4
//   s    [M][ceil(n/128)]    f32,  `s_ld` elements a row (0 = ceil(n/128))
//
// `hi8` additionally requires n % 128 == 0, as gated_quant_fp8_launch's `rot`. `wide` (with
// `hi8`) is IQ8R's group: one scale per 256 columns, n % 256 == 0, `s` [M][n/256]. `chain` (with
// `hi8`) takes the chained form at decode heights, where the row resolved with it.
static int rmsnorm_quant_launch(long x, long res, long w, long y, long q, long s, int M, int n,
                                float eps, float wadd, int w_f32, int x_ld, int r_ld, int y_ld,
                                int q_ld, int s_ld, long stream, bool hi8, bool wide = false,
                                bool chain = false) {
    if (!x || !w || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;              // a lane owns four columns; see the kernel
    if (hi8 && (n % (int)kGroup)) return -6;
    if (wide && (!hi8 || n % (2 * (int)kGroup))) return -6;
    if (!x_ld) x_ld = n;
    if (!r_ld) r_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    const int grp = wide ? 2 * (int)kGroup : (int)kGroup;
    const int nkb = (n + grp - 1) / grp;
    if (!s_ld) s_ld = nkb;

    // One workgroup per row, so a decode batch is 8-72 workgroups and the kernel is latency bound;
    // the workgroup width is the parallelism knob (crossover table in r4d_act_threads). The sum of
    // squares is folded in kR4dSumLanes' partition at every width, as rmsnorm_bf16 folds it.
    const int thr = r4d_act_threads(M, n);
    const bool ch = chain && hi8 && M <= 64;
#define R4D_RQ_K(K_)                                                                            \
    hipLaunchKernelGGL((K_), dim3((uint32_t)M), dim3((uint32_t)thr), 0,                          \
                       (hipStream_t)stream, (const uint16_t*)x, (uint16_t*)res, (const void*)w,  \
                       (uint16_t*)y, (uint8_t*)q, (float*)s, (uint32_t)n, (uint32_t)nkb, eps,    \
                       wadd, (uint32_t)x_ld, (uint32_t)r_ld, (uint32_t)y_ld, (uint32_t)q_ld,     \
                       (uint32_t)s_ld)
#define R4D_RQ_H(wf, rs, T, H, SUB) R4D_RQ_K((rmsnorm_quant_fp8_kernel<wf, rs, T, H, SUB>))
/* The rotated int8 forms, plain or chained. */
#define R4D_RQ_C(wf, rs, T, SUB)                                                                \
    do { if (ch) R4D_RQ_K((r4dchain_rmsnorm_quant<wf, rs, T, true, SUB>));                      \
         else R4D_RQ_H(wf, rs, T, true, SUB); } while (0)
#define R4D_RQ_T(wf, rs, T)                                                                     \
    do { if (wide) R4D_RQ_C(wf, rs, T, 2u);                                                    \
         else if (hi8) R4D_RQ_C(wf, rs, T, 1u); else R4D_RQ_H(wf, rs, T, false, 1u); } while (0)
#define R4D_RQ(wf, rs)                                                                          \
    do { if (thr == 1024)     R4D_RQ_T(wf, rs, 1024);                                           \
         else if (thr == 512) R4D_RQ_T(wf, rs, 512);                                            \
         else                 R4D_RQ_T(wf, rs, 256); } while (0)
    if (w_f32) { if (res) R4D_RQ(true, true);  else R4D_RQ(true, false); }
    else       { if (res) R4D_RQ(false, true); else R4D_RQ(false, false); }
#undef R4D_RQ
#undef R4D_RQ_T
#undef R4D_RQ_C
#undef R4D_RQ_H
#undef R4D_RQ_K
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_rmsnorm_quant_fp8(long x, long res, long w, long y, long q, long s, int M, int n,
                                     float eps, float wadd, int w_f32, int x_ld, int r_ld,
                                     int y_ld, int q_ld, int s_ld, long stream) {
    return rmsnorm_quant_launch(x, res, w, y, q, s, M, n, eps, wadd, w_f32, x_ld, r_ld, y_ld,
                                q_ld, s_ld, stream, false);
}

// The same norm with had_quant_act_i8g's quantiser on its pass: the width-128 Hadamard, then int8
// codes at amax / 127 per (row, 128). For a rotated int8 linear (dtype i8a8h); `q` is int8.
extern "C" int r4d_rmsnorm_had_quant_i8g(long x, long res, long w, long y, long q, long s, int M,
                                         int n, float eps, float wadd, int w_f32, int x_ld,
                                         int r_ld, int y_ld, int q_ld, int s_ld, long stream) {
    return rmsnorm_quant_launch(x, res, w, y, q, s, M, n, eps, wadd, w_f32, x_ld, r_ld, y_ld,
                                q_ld, s_ld, stream, true);
}

// `mode` is 0 for silu (down_proj's input) and 1 for sigmoid (a gated attention o_proj's), as in
// r4d_gated_had_quant_i8.
//
//   gu [M][2n]             bf16, gate first then up, `gu_ld` elements a row (0 = 2n)
//   y  [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q  [M][n]              E4M3 (int8 when `i8`), `q_ld` bytes a row (0 = n), rounded up to 4
//   s  [M][ceil(n/128)]    f32,  `s_ld` elements a row (0 = ceil(n/128))
//
// `rot` additionally requires n % 128 == 0: a ragged last group is fine unrotated (zero-padded,
// masked by the GEMM) but a butterfly mixing real values with padding has no matching weight block.
// `rot` with `i8` is had_quant_act_i8g's form (gated_had_quant_i8g), for a rotated int8 down
// projection; `rot` alone pairs with a rotated four-bit expert, which reads E4M3.
// `wide` (with `rot` and `i8`): IQ8R's group, one scale per 256 columns, n % 256 == 0. `chain`
// (with `rot`) takes the chained form at decode heights: to 64 rows at IQ8R's group, to 1024 at
// the 128 group, whose rows are a routed layer's (token, expert) slots.
static int gated_quant_fp8_launch(long gu, long y, long q, long s, int M, int n, int mode,
                                  int gu_ld, int y_ld, int q_ld, int s_ld, RadStream stream,
                                  bool rot, bool i8 = false, bool wide = false,
                                  bool chain = false) {
    int emit_rc = RAD_OK;
    if (!gu || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;
    if (mode != 0 && mode != 1) return -5;
    if (rot && (n % (int)kGroup)) return -6;
    if (wide && (!rot || !i8 || n % (2 * (int)kGroup))) return -6;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    const int grp = wide ? 2 * (int)kGroup : (int)kGroup;
    const int nkb = (n + grp - 1) / grp;
    if (!s_ld) s_ld = nkb;
    // One wave per group, capped at r4d_act_threads' width. The width does not affect the numerics:
    // there is no cross-row reduction and the rotation is wave-local (r4d_fwht.h).
    const int thr = std::min(nkb, r4d_act_threads(M, n) / (int)kWave) * (int)kWave;
    const bool ch = chain && rot && M <= (wide ? 64 : 64 * 16);
#define R4D_GQ_K(K_)                                                                            \
    emit_rc = R4D_EMIT(stream, K_, dim3((uint32_t)M), dim3(thr),                                 \
                       0, (const uint16_t*)gu, (uint16_t*)y, (uint8_t*)q,                      \
                       (float*)s, (uint32_t)n, (uint32_t)nkb, (uint32_t)gu_ld, (uint32_t)y_ld,  \
                       (uint32_t)q_ld, (uint32_t)s_ld)
#define R4D_GQ_R(a, R, I, SUB) R4D_GQ_K((gated_quant_fp8_kernel<a, R, I, SUB>))
/* The rotated forms (int8 codes or E4M3), plain or chained. */
#define R4D_GQ_C(a, I, SUB)                                                                     \
    do { if (ch) R4D_GQ_K((r4dchain_gated_quant<a, true, I, SUB>));                              \
         else R4D_GQ_R(a, true, I, SUB); } while (0)
#define R4D_GQ(a)                                                                               \
    do { if (wide) R4D_GQ_C(a, true, 2u);                                                       \
         else if (rot && i8) R4D_GQ_C(a, true, 1u);                                             \
         else if (rot) R4D_GQ_C(a, false, 1u);                                                  \
         else if (i8) R4D_GQ_R(a, false, true, 1u);                                             \
         else R4D_GQ_R(a, false, false, 1u); } while (0)
    if (mode == 0) R4D_GQ(0);
    else           R4D_GQ(1);
#undef R4D_GQ
#undef R4D_GQ_C
#undef R4D_GQ_R
#undef R4D_GQ_K
    if (emit_rc != RAD_OK) return -4;
    return (hipGetLastError() != hipSuccess) ? -4 : 0;
}

extern "C" int r4d_gated_quant_fp8(long gu, long y, long q, long s, int M, int n, int mode,
                                   int gu_ld, int y_ld, int q_ld, int s_ld, long stream) {
    return gated_quant_fp8_launch(gu, y, q, s, M, n, mode, gu_ld, y_ld, q_ld, s_ld,
                                  (RadStream)(uintptr_t)stream, false);
}

// The same product with had_quant_act_i8g's quantiser: `q` is int8.
extern "C" int r4d_gated_had_quant_i8g(long gu, long y, long q, long s, int M, int n, int mode,
                                       int gu_ld, int y_ld, int q_ld, int s_ld, long stream) {
    return gated_quant_fp8_launch(gu, y, q, s, M, n, mode, gu_ld, y_ld, q_ld, s_ld,
                                  (RadStream)(uintptr_t)stream, true, true);
}

// act . x with the gate and x in separate buffers; see the kernel for why a row is a (token, head)
// pair and why the activation is rounded to bf16 before the product. `mode` as in
// r4d_gated_quant_fp8.
//
//   g  [M][.]              bf16, `g_ld` elements a row; the gate starts at element 0 of the row,
//                          so a caller with a [q|gate] projection passes g already offset
//   x  [M][n]              bf16, `x_ld` elements a row (0 = n)
//   y  [M][n]              bf16, `y_ld` elements a row (0 = n), optional
//   q  [M][n]              E4M3 (int8 when `i8`), `q_ld` bytes a row (0 = n), rounded up to 4
//   s  [M][ceil(n/128)]    f32,  `s_ld` elements a row (0 = ceil(n/128))
// `wide` (with `rot`): IQ8R's group, one scale per 256 columns of a row, n % 256 == 0. `chain`
// (with `rot`) takes the chained form at decode heights.
static int gate_quant_fp8_emit(long g, long x, long y, long q, long s, int M, int n, int mode,
                               int g_ld, int x_ld, int y_ld, int q_ld, int s_ld,
                               RadStream stream, bool i8 = false, bool rot = false,
                               bool wide = false, bool chain = false) {
    int emit_rc = RAD_OK;
    if (!g || !x || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 4) return -3;
    if (rot && (!i8 || n % (int)kGroup)) return -3;
    if (wide && (!rot || n % (2 * (int)kGroup))) return -3;
    if (mode < 0 || mode > 2) return -5;
    if (!g_ld) g_ld = n;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    const int grp = wide ? 2 * (int)kGroup : (int)kGroup;
    const int nkb = (n + grp - 1) / grp;
    if (!s_ld) s_ld = nkb;
    // A wave a group, capped at r4d_act_threads' width, as gated_quant_fp8 above.
    const int thr = std::min(nkb, r4d_act_threads(M, n) / (int)kWave) * (int)kWave;
    /* A row is a (token, head) pair: decode heights are 64 tokens of up to 32 heads. */
    const bool ch = chain && rot && M <= 64 * 32;
#define R4D_GTQ_K(K_)                                                                           \
    R4D_EMIT(stream, K_, dim3((uint32_t)M), dim3(thr), 0, (const uint16_t*)g,                    \
             (const uint16_t*)x, (uint16_t*)y, (uint8_t*)q, (float*)s, (uint32_t)n,             \
             (uint32_t)nkb, (uint32_t)g_ld, (uint32_t)x_ld, (uint32_t)y_ld, (uint32_t)q_ld,     \
             (uint32_t)s_ld)
/* The rotated int8 forms (128, or IQ8R's 256), plain or chained; else the E4M3 or plain int8 one. */
#define R4D_GTQ_I(a, I)                                                                         \
    emit_rc = wide ? (ch ? R4D_GTQ_K((r4dchain_gate_quant<a, true, true, 2u>))                  \
                         : R4D_GTQ_K((gate_quant_fp8_kernel<a, true, true, 2u>)))               \
            : rot  ? (ch ? R4D_GTQ_K((r4dchain_gate_quant<a, true, true>))                      \
                         : R4D_GTQ_K((gate_quant_fp8_kernel<a, true, true>)))                   \
                   : R4D_GTQ_K((gate_quant_fp8_kernel<a, I>))
#define R4D_GTQ(a) do { if (i8) R4D_GTQ_I(a, true); else R4D_GTQ_I(a, false); } while (0)
    if (mode == 0) R4D_GTQ(0);
    else if (mode == 1) R4D_GTQ(1);
    else R4D_GTQ(2);
#undef R4D_GTQ
#undef R4D_GTQ_I
#undef R4D_GTQ_K
    if (emit_rc != RAD_OK) return -4;
    return (hipGetLastError() != hipSuccess) ? -4 : 0;
}

extern "C" int r4d_gate_quant_fp8(long g, long x, long y, long q, long s, int M, int n, int mode,
                                  int g_ld, int x_ld, int y_ld, int q_ld, int s_ld, long stream) {
    return gate_quant_fp8_emit(g, x, y, q, s, M, n, mode, g_ld, x_ld, y_ld, q_ld, s_ld,
                               (RadStream)(uintptr_t)stream);
}

// The fused all-reduce. The IPC arguments are r4d_ar_oneshot_2rank_exact's (double-buffered
// scratch, per-block flags, device-resident sequence counters); the rest are r4d_rmsnorm_quant_fp8's,
// except:
//
//   * `x` has no pitch: it is also the all-reduce message, pushed as a range of 16B words, and a
//     pitch would make the ranks disagree about a block's bytes.
//   * `nblocks` is clamped to M, since a block owns whole rows.
//
// `res` behaves as in r4d_rmsnorm_quant_fp8. Returns 0, or a negative code for an unserved shape.
extern "C" int r4d_ar_rmsnorm_quant_fp8_chunk_elems(void) { return ARQ_CE; }

// `hi8` is the rotated int8 form (ar_rmsnorm_had_quant_i8g): `q` is int8, n % 128 == 0, and there
// is no post-norm. `wide` (with `hi8`) is IQ8R's group: one scale per 256 columns, n % 256 == 0,
// `s` [M][n/256]. `chain` takes the chained form (r4d_chain.h), which the rotated int8 form has at
// the decode width of 1024 threads; any other launch stays plain.
static int ar_rmsnorm_quant_launch(long peer_scratch, long my_scratch, long peer_flags,
                                   long my_flags, long seq_ctrs, long slot_stride16,
                                   long x, long res, long w, long y, long q, long s,
                                   int M, int n, float eps, float wadd, int w_f32,
                                   int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                                   int nblocks, int nthreads, int drain, int acq, int wire,
                                   long loc_pack, long loc_scale_off, long wp, bool hi8,
                                   long stage, bool wide = false, bool chain = false) {
    if (!x || !w || !q || !s) return -1;
    if (wp && !res) return -10;                  // a post-norm needs the stream
    if (hi8 && wp) return -11;
    if (hi8 && (n % (int)kGroup)) return -12;
    if (wide && (!hi8 || n % (2 * (int)kGroup))) return -12;
    if (!peer_scratch || !my_scratch || !peer_flags || !my_flags || !seq_ctrs) return -1;
    if (M <= 0 || n <= 0) return -2;
    // n % 8: n bf16 is a whole number of 16B words (see the push), which also gives every lane its
    // four columns (see norm_quant_row).
    if (n % 8) return -5;
    if (wire != 0 && wire != 6) return -7;
    // A group may not straddle a row, or the ranks would disagree about a block's bytes; a row's
    // last chunk may be partial (its missing groups go out as zeros).
    if (wire == 6 && (n % ARQ_GROUP)) return -8;
    if (wire == 6 && (!loc_pack || loc_scale_off <= 0)) return -9;
    if (!r_ld) r_ld = n;
    if (!y_ld) y_ld = n;
    if (!q_ld) q_ld = n;
    const int grp = wide ? 2 * (int)kGroup : (int)kGroup;
    const int nkb = (n + grp - 1) / grp;
    if (!s_ld) s_ld = nkb;

    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;
    const int nbw = r4d_ar_max_blocks();       // the counter capacity; see r4d_ar_seq.h
    if (nb > nbw) nb = nbw;
    // The standalone norm's width; the sums fold in kR4dSumLanes' partition at any of the three.
    const int thr = nthreads ? nthreads : r4d_act_threads(M, n);
    if (thr != 256 && thr != 512 && thr != 1024) return -6;
    const bool pad = wire == 6 && (n % ARQ_CE) != 0;
    const bool ch = chain && hi8 && thr == 1024;
#define R4D_ARQ_HP(wf, rs, T, W, PO, H, PD, SUB, CH)                                             \
    hipLaunchKernelGGL((ar_rmsnorm_quant_fp8_kernel<wf, rs, T, W, PO, H, PD, SUB, CH>),           \
                       dim3((uint32_t)nb),                                                       \
                       dim3(T), 0,                                                               \
                       (hipStream_t)stream, (uint16_t*)x, (uint16_t*)res, (const void*)w,        \
                       (const void*)wp,                                                          \
                       (uint16_t*)y, (uint8_t*)q, (float*)s, (int4*)peer_scratch,                \
                       (const int4*)my_scratch, (unsigned int*)peer_flags,                       \
                       (unsigned int*)my_flags, (unsigned int*)seq_ctrs, (char*)loc_pack,        \
                       (long)loc_scale_off, M, (uint32_t)n,                                      \
                       (uint32_t)nkb, eps, wadd, (uint32_t)r_ld, (uint32_t)y_ld, (uint32_t)q_ld, \
                       (uint32_t)s_ld, (int)slot_stride16, drain, acq, (unsigned)nbw, stage)
#define R4D_ARQ_H(wf, rs, T, W, PO, H, SUB, CH)                                                  \
    do { if (pad) R4D_ARQ_HP(wf, rs, T, W, PO, H, (W == 6), SUB, CH);                            \
         else R4D_ARQ_HP(wf, rs, T, W, PO, H, false, SUB, CH); } while (0)
#define R4D_ARQ_P(wf, rs, T, W, PO) R4D_ARQ_H(wf, rs, T, W, PO, false, 1u, R4dUnchained)
#define R4D_ARQ_T(wf, rs, T, W, CH)                                                              \
    do { if (wide) R4D_ARQ_H(wf, rs, T, W, false, true, 2u, CH);                                 \
         else if (hi8) R4D_ARQ_H(wf, rs, T, W, false, true, 1u, CH);                             \
         else R4D_ARQ_P(wf, rs, T, W, false); } while (0)
#define R4D_ARQ_W(wf, rs, T, CH)                                                                 \
    do { if (wire == 6 && rs && wp) R4D_ARQ_P(wf, rs, T, 6, true);                               \
         else if (wire == 6) R4D_ARQ_T(wf, rs, T, 6, CH);                                        \
         else if (rs && wp) R4D_ARQ_P(wf, rs, T, 0, true);                                       \
         else R4D_ARQ_T(wf, rs, T, 0, CH); } while (0)
#define R4D_ARQ(wf, rs)                                                                          \
    do { if (ch)              R4D_ARQ_W(wf, rs, 1024, r4dchain);                                 \
         else if (thr == 1024) R4D_ARQ_W(wf, rs, 1024, R4dUnchained);                            \
         else if (thr == 512) R4D_ARQ_W(wf, rs, 512, R4dUnchained);                              \
         else                 R4D_ARQ_W(wf, rs, 256, R4dUnchained); } while (0)
    if (w_f32) { if (res) R4D_ARQ(true, true);  else R4D_ARQ(true, false); }
    else       { if (res) R4D_ARQ(false, true); else R4D_ARQ(false, false); }
#undef R4D_ARQ
#undef R4D_ARQ_W
#undef R4D_ARQ_T
#undef R4D_ARQ_P
#undef R4D_ARQ_H
#undef R4D_ARQ_HP
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_ar_rmsnorm_quant_fp8(long peer_scratch, long my_scratch, long peer_flags,
                                        long my_flags, long seq_ctrs, long slot_stride16,
                                        long x, long res, long w, long y, long q, long s,
                                        int M, int n, float eps, float wadd, int w_f32,
                                        int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                                        int nblocks, int nthreads, int drain, int acq, int wire,
                                        long loc_pack, long loc_scale_off, long wp, long stage) {
    return ar_rmsnorm_quant_launch(peer_scratch, my_scratch, peer_flags, my_flags, seq_ctrs,
                                   slot_stride16, x, res, w, y, q, s, M, n, eps, wadd, w_f32,
                                   r_ld, y_ld, q_ld, s_ld, stream, nblocks, nthreads, drain, acq,
                                   wire, loc_pack, loc_scale_off, wp, false, stage);
}

// The same with had_quant_act_i8g's quantiser on the norm's pass (no post-norm). `group` is the
// scale group: 128, or IQ8R's 256. `chain`: the chained form at the decode width.
extern "C" int r4d_ar_rmsnorm_had_quant_i8g(long peer_scratch, long my_scratch, long peer_flags,
                                            long my_flags, long seq_ctrs, long slot_stride16,
                                            long x, long res, long w, long y, long q, long s,
                                            int M, int n, float eps, float wadd, int w_f32,
                                            int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                                            int nblocks, int nthreads, int drain, int acq,
                                            int wire, long loc_pack, long loc_scale_off,
                                            long stage, int group, int chain) {
    if (group != (int)kGroup && group != 2 * (int)kGroup) return -12;
    return ar_rmsnorm_quant_launch(peer_scratch, my_scratch, peer_flags, my_flags, seq_ctrs,
                                   slot_stride16, x, res, w, y, q, s, M, n, eps, wadd, w_f32,
                                   r_ld, y_ld, q_ld, s_ld, stream, nblocks, nthreads, drain, acq,
                                   wire, loc_pack, loc_scale_off, 0, true, stage,
                                   group == 2 * (int)kGroup, chain != 0);
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h; see libr4d/r4d_args.h.
/* ================================================================== the two fused quantisers
 *
 * The standalone quantiser's shape rules, applied to the operand that produces the row. The bf16
 * output is optional so a layer without a bf16 consumer skips the store. */

/* `hi8`: rmsnorm_had_quant_i8g, whose codes are int8 and whose row is whole 128s (the rotation's
 * width). Separate ops for the reason given at r4d_rad_gated_had_quant_fp8. */
static int rad_rmsnorm_quant(const RadArgs* a, RadStream s, bool hi8) {
    const RadTensor* x  = r4d_opt(a, RQ_X);
    const RadTensor* rs = r4d_opt(a, RQ_RES);
    const RadTensor* w  = r4d_opt(a, RQ_W);
    const RadTensor* q  = r4d_opt(a, RQ_Q);
    const RadTensor* sc = r4d_opt(a, RQ_S);
    const RadTensor* y  = r4d_opt(a, RQ_Y);
    if (!x || !w || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (q->dtype != (hi8 ? RAD_I8 : RAD_F8E4M3)) return R4D_NO(RAD_E_DTYPE);
    /* Rank 2 on the input too, unlike the standalone quantiser: a norm has one gain per channel,
     * and folding trailing dims would spread a gain across heads. */
    if (x->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0], K = x->shape[1];
    /* 128, or IQ8R's 256 for the rotated int8 form. */
    const long long g0 = r4d_quant_act_fp8_group();
    const long long grp = r4d_geti_or(a, "group", g0);
    const bool wide = hi8 && grp == R4D_I8_GROUP_WIDE;
    if (grp != g0 && !wide) return R4D_NO(RAD_E_SHAPE);
    const long long nkb = (K + grp - 1) / grp;
    if (r4d_numel(w) != K) return R4D_NO(RAD_E_SHAPE);
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (q->shape[0] != M || q->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != K)) return R4D_NO(RAD_E_SHAPE);
    if (hi8 && K % grp) return R4D_NO(RAD_E_SHAPE);
    /* The residual has the input's extent and is read and written. It may alias `out_bf16` (see
     * the note at the top of the file). */
    if (rs && (rs->rank != 2 || rs->shape[0] != M || rs->shape[1] != K)) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = y ? r4d_row_pitch(y) : 0, rp = rs ? r4d_row_pitch(rs) : 0;
    if (xp < 0 || qp < 0 || sp < 0 || yp < 0 || rp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = rmsnorm_quant_launch(r4d_addr(x), rs ? r4d_addr(rs) : 0, r4d_addr(w),
                                       y ? r4d_addr(y) : 0, r4d_addr(q), r4d_addr(sc),
                                       (int)M, (int)K,
                                       (float)r4d_getf(a, "eps", 1e-6), (float)r4d_getf(a, "wadd", 0.0),
                                       w->dtype == RAD_F32, (int)xp, (int)rp, (int)yp, (int)qp,
                                       (int)sp, (long)(uintptr_t)s, hi8, wide,
                                       r4d_geti_or(a, "chain", 0) != 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_rmsnorm_quant_fp8(const RadArgs* a, RadStream s) {
    return rad_rmsnorm_quant(a, s, false);
}

extern "C" int r4d_rad_rmsnorm_had_quant_i8g(const RadArgs* a, RadStream s) {
    return rad_rmsnorm_quant(a, s, true);
}

/* The gate's activation: `act` by name overrides `mode` (libr4d's integer: 0 silu, 1 sigmoid) when
 * both are given, as for r4d_gated_had_quant_i8. Returns -1 for an unknown name. */
static int act_mode(const RadArgs* a, int dflt) {
    const char* act = r4d_gets(a, "act");
    if (!act) return (int)r4d_geti_or(a, "mode", dflt);
    if (!std::strcmp(act, "silu")) return 0;
    if (!std::strcmp(act, "sigmoid")) return 1;
    if (!std::strcmp(act, "none")) return 2;
    return -1;
}

/* `hi8` (with `rot`): gated_had_quant_i8g, int8 codes only. */
static int rad_gated_quant_fp8(const RadArgs* a, RadStream stream, bool rot, bool hi8 = false) {
    const RadTensor* gu = r4d_opt(a, GQ_GU);
    const RadTensor* q  = r4d_opt(a, GQ_Q);
    const RadTensor* sc = r4d_opt(a, GQ_S);
    const RadTensor* y  = r4d_opt(a, GQ_Y);
    if (!gu || !q || !sc) return R4D_NO(RAD_E_INVAL);
    /* The codes' dtype picks the quantiser: E4M3, or int8 for an int8 down projection. The two
     * rotated ops each take one: E4M3 for a four-bit expert, int8 for a rotated int8 weight. */
    if (q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    if (rot && q->dtype != (hi8 ? RAD_I8 : RAD_F8E4M3)) return R4D_NO(RAD_E_DTYPE);
    if (gu->rank != 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = gu->shape[0], K = q->shape[1];
    /* 128, or IQ8R's 256 for the rotated int8 form. */
    const long long g0 = r4d_quant_act_fp8_group();
    const long long grp = r4d_geti_or(a, "group", g0);
    const bool wide = rot && hi8 && grp == R4D_I8_GROUP_WIDE;
    if (grp != g0 && !wide) return R4D_NO(RAD_E_SHAPE);
    const long long nkb = (K + grp - 1) / grp;
    /* The gated operand is [M][2K], gate then up, as the fused gate_up projection writes it. */
    if (gu->shape[1] != 2 * K) return R4D_NO(RAD_E_SHAPE);
    if (q->shape[0] != M) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != K)) return R4D_NO(RAD_E_SHAPE);
    const long long gp = r4d_row_pitch(gu), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    const long long yp = y ? r4d_row_pitch(y) : 0;
    if (gp < 0 || qp < 0 || sp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    const int mode = act_mode(a, 0);                 /* silu: the gated MLP */
    if (mode < 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const int e = gated_quant_fp8_launch(r4d_addr(gu), y ? r4d_addr(y) : 0, r4d_addr(q),
                                         r4d_addr(sc), (int)M, (int)K, mode, (int)gp, (int)yp,
                                         (int)qp, (int)sp, stream, rot, q->dtype == RAD_I8, wide,
                                         r4d_geti_or(a, "chain", 0) != 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gated_quant_fp8(const RadArgs* a, RadStream stream) {
    return rad_gated_quant_fp8(a, stream, false);
}

/* Its int8 twin, had_quant_act_i8g's codes: pairs only with a weight stored at dtype i8a8h. */
extern "C" int r4d_rad_gated_had_quant_i8g(const RadArgs* a, RadStream stream) {
    return rad_gated_quant_fp8(a, stream, true, true);
}

/* Elements between consecutive `width`-wide rows, as r4d_gdn.cpp's gn_row_pitch: a
 * [.., heads, width] operand steps by the stride of the second-to-last axis; a flattened one is
 * contiguous and steps by `width`. This reads a column view of an interleaved projection (a gated
 * attention's gate). */
static long long tq_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] != width) return r4d_contig(t) ? width : -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 1) return width;
    /* Rows must also be uniformly spaced: a [T, n_head, head_dim] column view flattens to
     * T * n_head rows only if T's stride is n_head of them, which fails if an outer axis was also
     * sliced. Refuse otherwise; a wrong pitch would read the next head's gate. */
    for (int i = (int)t->rank - 2; i >= 1; --i)
        if (t->stride[i - 1] != t->shape[i] * t->stride[i]) return -1;
    return t->stride[t->rank - 2];
}

/* Two-operand form of gated_quant_fp8 for gated attention: `x` is the attention output and the
 * gate a column view of the [q|gate] projection. The gate offset comes from the view, so there is
 * no `gate_offset` parameter. */
static int rad_gate_quant(const RadArgs* a, RadStream stream, bool rot) {
    const RadTensor* g  = r4d_opt(a, TQ_G);
    const RadTensor* x  = r4d_opt(a, TQ_X);
    const RadTensor* q  = r4d_opt(a, TQ_Q);
    const RadTensor* sc = r4d_opt(a, TQ_S);
    const RadTensor* y  = r4d_opt(a, TQ_Y);
    if (!g || !x || !q || !sc) return R4D_NO(RAD_E_INVAL);
    if (q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    if (rot && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    /* M is numel(x) / n. The `M` parameter is the band's declared maximum, not this issue's, and
     * shape[0] counts tokens while a row here is a (token, head) pair. */
    const long long K = r4d_geti_or(a, "n", 0);
    /* 128, or IQ8R's 256 for the rotated form. */
    const long long g0 = r4d_quant_act_fp8_group();
    const long long grp = r4d_geti_or(a, "group", g0);
    const bool wide = rot && grp == R4D_I8_GROUP_WIDE;
    if (grp != g0 && !wide) return R4D_NO(RAD_E_SHAPE);
    if (K <= 0 || r4d_numel(x) % K) return R4D_NO(RAD_E_SHAPE);
    const long long M = r4d_numel(x) / K;
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long nkb = (K + grp - 1) / grp;
    if (r4d_numel(g) < M * K) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(q) < M * K || r4d_numel(sc) < M * nkb) return R4D_NO(RAD_E_SHAPE);
    if (y && r4d_numel(y) < M * K) return R4D_NO(RAD_E_SHAPE);
    const long long gp = tq_pitch(g, K), xp = tq_pitch(x, K);
    const long long qp = tq_pitch(q, K), sp = tq_pitch(sc, nkb);
    const long long yp = y ? tq_pitch(y, K) : 0;
    if (gp < 0 || xp < 0 || qp < 0 || sp < 0 || yp < 0) return R4D_NO(RAD_E_STRIDE);
    const int mode = act_mode(a, 1);                 /* sigmoid: gated attention */
    if (mode < 0) return R4D_NO(RAD_E_UNSUPPORTED);
    const int e = gate_quant_fp8_emit(r4d_addr(g), r4d_addr(x), y ? r4d_addr(y) : 0,
                                      r4d_addr(q), r4d_addr(sc), (int)M, (int)K, mode,
                                      (int)gp, (int)xp, (int)yp, (int)qp, (int)sp, stream,
                                      q->dtype == RAD_I8, rot, wide,
                                      r4d_geti_or(a, "chain", 0) != 0);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int r4d_rad_gate_quant_fp8(const RadArgs* a, RadStream stream) {
    return rad_gate_quant(a, stream, false);
}

/* gate_quant_fp8 at had_quant_act_i8g's codes, for an out projection stored rotated (i8a8h). */
extern "C" int r4d_rad_gate_had_quant_i8g(const RadArgs* a, RadStream stream) {
    return rad_gate_quant(a, stream, true);
}

