// docs/OPS.md's decomposed sampler chain, on the device. r4d_sample_chain_f32 does temperature,
// softmax, top_k, top_p, min_p and the draw in one workgroup in vLLM's fixed order. This engine's
// chain is eleven ops composed per request (core/sample/sampler.h), with penalties, DRY, a grammar
// bitmask, locally-typical and XTC stages the fused kernel lacks, and stages a request may skip.
// Since fusion is a selection (spec §2.3), both exist: this unit runs a served request's sampler;
// r4d_sample_chain_f32 serves the speculative-verify path and its three outputs.
//
// Every scalar is per row: `params` is an array of RadSampleParams, one row per sampled position,
// uploaded once a step. The struct is included from abi/rad_sample.h rather than restated, since it
// is a wire format between the core and this library.
//
// What is exact, and why. libref is the oracle (spec §17), and two samplers that disagree on one
// candidate's survival emit different tokens, which compounds:
//
//   * The top-k is exact, without ref's O(k*n_vocab) repeated-maximum scan: eight radix passes over
//     a monotone 32-bit key, four for the k-th largest logit and four over the token id to break a
//     tie at that logit by the lower id (ref's rule), then a bitonic sort of the k survivors, since
//     every later stage assumes descending order.
//
//   * The sequential stages stay sequential: `top_p` accumulates a cumulative probability in f32 in
//     candidate order and `sample_pick` walks an inverse CDF the same way. A parallel prefix sum
//     rounds differently and would flip a candidate near the cut. Both run on one thread over at
//     most a few thousand candidates.
//
//   * The draw is the specified one: splitmix64 over (seed * golden + pos), top 24 bits as a float
//     in [0,1), fixed by rad_sample.h so a host reference, the device and a rerun agree.
//
//   * expf / logf / fabsf are the accurate forms, not __expf, because ref uses libm.
//
// A dropped candidate is index -1, value -infinity, and the buffers never shrink. Kept candidates
// stay in place in descending order, so a later stage may stop at the first hole or skip it (these
// skip). `cand_val` carries logits, and each narrowing stage takes its own softmax over the live
// set, so the stages compose in any order.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_chain.h"

#include <cfloat>

// The params row is the engine's struct, included rather than redeclared: two declarations of one
// wire format drift silently (a sampler reading `top_p` out of `min_p` still produces text).
#include "rad_sample.h"

namespace {

constexpr uint32_t kWave    = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves   = kThreads / kWave;

// The vocabulary scan gets its own width. Every kernel after `topk_kernel` reads a candidate set
// of a thousand or two words, where 256 threads is plenty; `topk_kernel` reads the whole row
// (124160 floats on this model). At 256 threads that took 117.6 us a step for 3.97 MB, 34 GB/s on
// a part that streams 620: one block per row is eight blocks on a 64-CU card, and two waves per
// SIMD cannot keep enough loads in flight. 1024 threads is eight waves per SIMD on the same grid.
//
// It changes no bit, a property of this kernel: the scan reduces with max over a packed (key, ~id)
// word and the select counts histogram bins, both associative. The narrowing stages are not:
// `sum_block` folds lds[0..kWaves) in wave order and a float sum is not associative, so widening
// them would move the last bits of top_p and min_p. Hence a second constant, not a larger
// kThreads. argmax_kernel uses the same width for the same reason.
constexpr uint32_t kArgmaxThreads = 1024;
constexpr uint32_t kTopkThreads = 1024;
constexpr uint32_t kTopkWaves   = kTopkThreads / kWave;

// The largest candidate width this unit sorts. The bitonic network runs in LDS over a power-of-two
// padding, so the bound is 4096 x 8 bytes = 32 KiB, half a workgroup's 64 KiB. The engine declares
// 1024 (core/sample/sampler.h sizes it from max_num_seqs); the shim refuses anything wider rather
// than truncating a set.
constexpr uint32_t kMaxCand = 4096;

// ---------------------------------------------------------------------------- the params row
// The engine's declaration; see the note at the include.
using SP = RadSampleParams;

enum {
    SP_GREEDY  = 1 << 0,
    SP_PENALTY = 1 << 1,
    SP_MASK    = 1 << 2,
    SP_XTC     = 1 << 3,
    SP_DRY     = 1 << 4,
    SP_LOGPROBS = 1 << 5
};

__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float uniform01(uint64_t seed, uint64_t pos) {
    const uint64_t z = splitmix64(seed * 0x9E3779B97F4A7C15ull + pos);
    return (float)(uint32_t)(z >> 40) * (1.0f / 16777216.0f);
}

// A float's order-preserving 32-bit key: flip the sign bit of a positive and every bit of a
// negative, so unsigned comparison matches float comparison. -infinity maps to 0, so a dropped
// candidate sorts last without a branch.
__device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// The wave's largest (key, ~index), by an xor butterfly on r4d_xor_lane's exchange. An integer max
// is order-independent.
__device__ __forceinline__ unsigned long long wave_max_u64(unsigned long long b) {
    unsigned long long o;
    o = r4d_xor_lane_u64<16>(b); if (o > b) b = o;
    o = r4d_xor_lane_u64<8>(b);  if (o > b) b = o;
    o = r4d_xor_lane_u64<4>(b);  if (o > b) b = o;
    o = r4d_xor_lane_u64<2>(b);  if (o > b) b = o;
    o = r4d_xor_lane_u64<1>(b);  if (o > b) b = o;
    return b;
}

__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kWaves; ++w) s += lds[w];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float v, float* lds) {
    v = r4d_wave_max(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = -INFINITY;
#pragma unroll
    for (uint32_t w = 0; w < kWaves; ++w) s = fmaxf(s, lds[w]);
    __syncthreads();
    return s;
}

// ---------------------------------------------------------------------------- sample_temp
// ref_sample_temp. temp <= 0 leaves the row alone: greedy is `sample_argmax`, and a temperature of
// zero silently meaning something else would make the chain unreadable.
// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below.
__device__ __forceinline__ void temp_body(
    float* __restrict__ lg, const SP* __restrict__ sp, uint32_t n_vocab, uint32_t lg_ld) {
    const uint32_t m = blockIdx.x;
    const float t = sp[m].temp;
    if (!(t > 0.0f) || t == 1.0f) return;
    const float inv = 1.0f / t;
    float* __restrict__ row = lg + (size_t)m * lg_ld;
    for (uint32_t i = threadIdx.x + blockIdx.y * kThreads; i < n_vocab; i += kThreads * gridDim.y)
        row[i] *= inv;
}
__global__ __launch_bounds__(kThreads) void temp_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, uint32_t n_vocab, uint32_t lg_ld) {
    temp_body(lg, sp, n_vocab, lg_ld);
}
__global__ __launch_bounds__(kThreads) void r4dchain_sample_temp(
    float* __restrict__ lg, const SP* __restrict__ sp, uint32_t n_vocab, uint32_t lg_ld) {
    r4d_chain_wait();
    temp_body(lg, sp, n_vocab, lg_ld);
    r4d_chain_done();
}

// ---------------------------------------------------------------------------- sample_mask
// ref_sample_mask. A set bit means allowed: the mask is the grammar's admissible set. Word w bit b
// covers token 32w + b, least significant first. `mask_row` is -1 for an unconstrained sequence,
// which is why the plane is shared and shorter than M.
__global__ __launch_bounds__(kThreads) void mask_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, const uint32_t* __restrict__ bm,
    uint32_t n_vocab, uint32_t lg_ld, uint32_t words, int32_t mask_rows) {
    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    if (!(p.flags & SP_MASK)) return;
    if (p.mask_row < 0 || p.mask_row >= mask_rows) return;
    float* __restrict__ row = lg + (size_t)m * lg_ld;
    const uint32_t* __restrict__ mrow = bm + (size_t)p.mask_row * words;
    for (uint32_t i = threadIdx.x + blockIdx.y * kThreads; i < n_vocab; i += kThreads * gridDim.y)
        if (!((mrow[i >> 5] >> (i & 31)) & 1u)) row[i] = -INFINITY;
}

// ---------------------------------------------------------------------------- sample_penalties
// ref_sample_penalties, in llama.cpp's order:
//
//     repetition   logit = logit > 0 ? logit / rep : logit * rep    (once per DISTINCT token)
//     frequency    logit -= count * freq
//     presence     logit -= pres                                     (once per distinct token)
//
// The repetition penalty is multiplicative and sign-dependent: dividing a negative logit would make
// it larger.
//
// Parallel over the history and still race-free: only the first occurrence of a token updates it,
// so each distinct token has exactly one writing thread (the same `first` test ref needs anyway).
//
// History ids are global and the row is this rank's shard: a token's column is its id minus
// `voff`, and a token outside the shard is another rank's to penalise.

// The window both history stages read, ref's hist_window: the row's own span of the plane, clamped,
// then its last `last_n` tokens. last_n 0 disables the stage for the row (false); negative keeps
// the whole span.
__device__ __forceinline__ bool hist_window(const SP& p, int32_t last_n, uint32_t hmax,
                                            int32_t* ho_out, int32_t* hl_out) {
    if (last_n == 0) return false;
    int32_t hl = p.hist_len;
    int32_t ho = p.hist_off < 0 ? 0 : p.hist_off;
    if (ho > (int32_t)hmax) ho = (int32_t)hmax;
    if (hl > (int32_t)hmax - ho) hl = (int32_t)hmax - ho;
    if (last_n > 0 && hl > last_n) {
        ho += hl - last_n;
        hl = last_n;
    }
    *ho_out = ho;
    *hl_out = hl;
    return hl > 0;
}

__global__ __launch_bounds__(kThreads) void penalties_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, const int32_t* __restrict__ hist,
    uint32_t n_vocab, uint32_t lg_ld, uint32_t hmax, int32_t voff) {
    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    if (!(p.flags & SP_PENALTY)) return;
    const float rep = p.rep_penalty, freq = p.freq_penalty, pres = p.pres_penalty;
    if (rep == 1.0f && freq == 0.0f && pres == 0.0f) return;

    int32_t ho = 0, hl = 0;
    if (!hist_window(p, p.penalty_last_n, hmax, &ho, &hl)) return;

    const int32_t* __restrict__ h = hist + (size_t)m * hmax + ho;
    float* __restrict__ row = lg + (size_t)m * lg_ld;

    for (int32_t i = (int32_t)threadIdx.x; i < hl; i += (int32_t)kThreads) {
        const int32_t tok = h[i];
        const int64_t col = (int64_t)tok - voff;
        if (tok < 0 || col < 0 || col >= (int64_t)n_vocab) continue;
        bool  first = true;
        int32_t count = 0;
        for (int32_t j = 0; j < hl; ++j) {
            if (h[j] != tok) continue;
            if (j < i) { first = false; break; }
            ++count;
        }
        if (!first) continue;
        float v = row[col];
        v = v > 0.0f ? v / rep : v * rep;
        v -= (float)count * freq;
        v -= pres;
        row[col] = v;
    }
}

// ---------------------------------------------------------------------------- sample_dry
// ref_sample_dry, which is llama.cpp's DRY: every token that would extend a suffix of the window
// that occurred before is penalised once, by multiplier * base^(L - allowed) for the longest repeat
// L it would extend, with repeats capped at the row's dry_rep_limit; a position the optional `brk`
// plane marks (a sequence breaker) is never penalised.
//
// One block per row, two phases, no atomics. Phase one is parallel over earlier positions j: each
// thread measures how long a suffix of the window also ends at j and stores it in LDS (zero where
// the continuation after j is exempt or another rank's). Phase two picks, for each continuation
// token, one representative position (its longest repeat, earliest on a tie), and only that thread
// writes the token's logit. One writer and one subtraction per token is llama.cpp's rule and makes
// the result independent of thread order; a per-position or atomic sum would give neither.
//
// The cost is quadratic only for a genuinely repetitive history: phase one's match loop exits at
// the first mismatch, and phase two only scans for positions that already qualify. The penalty is
// formed by repeated multiplication stopping short of overflow, as ref forms it, so the two agree
// to the bit.
__device__ __forceinline__ float dry_penalty(float multiplier, float base, int32_t e) {
    float pen = multiplier;
    if (base == 1.0f) return pen;
    const float lim = FLT_MAX / base;
    for (int32_t i = 0; i < e; ++i) {
        if (pen > lim) break;
        pen *= base;
    }
    return pen;
}

__global__ __launch_bounds__(kThreads) void dry_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, const int32_t* __restrict__ hist,
    const uint8_t* __restrict__ brk, uint32_t n_vocab, uint32_t lg_ld, uint32_t hmax,
    int32_t voff) {
    extern __shared__ uint16_t len[];
    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    // Every test below reads only the row's params, so the whole block leaves together.
    if (!(p.flags & SP_DRY) || !(p.dry_multiplier > 0.0f)) return;
    const int32_t allowed = p.dry_allowed_length > 0 ? p.dry_allowed_length : 2;

    int32_t ho = 0, hl = 0;
    if (!hist_window(p, p.dry_penalty_last_n, hmax, &ho, &hl)) return;
    if (hl <= allowed) return;
    const int32_t cap = p.dry_rep_limit > 0 && p.dry_rep_limit < hl ? p.dry_rep_limit : hl;
    if (cap < allowed) return;

    const int32_t* __restrict__ h = hist + (size_t)m * hmax + ho;
    const uint8_t* __restrict__ f = brk ? brk + (size_t)m * hmax + ho : nullptr;
    float* __restrict__ row = lg + (size_t)m * lg_ld;

    for (int32_t j = (int32_t)threadIdx.x; j + 1 < hl; j += (int32_t)kThreads) {
        const int32_t lim = j + 1 < cap ? j + 1 : cap;
        int32_t n = 0;
        while (n < lim && h[j - n] == h[hl - 1 - n]) ++n;
        const int64_t col = (int64_t)h[j + 1] - voff;
        const bool skip = (f && f[j + 1]) || h[j + 1] < 0 || col < 0 || col >= (int64_t)n_vocab;
        len[j] = skip ? (uint16_t)0 : (uint16_t)n;
    }
    __syncthreads();

    const float base = p.dry_base > 0.0f ? p.dry_base : 1.75f;
    for (int32_t j = (int32_t)threadIdx.x; j + 1 < hl; j += (int32_t)kThreads) {
        const int32_t n = len[j];
        if (n < allowed) continue;
        const int32_t next = h[j + 1];
        bool rep = true;
        for (int32_t j2 = 0; j2 + 1 < hl; ++j2) {
            if (j2 == j || h[j2 + 1] != next) continue;
            const int32_t n2 = len[j2];
            if (n2 > n || (n2 == n && j2 < j)) { rep = false; break; }
        }
        if (!rep) continue;
        const int64_t col = (int64_t)next - voff;
        row[col] -= dry_penalty(p.dry_multiplier, base, n - allowed);
    }
}

// ---------------------------------------------------------------------------- sample_argmax
// ref_sample_argmax. Greedy, ties to the lower token id (the top-k selection's rule, so a greedy
// step and a top-1 step agree). Packing (key, ~id) into one 64-bit word makes that tie rule a plain
// unsigned max.
//
// One workgroup per row, so the row is read wide. A greedy step samples one row, so the grid is one
// workgroup regardless. At a 248320-entry vocabulary (993 KB), 256 threads of scalar loads took
// 170 us (5.8 GB/s, too few loads in flight); 4x the threads with float4 loads take 15.5 us.
//
// Splitting the row across workgroups is faster (7.3 us over 128 partials) but needs an ABI scratch
// plane and a second launch, and the per-dispatch floor of ~4.5 us brings it to ~12 us, not worth
// an extra operand.
//
// The vector path needs the row aligned (base and pitch together); the launcher chooses `VEC`, and
// the scalar body is the fallback.
template <uint32_t TH, bool VEC>
__global__ __launch_bounds__(TH) void argmax_kernel(
    const float* __restrict__ lg, int32_t* __restrict__ tok, uint32_t n_vocab, uint32_t lg_ld) {
    __shared__ unsigned long long lds[TH / kWave];
    const uint32_t m = blockIdx.x;
    const float* __restrict__ row = lg + (size_t)m * lg_ld;

    unsigned long long best = 0;
    if (VEC) {
        const uint32_t  n4 = n_vocab >> 2;
        const float4* __restrict__ r4 = (const float4*)row;
        for (uint32_t j = threadIdx.x; j < n4; j += TH) {
            const float4   v = r4[j];
            const uint32_t i = j << 2;
            unsigned long long e;
            e = ((unsigned long long)fkey(v.x) << 32) | (unsigned long long)(~(i + 0u));
            if (e > best) best = e;
            e = ((unsigned long long)fkey(v.y) << 32) | (unsigned long long)(~(i + 1u));
            if (e > best) best = e;
            e = ((unsigned long long)fkey(v.z) << 32) | (unsigned long long)(~(i + 2u));
            if (e > best) best = e;
            e = ((unsigned long long)fkey(v.w) << 32) | (unsigned long long)(~(i + 3u));
            if (e > best) best = e;
        }
        /* The up to 3 entries left when the vocabulary is not a multiple of four. */
        for (uint32_t i = (n4 << 2) + threadIdx.x; i < n_vocab; i += TH) {
            const unsigned long long e =
                ((unsigned long long)fkey(row[i]) << 32) | (unsigned long long)(~i);
            if (e > best) best = e;
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n_vocab; i += TH) {
            const unsigned long long e =
                ((unsigned long long)fkey(row[i]) << 32) | (unsigned long long)(~i);
            if (e > best) best = e;
        }
    }
    best = wave_max_u64(best);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = best;
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (uint32_t w = 1; w < TH / kWave; ++w) if (lds[w] > lds[0]) lds[0] = lds[w];
    tok[m] = (int32_t)(~(uint32_t)(lds[0] & 0xffffffffull));
}

// ---------------------------------------------------------------------------- sample_topk
//
// The radix select: four passes over the row, most significant byte first, narrowing a prefix of
// the monotone key until the k-th largest is pinned exactly. `remaining` is how many of the k are
// still owed once every strictly larger prefix is counted; at the end it is how many elements equal
// to the threshold belong in the set.
//
// Only when the tie group is larger than what is owed, four more passes over the token id within
// that group find which of them: the lowest ids, ref's rule. Ids are unique, so `idx <= I` takes
// exactly the right number.
//
// DESC selects on the logit key and walks the bins from 255 down; ASC selects on the token id and
// walks from 0 up. `gate` is the tie group the ASC pass is restricted to (only elements whose key
// equals it count); the DESC pass ignores it and sees the whole row.
//
// Loads are float4 where the shape allows (as in the top-1 path below), since bytes per load
// instruction set the pass's speed. The digit comes from a scan rather than one thread walking 256
// bins while 1023 wait: the bins' running sums in the walk's direction (suffix for DESC, prefix for
// ASC) are a DPP scan per wave plus one add across the eight waves, and the chosen bin is the one
// whose running sum first reaches `remaining` (exactly one, as the sums are monotone), with `cum`
// the sum before it. A walk that never stops ends on the last bin with every bin counted, and so
// does this. Counts are integers, so the digit and `cum` match the walk. `scan` holds 256 words and
// `wt` eight.
template <bool DESC>
__device__ uint32_t radix_select(const float* __restrict__ row, uint32_t n, uint32_t gate,
                                 uint32_t rank, uint32_t* hist, uint32_t* scan, uint32_t* wt,
                                 uint32_t* remaining_out, uint32_t* tie_out) {
    const uint32_t t = threadIdx.x;
    const int      lane = (int)(t & (kWave - 1)), wave = (int)(t / kWave);
    const bool     v4 = ((uintptr_t)row & 15u) == 0 && (n & 3u) == 0;
    uint32_t prefix = 0, mask = 0, remaining = rank;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t b = t; b < 256; b += kTopkThreads) hist[b] = 0;
        __syncthreads();
        auto count = [&](uint32_t key, uint32_t i) {
            if (!DESC && key != gate) return;
            const uint32_t v = DESC ? key : i;
            if ((v & mask) != prefix) return;
            atomicAdd(&hist[(v >> shift) & 0xffu], 1u);
        };
        if (v4) {
            const float4* __restrict__ r4 = (const float4*)row;
            for (uint32_t j = t; j < (n >> 2); j += kTopkThreads) {
                const float4 f = r4[j];
                count(fkey(f.x), 4 * j);
                count(fkey(f.y), 4 * j + 1);
                count(fkey(f.z), 4 * j + 2);
                count(fkey(f.w), 4 * j + 3);
            }
        } else {
            for (uint32_t i = t; i < n; i += kTopkThreads) count(fkey(row[i]), i);
        }
        __syncthreads();
        uint32_t x = 0;
        if (t < 256) {
            x = DESC ? r4d_wave_suffix_u32(hist[t], lane) : r4d_wave_prefix_u32(hist[t], lane);
            if (lane == (DESC ? 0 : kWave - 1)) wt[wave] = x;
        }
        __syncthreads();
        if (t < 256) {
            uint32_t add = 0;
            if (DESC) for (int w = wave + 1; w < 256 / (int)kWave; ++w) add += wt[w];
            else      for (int w = 0; w < wave; ++w) add += wt[w];
            scan[t] = x + add;
        }
        __syncthreads();
        if (t < 256) {
            const uint32_t before = DESC ? (t < 255 ? scan[t + 1] : 0u) : (t > 0 ? scan[t - 1] : 0u);
            if (scan[t] >= remaining && before < remaining) { hist[256] = t; hist[257] = before; }
        }
        if (t == 0) {
            const uint32_t total = DESC ? scan[0] : scan[255];
            if (total < remaining) { hist[256] = DESC ? 0u : 255u; hist[257] = total; }
        }
        __syncthreads();
        const uint32_t b = hist[256], cum = hist[257];
        if (tie_out) *tie_out = hist[b];
        remaining -= cum;
        prefix |= b << shift;
        mask |= 0xffu << shift;
        __syncthreads();
    }
    *remaining_out = remaining;
    return prefix;
}

// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below. A greedy row's whole
// block returns early, together.
__device__ __forceinline__ void topk_body(
    const float* __restrict__ lg, const SP* __restrict__ sp,
    int32_t* __restrict__ ci, float* __restrict__ cv, uint32_t* __restrict__ pr,
    uint32_t n_vocab, uint32_t n_cand, uint32_t lg_ld, uint32_t ci_ld, uint32_t cv_ld,
    uint32_t pr_ld, int32_t vocab_off, uint32_t pad) {
    extern __shared__ unsigned long long shared_cand[];
    __shared__ uint32_t hist[258];
    __shared__ uint32_t scan[256], wt[256 / kWave];
    __shared__ uint32_t fill;
    __shared__ unsigned long long lds[kTopkWaves];

    const uint32_t m = blockIdx.x;
    const float* __restrict__ row = lg + (size_t)m * lg_ld;
    int32_t* __restrict__ oi = ci + (size_t)m * ci_ld;
    float*   __restrict__ ov = cv + (size_t)m * cv_ld;

    // A greedy row has no candidates: its token is sample_argmax's, and every later stage skips it.
    // Holes are written without reading the row, avoiding this kernel's whole cost. The test reads
    // only the row's params, so the whole block leaves together.
    if (sp[m].flags & SP_GREEDY) {
        uint32_t* __restrict__ gp = pr ? pr + (size_t)m * pr_ld : nullptr;
        for (uint32_t r = threadIdx.x; r < n_cand; r += kTopkThreads) {
            oi[r] = -1;
            ov[r] = -INFINITY;
            if (gp) {
                gp[2 * r]     = 0xffffffffu;
                gp[2 * r + 1] = __builtin_bit_cast(uint32_t, -INFINITY);
            }
        }
        return;
    }

    // The row's own top_k, clamped to the buffer and to the vocabulary. 0 or negative means the
    // whole candidate width (a request that set no top_k).
    uint32_t k = (uint32_t)(sp[m].top_k > 0 ? sp[m].top_k : (int32_t)n_cand);
    if (k > n_cand) k = n_cand;
    if (k > n_vocab) k = n_vocab;

    // Top-1 is an argmax and needs no select. A greedy row over a split vocabulary arrives here at
    // k = 1, not at sample_argmax (core/sample/sampler.cpp stages it so the merge can map this
    // rank's local index to a global one), and the general path would spend eight vocabulary passes
    // and a full bitonic sort on one element (896 us a step on the 27B at tp 2). The packed
    // (key, ~id) word is argmax_kernel's, so the two agree, and its tie rule is the select's.
    uint32_t got;
    if (k == 1) {
        unsigned long long best = 0;
        // Four at a time where the row and pointer allow: the scan is latency bound (see
        // kTopkThreads), so a float4 puts four times the bytes in flight per wait. A row is 16-byte
        // aligned only when base and stride both are, and n_vocab need not be a multiple of four;
        // otherwise the scalar form runs.
        const bool v4 = ((uintptr_t)row & 15u) == 0 && (n_vocab & 3u) == 0;
        if (v4) {
            const float4* __restrict__ r4 = (const float4*)row;
            const uint32_t n4 = n_vocab >> 2;
            for (uint32_t j = threadIdx.x; j < n4; j += kTopkThreads) {
                const float4 f = r4[j];
                const uint32_t i0 = j << 2;
                const float fs[4] = { f.x, f.y, f.z, f.w };
#pragma unroll
                for (uint32_t c = 0; c < 4; ++c) {
                    const unsigned long long e = ((unsigned long long)fkey(fs[c]) << 32) |
                                                 (unsigned long long)(~(i0 + c));
                    if (e > best) best = e;
                }
            }
        } else {
            for (uint32_t i = threadIdx.x; i < n_vocab; i += kTopkThreads) {
                const unsigned long long e =
                    ((unsigned long long)fkey(row[i]) << 32) | (unsigned long long)(~i);
                if (e > best) best = e;
            }
        }
        best = wave_max_u64(best);
        const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
        if (lane == 0) lds[wave] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (uint32_t w = 1; w < kTopkWaves; ++w) if (lds[w] > lds[0]) lds[0] = lds[w];
            shared_cand[0] = lds[0];
        }
        __syncthreads();
        got = 1;
    } else {
        // `n_eq` is how many of the k are still owed once every strictly larger logit is counted:
        // the number to take from the threshold's tie group, always at least one. `n_tie` is that
        // group's size.
        uint32_t n_eq = 0, dummy = 0, n_tie = 0;
        const uint32_t T = radix_select<true>(row, n_vocab, 0, k, hist, scan, wt, &n_eq, &n_tie);

        // The tie-break pass runs only when there is a tie to break. When `n_tie == n_eq` the whole
        // group is in, and the four ASC passes (over 124160 logits) would find nothing new. With
        // distinct logits, the ordinary case, `i <= 0xffffffff` holds for every member.
        const uint32_t idx_cut = (n_tie == n_eq)
                                     ? 0xffffffffu
                                     : radix_select<false>(row, n_vocab, T, n_eq, hist, scan, wt,
                                                           &dummy, &n_tie);

        if (threadIdx.x == 0) fill = 0;
        __syncthreads();
        /* Four at a time, as in radix_select. Slots are claimed in atomic order, and the sort below
         * puts the set in (key, ~id) order, so the result does not depend on that order. */
        auto collect = [&](float f, uint32_t i) {
            const uint32_t key = fkey(f);
            const bool take = key > T || (key == T && i <= idx_cut);
            if (!take) return;
            const uint32_t slot = atomicAdd(&fill, 1u);
            if (slot < kMaxCand)
                shared_cand[slot] = ((unsigned long long)key << 32) | (unsigned long long)(~i);
        };
        if (((uintptr_t)row & 15u) == 0 && (n_vocab & 3u) == 0) {
            const float4* __restrict__ r4 = (const float4*)row;
            for (uint32_t j = threadIdx.x; j < (n_vocab >> 2); j += kTopkThreads) {
                const float4 f = r4[j];
                collect(f.x, 4 * j);
                collect(f.y, 4 * j + 1);
                collect(f.z, 4 * j + 2);
                collect(f.w, 4 * j + 3);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < n_vocab; i += kTopkThreads) collect(row[i], i);
        }
        __syncthreads();
        got = fill < kMaxCand ? fill : kMaxCand;

        // Sorted over the live set, not the candidate width: the select fills exactly `got` low
        // slots and the output reads none above them, so top_k = 20 sorts 32 words, not 4096.
        uint32_t sn = 1;
        while (sn < got) sn <<= 1;
        for (uint32_t i = got + threadIdx.x; i < sn; i += kTopkThreads) shared_cand[i] = 0;
        __syncthreads();

        // Bitonic sort, descending. The packed word is (key, ~id), so the largest logit comes
        // first and, among equal logits, the lower id (ref's tie rule). Padding words are 0, sort
        // to the tail, and are written as holes.
        for (uint32_t sz = 2; sz <= sn; sz <<= 1) {
            for (uint32_t st = sz >> 1; st > 0; st >>= 1) {
                for (uint32_t t = threadIdx.x; t < sn; t += kTopkThreads) {
                    const uint32_t partner = t ^ st;
                    if (partner <= t) continue;
                    const bool up = (t & sz) == 0;                 // `up` here means DESCENDING
                    const unsigned long long a = shared_cand[t], b = shared_cand[partner];
                    if (up ? (a < b) : (a > b)) { shared_cand[t] = b; shared_cand[partner] = a; }
                }
                __syncthreads();
            }
        }
    }
    (void)pad;

    // The candidate index is local and the pair plane global. ci/cv index `row`, this rank's slice
    // of the vocabulary, as every later stage reads them. `pr` is the vocab-parallel payload:
    // (u32 global id, f32 logit) pairs in sample_merge_topk's layout with vocab_off added, so the
    // all-gather moves opaque bytes; the single-rank path passes pr = null.
    uint32_t* __restrict__ op = pr ? pr + (size_t)m * pr_ld : nullptr;
    for (uint32_t r = threadIdx.x; r < n_cand; r += kTopkThreads) {
        const bool live = r < got && r < k;
        const unsigned long long e = live ? shared_cand[r] : 0ull;
        const uint32_t loc = (uint32_t)(~(uint32_t)(e & 0xffffffffull));
        const float    val = live ? row[loc] : -INFINITY;
        oi[r] = live ? (int32_t)loc : -1;
        ov[r] = val;
        if (op) {
            // 0xffffffff is the hole: the merge reads the id as signed and drops a negative one,
            // the convention ci carries as -1.
            op[2 * r]     = live ? (uint32_t)((int32_t)loc + vocab_off) : 0xffffffffu;
            op[2 * r + 1] = __builtin_bit_cast(uint32_t, val);
        }
    }
}
__global__ __launch_bounds__(kTopkThreads) void topk_kernel(
    const float* __restrict__ lg, const SP* __restrict__ sp,
    int32_t* __restrict__ ci, float* __restrict__ cv, uint32_t* __restrict__ pr,
    uint32_t n_vocab, uint32_t n_cand, uint32_t lg_ld, uint32_t ci_ld, uint32_t cv_ld,
    uint32_t pr_ld, int32_t vocab_off, uint32_t pad) {
    topk_body(lg, sp, ci, cv, pr, n_vocab, n_cand, lg_ld, ci_ld, cv_ld, pr_ld, vocab_off, pad);
}
__global__ __launch_bounds__(kTopkThreads) void r4dchain_sample_topk(
    const float* __restrict__ lg, const SP* __restrict__ sp,
    int32_t* __restrict__ ci, float* __restrict__ cv, uint32_t* __restrict__ pr,
    uint32_t n_vocab, uint32_t n_cand, uint32_t lg_ld, uint32_t ci_ld, uint32_t cv_ld,
    uint32_t pr_ld, int32_t vocab_off, uint32_t pad) {
    r4d_chain_wait();
    topk_body(lg, sp, ci, cv, pr, n_vocab, n_cand, lg_ld, ci_ld, cv_ld, pr_ld, vocab_off, pad);
    r4d_chain_done();
}

// ---------------------------------------------------------------------------- sample_logprobs
// ref_sample_logprobs. For a row with RAD_SP_LOGPROBS: [max, sum of exp(l - max), k (id, logit)
// pairs], descending, ties to the lower id, ids global (+ vocab_off) and written as floats. A row
// without the flag is not written, and its block exits before reading anything.
//
// The k largest come from topk_kernel's select at the op's fixed k rather than a row's top_k: the
// same radix_select passes, the tie-break pass only when needed, and the same packed (key, ~id)
// bitonic sort, so this set and a top-k candidate set over the same row agree to the bit. The max
// is the first of them; the sum is one more pass with accurate expf, folded wave by wave in a fixed
// order, so the same row gives the same bits every step.
//
// k is bounded by kMaxLogprobs, the LDS the sort runs in. The engine declares 21 (OpenAI's
// top_logprobs bound plus one, so a row is whole 16-byte words); the shim refuses a wider op
// rather than truncating.
constexpr uint32_t kMaxLogprobs = 64;

__global__ __launch_bounds__(kTopkThreads) void logprobs_kernel(
    const float* __restrict__ lg, const SP* __restrict__ sp, float* __restrict__ st,
    uint32_t n_vocab, uint32_t k, uint32_t lg_ld, uint32_t st_ld, int32_t vocab_off) {
    __shared__ unsigned long long cand[kMaxLogprobs];
    __shared__ uint32_t hist[258];
    __shared__ uint32_t scan[256], wt[256 / kWave];
    __shared__ uint32_t fill;
    __shared__ float red[kTopkWaves];

    const uint32_t m = blockIdx.x;
    if (sp && !(sp[m].flags & SP_LOGPROBS)) return;   /* no params: every row */
    const float* __restrict__ row = lg + (size_t)m * lg_ld;
    float* __restrict__ out = st + (size_t)m * st_ld;
    const uint32_t kk = k < n_vocab ? k : n_vocab;

    uint32_t n_eq = 0, dummy = 0, n_tie = 0;
    const uint32_t T = radix_select<true>(row, n_vocab, 0, kk, hist, scan, wt, &n_eq, &n_tie);
    const uint32_t idx_cut = (n_tie == n_eq)
                                 ? 0xffffffffu
                                 : radix_select<false>(row, n_vocab, T, n_eq, hist, scan, wt,
                                                       &dummy, &n_tie);
    if (threadIdx.x == 0) fill = 0;
    __syncthreads();
    auto collect = [&](float f, uint32_t i) {
        const uint32_t key = fkey(f);
        if (!(key > T || (key == T && i <= idx_cut))) return;
        const uint32_t slot = atomicAdd(&fill, 1u);
        if (slot < kMaxLogprobs) cand[slot] = ((unsigned long long)key << 32) | (unsigned long long)(~i);
    };
    const bool v4 = ((uintptr_t)row & 15u) == 0 && (n_vocab & 3u) == 0;
    if (v4) {
        const float4* __restrict__ r4 = (const float4*)row;
        for (uint32_t j = threadIdx.x; j < (n_vocab >> 2); j += kTopkThreads) {
            const float4 f = r4[j];
            collect(f.x, 4 * j);
            collect(f.y, 4 * j + 1);
            collect(f.z, 4 * j + 2);
            collect(f.w, 4 * j + 3);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n_vocab; i += kTopkThreads) collect(row[i], i);
    }
    __syncthreads();
    const uint32_t got = fill < kk ? fill : kk;
    uint32_t sn = 1;
    while (sn < got) sn <<= 1;
    for (uint32_t i = got + threadIdx.x; i < sn; i += kTopkThreads) cand[i] = 0;
    __syncthreads();
    for (uint32_t sz = 2; sz <= sn; sz <<= 1) {
        for (uint32_t stp = sz >> 1; stp > 0; stp >>= 1) {
            const uint32_t t = threadIdx.x;
            if (t < sn) {
                const uint32_t partner = t ^ stp;
                if (partner > t) {
                    const bool up = (t & sz) == 0;                 // `up` here means DESCENDING
                    const unsigned long long a = cand[t], b = cand[partner];
                    if (up ? (a < b) : (a > b)) { cand[t] = b; cand[partner] = a; }
                }
            }
            __syncthreads();
        }
    }

    const float mx = got ? row[~(uint32_t)(cand[0] & 0xffffffffull)] : -INFINITY;
    float s = 0.0f;
    if (v4) {
        const float4* __restrict__ r4 = (const float4*)row;
        for (uint32_t j = threadIdx.x; j < (n_vocab >> 2); j += kTopkThreads) {
            const float4 f = r4[j];
            s += expf(f.x - mx) + expf(f.y - mx) + expf(f.z - mx) + expf(f.w - mx);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n_vocab; i += kTopkThreads) s += expf(row[i] - mx);
    }
    s = r4d_wave_sum(s);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) red[wave] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t w = 0; w < kTopkWaves; ++w) sum += red[w];
        out[0] = mx;
        out[1] = sum;
    }
    for (uint32_t r = threadIdx.x; r < k; r += kTopkThreads) {
        const bool live = r < got;
        const uint32_t loc = live ? ~(uint32_t)(cand[r] & 0xffffffffull) : 0u;
        out[2 + 2 * r] = live ? (float)((int32_t)loc + vocab_off) : -1.0f;
        out[3 + 2 * r] = live ? row[loc] : -INFINITY;
    }
}

// ---------------------------------------------------------------------------- the narrowing stages
//
// All four read a fixed-width candidate set in place: index -1 and value -infinity mark a hole, and
// the live set keeps its descending order. n_cand is a thousand or two, so one block per row with a
// couple of block reductions is the whole cost.

enum { NR_TOPP = 0, NR_MINP, NR_TYPICAL, NR_XTC };

__global__ __launch_bounds__(kThreads) void narrow_kernel(
    int32_t* __restrict__ ci, float* __restrict__ cv, const SP* __restrict__ sp,
    uint32_t n_cand, uint32_t ci_ld, uint32_t cv_ld, int which) {
    __shared__ float red[kWaves];
    __shared__ float sh_mx, sh_inv, sh_H;
    __shared__ int   sh_cut_bits, sh_have_cut, sh_last, sh_above;

    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    // A greedy row has no candidates to narrow; the whole block leaves together.
    if (p.flags & SP_GREEDY) return;
    int32_t* __restrict__ oi = ci + (size_t)m * ci_ld;
    float*   __restrict__ ov = cv + (size_t)m * cv_ld;

    float mx = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
        if (oi[i] >= 0) mx = fmaxf(mx, ov[i]);
    mx = block_max(mx, red);
    if (mx == -INFINITY) return;
    if (threadIdx.x == 0) sh_mx = mx;
    __syncthreads();

    if (which == NR_MINP) {
        // Keep everything within a factor min_p of the most likely candidate. Comparing
        // exp(v - max) against p is the same test as prob >= p * max_prob without the normaliser:
        // one pass and one rounding fewer.
        if (!(p.min_p > 0.0f)) return;
        for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
            if (oi[i] < 0) continue;
            const float v = ov[i];
            if (v == mx) continue;                       // the max always survives
            if (expf(v - mx) < p.min_p) { oi[i] = -1; ov[i] = -INFINITY; }
        }
        return;
    }

    // The other three need the normalised probabilities.
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
        if (oi[i] >= 0) sum += expf(ov[i] - mx);
    sum = block_sum(sum, red);
    if (threadIdx.x == 0) sh_inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();
    const float inv = sh_inv;

    if (which == NR_TOPP) {
        // One thread finds the cut: the shortest prefix whose cumulative probability reaches
        // top_p, accumulated in f32 in candidate order, as ref does; a parallel prefix sum would
        // round differently at the boundary.
        //
        // Only the cut is serial; the whole block removes the candidates past it. Walking them on
        // one thread costs a dependent global load each (~50 us a row at 1024 candidates), on
        // every step of a greedy request over a split vocabulary, whose top-1 row reaches any top_p
        // at its first candidate.
        if (!(p.top_p < 1.0f) || !(p.top_p > 0.0f)) return;
        __shared__ uint32_t sh_cut;
        if (threadIdx.x == 0) {
            float cum = 0.0f;
            uint32_t cut = n_cand;
            for (uint32_t i = 0; i < n_cand; ++i) {
                if (oi[i] < 0) continue;
                cum += expf(ov[i] - mx) * inv;
                if (cum >= p.top_p) { cut = i + 1; break; }   // always keep at least one
            }
            sh_cut = cut;
        }
        __syncthreads();
        for (uint32_t i = sh_cut + threadIdx.x; i < n_cand; i += kThreads)
            if (oi[i] >= 0) { oi[i] = -1; ov[i] = -INFINITY; }
        return;
    }

    if (which == NR_TYPICAL) {
        // Locally typical: keep the candidates whose surprisal is closest to the entropy until
        // their mass reaches typical_p. Not a prefix (candidates are ordered by probability, this
        // by |surprisal - H|), so it is a threshold on the distance with ties kept, which can
        // overshoot p slightly, rather than an order that depends on the candidate index.
        if (!(p.typical_p < 1.0f) || !(p.typical_p > 0.0f)) return;
        float H = 0.0f;
        for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
            if (oi[i] < 0) continue;
            const float q = expf(ov[i] - mx) * inv;
            if (q > 0.0f) H -= q * logf(q);
        }
        H = block_sum(H, red);
        if (threadIdx.x == 0) {
            sh_H = H;
            /* A non-negative float's bit pattern orders like the float, so the minimum can be an
             * integer atomicMin. */
            sh_cut_bits = __float_as_int(INFINITY);
            sh_have_cut = 0;
        }
        __syncthreads();

        // The smallest distance threshold whose mass reaches p, over the distances that occur.
        // Quadratic in n_cand, parallel over the outer index; the reduction is a minimum, so it is
        // order-independent.
        float local_cut = INFINITY;
        bool  local_have = false;
        for (uint32_t c = threadIdx.x; c < n_cand; c += kThreads) {
            if (oi[c] < 0) continue;
            const float qc = expf(ov[c] - sh_mx) * inv;
            if (!(qc > 0.0f)) continue;
            const float dc = fabsf(-logf(qc) - sh_H);
            float mass = 0.0f;
            for (uint32_t i = 0; i < n_cand; ++i) {
                if (oi[i] < 0) continue;
                const float q = expf(ov[i] - sh_mx) * inv;
                if (!(q > 0.0f)) continue;
                if (fabsf(-logf(q) - sh_H) <= dc) mass += q;
            }
            if (mass >= p.typical_p && dc < local_cut) { local_cut = dc; local_have = true; }
        }
        if (local_have) {
            atomicMin(&sh_cut_bits, __float_as_int(local_cut));
            atomicOr(&sh_have_cut, 1);
        }
        __syncthreads();
        if (!sh_have_cut) return;
        const float cut = __int_as_float(sh_cut_bits);
        for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
            if (oi[i] < 0) continue;
            const float q = expf(ov[i] - sh_mx) * inv;
            if (!(q > 0.0f) || fabsf(-logf(q) - sh_H) > cut) { oi[i] = -1; ov[i] = -INFINITY; }
        }
        return;
    }

    // XTC ("exclude top choices"): with probability xtc_probability, drop every candidate at or
    // above xtc_threshold except the least likely of them, removing the obvious continuations. The
    // draw comes from the row's own stream, so it reproduces.
    if (!(p.xtc_probability > 0.0f) || !(p.xtc_threshold > 0.0f)) return;
    if (uniform01(p.seed ^ RAD_SAMPLE_KEY_XTC, p.pos) >= p.xtc_probability) return;

    if (threadIdx.x == 0) { sh_last = -1; sh_above = 0; }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
        if (oi[i] < 0) continue;
        if (expf(ov[i] - mx) * inv >= p.xtc_threshold) {
            atomicAdd(&sh_above, 1);
            atomicMax(&sh_last, (int)i);
        }
    }
    __syncthreads();
    // Only when at least two clear the threshold: dropping the only one would leave the row
    // without its own choice.
    if (sh_above < 2) return;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
        if (oi[i] < 0 || (int)i == sh_last) continue;
        if (expf(ov[i] - mx) * inv >= p.xtc_threshold) { oi[i] = -1; ov[i] = -INFINITY; }
    }
}

// ---------------------------------------------------------------------------- sample_pick
// ref_sample_pick: the inverse CDF over the candidate set, walked in candidate order with an f32
// sum, on one thread like top_p because the walk's rounding is the contract. The fallback is the
// last candidate above -infinity: `u` is below 1 but the probabilities only sum to 1 in exact
// arithmetic, and a draw that falls through must still yield an allowed token. A live candidate at
// -infinity is a grammar-masked logit top-k took to fill its width; it adds nothing to `cum`, so
// the comparison never chooses one, and the fallback must not either.
//
// A greedy row is not touched: its token is the argmax's, written before the chain ran.
// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below. Its returns are
// a whole block's, or every thread but the walker's after the last barrier.
__device__ __forceinline__ void pick_body(
    const int32_t* __restrict__ ci, const float* __restrict__ cv, const SP* __restrict__ sp,
    int32_t* __restrict__ tok, uint32_t n_cand, uint32_t ci_ld, uint32_t cv_ld) {
    __shared__ float red[kWaves];
    __shared__ float sh_mx, sh_inv;

    const uint32_t m = blockIdx.x;
    if (sp[m].flags & SP_GREEDY) return;
    const int32_t* __restrict__ oi = ci + (size_t)m * ci_ld;
    const float*   __restrict__ ov = cv + (size_t)m * cv_ld;

    float mx = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
        if (oi[i] >= 0) mx = fmaxf(mx, ov[i]);
    mx = block_max(mx, red);
    if (mx == -INFINITY) { if (threadIdx.x == 0) tok[m] = -1; return; }

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
        if (oi[i] >= 0) sum += expf(ov[i] - mx);
    sum = block_sum(sum, red);
    if (threadIdx.x == 0) { sh_mx = mx; sh_inv = sum > 0.0f ? 1.0f / sum : 0.0f; }
    __syncthreads();
    if (threadIdx.x != 0) return;

    const float u = uniform01(sp[m].seed ^ RAD_SAMPLE_KEY_PICK, sp[m].pos);
    float cum = 0.0f;
    int32_t chosen = -1, last = -1;
    for (uint32_t i = 0; i < n_cand; ++i) {
        const int32_t id = oi[i];
        if (id < 0) continue;
        const float v = ov[i];
        if (v > -INFINITY) last = id;
        cum += expf(v - sh_mx) * sh_inv;
        if (u < cum) { chosen = id; break; }
    }
    tok[m] = chosen >= 0 ? chosen : last;
}
__global__ __launch_bounds__(kThreads) void pick_kernel(
    const int32_t* __restrict__ ci, const float* __restrict__ cv, const SP* __restrict__ sp,
    int32_t* __restrict__ tok, uint32_t n_cand, uint32_t ci_ld, uint32_t cv_ld) {
    pick_body(ci, cv, sp, tok, n_cand, ci_ld, cv_ld);
}
__global__ __launch_bounds__(kThreads) void r4dchain_sample_pick(
    const int32_t* __restrict__ ci, const float* __restrict__ cv, const SP* __restrict__ sp,
    int32_t* __restrict__ tok, uint32_t n_cand, uint32_t ci_ld, uint32_t cv_ld) {
    r4d_chain_wait();
    pick_body(ci, cv, sp, tok, n_cand, ci_ld, cv_ld);
    r4d_chain_done();
}

// ---------------------------------------------------------------------------- sample_merge_topk
// The vocab-parallel merge. Each rank runs top-k over its own lm_head shard and the candidate sets
// are all-gathered (k * world_size pairs, kilobytes, instead of tens of megabytes of logits).
// `gathered` holds (u32 global id, f32 logit) pairs; each rank added its vocab offset before the
// gather, so this needs no shard layout.
//
// One block per row, with the top-k's packed (key, ~id) word, so descending order and the lower-id
// tie rule come from the same code. n_in is world_size * n_cand, a few thousand.
// The body; the plain kernel and its chained form (r4d_chain.h) wrap it below.
__device__ __forceinline__ void merge_topk_body(
    const uint32_t* __restrict__ gt, const SP* __restrict__ sp,
    int32_t* __restrict__ ci, float* __restrict__ cv,
    uint32_t n_in, uint32_t n_cand, uint32_t gt_ld, uint32_t ci_ld, uint32_t cv_ld, uint32_t pad) {
    extern __shared__ unsigned long long shared_cand[];
    const uint32_t m = blockIdx.x;
    const uint32_t* __restrict__ g = gt + (size_t)m * gt_ld;
    int32_t* __restrict__ oi = ci + (size_t)m * ci_ld;
    float*   __restrict__ ov = cv + (size_t)m * cv_ld;

    // A greedy row has no candidates, as in topk_kernel; the whole block leaves together.
    if (sp[m].flags & SP_GREEDY) {
        for (uint32_t r = threadIdx.x; r < n_cand; r += kTopkThreads) { oi[r] = -1; ov[r] = -INFINITY; }
        return;
    }

    /* The input is already world_size sorted runs: each rank wrote its n_cand pairs in descending
     * (key, ~id) order with holes at the tail, and adding a rank's vocab offset preserves that
     * order. When the runs are a power of two long and fill the padded width, that is the bitonic
     * network's state after its stages up to sz = n_cand, provided the odd runs are loaded reversed
     * (those stages leave them ascending), so only the log2(world_size) merge stages remain: 11
     * passes over two runs of 1024 instead of 66. Any other shape starts from the first stage. The
     * keys are distinct except for the all-zero holes, so the output order is unique. */
    const bool     runs  = (n_cand & (n_cand - 1)) == 0 && pad == n_in;
    const uint32_t first = runs ? 2 * n_cand : 2;
    for (uint32_t j = threadIdx.x; j < pad; j += kTopkThreads) {
        if (j >= n_in) { shared_cand[j] = 0ull; continue; }
        const uint32_t run = runs ? j / n_cand : 0;
        const uint32_t src = (run & 1u) ? run * n_cand + (n_cand - 1 - (j - run * n_cand)) : j;
        const uint32_t id = g[2 * src];
        const float    v  = __builtin_bit_cast(float, g[2 * src + 1]);
        shared_cand[j] = (int32_t)id < 0
                             ? 0ull
                             : (((unsigned long long)fkey(v) << 32) | (unsigned long long)(~id));
    }
    __syncthreads();

    for (uint32_t sz = first; sz <= pad; sz <<= 1) {
        for (uint32_t st = sz >> 1; st > 0; st >>= 1) {
            for (uint32_t t = threadIdx.x; t < pad; t += kTopkThreads) {
                const uint32_t partner = t ^ st;
                if (partner <= t) continue;
                const bool up = (t & sz) == 0;
                const unsigned long long a = shared_cand[t], b = shared_cand[partner];
                if (up ? (a < b) : (a > b)) { shared_cand[t] = b; shared_cand[partner] = a; }
            }
            __syncthreads();
        }
    }

    // The row's own top_k, re-applied. Each rank kept k before the gather, so the merged set holds
    // up to world_size * k, and nothing downstream narrows again. Leaving them live changes the
    // distribution, and at k = 1 the answer: a greedy row is staged at top_k = 1 so the chain
    // returns the argmax, and with world_size candidates live `sample_pick` would draw among them.
    // 0 or negative means the whole candidate width, as everywhere else.
    const uint32_t keep = (uint32_t)(sp[m].top_k > 0 ? sp[m].top_k : (int32_t)n_cand);
    for (uint32_t r = threadIdx.x; r < n_cand; r += kTopkThreads) {
        const unsigned long long e = (r < pad && r < keep) ? shared_cand[r] : 0ull;
        if (e == 0ull) { oi[r] = -1; ov[r] = -INFINITY; continue; }
        const uint32_t key = (uint32_t)(e >> 32);
        oi[r] = (int32_t)(~(uint32_t)(e & 0xffffffffull));
        ov[r] = __builtin_bit_cast(float, (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
    }
}
__global__ __launch_bounds__(kTopkThreads) void merge_topk_kernel(
    const uint32_t* __restrict__ gt, const SP* __restrict__ sp,
    int32_t* __restrict__ ci, float* __restrict__ cv,
    uint32_t n_in, uint32_t n_cand, uint32_t gt_ld, uint32_t ci_ld, uint32_t cv_ld, uint32_t pad) {
    merge_topk_body(gt, sp, ci, cv, n_in, n_cand, gt_ld, ci_ld, cv_ld, pad);
}
__global__ __launch_bounds__(kTopkThreads) void r4dchain_sample_merge_topk(
    const uint32_t* __restrict__ gt, const SP* __restrict__ sp,
    int32_t* __restrict__ ci, float* __restrict__ cv,
    uint32_t n_in, uint32_t n_cand, uint32_t gt_ld, uint32_t ci_ld, uint32_t cv_ld, uint32_t pad) {
    r4d_chain_wait();
    merge_topk_body(gt, sp, ci, cv, n_in, n_cand, gt_ld, ci_ld, cv_ld, pad);
    r4d_chain_done();
}

}  // namespace

// ================================================================================ entry points
//
// Every pitch is in elements and 0 means tight. `lg` is f32 whatever the model's activation dtype:
// a bf16 logit has 8 mantissa bits, and a nucleus cut compares neighbouring probabilities.

static uint32_t host_pow2(uint32_t v) { uint32_t p = 1; while (p < v) p <<= 1; return p; }

static int r4d_sample_temp_f32(long lg, long params, int M, int n_vocab, int lg_ld,
                                   long stream, bool chain = false) {
    if (!lg || !params) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_vocab <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!lg_ld) lg_ld = n_vocab;
    const uint32_t gy = ((uint32_t)n_vocab + kThreads - 1) / kThreads;
    hipLaunchKernelGGL((chain ? r4dchain_sample_temp : temp_kernel),
                       dim3((uint32_t)M, gy > 256u ? 256u : gy), dim3(kThreads), 0,
                       (hipStream_t)stream, (float*)lg, (const SP*)params,
                       (uint32_t)n_vocab, (uint32_t)lg_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

static int r4d_sample_mask_f32(long lg, long params, long bitmask, int M, int n_vocab,
                                   int mask_rows, int lg_ld, long stream) {
    if (!lg || !params || !bitmask) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_vocab <= 0 || mask_rows <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!lg_ld) lg_ld = n_vocab;
    const uint32_t words = ((uint32_t)n_vocab + 31u) / 32u;
    const uint32_t gy = ((uint32_t)n_vocab + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(mask_kernel, dim3((uint32_t)M, gy > 256u ? 256u : gy), dim3(kThreads), 0,
                       (hipStream_t)stream, (float*)lg, (const SP*)params,
                       (const uint32_t*)bitmask, (uint32_t)n_vocab, (uint32_t)lg_ld, words,
                       (int32_t)mask_rows);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

static int r4d_sample_penalties_f32(long lg, long params, long hist, int M, int n_vocab,
                                        int hmax, int lg_ld, int vocab_off, long stream) {
    if (!lg || !params || !hist) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_vocab <= 0 || hmax <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!lg_ld) lg_ld = n_vocab;
    hipLaunchKernelGGL(penalties_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (float*)lg, (const SP*)params, (const int32_t*)hist,
                       (uint32_t)n_vocab, (uint32_t)lg_ld, (uint32_t)hmax, (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

// The widest history row DRY takes: one u16 repeat length per position in LDS, so 16384 of them is
// 32 KiB, half a workgroup's LDS. The engine declares 2048.
constexpr uint32_t kDryMaxHist = 16384;

static int r4d_sample_dry_f32(long lg, long params, long hist, long brk, int M, int n_vocab,
                                  int hmax, int lg_ld, int vocab_off, long stream) {
    if (!lg || !params || !hist) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_vocab <= 0 || hmax <= 0) return R4D_NO(RAD_E_SHAPE);
    if ((uint32_t)hmax > kDryMaxHist) return R4D_NO(RAD_E_SHAPE);
    if (!lg_ld) lg_ld = n_vocab;
    hipLaunchKernelGGL(dry_kernel, dim3((uint32_t)M), dim3(kThreads),
                       (size_t)hmax * sizeof(uint16_t), (hipStream_t)stream,
                       (float*)lg, (const SP*)params, (const int32_t*)hist, (const uint8_t*)brk,
                       (uint32_t)n_vocab, (uint32_t)lg_ld, (uint32_t)hmax, (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

static int r4d_sample_argmax_f32(long lg, long tok, int M, int n_vocab, int lg_ld,
                                     long stream) {
    if (!lg || !tok) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_vocab <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!lg_ld) lg_ld = n_vocab;
    /* The pitch is part of the alignment: row m starts at `m * lg_ld` floats, so a pitch that is
     * not a multiple of four misaligns every row after the first. */
    const bool vec = ((uintptr_t)lg % (4 * sizeof(float)) == 0) && ((uint32_t)lg_ld % 4u == 0u);
    if (vec)
        hipLaunchKernelGGL((argmax_kernel<kArgmaxThreads, true>), dim3((uint32_t)M),
                           dim3(kArgmaxThreads), 0, (hipStream_t)stream,
                           (const float*)lg, (int32_t*)tok, (uint32_t)n_vocab, (uint32_t)lg_ld);
    else
        hipLaunchKernelGGL((argmax_kernel<kArgmaxThreads, false>), dim3((uint32_t)M),
                           dim3(kArgmaxThreads), 0, (hipStream_t)stream,
                           (const float*)lg, (int32_t*)tok, (uint32_t)n_vocab, (uint32_t)lg_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

static int r4d_sample_logprobs_f32(long lg, long params, long stats, int M, int n_vocab, int k,
                                   int lg_ld, int st_ld, int vocab_off, long stream) {
    if (!lg || !stats) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_vocab <= 0 || k <= 0) return R4D_NO(RAD_E_SHAPE);
    if ((uint32_t)k > kMaxLogprobs) return R4D_NO(RAD_E_SHAPE);
    if (!lg_ld) lg_ld = n_vocab;
    if (!st_ld) st_ld = 2 + 2 * k;
    hipLaunchKernelGGL(logprobs_kernel, dim3((uint32_t)M), dim3(kTopkThreads), 0,
                       (hipStream_t)stream, (const float*)lg, (const SP*)params, (float*)stats,
                       (uint32_t)n_vocab, (uint32_t)k, (uint32_t)lg_ld, (uint32_t)st_ld,
                       (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

static int r4d_sample_topk_f32(long lg, long params, long cand_idx, long cand_val, long pairs,
                                   int M, int n_vocab, int n_cand,
                                   int lg_ld, int ci_ld, int cv_ld, int pr_ld, int vocab_off,
                                   long stream, bool chain = false) {
    if (!lg || !params || !cand_idx || !cand_val) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_vocab <= 0 || n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    if ((uint32_t)n_cand > kMaxCand) return R4D_NO(RAD_E_SHAPE);
    if (!lg_ld) lg_ld = n_vocab;
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    if (!pr_ld) pr_ld = 2 * n_cand;
    const uint32_t pad = host_pow2((uint32_t)n_cand);
    hipLaunchKernelGGL((chain ? r4dchain_sample_topk : topk_kernel), dim3((uint32_t)M),
                       dim3(kTopkThreads),
                       pad * sizeof(unsigned long long), (hipStream_t)stream,
                       (const float*)lg, (const SP*)params, (int32_t*)cand_idx, (float*)cand_val,
                       (uint32_t*)pairs,
                       (uint32_t)n_vocab, (uint32_t)n_cand, (uint32_t)lg_ld, (uint32_t)ci_ld,
                       (uint32_t)cv_ld, (uint32_t)pr_ld, (int32_t)vocab_off, pad);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

// `which` is 0 top_p, 1 min_p, 2 typical, 3 xtc: one entry point because the four differ by a
// branch after the same two reductions, four rows in the table because the selector names ops.
static int r4d_sample_narrow_f32(long cand_idx, long cand_val, long params, int M, int n_cand,
                                     int which, int ci_ld, int cv_ld, long stream) {
    if (!cand_idx || !cand_val || !params) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    if (which < 0 || which > 3) return R4D_NO(RAD_E_SHAPE);
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    hipLaunchKernelGGL(narrow_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (int32_t*)cand_idx, (float*)cand_val, (const SP*)params,
                       (uint32_t)n_cand, (uint32_t)ci_ld, (uint32_t)cv_ld, which);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

static int r4d_sample_pick_f32(long cand_idx, long cand_val, long params, long tok,
                                   int M, int n_cand, int ci_ld, int cv_ld, long stream,
                                   bool chain = false) {
    if (!cand_idx || !cand_val || !params || !tok) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    hipLaunchKernelGGL((chain ? r4dchain_sample_pick : pick_kernel), dim3((uint32_t)M),
                       dim3(kThreads), 0, (hipStream_t)stream,
                       (const int32_t*)cand_idx, (const float*)cand_val, (const SP*)params,
                       (int32_t*)tok, (uint32_t)n_cand, (uint32_t)ci_ld, (uint32_t)cv_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

static int r4d_sample_merge_topk_f32(long gathered, long params, long cand_idx, long cand_val,
                                         int M, int n_in, int n_cand,
                                         int gt_ld, int ci_ld, int cv_ld, long stream,
                                         bool chain = false) {
    if (!gathered || !params || !cand_idx || !cand_val) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || n_in <= 0 || n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    if ((uint32_t)n_in > kMaxCand) return R4D_NO(RAD_E_SHAPE);
    if (!gt_ld) gt_ld = 2 * n_in;
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    const uint32_t pad = host_pow2((uint32_t)n_in);
    hipLaunchKernelGGL((chain ? r4dchain_sample_merge_topk : merge_topk_kernel), dim3((uint32_t)M),
                       dim3(kTopkThreads),
                       pad * sizeof(unsigned long long), (hipStream_t)stream,
                       (const uint32_t*)gathered, (const SP*)params,
                       (int32_t*)cand_idx, (float*)cand_val,
                       (uint32_t)n_in, (uint32_t)n_cand, (uint32_t)gt_ld, (uint32_t)ci_ld,
                       (uint32_t)cv_ld, pad);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

// ============================================================== the plugin ABI entry points
//
// Unpacking RadArgs and deriving the pitches is the kernel's own business, so it lives beside the
// kernel rather than in a translation layer an out-of-tree author could not write. The flattened
// entry points above are private to this unit. See the file header for why the chain is eleven ops
// beside r4d_sample_chain_f32.
//
// Every scalar is per row: `top_p` as a parameter would make two requests with different top_p two
// resolutions of one op, so a deployment could have only one (see rad_sample.h). The ops take `M`
// and `n_vocab` / `n_cand` as parameters and everything else as `params` data.

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

static inline int dt_f32(const RadTensor* t) {
    return (t && t->dtype == RAD_F32) ? RAD_OK : RAD_E_DTYPE;
}
static inline int dt_i32(const RadTensor* t) {
    return (t && (t->dtype == RAD_I32 || t->dtype == RAD_U32)) ? RAD_OK : RAD_E_DTYPE;
}

/* `params` is a byte blob viewed through a 32-bit dtype: the builder can only say "this many words
 * a row", so the extent is checked in words and the dtype must be 32-bit (rad_dtype_bytes is
 * core's, and a kernel plugin links only the ABI). */
static int params_ok(const RadTensor* t, long long M) {
    if (!t || !t->data) return R4D_NO(RAD_E_INVAL);
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(t)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(t) * 4 < M * (long long)sizeof(RadSampleParams)) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

/* Every [rows, n] operand is read with r4d_rows_pitch, which takes the row count from the operand:
 * `M` is ranged, so the arg holds the band's upper bound and a loop bounded by it would run off the
 * arena. */

/* ------------------------------------------------------------------ the vocabulary-wide stages */

enum { V_LOGITS = 0, V_PARAMS, V_AUX };

/* logits [M, n_vocab] f32, and the M that matters. */
static int logits_of(const RadArgs* a, const RadTensor** lg, long long* M, long long* n_vocab,
                     long long* ld) {
    *lg = r4d_opt(a, V_LOGITS);
    if (!*lg) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_f32(*lg));
    *n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(*lg, (int)(*lg)->rank - 1));
    if (*n_vocab <= 0) return R4D_NO(RAD_E_SHAPE);
    *ld = r4d_rows_pitch(*lg, *n_vocab, M);
    if (*ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (*M <= 0) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

extern "C" int r4d_rad_sample_temp(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* pr = r4d_opt(a, V_PARAMS);
    R4D_NEED(params_ok(pr, M));
    /* The chained form at decode heights, where the row resolved with `chain`. */
    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    return r4d_sample_temp_f32(r4d_addr(lg), r4d_addr(pr), (int)M, (int)nv, (int)ld,
                               (long)(uintptr_t)s, ch);
}

extern "C" int r4d_rad_sample_mask(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* pr = r4d_opt(a, V_PARAMS);
    const RadTensor* bm = r4d_opt(a, V_AUX);
    R4D_NEED(params_ok(pr, M));
    if (!bm) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(bm));
    if (!r4d_contig(bm)) return R4D_NO(RAD_E_STRIDE);
    /* The plane is [rows, ceil(n_vocab/32)] and shorter than M: constrained and unconstrained
     * sequences share it, and `mask_row` is -1 for the latter. */
    const long long words = (nv + 31) / 32;
    const long long rows = r4d_numel(bm) / (words > 0 ? words : 1);
    if (rows <= 0) return R4D_NO(RAD_E_SHAPE);
    return r4d_sample_mask_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(bm), (int)M, (int)nv,
                               (int)rows, (int)ld, (long)(uintptr_t)s);
}

/* `history` is [M, max_hist] int32 of global token ids, row m's window addressed by its hist_off /
 * hist_len. `vocab_off` is this rank's first vocabulary row (0 for a whole vocabulary). DRY's
 * optional fourth operand, `breakers`, is a u8 plane of the same shape, nonzero where the token is
 * a sequence breaker. */
static int penal(const RadArgs* a, int dry, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* pr = r4d_opt(a, V_PARAMS);
    const RadTensor* h  = r4d_opt(a, V_AUX);
    R4D_NEED(params_ok(pr, M));
    if (!h) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(h));
    if (!r4d_contig(h)) return R4D_NO(RAD_E_STRIDE);
    const long long hmax = r4d_numel(h) / M;
    if (hmax <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long voff = r4d_pgeti_or(a->p, a->n_p, "vocab_off", 0);
    if (dry) {
        const RadTensor* b = r4d_opt(a, V_AUX + 1);
        if (b) {
            if (b->dtype != RAD_U8 && b->dtype != RAD_I8 && b->dtype != RAD_BOOL)
                return R4D_NO(RAD_E_DTYPE);
            if (!r4d_contig(b)) return R4D_NO(RAD_E_STRIDE);
            if (r4d_numel(b) < M * hmax) return R4D_NO(RAD_E_SHAPE);
        }
        if (hmax > (long long)kDryMaxHist) return R4D_NO(RAD_E_SHAPE);
        return r4d_sample_dry_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(h),
                                  b ? r4d_addr(b) : 0, (int)M, (int)nv, (int)hmax, (int)ld,
                                  (int)voff, (long)(uintptr_t)s);
    }
    return r4d_sample_penalties_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(h), (int)M, (int)nv,
                                    (int)hmax, (int)ld, (int)voff, (long)(uintptr_t)s);
}

extern "C" int r4d_rad_sample_penalties(const RadArgs* a, RadStream s) {
    return penal(a, 0, s);
}
extern "C" int r4d_rad_sample_dry(const RadArgs* a, RadStream s) { return penal(a, 1, s); }

extern "C" int r4d_rad_sample_argmax(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* tok = r4d_opt(a, 2);
    if (!tok) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(tok));
    if (!r4d_contig(tok) || r4d_numel(tok) < M) return R4D_NO(RAD_E_SHAPE);
    return r4d_sample_argmax_f32(r4d_addr(lg), r4d_addr(tok), (int)M, (int)nv, (int)ld,
                                 (long)(uintptr_t)s);
}

/* stats [M, 2 + 2k] f32: max, sum of exp(l - max), then k (global id, logit) pairs. */
extern "C" int r4d_rad_sample_logprobs(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    /* Absent params: every row (a drafter's candidate logprobs). */
    const RadTensor* pr = r4d_opt(a, V_PARAMS);
    if (pr) R4D_NEED(params_ok(pr, M));
    const RadTensor* st = r4d_opt(a, 2);
    if (!st) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_f32(st));
    const long long k = r4d_pgeti_or(a->p, a->n_p, "k", 0);
    if (k <= 0 || k > (long long)kMaxLogprobs) return R4D_NO(RAD_E_SHAPE);
    long long Ms = 0;
    const long long st_ld = r4d_rows_pitch(st, 2 + 2 * k, &Ms);
    if (st_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Ms < M) return R4D_NO(RAD_E_SHAPE);
    const long long voff = r4d_pgeti_or(a->p, a->n_p, "vocab_off", 0);
    return r4d_sample_logprobs_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(st), (int)M, (int)nv,
                                   (int)k, (int)ld, (int)st_ld, (int)voff,
                                   (long)(uintptr_t)s);
}

/* ------------------------------------------------------------------ top-k */

enum { TK_LOGITS = 0, TK_PARAMS, TK_CI, TK_CV, TK_PAIRS };

extern "C" int r4d_rad_sample_topk(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* pr = r4d_opt(a, TK_PARAMS);
    const RadTensor* ci = r4d_opt(a, TK_CI);
    const RadTensor* cv = r4d_opt(a, TK_CV);
    R4D_NEED(params_ok(pr, M));
    if (!ci || !cv) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(ci));
    R4D_NEED(dt_f32(cv));

    /* The candidate width is the buffer's, not the request's: the deployment bound the arena was
     * sized for, read by every later stage. A narrower top_k leaves holes. */
    const long long n_cand = ci->rank >= 2 ? ci->shape[ci->rank - 1] : r4d_numel(ci) / M;
    if (n_cand <= 0 || n_cand > (int)kMaxCand) return R4D_NO(RAD_E_SHAPE);
    long long Mi = 0, Mv = 0;
    const long long ci_ld = r4d_rows_pitch(ci, n_cand, &Mi);
    const long long cv_ld = r4d_rows_pitch(cv, n_cand, &Mv);
    if (ci_ld < 0 || cv_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mi < M || Mv < M) return R4D_NO(RAD_E_SHAPE);

    /* The optional vocab-parallel pair plane, present only when lm_head is row-sharded: this rank's
     * candidates as (u32 global id, f32 logit) pairs, what the all-gather moves and
     * sample_merge_topk reads. The kernel adds `vocab_off`, so the gather needs no shard layout. */
    const RadTensor* pp = r4d_opt(a, TK_PAIRS);
    long long pr_ld = 0, Mp = 0;
    if (pp) {
        R4D_NEED(dt_i32(pp));
        pr_ld = r4d_rows_pitch(pp, 2 * n_cand, &Mp);
        if (pr_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (Mp < M) return R4D_NO(RAD_E_SHAPE);
    }
    const long long voff = r4d_pgeti_or(a->p, a->n_p, "vocab_off", 0);

    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    return r4d_sample_topk_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(ci), r4d_addr(cv),
                               pp ? r4d_addr(pp) : 0,
                               (int)M, (int)nv, (int)n_cand, (int)ld, (int)ci_ld, (int)cv_ld,
                               (int)pr_ld, (int)voff, (long)(uintptr_t)s, ch);
}

/* ------------------------------------------------------------------ the narrowing stages */

enum { NR_CI = 0, NR_CV, NR_PARAMS };

static int narrow(const RadArgs* a, int which, RadStream s) {
    const RadTensor* ci = r4d_opt(a, NR_CI);
    const RadTensor* cv = r4d_opt(a, NR_CV);
    const RadTensor* pr = r4d_opt(a, NR_PARAMS);
    if (!ci || !cv) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(ci));
    R4D_NEED(dt_f32(cv));

    const long long n_cand = r4d_geti_or(a, "n_cand", r4d_dim(ci, (int)ci->rank - 1));
    if (n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mv = 0;
    const long long ci_ld = r4d_rows_pitch(ci, n_cand, &M);
    const long long cv_ld = r4d_rows_pitch(cv, n_cand, &Mv);
    if (ci_ld < 0 || cv_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mv < M) return R4D_NO(RAD_E_SHAPE);
    R4D_NEED(params_ok(pr, M));

    return r4d_sample_narrow_f32(r4d_addr(ci), r4d_addr(cv), r4d_addr(pr), (int)M,
                                 (int)n_cand, which, (int)ci_ld, (int)cv_ld,
                                 (long)(uintptr_t)s);
}

extern "C" int r4d_rad_sample_topp(const RadArgs* a, RadStream s)    { return narrow(a, 0, s); }
extern "C" int r4d_rad_sample_minp(const RadArgs* a, RadStream s)    { return narrow(a, 1, s); }
extern "C" int r4d_rad_sample_typical(const RadArgs* a, RadStream s) { return narrow(a, 2, s); }
extern "C" int r4d_rad_sample_xtc(const RadArgs* a, RadStream s)     { return narrow(a, 3, s); }

/* ------------------------------------------------------------------ pick */

enum { PK_CI = 0, PK_CV, PK_PARAMS, PK_TOK };

extern "C" int r4d_rad_sample_pick(const RadArgs* a, RadStream s) {
    const RadTensor* ci  = r4d_opt(a, PK_CI);
    const RadTensor* cv  = r4d_opt(a, PK_CV);
    const RadTensor* pr  = r4d_opt(a, PK_PARAMS);
    const RadTensor* tok = r4d_opt(a, PK_TOK);
    if (!ci || !cv || !tok) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(ci));
    R4D_NEED(dt_f32(cv));
    R4D_NEED(dt_i32(tok));
    if (!r4d_contig(tok)) return R4D_NO(RAD_E_STRIDE);

    const long long n_cand = r4d_geti_or(a, "n_cand", r4d_dim(ci, (int)ci->rank - 1));
    if (n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mv = 0;
    const long long ci_ld = r4d_rows_pitch(ci, n_cand, &M);
    const long long cv_ld = r4d_rows_pitch(cv, n_cand, &Mv);
    if (ci_ld < 0 || cv_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mv < M || r4d_numel(tok) < M) return R4D_NO(RAD_E_SHAPE);
    R4D_NEED(params_ok(pr, M));

    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    return r4d_sample_pick_f32(r4d_addr(ci), r4d_addr(cv), r4d_addr(pr), r4d_addr(tok),
                               (int)M, (int)n_cand, (int)ci_ld, (int)cv_ld,
                               (long)(uintptr_t)s, ch);
}

/* ------------------------------------------------------------------ merge_topk */

enum { MG_GATHERED = 0, MG_PARAMS, MG_CI, MG_CV };

extern "C" int r4d_rad_sample_merge_topk(const RadArgs* a, RadStream s) {
    const RadTensor* gt = r4d_opt(a, MG_GATHERED);
    const RadTensor* pr = r4d_opt(a, MG_PARAMS);
    const RadTensor* ci = r4d_opt(a, MG_CI);
    const RadTensor* cv = r4d_opt(a, MG_CV);
    /* `params` carries the row's top_k, which the merge re-applies (see merge_topk_kernel). */
    if (!gt || !pr || !ci || !cv) return R4D_NO(RAD_E_INVAL);
    /* Each gathered entry is a (u32 id, f32 logit) pair in one buffer, so the dtype is a word type
     * and the logit is read by bit pattern. */
    R4D_NEED(dt_i32(gt));
    R4D_NEED(dt_i32(ci));
    R4D_NEED(dt_f32(cv));
    if (!r4d_contig(gt)) return R4D_NO(RAD_E_STRIDE);

    const long long n_cand = r4d_geti_or(a, "n_cand", r4d_dim(ci, (int)ci->rank - 1));
    if (n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mv = 0;
    const long long ci_ld = r4d_rows_pitch(ci, n_cand, &M);
    const long long cv_ld = r4d_rows_pitch(cv, n_cand, &Mv);
    if (ci_ld < 0 || cv_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mv < M) return R4D_NO(RAD_E_SHAPE);

    const long long per_row = r4d_numel(gt) / M;
    const long long n_in = per_row / 2;
    if (n_in <= 0 || n_in > (int)kMaxCand) return R4D_NO(RAD_E_SHAPE);

    const bool ch = r4d_geti_or(a, "chain", 0) != 0 && M <= 64;
    return r4d_sample_merge_topk_f32(r4d_addr(gt), r4d_addr(pr), r4d_addr(ci), r4d_addr(cv),
                                     (int)M, (int)n_in, (int)n_cand, (int)per_row, (int)ci_ld,
                                     (int)cv_ld, (long)(uintptr_t)s, ch);
}
