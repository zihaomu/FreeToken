// The QSA indexer's selection: the `topk` highest-scoring blocks of a query's key history, as a
// page-id table the paged attention reads directly, so attention reads at most `topk + 1` pages
// at any context length (docs/QSA.md).
//
// The output is page ids: the KV group is paged at the compress ratio, so a selected block is a
// page and the table is the attention's block table as is.
//
// The partial page is appended last and is not a candidate: the reference always attends the
// `(pos+1) % ratio` tokens after the last complete block (the query is among them). Every complete
// page is entirely in the query's past, so placing the partial page last lets the attention's
// own `kk <= q` bound do the remaining causal masking. The caller passes the effective query
// position `(nsel - 1) * ratio + pos % ratio`.
//
// Selection is a four-digit radix select, not a sort: attention is a softmax over a set, so table
// order within the set is irrelevant. The set is exact and deterministic: ties break to the lower
// block index by prefix count, never by atomic completion order.
//
// Returns a RAD status rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_emit.h"
#include "r4d_launch.h"   // R4D_DYN_LDS
#include "r4d_common.h"   // the DPP wave scans

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kThreads = 256;
constexpr int kWave    = 32;
constexpr int kWaves   = kThreads / kWave;
constexpr int kBins    = 256;
/* One thread per bin for the parallel threshold scan. */
static_assert(kThreads == kBins, "the bin scan gives one bin to one thread");

/* Key cache, gated on the launch width.
 *
 * Five passes read every key (four radix digits and the emit). Staging the keys in LDS once makes
 * four of them LDS reads, ~25% faster at decode (n = 5277, topk = 512).
 *
 * At prefill it costs occupancy: 32 KiB is half the per-workgroup LDS budget, and at M=2048 the
 * cache is ~7% slower. The crossover is 96-128 workgroups on a 64-CU GPU, so the gate is M <= 64.
 * Decode M is n_seq and prefill M is the chunk size; mixed steps are wide and take the uncached
 * path. */
constexpr int kLdsCap    = 8192;   /* keys, so 32 KiB of dynamic LDS */
constexpr int kCacheMaxM = 64;     /* launch width above which the cache costs more than it saves */

/* Cache size clamped to the device's per-workgroup LDS minus the static shared memory; deeper
 * queries take the global path. */
static inline int lds_cap_of() {
    static const int v = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return 0;
        hipDeviceProp_t pr{};
        if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return 0;
        const size_t stat = 4096;                      /* hist + suf + wsum + the scalars */
        if (pr.sharedMemPerBlock <= stat) return 0;
        const size_t room = (pr.sharedMemPerBlock - stat) / sizeof(uint32_t);
        return (int)(room < (size_t)kLdsCap ? room : (size_t)kLdsCap);
    }();
    return v;
}

/* The order key, by the op's `signed` parameter: the unsigned order (Signed = false, the
 * default) or the signed one. Key 0 means "no block" in both. */
template <bool Signed> __device__ __forceinline__ uint32_t keyof(float v, int page);

/* Order key. Scores are >= 0 or -inf, and non-negative float bit patterns order as unsigned.
 * `+1` keeps a live score of exactly zero above key 0, which means "no block". The clamp is a
 * select, not fmaxf: fmaxf(-0.0f, 0.0f) may return -0.0f, whose sign bit would rank it above
 * every positive score. */
template <> __device__ __forceinline__ uint32_t keyof<false>(float v, int page) {
    if (page < 0 || !(v > -1e30f)) return 0u;
    return __float_as_uint(v > 0.0f ? v : 0.0f) + 1u;
}

/* Signed order key, for indexers whose scores can be negative (GLM's head gates): the float's
 * total order over all 32 bits. Absent pages and nonfinite scores are key 0, which no finite
 * score maps to (only NaN bit patterns would); both signed zeros map to one key, so they tie. */
template <> __device__ __forceinline__ uint32_t keyof<true>(float v, int page) {
    if (page < 0 || !__builtin_isfinite(v)) return 0u;
    const uint32_t bits = v == 0 ? 0u : __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

/* Scans run in registers on DPP (five steps per wave, r4d_common.h); only wave totals go through
 * LDS, instead of step-doubling LDS rounds with barriers that wait on up to 32 waves.
 *
 * suf[b] = sum of hist[i] over i >= b for all kBins bins: a per-wave suffix scan in registers,
 * plus the totals of the waves above. Two barriers. The caller has a barrier between the
 * histogram and this; `wt` is kBins / 32 free words. */
template <int NT>
__device__ __forceinline__ void bin_suffix(const uint32_t* hist, uint32_t* suf, uint32_t* wt) {
    static_assert(NT >= kBins && kBins % kWave == 0, "the bins are whole waves of the workgroup");
    const int t = (int)threadIdx.x, lane = t & (kWave - 1), wave = t >> 5;
    uint32_t x = 0;
    if (t < kBins) {
        x = r4d_wave_suffix_u32(hist[t], lane);
        if (lane == 0) wt[wave] = x;
    }
    __syncthreads();
    if (t < kBins) {
        uint32_t above = 0;
        for (int w = wave + 1; w < kBins / kWave; ++w) above += wt[w];
        suf[t] = x + above;
    }
    __syncthreads();
}

/* Sum of the wave counts before this wave, and the total, from `wsum` (published before a
 * barrier); each wave scans the NW words in its own lanes. */
template <int NW>
__device__ __forceinline__ void counts_before(const uint32_t* wsum, int lane, int wave,
                                              uint32_t* before, uint32_t* total) {
    static_assert(NW <= kWave, "a wave count a lane");
    const uint32_t c   = lane < NW ? wsum[lane] : 0u;
    const uint32_t inc = r4d_wave_prefix_u32(c, lane);
    const int      w   = __builtin_amdgcn_readfirstlane(wave);
    *before = (uint32_t)__builtin_amdgcn_readlane((int)(inc - c), w);
    *total  = (uint32_t)__builtin_amdgcn_readlane((int)inc, NW - 1);
}

/* `m` is a query row, not a sequence (see r4d_qsa_score_bf16.hip); `cu` maps it to its sequence's
 * block table and complete-block count. The per-query bound is (pos + 1) / ratio: block b is
 * entirely in the query's past when b * ratio + ratio - 1 <= pos. At decode this equals the
 * sequence's complete count. */
#define QSA_SELECT_PARAMS                                                                       \
    const float* __restrict__ score, const int32_t* __restrict__ bt,                            \
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,                            \
    const int32_t* __restrict__ cu, int32_t* __restrict__ sel, int32_t* __restrict__ nsel,      \
    int32_t* __restrict__ seqused, int nb, int topk, int ratio, int nseq, int s_ld,             \
    int t_ld, int o_ld, int cap
template <bool Signed>
__device__ __forceinline__ void qsa_select_body(QSA_SELECT_PARAMS) {

    __shared__ uint32_t hist[kBins];
    __shared__ uint32_t suf[kBins];
    __shared__ uint32_t wsum[kWaves];
    __shared__ uint32_t sh_T, sh_need, sh_out;
    __shared__ int      sh_seq;
    R4D_DYN_LDS(uint32_t, sh_key);         /* `cap` keys, or nothing when the gate said no */

    const int m = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & (kWave - 1), wave = t >> 5;

    if (t == 0) sh_seq = 0;
    __syncthreads();
    for (int i = t; i < nseq; i += kThreads)
        if (m >= cu[i] && m < cu[i + 1]) sh_seq = i;
    __syncthreads();
    const int seq = sh_seq;
    const int p   = pos ? pos[m] : 0;
    const int own = (p + 1) / ratio;            /* blocks entirely in THIS query's past */
    int n = nc[seq] < nb ? nc[seq] : nb;        /* complete blocks; the rest is at most a partial */
    if (own < n) n = own;

    const float*   sc = score + (size_t)m * s_ld;
    const int32_t* tb = bt    + (size_t)seq * t_ld;
    int32_t*       ou = sel   + (size_t)m * o_ld;

    /* ---- the key cache: one global pass instead of four. */
    const bool cached = n <= cap;
    if (cached) {
        for (int b0 = 0; b0 < n; b0 += kThreads) {
            const int b = b0 + t;
            if (b < n) sh_key[b] = keyof<Signed>(sc[b], tb[b]);
        }
        __syncthreads();
    }

    /* ---- the threshold: T is the exact key of the topk-th largest and `need` how many entries
     * equal to T to take; together they define the set without ordering it. */
    uint32_t T = 0, need = 0;
    if (n > topk) {
        uint32_t fixed = 0, want = (uint32_t)topk;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = t; i < kBins; i += kThreads) hist[i] = 0;
            __syncthreads();
            const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
            for (int b = t; b < n; b += kThreads) {
                const uint32_t k = cached ? sh_key[b] : keyof<Signed>(sc[b], tb[b]);
                if (k != 0u && (k & mask) == (fixed & mask))
                    atomicAdd(&hist[(k >> shift) & 255u], 1u);
            }
            __syncthreads();
            /* ---- the digit: the largest bin whose suffix sum reaches `want`, found by a parallel
             * suffix scan instead of a serial walk from bin 255 (real scores occupy a few adjacent
             * exponents, so the walk mostly crosses empty bins). About a third faster at decode,
             * and faster at prefill too.
             *
             * suf is non-increasing, so suf[t] >= want && suf[t+1] < want holds for exactly one
             * bin. The t == 0 preset covers fewer live keys than `want` (enough padding). */
            bin_suffix<kThreads>(hist, suf, wsum);
            if (t == 0) { sh_T = fixed; sh_need = want - suf[0]; }
            __syncthreads();
            {
                const uint32_t after = (t + 1 < kBins) ? suf[t + 1] : 0u;
                if (suf[t] >= want && after < want) {
                    sh_T    = fixed | ((uint32_t)t << shift);
                    sh_need = want - after;
                }
            }
            __syncthreads();
            /* topk or fewer live keys: every one is taken, in block order, as libref takes them,
             * and T = 0 says so. Only the first digit counts every live key (a later one counts
             * one bin, which holds at least `want`, often exactly), and no later digit may run:
             * it would narrow into bin 0, where a score of exactly zero lives, and turn those
             * keys into a threshold to tie on. */
            const bool every = shift == 24 && suf[0] <= want;
            fixed = every ? 0u : sh_T;
            want  = sh_need;
            if (every) break;
        }
        T = fixed; need = want;
    }
    __syncthreads();
    if (t == 0) sh_out = 0;
    __syncthreads();

    /* ---- the emit. Entries strictly above the threshold are always in; entries equal to it are
     * in while slots remain, lowest block index first. Slots come from a workgroup prefix count,
     * so a block's slot depends only on how many qualifying blocks precede it. */
    uint32_t taken_eq = 0;
    for (int b0 = 0; b0 < n; b0 += kThreads) {
        const int b = b0 + t;
        uint32_t k = 0;
        if (b < n) k = cached ? sh_key[b] : keyof<Signed>(sc[b], tb[b]);
        const bool hi = k != 0u && (n <= topk || k > T);
        const bool eq = k != 0u && n > topk && k == T;

        /* High entries take slots from the running output count, ties from the count of ties
         * already taken. */
        const unsigned long long mh = __ballot(hi), me = __ballot(eq);
        const uint32_t rh = (uint32_t)__popcll(mh & ((1ull << lane) - 1ull));
        const uint32_t re = (uint32_t)__popcll(me & ((1ull << lane) - 1ull));
        if (lane == 0) wsum[wave] = (uint32_t)__popcll(mh);
        __syncthreads();
        uint32_t before_h, tot_h;
        counts_before<kWaves>(wsum, lane, wave, &before_h, &tot_h);
        const uint32_t base_h = sh_out + before_h;
        __syncthreads();
        if (lane == 0) wsum[wave] = (uint32_t)__popcll(me);
        __syncthreads();
        uint32_t before_e, tot_e;
        counts_before<kWaves>(wsum, lane, wave, &before_e, &tot_e);
        const uint32_t base_e = taken_eq + before_e;

        if (hi) {
            const uint32_t slot = base_h + rh;
            if (slot < (uint32_t)topk) ou[slot] = tb[b];
        }
        if (eq) {
            const uint32_t rank = base_e + re;
            if (rank < need) {
                const uint32_t slot = (uint32_t)topk - need + rank;
                ou[slot] = tb[b];
            }
        }
        __syncthreads();
        if (t == 0) sh_out += tot_h;
        taken_eq += tot_e;
        __syncthreads();
    }

    /* ---- the partial page, appended last. bt[n] is the page the query lives in, present exactly
     * when the query is not on the last token of a block.
     *
     * The `(p + 1) % ratio` test keeps prefill causal: mid-chunk, a query on a block boundary may
     * find bt[n] already holding later tokens of the same chunk (a live page entirely in its
     * future). At decode bt[nc] is -1 there, but the test must use the position, not whether a
     * page is allocated. */
    if (t == 0) {
        /* Above topk blocks the table is full (ties fill it), unless topk or fewer are live:
         * then T is 0, every live key was emitted, and the slots past them hold nothing. */
        uint32_t out = (n <= topk || T == 0u) ? sh_out : (uint32_t)topk;
        const int32_t part = (n < nb && (p + 1) % ratio != 0) ? tb[n] : -1;
        if (part >= 0) ou[out++] = part;
        for (uint32_t i = out; i < (uint32_t)(topk + 1); ++i) ou[i] = -1;
        nsel[m] = (int32_t)out;
        /* The attention's causal bound. attn_paged takes the block table and seqused_k as
         * ordinary operands, so QSA issues the same op with `sel` as the table and this as the
         * length; no new attention kernel is needed.
         *
         * The attention walks `out` pages of `ratio`, with the query at flat index
         * (out - 1) * ratio + pos % ratio (partial page last). Without a partial page
         * pos % ratio = ratio - 1, so this is out * ratio: every complete page fully visible. */
        const int32_t su = (int32_t)((out == 0) ? 0 : (out - 1) * (uint32_t)ratio +
                                                      (uint32_t)(p % ratio) + 1u);
        if (seqused) seqused[m] = su;
    }
}

/* The kernel around the body, whose blocks and threads return where their work ends: CH `r4dchain`
 * is the chained form (r4d_chain.h). */
template <bool Signed, class CH>
__global__ __launch_bounds__(kThreads)
void qsa_select_kernel(QSA_SELECT_PARAMS) {
    if (CH::on) r4d_chain_wait();
    qsa_select_body<Signed>(score, bt, nc, pos, cu, sel, nsel, seqused, nb, topk, ratio, nseq, s_ld,
                            t_ld, o_ld, cap);
    if (CH::on) r4d_chain_done();
}
#undef QSA_SELECT_PARAMS

/* ===================================================== the wide-launch path, for a prefill chunk
 *
 * Without the key cache the kernel above reads every key from global memory five times, and its
 * first 8-bit digit is the score's exponent: real scores share a few exponents, so that pass
 * funnels nearly every key into a few LDS bins and settles little (1.43 ms per call at 80K
 * context, growing with context).
 *
 * The wide path takes the top twelve bits first (exponent plus four mantissa bits, 4096 bins),
 * spreading the atomics and isolating one bin holding the threshold. A second pass copies that
 * bin's keys into LDS (over first-digit bins no longer needed), the remaining digits run there,
 * and a third pass emits; when the bin also fits with its block indices (kMergeCand), the second
 * pass is the last. A bin too large for LDS runs those digits over global memory.
 *
 * The table is bit-identical to the kernel above: the threshold (topk-th largest key and the tie
 * count) does not depend on the digit split, and the emit is the same. r4d_selftest compares the
 * two byte for byte. */
constexpr int kWideBits    = 12;                     /* the first digit */
/* The first digit's shift, by key order. */
template <bool Signed> struct WideKey;
template <> struct WideKey<false> {
    static constexpr int kWideShift   = 31 - kWideBits;         /* a key's bit 31 is clear: see keyof */
};
template <> struct WideKey<true> {
    static constexpr int kWideShift   = 32 - kWideBits;         /* the signed key uses all 32 bits */
};
constexpr int kWideBins    = 1 << kWideBits;
constexpr int kWideCand    = kWideBins - kBins;      /* keys the second pass may collect */
constexpr int kWidePer     = kWideBins / kThreads;   /* first-digit bins a thread scans */
static_assert(kWideBins % kThreads == 0 && kWideCand > 0, "the first digit's bins tile the block");

/* Merged form: when the threshold's bin fits with its block indices, the second pass compacts,
 * in block order, every block above that bin (all taken; fewer than topk) and every key in it with
 * its block. Once the threshold is fixed, the table is a merge of the two lists by block index,
 * with no third pass (1179 vs 1800 us per call at 150K context, 2048-query chunk). Same slots as
 * the three-pass form; the lists reuse the first digit's bins, so LDS use is unchanged. */
constexpr int kMergeAbove  = 512;                    /* keys above the threshold's bin */
constexpr int kMergeCand   = (kWideCand - kMergeAbove) / 2;   /* keys, and blocks, in it */
static_assert(kMergeCand > 0, "room for the bin's keys and their blocks");

__device__ __forceinline__ uint32_t lower_bound_u32(const uint32_t* v, uint32_t n, uint32_t x) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (v[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* Sampled first pass for long histories (37.5K keys at 150K context: 0.67 ms vs 1.2 ms per
 * layer). kSampRuns runs of kSampRun consecutive keys bound the threshold from below; one pass
 * collects every key at or above the bound with its block, in block order, and the exact select
 * and emit run on that list in LDS. Exact, not estimated: if at least topk keys clear the bound,
 * every taken key (ties included) is in the list and lands in the same slot. If fewer clear it, or
 * the list overflows, the kernel falls back to the two-pass form.
 *
 * Runs, not a stride, since a strided sample touches every cache line. The bound is the sample's
 * (2 topk S / n + 8)-th largest, about twice the expected count (~1000 entries of kSampCap). */
constexpr int kSampRun   = 32;
constexpr int kSampRuns  = 64;
constexpr int kSamp      = kSampRun * kSampRuns;             /* 2048 keys */
/* Below this the two-pass form wins (2048-query chunk: 174 vs 196 us at 5000 blocks; the sample
 * wins 221 vs 243 at 7500). */
constexpr int kSampMinN  = 3 * kSamp;
constexpr int kSampCap   = (kWideBins - kBins) / 2;          /* list entries: a key and a block */
static_assert(kSamp + kBins <= kWideBins, "the sample and its histogram share the bins");

/* The `want`-th largest of `cnt` LDS keys and the tie count, by the single-workgroup kernel's
 * four 8-bit digits. `want` or fewer live keys leave the threshold at 0. */
__device__ __forceinline__ void lds_kth(const uint32_t* keys, int cnt, uint32_t want,
                                        uint32_t* hist, uint32_t* suf, uint32_t* wsum,
                                        uint32_t* sh_T, uint32_t* sh_need, uint32_t* T,
                                        uint32_t* need) {
    const int t = (int)threadIdx.x;
    uint32_t fixed = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < kBins; i += kThreads) hist[i] = 0;
        __syncthreads();
        const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int i = t; i < cnt; i += kThreads) {
            const uint32_t k = keys[i];
            if (k != 0u && (k & mask) == (fixed & mask)) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        bin_suffix<kThreads>(hist, suf, wsum);
        if (t == 0) { *sh_T = fixed; *sh_need = want - suf[0]; }
        __syncthreads();
        {
            const uint32_t after = (t + 1 < kBins) ? suf[t + 1] : 0u;
            if (suf[t] >= want && after < want) {
                *sh_T    = fixed | ((uint32_t)t << shift);
                *sh_need = want - after;
            }
        }
        __syncthreads();
        /* `want` or fewer live keys stop at T = 0, as in the single-workgroup kernel. */
        const bool every = shift == 24 && suf[0] <= want;
        fixed = every ? 0u : *sh_T;
        want  = *sh_need;
        if (every) break;
    }
    *T = fixed;
    *need = want;
}

template <bool Signed>
__global__ __launch_bounds__(kThreads) void qsa_select_wide_kernel(
    const float* __restrict__ score, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu,
    int32_t* __restrict__ sel, int32_t* __restrict__ nsel, int32_t* __restrict__ seqused,
    int nb, int topk, int ratio, int nseq, int s_ld, int t_ld, int o_ld) {

    /* The first digit's 4096 bins; afterwards [0, kBins) is the later digits' histogram and the
     * rest holds the collected keys. */
    __shared__ uint32_t wide[kWideBins];
    __shared__ uint32_t suf[kBins];
    __shared__ uint32_t wsum[kWaves], wsum2[kWaves];
    __shared__ uint32_t sh_T, sh_need, sh_out, sh_cnt, sh_bin;
    __shared__ int      sh_seq;
    uint32_t* const hist = wide;
    uint32_t* const cand = wide + kBins;
    /* Merged-form lists over the same words: the bin's keys, their blocks, and the blocks above. */
    uint32_t* const mkey = wide + kBins;
    uint32_t* const mblk = mkey + kMergeCand;
    uint32_t* const mabv = mblk + kMergeCand;
    bool merged = false;

    const int m = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & (kWave - 1), wave = t >> 5;
    constexpr int kWideShift = WideKey<Signed>::kWideShift;

    if (t == 0) sh_seq = 0;
    __syncthreads();
    for (int i = t; i < nseq; i += kThreads)
        if (m >= cu[i] && m < cu[i + 1]) sh_seq = i;
    __syncthreads();
    const int seq = sh_seq;
    const int p   = pos ? pos[m] : 0;
    const int own = (p + 1) / ratio;
    int n = nc[seq] < nb ? nc[seq] : nb;
    if (own < n) n = own;

    const float*   sc = score + (size_t)m * s_ld;
    const int32_t* tb = bt    + (size_t)seq * t_ld;
    int32_t*       ou = sel   + (size_t)m * o_ld;

    uint32_t T = 0, need = 0;
    bool sampled = false;
    if (n > topk && n >= kSampMinN && topk <= kMergeAbove) {
        /* ---- the sample, and the bound from it. */
        uint32_t* const samp = wide;
        for (int i = t; i < kSamp; i += kThreads) {
            const int  r = i / kSampRun, j = i % kSampRun;
            const long b = (long)r * (n - kSampRun) / (kSampRuns - 1) + j;
            samp[i] = keyof<Signed>(sc[b], tb[b]);
        }
        __syncthreads();
        uint32_t ms = (uint32_t)(((uint64_t)topk * 2u * kSamp + (uint64_t)n - 1u) / (uint64_t)n) + 8u;
        if (ms > (uint32_t)kSamp) ms = (uint32_t)kSamp;
        uint32_t lo, lo_need;
        lds_kth(samp, kSamp, ms, wide + kSamp, suf, wsum, &sh_T, &sh_need, &lo, &lo_need);

        /* ---- one pass: every key at or above the bound, with its block, in block order. */
        uint32_t* const lkey = wide + kBins;
        uint32_t* const lblk = lkey + kSampCap;
        /* Eight consecutive keys per thread: vector loads and one workgroup prefix per 2048 keys
         * instead of per 256. Block order holds (consecutive keys, threads in order). */
        constexpr int kPer = 8;
        const bool vec = (((uintptr_t)sc | (uintptr_t)tb) & 15u) == 0u;
        uint32_t cnt = 0;
        for (int b0 = 0; b0 < n; b0 += kThreads * kPer) {
            const int bb = b0 + t * kPer;
            uint32_t kk[kPer];
            if (vec && bb + kPer <= n) {
                const float4 s0 = *(const float4*)(sc + bb), s1 = *(const float4*)(sc + bb + 4);
                const int4   p0 = *(const int4*)(tb + bb),   p1 = *(const int4*)(tb + bb + 4);
                kk[0] = keyof<Signed>(s0.x, p0.x); kk[1] = keyof<Signed>(s0.y, p0.y);
                kk[2] = keyof<Signed>(s0.z, p0.z); kk[3] = keyof<Signed>(s0.w, p0.w);
                kk[4] = keyof<Signed>(s1.x, p1.x); kk[5] = keyof<Signed>(s1.y, p1.y);
                kk[6] = keyof<Signed>(s1.z, p1.z); kk[7] = keyof<Signed>(s1.w, p1.w);
            } else {
#pragma unroll
                for (int e = 0; e < kPer; ++e)
                    kk[e] = bb + e < n ? keyof<Signed>(sc[bb + e], tb[bb + e]) : 0u;
            }
            uint32_t mine = 0;
#pragma unroll
            for (int e = 0; e < kPer; ++e) mine += (kk[e] != 0u && kk[e] >= lo) ? 1u : 0u;
            const uint32_t inc = r4d_wave_prefix_u32(mine, lane);
            if (lane == kWave - 1) wsum[wave] = inc;
            __syncthreads();
            uint32_t before, tot;
            counts_before<kWaves>(wsum, lane, wave, &before, &tot);
            uint32_t j = cnt + before + inc - mine;
#pragma unroll
            for (int e = 0; e < kPer; ++e) {
                if (kk[e] == 0u || kk[e] < lo) continue;
                if (j < (uint32_t)kSampCap) { lkey[j] = kk[e]; lblk[j] = (uint32_t)(bb + e); }
                ++j;
            }
            cnt += tot;
            __syncthreads();
        }

        /* ---- the table from the list, when the list holds the answer. */
        /* More than topk: exactly topk could be every live key, which takes no threshold. */
        if (cnt > (uint32_t)topk && cnt <= (uint32_t)kSampCap) {
            lds_kth(lkey, (int)cnt, (uint32_t)topk, hist, suf, wsum, &sh_T, &sh_need, &T, &need);
            uint32_t hbase = 0, ebase = 0;
            for (uint32_t i0 = 0; i0 < cnt; i0 += kThreads) {
                const uint32_t i = i0 + (uint32_t)t;
                const uint32_t k = i < cnt ? lkey[i] : 0u;
                const bool hi = i < cnt && k > T;
                const bool eq = i < cnt && k == T;
                const unsigned long long mh = __ballot(hi), me = __ballot(eq);
                if (lane == 0) {
                    wsum[wave]  = (uint32_t)__popcll(mh);
                    wsum2[wave] = (uint32_t)__popcll(me);
                }
                __syncthreads();
                uint32_t before_h, tot_h, before_e, tot_e;
                counts_before<kWaves>(wsum,  lane, wave, &before_h, &tot_h);
                counts_before<kWaves>(wsum2, lane, wave, &before_e, &tot_e);
                const uint32_t hp = hbase + before_h +
                                    (uint32_t)__popcll(mh & ((1ull << lane) - 1ull));
                const uint32_t ep = ebase + before_e +
                                    (uint32_t)__popcll(me & ((1ull << lane) - 1ull));
                if (hi) ou[hp] = tb[lblk[i]];
                if (eq && ep < need) ou[(uint32_t)topk - need + ep] = tb[lblk[i]];
                hbase += tot_h;
                ebase += tot_e;
                __syncthreads();
            }
            sampled = true;
        }
        __syncthreads();
    }
    if (n > topk && !sampled) {
        /* ---- the first digit, over global memory. */
        for (int i = t; i < kWideBins; i += kThreads) wide[i] = 0;
        __syncthreads();
        for (int b = t; b < n; b += kThreads) {
            const uint32_t k = keyof<Signed>(sc[b], tb[b]);
            if (k != 0u) atomicAdd(&wide[k >> kWideShift], 1u);
        }
        __syncthreads();

        /* The bin holding the topk-th largest key: the largest bin whose suffix count reaches
         * topk. A thread owns kWidePer consecutive bins, gets the count above them by a suffix
         * scan, and walks its own bins down. With topk or fewer keys live there is no search:
         * T stays 0 and every live key is taken, in block order, as libref takes them. */
        uint32_t loc[kWidePer], mine = 0;
#pragma unroll
        for (int j = 0; j < kWidePer; ++j) { loc[j] = wide[t * kWidePer + j]; mine += loc[j]; }
        const uint32_t x = r4d_wave_suffix_u32(mine, lane);
        if (lane == 0) wsum[wave] = x;
        __syncthreads();
        uint32_t above = 0, live = 0;
        for (int w = 0; w < kWaves; ++w) {
            live += wsum[w];
            if (w > wave) above += wsum[w];
        }
        uint32_t run = x - mine + above;
        if (t == 0) { sh_bin = 0xFFFFFFFFu; sh_T = 0; sh_need = 0; }
        __syncthreads();
        if (live > (uint32_t)topk) {
#pragma unroll
            for (int j = kWidePer - 1; j >= 0; --j) {
                if (run < (uint32_t)topk && run + loc[j] >= (uint32_t)topk) {
                    sh_bin  = (uint32_t)(t * kWidePer + j);
                    sh_need = (uint32_t)topk - run;
                    sh_cnt  = loc[j];
                }
                run += loc[j];
            }
        }
        __syncthreads();
        const uint32_t B = sh_bin;

        if (B != 0xFFFFFFFFu) {
            uint32_t fixed = B << kWideShift, want = sh_need;
            /* ---- the threshold's bin into LDS when it fits: with its blocks and the blocks above,
             * in block order, when those fit too (merged form); else its keys alone. */
            merged = sh_cnt <= (uint32_t)kMergeCand && topk <= kMergeAbove;
            const bool inlds = sh_cnt <= (uint32_t)kWideCand;
            __syncthreads();
            if (t == 0) sh_cnt = 0;
            __syncthreads();
            uint32_t n_abv = 0;
            if (merged) {
                uint32_t n_in = 0;
                for (int b0 = 0; b0 < n; b0 += kThreads) {
                    const int b = b0 + t;
                    const uint32_t k = b < n ? keyof<Signed>(sc[b], tb[b]) : 0u;
                    const bool abv = k != 0u && (k >> kWideShift) > B;
                    const bool inb = k != 0u && (k >> kWideShift) == B;
                    const unsigned long long ma = __ballot(abv), mi = __ballot(inb);
                    if (lane == 0) {
                        wsum[wave]  = (uint32_t)__popcll(ma);
                        wsum2[wave] = (uint32_t)__popcll(mi);
                    }
                    __syncthreads();
                    uint32_t before_a, tot_a, before_i, tot_i;
                    counts_before<kWaves>(wsum,  lane, wave, &before_a, &tot_a);
                    counts_before<kWaves>(wsum2, lane, wave, &before_i, &tot_i);
                    if (abv)
                        mabv[n_abv + before_a + (uint32_t)__popcll(ma & ((1ull << lane) - 1ull))] =
                            (uint32_t)b;
                    if (inb) {
                        const uint32_t j = n_in + before_i +
                                           (uint32_t)__popcll(mi & ((1ull << lane) - 1ull));
                        mkey[j] = k;
                        mblk[j] = (uint32_t)b;
                    }
                    n_abv += tot_a;
                    n_in  += tot_i;
                    __syncthreads();
                }
                if (t == 0) sh_cnt = n_in;
                __syncthreads();
            } else if (inlds) {
                for (int b = t; b < n; b += kThreads) {
                    const uint32_t k = keyof<Signed>(sc[b], tb[b]);
                    if (k != 0u && (k >> kWideShift) == B) cand[atomicAdd(&sh_cnt, 1u)] = k;
                }
                __syncthreads();
            }
            const int ncand = (merged || inlds) ? (int)sh_cnt : n;
            const uint32_t* const keys = merged ? mkey : cand;

            /* ---- the other 19 bits: digits of 8, 8 and 3 (20 bits and 8, 8, 4 when signed). */
            const int shifts[3] = { kWideShift - 8, kWideShift - 16, 0 };
            const int widths[3] = { 8, 8, kWideShift - 16 };
#pragma unroll
            for (int d = 0; d < 3; ++d) {
                const int      shift = shifts[d];
                const uint32_t dm    = (1u << widths[d]) - 1u;
                const uint32_t hm    = 0xFFFFFFFFu << (shift + widths[d]);
                for (int i = t; i < kBins; i += kThreads) hist[i] = 0;
                __syncthreads();
                for (int i = t; i < ncand; i += kThreads) {
                    const uint32_t k = (merged || inlds) ? keys[i] : keyof<Signed>(sc[i], tb[i]);
                    if (k != 0u && (k & hm) == (fixed & hm)) atomicAdd(&hist[(k >> shift) & dm], 1u);
                }
                __syncthreads();
                bin_suffix<kThreads>(hist, suf, wsum);
                if (t == 0) { sh_T = fixed; sh_need = want - suf[0]; }
                __syncthreads();
                {
                    const uint32_t after = (t + 1 < kBins) ? suf[t + 1] : 0u;
                    if (suf[t] >= want && after < want) {
                        sh_T    = fixed | ((uint32_t)t << shift);
                        sh_need = want - after;
                    }
                }
                __syncthreads();
                fixed = sh_T;
                want  = sh_need;
            }
            T = fixed; need = want;

            if (merged) {
                /* ---- emit from the lists. An in-bin key above the threshold takes the slot after
                 * every taken key before its block: in-bin ones by prefix count, above-bin ones by
                 * binary search. Its running count then replaces its key so above-bin keys can find
                 * their slots the same way. Ties fill from topk - need, in block order. */
                const uint32_t nin = (uint32_t)ncand;
                uint32_t hbase = 0, ebase = 0;
                for (uint32_t i0 = 0; i0 < nin; i0 += kThreads) {
                    const uint32_t i = i0 + (uint32_t)t;
                    const uint32_t k = i < nin ? mkey[i] : 0u;
                    const bool hi = i < nin && k > T;
                    const bool eq = i < nin && k == T;
                    const unsigned long long mh = __ballot(hi), me = __ballot(eq);
                    if (lane == 0) {
                        wsum[wave]  = (uint32_t)__popcll(mh);
                        wsum2[wave] = (uint32_t)__popcll(me);
                    }
                    __syncthreads();
                    uint32_t before_h, tot_h, before_e, tot_e;
                    counts_before<kWaves>(wsum,  lane, wave, &before_h, &tot_h);
                    counts_before<kWaves>(wsum2, lane, wave, &before_e, &tot_e);
                    const uint32_t hp = hbase + before_h +
                                        (uint32_t)__popcll(mh & ((1ull << lane) - 1ull));
                    const uint32_t ep = ebase + before_e +
                                        (uint32_t)__popcll(me & ((1ull << lane) - 1ull));
                    if (i < nin) {
                        const uint32_t b = mblk[i];
                        if (hi) ou[hp + lower_bound_u32(mabv, n_abv, b)] = tb[b];
                        if (eq && ep < need) ou[(uint32_t)topk - need + ep] = tb[b];
                        mkey[i] = hp;
                    }
                    hbase += tot_h;
                    ebase += tot_e;
                    __syncthreads();
                }
                for (uint32_t j = (uint32_t)t; j < n_abv; j += kThreads) {
                    const uint32_t b = mabv[j];
                    const uint32_t p = lower_bound_u32(mblk, nin, b);
                    ou[j + (p < nin ? mkey[p] : hbase)] = tb[b];
                }
            }
        } else {
            T = sh_T; need = sh_need;
        }
    }
    __syncthreads();
    if (t == 0) sh_out = 0;
    __syncthreads();

    /* ---- the emit, identical to the kernel above. */
    uint32_t taken_eq = 0;
    for (int b0 = 0; b0 < ((merged || sampled) ? 0 : n); b0 += kThreads) {
        const int b = b0 + t;
        uint32_t k = 0;
        if (b < n) k = keyof<Signed>(sc[b], tb[b]);
        const bool hi = k != 0u && (n <= topk || k > T);
        const bool eq = k != 0u && n > topk && k == T;

        const unsigned long long mh = __ballot(hi), me = __ballot(eq);
        const uint32_t rh = (uint32_t)__popcll(mh & ((1ull << lane) - 1ull));
        const uint32_t re = (uint32_t)__popcll(me & ((1ull << lane) - 1ull));
        if (lane == 0) wsum[wave] = (uint32_t)__popcll(mh);
        __syncthreads();
        uint32_t before_h, tot_h;
        counts_before<kWaves>(wsum, lane, wave, &before_h, &tot_h);
        const uint32_t base_h = sh_out + before_h;
        __syncthreads();
        if (lane == 0) wsum[wave] = (uint32_t)__popcll(me);
        __syncthreads();
        uint32_t before_e, tot_e;
        counts_before<kWaves>(wsum, lane, wave, &before_e, &tot_e);
        const uint32_t base_e = taken_eq + before_e;

        if (hi) {
            const uint32_t slot = base_h + rh;
            if (slot < (uint32_t)topk) ou[slot] = tb[b];
        }
        if (eq) {
            const uint32_t rank = base_e + re;
            if (rank < need) {
                const uint32_t slot = (uint32_t)topk - need + rank;
                ou[slot] = tb[b];
            }
        }
        __syncthreads();
        if (t == 0) sh_out += tot_h;
        taken_eq += tot_e;
        __syncthreads();
    }

    /* ---- the partial page, appended last, as above. */
    if (t == 0) {
        uint32_t out = (n <= topk || T == 0u) ? sh_out : (uint32_t)topk;
        const int32_t part = (n < nb && (p + 1) % ratio != 0) ? tb[n] : -1;
        if (part >= 0) ou[out++] = part;
        for (uint32_t i = out; i < (uint32_t)(topk + 1); ++i) ou[i] = -1;
        nsel[m] = (int32_t)out;
        const int32_t su = (int32_t)((out == 0) ? 0 : (out - 1) * (uint32_t)ratio +
                                                      (uint32_t)(p % ratio) + 1u);
        if (seqused) seqused[m] = su;
    }
}

/* Form override for tests: 0 = automatic, 1 = single-workgroup kernel, 2 = wide kernel. Does not
 * affect the split path. */
static int g_qsa_select_form = 0;

/* ===================================================== the split path, for a decode-shaped launch
 *
 * One workgroup per query leaves most of the GPU idle at decode (1 + n_spec workgroups, each
 * walking the whole block list five times; ~257 us per layer per step at 200K context).
 *
 * A chunk's local top-k contains every global top-k member of that chunk, so the block list can
 * be split exactly: if a block were in the global top-k but not its chunk's, k blocks of the chunk
 * would outrank it, hence k blocks globally. This needs the same total order on both sides: key
 * descending, then block index ascending, as the emit builds by prefix count.
 *
 * So up to kParWMax chunks per query find their local top-k in LDS, and one workgroup per query
 * merges the candidates and emits. Chunk lists are block-ordered and concatenated in chunk order,
 * so the tie rule needs no sort. The merge is not split further: its input is already small, and
 * splitting it would need a global prefix (two more kernels and dispatch gaps) for ~4% of the cost.
 *
 * Above kParM query rows the grid is wide enough and the single-workgroup kernel runs. */
constexpr int kParM    = 32;      /* query rows the split path serves; a prefill chunk is wider */
constexpr int kParWMax = 16;      /* chunks a query, and what the scratch is sized for */
constexpr int kParCap  = 8192;    /* keys a chunk may stage: 32 KiB, the same budget as above */
/* Set in a chunk's count when the chunk had more than topk live keys: the merge then knows the
 * whole list has more than topk, which the candidate count alone cannot tell at exactly topk. */
constexpr int32_t kParOver = 1 << 30;

/* The merge runs one workgroup per query (four at the decode shape), so its only parallelism is
 * within the workgroup: 1024 threads give eight waves per SIMD instead of two at 256, enough to
 * hide LDS and global latency (24 VGPRs, so occupancy is not the limit). The bin scan stays at
 * 256 threads. */
constexpr int kMergeThreads = 1024;
/* The local pass has many more workgroups (a query's chunks times the query rows), so a narrower
 * block -- except where the grid is small enough to sit on the card at once: up to
 * kLocalWideGrid workgroups (four verify rows over as many as kParWMax chunks) the wide block
 * hides the same latencies as the merge's. At 95K tokens of context, one sequence under the draft
 * head, the twelve local launches of a verify pass take 105 us wide against 126 (and 209 at 256
 * threads). */
constexpr int kLocalThreads = 512, kLocalWideThreads = 1024;
constexpr int kLocalWideGrid = 64;
static_assert(kParCap / kLocalThreads <= 32 && kLdsCap / kMergeThreads <= 32,
              "a thread's run of keys is a 32-bit mask (run_take)");

/* Chunks for a block list of nb (at most kParWMax). Both kernels index the scratch at the fixed
 * kParWMax stride, so they agree regardless of this value. */
static inline int par_chunks(long long nb) {
    long long w = (nb + 4095) / 4096;
    if (w < 1) w = 1;
    if (w > kParWMax) w = kParWMax;
    return (int)w;
}

/* The threshold over keys staged in LDS: the same four-digit radix and suffix scan as the
 * single-workgroup kernel, so split and unsplit selections are bit-identical.
 *
 * `every_at_want`: `want` or fewer live keys are all taken (T = 0), as the single-workgroup
 * kernel takes topk or fewer. The local pass sets it. The merge does not: its candidates can be
 * exactly the top-k of a longer list, whose ties still go last, and it never has fewer.
 *
 * Templated on the workgroup width (the merge is wider than the bin count). Above 256 threads the
 * surplus skips the bin scan; guards wrap the work, never a __syncthreads(). */
template <int NT>
__device__ __forceinline__ void radix_thresh(const uint32_t* __restrict__ key, int n,
                                             uint32_t want, bool every_at_want, uint32_t* hist,
                                             uint32_t* suf, uint32_t* wt, uint32_t* sh_T,
                                             uint32_t* sh_need) {
    const int t = (int)threadIdx.x;
    uint32_t fixed = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < kBins; i += NT) hist[i] = 0;
        __syncthreads();
        const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int b = t; b < n; b += NT) {
            const uint32_t k = key[b];
            if (k != 0u && (k & mask) == (fixed & mask))
                atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        bin_suffix<NT>(hist, suf, wt);
        if (t == 0) { *sh_T = fixed; *sh_need = want - suf[0]; }
        __syncthreads();
        if (t < kBins) {
            const uint32_t after = (t + 1 < kBins) ? suf[t + 1] : 0u;
            if (suf[t] >= want && after < want) {
                *sh_T    = fixed | ((uint32_t)t << shift);
                *sh_need = want - after;
            }
        }
        __syncthreads();
        /* Only the first digit counts every live key (see the single-workgroup kernel). */
        if (every_at_want && shift == 24 && suf[0] <= want) {
            if (t == 0) { *sh_T = 0u; *sh_need = 0u; }
            __syncthreads();
            break;
        }
        fixed = *sh_T;
        want  = *sh_need;
    }
}

/* The qualifying set by prefix count.
 *
 * The merge must reproduce the single-workgroup kernel's table exactly (entries above the
 * threshold first, ties last): the attention sums pages in table order and float addition is not
 * associative. The local pass instead writes in block order, since the merge's tie rule reads it
 * that way. Slots come from workgroup prefix counts, never from scheduling.
 *
 * A thread owns `run` consecutive keys of the list in LDS (threads in order, so block order holds)
 * and classes them into bit masks; three workgroup prefixes over the per-thread counts then place
 * every key, two barriers each whatever the list's length (a prefix per NT keys of the list costs
 * five barriers a round, eight rounds for a chunk). `run` is at most 32. */
struct RunTake {
    uint32_t hi, eq, take;             /* bit j: this thread's key j is above the threshold, a tie
                                        * at it, taken */
    uint32_t rank_eq;                  /* ties ahead of this thread's first key */
    uint32_t slot_hi, slot_take;       /* above-threshold and taken keys ahead of it */
    uint32_t n_hi, n_take;             /* the workgroup's totals */
};

/* The exclusive workgroup prefix of a per-thread count, and the total. */
template <int NT>
__device__ __forceinline__ uint32_t run_prefix(uint32_t mine, uint32_t* wsum, int lane, int wave,
                                               uint32_t* total) {
    constexpr int NW = NT / kWave;
    const uint32_t inc = r4d_wave_prefix_u32(mine, lane);
    if (lane == kWave - 1) wsum[wave] = inc;
    __syncthreads();
    uint32_t before, tot;
    counts_before<NW>(wsum, lane, wave, &before, &tot);
    __syncthreads();
    *total = tot;
    return before + inc - mine;
}

/* `all`: below the top-k, everything live is in. Else keys above `T` are in and the first `need`
 * ties at `T` in list order. */
template <int NT>
__device__ __forceinline__ RunTake run_take(const uint32_t* __restrict__ key, int n, int run,
                                            uint32_t T, uint32_t need, bool all, uint32_t* wsum,
                                            int lane, int wave) {
    RunTake r{};
    const int i0 = (int)threadIdx.x * run;
    for (int j = 0; j < run; ++j) {
        const uint32_t k = i0 + j < n ? key[i0 + j] : 0u;
        if (k != 0u && (all || k > T)) r.hi |= 1u << j;
        if (k != 0u && !all && k == T) r.eq |= 1u << j;
    }
    uint32_t n_eq = 0;
    r.rank_eq = run_prefix<NT>((uint32_t)__builtin_popcount(r.eq), wsum, lane, wave, &n_eq);
    /* The ties this thread may take: the first `need - rank_eq` of its own. */
    uint32_t left = r.rank_eq < need ? need - r.rank_eq : 0u, eq_take = 0u;
    for (int j = 0; j < run && left > 0u; ++j)
        if ((r.eq >> j) & 1u) { eq_take |= 1u << j; --left; }
    r.take = r.hi | eq_take;
    r.slot_hi   = run_prefix<NT>((uint32_t)__builtin_popcount(r.hi),   wsum, lane, wave, &r.n_hi);
    r.slot_take = run_prefix<NT>((uint32_t)__builtin_popcount(r.take), wsum, lane, wave, &r.n_take);
    return r;
}

/* ---- pass one: every chunk's own top-k, as (key, block index) in block order. */
#define QSA_LOCAL_PARAMS                                                                        \
    const float* __restrict__ score, const int32_t* __restrict__ bt,                            \
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,                            \
    const int32_t* __restrict__ cu,                                                             \
    uint32_t* __restrict__ ck, int32_t* __restrict__ ci, int32_t* __restrict__ cc,              \
    int nb, int topk, int ratio, int nseq, int s_ld, int t_ld, int chunk
template <bool Signed, int LT>
__device__ __forceinline__ void qsa_local_body(QSA_LOCAL_PARAMS) {

    __shared__ uint32_t hist[kBins], suf[kBins], wsum[LT / kWave];
    __shared__ uint32_t sh_T, sh_need;
    __shared__ int      sh_seq;
    R4D_DYN_LDS(uint32_t, sh_key);

    const int m = (int)blockIdx.y, c = (int)blockIdx.x;
    const int t = (int)threadIdx.x, lane = t & (kWave - 1), wave = t >> 5;

    if (t == 0) { sh_seq = 0; cc[(size_t)m * kParWMax + c] = 0; }
    __syncthreads();
    for (int i = t; i < nseq; i += LT)
        if (m >= cu[i] && m < cu[i + 1]) sh_seq = i;
    __syncthreads();
    const int seq = sh_seq;
    const int p   = pos ? pos[m] : 0;
    const int own = (p + 1) / ratio;
    int n = nc[seq] < nb ? nc[seq] : nb;
    if (own < n) n = own;

    const int lo = c * chunk;
    if (lo >= n) return;
    const int hi = (lo + chunk) < n ? (lo + chunk) : n;
    const int cn = hi - lo;

    const float*   sc = score + (size_t)m * s_ld;
    const int32_t* tb = bt    + (size_t)seq * t_ld;
    /* A thread's scores and pages (every LT-th of the chunk) are all loaded before the first key
     * is staged: one round trip instead of one a key. */
    {
        constexpr int kPre = kParCap / LT;
        float   sv[kPre];
        int32_t pv[kPre];
#pragma unroll
        for (int u = 0; u < kPre; ++u) {
            const int i = t + u * LT;
            sv[u] = 0.0f; pv[u] = -1;
            if (i < cn) { sv[u] = sc[lo + i]; pv[u] = tb[lo + i]; }
        }
#pragma unroll
        for (int u = 0; u < kPre; ++u) {
            const int i = t + u * LT;
            if (i < cn) sh_key[i] = keyof<Signed>(sv[u], pv[u]);
        }
    }
    __syncthreads();

    const bool all = cn <= topk;
    uint32_t T = 0u, need = 0u;
    if (!all) {
        radix_thresh<LT>(sh_key, cn, (uint32_t)topk, true, hist, suf, wsum, &sh_T, &sh_need);
        T = sh_T; need = sh_need;
    }
    /* A threshold means more than topk live keys here, so more than topk in the whole list. */
    const bool over = !all && T != 0u;

    /* In block order: the merge breaks ties by block index. */
    uint32_t* dk = ck + (size_t)(m * kParWMax + c) * topk;
    int32_t*  di = ci + (size_t)(m * kParWMax + c) * topk;
    const int run = (cn + LT - 1) / LT;
    const RunTake r = run_take<LT>(sh_key, cn, run, T, need, all, wsum, lane, wave);
    for (int j = 0; j < run; ++j) {
        if (!((r.take >> j) & 1u)) continue;
        const uint32_t slot = r.slot_take + (uint32_t)__builtin_popcount(r.take & ((1u << j) - 1u));
        if (slot >= (uint32_t)topk) continue;
        const int i = t * run + j;
        dk[slot] = sh_key[i];
        /* Store the page, not the block index: resolving it here spreads the dependent load
         * over many workgroups instead of the merge's few. The merge breaks ties by list
         * position (block-ordered), never by index. A taken candidate has k != 0, so its
         * page is live. */
        di[slot] = tb[lo + i];
    }
    if (t == 0) cc[(size_t)m * kParWMax + c] =
        (int32_t)(r.n_take < (uint32_t)topk ? r.n_take : (uint32_t)topk) | (over ? kParOver : 0);
}
/* The kernel around the body, whose blocks return where their chunk is empty: CH `r4dchain` is the
 * chained form (r4d_chain.h). */
template <bool Signed, int LT, class CH>
__global__ __launch_bounds__(LT) void qsa_local_kernel(QSA_LOCAL_PARAMS) {
    if (CH::on) r4d_chain_wait();
    qsa_local_body<Signed, LT>(score, bt, nc, pos, cu, ck, ci, cc, nb, topk, ratio, nseq, s_ld,
                               t_ld, chunk);
    if (CH::on) r4d_chain_done();
}
#undef QSA_LOCAL_PARAMS

/* ---- pass two: merge the chunk lists and emit the page-id table. */
template <bool Signed, class CH>
__global__ __launch_bounds__(kMergeThreads) void qsa_merge_kernel(
    const uint32_t* __restrict__ ck, const int32_t* __restrict__ ci,
    const int32_t* __restrict__ cc, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu,
    int32_t* __restrict__ sel, int32_t* __restrict__ nsel, int32_t* __restrict__ seqused,
    int nb, int topk, int ratio, int nseq, int t_ld, int o_ld, int nchunk) {
    /* CH `r4dchain` is the chained form (r4d_chain.h): no block or thread returns before the end. */
    if (CH::on) r4d_chain_wait();

    constexpr int NT = kMergeThreads;
    __shared__ uint32_t hist[kBins], suf[kBins], wsum[NT / kWave];
    __shared__ uint32_t sh_T, sh_need, sh_tot, sh_over;
    __shared__ int      sh_seq, sh_off[kParWMax + 1];
    R4D_DYN_LDS(uint32_t, sh_key);

    const int m = (int)blockIdx.x;
    const int t = (int)threadIdx.x, lane = t & (kWave - 1), wave = t >> 5;

    if (t == 0) sh_seq = 0;
    __syncthreads();
    for (int i = t; i < nseq; i += NT)
        if (m >= cu[i] && m < cu[i + 1]) sh_seq = i;
    __syncthreads();
    const int seq = sh_seq;
    const int p   = pos ? pos[m] : 0;
    const int own = (p + 1) / ratio;
    int n = nc[seq] < nb ? nc[seq] : nb;
    if (own < n) n = own;

    /* Chunk offsets: the first wave loads a chunk's count a lane and takes their prefix, one
     * round trip for all of them (nchunk <= kParWMax lanes). */
    {
        const int32_t  v   = t < nchunk ? cc[(size_t)m * kParWMax + t] : 0;
        const uint32_t cnt = (uint32_t)(v & ~kParOver);
        const uint32_t inc = r4d_wave_prefix_u32(cnt, lane);
        const unsigned long long ov = __ballot(t < nchunk && (v & kParOver) != 0);
        if (t < nchunk) sh_off[t] = (int)(inc - cnt);
        if (t == nchunk - 1) { sh_off[nchunk] = (int)inc; sh_tot = inc; }
        if (t == 0) sh_over = ov != 0ull ? 1u : 0u;
    }
    __syncthreads();
    const int tot = (int)sh_tot;

    /* The candidates into LDS in list order. A thread's keys (every NT-th of the list) are all
     * loaded before the first is stored: one round trip, where a pass per chunk made one for each
     * of the nchunk lists. */
    {
        constexpr int kPre = kLdsCap / NT;
        uint32_t kv[kPre];
#pragma unroll
        for (int u = 0; u < kPre; ++u) {
            const int i = t + u * NT;
            kv[u] = 0u;
            if (i < tot) {
                int c = nchunk - 1;
                while (c > 0 && sh_off[c] > i) --c;
                kv[u] = ck[(size_t)(m * kParWMax + c) * topk + (i - sh_off[c])];
            }
        }
#pragma unroll
        for (int u = 0; u < kPre; ++u) {
            const int i = t + u * NT;
            if (i < tot) sh_key[i] = kv[u];
        }
    }
    __syncthreads();

    /* Everything, in block order, exactly when the single-workgroup kernel takes everything: at
     * most topk blocks, or topk or fewer live. With no chunk over topk the candidates are every
     * live key, so their count is the live count; a chunk over topk means more than topk live,
     * and then exactly topk candidates are the top-k, whose ties go last. */
    const bool all = n <= topk || (!sh_over && tot <= topk);
    uint32_t T = 0u, need = 0u;
    if (!all) {
        radix_thresh<NT>(sh_key, tot, (uint32_t)topk, false, hist, suf, wsum, &sh_T, &sh_need);
        T = sh_T; need = sh_need;
    }

    const int32_t* tb = bt  + (size_t)seq * t_ld;
    int32_t*       ou = sel + (size_t)m * o_ld;
    const int run = (tot + NT - 1) / NT;
    const RunTake r = run_take<NT>(sh_key, tot, run, T, need, all, wsum, lane, wave);
    for (int j = 0; j < run; ++j) {
        if (!((r.take >> j) & 1u)) continue;
        const int i = t * run + j;
        /* The candidate's chunk, from the offsets. */
        int c = nchunk - 1;
        while (c > 0 && sh_off[c] > i) --c;
        /* Already a page (resolved by the local pass); tb is still needed for the partial
         * page. */
        const int32_t page = ci[(size_t)(m * kParWMax + c) * topk + (i - sh_off[c])];
        /* The single-workgroup kernel's layout: above the threshold first, ties last. */
        const uint32_t below = (1u << j) - 1u;
        if ((r.hi >> j) & 1u) {
            const uint32_t slot = r.slot_hi + (uint32_t)__builtin_popcount(r.hi & below);
            if (slot < (uint32_t)topk) ou[slot] = page;
        } else {
            ou[(uint32_t)topk - need + r.rank_eq + (uint32_t)__builtin_popcount(r.eq & below)] = page;
        }
    }
    const uint32_t n_hi = r.n_hi;

    /* The partial page, appended last, as in the single-workgroup kernel. */
    if (t == 0) {
        /* Use n, not the candidate count, as the single-workgroup kernel does: above topk the
         * table is full (ties fill it), and the merged count can fall short, which would let the
         * padding loop overwrite ties with -1. The exception is the same as there: topk or
         * fewer live blocks, all of them emitted. */
        uint32_t out = all ? n_hi : (uint32_t)topk;
        const int32_t part = (n < nb && (p + 1) % ratio != 0) ? tb[n] : -1;
        if (part >= 0) ou[out++] = part;
        for (uint32_t i = out; i < (uint32_t)(topk + 1); ++i) ou[i] = -1;
        nsel[m] = (int32_t)out;
        const int32_t su = (int32_t)((out == 0) ? 0
                                   : (out - 1) * (uint32_t)ratio + (uint32_t)(p % ratio) + 1u);
        seqused[m] = su;
    }
    if (CH::on) r4d_chain_done();
}
}  // namespace

/* Split-path scratch, sized from topk alone so declare (parameters only) and issue agree: two
 * candidate planes and a count per (query, chunk) at the fixed kParM x kParWMax stride (2.1 MiB
 * at topk 512). */
static inline int64_t par_scratch_bytes(long long topk) {
    if (topk <= 0) return 0;
    return (int64_t)kParM * kParWMax * topk * 8 + (int64_t)kParM * kParWMax * 4;
}

//   score [M][nb]        f32, `s_ld` elements a row (0 = nb); M is query rows
//   bt    [nseq][nb]     i32, `t_ld` elements a row (0 = nb); page ids, -1 = absent
//   nc    [nseq]         i32, complete blocks per sequence
//   pos   [M]            i32, each query's absolute position
//   cu    [nseq+1]       i32, cumulative query counts: which sequence a query row belongs to
//   sel   [M][topk+1]    i32, `o_ld` elements a row (0 = topk+1); -1 padded
//   nsel  [M]            i32
// Signed selects the key order (keyof), from the op's `signed` parameter.
template <bool Signed>
static int r4d_qsa_select_bf16_emit(long score, long bt, long nc, long pos, long cu, long sel,
                                   long nsel, long seqused, int M, int nb, int topk, int ratio,
                                   int nseq, int s_ld, int t_ld, int o_ld,
                                   void* scratch, long long scratch_bytes, RadStream stream,
                                   bool chain) {
    if (!score || !bt || !nc || !cu || !sel || !nsel || !seqused) return R4D_NO(RAD_E_INVAL);
    if (M <= 0 || nb <= 0 || topk <= 0 || ratio <= 0 || nseq <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!s_ld) s_ld = nb;
    if (!t_ld) t_ld = nb;
    if (!o_ld) o_ld = topk + 1;

    /* The split path for decode-shaped launches when chunks and candidate lists fit LDS and
     * scratch was given. The choice never changes the result. */
    const int   nchunk = par_chunks(nb);
    const int   chunk  = (int)((nb + nchunk - 1) / nchunk);
    const int   lds    = lds_cap_of();
    const bool  split  = M <= kParM && nchunk > 1 && scratch != nullptr &&
                         scratch_bytes >= par_scratch_bytes(topk) &&
                         chunk <= kParCap && chunk <= lds && nchunk * topk <= lds;
    if (split) {
        uint32_t* ck = (uint32_t*)scratch;
        int32_t*  ci = (int32_t*)(ck + (size_t)kParM * kParWMax * topk);
        int32_t*  cc = ci + (size_t)kParM * kParWMax * topk;
        /* Each form named at its own launch, as below. */
#define R4D_QLOC(LT_, CH_)                                                                      \
        R4D_EMIT(stream, (qsa_local_kernel<Signed, LT_, CH_>),                                  \
                 dim3((unsigned)nchunk, (unsigned)M), dim3(LT_),                                \
                 (size_t)chunk * sizeof(uint32_t),                                              \
                 (const float*)score, (const int32_t*)bt, (const int32_t*)nc,                   \
                 (const int32_t*)pos, (const int32_t*)cu, ck, ci, cc, nb, topk,                 \
                 ratio, nseq, s_ld, t_ld, chunk)
#define R4D_QMRG(CH_)                                                                           \
        R4D_EMIT(stream, (qsa_merge_kernel<Signed, CH_>), dim3((unsigned)M),                    \
                 dim3(kMergeThreads), (size_t)nchunk * topk * sizeof(uint32_t),                 \
                 (const uint32_t*)ck, (const int32_t*)ci, (const int32_t*)cc,                   \
                 (const int32_t*)bt, (const int32_t*)nc, (const int32_t*)pos,                   \
                 (const int32_t*)cu, (int32_t*)sel, (int32_t*)nsel, (int32_t*)seqused,          \
                 nb, topk, ratio, nseq, t_ld, o_ld, nchunk)
        const bool wide = (long)M * nchunk <= kLocalWideGrid;
        const int  rc = wide ? (chain ? R4D_QLOC(kLocalWideThreads, r4dchain)
                                      : R4D_QLOC(kLocalWideThreads, R4dUnchained))
                             : (chain ? R4D_QLOC(kLocalThreads, r4dchain)
                                      : R4D_QLOC(kLocalThreads, R4dUnchained));
        if (rc != RAD_OK) return rc;
        return chain ? R4D_QMRG(r4dchain) : R4D_QMRG(R4dUnchained);
#undef R4D_QLOC
#undef R4D_QMRG
    }

    /* A wide launch (prefill chunk, key cache off) takes the wide kernel. */
    if (g_qsa_select_form == 2 || (g_qsa_select_form == 0 && M > kCacheMaxM)) {
        return R4D_EMIT(stream, qsa_select_wide_kernel<Signed>, dim3((unsigned)M),
                        dim3(kThreads), 0,
                        (const float*)score, (const int32_t*)bt, (const int32_t*)nc,
                        (const int32_t*)pos, (const int32_t*)cu, (int32_t*)sel,
                        (int32_t*)nsel, (int32_t*)seqused, nb, topk, ratio, nseq, s_ld, t_ld,
                        o_ld);
    }

    /* Never more than the table width. */
    int cap = (M <= kCacheMaxM) ? lds_cap_of() : 0;
    if (cap > nb) cap = nb;
    /* Each form named at its own launch: a test reads the kernel a launch site names. */
#define R4D_QSEL(CH_)                                                                           \
    R4D_EMIT(stream, (qsa_select_kernel<Signed, CH_>), dim3((unsigned)M), dim3(kThreads),       \
             (size_t)cap * sizeof(uint32_t), (const float*)score, (const int32_t*)bt,           \
             (const int32_t*)nc, (const int32_t*)pos, (const int32_t*)cu, (int32_t*)sel,        \
             (int32_t*)nsel, (int32_t*)seqused, nb, topk, ratio, nseq, s_ld, t_ld, o_ld, cap)
    const int rc = chain ? R4D_QSEL(r4dchain) : R4D_QSEL(R4dUnchained);
#undef R4D_QSEL
    return rc;
}

extern "C" void r4d_qsa_select_pin_form(int form) { g_qsa_select_form = form; }

/* Split-path scratch, called at declare (parameters only), so it depends on topk alone. */
extern "C" int64_t r4d_rad_qsa_select_scratch(const RadArgs* a) {
    return par_scratch_bytes(r4d_geti_or(a, "topk", 0));
}

// ============================================================== the plugin ABI entry point

enum { QL_SCORE = 0, QL_BT = 1, QL_NC = 2, QL_POS = 3, QL_CU = 4, QL_SEL = 5, QL_NSEL = 6,
       QL_SEQ = 7 };

extern "C" int r4d_rad_qsa_select(const RadArgs* a, RadStream stream) {
    const RadTensor* sc = r4d_opt(a, QL_SCORE);
    const RadTensor* bt = r4d_opt(a, QL_BT);
    const RadTensor* nc = r4d_opt(a, QL_NC);
    const RadTensor* po = r4d_opt(a, QL_POS);
    const RadTensor* cu = r4d_opt(a, QL_CU);
    const RadTensor* se = r4d_opt(a, QL_SEL);
    const RadTensor* ns = r4d_opt(a, QL_NSEL);
    const RadTensor* su = r4d_opt(a, QL_SEQ);
    if (!sc || !bt || !nc || !po || !cu || !se || !ns || !su) return R4D_NO(RAD_E_INVAL);
    if (sc->rank != 2 || bt->rank != 2 || se->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long topk  = r4d_geti_or(a, "topk", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long sgn   = r4d_geti_or(a, "signed", 0);
    if (topk <= 0 || ratio <= 0) return R4D_NO(RAD_E_INVAL);
    if (sgn != 0 && sgn != 1) return R4D_NO(RAD_E_INVAL);
    /* nb = min of the two widths, as in r4d_qsa_score_bf16.hip. */
    const long long M = sc->shape[0];
    const long long nb = bt->shape[1] < sc->shape[1] ? bt->shape[1] : sc->shape[1];
    if (M < 1 || nb < 1) return R4D_NO(RAD_E_SHAPE);
    /* The sequence count comes from cu, not the table height (max_seqs). */
    if (cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const long long nseq = cu->shape[0] - 1;
    if (bt->shape[0] < nseq || r4d_numel(nc) < nseq) return R4D_NO(RAD_E_SHAPE);
    /* The output is topk + 1 wide: the extra slot is the partial page holding the query. */
    if (se->shape[0] != M || se->shape[1] != topk + 1) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(ns) < M) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(po) < M || r4d_numel(su) < M) return R4D_NO(RAD_E_SHAPE);
    const long long sp = r4d_row_pitch(sc), tp = r4d_row_pitch(bt), op = r4d_row_pitch(se);
    if (sp < 0 || tp < 0 || op < 0) return R4D_NO(RAD_E_STRIDE);
    /* Scratch is optional: without enough, the non-split kernels run. */
    auto* const emit = sgn ? r4d_qsa_select_bf16_emit<true> : r4d_qsa_select_bf16_emit<false>;
    return emit(r4d_addr(sc), r4d_addr(bt), r4d_addr(nc), r4d_addr(po),
                r4d_addr(cu), r4d_addr(se), r4d_addr(ns), r4d_addr(su),
                (int)M, (int)nb, (int)topk, (int)ratio, (int)nseq,
                (int)sp, (int)tp, (int)op, a->scratch, a->scratch_bytes,
                stream, r4d_geti_or(a, "chain", 0) != 0);
}

