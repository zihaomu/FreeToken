/* r4d_attn_paged_window.hip -- paged attention at head_dim 128, 256 and 512, 1/2/4/8 queries per
 * KV head (16 at 512), any sliding window, causal or not, bf16 or int8 cache, q at any row
 * pitch.
 *
 * head_dim 128 is Unlimited-OCR's decoder (10 query heads over 10 KV heads): the plain MHA case,
 * served here so its q slice of the fused q|k|v plane is read in place (attn_paged_qrow) instead
 * of being lifted into a dense buffer.
 *
 * Gemma 4's text layers: sliding layers are 16 query heads over 8 KV heads at head_dim 256 with a
 * 1024-token window (gqa 2 at every TP); full layers are 16 over 2 at head_dim 512 (gqa 8 at
 * TP1/TP2, 4 at TP4, 2 at TP8). The 12B's full layers are 16 over 1 (gqa 16 at TP1; its one KV
 * head is replicated at TP2/TP4, gqa 8/4). Where this overlaps the library's other h256 legs, the
 * row table decides (r4d_rows.cpp).
 *
 * Two kernels behind one entry point:
 *   - Scalar, for decode and verify: a workgroup owns one (sequence, KV head, slice of BQ query
 *     positions, split of the key range); each wave owns RW of the tile's R = BQ*gqa query rows.
 *     K/V go through LDS one or two paged blocks at a time, prefetched into registers while the
 *     previous tile is consumed. Dot products are scalar FMAs with a butterfly reduce-scatter that
 *     leaves key j's score in lane j; online softmax in the log2 domain. Memory-bound at decode.
 *   - WMMA, for prefill-shaped launches (see its header below).
 *
 * Both compute libref's ref_attn_paged, predicate for predicate:
 *   pos     = seqused[s] - q_len(s) + i                  (seqused INCLUDES this step's queries)
 *   causal  key j visible when j <= pos;  non-causal when j < seqused[s]
 *   window  causal: pos - j < window;  non-causal: |pos - j| < window;  0 = none
 *   ahead   non-causal, window > 0: only pos - j < window, the keys after pos unbounded (FWD, below)
 *   a block-table entry < 0 is a missing block: its keys are not visible
 *   no visible key at all -> a zero row
 *   k_descale / v_descale per (sequence, kv head), absent = 1
 *   attn_paged_sink: `sinks` [query heads] f32, one logit a head in the scaled-score domain, in
 *     the softmax's maximum and denominator and with no value (LiLiCorr's drafter). Folded in where
 *     a row is normalised: the unsplit kernel's store, the merge, and the WMMA prefill kernel's
 *     running max and denominator. A decode-sized sink launch stays scalar.
 *
 * Paged layout: kv_cache[n_blocks, kv_heads, 16, 2*head_dim], K then V per slot, so one block of
 * one KV head is a contiguous run of 16 * 2 * head_dim elements; a key tile is one or two runs,
 * fetched as 16-byte loads. The int8 cache (KV8 2, --kv-cache-dtype i8) appends four bytes to a
 * slot, the token's f16 scales of K and of V (kv_store_i8): a key's score is its integer codes'
 * dot times its K scale, and its row of V enters the sum at its probability times its V scale.
 *
 * Split-KV: with few (tile, kv head) pairs (decode, short verify) each tile's key range is cut into
 * `n_split` pieces, each writing an unnormalised partial (acc, max, sum) to scratch, merged by a
 * second kernel. The split count comes from host-side numbers only (rows, sequences, window or
 * max_ctx); each workgroup divides its own key range, so a split with no keys writes (0, -inf, 0)
 * and the merge needs no device-side context.
 *
 * Nothing here allocates, synchronises, or uses anything but hipLaunchKernel (docs/PLUGIN.md §5).
 * Wave32 (RDNA): the shuffles and the reduce-scatter are written for 32 lanes.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "r4d_chain.h"
#include "rad_plugin.h"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "this attention is written for wave32 (RDNA): build without -mwavefrontsize64"
#endif

namespace r4d_win {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;

/* Split-KV bookkeeping: at most kPartBudget (tile, kv head, split) triples carry partials and the
 * tile table holds at most kTinfoMax tiles. The launch stays under both; scratch is sized for them. */
constexpr int kPartBudget = 192;
constexpr int kTinfoMax = 128;
constexpr int kMaxSplit = 64;

template <int HD> struct Geo;
/* R query rows per workgroup (R / 8 per wave), BK keys per tile. BK * 2 * HD bf16 = 32 KiB of LDS
 * either way: two paged blocks at 256, one at 512, four at 128. */
template <> struct Geo<128> { static constexpr int R = 32, BK = 32; };
template <> struct Geo<256> { static constexpr int R = 32, BK = 32; };
template <> struct Geo<512> { static constexpr int R = 16, BK = 16; };

/* ------------------------------------------------------------------ device helpers */
__device__ __forceinline__ int lane_id() { return (int)(threadIdx.x & 31u); }

__device__ __forceinline__ float shfl_xor(float v, int m) {
    return __int_as_float(__builtin_amdgcn_ds_bpermute((lane_id() ^ m) << 2, __float_as_int(v)));
}
__device__ __forceinline__ int shfl_up(int v, int d) {
    const int l = lane_id();
    return __builtin_amdgcn_ds_bpermute((l >= d ? l - d : l) << 2, v);
}
__device__ __forceinline__ float readlane(float v, int l) {
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), l));
}
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v = __builtin_fmaxf(v, shfl_xor(v, d));
    return v;
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) v += shfl_xor(v, d);
    return v;
}
__device__ __forceinline__ float ex2(float x) {
#if __has_builtin(__builtin_amdgcn_exp2f)
    return __builtin_amdgcn_exp2f(x);   /* v_exp_f32: exp2(-inf) = 0 */
#else
    return exp2f(x);
#endif
}
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }

/* f(Ic<0>{}), ..., f(Ic<N-1>{}): a loop whose index is a compile-time constant in every body.
 * `#pragma unroll` is dropped past a size threshold (the fp8 row loop crosses it), and a register
 * array indexed by a surviving loop variable is demoted to scratch. This guarantees the unroll. */
template <int I> struct Ic { static constexpr int value = I; };
template <int I, int N> struct StaticFor {
    template <typename F> __device__ __forceinline__ static void run(F& f) {
        if constexpr (I < N) {
            f(Ic<I>{});
            StaticFor<I + 1, N>::run(f);
        }
    }
};

__device__ __forceinline__ void bf16x4(uint2 w, float (&f)[4]) {
    f[0] = __uint_as_float(w.x << 16);
    f[1] = __uint_as_float(w.x & 0xffff0000u);
    f[2] = __uint_as_float(w.y << 16);
    f[3] = __uint_as_float(w.y & 0xffff0000u);
}
/* OCP E4M3 (e4m3fn, bias 7, 0x7f/0xff NaN, no infinity), the KV cache format at
 * --kv-cache-dtype fp8 (docs/PLUGIN.md §10). Exact; subnormals kept. */
__device__ __forceinline__ float fp8e4m3(unsigned b) {
    const unsigned e = (b >> 3) & 15u, m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    if (e == 15u && m == 7u) v = __uint_as_float(0x7fc00000u);
    return __uint_as_float(__float_as_uint(v) | ((b & 0x80u) << 24));
}
__device__ __forceinline__ void fp8x4(unsigned w, float (&f)[4]) {
#if defined(__GFX12__)
    /* v_cvt_pk_f32_fp8: RDNA4 hardware OCP E4M3 conversion, two bytes per instruction. */
    const auto a = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, false);
    const auto b = __builtin_amdgcn_cvt_pk_f32_fp8((int)w, true);
    f[0] = a[0]; f[1] = a[1]; f[2] = b[0]; f[3] = b[1];
#else
    f[0] = fp8e4m3(w & 0xffu);
    f[1] = fp8e4m3((w >> 8) & 0xffu);
    f[2] = fp8e4m3((w >> 16) & 0xffu);
    f[3] = fp8e4m3(w >> 24);
#endif
}
/* f32 -> bf16, round to nearest even, NaN kept quiet (rad_plugin.h's rad_f32_to_bf16). */
__device__ __forceinline__ unsigned bf16_rtne(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}
__device__ __forceinline__ unsigned pack2(float a, float b) { return bf16_rtne(a) | (bf16_rtne(b) << 16); }

/* Four int8 codes as their values. */
__device__ __forceinline__ void i8x4(unsigned w, float (&f)[4]) {
    f[0] = (float)(signed char)(w & 0xffu);
    f[1] = (float)(signed char)((w >> 8) & 0xffu);
    f[2] = (float)(signed char)((w >> 16) & 0xffu);
    f[3] = (float)(signed char)(w >> 24);
}
/* Four bytes of an 8-bit cache: E4M3 (KV8 1) or int8 codes (KV8 2). */
template <int KV8>
__device__ __forceinline__ void kv8x4(unsigned w, float (&f)[4]) {
    if constexpr (KV8 == 2) i8x4(w, f);
    else fp8x4(w, f);
}
/* An f16 at `p` as a float: one of the int8 cache's scales. */
__device__ __forceinline__ float f16_at(const unsigned char* p) {
    return (float)__builtin_bit_cast(_Float16, *(const unsigned short*)p);
}

/* Four elements of one slot row, at element offset `e`, out of the LDS tile. */
template <int KV8>
__device__ __forceinline__ void lds4(const unsigned char* row, int e, float (&f)[4]) {
    if constexpr (KV8) kv8x4<KV8>(*(const unsigned*)(row + e), f);
    else bf16x4(*(const uint2*)(row + 2 * e), f);
}

/* Reduce-scatter: each lane holds a partial dot for every key of the tile; afterwards lane l holds
 * key l's full sum. Each step keeps the half its lane bit selects and adds the partner's copy:
 * 16 + 8 + 4 + 2 + 1 = 31 shuffles for 32 keys, instead of 32 * 5 for separate reductions. */
template <int H, int N>
__device__ __forceinline__ void reduce_scatter(float (&v)[N], int lane) {
    if constexpr (H >= 1) {
        const bool up = (lane & H) != 0;
#pragma unroll
        for (int i = 0; i < H; ++i) {
            /* Select values, not lvalues: `up ? v[i + H] : v[i]` selects an address, and an array
             * addressed by a lane-dependent pointer is demoted to scratch. */
            const float lo = v[i], hi = v[i + H];
            const float keep = up ? hi : lo;
            const float send = up ? lo : hi;
            v[i] = keep + shfl_xor(send, H);
        }
        reduce_scatter<H / 2, N>(v, lane);
    }
}

/* ------------------------------------------------------------------ the launch's arguments */
struct AttnArgs {
    const uint16_t*      q;      long long q_rs, q_hs;     /* elements */
    const unsigned char* kv;     long long kv_sb, kv_sh;   /* elements of the cache's dtype */
    const int*           bt;     long long bt_rs;
    const int*           seqused;
    const int*           cu;                               /* null: uniform q_len */
    const float*         kdes;
    const float*         vdes;
    uint16_t*            out;    long long o_rs, o_hs;
    float*               part;                             /* split-KV partials, or null */
    const float*         sinks;                            /* [query heads] or null */
    int*                 tinfo;                            /* [tile] = (first row, positions) */
    float qscale;                                          /* softmax scale * log2(e) */
    int   max_blocks, n_seq, q_len, tps, kv_heads, causal, window, n_split;
    int   shared_window;                                    /* all query rows use sequence anchor */
    /* 1: a non-causal window bounds the keys behind only (launch_attn's FWD) */
    int   ahead;
    /* A tree verify (attn_paged_qrow_tree): RadBatch::spec_parent, one a query row; read only by
     * the TREE instances. */
    const int*           tpar;
};

/* Tile `tile` of a ragged batch: its sequence and which BQ-position slice of that sequence's query
 * it is. Tiles are numbered sequence by sequence, ceil(q_len(s) / BQ) each, so this is a block-wide
 * prefix sum over cu_seqlens, NT sequences per pass. All threads get the same answer from LDS. */
template <int BQ, int NT = kThreads>
__device__ void find_tile(const int* cu, int n_seq, int tile, int* s_scan, int* s_res, int& s_out,
                          int& lt_out) {
    constexpr int NW = NT / 32;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    if (tid == 0) { s_res[0] = -1; s_res[1] = 0; }
    __syncthreads();
    int base = 0;
    for (int c0 = 0; c0 < n_seq; c0 += NT) {
        const int s = c0 + tid;
        int nt = 0;
        if (s < n_seq) {
            const int ql = cu[s + 1] - cu[s];
            nt = ql > 0 ? (ql + BQ - 1) / BQ : 0;
        }
        int x = nt;
        for (int d = 1; d < 32; d <<= 1) {
            const int y = shfl_up(x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) s_scan[wave] = x;
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < NW; ++w) {
            const int v = s_scan[w];
            before += (w < wave) ? v : 0;
            total += v;
        }
        const int incl = base + before + x;
        if (nt > 0 && tile >= incl - nt && tile < incl) { s_res[0] = s; s_res[1] = tile - (incl - nt); }
        base += total;
        __syncthreads();
        if (s_res[0] >= 0 || base > tile) break;
    }
    s_out = s_res[0];
    lt_out = s_res[1];
}

/* ------------------------------------------------------------------ the attention kernel */
/* The body; attn_kernel and its chained form r4dchain_attn_window (r4d_chain.h) wrap it below,
 * since its blocks return early. */
template <int HD, int GQA, int KV8, bool TREE = false>
__device__ __forceinline__ void attn_body(AttnArgs A) {
    constexpr int R = Geo<HD>::R, BK = Geo<HD>::BK, RW = R / kWaves, BQ = R / GQA;
    constexpr int NC = HD / 128;                     /* 4-element chunks a lane holds of a row */
    constexpr int ESZ = KV8 ? 1 : 2;
    constexpr int TAIL = KV8 == 2 ? 4 : 0;           /* the int8 cache's two f16 scales */
    constexpr int KEY_BYTES = 2 * HD * ESZ + TAIL;   /* one slot: K then V (then the scales) */
    constexpr int TILE_U4 = BK * KEY_BYTES / 16;
    /* 16-byte loads a thread, per tile. With the scales a tile is not whole threads: the last
     * round is short, and a block's loads are found by division. */
    constexpr int NPT = (TILE_U4 + kThreads - 1) / kThreads;
    constexpr int BLK_U4 = 16 * KEY_BYTES / 16;      /* one paged block of one kv head */
    constexpr int NB = BK / 16;                      /* paged blocks a key tile spans */
    static_assert(R % kWaves == 0 && R % GQA == 0, "rows must split over waves and heads");
    static_assert(TAIL != 0 || (TILE_U4 % kThreads == 0 && BLK_U4 % kThreads == 0),
                  "tile must split over threads");
    static_assert(BK == 32 || BK == 16, "the reduce-scatter is written for 16 or 32 keys");

    __shared__ uint4 s_kv[NPT * kThreads];
    __shared__ int s_scan[kWaves];
    __shared__ int s_res[2];

    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int tile = (int)blockIdx.x, kh = (int)blockIdx.y, z = (int)blockIdx.z;

    /* ---- which sequence, and which slice of its query */
    int s = -1, lt = 0;
    if (A.cu) {
        find_tile<BQ>(A.cu, A.n_seq, tile, s_scan, s_res, s, lt);
    } else {
        s = tile / A.tps;
        lt = tile - s * A.tps;
        if (s >= A.n_seq) s = -1;
    }
    int qlen = 0, tokb = 0, ctx = 0;
    if (s >= 0) {
        if (A.cu) { tokb = A.cu[s]; qlen = A.cu[s + 1] - tokb; }
        else      { tokb = s * A.q_len; qlen = A.q_len; }
        ctx = A.seqused[s];
    }
    const int pfirst = lt * BQ;
    const int npos = s >= 0 ? imax(0, imin(BQ, qlen - pfirst)) : 0;
    if (A.n_split > 1 && kh == 0 && z == 0 && tid == 0) {
        A.tinfo[2 * tile] = tokb + pfirst;
        A.tinfo[2 * tile + 1] = npos;
    }
    if (npos <= 0) return;                           /* uniform across the block */

    /* TREE: the sequence's first rows' parents staged once (a tree block is at most 64 rows), so
     * each row's walk below reads LDS rather than a chain of dependent global loads. */
    __shared__ int s_tpar[TREE ? 64 : 1];
    if constexpr (TREE) {
        if (A.tpar) {
            const int nst = imin(qlen, 64);
            for (int r = tid; r < nst; r += kThreads) s_tpar[r] = A.tpar[tokb + r];
            __syncthreads();
        }
    }

    const int tok0 = tokb + pfirst;
    const int qbase = ctx - qlen + pfirst;           /* absolute position of the tile's first query */
    const int plast = qbase + npos - 1;
    const int causal = A.causal, window = A.window;
    const int tbase = ctx - qlen;                    /* first query position, the block anchor */
    const int window_base = A.shared_window ? tbase : ((TREE && A.tpar) ? tbase : qbase);

    /* ---- the tile's key range, and this split's share of it */
    int hi = causal ? (A.shared_window ? tbase + 1 : plast + 1) : ctx;
    if (!causal && window > 0 && !A.shared_window && !A.ahead) hi = imin(hi, plast + window);
    hi = imin(imin(hi, ctx), A.max_blocks * 16);
    /* A tree row's window is measured from the block's first position plus its depth, which can
     * be below its own position; an anchor-window launch also uses the first row for every query. */
    const int lo = window > 0 ? imax(0, window_base - window + 1) : 0;
    const int lo_a = lo & ~(BK - 1);
    const int nkt = hi > lo_a ? (hi - lo_a + BK - 1) / BK : 0;
    const int per = (nkt + A.n_split - 1) / A.n_split;
    const int kt_lo = imin(nkt, z * per), kt_hi = imin(nkt, kt_lo + per);

    const float kds = A.kdes ? A.kdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float vds = A.vdes ? A.vdes[(long long)s * A.kv_heads + kh] : 1.0f;

    /* ---- this wave's query rows: row r is position r / GQA, head kh * GQA + r % GQA */
    float qf[RW][NC][4], o[RW][NC][4], m[RW], l[RW];
    int pos[RW];
    bool val[RW];
    /* TREE: each row's ancestors inside the block (a bit per row of the block, itself included),
     * whether it is a tree row at all, and the position its window is measured from. */
    uint64_t tm[RW];
    int wq[RW];
    bool tr[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        const int r = wave * RW + i, p = r / GQA, g = r - p * GQA;
        val[i] = p < npos;
        pos[i] = qbase + p;
        m[i] = -__builtin_inff();
        l[i] = 0.0f;
        tm[i] = 0;
        wq[i] = pos[i];
        tr[i] = false;
        if constexpr (TREE) {
            const int rel = pfirst + p;              /* the row within its sequence's rows */
            if (A.tpar && val[i] && rel < 64) {
                int pj = s_tpar[rel];
                if (pj != -2) {
                    uint64_t mk = 1ull << rel;
                    int dep = 0;
                    for (int at = rel; pj >= 0 && pj < at; ) {
                        mk |= 1ull << pj;
                        ++dep;
                        at = pj;
                        pj = s_tpar[pj];
                    }
                    tr[i] = true;
                    tm[i] = mk;
                    wq[i] = tbase + dep;
                }
            }
        }
        const uint16_t* qp = A.q + (long long)(tok0 + p) * A.q_rs + (long long)(kh * GQA + g) * A.q_hs;
        const float qs = A.qscale * kds;             /* k_descale folds into the query */
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (val[i]) bf16x4(*(const uint2*)(qp + c * 128 + lane * 4), f);
#pragma unroll
            for (int e = 0; e < 4; ++e) { qf[i][c][e] = f[e] * qs; o[i][c][e] = 0.0f; }
        }
    }

    /* ---- the key tiles, prefetched one ahead through registers */
    uint4 pre[NPT];
    bool pre_ok[NB];
    auto fetch = [&](int kt) __attribute__((always_inline)) {
        const int k0 = lo_a + kt * BK;
        const unsigned char* src[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int lb = (k0 >> 4) + b;
            int blk = -1;
            if (lb < A.max_blocks && (lb << 4) < hi) blk = A.bt[(long long)s * A.bt_rs + lb];
            pre_ok[b] = blk >= 0;
            /* Never null: selecting against null makes the pointer flat instead of global, and every
             * tile load a flat_load. A missing block points at the cache base; its value is
             * replaced by zeros below. */
            src[b] = A.kv + ((long long)(blk >= 0 ? blk : 0) * A.kv_sb + (long long)kh * A.kv_sh) * ESZ;
        }
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            if constexpr (TAIL) {
                const int gi = tid + i * kThreads;
                const int b = gi >= BLK_U4 ? NB - 1 : 0;
                const int off = gi - b * BLK_U4;
                pre[i] = make_uint4(0u, 0u, 0u, 0u);
                if (gi < TILE_U4 && pre_ok[b]) pre[i] = ((const uint4*)src[b])[off];
            } else {
                const int b = (i * kThreads) / BLK_U4;   /* constant: BLK_U4 is a multiple of 256 */
                const int off = tid + i * kThreads - b * BLK_U4;
                if (pre_ok[b]) pre[i] = ((const uint4*)src[b])[off];
                else           pre[i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };

    const unsigned char* sb = (const unsigned char*)s_kv;
    if (kt_lo < kt_hi) fetch(kt_lo);
    for (int kt = kt_lo; kt < kt_hi; ++kt) {
        __syncthreads();                             /* the previous tile is consumed */
#pragma unroll
        for (int i = 0; i < NPT; ++i) s_kv[tid + i * kThreads] = pre[i];
        const bool ok0 = pre_ok[0], ok1 = pre_ok[NB - 1];   /* values: see reduce_scatter */
        __syncthreads();
        const int k0 = lo_a + kt * BK;
        if (kt + 1 < kt_hi) fetch(kt + 1);

        const int jl = lane & (BK - 1);              /* the key this lane scores */
        const int kk = k0 + jl;
        const bool kok = (jl < 16 ? ok0 : ok1) && kk >= lo && kk < hi;

        auto row_step = [&](auto ic) __attribute__((always_inline)) {
            constexpr int i = decltype(ic)::value;
            if (!val[i]) return;                     /* uniform: one row, all lanes */
            float sc[BK];
#pragma unroll
            for (int j = 0; j < BK; ++j) {
                const unsigned char* row = sb + j * KEY_BYTES;
                float acc = 0.0f;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    float k[4];
                    lds4<KV8>(row, c * 128 + lane * 4, k);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc = __builtin_fmaf(qf[i][c][e], k[e], acc);
                }
                if constexpr (KV8 == 2) acc *= f16_at(row + 2 * HD);
                sc[j] = acc;
            }
            if constexpr (BK == 32) {
                reduce_scatter<16, BK>(sc, lane);
            } else {
#pragma unroll
                for (int j = 0; j < BK; ++j) sc[j] += shfl_xor(sc[j], 16);
                reduce_scatter<8, BK>(sc, lane);
            }
            const int pq = pos[i];
            bool vis = kok && (causal ? kk <= (A.shared_window ? tbase : pq) : true);
            if constexpr (TREE)
                if (tr[i] && vis && kk >= tbase) vis = ((tm[i] >> (kk - tbase)) & 1ull) != 0;
            const int wpq = A.shared_window ? tbase : (TREE ? wq[i] : pq);
            if (A.shared_window)
                vis = vis && (wpq - kk < window);
            else if (window > 0)
                vis = vis && (wpq - kk < window) && (causal || A.ahead || kk - wpq < window);
            const float sv = vis ? sc[0] : -__builtin_inff();
            const float mn = __builtin_fmaxf(m[i], wave_max(sv));
            if (mn == -__builtin_inff()) return;     /* nothing visible yet: uniform */
            const float corr = ex2(m[i] - mn);
            const float p = vis ? ex2(sv - mn) : 0.0f;
            l[i] = l[i] * corr + wave_sum((BK == 16 && lane >= 16) ? 0.0f : p);
#pragma unroll
            for (int c = 0; c < NC; ++c)
#pragma unroll
                for (int e = 0; e < 4; ++e) o[i][c][e] *= corr;
#pragma unroll
            for (int j = 0; j < BK; ++j) {
                const float pj = readlane(p, j);     /* uniform; a masked key's V is never read */
                if (pj != 0.0f) {
                    const unsigned char* row = sb + j * KEY_BYTES;
                    const float pv = KV8 == 2 ? pj * f16_at(row + 2 * HD + 2) : pj;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        float v[4];
                        lds4<KV8>(row, HD + c * 128 + lane * 4, v);
#pragma unroll
                        for (int e = 0; e < 4; ++e) o[i][c][e] = __builtin_fmaf(pv, v[e], o[i][c][e]);
                    }
                }
            }
            m[i] = mn;
        };
        StaticFor<0, RW>::run(row_step);
    }

    /* ---- out: normalised bf16 rows, or this split's partials */
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        if (!val[i]) continue;
        const int r = wave * RW + i, p = r / GQA, g = r - p * GQA;
        if (A.n_split == 1) {
            /* A sink joins the row's max and sum with no value (log2 domain, as m and l). */
            if (A.sinks && m[i] != -__builtin_inff()) {
                const float s2 = A.sinks[kh * GQA + g] * 1.4426950408889634f;
                if (s2 > m[i]) {
                    const float corr = ex2(m[i] - s2);
#pragma unroll
                    for (int c = 0; c < NC; ++c)
#pragma unroll
                        for (int e = 0; e < 4; ++e) o[i][c][e] *= corr;
                    l[i] = l[i] * corr + 1.0f;
                    m[i] = s2;
                } else {
                    l[i] += ex2(s2 - m[i]);
                }
            }
            const float inv = l[i] > 0.0f ? vds / l[i] : 0.0f;
            uint16_t* op = A.out + (long long)(tok0 + p) * A.o_rs + (long long)(kh * GQA + g) * A.o_hs;
#pragma unroll
            for (int c = 0; c < NC; ++c)
                *(uint2*)(op + c * 128 + lane * 4) =
                    make_uint2(pack2(o[i][c][0] * inv, o[i][c][1] * inv),
                               pack2(o[i][c][2] * inv, o[i][c][3] * inv));
        } else {
            const long long slot = (((long long)tile * A.kv_heads + kh) * A.n_split + z) * R + r;
            float* pp = A.part + slot * (HD + 4);
            if (lane == 0) { pp[HD] = m[i]; pp[HD + 1] = l[i]; }
            /* no key visible to this row in this split: the merge weighs it 0 and never reads its
             * accumulator, so it is not written (uniform: m is the wave's) */
            if (m[i] == -__builtin_inff()) continue;
#pragma unroll
            for (int c = 0; c < NC; ++c)
                *(float4*)(pp + c * 128 + lane * 4) =
                    make_float4(o[i][c][0] * vds, o[i][c][1] * vds, o[i][c][2] * vds, o[i][c][3] * vds);
        }
    }
}

/* ================================================================== the WMMA prefill kernel
 *
 * The scalar kernel reaches ~8 TFLOP/s on a long prefill, where global-layer attention is
 * quadratic and 46% of a 48k-token prefill. This kernel uses RDNA4's 16x16x16 WMMA instead. It is
 * chosen only for prefill-shaped steps (a sequence with >= kWmmaMinQ query rows) and computes the
 * same function.
 *
 * Layouts (wave32 v_wmma_f32_16x16x16_bf16): A[M][K] gives lane L row M = L % 16 and
 * K = 8 * (L / 16) .. + 7; B[K][N] gives lane L column N = L % 16, same K; the f32 result gives lane
 * L column N = L % 16, rows M = 8 * (L / 16) .. + 7. So the kernel computes S^T = K * Q^T: each lane
 * holds eight keys of one query row (row L % 16, keys 8h .. 8h + 7), the other eight are in lane
 * L ^ 16 (softmax statistics are one shuffle), and those values as f16 are the B operand of
 * O^T = V^T * P^T. O^T's lane L also holds only row L % 16, so the rescale is a per-lane scalar.
 *
 * Head split: O^T for 16 rows of a 512-dim head is 256 f32 per lane, the whole register file, so a
 * wave owns 16 rows by a 128-dim slice (kDQ): Q^T in 32 VGPRs, O^T in 64. Slices compute partial
 * S^T over their own dims; partials meet in LDS and every slice sums them in the same order, so all
 * see bit-identical scores. 16 waves: h512 is 4 row groups x 4 slices (64 rows), h256 8 x 2 (128).
 *
 * Tile = one paged block, 16 keys. K goes to LDS as [key][dim] bf16 with a 16-byte row pad so
 * A-fragment reads hit 32 banks. V goes transposed ([dim][key], 48-byte rows) because P.V reduces
 * over keys; two keys are packed per dword with the dword index rotated per lane group so a wave's
 * 32 writes hit 32 banks. An fp8 cache is widened on load (exact). Keys outside the range or in a
 * missing block are written as zeros, since 0 * NaN from a recycled slot is still NaN.
 *
 * P and V in f16, Q and K in bf16: bf16 P leaves up to 2^-9 |v| error per output (up to 0.25% of the
 * V range against f64, 4x the scalar kernel), while f16 runs at the same rate with 11 bits. bf16 V
 * is exact in f16 within [2^-14, 65504] (clamped beyond, NaN kept); fp8 V is always exact. The
 * denominator sums P as rounded, so a row is a true weighted mean of V.
 */
typedef short win_v8s __attribute__((ext_vector_type(8)));   /* 8 bf16, as the builtin takes them */
typedef float win_v8f __attribute__((ext_vector_type(8)));
typedef _Float16 win_v8h __attribute__((ext_vector_type(8)));

constexpr int kWThreads = 512;
constexpr int kWWaves = kWThreads / 32;
constexpr int kDQ = 128;            /* dims of a wave's slice of the head */
constexpr int kWmmaMinQ = 64;       /* a launch whose longest query is shorter stays scalar */

template <int HD> struct WGeo {
    static constexpr int NS = HD / kDQ;           /* slices: waves sharing one 16-row group */
    static constexpr int RG = kWWaves / NS;       /* row groups a workgroup */
    static constexpr int R = RG * 16;             /* rows a workgroup */
    static constexpr int KS = HD * 2 + 16;        /* bytes a key row of the K tile */
    static constexpr int VS = 48;                 /* bytes a dim row of the V^T tile: 16 keys + pad */
};

__device__ __forceinline__ win_v8f wmma_bf16(win_v8s a, win_v8s b, win_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ win_v8f wmma_f16(win_v8h a, win_v8h b, win_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
/* f32 -> f16 bits, round to nearest even; a finite value past f16's range saturates, NaN stays */
__device__ __forceinline__ unsigned f16_bits(float f) {
    if (f == f) f = __builtin_fminf(__builtin_fmaxf(f, -65504.0f), 65504.0f);
    return (unsigned)__builtin_bit_cast(unsigned short, (_Float16)f);
}
/* Two f32 values exact in f16 as an f16 pair, `a` low: v_cvt_pkrtz on gfx12, f16_bits elsewhere
 * (the host emulator), the same bits either way since neither rounds an exact value. */
__device__ __forceinline__ unsigned f16x2_exact(float a, float b) {
#if defined(__GFX12__)
    return __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_pkrtz(a, b));
#else
    return f16_bits(a) | f16_bits(b) << 16;
#endif
}
__device__ __forceinline__ void bf16x8_f16(const uint4 w, unsigned (&h)[8]) {
    const unsigned in[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        h[2 * q] = f16_bits(__uint_as_float(in[q] << 16));
        h[2 * q + 1] = f16_bits(__uint_as_float(in[q] & 0xffff0000u));
    }
}
/* fp8 e4m3 or int8 -> bf16 bits (the K tile), exact: the f32 a byte widens to has nothing below
 * bit 16 */
template <int KV8>
__device__ __forceinline__ void kv8x16_bf16(const uint4 w, unsigned (&h)[16]) {
    const unsigned in[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        float f[4];
        kv8x4<KV8>(in[q], f);
#pragma unroll
        for (int e = 0; e < 4; ++e) h[q * 4 + e] = __float_as_uint(f[e]) >> 16;
    }
}

template <int HD, int GQA, int KV8>
__global__ __launch_bounds__(kWThreads) void attn_wmma_kernel(AttnArgs A) {
    using G = WGeo<HD>;
    constexpr int NS = G::NS, R = G::R, BQ = R / GQA, KS = G::KS, VS = G::VS;
    constexpr int ESZ = KV8 ? 1 : 2;
    /* bytes of one key's slot: K then V, and at KV8 2 the token's two f16 scales, which make a
     * slot's 16-byte loads start on a 4-byte boundary */
    constexpr int SLOT = 2 * HD * ESZ + (KV8 == 2 ? 4 : 0);
    constexpr int CW = 16 / ESZ;                  /* cache elements one 16-byte load carries */
    constexpr int NCH = HD / CW;                  /* such chunks a head */
    constexpr int NKI = 16 * NCH;                 /* K items a tile: (key, chunk) */
    constexpr int NVI = 8 * NCH;                  /* V items a tile: (key pair, chunk) */
    constexpr int KPT = (NKI + kWThreads - 1) / kWThreads;
    constexpr int VPT = (NVI + kWThreads - 1) / kWThreads;
    static_assert(R % GQA == 0 && NS * G::RG == kWWaves, "rows must split over heads and waves");
    static_assert(KS % 16 == 0 && VS % 16 == 0, "16-byte LDS rows");

    __shared__ uint4 s_k[16 * KS / 16];
    __shared__ uint4 s_vt[HD * VS / 16];
    __shared__ float4 s_x[kWWaves * 32 * 2];
    __shared__ unsigned s_sc[KV8 == 2 ? 16 : 1];  /* a key's scales: K's f16 low, V's high */
    __shared__ int s_scan[kWWaves];
    __shared__ int s_res[2];

    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int rg = wave / NS, sl = wave - rg * NS;
    const int tile = (int)blockIdx.x, kh = (int)blockIdx.y;

    /* ---- which sequence, and which slice of its query: as the scalar kernel, at this BQ */
    int s = -1, lt = 0;
    if (A.cu) {
        find_tile<BQ, kWThreads>(A.cu, A.n_seq, tile, s_scan, s_res, s, lt);
    } else {
        s = tile / A.tps;
        lt = tile - s * A.tps;
        if (s >= A.n_seq) s = -1;
    }
    int qlen = 0, tokb = 0, ctx = 0;
    if (s >= 0) {
        if (A.cu) { tokb = A.cu[s]; qlen = A.cu[s + 1] - tokb; }
        else      { tokb = s * A.q_len; qlen = A.q_len; }
        ctx = A.seqused[s];
    }
    const int pfirst = lt * BQ;
    const int npos = s >= 0 ? imax(0, imin(BQ, qlen - pfirst)) : 0;
    if (npos <= 0) return;                           /* uniform across the block */

    const int tok0 = tokb + pfirst;
    const int qbase = ctx - qlen + pfirst;
    const int plast = qbase + npos - 1;
    const int causal = A.causal, window = A.window;
    int hi = causal ? plast + 1 : ctx;
    if (!causal && window > 0 && !A.ahead) hi = imin(hi, plast + window);
    hi = imin(imin(hi, ctx), A.max_blocks * 16);
    const int lo = window > 0 ? imax(0, qbase - window + 1) : 0;
    const int lo_a = lo & ~15;
    const int nkt = hi > lo_a ? (hi - lo_a + 15) / 16 : 0;

    const float kds = A.kdes ? A.kdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float vds = A.vdes ? A.vdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float sscale = A.qscale * kds;             /* log2-domain softmax scale, k_descale in */

    /* ---- this lane's query row: column n of every S^T and O^T tile */
    const int n = lane & 15, h = lane >> 4;
    const int r = rg * 16 + n, p = r / GQA, g = r - p * GQA;
    const bool rval = p < npos;
    const int pos = qbase + p;

    win_v8s qf[kDQ / 16];
    {
        const uint16_t* qp = A.q + (long long)(tok0 + (rval ? p : 0)) * A.q_rs +
                             (long long)(kh * GQA + g) * A.q_hs + sl * kDQ + 8 * h;
#pragma unroll
        for (int kc = 0; kc < kDQ / 16; ++kc) {
            const uint4 w = rval ? *(const uint4*)(qp + kc * 16) : make_uint4(0u, 0u, 0u, 0u);
            qf[kc] = __builtin_bit_cast(win_v8s, w);
        }
    }
    win_v8f oacc[kDQ / 16];
#pragma unroll
    for (int mc = 0; mc < kDQ / 16; ++mc)
#pragma unroll
        for (int e = 0; e < 8; ++e) oacc[mc][e] = 0.0f;
    /* The learned sink is a key with no value: it joins the running max and the denominator
     * (log2 domain). Keys are split across lane and lane^16, so only lane h=0 seeds the sink's
     * weight of one; the final lt2 adds the two lanes. */
    const float sink = A.sinks ? A.sinks[kh * GQA + g] * 1.4426950408889634f
                               : -__builtin_inff();
    float m = sink, l = (A.sinks && h == 0) ? 1.0f : 0.0f;

    /* ---- the key tiles, one paged block each, prefetched one ahead through registers */
    uint4 pk[KPT], pv[VPT][2];
    unsigned psc = 0u;
    bool pre_ok = false;
    auto fetch = [&](int kt) __attribute__((always_inline)) {
        const int k0 = lo_a + kt * 16, lb = k0 >> 4;
        int blk = -1;
        if (lb < A.max_blocks && k0 < hi) blk = A.bt[(long long)s * A.bt_rs + lb];
        pre_ok = blk >= 0;
        /* never a null: see the scalar kernel's fetch */
        const unsigned char* base =
            A.kv + ((long long)(blk >= 0 ? blk : 0) * A.kv_sb + (long long)kh * A.kv_sh) * ESZ;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int idx = tid + j * kWThreads;
            pk[j] = make_uint4(0u, 0u, 0u, 0u);
            if (pre_ok && idx < NKI) {
                const int key = idx / NCH, ch = idx - key * NCH;
                pk[j] = *(const uint4*)(base + key * SLOT + ch * 16);
            }
        }
        if constexpr (KV8 == 2) {
            psc = 0u;
            if (pre_ok && tid < 16) psc = *(const unsigned*)(base + tid * SLOT + 2 * HD);
        }
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = tid + j * kWThreads;
            pv[j][0] = pv[j][1] = make_uint4(0u, 0u, 0u, 0u);
            if (pre_ok && idx < NVI) {
                const int kp = idx & 7, ch = idx >> 3;
                const unsigned char* v0 = base + (2 * kp) * SLOT + HD * ESZ + ch * 16;
                pv[j][0] = *(const uint4*)v0;
                pv[j][1] = *(const uint4*)(v0 + SLOT);
            }
        }
    };

    unsigned char* kb = (unsigned char*)s_k;
    unsigned char* vb = (unsigned char*)s_vt;
    if (nkt > 0) fetch(0);
    for (int kt = 0; kt < nkt; ++kt) {
        const int k0 = lo_a + kt * 16;
        const bool tok = pre_ok;                     /* this tile's block is present */
        __syncthreads();                             /* the previous tile's LDS is consumed */

        /* ---- K: [key][dim] as bf16 */
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int idx = tid + j * kWThreads;
            if (idx < NKI) {
                const int key = idx / NCH, ch = idx - key * NCH;
                const int kk = k0 + key;
                const bool keep = tok && kk >= lo && kk < hi;
                const uint4 w = keep ? pk[j] : make_uint4(0u, 0u, 0u, 0u);
                if constexpr (KV8) {
                    unsigned hb[16];
                    kv8x16_bf16<KV8>(w, hb);
                    uint4* dst = (uint4*)(kb + key * KS + ch * 32);
                    dst[0] = make_uint4(hb[0] | hb[1] << 16, hb[2] | hb[3] << 16, hb[4] | hb[5] << 16, hb[6] | hb[7] << 16);
                    dst[1] = make_uint4(hb[8] | hb[9] << 16, hb[10] | hb[11] << 16, hb[12] | hb[13] << 16, hb[14] | hb[15] << 16);
                } else {
                    *(uint4*)(kb + key * KS + ch * 16) = w;
                }
            }
        }
        if constexpr (KV8 == 2) {
            if (tid < 16) s_sc[tid] = psc;
        }
        /* ---- V: [dim][key] as f16, two keys a dword; the row order rotated by 2 per 8-lane
         * group, so the four groups of a wave write four different rows mod 8 -- 32 banks */
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = tid + j * kWThreads;
            if (idx < NVI) {
                const int kp = idx & 7, ch = idx >> 3;
                const int ka = k0 + 2 * kp, kb1 = ka + 1;
                const uint4 wa = (tok && ka >= lo && ka < hi) ? pv[j][0] : make_uint4(0u, 0u, 0u, 0u);
                const uint4 wb = (tok && kb1 >= lo && kb1 < hi) ? pv[j][1] : make_uint4(0u, 0u, 0u, 0u);
                unsigned d[CW];                      /* f16 pairs: key 2kp low, 2kp + 1 high */
                if constexpr (KV8) {
                    /* every e4m3 value is exact in f16 (|v| <= 448, subnormals 2^-9 up), and so
                     * is an int8 code, so the pack's round-toward-zero and f16_bits' clamp leave
                     * the same bits */
                    const unsigned ia[4] = {wa.x, wa.y, wa.z, wa.w}, ib[4] = {wb.x, wb.y, wb.z, wb.w};
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float fa[4], fb[4];
                        kv8x4<KV8>(ia[q], fa);
                        kv8x4<KV8>(ib[q], fb);
#pragma unroll
                        for (int e = 0; e < 4; ++e) d[q * 4 + e] = f16x2_exact(fa[e], fb[e]);
                    }
                } else {
                    unsigned ha[8], hb2[8];
                    bf16x8_f16(wa, ha);
                    bf16x8_f16(wb, hb2);
#pragma unroll
                    for (int e = 0; e < CW; ++e) d[e] = ha[e] | hb2[e] << 16;
                }
                const int rot = 2 * ((lane >> 3) & 3);
                unsigned* col = (unsigned*)(vb + (ch * CW) * VS) + kp;
#pragma unroll
                for (int t = 0; t < CW; ++t) {
                    /* element (t + rot) % CW, chosen by value: rot is per lane */
                    const unsigned v0 = d[t], v2 = d[(t + 2) % CW], v4 = d[(t + 4) % CW], v6 = d[(t + 6) % CW];
                    const unsigned v = rot == 0 ? v0 : rot == 2 ? v2 : rot == 4 ? v4 : v6;
                    const int e = (t + rot) % CW;
                    col[e * (VS / 4)] = v;
                }
            }
        }
        __syncthreads();
        if (kt + 1 < nkt) fetch(kt + 1);

        /* ---- S^T partial over this wave's slice: A = K rows (key n), B = Q^T (row n) */
        win_v8f sacc;
#pragma unroll
        for (int e = 0; e < 8; ++e) sacc[e] = 0.0f;
        {
            /* One accumulator: a second hides the chain latency but costs 12 VGPRs; at 187 VGPRs
             * two 16-wave workgroups fit a WGP (8 waves per SIMD), at 200 only one, and the extra
             * waves hide latency better. */
            const unsigned char* ka = kb + n * KS + (sl * kDQ + 8 * h) * 2;
#pragma unroll
            for (int kc = 0; kc < kDQ / 16; ++kc)
                sacc = wmma_bf16(__builtin_bit_cast(win_v8s, *(const uint4*)(ka + kc * 32)), qf[kc], sacc);
        }
        /* ---- the slices' partials meet: every slice sums them in slice order */
        s_x[(wave * 2 + 0) * 32 + lane] = make_float4(sacc[0], sacc[1], sacc[2], sacc[3]);
        s_x[(wave * 2 + 1) * 32 + lane] = make_float4(sacc[4], sacc[5], sacc[6], sacc[7]);
        __syncthreads();
        float tot[8];
#pragma unroll
        for (int j = 0; j < NS; ++j) {
            float v[8];
            if (j == sl) {                           /* uniform: the wave's own, as written */
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = sacc[e];
            } else {
                const float4 a = s_x[((rg * NS + j) * 2 + 0) * 32 + lane];
                const float4 b = s_x[((rg * NS + j) * 2 + 1) * 32 + lane];
                v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
                v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) tot[e] = j == 0 ? v[e] : tot[e] + v[e];
        }

        /* The int8 cache: a key's dot is over its codes, times its K scale. */
        if constexpr (KV8 == 2) {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                tot[e] *= (float)__builtin_bit_cast(_Float16, (unsigned short)(s_sc[8 * h + e] & 0xffffu));
        }
        /* ---- online softmax, row n: eight keys here, eight in lane ^ 16 */
        float sv[8], mx = -__builtin_inff();
        /* A tile wholly inside [lo, hi) and, causal, at or before the block's first position
         * (every row's pos >= qbase) is visible to every valid row: the same scores, unmasked. */
        const bool interior = window <= 0 && k0 >= lo && k0 + 16 <= hi && (!causal || k0 + 15 <= qbase);
        if (interior) {
            const bool vis = rval && tok;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sv[e] = vis ? tot[e] * sscale : -__builtin_inff();
                mx = __builtin_fmaxf(mx, sv[e]);
            }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kk = k0 + 8 * h + e;
                bool vis = rval && tok && kk >= lo && kk < hi;
                if (causal) vis = vis && kk <= pos;
                if (window > 0) vis = vis && (pos - kk < window) && (causal || A.ahead || kk - pos < window);
                sv[e] = vis ? tot[e] * sscale : -__builtin_inff();
                mx = __builtin_fmaxf(mx, sv[e]);
            }
        }
        mx = __builtin_fmaxf(mx, shfl_xor(mx, 16));
        const float mn = __builtin_fmaxf(m, mx);
        const float corr = mn == -__builtin_inff() ? 1.0f : ex2(m - mn);
        /* P in f16, and the denominator summed from P as rounded (see the head of this kernel) */
        float ps = 0.0f;
        win_v8h pf;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float pe = sv[e] == -__builtin_inff() ? 0.0f : ex2(sv[e] - mn);
            pf[e] = (_Float16)pe;                    /* in [0, 1]: no saturation to worry about */
            ps += (float)pf[e];
        }
        l = l * corr + ps;
        m = mn;
        /* The int8 cache: a key's row of V codes enters at its probability times its V scale. */
        if constexpr (KV8 == 2) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                /* a masked key's scale is whatever its slot holds: its zero stays zero */
                const float pe = (float)pf[e];
                const float vs = (float)__builtin_bit_cast(_Float16, (unsigned short)(s_sc[8 * h + e] >> 16));
                pf[e] = pe != 0.0f ? (_Float16)(pe * vs) : (_Float16)0.0f;
            }
        }

        /* ---- O^T += V^T * P^T in f16: A = V^T rows (dim n of the chunk), keys 8h .. 8h + 7 */
        const unsigned char* va = vb + (sl * kDQ + n) * VS + h * 16;
        if constexpr (KV8) {
            /* x * 1 = x, so a wave whose rows all kept their max skips the rescale. With an fp8
             * cache that is ~4% at head_dim 512; with bf16 the split loop measured ~10% slower,
             * so it keeps the fused one. */
            if (__builtin_amdgcn_ballot_w32(corr != 1.0f) != 0) {
#pragma unroll
                for (int mc = 0; mc < kDQ / 16; ++mc) oacc[mc] *= corr;
            }
#pragma unroll
            for (int mc = 0; mc < kDQ / 16; ++mc)
                oacc[mc] = wmma_f16(__builtin_bit_cast(win_v8h, *(const uint4*)(va + mc * 16 * VS)), pf, oacc[mc]);
        } else {
#pragma unroll
            for (int mc = 0; mc < kDQ / 16; ++mc) {
                oacc[mc] *= corr;
                oacc[mc] = wmma_f16(__builtin_bit_cast(win_v8h, *(const uint4*)(va + mc * 16 * VS)), pf, oacc[mc]);
            }
        }
    }

    /* ---- out: the row's denominator is its two lanes' partial sums */
    const float lt2 = l + shfl_xor(l, 16);
    if (rval) {
        const float inv = lt2 > 0.0f ? vds / lt2 : 0.0f;
        uint16_t* op = A.out + (long long)(tok0 + p) * A.o_rs + (long long)(kh * GQA + g) * A.o_hs +
                       sl * kDQ + 8 * h;
#pragma unroll
        for (int mc = 0; mc < kDQ / 16; ++mc)
            *(uint4*)(op + mc * 16) =
                make_uint4(pack2(oacc[mc][0] * inv, oacc[mc][1] * inv), pack2(oacc[mc][2] * inv, oacc[mc][3] * inv),
                           pack2(oacc[mc][4] * inv, oacc[mc][5] * inv), pack2(oacc[mc][6] * inv, oacc[mc][7] * inv));
    }
}

/* ------------------------------------------------------------------ the split-KV merge
 *
 * One wave per row, splits across lanes: lane z holds split z's (max, sum) and z + 32's
 * (kMaxSplit is 64), so the row max and denominator are one wave reduction each. Then one sweep
 * over the splits that saw a key, four at a time, each lane accumulating its 4 * NC dims. */
template <int HD, int GQA>
__device__ __forceinline__ void merge_body(AttnArgs A) {
    constexpr int R = Geo<HD>::R, NC = HD / 128;
    static_assert(kMaxSplit <= 64, "two splits a lane");
    const int tile = (int)blockIdx.x, kh = (int)blockIdx.y;
    const int lane = lane_id(), wave = (int)(threadIdx.x >> 5);
    const int tok0 = A.tinfo[2 * tile], npos = A.tinfo[2 * tile + 1];
    const int ns = A.n_split;
    const float ninf = -__builtin_inff();
    for (int r = wave; r < npos * GQA; r += kWaves) {
        const int p = r / GQA, g = r - p * GQA;
        const long long base = ((long long)tile * A.kv_heads + kh) * ns * R + r;
        uint16_t* op = A.out + (long long)(tok0 + p) * A.o_rs + (long long)(kh * GQA + g) * A.o_hs;
        float m0 = ninf, m1 = ninf, l0 = 0.0f, l1 = 0.0f;
        if (lane < ns) {
            const float* pp = A.part + (base + (long long)lane * R) * (HD + 4);
            m0 = pp[HD]; l0 = pp[HD + 1];
        }
        if (lane + 32 < ns) {
            const float* pp = A.part + (base + (long long)(lane + 32) * R) * (HD + 4);
            m1 = pp[HD]; l1 = pp[HD + 1];
        }
        const float mt = wave_max(__builtin_fmaxf(m0, m1));
        if (mt == ninf) {                            /* no key visible to the row: a zero row */
#pragma unroll
            for (int c = 0; c < NC; ++c) *(uint2*)(op + c * 128 + lane * 4) = make_uint2(0u, 0u);
            continue;
        }
        /* A sink joins the max and the sum with no value (log2 domain). */
        const float s2 = A.sinks ? A.sinks[kh * GQA + g] * 1.4426950408889634f : ninf;
        const float ms = __builtin_fmaxf(mt, s2);
        const float w0 = ex2(m0 - ms), w1 = ex2(m1 - ms);    /* an empty split: ex2(-inf) = 0 */
        const float lt = wave_sum(__builtin_fmaf(l0, w0, l1 * w1)) + (A.sinks ? ex2(s2 - ms) : 0.0f);
        const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
        float acc[NC][4];
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[c][e] = 0.0f;
        /* Splits with weight, as a wave-uniform 64-bit set, taken four at a time so 4 * NC loads
         * are in flight before the first FMA. A short group repeats its first split at weight 0
         * (a written split, so 0 * finite). An empty split's accumulator is never read. */
        uint64_t live = (uint64_t)__builtin_amdgcn_ballot_w32(w0 != 0.0f) |
                        ((uint64_t)__builtin_amdgcn_ballot_w32(w1 != 0.0f) << 32);
        /* Group g + 1's loads go out before group g's FMAs, so a group's wait overlaps the one
         * before it instead of following it; the FMAs keep their order, so the bytes do too. */
        auto take = [&](int (&zi)[4], float (&wj)[4]) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool have = live != 0;
                zi[j] = have ? __builtin_ctzll(live) : zi[0];
                live &= have ? live - 1 : live;
                wj[j] = readlane(zi[j] < 32 ? w0 : w1, zi[j] & 31);
                if (!have) wj[j] = 0.0f;
            }
        };
        auto load = [&](const int (&zi)[4], float4 (&v)[4][NC]) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float* pp = A.part + (base + (long long)zi[j] * R) * (HD + 4);
#pragma unroll
                for (int c = 0; c < NC; ++c) v[j][c] = *(const float4*)(pp + c * 128 + lane * 4);
            }
        };
        auto fold = [&](const float (&wj)[4], const float4 (&v)[4][NC]) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    acc[c][0] = __builtin_fmaf(wj[j], v[j][c].x, acc[c][0]);
                    acc[c][1] = __builtin_fmaf(wj[j], v[j][c].y, acc[c][1]);
                    acc[c][2] = __builtin_fmaf(wj[j], v[j][c].z, acc[c][2]);
                    acc[c][3] = __builtin_fmaf(wj[j], v[j][c].w, acc[c][3]);
                }
        };
        if (live) {
            int za[4], zb[4];
            float wa[4], wb[4];
            float4 va[4][NC], vb[4][NC];
            take(za, wa);
            load(za, va);
            while (true) {
                const bool more_b = live != 0;
                if (more_b) { take(zb, wb); load(zb, vb); }
                fold(wa, va);
                if (!more_b) break;
                const bool more_a = live != 0;
                if (more_a) { take(za, wa); load(za, va); }
                fold(wb, vb);
                if (!more_a) break;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            *(uint2*)(op + c * 128 + lane * 4) =
                make_uint2(pack2(acc[c][0] * inv, acc[c][1] * inv), pack2(acc[c][2] * inv, acc[c][3] * inv));
    }
}

/* ------------------------------------------------------------------ host: the launch adapter */

/* q and out as [rows, heads, head_dim]: rank >= 3 (leading axes dense), rank 2
 * [rows, heads * head_dim] with its own row pitch (a column slice of a fused projection), or flat.
 * The last axis must be dense. */
static bool head_view(const RadTensor* t, int64_t heads, int64_t hd, int64_t* rows, int64_t* rs,
                      int64_t* hs) {
    if (t->rank >= 3) {
        const uint32_t k = t->rank;
        if (t->shape[k - 1] != hd || t->shape[k - 2] != heads || t->stride[k - 1] != 1) return false;
        int64_t n = 1;
        for (uint32_t i = 0; i + 2 < k; ++i) n *= t->shape[i];
        for (uint32_t i = 0; i + 3 < k; ++i)
            if (t->shape[i] > 1 && t->stride[i] != t->stride[i + 1] * t->shape[i + 1]) return false;
        *rows = n; *rs = t->stride[k - 3]; *hs = t->stride[k - 2];
        return true;
    }
    if (t->rank == 2) {
        if (t->shape[1] != heads * hd || t->stride[1] != 1) return false;
        *rows = t->shape[0]; *rs = t->stride[0]; *hs = hd;
        return true;
    }
    if (t->rank == 1) {
        if (t->shape[0] % (heads * hd)) return false;
        *rows = t->shape[0] / (heads * hd); *rs = heads * hd; *hs = hd;
        return true;
    }
    return false;
}

template <int HD>
static constexpr int64_t part_bytes() {
    return (int64_t)kPartBudget * Geo<HD>::R * (HD + 4) * (int64_t)sizeof(float);
}
template <int HD>
static constexpr int64_t scratch_need() {
    return part_bytes<HD>() + (int64_t)kTinfoMax * 2 * (int64_t)sizeof(int);
}

/* The plain kernels and their chained forms (r4dchain*, RadKernelChainFn): the chained ones wait
 * for the kernels before them on the queue in their prologue and count themselves in at the end. */
template <int HD, int GQA, int KV8, bool TREE = false>
__global__ __launch_bounds__(kThreads) void attn_kernel(AttnArgs A) {
    attn_body<HD, GQA, KV8, TREE>(A);
}
template <int HD, int GQA, int KV8, bool TREE = false>
__global__ __launch_bounds__(kThreads) void r4dchain_attn_window(AttnArgs A) {
    r4d_chain_wait();
    attn_body<HD, GQA, KV8, TREE>(A);
    r4d_chain_done();
}
template <int HD, int GQA>
__global__ __launch_bounds__(kThreads) void merge_kernel(AttnArgs A) { merge_body<HD, GQA>(A); }
template <int HD, int GQA>
__global__ __launch_bounds__(kThreads) void r4dchain_merge_window(AttnArgs A) {
    r4d_chain_wait();
    merge_body<HD, GQA>(A);
    r4d_chain_done();
}

template <int HD, int GQA, int KV8, bool TREE = false, bool SHARED = false, bool FWD = false>
static int launch_attn(const RadArgs* a, RadStream stream, bool sink = false) {
    constexpr int R = Geo<HD>::R, BK = Geo<HD>::BK, BQ = R / GQA, ESZ = KV8 ? 1 : 2;
    /* A slot's elements: K then V, and for the int8 cache its four bytes of scales
     * (abi/rad_builder.h, rad_kv_row_elems). */
    constexpr int SLOT_E = 2 * HD + (KV8 == 2 ? 4 : 0);
    if (!a || a->n_t < (TREE || sink ? 9 : 8)) return RAD_E_INVAL;
    /* attn_paged_sink keeps its output where the tree operand would be: never both. */
    if (TREE && sink) return RAD_E_INVAL;
    /* attn_paged_qrow_tree's ninth operand: one int32 a query row (docs/OPS.md). Causal only. */
    const RadTensor* tp = TREE ? rad_arg_in(a, 8) : nullptr;
    if (TREE && (!tp || !tp->data)) return RAD_E_INVAL;
    if (TREE && tp->dtype != RAD_I32) return RAD_E_DTYPE;
    if (TREE && rad_args_geti_or(a, "causal", 1) != 1) return RAD_E_UNSUPPORTED;
    const RadTensor* q  = rad_arg_in(a, 0);
    const RadTensor* kv = rad_arg_in(a, 1);
    const RadTensor* bt = rad_arg_in(a, 2);
    const RadTensor* su = rad_arg_in(a, 3);
    const RadTensor* kd = rad_arg_in(a, 4);
    const RadTensor* vd = rad_arg_in(a, 5);
    const RadTensor* cu = rad_arg_in(a, 6);
    const RadTensor* sk = sink ? rad_arg_in(a, 7) : nullptr;
    const RadTensor* o  = rad_arg_in(a, sink ? 8 : 7);
    if (!q || !kv || !bt || !su || !o || (sink && !sk)) return RAD_E_INVAL;

    if (rad_args_geti_or(a, "head_dim", HD) != HD || rad_args_geti_or(a, "gqa", GQA) != GQA ||
        rad_args_geti_or(a, "block_size", 16) != 16)
        return RAD_E_SHAPE;
    const long long causal = rad_args_geti_or(a, "causal", 1);
    const long long window = rad_args_geti_or(a, "window", 0);
    if ((causal != 0 && causal != 1) || window < 0 || window > (1ll << 30)) return RAD_E_SHAPE;
    /* The open-ended window is a non-causal one: a causal row has nothing ahead, and without a
     * window there is nothing to bound. */
    if (FWD && (causal != 0 || window == 0)) return RAD_E_SHAPE;

    if (q->dtype != RAD_BF16 || o->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (kv->dtype != (uint32_t)(KV8 ? RAD_I8 : RAD_BF16)) return RAD_E_DTYPE;
    if (bt->dtype != RAD_I32 || su->dtype != RAD_I32 || (cu && cu->dtype != RAD_I32)) return RAD_E_DTYPE;
    if ((kd && kd->dtype != RAD_F32) || (vd && vd->dtype != RAD_F32)) return RAD_E_DTYPE;
    if (sk && sk->dtype != RAD_F32) return RAD_E_DTYPE;

    /* the cache: [n_blocks, kv_heads, 16, 2 * head_dim (+ 4)], slots dense */
    if (kv->rank != 4 || kv->shape[2] != 16 || kv->shape[3] != SLOT_E) return RAD_E_SHAPE;
    if (kv->stride[3] != 1 || kv->stride[2] != SLOT_E) return RAD_E_STRIDE;
    const int64_t kvh = kv->shape[1];
    if (kvh <= 0 || kvh > 65535) return RAD_E_SHAPE;
    const int64_t qh = kvh * GQA;
    if (sk && rad_tensor_numel(sk) != qh) return RAD_E_SHAPE;

    int64_t rows = 0, q_rs = 0, q_hs = 0, orows = 0, o_rs = 0, o_hs = 0;
    if (!head_view(q, qh, HD, &rows, &q_rs, &q_hs) || !head_view(o, qh, HD, &orows, &o_rs, &o_hs))
        return RAD_E_SHAPE;
    if (orows < rows) return RAD_E_SHAPE;
    if (TREE && rad_tensor_numel(tp) < rows) return RAD_E_SHAPE;
    const int64_t n_seq = rad_tensor_numel(su);
    if (rows == 0) return RAD_OK;
    if (n_seq <= 0 || n_seq > (1 << 24)) return RAD_E_SHAPE;

    /* the block table: [n_seq, max_blocks] with a row pitch, or flat */
    int64_t max_blocks = 0, bt_rs = 0;
    if (bt->rank >= 2) {
        if (bt->stride[bt->rank - 1] != 1) return RAD_E_STRIDE;
        max_blocks = bt->shape[bt->rank - 1];
        bt_rs = bt->stride[bt->rank - 2];
        if (rad_tensor_numel(bt) / max_blocks < n_seq) return RAD_E_SHAPE;
    } else {
        max_blocks = rad_tensor_numel(bt) / n_seq;
        bt_rs = max_blocks;
    }
    if (max_blocks <= 0 || max_blocks > (1 << 26)) return RAD_E_SHAPE;
    if ((kd && rad_tensor_numel(kd) < n_seq * kvh) || (vd && rad_tensor_numel(vd) < n_seq * kvh))
        return RAD_E_SHAPE;

    /* 8-byte query/output chunks, 16-byte cache loads */
    if (!rad_ptr_aligned(q->data, 8) || !rad_ptr_aligned(o->data, 8) || (q_rs | q_hs | o_rs | o_hs) % 4)
        return RAD_E_ALIGN;
    if (!rad_ptr_aligned(kv->data, 16) || (kv->stride[0] * ESZ) % 16 || (kv->stride[1] * ESZ) % 16)
        return RAD_E_ALIGN;

    /* the tiles: uniform, or an upper bound over a ragged batch (empty tiles exit at once) */
    int64_t nt = 0, tps = 0, qlen_u = 0;
    if (!cu) {
        if (rows % n_seq) return RAD_E_SHAPE;
        qlen_u = rows / n_seq;
        tps = (qlen_u + BQ - 1) / BQ;
        nt = n_seq * tps;
    } else {
        if (rad_tensor_numel(cu) != n_seq + 1) return RAD_E_SHAPE;
        int64_t qmax = rad_args_geti_or(a, "q_len", rows);
        if (qmax <= 0 || qmax > rows) qmax = rows;
        const int64_t b1 = (rows + n_seq * (BQ - 1)) / BQ;
        const int64_t b2 = n_seq * ((qmax + BQ - 1) / BQ);
        nt = b1 < b2 ? b1 : b2;
    }
    if (nt <= 0) return RAD_OK;
    if (nt > 0x7fffffff) return RAD_E_SHAPE;

    /* WMMA path for prefill-shaped steps: longest query >= kWmmaMinQ rows and mean >= half that.
     * Decode and verify batches keep the scalar kernel and its split-KV. Needs 16-byte aligned
     * query and output rows. Both kernels agree with libref within 0.02 (Gemma 4 26B-A4B operands)
     * and differ from each other only in summation order. */
    {
        int64_t qmax_w = qlen_u;
        if (cu) {
            qmax_w = rad_args_geti_or(a, "q_len", rows);
            if (qmax_w <= 0 || qmax_w > rows) qmax_w = rows;
        }
        /* Never a tree: its verify blocks are decode-sized, and only the scalar kernel reads one. */
        const bool shaped = !TREE && !SHARED && qmax_w >= kWmmaMinQ &&
                            rows >= n_seq * (kWmmaMinQ / 2);
        const bool aligned = rad_ptr_aligned(q->data, 16) && rad_ptr_aligned(o->data, 16) &&
                             (q_rs | q_hs | o_rs | o_hs) % 8 == 0;
        if (shaped && aligned) {
            constexpr int BQW = WGeo<HD>::R / GQA;
            int64_t ntw = 0, tpsw = 0;
            if (!cu) {
                tpsw = (qlen_u + BQW - 1) / BQW;
                ntw = n_seq * tpsw;
            } else {
                const int64_t b1 = (rows + n_seq * (BQW - 1)) / BQW;
                const int64_t b2 = n_seq * ((qmax_w + BQW - 1) / BQW);
                ntw = b1 < b2 ? b1 : b2;
            }
            if (ntw <= 0) return RAD_OK;
            if (ntw > 0x7fffffff) return RAD_E_SHAPE;
            AttnArgs W{};
            W.q = (const uint16_t*)q->data;           W.q_rs = q_rs;  W.q_hs = q_hs;
            W.kv = (const unsigned char*)kv->data;    W.kv_sb = kv->stride[0];  W.kv_sh = kv->stride[1];
            W.bt = (const int*)bt->data;              W.bt_rs = bt_rs;
            W.seqused = (const int*)su->data;
            W.cu = cu ? (const int*)cu->data : nullptr;
            W.kdes = kd ? (const float*)kd->data : nullptr;
            W.vdes = vd ? (const float*)vd->data : nullptr;
            W.out = (uint16_t*)o->data;               W.o_rs = o_rs;  W.o_hs = o_hs;
            const double wscale = rad_args_getf_or(a, "scale", 1.0 / __builtin_sqrt((double)HD));
            W.qscale = (float)(wscale * 1.4426950408889634);
            W.max_blocks = (int)max_blocks;
            W.n_seq = (int)n_seq;
            W.q_len = (int)qlen_u;
            W.tps = (int)tpsw;
            W.kv_heads = (int)kvh;
            W.causal = (int)causal;
            W.window = (int)window;
            W.shared_window = SHARED;
            W.ahead = FWD ? 1 : 0;
            W.sinks = sk ? (const float*)sk->data : nullptr;
            W.n_split = 1;
            R4D_KLAUNCH(R4D_KNAME(attn_wmma_kernel<HD, GQA, KV8>), dim3((unsigned)ntw, (unsigned)kvh, 1u),
                       dim3(kWThreads), (hipStream_t)stream, W);
            return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
    }

    /* Split-KV only when (tile, kv head) pairs are too few to fill the GPU */
    int64_t ns = 1;
    const int64_t pairs = nt * kvh;
    if (pairs < kTinfoMax) {
        ns = kPartBudget / pairs;
        if (ns > kMaxSplit) ns = kMaxSplit;
        const int64_t span = window > 0 ? window + BQ
                                         : rad_args_geti_or(a, "max_ctx", max_blocks * 16);
        const int64_t nkt = (span + BK - 1) / BK + 1;
        if (ns > nkt / 2) ns = nkt / 2;
        if (ns < 1) ns = 1;
        if (ns > 1 && (!a->scratch || a->scratch_bytes < scratch_need<HD>() ||
                       !rad_ptr_aligned(a->scratch, 16)))
            ns = 1;   /* never wrong, only slower: the unsplit kernel writes the rows itself */
    }
    AttnArgs A{};
    A.q = (const uint16_t*)q->data;           A.q_rs = q_rs;  A.q_hs = q_hs;
    A.kv = (const unsigned char*)kv->data;    A.kv_sb = kv->stride[0];  A.kv_sh = kv->stride[1];
    A.bt = (const int*)bt->data;              A.bt_rs = bt_rs;
    A.seqused = (const int*)su->data;
    A.cu = cu ? (const int*)cu->data : nullptr;
    A.kdes = kd ? (const float*)kd->data : nullptr;
    A.vdes = vd ? (const float*)vd->data : nullptr;
    A.out = (uint16_t*)o->data;               A.o_rs = o_rs;  A.o_hs = o_hs;
    A.part = ns > 1 ? (float*)a->scratch : nullptr;
    A.sinks = sk ? (const float*)sk->data : nullptr;
    A.tinfo = ns > 1 ? (int*)((char*)a->scratch + part_bytes<HD>()) : nullptr;
    const double scale = rad_args_getf_or(a, "scale", 1.0 / __builtin_sqrt((double)HD));
    A.qscale = (float)(scale * 1.4426950408889634);
    A.max_blocks = (int)max_blocks;
    A.n_seq = (int)n_seq;
    A.q_len = (int)qlen_u;
    A.tps = (int)tps;
    A.kv_heads = (int)kvh;
    A.causal = (int)causal;
    A.window = (int)window;
    A.shared_window = SHARED;
    A.ahead = FWD ? 1 : 0;
    A.n_split = (int)ns;
    A.tpar = TREE ? (const int*)tp->data : nullptr;

    hipStream_t s = (hipStream_t)stream;
    /* The chained forms where the row resolved with `chain` (launches chain), at decode heights
     * only: the 31B's full layers at a verify. */
    const bool chain = rad_args_geti_or(a, "chain", 0) != 0 && rows <= 64;
    if (chain)
        R4D_KLAUNCH(R4D_KNAME(r4dchain_attn_window<HD, GQA, KV8, TREE>),
                   dim3((unsigned)nt, (unsigned)kvh, (unsigned)ns), dim3(kThreads), s, A);
    else
        R4D_KLAUNCH(R4D_KNAME(attn_kernel<HD, GQA, KV8, TREE>),
                   dim3((unsigned)nt, (unsigned)kvh, (unsigned)ns), dim3(kThreads), s, A);
    if (ns > 1) {
        if (chain)
            R4D_KLAUNCH(R4D_KNAME(r4dchain_merge_window<HD, GQA>), dim3((unsigned)nt, (unsigned)kvh),
                       dim3(kThreads), s, A);
        else
            R4D_KLAUNCH(R4D_KNAME(merge_kernel<HD, GQA>), dim3((unsigned)nt, (unsigned)kvh),
                       dim3(kThreads), s, A);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* namespace r4d_win */

/* ------------------------------------------------------------------ the exported entry points */
extern "C" {

/* Each geometry at the bf16 cache and at the int8 one (kv_store_i8's layout, KV8 2). */
#define R4D_WIN_DEF(HD, G)                                                                        \
    int r4d_rad_attn_window_h##HD##_gqa##G##_bf16(const RadArgs* a, RadStream s) {                \
        return r4d_win::launch_attn<HD, G, 0>(a, s);                                              \
    }                                                                                             \
    int r4d_rad_attn_window_h##HD##_gqa##G##_i8(const RadArgs* a, RadStream s) {                  \
        return r4d_win::launch_attn<HD, G, 2>(a, s);                                              \
    }
/* h128 gqa16: Muse Glimmer's sliding layers, 32 query heads over 2 KV heads; the same
 * launch_attn the sink variant below already runs at this geometry. */
R4D_WIN_DEF(128, 1) R4D_WIN_DEF(128, 16)
R4D_WIN_DEF(256, 1) R4D_WIN_DEF(256, 2) R4D_WIN_DEF(256, 4) R4D_WIN_DEF(256, 8)
R4D_WIN_DEF(512, 1) R4D_WIN_DEF(512, 2) R4D_WIN_DEF(512, 4) R4D_WIN_DEF(512, 8) R4D_WIN_DEF(512, 16)
#undef R4D_WIN_DEF

/* attn_paged_ahead and attn_paged_qrow_ahead: a non-causal window that bounds the keys behind only
 * (FWD), the sliding layers' mask over a picture's rows (docs/OPS.md). */
#define R4D_WIN_AHEAD_DEF(HD, G)                                                                  \
    int r4d_rad_attn_window_ahead_h##HD##_gqa##G##_bf16(const RadArgs* a, RadStream s) {          \
        return r4d_win::launch_attn<HD, G, 0, false, false, true>(a, s);                          \
    }                                                                                             \
    int r4d_rad_attn_window_ahead_h##HD##_gqa##G##_i8(const RadArgs* a, RadStream s) {            \
        return r4d_win::launch_attn<HD, G, 2, false, false, true>(a, s);                          \
    }
R4D_WIN_AHEAD_DEF(256, 1) R4D_WIN_AHEAD_DEF(256, 2) R4D_WIN_AHEAD_DEF(256, 4) R4D_WIN_AHEAD_DEF(256, 8)
#undef R4D_WIN_AHEAD_DEF

/* attn_paged_qrow_tree: the scalar kernel with a tree verify's mask (TREE). */
#define R4D_WIN_TREE_DEF(HD, G)                                                                   \
    int r4d_rad_attn_window_tree_h##HD##_gqa##G##_bf16(const RadArgs* a, RadStream s) {           \
        return r4d_win::launch_attn<HD, G, 0, true>(a, s);                                        \
    }                                                                                             \
    int r4d_rad_attn_window_tree_h##HD##_gqa##G##_i8(const RadArgs* a, RadStream s) {             \
        return r4d_win::launch_attn<HD, G, 2, true>(a, s);                                        \
    }
R4D_WIN_TREE_DEF(256, 1) R4D_WIN_TREE_DEF(256, 2) R4D_WIN_TREE_DEF(256, 4) R4D_WIN_TREE_DEF(256, 8)
R4D_WIN_TREE_DEF(512, 1) R4D_WIN_TREE_DEF(512, 2) R4D_WIN_TREE_DEF(512, 4) R4D_WIN_TREE_DEF(512, 8)
#undef R4D_WIN_TREE_DEF

/* Exact DSpark mask: every query row uses the first query position as the sliding-window anchor.
 * This is a distinct op name so a library without this semantic cannot silently resolve it. */
#define R4D_WIN_ANCHOR_DEF(HD, G)                                                                \
    int r4d_rad_attn_anchor_window_h##HD##_gqa##G##_bf16kv(const RadArgs* a, RadStream s) {      \
        return r4d_win::launch_attn<HD, G, 0, false, true>(a, s);                                \
    }                                                                                            \
    int r4d_rad_attn_anchor_window_h##HD##_gqa##G##_i8kv(const RadArgs* a, RadStream s) {        \
        return r4d_win::launch_attn<HD, G, 2, false, true>(a, s);                                \
    }
R4D_WIN_ANCHOR_DEF(128, 1)
R4D_WIN_ANCHOR_DEF(256, 1) R4D_WIN_ANCHOR_DEF(256, 2) R4D_WIN_ANCHOR_DEF(256, 4) R4D_WIN_ANCHOR_DEF(256, 8)
R4D_WIN_ANCHOR_DEF(512, 1) R4D_WIN_ANCHOR_DEF(512, 2) R4D_WIN_ANCHOR_DEF(512, 4) R4D_WIN_ANCHOR_DEF(512, 8)
#undef R4D_WIN_ANCHOR_DEF

/* attn_paged_sink: LiLiCorr's drafter, 32 query heads over 2 KV heads at head_dim 128 (16 over 1
 * at TP 2: gqa 16 either way; 8 over its replicated head at TP 4: gqa 8), non-causal over its
 * block, a 1024-token window. */
#define R4D_WINSINK_DEF(HD, G)                                                                    \
    int r4d_rad_attn_window_sink_h##HD##_gqa##G##_bf16(const RadArgs* a, RadStream s) {           \
        return r4d_win::launch_attn<HD, G, 0>(a, s, true);                                        \
    }                                                                                              \
    int r4d_rad_attn_window_sink_h##HD##_gqa##G##_i8(const RadArgs* a, RadStream s) {             \
        return r4d_win::launch_attn<HD, G, 2>(a, s, true);                                        \
    }
R4D_WINSINK_DEF(128, 8) R4D_WINSINK_DEF(128, 16)
#undef R4D_WINSINK_DEF

/* Called at declare with no tensors. Partials are capped at kPartBudget (tile, head, split)
 * triples for any batch, so the size is a constant. */
int64_t r4d_rad_attn_window_h128_scratch(const RadArgs*) { return r4d_win::scratch_need<128>(); }
int64_t r4d_rad_attn_window_h256_scratch(const RadArgs*) { return r4d_win::scratch_need<256>(); }
int64_t r4d_rad_attn_window_h512_scratch(const RadArgs*) { return r4d_win::scratch_need<512>(); }

}  /* extern "C" */
