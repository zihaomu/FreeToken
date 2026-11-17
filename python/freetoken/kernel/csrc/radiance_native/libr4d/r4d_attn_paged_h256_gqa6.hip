// R4D paged attention dispatch for head_dim 256, 6 queries per KV head, paged block 16. The
// KV-dtype variants are template instantiations of one device code. The geometry is a hard gate:
// anything else is rejected.
//
// The query is bf16 in every variant. A single-term 8-bit query is ~10x less accurate against an
// fp32 reference (2.58e-02 vs ~2.3e-03 relative). Two families ship, differing in KV dtype: bf16
// and the int8 cache (int8 K and V, an f16 scale each per token and head).
//
// `opt` is a bitmask of optimisation steps:
//     prefill 3954187 | KLATE = f16 operands | mask skip | folded query scale | contiguous-k fragments
//                     | prefetch 2 + scheduling barrier | dot2 row sum | block table in SGPRs
//                     | wide K fetch | V prefetch across the m-tile loop | LDS-scoped barrier
//                     | 8-bit K fetched a tile ahead (KLATE)
//     decode  3429947 = the same minus the wide K fetch (its K never reaches LDS), plus K read
//                       straight from the paged cache (bit 16) and f16 split-KV partials (32).
// Each bit is defined where it is used (r4d_attn_prefill_h256_gqa6.hip,
// r4d_attn_decode_h256_gqa6.hip) and can be turned off on its own after a geometry change.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"
#include "r4d_attn_prefill_split.h"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

// The geometry of every kernel in this unit. BLOCK_SIZE is not in the entry-point names because
// it is 16 for all of them.
#define A_HEAD_DIM   256
#define A_GQA          6
#define A_BLOCK_SIZE  16
#define A_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold

extern "C" void r4d_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows) {
    if (head_dim)        *head_dim        = A_HEAD_DIM;
    if (gqa)             *gqa             = A_GQA;
    if (block_size)      *block_size      = A_BLOCK_SIZE;
    if (max_decode_rows) *max_decode_rows = A_MAX_DECODE_ROWS;
}

// ---- prefill ---------------------------------------------------------------------------------
// LDS at head_dim 256, 16-bit K and V: sK = TILE*264*2, sV = 256*(TILE+8)*2. TILE 64 would need
// 68096 B > 65536, so TILE is 48.
#define P_NWARPS 24
#define P_TILE   48
// K prefetch 2, not 4: four fragments in flight make the m-tile loop spill at six waves per SIMD
// (36-48 bytes per lane, ~4% of the prefill). At 2 the spill is 20 bytes, outside the WMMA groups.
#define P_OPT    (3954187 | O_KLATE)

// ---- the prefill KV split (r4d_attn_prefill_split.h) -------------------------------------------
// Query positions a workgroup: 64.
#define P_BQ ((P_NWARPS * 16) / A_GQA)

// The law at this unit's geometry and this card's WGP count. A test hook (r4d_selftest), not a
// plugin entry.
extern "C" int r4d_attn_prefill_h256_gqa6_splits(int q_len, int total_q, int num_seqs,
                                                 int kv_heads, int max_ctx) {
    return r4d_prefill_splits_law(P_BQ, P_TILE, q_len, total_q, num_seqs, kv_heads, max_ctx,
                                  r4d_prefill_wgps());
}
// The split count this unit's last prefill launch used, 1 unsplit. A test hook (r4d_selftest).
extern "C" int r4d_attn_prefill_h256_gqa6_last_splits(void) { return g_prefill_last_splits; }

static int prefill_combine(const R4DArgs* a, int splits, hipStream_t s);

// KVP: 0 = the int8 cache (XOPT carries its option bits), 1 = bf16 cache.
// a->splits > 1 with a->scratch set (the plugin entries below, prefill_splits) is the split form:
// segments in the scratch, then the merge. Anything else is the plain form, unchanged.
template <int KVP, int XOPT = 0>
static int prefill_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    constexpr int BQ = (P_NWARPS * 16) / A_GQA;
    constexpr bool CAN_SPLIT = true;
    const int splits = (CAN_SPLIT && a->scratch && a->splits > 1) ? a->splits : 1;
    g_prefill_last_splits = splits;
    dim3 grid((a->q_len + BQ - 1) / BQ * splits, a->kv_heads, a->num_seqs);
    if constexpr (CAN_SPLIT) {
        if (splits > 1) {
            hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE,
                                                        KVP, P_OPT | XOPT | O_PSPLIT>),
                               grid, dim3(P_NWARPS * 32), 0, s, *a);
            CHK();
            return prefill_combine(a, splits, s);
        }
    }
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE,
                                                KVP, P_OPT | XOPT>),
                       grid, dim3(P_NWARPS * 32), 0, s, *a);
    CHK();
    return 0;
}

extern "C" int r4d_attn_prefill_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<1>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa6_i8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<0, O_V4 | O_K6 | O_I8>(a, s);
}
// ---- decode ----------------------------------------------------------------------------------
#define D_TILE 16
// Decode prefetches two K fragments per WMMA group, not four. Four costs 8 more VGPRs at the peak
// of the QK loop and spills 18, reloading part of the query from scratch on every tile. Two fits in
// 255 VGPRs and is 1.7-7.5% faster, with identical output.
#define D_OPT  3429947
// Output rows (waves) per block in the vectorised merge. 1, 2 and 4 are within 1%; 2 is best on
// average.
#define D_JPB 2
// Minimum output rows for the vectorised merge. It gives a row 32 lanes instead of 256, trading
// parallelism for 8x fewer loads, which pays only when there are enough rows to fill the GPU. At
// one request (96 rows, most segments) it loses up to 23%; the crossover is ~192 rows and above
// 288 it wins by 1.2-20%. The cut is four rows per CU.
#define D_VECMIN 256

// The prefill split's merge (r4d_prefill_combine), vectorised on the decode launcher's terms.
static int prefill_combine(const R4DArgs* a, int splits, hipStream_t s) {
    return r4d_prefill_combine<A_HEAD_DIM, D_JPB, D_VECMIN>(a, splits, P_TILE, s);
}

// Split law: r4d_decode_splits (r4d_attn_decode_h256_gqa6.hip).
extern "C" long r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a) {
    return r4d_decode_scratch(a, r4d_decode_splits(a, D_TILE), D_OPT);
}

template <int KVP, int XOPT = 0>
static int decode_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    const int rows = a->q_len * A_GQA;
    if (rows > A_MAX_DECODE_ROWS) return -4;    // verify width <= 10 at gqa 6
    const int nw = (rows + 15) / 16;            // gqa-aligned height; no dead m-tiles issued
    const int splits = r4d_decode_splits(a, D_TILE);

    /* `a->chain`: the chained forms (r4d_chain.h) of the attention kernel and of its merge. */
#define D_LAUNCH_K(K_, NW)                                                                    \
    hipLaunchKernelGGL((K_<NW, D_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, KVP, D_OPT | XOPT>),   \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
#define D_LAUNCH(NW)                                                                          \
    do { if (a->chain) D_LAUNCH_K(r4dchain_attn_decode, NW);                                  \
         else          D_LAUNCH_K(r4d_attn_decode_kernel, NW); } while (0)
    if      (nw == 1) D_LAUNCH(1);
    else if (nw == 2) D_LAUNCH(2);
    else if (nw == 3) D_LAUNCH(3);
    else              D_LAUNCH(4);
#undef D_LAUNCH
#undef D_LAUNCH_K
    CHK();

    // The vectorised merge's wave sum reproduces the general LDS tree only while splits <= 64.
    // Above that (only when kv_heads * num_seqs is 1, giving 128 splits) use the general form.
    // Both produce identical bytes; the choice is speed only.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    constexpr int PF16 = (D_OPT & 32) ? 1 : 0;
    if (splits <= 64 && jobs3 >= D_VECMIN) {
        const int blocks = (jobs3 + D_JPB - 1) / D_JPB;
        if (a->chain)
            hipLaunchKernelGGL((r4dchain_attn_splitkv_combine_vec<A_HEAD_DIM, 4, PF16, D_JPB>),
                               dim3(blocks), dim3(32 * D_JPB), 0, s, *a, splits, D_TILE);
        else
            hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<A_HEAD_DIM, 4, PF16, D_JPB>),
                               dim3(blocks), dim3(32 * D_JPB), 0, s, *a, splits, D_TILE);
    } else if (a->chain)
        hipLaunchKernelGGL((r4dchain_attn_splitkv_combine<A_HEAD_DIM, 4, PF16>), dim3(jobs3),
                           dim3(256), splits * sizeof(float), s, *a, splits, D_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<A_HEAD_DIM, 4, PF16>), dim3(jobs3),
                           dim3(256), splits * sizeof(float), s, *a, splits, D_TILE);
    CHK();
    return 0;
}

extern "C" int r4d_attn_decode_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<1>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa6_i8kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<0, O_V4 | O_K6 | O_I8>(a, s);
}

// ============================================================== the plugin ABI entry points
//
// The operand reader is r4d_attn_args.h, shared by every paged geometry.
//
// Note: entry points belong in this translation unit, not in the r4d_attn_prefill_*.hip or
// r4d_attn_decode_*.hip files it includes. An entry point added there does not get compiled, and
// the .so carries an undefined symbol that only r4d_selftest catches.
/* ------------------------------------------------------------------ paged prefill */

/* The launch's KV split count (r4d_prefill_splits) at this unit's geometry, 1 for the plain form. */
static int prefill_splits(const RadArgs* a, const R4DArgs* r) {
    return r4d_prefill_splits(a, r, P_BQ, P_TILE);
}

/* The split's f32 partials at declare (r4d_prefill_scratch: 23.8 MB is reached at this geometry,
 * at 384 rows x 5 and 640 x 3), by the cache's token row: bf16 (2 * head_dim elements) or int8
 * (2 * head_dim + 4 bytes). */
static int64_t prefill_scratch(const RadArgs* a, long long row) {
    return r4d_prefill_scratch(a, A_HEAD_DIM, row, P_BQ, P_TILE);
}
extern "C" int64_t r4d_rad_attn_prefill_h256_scratch(const RadArgs* a) {
    return prefill_scratch(a, 512);
}
extern "C" int64_t r4d_rad_attn_prefill_h256_i8_scratch(const RadArgs* a) {
    return prefill_scratch(a, 516);
}

extern "C" int r4d_rad_attn_prefill_h256_bf16(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    r.splits = prefill_splits(a, &r);
    return r4d_attn_prefill_h256_gqa6_bf16kv(&r, (hipStream_t)s);
}

/* ------------------------------------------------------------------ paged decode */

/* Split-KV partial buffer: written by the decode kernel, read by the combine, nothing else. Arena
 * scratch sized at declare from the band's upper bound (RadScratchFn).
 *
 * Declare passes parameters and no tensors (the batch is not known yet), so parameters are the
 * primary source and tensors the fallback. Reading tensors first would refuse every attn_paged
 * row with "invalid argument". q_len is the band's upper bound, since the region is sized once
 * for the worst step.
 *
 * The size is the maximum over every sequence count up to max_seqs and every query length up to
 * the band's: S * splits(S) is not monotone in S (at kv_heads 2 the h256 law gives 60 at twelve
 * sequences and 63 at nine, and the h128 law's step-up doubles splits at one count only), and the
 * h256 law's targets (r4d_decode_splits: 256 workgroups at q_len * gqa <= 32; above, 128 to 16384
 * tokens deep and more past it) can give a shorter query more splits (at gqa 6 and 16384 tokens, 5
 * rows a sequence need 1.67x the band bound 6's). The laws are closed forms, so the sweep is cheap. */
// Per-geometry byte counts over the shared partial layout, so one sweep serves all.
// which: 0 = h256/gqa6, 1 = h128/gqa4, 2 = h256/gqa2 (Gemma 4 windowed).
static int64_t decode_scratch_bytes_of(const R4DArgs* r, int which) {
    if (which == 1) return (int64_t)r4d_attn_decode_h128_gqa4_scratch_bytes(r);
    if (which == 2) return (int64_t)r4d_attn_decode_h256_gqa2_scratch_bytes(r);
    if (which == 3) return (int64_t)r4d_attn_decode_h128_gqa2w_scratch_bytes(r);
    return (int64_t)r4d_attn_decode_h256_gqa6_scratch_bytes(r);
}

/* `row`: a token's bytes in a (block, head) page (fill_paged_row), 0 for 2 * head_dim; `rep`: the
 * launch maps each kv head onto `rep` query groups (R4DArgs::kv_rep), so the split law counts
 * groups, gqa / rep of them a kv head; `q_pitched`: q may be a slice (attn_paged_qrow). */
static int64_t decode_scratch(const RadArgs* a, int head_dim, int which, long long row = 0,
                              int rep = 1, bool q_pitched = false) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);

    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        std::memset(&r, 0, sizeof(r));
        r.head_dim  = head_dim;
        r.q_heads   = (int)n_head;
        const long long gqa = r4d_geti_or(a, "gqa", 1);
        r.kv_heads  = (int)(gqa > 0 ? n_head / gqa : n_head) * rep;
        r.kv_rep    = rep;
        r.num_seqs  = (int)max_seqs;
        const long long q_hi = std::max(1LL, r4d_geti_or(a, "q_len", 1));
        r.max_ctx   = (int)r4d_geti_or(a, "max_ctx", 0);
        r.block_size = (int)r4d_geti_or(a, "block_size", 16);
        r.splits    = sp;
        /* max_ctx feeds only the split law. Zero would pick one split and undersize the buffer
         * for a launch that later picks more, so refuse. */
        if (r.max_ctx <= 0) return R4D_NO(RAD_E_INVAL);
        int64_t worst = 0;
        for (long long q = 1; q <= q_hi; ++q)
            for (long long n = 1; n <= max_seqs; ++n) {
                r.q_len    = (int)q;
                r.num_seqs = (int)n;
                const int64_t need = decode_scratch_bytes_of(&r, which);
                if (need > worst) worst = need;
            }
        return worst;
    }

    int rc = fill_paged_row(a, &r, head_dim, row > 0 ? row : 2LL * head_dim, q_pitched);
    if (rc != RAD_OK) return rc;
    r.kv_heads *= rep;
    r.kv_rep = rep;
    r.splits = sp;
    return decode_scratch_bytes_of(&r, which);
}

extern "C" int64_t r4d_rad_attn_decode_h256_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 0);
}

extern "C" int64_t r4d_rad_attn_decode_h128_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 1);
}

static int decode_common(const RadArgs* a, R4DArgs* r, int head_dim, int which, long long row = 0,
                         int rep = 1, bool q_pitched = false) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);
    int rc = fill_paged_row(a, r, head_dim, row > 0 ? row : 2LL * head_dim, q_pitched);
    if (rc != RAD_OK) return rc;
    /* a QLD instance reads q_ld whatever q is: dense is the pitch q_heads * head_dim */
    if (q_pitched && r->q_ld == 0) r->q_ld = (long)r->q_heads * head_dim;
    /* each cache head serves `rep` query groups: the kernel's grid and its gqa check count groups */
    r->kv_heads *= rep;
    r->kv_rep = rep;
    r->splits = sp;
    const int64_t need = decode_scratch_bytes_of(r, which);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    return RAD_OK;
}

/* The i8 cache: a token's row is 2 * 256 + 4 = 516 bytes (int8 K and V, an f16 scale each). */
extern "C" int r4d_rad_attn_prefill_h256_i8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = fill_paged_row(a, &r, 256, 516);
    if (rc != RAD_OK) return rc;
    r.splits = prefill_splits(a, &r);
    return r4d_attn_prefill_h256_gqa6_i8kv(&r, (hipStream_t)s);
}

extern "C" int64_t r4d_rad_attn_decode_h256_i8_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 0, 516);
}

extern "C" int r4d_rad_attn_decode_h256_i8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 0, 516);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa6_i8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h256_bf16(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 0);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa6_bf16kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h128_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_bf16kv(&r, (hipStream_t)s);
}

/* The i8 cache at this geometry: a token's row is 2 * 128 + 4 = 260 bytes. */
extern "C" int64_t r4d_rad_attn_decode_h128_i8_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 1, 260);
}

extern "C" int r4d_rad_attn_decode_h128_i8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1, 260);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_i8kv(&r, (hipStream_t)s);
}

/* Gemma 4 dense sliding window (head_dim 256, gqa 2, window 1024) decode; its prefill entry points
 * are in r4d_attn_paged_h256_gqa2.hip. No check_dims_once: this is not the geometry that reports
 * dims. */
extern "C" int64_t r4d_rad_attn_decode_h256_gqa2_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 2);
}

extern "C" int64_t r4d_rad_attn_decode_h256_gqa2_i8_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 2, 516);
}
extern "C" int r4d_rad_attn_decode_h256_gqa2_i8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2, 516);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_i8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_bf16kv(&r, (hipStream_t)s);
}

/* Gemma 3 sliding window (head_dim 128, gqa 2, window 1024) decode; its prefill entry points
 * are in r4d_attn_paged_h128_gqa2w.hip. */
extern "C" int64_t r4d_rad_attn_decode_h128_gqa2w_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 3);
}

extern "C" int64_t r4d_rad_attn_decode_h128_gqa2w_i8_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 3, 260);
}
extern "C" int r4d_rad_attn_decode_h128_gqa2w_i8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 3, 260);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa2w_i8kv(&r, (hipStream_t)s);
}

extern "C" int r4d_rad_attn_decode_h128_gqa2w_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 3);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa2w_bf16kv(&r, (hipStream_t)s);
}

/* The same decode under attn_paged_qrow: q read at its own row pitch (R4DArgs::q_ld), the sliding
 * layers' q where Gemma 4 31B's fused q|k|v projection left it. Same kernel, same arithmetic. */
extern "C" int64_t r4d_rad_attn_qrow_decode_h256_gqa2_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 2, 0, 1, true);
}
extern "C" int64_t r4d_rad_attn_qrow_decode_h256_gqa2_i8_scratch(const RadArgs* a) {
    return decode_scratch(a, 256, 2, 516, 1, true);
}
extern "C" int r4d_rad_attn_qrow_decode_h256_gqa2_i8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2, 516, 1, true);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_qld_i8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_qrow_decode_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2, 0, 1, true);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_qld_bf16kv(&r, (hipStream_t)s);
}

/* ...under attn_paged_qrow_tree: the same, with the tree verify's mask from `spec_parent` (operand
 * 8, one int32 a query row; docs/OPS.md). Causal only, as the op is. */
static int tree_operand(const RadArgs* a, R4DArgs* r) {
    const RadTensor* tp = a->n_t > 8 ? r4d_opt(a, 8) : nullptr;
    if (!tp || !tp->data) return R4D_NO(RAD_E_INVAL);
    if (tp->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(tp) < r4d_q_total(*r)) return R4D_NO(RAD_E_SHAPE);
    if (r->causal != 1 || r->causal_seq) return R4D_NO(RAD_E_UNSUPPORTED);
    r->spec_parent = (const int*)tp->data;
    return RAD_OK;
}
extern "C" int r4d_rad_attn_qrow_tree_decode_h256_gqa2_i8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2, 516, 1, true);
    if (rc == RAD_OK) rc = tree_operand(a, &r);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_qld_tree_i8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_qrow_tree_decode_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 256, 2, 0, 1, true);
    if (rc == RAD_OK) rc = tree_operand(a, &r);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h256_gqa2_qld_tree_bf16kv(&r, (hipStream_t)s);
}

/* Head_dim 128 at 8 queries per kv head, any window, causal or not, q_len up to 16, under
 * attn_paged_qrow: the gqa-4 kernel with each kv head mapped onto two query groups
 * (R4DArgs::kv_rep = 2), the 31B's DFlash drafter's attention over a cache that holds each kv head
 * once. The workgroup holds q_len * 4 rows, so a block of 16 fits where gqa 8 (128 rows) does not;
 * the two groups of a kv head read its blocks twice, as the copies they replace did. */
extern "C" int64_t r4d_rad_attn_qrow_h128_gqa8_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 1, 0, 2, true);
}
extern "C" int64_t r4d_rad_attn_qrow_h128_gqa8_i8_scratch(const RadArgs* a) {
    return decode_scratch(a, 128, 1, 260, 2, true);
}
extern "C" int r4d_rad_attn_qrow_h128_gqa8_i8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1, 260, 2, true);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_rep_i8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_qrow_h128_gqa8_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = decode_common(a, &r, 128, 1, 0, 2, true);
    if (rc != RAD_OK) return rc;
    return r4d_attn_decode_h128_gqa4_rep_bf16kv(&r, (hipStream_t)s);
}
