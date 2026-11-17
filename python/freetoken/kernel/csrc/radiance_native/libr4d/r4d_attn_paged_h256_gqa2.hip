// R4D attention dispatch for Gemma 4 dense's sliding-window layers: head_dim 256, 2 queries per KV
// head, paged block 16, window 1024 (50 of 60 layers). Instantiations of the h256/gqa6 kernel
// templates with WIN = 1, as the DFlash2 drafter leg does at head_dim 128.
//
// Both bands: decode (q_len * gqa <= 64, so q_len <= 32) and the query-tiled prefill kernel. Unlike
// the drafter leg, a trunk layer prefills, and the libref fallback is opt-in and refused by
// default (spec §17).
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
// The decode kernel uses helpers from the prefill unit, so prefill is included first. Both kernels
// are instantiated here with WIN = 1.
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"
#include "r4d_attn_prefill_split.h"

#define CHK4() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define B_HEAD_DIM   256
#define B_GQA          2
#define B_BLOCK_SIZE  16
#define B_TILE 16                   // decode: r4d_decode_launch_win (r4d_attn_decode_h256_gqa6.hip)
// The h256 decode option set, K prefetch 2 included. Same register use as gqa6 (256 VGPRs, small
// spill); not separately tuned for this geometry.
#define B_OPT  3429947

extern "C" long r4d_attn_decode_h256_gqa2_scratch_bytes(const R4DArgs* a) {
    return r4d_decode_scratch(a, r4d_decode_splits_win(a, B_TILE), B_OPT);
}

extern "C" int r4d_attn_decode_h256_gqa2_i8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, (B_OPT & ~O_KRAW) | O_KVI8, 0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa2_bf16kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT, 1>(a, s);
}
// The same with q at R4DArgs::q_ld (O_QLD), for attn_paged_qrow; its own instances, so the dense
// ones above keep their registers.
extern "C" int r4d_attn_decode_h256_gqa2_qld_i8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, (B_OPT & ~O_KRAW) | O_KVI8 | O_QLD, 0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa2_qld_bf16kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT | O_QLD, 1>(a, s);
}
// ...and a tree verify's mask (O_TREE, attn_paged_qrow_tree), again its own instances.
extern "C" int r4d_attn_decode_h256_gqa2_qld_tree_i8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, (B_OPT & ~O_KRAW) | O_KVI8 | O_QLD | O_TREE,
                                 0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa2_qld_tree_bf16kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT | O_QLD | O_TREE, 1>(a, s);
}

// ---- prefill ---------------------------------------------------------------------------------
// The h256/gqa6 prefill option set but at K prefetch 4 (P_OPT there is 3954187); O_KWIDE is valid
// at head_dim 256. At gqa 2 a block covers NWARPS * 16 / 2 = 192 query positions. WIN is on;
// window <= 0 gives full attention.
#define P_NWARPS 24
#define P_TILE   48
#define P_OPT    3955211

#define P_BQ ((P_NWARPS * 16) / B_GQA)

// ---- the prefill KV split (r4d_attn_prefill_split.h) -------------------------------------------
// Full-attention launches only (window <= 0). The split's partition has no window floor, so the
// split instance is the WIN = 0 form, which computes what WIN = 1 does at window <= 0. A windowed
// launch stays unsplit: its walk is bounded by the window (Gemma 4's 1024 keys: at most
// (1024 + 192) / 48 = 26 tiles a workgroup, whatever the context), so few workgroups cost little.
static bool prefill_h256_gqa2_can_split(const R4DArgs* r) { return r->window <= 0; }

// Test hooks (r4d_selftest), not plugin entries: the law at this unit's geometry and this card's
// WGP count, and the split count this unit's last prefill launch used.
extern "C" int r4d_attn_prefill_h256_gqa2_splits(int q_len, int total_q, int num_seqs,
                                                 int kv_heads, int max_ctx) {
    return r4d_prefill_splits_law(P_BQ, P_TILE, q_len, total_q, num_seqs, kv_heads, max_ctx,
                                  r4d_prefill_wgps());
}
extern "C" int r4d_attn_prefill_h256_gqa2_last_splits(void) { return g_prefill_last_splits; }

// a->splits > 1 with a->scratch set and no window is the split form: segments in the scratch, then
// the merge. Anything else is the plain (windowed) form, unchanged.
template <int KVP, int XOPT = 0>
static int prefill_launch_h256_gqa2(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != B_HEAD_DIM || a->block_size != B_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != B_GQA) return -2;
    constexpr int BQ = P_BQ;
    const int splits =
        (a->scratch && a->splits > 1 && prefill_h256_gqa2_can_split(a)) ? a->splits : 1;
    g_prefill_last_splits = splits;
    dim3 grid((a->q_len + BQ - 1) / BQ * splits, a->kv_heads, a->num_seqs);
    if (splits > 1) {
        hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, B_HEAD_DIM, B_GQA,
                                                    B_BLOCK_SIZE, KVP, P_OPT | XOPT | O_PSPLIT, 0>),
                           grid, dim3(P_NWARPS * 32), 0, s, *a);
        CHK4();
        // The full-attention legs' merge constants, as r4d_decode_launch_win takes them: two rows
        // a block, the wave form from 256 rows.
        return r4d_prefill_combine<B_HEAD_DIM, 2, 256>(a, splits, P_TILE, s);
    }
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, B_HEAD_DIM, B_GQA,
                                                B_BLOCK_SIZE, KVP, P_OPT | XOPT, 1>),
                       grid, dim3(P_NWARPS * 32), 0, s, *a);
    CHK4();
    return 0;
}

/* The split's f32 partials at declare (r4d_prefill_scratch). None for a windowed layer: `window`
 * is the op's parameter, the same at declare and at every launch, and a windowed launch never
 * splits. */
static int64_t prefill_scratch_h256_gqa2(const RadArgs* a, long long row) {
    if (r4d_geti_or(a, "window", 0) > 0) return 0;
    return r4d_prefill_scratch(a, B_HEAD_DIM, row, P_BQ, P_TILE);
}
extern "C" int64_t r4d_rad_attn_prefill_h256_gqa2_scratch(const RadArgs* a) {
    return prefill_scratch_h256_gqa2(a, 2LL * B_HEAD_DIM);
}
extern "C" int64_t r4d_rad_attn_prefill_h256_gqa2_i8_scratch(const RadArgs* a) {
    return prefill_scratch_h256_gqa2(a, 2LL * B_HEAD_DIM + 4);
}

extern "C" int r4d_attn_prefill_h256_gqa2_i8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch_h256_gqa2<0, O_KVI8>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa2_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch_h256_gqa2<1>(a, s);
}

/* Prefill RadArgs wrappers, defined in this unit rather than the h256 dispatch; the int8 cache's
 * token row is 2 * 256 + 4 bytes. */
extern "C" int r4d_rad_attn_prefill_h256_gqa2_i8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = fill_paged_row(a, &r, B_HEAD_DIM, 2 * B_HEAD_DIM + 4);
    if (rc != RAD_OK) return rc;
    if (prefill_h256_gqa2_can_split(&r)) r.splits = r4d_prefill_splits(a, &r, P_BQ, P_TILE);
    return r4d_attn_prefill_h256_gqa2_i8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_prefill_h256_gqa2_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = fill_paged(a, &r, B_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    if (prefill_h256_gqa2_can_split(&r)) r.splits = r4d_prefill_splits(a, &r, P_BQ, P_TILE);
    return r4d_attn_prefill_h256_gqa2_bf16kv(&r, (hipStream_t)s);
}
