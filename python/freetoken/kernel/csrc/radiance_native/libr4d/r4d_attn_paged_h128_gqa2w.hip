// R4D attention dispatch for Gemma 3's sliding-window layers (arch/gemma3, TranslateGemma 27B):
// head_dim 128, 2 queries per KV head (32 over 16, 16 over 8 at TP2), paged block 16, window 1024
// (52 of 62 layers). Instantiations of the h256/gqa6 kernel templates with WIN = 1, as
// r4d_attn_paged_h256_gqa2.hip does at head_dim 256 for Gemma 4.
//
// Both bands: decode (q_len * gqa <= 64, so q_len <= 32) and the query-tiled prefill kernel. The
// windowless h128/gqa2 rows (r4d_attn_paged_h128_gqa2.hip) keep window == 0; these rows take
// window >= 1 only, so the global layers stay on the tuned full-attention legs.
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

#define CHK4() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define B_HEAD_DIM   128
#define B_GQA          2
#define B_BLOCK_SIZE  16
#define B_TILE 16                   // decode: r4d_decode_launch_win (r4d_attn_decode_h256_gqa6.hip)
// The DFlash2 drafter's head_dim 128 windowed decode set (r4d_attn_paged_h128_gqa4.hip): K
// prefetch 2 and O_KRAW. Not separately tuned for gqa 2.
#define B_OPT  20207163

extern "C" long r4d_attn_decode_h128_gqa2w_scratch_bytes(const R4DArgs* a) {
    return r4d_decode_scratch(a, r4d_decode_splits_win(a, B_TILE), B_OPT);
}

// The int8 cache: K is widened from the paged cache a fragment at a time, so without the one-round
// raw K loads.
extern "C" int r4d_attn_decode_h128_gqa2w_i8kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, (B_OPT & ~O_KRAW) | O_KVI8, 0>(a, s);
}
extern "C" int r4d_attn_decode_h128_gqa2w_bf16kv(const R4DArgs* a, hipStream_t s) {
    return r4d_decode_launch_win<B_HEAD_DIM, B_GQA, B_OPT, 1>(a, s);
}

// ---- prefill ---------------------------------------------------------------------------------
// The windowless h128 prefill option set (r4d_attn_paged_gqa8.hip, G8_P_OPT_128: no O_KWIDE, which
// is a head_dim 256 fetch, K prefetch 4) with WIN on. At gqa 2 a block covers NWARPS * 16 / 2 = 192
// query positions. window <= 0 would give full attention, but the rows never send one here.
#define P_NWARPS 24
#define P_TILE   48
#define P_OPT    3430923

template <int KVP, int XOPT = 0>
static int prefill_launch_h128_gqa2w(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != B_HEAD_DIM || a->block_size != B_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != B_GQA) return -2;
    constexpr int BQ = (P_NWARPS * 16) / B_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, B_HEAD_DIM, B_GQA,
                                                B_BLOCK_SIZE, KVP, P_OPT | XOPT, 1>),
                       grid, dim3(P_NWARPS * 32), 0, s, *a);
    CHK4();
    return 0;
}

extern "C" int r4d_attn_prefill_h128_gqa2w_i8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch_h128_gqa2w<0, O_KVI8>(a, s);
}
extern "C" int r4d_attn_prefill_h128_gqa2w_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch_h128_gqa2w<1>(a, s);
}

/* Prefill RadArgs wrappers; the int8 cache's token row is 2 * 128 + 4 bytes. The decode wrappers
 * are in r4d_attn_paged_h256_gqa6.hip beside decode_common. */
extern "C" int r4d_rad_attn_prefill_h128_gqa2w_i8(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = fill_paged_row(a, &r, B_HEAD_DIM, 2 * B_HEAD_DIM + 4);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h128_gqa2w_i8kv(&r, (hipStream_t)s);
}
extern "C" int r4d_rad_attn_prefill_h128_gqa2w_bf16(const RadArgs* a, RadStream s) {
    R4DArgs r;
    int rc = fill_paged(a, &r, B_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    return r4d_attn_prefill_h128_gqa2w_bf16kv(&r, (hipStream_t)s);
}
