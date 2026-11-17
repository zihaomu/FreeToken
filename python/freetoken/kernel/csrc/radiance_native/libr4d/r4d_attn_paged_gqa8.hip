// Modified for FreeToken: native integration, diagnostics and Qwen FP8 KV support.
// R4D paged attention dispatch at eight queries per KV head, head_dim 256 and 128.
//
// Instantiations of the h256/gqa6 kernel templates (templated on head_dim and gqa), for the gqa-8
// models; the constraint tables gate on the exact ratio:
//
//   Qwen3.6-35B-A3B   16 query heads over 2 KV heads, head_dim 256
//   MiniCPM5-2B       16 query heads over 2 KV heads, head_dim 128
//
// Gqa changes only two bounds. A decode workgroup holds 64 rows of q_len * gqa, so the verify
// width is 8 here (10 at gqa 6); the selector bands on that value, stated in the row. A prefill
// block carries NWARPS * 16 / gqa query positions (48 here, 64 at gqa 6).
//
// Head_dim changes how K is fetched. O_KWIDE's thread-to-(key, channel) map covers exactly 16 x 16
// channels (head_dim 256); at 128 it would stage past the tile, so the h128 prefill runs without it.
// Its fragments take half the registers, so it keeps K prefetch 4 where h256 uses 2. All other
// option bits are as defined in r4d_attn_prefill_h256_gqa6.hip.
//
// No r4d_attn_dims here: that entry point (and check_dims_once) belongs to the h256/gqa6 unit,
// and a second definition would not link. Geometry is gated by the row constraints and the
// launchers below.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
// The decode kernel uses helpers defined in the prefill unit, so prefill is included first.
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"
#include "r4d_attn_prefill_split.h"

#define CHK8() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

/* Queries per KV head, and the symbol names built from it.
 *
 * Another ratio reuses this file: it defines these macros and includes this one (e.g.
 * r4d_attn_paged_h256_gqa12.hip), and the entry points paste the ratio into their names. Only
 * the two bounds change: decode verify width 64/gqa (8 here, 5 at gqa 12) and prefill block
 * 384/gqa query positions (48 here, 32 at gqa 12). Both must divide, and both are stated in the
 * row so the selector bands on them. */
#ifndef R4D_ATTN_GQA
#define R4D_ATTN_GQA  8
#define R4D_ATTN_H128 1      /* this ratio also serves head_dim 128 -- MiniCPM5-2B is 16 over 2 */
#endif
/* Head_dim 64 prefill leg, off unless the including unit enables it (LFM2: 32 query heads over 8,
 * r4d_attn_paged_h256_gqa4.hip). Uses the h128 option set (no O_KWIDE). No decode leg: the
 * split-KV merge holds a lane's columns as one or two dwords of codes (head_dim 128 or 256), so
 * the query-tiled kernel serves every query length at this width. */
#ifndef R4D_ATTN_H64
#define R4D_ATTN_H64 0
#endif
/* Head_dim 256 leg, on unless the including unit serves only head_dim 128 (as R4D_ATTN_H128 0
 * leaves the 128 leg out). */
#ifndef R4D_ATTN_H256
#define R4D_ATTN_H256 1
#endif
#define GQ_JOIN_(a, b, c) a##b##c
#define GQ_JOIN(a, b, c)  GQ_JOIN_(a, b, c)
#define GQ_SYM(pre, suf)  GQ_JOIN(pre, R4D_ATTN_GQA, suf)

#define G8_GQA         R4D_ATTN_GQA
/* KV page size in tokens; the kernel body is generic in it, so an including unit may override.
 * Qwen4-Exp's QSA uses 4: its indexer selects 4-token blocks, so the selection is just a block
 * table (docs/QSA.md). */
#ifndef G8_BLOCK_SIZE
#define G8_BLOCK_SIZE 16
#endif
/* q_len * gqa a single decode workgroup can hold: q_len <= 8 here. An including unit may raise it
 * to 128 (eight waves): gqa 16 then verifies 7 drafts in the decode kernel, as gqa 8 does. A unit
 * may also lower it (see the gqa7 unit: 63 rows and 21 warps at gqa 7), so each bound below is a
 * guard the including unit can win; the defaults are the numbers the row table states for gqa 8. */
#ifndef G8_MAX_DECODE_ROWS
#define G8_MAX_DECODE_ROWS 64
#endif

#ifndef G8_P_NWARPS
#define G8_P_NWARPS 24
#endif
#ifndef G8_P_TILE
#define G8_P_TILE   48
#endif
// K prefetch 2 at head_dim 256, as in the h256/gqa6 prefill (P_OPT there): 4 spills the m-tile
// loop. The head_dim-128 set keeps prefetch 4, which fits.
#define G8_P_OPT_256 3954187
#define G8_P_OPT_128 3430923        // without O_KWIDE, prefetch 4
#define G8_P_OPT_64  G8_P_OPT_128
#define G8_D_TILE   16
#define G8_D_OPT    3429947
#define G8_D_JPB     2
#define G8_D_VECMIN 256

// ---- prefill -----------------------------------------------------------------------------------
// Query positions a workgroup: 384 / gqa.
#define G8_P_BQ ((G8_P_NWARPS * 16) / G8_GQA)

// a->splits > 1 with a->scratch set is the KV split (r4d_attn_prefill_split.h): segments in the
// scratch, then the merge. Only the head_dim 256 entries set it (their rows declare the scratch);
// the 128 and 64 legs have no split instance and always run the plain form, unchanged.
template <int HEAD_DIM, int OPT, int KVP>
static int g8_prefill_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != HEAD_DIM || a->block_size != G8_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != G8_GQA) return -2;
    constexpr int BQ = G8_P_BQ;
    constexpr bool CAN_SPLIT = HEAD_DIM == 256;
    const int splits = (CAN_SPLIT && a->scratch && a->splits > 1) ? a->splits : 1;
    g_prefill_last_splits = splits;
    dim3 grid((a->q_len + BQ - 1) / BQ * splits, a->kv_heads, a->num_seqs);
    if constexpr (CAN_SPLIT) {
        if (splits > 1) {
            hipLaunchKernelGGL((r4d_attn_prefill_kernel<G8_P_NWARPS, G8_P_TILE, HEAD_DIM, G8_GQA,
                                                        G8_BLOCK_SIZE, KVP, OPT | O_PSPLIT>),
                               grid, dim3(G8_P_NWARPS * 32), 0, s, *a);
            CHK8();
            return r4d_prefill_combine<HEAD_DIM, G8_D_JPB, G8_D_VECMIN>(a, splits, G8_P_TILE, s);
        }
    }
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<G8_P_NWARPS, G8_P_TILE, HEAD_DIM, G8_GQA,
                                                G8_BLOCK_SIZE, KVP, OPT>),
                       grid, dim3(G8_P_NWARPS * 32), 0, s, *a);
    CHK8();
    return 0;
}

// ---- decode ------------------------------------------------------------------------------------
// The h256/gqa6 split law (r4d_decode_splits).
extern "C" long GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(const R4DArgs* a) {
    return r4d_decode_scratch(a, r4d_decode_splits(a, G8_D_TILE), G8_D_OPT);
}

template <int HEAD_DIM, int KVP, int XOPT = 0>
/* Two dispatches: the attention kernel and the split-KV combine. R4DArgs is passed by value (184
 * bytes) like any other kernel argument. Non-null `gq` adds the output gate and quantiser to the
 * combine (attn_paged_gate_quant; R4DGateQuant in r4d_attn_decode_h256_gqa6.hip). `XOPT` adds
 * option bits to the attention kernel's (O_KVI8: the int8 cache). */
static int g8_decode_launch(const R4DArgs* a, RadStream stream,
                            const R4DGateQuant* gq = nullptr) {
    int emit_rc = RAD_OK;
    if (a->head_dim != HEAD_DIM || a->block_size != G8_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != G8_GQA) return -2;
    const int rows = a->q_len * G8_GQA;
    if (rows > G8_MAX_DECODE_ROWS) return -4;
    const int nw = (rows + 15) / 16;
    const int splits = r4d_decode_splits(a, G8_D_TILE);

    /* `a->chain`: the chained forms (r4d_chain.h) of the attention kernel and of the merge
     * without a gate; the merge with a gate and quantiser has none. */
#define G8_LAUNCH_K(K_, NW, OPT_)                                                              \
    emit_rc = R4D_EMIT(stream, (K_<NW, G8_D_TILE, HEAD_DIM, G8_GQA, G8_BLOCK_SIZE, KVP,        \
                                   G8_D_OPT | OPT_ | XOPT>),                                   \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, *a, splits)
#define G8_LAUNCH(NW, OPT_)                                                                    \
    do { if (a->chain) G8_LAUNCH_K(r4dchain_attn_decode, NW, OPT_);                            \
         else          G8_LAUNCH_K(r4d_attn_decode_kernel, NW, OPT_); } while (0)
    /* One-wave workgroups stage V as codes (O_V8): same conversion count, half the LDS. The int8
     * cache stages V widened (the kernel drops O_V8 under it). */
    if      (nw == 1) G8_LAUNCH(1, O_V8);
    else if (nw == 2) G8_LAUNCH(2, 0);
    else if (nw == 3) G8_LAUNCH(3, 0);
#if G8_MAX_DECODE_ROWS > 64
    /* Five to eight waves take eight; a wave past the last query row computes a copy of it and
     * stores nothing. */
    else if (nw > 4)  G8_LAUNCH(8, 0);
#endif
    else              G8_LAUNCH(4, 0);
#undef G8_LAUNCH
#undef G8_LAUNCH_K
    if (emit_rc != RAD_OK) return -5;
    CHK8();

    // Merge form, as in the other dispatches: one wave per row when there are enough rows and the
    // segment count fits a wave, else the general LDS-tree form. Both give identical bytes.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (gq) {
        /* GQ 3: rotated int8 codes, GQ 2: int8 codes, GQ 1: E4M3; see R4DGateQuant. */
#define G8_GQ_VEC(PF, GQV)                                                                     \
        R4D_EMIT(stream,                                                                       \
                 (r4d_attn_splitkv_combine_vec_gq_kernel<HEAD_DIM, 4, PF, G8_D_JPB, 0, GQV>),   \
                 dim3((jobs3 + G8_D_JPB - 1) / G8_D_JPB), dim3(32 * G8_D_JPB), 0, *a, splits,   \
                 G8_D_TILE, *gq)
#define G8_GQ_GEN(PF, GQV)                                                                     \
        R4D_EMIT(stream, (r4d_attn_splitkv_combine_gq_kernel<HEAD_DIM, 4, PF, 0, GQV>),         \
                 dim3(jobs3), dim3(HEAD_DIM), splits * sizeof(float), *a, splits, G8_D_TILE,   \
                 *gq)
        const bool i8 = gq->i8 != 0, rot = gq->rot != 0;
        if (splits <= 64 && jobs3 >= G8_D_VECMIN) {
            if (G8_D_OPT & 32)
                emit_rc = rot ? G8_GQ_VEC(1, 3) : i8 ? G8_GQ_VEC(1, 2) : G8_GQ_VEC(1, 1);
            else
                emit_rc = rot ? G8_GQ_VEC(0, 3) : i8 ? G8_GQ_VEC(0, 2) : G8_GQ_VEC(0, 1);
        } else if (G8_D_OPT & 32) {
            emit_rc = rot ? G8_GQ_GEN(1, 3) : i8 ? G8_GQ_GEN(1, 2) : G8_GQ_GEN(1, 1);
        } else {
            emit_rc = rot ? G8_GQ_GEN(0, 3) : i8 ? G8_GQ_GEN(0, 2) : G8_GQ_GEN(0, 1);
        }
#undef G8_GQ_VEC
#undef G8_GQ_GEN
    } else if (splits <= 64 && jobs3 >= G8_D_VECMIN) {
        const int blocks = (jobs3 + G8_D_JPB - 1) / G8_D_JPB;
#define G8_VEC(K_, PF)                                                                         \
        emit_rc = R4D_EMIT(stream, (K_<HEAD_DIM, 4, PF, G8_D_JPB>), dim3(blocks),              \
                           dim3(32 * G8_D_JPB), 0, *a, splits, G8_D_TILE)
        if (G8_D_OPT & 32) {
            if (a->chain) G8_VEC(r4dchain_attn_splitkv_combine_vec, 1);
            else          G8_VEC(r4d_attn_splitkv_combine_vec_kernel, 1);
        } else {
            if (a->chain) G8_VEC(r4dchain_attn_splitkv_combine_vec, 0);
            else          G8_VEC(r4d_attn_splitkv_combine_vec_kernel, 0);
        }
#undef G8_VEC
    } else {
        /* Block of HEAD_DIM threads, not 256: the general combine is __launch_bounds__(HEAD_DIM),
         * one thread per channel, and the runtime refuses a larger block. */
#define G8_GEN(K_, PF)                                                                         \
        emit_rc = R4D_EMIT(stream, (K_<HEAD_DIM, 4, PF>), dim3(jobs3), dim3(HEAD_DIM),         \
                           splits * sizeof(float), *a, splits, G8_D_TILE)
        if (G8_D_OPT & 32) {
            if (a->chain) G8_GEN(r4dchain_attn_splitkv_combine, 1);
            else          G8_GEN(r4d_attn_splitkv_combine_kernel, 1);
        } else {
            if (a->chain) G8_GEN(r4dchain_attn_splitkv_combine, 0);
            else          G8_GEN(r4d_attn_splitkv_combine_kernel, 0);
        }
#undef G8_GEN
    }
    if (emit_rc != RAD_OK) return -5;
    CHK8();
    return 0;
}

// ============================================================== the plugin ABI entry points

#if R4D_ATTN_H256
/* The head_dim 256 prefill's KV split at this ratio's workgroup height (r4d_attn_prefill_split.h).
 * Test hooks (r4d_selftest), not plugin entries, named by ratio and page since gqa 6 exists at 16-
 * and 4-token pages in different units: the law at this card's WGP count, and the split count this
 * unit's last prefill launch used. */
#define G8_HOOK_(g, b, suf) r4d_attn_prefill_h256_gqa##g##_bs##b##suf
#define G8_HOOK(g, b, suf)  G8_HOOK_(g, b, suf)
extern "C" int G8_HOOK(R4D_ATTN_GQA, G8_BLOCK_SIZE, _splits)(int q_len, int total_q, int num_seqs,
                                                             int kv_heads, int max_ctx) {
    return r4d_prefill_splits_law(G8_P_BQ, G8_P_TILE, q_len, total_q, num_seqs, kv_heads, max_ctx,
                                  r4d_prefill_wgps());
}
extern "C" int G8_HOOK(R4D_ATTN_GQA, G8_BLOCK_SIZE, _last_splits)(void) {
    return g_prefill_last_splits;
}
#undef G8_HOOK
#undef G8_HOOK_

/* The split's f32 partials at declare, by the cache's token row (r4d_prefill_scratch). */
static int64_t g8_prefill_scratch(const RadArgs* a, long long row) {
    return r4d_prefill_scratch(a, 256, row, G8_P_BQ, G8_P_TILE);
}
extern "C" int64_t GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _scratch)(const RadArgs* a) {
    return g8_prefill_scratch(a, 512);
}

extern "C" int GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    r.splits = r4d_prefill_splits(a, &r, G8_P_BQ, G8_P_TILE);
    return g8_prefill_launch<256, G8_P_OPT_256, 1>(&r, (hipStream_t)s);
}
#endif
#if R4D_ATTN_H128
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h128_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<128, G8_P_OPT_128, 1>(&r, (hipStream_t)s);
}
#endif
#if R4D_ATTN_H64
/* The int8 cache at this width: a token's row is 2 * 64 + 4 = 132 bytes. */
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h64_gqa, _i8)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged_row(a, &r, 64, 2 * 64 + 4);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<64, G8_P_OPT_64 | O_KVI8, 0>(&r, (hipStream_t)s);
}
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h64_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 64);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<64, G8_P_OPT_64, 1>(&r, (hipStream_t)s);
}
#endif

/* Split-KV partial buffer size at declare time, from the band's upper bound. Declare passes
 * parameters and no tensors, so parameters come first and tensors are the fallback (see the
 * h256/gqa6 dispatch). The size is the maximum over every sequence count up to max_seqs and every
 * query length up to the band's: S * floor(128 / (kv_heads * S)) is not monotone in S, and the
 * split law's targets (r4d_decode_splits: 256 workgroups at q_len * gqa <= 32; above, 128 to 16384
 * tokens deep and more past it) can give a shorter query more splits: at gqa 8 a 4-row step needs
 * 1.6x the partials of the band bound 5. */
/* `row` is a token's bytes in a (block, head) page, 0 for the bf16 cache's 2 * head_dim elements
 * (fill_paged_row). */
static int64_t g8_decode_scratch(const RadArgs* a, int head_dim, long long row = 0) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);

    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        std::memset(&r, 0, sizeof(r));
        r.head_dim   = head_dim;
        r.q_heads    = (int)n_head;
        const long long gqa = r4d_geti_or(a, "gqa", 1);
        r.kv_heads   = (int)(gqa > 0 ? n_head / gqa : n_head);
        r.num_seqs   = (int)max_seqs;
        const long long q_hi = std::max(1LL, r4d_geti_or(a, "q_len", 1));
        r.max_ctx    = (int)r4d_geti_or(a, "max_ctx", 0);
        r.block_size = (int)r4d_geti_or(a, "block_size", 16);
        r.splits     = sp;
        if (r.max_ctx <= 0) return R4D_NO(RAD_E_INVAL);
        int64_t worst = 0;
        for (long long q = 1; q <= q_hi; ++q)
            for (long long n = 1; n <= max_seqs; ++n) {
                r.q_len    = (int)q;
                r.num_seqs = (int)n;
                const int64_t need = GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(&r);
                if (need > worst) worst = need;
            }
        return worst;
    }
    const int rc = fill_paged_row(a, &r, head_dim, row > 0 ? row : 2LL * head_dim);
    if (rc != RAD_OK) return rc;
    r.splits = sp;
    return GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(&r);
}

#if R4D_ATTN_H256
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h256_gqa, _scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 256);
}
#endif
#if R4D_ATTN_H128
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h128_gqa, _scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 128);
}
#endif

static int g8_decode_common(const RadArgs* a, R4DArgs* r, int head_dim, long long row = 0) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);
    const int rc = fill_paged_row(a, r, head_dim, row > 0 ? row : 2LL * head_dim);
    if (rc != RAD_OK) return rc;
    r->splits = sp;
    const int64_t need = GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(r);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    return RAD_OK;
}

#if R4D_ATTN_H256
/* ---- attn_paged_gate_quant: attn_paged's eight operands, then the gate, codes and scales.
 * The gate is a [T, n_head, head_dim] column view of the [q|gate] projection; its (token, head)
 * rows are one pitch apart only if the view narrowed the last axis alone. */
enum { AP_GQ_G = AP_OUT + 1, AP_GQ_Q, AP_GQ_S };

static long long gq_row_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] != width) return r4d_contig(t) ? width : -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 1) return width;
    for (int i = (int)t->rank - 2; i >= 1; --i)
        if (t->stride[i - 1] != t->shape[i] * t->stride[i]) return -1;
    return t->stride[t->rank - 2];
}

/* `rot`: attn_paged_gate_had_quant, whose codes are in the rotated domain. */
static int g8_gate_quant_args(const RadArgs* a, const R4DArgs& r, R4DGateQuant* gq,
                              bool rot = false) {
    const RadTensor* g = r4d_opt(a, AP_GQ_G);
    const RadTensor* q = r4d_opt(a, AP_GQ_Q);
    const RadTensor* s = r4d_opt(a, AP_GQ_S);
    if (!g || !q || !s) return R4D_NO(RAD_E_INVAL);
    /* The codes' dtype picks the quantiser: E4M3, or int8 for an int8 out projection. */
    if (g->dtype != RAD_BF16 || (q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) ||
        s->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (rot && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    const char* act = r4d_gets(a, "act");
    if (!act || std::strcmp(act, "sigmoid")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (r4d_geti_or(a, "group", kGqGroup) != kGqGroup) return R4D_NO(RAD_E_SHAPE);
    const long long n = r.head_dim;
    if (n % kGqGroup) return R4D_NO(RAD_E_SHAPE);
    const long long nkb  = n / kGqGroup;
    const long long rows = (long long)r4d_q_total(r) * r.q_heads;
    if (r4d_numel(g) < rows * n || r4d_numel(q) < rows * n || r4d_numel(s) < rows * nkb)
        return R4D_NO(RAD_E_SHAPE);
    const long long gp = gq_row_pitch(g, n), qp = gq_row_pitch(q, n), sp = gq_row_pitch(s, nkb);
    if (gp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    /* The vector merge reads a lane's gate in one 16-byte access and stores its codes in one
     * 8-byte access. */
    if ((r4d_addr(g) & 15) || (gp % 8) || (r4d_addr(q) & 7) || (qp % 8))
        return R4D_NO(RAD_E_STRIDE);
    gq->g    = (const uint16_t*)r4d_addr(g);
    gq->q    = (uint8_t*)r4d_addr(q);
    gq->s    = (float*)r4d_addr(s);
    gq->g_ld = (unsigned)gp;
    gq->q_ld = (unsigned)qp;
    gq->s_ld = (unsigned)sp;
    gq->i8   = q->dtype == RAD_I8 ? 1 : 0;
    gq->rot  = rot ? 1 : 0;
    return RAD_OK;
}

/* bf16-cache decode, plain and with the gate and quantiser in the merge (`_gq` E4M3 or int8 codes,
 * `_ghq` rotated int8 codes for an out projection stored rotated). */
extern "C" int GQ_SYM(r4d_rad_attn_decode_h256_gqa, _bf16)(const RadArgs* a, RadStream stream) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<256, 1>(&r, stream);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_gq_h256_gqa, _bf16)(const RadArgs* a, RadStream stream) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    R4DGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq);
    if (gc != RAD_OK) return gc;
    return g8_decode_launch<256, 1>(&r, stream, &gq);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_ghq_h256_gqa, _bf16)(const RadArgs* a, RadStream stream) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    R4DGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq, true);
    if (gc != RAD_OK) return gc;
    return g8_decode_launch<256, 1>(&r, stream, &gq);
}
#endif
#if R4D_ATTN_H128
extern "C" int GQ_SYM(r4d_rad_attn_decode_h128_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<128, 1>(&r, s);
}
#endif

#if R4D_ATTN_H128
/* ---- the int8 cache at head_dim 128: a token's row is 2 * 128 + 4 = 260 bytes. Prefill (K staged
 * eight dims a thread, having no wide fetch at this width), and the decode that also serves a
 * drafter's non-causal block. */
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h128_gqa, _i8)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged_row(a, &r, 128, 2 * 128 + 4);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<128, G8_P_OPT_128 | O_KVI8, 0>(&r, (hipStream_t)s);
}
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h128_gqa, _i8_scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 128, 2 * 128 + 4);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_h128_gqa, _i8)(const RadArgs* a, RadStream stream) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 128, 2 * 128 + 4);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<128, 0, O_KVI8>(&r, stream);
}
#endif

#if R4D_ATTN_H256
/* ---- the int8 cache at head_dim 256 (a token's row 2 * 256 + 4 = 516 bytes): prefill, decode
 * and decode with the gate and quantiser in the merge. `_ghq` is `_gq` with the rotated
 * quantiser. */
#define G8_I8_ENTRIES(TAG, ROWB, XOPT)                                                        \
extern "C" int64_t GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _##TAG##_scratch)(const RadArgs* a) { \
    return g8_prefill_scratch(a, ROWB);                                                        \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _##TAG)(const RadArgs* a, RadStream s) { \
    R4DArgs r;                                                                                 \
    const int rc = fill_paged_row(a, &r, 256, ROWB);                                           \
    if (rc != RAD_OK) return rc;                                                               \
    r.splits = r4d_prefill_splits(a, &r, G8_P_BQ, G8_P_TILE);                                  \
    return g8_prefill_launch<256, G8_P_OPT_256 | (XOPT), 0>(&r, (hipStream_t)s);               \
}                                                                                              \
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h256_gqa, _##TAG##_scratch)(const RadArgs* a) {  \
    return g8_decode_scratch(a, 256, ROWB);                                                    \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_decode_h256_gqa, _##TAG)(const RadArgs* a,                  \
                                                            RadStream stream) {                \
    R4DArgs r;                                                                                 \
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \
    if (rc != RAD_OK) return rc;                                                               \
    return g8_decode_launch<256, 0, (XOPT)>(&r, stream);                                       \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_decode_gq_h256_gqa, _##TAG)(const RadArgs* a,               \
                                                               RadStream stream) {             \
    R4DArgs r;                                                                                 \
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \
    if (rc != RAD_OK) return rc;                                                               \
    R4DGateQuant gq{};                                                                         \
    const int gc = g8_gate_quant_args(a, r, &gq);                                              \
    if (gc != RAD_OK) return gc;                                                               \
    return g8_decode_launch<256, 0, (XOPT)>(&r, stream, &gq);                                  \
}                                                                                              \
extern "C" int GQ_SYM(r4d_rad_attn_decode_ghq_h256_gqa, _##TAG)(const RadArgs* a,              \
                                                                RadStream stream) {            \
    R4DArgs r;                                                                                 \
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \
    if (rc != RAD_OK) return rc;                                                               \
    R4DGateQuant gq{};                                                                         \
    const int gc = g8_gate_quant_args(a, r, &gq, true);                                        \
    if (gc != RAD_OK) return gc;                                                               \
    return g8_decode_launch<256, 0, (XOPT)>(&r, stream, &gq);                                  \
}
G8_I8_ENTRIES(i8, 516, O_KVI8)
#if R4D_ATTN_GQA == 12
G8_I8_ENTRIES(fp8, 512, 0)
#endif
#undef G8_I8_ENTRIES
#endif
