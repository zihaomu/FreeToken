/* r4d_attn_args.h -- the paged-attention operand reader shared by every paged kernel geometry.
 *
 * The paged kernels (h256/gqa6, h128/gqa4, the gqa8/gqa12 instantiations; prefill and decode) take
 * the same operands in the same order, so one reader serves them all.
 *
 * R4DArgs uses RadTensor's convention (strides in elements, spec 2.1), so this is field extraction
 * plus the per-call validation a constraint cannot express: contiguity, alignment and buffer
 * sizes, each refused with the code that names the class.
 *
 * Kernel return codes for an unserved shape (-1 head_dim or block_size, -2 gqa, -4 too many decode
 * rows, -100-hipError for a launch failure) pass through unchanged. Getting one means a selection
 * bug, and the value says which.
 */
#ifndef R4D_ATTN_ARGS_H
#define R4D_ATTN_ARGS_H

#include "r4d.h"
#include "r4d_args.h"

#include <cmath>
#include <cstring>

/* Checks libr4d's compiled-in geometry against the row table, so a libr4d rebuilt for a
 * different head size cannot silently change what the rows mean. */
static inline int check_dims_once() {
    int hd = 0, gqa = 0, bs = 0, rows = 0;
    r4d_attn_dims(&hd, &gqa, &bs, &rows);
    return (hd == 256 && gqa == 6 && bs == 16 && rows == 64) ? RAD_OK : RAD_E_SHAPE;
}

/* Operand indices, from oAttnPaged in r4d_rows.cpp. */
enum { AP_Q = 0, AP_KV, AP_BT, AP_SEQ, AP_KDS, AP_VDS, AP_CU, AP_OUT };

/* Fill R4DArgs from a RadArgs. Returns RAD_OK, or the negative code naming what is wrong.
 *
 * Ragged batches: the kernels take a scalar q_len and an optional cu_seqlens. Without cu_seqlens
 * every sequence must have the same query length, which RAD_PHASE_MIXED steps (decode rows riding
 * on another sequence's prefill chunk, core/sched/scheduler.cpp) do not satisfy; refusing them
 * would abort the engine. So cu_seqlens is used whenever present, and the uniform arithmetic
 * (q_len = total_q / num_seqs) only when it is not. In the ragged case q_len is the batch maximum,
 * taken from the parameter the arch block issued the op with (batch->max_q_len).
 *
 * `q_pitched`: q may be a column slice of a wider row (attn_paged_qrow), its pitch in r->q_ld. Only
 * the split-KV decode kernel reads q_ld; every other caller leaves this off.
 */
/* `row` is a token's bytes (elements for a 16-bit cache) in a (block, head) page: 2 * head_dim,
 * or 25 * head_dim / 16 for the fp8_v4 cache (E4M3 K, int4 V, f16 group scales). */
static inline int fill_paged_row(const RadArgs* a, R4DArgs* r, int expect_head_dim, long long row,
                                 bool q_pitched = false) {
    const RadTensor* q  = r4d_opt(a, AP_Q);
    const RadTensor* kv = r4d_opt(a, AP_KV);
    const RadTensor* bt = r4d_opt(a, AP_BT);
    const RadTensor* su = r4d_opt(a, AP_SEQ);
    const RadTensor* cu = r4d_opt(a, AP_CU);
    const RadTensor* o  = r4d_opt(a, AP_OUT);
    if (!q || !kv || !bt || !su || !o) return R4D_NO(RAD_E_INVAL);

    if (q->rank < 2 || o->rank < 2 || bt->rank != 2 || su->rank < 1) return R4D_NO(RAD_E_SHAPE);
    if (kv->rank != 4) return R4D_NO(RAD_E_SHAPE);

    /* The query may be [tokens, n_head, head_dim] or flattened to [tokens, n_head*head_dim]
     * (a gated attention block's de-interleaved query). Both are the same dense block, so any rank
     * is accepted; head_dim comes from the op parameter, which makes the flattened form readable. */
    const long long head_dim = expect_head_dim;
    const long long total_q  = q->shape[0];
    const long long num_seqs = bt->shape[0];
    if (num_seqs <= 0 || total_q <= 0 || head_dim <= 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(q) % (total_q * head_dim)) return R4D_NO(RAD_E_SHAPE);
    const long long q_heads = r4d_numel(q) / (total_q * head_dim);
    if (q_heads <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Ragged exactly when cu_seqlens is present; see the note above fill_paged. */
    const bool ragged = (cu != nullptr) && cu->data;
    if (ragged) {
        if (cu->rank != 1 || cu->shape[0] != num_seqs + 1) return R4D_NO(RAD_E_SHAPE);
        if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);
    } else if (total_q % num_seqs) {
        return R4D_NO(RAD_E_SHAPE);
    }

    /* q and out are indexed as one dense [total_q, q_heads, head_dim] block; the kernel has no
     * pitch argument for either -- except q under q_pitched, whose rows may be apart: dense inside
     * a row, a pitch in whole 8-byte loads (the kernel reads q four elements at a time). */
    long long q_ld = 0;
    if (q_pitched && !r4d_contig(q)) {
        q_ld = r4d_row_pitch(q);
        if (q_ld < q_heads * head_dim || q_ld % 4) return R4D_NO(RAD_E_STRIDE);
    } else if (!r4d_contig(q)) {
        return R4D_NO(RAD_E_STRIDE);
    }
    if (!r4d_contig(o)) return R4D_NO(RAD_E_STRIDE);
    if (o->shape[0] != total_q || r4d_numel(o) != total_q * q_heads * head_dim)
        return R4D_NO(RAD_E_SHAPE);
    if (su->shape[0] != num_seqs) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(bt) || !r4d_contig(su)) return R4D_NO(RAD_E_STRIDE);

    /* The KV cache is pitched (kv_block_stride, kv_head_stride), so a padded cache is legal.
     * Each (block, head) slot must be dense, because the kernel walks it with vector loads. */
    const long long block_size = kv->shape[2];
    if (kv->stride[3] != 1) return R4D_NO(RAD_E_STRIDE);
    if (kv->stride[2] != row) return R4D_NO(RAD_E_STRIDE);
    if (kv->shape[3] != row) return R4D_NO(RAD_E_SHAPE);
    /* The kernels divide q_heads by kv_heads for the GQA factor. */
    if (kv->shape[1] <= 0) return R4D_NO(RAD_E_SHAPE);

    /* 16-byte alignment for 128-bit loads. */
    if (!r4d_aligned(q->data, 16) || !r4d_aligned(o->data, 16) || !r4d_aligned(kv->data, 16))
        return R4D_NO(RAD_E_ALIGN);

    const RadTensor* kds = r4d_opt(a, AP_KDS);
    const RadTensor* vds = r4d_opt(a, AP_VDS);

    std::memset(r, 0, sizeof(*r));
    r->q           = q->data;
    r->kv          = kv->data;
    r->block_table = (const int*)bt->data;
    r->seqused_k   = (const int*)su->data;
    r->out         = o->data;
    r->k_descale   = kds ? (const float*)kds->data : nullptr;
    r->v_descale   = vds ? (const float*)vds->data : nullptr;
    r->q_descale   = nullptr;                 /* the query is bf16 in every variant */
    r->q_ld        = (long)q_ld;              /* 0: dense */
    r->scratch     = a->scratch;
    r->num_seqs    = (int)num_seqs;
    r->cu_q        = ragged ? (const int*)cu->data : nullptr;
    r->total_q     = (int)total_q;
    /* q_len is exact for a uniform batch and an upper bound for a ragged one; it sizes the
     * prefill grid and the decode wave count, and the kernels read exact lengths from cu_seqlens.
     * The bound is min(band, total_q): the `q_len` parameter is the resolved band selected by
     * max_q_len, and total_q is a sum of lengths, so both are >= the true maximum. The min keeps
     * decode legal (band bound 10 * gqa 6 = 60 rows fits a 64-row workgroup; total_q alone would
     * be refused with -4). A loose bound only costs workgroups that exit on their first compare. */
    r->q_len       = (int)(total_q / num_seqs);
    if (ragged) {
        const long long band = r4d_geti_or(a, "q_len", total_q);
        r->q_len = (int)(band > 0 && band < total_q ? band : total_q);
    }
    r->q_heads     = (int)q_heads;
    r->kv_heads    = (int)kv->shape[1];
    r->head_dim    = (int)head_dim;
    r->block_size  = (int)block_size;
    r->max_blocks  = (int)bt->shape[1];
    r->kv_block_stride = (long)kv->stride[0];
    r->kv_head_stride  = (long)kv->stride[1];
    r->window      = (int)r4d_geti_or(a, "window", 0);
    r->causal      = (int)r4d_geti_or(a, "causal", 1);
    r->causal_seq  = nullptr;

    /* Softmax scale; defaults to 1/sqrt(head_dim). */
    r->scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)head_dim));

    /* max_ctx is the host-visible context bound for the split law (seqused_k is device-side).
     * Block-table width times block size is a safe upper bound. */
    long long mc = r4d_geti_or(a, "max_ctx", 0);
    if (mc <= 0) mc = (long long)r->max_blocks * block_size;
    r->max_ctx = (int)mc;

    /* Set by the caller from the tuned `splits`; 0 means libr4d's own split law. */
    r->splits = 0;
    r->chain = r4d_geti_or(a, "chain", 0) != 0 ? 1 : 0;
    return RAD_OK;
}

/* The decode rows declare a `splits` axis; its default is 0, which is that same law. */
static inline int split_of(const RadArgs* a) {
    R4dSplitCfg c;
    r4d_cfg_split(a, &c);
    return c.splits;
}


static inline int fill_paged(const RadArgs* a, R4DArgs* r, int expect_head_dim,
                             bool q_pitched = false) {
    return fill_paged_row(a, r, expect_head_dim, 2LL * expect_head_dim, q_pitched);
}

#endif  /* R4D_ATTN_ARGS_H */
