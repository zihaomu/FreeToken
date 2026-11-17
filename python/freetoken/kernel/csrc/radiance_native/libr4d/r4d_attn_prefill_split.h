// The prefill KV split, host side: the law, the scratch it declares and the merge launch, shared by
// every head-256 query-tiled prefill launcher (the h256/gqa6 unit, r4d_attn_paged_gqa8.hip's ratios
// 4, 6 over 4-token pages, 8 and 12, and the h256/gqa2 unit). The kernel side is O_PSPLIT in
// r4d_attn_prefill_h256_gqa6.hip. Include after r4d_attn_decode_h256_gqa6.hip, whose partial
// layout and merges it reuses. No entry points here: each unit defines its own, with its geometry.
//
// The query-tiled grid is ceil(q_len / BQ) x kv_heads x num_seqs, one 768-thread workgroup a WGP
// (BQ = NWARPS * 16 / gqa query positions: 192 at gqa 2, 96 at 4, 64 at 6, 48 at 8, 32 at 12), and
// each workgroup walks its whole causal range serially. Below one workgroup per WGP a launch
// therefore costs one full walk of the context whatever its row count: at gqa 6 with 2 KV heads a
// 53-row tail over 188k keys is 2 workgroups on 32 WGPs and takes about as long as a 1024-row
// chunk, and with one KV head a rank (gqa 12 at TP2) any chunk up to 32 rows is one workgroup.
// Such launches split the KV and finish in the split-KV merge.
#pragma once

#include "rad_prefill_split.h"
#include <atomic>

// Workgroups the card runs at once: one a WGP. Kernel launches use RadArgs::wgp_count, the same
// per-rank property used by pass-key callbacks. Declaration-time scratch sizing has no Ctx yet, so
// cache that query per HIP device rather than once process-wide (TP ranks can bind unlike cards).
static inline int r4d_prefill_wgps(const RadArgs* a = nullptr) {
    if (a && a->wgp_count > 0) return a->wgp_count;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0) return 32;
    static std::atomic<int> cached[64]{};
    if (dev < (int)(sizeof(cached) / sizeof(cached[0]))) {
        const int old = cached[dev].load(std::memory_order_relaxed);
        if (old > 0) return old;
    }
    hipDeviceProp_t pr{};
    if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return 32;
    const int count = pr.multiProcessorCount > 0 ? pr.multiProcessorCount : 32;
    if (dev < (int)(sizeof(cached) / sizeof(cached[0]))) {
        int expected = 0;
        cached[dev].compare_exchange_strong(expected, count, std::memory_order_relaxed);
        return cached[dev].load(std::memory_order_relaxed);
    }
    return count;
}

// The split law, at the launch's own geometry: `bq` query positions a workgroup, `tile` keys a KV
// tile. `units` counts the workgroups that do work: a ragged batch's grid is sized from its longest
// sequence (q_len), but a shorter sequence's tiles past its end exit at once, so the count is also
// bounded by sum ceil(q_s / bq) <= (total_q + (bq - 1) n) / bq. One S applies to every sequence of
// the launch. A launch over part of a step (cu past row 0) counts the step's rows in total_q, so
// the bound is loose there and may keep it unsplit.
// At `wgps` units or more the card is full: no split. Otherwise S rounds of equal workgroups cost
// rounds / S walks, so take one round (S1 = wgps / units) or two (S2 = 2 wgps / units), whichever
// is faster, the smaller S on a tie (fewer partials). At 32 WGPs, gqa 6 with 2 KV heads: 53 rows
// -> 16 splits, 80 -> 8, 345 -> 5, 576 -> 3, 1024 and up -> 1; gqa 12 with 1 KV head: up to 32
// rows -> 32, 53 -> 16, 345 -> 5. Each split keeps at least 8 tiles of the launch's live KV
// (seqused_k, falling back to max_ctx). Using the model bound here over-splits short prefills:
// a 128-token chunk at max_ctx 2048 became 4 splits of ~32 keys, while a 977-token unsplit
// launch of the same prompt did not, and the different reduction order moved logits. At most 64,
// the vectorised merge's limit.
static inline int r4d_prefill_splits_law(int bq, int tile, int q_len, int total_q, int num_seqs,
                                         int kv_heads, int max_ctx, int wgps) {
    return rad_prefill_splits_law(bq, tile, q_len, total_q, num_seqs, kv_heads, max_ctx, wgps);
}

// The split count of this unit's last prefill launch, 1 unsplit: read by the unit's test hook, which
// tells r4d_selftest a split launch from an unsplit one with the same bytes. Not atomic.
static int g_prefill_last_splits = 0;

/* RadArgs::kv_used_max_h is the host maximum of the launch's seqused operand (Ctx::issue_body),
 * -1 where it cannot be proven, which uses max_ctx below. In a decode, verify or draft pass, which is
 * recorded, it is the step's keyed context bound instead. Never
 * synchronously copy device seqused from a HIP queue independent of the AQL upload queue. */

/* The KV split's count before the scratch bound: the case's `splits` parameter when it names one
 * (r4d_selftest; the prefill rows declare no such axis, so the engine passes none), else the law,
 * at most 64. Use the runtime's host seqused maximum when exact for this launch; otherwise use the
 * declared model bound. The launcher and recording callback share that same rule. */
static inline int r4d_prefill_splits_want(const RadArgs* a, const R4DArgs* r, int bq, int tile) {
    const int sp0 = split_of(a);
    if (sp0 > 0) return std::min(sp0, 64);
    const int wgps = r4d_prefill_wgps(a);
    const int qtot = r4d_q_total(*r);
    const int used = a->kv_used_max_h;
    if (used > 0)
        return rad_prefill_splits_for_live(bq, tile, r->q_len, qtot, r->num_seqs, r->kv_heads,
                                           r->max_ctx, used, wgps);
    return r4d_prefill_splits_law(bq, tile, r->q_len, qtot, r->num_seqs, r->kv_heads,
                                  r->max_ctx, wgps);
}

/* The launch's split count, 1 for the plain form: what r4d_prefill_splits_want asks, cut to the
 * segments the caller's scratch holds, so a forced count past the declared worst case runs with as
 * many as fit rather than writing past the region. No scratch (a row without the hook, a direct
 * caller of the r4d_attn_prefill_* entry points) is the plain form. */
static inline int r4d_prefill_splits(const RadArgs* a, const R4DArgs* r, int bq, int tile) {
    if (!a->scratch || a->scratch_bytes <= 0) return 1;
    int sp = r4d_prefill_splits_want(a, r, bq, tile);
    const long per = r4d_decode_scratch(r, 1, 0);          /* one segment of every row, f32 */
    if (sp > 1 && per > 0 && a->scratch_bytes / per < sp) sp = (int)(a->scratch_bytes / per);
    return std::max(1, sp);
}

/* Bytes of the split's f32 partials: at declare, the most any launch in the band can take.
 *
 * Declare passes the band's upper bound as q_len and no tensors, so the worst case is swept over
 * every sequence count up to max_seqs and every row count up to the band, uniform (q_len = rows /
 * n) and ragged (q_len = min(band, rows), as fill_paged takes it); the law is not monotone in the
 * rows. The sweep stops where a launch fills the card. The law keeps units * S <= 2 wgps, so
 * rows * S * q_heads <= NWARPS * 16 * 2 wgps for every ratio: 24576 (row, head) pairs of 1032
 * bytes at head_dim 256 and 32 WGPs, 25.4 MB. A launch over part of a long step counts the whole
 * step's rows and may need more; r4d_prefill_splits then cuts S to what fits.
 *
 * Never an error, so the prefill row never loses its place: a shape it cannot read gets no
 * scratch, and its launches run the plain form. */
static inline int64_t r4d_prefill_scratch(const RadArgs* a, int head_dim, long long row, int bq,
                                          int tile) {
    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        const long long gqa = r4d_geti_or(a, "gqa", 1);
        const int kv   = (int)std::max(1LL, gqa > 0 ? n_head / gqa : n_head);
        const int band = (int)std::max(1LL, r4d_geti_or(a, "q_len", 1));
        const int mctx = (int)r4d_geti_or(a, "max_ctx", 0);
        const int w    = r4d_prefill_wgps(a);
        const int64_t per_row = (int64_t)n_head * (head_dim + 2) * 4;
        int64_t worst = 0;
        for (long long n = 1; n <= max_seqs; ++n)
            for (long long t = n; t <= band; ++t) {
                if ((long long)kv * ((t + bq - 1) / bq) >= w) break;
                int sp = r4d_prefill_splits_law(bq, tile, (int)std::min<long long>(band, t), (int)t,
                                                (int)n, kv, mctx, w);
                if (t % n == 0)
                    sp = std::max(sp, r4d_prefill_splits_law(bq, tile, (int)(t / n), (int)t, (int)n,
                                                              kv, mctx, w));
                if (sp > 1) worst = std::max(worst, (int64_t)t * sp * per_row);
            }
        return worst;
    }
    if (fill_paged_row(a, &r, head_dim, row) != RAD_OK) return 0;
    const int sp = r4d_prefill_splits_want(a, &r, bq, tile);
    return sp > 1 ? (int64_t)r4d_decode_scratch(&r, sp, 0) : 0;
}

// The split's merge: the decode merge over f32 partials (PF16 0), at the prefill tile, so it
// recomputes the kernel's partition; one wave a row from VECMIN rows (JPB rows a block), else the
// LDS-tree form, on the decode launchers' terms. SPAN, since the launch may cover part of a step's
// rows (ArPieces' pieces, a mixed step's prefill half).
template <int HEAD_DIM, int JPB, int VECMIN>
static inline int r4d_prefill_combine(const R4DArgs* a, int splits, int tile, hipStream_t s) {
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (splits <= 64 && jobs3 >= VECMIN)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_vec_kernel<HEAD_DIM, 4, 0, JPB, 0, 1>),
                           dim3((jobs3 + JPB - 1) / JPB), dim3(32 * JPB), 0, s, *a, splits, tile);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<HEAD_DIM, 4, 0, 0, 1>), dim3(jobs3),
                           dim3(HEAD_DIM), splits * sizeof(float), s, *a, splits, tile);
    const hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}
