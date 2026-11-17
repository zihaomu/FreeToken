/* rad_prefill_split.h -- host-pure live-KV prefill split law shared by libr4d launchers and
 * architecture pass-key callbacks. Keep this independent of HIP so callbacks can run before tape
 * lookup without a device copy or launch.
 */
#pragma once

#include <algorithm>
#include <cstdint>

/* `bq` query positions per workgroup, `tile` keys per KV tile. `units` counts the workgroups that
 * do work: a ragged batch's grid is sized from its longest sequence, but a shorter sequence's
 * tiles past its end exit at once. Each split keeps at least eight tiles of the supplied context.
 */
static inline int rad_prefill_splits_law(int bq, int tile, int q_len, int total_q, int num_seqs,
                                         int kv_heads, int max_ctx, int wgps) {
    const long n  = std::max(1, num_seqs);
    const long qt = std::min(n * ((std::max(1, q_len) + bq - 1) / bq),
                             (std::max(1, total_q) + (long)(bq - 1) * n) / bq);
    const long units = (long)std::max(1, kv_heads) * std::max(1L, qt);
    if (units >= wgps) return 1;
    const int s1 = (int)(wgps / units), s2 = (int)(2L * wgps / units);
    int sp = s2 > 2 * s1 ? s2 : s1;
    const int cap = ((max_ctx > 0 ? max_ctx : (1 << 30) - 1) + tile - 1) / tile / 8;
    sp = std::min(sp, std::max(1, cap));
    return std::max(1, std::min(sp, 64));
}

/* The launcher's live-context refinement of the initial model-bound law. */
static inline int rad_prefill_splits_for_live(int bq, int tile, int q_len, int total_q,
                                               int num_seqs, int kv_heads, int max_ctx,
                                               int used_ctx, int wgps) {
    int splits = rad_prefill_splits_law(bq, tile, q_len, total_q, num_seqs, kv_heads, max_ctx,
                                        wgps);
    if (splits > 1 && used_ctx > 0 && (max_ctx <= 0 || used_ctx < max_ctx))
        splits = rad_prefill_splits_law(bq, tile, q_len, total_q, num_seqs, kv_heads, used_ctx,
                                        wgps);
    return splits;
}

/* Fingerprint one actual launch tuple in packet-global sizes. Use this instead of the raw context
 * so equal split grids share a recording key as live context advances. */
static inline uint64_t rad_prefill_grid_key(int q_len, int bq, int splits, int kv_heads,
                                             int num_seqs, int threads) {
    const uint64_t gx = (uint64_t)((std::max(1, q_len) + bq - 1) / bq) *
                        (uint64_t)std::max(1, splits) * (uint64_t)threads;
    const uint64_t gy = (uint64_t)std::max(1, kv_heads);
    const uint64_t gz = (uint64_t)std::max(1, num_seqs);
    uint64_t h = 0x9e3779b97f4a7c15ull;
    const uint64_t grid[3] = {gx, gy, gz};
    for (uint64_t v : grid) {
        h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xbf58476d1ce4e5b9ull;
    }
    return h | 1;
}
