/* carve.h -- the streamed groups' part of KVManager::carve() (docs/KV-STREAMING.md §4.1, §4.2):
 * the resident windows and dense-read floors leave the budget before the other groups split the
 * rest; after the split each paged group's n_slots (and a streamed group's context and depth) is
 * written, refusing a streamed group with nothing to back its other blocks; and the pool report
 * gains a line naming both tiers. Arithmetic on the plans only.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rad {

struct KVGroupPlan;
class KVStreamStore;

/* Bytes the streamed groups' maps take from the KV pool (KVStreamStore::cost). They are reserved
 * before carve(), so the budget probe, which cannot reserve them, charges them in paged_split()
 * and the carve adds them back to its budget: both then see the same bytes. */
int64_t kvstream_maps_charged(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                              int64_t max_ctx, int64_t max_seqs);

/* Whether `p` is a dense-read group (rad_kv_group_dense_read: an indexer's block keys, read over
 * the whole history every step) while a sibling streams. Its pool must then hold every sequence's
 * context, since the sibling's mirror is sized for max_seqs. False when nothing streams: the cache
 * is then sized for one full-length sequence. */
bool kvstream_dense(const KVGroupPlan& p, const std::vector<KVGroupPlan>& plans,
                    const KVStreamStore& st);

/* The dense-read groups' full contexts in bytes, rounded as reserve() rounds them. */
int64_t kvstream_dense_charged(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                               int64_t max_ctx, int64_t max_seqs);

/* Take the streamed groups' windows, their maps and the dense-read groups' full contexts out of
 * the split before the other groups share the rest: `remaining` loses their bytes and
 * `out[group]` becomes the window's depth (or the dense-read context) in blocks. A streamed
 * group's context lives in host memory, so it does not compete for the pool. */
void kvstream_split_windows(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                            int64_t max_ctx, int64_t max_seqs, int64_t* remaining,
                            std::vector<int64_t>* out);

/* Refuse windows and dense-read floors that do not fit `budget`, before anything is split.
 * Otherwise the other groups would be left zero blocks and refused for a small pool, pointing the
 * operator at the wrong flag. The fixed costs were checked by the caller. A window group cannot
 * stream: its ids are its window's, not its context's. */
int kvstream_window_fit(const std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                        int64_t fixed, int64_t budget, int64_t max_ctx, int64_t max_seqs);

/* After the split: write every paged group's n_slots (its carve unless it streams) and, for a
 * streamed group, its context as n_blocks and its depth. Refuses a streamed group whose host
 * mirror or residency map is missing. */
int kvstream_apply(std::vector<KVGroupPlan>& plans, const KVStreamStore& st,
                   int64_t max_ctx, int64_t max_seqs);

/* The pool report's extra line for a streamed group, whose pool is smaller than its context. */
std::string kvstream_report_line(const KVGroupPlan& p);

}  /* namespace rad */
