/* bind.h -- the two views of a streamed group's one pool allocation.
 *
 * The stage view aliases every layer to offset 0 by reporting depth 1; `layer_slot` keeps each
 * layer's group-local position, because that is also its row in the host mirror
 * (docs/KV-STREAMING.md §3.5.1).
 *
 * A pass takes a view from its phase and row shape, before it is recorded or replayed. A decode
 * pass of at most 1 + n_spec rows a sequence takes the ring: its selection is rewritten to ring
 * slots and misses are copied in. Every other pass (prefill, mixed, and a decode-classified pass
 * with more rows a sequence, such as the serial drafter's history pass) takes the stage, the
 * group's context at block-id addressing, where no selection can overflow. `phase`, `n_tok`,
 * `n_seq` and `n_spec` are in the pass key, so a recorded pass replays against its view.
 */
#pragma once
#include "mem/kv.h"
#include "runtime/ctx.h"

namespace rad {

enum class KVView { Ring, Stage };

/* Fill both views from the carved plan and make the ring active. A non-paged group is left
 * untouched. */
void kv_bind_views(const KVGroupPlan& p, KVPoolBinding* b);

/* A layer's offset in the active view: 0 at depth 1. The `off + layer_stride <= bytes` bound in
 * issue.cpp holds for both views. */
int64_t kv_binding_offset(const KVPoolBinding& b, int64_t layer);

/* The view a pass with this phase and row shape takes (see the header). */
KVView kv_pass_view(int phase, int64_t n_tok, int64_t n_seq, int n_spec);

/* Make `view` active; the stage falls back to the ring when the group has none. */
void kv_binding_enter(KVPoolBinding& b, KVView view);

/* One rank's step view of a group's ring and mirror, copied into `out`; null when the group does
 * not stream. With `per_rank`, `src` is that rank's own manager's view: its maps live in that
 * rank's KV pool and its mirror is that rank's pinned allocation. Without (tests), `src` is rank
 * 0's and its map pointers are rebased from rank 0's pool (`kv0`) into this rank's (`kvr`). The
 * addresses are fixed at bring-up. */
const RadKVStream* kvstream_step_view(RadKVStream* out, const RadKVStream* src, bool per_rank,
                                      const void* kv0, const void* kvr);

/* A step view's identity for Ctx::pass_key: the memory a kernel reads, never the struct's address.
 * The batch copies the view into one of the builder's rotating Out slots, so keying on its address
 * would make every streamed decode step a new pass that never replays. */
uint64_t kvstream_view_key(const RadKVStream* v);

}  /* namespace rad */
