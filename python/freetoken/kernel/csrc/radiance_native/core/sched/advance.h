/* advance.h -- the card's own bookkeeping between passes: which drafts a verify step accepted, and
 * the inputs of every pass that follows from it.
 *
 * Why on the card: every pass after a verify depends on how many drafts it accepted (the draft
 * head continues from the accepted position, the next verify starts after it). Deciding that on
 * the host costs a readback and a sync per step with the card idle. On the card it is a compare
 * of a few integers per sequence, so the host can stage and issue every pass of a step, and the
 * next step, before any of them runs.
 *
 * State: one StepSlot per sequence slot, in device memory on every rank. It holds what the host
 * cannot know in advance: where the next verify starts, what it rolls back, which tokens it
 * verifies and where the draft head continues. Each rank computes the same numbers from the same
 * sampled tokens, so no rank reads another's memory.
 *
 * Device rows: a batch can mix host-staged rows (a prefill chunk, a history row at known
 * positions) with rows whose positions come from the card. `rows[0][i]` says which; the advance
 * leaves a host row as uploaded.
 *
 * On the host backend, device memory is host memory and the rows run on the issuing thread once
 * the stream has drained (advance_impl.h is shared by both builds).
 */
#pragma once
#include <cstdint>

#include "rad_device.h"
#include "draft.h"

namespace rad {

/* One sequence slot's state between passes. The accept step writes it; every later pass reads it. */
struct StepSlot {
    int32_t ctx;      /* the next verify step's first position: tokens committed so far */
    int32_t n_acc;    /* its one-based num_accepted: tokens the last verify step committed */
    int32_t head;     /* the last committed index, which the draft head's first round extends */
    int32_t acc;      /* drafts the last verify step accepted */
    /* The next verify step's input tokens: the bonus token, then the draft head's proposals. */
    int32_t tok[1 + RAD_SCHED_MAX_SPEC];
    /* A tree drafter's proposal (rad_drafter_tree): par[j] is the parent of node j + 1, a node
     * index in [0, j], where node 0 is the bonus token tok[0] and node j + 1 is tok[j + 1].
     * `ntree` is the number of nodes it describes, written by ADV_BTAKE from the drafter's
     * parents plane; 0 means tok[1..] is a chain. */
    int32_t par[RAD_SCHED_MAX_SPEC];
    int32_t ntree;
    /* The last verify's accepted root path: path[t] is the node committed (t + 1)-th, so the
     * committed tokens are tok[path[0]], tok[path[1]], ... and the step's anchor sits at
     * ctx - 1 - npath. ADV_COMPACT moves those nodes' K and V from where the verify stored them
     * (its row index) to where they belong (their depth), and ADV_CTXTREE positions the
     * drafter's context rows by it. 0 after a chain verify, which needs neither. */
    int32_t npath;
    int32_t path[RAD_SCHED_MAX_SPEC];
    /* The last verify step's row of its last committed position, in that step's own row order,
     * which a block drafter's context pass mirrors. ADV_BLOCK hands it to the query pass
     * (RadBatch::anchor_rows): LiLiCorr's correlator reads the context state there. */
    int32_t hrow;
    int32_t pad[1];
};

enum AdvMode : int32_t {
    ADV_ACCEPT = 1,   /* after a verify step's sampler: acceptance, and the state it leaves */
    ADV_TRUNK  = 2,   /* before a verify step: its tokens, positions and rollback count */
    ADV_HIST   = 3,   /* before the draft head's history pass: the verified positions */
    ADV_ROUND  = 4,   /* before draft round `round` >= 2: the next position, the last proposal */
    ADV_TAKE   = 5,   /* after the last round: its proposal into the state */
    /* A block drafter (RAD_DRAFT_BLOCK) on the card: */
    ADV_BLOCK  = 6,   /* before the query pass: the anchor at the committed end, then the masks */
    ADV_BTAKE  = 7,   /* after it: its `round` proposals into the state, behind the bonus token */
    /* A tree drafter (rad_drafter_tree), whose verify rows are a tree rather than a chain: */
    ADV_CTXTREE = 8,  /* before the context pass: the accepted path's rows at their depth, the
                       * rejected rows stored nowhere */
};

/* The most paged KV groups one ADV_COMPACT moves (the trunk's sliding and global groups and a
 * drafter's own, with room). */
enum { kAdvMaxPools = 6 };

/* One paged group's cache on one rank, as ADV_COMPACT addresses it: `n_layers` caches of
 * [n_blocks, n_head, block_size, row_bytes] bytes, `layer_stride` apart from `base`
 * (core/runtime/kvbind.cpp composes that shape). `grp` indexes the batch's AdvGroup array, whose
 * block table maps a position to a block. */
struct AdvPool {
    char*   base;
    int64_t layer_stride;
    int32_t n_layers;
    int32_t n_head;
    int32_t row_bytes;
    int32_t grp;
};

/* One KV group, as the builder stages it for this build. Offsets are bytes into the rank's slab,
 * which every rank carves identically. */
struct AdvGroup {
    int32_t block_size;
    int32_t paged;          /* 0: a state group, whose slots do not depend on the position */
    int32_t pitch;          /* this build's block-table pitch */
    int32_t pad;
    int64_t slot_off;       /* [n_tok] slot_mapping */
    int64_t used_off;       /* [n_seq] seqused */
    int64_t bt_off;         /* [n_seq][pitch] block table */
    int64_t first_off;      /* [n_seq] the position each row's block table starts at */
    /* The runtime window (KVManager::set_window / SeqWindow), so the card computes the same
     * `used` and slots the host builder staged: a ring sequence keeps its reference and rewrites
     * a fixed ring of blocks, so its held span stops at its own table and a position past its
     * ring_pos maps modulo ring_blocks. Without it (ring 0) the advance is the linear mapping, as
     * before. ring_pos and the table length are per sequence (each prompt has its own floor and
     * table), ring_blocks is the same for the group (one W). */
    int32_t ring;           /* 1: the group carries runtime windows */
    int32_t ring_blocks;    /* whole blocks the ring holds */
    int64_t rpos_off;       /* [n_seq] the sequence's block-aligned first ring position, 0 if none */
    int64_t tlen_off;       /* [n_seq] the sequence's block-table length in blocks, 0 if none */
};

/* One advance: a mode, the build it patches, and where its inputs are. Passed by value as the
 * kernel's argument. */
struct AdvanceArgs {
    int32_t mode = 0;
    int32_t round = 0;        /* ADV_ROUND: the draft round, from 2; ADV_TAKE: the last, from 1;
                               * ADV_BTAKE: how many proposals a row */
    int32_t n_seq = 0;
    int32_t n_groups = 0;
    char*   slab = nullptr;   /* this rank's device slab */
    int64_t tok_off = 0, pos_off = 0, ctx_off = 0, acc_off = 0, out_off = 0;
    int64_t arow_off = -1;    /* [n_seq] ADV_BLOCK: each row's StepSlot::hrow; -1 when not carved */
    int64_t sid_off = 0, cu_off = 0, q_off = 0;
    /* [4][row_pitch] int32: device-row flag; the verify step's token row of the row's first
     * verified position; its first sampled row, or -1 for a row that sampled nothing; and one
     * number the mode reads -- ADV_ACCEPT the row's draft count, ADV_HIST how many of the row's
     * positions come before the verified ones, whose tokens the host staged, ADV_BLOCK the token
     * id its mask rows embed. */
    int64_t row_off = 0;
    int32_t row_pitch = 0;
    int64_t grp_off = 0;      /* [n_groups] AdvGroup */
    StepSlot* st = nullptr;   /* [max_seqs], by scheduler slot */
    /* ADV_ACCEPT and ADV_HIST: the verify step's sampled tokens, one a row. ADV_ROUND and
     * ADV_TAKE: the last round's proposals, row i at i * src_pitch. ADV_BTAKE: the block's
     * proposals, row i's from i * src_pitch. */
    const int32_t* src = nullptr;
    int64_t src_pitch = 1;
    const int32_t* eog = nullptr;   /* end-of-generation ids, sorted */
    int32_t n_eog = 0;
    /* Rotary planes (RadBatch::rope_pos), [3][rope_pitch] at rope_off; rope_off -1 when the build
     * has none. A row the card moves keeps the host-staged offset between its position and its
     * rotary position: card-owned rows are decode and draft rows, all text, which rotate at their
     * index plus a per-sequence shift. A row left at the host's position keeps the host's three
     * components, as a media token needs. */
    int64_t rope_off = -1;
    int32_t rope_pitch = 0;
    /* A serial head with no cache of its own (rad_drafter_shared_kv), read by ADV_HIST and
     * ADV_ROUND only: its history pass is one row per sequence at the committed position, and
     * every round sits there too, with context ending at that position. `rot_shift` is the rotary
     * position of those rows, 0 or 1 past the committed position. Zero for every other head. */
    int32_t shared_kv = 0;
    int32_t rot_shift = 0;
    /* A tree drafter (rad_drafter_tree). ADV_BTAKE: the proposal's parents plane, row i's from
     * i * src_pitch as the tokens are; null for a chain drafter. */
    const int32_t* par = nullptr;
    /* ADV_TRUNK: RadBatch::spec_parent [n_tok] in the slab, or -1 when the build stages none.
     * A tree row's entry is its parent's row within the sequence's block (-1 for the bonus
     * token); every other row's is -2, an ordinary causal row. */
    int64_t parent_off = -1;
    /* ADV_COMPACT: every paged group's cache on this rank. */
    AdvPool pool[kAdvMaxPools] = {};
    int32_t n_pool = 0;
};

/* Issue one advance on `s`. The rows are independent, so the card runs them in parallel. */
int step_advance(const AdvanceArgs& a, RadStream s);

/* ADV_COMPACT, after ADV_ACCEPT on a tree verify and before anything reads the trunk's cache: for
 * every row whose StepSlot holds an accepted path, move the K and V each path node's verify row
 * stored (at the anchor's position + its row index) to its committed position (+ its depth), in
 * every pool of `a`, every layer and every head. A separate kernel from the advance because it
 * moves bytes across the whole cache rather than patching a row's few integers. */
int step_compact(const AdvanceArgs& a, RadStream s);

}  /* namespace rad */

/* The launcher in advance.hip: one kernel onto `stream` through hipLaunchKernel. */
extern "C" int rad_step_advance_launch(const rad::AdvanceArgs* args, void* stream);
/* And ADV_COMPACT's. */
extern "C" int rad_step_compact_launch(const rad::AdvanceArgs* args, void* stream);
