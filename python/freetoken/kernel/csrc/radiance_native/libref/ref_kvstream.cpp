/* ref_kvstream.cpp -- kv_resolve: the residency map between a streamed KV group's block ids and
 * the ring slots its device pool holds. The operands, layouts and contract are in docs/OPS.md
 * ("kv_resolve"); docs/KV-STREAMING.md §3.4 is the design.
 *
 * `qsa_select` writes pool page ids into `sel` and `attn_paged` uses each entry as a raw pool
 * index. When the group streams, the pool is `slots` blocks deep while the id space is `blocks`, so
 * kv_resolve rewrites every selected id as the ring slot holding it, copying a block in from the
 * host mirror when it is absent. `sel` is the only place the sparse path names KV storage, so the
 * attention is unchanged; `nsel` and `seqused` are token counts and stay as written.
 *
 * Parameters: `M` (ranged), `blocks` (the id space), `slots` (the declared ring), `sel_pitch`
 * (topk + 1) and `block_bytes` (one block of one layer: the mirror's row pitch, the ring's slot
 * pitch and every copy's length). Bytes rather than `block_size`, because the op never reads an
 * element and is the same code at every cache layout, and because Builder::find_kv_consumer binds
 * a group to the first op carrying `block_size`, which must stay the attention. The operands state
 * the pitches too, and must agree. `slot_block`'s extent is the ring the call may address; a call
 * deeper than the declared `slots` is refused.
 *
 * The miss list is the row's scratch, not an operand: a device row declares
 * `(next_pow2(slots) + slots) * 4` bytes (`miss_block[next_pow2(slots)]`, padded for an in-place
 * sort, then `miss_slot[slots]`). This reference sorts locally.
 *
 * State, which the caller brings up: `res` and `slot_block` filled with -1, `slot_state` and `ctl`
 * zeroed. There is no reset op; the maps live as long as the pool. A call entering with a `res`
 * entry of -2 or outside [-1, slots), a `slot_block` entry outside [-1, blocks), or a hand outside
 * [0, slots) is refused as a state fault. `ctl` is 16 i32s, 8-byte aligned:
 *
 *   [0] epoch of the last call   [1] CLOCK hand   [2] blocks placed by the last call
 *   [3] overflow, latched        [4..5] u64 misses   [6..7] u64 lookups   [8..9] u64 calls
 *   [10] largest union of one call (distinct blocks, hits + misses)
 *   [11] largest placement of one call
 *   [12] largest claim count of one call: the depth the miss list must hold, at most
 *        `M * sel_pitch`; a device kernel sizes its list from those operands, and this says
 *        whether that was enough
 *
 * The epoch is a wrapping i32 compared only for equality. `slot_state[s,1]` is a two-bit heat
 * counter in [0, 3], kept in the i32.
 *
 * One call:
 *
 *  1. epoch = ctl[0] + 1.
 *  2. One lookup per distinct block, rows then entries ascending, over [0, nsel[q]). A hit
 *     (res[b] >= 0) stamps the slot with the epoch and, once a slot a call, bumps its heat
 *     (clamped at 3), so heat counts calls that wanted the block, not selection width. A miss
 *     (res[b] == -1) is claimed: res[b] = -2 and b joins the miss list. -2 is already claimed; the
 *     map is its own set.
 *  3. The miss list is sorted by block id, so two implementations place the same blocks in the
 *     same slots whatever their thread order.
 *  4. One victim per miss, in that order. Identity first: slot b when b < slots and the slot is
 *     free or holds b, and not stamped by this call; the hand does not move. It never evicts, keeps
 *     a context shorter than the ring at identity, and puts adjacent ids in adjacent slots. Else
 *     walk from the hand: a free slot is taken before any resident block, a slot stamped this call
 *     is passed, and once a whole turn finds nothing free the walk is CLOCK with the heat counter
 *     as its reference: a counter above 0 is halved and passed, a cold slot is the victim. A 3
 *     survives two passes, so the sweep visits at most 3 * slots slots. The hand ends one past the
 *     last slot taken. The free scan changes nothing when it finds nothing.
 *  5. Re-point both maps as each victim is taken (res[old] = -1, slot_block[s] = b, res[b] = s,
 *     slot_state[s] = {epoch, min(heat + 1, 3)}): a placement is a use.
 *  6. Copy block_bytes from host[b] into slot s, as bytes.
 *  7. Rewrite every entry in [0, nsel[q]) as res[b]; the -1 padding past it is kept.
 *
 * Then ctl[0..2] take the epoch, hand and placement, the counters grow (lookups count entries), and
 * the maxima are updated.
 *
 * Overflow: the ring must hold every distinct block one call selects. At decode that is at most
 * `max_seqs * (sel_pitch - 1)`; a prefill launch above `qsa_exact_to` can select a chunk's history,
 * which is why the engine reads ctl[10] rather than assuming a bound. A miss the sweep cannot place
 * reverts to res[b] = -1, ctl[3] latches, and its entries are rewritten as slot 0: a defined
 * address inside the ring, where the block id would be a device address past it. The step's
 * numbers are then wrong, so a caller that finds ctl[3] set must discard the step.
 *
 * The op is stateful (epoch, hand, heat, counters), so two launches must never overlap: its rows
 * are not in libr4d's rad_kernel_concurrent list, which orders each behind everything before it.
 * It is still recordable: it is a function of device state whose addresses and pitches are fixed
 * from bring-up.
 *
 * This is the serial form of the contract, the oracle two device implementations must agree with
 * on which slot every block lands in.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cstring>
#include <vector>

using namespace ref;

namespace {

/* The heat counter's two writes, the same function in all three implementations: a use bumps by
 * one and saturates at 3, a sweep pass halves, and a negative value (never written by a legal call)
 * reads as cold rather than letting it survive every pass. */
int32_t heat_up(int32_t h) { return h < 0 ? 1 : (h < 3 ? h + 1 : 3); }
int32_t heat_down(int32_t h) { return h > 0 ? (h >> 1) : 0; }

/* `ctl`'s size, and the index of its first counter (three u64s from [4]). */
enum { kKvCtlInts = 16, kKvCtlMisses = 4, kKvCtlLookups = 6, kKvCtlCalls = 8 };
static_assert(kKvCtlLookups == kKvCtlMisses + 2 && kKvCtlCalls == kKvCtlMisses + 4,
              "the three counters are consecutive u64s from ctl[4]");

/* The byte pitch of a rank-2-or-deeper operand's rows, and whether its trailing extents fill
 * exactly `bytes` densely. Both the mirror and the pool are one block a row; a padded row would
 * copy a block's bytes over its neighbour's. */
int64_t row_bytes(const RadTensor* t, int64_t* dense) {
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) acc *= t->shape[i];
    *dense = acc;
    return t->stride[0] * rad_dtype_bytes(t->dtype, 1);
}

}  /* namespace */

int ref_kv_resolve(const RadArgs* a, RadStream) {
    if (!have(a, 8)) return RAD_E_INVAL;
    const RadTensor *sel = &a->t[0], *nsel = &a->t[1], *res = &a->t[2], *sblk = &a->t[3];
    const RadTensor *sst = &a->t[4], *ctl = &a->t[5], *host = &a->t[6], *pool = &a->t[7];
    const int64_t M = p_int(a, "M", 0), blocks = p_int(a, "blocks", 0);
    const int64_t slots_decl = p_int(a, "slots", 0), pitch = p_int(a, "sel_pitch", 0);
    const int64_t bbytes = p_int(a, "block_bytes", 0);
    /* Zero rows is not a call (docs/OPS.md). */
    if (M <= 0) return RAD_E_SHAPE;
    if (blocks <= 0 || slots_decl <= 0 || pitch <= 0 || bbytes <= 0) return RAD_E_SHAPE;

    /* The maps and counts are one format: i32, dense. */
    const RadTensor* state[] = { sel, nsel, res, sblk, sst, ctl };
    for (const RadTensor* t : state)
        if (t->dtype != RAD_I32) return RAD_E_DTYPE;
    if (sel->rank != 2 || sst->rank != 2) return RAD_E_SHAPE;
    if (sel->shape[0] < M || sel->shape[1] < pitch) return RAD_E_SHAPE;
    if (sel->stride[1] != 1 || sel->stride[0] != pitch) return RAD_E_STRIDE;
    if (numel(nsel) < M || nsel->stride[0] != 1) return RAD_E_SHAPE;
    if (numel(res) < blocks || res->stride[0] != 1) return RAD_E_SHAPE;
    /* The operand states the ring; the parameter is the ceiling it may not pass. */
    const int64_t slots = numel(sblk);
    if (slots < 1 || slots > slots_decl || sblk->stride[0] != 1) return RAD_E_SHAPE;
    if (sst->shape[0] < slots || sst->stride[1] != 1 || sst->stride[0] < 2) return RAD_E_STRIDE;
    if (numel(ctl) < kKvCtlInts || ctl->stride[0] != 1) return RAD_E_SHAPE;
    if (host->rank < 2 || pool->rank < 2) return RAD_E_SHAPE;
    if (host->shape[0] < blocks || pool->shape[0] < slots) return RAD_E_SHAPE;

    /* The mirror's row and the pool's slot are both `block_bytes`. */
    int64_t hdense = 0, pdense = 0;
    if (row_bytes(host, &hdense) != bbytes || hdense * rad_dtype_bytes(host->dtype, 1) != bbytes)
        return RAD_E_STRIDE;
    if (row_bytes(pool, &pdense) != bbytes || pdense * rad_dtype_bytes(pool->dtype, 1) != bbytes)
        return RAD_E_STRIDE;

    int32_t* const R  = (int32_t*)res->data;
    int32_t* const SB = (int32_t*)sblk->data;
    int32_t* const ST = (int32_t*)sst->data;
    int32_t* const C  = (int32_t*)ctl->data;
    const int64_t stp = sst->stride[0];
    /* The counters are read and written as u64 in an i32 block. */
    if (((uintptr_t)C & 7u) != 0) return RAD_E_STRIDE;

    /* ---- everything that can refuse, before anything is written; the selection is read once and
     * kept for step 7 ---- */
    std::vector<int64_t> off, ent;
    for (int64_t q = 0; q < M; ++q) {
        const int64_t n = ld_int(nsel->dtype, nsel->data, q);
        if (n < 0 || n > pitch) return RAD_E_SHAPE;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t b = ld_int(sel->dtype, sel->data, q * pitch + i);
            if (b < 0 || b >= blocks) return RAD_E_INVAL;
            off.push_back(q * pitch + i);
            ent.push_back(b);
        }
    }
    for (int64_t b = 0; b < blocks; ++b)
        if (R[b] < -1 || R[b] >= slots) return RAD_E_INVAL;   /* -2 is never left by a call */
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < -1 || SB[s] >= blocks) return RAD_E_INVAL;
    if (C[1] < 0 || C[1] >= slots) return RAD_E_INVAL;

    /* ---- 1. the epoch ---- */
    const int32_t epoch = (int32_t)((uint32_t)C[0] + 1u);

    /* ---- 2. hits, and one claim per distinct miss (the map is its own set) ---- */
    std::vector<int32_t> miss;
    for (size_t e = 0; e < ent.size(); ++e) {
        const int32_t b = (int32_t)ent[e];
        const int32_t s = R[b];
        if (s >= 0) {
            /* One bump a slot a call (step 2). */
            if (ST[s * stp] != epoch) {
                ST[s * stp] = epoch;
                ST[s * stp + 1] = heat_up(ST[s * stp + 1]);
            }
        } else if (s == -1) {
            R[b] = -2;
            miss.push_back(b);
        }
    }
    const int64_t lookups = (int64_t)ent.size();

    /* ---- 3. the canonical miss order ---- */
    std::sort(miss.begin(), miss.end());

    /* ---- 4. one victim per miss: identity, a free slot, then CLOCK, from the hand, never a slot
     * this call uses ---- */
    /* Identity first (step 4), leaving the hand alone. Then free slots, before any resident block
     * is evicted; a free scan that finds nothing changes nothing. A free count of zero holds for the
     * rest of the call (placements fill slots, evictions re-point them), so a full ring goes
     * straight to CLOCK. */
    int64_t free_left = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < 0 && ST[s * stp] != epoch) ++free_left;
    int64_t hand = C[1];
    int64_t placed = 0;
    bool overflow = false;
    for (size_t k = 0; k < miss.size(); ++k) {
        const int32_t b = miss[k];
        int64_t victim = -1;
        if ((int64_t)b < slots && ST[(int64_t)b * stp] != epoch && (SB[b] < 0 || SB[b] == b))
            victim = b;
        if (victim < 0 && free_left > 0) {
            int64_t h = hand;
            for (int64_t scanned = 0; scanned < slots; ++scanned) {
                const int64_t s = h;
                h = h + 1 == slots ? 0 : h + 1;
                if (ST[s * stp] == epoch) continue;                  /* used by this call */
                if (SB[s] >= 0) continue;                            /* holds a block */
                victim = s;
                hand = h;
                break;
            }
            if (victim < 0) free_left = 0;
        }
        if (victim < 0) {
            for (int64_t scanned = 0; scanned < 3 * slots; ++scanned) {
                const int64_t s = hand;
                hand = hand + 1 == slots ? 0 : hand + 1;
                if (ST[s * stp] == epoch) continue;                  /* used by this call */
                if (SB[s] >= 0 && ST[s * stp + 1] > 0) {             /* second chance */
                    ST[s * stp + 1] = heat_down(ST[s * stp + 1]);
                    continue;
                }
                victim = s;
                break;
            }
        }
        if (victim < 0) { overflow = true; break; }
        /* ---- 5. re-point both maps ---- */
        if (SB[victim] < 0 && free_left > 0) --free_left;            /* a free slot just filled */
        if (SB[victim] >= 0) R[SB[victim]] = -1;
        SB[victim] = b;
        ST[victim * stp] = epoch;
        ST[victim * stp + 1] = heat_up(ST[victim * stp + 1]);
        R[b] = (int32_t)victim;
        ++placed;
    }
    /* An unplaced miss is not resident: undo the claim so a later call may claim it again. */
    for (size_t k = (size_t)placed; k < miss.size(); ++k) R[miss[k]] = -1;

    /* ---- 6. the copies, one block each, in miss order ---- */
    const uint8_t* const hb = (const uint8_t*)host->data;
    uint8_t* const pb = (uint8_t*)pool->data;
    for (int64_t k = 0; k < placed; ++k) {
        const int32_t b = miss[(size_t)k];
        std::memcpy(pb + (int64_t)R[b] * bbytes, hb + (int64_t)b * bbytes, (size_t)bbytes);
    }

    /* ---- 7. the rewrite: every valid entry becomes the slot holding its block ---- */
    for (size_t e = 0; e < ent.size(); ++e) {
        const int32_t s = R[ent[e]];
        /* An unplaced miss's entry names slot 0, inside the ring, not its block id (see the header's
         * overflow note); ctl[3] says the table is unusable. */
        st_int(sel->dtype, sel->data, off[e], s >= 0 ? s : 0);
    }

    /* ---- the bookkeeping ---- */
    /* The union counts distinct blocks: the slots this call stamped (every hit and placement) plus
     * the misses it could not place. */
    int64_t used = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (ST[s * stp] == epoch) ++used;
    const int64_t uni = used + (int64_t)miss.size() - placed;
    C[0] = epoch;
    C[1] = (int32_t)hand;
    C[2] = (int32_t)placed;
    if (overflow) C[3] = 1;
    /* The maxima: the union this call selected and the part of it copied in. */
    if (uni > C[10]) C[10] = (int32_t)uni;
    if (placed > C[11]) C[11] = (int32_t)placed;
    if ((int64_t)miss.size() > C[12]) C[12] = (int32_t)miss.size();
    uint64_t* const ctr = (uint64_t*)(C + kKvCtlMisses);
    ctr[0] += (uint64_t)placed;
    ctr[1] += (uint64_t)lookups;
    ctr[2] += 1u;
    return RAD_OK;
}
