/* avx_kvstream.cpp -- kv_resolve: the residency map between a streamed KV group's block ids and
 * the ring slots its pool holds (docs/OPS.md "kv_resolve", docs/KV-STREAMING.md §3.4).
 *
 * libref/ref_kvstream.cpp is the contract, followed step for step: the same refusals in the same
 * order, the same epoch, miss order and CLOCK sweep over the same heat counter, so the same blocks
 * land in the same slots. A victim cannot be "close": `sel` becomes the attention's addressing.
 * A file of its own, as in libref, because the op belongs to no indexer: it moves bytes and keeps
 * state across calls.
 *
 * The rewrite of a `sel` row (a gather of `res[b]`) and the block copy (a whole number of 16-byte
 * lanes at every cache layout) are vector work. The lookup/claim pass, the claim gather and the
 * sweep are serial by nature: two lanes naming one absent block would both claim it, two naming
 * one slot would race on its epoch, and the sweep's moving hand is the placement. They cost the
 * selection and the slots, never `blocks`, and the copies dominate a miss.
 *
 * No OpenMP: the op's state makes one launch a serial unit, and two launches must never overlap.
 * Scratch is the arena (avx_common.h): `miss_block[slots]`, `miss_slot[slots]`, then
 * `claim[lookups]`, the gathered claim set this form sorts in place of a workgroup.
 */
#include "avx_common.h"
#include "avx_vec.h"

#include <algorithm>

using namespace avx;

namespace {

/* The heat counter's two writes, the same function in all three implementations (libref's
 * heat_up/heat_down): a use bumps by one and saturates at 3, a sweep pass halves, and a negative
 * value -- never written by a legal call -- reads as cold. */
int32_t kv_heat_up(int32_t h) { return h < 0 ? 1 : (h < 3 ? h + 1 : 3); }
int32_t kv_heat_down(int32_t h) { return h > 0 ? (h >> 1) : 0; }

/* `ctl`'s size and the index of its first u64 counter (three consecutive u64s from [4]). */
enum { kKvCtlInts = 16, kKvCtlMisses = 4, kKvCtlLookups = 6, kKvCtlCalls = 8 };
static_assert(kKvCtlLookups == kKvCtlMisses + 2 && kKvCtlCalls == kKvCtlMisses + 4,
              "the three counters are consecutive u64s from ctl[4]");

/* The byte pitch of a rank-2-or-deeper operand's rows, and whether its trailing extents fill that
 * pitch densely (libref's row_bytes). Both planes are one block a row: a padded row would copy a
 * block's bytes over its neighbour's. */
int64_t kv_row_bytes(const RadTensor* t, int64_t* dense) {
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) acc *= t->shape[i];
    *dense = acc;
    return t->stride[0] * rad_dtype_bytes(t->dtype, 1);
}

/* `block_bytes` of the mirror into a slot of the pool. A copy is a byte copy at every cache
 * layout (the op never reads an element), so the width is the ISA's, not the block's dtype. */
void kv_copy_block(uint8_t* dst, const uint8_t* src, int64_t n) {
    int64_t i = 0;
#if AVX_LEVEL >= 3
    for (; i + 64 <= n; i += 64)
        _mm512_storeu_si512((void*)(dst + i), _mm512_loadu_si512((const void*)(src + i)));
#elif AVX_LEVEL >= 2
    for (; i + 32 <= n; i += 32)
        _mm256_storeu_si256((__m256i*)(dst + i),
                            _mm256_loadu_si256((const __m256i*)(src + i)));
#elif AVX_LEVEL >= 1
    for (; i + 16 <= n; i += 16)
        _mm_storeu_si128((__m128i*)(dst + i), _mm_loadu_si128((const __m128i*)(src + i)));
#endif
    if (i < n) std::memcpy(dst + i, src + i, (size_t)(n - i));
}

/* The claim list: every distinct block the selection claimed (left -2 by the lookup pass), in
 * selection order, gathered from the `lookups` entries rather than a walk of `res` over the id
 * space. The map stays the dedup set: the first entry naming a claim marks it -3, and the marks
 * are restored before the sweep. `cap` is at least `lookups`, so every claim fits; the caller
 * sorts the list and the contract needs only its first `slots`. */
int64_t kv_claims(int32_t* res, int32_t* out, int64_t cap,
                  const int32_t* sel, const int32_t* nsel, int64_t M, int64_t pitch) {
    int64_t n = 0;
    for (int64_t q = 0; q < M; ++q) {
        const int32_t* const row = sel + q * pitch;
        for (int64_t i = 0; i < nsel[q]; ++i) {
            const int32_t b = row[i];
            if (res[b] != -2) continue;              /* a hit, or taken by an earlier entry */
            res[b] = -3;
            if (n < cap) out[n++] = b;
        }
    }
    for (int64_t k = 0; k < n; ++k) res[out[k]] = -2;
    return n;
}

/* Every claim the sweep could not place (still -2) goes back to -1, so a later call may claim it
 * again. Each is named by a selection entry, so one pass over the selection reaches all of them;
 * the write is idempotent. */
void kv_relax_claims(int32_t* res, const int32_t* sel, const int32_t* nsel, int64_t M,
                     int64_t pitch) {
    for (int64_t q = 0; q < M; ++q) {
        const int32_t* const row = sel + q * pitch;
        for (int64_t i = 0; i < nsel[q]; ++i)
            if (res[row[i]] == -2) res[row[i]] = -1;
    }
}

/* The rewrite: each entry below nsel[q] becomes its block's slot, or slot 0 for a block that
 * could not be placed (the id would be an address past the ring; ctl[3] marks the table unusable).
 * The gather is safe: validation refused every entry outside [0, blocks). */
void kv_rewrite_row(int32_t* row, int64_t n, const int32_t* res) {
    int64_t i = 0;
#if AVX_LEVEL >= 3
    const __m512i zero = _mm512_setzero_si512();
    for (; i + 16 <= n; i += 16) {
        const __m512i b = _mm512_loadu_si512((const void*)(row + i));
        const __m512i s = _mm512_mask_i32gather_epi32(zero, (__mmask16)0xFFFF, b,
                                                      (const void*)res, 4);
        /* A negative gather (-1 unplaced, -2 claimed) becomes slot 0, as the scalar tail writes it:
         * clamping the whole vector, since a masked store would leave the id in place. */
        _mm512_storeu_si512((void*)(row + i), _mm512_max_epi32(s, zero));
    }
#elif AVX_LEVEL >= 2
    const __m256i all = _mm256_set1_epi32(-1);
    const __m256i zero = _mm256_setzero_si256();
    for (; i + 8 <= n; i += 8) {
        const __m256i b = _mm256_loadu_si256((const __m256i*)(row + i));
        const __m256i s = _mm256_mask_i32gather_epi32(zero, res, b, all, 4);
        /* s > -1 is s >= 0; a negative lane becomes 0, never the block id. */
        const __m256i ok = _mm256_cmpgt_epi32(s, all);
        _mm256_storeu_si256((__m256i*)(row + i), _mm256_blendv_epi8(zero, s, ok));
    }
#endif
    for (; i < n; ++i) {
        const int32_t s = res[row[i]];
        row[i] = s >= 0 ? s : 0;
    }
}

}  /* namespace */

/* ------------------------------------------------------------------ kv_resolve */
/* One call, as libref's steps 1-7. `kv_cache` is inout: only the slots a miss lands in are
 * written, and every other slot keeps its resident block. */
AVX_KERNEL(kv_resolve) {
    if (!rad_args_have(a, 8)) return RAD_E_INVAL;
    const RadTensor *sel = &a->t[0], *nsel = &a->t[1], *res = &a->t[2], *sblk = &a->t[3];
    const RadTensor *sst = &a->t[4], *ctl = &a->t[5], *host = &a->t[6], *pool = &a->t[7];
    const int64_t M = p_int(a, "M", 0), blocks = p_int(a, "blocks", 0);
    const int64_t slots_decl = p_int(a, "slots", 0), pitch = p_int(a, "sel_pitch", 0);
    const int64_t bbytes = p_int(a, "block_bytes", 0);
    /* Zero rows is not a call (docs/OPS.md). */
    if (M <= 0) return RAD_E_SHAPE;
    if (blocks <= 0 || slots_decl <= 0 || pitch <= 0 || bbytes <= 0) return RAD_E_SHAPE;

    /* The maps and counts are one format: i32, dense. */
    const RadTensor* const state[] = { sel, nsel, res, sblk, sst, ctl };
    for (const RadTensor* t : state)
        if (t->dtype != RAD_I32) return RAD_E_DTYPE;
    if (sel->rank != 2 || sst->rank != 2) return RAD_E_SHAPE;
    if (sel->shape[0] < M || sel->shape[1] < pitch) return RAD_E_SHAPE;
    if (sel->stride[1] != 1 || sel->stride[0] != pitch) return RAD_E_STRIDE;
    if (numel(nsel) < M || nsel->stride[0] != 1) return RAD_E_SHAPE;
    if (numel(res) < blocks || res->stride[0] != 1) return RAD_E_SHAPE;
    /* The operand states the ring, the parameter its declared ceiling: the pool is planned after
     * declare, so a ring --kv-resident sized smaller arrives as a shorter `slot_block`. */
    const int64_t slots = numel(sblk);
    if (slots < 1 || slots > slots_decl || sblk->stride[0] != 1) return RAD_E_SHAPE;
    if (sst->shape[0] < slots || sst->stride[1] != 1 || sst->stride[0] < 2) return RAD_E_STRIDE;
    if (numel(ctl) < kKvCtlInts || ctl->stride[0] != 1) return RAD_E_SHAPE;
    if (host->rank < 2 || pool->rank < 2) return RAD_E_SHAPE;
    if (host->shape[0] < blocks || pool->shape[0] < slots) return RAD_E_SHAPE;

    /* The mirror's row and the pool's slot are both `block_bytes`, every copy's length. */
    int64_t hdense = 0, pdense = 0;
    if (kv_row_bytes(host, &hdense) != bbytes ||
        hdense * rad_dtype_bytes(host->dtype, 1) != bbytes)
        return RAD_E_STRIDE;
    if (kv_row_bytes(pool, &pdense) != bbytes ||
        pdense * rad_dtype_bytes(pool->dtype, 1) != bbytes)
        return RAD_E_STRIDE;

    int32_t* const R  = (int32_t*)res->data;
    int32_t* const SB = (int32_t*)sblk->data;
    int32_t* const ST = (int32_t*)sst->data;
    int32_t* const C  = (int32_t*)ctl->data;
    const int32_t* const NS = (const int32_t*)nsel->data;
    int32_t* const SE = (int32_t*)sel->data;
    const int64_t stp = sst->stride[0];
    /* The counters are read and written as u64 inside an i32 block. */
    if (((uintptr_t)C & 7u) != 0) return RAD_E_STRIDE;

    /* ---- everything that can refuse, before anything is written; the entry count summed here sizes
     * the counter and the claim list ---- */
    int64_t lookups = 0;
    for (int64_t q = 0; q < M; ++q) {
        const int64_t n = NS[q];
        if (n < 0 || n > pitch) return RAD_E_SHAPE;
        const int32_t* const row = SE + q * pitch;
        for (int64_t i = 0; i < n; ++i)
            if (row[i] < 0 || row[i] >= blocks) return RAD_E_INVAL;
        lookups += n;
    }
    /* The map's own domain, over the whole map: a state fault the contract refuses, and only a
     * look at the map can find an entry no selection names. Everything acting on a claim is
     * bounded by the selection. */
    for (int64_t b = 0; b < blocks; ++b)
        if (R[b] < -1 || R[b] >= slots) return RAD_E_INVAL;   /* -2 is never left by a call */
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < -1 || SB[s] >= blocks) return RAD_E_INVAL;
    if (C[1] < 0 || C[1] >= slots) return RAD_E_INVAL;

    /* ---- 1. the epoch ---- */
    const int32_t epoch = (int32_t)((uint32_t)C[0] + 1u);

    /* ---- 2. hits, and one claim per distinct miss (the map is its own set) ---- */
    for (int64_t q = 0; q < M; ++q) {
        const int64_t n = NS[q];
        const int32_t* const row = SE + q * pitch;
        for (int64_t i = 0; i < n; ++i) {
            const int32_t b = row[i];
            const int32_t s = R[b];
            if (s >= 0) {
                /* Stamp the slot and bump its heat once a slot a call (libref's step 2). */
                if (ST[s * stp] != epoch) {
                    ST[s * stp] = epoch;
                    ST[s * stp + 1] = kv_heat_up(ST[s * stp + 1]);
                }
            } else if (s == -1) {
                R[b] = -2;
            }
        }
    }

    /* ---- 3. the canonical miss order ---- */
    int32_t* const miss_block = (int32_t*)scratch_raw((size_t)(2 * slots + lookups) * sizeof(int32_t));
    int32_t* const miss_slot = miss_block + slots;
    int32_t* const claim = miss_slot + slots;
    const int64_t nclaims = kv_claims(R, claim, lookups, SE, NS, M, pitch);
    std::sort(claim, claim + nclaims);
    const int64_t nmiss = nclaims < slots ? nclaims : slots;
    for (int64_t k = 0; k < nmiss; ++k) miss_block[k] = claim[k];

    /* ---- 4. one victim per miss: identity, a free slot, then CLOCK, from the hand, never a slot
     * this call uses ---- */
    /* Identity first (libref's step 4): a context shorter than the ring gets res[b] = b, so the
     * attention's adjacent ids read adjacent slots. It takes only a free slot and leaves the hand
     * alone. Then free slots before any resident block; the free scan changes nothing when it
     * finds nothing, and a zero free count holds for the rest of the call, so a full ring does not
     * pay the scan per miss. */
    int64_t free_left = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < 0 && ST[s * stp] != epoch) ++free_left;
    int64_t hand = C[1];
    int64_t placed = 0;
    /* More claims than slots: some miss is unplaced by construction (the list is clipped at `slots`,
     * so the sweep cannot find out by running dry), and libref latches exactly that. */
    bool overflow = nclaims > slots;
    for (int64_t k = 0; k < nmiss; ++k) {
        const int32_t b = miss_block[k];
        int64_t victim = -1;
        if ((int64_t)b < slots && ST[(int64_t)b * stp] != epoch && (SB[b] < 0 || SB[b] == b))
            victim = b;
        if (victim < 0 && free_left > 0) {
            int64_t h = hand;
            for (int64_t scanned = 0; scanned < slots; ++scanned) {
                const int64_t s = h;
                h = h + 1 == slots ? 0 : h + 1;
                if (ST[s * stp] == epoch) continue;                 /* used by this call */
                if (SB[s] >= 0) continue;                           /* holds a block */
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
                if (ST[s * stp] == epoch) continue;                 /* used by this call */
                if (SB[s] >= 0 && ST[s * stp + 1] > 0) {             /* second chance */
                    ST[s * stp + 1] = kv_heat_down(ST[s * stp + 1]);
                    continue;
                }
                victim = s;
                break;
            }
        }
        if (victim < 0) {
            /* Every slot is stamped by this call: taking one would break the table it hands out. */
            overflow = true;
            break;
        }
        /* ---- 5. re-point both maps ---- */
        if (SB[victim] < 0 && free_left > 0) --free_left;            /* a free slot just filled */
        if (SB[victim] >= 0) R[SB[victim]] = -1;
        SB[victim] = b;
        ST[victim * stp] = epoch;
        ST[victim * stp + 1] = kv_heat_up(ST[victim * stp + 1]);
        R[b] = (int32_t)victim;
        miss_slot[k] = (int32_t)victim;
        ++placed;
    }
    kv_relax_claims(R, SE, NS, M, pitch);

    /* ---- 6. the copies, one block each, in miss order ---- */
    const uint8_t* const hb = (const uint8_t*)host->data;
    uint8_t* const pb = (uint8_t*)pool->data;
    for (int64_t k = 0; k < placed; ++k)
        kv_copy_block(pb + (int64_t)miss_slot[k] * bbytes,
                      hb + (int64_t)miss_block[k] * bbytes, bbytes);

    /* ---- 7. the rewrite: every valid entry becomes the slot holding its block ---- */
    for (int64_t q = 0; q < M; ++q) kv_rewrite_row(SE + q * pitch, NS[q], R);

    /* ---- the bookkeeping ---- */
    /* Distinct blocks: the slots this call stamped, plus the misses it could not place. Entry counts
     * would overcount, since rows share history. */
    int64_t used = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (ST[s * stp] == epoch) ++used;
    /* libref's union, including a call that overflowed, since the whole claim set is known. */
    const int64_t uni = used + nclaims - placed;
    C[0] = epoch;
    C[1] = (int32_t)hand;
    C[2] = (int32_t)placed;
    if (overflow) C[3] = 1;
    /* The maxima: the union, the part copied in, and the claim list's depth. */
    if (uni > C[10]) C[10] = (int32_t)uni;
    if (placed > C[11]) C[11] = (int32_t)placed;
    if (nclaims > C[12]) C[12] = (int32_t)nclaims;
    uint64_t* const ctr = (uint64_t*)(C + kKvCtlMisses);
    ctr[0] += (uint64_t)placed;
    ctr[1] += (uint64_t)lookups;
    ctr[2] += 1u;
    return RAD_OK;
}
