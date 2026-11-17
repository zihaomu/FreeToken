/* hostarena.cpp -- see hostarena.h. */
#include "mem/hostarena.h"

#include "rad_internal.h"

namespace rad {

HostArena::~HostArena() { close(); }

int HostArena::open(int64_t slot_bytes, int64_t cap_bytes) {
    close();
    if (slot_bytes <= 0 || cap_bytes < slot_bytes) return RAD_E_INVAL;
    const int64_t n = cap_bytes / slot_bytes;
    /* Pinned or nothing: an async copy to pageable memory serialises against the stream, so a
     * pageable fallback would run the store at a fraction of its budgeted rate. */
    base_ = rad_dev_alloc(n * slot_bytes, RAD_MEM_HOST_PINNED);
    if (!base_) return RAD_E_NOMEM;
    slot_bytes_ = slot_bytes;
    used_ = 0;
    cap_slots_ = n;
    cursor_ = 0;
    bits_.assign((size_t)((n + 63) / 64), 0ull);
    return RAD_OK;
}

void HostArena::close() {
    if (base_) rad_dev_free(base_, RAD_MEM_HOST_PINNED);
    base_ = nullptr;
    slot_bytes_ = 0;
    used_ = 0;
    cap_slots_ = 0;
    cursor_ = 0;
    bits_.clear();
}

/* First free run at or after the cursor, wrapping once. A fully used word is skipped in one
 * compare, so a nearly full arena costs one bitmap scan. The cursor makes consecutive calls hand
 * out consecutive slots, so a batch transfers as one copy per run. */
int32_t HostArena::alloc_run(int32_t want, int32_t* got) {
    *got = 0;
    if (want <= 0 || !base_ || full()) return -1;
    int64_t s = -1;
    for (int pass = 0; pass < 2 && s < 0; ++pass) {
        const int64_t lo = pass == 0 ? cursor_ : 0;
        const int64_t hi = pass == 0 ? cap_slots_ : cursor_;
        for (int64_t i = lo; i < hi;) {
            const uint64_t w = bits_[(size_t)(i >> 6)];
            if ((i & 63) == 0 && w == ~0ull) { i += 64; continue; }
            if (!is_used(i)) { s = i; break; }
            ++i;
        }
    }
    if (s < 0) return -1;
    int64_t n = 0;
    while (n < want && s + n < cap_slots_ && !is_used(s + n)) {
        bits_[(size_t)((s + n) >> 6)] |= 1ull << ((s + n) & 63);
        ++n;
    }
    used_ += n;
    cursor_ = s + n >= cap_slots_ ? 0 : s + n;
    *got = (int32_t)n;
    return (int32_t)s;
}

void HostArena::free(int32_t slot) {
    if (slot < 0 || slot >= cap_slots_) return;
    /* A double free is ignored so the used count, which `full` and the sessions view read, stays
     * correct. */
    if (!is_used(slot)) return;
    bits_[(size_t)(slot >> 6)] &= ~(1ull << (slot & 63));
    --used_;
}

void* HostArena::at(int32_t slot) const {
    if (!base_ || slot < 0 || slot >= cap_slots_) return nullptr;
    return (char*)base_ + (int64_t)slot * slot_bytes_;
}

}  /* namespace rad */
