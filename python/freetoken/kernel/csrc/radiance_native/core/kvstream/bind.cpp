/* bind.cpp -- see bind.h. */
#include "kvstream/bind.h"

#include "kvstream/plan.h"

namespace rad {

void kv_bind_views(const KVGroupPlan& p, KVPoolBinding* b) {
    if (!b || !p.paged()) return;
    b->base = p.base;

    b->ring.layer_stride = p.layer_stride();
    b->ring.bytes        = kv_ring_bytes(p, p.pool_blocks());
    b->ring.depth        = 0;
    b->ring.shape0       = p.pool_blocks();

    b->has_stage = p.depth == 1;
    if (b->has_stage) {
        b->stage.layer_stride = p.n_blocks * p.layer_bytes_per_block();
        b->stage.bytes        = b->stage.layer_stride;
        b->stage.depth        = 1;
        b->stage.shape0       = p.n_blocks;
    }

    kv_binding_enter(*b, KVView::Ring);
}

int64_t kv_binding_offset(const KVPoolBinding& b, int64_t layer) {
    return b.depth == 1 ? 0 : layer * b.layer_stride;
}

KVView kv_pass_view(int phase, int64_t n_tok, int64_t n_seq, int n_spec) {
    /* The serial drafter's history pass is labelled decode but runs over a prefill chunk's tokens,
     * whose union no ring bound covers, so it stages. */
    if (phase != RAD_PHASE_DECODE) return KVView::Stage;
    const int64_t rows = (int64_t)n_seq * (1 + (n_spec > 0 ? n_spec : 0));
    return n_tok > rows ? KVView::Stage : KVView::Ring;
}

void kv_binding_enter(KVPoolBinding& b, KVView view) {
    const KVViewState& v = (view == KVView::Stage && b.has_stage) ? b.stage : b.ring;
    b.layer_stride = v.layer_stride;
    b.bytes        = v.bytes;
    b.depth        = v.depth;
    b.shape[0]     = v.shape0;
}

namespace {
template <typename T>
T* on_rank(const void* base_r, const void* base_0, T* p0) {
    return p0 ? (T*)((char*)base_r + ((char*)p0 - (char*)base_0)) : nullptr;
}
}  /* namespace */

uint64_t kvstream_view_key(const RadKVStream* v) {
    if (!v) return 0;
    uint64_t h = 0x9e3779b97f4a7c15ull;
    auto mix = [&](uint64_t x) {
        h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h *= 0xbf58476d1ce4e5b9ull;
    };
    auto p = [&](const void* q) { mix((uint64_t)(uintptr_t)q); };
    p(v->host);       p(v->res);        p(v->slot_block);   p(v->slot_state);
    p(v->ctl);
    mix((uint64_t)v->block_bytes);      mix((uint64_t)v->host_layer_stride);
    mix((uint64_t)v->slots);            mix((uint64_t)v->host_bytes);
    return h | 1;
}

const RadKVStream* kvstream_step_view(RadKVStream* out, const RadKVStream* src, bool per_rank,
                                      const void* kv0, const void* kvr) {
    if (!src) return nullptr;
    *out = *src;
    /* Rebase a follower's maps into its own pool by the offset its arena buffers take. */
    if (!per_rank && kv0 && kvr) {
        out->res        = on_rank(kvr, kv0, src->res);
        out->slot_block = on_rank(kvr, kv0, src->slot_block);
        out->slot_state = on_rank(kvr, kv0, src->slot_state);
        out->ctl        = on_rank(kvr, kv0, src->ctl);
    }
    return out;
}

}  /* namespace rad */
