/* ref_collective.cpp -- all_reduce and all_gather on the host backend.
 *
 * At world_size 1 all_reduce is a no-op (inout) and all_gather a copy. Above 1 they rely on the
 * engine's shape: one process, one thread per rank, one address space (spec §1). A peer's buffer is
 * a pointer, so a collective is a rendezvous, a read of every rank's pointer, and a second
 * rendezvous before anyone overwrites theirs.
 *
 * This makes the host backend a fake multi-device backend, so tensor-parallel paths (sharded
 * declare, collective placement, distributed top-k) can run without an accelerator. It does not
 * test a device all-reduce's numerics.
 *
 * Reduction order is ascending rank, in f32, on every rank (libr4d's rule). Tensor-parallel ranks
 * hold replicated state that must stay bit-identical; different summation orders would drift apart
 * over a generation. -ffp-contract=off is part of the same rule (docs/OPS.md).
 *
 * exact=0 (caller accepts a lossy quantised wire) also gets the exact sum, since there is no wire;
 * the constraint only forbids handing a lossy result to a caller that asked for exact. `hops` is
 * ignored: one-shot and two-shot are the same walk when peers are pointers.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstring>

using namespace ref;

namespace {

enum { REF_MAX_RANKS = 8 };

/* One group per process. A process has one world size and the rank threads run the same graph in
 * lockstep, so every rank issues the same sequence of collectives and one generation-counting
 * barrier pairs them. A per-op group would need a key the ABI does not carry: init() cannot tell
 * two ops apart from two ranks of one op. */
struct Group {
    std::mutex              m;
    std::condition_variable cv;
    int          world   = 0;
    int          members = 0;   /* initialised instances not yet finalised (see join) */
    int          arrived = 0;
    unsigned     gen     = 0;
    const void*  slot[REF_MAX_RANKS] = {};
    int64_t      nel [REF_MAX_RANKS] = {};
    uint32_t     dt  [REF_MAX_RANKS] = {};
};

static Group& group() { static Group g; return g; }

/* How long a rank waits for its peers. A collective must block, but an unbounded wait would turn a
 * mismatched call sequence into a silent hang; the engine exits non-zero instead (spec §17). 30 s is
 * far longer than any correct step and shorter than a CI timeout. ref_collective_timeout_ms sets it
 * for a test. */
static std::atomic<int> g_timeout_ms{ 30000 };

/* Returns false on timeout. A rank that gives up withdraws its arrival, so the group's count is
 * again that of the ranks still waiting and the next collective pairs from a clean start. */
static bool barrier(Group& g) {
    std::unique_lock<std::mutex> lk(g.m);
    const unsigned mygen = g.gen;
    if (++g.arrived == g.world) {
        g.arrived = 0;
        ++g.gen;
        g.cv.notify_all();
        return true;
    }
    const bool met = g.cv.wait_for(lk, std::chrono::milliseconds(g_timeout_ms.load()),
                                   [&] { return g.gen != mygen; });
    if (!met) --g.arrived;
    return met;
}

/* The world size is fixed while it is in use: while an initialised instance is alive or a rank is
 * mid-collective. Otherwise the next join may state another one, so a process can run one group
 * after another (a re-initialised engine, a test). `member` counts an init, which fini releases. */
static int join(int rank, int world, bool member) {
    if (world <= 0 || world > REF_MAX_RANKS) return RAD_E_UNSUPPORTED;
    if (rank < 0 || rank >= world) return RAD_E_INVAL;
    Group& g = group();
    std::lock_guard<std::mutex> lk(g.m);
    if (g.world == 0 || (g.members == 0 && g.arrived == 0)) g.world = world;
    else if (g.world != world) return RAD_E_INVAL;
    if (member) ++g.members;
    return RAD_OK;
}

/* Publishes this rank's input and waits until every rank has published a matching one. Peers read
 * each other's input as `nel` contiguous elements, since only the pointer is published. */
static int publish(int rank, int world, const RadTensor* x, int64_t nel) {
    if (!rad_tensor_is_contiguous(x)) return RAD_E_STRIDE;
    RAD_REF_TRY(join(rank, world, false));
    Group& g = group();
    {
        std::lock_guard<std::mutex> lk(g.m);
        g.slot[rank] = x->data;
        g.nel [rank] = nel;
        g.dt  [rank] = x->dtype;
    }
    if (!barrier(g)) return RAD_E_DEVICE;
    for (int r = 0; r < world; ++r) {
        if (!g.slot[r] || g.nel[r] != nel) return RAD_E_SHAPE;
        if (g.dt[r] != x->dtype) return RAD_E_DTYPE;
    }
    return RAD_OK;
}

/* Staging, so a rank does not overwrite its buffer while a peer still reads it. Uses the arena's
 * scratch when large enough, else the per-thread workspace (for direct callers that pass none). */
static float* staging(const RadArgs* a, int64_t nel) {
    if (a->scratch && a->scratch_bytes >= nel * (int64_t)sizeof(float))
        return (float*)a->scratch;
    return ws_get((size_t)nel);
}

}  /* namespace */

int ref_collective_init(const RadParam* p, int n_p, int rank, int world_size,
                        void** out) {
    /* The geometry's world size wins: declare states it and RadBuildCtx divided the model by it
     * (spec §9). */
    int ws = world_size;
    for (int i = 0; i < n_p; ++i)
        if (p[i].key && std::strcmp(p[i].key, "world_size") == 0 && p[i].kind == RAD_P_INT)
            ws = (int)p[i].ival;
    if (out) *out = nullptr;
    if (ws <= 1) return RAD_OK;
    RAD_REF_TRY(join(rank, ws, true));
    if (out) *out = &group();
    return RAD_OK;
}

/* Pairs with an init that joined (its instance is the group); one that did not has none. */
void ref_collective_fini(void* inst) {
    if (!inst) return;
    Group& g = group();
    std::lock_guard<std::mutex> lk(g.m);
    if (g.members > 0) --g.members;
}

/* Test hook, exported beside the ABI entry points: how long a rank waits for its peers before
 * the collective returns RAD_E_DEVICE. A non-positive value restores the default. */
extern "C" void ref_collective_timeout_ms(int ms) { g_timeout_ms.store(ms > 0 ? ms : 30000); }

int64_t ref_all_reduce_scratch(const RadArgs* a) {
    const int64_t ws = p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (ws <= 1) return 0;
    const int64_t nel = p_int(a, "numel", a->n_t > 0 ? numel(&a->t[0]) : 0);
    return nel * (int64_t)sizeof(float);
}

/* ------------------------------------------------------------------ all_reduce */
/* `y` is optional; absent means in place (docs/OPS.md's form, all the one-shot kernels need).
 * libr4d's two-shot rows are not safe in place, since a rank gathers into the output while its
 * input is still read for the scatter, so a separate destination must be expressible. */
int ref_all_reduce(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const RadTensor* y = t_in(a, 1);
    /* The message is the operand; `numel` is the band. See band_rows in ref_common.h. */
    const int64_t nel = band_rows(numel(x), p_int(a, "numel", 0));
    if (nel <= 0) return RAD_E_SHAPE;
    if (y && numel(y) < nel) return RAD_E_SHAPE;
    const RadTensor* dst = y ? y : x;

    const int world = (int)p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (world <= 1) {
        /* x already holds the answer; copy it to a separate destination if one was given. */
        if (y) for (int64_t i = 0; i < nel; ++i) stt(y, offlin(y, i), ldt(x, offlin(x, i)));
        return RAD_OK;
    }
    RAD_REF_TRY(publish(a->rank, world, x, nel));
    Group& g = group();
    float* acc = staging(a, nel);
    for (int64_t i = 0; i < nel; ++i) {
        float s = 0.0f;
        for (int r = 0; r < world; ++r) s += ld_dt(g.dt[r], g.slot[r], i);
        acc[i] = s;
    }

    /* Every rank finishes reading before any writes; otherwise a fast rank overwrites input a
     * slower peer has not consumed, giving rank-dependent answers under load. */
    if (!barrier(g)) return RAD_E_DEVICE;
    for (int64_t i = 0; i < nel; ++i) stt(dst, offlin(dst, i), acc[i]);
    return RAD_OK;
}

/* ------------------------------------------------------------------ all_gather */
/* y is [world_size * numel] with rank r's contribution at offset r * numel.
 *
 * A gather moves bytes, not numbers. The vocab-parallel candidate merge gathers (id, score) pairs
 * as an opaque u32 payload of i32 ids and f32 bit patterns. Through float, the -1 empty-candidate
 * id (0xFFFFFFFF) becomes 4294967296.0f, which stores back as zero (undefined), and ids above 2^24
 * round. */
int ref_all_gather(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *y = &a->t[1];
    /* The contribution is the operand; `numel` is the band. See band_rows in ref_common.h. */
    const int64_t nel = band_rows(numel(x), p_int(a, "numel", 0));
    const int world = (int)p_int(a, "world_size", a->world_size > 0 ? a->world_size : 1);
    if (nel <= 0 || numel(y) < nel * world) return RAD_E_SHAPE;
    /* A byte copy needs whole-byte elements; sub-byte codes are refused (no caller gathers packed
     * weights). */
    const int64_t esz = rad_dtype_bytes(x->dtype, 1);
    if (esz <= 0 || y->dtype != x->dtype) return RAD_E_DTYPE;

    if (world <= 1) {
        for (int64_t i = 0; i < nel; ++i)
            std::memcpy((char*)y->data + offlin(y, i) * esz,
                        (const char*)x->data + offlin(x, i) * esz, (size_t)esz);
        return RAD_OK;
    }

    RAD_REF_TRY(publish(a->rank, world, x, nel));
    Group& g = group();
    /* Placement. `row` absent (or the whole message) is plain concatenation; present, it is
     * row-interleaved, x [rows][row] into y [rows][world][row], the form the vocab-parallel
     * candidate merge reads (one row of candidates per sampled position at one stride). */
    const int64_t row = p_int(a, "row", 0);
    if (row < 0 || (row > 0 && nel % row)) return RAD_E_SHAPE;
    for (int r = 0; r < world; ++r)
        for (int64_t i = 0; i < nel; ++i) {
            const int64_t o = (row > 0 && row < nel)
                                  ? (i / row) * (world * row) + (int64_t)r * row + (i % row)
                                  : (int64_t)r * nel + i;
            std::memcpy((char*)y->data + offlin(y, o) * esz,
                        (const char*)g.slot[r] + i * esz, (size_t)esz);
        }

    /* y is separate from x, but ranks must not start the next collective while a peer still reads
     * this one's input; the barrier also keeps call sequences paired. */
    if (!barrier(g)) return RAD_E_DEVICE;
    return RAD_OK;
}
