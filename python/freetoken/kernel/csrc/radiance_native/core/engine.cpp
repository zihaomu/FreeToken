// Modified for FreeToken: native integration, diagnostics and Qwen FP8 KV support.
/* engine.cpp -- the step loop and the server: DeviceRuntime construction, the per-rank threads, the
 * step loop and the server it runs under. Bring-up, the drafters and the buffer probe live in
 * their own files (engine_priv.h).
 */
#include "engine_priv.h"
#include "runtime/external_steps.h"
#include "gen/driver.h"
#include "gen/audio_driver.h"
#include "gen/video_driver.h"
#include "gen/shape_driver.h"
#include "kld.h"
#include "mm/processor.h"
#include "mm/media.h"

#include "build/startup.h"
#include "build/rad_build.h"
#include "device/device.h"
#include "format/radfile.h"
#include "gen_config.h"
#include "kvstream/stage.h"
#include "sample/accept.h"
#include "runtime/batch_state.h"
#include "server/adapters.h"
#include "server/front.h"
#include "server/server.h"
#include "server/sink.h"
#include "server/liveview.h"
#include "text/tokenizer.h"
#include "util/stallwatch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <algorithm>
#include <map>
#include <fstream>
#include <functional>
#include <utility>
#include <vector>
#include <thread>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/syscall.h>

#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
#include <string>
#endif

namespace rad {
std::vector<StorageRankView> DeviceRuntime::storage_rank_views() {
    std::vector<StorageRankView> out;
    for (auto& r : ranks_) out.push_back({r->device,&r->kv,r->ctx.stream()});
    return out;
}

#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
namespace {
struct KVRollbackTestEvidence {
    bool waiting_for_greedy = false;
    std::string report_path;
    uint64_t seq = 0;
    int step = -1;
    int proposals = 0;
    int accepted = -1;
    int32_t group = -1;
    int64_t prefix_before = 0;
    int64_t rollback_tokens_before = 0;
    int64_t rollback_to_tokens = 0;
    int64_t prefix_after_rollback = -1;
    int64_t first_block_pos = 0;
    int64_t block_size = 0;
    int64_t n_layers = 0;
    int64_t bytes_per_layer = 0;
    std::vector<int32_t> logical_blocks;
    std::vector<int32_t> physical_slots;
    std::vector<uint8_t> before;
    uint64_t after_hash = 0;
    bool bytes_equal = false;
    bool mapping_equal = false;
    bool negative_control_detected = false;
};

uint64_t kv_test_hash(const std::vector<uint8_t>& bytes) {
    uint64_t h = 14695981039346656037ull;
    for (uint8_t b : bytes) h = (h ^ b) * 1099511628211ull;
    return h;
}

bool kv_test_bytes_equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

template<class RankT>
bool kv_test_snapshot(RankT& rank, uint64_t seq, int32_t group, int64_t prefix_tokens,
                      int64_t* first_block_pos, std::vector<int32_t>* logical,
                      std::vector<int32_t>* physical, std::vector<uint8_t>* bytes) {
    const KVGroupPlan* p = rank.kv.plan(group);
    if (!p || p->zero || !p->paged() || p->kind != RAD_KV_FULL || p->streamed() ||
        p->n_layers <= 0 || p->block_size <= 0 || !p->base) return false;
    const auto& table = rank.kv.block_table(seq, group);
    const int64_t first_pos = rank.kv.first_block_pos(seq, group);
    if (first_pos < 0 || prefix_tokens < first_pos) return false;
    const int64_t complete = (prefix_tokens - first_pos) / p->block_size;
    if (complete <= 0 || complete > (int64_t)table.size()) return false;
    if (first_block_pos) *first_block_pos = first_pos;
    /* Snapshot every complete block in the retained pre-verify prefix. The trailing partial
     * block is omitted because accepted verify tokens may legitimately fill its unused slots. */
    const int64_t nblocks = complete;
    const int64_t layer_bytes = p->layer_bytes_per_block();
    const int64_t block_bytes = p->n_layers * layer_bytes;
    if (nblocks <= 0 || layer_bytes <= 0 || p->bytes_per_block != block_bytes) return false;
    if (rad_dev_set(rank.device) < 0 || rad_stream_sync(rank.ctx.stream()) < 0) return false;
    logical->clear();
    physical->clear();
    logical->reserve((size_t)nblocks);
    physical->reserve((size_t)nblocks);
    bytes->resize((size_t)nblocks * (size_t)block_bytes);
    for (int64_t b = 0; b < nblocks; ++b) {
        const int32_t id = table[(size_t)b];
        if (id < 0 || id >= p->pool_blocks()) return false;
        logical->push_back((int32_t)(first_pos / p->block_size + b));
        physical->push_back(id);
        for (int64_t layer = 0; layer < p->n_layers; ++layer) {
            const auto* src = static_cast<const uint8_t*>(p->base) +
                              layer * p->layer_stride() + (int64_t)id * layer_bytes;
            auto* dst = bytes->data() + b * block_bytes + layer * layer_bytes;
            if (rad_memcpy_async(dst, src, layer_bytes, rank.ctx.stream()) < 0) return false;
        }
    }
    return rad_stream_sync(rank.ctx.stream()) >= 0;
}

void kv_test_write_report(const KVRollbackTestEvidence& e, bool greedy, int greedy_token,
                          int64_t tokens_at_greedy) {
    if (e.report_path.empty()) return;
    std::ofstream f(e.report_path, std::ios::trunc);
    if (!f) return;
    f << "{\"partial_rejection_observed\":true,\"step\":" << e.step
      << ",\"sequence\":" << e.seq << ",\"proposals\":" << e.proposals
      << ",\"accepted\":" << e.accepted << ",\"group\":" << e.group
      << ",\"prefix_tokens_before_verify\":" << e.prefix_before
      << ",\"rollback_tokens_before\":" << e.rollback_tokens_before
      << ",\"rollback_to_tokens\":" << e.rollback_to_tokens
      << ",\"prefix_tokens_after\":" << e.prefix_after_rollback
      << ",\"first_block_pos\":" << e.first_block_pos
      << ",\"block_size\":" << e.block_size << ",\"layers\":" << e.n_layers
      << ",\"bytes_per_layer\":" << e.bytes_per_layer << ",\"logical_blocks\":[";
    for (size_t i = 0; i < e.logical_blocks.size(); ++i) f << (i ? "," : "") << e.logical_blocks[i];
    f << "],\"physical_slots\":[";
    for (size_t i = 0; i < e.physical_slots.size(); ++i) f << (i ? "," : "") << e.physical_slots[i];
    f << "],\"committed_prefix_bytes\":" << e.before.size()
      << ",\"before_hash\":" << kv_test_hash(e.before)
      << ",\"after_hash\":" << e.after_hash
      << ",\"bytes_equal\":" << (e.bytes_equal ? "true" : "false")
      << ",\"mapping_equal\":" << (e.mapping_equal ? "true" : "false")
      << ",\"negative_control_detected\":" << (e.negative_control_detected ? "true" : "false")
      << ",\"next_greedy_decode_observed\":" << (greedy ? "true" : "false")
      << ",\"tokens_at_next_greedy_decode\":" << (greedy ? tokens_at_greedy : -1)
      << ",\"next_greedy_token\":" << (greedy ? greedy_token : -1) << "}\n";
}

KVRollbackTestEvidence& kv_test_evidence() {
    static KVRollbackTestEvidence e;
    return e;
}

template<class Output, class Ranks>
void kv_test_before_commit(const StepPlan& plan, const Output& so, int step, Ranks& ranks) {
    auto& e = kv_test_evidence();
    if (!e.waiting_for_greedy && !e.report_path.empty() && e.accepted < 0) {
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& entry = plan.e[i];
            if (entry.seq == e.seq && step == e.step && entry.n_spec == e.proposals &&
                (so.accepted[i] <= 0 || so.accepted[i] >= e.proposals)) {
                e = KVRollbackTestEvidence{};
                return;
            }
        }
    }
    if (!e.waiting_for_greedy) return;
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& entry = plan.e[i];
        if (entry.seq == e.seq && entry.req && entry.req->sp.greedy() &&
            so.first_row[i] >= 0) {
            kv_test_write_report(e, true, so.token[i], ranks[0]->kv.seq_tokens(e.seq));
            e.waiting_for_greedy = false;
            break;
        }
    }
}

template<class Ranks>
void kv_test_prepare(const StepPlan& plan, int step, Ranks& ranks) {
    auto& e = kv_test_evidence();
    if (!e.report_path.empty() || e.waiting_for_greedy) return;
    const char* path = std::getenv("RADIANCE_TEST_KV_ROLLBACK_REPORT");
    if (!path || !*path) return;
    for (const StepEntry& entry : plan.e) {
        if (!entry.req || entry.n_spec != 7) continue;
        const int64_t prefix = entry.ctx_len;
        for (int32_t g = 0; g < ranks[0]->kv.n_groups(); ++g) {
            const KVGroupPlan* p = ranks[0]->kv.plan(g);
            if (!p || p->kind != RAD_KV_FULL || p->zero || p->streamed() ||
                prefix < p->block_size) continue;
            KVRollbackTestEvidence candidate;
            candidate.report_path = path;
            candidate.seq = entry.seq;
            candidate.step = step;
            candidate.proposals = entry.n_spec;
            candidate.group = g;
            candidate.prefix_before = prefix;
            candidate.block_size = p->block_size;
            candidate.n_layers = p->n_layers;
            candidate.bytes_per_layer = p->layer_bytes_per_block();
            if (!kv_test_snapshot(*ranks[0], entry.seq, g, prefix, &candidate.first_block_pos,
                                  &candidate.logical_blocks, &candidate.physical_slots,
                                  &candidate.before)) continue;
            e = std::move(candidate);
            return;
        }
    }
}

template<class RankT>
void kv_test_rollback(uint64_t seq, int proposals, int accepted, int64_t valid,
                      bool before, RankT& rank) {
    auto& e = kv_test_evidence();
    if (e.report_path.empty() || e.waiting_for_greedy || seq != e.seq ||
        proposals != e.proposals || proposals != 7 || accepted <= 0 || accepted >= proposals)
        return;
    if (before) {
        e.accepted = accepted;
        e.rollback_tokens_before = rank.kv.seq_tokens(seq);
        e.rollback_to_tokens = valid;
        return;
    }
    std::vector<int32_t> logical, physical;
    std::vector<uint8_t> after;
    int64_t first_block_pos = -1;
    const bool copied = kv_test_snapshot(rank, seq, e.group, std::min(e.prefix_before, valid),
                                         &first_block_pos, &logical, &physical, &after);
    bool mapping_equal = copied && first_block_pos == e.first_block_pos &&
                         logical == e.logical_blocks && physical == e.physical_slots;
    const bool bytes_equal = copied && kv_test_bytes_equal(after, e.before);
    e.after_hash = copied ? kv_test_hash(after) : 0;
    e.bytes_equal = bytes_equal;
    std::vector<uint8_t> mutated = e.before;
    if (!mutated.empty()) mutated[0] ^= 1;
    e.negative_control_detected = !mutated.empty() && !kv_test_bytes_equal(e.before, mutated);
    const int64_t after_tokens = rank.kv.seq_tokens(seq);
    mapping_equal = mapping_equal && after_tokens == e.rollback_to_tokens;
    e.mapping_equal = mapping_equal;
    e.prefix_after_rollback = after_tokens;
    kv_test_write_report(e, false, -1, -1);
    e.waiting_for_greedy = true;
    (void)valid;  // rollback's requested length is emitted by the manager as after_tokens
}
} // namespace
#endif

/* Owner ids for the device layer (engine_owner), one per engine; 0 is no engine. */
namespace {
std::atomic<int> g_next_owner{1};
}

/* The operator's RADIANCE_* names, captured at construction for the diagnostics report. */
DeviceRuntime::DeviceRuntime(Config cfg, std::unique_ptr<BatchState> batches)
    : cfg_(std::move(cfg)), batch_state_owner_(std::move(batches)), sched_(*batch_state_owner_),
      owner_(g_next_owner.fetch_add(1)),
      env_set_(diag::env_names(environ, "RADIANCE_")) {}

/* Every rank's compute stream, the mover's transfer stream and the KV tiers' copy stream,
 * synchronised. A tier copy reads the KV pool long after the step that issued it, and nothing on
 * the compute stream waits for it, so memory leaving the card must wait for it here: a gather over
 * a range just unbacked is a GPU memory fault. The compute stream goes first because the copy is
 * ordered after it. */
void DeviceRuntime::drain_streams() {
    const int dev = dev_current();
    for (auto& rk : ranks_) {
        (void)rad_stream_sync(rk->ctx.stream());
        if (RadStream t = rk->mover.transfer_stream()) (void)rad_stream_sync(t);
        rk->tierx.quiesce();
    }
    (void)rad_dev_set(dev);
}

std::vector<int> DeviceRuntime::rank_devices() const {
    std::vector<int> cards;
    cards.reserve(ranks_.size());
    for (auto& r : ranks_) cards.push_back(r->device);
    return cards;
}

/* The step loop's rest point for a park: idle, nothing issued that still runs. The streams are
 * drained before the loop answers, so the fleet may page the memory out at once. */
void DeviceRuntime::hold_point(StallWatch* watch) {
    if (!hold_req_.load(std::memory_order_acquire)) return;
    drain_streams();
    hold_ack_.store(true, std::memory_order_release);
    /* A park lasts as long as the fleet keeps the model off its cards. The loop sleeps until the
     * hold clears or the engine stops, waking once a second besides to mark the step watch, so a
     * model parked for minutes is not reported as a step that stopped (and is not woken a thousand
     * times a second to look). */
    {
        std::unique_lock<std::mutex> lk(hold_mu_);
        while (hold_req_.load(std::memory_order_acquire) && !stopping_.load(std::memory_order_acquire)) {
            hold_cv_.wait_for(lk, std::chrono::seconds(1));
            if (watch) watch->progress();
        }
    }
    hold_ack_.store(false, std::memory_order_release);
}

void DeviceRuntime::hold_release() {
    {
        std::lock_guard<std::mutex> lk(hold_mu_);
        hold_req_.store(false, std::memory_order_release);
    }
    hold_cv_.notify_all();
}

void DeviceRuntime::wake_hold() {
    { std::lock_guard<std::mutex> lk(hold_mu_); }
    hold_cv_.notify_all();
}

int DeviceRuntime::park(int64_t drain_ms, const PageOut& req, PageResult* got, std::string* why) {
    using clk = std::chrono::steady_clock;
    const auto secs = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    const auto t0 = clk::now();
    auto page_out = [&] {
        PageOut r = req;
        r.discard_elastic = true;
        return dev_page_out(owner_, r, got);
    };
    /* Already parked: more of it leaves. */
    if (parked_.load()) {
        EngineScope es(owner_, cfg_.served_model_name.c_str());
        const int rc = page_out();
        got->out_s = secs(t0, clk::now());
        if (rc < 0) {
            *why = std::string("paging out failed: ") + rad_dev_last_error();
            return rc;
        }
        return RAD_OK;
    }
    if (!VMem::supported()) {
        *why = "the card has no virtual memory manager, so nothing can leave it at the same address";
        return RAD_E_UNSUPPORTED;
    }
    if (!front_ || !srv_) {
        *why = "the engine is not serving behind a front";
        return RAD_E_STATE;
    }
    const std::string& name = cfg_.served_model_name;
    if (!front_->detach(name, (int)std::min<int64_t>(drain_ms, INT32_MAX))) {
        RAD_WARN("model %s: responses still open after %lld ms; ending them to park",
                 name.c_str(), (long long)drain_ms);
        srv_->end_streams(true);
        const bool gone = front_->detach(name, 60000);
        srv_->end_streams(false);
        if (!gone) {
            *why = "responses still held a minute after they were ended";
            front_->attach(name, front_aliases_, srv_);
            return RAD_E_STATE;
        }
    }
    hold_req_.store(true, std::memory_order_release);
    sched_.wake();
    for (int i = 0; i < 60000 && !hold_ack_.load(std::memory_order_acquire); ++i) {
        if (loop_status_.load(std::memory_order_acquire) < 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!hold_ack_.load(std::memory_order_acquire)) {
        hold_release();
        front_->attach(name, front_aliases_, srv_);
        *why = "the step loop did not come to rest";
        return RAD_E_STATE;
    }
    const auto t1 = clk::now();
    EngineScope es(owner_, name.c_str());
    /* What need not be copied: the expert slab once its units are back at their host copies
     * (the heat engine fills it again after the restore), the KV cache once its index is reset
     * (nothing runs, so no block is live), and the activation arena above its fixed region:
     * transients live within one step, and the loop rests between steps. A unit still in a slot
     * the arena lent keeps the arena copied. */
    std::vector<std::pair<const void*, int64_t>> spare;
    for (auto& r : ranks_) {
        if (rad_dev_set(r->device) < 0) continue;
        if (r->mover.flex_capacity() > 0) {
            (void)r->mover.reclaim(Mover::kLendKV, 0, &r->heat);
            (void)r->mover.reclaim(Mover::kLendArena, 0, &r->heat);
            (void)r->mover.reclaim(Mover::kLendFleet, 0, &r->heat);
        }
        /* The fleet's loan is unbacked, not copied: nothing in it outlives the recall. */
        if (r->xflex.committed() > 0 && r->mover.loan_held(Mover::kLendFleet) == 0)
            (void)r->xflex.decommit(0, r->xflex.size());
        if (r->heat.enabled() && r->mover.drop_slab(&r->heat, &spare) < 0)
            RAD_WARN("rank %d: the expert slab could not be emptied for the park; it is copied", r->index);
        const int64_t fixed = r->program.arena_fixed_bytes, top = r->ctx.arena_bytes();
        if (r->ctx.arena_base() && top > fixed && r->mover.loan_held(Mover::kLendArena) == 0)
            spare.push_back({ (const char*)r->ctx.arena_base() + fixed, top - fixed });
    }
    if (!spare.empty()) dev_page_discard(owner_, spare);
    (void)xflex_settle();
    if (cfg_.prefix_cache) (void)sched_.reset_prefix_cache();
    const auto t2 = clk::now();
    const int rc = page_out();
    got->quiet_s = secs(t0, t1);
    got->slab_s = secs(t1, t2);
    got->out_s = secs(t2, clk::now());
    if (rc < 0) {
        /* Whatever left comes back, and the model serves as before. */
        *why = std::string("paging out failed: ") + rad_dev_last_error();
        (void)page_back_in(nullptr);
        hold_release();
        front_->attach(name, front_aliases_, srv_);
        return rc;
    }
    parked_.store(true, std::memory_order_release);
    return RAD_OK;
}

int64_t DeviceRuntime::flex_bytes() const {
    const int64_t x = xflex_bytes_.load(std::memory_order_acquire);
    if (ranks_.empty() || !ranks_[0]->kv.flex()) return x;
    return x + std::max<int64_t>(ranks_[0]->pools.kv().committed() - flex_floor_bytes_, 0);
}

int64_t DeviceRuntime::flex_release(int64_t bytes, bool preempt, int64_t wait_ms, bool kv) {
    if (bytes <= 0 || ranks_.empty() || parked_.load()) return 0;
    const bool has_kv = kv && ranks_[0]->kv.flex();
    if (!has_kv && xflex_bytes_.load() <= 0) return 0;
    std::unique_lock<std::mutex> lk(flex_mu_);
    flex_want_ = bytes;
    flex_preempt_ = preempt;
    flex_kv_ = has_kv;
    flex_got_ = -1;
    flex_pending_.store(true, std::memory_order_release);
    flex_cv_.wait_for(lk, std::chrono::milliseconds(wait_ms), [&] {
        return flex_got_ >= 0 || stopping_.load(std::memory_order_acquire);
    });
    flex_pending_.store(false, std::memory_order_release);
    return flex_got_ > 0 ? flex_got_ : 0;
}

/* On the step loop's thread, between steps with nothing issued ahead: the expert slab's fleet loan
 * first, then (unless only the experts' flex was asked for) the cache's free top, the prefix
 * cache and, with `flex_preempt_`, the running requests, until `flex_want_` bytes can go; then the
 * mark moves on every rank and the stripes above it are unbacked. */
void DeviceRuntime::flex_service() {
    if (!flex_pending_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(flex_mu_);
    if (flex_got_ >= 0) return;
    drain_streams();
    /* The experts' loan first: giving it back is an address change for each unit in it. */
    const int64_t xgot = xflex_give(flex_want_);
    if (xgot >= flex_want_ || !flex_kv_) {
        flex_got_ = xgot;
        flex_cv_.notify_all();
        return;
    }
    KVManager& k0 = ranks_[0]->kv;
    const int64_t tb = std::max<int64_t>(k0.flex_token_bytes(), 1);
    const int64_t want = (flex_want_ - xgot + tb - 1) / tb;
    const int64_t floor_loan = k0.flex_floor_loan();
    const int64_t cur = sched_.locked([&] { return k0.loan_tokens(); });
    /* spare_tokens() counts from the top of the carve, and a flex carve's top is the unbacked
     * reservation: what is free below the live mark is that less the loan. Never below the floor
     * the claim pays for. */
    auto room_locked = [&] {
        return std::max<int64_t>(std::min(k0.spare_tokens() - cur, floor_loan - cur), 0);
    };
    auto room = [&] { return sched_.locked(room_locked); };
    int64_t spare = room();
    /* What the floor lets go however empty the cache is. Past that, evicting or preempting costs
     * conversations and frees nothing the fleet can have. */
    const int64_t goal = std::min(want, std::max<int64_t>(floor_loan - cur, 0));
    if (spare < goal && cfg_.prefix_cache) {
        (void)sched_.evict_prefix_until([&] { return room_locked() >= goal; });
        spare = room();
    }
    if (spare < goal && flex_preempt_) {
        const int n = sched_.preempt_until([&] { return room_locked() >= goal; });
        if (sched_.has_deferred()) sched_.release_deferred();
        if (n > 0)
            RAD_INFO("flex: %d running request(s) go back to the queue so a higher-priority model "
                     "takes the cache", n);
        spare = room();
    }
    const int64_t take = std::min(goal, spare);
    const int64_t before = ranks_[0]->pools.kv().committed();
    if (take > 0 && sched_.locked([&] { return k0.set_loan(cur + take); }) == RAD_OK) {
        for (size_t i = 1; i < ranks_.size(); ++i) (void)ranks_[i]->kv.set_loan(cur + take);
        for (auto& r : ranks_) {
            (void)rad_dev_set(r->device);
            if (const int rc = r->kv.flex_unback_above(cur + take); rc < 0)
                RAD_WARN("flex: rank %d could not unback its cache above the new mark: %s (%s)",
                         r->index, rad_strerror(rc), rad_dev_last_error());
        }
        (void)rad_dev_set(ranks_[0]->device);
    }
    flex_got_ = xgot + std::max<int64_t>(before - ranks_[0]->pools.kv().committed(), 0);
    RAD_INFO("flex: asked for %s, the cache could give %lld of %lld tokens, gave back %s",
             humanb(flex_want_).c_str(), (long long)take, (long long)want, humanb(flex_got_).c_str());
    flex_cv_.notify_all();
}

/* The slab's backed bytes a card after a recall, the most any rank still has backed; also clears
 * what the slab asks the fleet for. */
int64_t DeviceRuntime::xflex_settle() {
    int64_t left = 0;
    for (auto& r : ranks_) left = std::max(left, r->xflex.committed());
    xflex_bytes_.store(left, std::memory_order_release);
    xflex_want_.store(0, std::memory_order_release);
    return left;
}

/* On the step loop, every stream drained: the slab's units in the fleet's slots past the new loan
 * go back to their host copies (Mover::reclaim), and the granules they leave are unbacked. The
 * bytes given back on each card. The slab does not grow again for five seconds, so the model that
 * asked for the room gets it first. */
int64_t DeviceRuntime::xflex_give(int64_t bytes) {
    const int64_t cur = xflex_bytes_.load(std::memory_order_acquire);
    if (bytes <= 0 || cur <= 0) return 0;
    const int64_t g = std::max<int64_t>(VMem::granularity(), 1);
    int64_t to = std::max<int64_t>(cur - (bytes + g - 1) / g * g, 0);
    for (auto& r : ranks_) {
        if (rad_dev_set(r->device) < 0) continue;
        if (const int st = r->mover.reclaim(Mover::kLendFleet, to, &r->heat); st < 0)
            RAD_WARN("expert flex: rank %d could not recall its slab to %s: %s", r->index,
                     humanb(to).c_str(), rad_strerror(st));
        /* What a unit still holds stays backed. */
        to = std::max(to, (r->mover.loan_held(Mover::kLendFleet) + g - 1) / g * g);
    }
    for (auto& r : ranks_) {
        if (rad_dev_set(r->device) < 0) continue;
        const int64_t top = r->xflex.size();
        (void)r->mover.set_loan(Mover::kLendFleet, to);
        if (const int st = r->xflex.decommit(std::max<int64_t>(top - cur, 0), cur - to); st < 0)
            RAD_WARN("expert flex: rank %d could not unback %s: %s (%s)", r->index,
                     humanb(cur - to).c_str(), rad_strerror(st), rad_dev_last_error());
    }
    (void)rad_dev_set(ranks_[0]->device);
    const int64_t left = xflex_settle();
    xflex_retry_ns_ = rad_mono_ns() + 5000000000ull;
    const int64_t got = std::max<int64_t>(cur - left, 0);
    if (got > 0)
        RAD_INFO("expert flex: the slab gave back %s a card, keeps %s", humanb(got).c_str(),
                 humanb(left).c_str());
    return got;
}

/* Every tenth of a second on the step loop. While the engine has been busy in the last five
 * seconds, the slab grows a GiB at a time into memory the fleet says no model holds, and grows
 * again only once the units fill what it already has, leaving no more than the turnover slots and
 * half a step empty (an empty slot is VRAM nothing uses). A refusal is remembered, so the fleet
 * can make room for it (expert_flex_wanted), and asked again a second later; a backing that fails
 * waits five seconds. */
void DeviceRuntime::xflex_tick(bool busy) {
    if (xflex_cap_ <= 0 || !broker_) return;
    const uint64_t now = rad_mono_ns();
    if (busy) xflex_busy_ns_ = now;
    if (now - xflex_ns_ < 100000000ull) return;
    xflex_ns_ = now;
    const int64_t cur = xflex_bytes_.load(std::memory_order_acquire);
    if (cur >= xflex_cap_ || now - xflex_busy_ns_ > 5000000000ull) {
        xflex_want_.store(0, std::memory_order_release);
        return;
    }
    if (now < xflex_retry_ns_) return;
    const Mover& m0 = ranks_[0]->mover;
    const int64_t g = std::max<int64_t>(VMem::granularity(), 1);
    const int64_t step = std::min<int64_t>(((int64_t)1 << 30) / g * g, xflex_cap_ - cur);
    /* Still filling: what the loans offer and no unit holds, past the slots kept free for swaps. */
    const int64_t kept = m0.turnover_bytes();
    if (m0.flex_offered_bytes() - m0.flex_used_bytes() > kept + step / 2) return;
    if (!broker_->flex_grow(rank_devices(), step)) {
        xflex_want_.store(xflex_cap_ - cur, std::memory_order_release);
        xflex_retry_ns_ = now + 1000000000ull;
        return;
    }
    struct Landed { VramBroker* b; ~Landed() { b->flex_done(); } } landed{ broker_ };
    const int64_t to = cur + step;
    bool ok = true;
    for (auto& r : ranks_) {
        if (rad_dev_set(r->device) < 0) { ok = false; break; }
        const int64_t top = r->xflex.size();
        if (r->xflex.commit(std::max<int64_t>(top - to, 0), std::min(step, top)) < 0) { ok = false; break; }
    }
    if (!ok) {
        RAD_WARN("expert flex: backing %s more a card failed (%s); trying again later",
                 humanb(step).c_str(), rad_dev_last_error());
        for (auto& r : ranks_) {
            if (rad_dev_set(r->device) < 0) continue;
            const int64_t top = r->xflex.size();
            (void)r->xflex.decommit(std::max<int64_t>(top - to, 0), step);
        }
        (void)rad_dev_set(ranks_[0]->device);
        xflex_retry_ns_ = now + 5000000000ull;
        return;
    }
    for (auto& r : ranks_) {
        if (rad_dev_set(r->device) < 0) continue;
        (void)r->mover.set_loan(Mover::kLendFleet, to);
    }
    (void)rad_dev_set(ranks_[0]->device);
    xflex_bytes_.store(to, std::memory_order_release);
    xflex_want_.store(0, std::memory_order_release);
    if (to >= xflex_cap_ || cur == 0)
        RAD_INFO("expert flex: the slab grows into memory no model holds: %s a card of up to %s",
                 humanb(to).c_str(), humanb(xflex_cap_).c_str());
}

/* What the fleet could still grow the cache by, for admission, every tenth of a second. */
void DeviceRuntime::flex_room_tick(uint64_t now_ns) {
    if (!broker_ || ranks_.empty() || !ranks_[0]->kv.flex()) return;
    if (now_ns - flex_room_ns_ < 100000000ull) return;
    flex_room_ns_ = now_ns;
    const int64_t tb = std::max<int64_t>(ranks_[0]->kv.flex_token_bytes(), 1);
    ranks_[0]->kv.set_flex_room(broker_->flex_room(rank_devices()) / tb);
}

int64_t DeviceRuntime::release_host_spare() {
    if (!parked_.load() || ck_released_.load() || !ck_store_.enabled() || ck_store_.used_slots() > 0) return 0;
    EngineScope es(owner_, cfg_.served_model_name.c_str());
    ck_slot_ = ck_store_.slot_bytes();
    ck_cap_ = ck_store_.cap_bytes();
    ck_store_.close();
    ck_released_.store(true, std::memory_order_release);
    return ck_cap_;
}

int DeviceRuntime::repin_host_spare(std::string* why) {
    if (!ck_released_.load()) return RAD_OK;
    if (ck_store_.open(ck_slot_, ck_cap_) != RAD_OK) {
        *why = "the checkpoint slots' " + humanb(ck_cap_) + " of pinned host memory could not be pinned again";
        return RAD_E_NOMEM;
    }
    ck_released_.store(false, std::memory_order_release);
    return RAD_OK;
}

int DeviceRuntime::unpark(std::string* why) { return unpark_with(why, nullptr); }

int DeviceRuntime::page_back_in(int64_t* left) {
    std::vector<PageSpan> refill;
    if (const int rc = dev_page_in_begin(owner_, left, &refill); rc < 0) return rc;
    /* The elastic ranges (the KV cache, the expert flex) are backed, and the chunks held in host
     * memory copied back (q27: 6 GiB, 0.3 s), beside the refill, which reads the drive and writes
     * none of them. */
    int erc = RAD_OK;
    std::string elastic_why;   /* the device error is per thread: carried back to this one */
    std::thread elastic([&] {
        EngineScope es(owner_, cfg_.served_model_name.c_str());
        erc = dev_page_in_elastic(owner_);
        if (erc < 0) {
            const char* w = rad_dev_last_error();
            elastic_why = w && *w ? w : rad_strerror(erc);
        }
    });
    /* The refill's writes wait for the begin's clears of recycled chunks, which run beside its
     * first read. */
    const int frc = refill_weights(refill, "restore", [this] { return dev_page_in_cleared(owner_); });
    elastic.join();
    const int irc = dev_page_in_end(owner_);
    if (frc < 0) {
        device_set_error("the weights paged back empty could not be read from the container again (%s)",
                         rad_strerror(frc));
        return frc;
    }
    if (irc < 0) return irc;
    if (erc < 0)
        device_set_error("the KV cache and expert ranges could not be backed again: %s",
                         elastic_why.c_str());
    return erc;
}

int DeviceRuntime::unpark_live(std::string* why, int64_t* left) {
    *left = 0;
    return unpark_with(why, left);
}

/* unpark() with `left` null, unpark_live() otherwise: the memory comes back (whole, or live with
 * `left` bytes still in host memory), the step loop leaves its rest, and the front routes to the
 * model again. */
int DeviceRuntime::unpark_with(std::string* why, int64_t* left) {
    if (!parked_.load()) return RAD_OK;
    EngineScope es(owner_, cfg_.served_model_name.c_str());
    /* The snapshot store is pinned again beside the page-in (q27's 4 GiB takes 0.2-0.3 s), not
     * before it: neither waits on the other, and the step loop stays at rest until both are done.
     * On a failure of either the fleet unloads the model, so the order of the two is free. */
    std::string pin_why;
    int pin_rc = RAD_OK;
    std::thread pin;
    if (ck_released_.load())
        pin = std::thread([&] {
            EngineScope pes(owner_, cfg_.served_model_name.c_str());
            pin_rc = repin_host_spare(&pin_why);
        });
    const int prc = page_back_in(left);
    if (pin.joinable()) pin.join();
    if (prc < 0) {
        *why = std::string("paging in failed: ") + rad_dev_last_error();
        return prc;
    }
    if (pin_rc < 0) {
        *why = pin_why;
        return pin_rc;
    }
    parked_.store(false, std::memory_order_release);
    hold_release();
    while (hold_ack_.load(std::memory_order_acquire) && !stopping_.load())
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    front_->attach(cfg_.served_model_name, front_aliases_, srv_);
    return RAD_OK;
}

int DeviceRuntime::run_quiet(const std::function<int()>& fn) {
    std::unique_lock<std::mutex> lk(quiet_mu_);
    quiet_fn_ = &fn;
    quiet_done_ = false;
    quiet_pending_.store(true, std::memory_order_release);
    sched_.wake();
    /* Polled for a stop: the step loop may end without passing a step boundary again. */
    while (!quiet_cv_.wait_for(lk, std::chrono::milliseconds(50), [&] {
        return quiet_done_ || stopping_.load(std::memory_order_acquire);
    })) {}
    quiet_pending_.store(false, std::memory_order_release);
    quiet_fn_ = nullptr;
    return quiet_done_ ? quiet_rc_ : RAD_E_STATE;
}

/* On the step loop's thread, with nothing issued ahead: every rank's streams drained (the compute
 * stream and the mover's, as for a park), then run_quiet's function. */
void DeviceRuntime::quiet_service() {
    if (!quiet_pending_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(quiet_mu_);
    if (quiet_done_ || !quiet_fn_) return;
    const int dev = dev_current();
    drain_streams();
    quiet_rc_ = (*quiet_fn_)();
    (void)rad_dev_set(dev);
    quiet_done_ = true;
    quiet_cv_.notify_all();
}

int32_t* DeviceRuntime::Readback::reserve(int64_t count) {
    if (count <= n) return p;
    if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    p = static_cast<int32_t*>(rad_dev_alloc(count * (int64_t)sizeof(int32_t), RAD_MEM_HOST_PINNED));
    n = p ? count : 0;
    return p;
}

DeviceRuntime::Readback::~Readback() {
    if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
}

DeviceRuntime::~DeviceRuntime() {
    /* The control vector's per-rank direction copies. Uploaded at declare, outside any arena, so
     * nothing else frees them; each rank's copy is freed with that rank's device bound, since the
     * allocator is per device. */
    for (const auto& r : ranks_) {
        if (!r || (size_t) r->index >= cvec_views_.size()) continue;
        const float* d = cvec_views_[(size_t) r->index].dir;
        if (!d) continue;
        if (rad_dev_set(r->device) < 0) continue;
        rad_dev_free((void*) d, RAD_MEM_DEVICE);
        cvec_views_[(size_t) r->index].dir = nullptr;
    }
    cvec_views_.clear();
    /* --profile-ops prints its final table here, the last point where the program still holds
     * the op table the rows name. */
    if (cfg_.profile_ops) for (auto& r : ranks_) r->ctx.dump_profile();
    if (stream_ctl_p_) rad_dev_free(stream_ctl_p_, RAD_MEM_HOST_PINNED);
    for (float* p : kld_host_) if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    if (pooled_host_) rad_dev_free(pooled_host_, RAD_MEM_HOST_PINNED);
    for (auto& r : ranks_) if (r->score_host) rad_dev_free(r->score_host, RAD_MEM_HOST_PINNED);
    /* The tiering summary, here for the same reason, and only for a dynamic plan. Its per-layer
     * residency histogram shows whether the global ranking starved a layer, which a hit rate
     * cannot. */
    for (auto& r : ranks_) {
        if (!r->heat.enabled()) continue;
        LogRank lr(r->index);
        RAD_INFO("%s", r->heat.report().c_str());
        RAD_INFO("  routing evidence: %llu layer-histograms credited, %llu layer-sweeps found "
                 "nothing new, %llu dropped by a ring shallower than the run-ahead",
                 (unsigned long long)r->heat_credited, (unsigned long long)r->heat_idle,
                 (unsigned long long)r->ctx.route_dropped());
        RAD_INFO("%s", r->mover.report().c_str());
    }
    /* Drain the transfer streams before the contexts. Ctx::shutdown drains the compute streams
     * and frees the arena, but the mover's stream also copies into the arena (the prefill
     * stager's ring, lent slab slots) and would otherwise drain only later, with the mover. A fill
     * landing in a freed arena is a GPU memory fault. */
    for (auto& r : ranks_) {
        if (RadStream s = r->mover.transfer_stream()) (void)rad_stream_sync(s);
        r->tierx.quiesce();
    }
    for (auto& r : ranks_) r->ctx.shutdown();
    if (cards_claimed_) {
        const std::vector<int> cards = rank_devices();
        (void)rad_dev_cards_claim(cards.data(), (int)cards.size(), 0);
        cards_claimed_ = false;
    }
    /* The streamed groups' host mirrors are pinned outside every pool; freed after the contexts,
     * since a pass reads them. Safe after a failed bring-up. */
    for (auto& r : ranks_) r->kv.free_stream(rad_device_allocator());
    step_state_fini();
    /* The step batch's staging, events and media buffers (BatchBuilder), freed while the ranks'
     * KV managers the scheduler releases sequences into still exist. A process that unloads
     * models would otherwise keep every engine's. */
    sched_.fini();
    ranks_.clear();
    for (TableCopy& c : table_copies_) table_copy_free(&c);
    table_copies_.clear();
    for (TableCopy& c : table_files_) ::munmap(c.base, c.bytes);
    table_files_.clear();
    for (void* p : card_copies_) rad_dev_free(p, RAD_MEM_HOST_MAPPED);
    card_copies_.clear();
    for (TableShare& t : table_shares_) table_share_close(&t);
    table_shares_.clear();
    if (reg_) reg_->close();
    if (slot_base_ >= 0) rad_engine_release(plugins_.arch_plugin, slot_base_, cfg_.tp);
}

/* ================================================================== 7. serve */
/* The step loop, where every component meets. Threading per spec §1: HTTP threads feed a request
 * queue, one scheduler thread produces one step, and one thread per tensor-parallel rank owns one
 * device.
 *
 * The batch is shared by pointer and the handoff is a barrier (rank_barrier.h), entered twice per
 * step: to release the ranks against the batch, and to join them before the sampler's answer is
 * read. The scheduler thread takes part, so the width is tp + 1.
 */

int DeviceRuntime::declare_sampler(int rank, RadBuilder* b, int64_t logits_width, bool sizing) {
    Rank& r = *ranks_[(size_t)rank];
    /* A sizing declare uses a throwaway sampler: the chain must be in the Program so the buffer
     * list matches, but the rank's sampler keeps the real declare's handles. */
    Sampler scratch;
    Sampler& smp = sizing ? scratch : r.sampler;
    rad_buf scratch_tok = 0;
    rad_buf& tok = sizing ? scratch_tok : r.tok_buf;

    SamplerConfig sc{};
    sc.n_vocab       = meta_.n_vocab;
    /* This rank's vocabulary width, from the declared logits buffer (see rad_tools_declare).
     * meta_.n_vocab is the whole model's on every rank. */
    sc.n_vocab_local = logits_width > 0 ? logits_width : meta_.n_vocab;
    /* This rank's first vocabulary row, which the width cannot tell. The rule restates
     * arch/common/rad_arch.h's shard_begin rather than passing through the ABI, and is checked
     * against the declared width, so a plugin that shards differently is refused at startup
     * instead of sampling with the wrong ids. */
    sc.vocab_off = 0;
    if (sc.n_vocab_local < sc.n_vocab) {
        /* rad_arch.h vocab_per_rank: the even share when tp divides the vocabulary, otherwise
         * ceil(n / tp) rounded up to a 128-row block so a block-scaled head can be cut there; the
         * last rank is short. */
        const int64_t ceil_per = (sc.n_vocab + cfg_.tp - 1) / cfg_.tp;
        const int64_t per = (sc.n_vocab % cfg_.tp == 0) ? sc.n_vocab / cfg_.tp
                                                        : (ceil_per + 127) / 128 * 128;
        sc.vocab_off = per * rank;
        const int64_t want = std::min<int64_t>(per, sc.n_vocab - sc.vocab_off);
        if (sc.vocab_off >= sc.n_vocab || want != sc.n_vocab_local) {
            RAD_ERR("rank %d declared %lld logits columns of a %lld-token vocabulary at --tp %d; "
                    "the contiguous equal split this sampler assumes wants %lld. The plugin and "
                    "core/engine.cpp disagree about which rows this rank owns.",
                    rank, (long long)sc.n_vocab_local, (long long)sc.n_vocab, cfg_.tp,
                    (long long)want);
            return RAD_E_INVAL;
        }
    }
    sc.max_seqs      = cfg_.max_seqs;
    sc.max_spec      = cfg_.n_spec;
    /* The real step bound, not the sizing declare's level: the sampler's planes must be the same
     * buffers in every declare of the program (plan_level compares them), and a step can sample
     * no more rows than the real bound lets it carry tokens. */
    sc.max_tok       = cfg_.max_tok;
    sc.rank          = rank;
    sc.world_size    = cfg_.tp;
    sc.draft_coupling = cfg_.draft_coupling;
    RAD_TRY(smp.declare(b, sc));

    /* Where the chain writes the sampled ids. The engine declares it because rad-kbench drives the
     * same chain against its own buffer. PERSIST, since the step loop copies it back after the
     * step, when a transient's storage may belong to another buffer. One id per sampler row,
     * capped as the rows are (SamplerConfig::max_tok). */
    RadBufDecl d{};
    d.dtype    = RAD_I32;
    d.rank     = 1;
    d.shape[0] = cfg_.max_seqs * (cfg_.n_spec > 0 ? cfg_.n_spec + 1 : 1);
    if (cfg_.max_tok > 0)
        d.shape[0] = std::min<int64_t>(d.shape[0], std::max<int64_t>(cfg_.max_tok, cfg_.max_seqs));
    d.kind     = RAD_BUF_PERSIST;
    tok = rad_decl_buffer(b, "sampler.tokens", &d);
    return tok ? RAD_OK : RAD_E_STATE;
}

/* ------------------------------------------------------------------ the tiering seam
 *
 * The heat engine dispatches per step, not per layer: the mover relocates only between steps
 * (Ctx::run_step syncs the residency table once at its top), so a mid-step table edit would be
 * seen by some issues and not others.
 *
 * It runs once per step, after the step's forward pass is queued:
 *
 *   - After, so it runs under the pass instead of in front of it (ahead of a draft pass the
 *     device is idle). Its changes cannot reach the pass just issued: table edits apply at the
 *     next run_step, and a freed slot is quarantined behind an event recorded after this pass.
 *   - On the trunk pass only, since a swap rarely clears the hysteresis between passes of one
 *     step and each call ranks the whole movable set. Draft-pass histograms accumulate in the
 *     step's bank and are credited next step, weighted by pass count.
 *   - `begin_dispatch` only retires; the issue call below is the other half. Retiring without
 *     issuing stalls the mover (mover.h).
 *   - It credits the previous step's histograms, whose copies land asynchronously while that step
 *     runs (spec §5.5); `histogram_ready` says which have landed.
 *
 * On long prefill chunks the host runs many steps ahead of the card, so most histograms have not
 * landed and are skipped (about 2% credited with 2048-token chunks, nearly all at decode shapes).
 * The counters report this, and the link panel's expert-streaming share reads near zero for the
 * same reason.
 *
 * `heat_seen` prevents crediting a bank twice: a bank keeps its counts until the next report, so a
 * step where a layer did not route (a drafter pass) would re-credit the stale plane.
 */
/* Every landed histogram of a step before `before_step`, oldest first per layer, into
 * `r.heat_rows`; nothing is credited yet. */
void DeviceRuntime::find_histograms(Rank& r, int before_step) {
    const int n_layer = r.ctx.n_route_layers();
    if ((int)r.heat_seen.size() != n_layer) r.heat_seen.assign((size_t)n_layer, -1);
    /* Find all histograms first, then credit them in order. Each is a card-written row in host
     * memory (every line a cache miss), and one credit is too little work to hide the next fetch.
     * Prefetching two rows ahead keeps 32 lines in flight, about what one core sustains. */
    int8_t landed[kRouteBanks];
    std::memset(landed, -1, sizeof landed);
    std::vector<Rank::HeatRow>& rows = r.heat_rows;
    rows.clear();
    for (int l = 0; l < n_layer; ++l) {
        /* Drain every landed histogram for the layer. The host runs this once per issued step,
         * possibly far ahead of the card, so several may have landed since the last visit, and the
         * newest is often still in flight. */
        int seen = r.heat_seen[(size_t)l];
        bool any = false;
        for (;;) {
            const int bank = r.ctx.next_histogram(l, seen, before_step, landed);
            if (bank < 0) break;
            const int32_t* h = r.ctx.last_histogram(l, bank);
            if (!h) break;
            rows.push_back({l, bank, h});
            seen = r.ctx.last_histogram_step(l, bank);
            any = true;
        }
        if (!any) ++r.heat_idle;
    }
}

/* --expert-stats: one found histogram into the rank's tally (place/xstats.h), from the same row
 * the heat engine reads. The caller holds the tally. */
static void xstats_credit(XsTally* xs, Ctx& ctx, int layer, int bank, const int32_t* h) {
    const RadRouting* rt = ctx.last_routing(layer);
    const int64_t n = rt ? std::min<int64_t>(rt->n_expert, ctx.route_n_expert()) : 0;
    xs->credit(layer, ctx.last_histogram_step(layer, bank), h, n,
               ctx.last_histogram_reports(layer, bank), rt ? rt->top_k : 0);
}

/* The drain alone, for the tally, where the heat engine is off or the engine is stopping
 * (`before_step` INT_MAX once the streams are drained). */
void DeviceRuntime::xstats_drain(Rank& r, XsTally* xs, int before_step) {
    find_histograms(r, before_step);
    std::lock_guard<std::mutex> lk(xs->mutex());
    for (const Rank::HeatRow& hr : r.heat_rows) {
        xstats_credit(xs, r.ctx, hr.layer, hr.bank, hr.h);
        r.heat_seen[(size_t)hr.layer] = r.ctx.last_histogram_step(hr.layer, hr.bank);
        r.ctx.mark_credited(hr.layer, hr.bank);
        ++r.heat_credited;
    }
    xs->note_dropped(r.ctx.route_dropped());
}

void DeviceRuntime::tiering_dispatch(Rank& r, int step) {
    XsTally* const xs = xstats_ ? xstats_->tally(r.index) : nullptr;
    if (!r.heat.enabled()) {
        if (xs) xstats_drain(r, xs, step);
        return;
    }

    r.mover.begin_dispatch(step, &r.heat);

    find_histograms(r, step);
    std::vector<Rank::HeatRow>& rows = r.heat_rows;
    const int32_t n_ex = (int32_t)r.ctx.route_n_expert();
    constexpr size_t kAhead = 2;
    std::unique_lock<std::mutex> xl;
    if (xs) xl = std::unique_lock<std::mutex>(xs->mutex());
    for (size_t i = 0; i < rows.size() && i < kAhead; ++i) r.heat.prefetch(rows[i].h, n_ex);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i + kAhead < rows.size()) r.heat.prefetch(rows[i + kAhead].h, n_ex);
        const Rank::HeatRow& hr = rows[i];
        /* A bank can hold several passes (a draft head reports once per pass into one bank), so
         * weight it by the report count, or the layer's experts are underpriced and drained. */
        const int reps = r.ctx.last_histogram_reports(hr.layer, hr.bank);
        r.heat.observe(hr.layer, hr.h, n_ex, reps > 1 ? (float)reps : 1.0f);
        /* The row is in cache now. */
        if (xs) xstats_credit(xs, r.ctx, hr.layer, hr.bank, hr.h);
        r.heat_seen[(size_t)hr.layer] = r.ctx.last_histogram_step(hr.layer, hr.bank);
        r.ctx.mark_credited(hr.layer, hr.bank);
        ++r.heat_credited;
    }
    if (xs) {
        xs->note_dropped(r.ctx.route_dropped());
        xl.unlock();
    }

    /* No swaps behind a pass the stager armed: that step is bound by its own copies on the same
     * transfer stream, and the turnover slots are filled for it (below). Histograms are still
     * credited; the next unstaged step swaps on them. */
    std::vector<Swap> swaps;
    if (!r.stager.staged() && r.heat.step(&swaps) < 0) return;
    for (const Swap& s : swaps) {
        /* Contract: every proposed unit is busy until `issue` takes it or `abandon` returns it;
         * a dropped proposal stays busy forever. */
        if (r.mover.issue(s, &r.heat) < 0) r.heat.abandon(s);
    }

    /* Fill after the swaps: swaps are the policy's decisions, fills only use spare memory, and a
     * fill must not take a move record the next swap needs.
     *
     * Behind a staged pass the turnover slots are filled too, since that step swaps nothing; left
     * free, the stager would copy those residents instead (at TP2 a 160K-token prompt staged 9.6%
     * more and ran 3-5% slower).
     *
     * They stay filled until kTurnoverAfter consecutive unstaged steps (idle ticks included).
     * Otherwise a short answer or a pause drops them and the next prompt's first chunk copies them
     * back, scattering the lent residents through the pool runs the stager coalesces on (at TP2,
     * back-to-back 26K-token prompts: 33 copies per staged layer instead of 16). Decode reaches the
     * count in a few steps. */
    constexpr int kTurnoverAfter = 4;
    r.unstaged_steps = r.stager.staged() ? 0 : r.unstaged_steps + 1;
    const bool hold = r.unstaged_steps < kTurnoverAfter;
    r.mover.hold_turnover(hold);
    (void)r.mover.balance(&r.heat, !hold);
}

/* ------------------------------------------------------------------ --expert-stats
 *
 * Which model-wide expert each index of each rank's histogram is (place/xstats.h, XsMap): index e
 * is the rank's unit with group expert e, and the container's directory entry for that unit's
 * weight names the expert in the model's numbering. A checkpoint served directly has no such entry
 * and the map falls back to rank order. */
int DeviceRuntime::xstats_open() {
    if (cfg_.expert_stats.empty()) return RAD_OK;
    const int n_layer = ranks_[0]->ctx.n_route_layers();
    if (n_layer <= 0) {
        RAD_WARN("--expert-stats: '%s' routes no experts; there is nothing to count and %s is not "
                 "written", meta_.name ? meta_.name : "the model", cfg_.expert_stats.c_str());
        return RAD_OK;
    }
    constexpr int32_t kUnset = INT32_MIN, kClash = -3;
    std::vector<std::vector<std::vector<int32_t>>> ids(ranks_.size());
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        const Program& P = ranks_[ri]->program;
        auto& layers = ids[ri];
        layers.assign((size_t)n_layer, {});
        for (size_t w = 1; w < P.weights.size(); ++w) {
            const WeightInfo& wi = P.weights[w];
            const int32_t l = wi.decl.group.layer, e = wi.decl.group.expert;
            if (l < 0 || l >= n_layer || e < 0) continue;
            std::vector<int32_t>& row = layers[(size_t)l];
            if ((size_t)e >= row.size()) row.resize((size_t)e + 1, kUnset);
            const RadFileEntry* fe = file_ ? file_->find(wi.source) : nullptr;
            const int32_t g = fe && fe->layer == l && fe->expert >= 0 ? fe->expert : -1;
            int32_t& at = row[(size_t)e];
            /* Every weight of one unit must name the same expert. */
            at = at == kUnset ? g : at == g ? at : kClash;
        }
        for (auto& row : layers)
            for (int32_t& v : row) if (v == kUnset || v < 0) v = -1;
    }
    std::vector<int32_t> n_model((size_t)n_layer, 0);
    for (int l = 0; l < n_layer && l < meta_.n_layers; ++l) n_model[(size_t)l] = (int32_t)meta_.n_expert;
    XsMap map;
    RAD_TRY(xs_map_build(ids, n_model, (int32_t)meta_.n_layers, &map));

    XsSnapshot info;
    info.model = meta_.name ? meta_.name : "";
    info.arch = meta_.arch_id ? meta_.arch_id : "";
    info.container = cfg_.model;
    xstats_ = std::make_unique<ExpertStats>();
    if (const int st = xstats_->init(cfg_.expert_stats, map, std::move(info)); st < 0) {
        RAD_ERR("--expert-stats: cannot write %s", cfg_.expert_stats.c_str());
        xstats_.reset();
        return st;
    }
    std::map<std::string, int> how;
    int32_t mx = 0;
    for (int l = 0; l < n_layer; ++l) {
        if (map.n_expert[(size_t)l] <= 0) continue;
        ++how[map.how[(size_t)l]];
        mx = std::max(mx, map.n_expert[(size_t)l]);
    }
    std::string hs;
    for (const auto& [k, v] : how) hs += fmt("%s%d by %s", hs.empty() ? "" : ", ", v, k.c_str());
    int counting = 0;
    for (size_t ri = 0; ri < ranks_.size(); ++ri) counting += xstats_->tally((int)ri) != nullptr;
    RAD_INFO("expert stats: %d routed layers (up to %d experts; ids %s), counted on %d of %d "
             "rank%s, written to %s every 30 s and at shutdown",
             n_layer, (int)mx, hs.c_str(), counting, cfg_.tp, cfg_.tp == 1 ? "" : "s",
             cfg_.expert_stats.c_str());
    xstats_->start();
    return RAD_OK;
}

void DeviceRuntime::xstats_close() {
    if (!xstats_) return;
    xstats_->stop();
    /* The banks of the last steps are still in the ring: drain each counting rank's stream, then
     * credit them. Not after a step-loop error, where a stream may never drain. */
    if (loop_status_.load(std::memory_order_acquire) >= 0) {
        for (auto& r : ranks_) {
            XsTally* xs = xstats_->tally(r->index);
            if (!xs) continue;
            if (rad_dev_set(r->device) < 0 || rad_stream_sync(r->ctx.stream()) < 0) continue;
            xstats_drain(*r, xs, INT_MAX);
        }
        if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    }
    (void)xstats_->write(true);
}

/* ------------------------------------------------------------------ the rank thread */
void DeviceRuntime::rank_loop(int index) {
    Rank& r = *ranks_[(size_t)index];
    r.native_thread_id.store(syscall(SYS_gettid), std::memory_order_release);
    const char* delay_text = std::getenv("FREETOKEN_NATIVE_TEST_RANK_DELAY_US");
    long delay_us = 0;
    if (delay_text && *delay_text) {
        char* end = nullptr;
        delay_us = std::strtol(delay_text, &end, 10);
        if (!end || *end || delay_us < 0 || delay_us > 100000)
            fatal("FREETOKEN_NATIVE_TEST_RANK_DELAY_US must be 0..100000");
    }
    if (r.ctx.bind_thread() < 0) {
        RAD_ERR("rank %d: could not bind its device", index);
        step_status_.store(RAD_E_DEVICE, std::memory_order_release);
    }

    for (;;) {
        barrier_->arrive_and_wait();                    /* released against this step's batch */
        if (stopping_.load(std::memory_order_acquire)) break;

        const RadBatch* b = step_batch_.load(std::memory_order_acquire);
        if (b) {
            if (index == cfg_.tp - 1 && delay_us > 0) {
                RAD_INFO("native qualification: rank %d waits %ld us before its first pass", index, delay_us);
                std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
                delay_us = 0;
            }
            /* This rank's view of the step, pointing at this card's staging. Its upload was issued
             * on this rank's stream, so no cross-rank fence is needed. */
            const RadBatch* mine = gen_ ? gen_->batch_for(index) : sched_.batch_for(index);
            if (!mine) {
                RAD_ERR("rank %d: the builder has no batch for this rank", index);
                step_status_.store(RAD_E_STATE, std::memory_order_release);
                barrier_->arrive_and_wait();
                continue;
            }
            b = mine;
            const int what = issue_what_.load(std::memory_order_acquire);
            /* --expert-stats files each pass's step kind for the drain, which reads it a step or
             * more later. An encoder pass is not a step of the token stream. */
            if (xstats_ && what != kIssueSampler && b->enc == 0)
                if (XsTally* xs = xstats_->tally(index)) xs->issued(*b);
            int s = what == kIssueSampler ? RAD_OK : r.ctx.run_step(b);
            if (s < 0) {
                RAD_ERR("rank %d step %d: %s", index, b->step, r.ctx.step_error());
                step_status_.store(s, std::memory_order_release);
            } else if (b->n_out > 0 && what != kIssueModel) {
                /* Wait for the scheduler's sampler staging, which runs while the device executes
                 * the forward pass enqueued above, so the wait is hidden behind device work. */
                int bst = RAD_OK;
                {
                    std::unique_lock<std::mutex> lk(build_mu_);
                    build_cv_.wait(lk, [this] { return build_ready_; });
                    bst = build_status_;
                }
                /* The sampler chain goes on the same stream behind the model with no host sync
                 * (spec §13). */
                if (bst >= 0) {
                    /* The scored rows first: the sampler may rewrite a row in place. */
                    if (b->draft_pass == 0 && !score_rows_.empty()) s = issue_scored_rows(r);
                    if (s >= 0) s = r.sampler.run(&r.ctx, r.program.logits_buf, r.tok_buf);
                    if (s < 0) {
                        RAD_ERR("rank %d step %d: sampler: %s", index, b->step, rad_strerror(s));
                        step_status_.store(s, std::memory_order_release);
                    }
                }
            }
            /* Behind the pass, once a step: see tiering_dispatch. */
            if (s >= 0 && b->draft_pass == 0 && what != kIssueModel) tiering_dispatch(r, b->step);

        }
        barrier_->arrive_and_wait();                    /* joined; the scheduler may read */
    }
}

/* ------------------------------------------------------------------ the scheduler thread */
int DeviceRuntime::prepare_release(const RadBatch* b) {
    /* The arena level, recalling lent arena memory if this step needs it. Every rank thread is
     * parked, so the level can change and a recall can drain both streams. */
    if (const int as = arena_step(b); as < 0) return as;
    /* Grow the followers' pools to match rank 0's before the step's block table is used:
     * admission grew rank 0's cache and the new ids are already in the table. Done here, not on
     * the 100 ms arbiter tick, because the step is about to run; safe because every rank thread is
     * parked. */
    for (size_t i = 1; i < ranks_.size(); ++i) {
        if (rad_dev_set(ranks_[i]->device) < 0) continue;
        if (ranks_[i]->kv.follow(ranks_[0]->kv) < 0)
            step_status_.store(RAD_E_NOMEM, std::memory_order_release);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    /* The stage's ranges for every rank (kvstream/stage.h, "Volume"): the blocks the step's own
     * tables name. Rank 0's allocator hands out the block ids every rank uses, and the ranks are
     * parked, so no allocation races the read. */
    kvstream_step_runs(ranks_[0]->kv, b, sched_.step_seqs(b), &stage_runs_);
    for (auto& rk : ranks_) kvstream_bound_live(&rk->ctx, stage_runs_);
    return RAD_OK;
}

int DeviceRuntime::issue_ahead(const RadBatch** out, ExternalPermit* permit) {
    *out = nullptr;
    if (!pipelining_ || !sched_.supports_issue_ahead()) return RAD_OK;
    /* A flex release or a quiet window waits for a step boundary with nothing in flight. */
    if (flex_pending_.load(std::memory_order_acquire) || quiet_pending_.load(std::memory_order_acquire))
        return RAD_OK;
    ExternalPermit authorized;
    if (external_steps_ && !external_steps_->acquire(authorized, false, stopping_))
        return RAD_OK;
    for (int i = 0; i < serial_rows_ && serial_done_ > 0; ++i)
        sched_.expect_draft(serial_row_[(size_t)i].seq, serial_done_);
    for (int i = 0; i < block_rows_; ++i)
        sched_.expect_draft(block_row_[(size_t)i].seq, block_steps_);
    if (external_steps_) {
        const int prepared = external_steps_->prepare(authorized, sched_, true);
        if (prepared < 0) return prepared;
        if (prepared > 0) { external_steps_->defer(std::move(authorized)); return RAD_OK; }
    }
    const RadBatch* nb = sched_.step_next();
    if (!nb) {
        if (external_steps_) external_steps_->defer(std::move(authorized));
        return RAD_OK;
    }
    if (external_steps_) {
        RAD_TRY(external_steps_->validate(authorized, sched_.next_plan(), *nb));
        if (!permit) return RAD_E_STATE;
        *permit = std::move(authorized);
    }
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
    kv_test_prepare(sched_.plan(), nb->step, ranks_);
#endif
    /* Its rows' tokens, positions and slots, advanced on the card behind this step's acceptance
     * and drafts. */
    RAD_TRY(advance_all(ADV_TRUNK, 0, false));
    RAD_TRY(prepare_release(nb));
    issue_what_.store(kIssueModel, std::memory_order_release);
    step_batch_.store(nb, std::memory_order_release);
    barrier_->arrive_and_wait();
    barrier_->arrive_and_wait();
    const int st = step_status_.load(std::memory_order_acquire);
    if (st < 0) {
        RAD_ERR("step %d, issued ahead, failed: %s -- the engine cannot continue", nb->step,
                rad_strerror(st));
        return st;
    }
    *out = nb;
    return RAD_OK;
}

void DeviceRuntime::phase_mark(int p) {
    const auto now = std::chrono::steady_clock::now();
    phase_us_[p] += std::chrono::duration<double, std::micro>(now - phase_t_).count();
    phase_t_ = now;
}

/* Every error exit from the step loop goes through here, which does three things:
 *
 *   - aborts the barrier, so the parked rank threads see `stopping_`;
 *   - fails every queued or running request and answers its connection; the scheduler then
 *     refuses new submissions;
 *   - sets `loop_status_`, which run_server watches to stop HTTP, so the process exits with the
 *     error instead of queueing requests nothing will drain (spec §17). */
int DeviceRuntime::fail_loop(int st) {
    stopping_.store(true, std::memory_order_release);
    wake_hold();
    if (barrier_) barrier_->abort();
    sched_.shutdown("the engine stopped on an error and is not serving", st < 0 ? st : RAD_E_STATE);
    if (external_steps_) external_steps_->fault(st);
    loop_status_.store(st < 0 ? st : RAD_E_STATE, std::memory_order_release);
    return st;
}

int DeviceRuntime::step_loop() {
    /* The batch is built on rank 0's stream into rank 0's memory, so this thread binds rank 0's
     * device; the backend's current device is thread-local and starts at device 0. */
    if (rad_dev_set(ranks_[0]->device) < 0)
        RAD_ERR("step loop: could not bind rank 0's device");
    StepOut so;
    int idle = 0;
    std::vector<uint64_t> cancel;
    /* Not in KL mode: its scored rows are copied out of the logits buffer after collection, and a
     * pass issued ahead would overwrite them. */
    pipelining_ = !cfg_.kld() && !cfg_.calib() && !embedding_ &&
                  (serial_drafting() || block_drafting() || cfg_.n_spec <= 0);
    if (external_steps_ && !external_steps_->pipeline()) pipelining_ = false;
    early_ids_ = !cfg_.kld() && !embedding_;
    /* A tree's acceptance moves K and V on the card behind the step issued ahead. The KL mode
     * drafts nothing and an embedding model declares no drafter, so no server reaches this today;
     * it keeps a tree from running unverified should another path turn pipelining off. */
    if (tree_ && !pipelining_) {
        RAD_ERR("speculation: drafter '%s' proposes trees, which are verified only on steps issued "
                "ahead, and this server issues none; ask the plugin for a chain", drafter_.name);
        return fail_loop(RAD_E_UNSUPPORTED);
    }
    const bool t_on = (int)log_level() >= (int)Log::Debug;
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
    if (std::getenv("RADIANCE_TEST_KV_ROLLBACK_REPORT")) {
        sched_.set_kv_rollback_test_hook([this](uint64_t seq, int proposals, int accepted,
                                                int64_t valid, bool before) {
            if (!ranks_.empty()) kv_test_rollback(seq, proposals, accepted, valid, before,
                                                   *ranks_[0]);
        });
    }
#endif
    /* The next step's forward pass when already issued behind the previous step, before its
     * readback (Scheduler::step_next). */
    const RadBatch* ahead = nullptr;
    ExternalPermit ahead_permit;
    /* A step that runs for 20 s reports where the step loop waits: its syscall and its stack.
     * Every iteration, idle or not, marks progress, so only a stuck step is reported. */
    rad::StallWatch step_watch("step", 20000);
    while (!stopping_.load(std::memory_order_acquire)) {
        step_watch.progress();
        lora_epoch_.fetch_add(1, std::memory_order_acq_rel);
        phase_t_ = std::chrono::steady_clock::now();
        /* Encode media for waiting requests while nothing is in flight. The scheduler issues no
         * step ahead while a request waits, and a request with unencoded media waits. */
        if (!ahead && sched_.encodes_pending()) {
            if (const int es = run_encodes(); es < 0) return fail_loop(es);
        }
        if (!ahead) {
            flex_service();
            quiet_service();
            flex_room_tick(rad_mono_ns());
        }
        const bool issued = ahead != nullptr;
        const RadBatch* b = ahead;
        ExternalPermit permit;
        if (external_steps_) {
            if (issued) permit = std::move(ahead_permit);
            else if (!external_steps_->acquire(permit, true, stopping_)) break;
        }
        ahead = nullptr;
        /* Read before the step looks for work, so a request added after it found none still wakes
         * the idle wait below. */
        const uint64_t woke = issued ? 0 : sched_.wake_seq();
        if (!issued && external_steps_) {
            const int prepared = external_steps_->prepare(permit, sched_);
            if (prepared < 0) return fail_loop(prepared);
        }
        if (issued) sched_.promote_next();
        else        b = sched_.step();
        if (!b) {
            if (external_steps_) external_steps_->defer(std::move(permit));
            /* Idle: back off in small steps instead of spinning; the first wait is short so a new
             * request into an empty engine is not delayed. */
            step_batch_.store(nullptr, std::memory_order_release);
            drain_completed();
            sched_.publish(/*idle=*/true);
            /* Work is waiting only on the encoder: loop immediately. */
            if (sched_.encodes_pending()) { idle = 0; continue; }
            /* Release deferred by the last step. A request cancelled while a drafter pass was being
             * built keeps its blocks until the passes naming them run (Scheduler::cancel), and
             * commit() normally releases them; an idle engine makes no commit, and those blocks
             * may be what blocks the next admission. Drain the streams first so no in-flight pass
             * still reads them. */
            if (sched_.has_deferred()) {
                for (auto& rk : ranks_) (void)rad_stream_sync(rk->ctx.stream());
                sched_.release_deferred();
            }
            tier_tick();
            /* A park waits here once the tiers' copies have landed, before the idle work that
             * would touch the card (the loan's fills). */
            if (hold_req_.load(std::memory_order_acquire) && !(pump_ && !pump_->copy.empty())) {
                hold_point(&step_watch);
                continue;
            }
            expire_tick(rad_mono_ns());
            loan_tick(/*idle=*/true);
            xflex_tick(/*busy=*/false);
            /* A request (or a cancel, a park, a quiet call) wakes the wait at once; otherwise it
             * lengthens, up to 20 ms, which the periodic idle work above tolerates. Polling at
             * 1 ms woke an idle engine a thousand times a second. */
            if (idle < 200) ++idle;
            sched_.wait_idle(woke, std::chrono::microseconds(idle * 100));
            continue;
        }
        idle = 0;
        if (external_steps_) {
            const int checked = external_steps_->validate(permit, sched_.plan(), *b);
            if (checked < 0) return fail_loop(checked);
        }
        if (t_on) phase_mark(PH_PLAN);

        clear_fresh_states();

        /* Entries that want logits rows, in BatchBuilder's out_ids order: those with
         * produces_token, 1 + n_spec rows each. Row and sequence indices differ (a mid-prompt
         * chunk produces no row). */
        const StepPlan& plan = sched_.plan();
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
        kv_test_prepare(plan, b->step, ranks_);
#endif
        /* One entry per request; Sampler::build_step expands each into 1 + n_draft rows itself
         * (giving each position its RNG coordinate), so one push per row would stage (1+n)^2 rows.
         * `first_row` indexes sampled tokens, so it counts rows. */
        so.reqs.clear();
        so.first_row.assign(plan.e.size(), -1);
        int32_t logit_row = 0;
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (!e.produces_token) continue;
            so.first_row[i] = logit_row;
            logit_row += 1 + e.n_spec;
            so.reqs.push_back(const_cast<Request*>(e.req));
        }
        so.n_rows = logit_row;
        if (const int ss = stage_scored_rows(plan, so, cancel); ss < 0) return fail_loop(ss);

        /* The sampler's host work, run while the ranks enqueue (see build_mu_ in engine.h). It
         * touches only request state and host grammar automata; Sampler::run issues the uploads
         * after the gate opens.
         *
         * Cancellations are collected, not applied: the rank threads read scheduler state through
         * batch_for() meanwhile. They are applied after the join, before collect_step. */
        auto stage_sampler = [&]() -> int {
        if (!so.reqs.empty() && !embedding_) {
            /* Per-request sampler state is built on first sight: the grammar (compiled at
             * admission) is positioned and DRY sequence breakers are resolved against the
             * vocabulary. Without this, `grammar` and `dry_sequence_breakers` would silently do
             * nothing. Every rank does it, since each masks its own vocabulary shard.
             *
             * A failure (e.g. a left-recursive grammar) cancels the request, not the engine. The
             * server compiles at admission, so this should be unreachable. */
            for (Request* rq : so.reqs) {
                if (!rq) continue;
                bool failed = false;
                for (auto& rk : ranks_) {
                    if (rk->sampler.has_request(rq->id)) continue;
                    std::string err;
                    if (rk->sampler.begin_request(rq->id, rq->sp, &err) < 0) {
                        RAD_ERR("sampler: request %llu could not be started: %s; cancelling it",
                                (unsigned long long)rq->id, err.c_str());
                        failed = true;
                        break;
                    }
                }
                if (failed) {
                    for (auto& rk : ranks_) rk->sampler.end_request(rq->id);
                    cancel.push_back(rq->id);
                    continue;
                }
                rq->speculation_fixed = ranks_[0]->sampler.speculation_fixed(rq->id);
            }
            if (t_on) phase_mark(PH_BEGIN);
            for (auto& rk : ranks_) {
                const int s = rk->sampler.build_step(so.reqs, b->step);
                if (s < 0) { RAD_ERR("sampler build_step: %s", rad_strerror(s)); return s; }
                /* A request the sampler could not serve is cancelled; the step goes on. */
                for (uint64_t id : rk->sampler.take_failed()) cancel.push_back(id);
            }
            if (t_on) phase_mark(PH_BUILD);
        }
        return RAD_OK;
        };

        /* No clear here: `cancel` is empty when the step's staging begins (applied and cleared at
         * the join below), and stage_scored_rows may already have filled it — a scored request
         * whose pinned rows could not be allocated rides the same deferred path. */
        int bst = RAD_OK;

        /* RADIANCE_LOG_STEPS: one line per step with its phase and each sequence's slot, query
         * length and context. Off by default (one line per decode step). */
        static const bool log_steps = std::getenv("RADIANCE_LOG_STEPS") != nullptr;
        if (log_steps) {
            std::string s = fmt("step %d %s n_seq=%lld n_tok=%lld", b->step,
                                b->phase == RAD_PHASE_PREFILL ? "prefill" :
                                b->phase == RAD_PHASE_DECODE  ? "decode"  : "MIXED",
                                (long long)b->n_seq, (long long)b->n_tok);
            for (int i = 0; i < b->n_seq && i < 8; ++i)
                s += fmt("  [%d] slot=%d q=%d ctx=%d", i, plan.e[(size_t)i].slot,
                         plan.e[(size_t)i].n_tokens, plan.e[(size_t)i].ctx_len);
            RAD_INFO("%s", s.c_str());
        }

        /* Restore the recurrent half of a prefix hit before the step that reads it (the adopted
         * blocks carry the attention layers), copying from the snapshot's host slot. Enqueued on
         * each rank's stream before the release, so no synchronisation is needed.
         *
         * Under the scheduler's lock, because an HTTP thread's cancel mutates the sequence table.
         * Taken only when the step has a restore. */
        bool any_restore = false;
        for (const StepEntry& e : plan.e) if (e.ckpt_restore >= 0) { any_restore = true; break; }
        if (any_restore) {
            const int rs = sched_.locked([&]() -> int {
                for (const StepEntry& e : plan.e) {
                    if (e.ckpt_restore < 0) continue;
                    if (!state_slots_of(e.seq, &ckpt_slots_)) continue;
                    /* The window behind the hit, in the blocks admit_one seeded. */
                    if (ranks_[0]->kv.has_window_snapshot() &&
                        !ranks_[0]->kv.window_rows(e.seq, e.ctx_len, &ckpt_win_)) {
                        RAD_ERR("sequence %llu: no window rows at %d to restore into",
                                (unsigned long long)e.seq, (int)e.ctx_len);
                        return RAD_E_STATE;
                    }
                    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
                        auto& rk = ranks_[ri];
                        if (ranks_.size() > 1) (void)rad_dev_set(rk->device);
                        const int s = rk->kv.checkpoint_restore(ckpt_slots_.data(),
                                                                (int)ckpt_slots_.size(),
                                                                e.ckpt_restore, rk->ctx.stream(),
                                                                &ckpt_win_);
                        if (s < 0) {
                            RAD_ERR("could not restore sequence %llu from checkpoint slot %d on "
                                    "rank %d: %s", (unsigned long long)e.seq,
                                    (int)e.ckpt_restore, (int)ri, rad_strerror(s));
                            if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
                            return s;
                        }
                    }
                    const int pa = audit_prefix_pages(e.seq, e.ctx_len, e.ckpt_restore, false);
                    if (pa < 0) return pa;
                    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
                }
                return RAD_OK;
            });
            if (rs < 0) return fail_loop(rs);
        }

        /* Host time before the release, with the device idle; time between the two arrivals is
         * the ranks enqueueing. The split separates host-bound from device-bound steps. */
        if (t_on) phase_mark(PH_PRE);

        if (!issued) {
            /* Reset the gate while every rank is parked, so none sees a stale ready flag. */
            {
                std::lock_guard<std::mutex> lk(build_mu_);
                build_ready_  = false;
                build_status_ = RAD_OK;
            }
            if (const int ps = prepare_release(b); ps < 0) return fail_loop(ps);
            /* An embedding program has no sampler to issue behind its pass. */
            issue_what_.store(embedding_ ? kIssueModel : kIssueAll, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            barrier_->arrive_and_wait();      /* release the ranks */
            {
                /* Stage the sampler while the ranks enqueue and the device runs the pass. */
                bst = stage_sampler();
                {
                    std::lock_guard<std::mutex> lk(build_mu_);
                    build_ready_  = true;
                    build_status_ = bst;
                }
                build_cv_.notify_all();
            }
            barrier_->arrive_and_wait();      /* wait for them */
        } else {
            /* The forward pass was issued ahead and the previous step has now committed, so the
             * sampler is staged from the correct state and its chain queued behind the running
             * pass. */
            bst = stage_sampler();
            {
                std::lock_guard<std::mutex> lk(build_mu_);
                build_ready_  = true;
                build_status_ = bst;
            }
            issue_what_.store(kIssueSampler, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            barrier_->arrive_and_wait();
            barrier_->arrive_and_wait();
        }
        if (t_on) phase_mark(PH_RANKS);

        /* Deferred out of the staging, where the rank threads were reading the scheduler. */
        for (uint64_t id : cancel) sched_.cancel(id);
        cancel.clear();

        /* A staging failure is known only after the ranks were released; the gate passed it to
         * them, so they skipped the sampler and reached the join. */
        if (bst < 0) return fail_loop(bst);

        int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            /* A device fault is not recoverable in-process (spec §17). */
            RAD_ERR("step %d failed: %s -- the engine cannot continue", b->step, rad_strerror(st));
            return fail_loop(st);
        }

        /* Snapshot after the step that produced the state (save_checkpoints). Where a block
         * drafter's window rides in the snapshot, its context pass must store the step's rows
         * first, so the card's driver saves right behind that pass instead. */
        if (!(block_drafting() && drafter_snap_)) {
            const int ss = save_checkpoints();
            if (ss < 0) return fail_loop(ss);
        }

        if (t_on) phase_mark(PH_SAVE);
        /* Queue the drafter's passes behind the step before waiting: the card computes acceptance
         * itself and every pass reads it there (sched/advance.h), so one wait covers the whole
         * step. */
        st = block_drafting() ? issue_block_drafter(b->step, so)
                              : issue_serial_drafter(b->step, so);
        if (st < 0) return fail_loop(st);
        /* Behind the step's last resolve and before its end event, so collect_step's wait covers it. */
        st = stream_probe_issue(b);
        if (st < 0) return fail_loop(st);
        st = mark_step_end(b, so);
        if (st < 0) return fail_loop(st);
        /* Then the next step's forward pass, before this step is read back. Its rows continue from
         * the card's acceptance and verify the new drafts; its sampler waits for this step's
         * commit. The card runs straight on while the host waits, commits and stages. */
        st = issue_ahead(&ahead, &ahead_permit);
        if (st < 0) return fail_loop(st);
        st = embedding_ ? collect_pooled(b, so) : collect_step(b, so);
        if (t_on) phase_mark(PH_COLLECT);
        if (st < 0) return fail_loop(st);
        /* The step's sampler chain, verify op included, has returned: one line at the first
         * coupled verify that actually completed. */
        couple_verify_evidence(b->step);
        if (cfg_.kld()) {
            st = score_rows(b, so);
            if (st < 0) return fail_loop(st);
        }

        /* Buffer probe after the trunk step; drafters dump after their own passes. */
        dump_buffers(ranks_[0]->ctx, b->step);

        /* --profile-ops also prints during the run. Counters are cumulative, so each dump covers
         * the run so far. Every 32 steps by default (RADIANCE_PROFILE_EVERY): per-op timing
         * synchronises around every launch (over a thousand per step on a large model), so a
         * profiled step is several times slower. */
        static const int prof_every = [] {
            const char* v = std::getenv("RADIANCE_PROFILE_EVERY");
            const int n = (v && *v) ? (int)std::strtol(v, nullptr, 10) : 32;
            return n > 0 ? n : 32;
        }();
        if (cfg_.profile_ops && b->step > 0 && (b->step % prof_every) == 0)
            for (auto& rk : ranks_) rk->ctx.dump_profile();

        /* One line per step at -vv, the only log view of the loop itself. */
        RAD_TRACE("step %d: %lld tok / %lld seq, %lld logit row(s) -> tokens %s", b->step,
                  (long long)b->n_tok, (long long)b->n_seq, (long long)b->n_out,
                  so.token.empty() ? "(none)" : fmt("%d%s", so.token[0],
                                                    so.token.size() > 1 ? " ..." : "").c_str());
        /* And the input ids (first 32): wrong ids look like a wrong forward pass from the logits
         * alone. */
        if ((int)log_level() >= (int)Log::Trace && b->n_tok > 0 && b->token_ids) {
            std::string ids;
            const int64_t n = std::min<int64_t>(b->n_tok, 32);
            std::vector<int32_t> host((size_t)n);
            if (rad_memcpy_async(host.data(), b->token_ids, n * 4, ranks_[0]->ctx.stream()) >= 0 &&
                rad_stream_sync(ranks_[0]->ctx.stream()) >= 0) {
                for (int64_t i = 0; i < n; ++i) ids += fmt("%s%d", i ? " " : "", host[(size_t)i]);
                RAD_TRACE("step %d in: [%s%s]", b->step, ids.c_str(),
                          b->n_tok > n ? " ..." : "");
            }
        }

        BatchState::StepResult res{};
        res.n_seq      = (int)plan.e.size();
        res.n_accepted = so.accepted.data();
        res.token      = so.token.data();
        res.eos        = so.eos.data();
        if (!so.lp.empty()) {
            res.lp        = so.lp.data();
            res.first_row = so.first_row.data();
        }
        res.path = so.path.empty() ? nullptr : so.path.data();
#ifdef RAD_ENABLE_KV_ROLLBACK_TEST_HOOK
        kv_test_before_commit(plan, so, b->step, ranks_);
#endif
        st = sched_.commit(res);
        if (t_on) phase_mark(PH_COMMIT);
        if (st < 0) { RAD_ERR("commit: %s", rad_strerror(st)); return fail_loop(st); }
        /* The step has drained, so its counters are readable. A chunk that produced no row did not
         * wait, so its copy is read on a later step; a finished prompt or request always produced
         * a row. */
        if (so.n_rows > 0) {
            st = stream_probe_take(b, res.n_prefill_done > 0, res.n_finished > 0);
            if (st < 0) return fail_loop(st);
        }

        /* With the ids read ahead of the drafter's passes, the commit streamed the tokens while
         * the card ran them. Wait for the passes now, before the slots below are read and before
         * anything reuses what the commit released. */
        if (so.early) {
            st = rad_event_sync(step_end_);
            if (st < 0) { RAD_ERR("step %d: waiting for its draft passes: %s", b->step,
                                  rad_strerror(st)); return fail_loop(st); }
            so.early = false;
        }

        /* The drafter for the next step already ran on the card; what remains is to check its
         * acceptance against the host's and publish its proposals. Each driver returns
         * immediately for a kind that is not its own. */
        st = block_drafting() ? finish_block_drafter(so) : finish_serial_drafter(so);
        if (st < 0) return fail_loop(st);
        if (external_steps_) {
            st = external_steps_->completed(permit, plan, *b,
                serial_rows_>0?serial_done_:0,so.accepted.empty()?0:so.accepted[0]);
            if (st < 0) return fail_loop(st);
        }
        /* The step ends here, after its draft rounds, so step timing includes them
         * (Scheduler::step_finished). */
        sched_.step_finished();
        tier_tick();
        loan_tick(/*idle=*/false);
        xflex_tick(/*busy=*/true);
        const uint64_t now = rad_mono_ns();
        arena_tick(now);
        expire_tick(now);
        drain_completed();
        sched_.publish(/*idle=*/false);

        /* One line every 128 steps; the phases sum to the step. `ranks` is the enqueue; device time
         * shows up in `collect`, the step's only host synchronisation. */
        if (t_on) {
            phase_mark(PH_DRAFT);
            if (++phase_n_ == 128) {
                double tot = 0;
                for (int i = 0; i < PH_N; ++i) tot += phase_us_[i];
                RAD_DEBUG("step %s: %.2f ms = plan %.2f begin %.2f build %.2f pre %.2f "
                          "ranks %.2f save %.2f collect %.2f commit %.2f draft %.2f (mean/128)",
                          b->phase == RAD_PHASE_DECODE ? "decode" : "other", tot / 128000.0,
                          phase_us_[PH_PLAN] / 128000.0,  phase_us_[PH_BEGIN] / 128000.0,
                          phase_us_[PH_BUILD] / 128000.0, phase_us_[PH_PRE] / 128000.0,
                          phase_us_[PH_RANKS] / 128000.0, phase_us_[PH_SAVE] / 128000.0,
                          phase_us_[PH_COLLECT] / 128000.0, phase_us_[PH_COMMIT] / 128000.0,
                          phase_us_[PH_DRAFT] / 128000.0);
                for (int i = 0; i < PH_N; ++i) phase_us_[i] = 0;
                phase_n_ = 0;
            }
        }
    }
    return RAD_OK;
}

/* Issue the copy of the sampled ids into pinned memory behind the sampler, without waiting.
 * Issued by a drafter before its passes (a head sampling through the sampler's buffer overwrites
 * the ids), otherwise just before the wait.
 *
 * With `fence` (a drafter queueing its passes behind the step) the copy records tok_end_, so
 * collect_step waits for the ids alone (StepOut::early). Not at -vv, whose logits row is a
 * pageable copy read in collect_step. */
int DeviceRuntime::collect_issue(StepOut& so, bool fence) {
    so.rb = nullptr;
    so.lg.clear();
    so.early = false;
    if (so.n_rows <= 0) return RAD_OK;
    Rank& r0 = *ranks_[0];
    void* src = r0.ctx.buf_ptr(r0.tok_buf);
    if (!src) {
        RAD_ERR("the sampler's token buffer has no address -- it was declared but never planned");
        return RAD_E_STATE;
    }
    so.rb = readback_.reserve(so.n_rows);
    if (!so.rb) return RAD_E_NOMEM;
    RAD_TRY(rad_memcpy_async(so.rb, src, (int64_t)so.n_rows * 4, r0.ctx.stream()));

    /* The first logits row at -vv, the only direct view of the forward pass's output (rad-kbench
     * checks single ops, not whole layers). */
    const int64_t nv = meta_.n_vocab;
    if ((int)log_level() >= (int)Log::Trace && nv > 0) {
        void* lp = r0.ctx.buf_ptr(r0.program.logits_buf);
        if (lp) {
            so.lg.resize((size_t)nv);
            rad_memcpy_async(so.lg.data(), lp, nv * 4, r0.ctx.stream());
        }
        return RAD_OK;
    }
    if (fence && early_ids_) {
        RAD_TRY(rad_event_record(tok_end_, r0.ctx.stream()));
        so.early = true;
    }
    return RAD_OK;
}

/* ================================================================== KV streaming's counters
 *
 * Each streamed group's device control block holds the resolve's cumulative counters, the largest
 * union, misses and claims of one launch, and the overflow latch (libref/ref_kvstream.cpp, "The
 * control block"). They show at run time whether the ring's depth holds; a launch that overflowed
 * is failed, not served (docs/KV-STREAMING.md §3.4).
 *
 * The step path never waits on the card: the 64-byte copy per group is enqueued on rank 0's compute
 * stream behind the step's last resolve and before its end event, and the commit that already
 * waits for that event reads it. One rank's counters stand for all, since every rank resolves the
 * same block ids through identically built maps. The counters are cumulative, so the metrics are
 * charged the deltas. */
int DeviceRuntime::stream_probe_issue(const RadBatch* b) {
    if (!b || b->n_kv_groups <= 0 || ranks_.empty()) return RAD_OK;
    const size_t n = (size_t)b->n_kv_groups;
    if (stream_ctl_n_ < n) {
        if (stream_ctl_p_) rad_dev_free(stream_ctl_p_, RAD_MEM_HOST_PINNED);
        stream_ctl_p_ = static_cast<int32_t*>(
            rad_dev_alloc((int64_t)n * 16 * (int64_t)sizeof(int32_t), RAD_MEM_HOST_PINNED));
        stream_ctl_n_ = stream_ctl_p_ ? n : 0;
        if (!stream_ctl_p_) {
            RAD_ERR("kv streaming: no pinned host memory for the resolve's counters");
            return RAD_E_NOMEM;
        }
        /* Zeroed, so a group whose first copy has not landed reads as no calls yet. */
        std::memset(stream_ctl_p_, 0, n * 16 * sizeof(int32_t));
    }
    RadStream st = ranks_[0]->ctx.stream();
    for (size_t i = 0; i < n; ++i) {
        const RadKVStream* sv = b->kv[i].stream;
        if (!sv || !sv->ctl) continue;
        if (rad_memcpy_async(stream_ctl_p_ + i * 16, sv->ctl, 16 * 4, st) < 0) {
            RAD_ERR("kv streaming: the resolve's counters could not be copied to the host: %s",
                    rad_dev_last_error());
            return RAD_E_STATE;
        }
    }
    return RAD_OK;
}

/* Read what stream_probe_issue() enqueued, charge the metrics and, at a prompt's or request's
 * end, log the group's line. A newly latched overflow fails every request prefilling in the step
 * just committed and clears the latch on the card, so the next request is measured on its own;
 * the map stays consistent, because the resolve reverts a claim it could not place. */
int DeviceRuntime::stream_probe_take(const RadBatch* b, bool prefill_done, bool request_end) {
    if (!b || !stream_ctl_p_ || b->n_kv_groups <= 0) return RAD_OK;
    const size_t n = (size_t)b->n_kv_groups;
    if (stream_prev_.size() < n) stream_prev_.assign(n, StreamRead{});
    RadStream st = ranks_[0]->ctx.stream();
    bool       overflowed = false;
    std::string overflow_why;
    for (size_t i = 0; i < n; ++i) {
        const RadKVStream* sv = b->kv[i].stream;
        if (!sv) continue;
        const int32_t* c = stream_ctl_p_ + i * 16;
        /* The counters are three u64s from ctl[4], hence the 8-byte alignment. */
        const uint64_t* ctr = (const uint64_t*)(c + 4);
        StreamRead r;
        r.group       = (int32_t)b->kv[i].group;
        r.misses      = (int64_t)ctr[0];
        r.lookups     = (int64_t)ctr[1];
        r.resolves    = (int64_t)ctr[2];
        r.overflow    = c[3] != 0;
        r.max_union   = c[10];
        r.max_misses  = c[11];
        r.max_claims  = c[12];
        r.slots       = sv->slots;
        r.block_bytes = sv->block_bytes;
        const StreamRead& p = stream_prev_[i];
        const int64_t dm = r.misses - p.misses;
        ranks_[0]->kv.note_stream_traffic(dm, dm * r.block_bytes);
        if (request_end || prefill_done) stream_report(request_end ? "request end" : "prefill complete", r, p);
        if (r.overflow && !p.overflow && !overflowed) {
            overflowed = true;
            const KVGroupPlan* gp = ranks_[0]->kv.plan(r.group);
            const int64_t bs = gp && gp->block_size > 0 ? gp->block_size : 1;
            /* The largest union since bring-up, which for the failing request is the launch that
             * overflowed. */
            overflow_why = fmt(
                "kv streaming overflow: group '%s' ring holds %lld blocks (%lld tokens) and one "
                "resolve launch selected %lld distinct blocks, %lld of them not resident, so the "
                "request is failed; raise --kv-resident above %lld tokens",
                gp ? gp->name.c_str() : "?", (long long)r.slots, (long long)(r.slots * bs),
                (long long)r.max_union, (long long)r.max_misses,
                (long long)(r.max_union * bs));
            RAD_ERR("%s", overflow_why.c_str());
        }
        stream_prev_[i] = r;
    }
    if (overflowed) {
        (void)sched_.fail_prefilling(overflow_why.c_str());
        for (size_t i = 0; i < n; ++i) {
            const RadKVStream* sv = b->kv[i].stream;
            if (sv && sv->ctl) (void)rad_memset_async(sv->ctl + 3, 0, 4, st);
        }
    }
    return RAD_OK;
}

/* One line per group: the cumulative counters, the misses since the previous report and the three
 * largest-launch figures a ring and a claim list are sized from. A launch claims at most
 * `max_seqs * sel_pitch` blocks; a larger `largest launch claims` means that derivation is wrong. */
void DeviceRuntime::stream_report(const char* when, const StreamRead& r, const StreamRead& prev) {
    const KVGroupPlan* gp = ranks_.empty() ? nullptr : ranks_[0]->kv.plan(r.group);
    RAD_INFO("kv streaming: group '%s' %s: %lld resolves, %lld lookups, %lld misses (+%lld this "
             "interval); ring %lld blocks, largest launch union %lld distinct blocks, largest "
             "launch misses %lld, largest launch claims %lld, overflow %s",
             gp ? gp->name.c_str() : "?", when, (long long)r.resolves, (long long)r.lookups,
             (long long)r.misses, (long long)(r.misses - prev.misses), (long long)r.slots,
             (long long)r.max_union, (long long)r.max_misses, (long long)r.max_claims,
             r.overflow ? "LATCHED" : "no");
}

int DeviceRuntime::mark_step_end(const RadBatch* b, StepOut& so) {
    if (embedding_) RAD_TRY(collect_pooled_issue(b, so));
    else if (so.n_rows > 0 && !so.rb) RAD_TRY(collect_issue(so));
    return rad_event_record(step_end_, ranks_[0]->ctx.stream());
}

/* Issue the copy of an embedding step's pooled rows (the first n_seq) into pinned memory on rank
 * 0's stream; every rank holds the same vectors. collect_pooled waits. */
int DeviceRuntime::collect_pooled_issue(const RadBatch* b, StepOut& /*so*/) {
    if (b->n_seq <= 0) return RAD_OK;
    Rank& r0 = *ranks_[0];
    const void* src = r0.ctx.buf_ptr(r0.program.pooled_buf);
    if (!src) {
        RAD_ERR("the pooled output was declared but never planned");
        return RAD_E_STATE;
    }
    return rad_memcpy_async(pooled_host_, src,
                            (int64_t)b->n_seq * pooled_dim_ * (int64_t)sizeof(float),
                            r0.ctx.stream());
}

/* Hand each sequence's vector to its sink, and set eos for every sequence so the commit finishes
 * it: an embedding request is one step long. The token is never emitted (Request::embed). */
int DeviceRuntime::collect_pooled(const RadBatch* b, StepOut& so) {
    const StepPlan& plan = sched_.plan();
    so.token.assign(plan.e.size(), 0);
    so.accepted.assign(plan.e.size(), 0);
    so.eos.assign(plan.e.size(), 1);
    if (plan.e.empty()) return RAD_OK;
    RAD_TRY(rad_event_sync(step_end_));
    if ((int64_t)plan.e.size() != (int64_t)b->n_seq) {
        RAD_ERR("an embedding step planned %zu sequences and its batch carries %lld",
                plan.e.size(), (long long)b->n_seq);
        return RAD_E_STATE;
    }
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& e = plan.e[i];
        if (!e.req) continue;
        if (!e.produces_token) {
            RAD_ERR("an embedding step carried part of request %llu's input; an encoder must see "
                    "the whole of it in one step", (unsigned long long)e.req->id);
            return RAD_E_STATE;
        }
        const float* row = pooled_host_ + (int64_t)i * pooled_dim_;
        server::sink_embedding(e.req->sink, row, pooled_dim_);
    }
    return RAD_OK;
}

/* Snapshot after the step that produced the state. Linear state is what the chunk's last token
 * leaves, so the chunker ended the chunk at the checkpoint and copy 0 holds exactly that state; a
 * window snapshot is the window's rows behind the checkpoint, which may lie inside the chunk
 * (plan_chunk) and are still held when it ends. The copy to the host slot is issued before
 * collect_step's readback and before any step issued ahead, which orders the next write of the
 * state after it. Under the scheduler's lock, as the restore. A block drafter's window in the
 * snapshot (drafter_snap_) is saved behind the drafter's context pass (issue_block_drafter),
 * which writes only the drafter's groups. */
int DeviceRuntime::audit_prefix_pages(uint64_t seq, int64_t tokens, int32_t checkpoint, bool save) {
    static const bool enabled = std::getenv("FREETOKEN_NATIVE_DEBUG_PREFIX_PAGES") != nullptr;
    if (!enabled) return RAD_OK;
    for (auto& rk : ranks_) {
        if (rad_dev_set(rk->device) < 0 || rad_stream_sync(rk->ctx.stream()) < 0)
            return RAD_E_DEVICE;
        for (int32_t g : ranks_[0]->prefix.cached_groups()) {
            const auto* p = rk->kv.plan(g);
            if (!p || p->zero) continue;
            if (!p->base || p->streamed() || p->kind != RAD_KV_FULL || p->block_size <= 0)
                return RAD_E_UNSUPPORTED;
            // Followers share rank 0's logical block ids; their managers hold no sequences.
            const auto& table = ranks_[0]->kv.block_table(seq, g);
            const int64_t nb = tokens / p->block_size;
            if (nb <= 0 || nb > (int64_t)table.size()) return RAD_E_STATE;
            const int64_t lb = p->layer_bytes_per_block();
            const auto& layers = rk->ctx.program()->kv_groups[(size_t)g].layers;
            std::vector<uint64_t> hashes;
            std::vector<unsigned char> bytes((size_t)(nb * lb));
            for (size_t l = 0; l < layers.size(); ++l) {
                // The serial drafter fills its own layer after the trunk snapshot. Only trunk
                // layers belong to this invariant; hashing an unfilled draft layer is invalid.
                if (layers[l] >= meta_.n_layers) continue;
                for (int64_t b = 0; b < nb;) {
                    const int32_t id = table[(size_t)b];
                    int64_t run = 1;
                    while (b + run < nb && table[(size_t)(b + run)] == id + run) ++run;
                    if (id < 0 || id + run > rk->kv.live_blocks(g)) return RAD_E_STATE;
                    const char* src = (const char*)p->base + (int64_t)l * p->layer_stride() + id * lb;
                    const int rc = rad_memcpy_async(bytes.data() + b * lb, src, run * lb,
                                                    rk->ctx.stream());
                    if (rc < 0) return rc;
                    b += run;
                }
                if (rad_stream_sync(rk->ctx.stream()) < 0) return RAD_E_DEVICE;
                for (int64_t b = 0; b < nb; ++b) {
                    uint64_t h = 1469598103934665603ull;
                    for (int64_t k = 0; k < lb; ++k) { h ^= bytes[(size_t)(b * lb + k)]; h *= 1099511628211ull; }
                    hashes.push_back(h);
                }
            }
            const std::string key = std::to_string(rk->index) + ":" +
                                    std::to_string(checkpoint) + ":" + std::to_string(g);
            const auto it = prefix_page_hashes_.find(key);
            const bool equal = save || (it != prefix_page_hashes_.end() && it->second == hashes);
            size_t first = 0;
            if (!save && it != prefix_page_hashes_.end())
                while (first < hashes.size() && first < it->second.size() && hashes[first] == it->second[first]) ++first;
            RAD_INFO("prefix page audit: %s rank=%d ck=%d group=%s tokens=%lld blocks=%lld "
                     "checks=%zu exact=%d first_difference=%zu", save ? "save" : "restore",
                     rk->index, (int)checkpoint, p->name.c_str(), (long long)tokens,
                     (long long)nb, hashes.size(), equal ? 1 : 0, first);
            if (save) prefix_page_hashes_[key] = std::move(hashes);
            else if (!equal) return RAD_E_STATE;
        }
    }
    return rad_dev_set(ranks_[0]->device);
}

int DeviceRuntime::save_checkpoints() {
    const StepPlan& plan = sched_.plan();
    bool any_save = false;
    for (const StepEntry& e : plan.e)
        if (e.ckpt_slot >= 0 && e.ckpt_tok >= 0) { any_save = true; break; }
    if (any_save) {
        const int ss = sched_.locked([&]() -> int {
            for (const StepEntry& e : plan.e) {
                if (e.ckpt_slot < 0 || e.ckpt_tok < 0) continue;
                if (!state_slots_of(e.seq, &ckpt_slots_)) continue;
                /* The window behind the checkpoint position. This step's chunk reaches it, and
                 * a window slides only from a step's first query, so a miss is a broken
                 * invariant; skipping would commit a slot holding another sequence's pages. */
                if (ranks_[0]->kv.has_window_snapshot() &&
                    !ranks_[0]->kv.window_rows(e.seq, e.ctx_len + e.ckpt_tok + 1,
                                               &ckpt_win_)) {
                    RAD_ERR("sequence %llu: the window behind checkpoint %d is not held",
                            (unsigned long long)e.seq, (int)(e.ctx_len + e.ckpt_tok + 1));
                    return RAD_E_STATE;
                }
                for (auto& rk : ranks_) {
                    if (ranks_.size() > 1) (void)rad_dev_set(rk->device);
                    const int s = rk->kv.checkpoint_save(ckpt_slots_.data(),
                                                         (int)ckpt_slots_.size(),
                                                         e.ckpt_slot, rk->ctx.stream(),
                                                         &ckpt_win_);
                    if (s < 0) {
                        RAD_ERR("rank %d could not snapshot sequence %llu into checkpoint slot "
                                "%d: %s", rk->index, (unsigned long long)e.seq,
                                (int)e.ckpt_slot, rad_strerror(s));
                        if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
                        return s;
                    }
                }
                const int pa = audit_prefix_pages(e.seq, e.ctx_len + e.ckpt_tok + 1,
                                                   e.ckpt_slot, true);
                if (pa < 0) return pa;
                if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
            }
            return RAD_OK;
        });
        if (ss < 0) return ss;
    }
    return RAD_OK;
}

/* Scored rows (Request::token_scores): the logits row at a scored prompt's last position, copied
 * to the host before the sampler can rewrite it in place. stage_scored_rows picks the rows while
 * every rank is parked; issue_scored_rows copies one row per rank stream ahead of its sampler;
 * collect_scored_rows reads each request's ids after the step's wait, before the commit. A whole
 * row is copied (512 KB at a 128K vocabulary), which is cheaper than one copy per id.
 *
 * The pinned host buffer is grow-only and its growth is the one part of staging that can fail
 * on resources rather than on a broken program: then the scored requests are the ones cancelled
 * (into `cancel`, applied the way the sampler's cancellations are), not the engine — a request
 * whose scores the box cannot hold is that request's failure, and every other row on this step
 * and every other connection on the server keeps running. */
int DeviceRuntime::stage_scored_rows(const StepPlan& plan, const StepOut& so,
                              std::vector<uint64_t>& cancel) {
    score_rows_.clear();
    score_only_ = false;
    score_ids_.clear();
    if (embedding_) return RAD_OK;
    std::vector<uint64_t> scored;
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const Request* r = plan.e[i].req;
        if (so.first_row[i] < 0 || !r || !r->token_scores || !r->output.empty()) continue;
        score_rows_.push_back({ so.first_row[i], r->token_scores.get() });
        scored.push_back(r->id);
    }
    if (score_rows_.empty()) return RAD_OK;
    const int64_t n = (int64_t)score_rows_.size();

    /* A scored step reads back the hidden rows and the candidates' table rows instead of whole
     * logits rows (BatchBuilder set the flag on the same facts). */
    const RadBatch* b0 = sched_.batch_for(0);
    score_only_ = b0 && b0->score_only;
    if (score_only_) {
        const Program& p = ranks_[0]->program;
        score_embd_ = p.buffers[p.score_hidden].decl.shape[1];
        for (const auto& sr : score_rows_)
            for (int32_t t : sr.second->tokens) score_ids_.push_back(t);
        std::sort(score_ids_.begin(), score_ids_.end());
        score_ids_.erase(std::unique(score_ids_.begin(), score_ids_.end()), score_ids_.end());
    }
    bool host_failed = false;
    int64_t failed_rows = 0;
    for (auto& rk : ranks_) {
        const int64_t width = rk->sampler.config().n_vocab_local;
        /* In floats: n logits rows, or the bf16 hidden and table rows of a scored step. */
        int64_t rows = n;
        if (score_only_) {
            const int64_t need = ((n + (int64_t)score_ids_.size()) * score_embd_ + 1) / 2;
            rows = std::max(rows, (need + width - 1) / width);
        }
        if (rows <= rk->score_rows_cap) continue;
        if (ranks_.size() > 1) RAD_TRY(rad_dev_set(rk->device));
        if (rk->score_host) rad_dev_free(rk->score_host, RAD_MEM_HOST_PINNED);
        rk->score_host = static_cast<float*>(
            rad_dev_alloc(rows * width * (int64_t)sizeof(float), RAD_MEM_HOST_PINNED));
        rk->score_rows_cap = rk->score_host ? rows : 0;
        if (!rk->score_host) { host_failed = true; failed_rows = rows; break; }
    }
    if (ranks_.size() > 1) RAD_TRY(rad_dev_set(ranks_[0]->device));
    if (host_failed) {
        RAD_ERR("scored rows: no pinned memory for %lld logits rows; cancelling the scored "
                "requests, not the engine", (long long)failed_rows);
        cancel.insert(cancel.end(), scored.begin(), scored.end());
        score_rows_.clear();
        score_only_ = false;
        score_ids_.clear();
    }
    return RAD_OK;
}

int DeviceRuntime::issue_scored_rows(Rank& r) {
    const int64_t width = r.sampler.config().n_vocab_local;
    char* const logits = static_cast<char*>(r.ctx.buf_ptr(r.program.logits_buf));
    if (!logits || !r.score_host) return RAD_E_STATE;
    if (score_only_) {
        /* The plugin left the hidden rows and ran no lm_head: read back each scored row's hidden
         * state and the table rows its candidates name, then zero the logits rows the sampler is
         * about to read, so the token it draws (and the caller discards) is drawn from defined
         * values. One rank (BatchBuilder). */
        const int64_t rb = score_embd_ * (int64_t)sizeof(uint16_t);
        const char* hid = static_cast<const char*>(r.ctx.buf_ptr(r.program.score_hidden));
        const char* tab = static_cast<const char*>(r.ctx.weight_ptr(r.program.score_table));
        if (!hid || !tab) return RAD_E_STATE;
        char* host = reinterpret_cast<char*>(r.score_host);
        const int64_t n = (int64_t)score_rows_.size();
        for (int64_t k = 0; k < n; ++k)
            RAD_TRY(rad_memcpy_async(host + k * rb, hid + (int64_t)score_rows_[(size_t)k].first * rb,
                                     rb, r.ctx.stream()));
        for (size_t j = 0; j < score_ids_.size(); ++j)
            RAD_TRY(rad_memcpy_async(host + (n + (int64_t)j) * rb, tab + (int64_t)score_ids_[j] * rb,
                                     rb, r.ctx.stream()));
        return rad_memset_async(logits, 0, n * width * (int64_t)sizeof(float), r.ctx.stream());
    }
    for (size_t k = 0; k < score_rows_.size(); ++k)
        RAD_TRY(rad_memcpy_async(r.score_host + (int64_t)k * width,
                                 logits + (int64_t)score_rows_[k].first * width * 4,
                                 width * 4, r.ctx.stream()));
    return RAD_OK;
}

/* A scored step's logits: the dot product of a row's hidden state with each candidate's table row,
 * bf16 operands widened exactly and summed in f64, then rounded once to f32. */
static float score_dot(const uint16_t* h, const uint16_t* w, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float a = __builtin_bit_cast(float, (uint32_t)h[i] << 16);
        const float b = __builtin_bit_cast(float, (uint32_t)w[i] << 16);
        acc += (double)a * (double)b;
    }
    return (float)acc;
}

int DeviceRuntime::collect_scored_rows() {
    if (score_rows_.empty()) return RAD_OK;
    /* collect_step's wait fences rank 0's stream; every other rank's copies are fenced here. */
    for (size_t i = 1; i < ranks_.size(); ++i) {
        RAD_TRY(rad_dev_set(ranks_[i]->device));
        RAD_TRY(rad_stream_sync(ranks_[i]->ctx.stream()));
    }
    if (ranks_.size() > 1) RAD_TRY(rad_dev_set(ranks_[0]->device));
    if (score_only_) {
        const uint16_t* host = reinterpret_cast<const uint16_t*>(ranks_[0]->score_host);
        const int64_t n = (int64_t)score_rows_.size();
        for (int64_t k = 0; k < n; ++k) {
            TokenScores& ts = *score_rows_[(size_t)k].second;
            ts.logits.assign(ts.tokens.size(), 0.0f);
            for (size_t j = 0; j < ts.tokens.size(); ++j) {
                const auto it = std::lower_bound(score_ids_.begin(), score_ids_.end(), ts.tokens[j]);
                const int64_t c = it - score_ids_.begin();
                ts.logits[j] = score_dot(host + k * score_embd_, host + (n + c) * score_embd_,
                                         score_embd_);
            }
        }
        score_rows_.clear();
        score_only_ = false;
        return RAD_OK;
    }
    for (size_t k = 0; k < score_rows_.size(); ++k) {
        TokenScores& ts = *score_rows_[k].second;
        ts.logits.assign(ts.tokens.size(), 0.0f);
        for (size_t j = 0; j < ts.tokens.size(); ++j) {
            const int64_t t = ts.tokens[j];
            for (auto& rk : ranks_) {
                const SamplerConfig& sc = rk->sampler.config();
                if (t < sc.vocab_off || t >= sc.vocab_off + sc.n_vocab_local) continue;
                ts.logits[j] = rk->score_host[(int64_t)k * sc.n_vocab_local + (t - sc.vocab_off)];
                break;
            }
        }
    }
    score_rows_.clear();
    return RAD_OK;
}

/* One sampled row's logprobs into the step's table, sized on first use. A non-finite row
 * (RAD_E_INVAL) is left at token -1, and the scheduler fails only that request. A row outside the
 * step, or never staged for logprobs, is an engine error. */
int DeviceRuntime::row_logprobs(StepOut& so, int32_t row, int32_t token, int32_t at) {
    if (so.lp.empty()) so.lp.resize((size_t)so.n_rows);
    if (at < 0) at = row;
    if (row < 0 || row >= so.n_rows || at >= so.n_rows) return RAD_E_STATE;
    TokenLogprobs& lp = so.lp[(size_t)at];
    const int s = ranks_[0]->sampler.logprobs_for(row, token, &lp);
    if (s == RAD_E_INVAL) { lp = TokenLogprobs{}; return RAD_OK; }
    return s;
}

/* Wait for the sampled ids and build the scheduler's per-sequence arrays. The step path's only
 * host synchronisation: the host needs the tokens to commit and stream them (spec §5.4). */
int DeviceRuntime::collect_step(const RadBatch* b, StepOut& so) {
    const StepPlan& plan = sched_.plan();
    so.token.assign(plan.e.size(), 0);
    so.accepted.assign(plan.e.size(), 0);
    so.eos.assign(plan.e.size(), 0);
    so.lp.clear();
    so.path.clear();
    if (so.n_rows <= 0) return RAD_OK;

    so.rows.resize((size_t)so.n_rows);
    /* Wait on an event, not the stream: a step issued ahead is queued behind it. With early ids
     * that is the ids' own copy; the step loop waits for the rest after the commit. */
    RAD_TRY(rad_event_sync(so.early ? tok_end_ : step_end_));
    std::memcpy(so.rows.data(), so.rb, (size_t)so.n_rows * sizeof(int32_t));
    so.rb = nullptr;
    RAD_TRY(collect_scored_rows());
    const std::vector<float>& lg = so.lg;
    const int64_t nv = meta_.n_vocab;
    if (!lg.empty()) {
        /* Top three of the first sampled row, to tell a fixed spike in the distribution from a
         * sampler that ignores it. */
        int64_t i0 = 0, i1 = 0, i2 = 0;
        for (int64_t i = 1; i < nv; ++i) {
            if (lg[(size_t)i] > lg[(size_t)i0])      { i2 = i1; i1 = i0; i0 = i; }
            else if (lg[(size_t)i] > lg[(size_t)i1]) { i2 = i1; i1 = i; }
            else if (lg[(size_t)i] > lg[(size_t)i2]) { i2 = i; }
        }
        double sum = 0, amax = 0;
        for (int64_t i = 0; i < nv; ++i) { sum += lg[(size_t)i]; amax = std::max(amax, (double)std::fabs(lg[(size_t)i])); }
        RAD_TRACE("step %d logits: top %lld=%.4f  %lld=%.4f  %lld=%.4f   mean %.4f  amax %.4f",
                  b->step, (long long)i0, (double)lg[(size_t)i0], (long long)i1,
                  (double)lg[(size_t)i1], (long long)i2, (double)lg[(size_t)i2],
                  sum / (double)nv, amax);
    }

    for (size_t i = 0; i < plan.e.size(); ++i) {
        if (so.first_row[i] < 0) continue;
        const StepEntry& e = plan.e[i];

        if (e.n_spec > 0) {
            /* Accept draft[k] when it equals the token the target itself drew at position k. This
             * is exact at any temperature without rejection sampling or probability planes,
             * because the sampler gives every verified position its own RNG coordinate
             * (`p.pos = n_out + k`, sampler.cpp): y_k is the target's own draw at that output
             * position, and each later token is conditioned on a prefix accepted only because it
             * matched. The output is a draw from the target.
             *
             * It is not byte-identical to non-speculative decoding above temperature 0: a verify
             * step computes 1 + n_spec rows and split-K depends on M, so logits differ slightly
             * and a pick near a CDF boundary can flip. Greedy output matches exactly; both are
             * deterministic.
             *
             * The cost is a lower acceptance rate than rejection sampling (P(y == x) instead of
             * min(1, p/q)). `so.rows` holds the sampled tokens (the argmax at temperature 0). */
            if (!e.req) continue;
            int32_t out[RAD_SCHED_MAX_SPEC + 1] = { 0 };
            int32_t acc = 0;
            /* A tree step verifies every proposal as a tree (a chain one as the chain-shaped
             * tree), as the card's acceptance does; `path` names the accepted nodes. */
            int32_t path[RAD_SCHED_MAX_SPEC] = { 0 };
            const bool tree = plan.spec_is_tree;
            if (tree)
                RAD_TRY(accept_tree(e.draft, e.draft_parent, e.n_spec,
                                    &so.rows[(size_t)so.first_row[i]], out, path, &acc));
            else
                RAD_TRY(accept_greedy(e.draft, e.n_spec, &so.rows[(size_t)so.first_row[i]], out,
                                      &acc));

            /* Truncated at the first end-of-generation token, so `accepted` and `token` (which
             * drive the KV rollback, the conv cursor and the emitted text) say how far the step
             * really got. */
            acc = accept_cut_at_eog(out, acc,
                                    [&](int32_t t) { return vocab_ && vocab_->is_eog(t); });

            so.accepted[i] = acc;
            so.token[i]    = out[acc];
            so.eos[i]      = vocab_ && vocab_->is_eog(so.token[i]) ? 1 : 0;
            if (tree) {
                if (so.path.empty()) so.path.assign(plan.e.size() * RAD_SCHED_MAX_SPEC, 0);
                for (int32_t t = 0; t < acc; ++t) so.path[i * RAD_SCHED_MAX_SPEC + t] = path[t];
            }
            /* Each emitted token's logprobs come from its own row, the distribution it was drawn
             * from: on a tree, the row of the node it follows. They are filed as the entry's j-th
             * emitted token (StepResult::lp), whichever row they came from. */
            if (e.req && e.req->sp.n_probs >= 0)
                for (int32_t j = 0; j <= acc; ++j) {
                    const int32_t from = tree && j > 0 ? path[j - 1] : j;
                    RAD_TRY(row_logprobs(so, so.first_row[i] + from, out[j],
                                         so.first_row[i] + j));
                }
            /* Advance the grammar over every emitted token, accepted drafts included, up to the
             * truncated `acc`, so it sees exactly what the client sees. */
            if (e.req)
                for (auto& rk : ranks_)
                    for (int32_t j = 0; j <= acc; ++j) {
                        const int as = rk->sampler.accept_token(e.req->id, out[j]);
                        /* A parse that outgrew the grammar's bounds fails the request at its next
                         * mask; not an engine defect. */
                        if (as == RAD_E_FULL || as == RAD_E_NOMEM)
                            RAD_WARN("engine: req %llu: its grammar's parse outgrew the bounds a "
                                     "grammar may use at token %d; the request fails at its next "
                                     "step", (unsigned long long)e.req->id, out[j]);
                        else if (as < 0)
                            RAD_ERR("engine: req %llu emitted token %d (position %d of %d "
                                    "accepted, n_spec %d) that its grammar forbids -- the mask and "
                                    "the pick disagree. The same condition on the speculative WALK "
                                    "is expected and is a debug line in gbnf.cpp; only this one, "
                                    "on an ACCEPTED token, is a defect.",
                                    (unsigned long long)e.req->id, out[j], j, acc, e.n_spec);
                    }
            continue;
        }
        so.token[i] = so.rows[(size_t)so.first_row[i]];
        so.eos[i]   = vocab_ && vocab_->is_eog(so.token[i]) ? 1 : 0;
        if (e.req && e.req->sp.n_probs >= 0)
            RAD_TRY(row_logprobs(so, so.first_row[i], so.token[i]));
        /* Advance the grammar on every rank: each builds the mask for its own vocabulary shard from
         * its own automaton. */
        if (e.req)
            for (auto& rk : ranks_) rk->sampler.accept_token(e.req->id, so.token[i]);
    }

    /* Update may_speculate here, because a lazy grammar becomes constraining inside accept_token
     * above; a stale value would give a constrained request a draft the sampler refuses. */
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& e = plan.e[i];
        if (!e.req) continue;
        Request* r = const_cast<Request*>(e.req);
        r->may_speculate = ranks_[0]->sampler.allows_speculation(r->id);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the KV loan */
/* The only place the boundary between the KV cache and the expert slab moves.
 *
 * The budget divides the card once at startup, so an unused KV block is VRAM no expert can use.
 * The cache therefore lends what it does not hold at addresses it already owns: no allocation,
 * mapping or driver call. There is exactly one lender, so the same memory is never in two pools.
 *
 * The order is the safety argument:
 *
 *   1. the cache reports what it can spare (above its highest held block, less its buffer);
 *   2. the cache's mark is set to the larger of that and what any rank's slab actually holds, so
 *      a shrink completes only as units leave the slots;
 *   3. only then are the slabs told the offer, which they fill into or drain out of.
 *
 * A request that cannot wait for a shrink takes the blocks at once through a callback into every
 * slab (KVManager::set_reclaim, installed in start()). */
void DeviceRuntime::loan_tick(bool idle) {
    if (!lending_) return;
    const uint64_t now = rad_mono_ns();
    if (now - loan_last_ns_ < 100000000ull) return;    /* 100 ms */
    loan_last_ns_ = now;

    /* Under the scheduler's lock: the loan moves rank 0's live mark and free count, which an HTTP
     * thread's cancel changes and /metrics reads under that lock. */
    const auto lk = sched_.hold();
    Rank& lead = *ranks_[0];

    /* tier_tick has already released every entry whose host copy landed, so `spare` counts
     * finished conversations. The loan grows only in steps of at least a prefill chunk, so small
     * fluctuations do not move the boundary; it shrinks at once, the cache regaining blocks as the
     * slabs vacate them. */
    const int64_t cur   = lead.kv.loan_tokens();
    const int64_t spare = lead.kv.spare_tokens();
    const int64_t chunk = std::max<int64_t>(cfg_.max_tok, 1);
    int64_t offer = cur;
    if (spare < cur || spare >= cur + chunk) offer = spare;
    int64_t held = 0;
    for (auto& r : ranks_) held = std::max(held, r->mover.loan_held(Mover::kLendKV));

    /* The cache moves first when the loan grows, the slabs first when it shrinks, so no block is
     * ever both a cache block and a slab slot. */
    if (lead.kv.set_loan(std::max(offer, held)) < 0) {
        RAD_ERR("kv loan: the cache refused a loan of %lld tokens it said it could spare; lending "
                "stops", (long long)std::max(offer, held));
        lending_ = false;
        return;
    }
    for (size_t i = 1; i < ranks_.size(); ++i) (void)ranks_[i]->kv.follow(lead.kv);
    for (auto& r : ranks_) {
        if (rad_dev_set(r->device) < 0) continue;
        (void)r->mover.set_loan(Mover::kLendKV, offer);
        if (idle) {
            const int st = r->mover.idle_tick(&r->heat);
            if (st < 0) RAD_WARN("rank %d: filling the lent slots while idle: %s", r->index,
                                 rad_strerror(st));
        }
    }
    (void)rad_dev_set(ranks_[0]->device);
}

/* ------------------------------------------------------------------ the arena's level */
/* Pick the step's arena level and how much of the arena the slab may keep. A step's rows are its
 * tokens or, for the drafter's passes, one per drafted position per sequence; the smallest level
 * serving that many is used, else the full plan, which lends nothing.
 *
 * If the level needs memory the slab holds, it is reclaimed synchronously before the step: units
 * in those slots revert to their host copies (an address change, not a copy) and both streams
 * that can read a slot are drained. A prefill pays this on its first chunk only, since the loan
 * does not regrow until steps have been small for a while (arena_tick). */
int DeviceRuntime::arena_step(const RadBatch* b) {
    if (arena_rows_.size() <= 1 || !b) return RAD_OK;
    /* An encoder pass needs the full plan: its activations are sized by the pass, and a plugin
     * declares them small under shape_probe. */
    const int64_t need = b->enc ? INT64_MAX : std::max<int64_t>(
        b->n_tok, (int64_t)b->n_seq * (1 + std::max(cfg_.n_spec, 0)));
    const int level = arena_level(need);
    const uint64_t now = rad_mono_ns();
    arena_last_ = arena_rows_[(size_t)level];
    if (arena_last_ >= arena_floor_) { arena_floor_ = arena_last_; arena_floor_ns_ = now; }
    /* Whether the next step continues a prompt, so the stager stages its first layers at this
     * step's end. Never for a step issued ahead (a decode step whose plan the scheduler does not
     * hold). */
    const bool more = b->phase != RAD_PHASE_DECODE && sched_.prefill_continues();
    for (auto& r : ranks_) {
        const int64_t allow = r->arena_lent[(size_t)level];
        if (r->mover.loan(Mover::kLendArena) > allow ||
            r->mover.loan_held(Mover::kLendArena) > allow) {
            RAD_TRY(rad_dev_set(r->device));
            const int st = r->mover.reclaim(Mover::kLendArena, allow, &r->heat);
            if (st < 0) {
                RAD_ERR("rank %d: taking the arena back from the expert slab for a %lld-token step: "
                        "%s", r->index, (long long)b->n_tok, rad_strerror(st));
                return st;
            }
        }
        RAD_TRY(r->ctx.set_level(level));
        /* The full plan lends nothing, so its top (the staging region) is the step's ring; the
         * span's shared level keeps a place of its own for it (stager.h); every other level
         * lends both. */
        char* const ring = level == 0 ? r->stager.ring_full()
                         : arena_last_ == arena_span_rows_ && r->arena_low_off >= 0
                               ? r->stager.ring_low() : nullptr;
        r->stager.set_ring(ring);
        /* The next step's first layers go ahead only from the full plan, into the region at the
         * top that no level's buffers use. At the shared level the ring is memory the full plan
         * computes in, and a copy still going there when such a step starts would race it; a
         * pass's own copies are behind its own ops, which wait for them. */
        r->stager.set_next_prefill(more && level == 0);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    return RAD_OK;
}

int DeviceRuntime::arena_level(int64_t rows) const { return arena_level_for(arena_rows_, rows); }

/* The arena loan grows only after large steps stop. A prefill is chunks with decode steps between,
 * and lending after every chunk would refill and recall the slab each time. So the floor (the
 * largest step seen) holds for two seconds after it was last reached, then falls to the latest
 * step's size. */
void DeviceRuntime::arena_tick(uint64_t now_ns) {
    if (arena_rows_.size() <= 1) return;
    if (arena_last_ < arena_floor_ && now_ns - arena_floor_ns_ > 2000000000ull) {
        arena_floor_ = arena_last_;
        arena_floor_ns_ = now_ns;
    }
    const int level = arena_level(arena_floor_);
    for (auto& r : ranks_) {
        const int64_t want = r->arena_lent[(size_t)level];
        if (want <= r->mover.loan(Mover::kLendArena)) continue;
        if (rad_dev_set(r->device) < 0) continue;
        (void)r->mover.set_loan(Mover::kLendArena, want);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
}

/* ------------------------------------------------------------------ the idle-tier tick */
/* Runs every iteration, busy or idle. A host copy is reported once its event signals, so the
 * first tick after it lands collects it and hands out the next; a slower clock would only delay
 * releasing a finished conversation's VRAM. When the host tier has refused a copy for lack of
 * arena room, room is made at once.
 *
 * The scheduler's lock covers only the bookkeeping (planning and reporting touch rank 0's
 * manager, the prefix cache and the index an HTTP cancel changes). Issuing and polling touch none
 * of that, and a transfer's records stay pending while it is out. */
void DeviceRuntime::tier_tick() {
    Rank& lead = *ranks_[0];
    if (!pump_ || !lead.tierx.enabled()) return;
    TierPump& P = *pump_;
    const uint64_t now = rad_mono_ns();

    /* ---- a copy that has landed on every rank is reported ---------------------------------- */
    if (!P.copy.empty()) {
        bool all = true, good = true;
        for (size_t i = 0; i < ranks_.size(); ++i) {
            if (P.copy_st[i] == 0) P.copy_st[i] = ranks_[i]->tierx.copy_state();
            if (P.copy_st[i] == 0) all = false;
            else if (P.copy_st[i] < 0) good = false;
        }
        if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
        if (all) {
            /* RADIANCE_DEBUG_TIER_VERIFY: now that every rank's window has landed, each
             * follower's hash of the replicated groups it gathered is held against the window
             * the leader wrote (rad_kv_group_replicated's claim). The sums are taken either way,
             * so a failed batch leaves none behind. */
            if (lead.tierx.verifying_replicated()) {
                for (size_t i = 1; i < ranks_.size(); ++i)
                    for (const TierJob& j : P.copy) {
                        uint64_t sum = 0;
                        if (ranks_[i]->tierx.take_replicated_sum(j.key, &sum) && good)
                            (void)lead.tierx.verify_replicated(j, (int)i, sum);
                    }
            }
            /* A copy is good only if every rank's shard landed; otherwise the host copy is
             * partly stale, so the whole batch fails and every slot goes back. */
            const std::vector<char> ok(P.copy.size(), good ? 1 : 0);
            {
                const auto lk = sched_.hold();
                lead.tierx.report_pass(P.copy, ok, now);
            }
            P.copy.clear();
        }
    }

    /* ---- the next copy ---------------------------------------------------------------------- */
    if (P.copy.empty() && lead.tiers.writeback_waiting()) {
        {
            const auto lk = sched_.hold();
            lead.tiers.plan_writeback(P.max_copy, &P.copy);
        }
        if (!P.copy.empty()) {
            P.copy_st.assign(ranks_.size(), 0);
            for (size_t i = 0; i < ranks_.size(); ++i) {
                const int st = ranks_[i]->tierx.issue_copy(P.copy, ranks_[i]->ctx.stream());
                if (st < 0) P.copy_st[i] = st;
            }
            if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
        }
    }

    /* ---- what is not live leaves VRAM -------------------------------------------------------- */
    /* Release an entry once its host copy has landed, if the expert slab can use the blocks; a
     * later turn restores it with a copy. Otherwise (no lending, or the loan already covers every
     * slab slot) it stays in VRAM, and a request needing room takes it through trim. Snapshots
     * are already in host memory. Stragglers with no host copy coming go too (PrefixCache::
     * drop_clean): a few thousand such tokens near the top of the pool would otherwise hold the
     * whole loan at zero until the server restarts. */
    if (lending_) {
        int64_t ceiling = 0;
        for (auto& r : ranks_) ceiling = std::max(ceiling, r->mover.loan_ceiling(Mover::kLendKV));
        const auto lk = sched_.hold();
        if (lead.kv.loan_tokens() < ceiling) (void)lead.prefix.drop_clean(ceiling);
    }

    /* ---- capacity ---------------------------------------------------------------------------- */
    /* When the host arena refused copies, drop host-only entries worth less than them to make
     * room (IdleTiers::plan). Drops move no bytes, so the plan is reported at once. */
    if (lead.tiers.starved()) {
        std::vector<TierJob> jobs;
        lead.tiers.plan(P.slow_budget, &jobs);
        if (!jobs.empty()) {
            const std::vector<char> ok(jobs.size(), 1);
            const auto lk = sched_.hold();
            lead.tierx.report_pass(jobs, ok, now);
        }
    }
}

void DeviceRuntime::expire_tick(uint64_t now_ns) {
    if (cfg_.prefix_cache_ttl <= 0 || now_ns - expire_last_ns_ < 1000000000ull) return;
    expire_last_ns_ = now_ns;
    PrefixCache& pc = ranks_[0]->prefix;
    if (!pc.enabled()) return;
    const auto lk = sched_.hold();
    (void)pc.expire(now_ns, (uint64_t)cfg_.prefix_cache_ttl * 1000000000ull);
}

bool DeviceRuntime::flush_copies() {
    if (!pump_ || !ranks_[0]->tierx.enabled()) return false;
    TierPump& P = *pump_;
    Rank& lead = *ranks_[0];
    bool landed = false;
    const uint64_t t0 = rad_mono_ns();
    /* Counted only when something was in flight or waiting. */
    struct Note {
        IdleTiers& t; uint64_t t0; bool worked = false;
        ~Note() { if (worked) t.note_flush((int64_t)(rad_mono_ns() - t0)); }
    } note{lead.tiers, t0};
    /* Bounded: each round lands one batch of P.max_copy entries; a larger shortfall cannot be
     * relieved in one step. */
    for (int round = 0; round < 64; ++round) {
        if (!P.copy.empty()) {
            note.worked = true;
            bool good = true;
            for (size_t i = 0; i < ranks_.size(); ++i) {
                if (P.copy_st[i] == 0) P.copy_st[i] = ranks_[i]->tierx.wait_copy();
                if (P.copy_st[i] < 0) good = false;
            }
            if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
            const std::vector<char> ok(P.copy.size(), good ? 1 : 0);
            lead.tierx.report_pass(P.copy, ok, rad_mono_ns());
            P.copy.clear();
            landed = landed || good;
        }
        if (!lead.tiers.writeback_waiting()) break;
        lead.tiers.plan_writeback(P.max_copy, &P.copy);
        if (P.copy.empty()) break;
        P.copy_st.assign(ranks_.size(), 0);
        for (size_t i = 0; i < ranks_.size(); ++i) {
            const int st = ranks_[i]->tierx.issue_copy(P.copy, ranks_[i]->ctx.stream());
            if (st < 0) P.copy_st[i] = st;
        }
        if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
    }
    return landed;
}

/* Drop the sampler's per-request state for requests that reached a terminal state. The server
 * drains their text through the sink and calls reap(), which releases the Request; only the HTTP
 * thread knows when it has stopped writing. */
void DeviceRuntime::drain_completed() {
    /* Taken, not read: HTTP cancels append to the list, and nothing else consumes it, so each id
     * is seen exactly once. */
    sched_.take_completed(&done_);
    for (uint64_t id : done_) for (auto& rk : ranks_) rk->sampler.end_request(id);
}

/* Zero freshly assigned recurrent slots before the step that reads them. The manager records the
 * grab but cannot write (StateClear in core/mem/kv.h); the list is empty except on a sequence's
 * first step. Each rank clears by its own plan: the slot number is the scheduler's, but the state
 * size depends on the heads this rank holds, which can differ per rank (rad_arch.h
 * remainder_place). */
void DeviceRuntime::clear_fresh_states() {
    for (const StateClear& c : ranks_[0]->kv.state_clears()) {
        for (auto& rk : ranks_) {
            const KVGroupPlan* gp = rk->kv.plan(c.group);
            if (!gp || !gp->stateful()) continue;
            const int64_t bytes = gp->layer_bytes_per_state();
            for (int64_t l = 0; l < gp->n_layers; ++l)
                if (void* p = rk->kv.state_ptr(c.group, (int32_t)l, c.slot))
                    rad_memset_async(p, 0, bytes, rk->ctx.stream());
        }
    }
    ranks_[0]->kv.clear_state_clears();
}

int DeviceRuntime::serve(std::unique_ptr<StallWatch> watch) {
    return serve_impl(std::move(watch), {});
}

int DeviceRuntime::serve_headless(HeadlessFront front) {
    return serve_impl({}, front);
}

int DeviceRuntime::serve_impl(std::unique_ptr<StallWatch> watch, const HeadlessFront& front) {
    EngineScope es(owner_, cfg_.served_model_name.c_str());
    startup_watch_ = std::move(watch);
    if (ranks_.empty() || !ranks_[0]->ctx.program()) {
        RAD_ERR("serve: start() has not run, or stopped at --debug-graph");
        return RAD_E_STATE;
    }
    embedding_ = ranks_[0]->program.pooled_buf != RAD_NULL_HANDLE;
    const bool audio_gen = ranks_[0]->program.is_audio_generator();
    const bool video_gen = ranks_[0]->program.is_video_generator();
    const bool shape_gen = ranks_[0]->program.is_shape_generator();
    const bool generator = ranks_[0]->program.is_pass_driven();
    if (ranks_[0]->program.logits_buf == RAD_NULL_HANDLE && !embedding_ && !generator) {
        RAD_ERR("the '%s' plugin declared no logits buffer (rad_declare_logits), no pooled output "
                "(rad_declare_pooled) and no generator (rad_declare_generator, "
                "rad_declare_audio_generator, rad_declare_video_generator, "
                "rad_declare_shape_generator), so a step produces "
                "nothing anyone can read. An encoder-only plugin cannot serve.",
                plugins_.arch_plugin.c_str());
        return RAD_E_UNSUPPORTED;
    }
    if (generator) {
        const char* const what = audio_gen ? "audio" : video_gen ? "video"
                               : shape_gen ? "shapes" : "images";
        if (cfg_.kld() || cfg_.n_spec > 0) {
            RAD_ERR("the '%s' plugin makes %s: --kld and speculative decoding have no tokens to "
                    "work on", plugins_.arch_plugin.c_str(), what);
            return RAD_E_UNSUPPORTED;
        }
        for (size_t i = 1; i < ranks_.size(); ++i)
            if (ranks_[i]->program.is_audio_generator() != audio_gen ||
                ranks_[i]->program.is_video_generator() != video_gen ||
                ranks_[i]->program.is_shape_generator() != shape_gen ||
                !ranks_[i]->program.is_pass_driven()) {
                RAD_ERR("rank %zu declared a different generator from rank 0's", i);
                return RAD_E_STATE;
            }
        cfg_.prefix_cache = false;
        prefix_off_why_ = audio_gen
            ? "an audio generator: a prompt is read by its own request's frames and by nothing "
              "after them"
            : video_gen
            ? "a video generator: a prompt is read by its own request's steps and keeps no KV"
            : shape_gen
            ? "a shape generator: a picture is read by its own request's steps and keeps no KV"
            : "an image generator: a conditioning is read by its own request's images and by "
              "nothing after them";
        decide("--no-prefix-cache", "prefix caching off", prefix_off_why_);
    }
    if (embedding_) {
        if (cfg_.kld() || cfg_.n_spec > 0) {
            RAD_ERR("the '%s' plugin is an embedding model: it generates nothing, so --kld and "
                    "speculative decoding have nothing to work on", plugins_.arch_plugin.c_str());
            return RAD_E_UNSUPPORTED;
        }
        const Program& p0 = ranks_[0]->program;
        pooled_dim_ = p0.buffers[p0.pooled_buf].decl.shape[1];
        for (size_t i = 1; i < ranks_.size(); ++i) {
            const Program& pi = ranks_[i]->program;
            if (pi.pooled_buf == RAD_NULL_HANDLE ||
                pi.buffers[pi.pooled_buf].decl.shape[1] != pooled_dim_) {
                RAD_ERR("rank %zu declared a different pooled output from rank 0's", i);
                return RAD_E_STATE;
            }
        }
        pooled_host_ = static_cast<float*>(
            rad_dev_alloc(cfg_.max_seqs * pooled_dim_ * (int64_t)sizeof(float),
                          RAD_MEM_HOST_PINNED));
        if (!pooled_host_) {
            RAD_ERR("no pinned host memory for a step's pooled rows (%lld x %lld f32)",
                    (long long)cfg_.max_seqs, (long long)pooled_dim_);
            return RAD_E_NOMEM;
        }
        /* No prefix cache: an encoder keeps no KV, and with bidirectional attention a shared
         * prefix does not compute to the same rows. */
        cfg_.prefix_cache = false;
        prefix_off_why_ = "an embedding model: an encoder keeps no KV to reuse, and attending both "
                          "ways means a shared prefix does not compute to the same rows";
        decide("--no-prefix-cache", "prefix caching off", prefix_off_why_);
        RAD_INFO("embedding model: %lld-wide pooled vectors, one step an input, up to %lld "
                 "tokens an input", (long long)pooled_dim_, (long long)cfg_.max_tok);
    }

    /* The scheduler drives one block manager, rank 0's. Block ids are identical on every rank:
     * all carve the same block count (DeviceRuntime::configure_kv caps each at the smallest rank's), even
     * where a group's share differs in bytes or is zero (rad_kv_group_zero). Only the pool base
     * differs per rank (KVPoolBinding). Check the counts, since a mismatch would make one block
     * table mean different things on different cards. */
    for (size_t i = 1; i < ranks_.size(); ++i) {
        const auto& a = ranks_[0]->kv.plans();
        const auto& c = ranks_[i]->kv.plans();
        if (a.size() != c.size()) {
            RAD_ERR("rank %zu carved %zu KV groups and rank 0 carved %zu", i, c.size(), a.size());
            return RAD_E_STATE;
        }
        for (size_t g = 0; g < a.size(); ++g) {
            if (a[g].n_blocks == c[g].n_blocks && a[g].n_states == c[g].n_states) continue;
            RAD_ERR("KV group '%s': rank 0 carved %lld blocks / %lld states, rank %zu carved "
                    "%lld / %lld. One block table serves every rank, so they must agree.",
                    a[g].name.c_str(), (long long)a[g].n_blocks, (long long)a[g].n_states,
                    i, (long long)c[g].n_blocks, (long long)c[g].n_states);
            return RAD_E_STATE;
        }
    }

    /* Back to rank 0's device before building the scheduler: the loops above left this thread on
     * the last rank's device, and the scheduler's staging buffers and event must live on rank 0,
     * whose stream it is given (spec §1). */
    RAD_TRY(rad_dev_set(ranks_[0]->device));

    kv_iface_ = std::make_unique<KVBinding<KVManager>>(&ranks_[0]->kv);
    pc_iface_ = std::make_unique<PCBinding<PrefixCache>>(&ranks_[0]->prefix);
    /* The control vector's identity partitions the block cache (PCBinding::salt_id): a request
     * that runs with the projection on never adopts a block or a linear-state snapshot produced
     * without it, or the other way round. Zero with no vector loaded, and then every chain hashes
     * exactly as it did before the feature existed. */
    static_cast<PCBinding<PrefixCache>*>(pc_iface_.get())->salt_id = cvec_salt_;
    /* The restore path, installed only when the tiers are on. The scheduler calls it before the
     * admission lookup without knowing about tiers, so it is a callable rather than an include in
     * core/sched.
     *
     * Every rank restores its own shard. Rank 0's TierExec holds the policy: it allocates the
     * blocks and reports the job, and the others follow into the same block ids. A failure only
     * shortens the prefix, so no error is returned. */
    if (ranks_[0]->tierx.enabled()) {
        auto* b = static_cast<PCBinding<PrefixCache>*>(pc_iface_.get());
        /* Called inside Scheduler::step() under its lock, which makes touching the cache and rank
         * 0's manager safe; nothing here may call back into the scheduler. */
        b->restore_fn = [this, b](const Request& req, bool whole_only) -> RestoreResult {
            const uint64_t t0 = rad_mono_ns();
            PrefixCache& pc = ranks_[0]->prefix;
            /* Room is made before anything is planned (PCBinding::restore_plan): a chain the free
             * list cannot hold would otherwise come back only as far as it reaches. One block
             * table serves every rank, so what rank 0's cache gives up is room on every rank. */
            RestoreResult res = RestoreResult::Whole;
            const std::vector<BlockHash> want = b->restore_plan(
                req, cvec_salt_for(req.sp.use_control_vector), whole_only, &res);
            if (want.empty()) return res;
            /* The whole chain in one restore. With 4-token blocks a long conversation is tens of
             * thousands of entries; batched, it is one launch per (group, layer) per staging batch
             * and one copy per run of arena slots instead of a round trip per entry. */
            std::vector<TierJob> jobs;
            jobs.reserve(want.size());
            for (const BlockHash& h : want) {
                TierJob j;
                if (!ranks_[0]->tierx.plan_promote(h, &j)) break;
                jobs.push_back(std::move(j));
            }
            /* A record still in flight ends the chain here, as a short reservation does below. */
            if (jobs.size() < want.size()) res = RestoreResult::Cut;
            if (jobs.empty()) return res;
            std::vector<char> ok(jobs.size(), 1);
            for (auto& r : ranks_) {
                std::vector<char> mine;
                (void)r->tierx.issue_restore(jobs, r->ctx.stream(), &mine);
                for (size_t i = 0; i < ok.size(); ++i) ok[i] = ok[i] && i < mine.size() && mine[i];
            }
            /* Back to rank 0's device: the admission continues by growing rank 0's pool. */
            if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
            ranks_[0]->tierx.report_pass(jobs, ok, rad_mono_ns());
            /* Rebind every entry that came back, not only the leading run: the tiers already
             * marked each resident at its new blocks. A job's blocks are one per entry slot. */
            const std::vector<int32_t>& cg = pc.entry_groups();
            for (size_t i = 0; i < jobs.size(); ++i) {
                TierJob& j = jobs[i];
                if (!ok[i]) { res = RestoreResult::Cut; continue; }
                const int rb = pc.rebind(j.key, j.blocks);
                if (rb != RAD_OK) {
                    /* No entry to land in: release the reserved blocks and forget the tier record,
                     * which would otherwise release them later from under their next owner. */
                    ranks_[0]->tierx.quiesce();
                    for (size_t k = 0; k < cg.size() && k < j.blocks.size(); ++k)
                        if (j.blocks[k] >= 0) ranks_[0]->kv.release_block(cg[k], j.blocks[k]);
                    ranks_[0]->tiers.forget(j.key);
                }
            }
            ranks_[0]->tiers.note_restore((int64_t)(rad_mono_ns() - t0));
            return res;
        };
        /* An eviction that would lose an entry waits for its copy instead. Called by the prefix
         * cache binding under the scheduler's lock when dropping already-copied entries did not
         * free enough: all waiting copies are landed now, so those entries can leave VRAM. */
        b->flush_fn = [this]() -> bool { return flush_copies(); };
    }
    /* Per-rank card, stream and program. The batch is built once and shared by pointer on the
     * host (spec §1), but its device side is staged per rank: reading another card's staging is a
     * PCIe read and unordered against the copies that fill it (see Set in batch.h). */
    std::vector<BatchBuilder::RankIO> rio;
    rio.reserve(ranks_.size());
    rank_stream_views_.assign(ranks_.size(), {});
    for (size_t r = 0; r < ranks_.size(); ++r) {
        const size_t ng = ranks_[r]->program.kv_groups.size();
        rank_stream_views_[r].resize(ng, nullptr);
        for (size_t g = 0; g < ng; ++g)
            rank_stream_views_[r][g] = ranks_[r]->kv.stream_view((int32_t)g);
    }
    for (size_t r = 0; r < ranks_.size(); ++r) {
        auto& rk = ranks_[r];
        rio.push_back({ rk->device, rk->ctx.stream(), &rk->program, rk->pools.kv().base(),
                        rank_stream_views_[r].data() });
    }
    if (shape_gen) {
        /* As for the image generator below. */
        std::vector<gen::ShapeDriver::RankIO> gio;
        for (auto& rk : ranks_) gio.push_back({ rk->device, rk->ctx.stream() });
        auto d = std::make_unique<gen::ShapeDriver>();
        RAD_TRY(d->init(ranks_[0]->program, gio));
        shp_gen_ = d.get();
        gen_ = std::move(d);
        RAD_TRY(rad_dev_set(ranks_[0]->device));
    } else if (video_gen) {
        /* As for the image generator below; the plugin lays out each request's rows. */
        std::vector<gen::VideoDriver::RankIO> gio;
        for (auto& rk : ranks_) gio.push_back({ rk->device, rk->ctx.stream() });
        auto d = std::make_unique<gen::VideoDriver>();
        RAD_TRY(d->init(ranks_[0]->program, rad_tools_video_positions(plugins_), gio,
                        rad_tools_video_pack(plugins_), rad_tools_encode_geom(plugins_)));
        vid_gen_ = d.get();
        gen_ = std::move(d);
        RAD_TRY(rad_dev_set(ranks_[0]->device));
    } else if (audio_gen) {
        /* As for the image generator below. */
        std::vector<gen::AudioDriver::RankIO> gio;
        for (auto& rk : ranks_) gio.push_back({ rk->device, rk->ctx.stream() });
        auto d = std::make_unique<gen::AudioDriver>();
        RAD_TRY(d->init(ranks_[0]->program, kv_iface_.get(), gio));
        aud_gen_ = d.get();
        gen_ = std::move(d);
        RAD_TRY(rad_dev_set(ranks_[0]->device));
    } else if (generator) {
        /* The generator's driver replaces the scheduler, which is never initialised, so its
         * token-step staging is never allocated. */
        std::vector<gen::Driver::RankIO> gio;
        for (auto& rk : ranks_) gio.push_back({ rk->device, rk->ctx.stream() });
        auto d = std::make_unique<gen::Driver>();
        RAD_TRY(d->init(ranks_[0]->program, kv_iface_.get(), gio));
        img_gen_ = d.get();
        gen_ = std::move(d);
        RAD_TRY(rad_dev_set(ranks_[0]->device));
    } else
    RAD_TRY(sched_.init(ranks_[0]->program, cfg_, kv_iface_.get(), pc_iface_.get(), rio, &geo_));
    /* The smallest step the prefill stager overlaps. A chunk held back for other work
     * (--busy-prefill-chunk) stays at or above it; below it every routed layer reads its pooled
     * experts synchronously, which costs far more than sharing the step saves. The maximum over
     * ranks, since the step waits for the slowest. */
    {
        int64_t stage_rows = 0;
        for (auto& rk : ranks_)
            if (rk->stager.enabled()) stage_rows = std::max(stage_rows, rk->stager.min_rows());
        sched_.set_stage_rows(stage_rows);
        /* The shortest prompt a span pays for where routed experts are read from host memory: a
         * spanned step runs at the full plan, which lends the expert slab nothing, while every
         * other step runs at the span's shared level and lends it the span's extra rows. On
         * Qwen3.8-Flash-Next at TP2 (two R9700, 12 GiB host pool) the span loses to one-interval
         * steps up to 12K tokens and wins from 18K, hence eight intervals. With every expert on
         * the card there is no such cost and every lone prefill spans. */
        sched_.set_span_min(stage_rows > 0 && arena_span_rows_ > 0
                                ? 8 * std::max<int64_t>(geo_.checkpoint_interval, 0) : 0);
    }

    for (auto& r : ranks_) r->sampler.attach_vocab(vocab_view_.get());

    if (const int xs = xstats_open(); xs < 0) return xs;

    barrier_ = std::make_unique<RankBarrier>(cfg_.tp + 1);
    stopping_.store(false, std::memory_order_release);
    step_status_.store(RAD_OK, std::memory_order_release);
    loop_status_.store(RAD_OK, std::memory_order_release);
    done_.reserve((size_t)cfg_.max_seqs);

    rank_threads_.clear();
    for (int i = 0; i < cfg_.tp; ++i)
        rank_threads_.emplace_back([this, i] {
            EngineScope es(owner_, cfg_.served_model_name.c_str());
            rank_loop(i);
        });
    std::atomic<bool> stepped{false};
    std::thread stepper([this, &stepped] {
        EngineScope es(owner_, cfg_.served_model_name.c_str());
        if (gen_) gen_loop(); else step_loop();
        stepped.store(true, std::memory_order_release);
    });

    if (cfg_.kld() || cfg_.calib() || cfg_.diagnose_mode()) {
        print_startup_report(RAD_OK);
        startup_watch_.reset();
    }
    int s = cfg_.diagnose_mode() ? run_diagnose()
          : cfg_.kld()           ? run_kld()
          : cfg_.calib()         ? run_calib()
                                 : front ? front(sched_, vocab_, Json::parse(engine_info()))
                                         : run_server();
    startup_watch_.reset();

    /* Stop the step loop before the server's storage is destroyed: Request::sink points into it,
     * and an unfinished request still writes there (server.h). */
    stopping_.store(true, std::memory_order_release);
    wake_hold();
    barrier_->abort();
    /* A step loop that does not come back holds its cards and every model waiting for them, and
     * only a restart can free them, so past five minutes the process exits as for a GPU fault. */
    for (int t = 0; !stepped.load(std::memory_order_acquire); ++t) {
        if (t == 30000)
            fatal("model %s: the step loop has not come back five minutes after it was stopped; "
                  "exiting so a restart can bring the server back", cfg_.served_model_name.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (stepper.joinable()) stepper.join();
    for (auto& t : rank_threads_) if (t.joinable()) t.join();
    rank_threads_.clear();
    xstats_close();
    const char* manifest_dir = std::getenv("FREETOKEN_NATIVE_MANIFEST_DIR");
    if (manifest_dir && *manifest_dir) {
        const int saved = write_headless_manifest(manifest_dir);
        if (saved < 0 && s >= 0) s = saved;
    }
    /* A step-loop error takes precedence over the transport's status. */
    const int ls = loop_status_.load(std::memory_order_acquire);
    return ls < 0 ? ls : s;
}

/* The state slots a sequence owns, one per KV group, -1 where the group is not stateful. Read from
 * rank 0 and applied to every rank: only rank 0's KVManager tracks sequences (kv_iface_), and
 * every rank's pool has identical geometry, so a slot index means the same offset everywhere. */
bool DeviceRuntime::state_slots_of(uint64_t seq, std::vector<int32_t>* out) const {
    const KVManager& kv = ranks_[0]->kv;
    out->assign((size_t)kv.n_groups(), -1);
    bool any = false;
    for (int32_t g = 0; g < kv.n_groups(); ++g) {
        const KVGroupPlan* p = kv.plan(g);
        if (!p || !p->stateful()) continue;
        const int32_t s = kv.state_slot(seq, g);
        if (s < 0) return false;      /* the sequence is gone, or holds no state to copy */
        (*out)[(size_t)g] = s;
        any = true;
    }
    /* A window snapshot is a snapshot with no state slots (KVManager::has_window_snapshot). */
    return any || (kv.has_window_snapshot() && kv.has_sequence(seq));
}

/* ------------------------------------------------------------------ the HTTP frontend */
/* The control vector, for /stats and /server_info. `available` is
 * what a client should test before sending the field: without a loaded vector a request that asks
 * for it is refused, so a UI can hide the control rather than offer a switch that fails. */
std::string DeviceRuntime::cvec_json() const {
    Json o;
    o["available"] = cvec_.loaded();
    o["default"]   = cvec_default_.load();
    if (cvec_.loaded()) {
        /* The family the file was validated against, not a label the UI supplies: this is the same
         * `meta_.arch_id` the loader compared against the file's own model_hint. */
        o["arch"]    = meta_.arch_id ? meta_.arch_id : "";
        o["first"]   = cvec_.first;
        o["last"]    = cvec_.last;
        o["n_embd"]  = cvec_.n_embd;
        o["n_layers"] = cvec_.n_layers;
        o["scale"]   = cvec_.user_scale;
        o["file"]    = cvec_.path;
        o["mode"]    = "project";
    }
    return o.dump();
}

/* The engine's part of /server_info: model, cards, KV, drafting and wire, for a running process.
 * The server holds only interfaces, so the engine renders this object and the server splices it in
 * verbatim. */
std::string DeviceRuntime::engine_info() const {
    Json e;
    e["arch"]        = meta_.arch_id ? meta_.arch_id : "";
    e["quant"]       = meta_.quant ? meta_.quant : "";
    e["n_layers"]    = meta_.n_layers;
    e["n_embd"]      = meta_.n_embd;
    e["n_vocab"]     = meta_.n_vocab;
    e["n_ctx_train"] = meta_.n_ctx_train;
    e["container"]   = cfg_.model;

    e["tensor_parallel"]        = cfg_.tp;
    e["kv_cache_dtype"]         = cfg_.kv_cache_dtype;
    e["gpu_memory_mib"]         = cfg_.gpu_memory_mib;
    e["tp_wire"]                = cfg_.tp > 1 ? (cfg_.tp_wire_exact ? "exact" : "wht6") : "n/a";
    e["tp_wire_min_kb"]         = cfg_.tp_wire_exact ? 0 : cfg_.tp_wire_min_kb;
    e["tp_attn_split"]          = cfg_.tp_attn_rows ? "rows" : "heads";
    e["weight_prefetch"]        = cfg_.weight_prefetch;
    e["max_num_batched_tokens"] = cfg_.max_tok;
    e["placement"]              = cfg_.placement;
    /* Each lookup table's requested and resolved placement. */
    e["embedding_placement"]    = { { "asked", table_placement_name(cfg_.embedding_placement) },
                                    { "resolved", table_placement_name(cfg_.embedding_resolved) } };
    e["ngram_placement"]        = { { "asked", table_placement_name(cfg_.ngram_placement) },
                                    { "resolved", table_placement_name(cfg_.ngram_resolved) } };
    e["vision_tower_placement"] = { { "asked", table_placement_name(cfg_.tower_placement) },
                                    { "resolved", table_placement_name(cfg_.tower_resolved) } };
    e["prefix_cache"]           = cfg_.prefix_cache;
    e["control_vector"]         = Json::parse(cvec_json());
    e["checkpoint_slots"]       = cfg_.checkpoint_slots;
    e["deterministic"]          = cfg_.deterministic;

    Json devs = Json::array();
    for (const auto& r : ranks_) {
        RadDeviceProps p{};
        if (rad_dev_props(r->device, &p) < 0) continue;
        Json d;
        d["rank"]       = r->index;
        d["device"]     = r->device;
        d["name"]       = p.name;
        d["arch"]       = p.arch;
        d["vram_bytes"] = p.vram_bytes;
        d["vram_free"]  = p.vram_free;
        d["pci"]       = dev_pci_address(r->device);
        d["thread_id"] = r->native_thread_id.load(std::memory_order_acquire);
        devs.push_back(std::move(d));
    }
    e["devices"] = std::move(devs);

    /* Rank 0's; every rank declares the same graph (see the miss report in
     * engine_bringup.cpp). */
    const Program& p0 = ranks_[0]->program;
    e["ops"]     = (int64_t)p0.ops.size() - 1;
    e["weights"] = (int64_t)p0.weights.size() - 1;

    Json kvs = Json::array();
    const KVManager& kv = ranks_[0]->kv;
    for (int32_t g = 0; g < kv.n_groups(); ++g) {
        const KVGroupPlan* pl = kv.plan(g);
        /* Skip group 0, the sentinel that makes handle 0 mean "none". */
        if (!pl || pl->name.empty() || pl->n_layers <= 0) continue;
        Json k;
        k["name"]     = pl->name;
        k["kind"]     = bld_kv_kind_name(pl->kind);
        k["n_layers"] = pl->n_layers;
        k["bytes"]    = pl->pool_bytes;
        if (pl->paged()) {
            k["block_size"] = pl->block_size;
            k["blocks"]     = pl->n_blocks;
            k["tokens"]     = pl->n_blocks * pl->block_size;
        } else {
            k["states"] = pl->n_seq_states();
        }
        kvs.push_back(std::move(k));
    }
    e["kv_groups"] = std::move(kvs);
    e["prefix_key_namespace"] = !cfg_.prefix_cache ? "disabled"
        : geo_.exact && !geo_.window_ckpt ? "whole-cold-chunk-v1" : "token-prefix-v1";

    Json sp;
    if (drafter_.kind == RAD_DRAFT_NONE) {
        sp["enabled"] = false;
    } else {
        sp["enabled"] = true;
        sp["drafter"] = drafter_name_;
        sp["kind"]    = drafter_.kind == RAD_DRAFT_BLOCK ? "block" : "serial";
        sp["depth"]   = drafter_.depth;
    }
    e["speculation"] = std::move(sp);
    /* --expert-stats: where the counts are, for a client that reads them. */
    if (xstats_) e["expert_stats"] = { { "file", xstats_->path() }, { "interval_s", 30 } };
    return e.dump();
}

Json DeviceRuntime::headless_stats() const {
    Json j = {{"sampled_monotonic_ns", rad_mono_ns()}, {"allocator_high_water_bytes", dev_owner_high_water(owner_, -1)},
              {"host_pinned_budget_bytes", host_pinned_bytes()},
              {"host_pinned_budget_scope", "configured host/embedding/tower pools and prefix host tier; excludes shared ngram tables and checkpoint store"},
              {"memory_kind", "native owned device allocation/backing ledger; not torch reserved or sampled SMI VRAM"}};
    const char* kinds[] = {"device", "host_pinned", "host", "host_mapped"};
    j["process_allocations"] = Json::object();
    for (int kind = 0; kind < 4; ++kind) {
        int64_t bytes = 0, count = 0;
        rad_dev_alloc_totals(kind, &bytes, &count);
        j["process_allocations"][kinds[kind]] = {{"live_bytes", bytes},
                                                {"allocations_since_start", count}};
    }
    j["allocator_high_water_scope"] = "owner rad_dev_alloc device allocations plus elastic committed backing from engine construction through frees/rebuilds; excludes direct plugin hipMalloc; process physical VRAM is sampled separately";
    j["allocation_coverage"] = "process-wide rad_dev_alloc ledger; excludes direct plugin hipMalloc and is not RSS or allocator high-water";
    j["expert_counter_scope"] = "cumulative relaxed atomic snapshots; routed/resident and routed_stream_bytes are weighted routing-histogram observations, not measured PCIe traffic or a current resident-set inventory";
    j["ranks"] = Json::array();
    for (const auto& r : ranks_) {
        const auto ms = r->mover.stats();
        const auto hs = r->heat.stats();
        // Read the HTTP-safe tier mirror, never the dispatch thread's mutable tables.
        int64_t units[kTierCount] = {}, bytes[kTierCount] = {};
        for (size_t u = 0; u < r->plan.units.size(); ++u) {
            const auto& mu = r->plan.units[u];
            if (mu.expert < 0 || mu.layer < 0) continue;
            const Tier tier = r->heat.enabled() ? r->heat.tier_published((int32_t)u) : mu.tier;
            const int ti = int(tier) >= 0 && int(tier) < kTierCount ? int(tier) : int(Tier::SSD);
            ++units[ti]; bytes[ti] += mu.bytes;
        }
        Json placement = Json::object();
        for (int ti = 0; ti < kTierCount; ++ti)
            placement[tier_name(Tier(ti))] = {{"units", units[ti]}, {"bytes", bytes[ti]}};
        j["ranks"].push_back({{"rank", r->index}, {"native_committed_bytes", dev_owner_bytes(owner_, r->device)},
            {"native_high_water_bytes", dev_owner_high_water(owner_, r->device)},
            {"promotions", ms.promotions}, {"demotions", ms.demotions},
            {"h2d_bytes", ms.h2d_bytes}, {"d2h_bytes", ms.d2h_bytes},
            {"prefetched", ms.prefetched}, {"prefetch_bytes", ms.prefetch_bytes},
            {"staged_units", ms.staged_units}, {"staged_bytes", ms.staged_bytes},
            {"stage_overflow", ms.stage_overflow}, {"expert_placement", std::move(placement)},
            {"prefetch_useful_bytes", nullptr}, {"prefetch_wasted_bytes", nullptr},
            {"expert_heat", {{"dispatches", hs.dispatches}, {"proposed", hs.proposed},
                {"promotions", hs.promotions}, {"demotions", hs.demotions},
                {"refused", hs.refused}, {"no_candidate", hs.no_candidate},
                {"distinct_promoted", hs.distinct_promoted}, {"readmits", hs.readmits},
                {"routed", hs.routed}, {"routed_resident", hs.routed_resident},
                {"routed_stream_bytes", hs.stream_bytes}}}});
    }
    return j;
}

int DeviceRuntime::write_headless_manifest(const std::string& directory) const {
    Json root;
    root["engine"] = Json::parse(engine_info());
    root["pid"] = int(getpid());
    root["native_stats"] = headless_stats();
    root["ranks"] = Json::array();
    for (const auto& r : ranks_) {
        const auto stats = r->mover.stats();
        Json j = {{"rank", r->index}, {"device", r->device},
                  {"thread_id", r->native_thread_id.load(std::memory_order_acquire)},
                  {"pci", dev_pci_address(r->device)},
                  {"native_committed_bytes", dev_owner_bytes(owner_, r->device)},
            {"native_high_water_bytes", dev_owner_high_water(owner_, r->device)},
                  {"weight_pool_used", r->pools.weights().used()},
                  {"weight_pool_bytes", r->pools.weights().bytes()},
                  {"kv_pool_used", r->pools.kv().used()},
                  {"kv_pool_bytes", r->pools.kv().bytes()},
                  {"host_pool_used", r->pools.host_pinned().used()},
                  {"host_pool_bytes", r->pools.host_pinned().bytes()},
                  {"heat_enabled", r->heat.enabled()},
                  {"placement_owner", r->heat.owner()},
                  {"placement_control",Json::parse(r->heat.control_report())},
                  {"promotions", stats.promotions}, {"demotions", stats.demotions},
                  {"h2d_bytes", stats.h2d_bytes}, {"d2h_bytes", stats.d2h_bytes}};
        const auto& ps = r->prefix.stats();
        j["prefix_counters"] = {{"lookups", ps.lookups}, {"block_hits", ps.block_hits},
            {"block_misses", ps.block_misses}, {"checkpoint_hits", ps.ckpt_hits},
            {"checkpoint_misses", ps.ckpt_misses}, {"evictions", ps.evictions},
            {"checkpoint_evictions", ps.ckpt_evictions}, {"checkpoint_retired", ps.ckpt_retired},
            {"checkpoint_lru", ps.ckpt_lru}, {"checkpoint_refused", ps.ckpt_refused},
            {"drops_to_host", ps.drops}};
        // Rank threads have joined; these transfer counters can be read without a hot-path lock.
        const auto& tx = r->tierx.stats();
        const auto policy = r->tiers.stats();
        j["kv_tier_transfers"] = {{"passes", tx.passes}, {"jobs", tx.jobs},
            {"copy_out_bytes", tx.bytes_out}, {"restore_in_bytes", tx.bytes_in},
            {"replicated_checks", tx.rep_checked}, {"replicated_mismatches", tx.rep_mismatch}};
        j["kv_tier_policy"] = {{"to_host", policy.to_host}, {"dropped", policy.dropped},
            {"promoted", policy.promoted}, {"host_hits", policy.host_hits},
            {"failures", policy.failures}, {"host_drops", policy.host_drops}};
        j["state_groups"] = Json::array();
        for (size_t gi = 1; gi < r->program.kv_groups.size(); ++gi) {
            const auto& g = r->program.kv_groups[gi]; const auto& d = g.decl;
            j["state_groups"].push_back({{"name",g.name},{"kind",d.kind},{"dtype",d.dtype},
                {"heads",d.n_head_kv},{"head_dim",d.head_dim},{"state_dim",{d.state_dim[0],d.state_dim[1]}},
                {"conv_width",d.conv_width},{"block_tokens",g.block_size},{"layers",g.layers},
                {"bytes_per_block",g.bytes_per_block},{"bytes_per_state",g.bytes_per_state}});
        }
        root["ranks"].push_back(std::move(j));
        std::ofstream graph(directory + "/resolved-ops-rank" + std::to_string(r->index) + ".json");
        if (!graph) return RAD_E_IO;
        graph << dump_graph_json(r->program, reg_.get()) << '\n';
        if (!graph) return RAD_E_IO;
        std::ofstream passes(directory + "/collective-ledger-rank" + std::to_string(r->index) + ".json");
        passes << r->ctx.pass_ledger_json() << '\n';
        if (!passes) return RAD_E_IO;
        std::ofstream issued(directory + "/issued-ops-rank" + std::to_string(r->index) + ".jsonl");
        for (size_t i = 1; i < r->program.ops.size(); ++i) {
            const auto& op = r->program.ops[i];
            issued << Json({{"handle", i}, {"op", op.op}, {"geometry", op.base.str()},
                            {"issued", op.issued}}).dump() << '\n';
        }
        if (!issued) return RAD_E_IO;
        std::ofstream shards(directory + "/tp-weights-rank" + std::to_string(r->index) + ".jsonl");
        for (size_t i = 1; i < r->program.weights.size(); ++i) {
            const auto& w = r->program.weights[i];
            Json row = {{"handle", i}, {"name", w.name}, {"source", w.source},
                        {"shard", w.decl.shard}, {"shard_lo", w.shard_lo}, {"shard_hi", w.shard_hi},
                        {"part_lo", std::vector<int64_t>(w.part_lo, w.part_lo + w.n_part_spans)},
                        {"part_hi", std::vector<int64_t>(w.part_hi, w.part_hi + w.n_part_spans)},
                        {"part_total", std::vector<int64_t>(w.part_total, w.part_total + w.n_part_spans)},
                        {"logical_shape", std::vector<int64_t>(w.decl.shape, w.decl.shape + w.decl.rank)},
                        {"stored_shape", std::vector<int64_t>(w.stored_shape, w.stored_shape + w.stored_rank)},
                        {"layout", w.layout_tag}, {"stored_dtype", w.stored_dtype},
                        {"stored_bytes", w.stored_bytes}, {"tier", tier_name(w.tier)},
                        {"site", site_name(w.site)}};
            shards << row.dump() << '\n';
        }
        if (!shards) return RAD_E_IO;
    }
    std::ofstream memory(directory + "/native-manifest.json");
    if (!memory) return RAD_E_IO;
    memory << root.dump() << '\n';
    return memory ? RAD_OK : RAD_E_IO;
}

namespace {

/* ================================================================== sampling defaults */
/* The sampler for a request that sends no sampler settings. Parsing and range checks live in
 * core/gen_config.h; this is the precedence order and the container lookup:
 *
 *   --temp / --top-k / --top-p / --min-p    an operator overriding, field by field, and last
 *   --generation-config PATH                a generation_config.json named outright
 *   <model>.generation.json                 one placed beside the container
 *   the model's `generation.*` metadata     generation_config.json, as the container or the
 *                                          checkpoint carries it
 *   SamplingParams' own defaults            the untruncated floor
 *
 * The flags override field by field. Of the file sources, the first that states anything is used
 * whole, never blended with a lower one.
 *
 * `stated` and `source`, when given, receive what the source in force read and its name ("" for
 * none), which gen_sampling_sources turns into a reason per field. */
SamplingParams resolve_sampling_defaults(const RadModelMeta& meta, bool container,
                                         const Config& cfg, GenSampling* stated = nullptr,
                                         std::string* source = nullptr) {
    GenSampling g;
    std::string src;

    if (!cfg.generation_config.empty() &&
        gen_sampling_from_file(cfg.generation_config, /*required=*/true, &g))
        src = cfg.generation_config;

    if (!g.any()) {
        const std::string side = gen_sidecar_path(cfg.model);
        if (gen_sampling_from_file(side, /*required=*/false, &g)) src = side;
    }
    if (!g.any()) {
        auto get = [&meta](const char* field) {
            return rad_meta_gets(&meta, (std::string("generation.") + field).c_str(), nullptr);
        };
        if (gen_sampling_from_kv(get, &g)) src = container ? "the container" : "the checkpoint";
    }

    SamplingParams sp = g.onto(SamplingParams{});

    std::string over;
    if (cfg.sample_temp  >= 0.0f) { sp.temp  = cfg.sample_temp;  over += " --temp"; }
    if (cfg.sample_top_p >= 0.0f) { sp.top_p = cfg.sample_top_p; over += " --top-p"; }
    if (cfg.sample_min_p >= 0.0f) { sp.min_p = cfg.sample_min_p; over += " --min-p"; }
    if (cfg.sample_top_k >= 0)    { sp.top_k = cfg.sample_top_k; over += " --top-k"; }

    /* The remaining fields have no file source (generation_config.json supplies only the four
     * above): SamplingParams' defaults unless a flag sets them; request_overrides() logs those. */
    if (cfg.sample_typical_p)          sp.typical_p          = *cfg.sample_typical_p;
    if (cfg.sample_rep_penalty)        sp.rep_penalty        = *cfg.sample_rep_penalty;
    if (cfg.sample_pres_penalty)       sp.pres_penalty       = *cfg.sample_pres_penalty;
    if (cfg.sample_freq_penalty)       sp.freq_penalty       = *cfg.sample_freq_penalty;
    if (cfg.sample_penalty_last_n)     sp.penalty_last_n     = *cfg.sample_penalty_last_n;
    if (cfg.sample_dry_multiplier)     sp.dry_multiplier     = *cfg.sample_dry_multiplier;
    if (cfg.sample_dry_base)           sp.dry_base           = *cfg.sample_dry_base;
    if (cfg.sample_dry_allowed_length) sp.dry_allowed_length = *cfg.sample_dry_allowed_length;
    if (cfg.sample_dry_penalty_last_n) sp.dry_penalty_last_n = *cfg.sample_dry_penalty_last_n;
    if (cfg.sample_dry_seq_breakers)   sp.dry_seq_breakers   = *cfg.sample_dry_seq_breakers;
    if (cfg.sample_xtc_probability)    sp.xtc_probability    = *cfg.sample_xtc_probability;
    if (cfg.sample_xtc_threshold)      sp.xtc_threshold      = *cfg.sample_xtc_threshold;

    const std::string from = src.empty() ? (over.empty() ? "nothing stated it" : "the command line")
                                         : src;
    if (stated) *stated = g;
    if (source) *source = src;
    RAD_INFO("sampling defaults for requests that send none: %s (from %s%s%s)",
             GenSampling::describe(sp).c_str(), from.c_str(),
             over.empty() || src.empty() ? "" : ", overridden by", over.c_str());
    if (src.empty() && over.empty())
        RAD_WARN("this checkpoint states no sampling of its own and none was given, so a request "
                 "that sends no sampler fields is served with the whole vocabulary live at "
                 "temperature 1 -- which ends a reply early often enough to notice. Pass "
                 "--generation-config, or put the checkpoint's generation_config.json beside the "
                 "container as %s.", gen_sidecar_path(cfg.model).c_str());
    return sp;
}

/* "presence_penalty 1.5, max_tokens auto, ...": every request default or bound the command line
 * changed, in the request's field names; "" if none. Excludes the four fields
 * resolve_sampling_defaults reports. */
std::string request_overrides(const Config& c) {
    std::string out;
    auto add = [&out](const std::string& s) { out += (out.empty() ? "" : ", ") + s; };
    char b[64];
    auto f = [&](const char* k, const std::optional<float>& v) {
        if (v) { snprintf(b, sizeof b, "%s %g", k, (double)*v); add(b); }
    };
    auto i = [&](const char* k, const std::optional<int>& v) {
        if (v) add(std::string(k) + " " + std::to_string(*v));
    };
    f("typical_p", c.sample_typical_p);
    f("presence_penalty", c.sample_pres_penalty);
    f("frequency_penalty", c.sample_freq_penalty);
    f("repetition_penalty", c.sample_rep_penalty);
    i("repeat_last_n", c.sample_penalty_last_n);
    f("dry_multiplier", c.sample_dry_multiplier);
    f("dry_base", c.sample_dry_base);
    i("dry_allowed_length", c.sample_dry_allowed_length);
    i("dry_penalty_last_n", c.sample_dry_penalty_last_n);
    if (c.sample_dry_seq_breakers)
        add("dry_sequence_breakers " + std::to_string(c.sample_dry_seq_breakers->size()) +
            " strings");
    f("xtc_probability", c.sample_xtc_probability);
    f("xtc_threshold", c.sample_xtc_threshold);
    if (c.default_max_tokens == 0) add("max_tokens auto");
    else if (c.default_max_tokens > 0) add("max_tokens " + std::to_string(c.default_max_tokens));
    if (c.max_tokens_cap > 0) add("max_tokens capped at " + std::to_string(c.max_tokens_cap));
    if (c.max_n > 0) add("n up to " + std::to_string(c.max_n));
    if (c.max_stop_strings != 64 || c.max_stop_bytes != 4096)
        add("stop up to " + std::to_string(c.max_stop_strings) + " strings of " +
            std::to_string(c.max_stop_bytes) + " bytes");
    for (const auto& [k, v] : c.chat_template_kwargs) add("template " + k + "=" + v);
    if (c.reasoning_format != "auto") add("reasoning_format " + c.reasoning_format);
    if (c.thinking_token_budget >= 0)
        add("thinking_token_budget " + std::to_string(c.thinking_token_budget));
    return out;
}
}  /* namespace */

SamplingParams DeviceRuntime::sampling_defaults() const {
    return resolve_sampling_defaults(meta_, file_ != nullptr, cfg_);
}

/* The HTTP server over the engine. `/v1/completions`, `/v1/chat/completions`, `/v1/models`,
 * `/metrics` and `/health` are live. `/v1/embeddings` answers 501 unless the program is an
 * embedding program, and so do image, video and audio content parts unless the program declared an
 * encoder that accepts them (MEDIA, below). The chat endpoint answers 501 only when the model has
 * no chat template: a guessed template is a guessed prompt format. With one, the template's
 * grammar and lazy triggers reach the sampler and its derived reply parser comes back through
 * IChatTemplate::apply. */
int DeviceRuntime::run_server() {
    /* The rank-0 sampler lets admission refuse, by name, a request whose sampling stage has no
     * kernel (IScheduler::admit). The tiers are passed even when they move nothing, because they
     * hold the session table the sessions view reads. */
    server::SchedulerBridge sched_bridge(&sched_, ranks_.empty() ? nullptr : &ranks_[0]->sampler,
                                        ranks_.empty() ? nullptr : &ranks_[0]->tiers);
    server::TokenizerBridge tok_bridge(vocab_);
    server::ChatTemplateBridge chat_bridge;
    server::GrammarBridge grammar_bridge(vocab_view_.get());

    /* One snapshot function feeds both the live view (spec §16) and the web dashboard, so they
     * cannot disagree. It runs on the drawing thread and must not slow the engine: it reads the
     * scheduler's published copy (Scheduler::publish), the driver's VRAM counter, and rank 0's
     * plan, immutable after start().
     *
     * The cards' fixed facts are read once here, before serving: rad_dev_props enters the device
     * runtime (properties, device switch, free-memory query, peer walk) and also opens the counter
     * dev_vram_used reads. */
    std::vector<server::CardStat> card_facts;
    for (const auto& r : ranks_) {
        RadDeviceProps pr{};
        if (rad_dev_props(r->device, &pr) < 0) continue;
        server::CardStat c;
        c.index      = r->device;
        c.name       = pr.name;
        c.vram_total = pr.vram_bytes;
        card_facts.push_back(std::move(c));
    }
    server::LiveView::Source snap = [this, &sched_bridge, card_facts](server::LiveSnapshot& s) {
        server::snapshot_from(sched_bridge.metrics(), &s);
        s.model = meta_.name && *meta_.name ? meta_.name : "radiance";

        s.cards.clear();
        for (const server::CardStat& f : card_facts) {
            /* vram_used is sampled per call. A card whose counter cannot be read is left out.
             * Utilisation, power and temperature stay 0 ("not sampled"): the device layer does
             * not expose them. */
            int64_t used = 0;
            if (dev_vram_used(f.index, &used) != RAD_OK) continue;
            server::CardStat c = f;
            c.vram_used = used;
            s.cards.push_back(std::move(c));
        }

        /* The expert plane from rank 0's placement plan; empty for a dense model, and the
         * renderers then omit the panel. */
        s.experts = server::ExpertPlane{};
        if (!ranks_.empty()) {
            const Plan& p = ranks_[0]->plan;
            /* Read once for both the expert and link halves of the panel, so they agree. */
            const MoverStats ms = ranks_[0]->mover.stats();
            int32_t n_layer = 0, n_expert = 0;
            for (const MoveUnit& mu : p.units)
                if (mu.expert >= 0) {
                    n_layer  = std::max(n_layer,  mu.layer + 1);
                    n_expert = std::max(n_expert, mu.expert + 1);
                }
            if (n_layer > 0 && n_expert > 0) {
                s.experts.n_layers  = n_layer;
                s.experts.n_experts = n_expert;
                s.experts.tier.assign((size_t)n_layer * (size_t)n_expert, (uint8_t)Tier::VRAM);
                /* The live tier when the heat engine runs, else the plan's (where a unit started).
                 * tier_published is the HTTP-readable mirror HeatEngine::set_tier keeps; reading
                 * the dispatch thread's own tier_ here would be a data race. */
                const HeatEngine& heat = ranks_[0]->heat;
                const bool live_tier = heat.enabled();
                s.experts.heat_enabled = live_tier;
                s.experts.layer_units.assign((size_t)n_layer, 0);
                s.experts.layer_resident.assign((size_t)n_layer, 0);
                for (size_t u = 0; u < p.units.size(); ++u) {
                    const MoveUnit& mu = p.units[u];
                    if (mu.expert < 0 || mu.layer < 0) continue;
                    const Tier t = live_tier ? heat.tier_published((int32_t)u) : mu.tier;
                    s.experts.tier[(size_t)mu.layer * (size_t)n_expert + (size_t)mu.expert]
                        = (uint8_t)t;
                    /* Count residency over units, not plane cells: the plane is a rectangle a
                     * model need not fill (a draft head's expert count can differ), and empty
                     * cells default to VRAM. */
                    const int ti = (int)t >= 0 && (int)t < kTierCount ? (int)t : (int)Tier::SSD;
                    s.experts.units[ti]++;
                    s.experts.bytes[ti] += mu.bytes;
                    s.experts.n_units++;
                    s.experts.total_bytes += mu.bytes;
                    if (mu.movable) s.experts.movable++;
                    s.experts.layer_units[(size_t)mu.layer]++;
                    if (t == Tier::VRAM) s.experts.layer_resident[(size_t)mu.layer]++;
                }
                s.experts.promotions   = (int64_t)ms.promotions;
                s.experts.demotions    = (int64_t)ms.demotions;
                s.experts.no_record    = (int64_t)ms.no_record;
                s.experts.starved_slab = (int64_t)ms.starved_slab;
                s.experts.starved_pool = (int64_t)ms.starved_pool;
                s.experts.quarantined  = (int64_t)ms.quarantined;
                s.experts.stage_layers   = (int64_t)ms.stage_layers;
                s.experts.staged_units   = (int64_t)ms.staged_units;
                s.experts.staged_bytes   = (int64_t)ms.staged_bytes;
                s.experts.stage_overflow = (int64_t)ms.stage_overflow;
                s.experts.stage_copies   = (int64_t)ms.stage_copies;
                s.experts.flex_bytes    = ranks_[0]->mover.flex_used_bytes();
                s.experts.flex_capacity = ranks_[0]->mover.flex_offered_bytes();
                /* The heat engine's decision counts (the mover counts bytes): dispatches that
                 * proposed nothing and why, and whether a promotion was a first (investment) or a
                 * readmit of a just-demoted unit (pure loss). */
                const HeatStats hs = heat.stats();
                s.experts.dispatches        = (int64_t)hs.dispatches;
                s.experts.refused           = (int64_t)hs.refused;
                s.experts.no_candidate      = (int64_t)hs.no_candidate;
                s.experts.distinct_promoted = (int64_t)hs.distinct_promoted;
                s.experts.readmits          = (int64_t)hs.readmits;
                /* The routed hit rate: residency describes the plan, this describes the traffic. */
                s.experts.routed            = (int64_t)hs.routed;
                s.experts.routed_resident   = (int64_t)hs.routed_resident;

            }

            /* The link panel. All-reduce counts come from rank 0 only: the ranks are symmetric,
             * so summing would double them. Moves and streamed experts are summed over ranks:
             * each rank moves its own shard over its own link. `world` lets a reader recover the
             * per-card figure to compare against one link's ceiling. */
            s.passes.present = true;
            for (const auto& rk : ranks_) {
                const Ctx::PassCounts pc = rk->ctx.pass_counts();
                s.passes.played         += pc.played;
                s.passes.recorded       += pc.recorded;
                s.passes.audited        += pc.audited;
                s.passes.issued_unkeyed += pc.issued_unkeyed;
                s.passes.issued_refused += pc.issued_refused;
                s.passes.issued_first   += pc.issued_first;
                s.passes.dispatched     += pc.dispatched;
                s.passes.unordered      += pc.unordered;
            }

            s.link.present        = true;
            s.link.world          = (int)ranks_.size();
            s.link.ar_calls       = ranks_[0]->ctx.ar_calls();
            s.link.ar_bytes       = ranks_[0]->ctx.ar_bytes();
            s.link.h2d_bytes = s.link.d2h_bytes = s.link.ssd_reads = 0;
            s.link.ssd_bytes = s.link.prefetched = s.link.prefetch_bytes = 0;
            s.link.stream_bytes = s.link.staged_bytes = 0;
            s.link.ssd_staging_bytes = s.link.ssd_promoted_bytes = 0;
            s.link.ssd_deposits = s.link.ssd_deposit_bytes = 0;
            s.link.pool_evictions = 0;
            s.link.kv_stream_bytes = 0;
            s.link.kv_streaming = cfg_.kv_resident > 0;   /* zeroed here if the arch refused it */
            /* The ring's depth, not the flag: KVStreamStats::resident_blocks sums the device pool
             * of every group that really runs a window under its context, so this is zero when
             * --kv-resident was asked for and nothing streams anyway. */
            s.link.kv_stream_active = ranks_[0]->kv.stream_stats().resident_blocks > 0;
            for (const auto& rk : ranks_) {
                const MoverStats rms = rk->mover.stats();
                s.link.h2d_bytes      += (int64_t)rms.h2d_bytes;
                s.link.d2h_bytes      += (int64_t)rms.d2h_bytes;
                s.link.ssd_reads      += (int64_t)rms.ssd_reads;
                s.link.ssd_bytes      += (int64_t)rms.ssd_bytes;
                s.link.ssd_staging_bytes += (int64_t)rms.ssd_staging_bytes;
                s.link.ssd_promoted_bytes += (int64_t)rms.ssd_promoted_bytes;
                s.link.ssd_deposits    += (int64_t)rms.ssd_deposits;
                s.link.ssd_deposit_bytes += (int64_t)rms.ssd_deposit_bytes;
                s.link.pool_evictions   += (int64_t)rms.pool_evictions;
                s.link.prefetched     += (int64_t)rms.prefetched;
                s.link.prefetch_bytes += (int64_t)rms.prefetch_bytes;
                s.link.staged_bytes   += (int64_t)rms.staged_bytes;
                if (rk->heat.enabled()) s.link.stream_bytes += (int64_t)rk->heat.stats().stream_bytes;
            }
            /* KV streaming: only rank 0's resolve counters are read -- every rank sees the same
             * misses by design -- so the figure takes the world factor to reach all cards, the
             * way the summed move rows sit there. The counters ride the step's ctl read; nothing
             * here costs a step. */
            if (s.link.kv_streaming)
                s.link.kv_stream_bytes =
                    (int64_t)ranks_[0]->kv.stream_stats().bytes_in * (int64_t)ranks_.size();
        }
    };

    server::Deps d{};
    d.sched = &sched_bridge;
    d.background = &background_jobs_;
    d.front = front_;
    d.tok   = &tok_bridge;
    d.grammar = &grammar_bridge;
    d.live  = snap;
    d.stuck_ms = [] { return dev_stuck_ms(); };
    /* The control vector, when one was found. The admin endpoint flips the
     * default through this; nothing rebuilds a graph, so a toggle takes effect on the next request
     * and leaves the running ones alone. */
    if (cvec_.loaded()) {
        d.control_vector.available  = [this] { return cvec_loaded(); };
        d.control_vector.get_default = [this] { return cvec_default_on(); };
        d.control_vector.set_default = [this](bool on) { set_cvec_default(on); };
        d.control_vector.json        = [this] { return cvec_json(); };
    }

    /* An embedding model serves /v1/embeddings and no chat. Its named prompts are the checkpoint's
     * sentence-transformers configuration (`sentence_transformers.prompts.*`); a `messages` input
     * renders through its chat template. */
    struct EmbedInfo final : server::IEmbedder {
        int64_t d = 0, n = 0;
        const RadModelMeta* meta = nullptr;
        server::IChatTemplate* tpl = nullptr;
        int64_t dim() const override { return d; }
        int64_t max_input() const override { return n; }
        bool prompt(const std::string& name, std::string* text) const override {
            const std::string key = "sentence_transformers.prompts." + name;
            const char* v = rad_meta_gets(meta, key.c_str(), nullptr);
            if (!v) return false;
            *text = v;
            return true;
        }
        std::string default_prompt() const override {
            const char* v = rad_meta_gets(meta, "sentence_transformers.default_prompt_name", "");
            return v ? std::string(v) : std::string();
        }
        server::IChatTemplate* message_template() const override { return tpl; }
        const RadModelMeta* metadata() const override { return meta; }
    } embed_info;
    embed_info.d = pooled_dim_;
    embed_info.n = cfg_.max_tok;
    embed_info.meta = &meta_;
    if (embedding_) d.embed = &embed_info;
    std::string embed_why;
    const int embed_rc = server::check_embedding_template(
        d.embed, !cfg_.override_chat_template.empty(), &embed_why);
    if (embed_rc < 0) {
        RAD_ERR("%s", embed_why.c_str());
        return embed_rc;
    }

    /* The template loader warns on its own about a missing or broken template; this only decides
     * whether the chat endpoint exists. The reply format is the architecture plugin's when it
     * declares one, else derived from the template. */
    const RadChatFormat* declared_format = rad_tools_chat_format(plugins_, meta_);
    /* An image model's template is its conditioning network's; nothing is served as chat. */
    const bool have_tpl = !gen_ && vocab_ && !vocab_->chat_template().empty() &&
                          chat_bridge.init(*vocab_, declared_format, plugins_.arch_plugin) >= 0 &&
                          chat_bridge.ready();
    if (embedding_) {
        /* No chat endpoint: the model's template, when it has one, renders a `messages` input. */
        if (have_tpl) embed_info.tpl = &chat_bridge;
    } else if (have_tpl) {
        d.chat = &chat_bridge;
    } else if (!gen_) {
        RAD_WARN("no chat template in this container; /v1/chat/completions will answer 501 and "
                 "/v1/completions is the endpoint to use");
    }

    server::ServerOptions o{};
    o.embedding_model = embedding_;
    o.image_model = img_gen_ != nullptr;
    o.audio_model = aud_gen_ != nullptr;
    o.video_model = vid_gen_ != nullptr;
    o.shape_model = shp_gen_ != nullptr;
    o.decision_type = rad_meta_gets(&meta_, "decision.type", "");
    /* clef's learned scalars, which the server applies to what the head returns (server/clef.h);
     * rad-convert writes them beside decision.type. */
    if (o.decision_type == "clef") {
        const char* const keys[3] = { "decision.clef.prior_logit_scale",
                                      "decision.clef.joint_logit_scale",
                                      "decision.clef.residual_gate" };
        o.clef_scalars = true;
        for (const char* k : keys) o.clef_scalars = o.clef_scalars && rad_meta_gets(&meta_, k, nullptr);
        o.clef_prior_logit_scale = rad_meta_getf(&meta_, keys[0], 0.0);
        o.clef_joint_logit_scale = rad_meta_getf(&meta_, keys[1], 0.0);
        o.clef_residual_gate     = rad_meta_getf(&meta_, keys[2], 0.0);
    }
    /* Laya's sequence lengths, special tokens and fitted temperatures, which the server builds
     * and reads each question with (server/laya.h); rad-convert writes them from the checkpoint's
     * rl_agent_config.json and tokenizer/. */
    /* Mica's calibration: one temperature, and a bias on a noul's true logit (server/mica.h). */
    if (o.decision_type == "mica") {
        o.mica.temperature     = rad_meta_getf(&meta_, "decision.mica.temperature", 1.0);
        o.mica.noul_logit_bias = rad_meta_getf(&meta_, "decision.mica.noul_logit_bias", 0.0);
    }
    if (o.decision_type == "laya") {
        server::LayaSettings& l = o.laya;
        l.max_len      = rad_meta_geti(&meta_, "decision.laya.max_len", 0);
        l.head_max_len = rad_meta_geti(&meta_, "decision.laya.head_max_len", 0);
        l.cls  = (int32_t)rad_meta_geti(&meta_, "decision.laya.cls", -1);
        l.sep  = (int32_t)rad_meta_geti(&meta_, "decision.laya.sep", -1);
        l.mask = (int32_t)rad_meta_geti(&meta_, "decision.laya.mask", -1);
        l.mask_token = rad_meta_gets(&meta_, "decision.laya.mask_token", "");
        std::sscanf(rad_meta_gets(&meta_, "decision.laya.temperature", "1 1 1"), "%lf %lf %lf",
                    &l.temperature[0], &l.temperature[1], &l.temperature[2]);
        const std::string tbo = "decision.laya.temperature_by_options.";
        for (int i = 0; i < meta_.n_kv; ++i)
            if (std::strncmp(meta_.kv_key[i], tbo.c_str(), tbo.size()) == 0)
                l.temperature_by_options[meta_.kv_key[i] + tbo.size()] = std::strtod(meta_.kv_val[i], nullptr);
    }
    o.plugin_words = ranks_.empty() ? 0 : ranks_[0]->program.plugin_words;
    o.model_id = !cfg_.served_model_name.empty() ? cfg_.served_model_name
               : meta_.name && *meta_.name       ? meta_.name : "radiance";
    o.host     = cfg_.host;
    o.port     = cfg_.port;
    o.api_key  = cfg_.api_key;
    o.local_hosts = cfg_.allow_hosts;
    o.trusted_proxies = cfg_.trusted_proxies;
    o.max_seqs = cfg_.max_seqs;
    /* `n` fans one request out over the batch, so the default scales with --max-num-seqs; the
     * floor of 8 keeps a small batch from refusing what it would queue. */
    o.max_n    = cfg_.max_n > 0 ? (int)cfg_.max_n : (int)std::max<int64_t>(o.max_n, cfg_.max_seqs);
    if (cfg_.max_n <= 0)
        decide("--max-n", std::to_string(o.max_n), "the larger of 8 and --max-num-seqs");
    o.max_ctx  = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
    if (!gen_ && !ranks_.empty()) {
        o.kv_cache_dtype = cfg_.kv_cache_dtype;
        for (const KVGroupInfo& kg : ranks_[0]->program.kv_groups)
            if (kg.block_size > 0) { o.kv_block_size = kg.block_size; break; }
        o.prefix_cache = cfg_.prefix_cache;
        o.prefix_cache_mib = cfg_.prefix_cache_mib;
    }
    o.model_path = cfg_.model;
    o.default_reasoning_effort = cfg_.reasoning_effort;
    o.reasoning_effort_map = cfg_.reasoning_effort_map;
    o.default_thinking_budget = cfg_.thinking_token_budget;
    if (!gen_) {
        /* The four sampler defaults a file can state: the served value and, per field, its
         * source. A generator samples by its own rules and serves no text requests. */
        GenSampling stated;
        std::string src;
        o.default_sampling =
            resolve_sampling_defaults(meta_, file_ != nullptr, cfg_, &stated, &src);
        for (const GenFieldSource& f : gen_sampling_sources(stated, src, o.default_sampling, cfg_))
            decide(f.flag, f.value, f.why);
    }
    if (cfg_.served_model_name.empty())
        decide("--served-model-name", o.model_id,
               meta_.name && *meta_.name ? "the name the container states"
                                         : "the container states no name");
    o.default_dry_breakers_set = cfg_.sample_dry_seq_breakers.has_value();
    /* An embedding model samples nothing, so it has no logprobs to give. */
    o.supports_logprobs = !embedding_ && ranks_[0]->sampler.has_logprobs();
    /* -1 keeps ServerOptions' default, `auto` (0), resolved per request against the context, so
     * it needs a known context. */
    if (cfg_.default_max_tokens >= 0) o.default_max_tokens = cfg_.default_max_tokens;
    if (o.default_max_tokens == 0 && o.max_ctx <= 0) {
        RAD_ERR("a request that names no max_tokens gets what the context leaves after its prompt "
                "(--default-max-tokens auto), and this model states no context; pass "
                "--max-model-len, or --default-max-tokens N");
        return RAD_E_INVAL;
    }
    o.max_tokens_cap  = cfg_.max_tokens_cap;
    o.queue_depth     = cfg_.max_queued_requests;      /* 0: the server's 8 x max_seqs */
    o.max_stops       = cfg_.max_stop_strings;
    o.max_stop_bytes  = cfg_.max_stop_bytes;
    o.default_template_kwargs = cfg_.chat_template_kwargs;
    o.reasoning_format = cfg_.reasoning_format;
    o.n_threads       = server::idle_workers(cfg_.http_threads);
    if (cfg_.http_threads <= 0)
        decide("--http-threads", std::to_string(o.n_threads),
               "one a hardware thread, at least 4");
    o.read_timeout_s  = cfg_.http_read_timeout_s;
    o.write_timeout_s = cfg_.http_write_timeout_s;
    o.keep_alive_timeout_s = cfg_.http_keep_alive_timeout_s;
    o.max_body_bytes  = (size_t)cfg_.http_max_body_mib << 20;
    o.retry_after_s   = cfg_.retry_after_s;
    o.cors            = cfg_.cors;
    {
        const std::string moved = request_overrides(cfg_);
        if (!moved.empty()) RAD_INFO("request defaults and bounds from the command line: %s",
                                     moved.c_str());
    }
    o.engine_info = engine_info();
    /* Per request, since OpInfo::issued changes once the run phase runs
     * (ServerOptions::graph_json). */
    o.graph_json  = [this] { return dump_graph_json(ranks_[0]->program, reg_.get()); };
    /* Per request, from a reading at most a second old (DeviceRuntime::diagnostics_json). */
    o.diagnostics_json = [this] { return diagnostics_json(); };
    if (cfg_.lora.enabled() && ranks_[0]->program.is_pass_driven()) {
        /* A generator's request names its adapters in `loras`; `model` stays the model. */
        o.gen_lora_acquire = [this](const std::vector<std::pair<std::string, double>>& u,
                                    int32_t* slot, std::shared_ptr<void>* hold, std::string* note,
                                    std::string* e) {
            std::vector<LoraUse> uses;
            for (const auto& [n, w] : u) uses.push_back({ n, w });
            return gen_lora_acquire(uses, slot, hold, note, e);
        };
        o.gen_lora_names = [this] { return gen_lora_names(); };
        o.lora_load = [this](const std::string& n, const std::string& d, std::string* e) {
            return lora_load(n, d, e);
        };
        o.lora_unload = [this](const std::string& n, std::string* e, int* in_flight) {
            return lora_unload(n, e, in_flight);
        };
    } else if (cfg_.lora.enabled()) {
        /* `model` picks the adapter, and the served model's own name picks the base. */
        for (const auto& [name, dir] : cfg_.lora.adapters)
            if (name == o.model_id) {
                RAD_ERR("lora: the adapter '%s' has the served model's own name, so a request "
                        "naming it would run the base; rename it or give --served-model-name",
                        name.c_str());
                return RAD_E_INVAL;
            }
        o.lora_acquire = [this](const std::string& m, int32_t* slot, uint64_t* uid,
                                std::shared_ptr<void>* hold) {
            return lora_acquire(m, slot, uid, hold);
        };
        o.lora_names = [this] { return lora_names(); };
        o.lora_load = [this](const std::string& n, const std::string& d, std::string* e) {
            return lora_load(n, d, e);
        };
        o.lora_unload = [this](const std::string& n, std::string* e, int* in_flight) {
            return lora_unload(n, e, in_flight);
        };
    }

    /* Media, when the program declared an encoder. The processor's geometry comes from the
     * container and the encoder states its own; patch width and merge must agree, checked here
     * rather than found as garbage rows. */
    std::unique_ptr<mm::Processor> mm_proc;
    std::unique_ptr<server::MultimodalBridge> mm_bridge;
    /* How the model hears, for the speech routes: the plugin's RadAudioFormat copied out. */
    server::SpeechFormat speech_fmt;
    const RadEncoderDecl& encd = ranks_[0]->program.encoder;
    if (encd.modalities != 0) {
        mm::VisionConfig vc;
        std::string why;
        if (vc.from_meta(meta_, &why) < 0) {
            RAD_ERR("the '%s' encoder has no processor: %s", ranks_[0]->program.encoder_name.c_str(),
                    why.c_str());
            return RAD_E_INVAL;
        }
        vc.max_patches = encd.max_patches;
        /* Flags override the container field by field, and the result is checked as a whole (a
         * minimum given alone can exceed the container's maximum). Flags a processor would ignore
         * are refused by name: LFM2-VL takes pictures only, and a token-budget processor (Gemma
         * 4's) sizes by soft tokens (--image-tokens, --video-max-frame-tokens), not pixel
         * bands. */
        if ((vc.family == mm::Family::Lfm2 || vc.family == mm::Family::Glm5Next) &&
            (cfg_.video_min_pixels || cfg_.video_max_pixels || cfg_.video_min_frames ||
             cfg_.video_max_frames || cfg_.video_fps || cfg_.video_max_frame_tokens)) {
            RAD_ERR("media: this model's processor takes pictures, not video, so the --video-* "
                    "flags do nothing on it");
            return RAD_E_INVAL;
        }
        if (vc.family == mm::Family::Gemma4) {
            if (cfg_.image_min_pixels || cfg_.image_max_pixels || cfg_.video_min_pixels ||
                cfg_.video_max_pixels || cfg_.video_min_frames) {
                RAD_ERR("media: this model's processor sizes images and video frames by a token "
                        "budget, so --image-min/max-pixels, --video-min/max-pixels and "
                        "--video-min-frames do nothing; use --image-tokens and "
                        "--video-max-frame-tokens");
                return RAD_E_INVAL;
            }
            if (cfg_.image_tokens)           vc.image_soft_tokens = *cfg_.image_tokens;
            if (cfg_.video_max_frame_tokens) vc.video_soft_tokens = *cfg_.video_max_frame_tokens;
        } else if (vc.family == mm::Family::Glm5Next) {
            /* GLM-5.3 budgets an image in tokens (max_image_tokens), so --image-tokens is its
             * ceiling and the pixel bands do not apply. */
            if (cfg_.image_min_pixels || cfg_.image_max_pixels) {
                RAD_ERR("media: this model's processor sizes images by a token budget, so "
                        "--image-min-pixels and --image-max-pixels do nothing; use "
                        "--image-tokens");
                return RAD_E_INVAL;
            }
            if (cfg_.image_tokens) vc.max_image_tokens = *cfg_.image_tokens;
        } else if (vc.family == mm::Family::Radio) {
            /* NVIDIA's processor sizes a picture and a video frame by patch budgets
             * (min_num_patches..max_num_patches, video_target_num_patches), not pixel bands. */
            if (cfg_.image_min_pixels || cfg_.image_max_pixels || cfg_.image_tokens ||
                cfg_.video_min_pixels || cfg_.video_max_pixels || cfg_.video_max_frame_tokens) {
                RAD_ERR("media: this model's processor sizes images and video frames by its patch "
                        "budgets (min_num_patches, max_num_patches, video_target_num_patches), so "
                        "--image-min/max-pixels, --image-tokens, --video-min/max-pixels and "
                        "--video-max-frame-tokens do nothing on it");
                return RAD_E_INVAL;
            }
            if (cfg_.video_min_frames) vc.min_frames = *cfg_.video_min_frames;
        } else {
            if (cfg_.image_tokens) {
                RAD_ERR("media: --image-tokens is a token budget, and this model's processor "
                        "sizes images by pixels (--image-min-pixels, --image-max-pixels)");
                return RAD_E_INVAL;
            }
            if (cfg_.image_min_pixels)       vc.image_min_pixels  = *cfg_.image_min_pixels;
            if (cfg_.image_max_pixels)       vc.image_max_pixels  = *cfg_.image_max_pixels;
            if (cfg_.video_min_pixels)       vc.video_min_pixels  = *cfg_.video_min_pixels;
            if (cfg_.video_max_pixels)       vc.video_max_pixels  = *cfg_.video_max_pixels;
            if (cfg_.video_min_frames)       vc.min_frames        = *cfg_.video_min_frames;
            if (cfg_.video_max_frame_tokens) vc.max_frame_rows    = *cfg_.video_max_frame_tokens;
        }
        if (cfg_.video_fps)              vc.fps               = *cfg_.video_fps;
        if (cfg_.video_max_frames)       vc.max_frames        = *cfg_.video_max_frames;
        if (cfg_.max_source_pixels)      vc.max_source_pixels = *cfg_.max_source_pixels;
        if (cfg_.media_flags_set() && vc.check(&why) < 0) {
            RAD_ERR("media: the media flags and what the container states disagree: %s",
                    why.c_str());
            return RAD_E_INVAL;
        }
        /* The processor caps the image band at one encoder pass; warn if more was asked for. */
        const int64_t pass_px = encd.max_patches * vc.patch * vc.patch;
        if (cfg_.image_max_pixels && *cfg_.image_max_pixels > pass_px)
            RAD_WARN("media: --image-max-pixels %lld is above what one encoder pass takes (%lld "
                     "pixels at --mm-max-patches %lld); images are resized to at most that",
                     (long long)*cfg_.image_max_pixels, (long long)pass_px,
                     (long long)encd.max_patches);
        if (vc.patch_dim() != encd.patch_dim || (int64_t)vc.merge * vc.merge != encd.merge) {
            RAD_ERR("the container's processor cuts %lld-element patches merged %lld to a row; the "
                    "'%s' encoder takes %lld-element patches merged %lld", (long long)vc.patch_dim(),
                    (long long)vc.merge * vc.merge, ranks_[0]->program.encoder_name.c_str(),
                    (long long)encd.patch_dim, (long long)encd.merge);
            return RAD_E_INVAL;
        }
        /* An explicit budget past one pass is refused; the container's is lowered to fit
         * (Processor::init), as a pixel band is. */
        if (vc.family == mm::Family::Glm5Next && cfg_.image_tokens &&
            (int64_t)*cfg_.image_tokens * vc.merge * vc.merge > encd.max_patches) {
            RAD_ERR("media: --image-tokens %d is more patches than one encoder pass holds "
                    "(--mm-max-patches %lld, %lld tokens)", *cfg_.image_tokens,
                    (long long)encd.max_patches,
                    (long long)(encd.max_patches / ((int64_t)vc.merge * vc.merge)));
            return RAD_E_INVAL;
        }
        if (vc.family == mm::Family::Gemma4 &&
            (int64_t)std::max(vc.image_soft_tokens, vc.video_soft_tokens) * vc.merge * vc.merge >
                encd.max_patches) {
            RAD_ERR("media: an image of %d soft tokens or a video frame of %d is more patches than "
                    "one encoder pass holds (--mm-max-patches %lld)", vc.image_soft_tokens,
                    vc.video_soft_tokens, (long long)encd.max_patches);
            return RAD_E_INVAL;
        }
        mm_proc = std::make_unique<mm::Processor>();
        const int pi = mm_proc->init(
            vc,
            [&tok_bridge](const std::string& t) {
                return tok_bridge.encode(t, /*add_special=*/false, /*parse_special=*/false);
            },
            [this](const std::string& t) { return vocab_->find(t); });
        if (pi < 0) {
            RAD_ERR("media: the container's processor configuration does not fit this deployment "
                    "or this vocabulary (a marker it names is missing, or a segment is larger "
                    "than one encoder pass): %s", rad_strerror(pi));
            return pi;
        }
        o.allow_image = (encd.modalities & RAD_MM_IMAGE) != 0;
        o.allow_video = (encd.modalities & RAD_MM_VIDEO) != 0;
        if (!mm::decoder_available())
            RAD_WARN("media: the '%s' encoder is loaded but this build has no media decoder "
                     "(RAD_WITH_FFMPEG), so every media part will be refused",
                     ranks_[0]->program.encoder_name.c_str());
        else if (mm_proc->config().family == mm::Family::Lfm2)
            RAD_INFO("media: images through '%s'; a picture resizes to %lld..%lld pixels, or one "
                     "past %g times that is cut into %d..%d tiles of %d pixels a side plus a "
                     "thumbnail; %lld patches a pass", ranks_[0]->program.encoder_name.c_str(),
                     (long long)mm_proc->config().image_min_pixels,
                     (long long)mm_proc->config().image_max_pixels,
                     mm_proc->config().max_pixels_tolerance, mm_proc->config().min_tiles,
                     mm_proc->config().max_tiles, mm_proc->config().tile,
                     (long long)encd.max_patches);
        else if (mm_proc->config().family == mm::Family::Glm5Next) {
            RAD_INFO("media: images through '%s'; an image becomes %d..%d tokens, padded to its "
                     "canvas rather than stretched; %lld patches a pass",
                     ranks_[0]->program.encoder_name.c_str(), mm_proc->config().min_image_tokens,
                     mm_proc->config().max_image_tokens, (long long)encd.max_patches);
            if (mm_proc->config().max_image_tokens < vc.max_image_tokens)
                RAD_INFO("media: the budget of %d tokens an image is lowered to %d, what one "
                         "encoder pass holds; --mm-max-patches %lld serves it whole",
                         vc.max_image_tokens, mm_proc->config().max_image_tokens,
                         (long long)vc.max_image_tokens * vc.merge * vc.merge);
        }
        else if (mm_proc->config().family == mm::Family::Gemma4)
            RAD_INFO("media: %s%s through '%s'; an image becomes at most %d soft tokens (a "
                     "request's image_tokens may ask for 70..1120), a video %d a frame at %.3g "
                     "frames a second, at most %d frames; %lld patches a pass",
                     o.allow_image ? "images" : "", o.allow_video ? " and video" : "",
                     ranks_[0]->program.encoder_name.c_str(), mm_proc->config().image_soft_tokens,
                     mm_proc->config().video_soft_tokens, mm_proc->config().fps,
                     mm_proc->config().max_frames, (long long)encd.max_patches);
        else if (mm_proc->config().family == mm::Family::Radio)
            RAD_INFO("media: %s%s through '%s'; an image becomes one tile of %d..%d patches "
                     "(a quarter as many tokens), a video frame about %d patches, %d frames a "
                     "tubelet at %.3g frames a second, at most %d frames, no pruning; %lld "
                     "patches a pass", o.allow_image ? "images" : "",
                     o.allow_video ? " and video" : "", ranks_[0]->program.encoder_name.c_str(),
                     mm_proc->config().radio_min_patches, mm_proc->config().radio_max_patches,
                     mm_proc->config().radio_video_patches, mm_proc->config().radio_tubelet,
                     mm_proc->config().fps, mm_proc->config().max_frames,
                     (long long)encd.max_patches);
        else
            RAD_INFO("media: %s%s through '%s'; an image resizes to at most %lld pixels (%lld "
                     "patches a pass), a video to %lld pixels over all its frames",
                     o.allow_image ? "images" : "", o.allow_video ? " and video" : "",
                     ranks_[0]->program.encoder_name.c_str(),
                     (long long)mm_proc->config().image_max_pixels, (long long)encd.max_patches,
                     (long long)mm_proc->config().video_max_pixels);
    }
    /* Audio, when the program declared an audio encoder: framing comes from the container; the
     * encoder states the frame width and frames per output row. */
    const RadEncoderDecl& aend = ranks_[0]->program.audio_encoder;
    if (aend.modalities != 0) {
        mm::AudioConfig ac;
        std::string why;
        if (ac.from_meta(meta_, &why) < 0) {
            RAD_ERR("the '%s' encoder has no feature extractor: %s",
                    ranks_[0]->program.audio_encoder_name.c_str(), why.c_str());
            return RAD_E_INVAL;
        }
        if (ac.n_mels != aend.patch_dim) {
            RAD_ERR("the container's feature extractor makes %d-bin frames; the '%s' encoder takes "
                    "%lld", ac.n_mels, ranks_[0]->program.audio_encoder_name.c_str(),
                    (long long)aend.patch_dim);
            return RAD_E_INVAL;
        }
        /* The tokens around a clip: the config's ids where it names them (Gemma 4's boa/eoa),
         * else the spellings the architecture plugin states (RadAudioFormat). Each must be one
         * token of this vocabulary, or the prompt would carry text where the model was trained on
         * a marker. */
        const RadAudioFormat* afmt = rad_tools_audio_format(plugins_, meta_);
        auto one_token = [&](const char* spell, int32_t* id) -> int {
            if (*id >= 0 || !spell || !*spell) return RAD_OK;
            const std::vector<int32_t> t = tok_bridge.encode(spell, /*add_special=*/false,
                                                             /*parse_special=*/true);
            if (t.size() != 1) {
                RAD_ERR("the architecture plugin spells a clip's marker '%s', which this "
                        "vocabulary reads as %zu tokens rather than one", spell, t.size());
                return RAD_E_INVAL;
            }
            *id = t[0];
            return RAD_OK;
        };
        if (afmt) {
            RAD_TRY(one_token(afmt->audio_open, &ac.tok_start));
            RAD_TRY(one_token(afmt->audio_close, &ac.tok_end));
            auto str = [](const char* p) { return std::string(p ? p : ""); };
            speech_fmt.transcribe_prompt = str(afmt->transcribe_prompt);
            speech_fmt.language_prompt   = str(afmt->language_prompt);
            speech_fmt.transcript_open   = str(afmt->transcript_open);
            speech_fmt.language_lead     = str(afmt->language_lead);
            for (uint32_t i = 0; afmt->languages && i < afmt->n_languages; ++i)
                speech_fmt.languages.emplace_back(str(afmt->languages[2 * i]),
                                                  str(afmt->languages[2 * i + 1]));
        }
        if (ac.tok_start < 0 || ac.tok_end < 0) {
            RAD_ERR("the '%s' encoder's clips have no opening or closing token: the container "
                    "names none and the architecture plugin states none (rad_arch_audio_format)",
                    ranks_[0]->program.audio_encoder_name.c_str());
            return RAD_E_INVAL;
        }
        ac.merge = (int32_t)aend.merge;
        ac.max_frames = aend.max_patches;
        if (!mm_proc) mm_proc = std::make_unique<mm::Processor>();
        if (const int rc = mm_proc->init_audio(ac, &why); rc < 0) {
            RAD_ERR("the '%s' encoder's audio front end cannot be built: %s",
                    ranks_[0]->program.audio_encoder_name.c_str(), why.c_str());
            return rc;
        }
        o.allow_audio = true;
        if (mm::decoder_available() && ac.family == mm::AudioFamily::Gemma4Unified)
            RAD_INFO("media: audio through '%s'; a clip is cut at %.3g s, a token is a frame of %d "
                     "samples (%d ms); %lld frames a pass",
                     ranks_[0]->program.audio_encoder_name.c_str(),
                     (double)mm_proc->audio_config().max_samples / (double)ac.rate, ac.frame_length,
                     ac.frame_length * 1000 / ac.rate, (long long)aend.max_patches);
        else if (mm::decoder_available())
            RAD_INFO("media: audio through '%s'; a clip is cut at %.3g s, %d mel bins a %d ms frame, "
                     "%d frames a token; %lld frames a pass",
                     ranks_[0]->program.audio_encoder_name.c_str(),
                     (double)mm_proc->audio_config().max_samples / (double)ac.rate, ac.n_mels,
                     ac.hop * 1000 / ac.rate, ac.merge, (long long)aend.max_patches);
    }
    if (mm_proc) {
        mm_bridge = std::make_unique<server::MultimodalBridge>(
            mm_proc.get(), encd.modalities | aend.modalities);
        d.mm = mm_bridge.get();
        if (speech_fmt.transcribes() && o.allow_audio) {
            d.speech = &speech_fmt;
            RAD_INFO("speech: /v1/audio/transcriptions through '%s', %zu languages",
                     ranks_[0]->program.audio_encoder_name.c_str(), speech_fmt.languages.size());
        }
    }

    /* An image generator's endpoints: the driver's queue, the declared prompts and sizes, and the
     * processor that prepares a reference picture for the conditioning network. */
    struct ImageBridge final : server::IImageGen {
        gen::Driver*         drv = nullptr;
        const mm::Processor* proc = nullptr;
        server::ImageGenInfo inf;
        Tokenizer            aux;
        bool                 has_aux = false;
        const server::ImageGenInfo& info() const override { return inf; }
        int encode_aux(const std::string& text, std::vector<int32_t>* out) const override {
            if (!has_aux) return RAD_E_UNSUPPORTED;
            out->clear();
            return aux.encode(text, *out, /*add_special=*/true, /*parse_special=*/true);
        }
        int submit(std::unique_ptr<gen::Request> r, std::string* why) override {
            return drv->submit(std::move(r), why);
        }
        int picture(const mm::Frame& f, std::shared_ptr<mm::Item>* out, std::string* why) override {
            if (!proc) {
                if (why) *why = "this model reads no pictures";
                return RAD_E_UNSUPPORTED;
            }
            return proc->image_from(f, out, why);
        }
        int64_t queued() const override { return drv->queued(); }
        int64_t running() const override { return drv->running(); }
    } image_bridge;
    if (!img_gen_ && (cfg_.image_gen_max_pixels > 0 || cfg_.image_gen_max_refs > 0))
        RAD_WARN("--image-gen-max-pixels / --image-gen-max-refs: this model makes no images; "
                 "ignored");
    if (img_gen_) {
        const Program& pg = ranks_[0]->program;
        const RadGeneratorDecl& gd = pg.generator;
        server::ImageGenInfo& in = image_bridge.inf;
        /* The bounds are the plugin's to apply: one that keeps fixed buffers is named. */
        const int64_t px = gd.max_tokens * gd.latent_scale * gd.latent_scale;
        if (cfg_.image_gen_max_pixels > 0 && px > cfg_.image_gen_max_pixels)
            RAD_WARN("--image-gen-max-pixels %lld: '%s' sizes its buffers for %lld pixels and does "
                     "not take the bound", (long long)cfg_.image_gen_max_pixels,
                     pg.gen_str.name.c_str(), (long long)px);
        if (cfg_.image_gen_max_refs > 0 && gd.max_refs > cfg_.image_gen_max_refs)
            RAD_WARN("--image-gen-max-refs %lld: '%s' takes %lld reference pictures and does not "
                     "take the bound", (long long)cfg_.image_gen_max_refs, pg.gen_str.name.c_str(),
                     (long long)gd.max_refs);
        in.name = pg.gen_str.name;
        in.tmpl_text = pg.gen_str.tmpl_text;
        in.tmpl_edit = pg.gen_str.tmpl_edit;
        in.tmpl_image = pg.gen_str.tmpl_image;
        in.tmpl_image_sep = pg.gen_str.tmpl_image_sep;
        in.drop_prefix = pg.gen_str.drop_prefix;
        in.tmpl_transparent = pg.gen_str.tmpl_transparent;
        in.latent_scale = gd.latent_scale;
        in.size_multiple = gd.size_multiple;
        in.pixel_channels = gd.pixel_channels;
        in.max_tokens = gd.max_tokens;
        /* Edits need the conditioning network to read pictures, unless only the generator's
         * own encoder reads them. */
        in.refs_encoder_only = gd.layout == RAD_GEN_LAYOUT_JOINT_STEP;
        in.max_refs = mm_proc || in.refs_encoder_only ? gd.max_refs : 0;
        in.ref_shrink_only = gd.ref_shrink_only != 0;
        in.ref_area = gd.ref_area;
        in.default_width = gd.default_width;
        in.default_height = gd.default_height;
        in.default_steps = gd.default_steps;
        in.max_lanes = gd.max_lanes;
        in.max_cond_tokens = img_gen_->max_cond_tokens();
        in.fixed_schedule = !pg.gen_str.sigmas.empty();
        in.fixed_steps = (int64_t)pg.gen_str.sigmas.size();
        in.default_cfg = gd.default_cfg;
        in.cfg_original = gd.cfg_original != 0;
        in.has_default_negative = pg.gen_str.has_default_negative;
        in.default_negative = pg.gen_str.default_negative;
        in.min_tokens = gd.min_tokens;
        in.pad_token = gd.pad_token;
        in.aux_tokens = gd.aux_tokens != 0;
        in.aux_max_tokens = gd.aux_max_tokens;
        if (in.aux_tokens) {
            if (!aux_vocab_) {
                RAD_ERR("images: '%s' reads its prompt through a second tokenizer and %s carries "
                        "none; convert it with rad-convert --tokenizer2", in.name.c_str(),
                        cfg_.model.c_str());
                return RAD_E_FORMAT;
            }
            RAD_TRY(image_bridge.aux.init(aux_vocab_));
            image_bridge.has_aux = true;
        }
        in.samplers = gd.train_steps > 0 ? gd.samplers : 0;
        in.default_sampler = gd.default_sampler;
        in.default_scheduler = gd.default_scheduler;
        in.chunk_tokens = gd.chunk_tokens;
        in.chunk_bos = gd.chunk_bos;
        in.chunk_eos = gd.chunk_eos;
        in.chunk_pad = gd.pad_token;
        in.cond_opts = gd.cond_opts;
        image_bridge.drv = img_gen_;
        image_bridge.proc = mm_proc.get();
        d.images = &image_bridge;
        /* The served model id is the generator's own name (qwen-image-2.1), not the container's
         * architecture name. */
        if (cfg_.served_model_name.empty() && !in.name.empty()) {
            o.model_id = in.name;
            decide("--served-model-name", o.model_id, "an image generator: its own name");
        }
        RAD_INFO("images: '%s', %lldx%lld by default, up to %lld latent tokens, %lld picture(s) an "
                 "edit, %s", in.name.c_str(), (long long)in.default_width,
                 (long long)in.default_height, (long long)in.max_tokens, (long long)in.max_refs,
                 in.fixed_schedule ? (std::to_string(in.fixed_steps) + " fixed steps").c_str()
                                   : (std::to_string(in.default_steps) + " steps by default").c_str());
        if (in.samplers)
            RAD_INFO("images: sampler %s, scheduler %s by default", gen::sampler_name(in.default_sampler),
                     gen::scheduler_name(in.default_scheduler));
    }

    /* An audio generator's endpoint: the driver's queue, the declared prompt and limits, and the
     * plugin's rewrite of a request's texts. */
    struct AudioBridge final : server::IAudioGen {
        gen::AudioDriver*      drv = nullptr;
        const LoadedPlugins*   plugins = nullptr;
        server::AudioGenInfo   inf;
        const server::AudioGenInfo& info() const override { return inf; }
        int prepare(int field, const std::string& in, std::string* out,
                    std::string* why) const override {
            const int s = rad_tools_text_prepare(*plugins, field, in, out);
            if (s < 0 && why)
                *why = !out->empty()       ? *out
                       : s == RAD_E_INVAL ? "the model cannot take this text"
                                          : std::string(rad_strerror(s));
            return s;
        }
        int submit(std::unique_ptr<gen::AudioRequest> r, std::string* why) override {
            return drv->submit(std::move(r), why);
        }
        int64_t queued() const override { return drv->queued(); }
        int64_t running() const override { return drv->running(); }
    } audio_bridge;
    if (aud_gen_) {
        const Program& pg = ranks_[0]->program;
        const RadAudioGenDecl& ad = pg.audio_gen;
        server::AudioGenInfo& in = audio_bridge.inf;
        in.name = pg.agen_str.name;
        in.tmpl = pg.agen_str.tmpl;
        in.tmpl_plain = pg.agen_str.tmpl_plain;
        in.neg_tmpl = pg.agen_str.neg_tmpl;
        in.max_prompt = ad.max_prompt;
        in.max_frames = ad.max_frames;
        in.frame_rate_num = ad.frame_rate_num;
        in.frame_rate_den = ad.frame_rate_den;
        in.sample_rate = ad.sample_rate;
        in.channels = ad.channels;
        in.default_steps = ad.voc_chunk > 0 ? 0 : ad.default_steps;
        in.default_cfg = ad.default_cfg;
        in.max_lanes = ad.max_lanes;
        in.branches = ad.branches == 1 ? 1 : 2;
        in.input_special = ad.input_special != 0;
        in.tmpl_ref = pg.agen_str.tmpl_ref;
        in.tmpl_ref_instruct = pg.agen_str.tmpl_ref_instruct;
        in.neg_tmpl_ref = pg.agen_str.neg_tmpl_ref;
        in.ref_item = pg.agen_str.ref_item;
        in.ref_rate = ad.ref_rate;
        in.ref_hop = ad.ref_hop;
        in.ref_max_samples = ad.ref_max_samples;
        in.ref_max = ad.ref_max;
        audio_bridge.drv = aud_gen_;
        audio_bridge.plugins = &plugins_;
        d.audio = &audio_bridge;
        if (cfg_.served_model_name.empty() && !in.name.empty()) {
            o.model_id = in.name;
            decide("--served-model-name", o.model_id, "an audio generator: its own name");
        }
        RAD_INFO("audio: '%s', %lld Hz x %lld, up to %lld frames (%lld s), %lld at once, %s",
                 in.name.c_str(), (long long)in.sample_rate, (long long)in.channels,
                 (long long)in.max_frames,
                 (long long)(in.max_frames * in.frame_rate_den / in.frame_rate_num),
                 (long long)in.max_lanes,
                 ad.voc_chunk > 0
                     ? fmt("played %lld frames at a time as they are made",
                           (long long)ad.voc_chunk).c_str()
                     : fmt("%lld steps a window by default", (long long)in.default_steps).c_str());
    }

    /* A video generator's endpoints: the driver's queue and the declared prompt, sizes and
     * lengths. */
    struct ShapeBridge final : server::IShapeGen {
        gen::ShapeDriver* drv = nullptr;
        const gen::ShapeGenInfo& info() const override { return drv->info(); }
        int submit(std::unique_ptr<gen::ShapeRequest> r, std::string* why) override {
            return drv->submit(std::move(r), why);
        }
        int64_t queued() const override { return drv->queued(); }
        int64_t running() const override { return drv->running(); }
    } shape_bridge;
    if (shp_gen_) {
        shape_bridge.drv = shp_gen_;
        d.shapes = &shape_bridge;
        const gen::ShapeGenInfo& in = shp_gen_->info();
        if (cfg_.served_model_name.empty() && !in.name.empty()) {
            o.model_id = in.name;
            decide("--served-model-name", o.model_id, "a shape generator: its own name");
        }
        RAD_INFO("shape: '%s', a %lldx%lld picture to a field sampled at %lld cells a side by "
                 "default (%lld..%lld), %lld steps by default at guidance %g",
                 in.name.c_str(), (long long)in.canvas, (long long)in.canvas,
                 (long long)in.default_resolution, (long long)in.min_resolution,
                 (long long)in.max_resolution, (long long)in.default_steps, in.default_guidance);
    }
    struct VideoBridge final : server::IVideoGen {
        gen::VideoDriver*    drv = nullptr;
        server::VideoGenInfo inf;
        const server::VideoGenInfo& info() const override { return inf; }
        int submit(std::unique_ptr<gen::VideoRequest> r, std::string* why) override {
            return drv->submit(std::move(r), why);
        }
        int64_t queued() const override { return drv->queued(); }
        int64_t running() const override { return drv->running(); }
    } video_bridge;
    if (vid_gen_) {
        const Program& pg = ranks_[0]->program;
        const RadVideoGenDecl& vd = pg.video_gen;
        server::VideoGenInfo& in = video_bridge.inf;
        in.name = pg.vgen_str.name;
        in.tmpl = pg.vgen_str.tmpl;
        in.max_text = vd.max_text;
        in.fps_num = vd.fps_num;
        in.fps_den = vd.fps_den;
        in.frames_per_chunk = vd.frames_per_chunk;
        in.frames_tail = vd.frames_tail;
        in.min_chunks = vd.min_chunks;
        in.max_chunks = vd.max_chunks;
        in.size_multiple = vd.size_multiple;
        in.min_side = vd.min_side;
        in.max_area = vd.max_area;
        in.max_aspect = vd.max_aspect;
        in.default_width = vd.default_width;
        in.default_height = vd.default_height;
        in.default_chunks = vd.default_chunks;
        in.sample_rate = vd.sample_rate;
        in.channels = vd.audio_channels;
        in.default_steps = vd.default_steps;
        in.max_steps = vd.max_steps;
        in.fixed_steps = vd.n_rungs > 0;
        in.shift_video = vd.shift_video;
        in.shift_audio = vd.shift_audio;
        in.upscale_max = vd.upscale_max;
        in.upscale_refine = vd.upscale_refine;
        in.step_evals = vd.sched_kind == 1;
        in.guidance = vd.guidance != 0;
        for (int i = 0; i < 8; ++i) in.guidance_default[i] = vd.guidance_default[i];
        in.default_negative = pg.vgen_str.default_negative;
        in.image_cond = vd.image_cond != 0;
        in.image_crf = vd.image_crf;
        in.length_head = vd.duration_out != RAD_NULL_HANDLE;
        in.decoders = pg.vgen_str.decoders;
        in.decoder_max_area = pg.vgen_str.decoder_max_area;
        if (in.image_cond && in.image_crf > 0 && !mm::h264_roundtrip_available())
            RAD_WARN("video: '%s' prepares a first frame with an H.264 round trip at crf %d, which "
                     "needs an H.264 encoder and decoder; this build's libavcodec lacks one, so "
                     "input_reference is refused", in.name.c_str(), (int)in.image_crf);
        in.max_lanes = vd.max_lanes;
        in.max_ref_images = vd.max_ref_images;
        in.max_ref_videos = vd.max_ref_videos;
        in.max_ref_audios = vd.max_ref_audios;
        in.max_enc_frames = vd.max_enc_frames;
        in.max_enc_side = vd.max_enc_side;
        in.max_enc_samples = vd.max_enc_samples;
        in.enc_ratio = vd.enc_ratio;
        video_bridge.drv = vid_gen_;
        d.videos = &video_bridge;
        if (cfg_.served_model_name.empty() && !in.name.empty()) {
            o.model_id = in.name;
            decide("--served-model-name", o.model_id, "a video generator: its own name");
        }
        const double fps = (double)in.fps_num / (double)in.fps_den;
        RAD_INFO("video: '%s', %lldx%lld by default (up to %lld pixels), %.2f to %.2f s at %g fps "
                 "with %lld Hz x %lld sound, %lld steps by default",
                 in.name.c_str(), (long long)in.default_width, (long long)in.default_height,
                 (long long)in.max_area,
                 (double)(in.frames_per_chunk * in.min_chunks + in.frames_tail) / fps,
                 (double)(in.frames_per_chunk * in.max_chunks + in.frames_tail) / fps, fps,
                 (long long)in.sample_rate, (long long)in.channels, (long long)in.default_steps);
        if (in.upscale_max > 0.0)
            RAD_INFO("video: latent upscaling up to %gx, the last %lld evaluations at the full size "
                     "by default", in.upscale_max, (long long)in.upscale_refine);
    }

    if (encd.modalities == 0 && aend.modalities == 0 && cfg_.media_flags_set())
        RAD_WARN("media: the media flags do nothing here -- this model %s",
                 cfg_.mm_max_patches == 0 ? "is served text only (--mm-max-patches 0)"
                                          : "takes no images or video");

    server::Server srv(d, o);
    srv_ = &srv;
    struct ClearSrv { server::Server** p; ~ClearSrv() { *p = nullptr; } } clear_srv{ &srv_ };
    if (cfg_.max_queued_requests <= 0)
        decide("--max-queued-requests", std::to_string(srv.queue_cap()), "8 x --max-num-seqs");

    /* A default the template refuses would 400 every chat request that relies on it. Bad
     * --chat-template-kwargs stop startup; --reasoning-effort only warns, since clients that send
     * their own effort are unaffected. */
    if (d.chat && (!cfg_.chat_template_kwargs.empty() || !cfg_.reasoning_effort.empty())) {
        std::string kw_why, eff_why;
        srv.check_template_defaults(&kw_why, &eff_why);
        if (!kw_why.empty()) {
            RAD_ERR("--chat-template-kwargs: the chat template refuses a request carrying them: %s",
                    kw_why.c_str());
            return RAD_E_INVAL;
        }
        if (!eff_why.empty())
            RAD_WARN("--reasoning-effort %s: the chat template refuses it, so every chat request "
                     "that sends no reasoning effort of its own will be answered 400: %s",
                     cfg_.reasoning_effort.c_str(), eff_why.c_str());
    }
    /* Log the efforts the template accepts (a client sending another gets this set in its 400,
     * and /server_info shows it). A map target the template refuses stops startup. */
    if (d.chat) {
        const std::vector<std::string> efforts = srv.probe_reasoning_efforts();
        std::string mapped;
        for (const auto& [from, to] : cfg_.reasoning_effort_map)
            mapped += fmt("%s%s -> %s", mapped.empty() ? "" : ", ", from.c_str(), to.c_str());
        if (!efforts.empty()) {
            std::string list;
            for (const std::string& e : efforts) list += (list.empty() ? "" : ", ") + e;
            RAD_INFO("chat: the template takes reasoning_effort %s, and none or minimal turn "
                     "thinking off%s%s", list.c_str(), mapped.empty() ? "" : "; mapped: ",
                     mapped.c_str());
            /* What a request naming no effort gets, which a client cannot see: a template may
             * render a level into a system prompt (Qwen3.8's renders xhigh). */
            const std::string& dflt = srv.template_default_effort();
            if (cfg_.reasoning_effort.empty() && !dflt.empty())
                RAD_INFO("chat: a request naming no reasoning_effort renders as '%s', the "
                         "template's own default (--reasoning-effort sets another)", dflt.c_str());
        } else if (!mapped.empty()) {
            RAD_INFO("chat: reasoning_effort mapped: %s", mapped.c_str());
        }
        if (const std::string why = srv.check_reasoning_effort_map(); !why.empty()) {
            RAD_ERR("--reasoning-effort-map %s", why.c_str());
            return RAD_E_INVAL;
        }
    }
    /* A thinking budget the model cannot hold stops startup; it would silently apply to no
     * request. */
    if (const std::string why = srv.check_thinking_budget(); !why.empty()) {
        RAD_ERR("--thinking-token-budget %lld: %s", (long long)cfg_.thinking_token_budget,
                why.c_str());
        return RAD_E_INVAL;
    }
    if (!cfg_.report_once) print_startup_report(RAD_OK);
    const char* const keyed = o.api_key.empty() ? "" : "; a bearer key is required";
    if (embedding_)
        RAD_INFO("model %s serves  (/v1/embeddings; completions and chat are 501%s)",
                 cfg_.served_model_name.c_str(), keyed);
    else if (aud_gen_)
        RAD_INFO("model %s serves  (/v1/audio/speech; completions and chat are 501%s)",
                 cfg_.served_model_name.c_str(), keyed);
    else if (vid_gen_)
        RAD_INFO("model %s serves  (/v1/videos; completions and chat are 501%s)",
                 cfg_.served_model_name.c_str(), keyed);
    else if (shp_gen_)
        RAD_INFO("model %s serves  (/generate, /send and /status/{uid}; completions and "
                 "chat are 501%s)", cfg_.served_model_name.c_str(), keyed);
    else if (gen_)
        RAD_INFO("model %s serves  (/v1/images/generations%s; completions and chat are "
                 "501%s)", cfg_.served_model_name.c_str(),
                 d.images && d.images->info().max_refs > 0 ? " and /v1/images/edits" : "", keyed);
    else
        RAD_INFO("model %s serves  (/v1/completions%s%s)", cfg_.served_model_name.c_str(),
                 d.chat ? "; /v1/chat/completions with tools and reasoning" : "; chat is 501",
                 keyed);

    /* The live view (spec §16, off by default): the console's throughput heartbeat
     * (core/server/liveview.h). Redirected, it prints one key=value line per interval with no
     * escape sequences; the renderer is chosen once by isatty(). LiveView holds no data and only
     * calls `snap`. Scoped here so its thread is joined before Deps goes out of scope. */
    std::unique_ptr<server::LiveView> live;
    if (cfg_.live_view) {
        live = std::make_unique<server::LiveView>(snap);
        live->start();
    }

    /* A model serves behind the process's front (server/front.h); the process's signals are the
     * fleet's (fleet/fleet.cpp). This model ends when the front stops or the fleet asks it to. */
    if (!front_) {
        RAD_ERR("model %s: no front to serve behind", cfg_.served_model_name.c_str());
        if (live) live->stop();
        return RAD_E_STATE;
    }
    /* Startup ends here; a quiet server from now on is an idle one. */
    startup_watch_.reset();
    srv.note_ready(startup_begin_ns_);
    front_->attach(cfg_.served_model_name, front_aliases_, &srv);
    serving_.store(true, std::memory_order_release);
    RAD_INFO("model %s: ready", cfg_.served_model_name.c_str());
    while (!front_->stopping() && !stop_requested_.load(std::memory_order_acquire)) {
        if (loop_status_.load(std::memory_order_acquire) < 0) {
            RAD_ERR("model %s: stopping: the step loop failed with %s",
                    cfg_.served_model_name.c_str(),
                    rad_strerror(loop_status_.load(std::memory_order_acquire)));
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    serving_.store(false, std::memory_order_release);
    /* The responses already handed out run to their end, for as long as the stop allows: a
     * stream holds its lease until the client has it all or leaves. Past that the server ends
     * them (Server::retire stops every stream and job), and they let go at once. */
    const int64_t drain = stop_drain_ms_.load(std::memory_order_relaxed);
    if (!front_->detach(cfg_.served_model_name, (int)std::min<int64_t>(drain, INT32_MAX))) {
        RAD_WARN("model %s: responses still open after %lld ms; ending them",
                 cfg_.served_model_name.c_str(), (long long)drain);
        srv.retire();
        /* `srv` lives in this frame and a response names it to the last, so it cannot go while one
         * is held. Past five minutes the model is wedged, and the process exits as for a GPU fault
         * so a restart brings the others back. */
        for (int min = 1; !front_->detach(cfg_.served_model_name, 60000); ++min) {
            if (min >= 5)
                fatal("model %s: responses still held %d minutes after they were ended; exiting "
                      "so a restart can bring the server back", cfg_.served_model_name.c_str(), min);
            RAD_WARN("model %s: responses still held %d minute(s) after they were ended",
                     cfg_.served_model_name.c_str(), min);
        }
    }
    srv.retire();
    if (live) live->stop();
    const int st = loop_status_.load(std::memory_order_acquire);
    return st < 0 ? st : RAD_OK;
}

}  /* namespace rad */
