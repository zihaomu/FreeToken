/* fleet/fleet.cpp -- the models of one configuration, served by one process (fleet/fleet.h). */
#include "fleet/fleet.h"
#include "fleet/broker.h"
#include "fleet/policy.h"
#include "engine.h"
#include "engine_hostmem.h"
#include "hf_model.h"
#include "rad_internal.h"
#include "server/front.h"
#include "server/history.h"
#include "util/stallwatch.h"

#include "rad_device.h"
#include "device/device.h"

#include "util/json.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <set>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <pthread.h>
#include <mutex>
#include <optional>
#include <thread>

#include <sys/statvfs.h>
#include <unistd.h>

namespace rad::fleet {

namespace {

using json = rad::Json;
constexpr int64_t kMiB = (int64_t)1 << 20;

/* SIGINT and SIGTERM are handled so every `~Engine` runs: --profile-ops prints its table, the
 * placement summary prints, and in-flight requests drain instead of being cut. SIGHUP re-reads the
 * file. The handler only stores to lock-free atomics and the scheduler thread does the work,
 * because stopping takes locks and joins threads, which is not async-signal-safe. A second SIGINT
 * or SIGTERM gets the default disposition, so the operator can still force an immediate kill. */
std::atomic<bool> g_stop{false};
std::atomic<bool> g_hup{false};
void on_stop(int) { g_stop.store(true, std::memory_order_relaxed); }
void on_hup(int) { g_hup.store(true, std::memory_order_relaxed); }

/* The listener's settings. Every model carries the same server keys (the loader writes them into
 * each model's options), so the first model's are the process's. Worker threads cover every
 * model's admission ceiling, as one server's cover its own, plus the requests that wait for a
 * model to load. */
server::TransportOptions transport_options(const FleetConfig& fc) {
    const Config& c = fc.models.front().cfg;
    server::TransportOptions t;
    t.host = c.host;
    t.port = c.port;
    t.n_threads = server::idle_workers(c.http_threads);
    t.api_key = c.api_key;
    t.cors = c.cors;
    t.read_timeout_s = c.http_read_timeout_s;
    t.write_timeout_s = c.http_write_timeout_s;
    t.keep_alive_timeout_s = c.http_keep_alive_timeout_s;
    t.max_body_bytes = (size_t)c.http_max_body_mib << 20;
    t.trusted_proxies = c.trusted_proxies;
    const int64_t spare = 64;
    int64_t depth = 0;
    for (const ModelEntry& m : fc.models)
        depth += m.cfg.max_queued_requests > 0 ? m.cfg.max_queued_requests : m.cfg.max_seqs * 8;
    t.max_threads = depth + spare;
    t.max_queued = spare;
    return t;
}

/* The bytes a model's weights take on disk: its container, or the files of a checkpoint
 * directory. The card choice's first guess at a model never loaded before. */
int64_t source_bytes(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) return (int64_t)fs::file_size(path, ec);
    int64_t n = 0;
    if (fs::is_directory(path, ec))
        for (const auto& f : fs::recursive_directory_iterator(path, ec))
            if (f.is_regular_file(ec)) n += (int64_t)f.file_size(ec);
    return n;
}

/* The bytes a model is first planned to take on each card: its last claim, the claim its
 * configuration fixes, or for a model never loaded before its share of what it reads from disk
 * and a GiB. */
int64_t first_guess(int64_t last_claim, const Config& cfg) {
    int64_t guess = last_claim;
    if (cfg.given.count("--gpu-memory-mib")) guess = cfg.gpu_memory_mib * kMiB;
    if (guess <= 0) guess = source_bytes(cfg.model) / std::max(cfg.tp, 1) + 1024 * kMiB;
    return guess;
}

std::string cards_str(const std::vector<int>& v) {
    std::string s;
    for (int c : v) s += (s.empty() ? "" : ",") + std::to_string(c);
    return s;
}

double secs_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

/* Whether any of `cards` is a card of `need`. */
bool touches(const std::vector<int>& cards, const std::map<int, int64_t>& need) {
    for (int c : cards)
        if (need.count(c)) return true;
    return false;
}

/* The models a Resident stands for: its together group's members, or itself alone. */
std::vector<size_t> unit_ids(const Resident& r) {
    return r.members.empty() ? std::vector<size_t>{ r.id } : r.members;
}

/* The least VRAM of any of `cards`, INT64_MAX when none reports it. */
int64_t smallest_card(const std::vector<int>& cards) {
    int64_t n = INT64_MAX;
    for (int c : cards) {
        RadDeviceProps p{};
        if (rad_dev_props(c, &p) == RAD_OK && p.vram_bytes > 0) n = std::min(n, p.vram_bytes);
    }
    return n;
}

/* One number from a sysfs file, -1 when the file cannot be read. */
double sysfs_number(const std::string& path) {
    std::ifstream f(path);
    double v = -1;
    return f >> v ? v : -1;
}

/* A card's temperature in millidegrees: the sensor its driver labels `junction`, else temp1. amdgpu
 * labels three, edge (temp1), junction and mem, and throttles on the junction, the hottest spot on
 * the die: under a decode load an R9700 read 56 C at the edge and 75-77 C at the junction. A driver
 * that labels none, or no junction, keeps temp1. -1 when neither can be read. */
double card_temp_mc(const std::string& hwmon) {
    for (int i = 1; i <= 8; ++i) {
        const std::string t = hwmon + "/temp" + std::to_string(i);
        std::ifstream f(t + "_label");
        std::string label;
        if (!(f >> label)) continue;
        if (label != "junction") continue;
        if (const double mc = sysfs_number(t + "_input"); mc >= 0) return mc;
        break;
    }
    return sysfs_number(hwmon + "/temp1_input");
}

/* What the driver reports of a card right now: busy share, power and temperature, on the routes of
 * a dashboard being watched and nowhere on a request's path. A card that reports none adds none. */
void card_telemetry(int card, json& o) {
    namespace fs = std::filesystem;
    const std::string bdf = dev_pci_address(card);
    if (bdf.empty()) return;
    const std::string dev = "/sys/bus/pci/devices/" + bdf;
    if (const double busy = sysfs_number(dev + "/gpu_busy_percent"); busy >= 0) o["busy"] = busy / 100.0;
    std::error_code ec;
    for (const auto& h : fs::directory_iterator(dev + "/hwmon", ec)) {
        const std::string hp = h.path().string();
        double uw = sysfs_number(hp + "/power1_average");
        if (uw < 0) uw = sysfs_number(hp + "/power1_input");
        if (uw >= 0) o["power_w"] = uw / 1e6;
        if (const double mc = card_temp_mc(hp); mc >= 0) o["temp_c"] = mc / 1000.0;
        break;
    }
}

/* A fleet's hardware queues are opened while their card has room, never on a full one.
 *
 * A new queue's context save area is a range of shared virtual memory that the driver prefetches
 * into VRAM while it holds the process's memory map lock. On a full card that prefetch evicts
 * another such range and waits for the eviction, whose worker needs the same lock: with any other
 * thread waiting to change the map, neither moves until a kernel timeout, minutes later. A fleet's
 * cards fill up, and its restores open streams, so a queue opened late is opened on a full card.
 *
 * Closing a drained stream keeps its queue for the card's next stream, so a stream opens a queue
 * only when none is kept. A card therefore holds kReserveQueues kept ones whenever room was to be
 * had: at the start, and again before every load, when the room made for the model is still free
 * (Fleet::reopen_queues). The model's own streams take from them as it starts (two a rank, a
 * third where it runs a second lane), and what is left serves the streams of a restore and its
 * fills, which live for seconds. Each queue takes about 41 MiB of VRAM for as long as it is open
 * (24 of them 0.99 GiB of a card, measured), so no more are opened ahead once a card has
 * kMaxQueuesAhead: a file with that many streams at once opens the rest as its streams ask.
 * One model never parks, so its cards see no restores and it opens what it uses. */
constexpr int64_t kQueueBytes     = 41 * kMiB;
constexpr int     kReserveQueues  = 8;
constexpr int     kMaxQueuesAhead = 24;
/* What a model that needs nearly all of its cards leaves kept (Fleet::close_idle_queues). */
constexpr int     kKeepQueues     = 4;

/* `more` queues opened on card `c` beside the `idle` it keeps; how many were. A stream takes a
 * kept queue before it opens one, so the kept ones are held while the new ones are opened. */
int open_queues(int c, int more, int idle) {
    const int was = dev_current();
    if (more <= 0 || rad_dev_set(c) < 0) return 0;
    std::vector<RadStream> st((size_t)(more + idle), nullptr);
    size_t made = 0;
    while (made < st.size() && rad_stream_create(&st[made], 0) == RAD_OK) ++made;
    const int opened = std::max((int)made - idle, 0);
    if (made < st.size())
        RAD_WARN("fleet: card %d opened %d of %d queues ahead of its streams (%s)", c, opened, more,
                 rad_dev_last_error());
    for (size_t k = 0; k < made; ++k) rad_stream_destroy(st[k]);
    if (was >= 0) (void)rad_dev_set(was);
    return opened;
}

/* A flag raised for the life of a scope. */
struct Raised {
    std::atomic<bool>& f;
    explicit Raised(std::atomic<bool>& x) : f(x) { f.store(true); }
    ~Raised() { f.store(false); }
};

/* A count of work in flight, held for the life of a scope. */
struct InFlight {
    std::atomic<int>& n;
    explicit InFlight(std::atomic<int>& x) : n(x) { n.fetch_add(1); }
    ~InFlight() { n.fetch_sub(1); }
};

/* Set when a background write's thread returns, whichever way it returns. */
struct SetOnExit {
    std::atomic<bool>& e;
    ~SetOnExit() { e.store(true, std::memory_order_release); }
};

}  // namespace

std::string unsupported(const FleetConfig& fc) {
    std::string out;
    auto add = [&](const std::string& k) { out += (out.empty() ? "" : ", ") + k; };
    for (const ModelEntry& m : fc.models) {
        if (m.flex.kv_weight != 1.0) add("models." + m.name + ".flex.kv.weight");
        if (m.park.vram != ParkSpec::Vram::None) add("models." + m.name + ".park.vram");
    }
    return out;
}

namespace {

/* The host memory still to be had, in bytes: the host's MemAvailable, or less where a memory
 * cgroup the process is in (a container's --memory) leaves less, since its OOM killer acts on
 * the cgroup's limit whatever the host has free. A park never leaves less than kHostFloor of it,
 * whatever the budget says: pinned pages cannot be reclaimed, and a host out of them stops
 * responding. */
int64_t host_available() {
    long long kb = -1;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
        std::fclose(f);
    }
    return std::min(kb < 0 ? INT64_MAX : (int64_t)kb * 1024, cgroup_least_room());
}
constexpr int64_t kHostFloor = (int64_t)8 << 30;
/* Host room a fill from the drive leaves for the next load's pools and staging. */
constexpr int64_t kFillSlack = (int64_t)2 << 30;
/* dev_page_promote's unit: the paging chunk. */
constexpr int64_t kPromoteChunk = (int64_t)64 << 20;

class Fleet final {
public:
    explicit Fleet(FleetConfig fc);
    int run();

private:
    struct Model;

    /* The broker one load hands its engine: the fleet's grant, for that model. */
    struct LoadBroker final : VramBroker {
        Fleet* f = nullptr;
        Model* m = nullptr;
        int grant(const VramAsk& ask, int64_t* g, std::string* why) override { return f->grant(*m, ask, g, why); }
        bool host_room(int64_t bytes) override {
            if (!f->load_host_room(*m, bytes)) return false;
            m->pools_held.fetch_add(bytes);
            f->host_reserve_.fetch_add(bytes);
            return true;
        }
        bool host_pools_room() override {
            const int64_t own = m->pools_held.load();
            return f->load_host_room(*m, own, own);
        }
        int64_t host_margin() override {
            return kHostFloor + std::max<int64_t>(f->host_reserve_.load() - m->pools_held.load(), 0);
        }
        int64_t flex_kv_max_tokens() override {
            return m->entry.flex.kv_max_tokens >= 0 ? m->entry.flex.kv_max_tokens : -1;
        }
        bool    flex_grow(const std::vector<int>& c, int64_t b) override { return f->flex_grow(c, b); }
        void    flex_done() override { f->flex_done(); }
        int64_t flex_room(const std::vector<int>& c) override { return f->flex_room(c); }
        int64_t flex_experts_max_bytes() override {
            return m->entry.flex.experts_max_mib >= 0 ? m->entry.flex.experts_max_mib * kMiB : -1;
        }
    };

    /* What a model is doing, at most one thing at a time, so no other work on it starts meanwhile.
     * kEvict: it leaves or gives room (its ttl ran out, an admin park, or another worker taking
     * room from it). */
    enum Op { kIdle = 0, kLoad, kRestore, kEvict, kUnload, kFill, kYield };

    struct Model {
        ModelEntry entry;
        std::shared_ptr<Engine>            engine;
        std::unique_ptr<LoadBroker>        broker;
        std::thread                        serve;
        std::shared_ptr<std::atomic<bool>> ended;
        std::atomic<int>                   op{ kIdle };
        std::atomic<bool>                  waiting{ false };   /* in make_room_waiting */
        std::atomic<bool>                  waiting_host{ false };   /* in load_host_room's wait */
        /* Loaded by the warm-up and named by no request since (§8.6): its load takes room only
         * from free memory and other such models, and gives up quietly where there is none. */
        std::atomic<bool>                  warm{ false };
        uint64_t                           warm_uses = 0;      /* its uses (leases) when the warm-up began it */
        /* Between its engine's creation and its grant: a load waiting for the bring-up lock that
         * a granting load holds, so never something that grant may wait for. */
        std::atomic<bool>                  starting{ false };
        /* What host_reserve_ counts for this model's load in flight. */
        std::atomic<int64_t>               pools_held{ 0 };
        /* Its grant found the room held by loads still waiting for the bring-up lock it held:
         * the start ends, and the load is asked for again once they are in. */
        std::atomic<bool>                  room_retry{ false };
        int                                room_retries = 0;
        std::thread                        worker;
        uint64_t                           seq = 0;
        /* Under mu_ (HTTP threads read them). */
        std::vector<int> cards, last_cards;
        int64_t     charge = 0;          /* bytes it holds on each card when wholly on the cards */
        int64_t     claim = 0;           /* its grant, or the guess before one */
        /* A one-step model serving with weights read in place from host memory because its cards
         * had no room for them (§7.5): they are not room owed to its cards, and they move on only
         * into free memory. */
        bool        in_place = false;
        double      load_s = 0, park_s = 0, restore_s = 0;
        int64_t     filled = 0;          /* bytes read into host memory from the disk tier, this park */
        uint64_t    promote_ns = 0;      /* when a pass last tried to move its live rest onto the cards */
        double      fill_s = 0;
        /* kYield: what to free on each card, and for whose experts (Fleet::xflex_tick). */
        std::map<int, int64_t> yield_want;
        std::string yield_for;
        uint64_t    yield_ns = 0;         /* when it was last asked to yield */
        int64_t     loads = 0, unloads = 0, failures = 0, parks = 0, restores = 0;
        std::string error;
        bool        resident = false;    /* serving */
        bool        parked = false;      /* held, with part or all of it off the cards */
        std::map<int, int64_t> out;      /* parked: bytes off each card */
        uint64_t    parked_ns = 0;
        uint64_t    on_ns = 0;            /* when it last came onto its cards (load or restore) */
        /* Seconds its last page-in took: with what a park would write, a model whose way out and
         * back costs more than loading it again unloads instead of parking (Fleet::park). */
        double      trip_s = 0;
        /* memory.disk-tier: the background write of its parked host copies (dev_page_spill). */
        std::thread spill;
        std::shared_ptr<std::atomic<bool>> spill_abort, spill_done;
        std::shared_ptr<std::atomic<bool>> spill_end;   /* its thread has returned, done or not */

        /* Fresh flags for a new background write. */
        void new_spill_flags() {
            spill_abort = std::make_shared<std::atomic<bool>>(false);
            spill_done = std::make_shared<std::atomic<bool>>(false);
            spill_end = std::make_shared<std::atomic<bool>>(false);
        }
        /* Its background work (a spill, or a live restore's rest moving on) is still running. */
        bool spilling() const {
            return spill.joinable() && spill_end && !spill_end->load(std::memory_order_acquire);
        }
        /* The most of it parked off any one card. */
        int64_t most_out() const {
            int64_t o = 0;
            for (auto& [c, n] : out) o = std::max(o, n);
            return o;
        }
    };

    /* A load in flight: its claim on each card until its start returns, less what its engine has
     * allocated so far (dev_owner_bytes), so other loads and restores do not plan into it. */
    struct Reservation { const Model* m = nullptr; int owner = 0; std::vector<int> cards; int64_t claim = 0; };

    void   loop();
    Model* find(const std::string& name);
    void   demand(const std::string& name);
    void   post(std::function<void()> f);
    bool   start_op(Model& m, int op);
    void   load_op(Model& m);
    void   restore_op(Model& m);
    void   park_op(Model& m, const std::string& why);
    void   fill_op(Model& m);
    bool   quiet();
    void   warm_tick();
    void   fill_tick();
    void   promote_tick();
    /* An idle model makes way for a busy tiered model's experts (§6.7). */
    void   xflex_tick();
    void   yield_op(Model& m);
    int    grant(Model& m, const VramAsk& ask, int64_t* grant, std::string* why);
    bool   flex_grow(const std::vector<int>& cards, int64_t bytes);
    void   flex_done() { flex_landing_.fetch_sub(1, std::memory_order_acq_rel); }
    void   settle_flex();
    int64_t flex_room(const std::vector<int>& cards);

    /* Room decisions; under room_mu_. */
    int    make_room(Model& m, const std::map<int, int64_t>& need, std::string* why);
    /* make_room, retried while what stands in the way is passing (another model loading,
     * restoring or leaving on those cards, or its reserved room), up to the load timeout (or
     * `max_ms`, when shorter). Called without room_mu_ held. `commit` runs under the room lock in
     * the same hold as the make_room that found the room, so no other load can take that room
     * before the caller's reservation says it is taken. */
    int    make_room_waiting(Model& m, const std::map<int, int64_t>& need, std::string* why,
                             const std::function<void()>& commit = {}, int64_t max_ms = -1);
    /* 2: something on those cards is passing (a load, restore or park in flight, reserved room, a
     * model inside its quantum); 1: only other models waiting for room are; 0: nothing. */
    int    passing_on(const Model& m, const std::map<int, int64_t>& need);
    /* Whether `m`, on its cards and serving, is inside its quantum: the time it keeps its cards
     * against other models' demands once it came on, so a model under load serves a batch
     * instead of giving its cards back for each request of another (ping-pong). Four times what
     * coming back costs it, within [5 s, 30 s]. */
    bool   in_quantum(const Model& m, const server::Front::Activity& a, uint64_t now) const;
    bool   take_from(Model& victim, const std::map<int, int64_t>& want, const std::string& why);
    void   unload(Model& m, int64_t drain_ms, const std::string& why);
    /* 1 when `want` left, 0 when less could leave (nothing more to page out), -1 when the model
     * should unload instead (a park would cost more than its reload) or the park failed. */
    int    park(Model& m, const std::map<int, int64_t>& want, const std::string& why);
    void   reclaim_flex(const Model& m, const std::map<int, int64_t>& need);
    /* Host room for `need` bytes: kept copies go, then parked copies already on disk. With `force`
     * (a load about to pin its pools), parked copies not yet on disk are written there at once,
     * or their models unload without a disk tier. */
    /* `own` is what `need` may draw on of the reservation that counts it already. */
    bool   make_host_room(int64_t need, const Model* keep, bool force = false, bool unload_parked = false,
                          bool take_busy = false, int64_t own = 0, bool evict = true);
    /* Parked models give back their snapshot stores, in `parked` order, until `need` is free. */
    bool   release_spares(const std::vector<Model*>& parked, int64_t need);
    bool   load_host_room(Model& m, int64_t need, int64_t own = 0);
    bool   loading_beside(const Model& m);
    bool   alone_on_host(const Model& m);
    int64_t host_held();
    int64_t host_room();
    void   join_spill(Model& m);
    void   start_spill(Model& m);
    /* The chunks a live restore left in host memory move onto the cards, in the background. */
    void   start_promote(Model& m);
    void   refresh_out(Model& m);
    void   wait_pages(const std::map<int, int64_t>& before, const std::map<int, int64_t>& expect);
    void   trim_spares(const std::vector<int>& cards);
    bool   card_quiet(int card, const Model* but);
    int    close_idle_queues(const Model& m, const std::map<int, int64_t>& need);
    void   reopen_queues(const std::vector<int>& cards);

    void   fail(Model& m, const std::string& why);
    bool   choose_cards(Model& m, const Config& cfg, std::vector<int>* out, std::string* why);
    std::vector<CardRoom> rooms();
    /* The bytes a model may still claim on `card`: free past the headroom and what is pending. */
    int64_t room_on(int card);
    int64_t pending_on(int card);
    /* rooms() covers `need` on every card in it, without anything leaving. */
    bool   fits_free(const std::map<int, int64_t>& need);
    /* The least room on any of `cards`. */
    int64_t free_room(const std::vector<int>& cards);
    std::vector<Resident> residents(const Model* except);
    void   reap();
    void   ttl_tick();
    std::string reload();
    void   stop_all();
    void   add_routes();
    json   status();
    void   history_loop();
    json   placement(const std::string& name);
    std::string describe(const std::string& name);
    std::string metrics();
    std::vector<std::string> group_partners(const std::string& name, Group::Kind kind) const;
    void   keep_partners(const Model& m, std::vector<Resident>* res) const;
    std::vector<size_t> exclusive_victims(const Model& m);
    int64_t free_on(int card) const;
    bool   use_broker() const { return models_.size() > 1; }

    FleetConfig       fc_;
    /* The dashboard's hour of history (server/history.h), sampled by history_loop(). */
    server::History   history_;
    const std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
    server::Front     front_;
    std::vector<int>  cards_;
    int64_t           headroom_ = 96 * kMiB;
    /* §8.6: the models still to warm, in order (under mu_; the scheduler thread changes it). */
    std::deque<std::string> warm_;
    std::string             warming_;     /* the one loading now, under mu_ */
    std::atomic<uint64_t>   fill_wait_ns_{ 0 };   /* no host fill before this (rad_mono_ns) */
    uint64_t                xflex_ns_ = 0;        /* xflex_tick's last pass */
    int64_t                 warm_unloads_ = -1;   /* unloads when the warm-up began */
    int64_t           host_budget_ = 0;    /* memory.host-mib, or 75% of the host: all pinned memory */
    std::atomic<int64_t> host_reserve_{0}; /* host memory loads in flight will pin */
    std::atomic<int>  loads_in_flight_{0};
    std::atomic<uint64_t> seq_{0};

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::recursive_mutex room_mu_;
    std::mutex          flex_mu_;
    std::atomic<int>    flex_landing_{0};   /* flex_grow grants not yet backed (flex_done) */
    std::vector<std::unique_ptr<Model>> models_;
    std::vector<std::string>            demands_;
    std::deque<std::function<void()>>   cmds_;
    std::vector<Reservation>            reserved_;   /* under mu_ */
};

Fleet::Fleet(FleetConfig fc)
    : fc_(std::move(fc)), front_(transport_options(fc_), fc_.keepalive_ms) {
    dev_stuck_exit_ms.store(fc_.stuck_exit_ms, std::memory_order_relaxed);
    bool any = false;
    int64_t head = 0;
    for (const ModelEntry& e : fc_.models) {
        models_.push_back(std::make_unique<Model>());
        models_.back()->entry = e;
        if (!e.cfg.given.count("--gpu-memory-mib")) {
            head = std::max(head, e.cfg.gpu_headroom_mib * kMiB);
            any = true;
        }
    }
    if (any) headroom_ = head;
    if (fc_.host_mib > 0) {
        host_budget_ = fc_.host_mib * kMiB;
    } else {
        const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
        /* Three quarters of the host for everything the process pins (parked copies and the
         * engines' own pinned pools); the rest is the system's. Loads read the container with
         * direct I/O, so the page cache needs little, and host_room keeps kHostFloor of
         * MemAvailable free whatever the budget says. In a memory cgroup with less room than the
         * host has (a container's --memory), three quarters of that room. */
        const int64_t host = pages > 0 && page > 0 ? (int64_t)pages * page : 0;
        host_budget_ = std::min(host, cgroup_least_room()) / 4 * 3;
    }
}

Fleet::Model* Fleet::find(const std::string& name) {
    for (auto& m : models_)
        if (m->entry.name == name) return m.get();
    return nullptr;
}

void Fleet::demand(const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    if (std::find(demands_.begin(), demands_.end(), name) == demands_.end()) demands_.push_back(name);
    cv_.notify_all();
}

void Fleet::post(std::function<void()> f) {
    std::lock_guard<std::mutex> lk(mu_);
    cmds_.push_back(std::move(f));
    cv_.notify_all();
}

int64_t Fleet::free_on(int card) const {
    RadDeviceProps p{};
    if (rad_dev_props(card, &p) < 0) return 0;
    return p.vram_free;
}

int64_t Fleet::pending_on(int card) {
    int64_t n = 0;
    std::lock_guard<std::mutex> lk(mu_);
    for (const Reservation& r : reserved_)
        if (std::find(r.cards.begin(), r.cards.end(), card) != r.cards.end())
            n += std::max<int64_t>(r.claim - (r.owner ? dev_owner_bytes(r.owner, card) : 0), 0);
    /* A serving model's chunks still read from host memory are on their way back to this card,
     * unless they stay there until free memory takes them (in place, §7.5). */
    for (const auto& m : models_)
        if (m->engine && m->resident && !m->parked && !m->in_place)
            n += dev_page_live_bytes(m->engine->owner(), card);
    return n;
}

int64_t Fleet::room_on(int card) {
    return std::max<int64_t>(free_on(card) - headroom_ - pending_on(card), 0);
}

std::vector<CardRoom> Fleet::rooms() {
    std::vector<CardRoom> v;
    for (int c : cards_) v.push_back({ c, room_on(c) });
    return v;
}

bool Fleet::fits_free(const std::map<int, int64_t>& need) {
    for (const CardRoom& r : rooms())
        if (const auto it = need.find(r.card); it != need.end() && it->second > r.free) return false;
    return true;
}

int64_t Fleet::free_room(const std::vector<int>& cards) {
    int64_t n = INT64_MAX;
    for (const CardRoom& r : rooms())
        if (std::find(cards.begin(), cards.end(), r.card) != cards.end()) n = std::min(n, r.free);
    return n == INT64_MAX ? 0 : n;
}

std::vector<Resident> Fleet::residents(const Model* except) {
    std::vector<Resident> v;
    const uint64_t now = rad_mono_ns();
    /* The Front's activity is read without mu_ held: the Front calls demand() under its own lock,
     * so taking the two the other way round could deadlock. */
    std::vector<server::Front::Activity> act(models_.size());
    for (size_t i = 0; i < models_.size(); ++i) act[i] = front_.activity(models_[i]->entry.name);
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < models_.size(); ++i) {
        const Model& m = *models_[i];
        if (&m == except || !m.engine || (!m.resident && !m.parked)) continue;
        if (m.op.load() != kIdle) continue;   /* busy with work of its own */
        const server::Front::Activity& a = act[i];
        Resident r;
        r.id = i;
        r.cards = m.cards;
        /* A parked model gives only what of it is still pageable on each card: the rest is small
         * allocations that leave only with the whole model (make_room's last resort). */
        const std::map<int, int64_t> mapped =
            m.parked ? dev_page_mapped_bytes(m.engine->owner()) : std::map<int, int64_t>{};
        for (int c : m.cards) {
            const auto it = mapped.find(c);
            /* Weights read in place are not on the card, and not owed to it. */
            const int64_t away = m.in_place ? dev_page_live_bytes(m.engine->owner(), c) : 0;
            r.charges.push_back(m.parked ? (it == mapped.end() ? 0 : it->second)
                                         : m.charge + m.engine->flex_bytes() - away);
        }
        r.charge = r.charges.empty() ? 0 : r.charges[0];
        r.parked = m.parked;
        r.priority = m.entry.priority;
        r.pin = m.entry.pin;
        r.busy = a.leases > 0 || a.waiting > 0;
        r.keep = in_quantum(m, a, now);
        r.idle_s = a.last_used_ns && now > a.last_used_ns ? (double)(now - a.last_used_ns) * 1e-9 : 0;
        r.restore_s = m.parked ? std::max(m.restore_s, 0.1) : (m.load_s > 0 ? m.load_s : 5.0);
        v.push_back(r);
    }
    /* A together group leaves as one (docs/MULTIMODEL.md §8.5). */
    for (const Group& g : fc_.groups) {
        if (g.kind != Group::Kind::Together) continue;
        std::vector<size_t> at;
        for (size_t i = 0; i < v.size(); ++i)
            if (std::find(g.members.begin(), g.members.end(), models_[v[i].id]->entry.name) !=
                g.members.end())
                at.push_back(i);
        if (at.size() < 2) continue;
        Resident u = v[at[0]];
        u.members.clear();
        u.charges.clear();
        u.cards.clear();
        std::map<int, int64_t> per;
        for (size_t i : at) {
            const Resident& r = v[i];
            const std::vector<size_t> ids = unit_ids(r);
            u.members.insert(u.members.end(), ids.begin(), ids.end());
            for (int c : r.cards) per[c] += r.gives(c);
            u.priority = std::max(u.priority, r.priority);
            u.pin = u.pin || r.pin;
            u.busy = u.busy || r.busy;
            u.parked = u.parked && r.parked;
            u.idle_s = std::min(u.idle_s, r.idle_s);
            if (i != at[0]) u.restore_s += r.restore_s;
        }
        for (auto& [c, b] : per) { u.cards.push_back(c); u.charges.push_back(b); }
        for (size_t k = at.size(); k-- > 0;) v.erase(v.begin() + (long)at[k]);
        v.push_back(u);
    }
    return v;
}

void Fleet::keep_partners(const Model& m, std::vector<Resident>* res) const {
    const std::vector<std::string> partners = group_partners(m.entry.name, Group::Kind::Together);
    for (Resident& r : *res) {
        for (size_t id : unit_ids(r))
            if (std::find(partners.begin(), partners.end(), models_[id]->entry.name) != partners.end())
                r.keep = true;
    }
}

std::vector<std::string> Fleet::group_partners(const std::string& name, Group::Kind kind) const {
    std::vector<std::string> out;
    for (const Group& g : fc_.groups) {
        if (g.kind != kind) continue;
        if (std::find(g.members.begin(), g.members.end(), name) == g.members.end()) continue;
        for (const std::string& o : g.members)
            if (o != name && std::find(out.begin(), out.end(), o) == out.end()) out.push_back(o);
    }
    return out;
}

std::vector<size_t> Fleet::exclusive_victims(const Model& m) {
    std::vector<size_t> forced;
    std::lock_guard<std::mutex> lk(mu_);
    for (const std::string& p : group_partners(m.entry.name, Group::Kind::Exclusive))
        for (size_t i = 0; i < models_.size(); ++i)
            if (models_[i]->entry.name == p && models_[i]->engine && !models_[i]->parked)
                forced.push_back(i);
    return forced;
}

bool Fleet::choose_cards(Model& m, const Config& cfg, std::vector<int>* out, std::string* why) {
    const int tp = std::max(cfg.tp, 1);
    if (tp > (int)cards_.size()) {
        *why = "tp " + std::to_string(tp) + " needs " + std::to_string(tp) + " cards; " +
               std::to_string(cards_.size()) + " are available";
        return false;
    }
    if (!use_broker()) {
        out->assign(cards_.begin(), cards_.begin() + tp);
        return true;
    }
    const int64_t guess = first_guess(m.claim, cfg);
    std::vector<Resident> res = residents(&m);
    keep_partners(m, &res);
    RoomPlan plan;
    std::string w;
    if (plan_room(rooms(), res, tp, guess, m.entry.priority, exclusive_victims(m), m.last_cards,
                  nullptr, &plan, &w)) {
        *out = plan.cards;
        return true;
    }
    std::vector<std::pair<int64_t, int>> by;
    for (const CardRoom& c : rooms()) {
        int64_t room = c.free;
        for (const Resident& r : res)
            if (!r.pin && !r.keep && r.priority <= m.entry.priority) room += r.gives(c.card);
        by.push_back({ -room, c.card });
    }
    std::sort(by.begin(), by.end());
    out->clear();
    for (int i = 0; i < tp; ++i) out->push_back(by[(size_t)i].second);
    std::sort(out->begin(), out->end());
    return true;
}

/* Memory a cache or an expert slab may grow into: what the card has free past the headroom, the
 * loads in flight and a margin for what engines allocate on first use. Nothing while a model
 * loads, comes back or waits for room. */
int64_t Fleet::flex_room(const std::vector<int>& cards) {
    if (loads_in_flight_.load(std::memory_order_acquire) > 0) return 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& m : models_)
            if (m->waiting.load(std::memory_order_acquire)) return 0;
    }
    const int64_t margin = headroom_ + 512 * kMiB;
    int64_t room = INT64_MAX;
    for (int c : cards) room = std::min(room, free_on(c) - margin - pending_on(c));
    return room == INT64_MAX ? 0 : std::max<int64_t>(room, 0);
}

bool Fleet::flex_grow(const std::vector<int>& cards, int64_t bytes) {
    std::lock_guard<std::mutex> lk(flex_mu_);
    if (bytes <= 0 || flex_room(cards) < bytes) return false;
    flex_landing_.fetch_add(1, std::memory_order_acq_rel);
    return true;
}

/* A load or restore, its in-flight count raised (flex_room refuses from then on): growth granted
 * before that is still being backed, and the free memory the room plan reads does not show it yet.
 * The lock orders this after any grant under way; a grant after it sees the count and refuses. */
void Fleet::settle_flex() {
    { std::lock_guard<std::mutex> lk(flex_mu_); }
    const auto t0 = std::chrono::steady_clock::now();
    while (flex_landing_.load(std::memory_order_acquire) > 0 && secs_since(t0) < 2.0)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

/* ------------------------------------------------------------------ room */

void Fleet::wait_pages(const std::map<int, int64_t>& before, const std::map<int, int64_t>& expect) {
    /* The driver returns a freed buffer's pages once its fences have signalled, on a worker of
     * its own, so the card's free figure trails the frees by up to a second or so. The next
     * decision reads that figure, so wait for nine tenths of what each card gave (at most two
     * seconds). Only cards that gave a paging chunk or more are waited on: another model's
     * allocation lowers the free figure of a card this one gave nothing on, and that never comes
     * back, and the few MiB a fully parked model still held (16 MiB of Hunyuan3D-2's 5 GiB) are
     * within what the others move the figure by, so waiting for them ran out the two seconds. */
    constexpr int64_t kWorthWaiting = (int64_t)64 << 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        bool back = true;
        for (auto& [c, e] : expect) {
            const auto it = before.find(c);
            if (e >= kWorthWaiting && it != before.end() && free_on(c) < it->second + e * 9 / 10)
                back = false;
        }
        if (back || secs_since(t0) > 2.0) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

/* The spare device memory the backend kept from parks on `cards` (it counts as free) back to the
 * driver, and the free figure waited for as after a park. Before a fresh load, whose runtime
 * allocates outside the elastic calls, and after a page-in, which took what it could use. */
void Fleet::trim_spares(const std::vector<int>& cards) {
    std::map<int, int64_t> before, gave;
    for (int c : cards) {
        const int64_t f = free_on(c);
        const int64_t g = dev_vmem_trim(c);
        if (g > 0) { before[c] = f - g; gave[c] = g; }
    }
    if (!gave.empty()) wait_pages(before, gave);
}

/* Card `c` holds kReserveQueues kept queues again, as far as its free room goes; under room_mu_.
 * Before a load, whose room is made and not yet taken, and at the start. */
void Fleet::reopen_queues(const std::vector<int>& cards) {
    if (fc_.models.size() < 2) return;
    for (int c : cards) {
        int open = 0, idle = 0;
        dev_stream_counts(c, &open, &idle);
        int more = std::min(kReserveQueues - idle, kMaxQueuesAhead - open);
        more = (int)std::min<int64_t>(more, room_on(c) / kQueueBytes);
        if (more <= 0) continue;
        const int n = open_queues(c, more, idle);
        if (n > 0)
            RAD_INFO("fleet: card %d: %d queue(s) opened ahead of its next streams (%d open now)",
                     c, n, open + n);
    }
}

/* Whether nothing can launch on card `c` now but model `but` (null: nothing at all): no other
 * model has an engine with a rank there, parked or not, and none is on its way anywhere. Under
 * room_mu_, which a load or a return needs before it takes room. */
bool Fleet::card_quiet(int c, const Model* but) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& o : models_) {
        if (o.get() == but) continue;
        if (o->op.load() != kIdle || o->starting.load()) return false;
        if (o->engine && (o->cards.empty() ||
                          std::find(o->cards.begin(), o->cards.end(), c) != o->cards.end()))
            return false;
    }
    return true;
}

/* The last thing `m` can be given on the cards it is short on: their kept queues, closed down to
 * kKeepQueues; the number closed. MiniMax-H3 takes 29.2 GiB of a 31.9 GiB card and its plugin
 * 1.4 GiB more, so with eight kept queues on the card and what earlier models left behind there
 * it was refused although every other model had unloaded for it. Closing a queue beside a busy
 * one can reset the card (core/device/aql.cpp, g_spare), so only where `m` is the one model that
 * could launch there, and it is stopped at its grant; the device refuses while any queue of the
 * card holds a packet. The next load opens them again (reopen_queues). */
int Fleet::close_idle_queues(const Model& m, const std::map<int, int64_t>& need) {
    std::lock_guard<std::recursive_mutex> room(room_mu_);
    int n = 0;
    std::vector<int> closed;
    for (auto& [c, bytes] : need) {
        if (room_on(c) >= bytes || !card_quiet(c, &m)) continue;
        if (const int k = dev_stream_trim(c, kKeepQueues); k > 0) {
            n += k;
            closed.push_back(c);
        }
    }
    if (n > 0)
        RAD_INFO("model %s: %d kept queue(s) closed on card(s) %s for the room they held (%s)",
                 m.entry.name.c_str(), n, cards_str(closed).c_str(),
                 humanb((int64_t)n * kQueueBytes).c_str());
    return n;
}

void Fleet::refresh_out(Model& m) {
    std::map<int, int64_t> out = m.engine ? dev_page_out_bytes(m.engine->owner()) : std::map<int, int64_t>{};
    std::lock_guard<std::mutex> lk(mu_);
    m.out = out;
}

/* Two passes. The experts' flex of any model goes first, whatever its priority: it grew into
 * memory nobody held, and giving it back is an address change for each unit (§6.7). Then the
 * caches' flex of models of no higher priority, as §6.4 orders it. */
void Fleet::reclaim_flex(const Model& m, const std::map<int, int64_t>& need) {
    auto short_by = [&] {
        int64_t sh = 0;
        for (auto& [c, n] : need) sh = std::max(sh, n - room_on(c));
        return sh;
    };
    int64_t sh = short_by();
    for (int pass = 0; pass < 2 && sh > 0; ++pass) {
        const bool experts = pass == 0;
        std::vector<Model*> c;
        for (auto& mp : models_) {
            Model& o = *mp;
            if (&o == &m || !o.engine || o.parked || !o.resident || o.op.load() != kIdle) continue;
            if (experts ? o.engine->expert_flex_bytes() <= 0
                        : o.entry.priority > m.entry.priority || o.engine->flex_bytes() <= 0)
                continue;
            if (touches(o.cards, need)) c.push_back(&o);
        }
        std::sort(c.begin(), c.end(), [&](Model* a, Model* b) {
            if (a->entry.priority != b->entry.priority) return a->entry.priority < b->entry.priority;
            return front_.activity(a->entry.name).leases < front_.activity(b->entry.name).leases;
        });
        for (Model* o : c) {
            if (sh <= 0) break;
            const bool preempt = !experts && o->entry.priority < m.entry.priority;
            const int64_t got = o->engine->flex_release(sh, preempt, 10000, /*kv=*/!experts);
            if (got > 0)
                RAD_INFO("model %s: gave back %s of its %s flex for %s", o->entry.name.c_str(),
                         humanb(got).c_str(), experts ? "experts'" : "cache's", m.entry.name.c_str());
            for (int i = 0; i < 100 && short_by() > sh - got * 9 / 10; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            sh = short_by();
        }
    }
}

/* `need` bytes free on each card of the map for `m`: flex first, then the policy's models leave,
 * each only as much as is still missing (docs/MULTIMODEL.md §7: partial parking). */
int Fleet::make_room(Model& m, const std::map<int, int64_t>& need, std::string* why) {
    std::lock_guard<std::recursive_mutex> room(room_mu_);
    reclaim_flex(m, need);
    std::vector<int> cards;
    int64_t worst = 0;
    for (auto& [c, n] : need) { cards.push_back(c); worst = std::max(worst, n); }
    auto shortfall = [&] {
        std::map<int, int64_t> sh;
        for (const CardRoom& r : rooms())
            if (need.count(r.card)) {
                const int64_t s = need.at(r.card) - r.free;
                if (s > 0) sh[r.card] = s;
            }
        return sh;
    };
    std::vector<size_t> forced = exclusive_victims(m);
    /* A warm load (§8.6) displaces only other warm models: a model a request has used stays. */
    const bool warm = m.warm.load();
    if (warm)
        for (size_t v : forced)
            if (!models_[v]->warm.load()) {
                *why = models_[v]->entry.name + " is in use and shares an exclusive group with it";
                return RAD_E_NOMEM;
            }
    bool emptying = false;
    for (int attempt = 0; attempt < 6; ++attempt) {
        std::map<int, int64_t> sh = shortfall();
        if (sh.empty() && forced.empty()) return RAD_OK;
        std::vector<Resident> res = residents(&m);
        keep_partners(m, &res);
        if (warm)
            for (Resident& r : res)
                for (size_t id : unit_ids(r))
                    if (!models_[id]->warm.load()) r.keep = true;
        RoomPlan plan;
        std::string w;
        if (!plan_room(rooms(), res, (int)cards.size(), worst, m.entry.priority, forced, {}, &cards,
                       &plan, &w, &need)) {
            std::string staying;
            for (const Resident& r : res)
                if (r.pin || r.keep || r.priority > m.entry.priority)
                    staying += (staying.empty() ? "" : ", ") + models_[r.id]->entry.name;
            *why = "it needs " + std::to_string((worst + kMiB - 1) / kMiB) + " MiB on card(s) " +
                   cards_str(cards) + " and that much cannot be made free there" +
                   (staying.empty() ? std::string() : "; staying: " + staying);
            /* By the ledger the models that may leave do not hold enough. A model also holds
             * memory the ledger does not give it back for (what a park leaves of it on the card,
             * its plugin's own buffers), which comes free only when it unloads: a model that
             * needs nearly all of its cards, MiniMax-H3's 29.2 GiB of 31.9, was refused beside
             * ten parked models that held under a GiB each. With nothing staying, every model on
             * these cards parks and the last resort below unloads them before the demand fails.
             * A warm load takes no such step. */
            if (warm || !staying.empty()) return RAD_E_NOMEM;
            for (const Resident& r : res)
                for (size_t id : unit_ids(r)) {
                    Model& victim = *models_[id];
                    if (!victim.engine || victim.parked || !touches(victim.cards, need)) continue;
                    int expect = kIdle;
                    if (!victim.op.compare_exchange_strong(expect, kEvict)) continue;
                    (void)take_from(victim, {}, "making room for " + m.entry.name);
                    victim.op.store(kIdle);
                }
            emptying = true;
            break;
        }
        bool moved = false;
        for (size_t v : plan.victims) {
            Model& victim = *models_[v];
            const bool must = std::find(forced.begin(), forced.end(), v) != forced.end();
            sh = shortfall();
            if (sh.empty() && !must) break;
            /* What it gives: the shortfall on its cards, a little over so the next decision is not
             * a granule short; all of it when an exclusive group sends it away. */
            std::map<int, int64_t> want;
            if (!must)
                for (int c : victim.cards)
                    if (sh.count(c)) want[c] = sh[c] + 256 * kMiB;
            int expect = kIdle;
            if (!victim.op.compare_exchange_strong(expect, kEvict)) continue;
            moved = take_from(victim, want, "making room for " + m.entry.name) || moved;
            victim.op.store(kIdle);
        }
        forced.clear();
        if (!moved) break;   /* nobody could give more: the last resort below */
    }
    /* Last resort: parked models on these cards unload (what stays of them on the card is all that
     * is left to free), longest parked first. */
    std::vector<Model*> parked;
    for (auto& mp : models_) {
        Model& o = *mp;
        if (&o == &m || !o.engine || !o.parked || o.op.load() != kIdle ||
            o.entry.priority > m.entry.priority)
            continue;
        if (touches(o.cards, need)) parked.push_back(&o);
    }
    if (warm) parked.clear();
    std::sort(parked.begin(), parked.end(), [](Model* a, Model* b) { return a->parked_ns < b->parked_ns; });
    /* Only if what they still hold closes the gap: unloading them for less loses their copies
     * and frees nothing that matters (another model's room is what is missing). When the cards
     * are being emptied the ledger has already said it cannot tell. */
    if (!emptying) {
        const std::map<int, int64_t> sh = shortfall();
        for (auto& [c, n] : sh) {
            int64_t gives = 0;
            for (Model* o : parked) {
                if (std::find(o->cards.begin(), o->cards.end(), c) == o->cards.end()) continue;
                const auto it = o->out.find(c);
                gives += std::max<int64_t>(o->charge - (it == o->out.end() ? 0 : it->second), 0);
            }
            if (gives < n) parked.clear();
        }
    }
    for (Model* o : parked) {
        if (shortfall().empty()) break;
        int expect = kIdle;
        if (!o->op.compare_exchange_strong(expect, kUnload)) continue;
        unload(*o, 0, "making room for " + m.entry.name);
        o->op.store(kIdle);
    }
    if (shortfall().empty()) return RAD_OK;
    *why = "the models that may leave did not free enough on card(s) " + cards_str(cards);
    return RAD_E_NOMEM;
}

bool Fleet::in_quantum(const Model& m, const server::Front::Activity& a, uint64_t now) const {
    if (m.parked || !m.resident || !m.engine || m.entry.pin) return false;
    /* A job with no open request (a video) cannot be parked mid-way, and unloading throws away
     * what it has done, minutes of work: the model stays until the job ends. */
    if (m.engine->background_jobs() > 0) return true;
    if (a.leases <= 0 && a.waiting <= 0) return false;
    const double back = m.restore_s > 0 ? m.restore_s : (m.load_s > 0 ? m.load_s : 5.0);
    const double q = std::min(30.0, std::max(5.0, 4.0 * back));
    return now > m.on_ns && (double)(now - m.on_ns) * 1e-9 < q;
}

int Fleet::passing_on(const Model& m, const std::map<int, int64_t>& need) {
    const uint64_t now = rad_mono_ns();
    std::vector<server::Front::Activity> act(models_.size());
    for (size_t i = 0; i < models_.size(); ++i) act[i] = front_.activity(models_[i]->entry.name);
    std::lock_guard<std::mutex> lk(mu_);
    /* A model with no cards yet is loading and may land on any. */
    auto near = [&](const std::vector<int>& cards) {
        return cards.empty() || touches(cards, need);
    };
    bool waiters = false;
    for (auto& o : models_) {
        if (o.get() == &m || o->op.load() == kIdle || o->starting.load() || !near(o->cards)) continue;
        if (o->waiting.load()) waiters = true;
        else return 2;
    }
    for (size_t i = 0; i < models_.size(); ++i)
        if (models_[i].get() != &m && near(models_[i]->cards) && in_quantum(*models_[i], act[i], now))
            return 2;
    for (const Reservation& r : reserved_)
        if (r.m != &m && r.claim > 0 && !r.m->starting.load() && near(r.cards)) return 2;
    return waiters ? 1 : 0;
}

int Fleet::make_room_waiting(Model& m, const std::map<int, int64_t>& need, std::string* why,
                             const std::function<void()>& commit, int64_t max_ms) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto until = t0 + std::chrono::milliseconds(
        max_ms >= 0 ? std::min<int64_t>(max_ms, fc_.load_timeout_ms) : fc_.load_timeout_ms);
    auto still_since = t0;
    bool said = false, closed_queues = false;
    const Raised flag(m.waiting);
    /* The cards' free figures at the last look. Memory a model gave back is counted free only as
     * the driver's wipe reaches it, some GiB a second: after the tens of GiB a large model's
     * room takes (every other model unloaded for MiniMax-H3) the figure rises for longer than
     * the three seconds below, and a demand that gave up then was refused room that was on its
     * way. A figure still rising is something passing. */
    std::map<int, int64_t> free_was;
    for (auto& [c, n] : need) free_was[c] = free_on(c);
    for (;;) {
        int rc;
        {
            std::lock_guard<std::recursive_mutex> room(room_mu_);
            rc = make_room(m, need, why);
            if (rc >= 0 && commit) commit();
        }
        if (rc >= 0 || g_stop.load() || std::chrono::steady_clock::now() > until) return rc;
        for (auto& [c, was] : free_was) {
            const int64_t f = free_on(c);
            if (f > was + 16 * kMiB) still_since = std::chrono::steady_clock::now();
            was = f;
        }
        /* Nothing passing there: the residents cannot give it room. The fleet changes from one
         * moment to the next under load (a request ends, a quantum runs out), so that has to hold
         * for a few seconds before the demand fails. Only other models waiting for room: they may
         * be waiting on this one, and one of them gets room as the others' work ends, so that is
         * given a while to sort itself out, not the whole timeout. */
        const int p = passing_on(m, need);
        const auto now = std::chrono::steady_clock::now();
        if (p != 0) still_since = now;
        /* Nothing passing and the figures still for half a second: the kept queues of the cards
         * it has to itself go, once, before the wait for others that are not there. */
        if (!closed_queues && p == 0 && now - still_since > std::chrono::milliseconds(500) &&
            close_idle_queues(m, need) > 0) {
            closed_queues = true;
            still_since = now;
            continue;
        }
        if ((p == 0 && now - still_since > std::chrono::seconds(3)) ||
            (p == 1 && now - t0 > std::chrono::seconds(20)))
            return rc;
        if (!said) {
            std::vector<int> cards;
            for (auto& [c, n] : need) cards.push_back(c);
            RAD_INFO("model %s: waiting for room on card(s) %s while other models come and go (%s)",
                     m.entry.name.c_str(), cards_str(cards).c_str(), why->c_str());
            said = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

/* `victim` gives `want` (empty: all of it) by parking that much where it can, else by unloading. */
bool Fleet::take_from(Model& victim, const std::map<int, int64_t>& want, const std::string& why) {
    const int r = victim.entry.park.ram != ParkSpec::Ram::Never ? park(victim, want, why) : -1;
    if (r > 0) return true;
    /* Short: what is left of it on the card is not pageable; make_room unloads a parked model
     * only as the last resort, so it keeps its copy. */
    if (r == 0) return false;
    unload(victim, fc_.drain_timeout_ms, why);
    return true;
}

int Fleet::park(Model& m, const std::map<int, int64_t>& want, const std::string& why) {
    std::lock_guard<std::recursive_mutex> room(room_mu_);
    if (!m.engine) return -1;
    join_spill(m);
    int64_t dev_b = 0, host_b = 0;
    dev_page_stats(m.engine->owner(), &dev_b, &host_b);
    int64_t asked = 0;
    for (auto& [c, n] : want) asked += n;
    if (want.empty()) asked = dev_b;
    /* The copies this park makes: what it takes past the chunks that need none (dropped ones,
     * sealed ones already copied). Host room for them, kept copies of others going first; past the
     * budget they go to the disk tier, and without one the park stops where host memory does. */
    const int64_t uncopied = dev_page_uncopied(m.engine->owner());
    const int64_t copy = std::min(uncopied, std::max<int64_t>(asked - (dev_b - uncopied), 0));
    /* Room for the copy from what costs nothing to give (kept copies, snapshot stores, parked
     * copies with another copy), never from other parked models: unloading them for it took every
     * parked model off the host for one model's few GiB, and that model unloaded all the same
     * when the room was still short. */
    (void)make_host_room(copy, &m, false, false, false, 0, /*evict=*/false);
    const int64_t to_disk = std::max<int64_t>(copy - host_room(), 0);
    /* Writing to disk costs the load waiting on this room now, and reading it back costs the
     * model's own return; a model that loads from its container faster than the two together
     * unloads instead. */
    constexpr double kDiskWrite = 1.5e9, kDiskRead = 6.5e9;
    if (to_disk > 0 &&
        (fc_.disk_tier.empty() ||
         (double)to_disk / kDiskWrite + (double)to_disk / kDiskRead > std::max(m.load_s, 1.0))) {
        RAD_INFO("model %s: parking would write %s to disk and read it back, longer than its %.1f s "
                 "load; it unloads", m.entry.name.c_str(), humanb(to_disk).c_str(), m.load_s);
        return -1;
    }
    /* Its way back, from measured times: its last page-in and this park's disk writes, against its
     * load. Waits for room or for requests to drain are not counted: a load waits for those as
     * well. A park of under a quarter of it goes ahead regardless, since so small a part is
     * mostly unmapped against copies it has already. */
    const double trip = m.trip_s + (double)to_disk / kDiskWrite;
    /* `asked` is over all its cards, `charge` a card's. */
    const int64_t whole = m.charge * (int64_t)std::max<size_t>(m.cards.size(), 1);
    if (trip > 1.2 * m.load_s && m.load_s > 0 && (want.empty() || asked * 4 >= whole)) {
        RAD_INFO("model %s: its return would take %.1f s (its last page-in and what this park "
                 "writes), its load %.1f s; it unloads", m.entry.name.c_str(), trip, m.load_s);
        return -1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::map<int, int64_t> before;
    for (int c : m.cards) before[c] = free_on(c);
    PageOut req;
    req.want = want;
    req.host_budget = host_room();
    req.dir = fc_.disk_tier;
    PageResult got;
    std::string w;
    if (const int rc = m.engine->park(fc_.drain_timeout_ms, req, &got, &w); rc < 0) {
        RAD_WARN("model %s: could not park (%s)", m.entry.name.c_str(), w.c_str());
        return -1;
    }
    const auto t_freed = std::chrono::steady_clock::now();
    wait_pages(before, got.freed);
    const double settle_s = secs_since(t_freed);
    refresh_out(m);
    /* Short: some card freed more than 512 MiB less than it was asked for. make_room's asks carry
     * 256 MiB of slack, so a card short by up to as much again still counts as done. */
    bool short_of = false;
    for (auto& [c, n] : want) {
        const auto it = got.freed.find(c);
        if ((it == got.freed.end() ? 0 : it->second) + 512 * kMiB < n) short_of = true;
    }
    const bool was = m.parked;
    {
        std::lock_guard<std::mutex> lk(mu_);
        m.parked = true;
        m.resident = false;
        m.park_s = secs_since(t0);
        m.parked_ns = rad_mono_ns();
        ++m.parks;
        if (!was) { m.filled = 0; m.fill_s = 0; }
    }
    front_.set_parked(m.entry.name, true);
    int64_t out_total = 0;
    for (auto& [c, n] : m.out) out_total += n;
    RAD_INFO("model %s: %s in %.2f s (%s): %s off its cards; this time %s dropped, %s against "
             "earlier copies, %s against its container, %s copied to host memory, %s to disk "
             "(rest %.2f s, slab %.2f s, page-out %.2f s, the cards' free figure %.2f s); the fleet "
             "pins %s of %s",
             m.entry.name.c_str(), was ? "parked further" : want.empty() ? "parked" : "parked in part",
             m.park_s, why.c_str(), humanb(out_total).c_str(), humanb(got.dropped).c_str(),
             humanb(got.reused).c_str(), humanb(got.file).c_str(), humanb(got.host).c_str(),
             humanb(got.disk).c_str(),
             got.quiet_s, got.slab_s, got.out_s, settle_s, humanb(host_held()).c_str(),
             humanb(host_budget_).c_str());
    if (got.host > 0) start_spill(m);
    return short_of ? 0 : 1;
}

void Fleet::unload(Model& m, int64_t drain_ms, const std::string& why) {
    std::lock_guard<std::recursive_mutex> room(room_mu_);
    join_spill(m);
    if (!m.engine) return;
    const auto t0 = std::chrono::steady_clock::now();
    RAD_INFO("model %s: unloading: %s", m.entry.name.c_str(), why.c_str());
    std::map<int, int64_t> before;
    for (int c : m.cards) before[c] = free_on(c);
    /* What the engine holds on each card is at most its charge less what a park took off, and
     * at most what it has backed: a cache that grows (flex) holds less than its claim, and waiting
     * for the claim to come back ran out the two seconds wait_pages allows on every such unload. */
    std::map<int, int64_t> expect;
    const int owner = m.engine->owner();
    for (int c : m.cards) {
        const auto it = m.out.find(c);
        const int64_t out = m.parked && it != m.out.end() ? it->second : 0;
        expect[c] = std::min(std::max<int64_t>(m.charge - out, 0), dev_owner_bytes(owner, c));
    }
    m.engine->request_stop(drain_ms);
    if (m.serve.joinable()) m.serve.join();
    std::shared_ptr<Engine> gone;
    {
        std::lock_guard<std::mutex> lk(mu_);
        gone = std::move(m.engine);
    }
    gone.reset();
    front_.detach(m.entry.name, 0);
    front_.set_parked(m.entry.name, false);
    wait_pages(before, expect);
    const std::vector<int> was_on = m.cards;
    {
        std::lock_guard<std::mutex> lk(mu_);
        m.resident = false;
        m.parked = false;
        m.out.clear();
        m.cards.clear();
        ++m.unloads;
    }
    /* A card it leaves empty keeps the reserve of queues and no more: the queues its streams
     * held, and those of the models before it, are closed while nothing can launch there. */
    for (int c : was_on)
        if (card_quiet(c, &m)) (void)dev_stream_trim(c, kReserveQueues);
    RAD_INFO("model %s: unloaded in %.2f s", m.entry.name.c_str(), secs_since(t0));
}

void Fleet::join_spill(Model& m) {
    if (!m.spill.joinable()) return;
    if (m.spill_abort) m.spill_abort->store(true);
    m.spill.join();
}

/* After a park whose copies went to host memory: they are written to the disk tier in the
 * background (paused while a model loads, since its container reads share the drive), so a later
 * squeeze on host memory lets them go instead of the model. */
void Fleet::start_spill(Model& m) {
    if (fc_.disk_tier.empty() || !m.engine) return;
    struct statvfs st {};
    if (statvfs(fc_.disk_tier.c_str(), &st) != 0) return;
    if ((int64_t)st.f_bavail * (int64_t)st.f_frsize < ((int64_t)24 << 30)) return;
    join_spill(m);
    const int owner = m.engine->owner();
    const std::string dir = fc_.disk_tier, name = m.entry.name;
    std::atomic<int>* loading = &loads_in_flight_;
    m.new_spill_flags();
    m.spill = std::thread([owner, dir, name, loading, abort = m.spill_abort, done = m.spill_done,
                           end = m.spill_end] {
        const SetOnExit ended{ *end };
        const int64_t n = dev_page_spill(owner, dir, [loading, abort] {
            return abort->load() ? -1 : loading->load(std::memory_order_acquire) > 0 ? 1 : 0;
        });
        done->store(n >= 0);
        if (n < 0 && !abort->load())
            RAD_WARN("model %s: writing its parked copy to %s failed: %s", name.c_str(), dir.c_str(),
                     rad_dev_last_error());
    });
}

/* After a live restore, in the background: its rest moves onto the cards. */
void Fleet::start_promote(Model& m) {
    if (!m.engine) return;
    join_spill(m);
    const int owner = m.engine->owner();
    const std::string name = m.entry.name;
    Engine* const eng = m.engine.get();   /* park and unload join this thread first */
    /* Weights read in place move on only into free memory, never into room another model is
     * owed; unbounded otherwise (the restore's room holds them). */
    std::function<int64_t()> cap = [] { return INT64_MAX; };
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (m.in_place) cap = [this, cards = m.cards] { return free_room(cards); };
    }
    const bool capped = cap() != INT64_MAX;
    m.new_spill_flags();
    m.spill = std::thread([this, owner, name, eng, cap, capped, abort = m.spill_abort, done = m.spill_done,
                           end = m.spill_end] {
        const SetOnExit ended{ *end };
        /* A GiB at a time, each batch swapped in between two steps. One that finds no device
         * memory leaves the rest in host memory for now: the model serves on, slower, and the
         * next restore or a later pass moves it. */
        const auto t0 = std::chrono::steady_clock::now();
        int64_t moved = 0;
        int fails = 0;
        while (!abort->load()) {
            /* A capped pass runs under the room lock. A load or restore frees room and backs it
             * under that lock, and a restore holds no reservation: free memory read beside it is
             * the room it just made, and a pass there left it short of its own commit. While
             * another holds the lock the pass waits; a try, since park and unload join this
             * thread with the lock held. */
            std::unique_lock<std::recursive_mutex> held(room_mu_, std::defer_lock);
            if (capped && !held.try_lock()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            /* A pass may take one 64 MiB chunk past what it is given. */
            const int64_t room = cap();
            if (room < kPromoteChunk) break;
            const int64_t n = dev_page_promote(owner, room == INT64_MAX ? room : room - kPromoteChunk,
                                               [eng](const std::function<int()>& fn) { return eng->run_quiet(fn); });
            if (n == 0) break;
            if (n < 0) {
                if (++fails > 50) break;
                if (held.owns_lock()) held.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            moved += n;
        }
        const int64_t rest = dev_page_live_bytes(owner);
        if (moved > 0 || !capped)
            RAD_INFO("model %s: %s moved onto its cards in %.1f s while it served%s", name.c_str(),
                     humanb(moved).c_str(), secs_since(t0),
                     rest > 0 ? (", " + humanb(rest) + " still read from host memory").c_str() : "");
        done->store(!abort->load());
    });
}

/* All the pinned host memory the process holds (§6.5): its allocations, the host copies of parked
 * and kept chunks (host-backed mappings, not allocations), and what loads in flight will pin. */
int64_t Fleet::host_held() {
    int64_t pinned = 0, mapped = 0, n = 0;
    rad_dev_alloc_totals(RAD_MEM_HOST_PINNED, &pinned, &n);
    rad_dev_alloc_totals(RAD_MEM_HOST_MAPPED, &mapped, &n);
    return pinned + mapped + dev_page_host_bytes() + host_reserve_.load();
}

int64_t Fleet::host_room() {
    return std::max<int64_t>(std::min(host_budget_ - host_held(),
                                      host_available() - host_reserve_.load() - kHostFloor), 0);
}

bool Fleet::release_spares(const std::vector<Model*>& parked, int64_t need) {
    std::lock_guard<std::recursive_mutex> room(room_mu_);
    for (Model* m : parked) {
        if (host_room() >= need) return true;
        const int64_t n = m->engine->release_host_spare();
        if (n > 0)
            RAD_INFO("model %s: its snapshot store's %s of pinned host memory released while it is "
                     "parked", m->entry.name.c_str(), humanb(n).c_str());
    }
    return host_room() >= need;
}

/* Another model's load or return is in flight: its host pins are counted twice until it ends (its
 * reservation and what it has pinned so far), so host room read beside it is short of the truth. */
bool Fleet::loading_beside(const Model& m) {
    std::lock_guard<std::recursive_mutex> room(room_mu_);
    for (auto& o : models_) {
        const int op = o->op.load();
        if (o.get() != &m && (op == kLoad || op == kRestore)) return true;
    }
    return false;
}

/* No other model holds an engine and none is coming or going: nothing is left that could give host
 * memory back. */
bool Fleet::alone_on_host(const Model& m) {
    std::lock_guard<std::recursive_mutex> room(room_mu_);
    for (auto& o : models_)
        if (o.get() != &m && (o->engine || o->op.load() != kIdle)) return false;
    return true;
}

/* Room to pin `need` during a load. As in the load's first wait, beside another load in flight
 * parked and idle models go only after ten seconds, once a load counted twice has passed. */
bool Fleet::load_host_room(Model& m, int64_t need, int64_t own) {
    {
        /* There already: a warm-up needs no budget for a pin that leaves the host its floor. */
        std::lock_guard<std::recursive_mutex> room(room_mu_);
        if (host_available() - host_reserve_.load() + own - kHostFloor >= need) return true;
    }
    /* Before its grant a load holds a guess of VRAM, which what pins this memory may be waiting
     * for (a parked model coming back): they would wait for each other. The guess goes while it
     * waits; the grant makes room for the exact claim. */
    if (own == 0) {
        std::lock_guard<std::mutex> lk(mu_);
        for (Reservation& r : reserved_)
            if (r.m == &m) r.claim = 0;
    }
    const bool warm = m.warm.load();
    Engine* const eng = m.engine.get();
    if (eng) eng->bringup_pause();
    bool ok = false;
    /* Holding no VRAM it can wait as long as for VRAM. Past its grant, a minute at most: it holds its
     * VRAM here, which what pins the memory may be waiting for, and the failed start lets go of it. */
    const int64_t tries = warm ? 1 : own == 0 ? std::max<int64_t>(fc_.load_timeout_ms / 200, 300) : 300;
    const Raised flag(m.waiting_host);
    for (int64_t i = 0; i < tries && !ok && !g_stop.load(); ++i) {
        /* As in load_op: without a load in flight beside this one the figure is exact, and parked
         * and idle models give their host memory at once. */
        const bool exact = !warm && !loading_beside(m);
        ok = make_host_room(need, &m, !warm, !warm && (exact || i >= 50),
                            !warm && (i >= 150 || (exact && i >= 10)), own);
        if (ok || i + 1 >= tries) break;
        if (i == 0)
            RAD_INFO("model %s: waiting for %s of host memory to pin (%s available, %s of it kept "
                     "free)", m.entry.name.c_str(), humanb(need).c_str(),
                     humanb(host_available()).c_str(), humanb(kHostFloor).c_str());
        /* A host that every other model has left has no more to give, and the rest of the load
         * timeout would pass with nothing changing. */
        if ((i >= 150 || (exact && i >= 10)) && alone_on_host(m)) {
            RAD_INFO("model %s: %s of host memory to pin is not there with every other model "
                     "unloaded (%s available, %s of it kept free)", m.entry.name.c_str(),
                     humanb(need).c_str(), humanb(host_available()).c_str(),
                     humanb(kHostFloor).c_str());
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    if (eng) eng->bringup_resume();
    return ok;
}

bool Fleet::make_host_room(int64_t need, const Model* keep, bool force, bool unload_parked, bool take_busy,
                           int64_t own, bool evict) {
    std::lock_guard<std::recursive_mutex> room(room_mu_);
    /* A load is held to the host's floor alone; the budget (--host-mib) is what parks' copies may
     * fill, and pinned tables past it are no reason to clear out the other models. */
    auto fits = [&] {
        return force ? host_available() - host_reserve_.load() + own - kHostFloor >= need
                     : host_room() >= need;
    };
    if (fits()) return true;
    /* Kept copies first: they cost the next park a copy, nothing more. */
    for (auto& m : models_) {
        if (m.get() == keep || !m->engine || m->parked || m->op.load() != kIdle) continue;
        dev_page_drop_copies(m->engine->owner());
        if (fits()) return true;
    }
    std::vector<Model*> parked;
    for (auto& m : models_)
        if (m.get() != keep && m->engine && m->parked && m->op.load() == kIdle) parked.push_back(m.get());
    std::sort(parked.begin(), parked.end(), [](Model* a, Model* b) { return a->parked_ns < b->parked_ns; });
    /* Parked models' snapshot stores next: pinning one again costs its restore a fraction of a
     * second (Engine::release_host_spare). */
    (void)release_spares(parked, need);
    if (fits()) return true;
    /* Parked host copies with another copy (on the drive, or the container a weights pool is
     * refilled from) let their host memory go, longest parked first. */
    for (Model* m : parked) {
        dev_page_drop_host(m->engine->owner());
        if (fits()) return true;
    }
    /* Without a disk tier the longest-parked models unload. With one, a park that finds no host
     * room writes straight to disk instead. Not for a park's own copy (`evict` false): the model
     * being parked unloads when its copy has no room, which costs one model its place where this
     * cost every parked one theirs. */
    if ((fc_.disk_tier.empty() || force) && evict)
        for (Model* m : parked) {
            int expect = kIdle;
            if (!m->op.compare_exchange_strong(expect, kUnload)) continue;
            if (!fc_.disk_tier.empty()) {
                /* The background spill pauses while a model loads, so it is done here. */
                join_spill(*m);
                const auto t0 = std::chrono::steady_clock::now();
                const int64_t n = dev_page_spill(m->engine->owner(), fc_.disk_tier);
                if (n >= 0) {
                    dev_page_drop_host(m->engine->owner());
                    if (n > 0)
                        RAD_INFO("model %s: %s of its parked copies written to disk in %.1f s for host room",
                                 m->entry.name.c_str(), humanb(n).c_str(), secs_since(t0));
                } else {
                    unload(*m, 0, "its host memory is wanted and its copies could not go to disk");
                }
            } else {
                unload(*m, 0, "its host memory is wanted for another park");
            }
            m->op.store(kIdle);
            if (fits()) return true;
        }
    /* Last, what parked models still pin (an expert pool, a KV tier's arena): a parked model comes
     * back by loading, and the load asking for this could not start at all otherwise. */
    if (unload_parked)
        for (Model* m : parked) {
            int expect = kIdle;
            if (!m->op.compare_exchange_strong(expect, kUnload)) continue;
            join_spill(*m);
            unload(*m, 0, "a load needs the host memory it pins");
            m->op.store(kIdle);
            if (fits()) return true;
        }
    /* And a resident model with nothing to do, least recently used first, unless it outranks the
     * model loading. With `take_busy`, also one at work whose quantum has run out, as for VRAM: two
     * models that cannot share the host would otherwise never take turns. */
    if (unload_parked) {
        const uint64_t now = rad_mono_ns();
        std::vector<Model*> idle, busy;
        for (auto& m : models_) {
            if (m.get() == keep || !m->engine || m->parked || !m->resident || m->entry.pin ||
                m->op.load() != kIdle || (keep && m->entry.priority > keep->entry.priority))
                continue;
            const auto a = front_.activity(m->entry.name);
            if (a.leases == 0 && a.waiting == 0 && m->engine->background_jobs() == 0) {
                /* Just arrived and not yet used: the requests it came for are on their way. One
                 * that has answered since it came on is idle. */
                if (a.last_used_ns >= m->on_ns || now - m->on_ns > 10000000000ull)
                    idle.push_back(m.get());
            } else if (take_busy && !in_quantum(*m, a, now)) {
                busy.push_back(m.get());
            }
        }
        /* What frees the most first: unloading a small model for a load that wants tens of GiB
         * costs its next request a load and brings the room no nearer. */
        auto most_first = [&](Model* a, Model* b) {
            const int64_t x = a->engine->host_pinned_bytes(), y = b->engine->host_pinned_bytes();
            if (x != y) return x > y;
            return front_.activity(a->entry.name).last_used_ns < front_.activity(b->entry.name).last_used_ns;
        };
        std::sort(idle.begin(), idle.end(), most_first);
        std::sort(busy.begin(), busy.end(), most_first);
        idle.insert(idle.end(), busy.begin(), busy.end());
        for (Model* m : idle) {
            const bool at_work = std::find(busy.begin(), busy.end(), m) != busy.end();
            int expect = kIdle;
            if (!m->op.compare_exchange_strong(expect, kUnload)) continue;
            join_spill(*m);
            unload(*m, at_work ? fc_.drain_timeout_ms : 0, "a load needs the host memory it pins");
            m->op.store(kIdle);
            if (fits()) return true;
        }
    }
    return fits();
}

/* ------------------------------------------------------------------ work */

int Fleet::grant(Model& m, const VramAsk& ask, int64_t* grant, std::string* why) {
    const Config& cfg = m.entry.cfg;
    int64_t claim = 0;
    if (cfg.given.count("--gpu-memory-mib")) {
        claim = cfg.gpu_memory_mib * kMiB;
        if (claim < ask.need) {
            *why = "gpu-memory-mib " + std::to_string(cfg.gpu_memory_mib) + " is below the " +
                   std::to_string((ask.need + kMiB - 1) / kMiB) + " MiB it needs";
            return RAD_E_NOMEM;
        }
    } else {
        claim = flex_claim(ask.need, ask.want, ask.kv_fixed, ask.kv_ceiling, ask.kv_one,
                           ask.max_tokens, ask.experts, ask.expert_floor, m.entry.flex.kv_min_tokens,
                           m.entry.flex.experts_min_mib);
    }
    /* It no longer waits on anyone's bring-up: it holds the lock. Its own guess stops counting
     * while it plans the exact claim. */
    m.starting.store(false);
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (Reservation& r : reserved_)
            if (r.m == &m) r.claim = 0;
    }
    /* What the engine has allocated before its grant counts against its own claim already. */
    std::map<int, int64_t> need;
    for (int c : ask.cards)
        need[c] = std::max<int64_t>(claim - dev_owner_bytes(m.engine ? m.engine->owner() : 0, c), 0);
    /* Waiting for room, the engine lets other engines' bring-ups go on (it takes the lock back
     * before it carries on). */
    Engine* const eng = m.engine.get();
    if (eng) eng->bringup_pause();
    const int wrc = make_room_waiting(m, need, why, [&] {
        std::lock_guard<std::mutex> lk(mu_);
        m.claim = claim;
        for (Reservation& r : reserved_)
            if (r.m == &m) {
                r.owner = m.engine ? m.engine->owner() : 0;
                r.cards = ask.cards;
                r.claim = claim;
            }
    }, m.warm.load() ? 0 : -1);
    if (eng) eng->bringup_resume();
    if (wrc < 0) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const Reservation& r : reserved_)
            if (r.m != &m && r.claim > 0 && r.m->starting.load() && touches(r.cards, need))
                m.room_retry.store(true);
        return wrc;
    }
    *grant = claim;
    return RAD_OK;
}

void Fleet::fail(Model& m, const std::string& why) {
    /* A warm load that finds no room is not a failure: nobody asked for the model yet. */
    if (m.warm.exchange(false)) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            m.resident = false;
            m.cards.clear();
        }
        RAD_INFO("warm-up: %s stays cold: %s", m.entry.name.c_str(), why.c_str());
        front_.abandon(m.entry.name);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        m.error = why;
        ++m.failures;
        m.resident = false;
        m.cards.clear();
    }
    RAD_ERR("model %s: not loaded: %s", m.entry.name.c_str(), why.c_str());
    front_.fail(m.entry.name, why, 15000);
}

bool Fleet::start_op(Model& m, int op) {
    int expect = kIdle;
    if (!m.op.compare_exchange_strong(expect, op)) return false;
    if (m.worker.joinable()) m.worker.join();
    m.worker = std::thread([this, &m, op] {
        if (op == kLoad) load_op(m);
        else if (op == kRestore) restore_op(m);
        else if (op == kEvict) park_op(m, "idle for its ttl");
        else if (op == kFill) fill_op(m);
        else if (op == kYield) yield_op(m);
        m.op.store(kIdle);
        std::lock_guard<std::mutex> lk(mu_);
        cv_.notify_all();
    });
    return true;
}

void Fleet::park_op(Model& m, const std::string& why) {
    if (!m.engine || m.parked) return;
    if (m.entry.park.ram == ParkSpec::Ram::Never || park(m, {}, why) < 0) unload(m, fc_.drain_timeout_ms, why);
}

/* A parked model's chunks whose only copy is on the drive come into host memory a batch at a
 * time while the fleet is quiet (§8.6), so its next restore copies over the link. They are
 * copies of copies: make_host_room lets them go first. */
void Fleet::fill_op(Model& m) {
    if (!m.engine || !m.parked) return;
    /* A GiB at a time: a restore asked for meanwhile waits for one batch at most. */
    constexpr int64_t kBatch = (int64_t)1 << 30;
    if (host_room() - kFillSlack - Engine::kRefillHost < kBatch) {
        /* Snapshot stores of parked models are worth less than weights on the drive: pinning one
         * again takes a fraction of a second, reading a GiB back from the drive far longer. */
        std::vector<Model*> parked;
        for (auto& o : models_)
            if (o->engine && o->parked && (o.get() == &m || o->op.load() == kIdle)) parked.push_back(o.get());
        (void)release_spares(parked, kBatch + kFillSlack + Engine::kRefillHost);
    }
    /* The fill's own refill holds up to Engine::kRefillHost while it runs. */
    const int64_t room = host_room() - kFillSlack - Engine::kRefillHost;
    /* No room: asked again after a while, not every scheduler round. */
    if (room < kBatch / 4) {
        fill_wait_ns_.store(rad_mono_ns() + 5000000000ull);
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t n = m.engine->fill_host(std::min(room, kBatch));
    if (n < 0) {
        RAD_WARN("model %s: reading its parked copy into host memory failed: %s", m.entry.name.c_str(),
                 rad_dev_last_error());
        std::lock_guard<std::mutex> lk(mu_);
        m.filled = -1;   /* not tried again this park */
        return;
    }
    /* Every chunk filled has its drive copy, so the host copies may go whenever host memory is
     * wanted. */
    if (!m.spill.joinable() || (m.spill_end && m.spill_end->load())) {
        join_spill(m);
        m.spill_done = std::make_shared<std::atomic<bool>>(true);
    }
    const int64_t rest = dev_page_fillable(m.engine->owner());
    std::lock_guard<std::mutex> lk(mu_);
    m.filled += n;
    m.fill_s += secs_since(t0);
    if (rest == 0 || host_room() - kFillSlack < kBatch / 4)
        RAD_INFO("model %s: %s of its parked copy read into host memory in %.1f s%s", m.entry.name.c_str(),
                 humanb(m.filled).c_str(), m.fill_s,
                 rest > 0 ? (", " + humanb(rest) + " left on the drive (host memory is full)").c_str() : "");
}

void Fleet::restore_op(Model& m) {
    const auto t0 = std::chrono::steady_clock::now();
    front_.loading(m.entry.name);
    const InFlight in_flight(loads_in_flight_);
    settle_flex();
    join_spill(m);
    std::map<int, int64_t> need = dev_page_out_bytes(m.engine->owner());
    /* A one-step model (§7.5) whose cards lack the room for all of it right now serves at once,
     * its weights parked in host memory read in place, rather than wait for room or move another
     * model for it: each of its requests reads them over the link once, as the restore would. */
    bool in_place = false;
    if (m.engine->one_step() && m.entry.park.restore == ParkSpec::Restore::Auto) {
        const std::map<int, int64_t> part = dev_page_out_bytes(m.engine->owner(), true);
        in_place = part != need && !fits_free(need);
        if (in_place) need = part;
    }
    std::string why;
    /* A parked model comes back on its own cards. A fresh load could pick others, but needs as
     * much room; it is worth it only where other cards exist, and only after a while. */
    const bool elsewhere = (int)cards_.size() > (int)m.cards.size();
    std::unique_lock<std::recursive_mutex> room(room_mu_, std::defer_lock);
    int rrc = RAD_OK;
    bool yielded = false;
    for (;;) {
        /* Until it has given way, it waits a short while at a time, so the check below runs. */
        (void)make_room_waiting(m, need, &why, {}, yielded ? -1 : 2000);
        room.lock();
        rrc = make_room(m, need, &why);
        if (rrc >= 0 || g_stop.load()) break;
        const double waited = secs_since(t0);
        if ((elsewhere && waited > 20.0) || waited * 1000.0 > (double)fc_.load_timeout_ms) break;
        /* Two returning models can each hold what is left of them where the other needs room,
         * and neither leaves while it waits: this one gives all of its part on the cards up and
         * waits on. If both do, whichever asks next has the room. */
        if (!yielded && waited > 2.0) {
            yielded = true;
            RAD_INFO("model %s: waiting to come back beside another model that waits; it parks "
                     "the rest of itself meanwhile", m.entry.name.c_str());
            (void)park(m, {}, "giving way to another model's return");
            need = dev_page_out_bytes(m.engine->owner(), in_place);
        }
        room.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    /* Stopping: it stays parked, rather than start a fresh load the shutdown would wait for. */
    if (rrc < 0 && g_stop.load()) {
        room.unlock();
        return;
    }
    if (rrc < 0) {
        RAD_INFO("model %s: no room to come back on card(s) %s (%s); it loads fresh",
                 m.entry.name.c_str(), cards_str(m.cards).c_str(), why.c_str());
        unload(m, 0, "it could not come back on its cards");
        room.unlock();
        load_op(m);
        return;
    }
    const double room_s = secs_since(t0);
    std::string w;
    /* Its snapshot store, if a squeeze on host memory released it, is pinned again first. */
    if (const int64_t pin = m.engine->host_spare_released(); pin > 0) (void)make_host_room(pin, &m);
    /* park.restore live, or a one-step model read in place: it serves at once, what is parked in
     * host memory read in place until it is moved. */
    int64_t live = 0;
    /* Room for the refill of its weights from the container while it runs, held against other
     * pins meanwhile. */
    (void)make_host_room(Engine::kRefillHost, &m);
    host_reserve_.fetch_add(Engine::kRefillHost);
    const int urc = m.entry.park.restore == ParkSpec::Restore::Live || in_place
                        ? m.engine->unpark_live(&w, &live)
                        : m.engine->unpark(&w);
    host_reserve_.fetch_sub(Engine::kRefillHost);
    if (urc < 0) {
        RAD_ERR("model %s: could not come back: %s", m.entry.name.c_str(), w.c_str());
        unload(m, 0, "it could not come back");
        room.unlock();
        load_op(m);
        return;
    }
    /* What came back onto the cards: in place, `need` already left out what stays in host memory. */
    int64_t back = in_place ? 0 : -live;
    for (auto& [c, n] : need) back += n;
    const double took = secs_since(t0);
    {
        std::lock_guard<std::mutex> lk(mu_);
        m.parked = false;
        m.resident = true;
        m.out.clear();
        m.in_place = in_place && live > 0;
        m.restore_s = took;
        m.on_ns = rad_mono_ns();
        m.trip_s = took - room_s + (double)live / 25e9;   /* and the move behind it, over the link */
        ++m.restores;
        m.seq = ++seq_;
    }
    trim_spares(m.cards);
    room.unlock();
    RAD_INFO("model %s: back on card(s) %s in %.2f s (%s moved back; room %.2f s, page-in %.2f s)%s",
             m.entry.name.c_str(), cards_str(m.cards).c_str(), took, humanb(back).c_str(), room_s,
             took - room_s,
             live <= 0  ? ""
             : in_place ? ("; serving, " + humanb(live) + " read in place from host memory (no room for it "
                           "on its cards)").c_str()
                        : ("; serving, " + humanb(live) + " read from host memory until it is moved").c_str());
    if (live > 0) start_promote(m);
    for (const std::string& p : group_partners(m.entry.name, Group::Kind::Together)) demand(p);
}

void Fleet::load_op(Model& m) {
    const std::string name = m.entry.name;
    const auto t0 = std::chrono::steady_clock::now();
    front_.loading(name);
    Config cfg = m.entry.cfg;
    if (cfg.proceed_incompatible) Engine::print_unsupported_banner();
    if (!resolve_hf_model(cfg)) {
        fail(m, "its repository could not be fetched (see the log)");
        return;
    }
    const InFlight in_flight(loads_in_flight_);
    settle_flex();
    { std::lock_guard<std::mutex> lk(flex_mu_); }   /* a growth already decided lands first */
    std::vector<int> chosen;
    std::string why;
    /* Its room on `chosen` until its start returns (Reservation); one per model. */
    auto reserve = [&](int64_t claim) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const Reservation& r : reserved_)
            if (r.m == &m) return;
        Reservation r;
        r.m = &m;
        r.cards = chosen;
        r.claim = claim;
        reserved_.push_back(r);
    };
    auto drop_reservation = [&] {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < reserved_.size(); ++i)
            if (reserved_[i].m == &m) { reserved_.erase(reserved_.begin() + (long)i); break; }
    };
    /* Its host pools, and a GiB for the load's staging windows and the small pinned buffers of
     * bring-up. */
    const int64_t pools = cfg.host_pool_mib * kMiB * std::max(cfg.tp, 1) + ((int64_t)1 << 30);
    struct Unreserve {
        std::atomic<int64_t>* r;
        std::atomic<int64_t>* mine;
        ~Unreserve() { r->fetch_sub(mine->exchange(0)); }
    } unreserve{ &host_reserve_, &m.pools_held };
    {
        std::unique_lock<std::recursive_mutex> room(room_mu_);
        if (!choose_cards(m, cfg, &chosen, &why)) {
            fail(m, why);
            return;
        }
        if (use_broker()) {
            /* The engine pins its own host pools at bring-up (an expert host pool on every rank,
             * a checkpoint store): parked copies make way first, so the host keeps its floor. A
             * load that cannot have that room does not start: pinning past the floor gets the
             * process killed by the kernel. The budget is what parks' copies may fill; a load is
             * held to the floor alone. A load in flight counts twice until it ends (its
             * reservation and the pools it has already pinned), so the room is waited for, for as
             * long as VRAM's: a model at work holds its memory until its work ends. This comes before
             * any VRAM is reserved: a load holding VRAM while it waits for host memory, and a parked
             * model pinning that memory while it waits for that VRAM, would wait for each other. */
            int64_t free_host = 0;
            const bool warm = m.warm.load();
            const int64_t host_tries = warm ? 1 : std::max<int64_t>(fc_.load_timeout_ms / 200, 1);
            for (int64_t i = 0; i < host_tries; ++i) {
                /* A load in flight counts twice, so beside one parked models go only once that
                 * has had time to pass. With none in flight the figure is exact and they go at
                 * once: a load does not wait ten seconds for memory an idle model pins. */
                const bool exact = !loading_beside(m);
                (void)make_host_room(pools, &m, !warm, !warm && (exact || i >= 50),
                                     !warm && (i >= 150 || (exact && i >= 10)));
                free_host = host_available() - host_reserve_.load() - kHostFloor;
                if (free_host >= pools || g_stop.load() || warm) break;
                /* Every other model is unloaded: the host has no more to give. */
                if (exact && i >= 10 && alone_on_host(m)) break;
                room.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                room.lock();
            }
            if (free_host < pools) {
                const std::string w = "it pins " + humanb(pools) + " of host memory and only " +
                                      humanb(std::max<int64_t>(free_host, 0)) +
                                      " can be made free above the host's floor";
                room.unlock();
                fail(m, w);
                return;
            }
            m.pools_held.store(pools);
            host_reserve_.fetch_add(pools);
        }
        /* Room for the guess before the start: kernels allocate while the model declares (a
         * collective's buffers), before the grant could make room for the exact claim. Room
         * another model's load or park holds for the moment is waited for, not failed on. */
        if (use_broker()) {
            int64_t guess = first_guess(m.claim, cfg);
            /* A tiered plan keeps most of a large container in host memory, so the container is
             * no guess of its share of a card: never more than nine tenths of the smallest card. */
            if (const int64_t smallest = smallest_card(chosen); smallest != INT64_MAX)
                guess = std::min(guess, smallest / 10 * 9);
            std::map<int, int64_t> need;
            for (int c : chosen) need[c] = guess;
            std::string w;
            room.unlock();
            /* Its guess is reserved as soon as the room is found, so a load starting beside it
             * does not choose the same room; the grant replaces it with the exact claim. */
            const int grc = make_room_waiting(m, need, &w, [&] { reserve(guess); },
                                              m.warm.load() ? 0 : -1);
            room.lock();
            if (grc < 0 && m.warm.load()) {
                room.unlock();
                fail(m, w);
                return;
            }
            /* No room found: reserved all the same, since it starts anyway. */
            reserve(guess);
        }
    }
    cfg.devices = chosen;
    cfg.report_once = m.loads > 0;
    trim_spares(chosen);
    reopen_queues(chosen);
    RAD_INFO("model %s: loading on card(s) %s", name.c_str(), cards_str(chosen).c_str());
    auto e = std::make_shared<Engine>(cfg);
    if (use_broker()) {
        m.broker = std::make_unique<LoadBroker>();
        m.broker->f = this;
        m.broker->m = &m;
        e->set_broker(m.broker.get());
    }
    e->set_front(&front_, m.entry.aliases);
    {
        std::lock_guard<std::mutex> lk(mu_);
        m.engine = e;   /* the grant reads its owner id */
    }
    const Raised starting(m.starting);
    std::map<int, int64_t> start_free;
    for (int c : chosen) start_free[c] = free_on(c);
    auto watch = std::make_unique<StallWatch>("startup", 120000);
    m.room_retry.store(false);
    const int s = e->start();
    if (s < 0) {
        watch.reset();
        /* Out of memory while other loads were starting beside it: the room it was granted was
         * someone's in flight, so it starts again rather than fail what is queued for it. */
        const bool beside = s == RAD_E_NOMEM && loads_in_flight_.load() > 1;
        const bool retry = (m.room_retry.exchange(false) || beside) && m.room_retries < 20 && !m.warm.load();
        if (!retry && !m.warm.load()) e->print_startup_report(s);
        drop_reservation();
        std::map<int, int64_t> held;
        for (int c : chosen) held[c] = dev_owner_bytes(e->owner(), c);
        {
            std::lock_guard<std::mutex> lk(mu_);
            m.engine.reset();
        }
        e.reset();
        wait_pages(start_free, held);
        if (retry) {
            ++m.room_retries;
            RAD_INFO("model %s: its room is held by loads that start after it; it starts again once "
                     "they are in", name.c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            demand(name);
            return;
        }
        fail(m, std::string("startup failed with ") + rad_strerror(s) + " (see the log)");
        return;
    }
    m.room_retries = 0;
    Engine* raw = e.get();
    m.ended = std::make_shared<std::atomic<bool>>(false);
    m.serve = std::thread([raw, done = m.ended, w = std::move(watch)]() mutable {
        const int rc = raw->serve(std::move(w));
        if (rc < 0 && !raw->startup_report_printed()) raw->print_startup_report(rc);
        done->store(true);
    });
    while (!raw->serving() && !m.ended->load() && !g_stop.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    drop_reservation();
    if (!raw->serving()) {
        e->request_stop(0);
        if (m.serve.joinable()) m.serve.join();
        {
            std::lock_guard<std::mutex> lk(mu_);
            m.engine.reset();
        }
        e.reset();
        front_.detach(name, 0);
        fail(m, "it stopped before it served (see the log)");
        return;
    }
    /* What it holds on each card: everything its engine allocated, never below the grant. */
    int64_t charge = 0;
    for (int c : chosen) charge = std::max(charge, dev_owner_bytes(raw->owner(), c));
    const double took = secs_since(t0);
    {
        std::lock_guard<std::mutex> lk(mu_);
        m.cards = chosen;
        m.last_cards = chosen;
        m.charge = std::max<int64_t>(charge, m.broker ? m.claim : 0);
        m.load_s = took;
        m.on_ns = rad_mono_ns();
        ++m.loads;
        m.error.clear();
        m.resident = true;
        m.parked = false;
        m.out.clear();
        m.seq = ++seq_;
    }
    RAD_INFO("model %s: loaded on card(s) %s in %.1f s, holding %s a card (paging owner %d)", name.c_str(),
             cards_str(chosen).c_str(), took, humanb(m.charge).c_str(), m.engine ? m.engine->owner() : 0);
    for (const std::string& p : group_partners(name, Group::Kind::Together)) demand(p);
}

/* Nothing waits and nothing moves: no request queued for a model, no load, restore or park
 * running, no background write. */
bool Fleet::quiet() {
    if (loads_in_flight_.load() > 0) return false;
    for (auto& mp : models_) {
        const Model& m = *mp;
        if (m.op.load() != kIdle) return false;
        if (m.spilling()) return false;
        const server::Front::Activity a = front_.activity(m.entry.name);
        if (a.waiting > 0) return false;
    }
    return true;
}

/* §8.6: the next model of the warm-up loads while the fleet is quiet. It stops for good once a
 * load would have nowhere to put what it displaces, or once a warm load made something unload. */
void Fleet::warm_tick() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!warming_.empty()) {
            Model* w = find(warming_);
            if (w && w->op.load() != kIdle) return;
            warming_.clear();
        }
        if (warm_.empty()) return;
    }
    int64_t unloads = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& mp : models_) unloads += mp->unloads;
    }
    auto stop = [&](const std::string& why) {
        RAD_INFO("warm-up: stops: %s", why.c_str());
        std::lock_guard<std::mutex> lk(mu_);
        warm_.clear();
    };
    if (warm_unloads_ >= 0 && unloads > warm_unloads_) return stop("a warm load made another model unload");
    if (!quiet()) return;
    for (;;) {
        std::string name;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (warm_.empty()) break;
            name = warm_.front();
            warm_.pop_front();
        }
        Model* m = find(name);
        if (!m || m->engine || m->entry.pin) continue;
        /* What it displaces needs no room elsewhere: a parked model's weights leave by an
         * unmap and come back from their container (dev_page_refilled), and its KV cache and
         * transients are dropped. A warm load that still made something unload ends the warm-up
         * above. */
        if (warm_unloads_ < 0) warm_unloads_ = unloads;
        m->warm_uses = front_.activity(name).uses;
        m->warm.store(true);
        size_t left;
        {
            std::lock_guard<std::mutex> lk(mu_);
            left = warm_.size();
        }
        RAD_INFO("warm-up: loading %s (%zu more after it)", name.c_str(), left);
        if (!start_op(*m, kLoad)) {
            m->warm.store(false);
            continue;
        }
        std::lock_guard<std::mutex> lk(mu_);
        warming_ = name;
        return;
    }
    RAD_INFO("warm-up: every model has been loaded once");
}

/* §8.6: a parked model with copies only on the drive gets host memory while the fleet is quiet,
 * the highest priority first, then the most recently used. */
void Fleet::fill_tick() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!warm_.empty() || !warming_.empty()) return;
    }
    if (rad_mono_ns() < fill_wait_ns_.load() || !quiet()) return;
    Model* best = nullptr;
    uint64_t best_used = 0;
    for (auto& mp : models_) {
        Model& m = *mp;
        if (!m.engine || !m.parked || m.filled < 0 || m.op.load() != kIdle) continue;
        if (m.spilling()) continue;
        if (dev_page_fillable(m.engine->owner()) <= 0) continue;
        const uint64_t used = front_.activity(m.entry.name).last_used_ns;
        if (!best || m.entry.priority > best->entry.priority ||
            (m.entry.priority == best->entry.priority && used > best_used)) {
            best = &m;
            best_used = used;
        }
    }
    if (best) (void)start_op(*best, kFill);
}

/* A live restore's rest that found no device memory (an allocation of another model's took it
 * meanwhile) is tried again every few seconds, while the model serves from host memory. */
void Fleet::promote_tick() {
    const uint64_t now = rad_mono_ns();
    for (auto& mp : models_) {
        Model& m = *mp;
        if (!m.engine || !m.resident || m.parked || m.op.load() != kIdle) continue;
        if (now - m.promote_ns < 5000000000ull) continue;
        if (m.spilling()) continue;
        if (dev_page_live_bytes(m.engine->owner()) <= 0) continue;
        bool in_place;
        std::vector<int> cards;
        {
            std::lock_guard<std::mutex> lk(mu_);
            in_place = m.in_place;
            cards = m.cards;
        }
        if (in_place && free_room(cards) < kPromoteChunk) continue;
        m.promote_ns = now;
        start_promote(m);
    }
}

/* Once a second. A busy model whose expert slab could grow and finds no free memory on its cards
 * (Engine::expert_flex_wanted) takes room from a neighbour that has been idle a long time: one
 * that may park to host memory, of no higher priority, outside its together group, with no
 * request for at least 30 s and thirty times what its way back costs. A park of only what the
 * slab wants, so the neighbour's next request pays a short restore and takes the room back first
 * (reclaim_flex). A grown slab speeds a tiered model's decode by up to a tenth (two R9700 cards),
 * so the bar is set where the neighbour's restore is rare against that. One neighbour at a time,
 * each asked at most once a minute. */
void Fleet::xflex_tick() {
    const uint64_t now = rad_mono_ns();
    if (now - xflex_ns_ < 1000000000ull) return;
    xflex_ns_ = now;
    if (loads_in_flight_.load() > 0) return;
    for (auto& mp : models_)
        if (mp->op.load() == kYield) return;
    for (auto& mp : models_) {
        Model& m = *mp;
        if (!m.engine || m.parked || !m.resident || m.op.load() != kIdle) continue;
        const int64_t w = m.engine->expert_flex_wanted();
        if (w <= 0) continue;
        if (front_.activity(m.entry.name).leases <= 0) continue;
        const std::vector<std::string> partners = group_partners(m.entry.name, Group::Kind::Together);
        /* Card lists change under mu_ on the workers (a load, an unload). */
        std::vector<std::vector<int>> cards(models_.size());
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (size_t i = 0; i < models_.size(); ++i) cards[i] = models_[i]->cards;
        }
        const std::vector<int>& mine = cards[(size_t)(&mp - &models_[0])];
        Model* pick = nullptr;
        double pick_idle = 0;
        for (auto& op : models_) {
            Model& o = *op;
            if (&o == &m || !o.engine || o.parked || !o.resident || o.op.load() != kIdle) continue;
            if (o.entry.pin || o.entry.priority > m.entry.priority) continue;
            if (o.entry.park.ram == ParkSpec::Ram::Never) continue;
            if (std::find(partners.begin(), partners.end(), o.entry.name) != partners.end()) continue;
            if (o.yield_ns && now - o.yield_ns < 60000000000ull) continue;
            bool shared = false;
            for (int c : cards[(size_t)(&op - &models_[0])])
                shared = shared || std::find(mine.begin(), mine.end(), c) != mine.end();
            if (!shared) continue;
            const server::Front::Activity a = front_.activity(o.entry.name);
            if (a.leases > 0 || a.waiting > 0) continue;
            const uint64_t since = std::max(a.last_used_ns, o.on_ns);
            const double idle = since && now > since ? (double)(now - since) * 1e-9 : 0;
            const double back = std::max(o.restore_s, o.trip_s);
            if (idle < std::max(30.0, 30.0 * back)) continue;
            if (!pick || idle > pick_idle) { pick = &o; pick_idle = idle; }
        }
        if (!pick) continue;
        {
            std::lock_guard<std::mutex> lk(mu_);
            pick->yield_want.clear();
            for (int c : pick->cards)
                if (std::find(mine.begin(), mine.end(), c) != mine.end()) pick->yield_want[c] = w;
            pick->yield_for = m.entry.name;
            pick->yield_ns = now;
        }
        if (start_op(*pick, kYield)) return;
    }
}

void Fleet::yield_op(Model& m) {
    std::map<int, int64_t> want;
    std::string who;
    {
        std::lock_guard<std::mutex> lk(mu_);
        want = m.yield_want;
        who = m.yield_for;
    }
    if (!m.engine || m.parked || want.empty()) return;
    /* A park that would cost more than reloading it is not worth a faster slab: it stays. */
    (void)park(m, want, "idle while " + who + "'s experts can use its room");
}

void Fleet::reap() {
    for (auto& mp : models_) {
        Model& m = *mp;
        if (m.op.load() != kIdle || !m.engine || !m.ended || !m.ended->load()) continue;
        if (m.serve.joinable()) m.serve.join();
        {
            std::lock_guard<std::mutex> lk(mu_);
            m.engine.reset();
        }
        front_.detach(m.entry.name, 0);
        fail(m, "it stopped while serving (see the log)");
    }
}

void Fleet::ttl_tick() {
    const uint64_t now = rad_mono_ns();
    for (auto& mp : models_) {
        Model& m = *mp;
        if (!m.engine || m.parked || m.entry.pin || m.entry.ttl_ms <= 0 || m.op.load() != kIdle) continue;
        const server::Front::Activity a = front_.activity(m.entry.name);
        if (a.leases > 0 || a.waiting > 0 || !a.last_used_ns ||
            (m.engine && m.engine->background_jobs() > 0)) continue;
        if (now - a.last_used_ns < (uint64_t)m.entry.ttl_ms * 1000000ull) continue;
        start_op(m, kEvict);
    }
}

std::string Fleet::reload() {
    FleetConfig nf;
    if (fleet::load(fc_.path, &nf) != RAD_OK) return "the file does not load (see the log)";
    if (const std::string u = unsupported(nf); !u.empty())
        return "this build does not act on " + u + " yet";
    const Config& a = fc_.models.front().cfg;
    const Config& b = nf.models.front().cfg;
    if (a.host != b.host || a.port != b.port || a.api_key != b.api_key ||
        a.http_threads != b.http_threads || a.cors != b.cors || nf.cards != fc_.cards)
        return "server: and cards: changes need a restart";
    if (nf.log_set) log_set_level(nf.log);
    /* Work in flight finishes first. */
    for (auto& m : models_) {
        while (m->op.load() != kIdle) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (m->worker.joinable()) m->worker.join();
    }
    std::vector<std::string> keep;
    for (const ModelEntry& e : nf.models) keep.push_back(e.name);
    for (size_t i = 0; i < models_.size();) {
        Model& m = *models_[i];
        if (std::find(keep.begin(), keep.end(), m.entry.name) != keep.end()) { ++i; continue; }
        unload(m, fc_.drain_timeout_ms, "removed from the file");
        front_.remove(m.entry.name);
        history_.forget(m.entry.name);
        RAD_INFO("model %s: removed", m.entry.name.c_str());
        std::lock_guard<std::mutex> lk(mu_);
        models_.erase(models_.begin() + (long)i);
    }
    for (const ModelEntry& e : nf.models) {
        Model* m = find(e.name);
        const bool required = e.pin || nf.models.size() == 1;
        if (!m) {
            auto nm = std::make_unique<Model>();
            nm->entry = e;
            {
                std::lock_guard<std::mutex> lk(mu_);
                models_.push_back(std::move(nm));
            }
            front_.add(e.name, e.aliases, nf.load_timeout_ms, required);
            RAD_INFO("model %s: added", e.name.c_str());
            continue;
        }
        if (m->entry.args != e.args) {
            unload(*m, fc_.drain_timeout_ms, "its configuration changed");
            RAD_INFO("model %s: reconfigured", e.name.c_str());
            std::lock_guard<std::mutex> lk(mu_);
            m->claim = 0;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            m->entry = e;
        }
        front_.update(e.name, e.aliases, nf.load_timeout_ms, required);
    }
    fc_ = std::move(nf);
    dev_stuck_exit_ms.store(fc_.stuck_exit_ms, std::memory_order_relaxed);
    for (const ModelEntry& e : fc_.models)
        if (e.pin) demand(e.name);
    RAD_INFO("reload: %s applied, %zu model(s)", fc_.path.c_str(), fc_.models.size());
    return "";
}

std::string Fleet::describe(const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& mp : models_) {
        const Model& m = *mp;
        if (m.entry.name != name) continue;
        json j;
        j["priority"] = m.entry.priority;
        j["pinned"] = m.entry.pin;
        j["tp"] = m.entry.cfg.tp;
        if (m.resident || m.parked) {
            j["cards"] = m.cards;
            const int64_t out = m.most_out();
            j["vram_bytes_per_card"] = std::max<int64_t>(m.charge - out, 0);
            if (m.parked) j["parked_bytes_per_card"] = out;
            if (m.engine && m.resident) {
                j["flex_bytes_per_card"] = m.engine->flex_bytes();
                j["expert_flex_bytes_per_card"] = m.engine->expert_flex_bytes();
            }
        }
        /* What it holds on each card now, whatever its state: its claim and what its cache grew
         * by while it serves (less the weights it reads in place from host memory), what a park
         * left of it on that card, or, while it loads, what it has backed so far. A page that adds
         * these up per card is left with what no model holds. */
        if (m.engine) {
            json by = json::object();
            const int owner = m.engine->owner();
            if (m.resident || m.parked) {
                for (int c : m.cards) {
                    int64_t n = 0;
                    if (m.parked) {
                        const auto it = m.out.find(c);
                        n = m.charge - (it == m.out.end() ? 0 : it->second);
                    } else {
                        n = m.charge + m.engine->flex_bytes() -
                            (m.in_place ? dev_page_live_bytes(owner, c) : 0);
                    }
                    by[std::to_string(c)] = std::max<int64_t>(n, 0);
                }
            } else {
                for (int c : cards_)
                    if (const int64_t n = dev_owner_bytes(owner, c); n > 0) by[std::to_string(c)] = n;
            }
            if (!by.empty()) j["vram_bytes_by_card"] = by;
        }
        j["parks"] = m.parks;
        j["restores"] = m.restores;
        if (m.park_s > 0) j["park_seconds"] = m.park_s;
        if (m.restore_s > 0) j["restore_seconds"] = m.restore_s;
        j["loads"] = m.loads;
        j["unloads"] = m.unloads;
        if (m.load_s > 0) j["load_seconds"] = m.load_s;
        if (m.entry.ttl_ms > 0) j["ttl_seconds"] = (double)m.entry.ttl_ms / 1000.0;
        if (!m.error.empty()) j["error"] = m.error;
        return j.dump();
    }
    return "{}";
}

/* What /admin/fleet says beyond describe(), which also feeds every /v1/models card: where the
 * model's bytes are, its file and the policy it was configured with. */
json Fleet::placement(const std::string& name) {
    static const char* const kOps[] = { "", "loading", "restoring", "leaving", "unloading", "filling",
                                        "yielding" };
    std::lock_guard<std::mutex> lk(mu_);
    json j = json::object();
    Model* m = find(name);
    if (!m) return j;
    const ModelEntry& e = m->entry;
    j["file"] = std::filesystem::path(e.cfg.model).filename().string();
    if (!e.aliases.empty()) j["aliases"] = e.aliases;
    if (const int op = m->op.load(); op > 0 && op <= kYield) j["doing"] = kOps[op];
    if (m->waiting.load()) j["waiting_for"] = "accelerator";
    else if (m->waiting_host.load()) j["waiting_for"] = "host memory";
    if (m->in_place) j["in_place"] = true;
    if (m->warm.load()) j["warm"] = true;
    if (m->charge > 0) j["charge_bytes_per_card"] = m->charge;
    if (m->engine) {
        int64_t dev = 0, host = 0;
        dev_page_stats(m->engine->owner(), &dev, &host);
        j["host_bytes"] = host;
        /* What it pins beside its copies while it has an engine: its host pools, its tables in
         * host memory and its prefix cache's host tier. */
        j["host_pinned_bytes"] = m->engine->host_pinned_bytes();
        if (m->parked) j["drive_bytes"] = dev_page_fillable(m->engine->owner());
    }
    j["park"] = {
        { "ram", e.park.ram == ParkSpec::Ram::Required ? "required"
                 : e.park.ram == ParkSpec::Ram::Never  ? "never" : "allowed" },
        { "restore", e.park.restore == ParkSpec::Restore::Full ? "full"
                     : e.park.restore == ParkSpec::Restore::Live ? "live" : "auto" },
        { "vram", e.park.vram == ParkSpec::Vram::Statics ? "statics"
                  : e.park.vram == ParkSpec::Vram::Mib   ? "mib" : "none" }
    };
    return j;
}

std::string Fleet::metrics() {
    std::string o;
    /* Read before mu_ is taken: the Front calls demand() under its own lock. */
    std::vector<std::pair<std::string, server::Front::Activity>> act;
    for (const std::string& name : front_.names()) act.emplace_back(name, front_.activity(name));
    o += "# HELP radiance_fleet_wait_seconds How long requests waited for their model to open (a "
         "load or a restore; 0 when it was open).\n# TYPE radiance_fleet_wait_seconds histogram\n";
    for (const auto& [name, a] : act) {
        const std::string lbl = "model=\"" + name + "\"";
        for (size_t k = 0; k < server::Front::kWaitLe.size(); ++k)
            o += "radiance_fleet_wait_seconds_bucket{" + lbl + ",le=\"" +
                 fmt("%g", server::Front::kWaitLe[k]) + "\"} " + std::to_string(a.wait_le[k]) + "\n";
        o += "radiance_fleet_wait_seconds_bucket{" + lbl + ",le=\"+Inf\"} " + std::to_string(a.waits) + "\n";
        o += "radiance_fleet_wait_seconds_sum{" + lbl + "} " + fmt("%.3f", a.wait_s) + "\n";
        o += "radiance_fleet_wait_seconds_count{" + lbl + "} " + std::to_string(a.waits) + "\n";
    }
    std::lock_guard<std::mutex> lk(mu_);
    auto family = [&](const char* name, const char* type, const char* help,
                      const std::function<std::string(const Model&)>& v) {
        o += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " " + type + "\n";
        for (auto& m : models_) o += std::string(name) + "{model=\"" + m->entry.name + "\"} " + v(*m) + "\n";
    };
    family("radiance_fleet_model_resident", "gauge", "Whether the model is serving (1) or not (0).",
           [](const Model& m) { return std::string(m.resident ? "1" : "0"); });
    family("radiance_fleet_model_parked", "gauge", "Whether the model is parked, in part or whole (1) or not (0).",
           [](const Model& m) { return std::string(m.parked ? "1" : "0"); });
    family("radiance_fleet_restore_seconds", "gauge", "How long the model's last return from parking took.",
           [](const Model& m) { return std::to_string(m.restore_s); });
    family("radiance_fleet_loads_total", "counter", "Loads of the model since the process started.",
           [](const Model& m) { return std::to_string(m.loads); });
    family("radiance_fleet_unloads_total", "counter", "Unloads of the model since the process started.",
           [](const Model& m) { return std::to_string(m.unloads); });
    family("radiance_fleet_parks_total", "counter", "Parks of the model, in part or whole.",
           [](const Model& m) { return std::to_string(m.parks); });
    family("radiance_fleet_load_failures_total", "counter", "Failed loads of the model.",
           [](const Model& m) { return std::to_string(m.failures); });
    family("radiance_fleet_load_seconds", "gauge", "How long the model's last load took.",
           [](const Model& m) { return std::to_string(m.load_s); });
    family("radiance_fleet_vram_bytes", "gauge", "VRAM the model holds on each of its cards.",
           [](const Model& m) {
               if (!m.resident && !m.parked) return std::string("0");
               return std::to_string(std::max<int64_t>(m.charge - m.most_out(), 0));
           });
    return o;
}

/* The history's sampler (server/history.h): once a second the fleet's status and the /stats of
 * every open model, read in-process as a dashboard read, which is not a use of the model. On its
 * own thread, so a load or a restore the scheduler is busy with leaves no hole in the charts.
 * While a page is being watched (it asked within the last three seconds) the same two are read
 * and kept five times a second, so its charts move as its tables do; unwatched, a row a second.
 * A model seen open for the first time since it closed gets a row at once, so one that came,
 * answered and left between two seconds still has its figures. */
void Fleet::history_loop() {
    /* At the ordinary priority, although about a millisecond of work a second would not be missed
     * at the lowest: a sample holds the front's lock for an instant, and a thread the kernel may
     * leave waiting for a core while it holds that lock would hold up every request's admission. */
    (void)pthread_setname_np(pthread_self(), "rad-history");
    using clock = std::chrono::steady_clock;
    auto next_row = clock::now();
    std::set<std::string> open;   /* the models that had a row since they last opened */
    while (!g_stop.load() && !front_.stopping()) {
        const double t = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        const bool watched = history_.watched(3.0);
        const bool due = clock::now() >= next_row;
        const bool row = due || watched;
        if (row) {
            const json f = status();
            history_.fleet_sample(f, t, f.dump());
        }
        for (const std::string& n : front_.names()) {
            const bool is_open = front_.activity(n).state == server::Front::State::Open;
            if (!is_open) {
                if (open.erase(n) || row) history_.model_closed(n);
                continue;
            }
            const bool first = !open.count(n);
            if (!row && !first) continue;
            std::string body = front_.peek(n, "/stats");
            if (body.empty()) continue;
            const json s = json::parse(body, nullptr, false);
            if (!s.is_object()) continue;
            history_.model_sample(n, s, t, std::move(body));
            open.insert(n);
        }
        if (due) {
            next_row += std::chrono::seconds(1);
            /* After a stall the next row is a second from now, not a burst to catch up. */
            if (clock::now() > next_row) next_row = clock::now() + std::chrono::seconds(1);
        }
        const auto tick = clock::now() + std::chrono::milliseconds(200);
        while (clock::now() < std::min(tick, next_row) && !g_stop.load() && !front_.stopping())
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

/* What /admin/fleet answers: every model with its state, bytes and policy, every card with its
 * telemetry, the groups, the host budget and the warm-up queue. */
json Fleet::status() {
    json models = json::array();
    for (const std::string& n : front_.names()) {
        json j = json::parse(describe(n), nullptr, false);
        if (!j.is_object()) j = json::object();
        const server::Front::Activity a = front_.activity(n);
        j["id"] = n;
        j["state"] = server::Front::state_name(a.state);
        j["in_flight"] = a.leases;
        j["waiting"] = a.waiting;
        j.update(placement(n));
        /* A model that is not serving and waits for room is shown as waiting, whatever it was
         * doing when the wait began (a load, a return from its park). */
        if (j.contains("waiting_for") && a.state != server::Front::State::Open) j["state"] = "waiting";
        if (const std::string caps = front_.capabilities(n); !caps.empty()) j["capabilities"] = json::parse(caps, nullptr, false);
        models.push_back(j);
    }
    json cards = json::array();
    for (int c : cards_) {
        RadDeviceProps p{};
        const bool known = rad_dev_props(c, &p) == RAD_OK;
        json o = { { "card", c }, { "free_bytes", known ? p.vram_free : 0 } };
        if (known) {
            o["vram_bytes"] = p.vram_bytes;
            o["name"] = p.name;
        }
        int q_open = 0, q_idle = 0;
        dev_stream_counts(c, &q_open, &q_idle);
        o["queues"] = q_open;
        o["queues_idle"] = q_idle;
        o["queue_bytes"] = (int64_t)q_open * kQueueBytes;
        card_telemetry(c, o);
        cards.push_back(std::move(o));
    }
    json groups = json::array();
    for (const Group& g : fc_.groups)
        groups.push_back({ { "name", g.name },
                           { "kind", g.kind == Group::Kind::Together ? "together" : "exclusive" },
                           { "members", g.members } });
    /* `held_bytes` is all the process pins now (pools, tables, cache tiers, snapshot stores and
     * the copies), without what loads in flight have reserved: a load's pools are in its
     * reservation and, as it pins them, in the total as well, so the ledger's own figure reads a
     * load's pools twice until it ends. `copies_bytes` is the parked and kept copies alone. */
    json host = { { "budget_bytes", host_budget_ }, { "copies_bytes", dev_page_host_bytes() },
                  { "held_bytes", host_held() - host_reserve_.load() } };
    if (!fc_.disk_tier.empty()) host["disk_tier"] = fc_.disk_tier;
    json warm = json::object();
    {
        std::lock_guard<std::mutex> lk(mu_);
        warm["queued"] = warm_.size();
        warm["loading"] = warming_.empty() ? json() : json(warming_);
        int64_t fill = 0;
        for (auto& m : models_)
            if (m->engine && m->parked && m->filled >= 0) fill += dev_page_fillable(m->engine->owner());
        warm["drive_only_bytes"] = fill;
    }
    return json{ { "version", RAD_VERSION }, { "uptime_s", secs_since(started_) }, { "models", models },
                { "cards", cards }, { "groups", groups }, { "host", host }, { "warm_up", warm } };
}

void Fleet::add_routes() {
    /* The fleet's own routes. Without a bearer key they answer loopback callers only, as the
     * engine's other admin routes do. */
    auto guard = [this](const server::HttpRequest& q) -> std::optional<server::HttpResponse> {
        if (!fc_.models.front().cfg.api_key.empty() || q.peer_is_loopback()) return std::nullopt;
        return server::HttpResponse::json_body(
            403, server::error_body(q.path, 403, "permission_error",
                                    "the fleet's admin routes answer loopback callers only "
                                    "unless an api key is configured"));
    };
    auto ok = [](json j) { return server::HttpResponse::json_body(200, j.dump()); };
    auto bad = [](const server::HttpRequest& q, int st, const std::string& m) {
        return server::HttpResponse::json_body(st, server::error_body(q.path, st, "invalid_request_error", m));
    };
    front_.route("GET", "/admin/fleet", [this, guard, ok](const server::HttpRequest& q) {
        if (auto g = guard(q)) return *g;
        return ok(status());
    });
    /* The dashboard's history (server/history.h): with `model` the hour of that model's figures,
     * its finished requests and, as `live`, its latest /stats object; without it the fleet's, with
     * /admin/fleet's object as `live`. `since` (unix seconds) asks for what came after. Every byte
     * comes from the sampler's last read, so a page that polls this reads nothing of the engine:
     * it neither brings a model back nor costs a running one anything. A model's history is as
     * open as its /stats, the fleet's as /admin/fleet. */
    front_.route("GET", "/history", [this, guard, ok](const server::HttpRequest& q) {
        double since = 0;
        if (auto it = q.query.find("since"); it != q.query.end()) since = std::strtod(it->second.c_str(), nullptr);
        if (auto it = q.query.find("model"); it != q.query.end() && !it->second.empty())
            return server::HttpResponse::json_body(200, history_.model_body(it->second, since));
        if (auto g = guard(q)) return *g;
        return server::HttpResponse::json_body(200, history_.fleet_body(since));
    });
    /* A command run by the scheduler thread, its answer awaited. */
    auto run_cmd = [this](std::function<std::string()> f, int64_t wait_ms) -> std::pair<bool, std::string> {
        auto p = std::make_shared<std::promise<std::string>>();
        std::future<std::string> fu = p->get_future();
        post([p, f] { p->set_value(f()); });
        if (fu.wait_for(std::chrono::milliseconds(wait_ms)) != std::future_status::ready)
            return { false, "still running" };
        return { true, fu.get() };
    };
    front_.route("POST", "/admin/models/:name/:action",
                 [this, guard, ok, bad, run_cmd](const server::HttpRequest& q) {
        if (auto g = guard(q)) return *g;
        const std::string name = q.param("name"), action = q.param("action");
        bool known = false;
        for (const std::string& n : front_.names()) known |= n == name;
        if (!known) return bad(q, 404, "no model '" + name + "' is configured");
        if (action == "load") {
            demand(name);
            /* Waits for the load unless ?wait=0. */
            if (q.query.count("wait") && q.query.at("wait") == "0")
                return server::HttpResponse::json_body(202, json({ { "id", name }, { "state", "loading" } }).dump());
            const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(fc_.load_timeout_ms);
            for (;;) {
                const server::Front::Activity a = front_.activity(name);
                if (a.state == server::Front::State::Open) break;
                if (a.state == server::Front::State::Failed) {
                    std::lock_guard<std::mutex> lk(mu_);
                    Model* m = find(name);
                    return bad(q, 503, "it did not load: " + (m ? m->error : std::string("?")));
                }
                if (std::chrono::steady_clock::now() > until) return bad(q, 503, "it did not load in time");
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return ok({ { "id", name }, { "state", "resident" } });
        }
        if (action == "unload" || action == "park") {
            const bool unl = action == "unload";
            const auto r = run_cmd([this, name, unl] {
                Model* m = find(name);
                if (!m || !m->engine) return std::string(unl ? "" : "not loaded");
                if (m->entry.pin) return std::string("pinned");
                for (int i = 0; i < 6000 && m->op.load() != kIdle; ++i)
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                int expect = kIdle;
                if (!m->op.compare_exchange_strong(expect, unl ? kUnload : kEvict)) return std::string("busy");
                std::string res;
                if (unl) unload(*m, fc_.drain_timeout_ms, "asked by the admin route");
                else if (!m->parked && park(*m, {}, "asked by the admin route") < 0) res = "could not park (see the log)";
                m->op.store(kIdle);
                return res;
            }, fc_.drain_timeout_ms + 120000);
            if (!r.first)
                return server::HttpResponse::json_body(202, json({ { "id", name }, { "state", unl ? "unloading" : "parking" } }).dump());
            if (r.second == "pinned") return bad(q, 409, "model '" + name + "' is pinned; unpin it first");
            if (!r.second.empty()) return bad(q, 409, "model '" + name + "': " + r.second);
            return ok({ { "id", name }, { "state", unl ? "cold" : "parked" } });
        }
        if (action == "pin" || action == "unpin") {
            const bool pin = action == "pin";
            run_cmd([this, name, pin] {
                std::lock_guard<std::mutex> lk(mu_);
                if (Model* m = find(name)) m->entry.pin = pin;
                return std::string();
            }, 600000);
            if (pin) demand(name);
            return ok({ { "id", name }, { "pinned", pin } });
        }
        return bad(q, 404, "no admin action '" + action + "'; load, park, unload, pin or unpin");
    });
    front_.route("POST", "/admin/reload", [this, guard, ok, bad, run_cmd](const server::HttpRequest& q) {
        if (auto g = guard(q)) return *g;
        const auto r = run_cmd([this] { return reload(); }, 3600000);
        if (!r.second.empty()) return bad(q, 400, "reload refused: " + r.second);
        return ok({ { "reloaded", fc_.path } });
    });
}

void Fleet::loop() {
    std::vector<std::string> waiting;   /* demanded while busy: asked again each round */
    while (!g_stop.load() && !front_.stopping()) {
        std::deque<std::function<void()>> cmds;
        std::vector<std::string> want;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(100), [&] {
                return !cmds_.empty() || !demands_.empty() || g_stop.load() || g_hup.load();
            });
            cmds.swap(cmds_);
            want.swap(demands_);
        }
        if (g_hup.exchange(false)) {
            const std::string r = reload();
            if (!r.empty()) RAD_ERR("reload refused: %s", r.c_str());
        }
        for (auto& f : cmds) f();
        for (const std::string& n : waiting)
            if (std::find(want.begin(), want.end(), n) == want.end()) want.push_back(n);
        waiting.clear();
        for (const std::string& n : want) {
            if (g_stop.load()) break;
            Model* m = find(n);
            if (!m) continue;
            if (m->engine && !m->parked) continue;   /* serving already */
            /* Every model's work runs on its own worker: a restore never waits behind a load. */
            if (!start_op(*m, m->engine && m->parked ? kRestore : kLoad)) waiting.push_back(n);
        }
        for (auto& m : models_)
            if (m->op.load() == kIdle && m->worker.joinable()) m->worker.join();
        reap();
        ttl_tick();
        /* A request for a warm model makes it an ordinary one. */
        for (auto& m : models_)
            if (m->warm.load()) {
                const server::Front::Activity a = front_.activity(m->entry.name);
                if (a.leases > 0 || a.waiting > 0 || a.uses != m->warm_uses) m->warm.store(false);
            }
        warm_tick();
        fill_tick();
        promote_tick();
        xflex_tick();
    }
}

void Fleet::stop_all() {
    for (auto& m : models_) {
        while (m->op.load() != kIdle) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (m->worker.joinable()) m->worker.join();
        join_spill(*m);
    }
    for (auto& m : models_)
        if (m->engine) m->engine->request_stop(fc_.drain_timeout_ms);
    for (auto& m : models_)
        if (m->serve.joinable()) m->serve.join();
    front_.stop();
    /* Reverse load order: a later engine may share the plugin set an earlier one loaded. */
    std::vector<Model*> order;
    for (auto& m : models_)
        if (m->engine) order.push_back(m.get());
    std::sort(order.begin(), order.end(), [](Model* a, Model* b) { return a->seq > b->seq; });
    for (Model* m : order) {
        std::lock_guard<std::mutex> lk(mu_);
        m->engine.reset();
    }
    sweep_buffers_keep(false);
}

int Fleet::run() {
    std::vector<int> usable;
    if (Engine::usable_cards(&usable) < 0) {
        RAD_ERR("no usable card");
        return 1;
    }
    cards_ = usable;
    if (!fc_.cards.empty()) {
        cards_.clear();
        for (int c : fc_.cards) {
            if (std::find(usable.begin(), usable.end(), c) == usable.end()) {
                RAD_ERR("%s: cards lists %d, which is not a usable card here", fc_.path.c_str(), c);
                return 2;
            }
            cards_.push_back(c);
        }
    }
    for (const ModelEntry& e : fc_.models)
        if (std::max(e.cfg.tp, 1) > (int)cards_.size()) {
            RAD_ERR("models.%s: tp %d needs %d cards; %zu are available", e.name.c_str(), e.cfg.tp,
                    e.cfg.tp, cards_.size());
            return 2;
        }

    /* Park files a process that died left in the disk tier: radiance-park-PID-N.bin (pageable
     * chunks) and radiance-vmem-PID-N.bin (elastic ranges) of a PID that no longer runs. */
    if (!fc_.disk_tier.empty()) {
        namespace fs = std::filesystem;
        std::error_code ec;
        for (const auto& f : fs::directory_iterator(fc_.disk_tier, ec)) {
            const std::string n = f.path().filename().string();
            long long pid = 0;
            if ((std::sscanf(n.c_str(), "radiance-park-%lld-", &pid) == 1 ||
                 std::sscanf(n.c_str(), "radiance-vmem-%lld-", &pid) == 1) && pid > 0 &&
                kill((pid_t)pid, 0) != 0 && errno == ESRCH)
                fs::remove(f.path(), ec);
        }
    }
    struct sigaction sa {};
    sa.sa_handler = on_stop;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;      /* a second one takes the default disposition and kills */
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    struct sigaction sh {};
    sh.sa_handler = on_hup;
    sigemptyset(&sh.sa_mask);
    sigaction(SIGHUP, &sh, nullptr);

    /* Several models share the log: each line carries the model it is about. */
    if (fc_.models.size() > 1) log_model_tags(true);

    /* Pinned models, and the model of a one-model file, load before the listener opens; every
     * other model loads when a request first names it. */
    std::vector<std::string> preload;
    for (const ModelEntry& e : fc_.models) {
        const bool required = e.pin || fc_.models.size() == 1;
        front_.add(e.name, e.aliases, fc_.load_timeout_ms, required);
        if (required) preload.push_back(e.name);
    }
    front_.set_demand([this](const std::string& n) { demand(n); });
    front_.set_describe([this](const std::string& n) { return describe(n); });
    front_.set_metrics([this] { return metrics(); });
    add_routes();
    RAD_INFO("fleet: %zu model(s) on card(s) %s; %s", fc_.models.size(), cards_str(cards_).c_str(),
             preload.empty() ? "each loads when a request first names it"
                             : (std::to_string(preload.size()) + " load now, the rest on demand").c_str());

    {
        std::lock_guard<std::recursive_mutex> room(room_mu_);
        reopen_queues(cards_);
    }
    sweep_buffers_keep(fc_.models.size() > 1);

    size_t up = 0;
    for (const std::string& n : preload) {
        if (g_stop.load()) break;
        Model* m = find(n);
        if (!m) continue;
        m->op.store(kLoad);
        load_op(*m);
        m->op.store(kIdle);
        if (m->engine) ++up;
    }
    if (g_stop.load() || (!preload.empty() && up == 0)) {
        if (!g_stop.load()) RAD_ERR("no model started");
        stop_all();
        return g_stop.load() ? 0 : 1;
    }

    /* §8.6: the rest load in the background, lowest priority first (later in the file first among
     * equals), so the models that matter most are the ones still on the cards at the end. */
    if (fc_.warm_up && models_.size() > 1) {
        std::vector<size_t> order;
        for (size_t i = 0; i < models_.size(); ++i)
            if (!models_[i]->entry.pin && !models_[i]->engine) order.push_back(i);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const int pa = models_[a]->entry.priority, pb = models_[b]->entry.priority;
            return pa != pb ? pa < pb : a > b;
        });
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i : order) warm_.push_back(models_[i]->entry.name);
    }

    std::atomic<int> listen_rc{RAD_OK};
    std::thread listener([this, &listen_rc] {
        listen_rc.store(front_.run());
        if (listen_rc.load() < 0) g_stop.store(true);
    });
    std::thread history([this] { history_loop(); });
    loop();
    RAD_INFO("stopping: draining in-flight requests");
    if (history.joinable()) history.join();
    stop_all();
    if (listener.joinable()) listener.join();
    return listen_rc.load() < 0 ? 1 : 0;
}

}  // namespace

int run(FleetConfig& fc) {
    Fleet f(std::move(fc));
    return f.run();
}

}  // namespace rad::fleet
