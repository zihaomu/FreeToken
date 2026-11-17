/* engine_lora.cpp -- runtime LoRA on the engine side: which modules adapters may target (fixed
 * before declare), and loading an adapter into a slot of every rank's pools.
 *
 * A load packs the adapter for each rank (lora::load_adapter, host only), refuses a module no rank
 * placed, copies each rank's bytes into the slot on a stream of its own, waits, and only then
 * publishes the name. A request can name the adapter only after the publish, so no step reads a
 * slot while it is written.
 *
 * A generator keeps its adapter files in a library instead (--lora, --lora-dir, the load
 * endpoint), and a slot holds one combination of them at given strengths, named by the
 * combination: the first request that asks for it composes it (lora/adapter.h) into a free slot,
 * or the least recently used one no request holds, and later ones find it there. */
#include "engine_priv.h"
#include "lora/lora.h"

#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <thread>

namespace rad {

namespace {

/* adapter_config.json's target_modules as leaf names. PEFT also accepts a regex string and
 * "all-linear"; neither names leaves this engine can check before declare, so both are refused
 * with the flag that does. */
int read_targets(const std::string& dir, std::set<std::string>* out, std::string* err) {
    const std::string path = dir + "/adapter_config.json";
    std::ifstream f(path);
    if (!f) { *err = "cannot read " + path; return RAD_E_NOTFOUND; }
    Json j;
    try { f >> j; } catch (const std::exception& e) {
        *err = path + ": " + e.what();
        return RAD_E_INVAL;
    }
    const auto it = j.find("target_modules");
    if (it == j.end() || it->is_null()) { *err = path + " names no target_modules"; return RAD_E_INVAL; }
    if (it->is_string()) {
        *err = path + ": target_modules is '" + it->get<std::string>() +
               "', not a list of module names; give them with --lora-target-modules";
        return RAD_E_UNSUPPORTED;
    }
    if (!it->is_array()) { *err = path + ": target_modules is not a list"; return RAD_E_INVAL; }
    for (const auto& m : *it) {
        if (!m.is_string()) { *err = path + ": a target_modules entry is not a string"; return RAD_E_INVAL; }
        std::string s = m.get<std::string>();
        const size_t dot = s.rfind('.');
        if (dot != std::string::npos) s = s.substr(dot + 1);
        if (!s.empty()) out->insert(s);
    }
    return RAD_OK;
}

/* A generator's adapter name: a file stem under --lora-dir, possibly in a subdirectory, never
 * leaving it. */
bool valid_name(const std::string& n) {
    if (n.empty() || n.size() > 200 || n[0] == '/' || n[0] == '.') return false;
    if (n.find("..") != std::string::npos || n.find("//") != std::string::npos) return false;
    for (char c : n)
        if (!(std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.' || c == '/' ||
              c == ' ' || c == '(' || c == ')' || c == '+' || c == ','))
            return false;
    return true;
}

bool is_file(const std::string& p) {
    struct stat sb{};
    return stat(p.c_str(), &sb) == 0 && S_ISREG(sb.st_mode);
}

/* The combination's slot name: the adapters sorted by name, each at its strength. */
std::string combo_key(const std::vector<Engine::LoraUse>& uses) {
    std::string k;
    for (const Engine::LoraUse& u : uses) k += fmt("%s%s*%.6g", k.empty() ? "" : "+", u.name.c_str(), u.weight);
    return k;
}

}  /* namespace */

int Engine::lora_targets() {
    if (!cfg_.lora.enabled()) return RAD_OK;
    lora_targets_ = std::set<std::string>(cfg_.lora.targets.begin(), cfg_.lora.targets.end());
    if (lora_targets_.empty()) {
        /* A PEFT directory names its targets; an adapter file (a generator's) and a directory of
         * them do not, and a generator's default is every plain GEMM under its declared roots. A
         * text model refuses "*" at declare. */
        for (const auto& [name, dir] : cfg_.lora.adapters) {
            if (is_file(dir)) { lora_targets_.insert("*"); continue; }
            std::string err;
            const int s = read_targets(dir, &lora_targets_, &err);
            if (s < 0) {
                RAD_ERR("lora: adapter '%s': %s", name.c_str(), err.c_str());
                return s;
            }
        }
        if (cfg_.lora.adapters.empty()) lora_targets_.insert("*");
    }
    lora_table_.reset(cfg_.lora.slots(), &lora_epoch_);
    return RAD_OK;
}

int Engine::lora_start() {
    if (!cfg_.lora.enabled()) return RAD_OK;
    lora_gen_ = ranks_[0]->program.is_pass_driven();
    if (!lora_gen_ && cfg_.lora.slots() < (int)cfg_.lora.adapters.size()) {
        RAD_ERR("lora: --max-loras %d holds fewer adapters than the %zu that --lora names",
                cfg_.lora.slots(), cfg_.lora.adapters.size());
        return RAD_E_INVAL;
    }
    size_t n_sites = 0;
    for (const auto& r : ranks_) n_sites += r->program.lora_sites.size();
    if (n_sites == 0) {
        std::string t;
        for (const std::string& x : lora_targets_) t += (t.empty() ? "" : ",") + x;
        RAD_ERR("lora: no GEMM of this model multiplies a weight of the targeted modules (%s)",
                t.c_str());
        return RAD_E_UNSUPPORTED;
    }
    /* Every target is a module some rank carries adapters on. A leaf the model lacks, or holds
     * only outside the trunk (embedding, lm_head, a tower, the drafter), would target nothing. */
    if (!lora_targets_.count("*")) {
        std::set<std::string> carried;
        for (const auto& r : ranks_)
            for (const lora::Site& site : r->lora_sites)
                for (const lora::SiteSource& src : site.sources)
                    if (src.targeted) carried.insert(src.module.substr(src.module.rfind('.') + 1));
        for (const std::string& x : lora_targets_)
            if (!carried.count(x)) {
                RAD_ERR("lora: the target module '%s' is no trunk linear of this model, so no "
                        "adapter could land on it; leave it out of --lora-target-modules (or of "
                        "the adapters' target_modules)", x.c_str());
                return RAD_E_UNSUPPORTED;
            }
    }
    /* The adapter ops run in their GEMM's domain and read the pools on the card, so a targeted
     * GEMM placed on the host could serve no adapter request. On the host backend every op and
     * the pools are in host memory. */
    if (!device_is_host())
        for (const auto& r : ranks_)
            for (const LoraIssueSite& site : r->program.lora_sites)
                if (r->ctx.op_domain((rad_op)site.gemm) != RAD_DOMAIN_DEVICE) {
                    RAD_ERR("lora: rank %d: the GEMM over '%s' is placed on the host, and its "
                            "adapter ops read pools on the card; keep the targeted weights in "
                            "VRAM or leave the module out of --lora-target-modules", r->index,
                            r->program.lora_pools[(size_t)site.pool].weight.c_str());
                    return RAD_E_UNSUPPORTED;
                }
    {
        const Program& P = ranks_[0]->program;
        int64_t slot_bytes = 0;
        for (const LoraPool& p : P.lora_pools) slot_bytes += 2 * p.R * p.K + 2 * p.N * p.R + 4 * p.N;
        std::string t;
        for (const std::string& x : lora_targets_) t += (t.empty() ? "" : ",") + x;
        RAD_INFO("lora: %zu GEMMs over %zu weights carry adapters on rank 0 (%s); %d slots of rank "
                 "%d, %s a slot, %s on the card", P.lora_sites.size(), P.lora_pools.size(), t.c_str(),
                 cfg_.lora.slots(), cfg_.lora.rank(lora_gen_), humanb(slot_bytes).c_str(),
                 humanb(slot_bytes * cfg_.lora.slots()).c_str());
    }
    if (lora_gen_) return gen_lora_start();
    for (const auto& [name, dir] : cfg_.lora.adapters) {
        std::string err;
        const int s = lora_load(name, dir, &err);
        if (s < 0) {
            RAD_ERR("lora: adapter '%s' (%s): %s", name.c_str(), dir.c_str(), err.c_str());
            return s;
        }
    }
    return RAD_OK;
}

int Engine::lora_load(const std::string& name, const std::string& dir, std::string* err) {
    std::lock_guard<std::mutex> lk(lora_mu_);
    if (!cfg_.lora.enabled()) { *err = "the engine was started without adapters"; return RAD_E_STATE; }
    if (name.empty()) { *err = "an adapter needs a name"; return RAD_E_INVAL; }
    if (lora_gen_) {
        /* A generator's adapter goes into the library; a slot takes it when a request asks. */
        if (!valid_name(name)) { *err = "'" + name + "' is not an adapter name"; return RAD_E_INVAL; }
        {
            std::lock_guard<std::mutex> lk2(lora_lib_mu_);
            if (lora_lib_.count(name)) { *err = "an adapter named '" + name + "' is already loaded"; return RAD_E_INVAL; }
        }
        auto f = std::make_shared<lora::AdapterFile>();
        RAD_TRY(f->open(dir, err));
        int placed = 0, rank = 0;
        std::string first;
        for (const lora::FileModule& m : f->modules()) {
            std::string why;
            if (!lora_keys_->resolve(m.key, &why).empty()) { ++placed; continue; }
            if (first.empty()) first = m.key + ": " + why;
        }
        for (const lora::FileModule& m : f->modules())
            if (m.kind == lora::FileModule::LORA && m.t[0]) rank = std::max(rank, (int)m.t[0]->shape[0]);
        if (placed == 0) {
            *err = fmt("none of its %zu modules is a module of this model (%s)", f->modules().size(),
                       first.c_str());
            return RAD_E_NOTFOUND;
        }
        RAD_INFO("lora: '%s' -- %s, %d of %zu modules on this model%s%s%s", name.c_str(),
                 f->kind_name(), placed, f->modules().size(),
                 rank ? fmt(", rank %d", rank).c_str() : "", first.empty() ? "" : "; not placed: ",
                 first.c_str());
        std::lock_guard<std::mutex> lk2(lora_lib_mu_);
        lora_lib_[name] = { dir, f };
        return RAD_OK;
    }
    /* A slot that is still cooling (lora/slots.h) is waited for here rather than answered with a
     * retry: the step loop goes round idle or busy, so the wait is milliseconds, and a hot swap
     * (unload, then load at once) would otherwise fail under traffic. Only other loads wait on
     * lora_mu_ meanwhile; requests and unloads use the table alone. */
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    int slot = -1;
    for (;;) {
        bool cooling = false;
        const int s = lora_table_.reserve(name, &slot, &cooling, err);
        if (s >= 0) break;
        if (s != RAD_E_FULL || !cooling || stopping_.load(std::memory_order_acquire) ||
            std::chrono::steady_clock::now() >= give_up)
            return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    /* ---- pack, per rank (host only) */
    std::vector<lora::AdapterInfo> infos(ranks_.size());
    std::vector<std::vector<lora::PackedSite>> packed(ranks_.size());
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        const int s = lora::load_adapter(dir, ranks_[ri]->lora_sites, cfg_.lora.rank(false), &infos[ri],
                                         &packed[ri], err);
        if (s < 0) {
            if (ranks_.size() > 1) *err = fmt("rank %zu: ", ri) + *err;
            return s;
        }
    }
    {
        const int s = lora::check_placed(infos, err, &ranks_[0]->program, &lora_targets_);
        if (s < 0) return s;
    }

    RAD_TRY(lora_copy(slot, packed, err));

    lora_table_.publish(slot, name, dir);
    const lora::AdapterInfo& a = infos[0];
    RAD_INFO("lora: '%s' in slot %d -- %zu modules, rank %lld, alpha %g%s, base %s", name.c_str(),
             slot, a.modules.size(), (long long)a.max_module_rank, a.alpha, a.rslora ? " (rsLoRA)" : "",
             a.base_model.empty() ? "(unnamed)" : a.base_model.c_str());
    return RAD_OK;
}

int Engine::lora_copy(int slot, const std::vector<std::vector<lora::PackedSite>>& packed,
                      std::string* err) {
    /* Each rank on a stream of its own: the compute stream may be recording a pass. */
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        Rank& r = *ranks_[ri];
        const Program& P = r.program;
        if (packed[ri].size() != P.lora_pools.size()) {
            *err = fmt("rank %zu: the adapter was packed for %zu pools, the program has %zu", ri,
                       packed[ri].size(), P.lora_pools.size());
            return RAD_E_STATE;
        }
        if (rad_dev_set(r.device) < 0) { *err = fmt("rank %zu: cannot bind its device", ri); return RAD_E_DEVICE; }
        RadStream st{};
        if (rad_stream_create(&st, 0) < 0) { *err = fmt("rank %zu: no copy stream", ri); return RAD_E_DEVICE; }
        int rc = RAD_OK;
        for (size_t p = 0; p < P.lora_pools.size() && rc >= 0; ++p) {
            const LoraPool& lp = P.lora_pools[p];
            const lora::PackedSite& ps = packed[ri][p];
            const int64_t ab = 2 * lp.R * lp.K, bb = 2 * lp.N * lp.R, sb = 4 * lp.N;
            char* da = (char*)P.buffers[lp.a].ptr;
            char* db = (char*)P.buffers[lp.b].ptr;
            char* ds = (char*)P.buffers[lp.scale].ptr;
            if (!da || !db || !ds) {
                rc = RAD_E_STATE;
                *err = "the adapter pool of '" + lp.weight + "' has no buffer address";
                break;
            }
            da += (int64_t)slot * ab; db += (int64_t)slot * bb; ds += (int64_t)slot * sb;
            /* A pool this adapter does not touch is zeroed, not left as it was: the bytes are what
             * the slot's last adapter or the arena held, and a scale of 0 times a NaN is a NaN. */
            if (ps.empty) {
                if (rad_memset_async(da, 0, ab, st) < 0 || rad_memset_async(db, 0, bb, st) < 0 ||
                    rad_memset_async(ds, 0, sb, st) < 0)
                    rc = RAD_E_DEVICE;
                continue;
            }
            if ((int64_t)ps.a.size() * 2 != ab || (int64_t)ps.b.size() * 2 != bb ||
                (int64_t)ps.scale.size() * 4 != sb) {
                *err = fmt("rank %zu: pool '%s' was packed as %zu/%zu/%zu elements, its slot holds "
                           "%lld/%lld/%lld", ri, lp.weight.c_str(), ps.a.size(), ps.b.size(),
                           ps.scale.size(), (long long)(ab / 2), (long long)(bb / 2), (long long)(sb / 4));
                rc = RAD_E_STATE;
                break;
            }
            if (rad_memcpy_async(da, ps.a.data(), ab, st) < 0 ||
                rad_memcpy_async(db, ps.b.data(), bb, st) < 0 ||
                rad_memcpy_async(ds, ps.scale.data(), sb, st) < 0)
                rc = RAD_E_DEVICE;
        }
        const int sy = rad_stream_sync(st);
        rad_stream_destroy(st);
        if (rc >= 0 && sy < 0) rc = RAD_E_DEVICE;
        if (rc < 0) {
            if (err->empty()) *err = fmt("rank %zu: the copy into slot %d failed", ri, slot);
            return rc;
        }
    }
    return RAD_OK;
}

int Engine::lora_acquire(const std::string& name, int32_t* slot, uint64_t* uid,
                         std::shared_ptr<void>* hold) {
    return lora_table_.acquire(name, slot, uid, hold);
}

int Engine::lora_unload(const std::string& name, std::string* err, int* in_flight) {
    if (!lora_gen_) return lora_table_.unload(name, in_flight, err);
    /* A generator's adapter leaves the library, and every combination holding it its slot (at
     * once when idle, after its requests otherwise). */
    {
        std::lock_guard<std::mutex> lk(lora_lib_mu_);
        if (!lora_lib_.erase(name)) {
            *err = "no adapter named '" + name + "' is loaded";
            return RAD_E_NOTFOUND;
        }
    }
    *in_flight = 0;
    for (const std::string& k : lora_table_.names()) {
        bool uses = false;
        size_t at = 0;
        while (at <= k.size()) {
            const size_t plus = std::min(k.find('+', at), k.size());
            const std::string part = k.substr(at, plus - at);
            if (part.substr(0, part.rfind('*')) == name) uses = true;
            at = plus + 1;
        }
        if (!uses) continue;
        int n = 0;
        std::string e;
        lora_table_.unload(k, &n, &e);
        *in_flight += n;
    }
    return RAD_OK;
}

std::vector<std::string> Engine::lora_names() const {
    return lora_gen_ ? gen_lora_names() : lora_table_.names();
}

/* ================================================================== generator adapters */
int Engine::gen_lora_start() {
    /* Every slot's run of the slot-row table holds its own index (Program::lora_rows). */
    for (auto& rp : ranks_) {
        Rank& r = *rp;
        const Program& P = r.program;
        if (!P.lora_rows || P.lora_rows_n <= 0) continue;
        const int S = cfg_.lora.slots();
        std::vector<int32_t> v((size_t)(S * P.lora_rows_n));
        for (int s = 0; s < S; ++s)
            std::fill(v.begin() + (int64_t)s * P.lora_rows_n, v.begin() + (int64_t)(s + 1) * P.lora_rows_n, s);
        void* d = P.buffers[P.lora_rows].ptr;
        if (!d || rad_dev_set(r.device) < 0) {
            RAD_ERR("lora: rank %d: the slot-row table has no address", r.index);
            return RAD_E_STATE;
        }
        RadStream st{};
        if (rad_stream_create(&st, 0) < 0) return RAD_E_DEVICE;
        const int rc = rad_memcpy_async(d, v.data(), (int64_t)v.size() * 4, st);
        const int sy = rad_stream_sync(st);
        rad_stream_destroy(st);
        if (rc < 0 || sy < 0) return RAD_E_DEVICE;
    }
    lora_keys_ = std::make_unique<lora::KeyMap>(ranks_[0]->program, ranks_[0]->lora_sites);
    for (const auto& [name, path] : cfg_.lora.adapters) {
        std::string err;
        const int s = lora_load(name, path, &err);
        if (s < 0) {
            RAD_ERR("lora: adapter '%s' (%s): %s", name.c_str(), path.c_str(), err.c_str());
            return s;
        }
    }
    if (!cfg_.lora.dir.empty())
        RAD_INFO("lora: requests may name the %zu adapter files under %s", gen_lora_names().size(),
                 cfg_.lora.dir.c_str());
    return RAD_OK;
}

int Engine::gen_lora_open(const std::string& name, std::shared_ptr<const lora::AdapterFile>* out,
                          std::string* err) {
    {
        std::lock_guard<std::mutex> lk(lora_lib_mu_);
        auto it = lora_lib_.find(name);
        if (it != lora_lib_.end()) { *out = it->second.second; return RAD_OK; }
    }
    const std::string path = cfg_.lora.dir + "/" + name + ".safetensors";
    if (cfg_.lora.dir.empty() || !valid_name(name) || !is_file(path)) {
        *err = "no adapter named '" + name + "'" +
               (cfg_.lora.dir.empty() ? std::string(" is loaded")
                                      : " is loaded or under --lora-dir (GET /v1/loras lists them)");
        return RAD_E_NOTFOUND;
    }
    /* lora_load takes lora_mu_, which the caller holds: open here, as lora_load would. */
    auto f = std::make_shared<lora::AdapterFile>();
    RAD_TRY(f->open(path, err));
    RAD_INFO("lora: '%s' opened from --lora-dir -- %s, %zu modules", name.c_str(), f->kind_name(),
             f->modules().size());
    std::lock_guard<std::mutex> lk(lora_lib_mu_);
    lora_lib_[name] = { path, f };
    *out = f;
    return RAD_OK;
}

std::vector<std::string> Engine::gen_lora_names() const {
    std::set<std::string> n;
    {
        std::lock_guard<std::mutex> lk(lora_lib_mu_);
        for (const auto& [k, v] : lora_lib_) n.insert(k);
    }
    if (!cfg_.lora.dir.empty()) {
        /* Two levels: the directory and its subdirectories, as a UI's lora folder is laid out. */
        std::vector<std::string> dirs{ "" };
        for (size_t i = 0; i < dirs.size(); ++i) {
            const std::string rel = dirs[i];
            DIR* d = opendir((cfg_.lora.dir + "/" + rel).c_str());
            if (!d) continue;
            while (dirent* e = readdir(d)) {
                const std::string fn = e->d_name;
                if (fn.empty() || fn[0] == '.') continue;
                const std::string p = rel + fn;
                struct stat sb{};
                if (stat((cfg_.lora.dir + "/" + p).c_str(), &sb) != 0) continue;
                if (S_ISDIR(sb.st_mode)) { if (rel.empty()) dirs.push_back(p + "/"); continue; }
                const size_t k = fn.size();
                if (k > 12 && fn.compare(k - 12, 12, ".safetensors") == 0) {
                    const std::string stem = p.substr(0, p.size() - 12);
                    if (valid_name(stem)) n.insert(stem);
                }
            }
            closedir(d);
        }
    }
    return std::vector<std::string>(n.begin(), n.end());
}

int Engine::gen_lora_acquire(const std::vector<LoraUse>& uses_in, int32_t* slot,
                             std::shared_ptr<void>* hold, std::string* note, std::string* err) {
    *slot = -1;
    hold->reset();
    if (note) note->clear();
    if (!lora_gen_) { *err = "this model takes no adapters (start it with --lora or --lora-dir)"; return RAD_E_STATE; }
    /* One entry per adapter, strengths of a repeated name summed; strength 0 is no adapter. */
    std::map<std::string, double> by;
    for (const LoraUse& u : uses_in) {
        if (!std::isfinite(u.weight) || std::fabs(u.weight) > 16.0) {
            *err = fmt("adapter '%s': strength %g is outside [-16, 16]", u.name.c_str(), u.weight);
            return RAD_E_INVAL;
        }
        by[u.name] += u.weight;
    }
    std::vector<LoraUse> uses;
    for (const auto& [n, w] : by) if (w != 0.0) uses.push_back({ n, w });
    if (uses.empty()) return RAD_OK;
    if (uses.size() > 16) { *err = fmt("%zu adapters in one request; at most 16", uses.size()); return RAD_E_INVAL; }
    const std::string key = combo_key(uses);
    uint64_t uid = 0;
    if (lora_table_.acquire(key, slot, &uid, hold) == RAD_OK) return RAD_OK;

    std::lock_guard<std::mutex> lk(lora_mu_);
    if (lora_table_.acquire(key, slot, &uid, hold) == RAD_OK) return RAD_OK;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<lora::Strength> ss;
    for (const LoraUse& u : uses) {
        lora::Strength s;
        s.name = u.name;
        s.weight = u.weight;
        RAD_TRY(gen_lora_open(u.name, &s.file, err));
        ss.push_back(std::move(s));
    }
    /* A slot: free, or freed from the least recently used combination no request holds. */
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    int sl = -1;
    for (;;) {
        bool cooling = false;
        const int s = lora_table_.reserve(key, &sl, &cooling, err);
        if (s >= 0) break;
        if (s == RAD_E_FULL && !cooling) {
            std::string ev;
            if (lora_table_.evict_idle(&ev) == RAD_OK) continue;
            *err = fmt("all %d adapter slots are held by running requests with other adapters "
                       "(--max-loras); retry when one finishes", cfg_.lora.slots());
            return RAD_E_FULL;
        }
        if (s != RAD_E_FULL || stopping_.load(std::memory_order_acquire) ||
            std::chrono::steady_clock::now() >= give_up)
            return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const int rank = cfg_.lora.rank(true);
    std::map<std::string, lora::Composite> comp;
    lora::ComposeReport rep;
    RAD_TRY(lora::compose(*lora_keys_, ss, rank, &comp, &rep, err));
    if (rep.modules == 0) {
        *err = "none of the adapters' modules is a module of this model" +
               (rep.unplaced.empty() ? std::string() : " (" + rep.unplaced[0] + ")");
        return RAD_E_NOTFOUND;
    }
    const std::map<std::string, lora::AdModule> mods = lora::as_modules(comp, *lora_keys_);
    std::vector<std::vector<lora::PackedSite>> packed(ranks_.size());
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        lora::AdapterInfo info;
        const int s = lora::pack_modules(ranks_[ri]->lora_sites, rank, mods, &info, &packed[ri], err,
                                         /*compact=*/true);
        if (s < 0) {
            if (ranks_.size() > 1) *err = fmt("rank %zu: ", ri) + *err;
            return s;
        }
        /* Pools past the last one a site names hold nothing (lora_declare's dropped GEMMs). */
        packed[ri].resize(ranks_[ri]->program.lora_pools.size());
    }
    RAD_TRY(lora_copy(sl, packed, err));
    /* The ranks each pool takes in this slot, before a pass can name the slot (the publish). */
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        Program& P = ranks_[ri]->program;
        const size_t S = (size_t)cfg_.lora.slots();
        for (size_t p = 0; p < P.lora_pools.size() && p < packed[ri].size(); ++p)
            if (p * S + (size_t)sl < P.lora_rused.size())
                P.lora_rused[p * S + (size_t)sl] = packed[ri][p].empty ? 0 : (int32_t)packed[ri][p].r_used;
    }
    lora_table_.publish(sl, key, "");
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::string n;
    if (!rep.unplaced.empty())
        n += fmt("%zu adapter module%s not on this model (%s)", rep.unplaced.size(),
                 rep.unplaced.size() == 1 ? "" : "s", rep.unplaced[0].c_str());
    if (rep.exact < rep.modules)
        n += fmt("%s%lld module%s past rank %d were truncated to it, keeping at worst %.1f%% of the "
                 "delta (%s)", n.empty() ? "" : "; ", (long long)(rep.modules - rep.exact),
                 rep.modules - rep.exact == 1 ? "" : "s", rank, 100.0 * rep.worst_kept,
                 rep.worst_module.c_str());
    RAD_INFO("lora: slot %d <- %s: %lld modules (%lld exact) in %.0f ms%s%s", sl, key.c_str(),
             (long long)rep.modules, (long long)rep.exact, ms, n.empty() ? "" : "; ", n.c_str());
    if (note) *note = n;
    if (lora_table_.acquire(key, slot, &uid, hold) != RAD_OK) {
        *err = "the combination's slot was taken back before the request could hold it; retry";
        return RAD_E_STATE;
    }
    return RAD_OK;
}

}  /* namespace rad */
