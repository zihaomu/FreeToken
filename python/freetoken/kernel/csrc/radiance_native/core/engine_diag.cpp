/* engine_diag.cpp -- the startup report's engine half: facts only a started engine has, joined to
 * the machine's (core/diag/diag.h).
 *
 * Cards are joined by PCI address: the runtime numbers devices its own way (a visibility mask
 * renumbers them, an integrated GPU may come first), while the kernel names them by bus address.
 */
#include "engine_priv.h"
#include "device/device.h"
#include "mm/media.h"

#include <algorithm>

namespace rad {

namespace {
/* How long /diagnostics reuses a reading of the machine. */
constexpr uint64_t kDiagReuseNs = 1000000000ull;
}  /* namespace */

void Engine::identify_devices(const std::vector<std::string>& names) {
    dev_ids_.clear();
    runtime_version_ = dev_runtime_version();
    for (int d = 0; d < (int)names.size(); ++d) {
        DevId id;
        id.device = d;
        id.bdf = dev_pci_address(d);
        if (id.bdf.empty()) continue;
        id.name = names[(size_t)d];
        id.integrated = dev_integrated(d);
        dev_ids_.push_back(std::move(id));
    }
}

void Engine::print_unsupported_banner() {
    for (const std::string& l : diag::unsupported_banner()) RAD_WARN("%s", l.c_str());
}

int Engine::check_configuration(const std::vector<int>& devs) {
    diag::Facts f = hardware_facts(false);
    for (int i = 0; i < cfg_.tp && i < (int)devs.size(); ++i)
        for (diag::CardFacts& c : f.cards)
            if (c.device == devs[(size_t)i]) c.rank = i;
    int found = 0;
    for (const diag::Finding& x : diag::incompatible(f)) {
        if (x.level != diag::Finding::Incompatible) continue;
        ++found;
        if (cfg_.proceed_incompatible)
            RAD_WARN("incompatible configuration, started anyway: %s: %s. Fix: %s", x.id.c_str(),
                     x.what.c_str(), x.fix.c_str());
        else
            RAD_ERR("incompatible configuration: %s: %s -- %s. Fix: %s", x.id.c_str(),
                    x.what.c_str(), x.why.c_str(), x.fix.c_str());
    }
    if (!found || cfg_.proceed_incompatible) return RAD_OK;
    RAD_ERR("radiance does not start on a configuration it does not support: the fix for each is "
            "said above. --proceed-with-incompatible-configuration starts it anyway, unsupported "
            "and without any guarantee");
    return RAD_E_UNSUPPORTED;
}

diag::Facts Engine::hardware_facts(bool serving) const {
    std::vector<std::string> mine;
    for (const DevId& id : dev_ids_)
        for (const auto& r : ranks_)
            if (r->device == id.device) mine.push_back(id.bdf);
    diag::Facts f = diag::collect("", serving ? &mine : nullptr);
    f.host.runtime = runtime_version_;
    f.host.media = mm::decoder_version();
    if (probed_) {
        f.host.memlock_soft = memlock_soft_;
        f.host.io_uring = io_uring_;
    }
    for (const DevId& id : dev_ids_)
        for (diag::CardFacts& c : f.cards) {
            if (c.bdf != id.bdf) continue;
            c.name = id.name;
            c.device = id.device;
            c.integrated = id.integrated;
            for (const auto& r : ranks_)
                if (r->device == id.device) c.rank = r->index;
            if (c.rank >= 0) c.dma = dev_dma_lane(id.device);
        }
    return f;
}

diag::RunFacts Engine::run_facts(bool failed) const {
    diag::RunFacts r;
    r.version         = RAD_VERSION;
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    r.device_build    = true;
#else
    r.device_build    = false;
#endif
    r.tp              = cfg_.tp;
    r.host_backend    = device_is_host();
    r.failed          = failed;
    r.p2p             = cfg_.tp > 1 ? p2p_measured_ : -1;
    r.p2p_forced_off  = cfg_.p2p == RAD_P2P_OFF;
    r.p2p_blocked     = p2p_blocked_;
    if (!devnote_.empty()) r.ignored.push_back(devnote_);
    r.tp_wire_lossy   = !cfg_.tp_wire_exact;
    r.tp_wire_min_kb  = cfg_.tp_wire_min_kb;
    r.prefix_cache    = cfg_.prefix_cache;
    r.prefix_cache_off_why = prefix_off_why_;
    r.host_pool_mib   = cfg_.host_pool_mib;
    {
        int64_t pinned = 0, mapped = 0, n = 0;
        rad_dev_alloc_totals(RAD_MEM_HOST_PINNED, &pinned, &n);
        rad_dev_alloc_totals(RAD_MEM_HOST_MAPPED, &mapped, &n);
        r.pinned_bytes = pinned + mapped;
    }
    /* The most any rank streams from the host, since each step waits for the slowest rank.
     * Excludes host-placed tables (the embedding, read a few rows per step; the towers, once per
     * image). */
    for (const auto& rk : ranks_) {
        const Plan& p = rk->plan;
        const int64_t host = p.totals.bytes[(int)Tier::HostPinned] +
                             p.totals.bytes[(int)Tier::HostPageable] - p.forced_host_used;
        r.host_weight_bytes = std::max(r.host_weight_bytes, std::max<int64_t>(host, 0));
        r.file_weight_bytes = std::max(r.file_weight_bytes, p.totals.bytes[(int)Tier::SSD]);
    }
    for (const TableCopy& t : table_copies_) r.locked_bytes += (int64_t)t.bytes;
    r.ngram_placement   = has_ngram_ ? table_placement_name(cfg_.ngram_resolved) : "";
    r.reference_kernels = cfg_.accept_reference_kernels;
    r.profile_ops       = cfg_.profile_ops;
    r.deterministic     = cfg_.deterministic;
    r.unsupported       = cfg_.proceed_incompatible;
    r.all_options       = cfg_.given.count("-v") > 0;
    r.environment       = env_set_;
    return r;
}

void Engine::print_startup_report(int status) {
    report_printed_ = true;
    if (!probed_) {
        diag::HostFacts h;
        diag::probe_process(&h);
        memlock_soft_ = h.memlock_soft;
        io_uring_ = h.io_uring;
        probed_ = true;
    }
    std::vector<diag::ConfigRow> rows = diag::config_rows(cfg_);
    std::vector<diag::Decision> derived;
    diag::merge(&rows, decisions_, &derived);
    const diag::Facts f = hardware_facts(false);
    const diag::RunFacts r = run_facts(status < 0);
    const std::vector<diag::Finding> found = diag::check(f, r);
    /* One log call per line: the logger formats into a fixed buffer, and every line keeps the
     * level tag. */
    const std::string text = diag::render(rows, derived, f, r, found);
    size_t at = 0;
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        RAD_INFO("%s", text.substr(at, nl - at).c_str());
        at = nl + 1;
    }
    /* Each warning also on its own W line, for readers filtering a log by level. */
    for (const diag::Finding& x : found) {
        if (x.level == diag::Finding::Warn)
            RAD_WARN("startup report: %s: %s. Fix: %s", x.id.c_str(), x.what.c_str(), x.fix.c_str());
        /* A refusal is an E line; started anyway it is a W line, which the banner explains. */
        else if (x.level == diag::Finding::Incompatible && cfg_.proceed_incompatible)
            RAD_WARN("startup report: INCOMPATIBLE, started anyway: %s: %s. Fix: %s", x.id.c_str(),
                     x.what.c_str(), x.fix.c_str());
        else if (x.level == diag::Finding::Incompatible)
            RAD_ERR("startup report: INCOMPATIBLE: %s: %s. Fix: %s", x.id.c_str(), x.what.c_str(),
                    x.fix.c_str());
    }
    /* Repeat the failure last, since the original error line is now above the report. */
    if (status < 0)
        RAD_ERR("startup stopped: %s (the reason is said above the startup report)",
                rad_strerror(status));
}

std::string Engine::diagnostics_json() const {
    std::vector<diag::ConfigRow> rows = diag::config_rows(cfg_);
    std::vector<diag::Decision> derived;
    diag::merge(&rows, decisions_, &derived);
    diag::Facts f;
    {
        std::lock_guard<std::mutex> lk(diag_mu_);
        const uint64_t now = rad_mono_ns();
        if (diag_read_ns_ == 0 || now - diag_read_ns_ >= kDiagReuseNs) {
            diag_facts_ = hardware_facts(true);
            diag_read_ns_ = now;
        }
        f = diag_facts_;
    }
    const diag::RunFacts r = run_facts(false);
    return diag::to_json(rows, derived, f, r, diag::check(f, r));
}

}  /* namespace rad */
