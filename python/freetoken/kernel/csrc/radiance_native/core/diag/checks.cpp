/* checks.cpp -- what in the configuration or on the machine costs speed or correctness, and what to
 * do about it.
 *
 * Each check is a pure function of the facts (no file or device access), so tests pass a machine
 * described in a struct. A finding states what was seen with numbers, why it matters, and a fix.
 *
 * Incompatible: a machine setting startup refuses (incompatible()), unless
 * --proceed-with-incompatible-configuration starts it anyway. Warn: a cost or risk to act on. Note:
 * a trade-off, or worth knowing when reading a benchmark.
 * Load-dependent findings (a parked link) are raised only when the card was busy, since an idle
 * PCIe link parks at its lowest speed by design.
 */
#include "diag/diag.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>

namespace rad {
namespace diag {

namespace {

constexpr int64_t kGiB = (int64_t)1 << 30;
constexpr int64_t kMiB = (int64_t)1 << 20;

void add(std::vector<Finding>* out, Finding::Level lv, const char* id, std::string what,
         std::string why, std::string fix) {
    Finding f;
    f.level = lv;
    f.id    = id;
    f.what  = std::move(what);
    f.why   = std::move(why);
    f.fix   = std::move(fix);
    out->push_back(std::move(f));
}

std::vector<const CardFacts*> served(const Facts& f) {
    std::vector<const CardFacts*> out;
    for (const CardFacts& c : f.cards)
        if (c.rank >= 0) out.push_back(&c);
    std::sort(out.begin(), out.end(),
              [](const CardFacts* a, const CardFacts* b) { return a->rank < b->rank; });
    return out;
}

/* Whether the serving cards' DMA is translated: from their IOMMU groups where known, else the
 * machine's mode. */
bool translated(const Facts& f) {
    bool any = false, dma = false;
    for (const CardFacts* c : served(f)) {
        if (c->iommu_type.empty()) continue;
        any = true;
        if (c->iommu_type != "identity") dma = true;
    }
    return any ? dma : f.host.iommu == "translated";
}

/* An amdgpu device with at least 4 GiB of VRAM. Excludes small iGPU carve-outs, includes an APU
 * configured with tens of GiB. */
bool servable_gpu(const CardFacts& c) {
    return c.vendor == "0x1002" && c.driver == "amdgpu" && c.vram_total >= ((int64_t)4 << 30);
}

std::string card_name(const CardFacts& c) {
    return c.rank >= 0 ? fmt("card %s (rank %d)", c.bdf.c_str(), c.rank) : "card " + c.bdf;
}

std::string hop_link(const PciHop& h, bool cur) {
    return cur ? link_text(h.cur_speed, h.cur_width) : link_text(h.max_speed, h.max_width);
}

/* ---- the link ------------------------------------------------------------------------------ */

/* A hop below its own maximum while the card is busy: link power management stuck a level down,
 * or a link retrained narrower. Every hop is checked, because the card's endpoint can read full
 * speed while the root port above an on-card switch runs at half. */
void check_link_under_load(const Facts& f, std::vector<Finding>* out) {
    for (const CardFacts* c : served(f)) {
        if (c->busy_percent < kBusyPercent) continue;
        for (const PciHop& h : c->chain) {
            const double cur = link_gts(h.cur_speed), mx = link_gts(h.max_speed);
            const bool slow = cur > 0 && mx > 0 && cur < mx;
            const bool narrow = h.cur_width > 0 && h.max_width > 0 && h.cur_width < h.max_width;
            if (!slow && !narrow) continue;
            add(out, Finding::Warn, "link-below-max",
                fmt("%s: the PCIe link at %s runs %s of its %s while the card is %d%% busy",
                    card_name(*c).c_str(), h.bdf.c_str(), hop_link(h, true).c_str(),
                    hop_link(h, false).c_str(), c->busy_percent),
                "every copy between host memory and the card -- experts streamed from the host "
                "pool, the all-reduce through host memory, the embedding read in place -- and "
                "every kernel launch's packet fetch cross this link, so the card serves at the "
                "slower link's rate and a tensor-parallel step waits for it",
                "a link stuck one level down after long uptime has been cleared by a reboot; "
                "check that it recovers under load (GET /diagnostics during a long prefill), and "
                "reseat or move the card if it trains narrow");
            break;
        }
        if (c->dpm_pcie_cur >= 0 && c->dpm_pcie_cur + 1 < (int)c->dpm_pcie.size())
            add(out, Finding::Warn, "link-dpm-below-top",
                fmt("%s: the card's PCIe power level is %s, not its top level %s, while it is "
                    "%d%% busy", card_name(*c).c_str(), c->dpm_pcie[(size_t)c->dpm_pcie_cur].c_str(),
                    c->dpm_pcie.back().c_str(), c->busy_percent),
                "the SMU picks the link's speed from this table; a level below the top under load "
                "is half the host bandwidth or less",
                "a reboot has cleared a level stuck after long uptime; pp_dpm_pcie can pin a level "
                "(echo 2 | sudo tee /sys/class/drm/CARD/device/pp_dpm_pcie with "
                "power_dpm_force_performance_level manual), with the server stopped");
    }
}

/* A hop whose maximum is below the card's (older slot, riser, lane split). Often the board's
 * design, not a fault; a Note. */
void check_link_capped(const Facts& f, std::vector<Finding>* out) {
    for (const CardFacts* c : served(f)) {
        if (c->chain.size() < 2) continue;
        const PciHop& ep = c->chain.back();
        const double card = link_gts(ep.max_speed);
        for (size_t i = 0; i + 1 < c->chain.size(); ++i) {
            const PciHop& h = c->chain[i];
            const double mx = link_gts(h.max_speed);
            const bool slower = mx > 0 && card > 0 && mx < card;
            const bool narrower = h.max_width > 0 && ep.max_width > 0 && h.max_width < ep.max_width;
            if (!slower && !narrower) continue;
            add(out, Finding::Note, "link-capped",
                fmt("%s: the card's link is %s at most, and %s above it is %s at most",
                    card_name(*c).c_str(), hop_link(ep, false).c_str(), h.bdf.c_str(),
                    hop_link(h, false).c_str()),
                "the host side of the link bounds what the card can move to and from host memory: "
                "streamed experts, the host pool and the all-reduce when it goes through host memory",
                "a slot with the card's full width and speed (the board manual says which slots "
                "share lanes), or no riser; nothing to do if the board has no faster slot");
            break;
        }
    }
}

/* ---- the cards against each other ----------------------------------------------------------- */

/* A tensor-parallel step ends when its slowest rank does, so differently configured cards serve
 * at the slower one's pace. */
void check_cards_differ(const Facts& f, std::vector<Finding>* out) {
    const auto cs = served(f);
    if (cs.size() < 2) return;
    std::vector<std::string> diffs;
    auto field = [&](const char* what, auto get, auto show) {
        std::map<std::string, std::vector<std::string>> by;
        for (const CardFacts* c : cs) by[show(get(*c))].push_back(c->bdf);
        if (by.size() < 2) return;
        std::string d = std::string(what) + ":";
        for (const auto& [v, bdfs] : by) {
            d += " " + v + " on";
            for (const std::string& b : bdfs) d += " " + b;
            d += ";";
        }
        d.pop_back();
        diffs.push_back(d);
    };
    auto str = [](const std::string& s) { return s.empty() ? std::string("unknown") : s; };
    field("root port link", [](const CardFacts& c) {
              return c.chain.empty() ? std::string() : hop_link(c.chain.front(), false); }, str);
    field("power cap", [](const CardFacts& c) { return c.power_cap_uw; },
          watts_text);
    field("voltage offset", [](const CardFacts& c) { return c.vddgfx_offset; }, str);
    field("top PCIe power level", [](const CardFacts& c) {
              return c.dpm_pcie.empty() ? std::string() : c.dpm_pcie.back(); }, str);
    field("runtime power management", [](const CardFacts& c) { return c.runtime_control; }, str);
    field("target", [](const CardFacts& c) { return c.gfx; }, str);
    if (diffs.empty()) return;
    std::string what = "the cards serving this model are not set up alike -- ";
    for (size_t i = 0; i < diffs.size(); ++i) what += (i ? "; " : "") + diffs[i];
    add(out, Finding::Warn, "cards-differ", what,
        "a tensor-parallel step waits for its slowest rank in every all-reduce, so the whole "
        "server runs at the pace of the card with the lower limit or the slower link",
        "make the settings match (the power cap and voltage offset are per card under "
        "/sys/class/drm/cardN/device; set them with the server stopped)");
}

/* A power cap below the card's default. Compared to the default, not the maximum: stock cards
 * ship below the board maximum (300 W of 330 W on an R9700). */
void check_power_cap(const Facts& f, std::vector<Finding>* out) {
    for (const CardFacts* c : served(f)) {
        if (c->power_cap_uw <= 0 || c->power_cap_default_uw <= 0) continue;
        if (c->power_cap_uw >= c->power_cap_default_uw) continue;
        add(out, Finding::Note, "power-cap-below-default",
            fmt("%s: the power cap is %lld W, below the card's default of %lld W",
                card_name(*c).c_str(), (long long)(c->power_cap_uw / 1000000),
                (long long)(c->power_cap_default_uw / 1000000)),
            "under a sustained load the power cap is what sets the clock on these cards "
            "(scripts/cardclock.py shows which limit binds)",
            fmt("echo %lld | sudo tee /sys/class/drm/%s/device/hwmon/hwmon*/power1_cap restores "
                "the default, with the server stopped; nothing to do if the cap was lowered on "
                "purpose", (long long)c->power_cap_default_uw,
                c->drm.empty() ? "cardN" : c->drm.c_str()));
    }
}

void check_small_bar(const Facts& f, std::vector<Finding>* out) {
    for (const CardFacts* c : served(f)) {
        if (c->vram_visible <= 0 || c->vram_total <= 0) continue;
        if (c->vram_visible * 10 >= c->vram_total * 9) continue;
        add(out, Finding::Warn, "small-bar",
            fmt("%s: the host and the other cards see %s of its %s of VRAM", card_name(*c).c_str(),
                humanb(c->vram_visible).c_str(), humanb(c->vram_total).c_str()),
            "without Resizable BAR a peer can address only a window of the card's memory, which "
            "limits peer-to-peer collectives and copies into VRAM from the host; the engine also "
            "keeps that card's kernel arguments in pinned host memory, and every dispatch reads "
            "them across the link",
            "enable Resizable BAR (\"Re-Size BAR Support\" and \"Above 4G Decoding\") in the "
            "firmware setup");
    }
}

/* Overdrive offset with runtime suspend enabled. The driver re-uploads the offset on every resume,
 * and on RDNA4 that upload can fail ("Failed to upload customized OD settings"). Whether the
 * firmware kept it is not readable here; a card that lost it clocks lower at the same cap. */
void check_od_runtime_pm(const Facts& f, std::vector<Finding>* out) {
    for (const CardFacts* c : served(f)) {
        if (c->vddgfx_offset.empty() || c->vddgfx_offset == "0 mV") continue;
        if (c->runtime_control != "auto") continue;
        add(out, Finding::Note, "od-runtime-suspend",
            fmt("%s: a voltage offset of %s is set and the card runtime-suspends when idle "
                "(power/control auto)", card_name(*c).c_str(), c->vddgfx_offset.c_str()),
            "the driver re-uploads overdrive settings on every resume and can fail to (\"Failed "
            "to upload customized OD settings\" in the kernel log); a card that resumed without "
            "its offset clocks lower at the same power cap, and in a tensor-parallel world sets "
            "the pace for every card",
            fmt("echo on | sudo tee /sys/bus/pci/devices/%s/power/control keeps it resumed; check "
                "`journalctl -k | grep -i overdrive` for failed uploads", c->bdf.c_str()));
    }
}

/* ---- the host ------------------------------------------------------------------------------- */

/* Incompatible where cards serve (#42's class of hang); a host-backend run beside a loaded driver is
 * only told. */
void check_cwsr(const Facts& f, std::vector<Finding>* out) {
    if (f.host.cwsr_enable != 0) return;
    add(out, served(f).empty() ? Finding::Warn : Finding::Incompatible, "cwsr-off",
        "amdgpu cwsr_enable is 0",
        "the driver cannot preempt a running queue: any queue eviction (host memory the card maps, "
        "moved under memory pressure) waits for a queue a serve never lets drain, and ends in a "
        "GPU hang and reset",
        "amdgpu.cwsr_enable=1 as a module option, then rebuild the initramfs and reboot");
}

void check_huge_pages(const Facts& f, const RunFacts& r, std::vector<Finding>* out) {
    if (f.host.thp_enabled != "never") return;
    if (r.host_pool_mib <= 0 && r.pinned_bytes < kGiB) return;
    const bool tr = translated(f);
    add(out, tr ? Finding::Warn : Finding::Note, "thp-never",
        "transparent huge pages are off (/sys/kernel/mm/transparent_hugepage/enabled is never), so "
        "every pinned range is 4 KiB pages",
        tr ? "with an IOMMU translating the cards' reads each page is a translation of its "
                     "own; a host pool streamed at 4 KiB pages measured 10-21% slower prefill"
                   : "with the IOMMU in passthrough the page size has not measured slower; it "
                     "matters as soon as the IOMMU translates",
        "echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled (or "
        "transparent_hugepage=madvise on the kernel command line)");
}

void check_iommu(const Facts& f, const RunFacts& r, std::vector<Finding>* out) {
    if (!translated(f)) return;
    if (r.host_weight_bytes <= 0 && r.host_pool_mib <= 0 && r.tp < 2) return;
    std::string groups;
    for (const CardFacts* c : served(f))
        if (!c->iommu_type.empty()) groups += (groups.empty() ? "" : ", ") + c->bdf + " " + c->iommu_type;
    add(out, Finding::Note, "iommu-translated",
        "the IOMMU translates the cards' DMA (" + (groups.empty() ? f.host.iommu_why : groups) + ")",
        "every read a card makes of host memory -- the host pool, the embedding in RAM, the "
        "all-reduce through host memory -- goes through an IOMMU translation; with huge pages that "
        "is cheap, at 4 KiB pages it measured 10-21% prefill",
        std::string("iommu=pt on the kernel command line (amd_iommu=pt / intel_iommu=pt), if "
                    "nothing on the machine needs DMA isolation; otherwise keep transparent huge "
                    "pages at madvise or always") +
            (r.tp > 1 ? "; with more than one card, check that peer access survives the change "
                        "(--p2p on refuses to start without it): iommu=pt removed it on an Intel "
                        "VT-d host (GUIDE §7.2)"
                      : ""));
}

void check_memory(const Facts& f, const RunFacts& r, std::vector<Finding>* out) {
    const HostFacts& h = f.host;
    if (h.mem_available >= 0 && h.mem_total > 0 &&
        (h.mem_available < 2 * kGiB || h.mem_available * 20 < h.mem_total))
        add(out, Finding::Warn, "memory-low",
            fmt("%s of %s of RAM is available after startup", humanb(h.mem_available).c_str(),
                humanb(h.mem_total).c_str()),
            "the page cache is what a container read in place and the n-gram rows read from disk "
            "come through; with it starved every step waits on the drive, and the kernel starts "
            "reclaiming and swapping",
            "lower --host-pool-mib or --prefix-cache-mib, or stop what else holds memory");
    if (h.swap_total > 0 && h.swap_free >= 0 && h.swap_total - h.swap_free > 256 * kMiB)
        add(out, Finding::Note, "swap-in-use",
            fmt("%s of swap is in use", humanb(h.swap_total - h.swap_free).c_str()),
            "a host thread that faults on a swapped page stalls the step it is part of",
            "free memory, or turn swap off on a machine that only serves");
    if (h.cgroup_mem_limit > 0 && h.mem_total > 0 && h.cgroup_mem_limit < h.mem_total &&
        r.pinned_bytes > 0 && r.pinned_bytes * 10 > h.cgroup_mem_limit * 8)
        add(out, Finding::Warn, "cgroup-limit",
            fmt("the container's memory limit is %s and the process pins %s",
                humanb(h.cgroup_mem_limit).c_str(), humanb(r.pinned_bytes).c_str()),
            "pinned memory counts against the limit and cannot be reclaimed; what is left is all "
            "the page cache and the process's own allocations get before the OOM killer",
            "raise the container's memory limit or lower the pools");
}

void check_memlock(const Facts& f, const RunFacts& r, std::vector<Finding>* out) {
    if (f.host.memlock_soft == -2 || f.host.memlock_soft < 0) return;
    if (r.locked_bytes > 0 || r.ngram_placement != "disk") return;
    add(out, Finding::Note, "memlock-limited",
        fmt("RLIMIT_MEMLOCK is %s and the n-gram table is read from the container on disk",
            humanb(f.host.memlock_soft).c_str()),
        "a locked copy of the table in RAM is one of the faster tiers, and the memlock limit is "
        "one of the things that keeps --ngram-placement auto off it",
        "Docker: ulimits memlock -1 (--ulimit memlock=-1); systemd: LimitMEMLOCK=infinity");
}

void check_cpu(const Facts& f, std::vector<Finding>* out) {
    const HostFacts& h = f.host;
    if (h.cpu_governor.empty() || h.cpu_governor == "performance") return;
    if (h.cpu_governor == "powersave" && h.cpu_epp == "performance") return;
    if (h.cpu_governor == "schedutil") return;
    add(out, Finding::Note, "cpu-governor",
        fmt("the CPU governor is %s%s", h.cpu_governor.c_str(),
            h.cpu_epp.empty() ? "" : (" with energy preference " + h.cpu_epp).c_str()),
        "the host side of a step -- scheduling, sampling, the n-gram gather, launch bookkeeping -- "
        "is single-threaded per rank, and a core that ramps slowly shows as host gap between steps",
        "cpupower frequency-set -g performance, or echo performance | sudo tee "
        "/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference");
}

/* ---- the engine ----------------------------------------------------------------------------- */

void check_engine(const Facts& f, const RunFacts& r, std::vector<Finding>* out) {
    std::string gpus;
    for (const CardFacts& c : f.cards)
        if (servable_gpu(c))
            gpus += (gpus.empty() ? "" : ", ") + c.bdf + (c.gfx.empty() ? "" : " " + c.gfx);
    if (r.host_backend && !gpus.empty())
        add(out, Finding::Warn, "host-backend",
            "the engine serves on the CPU although this machine has an AMD GPU (" + gpus + ")",
            r.device_build ? "the device runtime found no card it could use, so every op runs on the host"
                           : "this build has no device runtime (built without HIP), so every op runs "
                             "on the host",
            r.device_build
                ? "the device lines at the top of the log say why: /dev/kfd and /dev/dri must be "
                  "passed to a container with the render and video groups, ROCm must support the "
                  "card, and HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES must not hide it"
                : "build with ROCm (RAD_WITH_HIP=ON, hipcc on the PATH), or run the image");
    for (const std::string& why : r.ignored)
        add(out, Finding::Note, "device-ignored", "a device was left out: " + why,
            "a rank placed on a slower or different card would set the pace of every step",
            "ROCR_VISIBLE_DEVICES names the cards to use explicitly");
    if (r.tp > 1 && r.p2p == 0 && !r.p2p_forced_off)
        add(out, Finding::Note, "p2p-off",
            "the cards cannot reach each other's memory directly" +
                (r.p2p_blocked.empty() ? std::string() : " (" + r.p2p_blocked + ")"),
            "the collectives exchange through pinned host memory, which crosses each card's link "
            "twice",
            "peer access needs Resizable BAR and the cards under one root complex; whether the "
            "IOMMU's mode allows it depends on the platform -- iommu=pt removed it on an Intel "
            "VT-d host (GUIDE §7.2) -- so check `--p2p on`, which refuses to start without it, "
            "before and after changing the mode");
    if (r.tp > 1 && r.tp_wire_lossy)
        add(out, Finding::Note, "tp-wire-lossy",
            fmt("--tp-wire wht6: all-reduce messages from %lld KiB up carry a quantised payload",
                (long long)r.tp_wire_min_kb),
            "faster prefill, but the model does not compute the exact function; two runs on "
            "different world sizes are not comparable token for token",
            "--tp-wire exact when exactness matters more than prefill speed");
    /* Only when the operator turned it off; an engine-side reason is shown in the configuration. */
    if (!r.prefix_cache && r.prefix_cache_off_why.empty())
        add(out, Finding::Note, "prefix-cache-off", "prefix caching is off (--no-prefix-cache)",
            "every request prefills its whole prompt, including the conversation it shares with "
            "the turn before",
            "drop --no-prefix-cache");
    if (r.host_weight_bytes > 0)
        add(out, Finding::Note, "experts-from-host",
            fmt("%s of weights a rank live in the pinned host pool and cross PCIe when they are "
                "used", humanb(r.host_weight_bytes).c_str()),
            "a prefill step streams every routed layer's pooled experts onto the card, so its speed "
            "is bounded by the link; decode reads the ones it routes to",
            "more VRAM for the experts: a shorter --max-model-len or fewer --max-num-seqs, "
            "--embedding-placement ram, --kv-cache-dtype i8, or more cards (--tp)");
    if (r.file_weight_bytes > 0)
        add(out, Finding::Warn, "weights-from-disk",
            fmt("%s of weights a rank are read from the container on disk as they are needed",
                humanb(r.file_weight_bytes).c_str()),
            "a step that needs them is paced by the drive",
            "a larger --host-pool-mib (if RAM allows) or more VRAM for the weights");
    if (r.reference_kernels)
        add(out, Finding::Warn, "reference-kernels",
            "--debug-accept-reference-kernels: ops may run on the reference implementation",
            "the reference library is an oracle, orders of magnitude slower than a device kernel",
            "install a kernel library for this card and drop the flag");
    if (r.profile_ops)
        add(out, Finding::Warn, "profile-ops",
            "--profile-ops times every op, which restores synchronisations",
            "the run is slower and its numbers are not comparable against an unprofiled one",
            "drop --profile-ops for anything measured");
}

}  /* namespace */

std::vector<Finding> incompatible(const Facts& f) {
    std::vector<Finding> out;
    check_cwsr(f, &out);
    std::stable_sort(out.begin(), out.end(),
                     [](const Finding& a, const Finding& b) { return a.level > b.level; });
    return out;
}

std::vector<std::string> unsupported_banner() {
    const char* text[] = {
        "UNSUPPORTED CONFIGURATION -- started with --proceed-with-incompatible-configuration.",
        "This machine is set up in a way radiance refuses to run on, and it was started anyway.",
        "ALL GUARANTEES ARE OFF: correctness, speed and stability are not tested here, and",
        "problems seen on this configuration are not supported. The startup report lists what",
        "is incompatible and how to fix it.",
    };
    size_t w = 0;
    for (const char* t : text) w = std::max(w, std::strlen(t));
    const std::string bar(w + 8, '!');
    std::vector<std::string> out{ bar };
    for (const char* t : text)
        out.push_back("!!  " + std::string(t) + std::string(w - std::strlen(t), ' ') + "  !!");
    out.push_back(bar);
    return out;
}

std::vector<Finding> check(const Facts& f, const RunFacts& r) {
    std::vector<Finding> out = incompatible(f);
    check_engine(f, r, &out);
    check_link_under_load(f, &out);
    check_cards_differ(f, &out);
    check_link_capped(f, &out);
    check_small_bar(f, &out);
    check_power_cap(f, &out);
    check_od_runtime_pm(f, &out);
    check_memory(f, r, &out);
    check_huge_pages(f, r, &out);
    check_iommu(f, r, &out);
    check_memlock(f, r, &out);
    check_cpu(f, &out);
    /* Refusals, then warnings, then notes; stable, so each group keeps the order above. */
    std::stable_sort(out.begin(), out.end(),
                     [](const Finding& a, const Finding& b) { return a.level > b.level; });
    return out;
}

std::string link_text(const std::string& speed, int width) {
    std::string s = speed;
    const size_t p = s.find(" PCIe");
    if (p != std::string::npos) s.erase(p);
    if (s.empty()) s = "?";
    return width > 0 ? fmt("%s x%d", s.c_str(), width) : s;
}

std::string watts_text(int64_t uw) {
    return uw < 0 ? std::string("unknown") : fmt("%lld W", (long long)(uw / 1000000));
}

double link_gts(const std::string& speed) {
    return speed.empty() ? 0.0 : std::strtod(speed.c_str(), nullptr);
}

}  /* namespace diag */
}  /* namespace rad */
