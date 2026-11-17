/* report.cpp -- the startup report, as text for the log and as JSON for /diagnostics.
 *
 * Text layout: one line per option with value and source (the reason below any derived value),
 * the host in a few lines, a block per card, then findings with fixes. Text and JSON render from
 * the same structs.
 */
#include "diag/diag.h"
#include "rad_internal.h"

#include "util/json.h"

#include <algorithm>

namespace rad {
namespace diag {

void decide(std::vector<Decision>* list, const std::string& key, const std::string& value,
            const std::string& why) {
    for (Decision& d : *list)
        if (d.key == key) { d.value = value; d.why = why; return; }
    list->push_back({ key, value, why });
}

void merge(std::vector<ConfigRow>* rows, const std::vector<Decision>& decisions,
           std::vector<Decision>* derived) {
    derived->clear();
    for (const Decision& d : decisions) {
        auto it = std::find_if(rows->begin(), rows->end(),
                               [&](const ConfigRow& r) { return r.flag == d.key; });
        if (it == rows->end()) { derived->push_back(d); continue; }
        it->value = d.value;
        it->why   = d.why;
        if (it->source == "default") it->source = "auto";
    }
}

namespace {

std::string or_unknown(const std::string& s) { return s.empty() ? "unknown" : s; }
std::string bytes_or_unknown(int64_t b) { return b < 0 ? "unknown" : humanb(b); }

/* Continuation lines of a multi-line value, indented under the column they belong to. */
void indented(std::string* out, const std::string& text, int indent) {
    size_t at = 0;
    while (at <= text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        const std::string l = text.substr(at, nl - at);
        if (!l.empty()) *out += std::string((size_t)indent, ' ') + l + "\n";
        at = nl + 1;
    }
}

std::string short_bdf(const std::string& b) {
    return b.rfind("0000:", 0) == 0 ? b.substr(5) : b;
}


std::string host_lines(const HostFacts& h, const RunFacts& r) {
    std::string o;
    o += fmt("  host      %s | %s cores, %s threads | RAM %s, %s available | swap %s\n",
             or_unknown(h.cpu_model).c_str(),
             h.cores > 0 ? std::to_string(h.cores).c_str() : "?",
             h.threads > 0 ? std::to_string(h.threads).c_str() : "?",
             bytes_or_unknown(h.mem_total).c_str(), bytes_or_unknown(h.mem_available).c_str(),
             h.swap_total == 0 ? "none"
             : h.swap_total < 0 ? "unknown"
             : fmt("%s, %s used", humanb(h.swap_total).c_str(),
                   humanb(h.swap_total - std::max<int64_t>(h.swap_free, 0)).c_str()).c_str());
    o += fmt("  system    kernel %s | %s | %s | cgroup memory %s | memlock %s\n",
             or_unknown(h.kernel).c_str(), or_unknown(h.distro).c_str(),
             h.container.empty() ? "no container seen" : ("in a " + h.container + " container").c_str(),
             h.cgroup_mem_limit == -2 ? "unlimited" : bytes_or_unknown(h.cgroup_mem_limit).c_str(),
             h.memlock_soft == -2 ? "unlimited" : bytes_or_unknown(h.memlock_soft).c_str());
    o += fmt("  memory    huge pages %s (defrag %s, shmem %s) | ttm.pages_limit %s "
             "| pinned by this process %s\n",
             or_unknown(h.thp_enabled).c_str(), or_unknown(h.thp_defrag).c_str(),
             or_unknown(h.thp_shmem).c_str(),
             h.ttm_pages_limit > 0 ? humanb(h.ttm_pages_limit * h.page_size).c_str() : "unknown",
             bytes_or_unknown(r.pinned_bytes).c_str());
    o += fmt("  iommu     %s (%s)\n", or_unknown(h.iommu).c_str(), or_unknown(h.iommu_why).c_str());
    o += fmt("  amdgpu    cwsr_enable %s | boot parameters: %s\n",
             h.cwsr_enable < 0 ? "unknown" : std::to_string(h.cwsr_enable).c_str(),
             h.cmdline.empty() ? "none that bear on a card" : h.cmdline.c_str());
    o += fmt("  cpu       governor %s%s%s | io_uring %s\n", or_unknown(h.cpu_governor).c_str(),
             h.cpu_epp.empty() ? "" : (", energy preference " + h.cpu_epp).c_str(),
             h.cpu_driver.empty() ? "" : (" (" + h.cpu_driver + ")").c_str(),
             or_unknown(h.io_uring).c_str());
    o += fmt("  runtime   %s | backend %s\n",
             h.runtime.empty() ? "no device runtime" : h.runtime.c_str(),
             r.host_backend ? "host (CPU)" : "device");
    if (!h.media.empty())
        o += fmt("  media     decoders %s\n", h.media.c_str());
    return o;
}

const char* level_tag(Finding::Level l) {
    return l == Finding::Incompatible ? "INCOMPATIBLE" : l == Finding::Warn ? "WARN" : "note";
}

const char* level_json(Finding::Level l) {
    return l == Finding::Incompatible ? "incompatible" : l == Finding::Warn ? "warn" : "note";
}

std::string card_lines(const CardFacts& c) {
    std::string o;
    std::string who = c.rank >= 0 ? fmt("rank %d (device %d)", c.rank, c.device)
                    : c.device >= 0 ? fmt("device %d, not serving", c.device) : "not serving";
    o += fmt("  card %s  %s  %s%s%s  %s VRAM",
             c.bdf.c_str(), who.c_str(), c.gfx.empty() ? (c.vendor + ":" + c.pci_device).c_str() : c.gfx.c_str(),
             c.name.empty() ? "" : " ", c.name.empty() ? "" : ("\"" + c.name + "\"").c_str(),
             bytes_or_unknown(c.vram_total).c_str());
    if (c.vram_visible > 0 && c.vram_visible != c.vram_total)
        o += fmt(" (BAR %s)", humanb(c.vram_visible).c_str());
    o += fmt("  NUMA %d  driver %s%s%s\n", c.numa_node, or_unknown(c.driver).c_str(),
             c.drm.empty() ? "" : " ", c.drm.c_str());
    std::string chain;
    for (const PciHop& h : c.chain)
        chain += fmt("%s%s %s (max %s)", chain.empty() ? "" : " -> ", short_bdf(h.bdf).c_str(),
                     link_text(h.cur_speed, h.cur_width).c_str(),
                     link_text(h.max_speed, h.max_width).c_str());
    o += fmt("      link   %s  [read at %s%% busy%s]\n", chain.empty() ? "unknown" : chain.c_str(),
             c.busy_percent < 0 ? "?" : std::to_string(c.busy_percent).c_str(),
             c.busy_percent >= 0 && c.busy_percent < kBusyPercent
                 ? ": a link at rest parks below its maximum" : "");
    if (!c.dpm_pcie.empty()) {
        std::string t;
        for (size_t i = 0; i < c.dpm_pcie.size(); ++i)
            t += fmt("%s%s%s", i ? " | " : "", (int)i == c.dpm_pcie_cur ? "*" : "", c.dpm_pcie[i].c_str());
        o += "      pcie   " + t + "\n";
    }
    if (!c.dma.empty()) o += "      dma    " + c.dma + "\n";
    o += fmt("      power  cap %s (default %s, max %s) | voltage offset %s | sclk %s | mclk %s | "
             "runtime pm %s (control %s) | iommu group %s\n",
             watts_text(c.power_cap_uw).c_str(), watts_text(c.power_cap_default_uw).c_str(),
             watts_text(c.power_cap_max_uw).c_str(), or_unknown(c.vddgfx_offset).c_str(),
             or_unknown(c.sclk).c_str(), or_unknown(c.mclk).c_str(),
             or_unknown(c.runtime_status).c_str(), or_unknown(c.runtime_control).c_str(),
             c.iommu_type.empty() ? "none" : c.iommu_type.c_str());
    return o;
}

}  /* namespace */

std::string render(const std::vector<ConfigRow>& config, const std::vector<Decision>& derived,
                   const Facts& f, const RunFacts& r, const std::vector<Finding>& findings) {
    std::string o;
    if (r.unsupported)
        for (const std::string& l : unsupported_banner()) o += l + "\n";
    o += "STARTUP REPORT" + (r.version.empty() ? std::string() : " -- radiance " + r.version) +
         (r.failed ? " -- STARTUP STOPPED; this is what it had resolved when it did\n" : "\n");
    o += r.all_options
             ? "CONFIGURATION -- every option, the value in force, and where it came from\n"
             : "CONFIGURATION -- every option not at its default, the value in force, and where it "
               "came from\n";
    size_t w = 0;
    for (const ConfigRow& c : config) w = std::max(w, c.flag.size());
    w = std::min<size_t>(w + 2, 34);
    int defaults = 0;
    for (const ConfigRow& c : config) {
        if (!r.all_options && c.source == "default") { ++defaults; continue; }
        const std::string v = c.value.empty() ? "\"\"" : c.value;
        const std::string first = v.substr(0, v.find('\n'));
        std::string l = fmt("  %-*s %s", (int)w, c.flag.c_str(), first.c_str());
        const size_t col = 2 + w + 1 + 44;
        l += l.size() < col ? std::string(col - l.size(), ' ') : std::string("  ");
        o += l + "[" + c.source + "]\n";
        if (first.size() != v.size()) indented(&o, v.substr(first.size() + 1), (int)(3 + w));
        if (!c.why.empty()) indented(&o, c.why, (int)(5 + w));
    }
    if (defaults)
        o += fmt("  %d option%s at %s default (-v lists every option; /diagnostics carries them "
                 "all)\n", defaults, defaults == 1 ? "" : "s", defaults == 1 ? "its" : "their");
    o += "ENVIRONMENT -- the RADIANCE_* variables set, by name; their values are not shown\n";
    if (r.environment.empty()) o += "  none\n";
    for (const std::string& n : r.environment) o += "  " + n + "\n";
    if (!derived.empty()) {
        o += "DERIVED -- what startup worked out that no single option states\n";
        for (const Decision& d : derived) {
            o += fmt("  %-*s %s\n", (int)w, d.key.c_str(), d.value.c_str());
            if (!d.why.empty()) indented(&o, d.why, (int)(5 + w));
        }
    }
    o += "HARDWARE\n";
    o += host_lines(f.host, r);
    for (const CardFacts& c : f.cards) o += card_lines(c);
    if (f.cards.empty()) o += "  no display-class PCI device is visible\n";
    int ni = 0, nw = 0, nn = 0;
    for (const Finding& x : findings)
        (x.level == Finding::Incompatible ? ni : x.level == Finding::Warn ? nw : nn)++;
    o += fmt("FINDINGS -- %s%d warning%s, %d note%s\n",
             ni ? fmt("%d incompatible%s, ", ni, r.unsupported ? " (started anyway)" : "").c_str() : "",
             nw, nw == 1 ? "" : "s", nn, nn == 1 ? "" : "s");
    if (findings.empty()) o += "  nothing found that costs speed or correctness\n";
    for (const Finding& x : findings) {
        o += fmt("  %s %s: %s\n", level_tag(x.level), x.id.c_str(), x.what.c_str());
        o += "       why: " + x.why + "\n";
        o += "       fix: " + x.fix + "\n";
    }
    return o;
}

std::string to_json(const std::vector<ConfigRow>& config, const std::vector<Decision>& derived,
                    const Facts& f, const RunFacts& r, const std::vector<Finding>& findings) {
    using J = Json;
    J o;
    o["startup_failed"] = r.failed;
    o["unsupported_configuration"] = r.unsupported;
    J cfg = J::array();
    for (const ConfigRow& c : config) {
        J e;
        e["option"] = c.flag;
        e["value"]  = c.value;
        e["source"] = c.source;
        if (!c.why.empty()) e["why"] = c.why;
        cfg.push_back(std::move(e));
    }
    o["configuration"] = std::move(cfg);
    o["environment"] = r.environment;
    J der = J::array();
    for (const Decision& d : derived) der.push_back({ { "topic", d.key }, { "value", d.value }, { "why", d.why } });
    o["derived"] = std::move(der);

    const HostFacts& h = f.host;
    J host;
    host["cpu"]              = h.cpu_model;
    host["cores"]            = h.cores;
    host["threads"]          = h.threads;
    host["mem_total"]        = h.mem_total;
    host["mem_available"]    = h.mem_available;
    host["swap_total"]       = h.swap_total;
    host["swap_free"]        = h.swap_free;
    host["kernel"]           = h.kernel;
    host["distro"]           = h.distro;
    host["container"]        = h.container;
    host["cgroup_mem_limit"] = h.cgroup_mem_limit;
    host["memlock"]          = h.memlock_soft;
    host["thp"]              = { { "enabled", h.thp_enabled }, { "defrag", h.thp_defrag },
                                 { "shmem_enabled", h.thp_shmem } };
    host["iommu"]            = { { "mode", h.iommu }, { "why", h.iommu_why } };
    host["boot_parameters"]  = h.cmdline;
    host["ttm_pages_limit"]  = h.ttm_pages_limit;
    host["page_size"]        = h.page_size;
    host["cwsr_enable"]      = h.cwsr_enable;
    host["io_uring"]         = h.io_uring;
    host["cpu_governor"]     = h.cpu_governor;
    host["cpu_epp"]          = h.cpu_epp;
    host["cpu_driver"]       = h.cpu_driver;
    host["runtime"]          = h.runtime;
    host["media"]            = h.media;
    J cards = J::array();
    for (const CardFacts& c : f.cards) {
        J e;
        e["bdf"] = c.bdf;
        e["rank"] = c.rank;
        e["device"] = c.device;
        e["name"] = c.name;
        e["vendor"] = c.vendor;
        e["pci_device"] = c.pci_device;
        e["gfx"] = c.gfx;
        e["simd"] = c.simd;
        e["driver"] = c.driver;
        e["drm"] = c.drm;
        e["numa_node"] = c.numa_node;
        e["vram_total"] = c.vram_total;
        e["vram_visible"] = c.vram_visible;
        e["busy_percent"] = c.busy_percent;
        e["runtime_status"] = c.runtime_status;
        e["runtime_control"] = c.runtime_control;
        e["iommu_group_type"] = c.iommu_type;
        e["dma"] = c.dma;
        J chain = J::array();
        for (const PciHop& hp : c.chain)
            chain.push_back({ { "bdf", hp.bdf }, { "speed", hp.cur_speed }, { "width", hp.cur_width },
                              { "max_speed", hp.max_speed }, { "max_width", hp.max_width } });
        e["link_chain"] = std::move(chain);
        e["pcie_dpm"] = c.dpm_pcie;
        e["pcie_dpm_current"] = c.dpm_pcie_cur;
        e["sclk"] = c.sclk;
        e["mclk"] = c.mclk;
        e["power_cap_uw"] = c.power_cap_uw;
        e["power_cap_default_uw"] = c.power_cap_default_uw;
        e["power_cap_max_uw"] = c.power_cap_max_uw;
        e["vddgfx_offset"] = c.vddgfx_offset;
        cards.push_back(std::move(e));
    }
    J run;
    run["version"] = r.version;
    run["device_build"] = r.device_build;
    run["tp"] = r.tp;
    run["host_backend"] = r.host_backend;
    run["p2p"] = r.p2p;
    run["p2p_blocked"] = r.p2p_blocked;
    run["ignored_devices"] = r.ignored;
    run["pinned_bytes"] = r.pinned_bytes;
    run["host_weight_bytes"] = r.host_weight_bytes;
    run["file_weight_bytes"] = r.file_weight_bytes;
    o["hardware"] = { { "host", std::move(host) }, { "cards", std::move(cards) }, { "run", std::move(run) } };
    J fs = J::array();
    for (const Finding& x : findings)
        fs.push_back({ { "level", level_json(x.level) }, { "id", x.id },
                       { "what", x.what }, { "why", x.why }, { "fix", x.fix } });
    o["findings"] = std::move(fs);
    /* Replace invalid UTF-8 instead of throwing: paths, boot parameters and product names are
     * not guaranteed to be UTF-8. */
    return o.dump(-1, ' ', false, J::error_handler_t::replace);
}

}  /* namespace diag */
}  /* namespace rad */
