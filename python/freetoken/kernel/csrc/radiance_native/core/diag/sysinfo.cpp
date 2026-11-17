/* sysinfo.cpp -- the machine, as /proc and /sys describe it.
 *
 * Every read is a file under a root; no tool or library is used. The production image is built
 * FROM scratch (no shell, lspci or rocm-smi), and tests pass a directory tree describing the
 * machine. What a file does not say is left unknown, never guessed.
 */
#include "diag/diag.h"
#include "rad_internal.h"

#include <cerrno>
#include <cctype>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rad {
namespace diag {

namespace fs = std::filesystem;

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) return std::string();
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
        s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

/* The first line of a sysfs attribute, trimmed; "" when it cannot be read. */
std::string line(const std::string& path) {
    const std::string s = slurp(path);
    const size_t nl = s.find('\n');
    return trim(nl == std::string::npos ? s : s.substr(0, nl));
}

int64_t num(const std::string& path, int64_t dflt = -1) {
    const std::string s = line(path);
    if (s.empty()) return dflt;
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 0);
    return end && end != s.c_str() ? (int64_t)v : dflt;
}

/* The bracketed word of a sysfs choice list: "always [madvise] never" -> "madvise". */
std::string selected(const std::string& path) {
    const std::string s = line(path);
    const size_t a = s.find('['), b = s.find(']');
    if (a == std::string::npos || b == std::string::npos || b < a) return s;
    return s.substr(a + 1, b - a - 1);
}

bool exists(const std::string& p) {
    std::error_code ec;
    return fs::exists(p, ec);
}

std::vector<std::string> list(const std::string& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
        out.push_back(it->path().filename().string());
    std::sort(out.begin(), out.end());
    return out;
}

bool is_bdf(const std::string& s) {
    unsigned d, b, dv, f;
    char tail;
    return s.size() == 12 && std::sscanf(s.c_str(), "%4x:%2x:%2x.%1x%c", &d, &b, &dv, &f, &tail) == 4;
}

/* "0x7551" from a sysfs id file, which the kernel writes with its prefix. */
std::string hexid(const std::string& path) { return line(path); }

/* The lowest and highest level of a pp_dpm_* table, "500-2350 MHz"; the levels are listed in
 * order, one a line, as "N: 2350Mhz *". */
std::string dpm_range(const std::string& path) {
    std::istringstream in(slurp(path));
    std::string l;
    long lo = -1, hi = -1;
    while (std::getline(in, l)) {
        const size_t c = l.find(':');
        const char* p = l.c_str() + (c == std::string::npos ? 0 : c + 1);
        long v = std::strtol(p, nullptr, 10);
        if (v <= 0) continue;
        if (lo < 0 || v < lo) lo = v;
        if (v > hi) hi = v;
    }
    if (lo < 0) return std::string();
    return fmt("%ld-%ld MHz", lo, hi);
}

/* `pm`: whether the power-management files are read as well (collect's `managed`). */
void read_card(const std::string& root, const std::string& bdf, bool pm, CardFacts* c) {
    const std::string dev = root + "/sys/bus/pci/devices/" + bdf;
    c->bdf    = bdf;
    c->vendor = hexid(dev + "/vendor");
    c->pci_device = hexid(dev + "/device");
    {
        std::error_code ec;
        const fs::path drv = fs::read_symlink(dev + "/driver", ec);
        if (!ec) c->driver = drv.filename().string();
    }
    c->numa_node       = (int)num(dev + "/numa_node", -1);
    c->runtime_status  = line(dev + "/power/runtime_status");
    c->runtime_control = line(dev + "/power/control");
    c->iommu_type      = line(dev + "/iommu_group/type");
    for (const std::string& d : list(dev + "/drm"))
        if (d.rfind("card", 0) == 0 && d.size() > 4 && std::isdigit((unsigned char)d[4])) c->drm = d;

    /* Every hop up to the root port, not just the card's link: a card with an on-board switch
     * reports its endpoint at full speed regardless of the slot, so a parked root port shows only
     * at the root port (and in the card's DPM table). */
    auto hop = [&](const std::string& at, const std::string& hop_bdf) {
        PciHop h;
        h.bdf       = hop_bdf;
        h.cur_speed = line(at + "/current_link_speed");
        h.max_speed = line(at + "/max_link_speed");
        h.cur_width = (int)num(at + "/current_link_width", -1);
        h.max_width = (int)num(at + "/max_link_width", -1);
        c->chain.push_back(h);
    };
    std::error_code ec;
    const fs::path real = fs::canonical(dev, ec);
    if (!ec) {
        for (const fs::path& part : real) {
            const std::string p = part.string();
            if (is_bdf(p)) hop(root + "/sys/bus/pci/devices/" + p, p);
        }
    }
    if (c->chain.empty()) hop(dev, bdf);

    c->vram_total   = num(dev + "/mem_info_vram_total", -1);
    c->vram_visible = num(dev + "/mem_info_vis_vram_total", -1);
    if (!pm) return;
    c->busy_percent = (int)num(dev + "/gpu_busy_percent", -1);
    {
        std::istringstream in(slurp(dev + "/pp_dpm_pcie"));
        std::string l;
        while (std::getline(in, l)) {
            l = trim(l);
            if (l.empty()) continue;
            if (l.find('*') != std::string::npos) c->dpm_pcie_cur = (int)c->dpm_pcie.size();
            l.erase(std::remove(l.begin(), l.end(), '*'), l.end());
            c->dpm_pcie.push_back(trim(l));
        }
    }
    c->sclk = dpm_range(dev + "/pp_dpm_sclk");
    c->mclk = dpm_range(dev + "/pp_dpm_mclk");
    for (const std::string& h : list(dev + "/hwmon")) {
        const std::string hp = dev + "/hwmon/" + h;
        if (!exists(hp + "/power1_cap")) continue;
        c->power_cap_uw         = num(hp + "/power1_cap", -1);
        c->power_cap_default_uw = num(hp + "/power1_cap_default", -1);
        c->power_cap_max_uw     = num(hp + "/power1_cap_max", -1);
        break;
    }
    {
        /* "OD_VDDGFX_OFFSET:\n-75mV": the offset the driver holds and uploads; whether the
         * firmware applied it is not in this file. */
        const std::string od = slurp(dev + "/pp_od_clk_voltage");
        const size_t at = od.find("OD_VDDGFX_OFFSET:");
        if (at != std::string::npos) {
            const char* p = od.c_str() + at + std::strlen("OD_VDDGFX_OFFSET:");
            char* end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end && end != p) c->vddgfx_offset = fmt("%ld mV", v);
        }
    }
}

/* Each KFD topology GPU node's target and SIMD count, keyed by PCI location:
 * `location_id` is (bus << 8) | (device << 3) | function. */
void read_kfd(const std::string& root, std::vector<CardFacts>* cards) {
    const std::string nodes = root + "/sys/class/kfd/kfd/topology/nodes";
    for (const std::string& n : list(nodes)) {
        /* Line by line, parsing only the needed keys: some values (unique_id, a 64-bit hash)
         * overflow int64, and a failed stream extraction would stop reading all later keys. */
        std::istringstream in(slurp(nodes + "/" + n + "/properties"));
        std::string l;
        long long loc = -1, dom = 0, gfx = 0, simd = -1;
        while (std::getline(in, l)) {
            const size_t sp = l.find_first_of(" \t");
            if (sp == std::string::npos) continue;
            const std::string k = l.substr(0, sp);
            long long* into = k == "location_id"        ? &loc
                            : k == "domain"             ? &dom
                            : k == "gfx_target_version" ? &gfx
                            : k == "simd_count"         ? &simd
                                                        : nullptr;
            if (!into) continue;
            const char* p = l.c_str() + sp + 1;
            char* end = nullptr;
            errno = 0;
            const long long v = std::strtoll(p, &end, 10);
            if (end != p && errno == 0) *into = v;
        }
        if (gfx <= 0 || loc < 0) continue;
        const std::string bdf = fmt("%04llx:%02llx:%02llx.%llx", dom, (loc >> 8) & 0xff,
                                    (loc >> 3) & 0x1f, loc & 7);
        for (CardFacts& c : *cards) {
            if (c.bdf != bdf) continue;
            c.gfx  = fmt("gfx%lld%llx%llx", gfx / 10000, (gfx / 100) % 100, gfx % 100);
            c.simd = (int)simd;
        }
    }
}

/* The binding one of two limits in HostFacts' encoding (bytes, -2 unlimited, -1 unknown): the
 * smaller number, else unlimited over unknown. */
int64_t tighter(int64_t a, int64_t b) {
    if (a >= 0 && b >= 0) return std::min(a, b);
    if (a >= 0 || b >= 0) return std::max(a, b);
    return a == -2 || b == -2 ? -2 : -1;
}

/* The tightest memory limit from a cgroup up to the mount's root. A limit binds every cgroup
 * below it, so the process's group may say "max" while a parent slice holds the real limit.
 * `rel` is the /proc/self/cgroup path under `mount`; every level is read, the root included
 * (inside a cgroup namespace that is the container's own group). */
int64_t cgroup_limit(const std::string& mount, std::string rel, const char* file) {
    int64_t lim = -1;
    while (!rel.empty() && rel.back() == '/') rel.pop_back();
    for (;;) {
        const std::string v = line(mount + rel + "/" + file);
        if (v == "max") {
            lim = tighter(lim, -2);
        } else if (!v.empty()) {
            char* end = nullptr;
            errno = 0;
            const long long n = std::strtoll(v.c_str(), &end, 10);
            /* Version 1 writes "no limit" as the largest page-aligned number below 2^63. */
            if (end != v.c_str() && errno == 0 && n >= 0)
                lim = tighter(lim, n >= ((long long)1 << 62) ? -2 : (int64_t)n);
        }
        if (rel.empty()) break;
        const size_t up = rel.rfind('/');
        rel.erase(up == std::string::npos ? 0 : up);
    }
    return lim;
}

void read_host(const std::string& root, HostFacts* h) {
    {
        std::istringstream in(slurp(root + "/proc/cpuinfo"));
        std::string l;
        std::set<std::pair<std::string, std::string>> cores;
        std::string phys;
        int threads = 0;
        while (std::getline(in, l)) {
            const size_t c = l.find(':');
            if (c == std::string::npos) continue;
            const std::string k = trim(l.substr(0, c)), v = trim(l.substr(c + 1));
            if (k == "processor") ++threads;
            else if (k == "model name" && h->cpu_model.empty()) h->cpu_model = v;
            else if (k == "physical id") phys = v;
            else if (k == "core id") cores.insert({ phys, v });
        }
        if (threads > 0) h->threads = threads;
        if (!cores.empty()) h->cores = (int)cores.size();
    }
    {
        std::istringstream in(slurp(root + "/proc/meminfo"));
        std::string k;
        long long v;
        std::string unit;
        while (in >> k >> v) {
            std::getline(in, unit);
            const int64_t b = (int64_t)v * 1024;
            if (k == "MemTotal:") h->mem_total = b;
            else if (k == "MemAvailable:") h->mem_available = b;
            else if (k == "SwapTotal:") h->swap_total = b;
            else if (k == "SwapFree:") h->swap_free = b;
        }
    }
    h->kernel = line(root + "/proc/sys/kernel/osrelease");
    {
        std::istringstream in(slurp(root + "/etc/os-release"));
        std::string l;
        while (std::getline(in, l))
            if (l.rfind("PRETTY_NAME=", 0) == 0) {
                h->distro = l.substr(12);
                if (h->distro.size() >= 2 && h->distro.front() == '"') h->distro = h->distro.substr(1, h->distro.size() - 2);
            }
    }
    if (exists(root + "/.dockerenv")) h->container = "docker";
    else if (exists(root + "/run/.containerenv")) h->container = "podman";
    else {
        const std::string cg = slurp(root + "/proc/1/cgroup");
        if (cg.find("docker") != std::string::npos) h->container = "docker";
        else if (cg.find("kubepods") != std::string::npos) h->container = "kubernetes";
        else if (cg.find("containerd") != std::string::npos) h->container = "containerd";
    }
    {
        /* The process's cgroup ("/" inside a container's namespace) and its ancestors. v2 is
         * the "0::" line; v1's memory controller is ":memory:", where unlimited is a number
         * near 2^63. */
        std::istringstream in(slurp(root + "/proc/self/cgroup"));
        std::string l;
        int64_t lim = -1;
        while (std::getline(in, l)) {
            int64_t v = -1;
            if (l.rfind("0::", 0) == 0)
                v = cgroup_limit(root + "/sys/fs/cgroup", l.substr(3), "memory.max");
            else if (l.find(":memory:") != std::string::npos)
                v = cgroup_limit(root + "/sys/fs/cgroup/memory", l.substr(l.find(":memory:") + 8),
                                 "memory.limit_in_bytes");
            lim = tighter(lim, v);
        }
        h->cgroup_mem_limit = lim;
    }
    const std::string thp = root + "/sys/kernel/mm/transparent_hugepage";
    h->thp_enabled = selected(thp + "/enabled");
    h->thp_defrag  = selected(thp + "/defrag");
    h->thp_shmem   = selected(thp + "/shmem_enabled");
    {
        std::istringstream in(line(root + "/proc/cmdline"));
        std::string t;
        static const char* const kKeep[] = { "iommu", "amd_iommu", "intel_iommu", "amdgpu.", "ttm.",
                                             "transparent_hugepage", "hugepages", "pcie_aspm",
                                             "pci=", "default_hugepagesz" };
        while (in >> t)
            for (const char* k : kKeep)
                if (t.rfind(k, 0) == 0) { h->cmdline += (h->cmdline.empty() ? "" : " ") + t; break; }
    }
    {
        /* Passthrough or translated. The cards' IOMMU group type (identity = passthrough) is the
         * kernel's own answer; IOMMU presence and boot parameters are the fallback for a card
         * with no group. */
        const bool any_iommu = !list(root + "/sys/class/iommu").empty();
        const bool groups = !list(root + "/sys/kernel/iommu_groups").empty();
        const std::string cl = " " + h->cmdline + " ";
        if (!any_iommu && !groups) {
            h->iommu = "off";
            h->iommu_why = "no IOMMU is registered (/sys/class/iommu is empty)";
        } else if (cl.find(" iommu=pt ") != std::string::npos ||
                   cl.find(" amd_iommu=pt ") != std::string::npos ||
                   cl.find(" intel_iommu=pt ") != std::string::npos) {
            h->iommu = "passthrough";
            h->iommu_why = "iommu=pt on the kernel command line";
        } else if (cl.find("iommu=off") != std::string::npos) {
            h->iommu = "off";
            h->iommu_why = "the kernel command line turns it off";
        } else {
            h->iommu = "unknown";
            h->iommu_why = "an IOMMU is present; its mode is read from the cards' groups below";
        }
    }
    {
        int64_t lim = num(root + "/sys/module/ttm/parameters/pages_limit", -1);
        /* 0 means the driver default, half of RAM. */
        if (lim == 0 && h->mem_total > 0) lim = h->mem_total / 2 / h->page_size;
        h->ttm_pages_limit = lim;
    }
    h->cwsr_enable = (int)num(root + "/sys/module/amdgpu/parameters/cwsr_enable", -1);
    {
        const int64_t d = num(root + "/proc/sys/kernel/io_uring_disabled", -1);
        h->io_uring = d == 0 ? "available" : d == 1 ? "restricted to a group (io_uring_disabled 1)"
                    : d == 2 ? "refused (io_uring_disabled 2)" : "unknown";
    }
    const std::string cpu = root + "/sys/devices/system/cpu/cpu0/cpufreq";
    h->cpu_governor = line(cpu + "/scaling_governor");
    h->cpu_epp      = line(cpu + "/energy_performance_preference");
    h->cpu_driver   = line(cpu + "/scaling_driver");
}

}  /* namespace */

Facts collect(const std::string& root, const std::vector<std::string>* managed) {
    Facts f;
    if (root.empty()) {
        const long ps = sysconf(_SC_PAGE_SIZE);
        if (ps > 0) f.host.page_size = ps;
    }
    read_host(root, &f.host);
    /* Every display-class device, used by the engine or not (an iGPU, an excluded card). */
    for (const std::string& d : list(root + "/sys/bus/pci/devices")) {
        const std::string cls = line(root + "/sys/bus/pci/devices/" + d + "/class");
        if (cls.rfind("0x03", 0) != 0) continue;
        const bool pm =
            !managed || std::find(managed->begin(), managed->end(), d) != managed->end();
        CardFacts c;
        read_card(root, d, pm, &c);
        f.cards.push_back(std::move(c));
    }
    read_kfd(root, &f.cards);
    /* The IOMMU mode as the cards see it, which beats the command line where both exist:
     * identity is passthrough, DMA / DMA-FQ translated. A machine can mix both (an
     * identity-mapped iGPU beside a translated discrete card). */
    std::map<std::string, std::string> by;
    for (const CardFacts& c : f.cards)
        if (!c.iommu_type.empty()) by[c.iommu_type] += (by[c.iommu_type].empty() ? "" : ", ") + c.bdf;
    if (by.size() == 1) {
        const bool id = by.begin()->first == "identity";
        f.host.iommu = id ? "passthrough" : "translated";
        f.host.iommu_why = "the display devices' IOMMU groups are of type " + by.begin()->first;
    } else if (by.size() > 1) {
        f.host.iommu = "mixed";
        f.host.iommu_why.clear();
        for (const auto& [t, bdfs] : by)
            f.host.iommu_why += (f.host.iommu_why.empty() ? "" : "; ") + t + " for " + bdfs;
    }
    return f;
}

std::vector<CardSample> sample_cards(const std::string& root, const std::vector<std::string>& bdfs) {
    /* The line the driver marks with '*', without the mark and the level's index: "2350Mhz". */
    auto current = [](const std::string& path) {
        std::istringstream in(slurp(path));
        std::string l;
        while (std::getline(in, l)) {
            if (l.find('*') == std::string::npos) continue;
            l.erase(std::remove(l.begin(), l.end(), '*'), l.end());
            const size_t colon = l.find(':');
            return trim(colon == std::string::npos ? l : l.substr(colon + 1));
        }
        return std::string();
    };
    std::vector<CardSample> out;
    for (const std::string& bdf : bdfs) {
        const std::string dev = root + "/sys/bus/pci/devices/" + bdf;
        CardSample c;
        c.bdf          = bdf;
        c.perf_level   = line(dev + "/power_dpm_force_performance_level");
        c.sclk_cur     = current(dev + "/pp_dpm_sclk");
        c.mclk_cur     = current(dev + "/pp_dpm_mclk");
        c.busy_percent = (int)num(dev + "/gpu_busy_percent", -1);
        for (const std::string& h : list(dev + "/hwmon")) {
            const std::string hp = dev + "/hwmon/" + h;
            /* amdgpu labels its sensors; an unlabelled temp1 is the edge sensor. */
            for (int t = 1; t <= 8; ++t) {
                const std::string base = hp + "/temp" + std::to_string(t);
                const int64_t v = num(base + "_input", -1);
                if (v < 0) continue;
                const std::string label = line(base + "_label");
                if (label == "edge" || (label.empty() && t == 1)) c.edge_mc = v;
                else if (label == "junction")                     c.junction_mc = v;
                else if (label == "mem")                          c.mem_mc = v;
            }
            if (c.power_uw < 0) c.power_uw = num(hp + "/power1_average", -1);
            if (c.power_uw < 0) c.power_uw = num(hp + "/power1_input", -1);
        }
        out.push_back(std::move(c));
    }
    return out;
}

void probe_process(HostFacts* h) {
    struct rlimit rl{};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0)
        h->memlock_soft = rl.rlim_cur == RLIM_INFINITY ? -2 : (int64_t)rl.rlim_cur;
#ifdef SYS_io_uring_setup
    /* Try the call: a seccomp profile (Docker's default) refuses it whatever the sysctl says.
     * One entry, closed at once. */
    struct { uint32_t sq_entries, cq_entries, flags, sq_thread_cpu, sq_thread_idle, features,
             wq_fd, resv[3]; uint64_t sq_off[5]; uint64_t cq_off[5]; } p{};
    static_assert(sizeof(p) == 120, "io_uring_params is 120 bytes");
    const long fd = syscall(SYS_io_uring_setup, 1, &p);
    if (fd >= 0) {
        close((int)fd);
        h->io_uring = "available";
    } else {
        h->io_uring = fmt("refused (%s)", std::strerror(errno));
    }
#endif
}

std::vector<std::string> env_names(char* const* env, const char* prefix) {
    std::vector<std::string> out;
    const size_t n = std::strlen(prefix);
    for (char* const* e = env; e && *e; ++e) {
        if (std::strncmp(*e, prefix, n) != 0) continue;
        const char* eq = std::strchr(*e, '=');
        out.emplace_back(*e, eq ? (size_t)(eq - *e) : std::strlen(*e));
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  /* namespace diag */
}  /* namespace rad */
