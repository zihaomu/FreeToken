/* diag.h -- the startup report: what was configured, what it runs on, and what looks wrong.
 *
 * One block at the end of startup, instead of decisions scattered through the log (defaults are
 * never logged at all): every option with its value and source, the machine as the kernel
 * describes it, and findings that cost speed or correctness, each with a fix.
 *
 * Each part is built from data, not a list kept beside the code:
 *   Configuration  every row of the option table (core/config.cpp); a value startup resolved
 *                  carries its reason (Decision).
 *   Hardware       read from /proc and /sys under a root, so tests can supply a tree and a
 *                  FROM-scratch container (no shell, lspci or rocm-smi) reads the same files.
 *   Findings       small functions over the two above (checks.cpp), testable without a card.
 *
 * A field that could not be read is reported as unknown, never omitted.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rad {
struct Config;
namespace diag {

/* ------------------------------------------------------------------ decisions */
/* One value startup resolved, and why. `key` is the option it resolves (`--placement`), shown on
 * that option's row, or a topic of its own (`vram split`). Recording a key again replaces the
 * earlier answer, so a later stage can revise a choice. */
struct Decision {
    std::string key;
    std::string value;
    std::string why;
};

void decide(std::vector<Decision>* list, const std::string& key, const std::string& value,
            const std::string& why);

/* ------------------------------------------------------------------ the configuration table */
/* One option: the value in force, its source (`default`, `command line`, an environment
 * variable, or `auto` when startup derived it), and the reason when derived. */
struct ConfigRow {
    std::string flag;
    std::string value;
    std::string source;
    std::string why;
};

/* Every option in core/config.cpp's table, in order, with the value `c` holds and its source
 * from Config::given. Secrets are not shown: --api-key is "set" or "not set". */
std::vector<ConfigRow> config_rows(const Config& c);

/* The options the table holds, for a test that every one reaches the report. */
std::vector<std::string> config_option_names();

/* ------------------------------------------------------------------ hardware */
/* One PCIe link between the root port and a card. Speeds are the kernel's strings
 * ("32.0 GT/s PCIe"), widths in lanes; -1 / "" where the file is absent (some bridges inside a
 * card's own switch). */
struct PciHop {
    std::string bdf;
    std::string cur_speed, max_speed;
    int         cur_width = -1, max_width = -1;
};

struct CardFacts {
    std::string bdf;                 /* 0000:03:00.0 */
    std::string vendor, pci_device;  /* 0x1002, 0x7551 */
    std::string driver;              /* the bound kernel driver, "" for none */
    std::string drm;                 /* card1, "" when the driver made no DRM node */
    std::string gfx;                 /* gfx1201, from the KFD topology; "" when unknown */
    int         simd = -1;           /* SIMDs the KFD topology reports */
    int         numa_node = -1;
    int64_t     vram_total = -1, vram_visible = -1;   /* the BAR the host and the peers see */
    int         busy_percent = -1;
    std::string runtime_status;      /* active | suspended, "" when unknown */
    std::string runtime_control;     /* auto | on */
    std::string iommu_type;          /* the card's IOMMU group: identity, DMA, DMA-FQ */
    std::vector<PciHop> chain;       /* root port first, the card's own link last */
    std::vector<std::string> dpm_pcie;   /* the SMU's link levels, as listed */
    int         dpm_pcie_cur = -1;   /* the level marked current, -1 unknown */
    std::string sclk, mclk;          /* "500-2350 MHz", the lowest and highest DPM level */
    int64_t     power_cap_uw = -1, power_cap_default_uw = -1, power_cap_max_uw = -1;
    std::string vddgfx_offset;       /* "-75 mV" from pp_od_clk_voltage, "" when unknown */
    bool        integrated = false;  /* the KFD topology's node has no local memory of its own */

    /* Filled by the engine for the cards it uses. */
    std::string name;                /* the runtime's product name */
    int         rank = -1;           /* the tensor-parallel rank it serves, -1 none */
    int         device = -1;         /* the backend's device index, -1 none */
    std::string dma;                 /* what carries its copy lists (dev_dma_lane), "" unknown */
};

struct HostFacts {
    std::string cpu_model;
    int         cores = -1, threads = -1;
    int64_t     mem_total = -1, mem_available = -1, swap_total = -1, swap_free = -1;
    std::string kernel, distro;
    std::string container;           /* "docker", "podman", "" for none seen */
    int64_t     cgroup_mem_limit = -1;   /* bytes; -2 is "max", -1 unknown */
    int64_t     memlock_soft = -1;       /* bytes; -2 is unlimited, -1 unknown */
    std::string thp_enabled, thp_defrag, thp_shmem;   /* the selected word of each */
    std::string iommu;               /* passthrough | translated | off | unknown */
    std::string iommu_why;           /* what that was read from */
    std::string cmdline;             /* the boot parameters that bear on a card, space separated */
    int64_t     ttm_pages_limit = -1;    /* pages; TTM's limit on the system memory it allocates */
    int64_t     page_size = 4096;
    int         cwsr_enable = -1;
    std::string io_uring;            /* available | refused (why) | unknown */
    std::string cpu_governor, cpu_epp, cpu_driver;
    std::string runtime;             /* the device runtime's version, "" for none */
    std::string media;               /* "libavformat <v>" / "absent (built without libav*)" */
};

struct Facts {
    HostFacts              host;
    std::vector<CardFacts> cards;    /* every display-class PCI device, in address order */
};

/* Reads /proc and /sys under `root` ("" is the live system). Never fails: unreadable fields are
 * left unknown.
 *
 * `managed`, when given, lists the cards (by PCI address) whose power-management files are read:
 * DPM tables, overdrive, busy percentage, hwmon power limits. The driver answers those from the
 * SMU under its power-management lock and wakes a runtime-suspended card to do so, so a serving
 * process reads them only for its own cards. Null reads them for every card. */
Facts collect(const std::string& root, const std::vector<std::string>* managed = nullptr);

/* A CARD'S STATE AT ONE MOMENT, for --diagnose to read before and after each run: a run that
 * starts hotter, or under a different performance level, than the one it is compared with is
 * measuring a different card. Read under the same root as collect(); unreadable fields stay unknown.
 * The power-management files are answered by the SMU under its lock, so this is read between
 * runs, never during one. */
struct CardSample {
    std::string bdf;
    int64_t     edge_mc = -1, junction_mc = -1, mem_mc = -1;   /* hwmon temperatures, millidegrees C */
    int64_t     power_uw = -1;      /* power1_average, else power1_input */
    std::string perf_level;         /* power_dpm_force_performance_level: auto, high, profile_peak, ... */
    std::string sclk_cur, mclk_cur; /* the DPM level marked current, as the driver lists it */
    int         busy_percent = -1;
};

std::vector<CardSample> sample_cards(const std::string& root, const std::vector<std::string>& bdfs);

/* Fills what only a system call answers for the live process: the memlock limit and whether
 * io_uring is allowed (a container's seccomp profile may refuse it). */
void probe_process(HostFacts* h);

/* Sorted names of the variables in `env` (environ-shaped, null-terminated) starting with
 * `prefix`. Names only: values may be paths, host names or tokens, and the report gets pasted
 * into issues. */
std::vector<std::string> env_names(char* const* env, const char* prefix);

/* ------------------------------------------------------------------ what the engine did */
struct RunFacts {
    std::string version;                 /* the engine's, for the report's first line */
    bool        device_build = true;     /* built with a device runtime (HIP) */
    int         tp = 1;
    bool        host_backend = false;
    bool        failed = false;          /* startup stopped; the report is what it got to */
    int         p2p = -1;                /* 1 peer to peer, 0 through host memory, -1 n/a */
    bool        p2p_forced_off = false;  /* --p2p off */
    std::string p2p_blocked;             /* the pairs the driver refused */
    std::vector<std::string> ignored;    /* devices left out, each with the reason */
    bool        tp_wire_lossy = false;
    int64_t     tp_wire_min_kb = 0;
    bool        prefix_cache = true;
    /* Why the engine disabled prefix caching itself (KL mode, an embedding model); "" when it is
     * on or the operator turned it off. */
    std::string prefix_cache_off_why;
    int64_t     host_pool_mib = 0;       /* a rank, as configured */
    int64_t     pinned_bytes = -1;       /* every pinned host allocation the process made */
    int64_t     host_weight_bytes = 0;   /* weights a rank reads from the pinned host pool */
    int64_t     file_weight_bytes = 0;   /* and from the container on disk */
    int64_t     locked_bytes = 0;        /* memory locked for a host gather (an n-gram copy) */
    std::string ngram_placement;         /* where an n-gram table went, "" for a model without one */
    bool        reference_kernels = false;
    bool        profile_ops = false;
    bool        deterministic = false;
    /* Started with --proceed-with-incompatible-configuration: the report opens with
     * unsupported_banner() and marks each incompatible finding as started anyway. */
    bool        unsupported = false;
    /* -v: the CONFIGURATION section lists every option; without it, the options not at their
     * default and a count of the rest (/diagnostics always carries every one). */
    bool        all_options = false;
    /* Names of the RADIANCE_* variables set (env_names): settings outside the option table,
     * such as the log level, a pinned tuning or an instrument left on. */
    std::vector<std::string> environment;
};

/* ------------------------------------------------------------------ findings */
/* Incompatible: a machine setting known to break radiance; startup stops on it with the fix unless
 * --proceed-with-incompatible-configuration starts it anyway (incompatible()). */
struct Finding {
    enum Level { Note = 0, Warn = 1, Incompatible = 2 };
    Level       level = Note;
    std::string id;     /* stable, for a test or a grep: "link-below-max" */
    std::string what;   /* what was seen, with the numbers */
    std::string why;    /* what it costs */
    std::string fix;    /* what to do: a flag, a setting, a command */
};

std::vector<Finding> check(const Facts& f, const RunFacts& r);

/* The incompatible findings alone, from the machine only, so startup can refuse before it loads
 * anything: the serving cards (rank >= 0) and their host. check() includes them. */
std::vector<Finding> incompatible(const Facts& f);

/* The box --proceed-with-incompatible-configuration puts at the top of the log and the startup
 * report, one log line an element. */
std::vector<std::string> unsupported_banner();

/* ------------------------------------------------------------------ the report */
std::string render(const std::vector<ConfigRow>& config, const std::vector<Decision>& derived,
                   const Facts& f, const RunFacts& r, const std::vector<Finding>& findings);

/* The same, as a JSON object in text: {"configuration", "derived", "hardware", "findings"}. */
std::string to_json(const std::vector<ConfigRow>& config, const std::vector<Decision>& derived,
                    const Facts& f, const RunFacts& r, const std::vector<Finding>& findings);

/* Merges the decisions into the option rows: a decision whose key is an option's flag supplies
 * that row's value and reason (and `auto` as its source when the option was not given). The
 * decisions no row took are returned in `derived`, in the order they were made. */
void merge(std::vector<ConfigRow>* rows, const std::vector<Decision>& decisions,
           std::vector<Decision>* derived);

/* "32.0 GT/s PCIe" -> 32.0; 0 when it does not parse. */
double link_gts(const std::string& speed);

/* "32.0 GT/s x16" from a sysfs speed and width; "?" for an unknown speed. */
std::string link_text(const std::string& speed, int width);

/* "300 W" from microwatts; "unknown" when negative. */
std::string watts_text(int64_t uw);

/* Busy percentage above which a link below its maximum is not idle parking. */
constexpr int kBusyPercent = 30;

}  /* namespace diag */
}  /* namespace rad */
