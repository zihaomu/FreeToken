#include "liveview.h"
#include "iface.h"
#include "rad_internal.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <sys/ioctl.h>
#include <unistd.h>

namespace rad {
namespace server {

static_assert((int)kLiveTiers == kTierCount,
              "liveview.h restates rad::kTierCount; the tier enum grew and the snapshot did not");

void snapshot_from(const SchedMetrics& m, LiveSnapshot* out) {
    out->waiting = m.waiting;
    out->running = m.running;
    out->preempted = m.preempted;
    out->kv_util = m.kv_util;
    out->prefix_hit_rate = m.prefix_hit_rate;
    out->prefix_evictions = m.prefix_evictions;
    out->prefill_tps = m.prefill_tps;
    out->decode_tps = m.decode_tps;
    out->draft_acceptance = m.draft_acceptance;
    out->total_requests = m.total_requests;
    out->preemptions = m.preemptions;
    /* Throughput decays toward zero on its own; a step count frozen between two draws is what
     * shows the loop itself stopped. */
    out->step = m.steps;
}

/* ------------------------------------------------------------------ small formatting */

namespace {

/* Tier -> colour, in rad::Tier order: VRAM, HostPinned, HostPageable, SSD, Mapped. Green is
 * resident, red is on disk, so the plane shows at a glance where the model lives. */
const char* tier_colour(uint8_t t) {
    switch (t) {
        case 0: return "\033[32m";   /* VRAM */
        case 1: return "\033[33m";   /* host, pinned */
        case 2: return "\033[35m";   /* host, pageable */
        case 4: return "\033[36m";   /* the container's mapping */
        default: return "\033[31m";  /* SSD */
    }
}
/* Tier names, in rad::Tier order. */
const char* const kTierNames[kLiveTiers] = {"vram", "pinned", "host", "ssd", "mapped"};

/* How slow a tier is to read, in rad::Tier order, for a column's worst tier: SSD is the slowest,
 * so it outranks the mapping though it comes first in the enum. */
const uint8_t kTierRank[kLiveTiers] = {0, 1, 2, 4, 3};

char tier_glyph(uint8_t t) {
    switch (t) {
        case 0: return '#';
        case 1: return '+';
        case 2: return '-';
        default: return '.';
    }
}

std::string bar(double frac, int width, bool colour) {
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    int fill = (int)std::lround(frac * width);
    std::string s;
    if (colour) s += frac > 0.9 ? "\033[31m" : (frac > 0.7 ? "\033[33m" : "\033[32m");
    s.append((size_t)fill, '=');
    if (colour) s += "\033[0m";
    s.append((size_t)(width - fill), ' ');
    return s;
}

std::string gib(int64_t bytes) {
    char b[48];
    snprintf(b, sizeof b, "%.1fG", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    return b;
}

std::string f2(double v) {
    char b[32];
    snprintf(b, sizeof b, "%.2f", v);
    return b;
}

/* Tokens per second, never SI-scaled: rates are compared digit by digit, and 6.12k beside 5.98k
 * hides what 6116 beside 5979 shows. */
std::string tps(double v) {
    char b[32];
    snprintf(b, sizeof b, v >= 99.95 ? "%.0f" : "%.1f", v);
    return b;
}

/* ---- the terminal form's figures.
 *
 * Same rules as the web page (dashboard.cpp), so both views print the same figure. Three
 * significant figures and at most two decimals, rounded before the unit is chosen (otherwise
 * 1023.7 MiB prints as "1024 MiB"). Sizes in binary units, as the card and every --*-mib flag
 * count. Counters exact, grouped in threes by a narrow space past four digits, as SI writes them.
 * A share never rounds onto 0% or 100%.
 *
 * The piped form keeps its own formatting, since log consumers parse its keys and value shapes. */

/* The digits of 0 <= a < 999.5; the thresholds are where the rounded value gains a digit. */
std::string sig3(double a) {
    char b[32];
    snprintf(b, sizeof b, "%.*f", a >= 99.95 ? 0 : a >= 9.995 ? 1 : 2, a);
    return b;
}

/* A value on a unit ladder: steps up while the value would round to four digits. `exact` prints
 * the first rung as a whole number (no "5.00 B"). */
std::string ladder(double v, double base, const char* const* units, int n_units, bool exact,
                   const char* sep) {
    if (!std::isfinite(v)) return "-";
    const char* u = units[0];
    std::string n;
    if (v == 0.0) {
        n = "0";
    } else {
        double a = std::fabs(v);
        int i = 0;
        while (a >= 999.5 && i < n_units - 1) { a /= base; ++i; }
        u = units[i];
        n = (i == 0 && exact) ? std::to_string(std::llround(a)) : sig3(a);
        if (v < 0) n = "-" + n;
    }
    return *u ? n + sep + u : n;
}

const char* const kIec[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };
const char* const kPfx[] = { "", "k", "M", "G", "T", "P" };

std::string size_iec(int64_t bytes) { return ladder((double)bytes, 1024.0, kIec, 6, true, " "); }

/* Used and total of one quantity, the unit said once when it is the same unit. */
std::string size_of(int64_t used, int64_t total) {
    std::string a = size_iec(used), b = size_iec(total);
    const size_t ia = a.rfind(' '), ib = b.rfind(' ');
    if (ia != std::string::npos && ib != std::string::npos && a.substr(ia) == b.substr(ib))
        return a.substr(0, ia) + " / " + b;
    return a + " / " + b;
}

/* A continuous quantity (watts, degrees) to three figures and an SI prefix. Not for tokens per
 * second, which tps() prints whole. */
std::string num_si(double v) { return ladder(v, 1000.0, kPfx, 6, false, ""); }

/* A counter, exact, grouped in threes by U+202F past four digits. */
std::string grouped(int64_t n) {
    std::string d = std::to_string(n < 0 ? -n : n), o;
    for (size_t i = 0; i < d.size(); ++i) {
        if (i && d.size() > 4 && (d.size() - i) % 3 == 0) o += " ";
        o += d[i];
    }
    return n < 0 ? "-" + o : o;
}

/* A fraction as a percentage to one decimal. Only exactly 0 reads 0 and only exactly 1 reads
 * 100. */
std::string share(double f) {
    if (!std::isfinite(f)) return "-";
    char b[32];
    snprintf(b, sizeof b, "%.1f", 100.0 * f);
    if (f > 0 && strcmp(b, "0.0") == 0) return "<0.1%";
    if (f < 1 && strcmp(b, "100.0") == 0) return ">99.9%";
    return std::string(b) + "%";
}

}  /* namespace */

/* ------------------------------------------------------------------ construction */

LiveView::LiveView(Source src, int interval_ms, FILE* out)
    : src_(std::move(src)), interval_ms_(interval_ms), out_(out ? out : stdout) {
    tty_ = isatty(fileno(out_)) != 0;
    if (tty_) {
        struct winsize ws;
        if (ioctl(fileno(out_), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 20) cols_ = ws.ws_col;
    }
}

LiveView::~LiveView() { stop(); }

void LiveView::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread([this] { loop(); });
}

void LiveView::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    if (tty_) {
        /* Leave the terminal as we found it: cursor visible, no leftover attributes. */
        fputs("\033[?25h\033[0m\n", out_);
        fflush(out_);
    }
}

void LiveView::loop() {
    if (tty_) fputs("\033[?25l", out_);   /* hide the cursor; the redraw is what moves it */
    LiveSnapshot s;
    while (running_.load()) {
        s = LiveSnapshot{};
        if (src_) src_(s);
        std::string frame = tty_ ? render_screen(s) : render_line(s);
        fwrite(frame.data(), 1, frame.size(), out_);
        fflush(out_);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
}

/* ------------------------------------------------------------------ the piped form
 *
 * One line per interval, key=value, no escape sequences or cursor movement, so a log file or a
 * `| tee` stays greppable and diffable.
 */
std::string LiveView::render_line(const LiveSnapshot& s) const {
    std::string o = "live";
    o += " step=" + std::to_string(s.step);
    o += " run=" + std::to_string(s.running);
    o += " wait=" + std::to_string(s.waiting);
    if (s.preempted) o += " preempt=" + std::to_string(s.preempted);
    o += " kv=" + f2(s.kv_util);
    o += " prefix_hit=" + f2(s.prefix_hit_rate);
    if (s.prefix_evictions) o += " prefix_evict=" + std::to_string(s.prefix_evictions);
    o += " prefill_tps=" + tps(s.prefill_tps);
    o += " decode_tps=" + tps(s.decode_tps);
    if (s.draft_acceptance > 0.0) o += " draft_accept=" + f2(s.draft_acceptance);

    for (const auto& c : s.cards) {
        o += " gpu" + std::to_string(c.index) + "=";
        o += f2(c.util);
        o += "/vram:" + gib(c.vram_used) + "of" + gib(c.vram_total);
        if (c.power_w > 0) { o += "/pw:" + f2(c.power_w); }
        if (c.temp_c > 0)  { o += "/t:" + f2(c.temp_c); }
    }

    if (s.experts.n_units > 0) {
        /* The snapshot's own unit tally, not a walk of the plane. The plane is a rectangle a
         * model need not fill (a draft head with fewer experts than the trunk leaves empty
         * cells), so counting cells would charge empty ones to some tier. */
        o += " experts";
        for (int t = 0; t < kLiveTiers; ++t)
            o += std::string(" ") + kTierNames[t] + "=" + std::to_string(s.experts.units[t]);
        o += " resident=" + f2(s.experts.n_units
                                   ? 100.0 * (double)s.experts.units[0] / (double)s.experts.n_units
                                   : 0.0) + "%";
        /* Hit rate, which residency does not imply: a plan holding most units still misses on
         * most tokens if the hot ones are outside. */
        if (s.experts.routed > 0)
            o += " hit=" + f2(100.0 * (double)s.experts.routed_resident /
                              (double)s.experts.routed) + "%";
        if (s.experts.promotions || s.experts.demotions) {
            o += " promo=" + std::to_string(s.experts.promotions);
            o += " demo=" + std::to_string(s.experts.demotions);
            o += " readmit=" + std::to_string(s.experts.readmits);
        }
    }
    /* Link traffic by consumer (collective, mover, MoE GEMM streaming, KV streaming), cumulative
     * so a `| tee` of these lines can be differenced into rates. */
    if (s.link.present &&
        (s.link.ar_bytes || s.link.h2d_bytes || s.link.d2h_bytes || s.link.stream_bytes ||
         s.link.kv_streaming)) {
        o += " link ar=" + std::to_string(s.link.ar_bytes);
        o += " h2d=" + std::to_string(s.link.h2d_bytes);
        o += " d2h=" + std::to_string(s.link.d2h_bytes);
        if (s.link.stream_bytes) o += " stream=" + std::to_string(s.link.stream_bytes);
        if (s.link.ssd_bytes) o += " ssd=" + std::to_string(s.link.ssd_bytes);
        /* Printed at zero when the switch is on: the zero is the measurement. */
        if (s.link.kv_streaming) o += " kv=" + std::to_string(s.link.kv_stream_bytes);
    }
    o += "\n";
    return o;
}

/* ------------------------------------------------------------------ the terminal form */

std::string LiveView::render_screen(const LiveSnapshot& s) const {
    const int W = cols_;
    std::string o;
    o.reserve(4096);
    o += "\033[H\033[2J";          /* home, clear */

    o += "\033[1mradiance\033[0m  ";
    o += s.model.empty() ? "-" : s.model;
    o += "   step " + grouped(s.step);
    o += "   requests " + grouped(s.total_requests) + "\n\n";

    for (const auto& c : s.cards) {
        char head[96];
        snprintf(head, sizeof head, "  gpu%d %-14.14s ", c.index, c.name.c_str());
        o += head;
        o += "util [" + bar(c.util, std::max(10, W / 4), true) + "] ";
        o += share(c.util) + "  ";
        double vf = c.vram_total > 0 ? (double)c.vram_used / (double)c.vram_total : 0.0;
        o += "vram [" + bar(vf, std::max(10, W / 6), true) + "] ";
        o += size_of(c.vram_used, c.vram_total);
        if (c.power_w > 0) o += "  " + num_si(c.power_w) + " W";
        if (c.temp_c > 0)  o += "  " + num_si(c.temp_c) + " \u00b0C";
        o += "\n";
    }
    if (!s.cards.empty()) o += "\n";

    o += "  running " + std::to_string(s.running) +
         "   waiting " + std::to_string(s.waiting);
    if (s.preempted) o += "   preempted " + std::to_string(s.preempted);
    o += "\n";
    o += "  kv      [" + bar(s.kv_util, std::max(10, W / 4), true) + "] " +
         share(s.kv_util) + "   prefix hit " + share(s.prefix_hit_rate);
    /* Evictions predict a prefix-cache collapse that the hit rate reports only afterwards: past
     * ~75% kv the cache drops prefixes the next turn wants. Shown only when non-zero. */
    if (s.prefix_evictions) o += "   evicted " + grouped(s.prefix_evictions);
    o += "\n";
    o += "  prefill " + tps(s.prefill_tps) + " tok/s    decode " + tps(s.decode_tps) + " tok/s";
    if (s.draft_acceptance > 0.0)
        o += "    draft accept " + share(s.draft_acceptance);
    o += "\n";

    if (s.experts.n_layers > 0 && s.experts.n_experts > 0) {
        o += "\n  experts by tier  " + grouped(s.experts.n_units) + " units  " +
             size_iec(s.experts.total_bytes) + "\n  ";
        /* Residency per tier in the legend: units, their share, and bytes. Unit share and byte
         * share differ because the tiers hold different weight classes. */
        char tl[192];
        for (int t = 0; t < kLiveTiers; ++t) {
            if (!s.experts.units[t]) continue;
            snprintf(tl, sizeof tl, "%s%c %-6s\033[0m %s  %s  %s   ",
                     tier_colour((uint8_t)t), tier_glyph((uint8_t)t),
                     kTierNames[t],
                     grouped(s.experts.units[t]).c_str(),
                     share((double)s.experts.units[t] / (double)s.experts.n_units).c_str(),
                     size_iec(s.experts.bytes[t]).c_str());
            o += tl;
        }
        o += "\n";

        /* The plane is usually wider than the terminal (128 experts x 48 layers is typical), so
         * each column group shows its worst tier, not the average: one SSD expert makes the
         * group slow. */
        const int avail = W - 8;
        const int stride = std::max(1, (s.experts.n_experts + avail - 1) / std::max(1, avail));
        for (int l = 0; l < s.experts.n_layers; ++l) {
            char lb[16];
            snprintf(lb, sizeof lb, "  %3d ", l);
            o += lb;
            for (int e = 0; e < s.experts.n_experts; e += stride) {
                uint8_t worst = 0;
                for (int k = 0; k < stride && e + k < s.experts.n_experts; ++k) {
                    size_t idx = (size_t)l * s.experts.n_experts + e + k;
                    if (idx >= s.experts.tier.size()) continue;
                    const uint8_t t = s.experts.tier[idx];
                    if (t < kLiveTiers && kTierRank[t] > kTierRank[worst]) worst = t;
                }
                o += tier_colour(worst);
                o += tier_glyph(worst);
            }
            o += "\033[0m\n";
        }
        if (s.experts.promotions || s.experts.demotions) {
            o += "  moves  +" + grouped(s.experts.promotions) +
                 "  -" + grouped(s.experts.demotions) + "\n";
        }
    }
    return o;
}

}  /* namespace server */
}  /* namespace rad */
