/* rad_util.cpp -- logging, string formatting, Geometry and constraint matching. */
#include <atomic>
#include <chrono>
#include "../rad_internal.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cinttypes>
#include <ctime>
#include <iterator>
#include <unistd.h>

namespace rad {

/* ------------------------------------------------------------------ logging */

/* The initial level comes from RADIANCE_LOG=error|warn|info|debug|trace (default info), so
 * RAD_DEBUG / RAD_TRACE sites are reachable in any process. Read once at startup; log_set_level
 * overrides it (command-line -v, tests, embedding hosts). */
static Log level_from_env() {
    const char* v = std::getenv("RADIANCE_LOG");
    if (!v || !*v) return Log::Info;
    if (!std::strcmp(v, "error")) return Log::Error;
    if (!std::strcmp(v, "warn"))  return Log::Warn;
    if (!std::strcmp(v, "info"))  return Log::Info;
    if (!std::strcmp(v, "debug")) return Log::Debug;
    if (!std::strcmp(v, "trace")) return Log::Trace;
    /* Report a misspelt level instead of silently ignoring it. */
    std::fprintf(stderr, "E RADIANCE_LOG='%s' is not error|warn|info|debug|trace; using info\n", v);
    return Log::Info;
}

static Log g_level = level_from_env();
void log_set_level(Log l) { g_level = l; }
Log  log_level()          { return g_level; }

static const char* level_tag(Log l) {
    switch (l) {
        case Log::Error: return "E";
        case Log::Warn:  return "W";
        case Log::Info:  return "I";
        case Log::Debug: return "D";
        default:         return "T";
    }
}

/* Colour only when stderr is a terminal, as the live view does (spec §16). */
static const char* level_colour(Log l) {
    if (!isatty(2)) return "";
    switch (l) {
        case Log::Error: return "\033[31m";
        case Log::Warn:  return "\033[33m";
        case Log::Debug: return "\033[36m";
        case Log::Trace: return "\033[90m";
        default:         return "";
    }
}

/* Thread-local: the ranks' declare threads run concurrently and each tags its own lines. */
static thread_local int  g_log_rank  = -1;
static bool              g_rank_tags = false;

void log_rank_tags(bool on) { g_rank_tags = on; }
LogRank::LogRank(int rank) : prev_(g_log_rank) { g_log_rank = rank; }
LogRank::~LogRank() { g_log_rank = prev_; }

static thread_local int         g_owner = 0;
static thread_local const char* g_model = nullptr;
static bool                     g_model_tags = false;

void log_model_tags(bool on) { g_model_tags = on; }
int  engine_owner() { return g_owner; }
EngineScope::EngineScope(int owner, const char* model) : prev_owner_(g_owner), prev_model_(g_model) {
    g_owner = owner;
    g_model = model;
}
EngineScope::~EngineScope() {
    g_owner = prev_owner_;
    g_model = prev_model_;
}

static std::atomic<uint64_t> g_last_line_ns{0};
uint64_t log_last_line_ns() { return g_last_line_ns.load(std::memory_order_relaxed); }

std::string log_sanitize(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        const unsigned char b = (unsigned char)c;
        if ((b < 0x20 && b != '\t') || b == 0x7f) c = '?';
    }
    return out;
}

void log_emit(Log l, const char* file, int line, const char* fmt, ...) {
    const char* base = std::strrchr(file, '/');
    base = base ? base + 1 : file;

    /* A line longer than the stack buffer is formatted again in full, so a trace row of many
     * values (RADIANCE_DUMP_HEAD) is never cut at 4 KiB. */
    char buf[4096];
    std::string big;
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n >= (int)sizeof buf) {
        big.resize((size_t)n + 1);
        vsnprintf(big.data(), big.size(), fmt, ap2);
        big.resize((size_t)n);
    }
    va_end(ap2);
    const char* msg = big.empty() ? buf : big.c_str();

    const char* col = level_colour(l);
    const char* rst = col[0] ? "\033[0m" : "";

    char rank[80] = "";
    const bool tag_rank = g_rank_tags && g_log_rank >= 0;
    if (g_model_tags && g_model) {
        if (tag_rank) snprintf(rank, sizeof rank, "[%.60s/r%d]", g_model, g_log_rank);
        else          snprintf(rank, sizeof rank, "[%.60s]", g_model);
    } else if (tag_rank) {
        snprintf(rank, sizeof rank, "[%d]", g_log_rank);
    }

    /* Debug and Trace carry the source site; Info and above do not. */
    if ((int)l >= (int)Log::Debug)
        fprintf(stderr, "%s%s%s%s %s:%d  %s\n", col, level_tag(l), rank, rst, base, line, msg);
    else
        fprintf(stderr, "%s%s%s%s %s\n", col, level_tag(l), rank, rst, msg);
    g_last_line_ns.store(rad_mono_ns(), std::memory_order_relaxed);
}

void fatal(const char* fmt, ...) {
    char msg[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%sradiance: fatal: %s%s\n", isatty(2) ? "\033[31m" : "", msg,
            isatty(2) ? "\033[0m" : "");
    fflush(stderr);
    _exit(70);
}

std::string fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(nullptr, 0, f, ap);
    va_end(ap);
    std::string s;
    if (n > 0) {
        s.resize((size_t)n);
        vsnprintf(s.data(), (size_t)n + 1, f, ap2);
    }
    va_end(ap2);
    return s;
}

uint64_t rad_mono_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string humanb(int64_t b) {
    static const char* u[] = { "B", "KiB", "MiB", "GiB", "TiB" };
    double v = (double)b;
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; ++i; }
    return i == 0 ? fmt("%lld B", (long long)b) : fmt("%.2f %s", v, u[i]);
}

std::vector<std::string_view> split_ws(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == ',')) ++i;
        size_t j = i;
        while (j < s.size() && s[j] != ' ' && s[j] != '\t' && s[j] != ',') ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

/* ------------------------------------------------------------------ Geometry */
/* Linear scan on purpose: an op has a handful of parameters and the selector reads this at
 * declare, not per step. */
size_t Geometry::slot(std::string_view k, int kind) {
    size_t i = 0;
    while (i < keys_.size() && keys_[i] != k) ++i;
    if (i == keys_.size()) {
        keys_.emplace_back(k);
        svals_.emplace_back();
        ivals_.push_back(0);
        dvals_.push_back(0.0);
        kinds_.push_back(kind);
    }
    kinds_[i] = kind;
    return i;
}

void Geometry::set_i(std::string_view k, long long v) { ivals_[slot(k, RAD_P_INT)] = v; rebuild(); }
void Geometry::set_f(std::string_view k, double v)    { dvals_[slot(k, RAD_P_F64)] = v; rebuild(); }
void Geometry::set_s(std::string_view k, std::string_view v) {
    svals_[slot(k, RAD_P_STR)] = std::string(v);
    rebuild();
}

bool Geometry::get_i(std::string_view k, long long* out) const {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k && kinds_[i] == RAD_P_INT) { if (out) *out = ivals_[i]; return true; }
    return false;
}

bool Geometry::get_f(std::string_view k, double* out) const {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k && kinds_[i] == RAD_P_F64) { if (out) *out = dvals_[i]; return true; }
    return false;
}

const char* Geometry::get_s(std::string_view k) const {
    for (size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == k && kinds_[i] == RAD_P_STR) return svals_[i].c_str();
    return nullptr;
}

bool Geometry::has(std::string_view k) const {
    for (auto& s : keys_) if (s == k) return true;
    return false;
}

/* Rebuild the whole view: an append can reallocate keys_ and svals_, and short strings live
 * inside their element, so every c_str() may have moved. */
void Geometry::rebuild() {
    view_.clear();
    view_.reserve(keys_.size());
    for (size_t i = 0; i < keys_.size(); ++i) {
        RadParam p{};
        p.key  = keys_[i].c_str();
        p.kind = kinds_[i];
        p.ival = ivals_[i];
        p.dval = dvals_[i];
        p.sval = kinds_[i] == RAD_P_STR ? svals_[i].c_str() : nullptr;
        view_.push_back(p);
    }
}

std::string Geometry::str() const {
    std::string s;
    for (size_t i = 0; i < keys_.size(); ++i) {
        if (i) s += ' ';
        s += keys_[i];
        s += '=';
        if (kinds_[i] == RAD_P_STR)      s += svals_[i];
        else if (kinds_[i] == RAD_P_F64) s += fmt("%g", dvals_[i]);
        else                             s += fmt("%lld", ivals_[i]);
    }
    return s;
}

/* ------------------------------------------------------------------ constraint matching */
/* A constraint on a key the geometry lacks does not hold (libr4d's rule): a row constraining an
 * optional parameter serves only callers that state it, and others get a row without it. */
bool constraint_holds(const RadConstraint& c, const Geometry& g) {
    if (c.op == RAD_C_IN) {
        const char* v = g.get_s(c.key);
        if (!v) {
            /* An integer geometry against a string set: compare its decimal form, so
             * C_IN("causal", "0 1") works against causal=1. */
            long long iv;
            if (!g.get_i(c.key, &iv)) return false;
            std::string s = fmt("%lld", iv);
            for (auto tok : split_ws(c.sval ? c.sval : "")) if (tok == s) return true;
            return false;
        }
        for (auto tok : split_ws(c.sval ? c.sval : "")) if (tok == v) return true;
        return false;
    }

    long long v;
    if (!g.get_i(c.key, &v)) return false;
    switch (c.op) {
        case RAD_C_EQ:  return v == c.ival;
        case RAD_C_LE:  return v <= c.ival;
        case RAD_C_GE:  return v >= c.ival;
        case RAD_C_DIV: return c.ival != 0 && (v % c.ival) == 0;
        default:        return false;
    }
}

std::string constraint_why(const RadConstraint& c, const Geometry& g) {
    static const char* opname[] = { "==", "<=", ">=", "%", "in" };
    const char* on = (c.op >= 0 && c.op < (int)std::size(opname)) ? opname[c.op] : "?";

    if (c.op == RAD_C_IN) {
        const char* v = g.get_s(c.key);
        long long iv;
        std::string have = v ? std::string(v)
                             : (g.get_i(c.key, &iv) ? fmt("%lld", iv) : std::string("<absent>"));
        return fmt("%s in {%s}, have %s", c.key, c.sval ? c.sval : "", have.c_str());
    }
    long long v;
    if (!g.get_i(c.key, &v)) return fmt("%s %s %lld, but the query did not name %s",
                                        c.key, on, c.ival, c.key);
    if (c.op == RAD_C_DIV) return fmt("%s %% %lld == 0, have %lld", c.key, c.ival, v);
    return fmt("%s %s %lld, have %lld", c.key, on, c.ival, v);
}

}  /* namespace rad */

/* ------------------------------------------------------------------ the C surface
 *
 * Intentionally empty. rad_strerror, the dtype and tensor helpers and rad_args_geti/gets are
 * `static inline` in the ABI headers; defined in rad_core they would be unreachable from plugins
 * (see the note above rad_strerror in rad_types.h). A helper plugins need goes in the header; a
 * core-only helper goes in namespace rad, not behind extern "C". */
