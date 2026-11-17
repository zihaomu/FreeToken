/* rad_internal.h -- shared by everything under core/, tools/ and tests/. Not installed and not
 * visible to plugins, which see only abi/. C++20 is fine here.
 */
#pragma once
#include "rad_abi.h"
#include "rad_device.h"
#include "rad_builder.h"
#include "rad_runtime.h"
#include "rad_format.h"

#include <cstdio>
#include <cstdarg>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

namespace rad {

/* ------------------------------------------------------------------ logging */
/* Normal operation is a few lines. Everything verbose is off by default (spec §16). */
enum class Log { Error = 0, Warn = 1, Info = 2, Debug = 3, Trace = 4 };

void log_set_level(Log l);
Log  log_level();
void log_emit(Log l, const char* file, int line, const char* fmt, ...)
     __attribute__((format(printf, 4, 5)));
/* rad_mono_ns of the last line written at any level; StallWatch reads it as progress. 0 before
 * the first. */
uint64_t log_last_line_ns();

/* A copy safe to interpolate into a log line: every byte below 0x20 but '\t', and DEL, becomes
 * '?'. Client-controlled text reaches the log -- session labels, echoed request fields -- and an
 * unfiltered newline forges a line while an ESC sequence drives the terminal that reads it.
 * Multi-byte UTF-8 is left alone; only control bytes are replaced. */
std::string log_sanitize(const std::string& s);

/* Rank tags on log lines. Per-rank components (pools, planner, mover, sampler) print the same
 * line once per rank, so under TP each line needs its rank. LogRank sets the tag as a scope
 * because the printers are several layers down, where threading a rank argument through every
 * call is error-prone. Off unless log_rank_tags(true): at tp 1, and in tools, there is one rank. */
void log_rank_tags(bool on);

class LogRank {
public:
    explicit LogRank(int rank);
    ~LogRank();
    LogRank(const LogRank&) = delete;
    LogRank& operator=(const LogRank&) = delete;
private:
    int prev_;
};

/* The engine a thread works for, in a process serving several models (docs/MULTIMODEL.md §3.3):
 * its log lines carry the model's name, [name] or [name/r0] with rank tags, and the device layer
 * files the thread's device allocations under the engine's owner id so they can be parked with it
 * (rad_dev_page_out). EngineScope sets both for a scope; every thread an engine starts opens one.
 * Owner 0 is no engine. Names show only once log_model_tags(true): a process of one model prints
 * as it always has. */
void log_model_tags(bool on);
int  engine_owner();
class EngineScope {
public:
    EngineScope(int owner, const char* model);
    ~EngineScope();
    EngineScope(const EngineScope&) = delete;
    EngineScope& operator=(const EngineScope&) = delete;
private:
    int         prev_owner_;
    const char* prev_model_;
};

#define RAD_LOG(lvl, ...)                                                        \
    do { if ((int)::rad::Log::lvl <= (int)::rad::log_level())                     \
             ::rad::log_emit(::rad::Log::lvl, __FILE__, __LINE__, __VA_ARGS__); } while (0)

#define RAD_ERR(...)   RAD_LOG(Error, __VA_ARGS__)
#define RAD_WARN(...)  RAD_LOG(Warn,  __VA_ARGS__)
#define RAD_INFO(...)  RAD_LOG(Info,  __VA_ARGS__)
#define RAD_DEBUG(...) RAD_LOG(Debug, __VA_ARGS__)
#define RAD_TRACE(...) RAD_LOG(Trace, __VA_ARGS__)

/* RAD_LOG with the level as a runtime value, for sites where the level depends on what was
 * found. */
#define RAD_LOG_AT(lvl, ...)                                                     \
    do { ::rad::Log _l = (lvl);                                                  \
         if ((int)_l <= (int)::rad::log_level())                                 \
             ::rad::log_emit(_l, __FILE__, __LINE__, __VA_ARGS__); } while (0)

/* Propagate a negative status, logging where it came from. */
#define RAD_TRY(expr)                                                            \
    do { int _s = (expr);                                                        \
         if (_s < 0) { RAD_DEBUG("%s -> %s", #expr, rad_strerror(_s)); return _s; } } while (0)

/* An unrecoverable condition: a device fault, a corrupt container, an invariant broken. Prints
 * and exits non-zero rather than serving from a wedged queue (spec §17). */
[[noreturn]] void fatal(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

/* ------------------------------------------------------------------ small helpers */
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));

inline int64_t align_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

/* Human-readable byte counts, for every report that prints a size. */
std::string humanb(int64_t bytes);

/* The one monotonic clock for all elapsed-time logic. The idle-session tiers stamp an entry on
 * use and compare against the stamp in another translation unit, so both sides must use the same
 * clock. Monotonic so an NTP step cannot age entries. */
uint64_t rad_mono_ns();

/* Split on spaces, tabs and commas: the encoding of RAD_C_IN's set. */
std::vector<std::string_view> split_ws(std::string_view s);

/* The strings joined with `sep` between them. */
inline std::string join(const std::vector<std::string>& v, const char* sep = ", ") {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += sep; s += v[i]; }
    return s;
}

/* ------------------------------------------------------------------ geometry */
/* A resolved parameter set: the geometry after ranges are collapsed to a band's upper bound.
 * Owns its strings, because a RadParam's char* may point into a plugin the core outlives. */
class Geometry {
public:
    Geometry() = default;

    /* Every write keeps the RadParam view current; it is never built lazily on read. A lazy
     * rebuild from a const reader is a write, and concurrent readers exist (the sizing declares
     * in Engine::probe_arena_levels read the same geometries in parallel). So params() writes
     * nothing, and an unwritten Geometry is safe to read from any number of threads.
     *
     * The view holds char* into this object's strings. A copy rebuilds its own view. A move
     * takes the view too, since the vectors hand over their arrays and the pointers stay valid;
     * the source is cleared, view included, leaving a valid empty geometry. */
    Geometry(const Geometry& o)
        : keys_(o.keys_), svals_(o.svals_), ivals_(o.ivals_), dvals_(o.dvals_), kinds_(o.kinds_) {
        rebuild();
    }

    Geometry& operator=(const Geometry& o) {
        if (this != &o) {
            keys_ = o.keys_; svals_ = o.svals_; ivals_ = o.ivals_; dvals_ = o.dvals_;
            kinds_ = o.kinds_;
            rebuild();
        }
        return *this;
    }

    Geometry(Geometry&& o) noexcept
        : keys_(std::move(o.keys_)), svals_(std::move(o.svals_)),
          view_(std::move(o.view_)), ivals_(std::move(o.ivals_)),
          dvals_(std::move(o.dvals_)), kinds_(std::move(o.kinds_)) {
        o.clear();
    }

    Geometry& operator=(Geometry&& o) noexcept {
        if (this != &o) {
            keys_ = std::move(o.keys_); svals_ = std::move(o.svals_);
            view_ = std::move(o.view_);
            ivals_ = std::move(o.ivals_); dvals_ = std::move(o.dvals_);
            kinds_ = std::move(o.kinds_);
            o.clear();
        }
        return *this;
    }

    void set_i(std::string_view k, long long v);
    void set_s(std::string_view k, std::string_view v);
    /* Floats (eps, rope theta, an attention scale) reach RadArgs.p but are never matched on; see
     * constraint_holds(). */
    void set_f(std::string_view k, double v);
    bool get_i(std::string_view k, long long* out) const;
    bool get_f(std::string_view k, double* out) const;
    const char* get_s(std::string_view k) const;
    bool has(std::string_view k) const;

    /* A view suitable for RadArgs.p. Valid while this object lives and is not written. */
    const RadParam* params() const { return view_.data(); }
    int             n_params() const { return (int)view_.size(); }

    std::string str() const;      /* "M=64 N=8192 K=5120 dtype=w4a8", for diagnostics */

private:
    std::vector<std::string> keys_, svals_;
    std::vector<RadParam>    view_;
    size_t slot(std::string_view k, int kind);   /* find or append k, set its kind */
    void rebuild();
    void clear() noexcept {
        keys_.clear(); svals_.clear(); view_.clear(); ivals_.clear(); dvals_.clear();
        kinds_.clear();
    }
    std::vector<long long>   ivals_;
    std::vector<double>      dvals_;
    std::vector<int>         kinds_;
};

/* Whether the geometry satisfies the constraint; the one place the matching rule lives. A
 * constraint on a float key never holds (get_i and get_s miss RAD_P_F64 slots): a tolerance is
 * not a predicate, and a kernel that discriminates on eps that precisely has a bug. */
bool constraint_holds(const RadConstraint& c, const Geometry& g);
/* Why not, for the selection table's miss column (spec §16). */
std::string constraint_why(const RadConstraint& c, const Geometry& g);

}  /* namespace rad */
