/* tunecache.h -- the single tuning cache: rad-tune writes it, the selector reads it at declare.
 *
 * Constraint matching finds a kernel that is correct for a geometry; within a kernel the best tile
 * configuration depends on the shape and the machine (spec §15). One cache and one format for all
 * plugins, so the engine can always say how a shape was tuned.
 *
 * ================================================================= the format
 *
 *   $RADIANCE_HOME/tune/<machine>.tune
 *
 * One file per machine (a cache from another card is wrong, not just slow). UTF-8 text,
 * LF-terminated, so it can be read, diffed, edited and versioned. One row per (kernel, shape) the
 * declared graph asks for.
 *
 *   line 1        `radtune 2`                        magic and format version
 *   line 2        `machine <machine-key>`            must equal the reader's own key
 *   `#...`        comment, ignored
 *   otherwise     `<kernel> <plugin> <choice> <us> <measured> <geometry...>`
 *
 *     kernel     RadKernelInfo.name, the entry point. No spaces.
 *     plugin     RadPluginInfo.name that supplied it. No spaces.
 *     choice     the winning tuned axes, `key=value` comma separated, sorted by key (e.g.
 *                `mb=3,npw=4,sk=8`); `-` for a kernel with nothing to tune. A point in the
 *                RadTunable space, not a variant name, so libraries may add or widen axes. An axis
 *                no longer declared is ignored on lookup; a declared axis the row omits takes its
 *                default.
 *     us         median microseconds per launch, `%.6f`.
 *     measured   ISO-8601 UTC, e.g. `2026-09-04T09:11:02Z`; --debug-graph prints it, since a
 *                measurement may predate a driver or library change.
 *     geometry   the resolved parameters, `key=value` sorted by key, space separated, with ranges
 *                collapsed to the band's upper bound (a bucket table row is what was measured).
 *
 * The lookup key is (kernel, plugin, geometry), not the op: kernels serving one op tune
 * independently. Rows for missing kernels are kept on rewrite and ignored on lookup, so a plugin
 * downgrade keeps the measurements.
 *
 * A missing file is not an error: every RadTunable has its own `deflt` (spec §15), and the cache
 * only moves shapes off those defaults.
 */
#pragma once
#include "../rad_internal.h"

#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace rad {

struct TuneRow {
    std::string kernel, plugin, choice, geom;
    double      us = 0.0;
    std::string measured;     /* ISO-8601 UTC, exactly as written */
};

class TuneCache {
public:
    /* Missing file: RAD_OK, empty. Malformed line: skipped with a warning naming it; a bad cache
     * must not stop a server. */
    int load(const std::string& radiance_home, const std::string& machine);
    int save(const std::string& radiance_home) const;

    const std::string& machine() const { return machine_; }
    void set_machine(std::string m) { machine_ = std::move(m); }

    const TuneRow* find(std::string_view kernel, std::string_view plugin,
                        std::string_view geom) const;
    void put(const TuneRow& r);

    const std::vector<TuneRow>& rows() const { return rows_; }
    bool empty() const { return rows_.empty(); }

    /* `key=value` sorted by key, space separated: the canonical spelling of a shape. Every
     * component keying on shapes must use it. */
    static std::string canon_geometry(const Geometry& g);

    /* The `choice` column in both directions (`key=value` comma separated, sorted by key), shared
     * by tuner and selector so the cache always hits. */
    static std::string canon_choice(const std::vector<std::pair<std::string, int64_t>>& c);
    /* Unparseable pairs are dropped, not an error (forward compatibility); the caller filters by
     * its kernel's declared keys anyway. */
    static std::vector<std::pair<std::string, int64_t>> parse_choice(std::string_view s);

    static std::string path_for(const std::string& radiance_home, const std::string& machine);
    /* The same path relative to a home, for finding the first home on a search path that has one. */
    static std::string rel_path(const std::string& machine);
    static std::string now_iso8601();

private:
    static std::string index_key(std::string_view kernel, std::string_view plugin,
                                 std::string_view geom);

    std::string                                    machine_;
    std::vector<TuneRow>                           rows_;
    std::unordered_map<std::string, size_t>        index_;
};

/* The machine half of the key, from the device layer (machine_key.cpp), kept apart so the cache
 * stays pure file I/O. */
std::string rad_machine_key();
/* The key for `n` devices, the first described by `dev` (null when there is none to describe). */
std::string rad_machine_key_of(int n, const RadDeviceProps* dev);

}  /* namespace rad */
