/* fleet/config.h -- the YAML file that configures a radiance process (docs/MULTIMODEL.md §4).
 *
 * One file names every model the process serves and the settings they share. Each former command
 * line flag is a key with the same name and no leading dashes, at the level it belongs to:
 *
 *   server:     the listener, authentication, logging and plugin search path (§4.3)
 *   cards:      the cards models may use
 *   memory:     per-card VRAM headroom and the host budgets
 *   scheduler:  load and drain timeouts, the keepalive interval
 *   defaults:   per-model keys merged under every model
 *   models:     one entry per model, keyed by its id
 *   groups:     together / exclusive co-residency rules
 *
 * Top-level keys that start with `x-` are ignored, so a file can keep anchors there and merge them
 * into models with `<<: *name`.
 *
 * A model's keys are turned back into the flag list config_parse reads, so a key and its flag are
 * validated by the same code and cannot drift. The loader adds only what the flag table cannot
 * express: levels, the defaults merge, the fleet's own keys, and provenance (which YAML path set
 * each option, for the startup report).
 */
#pragma once
#include "rad_core.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace rad::fleet {

/* A model's share of the card's flex pool above its fixed part (§6.3). -1 is "not set": the
 * floor defaults to one sequence at the model's context bound and the ceiling to everything the
 * card can lend. */
struct FlexSpec {
    int64_t kv_min_tokens   = -1;
    int64_t kv_max_tokens   = -1;
    double  kv_weight       = 1.0;
    int64_t experts_min_mib = -1;
    int64_t experts_max_mib = -1;

    bool operator==(const FlexSpec&) const = default;
};

/* What a parked model keeps (§7.2). */
struct ParkSpec {
    enum class Vram { None, Statics, Mib };
    enum class Ram  { Required, Allowed, Never };
    /* How it comes back from host memory: every byte onto the card before it serves, or serving
     * at once with the rest read in place over the link while it moves (dev_page_in_live). Live
     * suits a model whose steps read a small part of what is parked (a tiered MoE's experts) or
     * that computes long per byte read; a dense model decoding reads all of it every step, so its
     * reads share the link with the move and both run slower (docs/MULTIMODEL.md §7.4). Auto is
     * Full, except that a one-step model (an embedding program) whose cards lack the room for all
     * of it serves at once with its weights read in place until free memory takes them (§7.5). */
    enum class Restore { Auto, Full, Live };
    Vram    vram     = Vram::None;
    int64_t vram_mib = 0;          /* with Vram::Mib: the head of the first-use order kept */
    Ram     ram      = Ram::Allowed;
    Restore restore  = Restore::Auto;

    bool operator==(const ParkSpec&) const = default;
};

struct ModelEntry {
    std::string              name;       /* the model id requests name; never contains ':' */
    std::vector<std::string> aliases;
    int                      priority = 0;
    bool                     pin      = false;
    int64_t                  ttl_ms   = 0;     /* idle time before parking; 0 = never */
    FlexSpec                 flex;
    ParkSpec                 park;

    /* The flag list `cfg` was parsed from, without the program name, in a fixed order: server
     * keys, then the model's own and inherited keys sorted by name. Two entries with equal args
     * configure identical engines, which is what hot reload compares (§10). */
    std::vector<std::string> args;
    Config                   cfg;
};

struct Group {
    enum class Kind { Together, Exclusive };
    std::string              name;
    Kind                     kind = Kind::Together;
    std::vector<std::string> members;
};

struct FleetConfig {
    std::string path;               /* the file read, for messages */

    /* server.log: Info, Debug or Trace. Unset (`log_set` false), the level RADIANCE_LOG chose
     * stands. */
    Log  log = Log::Info;
    bool log_set = false;

    /* cards: empty means every card. */
    std::vector<int> cards;

    /* memory: the per-card keys travel inside every model's flag list; these are the fleet's own. */
    int64_t     host_mib = 0;       /* 0 = not set */
    std::string disk_tier;

    /* scheduler: */
    int64_t load_timeout_ms  = 120000;
    int64_t drain_timeout_ms = 30000;
    int64_t keepalive_ms     = 10000;
    /* A device that has advanced nothing for this long ends the process, so a restart policy can
     * bring the server back (dev_stuck_exit_ms); 0 never. */
    int64_t stuck_exit_ms    = 60000;
    /* Once the listener is open and nothing waits, unpinned models load one at a time into free
     * VRAM and park to host memory and the disk tier as later ones displace them, so a first
     * request finds its model parked rather than cold (docs/MULTIMODEL.md §8.6). */
    bool    warm_up          = true;

    std::vector<ModelEntry> models;
    std::vector<Group>      groups;

    /* The model a name or alias resolves to, or null. */
    const ModelEntry* find(const std::string& name_or_alias) const;
};

/* Read and validate `path`. RAD_OK, or RAD_E_INVAL after RAD_ERR lines that name the file, the
 * line and the key. Nothing is half-applied: on failure *out is left as it was. */
int load(const std::string& path, FleetConfig* out);

/* The same from text already in memory; `origin` names it in messages. */
int parse(const std::string& text, const std::string& origin, FleetConfig* out);

/* The YAML equivalent of a command line, for the message that refuses the removed flags: the
 * flags as one model named after the container, server and memory keys at their levels. A word
 * that is not an option is kept as a model key, so loading the result refuses it by name. */
std::string yaml_from_args(int argc, char** argv);

/* The option keys a level holds ("server" or "memory"), without the fleet's own keys. */
std::vector<std::string> level_keys(const std::string& level);

/* The key reference `radiance --help` prints: the levels, then every key with its argument and
 * help, grouped by level. */
void print_reference(FILE* to);

/* Durations: "250ms", "30s", "30m", "2h"; a bare number is seconds. False on anything else. */
bool parse_duration_ms(const std::string& s, int64_t* out_ms);

}  // namespace rad::fleet
