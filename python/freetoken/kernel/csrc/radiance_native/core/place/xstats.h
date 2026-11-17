/* xstats.h -- --expert-stats FILE: which routed experts a served model selects, per layer and per
 * step kind, counted on the host from the routing histograms the core already copies back.
 *
 * ---- where the counts come from ---------------------------------------------------------------
 *
 * Every routed layer hands its per-expert counts to the core (rad_route_report), which copies the
 * pass's rows to a host bank once a pass (Ctx::route_flush) whether or not anything reads them. The
 * heat engine drains the landed banks once a step, one step behind (Engine::tiering_dispatch). The
 * tally here is credited on that same drain from the same rows, so it adds no device work, no copy
 * and no synchronisation; where the heat engine is off (every expert in VRAM, a static plan) the
 * drain runs for the tally alone. At shutdown the streams are drained and the banks still in the
 * ring are credited, so the file covers every step the engine ran.
 *
 * ---- what a histogram row is ------------------------------------------------------------------
 *
 * Index e of a rank's row is the rank's own expert e: the unit it declared with group expert e,
 * the contract the heat engine relies on too. Routing is replicated (every rank scores every
 * expert and picks the same top-k), but under expert parallelism a rank counts only the experts it
 * holds, numbered from zero (moe_scatter subtracts the rank's first expert). So a row is the
 * model's histogram on a rank that holds every expert (one width slice of each), and this rank's
 * slice of it otherwise. XsMap says, per rank and layer, which model-wide expert each index is:
 * from the container's directory where it numbers the unit's weights (exact for any split), else
 * by rank order (the contiguous runs every in-tree expert-parallel split uses). An expert several
 * ranks hold is counted on the lowest of them only, since each sees the same routing.
 *
 * ---- step kinds -------------------------------------------------------------------------------
 *
 * A bank holds one step. Trunk layers are filed under the step's kind: prefill (no decode rows),
 * decode (only decode rows) or mixed (both, which one histogram cannot split). Routed layers past
 * the trunk (an MTP or DSpark head, Ctx::bind_routing) are filed under draft. A head reports once a
 * draft pass into the step's bank, each pass overwriting the last, so its bank holds the last pass
 * of the step; `passes` counts every pass reported, `tokens` the rows of the pass the bank holds.
 * Router weights are not summed: they stay on the card (RadRouting::expert_w is device memory, and
 * the in-tree MoE block leaves it null), and copying them back would be new device work.
 */
#pragma once
#include "../rad_core.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct RadBatch;

namespace rad {

enum XsKind : int { kXsPrefill = 0, kXsDecode = 1, kXsMixed = 2, kXsDraft = 3 };
constexpr int kXsKinds = 4;
const char* xs_kind_name(int kind);

/* A trunk pass's kind from its batch shape (RadBatch::n_seq_decode). */
inline int xs_trunk_kind(int64_t n_seq, int64_t n_seq_decode) {
    return n_seq_decode <= 0 ? kXsPrefill : n_seq_decode >= n_seq ? kXsDecode : kXsMixed;
}

/* Which model-wide expert each index of each rank's histogram row is. */
struct XsMap {
    static constexpr int32_t kElsewhere = -1;   /* a lower rank holds and counts this expert */
    static constexpr int32_t kUnknown   = -2;   /* no model-wide id: counted as unmapped */

    int32_t n_rank  = 0;
    int32_t n_layer = 0;      /* routed layers, any head past the trunk included */
    int32_t n_trunk = 0;      /* layers [0, n_trunk) are the trunk's, the rest draft heads */
    std::vector<int32_t> n_expert;                      /* [layer]: the model's, 0 where none */
    std::vector<std::vector<std::vector<int32_t>>> gid; /* [rank][layer][local] */
    std::vector<std::string> how;                       /* [layer]: "container", "rank order",
                                                         * "replicated" or "none" */
};

/* Builds the map from what each rank declared.
 *
 *   ids[r][l][e]  the model-wide id the container gives rank r's expert e of layer l (the
 *                 directory entry of the unit's weight), or -1 where it gives none. Its size per
 *                 layer is the rank's expert count there; a hole (no unit at e) is -1 too.
 *   n_model[l]    the model's expert count where the caller knows it (the container's n_expert
 *                 for trunk layers), 0 where not.
 *
 * Per layer, when every index of every rank has an id, the ids are used, each counted on the
 * lowest rank that holds it. Otherwise, at one rank or when every rank holds n_model experts, the
 * indices are the model's and rank 0 counts them; else the ranks' rows are concatenated in rank
 * order. A layer's ids must be distinct within a rank and below 2^20, or the layer falls back. */
int xs_map_build(const std::vector<std::vector<std::vector<int32_t>>>& ids,
                 const std::vector<int32_t>& n_model, int32_t n_trunk, XsMap* out);

/* The merged tally a write renders. Counts are model-wide expert ids. */
struct XsSnapshot {
    std::string model, arch, container;
    int32_t     tp = 1;
    int32_t     n_layer = 0, n_trunk = 0, n_expert = 0;
    std::vector<int32_t>     layer_experts, top_k;
    std::vector<std::string> how;
    /* Issued, from rank 0's passes: trunk steps and their tokens by kind (the mixed kind's decode
     * tokens apart), draft passes and their rows. */
    uint64_t steps[kXsKinds] = {}, tokens[kXsKinds] = {}, mixed_decode_tokens = 0;
    uint64_t draft_passes = 0, draft_rows = 0;
    /* Observed, per kind and layer: histograms credited, passes they stand for, the tokens of the
     * passes they hold (rank-0 view of the layer's lowest counting rank), and the selections,
     * which are the counts' sum. */
    std::vector<uint64_t> histograms[kXsKinds], passes[kXsKinds], obs_tokens[kXsKinds],
                          selections[kXsKinds];
    std::vector<std::vector<uint64_t>> counts[kXsKinds];   /* [layer][expert] */
    /* Per rank: histograms credited, selections with no model-wide id, histograms whose step had
     * no record (filed as mixed), and the ring's lost histograms (Ctx::route_dropped). */
    std::vector<uint64_t> rank_credited, rank_unmapped, rank_unclassified, rank_dropped;
    double   uptime_s = 0.0;
    int64_t  written_unix = 0;
    bool     final_ = false;
};

/* One rank's tally. Written by its rank thread (issued, credit) and read by the writer
 * (snapshot_into) under the tally's lock, which the rank takes once a pass and once a drain;
 * the writer holds it for a copy every 30 s, so it is never contended in practice. */
class XsTally {
public:
    void init(const XsMap* map, int rank);

    /* At every pass the rank issues, before it runs: files the step's kind, or counts a draft
     * pass, for the drain to find a step or more later. Not for an encoder pass. */
    void issued(const RadBatch& b);
    void issued(int step, int draft_pass, int64_t n_seq, int64_t n_seq_decode, int64_t n_tok,
                int64_t n_tok_decode);

    /* Taken by the drain around its credits, and by snapshot_into. */
    std::mutex& mutex() { return mu_; }
    /* One landed histogram of `layer` from step `step`: `n` counts (the layer's reported expert
     * count), standing for `reports` passes. Caller holds mutex(). */
    void credit(int layer, int step, const int32_t* counts, int64_t n, int reports, int64_t top_k);

    /* The route ring's losses so far (Ctx::route_dropped), noted by the drain. Caller holds
     * mutex(). */
    void note_dropped(uint64_t n) { dropped_ = n; }

    /* Adds this rank's tally into `s`, sized by the caller from the map. The per-layer token and
     * histogram counts come from the layer's lowest counting rank (`token_rank`), since every rank
     * counting in a layer sees the same steps, and the issued totals from the lowest counting rank
     * of all (`issued`). Takes the lock. */
    void snapshot_into(XsSnapshot* s, const std::vector<int32_t>& token_rank, bool issued);

    uint64_t credited() const { return credited_; }

private:
    struct Rec {
        int32_t  step = -1;
        uint8_t  kind = 0;
        bool     trunk = false;     /* the step's trunk pass was filed */
        int64_t  n_tok = 0;
        int64_t  draft_rows = 0;    /* the step's last draft pass's rows */
    };
    static constexpr int kRing = 256;   /* > the route ring's 32 banks, with room for idle layers */

    const XsMap* map_ = nullptr;
    int          rank_ = 0;
    std::mutex   mu_;
    std::vector<Rec> ring_;
    std::vector<size_t> off_;           /* [layer]: offset of the layer's run in a counts plane */
    std::vector<uint64_t> counts_[kXsKinds];
    std::vector<uint64_t> hist_[kXsKinds], passes_[kXsKinds], tok_[kXsKinds], sel_[kXsKinds];
    std::vector<int32_t>  top_k_;
    uint64_t steps_[kXsKinds] = {}, tokens_[kXsKinds] = {}, mixed_decode_ = 0;
    uint64_t draft_passes_ = 0, draft_rows_ = 0;
    uint64_t credited_ = 0, unmapped_ = 0, unclassified_ = 0, dropped_ = 0;
};

/* Renders a snapshot as the file's JSON. */
std::string xs_json(const XsSnapshot& s);
/* Writes `text` to `path` through `path`.tmp and a rename, so a reader never sees half a file. */
int xs_write_atomic(const std::string& path, const std::string& text);

/* The engine's side: the map, one tally a rank, and a thread that writes the file every
 * `interval_s` seconds until stop(). */
class ExpertStats {
public:
    ~ExpertStats();
    /* Writes the file once, empty, so a path that cannot be written fails at boot. */
    int  init(const std::string& path, const XsMap& map, XsSnapshot info, double interval_s = 30.0);
    /* Null for a rank that counts nothing (every expert it holds a lower rank counts): its rank
     * thread then neither files steps nor drains for the tally. */
    XsTally* tally(int rank) {
        return rank >= 0 && (size_t)rank < tallies_.size() && counts_on_[(size_t)rank]
                   ? tallies_[(size_t)rank].get() : nullptr;
    }
    const XsMap& map() const { return map_; }
    const std::string& path() const { return path_; }

    void start();
    void stop();
    /* Merges every rank's tally and writes the file. */
    int        write(bool final_);
    XsSnapshot snapshot();

private:
    std::string path_;
    XsMap       map_;
    XsSnapshot  info_;
    double      interval_s_ = 30.0;
    std::vector<std::unique_ptr<XsTally>> tallies_;
    std::vector<uint8_t> counts_on_;      /* [rank]: holds an expert it counts */
    std::vector<int32_t> token_rank_;     /* [layer]: the lowest rank counting in it */
    int32_t     issued_rank_ = -1;        /* the lowest rank counting at all */
    uint64_t    t0_ns_ = 0;

    std::thread             writer_;
    std::mutex              wmu_;
    std::condition_variable wcv_;
    bool                    wstop_ = false;
    std::mutex              file_mu_;     /* one write at a time */
};

/* --expert-profile FILE: the planner's ranking, from a radiance-expert-stats JSON's `decode`
 * counts. One entry per (layer, expert) with a positive count, its share that count over the
 * layer's decode selections, so a layer's hottest expert ranks above another layer's merely busy
 * one. `n_trunk` is the file's trunk layer count, or -1 where it names none. Returns RAD_E_IO on a
 * missing or unreadable file and RAD_E_INVAL on one that is not this format. */
int xs_profile_load(const std::string& path, std::vector<RadFileProfile>* out, int32_t* n_trunk);

}  /* namespace rad */
