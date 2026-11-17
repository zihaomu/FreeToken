/* core/server/history.h -- an hour of the dashboard's figures, kept by the process.
 *
 * A page that keeps its own samples shows nothing of the time before it was opened. The process
 * samples once a second instead and keeps an hour: for each model the figures its Performance page
 * charts, derived from two consecutive /stats objects, and the requests that finished; for the
 * fleet every card's telemetry, every model's state and its bytes on each card. A page loads the
 * hour when it opens (`since` 0) and asks for what is new after its last sample from then on.
 *
 * The sampler is the only reader of the engine on the dashboard's behalf. Every answer here comes
 * from what it last read (`live` is the /stats or /admin/fleet object of the latest sample, as
 * text), so any number of open pages costs the serving path nothing beyond the one sample a second.
 *
 * A row is a set of named values at one time; a value a sample did not yield is a gap, not a zero.
 * Rates are counter differences over the time between two samples of one start of the model: the
 * first sample after a load yields only the instantaneous figures.
 */
#pragma once
#include "server/iface.h"

#include <deque>
#include <map>
#include <mutex>
#include <atomic>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rad {
namespace server {

class History {
public:
    static constexpr int kKeepS = 3600;      /* seconds of samples kept */
    /* Rows a ring holds: the hour at the five rows a second a watched page is given. */
    static constexpr int kRows = kKeepS * 5;
    static constexpr int kRequests = 500;    /* finished requests kept for a model */

    /* One sample of a model from its /stats object (`body` its text), `t` in unix seconds. */
    void model_sample(const std::string& id, const json& stats, double t, std::string body = std::string());
    /* One sample of the fleet from /admin/fleet's object. */
    void fleet_sample(const json& fleet, double t, std::string body = std::string());
    /* A page asked for a body within the last `seconds`. */
    bool watched(double seconds) const;
    /* A model that is not open right now: its history stays, its live object goes. */
    void model_closed(const std::string& id);
    /* A model the process no longer serves. */
    void forget(const std::string& id);

    /* {"keep_s", "t": [...], "c": {name: [...]}, "requests": [...]}: the samples later than
     * `since` and the requests that finished after it. A column holds a value or null for every
     * time in `t`; a column with no value in the range is left out. An unknown model answers with
     * empty arrays. */
    json model_since(const std::string& id, double since) const;
    /* The fleet's, without `requests`. */
    json fleet_since(double since) const;
    /* The same as response bodies, with `"live"`: the latest sample's object, null for a model
     * that is not open. */
    std::string model_body(const std::string& id, double since) const;
    std::string fleet_body(double since) const;

    using Row = std::vector<std::pair<std::string, double>>;
    /* The row two consecutive /stats objects of one model yield (`prev` null for the first, `dt`
     * the seconds between them). Exposed for the tests. */
    static Row model_row(const json& now, const json* prev, double dt);
    /* The row /admin/fleet's object yields. */
    static Row fleet_row(const json& fleet);

private:
    /* A ring of rows in columns. */
    struct Series {
        std::vector<double> t;
        std::map<std::string, std::vector<float>> c;
        size_t head = 0, n = 0;
        void push(double at, const Row& row);
        void write(json& out, double since) const;
    };
    struct Model {
        Series series;
        json   prev;                 /* the last /stats object */
        double prev_t = 0;
        /* Requests seen and not yet finished, by id, and the finished ones, oldest first. */
        std::map<std::string, json> live;
        std::set<std::string>       done;
        std::deque<json>            finished;
        std::string                 latest;     /* the latest /stats text; empty while not open */
    };
    void track_requests(Model& m, const json& stats, double t);

    mutable std::mutex           mu_;
    std::map<std::string, Model> models_;
    Series                       fleet_;
    std::string                  fleet_live_;
    /* steady_clock nanoseconds of the last body asked for; 0 before the first. */
    mutable std::atomic<int64_t> asked_ns_{ 0 };
};

}  // namespace server
}  // namespace rad
