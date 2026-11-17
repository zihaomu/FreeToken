/* front.h -- the process's one listener in front of every model's server (docs/MULTIMODEL.md §5).
 *
 * Each engine builds its own Server, as a process serving one model always has: its routes, its
 * limits, its tokenizer and template, its metrics. The Front owns the socket. It registers the
 * routes every model server registers, reads which model a request names, and hands the request
 * to that model's router. The model servers never listen.
 *
 * A request picks its model by, in order: a `model` query parameter, a `model` form field, the
 * body's top-level `model` string. `base:adapter` names an adapter of the model `base`; the server
 * then sees the adapter's own name, as a one-model process would. A request naming no model goes
 * to the model named or aliased `default`, or to the only model when there is one. A request that
 * names a job by id (a video, a shape) and no model asks every loaded model, and the one that knows
 * the id answers.
 *
 * Every model the configuration names is known to the Front from the start, loaded or not (§8.1):
 *
 *   Cold     not loaded. A request for it asks the fleet to load it (the demand callback) and
 *            waits.
 *   Loading  the fleet is loading it; requests wait.
 *   Open     its server takes requests.
 *   Closing  it is leaving (an eviction, an unload, a reload); new requests wait for it to come
 *            back, the responses already handed out finish.
 *   Failed   its last load failed; requests are refused with the reason until the retry time,
 *            and the first request after that asks for a load again.
 *   Parked   loaded but out of VRAM (§7): its memory waits in host RAM. A request asks the fleet
 *            to bring it back, as for a Cold model, and waits.
 *
 * A request naming a finished job by id is answered here, whatever state its model is in
 * (keep_job).
 *
 * A waiting request waits at most its model's load timeout, then gets 503 with Retry-After. A
 * streaming request (`"stream": true`) is answered at once with an event stream that carries an
 * SSE comment every keepalive interval while it waits, so proxies and clients keep the connection;
 * the model's own stream follows on the same connection (§8.2).
 *
 * The Front answers itself for the routes that describe the whole process: /v1/models (every model
 * and alias, with its state), /health, /metrics (every loaded model's series, each family once,
 * plus the fleet's).
 *
 * Lifetime: a model attaches once its server is ready and detaches before the server goes away.
 * detach() waits for every response handed out for that model to be finished with (the response
 * object, and so its stream, holds a lease), so a server is never destroyed under a stream.
 */
#pragma once
#include "server/http.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace server {

class Server;

class Front {
public:
    enum class State { Cold, Loading, Open, Closing, Failed, Parked };
    static const char* state_name(State s);

    /* Upper bounds, in seconds, of the buckets of a request's wait for its model to open
     * (Activity::wait_le; the last bucket, +Inf, is `waits`). */
    static constexpr std::array<double, 9> kWaitLe = { 0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60 };

    /* What the fleet reads to schedule (docs/MULTIMODEL.md §8.3) and reports. */
    struct Activity {
        State    state = State::Cold;
        int64_t  leases = 0;        /* responses handed out and not yet finished */
        int64_t  waiting = 0;       /* requests waiting for it to open */
        uint64_t last_used_ns = 0;  /* rad_mono_ns of the last lease taken or returned */
        uint64_t uses = 0;          /* leases taken since it was added */
        /* How long the leased requests waited for the model to open (a load or a restore; 0 for
         * one that found it open): how many, their sum, and the cumulative count at or under each
         * bound. */
        uint64_t waits = 0;
        double   wait_s = 0;
        std::array<uint64_t, kWaitLe.size()> wait_le{};
    };

    /* `keepalive_ms` is the SSE comment interval for streaming requests that wait; 0 sends none. */
    Front(TransportOptions t, int64_t keepalive_ms);
    ~Front();
    Front(const Front&) = delete;
    Front& operator=(const Front&) = delete;

    /* A model the process serves, Cold. A request for it waits at most `wait_ms` for it to open.
     * `required` models must be open for /health to answer 200 (pinned and preloaded ones). */
    void add(const std::string& name, const std::vector<std::string>& aliases, int64_t wait_ms,
             bool required);
    /* Forgets a model (hot reload removed it): requests for it are 404 from now on. Waits for its
     * leases like detach(). */
    void remove(const std::string& name);
    /* The model's aliases, wait and health requirement change (hot reload). */
    void update(const std::string& name, const std::vector<std::string>& aliases, int64_t wait_ms,
                bool required);
    /* Called when a request needs a Cold or Parked model, or a Failed one past its retry time, and
     * again on every wake-up while the request waits for it. It runs under the Front's lock, so it
     * must not block or call back into the Front; the fleet queues a load. */
    void set_demand(std::function<void(const std::string&)> f) { demand_ = std::move(f); }
    /* Extra JSON members for a model's /v1/models card (state detail, cards), and extra metrics
     * text; both from the fleet. */
    void set_describe(std::function<std::string(const std::string&)> f) { describe_ = std::move(f); }
    void set_metrics(std::function<std::string()> f) { fleet_metrics_ = std::move(f); }

    /* What the model does (Server::capabilities_json), from when it last opened: it stays known
     * while the model is parked, when its server cannot be asked. Empty before its first load. */
    std::string capabilities(const std::string& name) const;

    /* The fleet is loading it. */
    void loading(const std::string& name);
    /* A load nobody asked for gave up (the fleet's warm-up): a loading model is cold again, and
     * any request that came meanwhile asks for it anew. */
    void abandon(const std::string& name);
    /* A Cold model is parked (its memory waits in host RAM), or no longer is. */
    void set_parked(const std::string& name, bool parked);
    /* The load failed: requests waiting get `why` as a 503, and so do new ones until `retry_ms`
     * from now; after that a request asks for a load again. */
    void fail(const std::string& name, const std::string& why, int64_t retry_ms);
    /* `s` serves the model `name` from now on. */
    void attach(const std::string& name, const std::vector<std::string>& aliases, Server* s);
    /* No new request reaches `name` (they wait for it to come back); returns once every response
     * handed out for it is gone, or after `wait_ms`, whichever is first. False when responses were
     * still held at the deadline, and the model stays Closing; a later call waits again. On true
     * the model is Cold. */
    bool detach(const std::string& name, int wait_ms);

    Activity activity(const std::string& name) const;
    /* Every model's name, in the order added. */
    std::vector<std::string> names() const;

    /* A finished job of a model (a video, a shape), named by its id in the path (/v1/videos/{id},
     * /status/{id}). The model's memory may go at any moment after it finishes, and with it the
     * server that made the job, so from here on `answer` serves every request naming that id, and
     * a client polling for its result is no reason to bring the model back. Kept until `expires_at`
     * (unix seconds) and among the newest `max_kept`; a DELETE is answered and forgets it. */
    void keep_job(const std::string& id, int64_t expires_at, int64_t max_kept, Handler answer);
    /* What answers the kept job a request's path names; empty when there is none. */
    Handler kept_job(const HttpRequest& q);

    /* A route the Front answers itself (the fleet's /admin routes). Before run(). */
    void route(const std::string& method, const std::string& path, Handler h);

    /* A dashboard read of an open model made in-process (the fleet's history reads /stats this
     * way): the body of `path` as the model's server answers it, or empty when the model is not
     * open or does not answer 200. Like a dashboard read from the socket it is not a use: it
     * neither brings a model back nor keeps one from going idle. */
    std::string peek(const std::string& model, const std::string& path);

    /* Blocks serving until stop(). */
    int  run();
    void stop();
    bool stopping() const { return stopping_.load(std::memory_order_acquire); }

    /* The model a request names, as read before any parsing (see the file header), or "". */
    static std::string requested_model(const HttpRequest& q);
    /* A read the dashboard polls. It reports on a model and is no reason to bring one back, so it
     * is refused while that model is not open, where a request for work would wait for it. */
    static bool dashboard_read(const HttpRequest& q);

private:
    struct Entry {
        std::string              name;
        std::vector<std::string> aliases;
        Server*                  server = nullptr;
        State                    state = State::Cold;
        std::string              why;               /* State::Failed: the reason */
        std::string              caps;              /* what it does, from when it last opened */
        uint64_t                 retry_ns = 0;      /* State::Failed: when a request may ask again */
        int64_t                  wait_ms = 0;
        bool                     required = false;
        bool                     removed = false;
        int64_t                  leases = 0;
        int64_t                  waiting = 0;
        uint64_t                 last_used_ns = 0;
        uint64_t                 uses = 0;
        uint64_t                 waits = 0;
        double                   wait_s = 0;
        std::array<uint64_t, kWaitLe.size()> wait_le{};
    };
    struct KeptJob {
        std::string id;
        int64_t     expires_at = 0;
        Handler     answer;
    };
    enum class Got { Leased, Refused, Waiting };

    /* The entry `model` resolves to (a name, an alias, or `base` of base:adapter), under mu_. */
    Entry* resolve(const std::string& model);
    /* A lease on the entry `model` resolves to, waiting for it to open for at most `wait_ms`
     * (-1: until the whole wait's end). *until_ns (rad_mono_ns) is the end of the whole wait; a
     * zero there, or a null `until_ns`, starts the wait now with the entry's load timeout, and the
     * end is written back so later calls continue the same wait. Refused with *refusal set when
     * there is no such model, it failed, or the whole wait passed; Waiting when only `wait_ms`
     * passed. */
    Got lease(const std::string& model, const std::string& path, int64_t wait_ms,
              uint64_t* until_ns, Entry** out, std::shared_ptr<void>* held, HttpResponse* refusal);
    std::shared_ptr<void> lease_entry(Entry* e);
    std::shared_ptr<void> lease_quiet(Entry* e);
    HttpResponse forward(HttpRequest q);
    HttpResponse forward_waiting(HttpRequest q, const std::string& model);
    void         forward_socket(const HttpRequest& q, WsConn& c);
    HttpResponse models(const HttpRequest& q);
    HttpResponse health(const HttpRequest& q);
    HttpResponse metrics(const HttpRequest& q);
    void         install_routes(const Router& model_router);
    /* Leases on every Open entry, for the fleet-wide routes. */
    std::vector<std::pair<Entry*, std::shared_ptr<void>>> hold_open();

    TransportOptions      topts_;
    int64_t               keepalive_ms_ = 0;
    Router                router_;
    HttpTransport         transport_;
    std::atomic<bool>     stopping_{false};
    std::function<void(const std::string&)>        demand_;
    std::function<std::string(const std::string&)> describe_;
    std::function<std::string()>                   fleet_metrics_;

    mutable std::mutex      mu_;
    /* Finished jobs, oldest first; their own lock, since one is answered without a model. */
    std::mutex              jobs_mu_;
    /* A model's last /sessions answer and when it was read. Listing the
     * conversations takes the idle tiers' lock in slices, so however many pages show them, the
     * engine is asked at most once in kSessionsKeepNs. Times are rad_mono_ns. */
    static constexpr uint64_t kSessionsKeepNs = 5ull * 1000 * 1000 * 1000;
    std::mutex sessions_mu_;
    std::map<std::string, std::pair<uint64_t, std::string>> sessions_read_;
    std::vector<KeptJob>    jobs_;
    std::condition_variable changed_;
    std::vector<std::unique_ptr<Entry>> entries_;
};

/* The top-level string member `key` of the JSON object `body`: true with its raw (still escaped)
 * text in [*begin, *end) of `body`, false when absent or not a string. Reads only as far as the
 * key, skipping nested values, so a body of hundreds of MB costs no parse. */
bool json_top_string(const std::string& body, const char* key, size_t* begin, size_t* end);
/* The same for any value: its raw text, e.g. `true` or `{...}`. */
bool json_top_value(const std::string& body, const char* key, size_t* begin, size_t* end);

/* Prometheus text from several sources as one exposition: each family's HELP and TYPE once, its
 * samples together. */
std::string merge_metrics(const std::vector<std::string>& texts);

}  /* namespace server */
}  /* namespace rad */
