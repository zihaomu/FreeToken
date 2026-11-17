/* front.cpp -- the listener in front of every model's server (front.h, docs/MULTIMODEL.md §5). */
#include "server/front.h"
#include "server/server.h"
#include "rad_internal.h"

#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstring>
#include <map>

namespace rad {
namespace server {

namespace {

/* Requests that name a job by its id in the path. Without a model they are asked of every model,
 * and the one that knows the id answers; the others say 404. */
bool id_route(const std::string& path) {
    return path.rfind("/v1/videos/", 0) == 0 || path.rfind("/status/", 0) == 0;
}

/* The id in such a path: the segment after the route's prefix. */
std::string route_id(const std::string& path) {
    const size_t at = path.rfind("/status/", 0) == 0 ? 8 : 11;
    const size_t end = path.find('/', at);
    return path.substr(at, end == std::string::npos ? std::string::npos : end - at);
}

HttpResponse refuse(const std::string& path, int status, const char* type, const std::string& msg) {
    return HttpResponse::json_body(status, error_body(path, status, type, msg));
}

/* `r` holds `lease` until the transport has finished with it (on_close), so the model's server
 * outlives the response and its stream. */
void hold_until_close(HttpResponse& r, std::shared_ptr<void> lease) {
    std::function<void(bool)> prev = std::move(r.on_close);
    r.on_close = [prev, lease](bool ok) { if (prev) prev(ok); };
}

}  // namespace

/* ------------------------------------------------------------------ reading the model */

/* The member `key` of the top-level object: its raw value in [*begin, *end). `string_only` refuses
 * a value that is not a string and returns a string's text without its quotes. */
static bool json_top(const std::string& s, const char* key, bool string_only, size_t* begin,
                     size_t* end) {
    const size_t n = s.size();
    size_t i = 0;
    auto ws = [&] { while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i; };
    /* A string starting at s[i] == '"': i ends past the closing quote. False if unterminated. */
    auto skip_string = [&]() -> bool {
        for (++i; i < n; ++i) {
            if (s[i] == '\\') { ++i; continue; }
            if (s[i] == '"') { ++i; return true; }
        }
        return false;
    };
    /* Any value at s[i]: i ends past it. */
    auto skip_value = [&]() -> bool {
        if (i >= n) return false;
        if (s[i] == '"') return skip_string();
        if (s[i] == '{' || s[i] == '[') {
            int depth = 0;
            while (i < n) {
                const char c = s[i];
                if (c == '"') { if (!skip_string()) return false; continue; }
                if (c == '{' || c == '[') ++depth;
                else if (c == '}' || c == ']') { if (--depth == 0) { ++i; return true; } }
                ++i;
            }
            return false;
        }
        while (i < n && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
        return true;
    };
    const size_t klen = std::strlen(key);
    ws();
    if (i >= n || s[i] != '{') return false;
    ++i;
    for (;;) {
        ws();
        if (i >= n || s[i] != '"') return false;
        const size_t kb = i + 1;
        if (!skip_string()) return false;
        const size_t ke = i - 1;
        ws();
        if (i >= n || s[i] != ':') return false;
        ++i;
        ws();
        if (ke - kb == klen && s.compare(kb, klen, key) == 0) {
            if (string_only) {
                if (i >= n || s[i] != '"') return false;
                const size_t vb = i + 1;
                if (!skip_string()) return false;
                *begin = vb;
                *end = i - 1;
                return true;
            }
            const size_t vb = i;
            if (!skip_value()) return false;
            size_t ve = i;
            while (ve > vb && (s[ve - 1] == ' ' || s[ve - 1] == '\t' || s[ve - 1] == '\n' ||
                               s[ve - 1] == '\r'))
                --ve;
            *begin = vb;
            *end = ve;
            return true;
        }
        if (!skip_value()) return false;
        ws();
        if (i < n && s[i] == ',') { ++i; continue; }
        return false;
    }
}

bool json_top_string(const std::string& s, const char* key, size_t* begin, size_t* end) {
    return json_top(s, key, true, begin, end);
}

bool json_top_value(const std::string& s, const char* key, size_t* begin, size_t* end) {
    return json_top(s, key, false, begin, end);
}

std::string Front::requested_model(const HttpRequest& q) {
    auto it = q.query.find("model");
    if (it != q.query.end()) return it->second;
    if (const FormPart* p = q.part("model")) return p->data;
    size_t b = 0, e = 0;
    if (!q.body.empty() && json_top_string(q.body, "model", &b, &e)) {
        const std::string raw = q.body.substr(b, e - b);
        if (raw.find('\\') == std::string::npos) return raw;
        /* An escaped name: decode it as JSON does. */
        const json j = json::parse("\"" + raw + "\"", nullptr, false);
        return j.is_string() ? j.get<std::string>() : std::string();
    }
    return std::string();
}

/* ------------------------------------------------------------------ metrics */

std::string merge_metrics(const std::vector<std::string>& texts) {
    struct Family { std::string help, type; std::vector<std::string> samples; };
    std::vector<std::string> order;
    std::map<std::string, Family> fam;
    auto family = [&](const std::string& name) -> Family& {
        auto it = fam.find(name);
        if (it == fam.end()) { order.push_back(name); it = fam.emplace(name, Family{}).first; }
        return it->second;
    };
    for (const std::string& t : texts) {
        std::string current;
        size_t at = 0;
        while (at < t.size()) {
            size_t nl = t.find('\n', at);
            if (nl == std::string::npos) nl = t.size();
            const std::string line = t.substr(at, nl - at);
            at = nl + 1;
            if (line.empty()) continue;
            if (line.rfind("# HELP ", 0) == 0 || line.rfind("# TYPE ", 0) == 0) {
                const size_t ns = 7, ne = line.find(' ', ns);
                current = line.substr(ns, ne == std::string::npos ? std::string::npos : ne - ns);
                Family& f = family(current);
                std::string& slot = line[2] == 'H' ? f.help : f.type;
                if (slot.empty()) slot = line;
                continue;
            }
            if (line[0] == '#') continue;
            const size_t ne = line.find_first_of("{ ");
            const std::string name = line.substr(0, ne);
            /* _bucket, _sum, _count belong to the histogram declared above them. */
            const std::string owner =
                !current.empty() && name.compare(0, current.size(), current) == 0 ? current : name;
            family(owner).samples.push_back(line);
        }
    }
    std::string out;
    for (const std::string& name : order) {
        const Family& f = fam[name];
        if (!f.help.empty()) out += f.help + "\n";
        if (!f.type.empty()) out += f.type + "\n";
        for (const std::string& s : f.samples) out += s + "\n";
    }
    return out;
}

/* ------------------------------------------------------------------ the Front */

const char* Front::state_name(State s) {
    switch (s) {
    case State::Cold:    return "cold";
    case State::Loading: return "loading";
    case State::Open:    return "resident";
    case State::Closing: return "closing";
    case State::Failed:  return "failed";
    case State::Parked:  return "parked";
    }
    return "?";
}

Front::Front(TransportOptions t, int64_t keepalive_ms)
    : topts_(std::move(t)), keepalive_ms_(keepalive_ms) {
    /* Every model server registers the same routes whatever it serves (Server::register_routes),
     * so a server built on nothing gives the table, and the Front can listen before any model is
     * loaded. Its handlers are never called. */
    Server proto(Deps{}, ServerOptions{});
    install_routes(proto.router());
}

Front::~Front() { stop(); }

void Front::install_routes(const Router& model_router) {
    /* The GET routes that describe the whole process: answered here, never forwarded. */
    using Own = HttpResponse (Front::*)(const HttpRequest&);
    static const std::pair<const char*, Own> own[] = {
        { "/v1/models", &Front::models }, { "/models", &Front::models },
        { "/v1/models/*id", &Front::models }, { "/health", &Front::health },
        { "/v1/health", &Front::health }, { "/metrics", &Front::metrics },
    };
    for (const std::string& sig : model_router.routes()) {
        const size_t sp = sig.find(' ');
        const std::string method = sig.substr(0, sp), path = sig.substr(sp + 1);
        bool fleet = method == "GET" && (path == "/" || path == "/dashboard");
        for (const auto& [p, fn] : own) fleet |= method == "GET" && path == p;
        if (fleet) continue;
        router_.add(method, path, [this](const HttpRequest& q) { return forward(q); });
    }
    for (const auto& [p, fn] : own)
        router_.get(p, [this, fn = fn](const HttpRequest& q) { return (this->*fn)(q); });
    /* The page itself, from here: it names no model, and asking one for it would wake that one. */
    router_.get("/", [](const HttpRequest&) { return dashboard_page(); });
    router_.get("/dashboard", [](const HttpRequest&) { return dashboard_page(); });
    for (const std::string& path : model_router.socket_routes())
        router_.socket(path, [this](const HttpRequest& q, WsConn& c) { forward_socket(q, c); });
}

void Front::route(const std::string& method, const std::string& path, Handler h) {
    router_.add(method, path, std::move(h));
}

void Front::add(const std::string& name, const std::vector<std::string>& aliases, int64_t wait_ms,
                bool required) {
    std::lock_guard<std::mutex> lk(mu_);
    auto e = std::make_unique<Entry>();
    e->name = name;
    e->aliases = aliases;
    e->wait_ms = wait_ms;
    e->required = required;
    entries_.push_back(std::move(e));
}

void Front::update(const std::string& name, const std::vector<std::string>& aliases,
                   int64_t wait_ms, bool required) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& p : entries_)
        if (p->name == name && !p->removed) {
            p->aliases = aliases;
            p->wait_ms = wait_ms;
            p->required = required;
        }
}

void Front::remove(const std::string& name) {
    std::unique_lock<std::mutex> lk(mu_);
    for (size_t i = 0; i < entries_.size(); ++i) {
        Entry* e = entries_[i].get();
        if (e->name != name || e->removed) continue;
        e->removed = true;
        changed_.notify_all();
        changed_.wait(lk, [&] { return e->leases == 0 && e->waiting == 0; });
        for (size_t j = 0; j < entries_.size(); ++j)
            if (entries_[j].get() == e) { entries_.erase(entries_.begin() + (long)j); break; }
        return;
    }
}

void Front::loading(const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& p : entries_)
        if (p->name == name && !p->removed) p->state = State::Loading;
    changed_.notify_all();
}

void Front::abandon(const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& p : entries_)
        if (p->name == name && !p->removed && p->state == State::Loading) p->state = State::Cold;
    changed_.notify_all();
}

void Front::set_parked(const std::string& name, bool parked) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& p : entries_)
        if (p->name == name && !p->removed) {
            if (parked && p->state == State::Cold) p->state = State::Parked;
            else if (!parked && p->state == State::Parked) p->state = State::Cold;
        }
    changed_.notify_all();
}

void Front::fail(const std::string& name, const std::string& why, int64_t retry_ms) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& p : entries_)
        if (p->name == name && !p->removed) {
            p->state = State::Failed;
            p->why = why;
            p->retry_ns = rad_mono_ns() + (uint64_t)std::max<int64_t>(retry_ms, 0) * 1000000ull;
            p->server = nullptr;
        }
    changed_.notify_all();
}

void Front::attach(const std::string& name, const std::vector<std::string>& aliases, Server* s) {
    std::lock_guard<std::mutex> lk(mu_);
    Entry* e = nullptr;
    for (auto& p : entries_)
        if (p->name == name && !p->removed) e = p.get();
    if (!e) {
        entries_.push_back(std::make_unique<Entry>());
        e = entries_.back().get();
        e->name = name;
    }
    e->aliases = aliases;
    e->server = s;
    if (s) e->caps = s->capabilities_json();
    e->state = State::Open;
    e->why.clear();
    e->last_used_ns = rad_mono_ns();
    changed_.notify_all();
}

bool Front::detach(const std::string& name, int wait_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    Entry* e = nullptr;
    for (auto& p : entries_)
        if (p->name == name && (p->state == State::Open || p->state == State::Closing)) e = p.get();
    if (!e) return true;
    e->state = State::Closing;
    changed_.notify_all();
    const bool idle = changed_.wait_for(lk, std::chrono::milliseconds(wait_ms),
                                        [&] { return e->leases == 0; });
    if (idle) {
        e->state = State::Cold;
        e->server = nullptr;
        changed_.notify_all();
    }
    return idle;
}

std::string Front::capabilities(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& p : entries_)
        if (p->name == name && !p->removed) return p->caps;
    return std::string();
}

Front::Activity Front::activity(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    Activity a;
    for (auto& p : entries_)
        if (p->name == name && !p->removed) {
            a.state = p->state;
            a.leases = p->leases;
            a.waiting = p->waiting;
            a.last_used_ns = p->last_used_ns;
            a.uses = p->uses;
            a.waits = p->waits;
            a.wait_s = p->wait_s;
            a.wait_le = p->wait_le;
        }
    return a;
}

std::vector<std::string> Front::names() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> v;
    for (auto& p : entries_)
        if (!p->removed) v.push_back(p->name);
    return v;
}

std::shared_ptr<void> Front::lease_entry(Entry* e) {
    ++e->leases;
    ++e->uses;
    e->last_used_ns = rad_mono_ns();
    return std::shared_ptr<void>(static_cast<void*>(e), [this](void* p) {
        std::lock_guard<std::mutex> lk(mu_);
        Entry* x = static_cast<Entry*>(p);
        --x->leases;
        x->last_used_ns = rad_mono_ns();
        changed_.notify_all();
    });
}

/* A hold on an open model's server that is not a use of the model: detach() waits for it, and the
 * idle clock and the use count do not see it. For the dashboard's reads. Under mu_. */
std::shared_ptr<void> Front::lease_quiet(Entry* e) {
    ++e->leases;
    return std::shared_ptr<void>(static_cast<void*>(e), [this](void* p) {
        std::lock_guard<std::mutex> lk(mu_);
        --static_cast<Entry*>(p)->leases;
        changed_.notify_all();
    });
}

std::string Front::peek(const std::string& model, const std::string& path) {
    Entry* e = nullptr;
    std::shared_ptr<void> hold;
    {
        std::lock_guard<std::mutex> lk(mu_);
        e = resolve(model);
        if (!e || e->removed || e->state != State::Open || !e->server) return std::string();
        hold = lease_quiet(e);
    }
    HttpRequest q;
    q.path = path;
    HttpResponse r = e->server->router().dispatch(std::move(q));
    return r.status == 200 && !r.streaming() ? std::move(r.body) : std::string();
}

Front::Entry* Front::resolve(const std::string& model) {
    auto names = [](const Entry& e, const std::string& m) {
        if (e.name == m) return true;
        for (const std::string& a : e.aliases)
            if (a == m) return true;
        return false;
    };
    const std::string want = model.empty() ? std::string("default") : model;
    for (auto& p : entries_)
        if (!p->removed && names(*p, want)) return p.get();
    if (!model.empty()) {
        const size_t colon = model.find(':');
        if (colon != std::string::npos)
            for (auto& p : entries_)
                if (!p->removed && names(*p, model.substr(0, colon))) return p.get();
        return nullptr;
    }
    Entry* only = nullptr;
    int n = 0;
    for (auto& p : entries_)
        if (!p->removed) { only = p.get(); ++n; }
    return n == 1 ? only : nullptr;
}

Front::Got Front::lease(const std::string& model, const std::string& path, int64_t wait_ms,
                        uint64_t* until_ns, Entry** out, std::shared_ptr<void>* held,
                        HttpResponse* refusal) {
    std::unique_lock<std::mutex> lk(mu_);
    *out = nullptr;
    Entry* e = resolve(model);
    if (!e) {
        std::string serving;
        for (auto& p : entries_)
            if (!p->removed) serving += (serving.empty() ? "" : ", ") + p->name;
        *refusal = model.empty()
            ? refuse(path, 400, "invalid_request_error",
                     "this server serves several models (" + serving + "); name one with `model`")
            : refuse(path, 404, "not_found_error",
                     "no model '" + model + "' is served here (serving: " + serving + ")");
        return Got::Refused;
    }
    /* A model not open: ask for it and wait (docs/MULTIMODEL.md §8.2). The entry's own timeout
     * bounds the whole wait; `wait_ms` only a slice of it, for the keepalive. */
    const uint64_t t0 = rad_mono_ns();
    uint64_t own = 0;
    if (!until_ns) until_ns = &own;
    if (*until_ns == 0) *until_ns = t0 + (uint64_t)std::max<int64_t>(e->wait_ms, 0) * 1000000ull;
    const auto now = std::chrono::steady_clock::now();
    const auto limit = now + std::chrono::nanoseconds(*until_ns > t0 ? *until_ns - t0 : 0);
    const auto slice = wait_ms < 0 ? limit : std::min(limit, now + std::chrono::milliseconds(wait_ms));
    ++e->waiting;
    for (;;) {
        if (e->removed) break;
        if (e->state == State::Open) break;
        if (e->state == State::Failed) {
            if (rad_mono_ns() < e->retry_ns) break;
            e->state = State::Cold;
        }
        if ((e->state == State::Cold || e->state == State::Parked) && demand_) demand_(e->name);
        if (stopping()) break;
        if (changed_.wait_until(lk, slice) == std::cv_status::timeout) break;
    }
    --e->waiting;
    changed_.notify_all();
    if (e->removed) {
        *refusal = refuse(path, 404, "not_found_error", "model '" + e->name + "' is no longer served here");
        return Got::Refused;
    }
    if (e->state == State::Failed) {
        *refusal = refuse(path, 503, "service_unavailable",
                          "model '" + e->name + "' could not be loaded: " + e->why);
        const uint64_t t = rad_mono_ns();
        refusal->set_header("Retry-After",
                            std::to_string(e->retry_ns > t ? (e->retry_ns - t) / 1000000000ull + 1 : 1));
        return Got::Refused;
    }
    if (e->state != State::Open) {
        if (std::chrono::steady_clock::now() < limit && !stopping()) return Got::Waiting;
        *refusal = refuse(path, 503, "service_unavailable",
                          "model '" + e->name + "' is " + state_name(e->state) +
                              " and did not open within the load timeout; try again");
        refusal->set_header("Retry-After", "5");
        return Got::Refused;
    }
    *out = e;
    *held = lease_entry(e);
    /* The whole wait began where *until_ns was set, the entry's timeout before its end. */
    const uint64_t began = *until_ns - (uint64_t)std::max<int64_t>(e->wait_ms, 0) * 1000000ull;
    const double waited = (double)(rad_mono_ns() - std::min(began, rad_mono_ns())) * 1e-9;
    ++e->waits;
    e->wait_s += waited;
    for (size_t k = 0; k < kWaitLe.size(); ++k)
        if (waited <= kWaitLe[k]) ++e->wait_le[k];
    return Got::Leased;
}

/* `base:adapter` reaches the model's server as the adapter's own name, the spelling a one-model
 * process takes: in the query, the form field, or the body's top-level string, in place. */
static void strip_base(HttpRequest* q, const std::string& model) {
    const size_t colon = model.find(':');
    if (colon == std::string::npos) return;
    const std::string adapter = model.substr(colon + 1);
    auto it = q->query.find("model");
    if (it != q->query.end()) { it->second = adapter; return; }
    for (FormPart& p : q->form)
        if (p.name == "model") { p.data = adapter; return; }
    size_t b = 0, e = 0;
    if (json_top_string(q->body, "model", &b, &e)) {
        const std::string raw = q->body.substr(b, e - b);
        const size_t c = raw.find(':');
        if (c != std::string::npos) q->body.replace(b, e - b, raw.substr(c + 1));
    }
}

/* The body asks for a streamed answer: its top-level `stream` is true. */
static bool wants_stream(const HttpRequest& q) {
    size_t b = 0, e = 0;
    return !q.body.empty() && json_top_value(q.body, "stream", &b, &e) &&
           q.body.compare(b, e - b, "true") == 0;
}

std::vector<std::pair<Front::Entry*, std::shared_ptr<void>>> Front::hold_open() {
    std::vector<std::pair<Entry*, std::shared_ptr<void>>> held;
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& p : entries_)
        if (!p->removed && p->state == State::Open) held.emplace_back(p.get(), lease_entry(p.get()));
    return held;
}

bool Front::dashboard_read(const HttpRequest& q) {
    if (q.method != "GET") return false;
    return q.path == "/stats" || q.path == "/sessions" || q.path == "/graph" ||
           q.path == "/server_info" || q.path == "/admin/control-vector";
}

void Front::keep_job(const std::string& id, int64_t expires_at, int64_t max_kept, Handler answer) {
    std::lock_guard<std::mutex> lk(jobs_mu_);
    const int64_t now = (int64_t)std::time(nullptr);
    jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(),
                               [&](const KeptJob& j) { return j.expires_at <= now || j.id == id; }),
                jobs_.end());
    jobs_.push_back(KeptJob{ id, expires_at, std::move(answer) });
    const size_t cap = (size_t)std::max<int64_t>(max_kept, 1);
    if (jobs_.size() > cap) jobs_.erase(jobs_.begin(), jobs_.end() - (std::ptrdiff_t)cap);
}

Handler Front::kept_job(const HttpRequest& q) {
    if (!id_route(q.path)) return nullptr;
    const std::string id = route_id(q.path);
    const int64_t now = (int64_t)std::time(nullptr);
    std::lock_guard<std::mutex> lk(jobs_mu_);
    for (size_t i = 0; i < jobs_.size(); ++i) {
        if (jobs_[i].id != id) continue;
        if (jobs_[i].expires_at <= now) {
            jobs_.erase(jobs_.begin() + (std::ptrdiff_t)i);
            return nullptr;
        }
        Handler h = jobs_[i].answer;
        if (q.method == "DELETE") jobs_.erase(jobs_.begin() + (std::ptrdiff_t)i);
        return h;
    }
    return nullptr;
}

HttpResponse Front::forward(HttpRequest q) {
    if (Handler kept = kept_job(q)) return kept(q);
    const std::string model = requested_model(q);
    /* A dashboard read is not a use of the model: it does not bring one back, and it does not
     * keep one that nobody asks anything of from going idle. */
    if (dashboard_read(q)) {
        Entry* r = nullptr;
        std::shared_ptr<void> hold;
        {
            std::lock_guard<std::mutex> lk(mu_);
            r = resolve(model);
            if (r && r->state != State::Open)
                return refuse(q.path, 409, "model_not_open", "model '" + r->name + "' is " +
                                                                 state_name(r->state) +
                                                                 "; this read does not bring it back");
            if (r && r->server) hold = lease_quiet(r);
        }
        if (hold) {
            const bool sessions = q.path == "/sessions";
            const std::string name = r->name;
            const uint64_t now = rad_mono_ns();
            if (sessions) {
                std::lock_guard<std::mutex> lk(sessions_mu_);
                const auto it = sessions_read_.find(name);
                if (it != sessions_read_.end() && now - it->second.first < kSessionsKeepNs) {
                    HttpResponse kept;
                    kept.status = 200;
                    kept.content_type = "application/json";
                    kept.body = it->second.second;
                    kept.set_header("Cache-Control", "no-store");
                    return kept;
                }
            }
            strip_base(&q, model);
            HttpResponse resp = r->server->router().dispatch(std::move(q));
            if (sessions && resp.status == 200) {
                std::lock_guard<std::mutex> lk(sessions_mu_);
                sessions_read_[name] = { now, resp.body };
            }
            hold_until_close(resp, hold);
            return resp;
        }
    }
    Entry* e = nullptr;
    std::shared_ptr<void> l;
    HttpResponse refusal;
    /* A streaming request for a model that is not open gets its event stream at once. */
    if (keepalive_ms_ > 0 && wants_stream(q)) {
        bool open = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            Entry* r = resolve(model);
            open = !r || r->state == State::Open;
        }
        if (!open) return forward_waiting(std::move(q), model);
    }
    const Got g = lease(model, q.path, -1, nullptr, &e, &l, &refusal);
    if (g != Got::Leased) {
        if (model.empty() && id_route(q.path) && refusal.status == 400) {
            /* Every loaded model that might hold the job, in turn; the first that knows it
             * answers. */
            for (auto& [ent, hold] : hold_open()) {
                HttpResponse r = ent->server->router().dispatch(q);
                if (r.status == 404) continue;
                hold_until_close(r, hold);
                return r;
            }
            return refuse(q.path, 404, "not_found_error", "no model holds " + q.path);
        }
        return refusal;
    }
    strip_base(&q, model);
    HttpResponse r = e->server->router().dispatch(std::move(q));
    hold_until_close(r, l);
    return r;
}

HttpResponse Front::forward_waiting(HttpRequest q, const std::string& model) {
    struct Wait {
        HttpRequest           q;
        std::string           model;
        std::shared_ptr<void> lease;
        HttpResponse          inner;
        bool                  relaying = false;
        bool                  done = false;
        uint64_t              until = 0;   /* the whole wait's end, set by the first lease() */
    };
    auto w = std::make_shared<Wait>();
    w->q = std::move(q);
    w->model = model;
    HttpResponse r;
    r.status = 200;
    r.content_type = "text/event-stream";
    r.set_header("Cache-Control", "no-cache");
    r.next = [this, w](std::string& chunk) -> bool {
        chunk.clear();
        if (w->done) return false;
        if (w->relaying) {
            if (!w->inner.next) { w->done = true; return false; }
            const bool more = w->inner.next(chunk);
            if (!more) w->done = true;
            return more || !chunk.empty();
        }
        Entry* e = nullptr;
        HttpResponse refusal;
        const Got g = lease(w->model, w->q.path, keepalive_ms_, &w->until, &e, &w->lease, &refusal);
        if (g == Got::Waiting) {
            chunk = ": loading model " + w->model + "\n\n";
            return true;
        }
        if (g == Got::Refused) {
            chunk = "data: " + refusal.body + "\n\n";
            w->done = true;
            return true;
        }
        strip_base(&w->q, w->model);
        w->inner = e->server->router().dispatch(std::move(w->q));
        if (!w->inner.streaming()) {
            /* The model answered whole (a refusal, most likely): one event carrying it. */
            chunk = "data: " + w->inner.body + "\n\n";
            w->done = true;
            return true;
        }
        w->relaying = true;
        return true;
    };
    r.on_close = [w](bool ok) {
        if (w->inner.on_close) w->inner.on_close(ok);
        w->inner.on_close = nullptr;
        w->lease.reset();
    };
    return r;
}

void Front::forward_socket(const HttpRequest& q, WsConn& c) {
    const std::string model = requested_model(q);
    Entry* e = nullptr;
    std::shared_ptr<void> l;
    HttpResponse refusal;
    if (lease(model, q.path, -1, nullptr, &e, &l, &refusal) != Got::Leased) {
        c.close(1008, model.empty() ? "name a model with ?model=" : "model '" + model + "' is not served here");
        return;
    }
    const SocketHandler* h = e->server->router().socket_for(q.path);
    if (!h) { c.close(1008, "this model serves no " + q.path); return; }
    (*h)(q, c);
}

HttpResponse Front::models(const HttpRequest& q) {
    /* One model's card: asked of the model it names, under its own id. A model not loaded is
     * described without loading it. */
    if (q.path.rfind("/v1/models/", 0) == 0 && q.path.size() > 11) {
        const std::string id = q.path.substr(11);
        Entry* e = nullptr;
        std::shared_ptr<void> l;
        {
            std::lock_guard<std::mutex> lk(mu_);
            Entry* r = resolve(id);
            if (!r)
                return refuse(q.path, 404, "not_found_error", "no model '" + id + "' is served here");
            if (r->state != State::Open) {
                json card = {{"id", r->name}, {"object", "model"}, {"owned_by", "radiance"},
                             {"state", state_name(r->state)}};
                return HttpResponse::json_body(200, card.dump());
            }
            e = r;
            l = lease_entry(r);
        }
        HttpRequest one = q;
        const size_t colon = id.find(':');
        one.path = "/v1/models/" + (colon != std::string::npos ? id.substr(colon + 1) : e->name);
        return e->server->router().dispatch(std::move(one));
    }
    struct Row { Entry* e; std::shared_ptr<void> hold; std::string name; std::vector<std::string> aliases; State st; };
    std::vector<Row> rows;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& p : entries_) {
            if (p->removed) continue;
            Row r{p.get(), nullptr, p->name, p->aliases, p->state};
            if (p->state == State::Open) r.hold = lease_entry(p.get());
            rows.push_back(std::move(r));
        }
    }
    json data = json::array();
    for (Row& row : rows) {
        json extra = json::object();
        if (describe_) {
            const json j = json::parse(describe_(row.name), nullptr, false);
            if (j.is_object()) extra = j;
        }
        std::vector<json> cards;
        if (row.hold) {
            HttpRequest one = q;
            one.path = "/v1/models";
            const HttpResponse r = row.e->server->router().dispatch(std::move(one));
            const json j = r.status == 200 ? json::parse(r.body, nullptr, false) : json();
            if (!j.is_discarded() && j.is_object() && j.contains("data"))
                for (json card : j["data"]) {
                    /* An adapter's card is addressed base:adapter in a process of several models. */
                    if (card.contains("parent") && card.value("id", "") != row.name) {
                        card["id"] = row.name + ":" + card.value("id", "");
                        card["parent"] = row.name;
                    }
                    cards.push_back(card);
                }
        }
        if (cards.empty())
            cards.push_back({{"id", row.name}, {"object", "model"}, {"owned_by", "radiance"}});
        for (json& card : cards) {
            const bool own = card.value("id", "") == row.name;
            if (own) {
                card["state"] = state_name(row.st);
                for (const auto& [k, v] : extra.items()) card[k] = v;
            }
            data.push_back(card);
            if (own)
                for (const std::string& a : row.aliases) {
                    json alias = card;
                    alias["id"] = a;
                    alias["parent"] = row.name;
                    data.push_back(alias);
                }
        }
    }
    json out;
    out["object"] = "list";
    out["data"] = data;
    return HttpResponse::json_body(200, out.dump());
}

HttpResponse Front::health(const HttpRequest& q) {
    /* One model: its own answer when loaded, its state when not; asking does not load it. */
    if (auto it = q.query.find("model"); it != q.query.end()) {
        Entry* e = nullptr;
        std::shared_ptr<void> l;
        {
            std::lock_guard<std::mutex> lk(mu_);
            e = resolve(it->second);
            if (!e) return refuse(q.path, 404, "not_found_error", "no model '" + it->second + "' is served here");
            if (e->state != State::Open) {
                json j = {{"status", state_name(e->state)}, {"model", e->name}};
                if (e->state == State::Failed) j["error"] = e->why;
                return HttpResponse::json_body(503, j.dump());
            }
            l = lease_entry(e);
        }
        return e->server->router().dispatch(q);
    }
    /* The process: healthy once every model that must be loaded (pinned or preloaded) is, and
     * each of those says so. */
    std::vector<std::pair<Entry*, std::shared_ptr<void>>> held;
    json states = json::object();
    std::string missing;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& p : entries_) {
            if (p->removed) continue;
            states[p->name] = state_name(p->state);
            if (!p->required) continue;
            if (p->state == State::Open) held.emplace_back(p.get(), lease_entry(p.get()));
            else missing += (missing.empty() ? "" : ", ") + p->name;
        }
    }
    if (!missing.empty()) {
        json j = {{"status", "loading"}, {"waiting_for", missing}, {"models", states}};
        return HttpResponse::json_body(503, j.dump());
    }
    for (auto& [e, hold] : held) {
        HttpResponse r = e->server->router().dispatch(q);
        if (r.status != 200) return r;
    }
    json j = {{"status", "ok"}, {"models", states}};
    return HttpResponse::json_body(200, j.dump());
}

HttpResponse Front::metrics(const HttpRequest& q) {
    std::vector<std::string> texts;
    std::string type;
    for (auto& [e, hold] : hold_open()) {
        HttpResponse r = e->server->router().dispatch(q);
        if (r.status != 200) continue;
        if (type.empty()) type = r.content_type;
        texts.push_back(std::move(r.body));
    }
    if (fleet_metrics_) texts.push_back(fleet_metrics_());
    HttpResponse r = HttpResponse::text(200, merge_metrics(texts));
    r.content_type = type.empty() ? "text/plain; version=0.0.4; charset=utf-8" : type;
    return r;
}

int Front::run() { return transport_.serve(router_, topts_); }

void Front::stop() {
    stopping_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(mu_);
        changed_.notify_all();
    }
    transport_.stop();
}

}  /* namespace server */
}  /* namespace rad */
