/* core/server/history.cpp -- see history.h. */
#include "server/history.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace rad {
namespace server {

namespace {

const double kNone = std::numeric_limits<double>::quiet_NaN();

/* A member's number, or NaN where the object has none. */
double num(const json& o, const char* k) {
    if (!o.is_object() || !o.contains(k)) return kNone;
    const json& v = o[k];
    return v.is_number() ? v.get<double>() : kNone;
}
double num0(const json& o, const char* k) {
    const double v = num(o, k);
    return std::isfinite(v) ? v : 0.0;
}
const json* member(const json& o, const char* k) {
    return o.is_object() && o.contains(k) ? &o[k] : nullptr;
}
std::string text(const json& o, const char* k) {
    if (!o.is_object() || !o.contains(k)) return std::string();
    const json& v = o[k];
    return v.is_string() ? v.get<std::string>() : v.dump();
}

/* Four significant digits: what a chart resolves, and a third of the bytes on the wire. */
double round4(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.4g", v);
    return std::strtod(b, nullptr);
}

void put(History::Row& r, const char* k, double v) {
    if (std::isfinite(v)) r.emplace_back(k, v);
}

/* A model's state as a level a chart can draw; an unknown state is 0. */
double state_code(const std::string& s) {
    if (s == "resident") return 3;
    if (s == "loading" || s == "closing") return 2;
    if (s == "parked") return 1;
    if (s == "failed") return 4;
    if (s == "waiting") return 5;
    return 0;
}

}  // namespace

/* ------------------------------------------------------------------ rows */

History::Row History::model_row(const json& s, const json* prev, double dt) {
    Row row;
    put(row, "decode", num(s, "decode_tps"));
    put(row, "prefill", num(s, "prefill_tps"));
    put(row, "running", num(s, "running"));
    put(row, "waiting", num(s, "waiting"));
    put(row, "preempted", num(s, "preempted"));
    put(row, "flight", num(s, "in_flight"));
    put(row, "hit", num(s, "prefix_hit_rate"));
    put(row, "kvUsed", num(s, "kv_bytes_used"));
    put(row, "kvCap", num(s, "kv_bytes_total"));
    /* The pool's bytes split by tokens: what finished turns keep for reuse and what running
     * requests read. */
    {
        const double held = num0(s, "kv_bytes_used"), used = num0(s, "kv_tokens_used");
        const double kept = std::min(num0(s, "kv_tokens_cached"), used);
        const double kept_b = used > 0 ? held * kept / used : 0.0;
        put(row, "kvKept", kept_b);
        put(row, "kvRead", std::max(0.0, held - kept_b));
    }
    if (const double best = num(s, "step_ns_decode_best"); best > 0) put(row, "stepBest", best / 1e6);
    const json* jobs = member(s, "jobs");
    if (jobs) {
        put(row, "jobRun", num(*jobs, "running"));
        put(row, "jobDone", num(*jobs, "done"));
    }
    /* A prompt's fresh tokens over its own prefill time, for the requests whose prefill ended
     * since the last sample: the scheduler's rate divides by everything the step loop did, so
     * one short prompt among long decodes would read as a few tokens a second. */
    if (const json* rq = member(s, "requests"); rq && rq->is_array()) {
        std::set<std::string> seen;
        if (prev)
            if (const json* pq = member(*prev, "requests"); pq && pq->is_array())
                for (const json& r : *pq)
                    if (num(r, "prefill_s") > 0) seen.insert(text(r, "id"));
        double sum = 0;
        int n = 0;
        for (const json& r : *rq) {
            const double pf = num(r, "prefill_s"), fresh = num0(r, "prompt") - num0(r, "cached");
            if (pf > 0 && fresh > 0 && !seen.count(text(r, "id"))) {
                sum += fresh / pf;
                ++n;
            }
        }
        if (n) put(row, "prefillRate", sum / n);
    }
    if (!prev || !(dt > 0)) return row;
    const json& p = *prev;

    if (jobs)
        if (const json* pj = member(p, "jobs"); pj && num0(*jobs, "done") > num0(*pj, "done"))
            put(row, "jobTime", num(*jobs, "last_ms") / 1000.0);
    put(row, "reqRate", std::max(0.0, (num0(s, "total_requests") - num0(p, "total_requests")) / dt));
    if (const double steps = num0(s, "steps") - num0(p, "steps"); steps > 0)
        put(row, "stepAny", (num0(s, "step_ns") - num0(p, "step_ns")) / steps / 1e6);
    const double ds = num0(s, "steps_decode") - num0(p, "steps_decode");
    if (ds > 0) put(row, "step", (num0(s, "step_ns_decode") - num0(p, "step_ns_decode")) / ds / 1e6);
    /* Acceptance is accepted over reached, summed over the draft positions that were verified. */
    {
        const json* dp = member(s, "draft_positions");
        const json* pp = member(p, "draft_positions");
        if (dp && pp && dp->is_array() && pp->is_array() && dp->size() == pp->size()) {
            double reached = 0, accepted = 0;
            for (size_t i = 0; i < dp->size(); ++i) {
                const double dr = num0((*dp)[i], "reached") - num0((*pp)[i], "reached");
                const double da = num0((*dp)[i], "accepted") - num0((*pp)[i], "accepted");
                if (dr > 0) {
                    reached += dr;
                    accepted += da;
                }
            }
            if (reached > 0) put(row, "accept", accepted / reached);
            if (reached > 0 && ds > 0) put(row, "tokStep", 1.0 + accepted / ds);
        }
    }
    {
        const json* lk = member(s, "link");
        const json* lp = member(p, "link");
        if (lk && lp) {
            auto d = [&](const char* f) { return std::max(0.0, (num0(*lk, f) - num0(*lp, f)) / dt); };
            put(row, "lkAr", d("ar_bytes"));
            put(row, "lkUp", d("h2d_bytes"));
            put(row, "lkDn", d("d2h_bytes"));
            put(row, "lkSsd", d("ssd_bytes"));
            put(row, "lkStream", d("stream_bytes") + d("kv_stream_bytes"));
        }
    }
    {
        const json* ex = member(s, "experts");
        const json* xp = member(p, "experts");
        if (ex && xp) {
            if (const double routed = num0(*ex, "routed") - num0(*xp, "routed"); routed > 0)
                put(row, "expHit", (num0(*ex, "routed_res") - num0(*xp, "routed_res")) / routed);
            put(row, "expIn", std::max(0.0, (num0(*ex, "promotions") - num0(*xp, "promotions")) / dt));
            put(row, "expOut", std::max(0.0, (num0(*ex, "demotions") - num0(*xp, "demotions")) / dt));
        }
    }
    return row;
}

History::Row History::fleet_row(const json& f) {
    Row row;
    const json* models = member(f, "models");
    const json* cards = member(f, "cards");
    if (models && models->is_array())
        for (const json& m : *models) row.emplace_back("s:" + text(m, "id"), state_code(text(m, "state")));
    if (!cards || !cards->is_array()) return row;
    for (const json& c : *cards) {
        const std::string card = text(c, "card");
        auto name = [&](const char* k) { return std::string(k) + card; };
        if (const double v = num(c, "busy"); std::isfinite(v)) row.emplace_back(name("busy"), v);
        if (const double v = num(c, "power_w"); std::isfinite(v)) row.emplace_back(name("power"), v);
        if (const double v = num(c, "temp_c"); std::isfinite(v)) row.emplace_back(name("temp"), v);
        if (const double v = num(c, "free_bytes"); std::isfinite(v)) row.emplace_back(name("free"), v);
        if (!models || !models->is_array()) continue;
        /* A model's bytes on this card, 0 where it holds none: the stacked memory chart draws a
         * model that left as an empty band, not as a gap. */
        const double ci = num(c, "card");
        for (const json& m : *models) {
            double bytes = 0;
            /* What the model holds on this card now (a loading or partly parked one too), else
             * its claim on each of its cards. */
            const json* by = member(m, "vram_bytes_by_card");
            if (by && by->is_object()) {
                if (const auto it = by->find(card); it != by->end() && it->is_number()) bytes = it->get<double>();
            } else if (const json* on = member(m, "cards"); on && on->is_array()) {
                for (const json& x : *on)
                    if (x.is_number() && x.get<double>() == ci) bytes = num0(m, "vram_bytes_per_card");
            }
            row.emplace_back("c" + card + ":" + text(m, "id"), bytes);
        }
    }
    return row;
}

/* ------------------------------------------------------------------ the ring */

void History::Series::push(double at, const Row& row) {
    if (t.empty()) t.assign((size_t)kRows, 0.0);
    const size_t i = head;
    /* Kept as it is written out, to the millisecond: a page asks for what came after the last
     * time it was given, and a time kept finer than it was sent would be later than itself. */
    t[i] = std::round(at * 1000.0) / 1000.0;
    for (auto& [k, col] : c) col[i] = std::numeric_limits<float>::quiet_NaN();
    for (const auto& [k, v] : row) {
        auto it = c.find(k);
        if (it == c.end())
            it = c.emplace(k, std::vector<float>((size_t)kRows, std::numeric_limits<float>::quiet_NaN())).first;
        it->second[i] = (float)v;
    }
    head = (head + 1) % (size_t)kRows;
    if (n < (size_t)kRows) ++n;
}

void History::Series::write(json& out, double since) const {
    json times = json::array();
    json cols = json::object();
    std::vector<size_t> at;
    const size_t first = (head + (size_t)kRows - n) % (size_t)kRows;
    /* The ring holds the hour at its fastest rate, so at one row a second it holds more than the
     * hour: only what is within kKeepS of the latest row is given. */
    const double oldest = n ? t[(head + (size_t)kRows - 1) % (size_t)kRows] - (double)kKeepS : 0.0;
    for (size_t k = 0; k < n; ++k) {
        const size_t i = (first + k) % (size_t)kRows;
        if (t[i] > since && t[i] > oldest) at.push_back(i);
    }
    for (size_t i : at) times.push_back(t[i]);
    for (const auto& [name, col] : c) {
        bool any = false;
        for (size_t i : at) any = any || std::isfinite(col[i]);
        if (!any) continue;
        json a = json::array();
        for (size_t i : at) {
            if (std::isfinite(col[i])) a.push_back(round4((double)col[i]));
            else a.push_back(nullptr);
        }
        cols[name] = std::move(a);
    }
    out["keep_s"] = kKeepS;
    out["t"] = std::move(times);
    out["c"] = std::move(cols);
}

/* ------------------------------------------------------------------ samples */

void History::track_requests(Model& m, const json& stats, double t) {
    auto finish = [&](const json& r) {
        const double fresh = num0(r, "prompt") - num0(r, "cached");
        const double pf = num(r, "prefill_s"), ttft = num(r, "ttft_s"), tps = num(r, "tps");
        json h;
        h["id"] = text(r, "id");
        h["prompt"] = num0(r, "prompt");
        h["out"] = num0(r, "output");
        h["ctx"] = num0(r, "ctx");
        if (std::isfinite(ttft)) h["ttft"] = ttft; else h["ttft"] = nullptr;
        if (pf > 0 && fresh > 0) h["pf"] = round4(fresh / pf); else h["pf"] = nullptr;
        if (tps > 0) h["tps"] = round4(tps); else h["tps"] = nullptr;
        h["at"] = (int64_t)std::llround(t * 1000.0);
        m.finished.push_back(std::move(h));
        if (m.finished.size() > (size_t)kRequests) m.finished.pop_front();
    };
    std::set<std::string> seen;
    if (const json* rq = member(stats, "requests"); rq && rq->is_array())
        for (const json& r : *rq) {
            const std::string id = text(r, "id");
            if (id.empty()) continue;
            seen.insert(id);
            m.live[id] = r;
            if (text(r, "state") == "done" && m.done.insert(id).second) finish(r);
        }
    /* A request that left the table without being seen done ended between two samples. */
    for (auto it = m.live.begin(); it != m.live.end();) {
        if (seen.count(it->first)) {
            ++it;
            continue;
        }
        if (!m.done.count(it->first)) finish(it->second);
        m.done.erase(it->first);
        it = m.live.erase(it);
    }
}

void History::model_sample(const std::string& id, const json& stats, double t, std::string body) {
    std::lock_guard<std::mutex> lk(mu_);
    Model& m = models_[id];
    m.latest = std::move(body);
    /* Counters start over with the model: a sample of another start is no predecessor. */
    const bool same = m.prev.is_object() && num(m.prev, "started_at") == num(stats, "started_at");
    if (!same) {
        m.live.clear();
        m.done.clear();
    }
    m.series.push(t, model_row(stats, same ? &m.prev : nullptr, t - m.prev_t));
    track_requests(m, stats, t);
    m.prev = stats;
    m.prev_t = t;
}

void History::fleet_sample(const json& fleet, double t, std::string body) {
    std::lock_guard<std::mutex> lk(mu_);
    fleet_.push(t, fleet_row(fleet));
    fleet_live_ = std::move(body);
}

namespace {
int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

bool History::watched(double seconds) const {
    const int64_t at = asked_ns_.load(std::memory_order_relaxed);
    return at != 0 && (double)(steady_ns() - at) * 1e-9 < seconds;
}

void History::model_closed(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    if (auto it = models_.find(id); it != models_.end()) it->second.latest.clear();
}

void History::forget(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    models_.erase(id);
}

json History::model_since(const std::string& id, double since) const {
    std::lock_guard<std::mutex> lk(mu_);
    json out = json::object();
    json reqs = json::array();
    auto it = models_.find(id);
    if (it == models_.end()) {
        Series().write(out, since);
    } else {
        it->second.series.write(out, since);
        for (const json& h : it->second.finished)
            /* In whole milliseconds on both sides, as `at` is. */
            if (num0(h, "at") > (double)std::llround(since * 1000.0)) reqs.push_back(h);
    }
    out["requests"] = std::move(reqs);
    return out;
}

json History::fleet_since(double since) const {
    std::lock_guard<std::mutex> lk(mu_);
    json out = json::object();
    fleet_.write(out, since);
    return out;
}

namespace {
/* `{...}` with `"live": <text>` appended, the text spliced in as it was read. */
std::string with_live(const json& o, const std::string& live) {
    std::string b = o.dump();
    b.pop_back();
    b += ",\"live\":";
    b += live.empty() ? "null" : live;
    b += '}';
    return b;
}
}  // namespace

std::string History::model_body(const std::string& id, double since) const {
    asked_ns_.store(steady_ns(), std::memory_order_relaxed);
    const json o = model_since(id, since);
    std::string live;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (auto it = models_.find(id); it != models_.end()) live = it->second.latest;
    }
    return with_live(o, live);
}

std::string History::fleet_body(double since) const {
    asked_ns_.store(steady_ns(), std::memory_order_relaxed);
    const json o = fleet_since(since);
    std::string live;
    {
        std::lock_guard<std::mutex> lk(mu_);
        live = fleet_live_;
    }
    return with_live(o, live);
}

}  // namespace server
}  // namespace rad
