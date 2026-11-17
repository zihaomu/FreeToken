/* xstats.cpp -- the --expert-stats tally, its merge across ranks and its file. See xstats.h. */
#include "xstats.h"

#include "util/json.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <set>
#include <unistd.h>

namespace rad {

const char* xs_kind_name(int kind) {
    switch (kind) {
        case kXsPrefill: return "prefill";
        case kXsDecode:  return "decode";
        case kXsMixed:   return "mixed";
        case kXsDraft:   return "draft";
        default:         return "?";
    }
}

/* ------------------------------------------------------------------ the map */

int xs_map_build(const std::vector<std::vector<std::vector<int32_t>>>& ids,
                 const std::vector<int32_t>& n_model, int32_t n_trunk, XsMap* out) {
    if (!out || ids.empty()) return RAD_E_INVAL;
    const int32_t R = (int32_t)ids.size();
    int32_t L = 0;
    for (const auto& r : ids) L = std::max(L, (int32_t)r.size());
    out->n_rank = R;
    out->n_layer = L;
    out->n_trunk = std::max(0, std::min(n_trunk, L));
    out->n_expert.assign((size_t)L, 0);
    out->how.assign((size_t)L, "none");
    out->gid.assign((size_t)R, std::vector<std::vector<int32_t>>((size_t)L));
    constexpr int32_t kMaxId = 1 << 20;

    for (int32_t l = 0; l < L; ++l) {
        std::vector<size_t> n((size_t)R, 0);
        bool any = false;
        for (int32_t r = 0; r < R; ++r) {
            n[(size_t)r] = (size_t)l < ids[(size_t)r].size() ? ids[(size_t)r][(size_t)l].size() : 0;
            any = any || n[(size_t)r] > 0;
        }
        if (!any) continue;
        for (int32_t r = 0; r < R; ++r) out->gid[(size_t)r][(size_t)l].assign(n[(size_t)r], XsMap::kUnknown);

        /* The container's ids, when every index of every rank has one, distinct within the rank. */
        bool ok = true;
        int32_t mx = -1;
        for (int32_t r = 0; r < R && ok; ++r) {
            std::set<int32_t> seen;
            for (size_t e = 0; e < n[(size_t)r] && ok; ++e) {
                const int32_t g = ids[(size_t)r][(size_t)l][e];
                if (g < 0 || g >= kMaxId || !seen.insert(g).second) ok = false;
                else mx = std::max(mx, g);
            }
        }
        if (ok) {
            /* An expert several ranks hold (a width slice each) is counted on the lowest. */
            std::vector<uint8_t> owned((size_t)mx + 1, 0);
            for (int32_t r = 0; r < R; ++r)
                for (size_t e = 0; e < n[(size_t)r]; ++e) {
                    const int32_t g = ids[(size_t)r][(size_t)l][e];
                    if (owned[(size_t)g]) {
                        out->gid[(size_t)r][(size_t)l][e] = XsMap::kElsewhere;
                    } else {
                        owned[(size_t)g] = 1;
                        out->gid[(size_t)r][(size_t)l][e] = g;
                    }
                }
            out->n_expert[(size_t)l] = mx + 1;
            out->how[(size_t)l] = "container";
            continue;
        }

        /* No ids: the indices are the model's on one rank, or where every rank holds the model's
         * count; otherwise each rank holds a run of its own, in rank order. */
        bool same = true;
        for (int32_t r = 1; r < R; ++r) same = same && n[(size_t)r] == n[0];
        const int64_t nm = (size_t)l < n_model.size() ? n_model[(size_t)l] : 0;
        if (R == 1 || (same && nm > 0 && (int64_t)n[0] == nm)) {
            for (int32_t r = 0; r < R; ++r)
                for (size_t e = 0; e < n[(size_t)r]; ++e)
                    out->gid[(size_t)r][(size_t)l][e] = r == 0 ? (int32_t)e : XsMap::kElsewhere;
            out->n_expert[(size_t)l] = (int32_t)n[0];
            out->how[(size_t)l] = R == 1 ? "index" : "replicated";
        } else {
            int64_t base = 0;
            for (int32_t r = 0; r < R; ++r) {
                for (size_t e = 0; e < n[(size_t)r]; ++e)
                    out->gid[(size_t)r][(size_t)l][e] = (int32_t)(base + (int64_t)e);
                base += (int64_t)n[(size_t)r];
            }
            out->n_expert[(size_t)l] = (int32_t)base;
            out->how[(size_t)l] = "rank order";
            if (nm > 0 && base != nm)
                RAD_WARN("--expert-stats: layer %d's ranks hold %lld experts between them, the model "
                         "%lld; its counts are numbered in rank order regardless",
                         (int)l, (long long)base, (long long)nm);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ one rank's tally */

void XsTally::init(const XsMap* map, int rank) {
    map_ = map;
    rank_ = rank;
    ring_.assign((size_t)kRing, Rec{});
    const size_t L = (size_t)map->n_layer;
    off_.assign(L + 1, 0);
    for (size_t l = 0; l < L; ++l) off_[l + 1] = off_[l] + (size_t)map->n_expert[l];
    for (int k = 0; k < kXsKinds; ++k) {
        counts_[k].assign(off_[L], 0);
        hist_[k].assign(L, 0);
        passes_[k].assign(L, 0);
        tok_[k].assign(L, 0);
        sel_[k].assign(L, 0);
        steps_[k] = tokens_[k] = 0;
    }
    top_k_.assign(L, 0);
    mixed_decode_ = draft_passes_ = draft_rows_ = 0;
    credited_ = unmapped_ = unclassified_ = dropped_ = 0;
}

void XsTally::issued(const RadBatch& b) {
    issued(b.step, b.draft_pass, b.n_seq, b.n_seq_decode, b.n_tok, b.n_tok_decode);
}

void XsTally::issued(int step, int draft_pass, int64_t n_seq, int64_t n_seq_decode, int64_t n_tok,
                     int64_t n_tok_decode) {
    if (step < 0) return;
    std::lock_guard<std::mutex> lk(mu_);
    Rec& r = ring_[(size_t)(step % kRing)];
    if (r.step != step) {
        r = Rec{};
        r.step = step;
    }
    if (draft_pass == 0) {
        const int k = xs_trunk_kind(n_seq, n_seq_decode);
        r.kind = (uint8_t)k;
        r.trunk = true;
        r.n_tok = n_tok;
        ++steps_[k];
        tokens_[k] += (uint64_t)std::max<int64_t>(n_tok, 0);
        if (k == kXsMixed) mixed_decode_ += (uint64_t)std::max<int64_t>(n_tok_decode, 0);
    } else {
        /* Each draft pass overwrites the head's row, so the bank holds the last pass's rows. */
        r.draft_rows = n_tok;
        ++draft_passes_;
        draft_rows_ += (uint64_t)std::max<int64_t>(n_tok, 0);
    }
}

void XsTally::credit(int layer, int step, const int32_t* counts, int64_t n, int reports,
                     int64_t top_k) {
    if (!map_ || !counts || layer < 0 || layer >= map_->n_layer || step < 0) return;
    const Rec& r = ring_[(size_t)(step % kRing)];
    const bool known = r.step == step;
    int kind;
    int64_t toks = 0;
    if (layer >= map_->n_trunk) {
        kind = kXsDraft;
        if (known) toks = r.draft_rows;
        else ++unclassified_;
    } else if (known && r.trunk) {
        kind = r.kind;
        toks = r.n_tok;
    } else {
        kind = kXsMixed;
        ++unclassified_;
    }

    const std::vector<int32_t>& g = map_->gid[(size_t)rank_][(size_t)layer];
    const int64_t m = std::min<int64_t>(n, (int64_t)g.size());
    uint64_t* const plane = counts_[kind].data() + off_[(size_t)layer];
    uint64_t sel = 0;
    for (int64_t e = 0; e < m; ++e) {
        const int32_t c = counts[e];
        if (c <= 0) continue;
        const int32_t gi = g[(size_t)e];
        if (gi >= 0) {
            plane[gi] += (uint64_t)c;
            sel += (uint64_t)c;
        } else if (gi == XsMap::kUnknown) {
            unmapped_ += (uint64_t)c;
        }
    }
    for (int64_t e = m; e < n; ++e)
        if (counts[e] > 0) unmapped_ += (uint64_t)counts[e];

    ++hist_[kind][(size_t)layer];
    passes_[kind][(size_t)layer] += (uint64_t)std::max(reports, 1);
    tok_[kind][(size_t)layer] += (uint64_t)std::max<int64_t>(toks, 0);
    sel_[kind][(size_t)layer] += sel;
    if (top_k > top_k_[(size_t)layer]) top_k_[(size_t)layer] = (int32_t)top_k;
    ++credited_;
}

void XsTally::snapshot_into(XsSnapshot* s, const std::vector<int32_t>& token_rank, bool issued) {
    std::lock_guard<std::mutex> lk(mu_);
    const size_t L = (size_t)map_->n_layer;
    for (int k = 0; k < kXsKinds; ++k) {
        for (size_t l = 0; l < L; ++l) {
            std::vector<uint64_t>& row = s->counts[k][l];
            const uint64_t* src = counts_[k].data() + off_[l];
            for (size_t e = 0; e < row.size(); ++e) row[e] += src[e];
            s->selections[k][l] += sel_[k][l];
            if (l < token_rank.size() && token_rank[l] == rank_) {
                s->histograms[k][l] = hist_[k][l];
                s->passes[k][l] = passes_[k][l];
                s->obs_tokens[k][l] = tok_[k][l];
            }
        }
        if (issued) {
            s->steps[k] = steps_[k];
            s->tokens[k] = tokens_[k];
        }
    }
    if (issued) {
        s->mixed_decode_tokens = mixed_decode_;
        s->draft_passes = draft_passes_;
        s->draft_rows = draft_rows_;
    }
    for (size_t l = 0; l < L; ++l) s->top_k[l] = std::max(s->top_k[l], top_k_[l]);
    s->rank_credited[(size_t)rank_] = credited_;
    s->rank_unmapped[(size_t)rank_] = unmapped_;
    s->rank_unclassified[(size_t)rank_] = unclassified_;
    s->rank_dropped[(size_t)rank_] = dropped_;
}

/* ------------------------------------------------------------------ the file */

namespace {

void put_u64s(std::string& o, const std::vector<uint64_t>& v) {
    o += '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o += ',';
        o += std::to_string(v[i]);
    }
    o += ']';
}

void put_i32s(std::string& o, const std::vector<int32_t>& v) {
    o += '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o += ',';
        o += std::to_string(v[i]);
    }
    o += ']';
}

std::string jstr(const std::string& s) { return Json(s).dump(); }

/* {"prefill": f(0), "decode": f(1), ...} on one line. */
template <typename F>
void put_kinds(std::string& o, F f) {
    o += '{';
    for (int k = 0; k < kXsKinds; ++k) {
        if (k) o += ", ";
        o += jstr(xs_kind_name(k));
        o += ": ";
        f(o, k);
    }
    o += '}';
}

}  /* namespace */

std::string xs_json(const XsSnapshot& s) {
    std::string o;
    o.reserve(4096 + (size_t)s.n_layer * (size_t)std::max(s.n_expert, 1) * kXsKinds * 4);
    o += "{\n";
    o += "  \"format\": \"radiance-expert-stats/1\",\n";
    o += "  \"model\": " + jstr(s.model) + ",\n";
    o += "  \"arch\": " + jstr(s.arch) + ",\n";
    o += "  \"container\": " + jstr(s.container) + ",\n";
    o += "  \"tp\": " + std::to_string(s.tp) + ",\n";
    o += std::string("  \"final\": ") + (s.final_ ? "true" : "false") + ",\n";
    o += "  \"written_unix\": " + std::to_string(s.written_unix) + ",\n";
    char up[64];
    snprintf(up, sizeof up, "%.1f", s.uptime_s);
    o += std::string("  \"uptime_s\": ") + up + ",\n";
    o += "  \"n_layers\": " + std::to_string(s.n_layer) + ",\n";
    o += "  \"n_trunk_layers\": " + std::to_string(s.n_trunk) + ",\n";
    o += "  \"n_experts\": " + std::to_string(s.n_expert) + ",\n";
    o += "  \"layer_experts\": ";
    put_i32s(o, s.layer_experts);
    o += ",\n  \"top_k\": ";
    put_i32s(o, s.top_k);
    o += ",\n  \"map\": [";
    for (size_t l = 0; l < s.how.size(); ++l) o += (l ? "," : "") + jstr(s.how[l]);
    o += "],\n  \"kinds\": [";
    for (int k = 0; k < kXsKinds; ++k) o += (k ? "," : "") + jstr(xs_kind_name(k));
    o += "],\n";
    /* Issued: trunk steps and tokens by kind; the draft kind's are passes and rows. */
    o += "  \"steps\": ";
    put_kinds(o, [&](std::string& x, int k) {
        x += std::to_string(k == kXsDraft ? s.draft_passes : s.steps[k]);
    });
    o += ",\n  \"tokens\": ";
    put_kinds(o, [&](std::string& x, int k) {
        x += std::to_string(k == kXsDraft ? s.draft_rows : s.tokens[k]);
    });
    o += ",\n  \"mixed_decode_tokens\": " + std::to_string(s.mixed_decode_tokens);
    auto per_layer = [&](const char* name, const std::vector<uint64_t>* v) {
        o += ",\n  \"";
        o += name;
        o += "\": ";
        put_kinds(o, [&](std::string& x, int k) { put_u64s(x, v[k]); });
    };
    per_layer("histograms", s.histograms);
    per_layer("passes", s.passes);
    per_layer("observed_tokens", s.obs_tokens);
    per_layer("selections", s.selections);
    o += ",\n  \"ranks\": {\"credited\": ";
    put_u64s(o, s.rank_credited);
    o += ", \"unmapped\": ";
    put_u64s(o, s.rank_unmapped);
    o += ", \"unclassified\": ";
    put_u64s(o, s.rank_unclassified);
    o += ", \"dropped\": ";
    put_u64s(o, s.rank_dropped);
    o += "},\n  \"counts\": {";
    for (int k = 0; k < kXsKinds; ++k) {
        o += k ? ",\n    " : "\n    ";
        o += jstr(xs_kind_name(k));
        o += ": [";
        for (size_t l = 0; l < s.counts[k].size(); ++l) {
            o += l ? ",\n      " : "\n      ";
            put_u64s(o, s.counts[k][l]);
        }
        o += "\n    ]";
    }
    o += "\n  }\n}\n";
    return o;
}

int xs_write_atomic(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp." + std::to_string((long long)getpid());
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) {
        RAD_WARN("--expert-stats: cannot write %s: %s", tmp.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    const bool wrote = std::fwrite(text.data(), 1, text.size(), f) == text.size() &&
                       std::fflush(f) == 0 && fsync(fileno(f)) == 0;
    const int err = errno;
    if (std::fclose(f) != 0 || !wrote) {
        RAD_WARN("--expert-stats: writing %s failed: %s", tmp.c_str(), std::strerror(wrote ? errno : err));
        ::unlink(tmp.c_str());
        return RAD_E_IO;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        RAD_WARN("--expert-stats: %s -> %s: %s", tmp.c_str(), path.c_str(), std::strerror(errno));
        ::unlink(tmp.c_str());
        return RAD_E_IO;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the engine's side */

int ExpertStats::init(const std::string& path, const XsMap& map, XsSnapshot info,
                      double interval_s) {
    if (path.empty() || map.n_rank <= 0) return RAD_E_INVAL;
    path_ = path;
    map_ = map;
    info_ = std::move(info);
    interval_s_ = interval_s > 0.0 ? interval_s : 30.0;
    t0_ns_ = rad_mono_ns();

    const size_t R = (size_t)map_.n_rank, L = (size_t)map_.n_layer;
    tallies_.clear();
    counts_on_.assign(R, 0);
    token_rank_.assign(L, -1);
    for (size_t r = 0; r < R; ++r) {
        tallies_.push_back(std::make_unique<XsTally>());
        tallies_.back()->init(&map_, (int)r);
        for (size_t l = 0; l < L; ++l)
            for (int32_t g : map_.gid[r][l])
                if (g != XsMap::kElsewhere) {
                    counts_on_[r] = 1;
                    if (token_rank_[l] < 0) token_rank_[l] = (int32_t)r;
                    break;
                }
    }
    issued_rank_ = -1;
    for (size_t r = 0; r < R; ++r)
        if (counts_on_[r]) { issued_rank_ = (int32_t)r; break; }
    /* The file appears at boot, so a path that cannot be written fails here, not 30 s in. */
    return write(false);
}

XsSnapshot ExpertStats::snapshot() {
    XsSnapshot s = info_;
    const size_t R = (size_t)map_.n_rank, L = (size_t)map_.n_layer;
    s.tp = map_.n_rank;
    s.n_layer = map_.n_layer;
    s.n_trunk = map_.n_trunk;
    s.n_expert = 0;
    for (int32_t n : map_.n_expert) s.n_expert = std::max(s.n_expert, n);
    s.layer_experts = map_.n_expert;
    s.how = map_.how;
    s.top_k.assign(L, 0);
    for (int k = 0; k < kXsKinds; ++k) {
        s.histograms[k].assign(L, 0);
        s.passes[k].assign(L, 0);
        s.obs_tokens[k].assign(L, 0);
        s.selections[k].assign(L, 0);
        s.counts[k].assign(L, {});
        for (size_t l = 0; l < L; ++l) s.counts[k][l].assign((size_t)map_.n_expert[l], 0);
    }
    s.rank_credited.assign(R, 0);
    s.rank_unmapped.assign(R, 0);
    s.rank_unclassified.assign(R, 0);
    s.rank_dropped.assign(R, 0);
    for (size_t r = 0; r < R; ++r)
        if (counts_on_[r]) tallies_[r]->snapshot_into(&s, token_rank_, (int32_t)r == issued_rank_);
    s.uptime_s = (double)(rad_mono_ns() - t0_ns_) * 1e-9;
    s.written_unix = (int64_t)std::time(nullptr);
    return s;
}

int ExpertStats::write(bool final_) {
    std::lock_guard<std::mutex> lk(file_mu_);
    XsSnapshot s = snapshot();
    s.final_ = final_;
    const int st = xs_write_atomic(path_, xs_json(s));
    if (final_ && st >= 0) {
        uint64_t hist = 0, sel = 0;
        for (uint64_t v : s.rank_credited) hist += v;
        for (int k = 0; k < kXsKinds; ++k)
            for (uint64_t v : s.selections[k]) sel += v;
        RAD_INFO("expert stats: %llu layer histograms, %llu selections (prefill %llu, decode %llu, "
                 "mixed %llu tokens; %llu draft passes) written to %s",
                 (unsigned long long)hist, (unsigned long long)sel,
                 (unsigned long long)s.tokens[kXsPrefill], (unsigned long long)s.tokens[kXsDecode],
                 (unsigned long long)s.tokens[kXsMixed], (unsigned long long)s.draft_passes,
                 path_.c_str());
    }
    return st;
}

void ExpertStats::start() {
    if (writer_.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(wmu_);
        wstop_ = false;
    }
    writer_ = std::thread([this] {
        const auto every = std::chrono::duration<double>(interval_s_);
        for (;;) {
            std::unique_lock<std::mutex> lk(wmu_);
            if (wcv_.wait_for(lk, every, [this] { return wstop_; })) break;
            lk.unlock();
            (void)write(false);
        }
    });
}

void ExpertStats::stop() {
    {
        std::lock_guard<std::mutex> lk(wmu_);
        wstop_ = true;
    }
    wcv_.notify_all();
    if (writer_.joinable()) writer_.join();
}

ExpertStats::~ExpertStats() { stop(); }

/* ------------------------------------------------------------------ --expert-profile */

int xs_profile_load(const std::string& path, std::vector<RadFileProfile>* out, int32_t* n_trunk) {
    if (!out) return RAD_E_INVAL;
    out->clear();
    if (n_trunk) *n_trunk = -1;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) {
        RAD_ERR("--expert-profile: cannot read %s: %s", path.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    std::string text;
    char buf[1 << 16];
    while (const size_t n = std::fread(buf, 1, sizeof buf, f)) text.append(buf, n);
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) {
        RAD_ERR("--expert-profile: reading %s failed", path.c_str());
        return RAD_E_IO;
    }
    Json j;
    try { j = Json::parse(text); }
    catch (const Json::exception& e) {
        RAD_ERR("--expert-profile: %s is not JSON: %s", path.c_str(), e.what());
        return RAD_E_INVAL;
    }
    if (!j.is_object() || j.value("format", "") != "radiance-expert-stats/1" ||
        !j.contains("counts") || !j["counts"].contains("decode") ||
        !j["counts"]["decode"].is_array()) {
        RAD_ERR("--expert-profile: %s is not a radiance-expert-stats/1 file with decode counts",
                path.c_str());
        return RAD_E_INVAL;
    }
    if (n_trunk && j.contains("n_trunk_layers") && j["n_trunk_layers"].is_number_integer())
        *n_trunk = j["n_trunk_layers"].get<int32_t>();
    const auto& rows = j["counts"]["decode"];
    int64_t kept = 0;
    for (size_t l = 0; l < rows.size(); ++l) {
        if (!rows[l].is_array()) continue;
        uint64_t sum = 0;
        for (const auto& c : rows[l])
            if (c.is_number_unsigned() || c.is_number_integer())
                sum += c.get<uint64_t>();
        if (sum == 0) continue;
        for (size_t e = 0; e < rows[l].size(); ++e) {
            const auto& c = rows[l][e];
            if (!(c.is_number_unsigned() || c.is_number_integer())) continue;
            const uint64_t n = c.get<uint64_t>();
            if (n == 0) continue;
            RadFileProfile p{};
            p.layer = (int32_t)l;
            p.expert = (int32_t)e;
            p.share = (float)((double)n / (double)sum);
            out->push_back(p);
            ++kept;
        }
    }
    if (kept == 0)
        RAD_WARN("--expert-profile: %s has no decode selections; the planner ranks from the "
                 "container's profile", path.c_str());
    else
        RAD_INFO("--expert-profile: %lld decode-hot entries over %zu layers from %s",
                 (long long)kept, rows.size(), path.c_str());
    return RAD_OK;
}

}  /* namespace rad */
