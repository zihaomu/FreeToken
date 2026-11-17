/* lora.cpp -- see lora.h. Pure host code: nothing here reaches a device. */
#include "lora.h"

#include "../format/safetensors.h"
#include "../format/share.h"
#include "rad_plugin.h"

#include "util/json.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <functional>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace rad {
namespace lora {

namespace {

using ojson = Json;

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

int fail(std::string* err, int code, const std::string& m) {
    if (err) *err = m;
    return code;
}

std::vector<std::string> split_dots(const std::string& s) {
    std::vector<std::string> v;
    size_t at = 0;
    while (true) {
        const size_t d = s.find('.', at);
        v.push_back(s.substr(at, d == std::string::npos ? std::string::npos : d - at));
        if (d == std::string::npos) break;
        at = d + 1;
    }
    return v;
}

/* `<module>.weight` -> module and its leaf; "" when the source is not a `.weight` tensor. */
std::string module_of(const std::string& src, std::string* leaf) {
    if (!src.ends_with(".weight")) return "";
    std::string m = src.substr(0, src.size() - 7);
    if (m.empty()) return "";
    const size_t d = m.rfind('.');
    if (leaf) *leaf = d == std::string::npos ? m : m.substr(d + 1);
    return m;
}

/* ---------------------------------------------------------------- scope rule (lora.h) */
bool excluded_source(const std::string& src) {
    static const char* const kTower[] = { "visual", "vision_tower", "vision_model", "audio_tower",
                                          "audio_model", "embed_vision", "embed_audio",
                                          "multi_modal_projector", "mm_projector" };
    const std::vector<std::string> c = split_dots(src);
    if (c.empty()) return true;
    if (c[0] == "mtp" || c[0] == "draft") return true;
    for (const std::string& x : c)
        for (const char* t : kTower)
            if (x == t) return true;
    std::string leaf;
    if (!module_of(src, &leaf).empty() && (leaf == "embed_tokens" || leaf == "lm_head")) return true;
    return false;
}

/* A generator that declared adapter key names (rad_lora_root): its scope is those roots, every
 * matrix under one of them, and none of the token model's rules below apply. */
bool gen_scope(const Program& prog) { return !prog.lora_roots.empty(); }

const char* excluded_weight(const Program& prog, const WeightInfo& w) {
    if (w.decl.access == RAD_ACCESS_VOCAB) return "a vocabulary weight (embedding / lm_head)";
    if (gen_scope(prog)) return w.decl.rank == 2 ? nullptr : "not a matrix";
    if (w.decl.access == RAD_ACCESS_PER_REQUEST) return "an encoder tower weight (per-request access)";
    if (prog.meta.n_layers > 0 && w.decl.group.layer >= prog.meta.n_layers)
        return "a drafter / MTP layer (layer group at or above meta.n_layers)";
    static const char* const kScopes[] = { "draft.", "vision.", "audio.", "mtp." };
    for (const char* s : kScopes)
        if (w.name.starts_with(s)) return "a non-trunk builder scope";
    return nullptr;
}

/* The name-map entries for a declared logical weight, found walking back from the end. CONCAT
 * extension pieces are declared one after another (map_concat_shards), so the run of equal
 * entries around the hit is the whole map, in call order. */
void scan_maps(const Program& prog, const std::string& declared,
               std::vector<const RadNameMap*>* out) {
    out->clear();
    const auto& nm = prog.name_map;
    for (size_t i = nm.size(); i-- > 0;) {
        if (!nm[i].declared || declared != nm[i].declared) continue;
        size_t lo = i, hi = i;
        while (lo > 0 && nm[lo - 1].declared && declared == nm[lo - 1].declared) --lo;
        while (hi + 1 < nm.size() && nm[hi + 1].declared && declared == nm[hi + 1].declared) ++hi;
        for (size_t k = lo; k <= hi; ++k) out->push_back(&nm[k]);
        return;
    }
}

/* Whole-program index, for the callers that walk every op. Same answer as scan_maps for every
 * map the builder accepts (pieces of one map are all collected, in order). */
struct MapIndex {
    std::unordered_map<std::string, std::vector<const RadNameMap*>> by;
    explicit MapIndex(const Program& prog) {
        /* An entry repeated whole (a plugin's encoding probe maps the weight it then declares) is
         * one entry. */
        auto same = [](const RadNameMap& a, const RadNameMap& b) {
            if (a.mode != b.mode || a.n_src != b.n_src || a.concat_dim != b.concat_dim) return false;
            for (int k = 0; k < a.n_src && k < 8; ++k)
                if (std::strcmp(a.src[k] ? a.src[k] : "", b.src[k] ? b.src[k] : "") != 0 ||
                    a.src_index[k] != b.src_index[k] || a.src_rows[k] != b.src_rows[k])
                    return false;
            return true;
        };
        for (const RadNameMap& m : prog.name_map) {
            if (!m.declared) continue;
            std::vector<const RadNameMap*>& v = by[m.declared];
            bool dup = false;
            for (const RadNameMap* x : v) dup = dup || same(*x, m);
            if (!dup) v.push_back(&m);
        }
    }
    const std::vector<const RadNameMap*>* find(const std::string& d) const {
        auto it = by.find(d);
        return it == by.end() ? nullptr : &it->second;
    }
};

/* Is checkpoint tensor `src` an adapter target? A token model: its leaf is in the target list and
 * it is in the trunk. A generator: it is under a declared root, and its leaf is in the list or the
 * list is "*". */
bool source_targeted(const Program& prog, const TargetSet& targets, const std::string& src) {
    std::string leaf;
    if (module_of(src, &leaf).empty()) return false;
    if (!gen_scope(prog)) return targets.count(leaf) && !excluded_source(src);
    if (!targets.count("*") && !targets.count(leaf)) return false;
    for (const LoraRoot& r : prog.lora_roots)
        if (src.compare(0, r.native.size(), r.native) == 0) return true;
    return false;
}

/* weight_targeted(), over the maps already found. */
bool targeted_core(const Program& prog, const WeightInfo& w,
                   const std::vector<const RadNameMap*>& maps, const TargetSet& targets) {
    if (targets.empty() || maps.empty()) return false;
    if (excluded_weight(prog, w)) return false;
    for (const RadNameMap* m : maps)
        for (int k = 0; k < m->n_src && k < 8; ++k)
            if (m->src[k] && source_targeted(prog, targets, m->src[k])) return true;
    return false;
}

bool targeted_idx(const Program& prog, const MapIndex& mi, rad_weight h, const TargetSet& t) {
    if (h == 0 || h >= prog.weights.size()) return false;
    const WeightInfo& w = prog.weights[h];
    const auto* maps = mi.find(w.source);
    return maps ? targeted_core(prog, w, *maps, t) : false;
}

/* An op the builder recorded and could not resolve is never issued: the plugin was handed a null
 * handle and declared its fallback. A fixture's op with no bands at all counts as live. */
bool op_live(const OpInfo& op) {
    if (op.bands.empty()) return true;
    for (const Band& b : op.bands)
        for (int d = 0; d < RAD_N_DOMAINS; ++d)
            if (b.dom[d]) return true;
    return false;
}

/* The schema's operand position of `name`, or the built-in layout of the four capable ops when
 * there is no schema on the OpInfo (a fixture). */
int operand_pos(const OpInfo& op, const char* name) {
    if (op.schema && op.schema->operands) {
        for (int i = 0; i < op.schema->n_operands; ++i)
            if (op.schema->operands[i].name && !std::strcmp(op.schema->operands[i].name, name))
                return i;
        return -1;
    }
    /* docs/OPS.md GEMM table. */
    if (!std::strcmp(name, "b")) {
        if (op.op == "gemm_nt" || op.op == "gemm_nt_bias") return 1;
        if (op.op == "gemm_nt_q" || op.op == "gemm_nt_q_bias") return 2;
    }
    return -1;
}

rad_weight b_weight(const OpInfo& op) {
    const int pos = operand_pos(op, "b");
    if (pos >= 0 && op.weight_opd.size() == op.weights.size()) {
        for (size_t i = 0; i < op.weights.size(); ++i)
            if (op.weight_opd[i] == pos) return op.weights[i];
        return 0;
    }
    return op.weights.empty() ? 0 : op.weights[0];
}

bool read_file(const std::string& path, std::string* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    *out = ss.str();
    return true;
}

std::string shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

float elem_f32(const StTensor& t, int64_t i) {
    switch (t.dtype) {
        case RAD_BF16: { uint16_t v; std::memcpy(&v, t.data + 2 * i, 2); return rad_bf16_to_f32(v); }
        case RAD_F16:  { uint16_t v; std::memcpy(&v, t.data + 2 * i, 2); return rad_f16_to_f32(v); }
        default:       { float v; std::memcpy(&v, t.data + 4 * i, 4); return v; }
    }
}
/* bf16 RNE of one element; a bf16 source is copied as is. */
uint16_t elem_bf16(const StTensor& t, int64_t i) {
    if (t.dtype == RAD_BF16) { uint16_t v; std::memcpy(&v, t.data + 2 * i, 2); return v; }
    return rad_f32_to_bf16(elem_f32(t, i));
}

/* ---------------------------------------------------------------- segments */
/* Runs of whole parts: source s spans parts [g0, g1); its rows on this rank are part g's
 * [lo_g, hi_g) shifted by the container rows of its earlier parts in the run. */
int segments_runs(const Site& s, const std::vector<std::pair<int, int>>& runs,
                  const std::vector<int64_t>& T, std::vector<SitePart>* out, std::string* err) {
    out->clear();
    for (size_t si = 0; si < s.sources.size(); ++si) {
        const SiteSource& src = s.sources[si];
        if (!src.targeted) continue;
        const int g0 = runs[si].first, g1 = runs[si].second;
        int64_t cum = 0;
        for (int g = g0; g < g1; ++g) {
            if (g > g0) {
                if (T[(size_t)g - 1] <= 0)
                    return fail(err, RAD_E_SHAPE,
                                fmt("weight '%s': module %s spans row parts %d..%d and the rows of "
                                    "part %d in the container are not recorded",
                                    s.weight.c_str(), src.module.c_str(), g0, g1 - 1, g - 1));
                cum += T[(size_t)g - 1];
            }
            SitePart p;
            p.module = src.module;
            p.src_lo = cum + s.part_lo[(size_t)g];
            p.src_hi = cum + s.part_hi[(size_t)g];
            p.y_col  = s.part_y[(size_t)g];
            p.t_off  = (int64_t)src.slot * s.max_rank;
            p.row_part = g;
            if (p.src_hi > p.src_lo) out->push_back(p);
        }
    }
    return RAD_OK;
}

/* Several sources inside ONE part (CONCAT at n_row_parts <= 1): source s is part-local rows
 * [off_s, off_s + rows_s); this rank holds [lo, hi) of the part. */
int segments_within(const Site& s, const std::vector<int64_t>& rows, const std::vector<bool>& want,
                    std::vector<SitePart>* out, std::string* err) {
    out->clear();
    const int64_t lo = s.part_lo[0], hi = s.part_hi[0];
    int64_t off = 0;
    for (size_t si = 0; si < s.sources.size(); ++si) {
        const SiteSource& src = s.sources[si];
        if (src.targeted && want[si]) {
            if (off < 0)
                return fail(err, RAD_E_SHAPE,
                            fmt("weight '%s': the rows of a source before %s are not known, so "
                                "where it starts in the fused weight is not either",
                                s.weight.c_str(), src.module.c_str()));
            if (rows[si] < 0)
                return fail(err, RAD_E_SHAPE, fmt("weight '%s': the rows of %s are not known",
                                                  s.weight.c_str(), src.module.c_str()));
            const int64_t a = std::max(lo, off), b = std::min(hi, off + rows[si]);
            if (a < b) {
                SitePart p;
                p.module = src.module;
                p.src_lo = a - off;
                p.src_hi = b - off;
                p.y_col  = s.part_y[0] + (a - lo);
                p.t_off  = (int64_t)src.slot * s.max_rank;
                p.row_part = 0;
                out->push_back(p);
            }
        }
        off = (off < 0 || rows[si] < 0) ? -1 : off + rows[si];
    }
    return RAD_OK;
}

/* Every way to cut P parts into n contiguous non-empty runs. */
void compositions(int P, int n, std::vector<std::vector<std::pair<int, int>>>* out) {
    std::vector<std::pair<int, int>> cur;
    std::function<void(int, int)> rec = [&](int start, int left) {
        if (left == 1) {
            cur.push_back({ start, P });
            out->push_back(cur);
            cur.pop_back();
            return;
        }
        for (int end = start + 1; end <= P - (left - 1); ++end) {
            cur.push_back({ start, end });
            rec(end, left - 1);
            cur.pop_back();
        }
    };
    if (n >= 1 && n <= P) rec(0, n);
}

bool same_layout(const Site& a, const Site& b) {
    if (a.weight != b.weight || a.shard != b.shard || a.K_r != b.K_r || a.k_lo != b.k_lo ||
        a.K_full != b.K_full || a.N_r != b.N_r || a.R != b.R || a.n_modules != b.n_modules ||
        a.parts_resolved != b.parts_resolved || a.part_total != b.part_total ||
        a.part_lo != b.part_lo || a.part_hi != b.part_hi || a.part_y != b.part_y ||
        a.parts.size() != b.parts.size() || a.sources.size() != b.sources.size())
        return false;
    for (size_t i = 0; i < a.parts.size(); ++i) {
        const SitePart &x = a.parts[i], &y = b.parts[i];
        if (x.module != y.module || x.src_lo != y.src_lo || x.src_hi != y.src_hi ||
            x.y_col != y.y_col || x.t_off != y.t_off)
            return false;
    }
    for (size_t i = 0; i < a.sources.size(); ++i)
        if (a.sources[i].module != b.sources[i].module || a.sources[i].slot != b.sources[i].slot ||
            a.sources[i].part_lo != b.sources[i].part_lo ||
            a.sources[i].part_hi != b.sources[i].part_hi)
            return false;
    return true;
}

/* One site's geometry from the weight's recorded facts. */
int make_site(const Program& prog, const MapIndex& mi, const TargetSet& targets, int max_rank,
              uint32_t op_index, rad_weight wh, Site* out, std::string* err) {
    const OpInfo& op = prog.ops[op_index];
    const WeightInfo& W = prog.weights[wh];
    Site s;
    s.gemm = op_index;
    s.op = op.op;
    s.weight = W.name;
    s.shard = W.decl.shard;
    s.max_rank = max_rank;

    if (W.decl.rank != 2)
        return fail(err, RAD_E_SHAPE, fmt("weight '%s' is rank %u; a LoRA site needs a matrix",
                                          W.name.c_str(), W.decl.rank));
    s.N_r = W.decl.shape[0];
    s.K_r = W.decl.shape[1];
    long long pn = 0, pk = 0;
    if ((op.base.get_i("N", &pn) && pn != s.N_r) || (op.base.get_i("K", &pk) && pk != s.K_r))
        return fail(err, RAD_E_SHAPE,
                    fmt("op %u (%s) over '%s' declares N=%lld K=%lld and the weight is [%lld, %lld]",
                        op_index, op.op.c_str(), W.name.c_str(), pn, pk, (long long)s.N_r,
                        (long long)s.K_r));

    const int world = prog.ctx.world_size > 0 ? prog.ctx.world_size : 1;
    const int index = prog.ctx.rank;
    const bool whole = world <= 1 || W.decl.shard == RAD_SHARD_NONE;

    /* ---- K: share.cpp's column cut. */
    if (whole || W.decl.shard == RAD_SHARD_ROW) {
        s.k_lo = 0;
        s.K_full = s.K_r;
    } else {   /* COL */
        const bool span = W.shard_hi > 0 || W.n_part_spans > 0;
        if (span) {
            s.k_lo = W.shard_lo;
            if (W.shard_hi - W.shard_lo != s.K_r)
                return fail(err, RAD_E_SHAPE, fmt("weight '%s': column span [%lld, %lld) is not its "
                                                  "%lld declared columns", W.name.c_str(),
                                                  (long long)W.shard_lo, (long long)W.shard_hi,
                                                  (long long)s.K_r));
            s.K_full = 0;   /* not recorded on the weight: lora_A's columns say */
        } else {
            s.k_lo = (int64_t)index * s.K_r;
            s.K_full = s.K_r * world;
        }
    }

    /* ---- rows: share.cpp's per-part cut, in each part's own units. */
    const int P = W.decl.n_row_parts > 1 ? W.decl.n_row_parts : 1;
    std::vector<int64_t> rp((size_t)P);
    int64_t sum = 0;
    for (int g = 0; g < P; ++g) {
        rp[(size_t)g] = P > 1 ? W.decl.row_parts[g] : s.N_r;
        sum += rp[(size_t)g];
    }
    if (sum != s.N_r)
        return fail(err, RAD_E_SHAPE, fmt("weight '%s': row parts sum to %lld, declared rows %lld",
                                          W.name.c_str(), (long long)sum, (long long)s.N_r));
    s.part_total.assign((size_t)P, 0);
    s.part_lo.assign((size_t)P, 0);
    s.part_hi.assign((size_t)P, 0);
    s.part_y.assign((size_t)P, 0);
    int64_t y = 0;
    for (int g = 0; g < P; ++g) {
        const size_t G = (size_t)g;
        s.part_y[G] = y;
        y += rp[G];
        if (whole || W.decl.shard == RAD_SHARD_COL) {
            s.part_total[G] = rp[G]; s.part_lo[G] = 0; s.part_hi[G] = rp[G];
        } else if (W.n_part_spans > 0) {
            if (W.n_part_spans != P)
                return fail(err, RAD_E_SHAPE, fmt("weight '%s': %d part spans for %d parts",
                                                  W.name.c_str(), W.n_part_spans, P));
            s.part_total[G] = W.part_total[g]; s.part_lo[G] = W.part_lo[g]; s.part_hi[G] = W.part_hi[g];
        } else if (W.shard_hi > 0) {
            s.part_total[G] = 0;   /* VR / P: the container's rows, not recorded on the weight */
            s.part_lo[G] = W.shard_lo; s.part_hi[G] = W.shard_hi;
        } else {
            s.part_total[G] = rp[G] * world;
            s.part_lo[G] = (int64_t)index * rp[G];
            s.part_hi[G] = s.part_lo[G] + rp[G];
        }
        if (s.part_hi[G] - s.part_lo[G] != rp[G])
            return fail(err, RAD_E_SHAPE, fmt("weight '%s': part %d takes rows [%lld, %lld) and "
                                              "declares %lld", W.name.c_str(), g,
                                              (long long)s.part_lo[G], (long long)s.part_hi[G],
                                              (long long)rp[G]));
    }
    int64_t N_full = 0;
    for (int64_t t : s.part_total) N_full = (t > 0 && N_full >= 0) ? N_full + t : -1;

    /* ---- the loader's own cut must agree, or this is not a site: weight_share, which puts the
     * base weight's bytes on this rank, must name for a plain [N_full, K_full] entry the
     * rectangles this geometry names. */
    if (N_full > 0 && s.K_full > 0) {
        WeightInfo v = W;
        v.n_sel = 1;
        v.sel[0] = 0;
        v.sel_rows[0] = s.N_r;
        v.sel_cols[0] = s.K_r;
        const RadEncoding enc = rad_enc_plain(RAD_BF16);
        const int64_t shape[RAD_MAX_RANK] = { N_full, s.K_full };
        PlaneShare ps[RAD_ENC_MAX_PLANES];
        std::string why;
        if (weight_share(v, enc, 2, shape, world, index, ps, &why) != RAD_OK)
            return fail(err, RAD_E_SHAPE, fmt("weight '%s': weight_share refuses the LoRA view: %s",
                                              W.name.c_str(), why.c_str()));
        std::vector<PlaneShare::Rect> want;
        int64_t base = 0;
        for (int g = 0; g < P; ++g) {
            PlaneShare::Rect r;
            r.row0 = base + s.part_lo[(size_t)g];
            r.rows = s.part_hi[(size_t)g] - s.part_lo[(size_t)g];
            r.byte0 = s.k_lo * 2;
            r.bytes = s.K_r * 2;
            base += s.part_total[(size_t)g];
            if (whole || W.decl.shard == RAD_SHARD_COL) {   /* one rectangle, every row */
                if (g == 0) { r.row0 = 0; r.rows = s.N_r; want.push_back(r); }
                continue;
            }
            want.push_back(r);
        }
        bool same = ps[0].rects.size() == want.size();
        for (size_t i = 0; same && i < want.size(); ++i)
            same = ps[0].rects[i].row0 == want[i].row0 && ps[0].rects[i].rows == want[i].rows &&
                   ps[0].rects[i].byte0 == want[i].byte0 && ps[0].rects[i].bytes == want[i].bytes;
        if (!same)
            return fail(err, RAD_E_SHAPE, fmt("weight '%s': the LoRA geometry and weight_share "
                                              "disagree about this rank's rows/columns",
                                              W.name.c_str()));
    }

    /* ---- sources, in concatenation order. */
    const auto* maps = mi.find(W.source);
    if (!maps || maps->empty())
        return fail(err, RAD_E_NOTFOUND, fmt("weight '%s' has no name-map entry", W.name.c_str()));
    std::vector<std::string> srcs;
    /* A row piece of a checkpoint tensor (RAD_MAP_ROWS) is a source of its own, named
     * `<module>@<first row>+<rows>`: the adapter's B rows it takes are that range of the
     * module's. */
    std::vector<std::string> piece;
    const bool alternatives = maps->size() == 1 && (*maps)[0]->mode == RAD_MAP_COPY &&
                              (*maps)[0]->n_src > 1;
    for (const RadNameMap* m : *maps)
        for (int k = 0; k < m->n_src && k < 8; ++k) {
            const bool rows = m->mode == RAD_MAP_ROWS;
            if (!rows && m->src_index[k] >= 0)
                return fail(err, RAD_E_UNSUPPORTED,
                            fmt("weight '%s' is slice %d of the stacked checkpoint tensor %s; a "
                                "LoRA on a stacked (expert) tensor is not supported",
                                W.name.c_str(), m->src_index[k], m->src[k]));
            srcs.push_back(m->src[k] ? m->src[k] : "");
            piece.push_back(rows ? fmt("@%d+%lld", m->src_index[k], (long long)m->src_rows[k]) : "");
        }
    if (alternatives) {
        /* COPY with alternatives: the first one the container holds is the weight, and which one
         * that is is not recorded. Usable only when exactly one alternative is targeted. */
        std::vector<std::string> hit;
        for (const std::string& x : srcs)
            if (source_targeted(prog, targets, x)) hit.push_back(x);
        if (hit.size() != 1)
            return fail(err, RAD_E_UNSUPPORTED,
                        fmt("weight '%s' is one of %zu alternative checkpoint tensors and %zu of "
                            "them are targeted; which one the container holds is not recorded",
                            W.name.c_str(), srcs.size(), hit.size()));
        srcs = hit;
        piece.assign(srcs.size(), "");
    }
    int slot = 0;
    for (size_t si = 0; si < srcs.size(); ++si) {
        const std::string& x = srcs[si];
        SiteSource ss;
        std::string leaf;
        ss.module = module_of(x, &leaf);
        if (ss.module.empty()) ss.module = x;
        ss.module += piece[si];
        ss.targeted = source_targeted(prog, targets, x);
        if (ss.targeted) ss.slot = slot++;
        s.sources.push_back(ss);
    }
    s.n_modules = slot;
    s.R = (int64_t)slot * max_rank;
    const int n = (int)s.sources.size();

    /* ---- which parts each source covers, when the declaration alone decides it. */
    std::vector<std::pair<int, int>> runs((size_t)n, { -1, -1 });
    bool decided = false;
    if (P == 1 && n == 1) {
        runs[0] = { 0, 1 };
        decided = true;
    } else if (P >= 2 && n == P) {
        for (int i = 0; i < n; ++i) runs[(size_t)i] = { i, i + 1 };
        decided = true;
    } else if (P >= 2 && n == 1) {
        runs[0] = { 0, P };
        decided = true;
        for (int g = 0; g + 1 < P; ++g) if (s.part_total[(size_t)g] <= 0) decided = false;
    } else if (P >= 2 && n > P) {
        return fail(err, RAD_E_UNSUPPORTED,
                    fmt("weight '%s' concatenates %d checkpoint tensors over %d row parts; a "
                        "source cannot be a whole number of parts", W.name.c_str(), n, P));
    }
    /* (P == 1 && n > 1) and (1 < n < P) wait for the adapter's B rows. */
    if (decided) {
        for (int i = 0; i < n; ++i) {
            s.sources[(size_t)i].part_lo = runs[(size_t)i].first;
            s.sources[(size_t)i].part_hi = runs[(size_t)i].second;
        }
        int st = segments_runs(s, runs, s.part_total, &s.parts, err);
        if (st < 0) return st;
        s.parts_resolved = true;
    } else {
        s.parts_resolved = false;
    }
    *out = std::move(s);
    return RAD_OK;
}

}  /* namespace */

/* ================================================================== predicates */
bool weight_targeted(const Program& prog, rad_weight w, const TargetSet& targets) {
    if (targets.empty() || w == 0 || w >= prog.weights.size()) return false;
    /* A generator's fused GEMMs are kept: its scope is the plain GEMMs (build_sites). */
    if (gen_scope(prog)) return false;
    const WeightInfo& wi = prog.weights[w];
    if (excluded_weight(prog, wi)) return false;
    std::vector<const RadNameMap*> maps;
    scan_maps(prog, wi.source, &maps);
    return targeted_core(prog, wi, maps, targets);
}

bool op_is_site(const Program& prog, const OpInfo& op, rad_weight* b_out, std::string* why) {
    auto no = [&](const std::string& m) { if (why) *why = m; return false; };
    if (b_out) *b_out = 0;
    const bool g = op.op == "gemm_nt", gb = op.op == "gemm_nt_bias", gq = op.op == "gemm_nt_q",
               gqb = op.op == "gemm_nt_q_bias";
    if (!g && !gb && !gq && !gqb) return no("op '" + op.op + "' is not a LoRA-capable GEMM");
    const rad_weight b = b_weight(op);
    if (b == 0 || b >= prog.weights.size()) return no("op '" + op.op + "' names no `b` weight");
    if (b_out) *b_out = b;
    if (gb) {
        const char* act = op.base.get_s("act");
        if (act && *act && std::strcmp(act, "none") != 0)
            return no(fmt("gemm_nt_bias applies act=%s after the product", act));
    }
    if (gq) {
        long long p1 = 0, p2 = 0;
        op.base.get_i("pb1", &p1);
        op.base.get_i("pb2", &p2);
        if (p1 > 0 || p2 > 0)
            return no(fmt("gemm_nt_q over rotated parts (pb1=%lld pb2=%lld): its input is the "
                          "rotated activation", p1, p2));
    }
    if (gq || gqb) {
        /* At w4a8, w8a8 and w2a8 `a` is quant_act_i8's codes in the int8 WMMA's step order (the
         * middle dwords of every 16-byte step exchanged, docs/OPS.md), which lora_shrink would
         * read as the activation's columns. */
        const char* dt = op.base.get_s("dtype");
        if (dt && (!std::strcmp(dt, "w4a8") || !std::strcmp(dt, "w8a8") || !std::strcmp(dt, "w2a8")))
            return no(fmt("%s at %s reads quant_act_i8's codes in the int8 WMMA's step order, not "
                          "in the activation's column order", op.op.c_str(), dt));
    }
    const WeightInfo& w = prog.weights[b];
    if (gq || gqb) {
        long long k = 0;
        if (w.decl.rank == 2 && op.base.get_i("K", &k) && k != w.decl.shape[1])
            return no(fmt("%s reads K=%lld against a weight of %lld columns (a widened, rotated "
                          "input)", op.op.c_str(), k, (long long)w.decl.shape[1]));
    }
    if (w.enc_known && w.enc.transform[0])
        return no(fmt("weight '%s' is stored in the '%.*s' transform basis, so its GEMM is fed a "
                      "transformed activation", w.name.c_str(), (int)RAD_ENC_STR, w.enc.transform));
    return true;
}

int find_unsupported_uses(const Program& prog, const TargetSet& targets,
                          std::vector<UnsupportedUse>* out) {
    if (out) out->clear();
    if (targets.empty() || gen_scope(prog)) return 0;
    const MapIndex mi(prog);
    int n = 0;
    for (size_t i = 1; i < prog.ops.size(); ++i) {
        const OpInfo& op = prog.ops[i];
        if (op.weights.empty() || !op_live(op)) continue;
        rad_weight b = 0;
        std::string why;
        const bool site = op_is_site(prog, op, &b, &why);
        for (rad_weight h : op.weights) {
            if (!targeted_idx(prog, mi, h, targets)) continue;
            /* Every view of a targeted weight read by a site over the same logical weight is the
             * site's own (codes, scale, ref). Anything else reading it is not a site. */
            if (site && b && prog.weights[b].source == prog.weights[h].source) continue;
            UnsupportedUse u;
            u.weight = prog.weights[h].name;
            u.op = op.op;
            u.op_index = (uint32_t)i;
            u.why = site ? "the targeted weight is not this GEMM's `b` operand" : why;
            if (const auto* maps = mi.find(prog.weights[h].source))
                for (const RadNameMap* m : *maps)
                    for (int k = 0; k < m->n_src && k < 8; ++k) {
                        std::string leaf;
                        const std::string src = m->src[k] ? m->src[k] : "";
                        const std::string mod = module_of(src, &leaf);
                        if (!mod.empty() && targets.count(leaf) && !excluded_source(src))
                            u.module += (u.module.empty() ? "" : ",") + mod;
                    }
            if (out) out->push_back(u);
            ++n;
            break;   /* one report per op */
        }
    }
    return n;
}

int build_sites(const Program& prog, const TargetSet& targets, int max_rank,
                std::vector<Site>* out, std::string* err, std::vector<std::string>* pool_weights) {
    if (!out) return fail(err, RAD_E_INVAL, "build_sites: null output");
    out->clear();
    if (pool_weights) pool_weights->clear();
    if (max_rank <= 0) return fail(err, RAD_E_INVAL, fmt("max LoRA rank %d", max_rank));
    if (targets.empty()) return RAD_OK;
    const MapIndex mi(prog);
    /* A generator's weight carries an adapter only when every op reading it is a site over it; one
     * read elsewhere (a fused GEMM, a non-GEMM op) stays the base everywhere. */
    std::set<std::string> other_reader;
    if (gen_scope(prog))
        for (size_t i = 1; i < prog.ops.size(); ++i) {
            const OpInfo& op = prog.ops[i];
            if (op.weights.empty() || !op_live(op)) continue;
            rad_weight b = 0;
            const bool site = op_is_site(prog, op, &b, nullptr);
            for (rad_weight h : op.weights)
                if (h && h < prog.weights.size() &&
                    (!site || !b || prog.weights[b].source != prog.weights[h].source))
                    other_reader.insert(prog.weights[h].source);
        }
    std::map<rad_weight, int32_t> pool_of;
    std::vector<size_t> first_site;
    for (size_t i = 1; i < prog.ops.size(); ++i) {
        rad_weight b = 0;
        if (!op_live(prog.ops[i]) || !op_is_site(prog, prog.ops[i], &b, nullptr)) continue;
        if (!targeted_idx(prog, mi, b, targets)) continue;
        if (other_reader.count(prog.weights[b].source)) continue;
        if (gen_scope(prog)) {
            /* Checkpoint tensors copied, joined, transposed or cut into row pieces carry a
             * generator's adapter; a folded or derived one (weight norm, EMA) stays the base. */
            bool whole = true;
            if (const auto* maps = mi.find(prog.weights[b].source))
                for (const RadNameMap* m : *maps)
                    whole = whole && (m->mode == RAD_MAP_COPY || m->mode == RAD_MAP_CONCAT ||
                                      m->mode == RAD_MAP_TRANSPOSE || m->mode == RAD_MAP_ROWS);
            if (!whole) continue;
        }
        Site s;
        const int st = make_site(prog, mi, targets, max_rank, (uint32_t)i, b, &s, err);
        if (st < 0 && gen_scope(prog)) {
            /* A generator's every plain GEMM is a candidate; one whose geometry cannot be told
             * stays the base, as a fused one does. */
            RAD_WARN("lora: '%s' carries no adapter: %s", prog.weights[b].name.c_str(),
                     err ? err->c_str() : "");
            if (err) err->clear();
            continue;
        }
        if (st < 0) return st;
        auto it = pool_of.find(b);
        if (it == pool_of.end()) {
            s.pool = (int32_t)first_site.size();
            pool_of.emplace(b, s.pool);
            first_site.push_back(out->size());
            if (pool_weights) pool_weights->push_back(s.weight);
        } else {
            s.pool = it->second;
            const Site& f = (*out)[first_site[(size_t)s.pool]];
            if (!same_layout(f, s))
                return fail(err, RAD_E_STATE,
                            fmt("weight '%s': op %u (%s) and op %u (%s) read it with different "
                                "LoRA geometry; one pool per weight cannot serve both",
                                s.weight.c_str(), f.gemm, f.op.c_str(), s.gemm, s.op.c_str()));
        }
        out->push_back(std::move(s));
    }
    return RAD_OK;
}

std::string explain_module(const Program& prog, const TargetSet& targets, const std::string& module) {
    const std::string want[3] = {
        module,
        module.starts_with("model.") && !module.starts_with("model.language_model.")
            ? "model.language_model." + module.substr(6) : "",
        module.starts_with("model.language_model.") ? "model." + module.substr(21) : "" };
    std::string first;
    bool any = false;
    for (const RadNameMap& m : prog.name_map) {
        std::string src;
        for (int k = 0; k < m.n_src && k < 8 && src.empty(); ++k)
            for (const std::string& c : want)
                if (!c.empty() && m.src[k] && c + ".weight" == m.src[k]) src = m.src[k];
        if (src.empty() || !m.declared) continue;
        any = true;
        std::string why;
        std::string leaf;
        module_of(src, &leaf);
        rad_weight w = 0;
        for (size_t h = 1; h < prog.weights.size() && !w; ++h)
            if (prog.weights[h].name == m.declared && prog.weights[h].source == m.declared)
                w = (rad_weight)h;
        if (excluded_source(src)) {
            why = src + " is in a non-trunk scope (drafter / MTP / tower / vocab)";
        } else if (!targets.count(leaf)) {
            why = "its leaf '" + leaf + "' is not in the target module list";
        } else if (!w) {
            why = std::string("its declared weight '") + m.declared + "' is not on this rank";
        } else if (const char* ex = excluded_weight(prog, prog.weights[w])) {
            why = std::string("'") + m.declared + "' is " + ex;
        } else {
            std::string reads;
            bool site = false;
            for (size_t i = 1; i < prog.ops.size(); ++i) {
                const OpInfo& op = prog.ops[i];
                if (!op_live(op)) continue;
                bool uses = false;
                for (rad_weight x : op.weights)
                    if (x && x < prog.weights.size() && prog.weights[x].source == m.declared) uses = true;
                if (!uses) continue;
                rad_weight b = 0;
                std::string no;
                if (op_is_site(prog, op, &b, &no) && b == w) { site = true; continue; }
                reads += fmt("%sop %zu %s (%s)", reads.empty() ? "" : "; ", i, op.op.c_str(), no.c_str());
            }
            if (!reads.empty()) why = std::string("'") + m.declared + "' is read by " + reads;
            else if (!site) why = std::string("no op reads '") + m.declared + "'";
        }
        if (why.empty()) return "";
        if (first.empty()) first = why;
    }
    if (!any) return "the base has no tensor " + module + ".weight in any name map on this rank";
    return first;
}

/* ================================================================== the adapter */
namespace {

struct Slot { int32_t pool; int32_t source; };

}  /* namespace */

int load_adapter(const std::string& dir, const std::vector<Site>& sites, int max_rank,
                 AdapterInfo* info, std::vector<PackedSite>* packed, std::string* err) {
    if (!info || !packed) return fail(err, RAD_E_INVAL, "load_adapter: null output");
    *info = AdapterInfo{};
    packed->clear();
    info->dir = dir;
    if (max_rank <= 0) return fail(err, RAD_E_INVAL, fmt("max LoRA rank %d", max_rank));

    /* ---- the config */
    const std::string cfg_path = dir + "/adapter_config.json";
    std::string text;
    if (!read_file(cfg_path, &text))
        return fail(err, RAD_E_IO, cfg_path + ": cannot read (a PEFT adapter directory has one)");
    const ojson cfg = ojson::parse(text, nullptr, false);
    if (cfg.is_discarded() || !cfg.is_object())
        return fail(err, RAD_E_FORMAT, cfg_path + ": not a JSON object");
    auto truthy = [&](const char* k) {
        if (!cfg.contains(k) || cfg[k].is_null()) return false;
        const ojson& v = cfg[k];
        if (v.is_boolean()) return v.get<bool>();
        if (v.is_number()) return v.get<double>() != 0.0;
        if (v.is_string()) return !v.get<std::string>().empty();
        if (v.is_array() || v.is_object()) return !v.empty();
        return true;
    };
    info->peft_type = cfg.contains("peft_type") && cfg["peft_type"].is_string()
                          ? cfg["peft_type"].get<std::string>() : "LORA";
    if (info->peft_type != "LORA")
        return fail(err, RAD_E_UNSUPPORTED, fmt("%s: peft_type is %s; only LORA adapters are served",
                                                cfg_path.c_str(), info->peft_type.c_str()));
    if (truthy("use_dora"))
        return fail(err, RAD_E_UNSUPPORTED, cfg_path + ": use_dora -- DoRA adapters are not served "
                                                       "(merge with PEFT's merge_and_unload)");
    if (truthy("modules_to_save"))
        return fail(err, RAD_E_UNSUPPORTED, cfg_path + ": modules_to_save -- an adapter that "
                                                       "replaces whole modules is not served");
    if (cfg.contains("bias") && cfg["bias"].is_string() && cfg["bias"].get<std::string>() != "none")
        return fail(err, RAD_E_UNSUPPORTED,
                    cfg_path + ": bias = \"" + cfg["bias"].get<std::string>() +
                        "\" -- trained biases are not served");
    if (truthy("lora_bias"))
        return fail(err, RAD_E_UNSUPPORTED, cfg_path + ": lora_bias -- a bias on lora_B is not served");
    if (truthy("fan_in_fan_out"))
        return fail(err, RAD_E_UNSUPPORTED,
                    cfg_path + ": fan_in_fan_out -- Conv1D-style (transposed) bases are not served");
    if (truthy("target_parameters"))
        return fail(err, RAD_E_UNSUPPORTED, cfg_path + ": target_parameters -- parameter (expert) "
                                                       "LoRA is not served");
    if (truthy("trainable_token_indices"))
        return fail(err, RAD_E_UNSUPPORTED, cfg_path + ": trainable_token_indices -- embedding "
                                                       "updates are not served");
    /* PEFT options that change the forward pass itself, not only the weights. */
    if (truthy("layer_replication"))
        return fail(err, RAD_E_UNSUPPORTED, cfg_path + ": layer_replication -- an adapter over "
                                                       "repeated base layers is not served");
    if (truthy("use_qalora"))
        return fail(err, RAD_E_UNSUPPORTED, cfg_path + ": use_qalora -- QA-LoRA's pooled input is "
                                                       "not served");
    if (truthy("alora_invocation_tokens"))
        return fail(err, RAD_E_UNSUPPORTED, cfg_path + ": alora_invocation_tokens -- an activated "
                                                       "LoRA, which applies only after its "
                                                       "invocation, is not served");
    if (!cfg.contains("r") || !cfg["r"].is_number_integer() || cfg["r"].get<int64_t>() <= 0)
        return fail(err, RAD_E_FORMAT, cfg_path + ": no positive integer `r`");
    if (!cfg.contains("lora_alpha") || !cfg["lora_alpha"].is_number())
        return fail(err, RAD_E_FORMAT, cfg_path + ": no numeric `lora_alpha`");
    info->r = cfg["r"].get<int64_t>();
    info->alpha = cfg["lora_alpha"].get<double>();
    info->rslora = cfg.contains("use_rslora") && cfg["use_rslora"].is_boolean() &&
                   cfg["use_rslora"].get<bool>();
    if (cfg.contains("base_model_name_or_path") && cfg["base_model_name_or_path"].is_string())
        info->base_model = cfg["base_model_name_or_path"].get<std::string>();
    if (cfg.contains("target_modules")) {
        const ojson& t = cfg["target_modules"];
        if (t.is_string()) info->target_modules.push_back(t.get<std::string>());
        else if (t.is_array())
            for (const ojson& x : t) if (x.is_string()) info->target_modules.push_back(x.get<std::string>());
    }
    /* PEFT's get_pattern_key: a key matches a module whose name ends in it on a dot boundary,
     * `(.*\.)?(key)$` from the start of the name, and the first matching key in the file's order
     * wins. */
    struct Pat { std::string key; std::regex re; double v; };
    std::vector<Pat> rpat, apat;
    for (const char* which : { "rank_pattern", "alpha_pattern" }) {
        if (!cfg.contains(which) || cfg[which].is_null()) continue;
        if (!cfg[which].is_object())
            return fail(err, RAD_E_FORMAT, cfg_path + ": " + which + " is not an object");
        for (auto it = cfg[which].begin(); it != cfg[which].end(); ++it) {
            if (!it.value().is_number())
                return fail(err, RAD_E_FORMAT, cfg_path + ": " + which + "[" + it.key() +
                                                   "] is not a number");
            Pat p;
            p.key = it.key();
            try {
                p.re = std::regex("(.*\\.)?(" + p.key + ")$", std::regex::ECMAScript);
            } catch (const std::regex_error& e) {
                return fail(err, RAD_E_FORMAT, cfg_path + ": " + which + " key '" + p.key +
                                                   "' is not a regex this engine reads (" + e.what() + ")");
            }
            p.v = it.value().get<double>();
            if (!std::isfinite(p.v))
                return fail(err, RAD_E_FORMAT, cfg_path + ": " + which + "[" + p.key +
                                                   "] is not a finite number");
            if (!std::strcmp(which, "rank_pattern")) {
                if (!it.value().is_number_integer() || p.v <= 0 || p.v > (double)INT32_MAX)
                    return fail(err, RAD_E_FORMAT, cfg_path + ": rank_pattern[" + p.key +
                                                       "] is not a positive integer");
                info->rank_pattern.push_back({ p.key, (int64_t)p.v });
                rpat.push_back(std::move(p));
            } else {
                info->alpha_pattern.push_back({ p.key, p.v });
                apat.push_back(std::move(p));
            }
        }
    }

    /* ---- the tensors */
    const std::string st_path = dir + "/adapter_model.safetensors";
    SafeTensorsFile st;
    if (!std::ifstream(st_path).good() && std::ifstream(dir + "/adapter_model.bin").good())
        return fail(err, RAD_E_UNSUPPORTED, dir + ": adapter_model.bin (pickle) -- save the "
                                                  "adapter as safetensors");
    if (st.open(st_path) != RAD_OK) return fail(err, RAD_E_IO, st_path + ": cannot open as safetensors");

    std::vector<const StTensor*> order;
    for (const StTensor& t : st.tensors()) order.push_back(&t);
    std::sort(order.begin(), order.end(),
              [](const StTensor* a, const StTensor* b) { return a->name < b->name; });

    static const std::regex key_re(R"(^(.*)\.lora_([AB])\.(weight|default\.weight)$)");
    std::map<std::string, AdModule> mods;
    for (const StTensor* t : order) {
        const std::string& k = t->name;
        std::smatch m;
        if (!std::regex_match(k, m, key_re)) {
            std::string why = "is not a linear LoRA A/B tensor";
            if (k.find("lora_embedding_") != std::string::npos) why = "is an embedding LoRA tensor";
            else if (k.ends_with(".bias")) why = "is a trained bias";
            else if (k.find("modules_to_save") != std::string::npos) why = "is a modules_to_save copy";
            else if (k.find("lora_magnitude_vector") != std::string::npos) why = "is a DoRA magnitude";
            return fail(err, RAD_E_UNSUPPORTED, st_path + ": tensor " + k + " " + why +
                                                    "; only linear LoRA A/B pairs are served");
        }
        std::string mod = m[1].str();
        const size_t at = mod.find("base_model.model.");   /* the first one, as rad-export strips it */
        if (at != std::string::npos) mod.erase(at, 17);
        AdModule& am = mods[mod];
        am.name = mod;
        const StTensor*& slot = m[2].str() == "A" ? am.A : am.B;
        if (slot)
            return fail(err, RAD_E_FORMAT, st_path + ": module " + mod + " has two lora_" +
                                               m[2].str() + " tensors");
        slot = t;
    }
    if (mods.empty()) return fail(err, RAD_E_FORMAT, st_path + ": no LoRA tensors");

    for (auto& kv : mods) {
        AdModule& am = kv.second;
        if (!am.A || !am.B)
            return fail(err, RAD_E_FORMAT, fmt("%s: module %s has lora_%s and no lora_%s",
                                               st_path.c_str(), am.name.c_str(), am.A ? "A" : "B",
                                               am.A ? "B" : "A"));
        for (const StTensor* t : { am.A, am.B }) {
            if (t->dtype != RAD_BF16 && t->dtype != RAD_F16 && t->dtype != RAD_F32)
                return fail(err, RAD_E_DTYPE, fmt("module %s: %s is %s; expected BF16, F16 or F32",
                                                  am.name.c_str(), t->name.c_str(),
                                                  t->dtype_str.c_str()));
            if (t->shape.size() != 2 || !t->data)
                return fail(err, RAD_E_SHAPE, fmt("module %s: %s has shape %s; expected a matrix",
                                                  am.name.c_str(), t->name.c_str(),
                                                  shape_str(t->shape).c_str()));
        }
        int64_t rr = info->r;
        double  aa = info->alpha;
        for (const Pat& p : rpat) if (std::regex_match(am.name, p.re)) { rr = (int64_t)p.v; break; }
        for (const Pat& p : apat) if (std::regex_match(am.name, p.re)) { aa = p.v; break; }
        if (am.A->shape[0] != rr || am.B->shape[1] != rr)
            return fail(err, RAD_E_SHAPE,
                        fmt("module %s: expected rank %lld (r / rank_pattern), found lora_A %s and "
                            "lora_B %s", am.name.c_str(), (long long)rr,
                            shape_str(am.A->shape).c_str(), shape_str(am.B->shape).c_str()));
        if (rr > max_rank)
            return fail(err, RAD_E_FULL, fmt("module %s: rank %lld is above --max-lora-rank %d",
                                             am.name.c_str(), (long long)rr, max_rank));
        am.rank = rr;
        am.alpha = aa;
        am.scale = (float)(info->rslora ? aa / std::sqrt((double)rr) : aa / (double)rr);
        if (!std::isfinite(am.scale))
            return fail(err, RAD_E_FORMAT, fmt("module %s: alpha %g over rank %lld is no finite "
                                               "f32 scale", am.name.c_str(), aa, (long long)rr));
        info->max_module_rank = std::max(info->max_module_rank, rr);
        const size_t d = am.name.rfind('.');
        info->leaves.insert(d == std::string::npos ? am.name : am.name.substr(d + 1));
    }

    return pack_modules(sites, max_rank, mods, info, packed, err);
}

int pack_modules(const std::vector<Site>& sites, int max_rank,
                 const std::map<std::string, AdModule>& mods, AdapterInfo* info,
                 std::vector<PackedSite>* packed, std::string* err, bool compact) {
    /* ---- the sites, by pool: the representative of each pool and where its modules sit. */
    int32_t n_pools = 0;
    for (const Site& s : sites) {
        if (s.pool < 0) return fail(err, RAD_E_INVAL, "site '" + s.weight + "' has no pool");
        if (s.max_rank != max_rank)
            return fail(err, RAD_E_INVAL, fmt("sites were built at max rank %d, the load asks %d",
                                              s.max_rank, max_rank));
        n_pools = std::max(n_pools, s.pool + 1);
    }
    std::vector<const Site*> rep((size_t)n_pools, nullptr);
    for (const Site& s : sites) {
        const Site*& r = rep[(size_t)s.pool];
        if (!r) r = &s;
        else if (!same_layout(*r, s))
            return fail(err, RAD_E_STATE, "pool of '" + s.weight + "' holds sites of two layouts");
    }
    std::unordered_map<std::string, std::vector<Slot>> where;
    for (int32_t p = 0; p < n_pools; ++p) {
        /* A pool whose only GEMM could carry no adapter ops (lora_declare drops it) holds
         * nothing. */
        if (!rep[(size_t)p]) continue;
        const Site& s = *rep[(size_t)p];
        for (size_t i = 0; i < s.sources.size(); ++i)
            if (s.sources[i].targeted) where[s.sources[i].module].push_back({ p, (int32_t)i });
    }

    /* ---- module -> base spelling (VLM bases keep the text model under model.language_model.) */
    std::map<std::pair<int32_t, int32_t>, const AdModule*> placed;   /* (pool, source) -> module */
    for (auto& kv : mods) {
        const AdModule& am = kv.second;
        std::vector<std::string> cand{ am.name };
        if (am.name.starts_with("model.") && !am.name.starts_with("model.language_model."))
            cand.push_back("model.language_model." + am.name.substr(6));
        if (am.name.starts_with("model.language_model."))
            cand.push_back("model." + am.name.substr(21));
        std::vector<std::string> hit;
        for (const std::string& c : cand) if (where.count(c)) hit.push_back(c);
        AdapterInfo::Module mi;
        mi.name = am.name;
        mi.rank = am.rank;
        mi.alpha = am.alpha;
        mi.scale = am.scale;
        if (hit.size() > 1)
            return fail(err, RAD_E_INVAL, fmt("module %s matches both %s and %s in the base; the "
                                              "spelling is ambiguous", am.name.c_str(),
                                              hit[0].c_str(), hit[1].c_str()));
        if (hit.empty()) {
            info->absent.push_back(am.name);
        } else {
            mi.base = hit[0];
            for (const Slot& sl : where[hit[0]]) {
                const AdModule*& to = placed[{ sl.pool, sl.source }];
                if (to)
                    return fail(err, RAD_E_INVAL, fmt("modules %s and %s both name the base's %s",
                                                      to->name.c_str(), am.name.c_str(),
                                                      hit[0].c_str()));
                to = &am;
                if (mi.pool < 0) mi.pool = sl.pool;
            }
        }
        info->modules.push_back(mi);
    }

    /* ---- pack, pool by pool */
    packed->resize((size_t)n_pools);
    for (int32_t p = 0; p < n_pools; ++p) {
        PackedSite& out = (*packed)[(size_t)p];
        out.pool = p;
        if (!rep[(size_t)p]) continue;
        const Site& s = *rep[(size_t)p];
        out.a.assign((size_t)(s.R * s.K_r), 0);
        out.b.assign((size_t)(s.N_r * s.R), 0);
        out.scale.assign((size_t)s.N_r, 0.0f);
        const size_t n = s.sources.size();
        std::vector<const AdModule*> am(n, nullptr);
        bool any = false;
        for (size_t i = 0; i < n; ++i) {
            auto it = placed.find({ p, (int32_t)i });
            if (it != placed.end()) { am[i] = it->second; any = true; }
        }
        if (!any) continue;
        out.empty = false;
        const int P = (int)s.part_total.size();

        /* A: [r, K_full] -- the whole of K for a row/whole site, this rank's columns for COL. */
        for (size_t i = 0; i < n; ++i) {
            if (!am[i]) continue;
            const StTensor& A = *am[i]->A;
            const int64_t kc = A.shape[1];
            if (s.K_full > 0 ? kc != s.K_full : kc < s.k_lo + s.K_r)
                return fail(err, RAD_E_SHAPE,
                            fmt("module %s: lora_A is %s; site '%s' expects [%lld, %s%lld]",
                                am[i]->name.c_str(), shape_str(A.shape).c_str(), s.weight.c_str(),
                                (long long)am[i]->rank, s.K_full > 0 ? "" : ">= ",
                                (long long)(s.K_full > 0 ? s.K_full : s.k_lo + s.K_r)));
        }

        /* The parts: fixed at build, or decided now by the B rows. */
        std::vector<SitePart> parts;
        std::vector<int64_t> T = s.part_total;
        std::vector<std::pair<int, int>> runs(n, { -1, -1 });
        for (size_t i = 0; i < n; ++i) runs[i] = { s.sources[i].part_lo, s.sources[i].part_hi };
        if (s.parts_resolved) {
            parts = s.parts;
        } else if (P == 1) {
            /* several sources inside one part: each one's rows from its B, the one unknown (if
             * any) from the part's total. */
            std::vector<int64_t> rows(n, -1);
            int unknown = 0;
            int64_t known = 0;
            for (size_t i = 0; i < n; ++i) {
                if (am[i]) { rows[i] = am[i]->B->shape[0]; known += rows[i]; }
                else ++unknown;
            }
            if (T[0] > 0) {
                if (unknown == 0 && known != T[0])
                    return fail(err, RAD_E_SHAPE, fmt("weight '%s': its sources' lora_B rows sum to "
                                                      "%lld and the weight has %lld rows",
                                                      s.weight.c_str(), (long long)known,
                                                      (long long)T[0]));
                if (unknown == 1)
                    for (size_t i = 0; i < n; ++i) if (rows[i] < 0) rows[i] = T[0] - known;
            }
            std::vector<bool> want(n);
            for (size_t i = 0; i < n; ++i) want[i] = am[i] != nullptr;
            const int st = segments_within(s, rows, want, &parts, err);
            if (st < 0) return st;
        } else {
            std::vector<std::vector<std::pair<int, int>>> all;
            compositions(P, (int)n, &all);
            std::vector<std::pair<std::vector<std::pair<int, int>>, std::vector<int64_t>>> ok;
            for (const auto& c : all) {
                std::vector<int64_t> t = T;
                bool unknown_t = false;
                for (int64_t x : t) if (x <= 0) unknown_t = true;
                bool good = true;
                if (unknown_t) {   /* an equal-part span: one common part height */
                    int64_t tc = 0;
                    for (size_t i = 0; i < n && good; ++i) {
                        if (!am[i]) continue;
                        const int64_t m = c[i].second - c[i].first, b = am[i]->B->shape[0];
                        if (b % m) { good = false; break; }
                        if (!tc) tc = b / m;
                        else if (tc != b / m) good = false;
                    }
                    for (int64_t& x : t) x = tc;
                } else {
                    for (size_t i = 0; i < n && good; ++i) {
                        if (!am[i]) continue;
                        int64_t sum = 0;
                        for (int g = c[i].first; g < c[i].second; ++g) sum += t[(size_t)g];
                        good = sum == am[i]->B->shape[0];
                    }
                }
                if (good) ok.push_back({ c, t });
            }
            if (ok.size() != 1) {
                std::string got;
                for (size_t i = 0; i < n; ++i)
                    if (am[i]) got += fmt("%s%s %lld rows", got.empty() ? "" : ", ",
                                          am[i]->name.c_str(), (long long)am[i]->B->shape[0]);
                return fail(err, RAD_E_SHAPE,
                            fmt("weight '%s': %zu ways to lay its %zu sources over its %d row parts "
                                "fit the adapter (%s); exactly one must", s.weight.c_str(),
                                ok.size(), n, P, got.c_str()));
            }
            runs = ok[0].first;
            T = ok[0].second;
            const int st = segments_runs(s, runs, T, &parts, err);
            if (st < 0) return st;
        }

        /* B rows against the runs the parts came from. */
        for (size_t i = 0; i < n; ++i) {
            if (!am[i] || runs[i].first < 0) continue;
            int64_t sum = 0;
            bool all_known = true;
            for (int g = runs[i].first; g < runs[i].second; ++g) {
                if (T[(size_t)g] <= 0) all_known = false;
                sum += T[(size_t)g];
            }
            const int64_t b = am[i]->B->shape[0];
            if (all_known ? b != sum : false)
                return fail(err, RAD_E_SHAPE, fmt("module %s: lora_B is %s; site '%s' expects "
                                                  "[%lld, %lld]", am[i]->name.c_str(),
                                                  shape_str(am[i]->B->shape).c_str(),
                                                  s.weight.c_str(), (long long)sum,
                                                  (long long)am[i]->rank));
        }
        for (const SitePart& sp : parts) {
            size_t i = 0;
            while (i < n && s.sources[i].module != sp.module) ++i;
            if (i == n || !am[i]) continue;
            if (sp.src_hi > am[i]->B->shape[0])
                return fail(err, RAD_E_SHAPE, fmt("module %s: lora_B is %s; this rank reads its rows "
                                                  "[%lld, %lld)", am[i]->name.c_str(),
                                                  shape_str(am[i]->B->shape).c_str(),
                                                  (long long)sp.src_lo, (long long)sp.src_hi));
        }

        /* Pack. A: rows t_off.. of the module's block; B: rows y_col.., columns t_off..; scale on
         * the module's rows. Compact: the modules' ranks one after another from 0. */
        std::vector<int64_t> off(n, 0);
        int64_t used = 0;
        for (size_t i = 0; i < n; ++i) {
            off[i] = compact ? used : (int64_t)s.sources[i].slot * max_rank;
            if (am[i]) used += am[i]->rank;
        }
        out.r_used = compact ? used : s.R;
        for (size_t i = 0; i < n; ++i) {
            if (!am[i]) continue;
            const StTensor& A = *am[i]->A;
            const int64_t r = am[i]->rank, kc = A.shape[1];
            const int64_t t0 = off[i];
            for (int64_t j = 0; j < r; ++j)
                for (int64_t k = 0; k < s.K_r; ++k)
                    out.a[(size_t)((t0 + j) * s.K_r + k)] = elem_bf16(A, j * kc + s.k_lo + k);
        }
        for (const SitePart& sp : parts) {
            size_t i = 0;
            while (i < n && s.sources[i].module != sp.module) ++i;
            if (i == n || !am[i]) continue;
            const StTensor& B = *am[i]->B;
            const int64_t r = am[i]->rank;
            for (int64_t row = sp.src_lo; row < sp.src_hi; ++row) {
                const int64_t yr = sp.y_col + (row - sp.src_lo);
                if (yr < 0 || yr >= s.N_r)
                    return fail(err, RAD_E_STATE, fmt("weight '%s': a part lands outside y", s.weight.c_str()));
                for (int64_t j = 0; j < r; ++j)
                    out.b[(size_t)(yr * s.R + off[i] + j)] = elem_bf16(B, row * r + j);
                out.scale[(size_t)yr] = am[i]->scale;
            }
        }
    }
    return RAD_OK;
}

int check_placed(const std::vector<AdapterInfo>& per_rank, std::string* err, const Program* prog,
                 const TargetSet* targets) {
    if (per_rank.empty()) return RAD_OK;
    for (const AdapterInfo::Module& m : per_rank[0].modules) {
        bool anywhere = false;
        for (const AdapterInfo& a : per_rank)
            for (const AdapterInfo::Module& x : a.modules)
                if (x.name == m.name && x.pool >= 0) anywhere = true;
        if (anywhere) continue;
        std::string why = prog && targets ? explain_module(*prog, *targets, m.name) : "";
        return fail(err, RAD_E_NOTFOUND,
                    "adapter module " + m.name + " is not a LoRA site on any rank" +
                        (why.empty() ? std::string() : ": " + why));
    }
    return RAD_OK;
}

}  /* namespace lora */
}  /* namespace rad */
