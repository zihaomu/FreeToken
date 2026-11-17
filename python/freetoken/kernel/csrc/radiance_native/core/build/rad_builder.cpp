/* rad_builder.cpp -- the Builder, and the C surface of rad_builder.h that drives it.
 *
 * Handles are 1-based indices into the Program's tables; index 0 is a sentinel, so a failed
 * decl_* is falsy and an architecture plugin can write
 *
 *     rad_op h = decl_op(b, "rmsnorm_had_quant_i8", ...);
 *     if (!h) { ... emit the unfused sequence ... }
 *
 * which makes fusion a selection rather than a compiler pass (spec §2.3).
 */
#include "rad_build.h"
#include "kvstream/bringup.h"
#include "lora/lora.h"
#include "../device/device.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <set>

namespace rad {

/* ================================================================== interning */
const char* intern(std::string_view s) {
    /* A node-based set: insertion never moves an element, so returned pointers stay valid. Locked
     * because each rank declares on its own thread (spec §1). */
    static std::unordered_set<std::string> pool;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    return pool.emplace(s).first->c_str();
}

/* ================================================================== small helpers */
const char* bld_domain_name(int d) { return d == RAD_DOMAIN_HOST ? "host" : "device"; }

/* Defined here so the domain is spelled by bld_domain_name. */
std::string Program::Miss::line() const {
    return fmt("%-22s %s  [%s] -> %s", op.c_str(), geom.c_str(), bld_domain_name(domain),
               why.c_str());
}

std::vector<const Program::Miss*> Program::device_holes(bool host_backend) const {
    /* Per instance, like bands: two declarations of one op can differ in geometry and in whether
     * they carry a weight, which decides whether the planner can move the op to the host. */
    std::set<std::pair<int32_t, std::string>> no_host;
    for (const Miss& m : misses)
        if (m.domain == RAD_DOMAIN_HOST) no_host.emplace(m.index, m.span);
    std::vector<const Miss*> out;
    for (const Miss& m : misses) {
        if (m.domain == RAD_DOMAIN_HOST) continue;
        const bool host_kernel = no_host.count(std::make_pair(m.index, m.span)) == 0;
        /* An instance with no weight has nothing the planner places, so nothing moves it. */
        const bool weightless = m.index > 0 && (size_t)m.index < ops.size() &&
                                ops[(size_t)m.index].weights.empty();
        if (host_kernel && (host_backend || !weightless)) continue;
        out.push_back(&m);
    }
    return out;
}

const char* bld_access_name(int a) {
    switch (a) {
        case RAD_ACCESS_PER_TOKEN:   return "per-token";
        case RAD_ACCESS_PER_REQUEST: return "per-request";
        case RAD_ACCESS_CONDITIONAL: return "conditional";
        case RAD_ACCESS_RARE:        return "rare";
        case RAD_ACCESS_VOCAB:       return "vocab";
        default:                     return "?";
    }
}

const char* bld_buf_kind_name(int k) {
    switch (k) {
        case RAD_BUF_TRANSIENT: return "transient";
        case RAD_BUF_PERSIST:   return "persist";
        case RAD_BUF_DERIVED:   return "derived";
        default:                return "?";
    }
}

const char* bld_kv_kind_name(int k) {
    switch (k) {
        case RAD_KV_FULL:   return "full";
        case RAD_KV_WINDOW: return "window";
        case RAD_KV_LINEAR: return "linear";
        case RAD_KV_CONV:   return "conv";
        case RAD_KV_RING:   return "ring";
        default:            return "?";
    }
}

static int64_t decl_numel(uint32_t rank, const int64_t* shape) {
    if (rank == 0 || rank > RAD_MAX_RANK) return 0;
    int64_t n = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] <= 0) return 0;
        n *= shape[i];
    }
    return n;
}

/* ================================================================== construction */
Builder::Builder(Registry& reg, const RadModelMeta& meta, const RadBuildCtx& ctx)
    : reg_(reg) {
    prog_.meta = meta;
    prog_.ctx  = ctx;
    scope_ = ctx.scope ? ctx.scope : "";

    /* Sentinels: handle 0 is never valid, so a zeroed struct is an unset handle. */
    prog_.weights.emplace_back();
    prog_.buffers.emplace_back();
    prog_.ops.emplace_back();
    prog_.kv_groups.emplace_back();
    op_bufs_.emplace_back();
    kv_decl_at_.push_back(-1);
    kv_consumer_.push_back(-1);
}

uint32_t Builder::err(int code, const char* f, ...) {
    char msg[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(msg, sizeof msg, f, ap);
    va_end(ap);
    errors_.emplace_back(msg);
    /* First class wins; later errors are usually consequences, and the list keeps them all. */
    if (status_ == RAD_OK) status_ = code;
    RAD_ERR("declare: %s", msg);
    return RAD_NULL_HANDLE;
}

/* The scope prefixes declared names so a drafter, a vision tower and the language model can share
 * one builder (spec §10, §11). Op names are not scoped: the selector matches on them. */
std::string Builder::scoped(const char* name) const {
    if (scope_.empty()) return name ? name : "";
    return scope_ + "." + (name ? name : "");
}

/* ================================================================== weights */
rad_weight Builder::decl_weight(const char* name, const RadWeightDecl* d) {
    if (finished_) return err(RAD_E_STATE, "decl_weight('%s') after declare finished", name ? name : "");
    if (!name || !*name || !d) return err(RAD_E_INVAL, "decl_weight: null name or declaration");

    std::string n = scoped(name);
    if (weight_by_name_.count(n))
        return err(RAD_E_DUPLICATE, "weight '%s' declared twice; the container has one entry "
                                    "for that name and two declarations cannot both own it",
                   n.c_str());

    int64_t numel = decl_numel(d->rank, d->shape);
    if (numel == 0)
        return err(RAD_E_SHAPE, "weight '%s': rank %u with a zero or negative extent",
                   n.c_str(), d->rank);

    WeightInfo w;
    w.name = n;
    w.decl = *d;
    /* Under tensor parallelism the plugin declares dimensions already divided (spec §9), so this
     * is what this rank holds. The core never infers or divides shards. */
    w.logical_bytes = rad_dtype_bytes(d->dtype, numel);
    if (w.logical_bytes == 0 && d->dtype < RAD_DT_PLUGIN_BASE)
        return err(RAD_E_DTYPE, "weight '%s': dtype %u is not one the core can size", n.c_str(),
                   d->dtype);

    /* The view's source (spec §4.3): its encoding as the model holds it, and the planes this
     * declaration takes. With no source (tests, synthetic tools) the weight is one plain plane in
     * the declared dtype. */
    w.source = (d->source && *d->source) ? scoped(d->source) : n;
    WeightSource ws;
    w.enc_known = lookup_source(w.source, &ws) == RAD_OK;
    w.enc = w.enc_known ? ws.enc : rad_enc_plain(d->dtype);
    int64_t vr = 0, vc = 0;            /* the declared extents as a [rows, cols] view */
    rad_enc_view(d->rank, d->shape, &vr, &vc);
    if (!w.enc_known) {
        w.n_sel = 1;
        w.sel[0] = 0;
    } else if (!d->planes || !*d->planes) {
        w.n_sel = w.enc.n_planes;
        for (int k = 0; k < w.n_sel; ++k) w.sel[k] = k;
    } else {
        std::string list(d->planes);
        size_t at = 0;
        while (at <= list.size()) {
            size_t comma = list.find(',', at);
            if (comma == std::string::npos) comma = list.size();
            std::string role = list.substr(at, comma - at);
            while (!role.empty() && role.front() == ' ') role.erase(role.begin());
            while (!role.empty() && role.back() == ' ') role.pop_back();
            const int k = rad_enc_find(&w.enc, role.c_str());
            if (k < 0) {
                char e[256];
                rad_enc_format(&w.enc, e, sizeof e);
                return err(RAD_E_SHAPE, "weight '%s' takes plane '%s' of '%s', which is %s and has "
                                        "no such plane", n.c_str(), role.c_str(), w.source.c_str(), e);
            }
            if (w.n_sel == RAD_ENC_MAX_PLANES)
                return err(RAD_E_INVAL, "weight '%s': more than %d planes", n.c_str(),
                           RAD_ENC_MAX_PLANES);
            w.sel[w.n_sel++] = k;
            at = comma + 1;
        }
    }

    /* Check the declaration against the planes. One plane is declared at its own dtype and extents
     * (this rank's share under a shard); several at the codes' dtype and the logical extents. A
     * different dtype must be an exact conversion the loader performs; a quantisation belongs to
     * rad-convert and is refused. */
    const RadEncPlane& p0 = w.enc.plane[w.sel[0]];
    if (p0.dtype != d->dtype) {
        const bool widen = w.n_sel == 1 && d->dtype == RAD_F32 &&
                           (p0.dtype == RAD_BF16 || p0.dtype == RAD_F16);
        /* A bf16 readout head over an f32 checkpoint tensor: the loader rounds it to bf16. */
        const bool narrow = w.n_sel == 1 && d->dtype == RAD_BF16 && p0.dtype == RAD_F32;
        if (!widen && !narrow) {
            char e[256];
            rad_enc_format(&w.enc, e, sizeof e);
            return err(RAD_E_DTYPE, "weight '%s' is declared %s, and '%s' holds %s as %s: only an "
                                    "exact widening or narrowing is done at load; anything else "
                                    "is a quantisation, which a recipe does before the container "
                                    "is written (spec §4.4)", n.c_str(), rad_dtype_name(d->dtype),
                       w.source.c_str(), p0.role, e);
        }
        w.widen = widen;
        w.narrow = narrow;
    }
    int64_t sr = 0, sc = 0;            /* the source's logical [rows, cols] */
    if (w.enc_known) rad_enc_view(ws.rank, ws.shape, &sr, &sc);
    for (int k = 0; k < w.n_sel; ++k) {
        const RadEncPlane& pk = w.enc.plane[w.sel[k]];
        int64_t fr = vr, fc = vc;      /* the plane's whole extents */
        if (w.n_sel == 1) {
            w.sel_rows[k] = vr;
            w.sel_cols[k] = vc;
            if (w.enc_known) rad_enc_plane_dims(&pk, sr, sc, &fr, &fc);
        } else {
            rad_enc_plane_dims(&pk, vr, vc, &w.sel_rows[k], &w.sel_cols[k]);
            fr = sr; fc = sc;
        }
        /* Unsharded, the declaration must match the plane (or, for several, the logical weight);
         * a mismatch is reported now rather than as a byte count at load. The loader checks a
         * shard's share. */
        const bool whole = d->shard == RAD_SHARD_NONE || prog_.ctx.world_size <= 1;
        /* A reshape is allowed: one whole plane of single elements (a byte or more each) declared
         * under another shape with the same count, e.g. a conv kernel [O, C, T, H, W] read as
         * [O, C*T*H*W]. Row-major bytes are unchanged. */
        const bool reshape = w.n_sel == 1 && whole && pk.kind == RAD_PLANE_TILED &&
                             pk.block[0] == 1 && pk.block[1] == 1 &&
                             rad_dtype_bits(pk.dtype) >= 8 && fr * fc == vr * vc;
        if (w.enc_known && whole && !reshape && (k == 0 || w.n_sel == 1) &&
            (w.n_sel == 1 ? (fr != vr || fc != vc) : (sr != vr || sc != vc)))
            return err(RAD_E_SHAPE, "weight '%s' is declared [%lld, %lld], and %s of '%s' is "
                                    "[%lld, %lld]", n.c_str(), (long long)vr, (long long)vc,
                       w.n_sel == 1 ? pk.role : "the logical extent", w.source.c_str(),
                       (long long)(w.n_sel == 1 ? fr : sr), (long long)(w.n_sel == 1 ? fc : sc));
    }

    /* The stored form defaults to the selected plane unchanged. finish_layouts() overwrites it when
     * the resolved kernel's layout hook asks for another (spec §4.3); no hook, or
     * RAD_E_UNSUPPORTED, keeps this. */
    w.identity     = true;
    const bool converted = w.widen || w.narrow;
    w.stored_dtype = converted ? d->dtype : p0.dtype;
    w.stored_bytes = converted ? rad_dtype_bytes(d->dtype, vr * vc)
                               : vr * rad_enc_row_bytes(p0.dtype, vc);
    w.stored_rank  = d->rank;
    for (uint32_t i = 0; i < RAD_MAX_RANK; ++i)
        w.stored_shape[i] = i < d->rank ? d->shape[i] : 0;
    w.align        = RAD_ALIGN_UNIT;

    prog_.weights.push_back(std::move(w));
    const rad_weight h = (rad_weight)(prog_.weights.size() - 1);
    weight_by_name_.emplace(prog_.weights.back().name, h);
    return h;
}

int Builder::lookup_source(const std::string& name, WeightSource* out) {
    if (!sources_) return RAD_E_NOTFOUND;
    auto it = source_cache_.find(name);
    if (it != source_cache_.end()) {
        *out = it->second.second;
        return it->second.first;
    }
    WeightSource ws;
    const int st = sources_(name, prog_, &ws);
    /* Cache only successes: a weight not found before its name-map entry is declared may be found
     * after. */
    if (st == RAD_OK) source_cache_.emplace(name, std::make_pair(st, ws));
    *out = ws;
    return st;
}

int Builder::weight_encoding(const char* source, RadEncoding* out, int64_t* shape,
                             uint32_t* rank) {
    if (!source || !*source || !out) return RAD_E_INVAL;
    WeightSource ws;
    const int st = lookup_source(scoped(source), &ws);
    if (st != RAD_OK) return st;
    *out = ws.enc;
    if (rank) *rank = ws.rank;
    if (shape)
        for (int i = 0; i < RAD_MAX_RANK; ++i) shape[i] = i < (int)ws.rank ? ws.shape[i] : 0;
    return RAD_OK;
}

/* ================================================================== buffers */
int64_t buffer_bytes(const RadBufDecl& d) {
    int64_t numel = decl_numel(d.rank, d.shape);
    if (numel == 0) return 0;
    return rad_dtype_bytes(d.dtype, numel);
}

rad_buf Builder::decl_buffer(const char* name, const RadBufDecl* d) {
    if (finished_) return err(RAD_E_STATE, "decl_buffer('%s') after declare finished", name ? name : "");
    if (!name || !*name || !d) return err(RAD_E_INVAL, "decl_buffer: null name or declaration");

    std::string n = scoped(name);
    for (size_t i = 1; i < prog_.buffers.size(); ++i)
        if (prog_.buffers[i].name == n)
            return err(RAD_E_DUPLICATE, "buffer '%s' declared twice", n.c_str());

    if (d->kind < RAD_BUF_TRANSIENT || d->kind > RAD_BUF_DERIVED)
        return err(RAD_E_INVAL, "buffer '%s': unknown kind %d", n.c_str(), d->kind);
    if (d->domain != RAD_DOMAIN_DEVICE && d->domain != RAD_DOMAIN_HOST)
        return err(RAD_E_INVAL, "buffer '%s': unknown domain %d", n.c_str(), d->domain);

    BufferInfo b;
    b.name = n;
    b.decl = *d;
    b.bytes = buffer_bytes(*d);
    if (b.bytes == 0)
        return err(RAD_E_SHAPE, "buffer '%s': the core cannot size dtype %u at rank %u -- an "
                                "activation must be a dtype the buffer planner understands",
                   n.c_str(), d->dtype, d->rank);

    prog_.buffers.push_back(std::move(b));
    return (rad_buf)(prog_.buffers.size() - 1);
}

/* ================================================================== KV groups */
rad_kvgroup Builder::decl_kv_group(const char* name, const RadKVGroupDecl* d) {
    if (finished_) return err(RAD_E_STATE, "decl_kv_group('%s') after declare finished", name ? name : "");
    if (!name || !*name || !d) return err(RAD_E_INVAL, "decl_kv_group: null name or declaration");
    if (d->kind < RAD_KV_FULL || d->kind > RAD_KV_RING)
        return err(RAD_E_INVAL, "kv group '%s': unknown kind %d", name, d->kind);

    std::string n = scoped(name);
    for (size_t i = 1; i < prog_.kv_groups.size(); ++i)
        if (prog_.kv_groups[i].name == n)
            return err(RAD_E_DUPLICATE, "kv group '%s' declared twice", n.c_str());

    KVGroupInfo g;
    g.name = n;
    g.decl = *d;
    prog_.kv_groups.push_back(std::move(g));
    kv_decl_at_.push_back((int32_t)prog_.ops.size());   /* current position in the op list */
    kv_consumer_.push_back(-1);
    return (rad_kvgroup)(prog_.kv_groups.size() - 1);
}

int Builder::bind_layer_kv(int layer, rad_kvgroup g) {
    if (g == 0 || g >= prog_.kv_groups.size()) {
        err(RAD_E_INVAL, "bind_layer_kv(layer %d): kv group handle %u is not one this builder "
                         "handed out", layer, g);
        return RAD_E_INVAL;
    }
    if (layer < 0) { err(RAD_E_INVAL, "bind_layer_kv: layer %d", layer); return RAD_E_INVAL; }
    auto& v = prog_.kv_groups[g].layers;
    if (std::find(v.begin(), v.end(), layer) == v.end()) v.push_back(layer);
    return RAD_OK;
}

/* Which op reads this group. The ABI does not link an op to a KV group, so declaration order is
 * the binding, as documented: "declare the group, declare the op, then ask rad_kv_block_size()"
 * (docs/OPS.md, spec §7.2). The consumer is the first unclaimed op after the group that carries a
 * block_size; an op serves at most one group, so a hybrid model's full and sliding-window groups
 * each claim their own attention op.
 *
 * An op carries a block_size if it declared one, its schema names one (attn_paged does, so it is
 * found even when the plugin omits the value it is about to ask for), or its resolved kernel
 * constrains one. */
int32_t Builder::find_kv_consumer(rad_kvgroup g) {
    if (kv_consumer_[g] >= 0) return kv_consumer_[g];

    for (int32_t i = kv_decl_at_[g]; i < (int32_t)prog_.ops.size(); ++i) {
        bool claimed = false;
        for (size_t k = 1; k < kv_consumer_.size(); ++k) if (kv_consumer_[k] == i) claimed = true;
        if (claimed) continue;

        const OpInfo& o = prog_.ops[i];
        bool carries = o.base.has("block_size");
        if (!carries && o.schema && o.schema->params)
            for (int c = 0; c < o.schema->n_params && !carries; ++c)
                if (o.schema->params[c].key && !std::strcmp(o.schema->params[c].key, "block_size"))
                    carries = true;
        if (!carries)
            for (const Band& b : o.bands)
                for (int d = 0; d < RAD_N_DOMAINS && !carries; ++d)
                    if (b.dom[d] && b.dom[d].row && b.dom[d].row->info)
                        for (int c = 0; c < b.dom[d].row->info->n_constraints; ++c)
                            if (!std::strcmp(b.dom[d].row->info->constraints[c].key, "block_size"))
                                carries = true;
        if (carries) { kv_consumer_[g] = i; return i; }
    }
    return -1;
}

int64_t Builder::kv_block_size(rad_kvgroup g) {
    if (g == 0 || g >= prog_.kv_groups.size()) {
        err(RAD_E_INVAL, "rad_kv_block_size: handle %u is not a kv group this builder handed out", g);
        return 0;
    }
    KVGroupInfo& gi = prog_.kv_groups[g];

    /* Per-sequence state has no block size; warn and return 0 (the caller wants
     * bytes_per_state). */
    if (gi.decl.kind == RAD_KV_LINEAR || gi.decl.kind == RAD_KV_CONV) {
        RAD_WARN("kv group '%s' is per-sequence state; it has no block size", gi.name.c_str());
        return 0;
    }
    if (gi.block_size) return gi.block_size;

    int32_t consumer = find_kv_consumer(g);
    if (consumer < 0) {
        /* An error, not a silent 0, which would become a division by zero in the block
         * manager. */
        err(RAD_E_STATE,
            "rad_kv_block_size('%s'): no op that reads this group has been declared yet. The "
            "block size comes off the RESOLVED attention kernel's block_size constraint, so the "
            "attention op has to exist before there is an answer (spec §7.2)", gi.name.c_str());
        return 0;
    }

    const OpInfo& o = prog_.ops[consumer];

    /* The resolved kernel's block_size constraint is the answer, read with core/plugin/'s
     * accessor; the core never picks a block size (spec §7.2). */
    for (const Band& b : o.bands)
        for (int d = 0; d < RAD_N_DOMAINS; ++d) {
            if (!b.dom[d] || !b.dom[d].row) continue;
            long long bs = 0;
            if (kernel_constraint(*b.dom[d].row, "block_size", RAD_C_EQ, &bs) && bs > 0) {
                gi.block_size = bs;
                return gi.block_size;
            }
        }

    /* No kernel constrains it (e.g. libref, which serves any block size, spec §17): use the
     * plugin's declared parameter. */
    long long v = 0;
    if (o.base.get_i("block_size", &v) && v > 0) { gi.block_size = v; return gi.block_size; }

    /* Nothing to read it from. The usual cause: a constraint on a key the geometry lacks does not
     * hold, so a kernel constrained to `block_size == 16` is never selected for an op declared
     * without block_size. The plugin must declare the block size it wants; the resolved kernel's
     * constraint then confirms or refuses it (spec §7.2). */
    bool resolved_anywhere = false;
    for (const Band& b : o.bands)
        for (int d = 0; d < RAD_N_DOMAINS; ++d) if (b.dom[d]) resolved_anywhere = true;

    err(RAD_E_STATE,
        "rad_kv_block_size('%s'): op '%s' consumes this group but %s. Declare block_size on the "
        "op with the value this model wants: a kernel constrained to a block size cannot be "
        "selected by a query that omits it, so leaving it off means the resolution it was to be "
        "read from never happens (spec §7.2)",
        gi.name.c_str(), o.op.c_str(),
        resolved_anywhere ? "neither its resolved kernel nor its declared parameters name a "
                            "block_size"
                          : "it resolved to no kernel at all, so there is no constraint to read");
    return 0;
}

/* One layer's bytes of a paged group's block: what a kernel addressing one layer sees, and the
 * pitch of a streamed group's host mirror rows. */
int64_t Builder::kv_layer_bytes(rad_kvgroup g) {
    if (g == 0 || g >= prog_.kv_groups.size()) return 0;
    KVGroupInfo& gi = prog_.kv_groups[g];
    if (gi.decl.kind != RAD_KV_FULL && gi.decl.kind != RAD_KV_WINDOW) return 0;
    const int64_t bs = kv_block_size(g);
    if (bs <= 0) return 0;
    const int64_t nh = gi.decl.n_head_kv > 0 ? gi.decl.n_head_kv : 1;
    return rad_dtype_bytes(gi.decl.dtype,
                           bs * nh * rad_kv_row_elems(gi.decl.dtype, gi.decl.head_dim));
}

/* ================================================================== ops */
/* A kernel instantiation: (kernel, plugin, resolved input params). Keyed on geom_key, not
 * geom.str(), so tuner and selector key identically: geom_key is tunecache.h's canonical sorted
 * spelling without the tuned axes, while geom.str() keeps declaration order and includes them. */
static std::string tune_key(const Resolved& r) {
    return r.row->info->name + std::string("\x1f") + r.row->plugin + std::string("\x1f") +
           r.geom_key;
}

void Builder::record_tune_request(const Resolved& r) {
    if (!r.row || !r.row->info || r.row->info->n_tunables <= 0) return;
    /* The same shape requested by many layers is one benchmark (spec §15). */
    if (!tune_index_.emplace(tune_key(r), prog_.tune_requests.size()).second) return;

    Program::TuneRequest req;
    req.kernel     = r.row->info->name;
    req.plugin     = r.row->plugin;
    req.geom       = r.geom;
    req.geom_key   = r.geom_key;
    req.info       = r.row->info;
    req.instance   = r.instance;
    prog_.tune_requests.push_back(std::move(req));
}

/* The geometry a band is displayed as: the fixed parameters plus the ranged one at the band's upper
 * bound. Diagnostics only; the kernel receives the geometry the registry froze in Resolved. */
static Geometry display_geometry(const OpInfo& oi, const Band& b) {
    Geometry g = oi.base;
    if (!oi.ranged_key.empty() && b.hi != INT64_MAX) g.set_i(oi.ranged_key, b.hi);
    return g;
}

void Builder::resolve_bands(OpInfo& oi) {
    /* The bucket table comes from core/plugin/: boundaries are the union of candidate kernels'
     * constraint values on the ranged parameter, clamped to the declared range, and both domains
     * resolve at each band's upper bound (the worst case, the only safe value to select and size
     * on). No ranged parameter yields one band with hi INT64_MAX, which band_for() always finds.
     *
     * A kernel that cannot read a weight's encoding is not a candidate: its layout hook returns
     * RAD_E_DTYPE, and a hookless device row reads only a single plane as stored. So a bf16 head
     * goes to the bf16 GEMM instead of being refused by the fp8 one, and one declaration serves a
     * checkpoint and its quantised container. Hookless host rows are libref, which reads
     * canonically and is gated elsewhere. */
    const Registry::RowFilter accept = [&](const KernelRow& r, const Geometry& g) -> std::string {
        const RadKernelInfo* ki = r.info;
        for (size_t wi = 0; wi < oi.weights.size(); ++wi) {
            const rad_weight h = oi.weights[wi];
            if (h == 0 || h >= prog_.weights.size()) continue;
            const WeightInfo& w = prog_.weights[h];
            if (sources_ && !w.enc_known && w.decl.optional) continue;
            const int operand = wi < oi.weight_opd.size() && oi.weight_opd[wi] >= 0
                                    ? (int)oi.weight_opd[wi] : (int)wi;
            RadTensor t[RAD_ENC_MAX_PLANES];
            weight_planes(w, t);
            std::string why = weight_refusal(ki, g, operand, w.enc, w.sel, t, w.n_sel, w.name);
            if (!why.empty()) return why;
        }
        return std::string();
    };
    oi.bands = reg_.build_bands(oi.op, oi.ranged_key, oi.range_lo, oi.range_hi, oi.base,
                                oi.weights.empty() ? nullptr : &accept);

    if (oi.bands.empty()) {
        /* No rows for this op at all; add one band so the miss report names it. */
        oi.bands.emplace_back();
        oi.bands.back().hi = oi.ranged_key.empty() ? INT64_MAX : oi.range_hi;
        for (int d = 0; d < RAD_N_DOMAINS; ++d)
            oi.bands.back().miss[d] = "no kernel implements this op in this domain";
    }

    for (Band& b : oi.bands) {
        for (int d = 0; d < RAD_N_DOMAINS; ++d) {
            Resolved& r = b.dom[d];

            if (!r) {
                /* One miss per op, band and domain, with the refusing constraint. Recorded, not
                 * fatal: whether a miss matters is decided later (spec §3.1). */
                prog_.misses.push_back({ oi.op, display_geometry(oi, b).str(), b.span,
                                         b.miss[d].empty() ? "no kernel matched" : b.miss[d],
                                         d, oi.index });
                continue;
            }

            /* Scratch sized at r.geom, the frozen geometry the band was selected at. RadArgs
             * carries no tensors: a scratch hook depends only on geometry, so it can run at
             * declare. */
            if (r.row->info && r.row->info->scratch) {
                RadArgs a{};
                a.p          = r.geom.params();
                a.n_p        = r.geom.n_params();
                a.rank       = prog_.ctx.rank;
                a.world_size = prog_.ctx.world_size;
                int64_t s = r.row->info->scratch(&a);
                if (s < 0)
                    err(RAD_E_SCRATCH, "%s: kernel %s refused to size its scratch for %s: %s",
                        oi.op.c_str(), r.row->info->name, r.geom.str().c_str(),
                        rad_strerror((int)s));
                else if (s > r.scratch_bytes)
                    r.scratch_bytes = s;
            }
            if (r.scratch_bytes > prog_.scratch_bytes) prog_.scratch_bytes = r.scratch_bytes;

            record_tune_request(r);
        }
    }
}

/* init() once per resolved instance (op, band, domain), after the whole graph has resolved
 * (spec §2.1; see rad_build.h). A negative init means the kernel refused an instance it matched:
 * the band becomes a miss reported here rather than at the first launch. */
void Builder::finish_instances() {
    for (size_t i = 1; i < prog_.ops.size(); ++i) {
        OpInfo& oi = prog_.ops[i];
        for (Band& b : oi.bands) {
            for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                Resolved& r = b.dom[d];
                if (!r) continue;
                const int s = reg_.init_instance(r, prog_.ctx.rank, prog_.ctx.world_size);
                if (s >= 0) continue;

                err(s, "%s: kernel %s matched %s and then its init() refused the instance: %s",
                    oi.op.c_str(), r.row->info ? r.row->info->name : "?",
                    r.geom.str().c_str(), rad_strerror(s));
                b.miss[d] = fmt("init() refused the instance: %s", rad_strerror(s));
                prog_.misses.push_back({ oi.op, display_geometry(oi, b).str(), b.span,
                                         b.miss[d], d, oi.index });
                r = Resolved{};
            }
        }
    }
}

/* Hand each tune request the instance finish_instances() made for its kernel: a collective's
 * launch reaches its peer bundle (scratch, flags, sequence counters) only through the instance, so
 * the tuner forms its launches from this. */
void Builder::fill_tune_instances() {
    if (prog_.tune_requests.empty()) return;
    for (size_t i = 1; i < prog_.ops.size(); ++i)
        for (const Band& b : prog_.ops[i].bands)
            for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                const Resolved& r = b.dom[d];
                if (!r || !r.row || !r.row->info || r.row->info->n_tunables <= 0) continue;
                const auto it = tune_index_.find(tune_key(r));
                if (it != tune_index_.end()) prog_.tune_requests[it->second].instance = r.instance;
            }
}

rad_op Builder::decl_op(const char* opname, const RadParam* p, int n_p,
                        const rad_weight* w, int n_w) {
    if (finished_) return err(RAD_E_STATE, "decl_op('%s') after declare finished", opname ? opname : "");
    if (!opname || !*opname) return err(RAD_E_INVAL, "decl_op: null op name");
    if (n_p < 0 || n_w < 0) return err(RAD_E_INVAL, "decl_op('%s'): negative count", opname);

    /* ---- the schema, checked by its owner. Registry::validate covers the parameter side: unknown
     * op (naming the near miss), unknown or repeated key, type mismatch, empty range, and the
     * one-ranged-parameter rule. Its message is written for plugin authors and passed through
     * unchanged. */
    std::string verr;
    const int v = reg_.validate(opname, p, n_p, &verr);
    if (v < 0) return err(v, "%s", verr.c_str());

    const RadOpSchema* sc = reg_.schema(opname);
    if (!sc) return err(RAD_E_NOSCHEMA, "op '%s' has no declared schema", opname);

    /* ---- parameters, already validated: split the fixed ones from the single ranged one. */
    Geometry base;
    std::string ranged_key;
    int64_t lo = 0, hi = 0;

    for (int i = 0; i < n_p; ++i) {
        const RadParam& q = p[i];
        switch (q.kind) {
            case RAD_P_INT:   base.set_i(q.key, q.ival); break;
            case RAD_P_STR:   base.set_s(q.key, q.sval); break;
            /* A float reaches the kernel through the geometry but is never matched on or made a
             * band boundary (docs/OPS.md; see constraint_holds). */
            case RAD_P_F64:   base.set_f(q.key, q.dval); break;
            case RAD_P_RANGE: ranged_key = q.key; lo = q.ival; hi = q.ihi; break;
            default:
                return err(RAD_E_INVAL, "op '%s': parameter '%s' has kind %d, which is not INT, "
                                        "F64, STR or RANGE", opname, q.key, q.kind);
        }
    }

    /* ---- weight operand count, which validate() does not see. Arguments are positional, so a
     * count that disagrees with the schema produces silent garbage, not an error (spec §2.3). Only
     * weight operands are named at declare; the full count is checked at issue. */
    size_t before = errors_.size();
    int w_req = 0, w_max = 0, n_tab = 0;
    for (int i = 0; sc->operands && i < sc->n_operands; ++i) {
        if (sc->operands[i].role == RAD_OPD_WTAB) { ++n_tab; continue; }
        if (sc->operands[i].role != RAD_OPD_WEIGHT) continue;
        ++w_max;
        if (!sc->operands[i].optional) ++w_req;
    }
    /* ---- table run lengths. A RAD_OPD_WTAB position consumes a run of declared weights. After
     * each fixed weight position takes one, the rest is divided equally across the table
     * positions in declaration order (a grouped GEMM has one codes and one scales plane per
     * expert). If it does not divide, the first positions take one more, which matches n experts
     * split into C classes by number mod C. Any other grouping is refused at issue, where the
     * plugin names each run. Runs must also be contiguous (checked when tables are bound) and of
     * one shape (checked at first issue). Too few weights for the tables is refused here. */
    int per_tab = 0, tab_extra = 0;
    if (n_tab > 0) {
        const int rest = n_w - w_max;
        if (rest < n_tab) {
            err(RAD_E_SCHEMA,
                "op '%s': %d weight operands declared; the schema has %d fixed weight operands and "
                "%d weight TABLE operands, so %d leaves a table empty. A table takes one weight "
                "per expert", opname, n_w, w_max, n_tab, rest);
        } else {
            per_tab = rest / n_tab;
            tab_extra = rest % n_tab;
        }
    } else if (n_w < w_req || n_w > w_max) {
        err(RAD_E_SCHEMA, "op '%s': %d weight operands declared, but the schema has %d (%d "
                          "required) among its %d operands", opname, n_w, w_max, w_req,
            sc->n_operands);
    }

    for (int i = 0; i < n_w; ++i)
        if (w[i] == 0 || w[i] >= prog_.weights.size())
            err(RAD_E_INVAL, "op '%s': weight operand %d is handle %u, which this builder did not "
                             "hand out", opname, i, w[i]);

    if (errors_.size() != before) return RAD_NULL_HANDLE;

    /* ---- resolve. */
    OpInfo oi;
    oi.op         = opname;
    oi.base       = std::move(base);
    oi.schema     = sc;
    oi.index      = (int32_t)prog_.ops.size();
    if (!ranged_key.empty()) {
        oi.ranged_key = ranged_key;
        oi.range_lo   = lo;
        oi.range_hi   = hi;
    }
    oi.weights.assign(w, w + n_w);
    /* Declared weight -> schema operand position (OpInfo::weight_opd). A fixed position takes one
     * weight; a table position takes `per_tab`, plus one for the first `tab_extra` tables. */
    oi.weight_opd.assign((size_t)n_w, -1);
    {
        int wi = 0, ti = 0;
        for (int k = 0; sc->operands && k < sc->n_operands && wi < n_w; ++k) {
            const int role = sc->operands[k].role;
            if (role == RAD_OPD_WTAB) {
                const int run = per_tab + (ti++ < tab_extra ? 1 : 0);
                for (int j = 0; j < run && wi < n_w; ++j) oi.weight_opd[(size_t)wi++] = k;
            } else if (role == RAD_OPD_WEIGHT) {
                oi.weight_opd[(size_t)wi++] = k;
            }
        }
    }

    /* Runtime LoRA: a fused GEMM over a targeted weight is declined. Every `gemm_*` op other than
     * the four an adapter delta can follow (lora::op_is_site) fuses something into its epilogue
     * (the gated fold quantises the SiLU product in place), so the delta has no y to land in. The
     * op is recorded unresolved, as a library without the kernel would leave it, and the plugin
     * runs its unfused chain. The decision reads only the weight's own declaration and name map,
     * so the sizing declare makes the same one. */
    bool lora_declined = false;
    if (lora_targets_ && n_w > 0 && oi.op.rfind("gemm_", 0) == 0 && oi.op != "gemm_nt" &&
        oi.op != "gemm_nt_bias" && oi.op != "gemm_nt_q" && oi.op != "gemm_nt_q_bias") {
        for (int i = 0; i < n_w && !lora_declined; ++i)
            if (w[i] > 0 && w[i] < prog_.weights.size() &&
                lora::weight_targeted(prog_, w[i], *lora_targets_))
                lora_declined = true;
    }

    if (lora_declined) {
        Band none;
        none.hi = oi.ranged_key.empty() ? INT64_MAX : oi.range_hi;
        for (int d = 0; d < RAD_N_DOMAINS; ++d)
            none.miss[d] = "declined: its weight is a runtime-LoRA target, and a fused GEMM has no "
                           "output for the adapter's delta (the plugin runs the unfused chain)";
        oi.bands.assign(1, none);
        oi.lora_declined = true;
        /* It owns no weight: the planner reads an op's weights, and an op with no kernel anywhere
         * would tell it the weight cannot be read on the card, moving the gate_up the unfused
         * chain multiplies off the card. The op stays in the list by name so the sizing declare
         * lines up. */
        oi.weights.clear();
        oi.weight_opd.clear();
        prog_.ops.push_back(std::move(oi));
        op_bufs_.emplace_back();
        return RAD_NULL_HANDLE;
    }

    if (ref_) {
        const size_t at = prog_.ops.size();
        if (at >= ref_->ops.size() || ref_->ops[at].op != oi.op)
            return err(RAD_E_STATE, "sizing declare: op %zu is '%s', and the real declare had "
                                    "'%s' there -- the plugin declares a different graph at a "
                                    "smaller max_tok", at, opname,
                       at < ref_->ops.size() ? ref_->ops[at].op.c_str() : "nothing");
        /* Fixed parameters must match too: the kernel runs with the real declare's. A fixed row
         * count that follows max_tok would be launched at the real size over the smaller level's
         * buffers and write past them into lent arena memory. */
        const Geometry& rb = ref_->ops[at].base;
        auto role_of = [&](const char* key) {
            for (int k = 0; sc->params && k < sc->n_params; ++k)
                if (sc->params[k].key && std::strcmp(sc->params[k].key, key) == 0)
                    return sc->params[k].role;
            return (int)RAD_PROLE_NONE;
        };
        const RadParam* ra = rb.params();
        const RadParam* rq = oi.base.params();
        const int na = rb.n_params(), nq = oi.base.n_params();
        std::string moved = na == nq ? "" : "the parameter list itself";
        for (int x = 0; x < na && moved.empty(); ++x) {
            const RadParam* y = nullptr;
            for (int k = 0; k < nq; ++k)
                if (std::strcmp(rq[k].key, ra[x].key) == 0) { y = &rq[k]; break; }
            if (!y || y->kind != ra[x].kind) { moved = ra[x].key; break; }
            bool same = false;
            switch (ra[x].kind) {
                case RAD_P_INT:
                    /* A capacity may shrink: its kernel writes by its operands' extents. */
                    same = y->ival == ra[x].ival ||
                           (role_of(ra[x].key) == RAD_PROLE_CAPACITY && y->ival <= ra[x].ival);
                    break;
                case RAD_P_STR: same = std::strcmp(y->sval ? y->sval : "",
                                                   ra[x].sval ? ra[x].sval : "") == 0; break;
                default:        same = y->dval == ra[x].dval; break;
            }
            if (!same) moved = ra[x].key;
        }
        if (!moved.empty())
            return err(RAD_E_STATE, "sizing declare: op %zu ('%s') changes '%s' with the step size "
                                    "(%s at the smaller max_tok, %s at the real one), and its "
                                    "kernel was chosen and runs with the real one", at, opname,
                       moved.c_str(), oi.base.str().c_str(), rb.str().c_str());
        oi.bands = ref_->ops[at].bands;
    } else {
        resolve_bands(oi);
    }

    rad_op h = (rad_op)prog_.ops.size();
    bool resolved = true;
    for (const Band& b : oi.bands)
        if (!b.dom[RAD_DOMAIN_DEVICE] && !b.dom[RAD_DOMAIN_HOST]) resolved = false;

    prog_.ops.push_back(std::move(oi));
    op_bufs_.emplace_back();

    /* An unresolved op is recorded (the graph dump shows it) but returns a null handle, so the
     * plugin can declare the unfused sequence instead. It becomes an error only if issued
     * (spec §2.3, §3.2). */
    return resolved ? h : RAD_NULL_HANDLE;
}

int Builder::op_declared(const char* op) const {
    if (kvstream_op_gated(prog_.ctx, op)) return 0; /* kvstream_op_gated */
    return op && reg_.schema(op) ? 1 : 0;
}

int Builder::op_resolved(rad_op h) const {
    if (h == 0 || h >= prog_.ops.size()) return 0;
    const OpInfo& o = prog_.ops[h];
    if (o.bands.empty()) return 0;
    for (const Band& b : o.bands)
        if (!b.dom[RAD_DOMAIN_DEVICE] && !b.dom[RAD_DOMAIN_HOST]) return 0;
    return 1;
}

int Builder::op_reads(rad_op h, const rad_buf* b, int n) {
    if (h == 0 || h >= prog_.ops.size()) return RAD_E_INVAL;
    for (int i = 0; i < n; ++i) {
        if (b[i] == 0 || b[i] >= prog_.buffers.size()) return RAD_E_INVAL;
        op_bufs_[h].reads.push_back(b[i]);
    }
    return RAD_OK;
}

int Builder::op_writes(rad_op h, const rad_buf* b, int n) {
    if (h == 0 || h >= prog_.ops.size()) return RAD_E_INVAL;
    for (int i = 0; i < n; ++i) {
        if (b[i] == 0 || b[i] >= prog_.buffers.size()) return RAD_E_INVAL;
        op_bufs_[h].writes.push_back(b[i]);
    }
    return RAD_OK;
}

/* See rad_buf_concurrent. Recorded here and applied by liveness at finish(). */
int Builder::buf_concurrent(rad_buf b) {
    if (finished_) { err(RAD_E_STATE, "buf_concurrent after declare finished"); return RAD_E_STATE; }
    if (b == 0 || b >= prog_.buffers.size()) return RAD_E_INVAL;
    concurrent_bufs_.insert(b);
    return RAD_OK;
}

/* See rad_lane1_window_begin. The window is the ops declared between the two calls. */
int Builder::lane1_window(bool end) {
    if (finished_) { err(RAD_E_STATE, "rad_lane1_window after declare finished"); return RAD_E_STATE; }
    const int32_t next = (int32_t)prog_.ops.size();   /* the next op's handle */
    if (!end) {
        if (lane1_w0_ >= 0) {
            err(RAD_E_STATE, "rad_lane1_window_begin called twice: one window a program");
            return RAD_E_STATE;
        }
        lane1_w0_ = next;
    } else {
        if (lane1_w0_ < 0 || lane1_w1_ >= 0 || next - 1 < lane1_w0_) {
            err(RAD_E_STATE, "rad_lane1_window_end without a begin, twice, or around no op");
            return RAD_E_STATE;
        }
        lane1_w1_ = next - 1;
    }
    return RAD_OK;
}

/* See rad_weight_shard_span. Checked against the declaration here, where the plugin can be named;
 * the loader checks it against the container. */
int Builder::weight_shard_span(rad_weight h, int64_t lo, int64_t hi) {
    if (finished_) { err(RAD_E_STATE, "weight_shard_span after declare finished"); return RAD_E_STATE; }
    if (h == 0 || h >= prog_.weights.size()) return RAD_E_INVAL;
    WeightInfo& w = prog_.weights[h];
    const RadWeightDecl& d = w.decl;
    if (d.shard != RAD_SHARD_ROW && d.shard != RAD_SHARD_COL) {
        err(RAD_E_INVAL, "weight '%s': a shard span on a weight that is not row- or column-sharded",
            w.name.c_str());
        return RAD_E_INVAL;
    }
    if (lo < 0 || hi <= lo) {
        err(RAD_E_INVAL, "weight '%s': shard span [%lld, %lld) is empty", w.name.c_str(),
            (long long)lo, (long long)hi);
        return RAD_E_INVAL;
    }
    /* The declared extent is this rank's, so it must equal the span's length: each stacked part's
     * rows for ROW, the columns for COL. */
    bool ok;
    if (d.shard == RAD_SHARD_ROW) {
        if (d.n_row_parts > 1) {
            ok = true;
            for (int g = 0; g < d.n_row_parts; ++g) ok = ok && d.row_parts[g] == hi - lo;
        } else {
            ok = d.rank >= 1 && d.shape[0] == hi - lo;
        }
    } else {
        ok = d.rank == 2 && d.shape[1] == hi - lo;
    }
    if (!ok) {
        err(RAD_E_SHAPE, "weight '%s': shard span [%lld, %lld) is %lld wide and the declared "
                         "extent of the split dimension is not", w.name.c_str(), (long long)lo,
            (long long)hi, (long long)(hi - lo));
        return RAD_E_SHAPE;
    }
    w.shard_lo = lo;
    w.shard_hi = hi;
    return RAD_OK;
}

int Builder::weight_shard_span_parts(rad_weight h, const int64_t* lo, const int64_t* hi,
                                     const int64_t* total, int n) {
    if (finished_) { err(RAD_E_STATE, "weight_shard_span_parts after declare finished"); return RAD_E_STATE; }
    if (h == 0 || h >= prog_.weights.size() || !lo || !hi || !total) return RAD_E_INVAL;
    WeightInfo& w = prog_.weights[h];
    const RadWeightDecl& d = w.decl;
    if (d.shard != RAD_SHARD_ROW || d.n_row_parts < 2 || n != d.n_row_parts) {
        err(RAD_E_INVAL, "weight '%s': per-part spans need a row-sharded weight of %d stacked "
                         "parts, and %d span(s) were given", w.name.c_str(), d.n_row_parts, n);
        return RAD_E_INVAL;
    }
    for (int g = 0; g < n; ++g) {
        if (lo[g] < 0 || hi[g] <= lo[g] || hi[g] > total[g]) {
            err(RAD_E_INVAL, "weight '%s': part %d span [%lld, %lld) is empty or outside its "
                             "%lld rows", w.name.c_str(), g, (long long)lo[g], (long long)hi[g],
                (long long)total[g]);
            return RAD_E_INVAL;
        }
        if (d.row_parts[g] != hi[g] - lo[g]) {
            err(RAD_E_SHAPE, "weight '%s': part %d span [%lld, %lld) is %lld wide and the "
                             "declared part is %lld", w.name.c_str(), g, (long long)lo[g],
                (long long)hi[g], (long long)(hi[g] - lo[g]), (long long)d.row_parts[g]);
            return RAD_E_SHAPE;
        }
        w.part_lo[g] = lo[g];
        w.part_hi[g] = hi[g];
        w.part_total[g] = total[g];
    }
    w.n_part_spans = n;
    w.shard_lo = 0;
    w.shard_hi = 0;
    return RAD_OK;
}

int Builder::kv_group_zero(rad_kvgroup h) {
    if (finished_) { err(RAD_E_STATE, "kv_group_zero after declare finished"); return RAD_E_STATE; }
    if (h == 0 || h >= prog_.kv_groups.size()) return RAD_E_INVAL;
    /* Paged groups only. A zero group is one this rank never reads or writes; for a paged group
     * the other ranks just index the shared block table. A linear or conv group's per-sequence
     * state is saved and cleared on every rank, so a zero-sized one would mismatch those
     * copies. */
    const int kind = prog_.kv_groups[h].decl.kind;
    if (kind != RAD_KV_FULL && kind != RAD_KV_WINDOW && kind != RAD_KV_RING) {
        err(RAD_E_INVAL, "kv group '%s': kv_group_zero on a group that is not paged",
            prog_.kv_groups[h].name.c_str());
        return RAD_E_INVAL;
    }
    if (prog_.kv_groups[h].replicated) {
        err(RAD_E_INVAL, "kv group '%s': kv_group_zero on a group declared replicated; a "
            "replicated group is held whole on every rank", prog_.kv_groups[h].name.c_str());
        return RAD_E_INVAL;
    }
    prog_.kv_groups[h].zero = true;
    return RAD_OK;
}

int Builder::kv_group_dense_read(rad_kvgroup h) {
    if (finished_) { err(RAD_E_STATE, "kv_group_dense_read after declare finished"); return RAD_E_STATE; }
    if (h == 0 || h >= prog_.kv_groups.size()) return RAD_E_INVAL;
    /* Full groups only: a state slot is already per sequence, and a window bounds its own
     * history. */
    const KVGroupInfo& gi = prog_.kv_groups[h];
    if (gi.decl.kind != RAD_KV_FULL) {
        err(RAD_E_INVAL, "kv group '%s': dense_read on a group that is not a full attention group",
            gi.name.c_str());
        return RAD_E_INVAL;
    }
    prog_.kv_groups[h].dense_read = true;
    return RAD_OK;
}

int Builder::kv_group_replicated(rad_kvgroup h) {
    if (finished_) {
        err(RAD_E_STATE, "kv_group_replicated after declare finished");
        return RAD_E_STATE;
    }
    if (h == 0 || h >= prog_.kv_groups.size()) {
        err(RAD_E_INVAL, "kv_group_replicated: %u is not a declared KV group", (unsigned)h);
        return RAD_E_INVAL;
    }
    /* Paged groups only: the tiers move blocks. A linear or conv group's state travels in the
     * checkpoint store, which has no such window. */
    const int kind = prog_.kv_groups[h].decl.kind;
    if (kind != RAD_KV_FULL && kind != RAD_KV_WINDOW) {
        err(RAD_E_INVAL, "kv group '%s': kv_group_replicated on a group that is not paged",
            prog_.kv_groups[h].name.c_str());
        return RAD_E_INVAL;
    }
    if (prog_.kv_groups[h].zero) {
        err(RAD_E_INVAL, "kv group '%s': kv_group_replicated on a group declared zero; a zero "
            "rank holds none of it", prog_.kv_groups[h].name.c_str());
        return RAD_E_INVAL;
    }
    prog_.kv_groups[h].replicated = true;
    return RAD_OK;
}

/* ================================================================== name map */
int Builder::decl_name_map(const RadNameMap* m) {
    if (finished_) { err(RAD_E_STATE, "decl_name_map after declare finished"); return RAD_E_STATE; }
    if (!m || !m->declared) { err(RAD_E_INVAL, "decl_name_map: null map"); return RAD_E_INVAL; }
    if (m->n_src < 1 || m->n_src > 8) {
        err(RAD_E_INVAL, "name map for '%s': %d source tensors, 1..8 allowed", m->declared, m->n_src);
        return RAD_E_INVAL;
    }
    if (m->mode != RAD_MAP_COPY && m->mode != RAD_MAP_CONCAT && m->mode != RAD_MAP_WNORM &&
        m->mode != RAD_MAP_ROWS && m->mode != RAD_MAP_TRANSPOSE && m->mode != RAD_MAP_EMA &&
        m->mode != RAD_MAP_TAP && m->mode != RAD_MAP_COLS) {
        err(RAD_E_INVAL, "name map for '%s': unknown mode %d", m->declared, m->mode);
        return RAD_E_INVAL;
    }
    if ((m->mode == RAD_MAP_WNORM || m->mode == RAD_MAP_EMA) && m->n_src != 2) {
        err(RAD_E_INVAL, "name map for '%s': a %s map names two sources (%s), not %d",
            m->declared, m->mode == RAD_MAP_EMA ? "codebook" : "weight-norm",
            m->mode == RAD_MAP_EMA ? "the embedding sums and the cluster usage"
                                   : "the magnitude and the direction", m->n_src);
        return RAD_E_INVAL;
    }
    if ((m->mode == RAD_MAP_TRANSPOSE || m->mode == RAD_MAP_TAP || m->mode == RAD_MAP_COLS) &&
        m->n_src != 1) {
        err(RAD_E_INVAL, "name map for '%s': a %s map names one source, not %d", m->declared,
            m->mode == RAD_MAP_TAP ? "tap" : m->mode == RAD_MAP_COLS ? "column" : "transpose",
            m->n_src);
        return RAD_E_INVAL;
    }

    RadNameMap c{};
    c.declared   = intern(scoped(m->declared));
    c.mode       = m->mode;
    c.n_src      = m->n_src;
    c.concat_dim = m->concat_dim;
    for (int i = 0; i < m->n_src; ++i) {
        if (!m->src[i]) {
            err(RAD_E_INVAL, "name map for '%s': source %d is null", m->declared, i);
            return RAD_E_INVAL;
        }
        c.src[i] = intern(m->src[i]);
    }
    /* This copy is field by field: a field added to RadNameMap must be added here too. A missed
     * src_index would stay 0 ("sub-tensor 0") instead of the ABI default -1 ("whole tensor",
     * also map_base() in arch/common/rad_arch.h), so every map would ask for slice 0. */
    for (int i = 0; i < m->n_src; ++i) c.src_index[i] = m->src_index[i];
    for (int i = m->n_src; i < 8; ++i) c.src_index[i] = -1;
    for (int i = 0; i < 8; ++i) c.src_rows[i] = i < m->n_src ? m->src_rows[i] : 0;
    if (m->mode == RAD_MAP_ROWS)
        for (int i = 0; i < m->n_src; ++i)
            if (m->src_index[i] < 0 || m->src_rows[i] <= 0) {
                err(RAD_E_INVAL, "name map for '%s': row piece %d starts at %d and holds %lld rows",
                    m->declared, i, m->src_index[i], (long long)m->src_rows[i]);
                return RAD_E_INVAL;
            }
    if (m->mode == RAD_MAP_COLS && (m->src_index[0] < 0 || m->src_rows[0] <= 0)) {
        err(RAD_E_INVAL, "name map for '%s': the column range starts at %d and holds %lld columns",
            m->declared, m->src_index[0], (long long)m->src_rows[0]);
        return RAD_E_INVAL;
    }

    /* A CONCAT, ROWS or EMA map may be declared in pieces; any other repeat is an error.
     *
     * `src[8]` is the ABI's cap, and a sharded checkpoint exceeds it: Qwen3.8-Flash-Next stores
     * its n-gram embedding as 128 tensors `...ngram_embedding.shard_K.weight`. Growing the array
     * would cost ~80 MB of Program (one map per declared weight, ~50K weights). So successive
     * CONCAT maps for one name extend it in call order, and rad-convert joins their sources end
     * to end (map_concat_shards() in arch/common/rad_arch.h). ROWS maps extend the same way (a
     * row permutation is one piece a row).
     *
     * Other repeats are refused: a lookup built with std::map::emplace would silently keep the
     * first, so a copy-paste mistake would convert weights from the wrong tensors. */
    auto seen = name_map_first_.find(c.declared);
    if (seen != name_map_first_.end()) {
        const RadNameMap& first = prog_.name_map[seen->second];
        const bool joins = (c.mode == RAD_MAP_CONCAT && first.mode == RAD_MAP_CONCAT) ||
                           (c.mode == RAD_MAP_EMA && first.mode == RAD_MAP_EMA) ||
                           (c.mode == RAD_MAP_ROWS && first.mode == RAD_MAP_ROWS);
        if (!joins) {
            err(RAD_E_INVAL, "name map for '%s': declared twice, and only a CONCAT, ROWS or EMA "
                             "map may be extended -- a repeated COPY map means two sources for one "
                             "weight, and which one wins is not something a declaration can express",
                c.declared);
            return RAD_E_INVAL;
        }
        if (c.concat_dim != first.concat_dim) {
            err(RAD_E_INVAL, "name map for '%s': extended at concat_dim %d having been declared at "
                             "%d; the pieces of one concatenation join along one axis",
                c.declared, c.concat_dim, first.concat_dim);
            return RAD_E_INVAL;
        }
    } else {
        name_map_first_.emplace(c.declared, prog_.name_map.size());
    }

    prog_.name_map.push_back(c);
    return RAD_OK;
}

/* The buffer the core samples from. A second call is refused rather than winning: a target and
 * its drafter declaring into one builder (spec §10) both have logits, and keeping the second would
 * sample the draft distribution. The drafter's logits are reached through its scope. */
int Builder::declare_scores(rad_buf hidden, rad_weight table) {
    if (finished_) { err(RAD_E_STATE, "declare_scores after declare finished"); return RAD_E_STATE; }
    if (hidden == RAD_NULL_HANDLE || (size_t)hidden >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_scores: %u is not a declared buffer", (unsigned)hidden);
        return RAD_E_INVAL;
    }
    if (table == RAD_NULL_HANDLE || (size_t)table >= prog_.weights.size()) {
        err(RAD_E_INVAL, "declare_scores: %u is not a declared weight", (unsigned)table);
        return RAD_E_INVAL;
    }
    const RadBufDecl& h = prog_.buffers[hidden].decl;
    const auto& w = prog_.weights[table];
    if (h.dtype != RAD_BF16 || h.rank != 2 || w.decl.dtype != RAD_BF16 || w.decl.rank != 2 ||
        w.decl.shape[1] != h.shape[1]) {
        err(RAD_E_INVAL, "declare_scores: the hidden rows ('%s') and the table ('%s') must be bf16 "
            "[rows, n_embd] and [n_vocab, n_embd]", prog_.buffers[hidden].name.c_str(),
            w.name.c_str());
        return RAD_E_INVAL;
    }
    /* The table must cover the vocabulary: the scored readout indexes it by token id (the
     * scheduler bounds candidate ids by meta.n_vocab), and a table a few thousand rows short
     * would turn the last block of the vocabulary into a read off the end of a weight — a device
     * fault, not a refusal. More rows than the vocabulary is padding and reads fine. */
    if (w.decl.shape[0] < prog_.meta.n_vocab) {
        err(RAD_E_INVAL, "declare_scores: the table ('%s') has %lld rows, under the %lld-token "
            "vocabulary the scored readout indexes it by", w.name.c_str(),
            (long long)w.decl.shape[0], (long long)prog_.meta.n_vocab);
        return RAD_E_INVAL;
    }
    prog_.score_hidden = hidden;
    prog_.score_table  = table;
    return RAD_OK;
}

int Builder::declare_logits(rad_buf b) {
    if (finished_) { err(RAD_E_STATE, "declare_logits after declare finished"); return RAD_E_STATE; }
    if (b == RAD_NULL_HANDLE || (size_t)b >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_logits: %u is not a declared buffer", (unsigned)b);
        return RAD_E_INVAL;
    }
    if (prog_.logits_buf != RAD_NULL_HANDLE && prog_.logits_buf != b) {
        err(RAD_E_STATE, "declare_logits: '%s' was already named as the logits buffer; '%s' is a "
                         "second one. A scoped sub-model's logits are its own, not the step's.",
            prog_.buffers[prog_.logits_buf].name.c_str(), prog_.buffers[b].name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.pooled_buf != RAD_NULL_HANDLE) {
        err(RAD_E_STATE, "declare_logits: this program already declared '%s' as its pooled "
                         "output; a program generates or embeds, not both",
            prog_.buffers[prog_.pooled_buf].name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.generator.layout != 0) {
        err(RAD_E_STATE, "declare_logits: this program already declared the image generator '%s'",
            prog_.gen_str.name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.is_audio_generator()) {
        err(RAD_E_STATE, "declare_logits: this program already declared the audio generator '%s'",
            prog_.agen_str.name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.is_video_generator()) {
        err(RAD_E_STATE, "declare_logits: this program already declared the video generator '%s'",
            prog_.vgen_str.name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.is_shape_generator()) {
        err(RAD_E_STATE, "declare_logits: this program already declared the shape generator '%s'",
            prog_.sgen_name.c_str());
        return RAD_E_STATE;
    }
    prog_.logits_buf = b;
    return RAD_OK;
}

/* The embedding program's output (rad_declare_pooled). Checked as declare_logits is, and for
 * shape: the core reads `n_seq` f32 rows from the device after every step, so the wrong dtype,
 * domain or too few rows is refused now rather than misread at the first request. */
int Builder::declare_pooled(rad_buf b) {
    if (finished_) { err(RAD_E_STATE, "declare_pooled after declare finished"); return RAD_E_STATE; }
    if (b == RAD_NULL_HANDLE || (size_t)b >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_pooled: %u is not a declared buffer", (unsigned)b);
        return RAD_E_INVAL;
    }
    if (prog_.pooled_buf != RAD_NULL_HANDLE && prog_.pooled_buf != b) {
        err(RAD_E_STATE, "declare_pooled: '%s' was already named as the pooled output; '%s' is a "
                         "second one", prog_.buffers[prog_.pooled_buf].name.c_str(),
            prog_.buffers[b].name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.logits_buf != RAD_NULL_HANDLE) {
        err(RAD_E_STATE, "declare_pooled: this program already declared '%s' as its logits; a "
                         "program generates or embeds, not both",
            prog_.buffers[prog_.logits_buf].name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.generator.layout != 0) {
        err(RAD_E_STATE, "declare_pooled: this program already declared the image generator '%s'",
            prog_.gen_str.name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.is_audio_generator()) {
        err(RAD_E_STATE, "declare_pooled: this program already declared the audio generator '%s'",
            prog_.agen_str.name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.is_video_generator()) {
        err(RAD_E_STATE, "declare_pooled: this program already declared the video generator '%s'",
            prog_.vgen_str.name.c_str());
        return RAD_E_STATE;
    }
    if (prog_.is_shape_generator()) {
        err(RAD_E_STATE, "declare_pooled: this program already declared the shape generator '%s'",
            prog_.sgen_name.c_str());
        return RAD_E_STATE;
    }
    const RadBufDecl& d = prog_.buffers[b].decl;
    if (d.dtype != RAD_F32 || d.domain != RAD_DOMAIN_DEVICE || d.rank != 2 ||
        d.shape[0] < prog_.ctx.max_seqs || d.shape[1] <= 0) {
        err(RAD_E_INVAL, "declare_pooled: '%s' must be a device f32 buffer of [>= %lld, dim] -- "
                         "one row a sequence of the step",
            prog_.buffers[b].name.c_str(), (long long)prog_.ctx.max_seqs);
        return RAD_E_INVAL;
    }
    prog_.pooled_buf = b;
    return RAD_OK;
}

/* The words a request may hand the plugin (rad_declare_plugin_words). A bound, not a buffer: the
 * batch builder stages them per sequence beside the step's other arrays. */
int Builder::declare_plugin_words(int64_t max_words) {
    if (finished_) { err(RAD_E_STATE, "declare_plugin_words after declare finished"); return RAD_E_STATE; }
    if (max_words < 0 || max_words > (1 << 20)) {
        err(RAD_E_INVAL, "declare_plugin_words: %lld words a request; the bound is 0 to %d",
            (long long)max_words, 1 << 20);
        return RAD_E_INVAL;
    }
    prog_.plugin_words = max_words;
    return RAD_OK;
}

/* Validates the encoder at declare: it runs only when a request carries media, so a bad
 * declaration would otherwise surface on the first image as an out-of-bounds pass.
 *
 * `max_patches` must equal RadBuildCtx::max_enc_patches (or max_audio_frames), which the plugin
 * sized its activations with and the core stages up to. `out` must hold a whole pass's rows, which
 * the core reads back. */
int Builder::declare_encoder(const RadEncoderDecl* d) {
    if (finished_) { err(RAD_E_STATE, "declare_encoder after declare finished"); return RAD_E_STATE; }
    if (!d) { err(RAD_E_INVAL, "declare_encoder: null declaration"); return RAD_E_INVAL; }
    if (d->modalities == 0) return RAD_OK;   /* declaring nothing is declaring nothing */
    const char* nm = d->name && *d->name ? d->name : "(unnamed)";
    /* One vision encoder (images and video) and one audio encoder, each at most once. */
    const bool audio = d->modalities == RAD_MM_AUDIO;
    if (!audio && (d->modalities & ~(uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO)) != 0) {
        err(RAD_E_INVAL, "declare_encoder '%s': modalities 0x%x names a kind this engine has no "
                         "processor for (an audio encoder serves audio alone)", nm,
            (unsigned)d->modalities);
        return RAD_E_INVAL;
    }
    RadEncoderDecl& slot = audio ? prog_.audio_encoder : prog_.encoder;
    std::string& slot_name = audio ? prog_.audio_encoder_name : prog_.encoder_name;
    if (slot.modalities != 0) {
        err(RAD_E_STATE, "declare_encoder: '%s' already declared this program's %s encoder; '%s' "
                         "is a second one", slot_name.c_str(), audio ? "audio" : "image and video",
            nm);
        return RAD_E_STATE;
    }
    const int64_t want = audio ? prog_.ctx.max_audio_frames : prog_.ctx.max_enc_patches;
    if (d->patch_dim <= 0 || d->merge <= 0 || d->n_embd <= 0) {
        err(RAD_E_INVAL, "declare_encoder '%s': patch_dim %lld, merge %lld, n_embd %lld -- all "
                         "three must be positive", nm, (long long)d->patch_dim,
            (long long)d->merge, (long long)d->n_embd);
        return RAD_E_INVAL;
    }
    if (d->max_patches <= 0 || d->max_patches != want || d->max_patches % d->merge != 0) {
        err(RAD_E_INVAL, "declare_encoder '%s': max_patches %lld, but this program was declared "
                         "with %s %lld and a merge of %lld. The plugin sized its activations with "
                         "that and the core stages up to that many patches a pass; they cannot "
                         "differ, and a pass's patches must divide into whole output rows.", nm,
            (long long)d->max_patches, audio ? "max_audio_frames" : "max_enc_patches",
            (long long)want, (long long)d->merge);
        return RAD_E_INVAL;
    }
    if (d->out == RAD_NULL_HANDLE || (size_t)d->out >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_encoder '%s': out %u is not a declared buffer", nm,
            (unsigned)d->out);
        return RAD_E_INVAL;
    }
    const RadBufDecl& ob = prog_.buffers[d->out].decl;
    const int64_t numel = decl_numel(ob.rank, ob.shape);
    if (ob.dtype != RAD_BF16 || ob.domain != RAD_DOMAIN_DEVICE ||
        numel < d->max_patches / d->merge * d->n_embd) {
        err(RAD_E_INVAL, "declare_encoder '%s': out '%s' must be a device bf16 buffer of at least "
                         "%lld x %lld elements -- one pass's rows", nm,
            prog_.buffers[d->out].name.c_str(), (long long)(d->max_patches / d->merge),
            (long long)d->n_embd);
        return RAD_E_INVAL;
    }
    if (d->one_grid != 0 && d->one_grid != 1) {
        err(RAD_E_INVAL, "declare_encoder '%s': one_grid %d is neither 0 nor 1", nm,
            (int)d->one_grid);
        return RAD_E_INVAL;
    }
    if (d->bidirectional != 0 && (d->bidirectional != 1 || audio)) {
        err(RAD_E_INVAL, "declare_encoder '%s': bidirectional %d -- 0, or 1 for a picture and video "
                         "encoder", nm, (int)d->bidirectional);
        return RAD_E_INVAL;
    }
    slot = *d;
    slot_name = nm;
    slot.name = nullptr;   /* the plugin's storage; the name is the copy */
    return RAD_OK;
}

/* Validates the whole generator at declare rather than at the first request (e.g. an image
 * buffer too small for the largest picture). The strings are copied because the Program outlives
 * the plugin's declare-time storage. */
int Builder::declare_generator(const RadGeneratorDecl* d) {
    if (finished_) { err(RAD_E_STATE, "declare_generator after declare finished"); return RAD_E_STATE; }
    if (!d) { err(RAD_E_INVAL, "declare_generator: null declaration"); return RAD_E_INVAL; }
    const char* nm = d->name && *d->name ? d->name : "(unnamed)";
    if (prog_.generator.layout != 0) {
        err(RAD_E_STATE, "declare_generator: '%s' was already declared; '%s' is a second one",
            prog_.gen_str.name.c_str(), nm);
        return RAD_E_STATE;
    }
    if (prog_.logits_buf != RAD_NULL_HANDLE || prog_.pooled_buf != RAD_NULL_HANDLE ||
        prog_.is_audio_generator() || prog_.is_video_generator() || prog_.is_shape_generator()) {
        err(RAD_E_STATE, "declare_generator '%s': this program already declared %s; a program "
                         "generates text, embeds, makes images, audio, video or shapes -- one of "
                         "the six", nm, prog_.logits_buf != RAD_NULL_HANDLE ? "logits"
                : prog_.pooled_buf != RAD_NULL_HANDLE ? "a pooled output"
                : prog_.is_audio_generator() ? "an audio generator"
                : prog_.is_video_generator() ? "a video generator" : "a shape generator");
        return RAD_E_STATE;
    }
    if (d->layout != RAD_GEN_LAYOUT_JOINT_GRID && d->layout != RAD_GEN_LAYOUT_JOINT_STEP) {
        err(RAD_E_INVAL, "declare_generator '%s': layout %d is not one this engine builds", nm,
            (int)d->layout);
        return RAD_E_INVAL;
    }
    const bool joint = d->layout == RAD_GEN_LAYOUT_JOINT_STEP;
    auto has = [](const char* t, const char* what) { return t && std::strstr(t, what); };
    if (joint ? !has(d->tmpl_text, "{prompt}") || !has(d->tmpl_edit, "{prompt}")
              : !has(d->tmpl_text, "{prompt}") || !has(d->tmpl_edit, "{prompt}") ||
                    !has(d->tmpl_edit, "{images}") || !has(d->tmpl_image, "{image}")) {
        err(RAD_E_INVAL, "declare_generator '%s': tmpl_text and tmpl_edit need {prompt}, and "
                         "with the pictures in the prompt tmpl_edit needs {images} and tmpl_image "
                         "{image}, where the picture's tokens go", nm);
        return RAD_E_INVAL;
    }
    if (joint && (d->cond_rows <= 0 || !(d->rope_hw_scale > 0) || d->pad_multiple < 0 ||
                  (d->pad_multiple > 0 && d->pad_token < 0))) {
        err(RAD_E_INVAL, "declare_generator '%s': the joint layout needs cond_rows, a positive "
                         "rope_hw_scale and, with padding, a pad token (cond_rows %lld, scale %g, "
                         "pad %d to a multiple of %d)", nm, (long long)d->cond_rows,
            d->rope_hw_scale, (int)d->pad_token, (int)d->pad_multiple);
        return RAD_E_INVAL;
    }
    if (d->latent_channels <= 0 || d->latent_scale <= 0 || d->size_multiple <= 0 ||
        d->size_multiple % d->latent_scale != 0 || d->pixel_channels <= 0 ||
        d->pixel_channels > 4 || d->slot_rows <= 0 || d->max_tokens <= 0 || d->max_refs < 0 ||
        d->ref_area <= 0 || d->default_width <= 0 || d->default_height <= 0 ||
        d->default_width % d->size_multiple != 0 || d->default_height % d->size_multiple != 0 ||
        (d->default_width / d->latent_scale) * (d->default_height / d->latent_scale) > d->max_tokens ||
        d->max_lanes <= 0) {
        err(RAD_E_INVAL, "declare_generator '%s': a geometry field is out of range (latent %lld x "
                         "%lld px, multiple %lld, %lld channels, slot rows %lld, max %lld tokens, "
                         "default %lldx%lld, %lld lanes)", nm, (long long)d->latent_channels,
            (long long)d->latent_scale, (long long)d->size_multiple, (long long)d->pixel_channels,
            (long long)d->slot_rows, (long long)d->max_tokens, (long long)d->default_width,
            (long long)d->default_height, (long long)d->max_lanes);
        return RAD_E_INVAL;
    }
    if (d->train_steps < 0) {
        err(RAD_E_INVAL, "declare_generator '%s': train_steps %d is negative", nm,
            (int)d->train_steps);
        return RAD_E_INVAL;
    }
    if (d->train_steps > 0) {
        /* A discrete-time denoiser: its own schedule, a sampler it serves by default. */
        const uint32_t all = (1u << RAD_GEN_SAMPLER_COUNT) - 1;
        if (d->train_steps < 2 || !(d->beta_start > 0) || !(d->beta_end > d->beta_start) ||
            !(d->beta_end < 1) ||
            (d->beta_schedule != RAD_GEN_BETAS_SCALED_LINEAR &&
             d->beta_schedule != RAD_GEN_BETAS_LINEAR) ||
            d->n_sigmas != 0 || d->default_steps <= 0 || (d->samplers & ~all) || !d->samplers ||
            d->default_sampler < 0 || d->default_sampler >= RAD_GEN_SAMPLER_COUNT ||
            !(d->samplers & (1u << d->default_sampler)) || d->default_scheduler < 0 ||
            d->default_scheduler >= RAD_GEN_SCHED_COUNT ||
            !(d->default_cfg >= (d->cfg_original ? 0.0 : 1.0))) {
            err(RAD_E_INVAL, "declare_generator '%s': the discrete-time sampler is incomplete (%d "
                             "timesteps, betas %g..%g kind %d, %lld default steps, samplers 0x%x "
                             "default %d, scheduler %d, %d fixed sigmas, guidance %g)", nm,
                (int)d->train_steps, d->beta_start, d->beta_end, (int)d->beta_schedule,
                (long long)d->default_steps, (unsigned)d->samplers, (int)d->default_sampler,
                (int)d->default_scheduler, (int)d->n_sigmas, d->default_cfg);
            return RAD_E_INVAL;
        }
    } else if (d->n_sigmas < 0 || (d->n_sigmas > 0 && !d->sigmas) ||
        (d->n_sigmas == 0 && d->default_steps <= 0) ||
        !(d->default_cfg >= (d->cfg_original ? 0.0 : 1.0)) ||
        (d->dyn_shift && (d->max_seq <= d->base_seq || d->base_seq <= 0)) ||
        (!d->dyn_shift && !(d->shift > 0)) || d->shift_terminal < 0 || d->shift_terminal >= 1) {
        err(RAD_E_INVAL, "declare_generator '%s': the sampler is incomplete (%d fixed sigmas, %lld "
                         "default steps, shift %g, dynamic %d over %lld..%lld, terminal %g, "
                         "guidance %g)", nm, (int)d->n_sigmas, (long long)d->default_steps,
            d->shift, (int)d->dyn_shift, (long long)d->base_seq, (long long)d->max_seq,
            d->shift_terminal, d->default_cfg);
        return RAD_E_INVAL;
    }
    if (d->sigma_grid < 0 || d->min_tokens < 0 || (d->aux_tokens && d->aux_max_tokens <= 0)) {
        err(RAD_E_INVAL, "declare_generator '%s': sigma_grid %d, min_tokens %d and a second "
                         "tokenizer of at most %lld tokens are not a schedule and a prompt it can "
                         "read", nm, (int)d->sigma_grid, (int)d->min_tokens,
            (long long)d->aux_max_tokens);
        return RAD_E_INVAL;
    }
    if (d->chunk_tokens < 0 ||
        (d->chunk_tokens > 0 &&
         (!joint || d->pad_multiple != 0 || d->pad_token < 0 || d->chunk_bos < 0 ||
          d->chunk_eos < 0 || d->cond_rows % (d->chunk_tokens + 2) != 0))) {
        err(RAD_E_INVAL, "declare_generator '%s': a prompt in windows of %d tokens needs the joint "
                         "layout without pad_multiple, bos, eos and pad tokens, and cond_rows a "
                         "whole number of windows (cond_rows %lld)", nm, (int)d->chunk_tokens,
            (long long)d->cond_rows);
        return RAD_E_INVAL;
    }
    for (int i = 0; i < d->n_sigmas; ++i)
        if (!(d->sigmas[i] > 0.0 && d->sigmas[i] <= 1.0) ||
            (i > 0 && !(d->sigmas[i] < d->sigmas[i - 1]))) {
            err(RAD_E_INVAL, "declare_generator '%s': fixed sigma %d (%g) is not in (0, 1] below "
                             "the one before it", nm, i, d->sigmas[i]);
            return RAD_E_INVAL;
        }
    if (d->image_out == RAD_NULL_HANDLE || (size_t)d->image_out >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_generator '%s': image_out %u is not a declared buffer", nm,
            (unsigned)d->image_out);
        return RAD_E_INVAL;
    }
    const RadBufDecl& ob = prog_.buffers[d->image_out].decl;
    const int64_t numel = decl_numel(ob.rank, ob.shape);
    const int64_t px = d->max_tokens * d->latent_scale * d->latent_scale;
    if (ob.dtype != RAD_U8 || ob.domain != RAD_DOMAIN_DEVICE || numel < px * d->pixel_channels) {
        err(RAD_E_INVAL, "declare_generator '%s': image_out '%s' must be a device u8 buffer of at "
                         "least %lld x %lld -- the largest image", nm,
            prog_.buffers[d->image_out].name.c_str(), (long long)px, (long long)d->pixel_channels);
        return RAD_E_INVAL;
    }
    Program::GenStrings& g = prog_.gen_str;
    g.name             = nm;
    g.tmpl_text        = d->tmpl_text;
    g.tmpl_edit        = d->tmpl_edit;
    g.tmpl_image       = d->tmpl_image ? d->tmpl_image : "";
    g.tmpl_image_sep   = d->tmpl_image_sep ? d->tmpl_image_sep : "";
    g.drop_prefix      = d->drop_prefix ? d->drop_prefix : "";
    g.tmpl_transparent = d->tmpl_transparent ? d->tmpl_transparent : "";
    g.has_default_negative = d->default_negative != nullptr;
    g.default_negative = d->default_negative ? d->default_negative : "";
    g.sigmas.assign(d->sigmas, d->sigmas + d->n_sigmas);
    prog_.generator = *d;
    /* The Program is moved after declare, which would invalidate pointers into gen_str, so the
     * decl's string fields are cleared; read them only through gen_str. */
    RadGeneratorDecl& c = prog_.generator;
    c.name = c.tmpl_text = c.tmpl_edit = c.tmpl_image = c.tmpl_image_sep = nullptr;
    c.drop_prefix = c.tmpl_transparent = c.default_negative = nullptr;
    c.sigmas = nullptr;
    return RAD_OK;
}

/* Validated at declare for the reason the image generator is: a buffer too small for the longest
 * window, or a window that never ends, would otherwise show at the first request. */
int Builder::declare_audio_generator(const RadAudioGenDecl* d) {
    if (finished_) {
        err(RAD_E_STATE, "declare_audio_generator after declare finished");
        return RAD_E_STATE;
    }
    if (!d) { err(RAD_E_INVAL, "declare_audio_generator: null declaration"); return RAD_E_INVAL; }
    const char* nm = d->name && *d->name ? d->name : "(unnamed)";
    if (prog_.is_audio_generator()) {
        err(RAD_E_STATE, "declare_audio_generator: '%s' was already declared; '%s' is a second one",
            prog_.agen_str.name.c_str(), nm);
        return RAD_E_STATE;
    }
    if (prog_.logits_buf != RAD_NULL_HANDLE || prog_.pooled_buf != RAD_NULL_HANDLE ||
        prog_.generator.layout != 0 || prog_.is_video_generator() || prog_.is_shape_generator()) {
        err(RAD_E_STATE, "declare_audio_generator '%s': this program already declared %s; a "
                         "program generates text, embeds, makes images, audio, video or shapes -- "
                         "one of the six", nm,
            prog_.logits_buf != RAD_NULL_HANDLE ? "logits"
                : prog_.pooled_buf != RAD_NULL_HANDLE ? "a pooled output"
                : prog_.generator.layout != 0 ? "an image generator"
                : prog_.is_video_generator() ? "a video generator" : "a shape generator");
        return RAD_E_STATE;
    }
    if (!d->tmpl && !d->tmpl_plain) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': no prompt template (tmpl or tmpl_plain)",
            nm);
        return RAD_E_INVAL;
    }
    if (d->tmpl && (!std::strstr(d->tmpl, "{input}") || !std::strstr(d->tmpl, "{instructions}"))) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': tmpl needs {instructions} and {input}", nm);
        return RAD_E_INVAL;
    }
    if (d->tmpl_plain &&
        (!std::strstr(d->tmpl_plain, "{input}") || std::strstr(d->tmpl_plain, "{instructions}"))) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': tmpl_plain is the prompt without "
                         "instructions and needs {input} alone", nm);
        return RAD_E_INVAL;
    }
    /* References: a template with both of its slots, an item with one transcript before its one
     * clip, and the clips' geometry. Two branches may repeat the same reference prompt at
     * guidance 1; reference CFG would need a separate unconditional frame map and is refused. */
    if (d->tmpl_ref) {
        auto once = [](const char* t, const char* k) {
            const char* a = t ? std::strstr(t, k) : nullptr;
            return a && !std::strstr(a + 1, k);
        };
        const char* au = d->ref_item ? std::strstr(d->ref_item, "{audio}") : nullptr;
        const char* tx = d->ref_item ? std::strstr(d->ref_item, "{text}") : nullptr;
        if (!once(d->tmpl_ref, "{references}") || !once(d->tmpl_ref, "{input}") ||
            std::strstr(d->tmpl_ref, "{instructions}") || !once(d->ref_item, "{audio}") ||
            !once(d->ref_item, "{text}") || !au || !tx || tx > au) {
            err(RAD_E_INVAL, "declare_audio_generator '%s': tmpl_ref needs {references} and "
                             "{input} once each, and ref_item {text} before its one {audio}", nm);
            return RAD_E_INVAL;
        }
        if (!d->prompt_whole || d->ref_rate <= 0 || d->ref_hop <= 0 ||
            d->ref_max <= 0 || d->ref_max_samples < d->ref_hop ||
            (d->branches != 1 && (!d->neg_tmpl || d->default_cfg != 1.0 ||
                                  std::strstr(d->neg_tmpl, "{references}")))) {
            err(RAD_E_INVAL, "declare_audio_generator '%s': references need whole prompts and "
                             "guidance 1 (two branches need a plain negative template); clips at "
                             "a rate (%lld Hz) in frames of "
                             "%lld samples, at most %d of them and %lld samples a request", nm,
                (long long)d->ref_rate, (long long)d->ref_hop, (int)d->ref_max,
                (long long)d->ref_max_samples);
            return RAD_E_INVAL;
        }
    }
    if (d->tmpl_ref_instruct || d->neg_tmpl_ref) {
        auto once = [](const char* t, const char* key) {
            const char* p = t ? std::strstr(t, key) : nullptr;
            return p && !std::strstr(p + 1, key);
        };
        if (!d->tmpl_ref || !d->tmpl_ref_instruct || !d->neg_tmpl_ref || d->branches == 1 ||
            !once(d->tmpl_ref_instruct, "{references}") ||
            !once(d->tmpl_ref_instruct, "{input}") ||
            !once(d->tmpl_ref_instruct, "{instructions}") ||
            !once(d->neg_tmpl_ref, "{references}") || !once(d->neg_tmpl_ref, "{input}") ||
            std::strstr(d->neg_tmpl_ref, "{instructions}")) {
            err(RAD_E_INVAL, "declare_audio_generator '%s': reference direction needs paired "
                             "conditional/reference-only templates with one references/input "
                             "slot each and instructions only in the conditional template", nm);
            return RAD_E_INVAL;
        }
    }
    if (d->branches != 0 && d->branches != 1 && d->branches != 2) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': branches is %d; a request runs 1 sequence "
                         "or 2 (0 means 2)", nm, (int)d->branches);
        return RAD_E_INVAL;
    }
    /* One branch has no unconditional prompt to render and no guidance to apply. */
    if (d->branches == 1 && (d->neg_tmpl || d->default_cfg != 1.0)) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': a model of one branch has no "
                         "unconditional template and no guidance (default_cfg %g)", nm,
            d->default_cfg);
        return RAD_E_INVAL;
    }
    if (d->neg_tmpl && !d->prompt_whole) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': an unconditional template makes branches "
                         "of different lengths, which step together only as whole prompts; set "
                         "prompt_whole", nm);
        return RAD_E_INVAL;
    }
    if (d->max_prompt <= 0 || d->first_frame < 0 ||
        (!d->neg_tmpl && (d->neg_keep_head < 0 || d->neg_keep_tail < 0 || d->neg_token < 0 ||
                          d->neg_keep_head + d->neg_keep_tail >= d->max_prompt))) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': the prompt is out of range (at most %lld "
                         "tokens, unconditional branch token %d keeping %d + %d)", nm,
            (long long)d->max_prompt, (int)d->neg_token, (int)d->neg_keep_head,
            (int)d->neg_keep_tail);
        return RAD_E_INVAL;
    }
    const bool chunked = d->voc_chunk > 0;
    if (chunked && (d->voc_context < 0 || d->lat_num != 1 || d->lat_den != 1)) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': a vocoder that plays frames as they are "
                         "made reads a frame as one latent (lat_num = lat_den = 1) and a context "
                         "of >= 0 frames; got %lld/%lld and %lld", nm, (long long)d->lat_num,
            (long long)d->lat_den, (long long)d->voc_context);
        return RAD_E_INVAL;
    }
    if (d->max_frames <= 0 || d->frame_rate_num <= 0 || d->frame_rate_den <= 0 ||
        (!chunked && (d->win_frames <= 0 || d->win_hop <= 0 || d->win_hop > d->win_frames ||
                      d->win_lead < 0 || d->latent_channels <= 0 || d->win_crop_left < 0 ||
                      d->win_crop_right < 0)) ||
        d->lat_num <= 0 || d->lat_den <= 0 || d->samples_per_latent <= 0 || d->sample_rate <= 0 ||
        d->channels <= 0 || d->max_lanes <= 0) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': a geometry field is out of range (%lld "
                         "frames at %lld/%lld a second, windows of %lld every %lld, %lld lead "
                         "rows, %lld/%lld latents a frame of %lld channels, crops %lld + %lld, "
                         "%lld samples a latent at %lld Hz x %lld, %lld lanes)", nm,
            (long long)d->max_frames, (long long)d->frame_rate_num, (long long)d->frame_rate_den,
            (long long)d->win_frames, (long long)d->win_hop, (long long)d->win_lead,
            (long long)d->lat_num, (long long)d->lat_den, (long long)d->latent_channels,
            (long long)d->win_crop_left, (long long)d->win_crop_right,
            (long long)d->samples_per_latent, (long long)d->sample_rate, (long long)d->channels,
            (long long)d->max_lanes);
        return RAD_E_INVAL;
    }
    /* A middle window plays what is left after both crops, so they must leave some of it. */
    if (!chunked &&
        d->win_crop_left + d->win_crop_right >= d->win_frames * d->lat_num / d->lat_den) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': the crops (%lld + %lld latents) leave a "
                         "window of %lld latents nothing to play", nm, (long long)d->win_crop_left,
            (long long)d->win_crop_right, (long long)(d->win_frames * d->lat_num / d->lat_den));
        return RAD_E_INVAL;
    }
    if ((!chunked && d->default_steps <= 0) || !(d->default_cfg >= 1.0)) {
        err(RAD_E_INVAL, "declare_audio_generator '%s': the sampler is incomplete (%lld default "
                         "steps, guidance %g)", nm, (long long)d->default_steps, d->default_cfg);
        return RAD_E_INVAL;
    }
    auto buf_ok = [&](rad_buf h, uint32_t dtype, int64_t numel, const char* what) {
        if (h == RAD_NULL_HANDLE || (size_t)h >= prog_.buffers.size()) {
            err(RAD_E_INVAL, "declare_audio_generator '%s': %s %u is not a declared buffer", nm,
                what, (unsigned)h);
            return false;
        }
        const RadBufDecl& ob = prog_.buffers[h].decl;
        if (ob.dtype != dtype || ob.domain != RAD_DOMAIN_DEVICE ||
            decl_numel(ob.rank, ob.shape) < numel) {
            err(RAD_E_INVAL, "declare_audio_generator '%s': %s '%s' must be a device %s buffer of "
                             "at least %lld elements", nm, what, prog_.buffers[h].name.c_str(),
                dtype == RAD_I32 ? "i32" : "f32", (long long)numel);
            return false;
        }
        return true;
    };
    /* audio_out holds what one VOCODE pass plays: a window, or a chunk of frames. */
    const int64_t win_samples = (chunked ? d->voc_chunk : d->win_frames * d->lat_num / d->lat_den) *
                                d->samples_per_latent;
    if (!buf_ok(d->status, RAD_I32, d->max_lanes, "status") ||
        !buf_ok(d->audio_out, RAD_F32, win_samples * d->channels, "audio_out"))
        return RAD_E_INVAL;
    prog_.agen_str.name = nm;
    prog_.agen_str.tmpl = d->tmpl ? d->tmpl : "";
    prog_.agen_str.tmpl_plain = d->tmpl_plain ? d->tmpl_plain : "";
    prog_.agen_str.neg_tmpl = d->neg_tmpl ? d->neg_tmpl : "";
    prog_.agen_str.tmpl_ref = d->tmpl_ref ? d->tmpl_ref : "";
    prog_.agen_str.tmpl_ref_instruct = d->tmpl_ref_instruct ? d->tmpl_ref_instruct : "";
    prog_.agen_str.neg_tmpl_ref = d->neg_tmpl_ref ? d->neg_tmpl_ref : "";
    prog_.agen_str.ref_item = d->tmpl_ref && d->ref_item ? d->ref_item : "";
    prog_.audio_gen = *d;
    /* As for the image generator: the Program is moved after declare; strings live in agen_str. */
    prog_.audio_gen.name = nullptr;
    prog_.audio_gen.tmpl = nullptr;
    prog_.audio_gen.tmpl_plain = nullptr;
    prog_.audio_gen.neg_tmpl = nullptr;
    prog_.audio_gen.tmpl_ref = nullptr;
    prog_.audio_gen.tmpl_ref_instruct = nullptr;
    prog_.audio_gen.neg_tmpl_ref = nullptr;
    prog_.audio_gen.ref_item = nullptr;
    /* Without references the clips' geometry is nothing: the driver stages none. */
    if (!d->tmpl_ref) {
        prog_.audio_gen.ref_rate = prog_.audio_gen.ref_hop = prog_.audio_gen.ref_max_samples = 0;
        prog_.audio_gen.ref_max = 0;
    }
    return RAD_OK;
}

/* Validated at declare for the reasons the other generators are: an output buffer too small for
 * the largest chunk or the longest soundtrack, or a geometry whose frame count does not follow
 * from its chunks, would otherwise show at the first request. */
int Builder::declare_video_generator(const RadVideoGenDecl* d) {
    if (finished_) {
        err(RAD_E_STATE, "declare_video_generator after declare finished");
        return RAD_E_STATE;
    }
    if (!d) { err(RAD_E_INVAL, "declare_video_generator: null declaration"); return RAD_E_INVAL; }
    const char* nm = d->name && *d->name ? d->name : "(unnamed)";
    if (prog_.is_video_generator()) {
        err(RAD_E_STATE, "declare_video_generator: '%s' was already declared; '%s' is a second one",
            prog_.vgen_str.name.c_str(), nm);
        return RAD_E_STATE;
    }
    if (prog_.logits_buf != RAD_NULL_HANDLE || prog_.pooled_buf != RAD_NULL_HANDLE ||
        prog_.generator.layout != 0 || prog_.is_audio_generator() || prog_.is_shape_generator()) {
        err(RAD_E_STATE, "declare_video_generator '%s': this program already declared %s; a "
                         "program generates text, embeds, makes images, audio, video or shapes -- "
                         "one of the six", nm,
            prog_.logits_buf != RAD_NULL_HANDLE ? "logits"
                : prog_.pooled_buf != RAD_NULL_HANDLE ? "a pooled output"
                : prog_.generator.layout != 0 ? "an image generator"
                : prog_.is_audio_generator() ? "an audio generator" : "a shape generator");
        return RAD_E_STATE;
    }
    if (!d->tmpl || !std::strstr(d->tmpl, "{prompt}") || d->max_text <= 0) {
        err(RAD_E_INVAL, "declare_video_generator '%s': tmpl needs {prompt} and max_text a "
                         "length (%lld)", nm, (long long)d->max_text);
        return RAD_E_INVAL;
    }
    if (d->fps_num <= 0 || d->fps_den <= 0 || d->frames_per_chunk <= 0 || d->frames_tail < 0 ||
        d->latents_per_chunk <= 0 || d->latents_tail < 0 || d->dec_latents < d->latents_per_chunk ||
        d->min_chunks <= 0 || d->max_chunks < d->min_chunks || d->default_chunks < d->min_chunks ||
        d->default_chunks > d->max_chunks ||
        d->latents_per_chunk * (d->max_chunks - 1) + d->dec_latents >
            d->latents_per_chunk * d->max_chunks + d->latents_tail) {
        err(RAD_E_INVAL, "declare_video_generator '%s': the temporal geometry is out of range "
                         "(%lld/%lld fps, %lld n + %lld frames over %lld n + %lld latents, chunks of "
                         "%lld latents, %lld..%lld chunks, %lld by default)", nm,
            (long long)d->fps_num, (long long)d->fps_den, (long long)d->frames_per_chunk,
            (long long)d->frames_tail, (long long)d->latents_per_chunk, (long long)d->latents_tail,
            (long long)d->dec_latents, (long long)d->min_chunks, (long long)d->max_chunks,
            (long long)d->default_chunks);
        return RAD_E_INVAL;
    }
    const int64_t unit = d->latent_scale * d->patch;
    if (d->latent_scale <= 0 || d->latent_channels <= 0 || d->patch <= 0 ||
        d->size_multiple <= 0 || d->size_multiple % unit != 0 || d->min_side < d->size_multiple ||
        d->max_area < d->min_side * d->min_side || d->max_aspect < 1 ||
        d->default_width % d->size_multiple != 0 || d->default_height % d->size_multiple != 0 ||
        d->default_width < d->min_side || d->default_height < d->min_side ||
        d->default_width * d->default_height > d->max_area) {
        err(RAD_E_INVAL, "declare_video_generator '%s': the spatial geometry is out of range "
                         "(latents of %lld px in %lld x %lld patches, sides multiples of %lld from "
                         "%lld, area to %lld, aspect to %lld, %lld x %lld by default)", nm,
            (long long)d->latent_scale, (long long)d->patch, (long long)d->patch,
            (long long)d->size_multiple, (long long)d->min_side, (long long)d->max_area,
            (long long)d->max_aspect, (long long)d->default_width, (long long)d->default_height);
        return RAD_E_INVAL;
    }
    if (d->sample_rate <= 0 || d->audio_rate <= 0 || d->sample_rate % d->audio_rate != 0 ||
        d->audio_channels <= 0 || d->audio_latent_channels <= 0) {
        err(RAD_E_INVAL, "declare_video_generator '%s': the soundtrack is out of range (%lld Hz "
                         "at %lld latents a second, %lld channels of %lld)", nm,
            (long long)d->sample_rate, (long long)d->audio_rate, (long long)d->audio_channels,
            (long long)d->audio_latent_channels);
        return RAD_E_INVAL;
    }
    /* Form 1 (LTX-2's FlowMatchEulerDiscrete) steps both modalities the same sigmas and reads
     * no per-modality shift. */
    const bool shifts = d->sched_kind == 1 || (d->shift_video > 0.0 && d->shift_audio > 0.0);
    if (d->default_steps <= 0 || d->max_steps < d->default_steps || !shifts || d->max_lanes <= 0) {
        err(RAD_E_INVAL, "declare_video_generator '%s': the sampler is incomplete (%lld steps by "
                         "default, %lld at most, shifts %g and %g, %lld lanes)", nm,
            (long long)d->default_steps, (long long)d->max_steps, d->shift_video, d->shift_audio,
            (long long)d->max_lanes);
        return RAD_E_INVAL;
    }
    /* Form 0 counts grid points, form 1 evaluations. */
    const int64_t rung_steps = d->sched_kind == 1 ? d->n_rungs : d->n_rungs + 1;
    if (d->sched_kind < 0 || d->sched_kind > 1 || d->n_rungs < 0 || (d->n_rungs > 0 && !d->rungs) ||
        (d->n_rungs > 0 && (d->default_steps != rung_steps || d->max_steps != rung_steps))) {
        err(RAD_E_INVAL, "declare_video_generator '%s': a trained schedule of %d rungs runs %lld "
                         "steps, which default_steps and max_steps must both be (%lld, %lld)", nm,
            (int)d->n_rungs, (long long)rung_steps, (long long)d->default_steps,
            (long long)d->max_steps);
        return RAD_E_INVAL;
    }
    if (d->sched_kind == 1 && d->dyn_shift &&
        (!(d->max_seq > d->base_seq) || !std::isfinite(d->base_shift) ||
         !std::isfinite(d->max_shift))) {
        err(RAD_E_INVAL, "declare_video_generator '%s': the dynamic shift runs from %lld tokens "
                         "(%g) to %lld (%g); the second length must be the larger", nm,
            (long long)d->base_seq, d->base_shift, (long long)d->max_seq, d->max_shift);
        return RAD_E_INVAL;
    }
    if (!(d->shift_terminal >= 0.0 && d->shift_terminal < 1.0)) {
        err(RAD_E_INVAL, "declare_video_generator '%s': shift_terminal %g is outside [0, 1)", nm,
            d->shift_terminal);
        return RAD_E_INVAL;
    }
    if (d->n_refine_rungs < 0 || (d->n_refine_rungs > 0 && (!d->refine_rungs ||
        d->sched_kind != 1 || d->n_rungs == 0 || !(d->upscale_max > 1.0)))) {
        err(RAD_E_INVAL, "declare_video_generator '%s': refinement rungs need a trained schedule of "
                         "form 1 and a latent upscaler", nm);
        return RAD_E_INVAL;
    }
    for (int i = 0; i < d->n_refine_rungs; ++i)
        if (!(d->refine_rungs[i] > 0.0 && d->refine_rungs[i] <= 1.0) ||
            (i > 0 && !(d->refine_rungs[i] < d->refine_rungs[i - 1]))) {
            err(RAD_E_INVAL, "declare_video_generator '%s': refinement rung %d (%g) is outside "
                             "(0, 1] or not below the one before it", nm, i, d->refine_rungs[i]);
            return RAD_E_INVAL;
        }
    if (d->step_text_rows < 0 || d->step_text_rows > d->max_text * 64 || d->image_crf < 0 ||
        d->image_crf > 51 || (d->image_cond != 0 && d->image_cond != 1) ||
        (d->guidance != 0 && d->guidance != 1)) {
        err(RAD_E_INVAL, "declare_video_generator '%s': step_text_rows %lld, image_cond %d, "
                         "image_crf %d (0..51) or guidance %d is out of range", nm,
            (long long)d->step_text_rows, (int)d->image_cond, (int)d->image_crf, (int)d->guidance);
        return RAD_E_INVAL;
    }
    if (d->text_left_pad < 0 || (d->text_left_pad > 0 && d->text_left_pad < d->max_text) ||
        d->audio_streams < 0 ||
        (d->upscale_exact && (!(d->upscale_max > 1.0) ||
                              d->upscale_max != std::floor(d->upscale_max)))) {
        err(RAD_E_INVAL, "declare_video_generator '%s': text_left_pad %lld must hold the longest "
                         "prompt (%lld), audio_streams %d may not be negative, and an exact "
                         "upscaler's factor (%g) is a whole number above 1", nm,
            (long long)d->text_left_pad, (long long)d->max_text, (int)d->audio_streams,
            d->upscale_max);
        return RAD_E_INVAL;
    }
    if (d->guidance)
        for (int i = 0; i < 8; ++i)
            if (!std::isfinite(d->guidance_default[i]) || d->guidance_default[i] < 0.0) {
                err(RAD_E_INVAL, "declare_video_generator '%s': default guidance term %d (%g) must "
                                 "be finite and not negative", nm, i, d->guidance_default[i]);
                return RAD_E_INVAL;
            }
    for (int i = 0; i < d->n_rungs; ++i)
        if (!(d->rungs[i] > 0.0 && d->rungs[i] <= 1.0) ||
            (i > 0 && !(d->rungs[i] < d->rungs[i - 1]))) {
            err(RAD_E_INVAL, "declare_video_generator '%s': rung %d (%g) is outside (0, 1] or not "
                             "below the one before it", nm, i, d->rungs[i]);
            return RAD_E_INVAL;
        }
    /* An upscaler needs a schedule it can split: one evaluation at the small size at least, and
     * `upscale_refine` of them after it by default. */
    const int64_t def_evals = d->n_rungs > 0 ? d->n_rungs
                            : d->sched_kind == 1 ? d->default_steps : d->default_steps - 1;
    const bool own_refine = d->n_refine_rungs > 0;
    if (d->upscale_max != 0.0 &&
        (!(d->upscale_max > 1.0) ||
         (!own_refine && (d->upscale_refine < 1 || d->upscale_refine >= def_evals)) ||
         (own_refine && d->upscale_refine != d->n_refine_rungs))) {
        err(RAD_E_INVAL, "declare_video_generator '%s': a latent upscaler of factor %g at most, "
                         "refining %d evaluations of %lld by default; the factor must be above 1 "
                         "and the refinement 1 to %lld", nm, d->upscale_max, (int)d->upscale_refine,
            (long long)def_evals, (long long)(def_evals - 1));
        return RAD_E_INVAL;
    }
    auto buf_ok = [&](rad_buf h, uint32_t dtype, int64_t numel, const char* what) {
        if (h == RAD_NULL_HANDLE || (size_t)h >= prog_.buffers.size()) {
            err(RAD_E_INVAL, "declare_video_generator '%s': %s %u is not a declared buffer", nm,
                what, (unsigned)h);
            return false;
        }
        const RadBufDecl& ob = prog_.buffers[h].decl;
        if (ob.dtype != dtype || ob.domain != RAD_DOMAIN_DEVICE ||
            decl_numel(ob.rank, ob.shape) < numel) {
            err(RAD_E_INVAL, "declare_video_generator '%s': %s '%s' must be a device %s buffer of "
                             "at least %lld elements", nm, what, prog_.buffers[h].name.c_str(),
                dtype == RAD_U8 ? "u8" : "f32", (long long)numel);
            return false;
        }
        return true;
    };
    /* frames_out holds the last chunk, the largest, at the largest area; audio_out the longest
     * soundtrack. */
    const int64_t chunk_px = (d->frames_per_chunk + d->frames_tail) * d->max_area * 3;
    const int64_t max_frames = d->frames_per_chunk * d->max_chunks + d->frames_tail;
    const int64_t max_lat = (max_frames * d->fps_den * d->audio_rate * 2 + d->fps_num) /
                            (2 * d->fps_num) + 1;
    const int64_t max_samples = max_lat * (d->sample_rate / d->audio_rate);
    if (!buf_ok(d->frames_out, RAD_U8, chunk_px, "frames_out") ||
        !buf_ok(d->audio_out, RAD_F32, max_samples * d->audio_channels, "audio_out"))
        return RAD_E_INVAL;
    if (d->duration_out != RAD_NULL_HANDLE) {
        if (!buf_ok(d->duration_out, RAD_F32, 1, "duration_out")) return RAD_E_INVAL;
        if (!(d->duration_min > 0.0 && d->duration_max > d->duration_min) ||
            !std::isfinite(d->duration_max) ||
            (d->duration_dtype != RAD_F32 && d->duration_dtype != RAD_BF16)) {
            err(RAD_E_INVAL, "declare_video_generator '%s': a length head clamps to "
                             "[%g, %g] seconds in dtype %u; the bounds must be positive, finite "
                             "and in order, the dtype f32 or bf16", nm, d->duration_min,
                d->duration_max, (unsigned)d->duration_dtype);
            return RAD_E_INVAL;
        }
    }
    prog_.vgen_str.name = nm;
    prog_.vgen_str.tmpl = d->tmpl;
    prog_.vgen_str.rungs.assign(d->rungs, d->rungs + d->n_rungs);
    prog_.vgen_str.refine_rungs.assign(d->refine_rungs, d->refine_rungs + d->n_refine_rungs);
    prog_.vgen_str.default_negative = d->default_negative ? d->default_negative : "";
    prog_.vgen_str.decoders.clear();
    prog_.vgen_str.decoder_max_area.clear();
    for (const char* s = d->decoders; s && *s;) {
        while (*s == ' ') ++s;
        const char* e = s;
        while (*e && *e != ' ') ++e;
        if (e > s) prog_.vgen_str.decoders.emplace_back(s, e);
        s = e;
    }
    for (size_t i = 0; i < prog_.vgen_str.decoders.size(); ++i) {
        const int64_t a = d->decoder_max_area ? d->decoder_max_area[i] : 0;
        if (a < 0 || a > d->max_area) {
            err(RAD_E_INVAL, "declare_video_generator '%s': decoder '%s' makes frames of %lld pixels "
                             "at most, outside 0..max_area", nm,
                prog_.vgen_str.decoders[i].c_str(), (long long)a);
            return RAD_E_INVAL;
        }
        prog_.vgen_str.decoder_max_area.push_back(a > 0 ? a : d->max_area);
    }
    prog_.video_gen = *d;
    /* As for the other generators: the Program is moved after declare; strings and the rungs live
     * in vgen_str. */
    prog_.video_gen.name = nullptr;
    prog_.video_gen.tmpl = nullptr;
    prog_.video_gen.rungs = nullptr;
    prog_.video_gen.refine_rungs = nullptr;
    prog_.video_gen.default_negative = nullptr;
    prog_.video_gen.decoders = nullptr;
    prog_.video_gen.decoder_max_area = nullptr;
    return RAD_OK;
}

/* The shape generator (rad_declare_shape_generator). Checked here rather than at the first request:
 * the picture's geometry, the sampler, the grid and the field's buffer, which the core reads back
 * after every query pass. */
int Builder::declare_shape_generator(const RadShapeGenDecl* d) {
    if (finished_) {
        err(RAD_E_STATE, "declare_shape_generator after declare finished");
        return RAD_E_STATE;
    }
    if (!d) { err(RAD_E_INVAL, "declare_shape_generator: null declaration"); return RAD_E_INVAL; }
    const char* nm = d->name && *d->name ? d->name : "(unnamed)";
    if (prog_.is_shape_generator()) {
        err(RAD_E_STATE, "declare_shape_generator: '%s' was already declared; '%s' is a second one",
            prog_.sgen_name.c_str(), nm);
        return RAD_E_STATE;
    }
    if (prog_.logits_buf != RAD_NULL_HANDLE || prog_.pooled_buf != RAD_NULL_HANDLE ||
        prog_.generator.layout != 0 || prog_.is_audio_generator() || prog_.is_video_generator()) {
        err(RAD_E_STATE, "declare_shape_generator '%s': this program already declared %s; a "
                         "program generates text, embeds, makes images, audio, video or shapes -- "
                         "one of the six", nm,
            prog_.logits_buf != RAD_NULL_HANDLE ? "logits"
                : prog_.pooled_buf != RAD_NULL_HANDLE ? "a pooled output"
                : prog_.generator.layout != 0 ? "an image generator"
                : prog_.is_audio_generator() ? "an audio generator" : "a video generator");
        return RAD_E_STATE;
    }
    if (d->canvas < 1 || d->canvas > 4096 || !(d->border_ratio >= 0.0 && d->border_ratio < 1.0) ||
        d->enc_patch < 1 || d->enc_size < d->enc_patch || d->enc_size > 4096 ||
        d->enc_size % d->enc_patch != 0) {
        err(RAD_E_INVAL, "declare_shape_generator '%s': the picture is out of range (canvas %lld, "
                         "border %g, encoder %lld in patches of %lld): the canvas 1..4096, the "
                         "border in [0, 1), the encoder's side 1..4096 and a whole number of "
                         "patches", nm, (long long)d->canvas, d->border_ratio,
            (long long)d->enc_size, (long long)d->enc_patch);
        return RAD_E_INVAL;
    }
    for (int c = 0; c < 3; ++c)
        if (!std::isfinite(d->pix_mean[c]) || !(d->pix_std[c] > 0.0) ||
            !std::isfinite(d->pix_std[c])) {
            err(RAD_E_INVAL, "declare_shape_generator '%s': channel %d's mean %g and deviation %g "
                             "must be finite and the deviation positive", nm, c, d->pix_mean[c],
                d->pix_std[c]);
            return RAD_E_INVAL;
        }
    if (d->latent_rows < 1 || d->latent_channels < 1 || d->step_rows < 2 * d->latent_rows) {
        err(RAD_E_INVAL, "declare_shape_generator '%s': a latent of %lld rows of %lld channels "
                         "and a step of %lld rows; a step holds both branches' latents", nm,
            (long long)d->latent_rows, (long long)d->latent_channels, (long long)d->step_rows);
        return RAD_E_INVAL;
    }
    if (d->default_steps < 2 || d->max_steps < d->default_steps || d->max_steps > 1000 ||
        !(d->default_guidance >= 0.0) || !std::isfinite(d->default_guidance)) {
        err(RAD_E_INVAL, "declare_shape_generator '%s': the sampler is out of range (%lld steps "
                         "by default, %lld at most, guidance %g): at least 2 grid points, at most "
                         "1000, the guidance finite and not negative", nm,
            (long long)d->default_steps, (long long)d->max_steps, d->default_guidance);
        return RAD_E_INVAL;
    }
    if (!(d->bounds > 0.0) || !std::isfinite(d->bounds) || !std::isfinite(d->level) ||
        d->min_resolution < 2 || d->default_resolution < d->min_resolution ||
        d->max_resolution < d->default_resolution || d->max_resolution > 2048) {
        err(RAD_E_INVAL, "declare_shape_generator '%s': the field is out of range (bounds %g, "
                         "level %g, resolution %lld in %lld..%lld): the bounds positive, the level "
                         "finite, the resolutions in order from 2 to 2048", nm, d->bounds,
            d->level, (long long)d->default_resolution, (long long)d->min_resolution,
            (long long)d->max_resolution);
        return RAD_E_INVAL;
    }
    if (d->max_lanes < 1 || d->max_query < 1) {
        err(RAD_E_INVAL, "declare_shape_generator '%s': %lld lanes and %lld points a query pass; "
                         "both at least 1", nm, (long long)d->max_lanes, (long long)d->max_query);
        return RAD_E_INVAL;
    }
    const int64_t n_patch = (d->enc_size / d->enc_patch) * (d->enc_size / d->enc_patch);
    const int64_t rows = std::max({ d->step_rows, d->max_query, n_patch, d->latent_rows });
    if (prog_.ctx.max_tok < rows) {
        err(RAD_E_INVAL, "declare_shape_generator '%s': a pass carries %lld rows and this one's "
                         "largest is %lld", nm, (long long)prog_.ctx.max_tok, (long long)rows);
        return RAD_E_INVAL;
    }
    const rad_buf h = d->field_out;
    if (h == RAD_NULL_HANDLE || (size_t)h >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_shape_generator '%s': field_out %u is not a declared buffer", nm,
            (unsigned)h);
        return RAD_E_INVAL;
    }
    const RadBufDecl& ob = prog_.buffers[h].decl;
    if (ob.dtype != RAD_F32 || ob.domain != RAD_DOMAIN_DEVICE ||
        decl_numel(ob.rank, ob.shape) < d->max_query) {
        err(RAD_E_INVAL, "declare_shape_generator '%s': field_out '%s' must be a device f32 "
                         "buffer of at least %lld elements", nm, prog_.buffers[h].name.c_str(),
            (long long)d->max_query);
        return RAD_E_INVAL;
    }
    prog_.sgen_name = nm;
    prog_.shape_gen = *d;
    /* As for the other generators: the Program is moved after declare; the name lives in
     * sgen_name. */
    prog_.shape_gen.name = nullptr;
    return RAD_OK;
}

/* Validates the draft head at declare rather than at the first draft pass, where a bad decl would
 * only show as zero acceptance. `depth` must equal RadBuildCtx::max_spec, which the plugin sized
 * its buffers with. */
int Builder::declare_drafter(const RadDrafterDecl* d) {
    if (finished_) {
        err(RAD_E_STATE, "declare_drafter after declare finished");
        return RAD_E_STATE;
    }
    if (!d) { err(RAD_E_INVAL, "declare_drafter: null declaration"); return RAD_E_INVAL; }
    if (d->kind == RAD_DRAFT_NONE) return RAD_OK;   /* declaring nothing is declaring nothing */

    const char* nm = d->name && *d->name ? d->name : "(unnamed)";
    if (d->kind != RAD_DRAFT_SERIAL && d->kind != RAD_DRAFT_BLOCK) {
        err(RAD_E_INVAL, "declare_drafter '%s': kind %d is neither RAD_DRAFT_SERIAL nor "
                         "RAD_DRAFT_BLOCK", nm, d->kind);
        return RAD_E_INVAL;
    }
    if (prog_.drafter.kind != RAD_DRAFT_NONE) {
        err(RAD_E_STATE, "declare_drafter: '%s' already declared this program's drafter; '%s' is a "
                         "second one. One step has one draft head -- `draft_pass` carries a sign "
                         "and a round, and it has no room for a third meaning.",
            prog_.drafter_name.c_str(), nm);
        return RAD_E_STATE;
    }
    if (d->depth <= 0) {
        err(RAD_E_INVAL, "declare_drafter '%s': depth %d. A drafter that proposes nothing is not "
                         "declared at all -- leave the call out.", nm, d->depth);
        return RAD_E_INVAL;
    }
    if (prog_.ctx.max_spec > 0 && d->depth != prog_.ctx.max_spec) {
        err(RAD_E_INVAL, "declare_drafter '%s': depth %d, but this program was declared with "
                         "max_spec %d. The plugin sized its own buffers with max_spec and the "
                         "engine drives this depth; they cannot differ.",
            nm, d->depth, prog_.ctx.max_spec);
        return RAD_E_INVAL;
    }
    if (d->proposal != RAD_NULL_HANDLE && (size_t)d->proposal >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_drafter '%s': proposal buffer %u is not a declared buffer",
            nm, (unsigned)d->proposal);
        return RAD_E_INVAL;
    }
    if (d->proposal_pitch < 0) {
        err(RAD_E_INVAL, "declare_drafter '%s': proposal_pitch %lld", nm,
            (long long)d->proposal_pitch);
        return RAD_E_INVAL;
    }
    if (d->kind == RAD_DRAFT_BLOCK && d->proposal_pitch > 0 && d->proposal_pitch < d->depth) {
        err(RAD_E_INVAL, "declare_drafter '%s': proposal_pitch %lld under depth %d. A block pass "
                         "writes each sequence's %d proposals into one row of the proposal buffer, "
                         "so a row must hold them.", nm, (long long)d->proposal_pitch, d->depth,
            d->depth);
        return RAD_E_INVAL;
    }
    /* A block pass embeds the mask id at every unfilled position, so it needs one. */
    if (d->kind == RAD_DRAFT_BLOCK && d->mask_token < 0) {
        err(RAD_E_INVAL, "declare_drafter '%s': RAD_DRAFT_BLOCK with no mask_token. Every "
                         "position the block has not filled embeds it; there is nothing to run.",
            nm);
        return RAD_E_INVAL;
    }
    /* A block drafter's proposal cannot come from the sampler, which writes one token per row
     * while one pass fills the whole block. A serial drafter may use either. */
    if (d->kind == RAD_DRAFT_BLOCK && d->proposal == RAD_NULL_HANDLE) {
        err(RAD_E_INVAL, "declare_drafter '%s': RAD_DRAFT_BLOCK with no proposal buffer. One pass "
                         "fills %d positions and the sampler writes one token a row, so there is "
                         "nowhere for the rest of the block to arrive.", nm, d->depth);
        return RAD_E_INVAL;
    }
    /* The block's row count, defaulted to depth + 1. The head runs over the last `depth` rows
     * (rad_builder.h), so the block must be depth or depth + 1 rows; anything else shifts the
     * proposals. */
    int64_t block = d->block;
    if (d->kind == RAD_DRAFT_BLOCK) {
        if (block == 0) block = (int64_t)d->depth + 1;
        /* A tree's (rad_drafter_tree) node count is not its block's draft count: more nodes
         * than draft positions (a wider tree) or fewer (a block longer than the tree is deep on
         * most steps). Checked at finish, once the tree is or is not declared. */
        if (block < 2) {
            err(RAD_E_INVAL, "declare_drafter '%s': block %lld rows at depth %d. The head runs "
                             "over the last %d rows of the block, so the block is either %d rows "
                             "(the anchor predicts nothing) or %d (it does).",
                nm, (long long)block, d->depth, d->depth, d->depth + 1, d->depth);
            return RAD_E_INVAL;
        }
    } else if (block != 0) {
        err(RAD_E_INVAL, "declare_drafter '%s': block %lld on a SERIAL drafter. A serial drafter "
                         "runs one row a round; there is no block to size.",
            nm, (long long)block);
        return RAD_E_INVAL;
    }

    /* The name lives only in drafter_name. `drafter.name` points into plugin storage, and
     * pointing it at drafter_name would not survive moving the Program (a short string is stored
     * inline in the std::string). Left null, a careless reader gets nothing instead of freed
     * memory; the engine points its own copy of the decl at its own string. */
    prog_.drafter      = *d;
    prog_.drafter_name = nm;
    prog_.drafter.name = nullptr;
    prog_.drafter.block = block;
    /* A block drafter's proposal row holds its whole block's tokens, and the engine reads row s
     * at s * pitch (a short block fills only its front), so its pitch defaults to depth; a serial
     * one writes a token a pass, at pitch 1 by default. */
    if (prog_.drafter.proposal_pitch == 0)
        prog_.drafter.proposal_pitch = prog_.drafter.kind == RAD_DRAFT_BLOCK ? prog_.drafter.depth : 1;
    return RAD_OK;
}

/* A serial head that reads the trunk's cache and keeps none (rad_builder.h). Validated at declare
 * for the same reason as declare_drafter. */
int Builder::drafter_shared_kv(int pos_offset) {
    if (finished_) {
        err(RAD_E_STATE, "drafter_shared_kv after declare finished");
        return RAD_E_STATE;
    }
    if (prog_.drafter.kind != RAD_DRAFT_SERIAL) {
        err(RAD_E_STATE, "drafter_shared_kv: %s. Only a SERIAL head runs round after round at one "
                         "position; declare it with rad_declare_drafter first.",
            prog_.drafter.kind == RAD_DRAFT_NONE ? "no drafter is declared"
                                                 : "the declared drafter is a BLOCK drafter");
        return RAD_E_STATE;
    }
    if (pos_offset != 0 && pos_offset != 1) {
        err(RAD_E_INVAL, "drafter_shared_kv '%s': pos_offset %d. The rows sit at the committed "
                         "position (0) or one past it (1); nothing else names a position the head "
                         "was trained at.", prog_.drafter_name.c_str(), pos_offset);
        return RAD_E_INVAL;
    }
    prog_.drafter_shared_kv  = true;
    prog_.drafter_pos_offset = pos_offset;
    return RAD_OK;
}

int Builder::lora_root(const char* native, const char* spellings) {
    if (finished_) {
        err(RAD_E_STATE, "lora_root after declare finished");
        return RAD_E_STATE;
    }
    if (!native || !*native) {
        err(RAD_E_INVAL, "lora_root: an empty checkpoint prefix would put every tensor in one "
                         "network; name the prefix the network's tensors share");
        return RAD_E_INVAL;
    }
    for (const LoraRoot& r : prog_.lora_roots)
        if (r.native == native) {
            err(RAD_E_INVAL, "lora_root '%s' declared twice", native);
            return RAD_E_INVAL;
        }
    LoraRoot r;
    r.native = native;
    std::string cur;
    for (const char* p = spellings ? spellings : "";; ++p) {
        if (*p && *p != ' ') { cur += *p; continue; }
        if (!cur.empty()) r.spellings.push_back(cur == "-" ? std::string() : cur);
        cur.clear();
        if (!*p) break;
    }
    prog_.lora_roots.push_back(std::move(r));
    return RAD_OK;
}

int Builder::lora_alias(const char* native, const char* from, const char* to, int64_t from_lo,
                        int64_t from_hi, int64_t to_lo) {
    if (finished_) {
        err(RAD_E_STATE, "lora_alias after declare finished");
        return RAD_E_STATE;
    }
    bool root = false;
    for (const LoraRoot& r : prog_.lora_roots) root = root || (native && r.native == native);
    if (!root) {
        err(RAD_E_INVAL, "lora_alias '%s': its root '%s' is not declared (rad_lora_root first)",
            from ? from : "", native ? native : "");
        return RAD_E_INVAL;
    }
    if (!from || !*from || !to || !*to) {
        err(RAD_E_INVAL, "lora_alias under '%s': an empty module path", native);
        return RAD_E_INVAL;
    }
    auto hashes = [](const char* s) { int n = 0; for (; *s; ++s) n += *s == '#'; return n; };
    if (hashes(from) != hashes(to)) {
        err(RAD_E_INVAL, "lora_alias '%s' -> '%s': %d index placeholders against %d", from, to,
            hashes(from), hashes(to));
        return RAD_E_INVAL;
    }
    if (from_lo < 0 || to_lo < 0 || (from_hi != 0 && from_hi <= from_lo)) {
        err(RAD_E_INVAL, "lora_alias '%s' -> '%s': rows [%lld, %lld) to %lld", from, to,
            (long long)from_lo, (long long)from_hi, (long long)to_lo);
        return RAD_E_INVAL;
    }
    LoraAlias a;
    a.root = native; a.from = from; a.to = to;
    a.from_lo = from_lo; a.from_hi = from_hi; a.to_lo = to_lo;
    prog_.lora_aliases.push_back(std::move(a));
    return RAD_OK;
}

/* A block drafter's shorter block (rad_builder.h). Validated at declare: the core stages the block
 * from it, and a block the plugin did not declare its ops at would fail the step. */
int Builder::drafter_short_block(int64_t block) {
    if (finished_) {
        err(RAD_E_STATE, "drafter_short_block after declare finished");
        return RAD_E_STATE;
    }
    if (prog_.drafter.kind != RAD_DRAFT_BLOCK) {
        err(RAD_E_STATE, "drafter_short_block: %s. Only a BLOCK drafter proposes a block; declare "
                         "it with rad_declare_drafter first.",
            prog_.drafter.kind == RAD_DRAFT_NONE ? "no drafter is declared"
                                                 : "the declared drafter is a SERIAL head");
        return RAD_E_STATE;
    }
    const int64_t full = prog_.drafter.block > 0 ? prog_.drafter.block : prog_.drafter.depth + 1;
    const int64_t lead = full - prog_.drafter.depth;     /* rows ahead of the proposing ones */
    /* A tree (rad_drafter_tree, declared first) grows a tree of its own over the shorter pass, up
     * to block - 1 nodes whatever its own node count, so it needs only two rows. */
    if (prog_.drafter_tree ? (block < 2 || block >= full) : (block >= full || block - lead < 1)) {
        err(RAD_E_INVAL, "drafter_short_block '%s': %lld rows. The short block must be shorter "
                         "than the drafter's own (%lld) and still propose a token (%s %lld rows).",
            prog_.drafter_name.c_str(), (long long)block, (long long)full,
            prog_.drafter_tree ? "at least" : "more than",
            (long long)(prog_.drafter_tree ? 2 : lead));
        return RAD_E_INVAL;
    }
    prog_.drafter_short_block = block;
    return RAD_OK;
}

/* A block drafter whose proposal is a tree (rad_builder.h). */
int Builder::drafter_tree(rad_buf parents) {
    if (finished_) {
        err(RAD_E_STATE, "drafter_tree after declare finished");
        return RAD_E_STATE;
    }
    if (prog_.drafter.kind != RAD_DRAFT_BLOCK) {
        err(RAD_E_STATE, "drafter_tree: %s. Only a BLOCK drafter's one pass proposes every node at "
                         "once; declare it with rad_declare_drafter first.",
            prog_.drafter.kind == RAD_DRAFT_NONE ? "no drafter is declared"
                                                 : "the declared drafter is a SERIAL head");
        return RAD_E_STATE;
    }
    if (parents == RAD_NULL_HANDLE || (size_t)parents >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "drafter_tree '%s': parents buffer %u is not a declared buffer",
            prog_.drafter_name.c_str(), (unsigned)parents);
        return RAD_E_INVAL;
    }
    if (prog_.buffers[parents].decl.dtype != RAD_I32) {
        err(RAD_E_DTYPE, "drafter_tree '%s': parents buffer '%s' is not i32",
            prog_.drafter_name.c_str(), prog_.buffers[parents].name.c_str());
        return RAD_E_DTYPE;
    }
    prog_.drafter_tree    = true;
    prog_.drafter_parents = parents;
    return RAD_OK;
}

/* The routing counts' storage (rad_builder.h). The row count is checked in Ctx::bind_routing,
 * which knows the reporting layers. */
int Builder::route_table(rad_buf table) {
    if (finished_) {
        err(RAD_E_STATE, "route_table after declare finished");
        return RAD_E_STATE;
    }
    if (prog_.route_table != RAD_NULL_HANDLE) {
        err(RAD_E_STATE, "route_table: buffer '%s' is already the routing table",
            prog_.buffers[prog_.route_table].name.c_str());
        return RAD_E_STATE;
    }
    if (table == RAD_NULL_HANDLE || (size_t)table >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "route_table: buffer %u is not a declared buffer", (unsigned)table);
        return RAD_E_INVAL;
    }
    const BufferInfo& bi = prog_.buffers[table];
    if (bi.decl.dtype != RAD_I32) {
        err(RAD_E_DTYPE, "route_table: buffer '%s' is not i32", bi.name.c_str());
        return RAD_E_DTYPE;
    }
    if (bi.decl.domain != RAD_DOMAIN_DEVICE || bi.decl.kind != RAD_BUF_PERSIST) {
        err(RAD_E_INVAL, "route_table: buffer '%s' must be a RAD_BUF_PERSIST device buffer: the "
                         "core reads its rows at the end of every pass and between passes, where a "
                         "transient's bytes belong to whichever buffer shares them",
            bi.name.c_str());
        return RAD_E_INVAL;
    }
    prog_.route_table = table;
    return RAD_OK;
}

void Builder::note_v(const char* f, va_list ap) {
    char msg[1024];
    vsnprintf(msg, sizeof msg, f ? f : "", ap);
    prog_.notes.emplace_back(msg);
}

/* ================================================================== finalise */
void Builder::finish_weight_uses() {
    /* Declare enumerated every op and its weights in order, so the access sequence is known
     * statically and prefetch is exact (spec §5). Unresolved ops still count, which can only widen
     * an interval (prefetch earlier, free later): the safe direction. */
    for (size_t i = 1; i < prog_.ops.size(); ++i) {
        const OpInfo& o = prog_.ops[i];
        for (rad_weight h : o.weights) {
            if (h == 0 || h >= prog_.weights.size()) continue;
            WeightInfo& w = prog_.weights[h];
            if (w.first_use_op < 0) w.first_use_op = o.index;
            w.last_use_op = o.index;
        }
    }
}

/* The weight layout pass (spec §4.3). Each kernel's layout hook, given the weight's encoding and
 * this rank's planes, states the stored form it wants (e.g. libr4d's gemm_w4a8 wants WMMA fragment
 * order); the manifest records it and the loader produces it with the same kernel's relayout. A
 * hook may also refuse an encoding, which becomes a declare error naming weight, encoding and
 * kernel.
 *
 * A weight is stored once, so two kernels wanting different stored forms is an error, reported
 * here rather than as a shape mismatch at load.
 *
 * Device rows claim first, over every op, and host rows after them. A host row only claims a
 * weight no device row did: on a card a host row is the fallback of a device kernel that may
 * relayout, and a different reading there is not an error but a reason the op cannot move to the
 * host, which the runtime's check_domain_layout decides. On the host backend there are no device
 * rows, and the host rows' hooks decide the stored form (a weight over several planes is joined
 * by them, abi/rad_planes.h). */
void Builder::finish_layouts() {
    /* Who asked for each weight's layout, so a conflict names both kernels, and on which domain. */
    std::vector<std::string> claimant(prog_.weights.size());
    std::vector<int> claim_dom(prog_.weights.size(), -1);
    /* The host kernels that read a weight over several planes without joining them, for the error
     * when nothing else does. */
    std::vector<std::string> unjoined(prog_.weights.size());

    for (int d : { (int)RAD_DOMAIN_DEVICE, (int)RAD_DOMAIN_HOST })
    for (size_t i = 1; i < prog_.ops.size(); ++i) {
        const OpInfo& o = prog_.ops[i];
        if (o.weights.empty()) continue;

        /* The hook indexes the schema's positional operand list; OpInfo::weight_opd maps weights
         * to it. */
        for (size_t wi = 0; wi < o.weights.size(); ++wi) {
            rad_weight h = o.weights[wi];
            if (h == 0 || h >= prog_.weights.size()) continue;
            /* An optional weight the source lacks is absent; no kernel reads it. */
            if (sources_ && !prog_.weights[h].enc_known && prog_.weights[h].decl.optional)
                continue;
            int operand = wi < o.weight_opd.size() && o.weight_opd[wi] >= 0
                              ? (int)o.weight_opd[wi] : (int)wi;

            for (size_t bi = 0; bi < o.bands.size(); ++bi) {
                const Band& b = o.bands[bi];
                {
                    const Resolved& r = b.dom[d];
                    if (!r || !r.row->info) continue;
                    const RadKernelInfo* ki = r.row->info;
                    WeightInfo& w = prog_.weights[h];

                    /* This rank's planes as geometry, without data. */
                    RadTensor t[RAD_ENC_MAX_PLANES];
                    weight_planes(w, t);

                    /* A device row with no layout hook claims the selected plane as stored; it is
                     * not "no opinion". All bands of an op read the same stored bytes, so if one
                     * band's kernel relayouts and another's does not, the second misreads them
                     * with a matching byte count. Example: on an fp8 lm_head, logits_gemm_fp8
                     * (M <= 32) packs codes and scales per row, while logits_gemm_bf16 above it
                     * would read the E4M3 codes as bf16 and return empty completions once
                     * batches grow.
                     *
                     * A host row with no hook, or one declining the operand, is exempt: it reads
                     * the planes as the container stores them, and counting it would make every
                     * relayouting device op conflict with its own fallback. */
                    RadLayout L{};
                    const int st = ki->layout ? ki->layout(r.geom.params(), r.geom.n_params(),
                                                           operand, &w.enc, w.sel, t, w.n_sel, &L)
                                              : RAD_E_UNSUPPORTED;
                    /* Reads the bytes uninterpreted (a prefetch): no claim (rad_abi.h). */
                    if (st == RAD_LAYOUT_ANY) continue;
                    char enc[256];
                    rad_enc_format(&w.enc, enc, sizeof enc);
                    if (st == RAD_E_DTYPE) {
                        err(st, "%s: kernel %s does not read '%s' (operand %d) encoded %s -- "
                                "convert it to an encoding the kernel reads, or select another "
                                "kernel", o.op.c_str(), ki->name, w.name.c_str(), operand, enc);
                        continue;
                    }
                    if (st < 0 && st != RAD_E_UNSUPPORTED) {
                        err(st, "%s: kernel %s could not describe what it stores for operand %d "
                                "('%s', %s): %s", o.op.c_str(), ki->name, operand, w.name.c_str(),
                            enc, rad_strerror(st));
                        continue;
                    }
                    if (st == RAD_E_UNSUPPORTED && d != RAD_DOMAIN_DEVICE) {
                        if (w.n_sel > 1 && unjoined[h].find(ki->name) == std::string::npos)
                            unjoined[h] += (unjoined[h].empty() ? "" : ", ") + std::string(ki->name);
                        continue;
                    }

                    const std::string tag = st == RAD_OK ? (L.tag && *L.tag ? L.tag : ki->name)
                                                         : std::string();
                    /* A device row's form stands; this host row then cannot take the op. */
                    if (d == RAD_DOMAIN_HOST && claim_dom[h] == RAD_DOMAIN_DEVICE) continue;
                    if (!claimant[h].empty() && w.layout_tag != tag) {
                        err(RAD_E_SHAPE,
                            "weight '%s' is wanted in two stored forms: '%s' by %s and '%s' by %s. "
                            "A weight is stored once. Bands of the same op resolving to different "
                            "kernels is the usual cause: every band reads the same stored bytes, "
                            "so a kernel that relayouts has to cover the whole range or none of it",
                            w.name.c_str(),
                            w.layout_tag.empty() ? "(as it is)" : w.layout_tag.c_str(),
                            claimant[h].c_str(), tag.empty() ? "(as it is)" : tag.c_str(),
                            ki->name);
                        continue;
                    }
                    if (!claimant[h].empty()) continue;   /* the same form, asked again */
                    claimant[h] = ki->name;
                    claim_dom[h] = d;

                    if (tag.empty()) {
                        if (w.n_sel != 1)
                            err(RAD_E_SHAPE, "%s: kernel %s reads '%s' as it is, but the "
                                             "declaration takes %d planes of '%s' as one operand, "
                                             "and only a kernel's layout can join planes",
                                o.op.c_str(), ki->name, w.name.c_str(), w.n_sel,
                                w.source.c_str());
                        continue;   /* the stored defaults decl_weight set stand */
                    }
                    if (w.widen || w.narrow) {
                        err(RAD_E_DTYPE, "'%s' is declared %s over a %s plane and kernel %s "
                                         "relayouts it; a %s and a relayout are not both "
                                         "done at load", w.name.c_str(),
                            rad_dtype_name(w.decl.dtype),
                            rad_dtype_name(w.enc.plane[w.sel[0]].dtype), ki->name,
                            w.widen ? "widening" : "narrowing");
                        continue;
                    }
                    w.layout_tag   = tag;
                    w.identity     = false;
                    w.lay_op       = (int32_t)i;
                    w.lay_band     = (int32_t)bi;
                    w.lay_dom      = d;
                    w.lay_operand  = operand;
                    w.stored_dtype = L.dtype ? L.dtype : w.decl.dtype;
                    w.stored_rank  = L.rank;
                    for (uint32_t k = 0; k < RAD_MAX_RANK; ++k)
                        w.stored_shape[k] = k < L.rank ? L.shape[k] : 0;
                    if (L.align > 0) w.align = L.align;
                    /* The hook's byte count wins: it includes layout padding the core cannot derive
                     * from a shape. */
                    if (L.bytes > 0) {
                        w.stored_bytes = L.bytes;
                    } else {
                        int64_t nel = 1;
                        for (uint32_t k = 0; k < L.rank && k < RAD_MAX_RANK; ++k) nel *= L.shape[k];
                        const int64_t bs = rad_dtype_bytes(w.stored_dtype, nel);
                        if (bs <= 0) {
                            err(RAD_E_SHAPE, "%s: kernel %s stores '%s' as dtype %u and gave no "
                                             "byte count, which the core cannot size",
                                o.op.c_str(), ki->name, w.name.c_str(), w.stored_dtype);
                            continue;
                        }
                        w.stored_bytes = bs;
                    }
                }
            }
        }
    }

    /* A weight taking several planes as one operand needs a kernel to join them; one no op reads
     * has none, and neither has one read only by host kernels without a hook for it. */
    for (size_t h = 1; h < prog_.weights.size(); ++h) {
        const WeightInfo& w = prog_.weights[h];
        if (w.n_sel <= 1 || !w.identity || !claimant[h].empty()) continue;
        char en[256];
        rad_enc_format(&w.enc, en, sizeof en);
        if (unjoined[h].empty())
            err(RAD_E_SHAPE, "'%s' takes %d planes of '%s' (%s) as one weight and no op reads it, "
                             "so nothing says how they are stored", w.name.c_str(), w.n_sel,
                w.source.c_str(), en);
        else
            err(RAD_E_SHAPE, "'%s' takes %d planes of '%s' as one weight, and the host kernels "
                             "reading it (%s) have no layout that joins them (abi/rad_planes.h)",
                w.name.c_str(), w.n_sel, w.source.c_str(), unjoined[h].c_str());
    }
}

void Builder::finish_kv_groups() {
    for (size_t g = 1; g < prog_.kv_groups.size(); ++g) {
        KVGroupInfo& gi = prog_.kv_groups[g];
        const RadKVGroupDecl& d = gi.decl;

        if ((d.kind == RAD_KV_FULL || d.kind == RAD_KV_WINDOW || d.kind == RAD_KV_RING) &&
            gi.block_size == 0) {
            /* The plugin may never have asked; the manifest still needs the answer. */
            kv_block_size((rad_kvgroup)g);
        }

        int64_t elem_n_head = d.n_head_kv > 0 ? d.n_head_kv : 1;
        /* Sizes cover every layer bound to the group: KVManager defines a page as covering all of
         * them and divides it across the bound layers. A one-layer figure would make the per-layer
         * stride a fraction of a layer (misaligned from the second bound layer on). */
        const int64_t n_bound = gi.layers.empty() ? 1 : (int64_t)gi.layers.size();
        switch (d.kind) {
            case RAD_KV_FULL:
            case RAD_KV_WINDOW:
            case RAD_KV_RING:
                /* One block holds K and V for block_size tokens, the block manager's unit. */
                gi.bytes_per_block =
                    rad_dtype_bytes(d.dtype, gi.block_size * elem_n_head *
                                             rad_kv_row_elems(d.dtype, d.head_dim)) * n_bound;
                gi.bytes_per_state = 0;
                break;
            case RAD_KV_LINEAR:
                /* Per sequence, fixed size (32 heads of 128x128 fp32 is 2 MiB a layer), so it
                 * cannot be snapshotted per block (spec §7.3). */
                gi.bytes_per_block = 0;
                gi.bytes_per_state = rad_dtype_bytes(
                    d.dtype, elem_n_head * (d.state_dim[0] ? d.state_dim[0] : 1)
                                         * (d.state_dim[1] ? d.state_dim[1] : 1)) * n_bound;
                break;
            case RAD_KV_CONV: {
                /* A rolling window of conv_width-1 + max_spec entries, so a speculative rejection
                 * moves the read offset instead of recomputing (spec §10). Must be exactly
                 * conv_window_entries(), the state tensor's extent; see its comment. */
                int64_t entries = conv_window_entries(d, prog_.ctx.max_spec);
                if (entries <= 0) entries = 1;
                gi.bytes_per_block = 0;
                gi.bytes_per_state =
                    rad_dtype_bytes(d.dtype, elem_n_head * d.head_dim * entries) * n_bound;
                break;
            }
            default: break;
        }
        std::sort(gi.layers.begin(), gi.layers.end());
        gi.drafter = prog_.meta.n_layers > 0 && !gi.layers.empty() &&
                     gi.layers.front() >= prog_.meta.n_layers;

        /* A group with no bound layers cannot be sized (the layer count is the number of state
         * instances per sequence). KVManager::configure would refuse it; refuse here, where the
         * missing call is obvious. */
        if (gi.layers.empty())
            err(RAD_E_STATE,
                "kv group '%s' has no layers bound: call rad_bind_layer_kv(b, layer, g) for every "
                "layer that owns state in this group, or the block manager cannot tell how many "
                "state instances a sequence holds", gi.name.c_str());
    }
}

void Builder::finish_buffer_liveness() {
    const int32_t last_op = std::max<int32_t>((int32_t)prog_.ops.size() - 1, 0);
    /* A second lane, signalled by any concurrent buffer, gets a second scratch region
     * (Program::scratch_regions). */
    prog_.scratch_regions = concurrent_bufs_.empty() ? 1 : 2;
    if (lane1_w0_ >= 0 && lane1_w1_ < 0)
        err(RAD_E_STATE, "rad_lane1_window_begin without rad_lane1_window_end");
    prog_.lane1_w0 = lane1_w1_ >= 0 ? lane1_w0_ : -1;
    prog_.lane1_w1 = lane1_w1_;

    /* Each buffer's first and last declaring op, in one pass over the op lists. */
    const size_t nb = prog_.buffers.size();
    std::vector<int32_t> first(nb, -1), last(nb, -1);
    for (size_t o = 1; o < op_bufs_.size(); ++o)
        for (const std::vector<rad_buf>* v : { &op_bufs_[o].reads, &op_bufs_[o].writes })
            for (rad_buf h : *v) {
                if (first[h] < 0) first[h] = (int32_t)o;
                last[h] = (int32_t)o;
            }

    for (size_t i = 1; i < nb; ++i) {
        BufferInfo& b = prog_.buffers[i];
        /* Live at every op: PERSIST and DERIVED (set explicitly so the plan and the dump read one
         * rule); a buffer on the second lane, whose declared range does not bound when its bytes
         * are in use; and one with no declared use, since an op may issue any handle in any
         * order (--debug-graph reports the cost). */
        if (b.decl.kind != RAD_BUF_TRANSIENT || first[i] < 0 ||
            (concurrent_bufs_.count((rad_buf)i) && lane1_w1_ < 0)) {
            b.first_def = 0;
            b.last_use  = last_op;
            continue;
        }
        /* A concurrent buffer under a declared window (rad_lane1_window_begin): live across the
         * window, when lane 1 may touch it, and over its own uses. */
        if (concurrent_bufs_.count((rad_buf)i)) {
            b.first_def = std::min(first[i], lane1_w0_);
            b.last_use  = std::max(last[i], lane1_w1_);
            continue;
        }
        b.first_def = first[i];
        b.last_use  = last[i];

        /* The logits buffer is read after the declared op list: the sampler (whose ops declare no
         * buffer lists; its planes are PERSIST), the logprob readback and the engine. Extend its
         * life to the last op. The lm_head is at the end of the graph, so this costs almost no
         * sharing; the undeclared-use guard in core/runtime/issue.cpp enforces it. The pooled
         * output likewise: the engine reads it after the last op. */
        if ((rad_buf)i == prog_.logits_buf || (rad_buf)i == prog_.pooled_buf) b.last_use = last_op;
    }

    /* Region 1 holds what an op the second lane may issue asks for (Program::scratch_bytes_lane1):
     * every transient buffer of such an op is concurrent (Ctx::lane1_ rule 3). Ctx::issue refuses a
     * larger request on lane 1 by name, so an op that breaks the rule fails rather than overruns. */
    prog_.scratch_bytes_lane1 = 0;
    if (prog_.scratch_regions > 1)
        for (size_t o = 1; o < prog_.ops.size() && o < op_bufs_.size(); ++o) {
            bool lane1 = true;
            for (const std::vector<rad_buf>* v : { &op_bufs_[o].reads, &op_bufs_[o].writes })
                for (rad_buf h : *v)
                    if (prog_.buffers[h].decl.kind == RAD_BUF_TRANSIENT && !concurrent_bufs_.count(h))
                        lane1 = false;
            if (!lane1) continue;
            for (const Band& bd : prog_.ops[o].bands)
                if (bd.dom[RAD_DOMAIN_DEVICE] &&
                    bd.dom[RAD_DOMAIN_DEVICE].scratch_bytes > prog_.scratch_bytes_lane1)
                    prog_.scratch_bytes_lane1 = bd.dom[RAD_DOMAIN_DEVICE].scratch_bytes;
        }
    if (prog_.scratch_bytes_lane1 > prog_.scratch_bytes) prog_.scratch_bytes_lane1 = prog_.scratch_bytes;
}

int Builder::finish() {
    if (finished_) return status_;
    finished_ = true;

    /* A sizing declare needs only buffers and liveness; its kernels are the real declare's and its
     * weights and KV groups are unused. */
    if (ref_) {
        finish_buffer_liveness();
        return status_;
    }

    /* A weight declared before its name-map entry was looked up and not found (sources are
     * searched through the name map), so it was read as plain and its kernels chosen for that. If
     * the source answers now, report the ordering rather than serve the wrong encoding. */
    if (sources_)
        for (size_t h = 1; h < prog_.weights.size(); ++h) {
            const WeightInfo& w = prog_.weights[h];
            WeightSource ws;
            if (!w.enc_known && lookup_source(w.source, &ws) == RAD_OK)
                err(RAD_E_STATE, "weight '%s' was declared before the name map entry for '%s', so "
                                 "it was read as plain %s and its kernels chosen for that -- "
                                 "declare the map first (rad_weight_encoding)", w.name.c_str(),
                    w.source.c_str(), rad_dtype_name(w.decl.dtype));
        }

    if (prog_.drafter.kind == RAD_DRAFT_BLOCK && !prog_.drafter_tree &&
        (prog_.drafter.block < prog_.drafter.depth ||
         prog_.drafter.block > (int64_t)prog_.drafter.depth + 1))
        err(RAD_E_INVAL, "declare_drafter '%s': block %lld rows at depth %d. The head runs over the "
                         "last %d rows of the block, so the block is either %d rows (the anchor "
                         "predicts nothing) or %d (it does); only a tree (rad_drafter_tree) "
                         "proposes a node count other than its block's drafts.",
            prog_.drafter_name.c_str(), (long long)prog_.drafter.block, prog_.drafter.depth,
            prog_.drafter.depth, prog_.drafter.depth + 1, prog_.drafter.depth);

    if (instances_) finish_instances();
    fill_tune_instances();
    finish_weight_uses();
    finish_layouts();
    finish_kv_groups();
    /* A head with no cache of its own (rad_drafter_shared_kv) gets no history pass, so a KV group
     * bound at one of its layers (meta.n_layers and above) would be read but never written. */
    if (prog_.drafter_shared_kv)
        for (size_t g = 1; g < prog_.kv_groups.size(); ++g)
            for (int l : prog_.kv_groups[g].layers)
                if (l >= prog_.meta.n_layers) {
                    err(RAD_E_STATE, "drafter '%s' keeps no cache of its own (rad_drafter_shared_kv) "
                                     "and its layer %d is bound to KV group '%s', which no pass "
                                     "would ever write", prog_.drafter_name.c_str(), l,
                        prog_.kv_groups[g].name.c_str());
                    break;
                }
    finish_buffer_liveness();

    int s = plan_buffers(prog_, &bufrep_);
    if (s < 0 && status_ == RAD_OK) status_ = s;

    if (!prog_.misses.empty()) {
        /* Reported whole at the end (spec §1 step 4); the engine decides what is fatal. Split by
         * domain, since the tools read only this line. A device miss a host kernel stands in for
         * is not a hole (Program::device_holes; engine_bringup.cpp's miss_report). */
        const size_t dev = prog_.device_holes(device_is_host()).size();
        /* Host-only misses are not a warning (see engine_bringup.cpp). */
        RAD_LOG_AT(dev ? Log::Warn : Log::Debug,
                   "declare: %zu device band(s) and %zu host band(s) resolved to nothing; "
                   "run with --debug-graph for the full list", dev, prog_.misses.size() - dev);
    }
    return status_;
}

}  /* namespace rad */

/* ================================================================== the C surface */
/* Every entry point in rad_builder.h: a static_cast and a forward, since the handle is the
 * builder. */
extern "C" {

static rad::Builder* B(RadBuilder* b) { return static_cast<rad::Builder*>(b); }

rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d) {
    if (!b) return RAD_NULL_HANDLE;
    return B(b)->decl_weight(name, d);
}

int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t* shape,
                        uint32_t* rank) {
    if (!b) return RAD_E_INVAL;
    return B(b)->weight_encoding(source, out, shape, rank);
}

rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d) {
    if (!b) return RAD_NULL_HANDLE;
    return B(b)->decl_buffer(name, d);
}

rad_kvgroup rad_decl_kv_group(RadBuilder* b, const char* name, const RadKVGroupDecl* d) {
    if (!b) return RAD_NULL_HANDLE;
    return B(b)->decl_kv_group(name, d);
}

int64_t rad_kv_block_size(RadBuilder* b, rad_kvgroup g) {
    if (!b) return 0;
    return B(b)->kv_block_size(g);
}

int64_t rad_kv_layer_bytes(RadBuilder* b, rad_kvgroup g) {
    if (!b) return 0;
    return B(b)->kv_layer_bytes(g);
}

int rad_bind_layer_kv(RadBuilder* b, int layer, rad_kvgroup g) {
    if (!b) return RAD_E_INVAL;
    return B(b)->bind_layer_kv(layer, g);
}

rad_op rad_decl_op(RadBuilder* b, const char* op, const RadParam* params, int n_params,
                   const rad_weight* weights, int n_weights) {
    if (!b) return RAD_NULL_HANDLE;
    return B(b)->decl_op(op, params, n_params, weights, n_weights);
}

int rad_op_resolved(RadBuilder* b, rad_op h) {
    if (!b) return 0;
    return B(b)->op_resolved(h);
}

int rad_op_declared(RadBuilder* b, const char* op) {
    if (!b) return 0;
    return B(b)->op_declared(op);
}

int rad_op_reads(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!b || (!bufs && n)) return RAD_E_INVAL;
    return B(b)->op_reads(h, bufs, n);
}

int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!b || (!bufs && n)) return RAD_E_INVAL;
    return B(b)->op_writes(h, bufs, n);
}

int rad_buf_concurrent(RadBuilder* b, rad_buf buf) {
    if (!b) return RAD_E_INVAL;
    return B(b)->buf_concurrent(buf);
}

int rad_lane1_window_begin(RadBuilder* b) {
    if (!b) return RAD_E_INVAL;
    return B(b)->lane1_window(false);
}

int rad_lane1_window_end(RadBuilder* b) {
    if (!b) return RAD_E_INVAL;
    return B(b)->lane1_window(true);
}

int rad_weight_shard_span(RadBuilder* b, rad_weight w, int64_t lo, int64_t hi) {
    if (!b) return RAD_E_INVAL;
    return B(b)->weight_shard_span(w, lo, hi);
}

int rad_weight_shard_span_parts(RadBuilder* b, rad_weight w, const int64_t* lo, const int64_t* hi,
                                const int64_t* total, int n) {
    if (!b) return RAD_E_INVAL;
    return B(b)->weight_shard_span_parts(w, lo, hi, total, n);
}

int rad_kv_group_zero(RadBuilder* b, rad_kvgroup g) {
    if (!b) return RAD_E_INVAL;
    return B(b)->kv_group_zero(g);
}

int rad_kv_group_dense_read(RadBuilder* b, rad_kvgroup g) {
    if (!b) return RAD_E_INVAL;
    return B(b)->kv_group_dense_read(g);
}

int rad_kv_group_replicated(RadBuilder* b, rad_kvgroup g) {
    if (!b) return RAD_E_INVAL;
    return B(b)->kv_group_replicated(g);
}

int rad_decl_name_map(RadBuilder* b, const RadNameMap* m) {
    if (!b) return RAD_E_INVAL;
    return B(b)->decl_name_map(m);
}

int rad_declare_logits(RadBuilder* b, rad_buf logits) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_logits(logits);
}

int rad_declare_scores(RadBuilder* b, rad_buf hidden, rad_weight table) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_scores(hidden, table);
}

int rad_declare_pooled(RadBuilder* b, rad_buf pooled) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_pooled(pooled);
}

int rad_declare_plugin_words(RadBuilder* b, int64_t max_words) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_plugin_words(max_words);
}

int rad_declare_drafter(RadBuilder* b, const RadDrafterDecl* d) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_drafter(d);
}

int rad_drafter_shared_kv(RadBuilder* b, int pos_offset) {
    if (!b) return RAD_E_INVAL;
    return B(b)->drafter_shared_kv(pos_offset);
}

int rad_lora_root(RadBuilder* b, const char* native, const char* spellings) {
    if (!b) return RAD_E_INVAL;
    return B(b)->lora_root(native, spellings);
}

int rad_lora_alias(RadBuilder* b, const char* native, const char* from, const char* to,
                   int64_t from_lo, int64_t from_hi, int64_t to_lo) {
    if (!b) return RAD_E_INVAL;
    return B(b)->lora_alias(native, from, to, from_lo, from_hi, to_lo);
}

int rad_drafter_tree(RadBuilder* b, rad_buf parents) {
    if (!b) return RAD_E_INVAL;
    return B(b)->drafter_tree(parents);
}

int rad_route_table(RadBuilder* b, rad_buf table) {
    if (!b) return RAD_E_INVAL;
    return B(b)->route_table(table);
}

int rad_chain_on(void) { return rad_dev_chain_enabled(); }

int rad_drafter_short_block(RadBuilder* b, int64_t block) {
    if (!b) return RAD_E_INVAL;
    return B(b)->drafter_short_block(block);
}

int rad_declare_encoder(RadBuilder* b, const RadEncoderDecl* d) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_encoder(d);
}

int rad_declare_generator(RadBuilder* b, const RadGeneratorDecl* d) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_generator(d);
}

int rad_declare_audio_generator(RadBuilder* b, const RadAudioGenDecl* d) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_audio_generator(d);
}

int rad_declare_video_generator(RadBuilder* b, const RadVideoGenDecl* d) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_video_generator(d);
}

int rad_declare_shape_generator(RadBuilder* b, const RadShapeGenDecl* d) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_shape_generator(d);
}

void rad_note(RadBuilder* b, const char* f, ...) {
    if (!b) return;
    va_list ap;
    va_start(ap, f);
    B(b)->note_v(f, ap);
    va_end(ap);
}

/* ---- model metadata accessors for the container header's free-form key/values. The core never
 * interprets these; a plugin reads what it needs. */
static const char* meta_lookup(const RadModelMeta* m, const char* key) {
    if (!m || !key || !m->kv_key || !m->kv_val) return nullptr;
    for (int i = 0; i < m->n_kv; ++i)
        if (m->kv_key[i] && !std::strcmp(m->kv_key[i], key)) return m->kv_val[i];
    return nullptr;
}

long long rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt) {
    const char* v = meta_lookup(m, key);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    long long r = std::strtoll(v, &end, 0);
    return (end && end != v) ? r : dflt;
}

double rad_meta_getf(const RadModelMeta* m, const char* key, double dflt) {
    const char* v = meta_lookup(m, key);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    double r = std::strtod(v, &end);
    return (end && end != v) ? r : dflt;
}

const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt) {
    const char* v = meta_lookup(m, key);
    return v ? v : dflt;
}

}  /* extern "C" */
