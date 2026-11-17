/* rad_arch.h -- the scaffolding an architecture plugin sits on.
 *
 * spec.md §2.4 has one plugin per (architecture, quantisation): a w4a8 model wants
 * rmsnorm_had_quant_i8 where a bf16 model wants a plain rmsnorm, and a declare function without
 * branches on a quant descriptor reads as the list of what the model does, in order.
 *
 * This header-only C++ library keeps that from multiplying files: one small struct per
 * transformer block with a declare() and a step(), composed by the plugin. Nothing here crosses
 * an .so boundary; only the exported C symbols do.
 */
#ifndef RAD_ARCH_H
#define RAD_ARCH_H

#include "rad_builder.h"
#include "rad_device.h"
#include "rad_runtime.h"
#include "rad_yarn.h"
#include "rad_abi_layout.h"

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace rad {
namespace arch {

/* One thread per tensor-parallel rank in one process (spec §1), and declare runs once per rank,
 * so declared handles are per-instance state; a single global would let rank 1 overwrite rank 0.
 * libr4d all-reduces over 2/4/8 ranks, hence eight ranks. Several engines may run in the process
 * (docs/MULTIMODEL.md), each with its own ranks, so state is indexed by the instance's slot
 * (rad_build_slot at declare, rad_slot in step), unique across all of them, never by the rank two
 * engines share. */
enum { MAX_RANKS = 8, MAX_SLOTS = RAD_MAX_SLOTS };

/* ================================================================== a debug residual probe
 *
 * Finds composition faults: rad-kbench checks each op against the oracle, so a graph that wires
 * the right ops to the wrong buffers, or a block whose output has the wrong size, passes kbench
 * and still serves a different model. The residual stream's magnitude down the stack is smooth in
 * a working model and steps in a broken one.
 *
 * It synchronises the stream and copies one row, so it is off unless RADIANCE_DEBUG_RESID is set,
 * and prints each (layer, site) once. bf16 buffers only, as is every activation in this tree. */
struct DbgStat { double rms = -1.0, med = -1.0, max = -1.0; };

/* RMS, median and max of |x| over one row. A few "massive activation" channels sit two orders of
 * magnitude above the rest and dominate the RMS; the median is the bulk, which is what a block's
 * delta should be compared against. */
inline DbgStat dbg_row_stat(RadCtx* c, rad_buf b, int64_t n) {
    DbgStat s;
    if (!b || n <= 0) return s;
    void* p = rad_buf_ptr(c, b);
    if (!p) return s;
    std::vector<uint16_t> h((size_t)n);
    if (rad_memcpy_async(h.data(), p, n * 2, rad_stream(c)) < 0) return s;
    rad_stream_sync(rad_stream(c));
    std::vector<float> v((size_t)n);
    double a = 0.0, mx = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const uint32_t u = (uint32_t)h[(size_t)i] << 16;
        float f;
        std::memcpy(&f, &u, 4);
        a += (double)f * (double)f;
        v[(size_t)i] = std::fabs(f);
        if (v[(size_t)i] > mx) mx = v[(size_t)i];
    }
    std::sort(v.begin(), v.end());
    s.rms = std::sqrt(a / (double)n);
    s.med = v[(size_t)(n / 2)];
    s.max = mx;
    return s;
}

inline bool dbg_resid_on() {
    static const bool on = std::getenv("RADIANCE_DEBUG_RESID") != nullptr;
    return on;
}

/* True the first time any rank thread reaches (layer, site), so a probe prints each once. */
inline bool dbg_first(int layer, const char* site) {
    static std::mutex mu;
    static std::set<std::pair<int, std::string>> seen;
    std::lock_guard<std::mutex> lk(mu);
    return seen.emplace(layer, site).second;
}

inline void dbg_resid(RadCtx* c, int layer, const char* site, int64_t n_embd,
                      rad_buf x, rad_buf delta) {
    if (!dbg_resid_on() || !dbg_first(layer, site)) return;
    const DbgStat sx = dbg_row_stat(c, x, n_embd), sd = dbg_row_stat(c, delta, n_embd);
    std::fprintf(stderr, "D resid layer %2d %-4s: x rms %8.4f med %8.5f max %9.3f | "
                         "delta rms %8.4f med %8.5f max %9.3f\n",
                 layer, site, sx.rms, sx.med, sx.max, sd.rms, sd.med, sd.max);
}
/* ================================================================== the GPTQ calibration tap
 *
 * A block that characterises one of its activations declares an f32 [K, K] PERSIST buffer and a
 * `gram_accum` op into it, then calls calib_write() when the corpus is done. These helpers are
 * the block-independent parts: whether the tap is on, when to stop, and the file.
 *
 * The accumulator must be PERSIST: a transient buffer may share storage with other transients
 * and would be overwritten between the launches it sums.
 *
 * rad-convert reads these files in a separate process, because GPTQ needs the Hessian of a
 * layer's real input and so a running model. The bootstrap is: convert with plain absmax, serve
 * that, calibrate on it, convert again with the Hessians. The second moment is robust to the
 * first container's small quantisation error.
 *
 * Files are per rank and the consumer sums them. Under expert parallelism a rank's `down` input
 * covers only its own experts' tokens. `gate_up` input is the full token set on every rank, so
 * the sum scales H by `world`; GPTQ is invariant to a positive scale (damping is a fraction of
 * the trace and the update uses ratios). */
inline const char* calib_dir() {
    static const char* d = std::getenv("RADIANCE_CALIB_DIR");
    return d && *d ? d : nullptr;
}

/* Whether layer `layer` taps (RADIANCE_CALIB_LAYERS: comma-separated layers and "lo-hi" ranges;
 * unset, every layer). A model whose per-expert Hessians do not fit the host at once (Nemotron-3.5
 * Bolt's are 169 GiB) calibrates in several runs over disjoint ranges; each run writes only its
 * layers' files, so the runs share one directory. */
inline bool calib_layer_on(int layer) {
    static const std::vector<std::pair<int, int>> ranges = [] {
        std::vector<std::pair<int, int>> r;
        const char* s = std::getenv("RADIANCE_CALIB_LAYERS");
        while (s && *s) {
            char* e = nullptr;
            const long lo = std::strtol(s, &e, 10);
            if (e == s) break;
            long hi = lo;
            if (*e == '-') {
                const char* h = e + 1;
                hi = std::strtol(h, &e, 10);
                if (e == h) break;
            }
            r.emplace_back((int)lo, (int)hi);
            s = *e == ',' ? e + 1 : e;
            if (*s && *s != ',' && !std::isdigit((unsigned char)*s)) break;
        }
        return r;
    }();
    if (!std::getenv("RADIANCE_CALIB_LAYERS")) return true;
    for (const auto& r : ranges)
        if (layer >= r.first && layer <= r.second) return true;
    return false;
}

/* Token count at which the tap dumps and switches itself off (RADIANCE_CALIB_TOKENS, default
 * 65536). Whether the tap runs at all is calib_dir()'s decision. */
inline long long calib_target() {
    static const long long n = [] {
        const char* s = std::getenv("RADIANCE_CALIB_TOKENS");
        const long long v = s ? std::atoll(s) : 0;
        return v > 0 ? v : 65536LL;
    }();
    return n;
}

/* <dir>/gram.r<rank>.<what>.bin, as a 32-byte header and k*k f32 in row-major order:
 *
 *   char     magic[8] = "RADGRAM2"
 *   uint32_t k          the dimension, on both axes
 *   uint32_t rank       which rank wrote it
 *   uint64_t rows       activation rows accumulated, for provenance only
 *   uint32_t domain     0: the model's own activation; N: that activation rotated by the
 *                       unnormalised Hadamard of width N, block by block -- what a rotated
 *                       weight's kernel multiplies
 *   uint32_t pad
 *
 * `rows` is not a normaliser (H is used only up to a positive scale). It tells a full
 * calibration from a killed one, and for a per-expert file (CalibExperts) a barely-routed expert
 * from an exercised one: libquant reads a per-expert file only when it holds several times K
 * rows. `domain` is what the tap summed; a reader quantising in another domain converts the file
 * itself (libquant's calib_sum). */
inline bool calib_write_host(const float* h, int64_t k, long long rows, const char* what,
                             uint32_t domain, int rank) {
    const char* dir = calib_dir();
    if (!dir || !h || k <= 0) return false;
    char path[512];
    std::snprintf(path, sizeof path, "%s/gram.r%d.%s.bin", dir, rank, what);
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "radiance: calibration cannot write %s\n", path);
        return false;
    }
    char hdr[32] = {};
    std::memcpy(hdr, "RADGRAM2", 8);
    const uint32_t kk = (uint32_t)k, rk = (uint32_t)rank;
    const uint64_t nr = (uint64_t)rows;
    std::memcpy(hdr + 8, &kk, 4);
    std::memcpy(hdr + 12, &rk, 4);
    std::memcpy(hdr + 16, &nr, 8);
    std::memcpy(hdr + 24, &domain, 4);
    const size_t n = (size_t)(k * k);
    const bool okw = std::fwrite(hdr, 1, sizeof hdr, f) == sizeof hdr &&
                     std::fwrite(h, 4, n, f) == n;
    std::fclose(f);
    if (!okw) std::fprintf(stderr, "radiance: calibration write to %s was short\n", path);
    return okw;
}

/* The same file from a device accumulator: copied back on the block's stream, then written. */
inline bool calib_write(RadCtx* c, rad_buf b, int64_t k, long long rows, const char* what,
                        uint32_t domain) {
    if (!calib_dir() || !b || k <= 0) return false;
    void* p = rad_buf_ptr(c, b);
    if (!p) return false;
    std::vector<float> h((size_t)(k * k));
    if (rad_memcpy_async(h.data(), p, (int64_t)h.size() * 4, rad_stream(c)) < 0) return false;
    rad_stream_sync(rad_stream(c));
    return calib_write_host(h.data(), k, rows, what, domain, rad_rank(c));
}

/* A pinned host buffer of at least `bytes`, one per instance (slot), held for the process
 * lifetime, so a tap that copies a scratch back every pass runs at link speed. Null (and logged)
 * on failure. */
inline float* calib_staging(RadCtx* c, int64_t bytes) {
    static std::mutex mu;
    static std::map<int, std::pair<void*, int64_t>> per_slot;
    std::lock_guard<std::mutex> lk(mu);
    std::pair<void*, int64_t>& s = per_slot[rad_slot(c)];
    if (s.second >= bytes) return (float*)s.first;
    if (s.first) rad_dev_free(s.first, RAD_MEM_HOST_PINNED);
    s.first = rad_dev_alloc(bytes, RAD_MEM_HOST_PINNED);
    s.second = s.first ? bytes : 0;
    if (!s.first)
        std::fprintf(stderr, "radiance: calibration could not pin %lld MiB of host staging on "
                             "rank %d\n", (long long)(bytes >> 20), rad_rank(c));
    return (float*)s.first;
}

/* One Hessian per expert, for a routed expert's `down` projection. Its input is the expert's own
 * intermediate units, so channel i means something different in each expert and a pooled Gram
 * describes none of them; it is dominated by the highest-energy expert (a quarter of some layers'
 * energy on Qwen3.8-Flash-Next), and GPTQ fitted to it spends every expert's error budget there.
 *
 * Kept on the host: 640^2 f32 for 256 experts per rank is 419 MB a layer, 20 GB over 48 layers.
 * A pass sums each expert's rows into one layer-sized device scratch, which comes back through
 * calib_staging() and is added here as packed upper triangles (half the bytes, 10 GB per rank for
 * that model). The cost is one stream sync per layer.
 *
 * f32 accumulation suffices: each pass adds one partial sum per expert, so a 10M-token corpus is
 * a few hundred adds deep, and the consumer damps H by a percent of its trace. */
struct CalibExperts {
    int64_t ne = 0, k = 0;
    std::vector<float>   acc;    /* `ne` upper triangles of k*(k+1)/2, row by row */
    std::vector<int64_t> rows;   /* rows each expert was routed */

    void init(int64_t n_expert, int64_t width) {
        ne = n_expert;
        k = width;
        acc.assign((size_t)(ne * (k * (k + 1) / 2)), 0.0f);
        rows.assign((size_t)ne, 0);
    }

    /* `full` is `ne` row-major k x k Grams back to back, as the device scratch holds them. */
    void add(const float* full) {
        const int64_t tri = k * (k + 1) / 2;
        const unsigned hw = std::thread::hardware_concurrency();
        const unsigned nt = std::max(1u, std::min(8u, hw / 4));
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([this, full, tri, nt, t] {
                for (int64_t e = t; e < ne; e += nt) {
                    const float* s = full + (size_t)(e * k * k);
                    float* d = acc.data() + (size_t)(e * tri);
                    for (int64_t i = 0; i < k; ++i) {
                        const float* sr = s + (size_t)(i * k + i);
                        for (int64_t j = 0; j < k - i; ++j) d[j] += sr[j];
                        d += k - i;
                    }
                }
            });
        for (std::thread& x : th) x.join();
    }

    /* Expert `e`'s Gram, whole and symmetric, into `h` (k x k row-major). */
    void unpack(int64_t e, float* h) const {
        const float* d = acc.data() + (size_t)(e * (k * (k + 1) / 2));
        for (int64_t i = 0; i < k; ++i)
            for (int64_t j = i; j < k; ++j, ++d) {
                h[(size_t)(i * k + j)] = *d;
                h[(size_t)(j * k + i)] = *d;
            }
    }

    /* One file per expert, `<stem>.<global expert>`, where stem is the packed weight's name minus
     * `.weight`; libquant looks for it before the pooled `<stem>`. A never-routed expert's file
     * has `rows` 0. */
    bool write(const char* stem, int64_t expert_base, uint32_t domain, int rank) const {
        std::vector<float> h((size_t)(k * k));
        bool ok = true;
        for (int64_t e = 0; e < ne; ++e) {
            unpack(e, h.data());
            char what[256];
            std::snprintf(what, sizeof what, "%s.%lld", stem, (long long)(expert_base + e));
            ok = calib_write_host(h.data(), k, rows[(size_t)e], what, domain, rank) && ok;
        }
        return ok;
    }
};

/* ================================================================== names */
/* Declared names outlive the call that built them (the weight manifest, graph dump and name map
 * hold the pointer). A deque keeps every c_str() stable for the plugin's life; a vector would
 * reallocate.
 *
 * The scope prefix comes from RadBuildCtx, so a drafter or encoder tower in the same builder does
 * not collide with the language model's names (spec §10, §11). */
class Names {
public:
    explicit Names(const char* scope = "")
        : scope_(scope && *scope ? std::string(scope) + "." : std::string()) {}

    /* A declared name, scope-prefixed. */
    const char* f(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        char buf[256];
        va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
        pool_.emplace_back(scope_ + buf);
        return pool_.back().c_str();
    }
    /* A checkpoint-side name. Never scope-prefixed: the container says what it says. */
    const char* ckpt(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        char buf[256];
        va_list ap; va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
        pool_.emplace_back(buf);
        return pool_.back().c_str();
    }

private:
    std::deque<std::string> pool_;
    std::string             scope_;
};

/* Returns the first negative status from a declare step. */
#define RAD_ARCH_TRY(expr)  do { int _s = (expr); if (_s < 0) return _s; } while (0)

/* ================================================================== C++-side decl helpers */
/* C++ forms of rad_group_layer() and friends, which are C compound literals (a GNU extension in
 * C++; rad_builder.h points here). */
inline RadWeightGroup grp_model()               { RadWeightGroup g{}; g.layer = -1; g.expert = -1; return g; }
inline RadWeightGroup grp_layer(int l)          { RadWeightGroup g{}; g.layer = l;  g.expert = -1; return g; }
inline RadWeightGroup grp_expert(int l, int e)  { RadWeightGroup g{}; g.layer = l;  g.expert = e;  return g; }
/* An expert with more than one tensor needs a slot per tensor. The container writes a layer's
 * experts as fixed-stride units whose slots must arrive 0..n-1 (core/format/radfile.cpp,
 * check_unit_invariant); two tensors both at slot 0 are refused at convert time. A routed fp8
 * expert has four: gate_up codes, their scales, down codes, their scales. */
inline RadWeightGroup grp_expert_slot(int l, int e, int s) {
    RadWeightGroup g{}; g.layer = l; g.expert = e; g.slot = s; return g;
}

/* `row_parts` is this rank's rows per stacked projection (RadWeightDecl). A single part equals
 * none and is not recorded, keeping the load path on its plain-rows branch. */
inline RadWeightDecl weight_decl(uint32_t dtype, std::initializer_list<int64_t> shape, int access,
                                 int shard, RadWeightGroup group, const int64_t* row_parts,
                                 int n_row_parts) {
    RadWeightDecl d{};
    d.dtype = dtype;
    d.rank  = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) d.shape[i++] = s;
    d.access = access;
    d.shard  = shard;
    d.group  = group;
    if (row_parts && n_row_parts > 1) {
        d.n_row_parts = n_row_parts < RAD_MAX_ROW_PARTS ? n_row_parts : RAD_MAX_ROW_PARTS;
        for (int j = 0; j < d.n_row_parts; ++j) d.row_parts[j] = row_parts[j];
    }
    return d;
}

/* The pointer form of decl_w's row parts, for a caller that derives them (the fp8 linear divides
 * its weight's parts for the scale plane). */
inline rad_weight decl_w_n(RadBuilder* b, const char* name, uint32_t dtype,
                           std::initializer_list<int64_t> shape,
                           int access, int shard, RadWeightGroup group, int optional,
                           const int64_t* row_parts, int n_row_parts) {
    RadWeightDecl d = weight_decl(dtype, shape, access, shard, group, row_parts, n_row_parts);
    d.optional = optional;
    return rad_decl_weight(b, name, &d);
}

inline rad_weight decl_w(RadBuilder* b, const char* name, uint32_t dtype,
                         std::initializer_list<int64_t> shape,
                         int access, int shard, RadWeightGroup group, int optional = 0,
                         std::initializer_list<int64_t> row_parts = {}) {
    return decl_w_n(b, name, dtype, shape, access, shard, group, optional, row_parts.begin(),
                    (int)row_parts.size());
}

/* A view: `planes` of the logical weight `source` (null: the weight named `name`), declared at
 * what they select: one plane's own dtype and extents, or for several planes the codes dtype and
 * the logical extents (RadWeightDecl::source, spec §4.3). A block-fp8 linear is one stored weight
 * that gemm_nt_q takes as two operands, so it is declared twice over `<base>.weight`: codes as
 * `<base>.weight`, scales as `<base>.scale`. Every view takes the weight's movement group. */
inline rad_weight decl_view(RadBuilder* b, const char* name, const char* source,
                            const char* planes, uint32_t dtype,
                            std::initializer_list<int64_t> shape, int access, int shard,
                            RadWeightGroup group, const int64_t* row_parts = nullptr,
                            int n_row_parts = 0) {
    RadWeightDecl d = weight_decl(dtype, shape, access, shard, group, row_parts, n_row_parts);
    d.source = source;
    d.planes = planes;
    return rad_decl_weight(b, name, &d);
}

/* The encoding `source` is held in (from the container, the checkpoint, or rad-convert's recipe),
 * or false when unknown: a builder with no model behind it, or a name not yet in the name map
 * (declare the map first). */
inline bool weight_enc(RadBuilder* b, const char* source, RadEncoding* out) {
    return rad_weight_encoding(b, source, out, nullptr, nullptr) == RAD_OK;
}

/* The dtype for a view of all of `source`'s planes: the codes' dtype when the encoding is known,
 * else `dflt`. */
inline uint32_t weight_codes_dtype(RadBuilder* b, const char* source, uint32_t dflt) {
    RadEncoding e{};
    return weight_enc(b, source, &e) && e.n_planes > 0 ? e.plane[0].dtype : dflt;
}

inline rad_buf decl_b(RadBuilder* b, const char* name, uint32_t dtype,
                      std::initializer_list<int64_t> shape,
                      int kind = RAD_BUF_TRANSIENT, int domain = RAD_DOMAIN_DEVICE) {
    RadBufDecl d{};
    d.dtype = dtype;
    d.rank  = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) d.shape[i++] = s;
    d.kind   = kind;
    d.domain = domain;
    return rad_decl_buffer(b, name, &d);
}

/* src_index starts at -1, "the whole tensor"; 0 would silently select sub-tensor 0 of every
 * source. */
inline RadNameMap map_base(const char* declared, int mode) {
    RadNameMap m{};
    m.declared = declared;
    m.mode     = mode;
    for (int i = 0; i < 8; ++i) m.src_index[i] = -1;
    return m;
}

/* Row ranges of checkpoint tensors joined along dim 0 (RAD_MAP_ROWS): each piece is (source,
 * first row, rows). */
struct RowPiece { const char* src; int first; int64_t rows; };
inline int map_rows(RadBuilder* b, const char* declared, std::initializer_list<RowPiece> pieces) {
    RadNameMap m = map_base(declared, RAD_MAP_ROWS);
    for (const RowPiece& p : pieces) {
        if (m.n_src >= 8) return RAD_E_INVAL;
        m.src[m.n_src] = p.src;
        m.src_index[m.n_src] = p.first;
        m.src_rows[m.n_src] = p.rows;
        ++m.n_src;
    }
    return rad_decl_name_map(b, &m);
}

/* How adapter files name a generator's modules (rad_lora_root, rad_lora_alias): one network
 * under the checkpoint prefix `native`, the prefixes adapter keys use for it, and the module paths
 * that differ, '#' standing for a block index. */
struct LoraName {
    std::string from, to;
    int64_t     from_lo = 0, from_hi = 0, to_lo = 0;
};
inline int lora_names(RadBuilder* b, const char* native, const char* spellings,
                      const std::vector<LoraName>& names) {
    const int s = rad_lora_root(b, native, spellings);
    if (s < 0) return s;
    for (const LoraName& n : names) {
        const int a = rad_lora_alias(b, native, n.from.c_str(), n.to.c_str(), n.from_lo, n.from_hi,
                                     n.to_lo);
        if (a < 0) return a;
    }
    return RAD_OK;
}
/* The prefixes ComfyUI, kohya and bare-key adapter files put before a diffusion model's modules. */
inline const char* kLoraDiffusionSpellings = "diffusion_model. model.diffusion_model. lora_unet_ "
                                             "base_model.model. -";

/* A weight-normalised checkpoint tensor folded into its plain weight (RAD_MAP_WNORM). */
inline int map_wnorm(RadBuilder* b, const char* declared, const char* g, const char* v) {
    RadNameMap m = map_base(declared, RAD_MAP_WNORM);
    m.n_src  = 2;
    m.src[0] = g;
    m.src[1] = v;
    return rad_decl_name_map(b, &m);
}

/* A linear the checkpoint keeps as (in, out), read as (out, in): src's last two dimensions
 * swapped (RAD_MAP_TRANSPOSE). */
inline int map_transpose(RadBuilder* b, const char* declared, const char* src) {
    RadNameMap m = map_base(declared, RAD_MAP_TRANSPOSE);
    m.n_src  = 1;
    m.src[0] = src;
    return rad_decl_name_map(b, &m);
}

/* One index of one dimension of a checkpoint tensor (RAD_MAP_TAP): a causal 3-D convolution's
 * last temporal tap is map_tap(b, name, src, 2, -1). */
inline int map_tap(RadBuilder* b, const char* declared, const char* src, int dim, int index) {
    RadNameMap m = map_base(declared, RAD_MAP_TAP);
    m.n_src        = 1;
    m.src[0]       = src;
    m.src_index[0] = index;
    m.concat_dim   = dim;
    return rad_decl_name_map(b, &m);
}

/* An EMA codebook: the embedding sums over max(cluster usage, 1e-5), row by row (RAD_MAP_EMA). */
inline int map_ema(RadBuilder* b, const char* declared, const char* sums, const char* usage) {
    RadNameMap m = map_base(declared, RAD_MAP_EMA);
    m.n_src  = 2;
    m.src[0] = sums;
    m.src[1] = usage;
    return rad_decl_name_map(b, &m);
}

/* A checkpoint tensor copied through unchanged. */
inline int map_copy(RadBuilder* b, const char* declared, const char* src) {
    RadNameMap m = map_base(declared, RAD_MAP_COPY);
    m.n_src    = 1;
    m.src[0]   = src;
    return rad_decl_name_map(b, &m);
}

/* A copy with an alternative source; the first one the container holds wins. Expresses a tied
 * lm_head: Qwen3 0.6B and 1.7B ship no output.weight and reuse the embedding, 8B and up ship
 * one. */
inline int map_copy_alt(RadBuilder* b, const char* declared, const char* a, const char* c) {
    RadNameMap m = map_base(declared, RAD_MAP_COPY);
    m.n_src    = 2;
    m.src[0]   = a;
    m.src[1]   = c;
    return rad_decl_name_map(b, &m);
}

/* One expert's slice of a stacked checkpoint tensor. GGUF stores a MoE layer's experts as one
 * ffn_gate_exps [.., n_expert], while placement moves one expert at a time and so needs one
 * declared weight per expert (spec §4.2). src_index carries the expert index explicitly.
 *
 * `sub_rows` is for a stack flattened into the rows of a rank-2 tensor (Nomic's
 * experts.mlp.w1 is [n_expert * n_ff, n_embd]): it is the sub-tensor's row count, so the slice is
 * rows [index * sub_rows, (index + 1) * sub_rows). Left 0, a rank-3 stack takes index as its
 * dim-0 sub-tensor. */
inline int map_slice(RadBuilder* b, const char* declared, const char* src, int index,
                     int64_t sub_rows = 0) {
    RadNameMap m = map_base(declared, RAD_MAP_COPY);
    m.n_src       = 1;
    m.src[0]      = src;
    m.src_index[0]= index;
    m.src_rows[0] = sub_rows;
    return rad_decl_name_map(b, &m);
}

/* One expert's slice of a stacked tensor, transposed: the per-expert table a grouped GEMM reads
 * when the checkpoint keeps it [in, out]. A rank-3 stack takes index as its dim-0 sub-tensor. A
 * rank-2 stack flattened over its rows takes `sub_rows` as the sub-tensor's row count, as
 * map_slice's. */
inline int map_transpose_slice(RadBuilder* b, const char* declared, const char* src, int index,
                               int64_t sub_rows = 0) {
    RadNameMap m = map_base(declared, RAD_MAP_TRANSPOSE);
    m.n_src       = 1;
    m.src[0]      = src;
    m.src_index[0]= index;
    m.src_rows[0] = sub_rows;
    return rad_decl_name_map(b, &m);
}

/* Two stacked tensors, the same expert out of each, concatenated: gate and up into gate_up. */
inline int map_slice_concat(RadBuilder* b, const char* declared,
                            std::initializer_list<const char*> src, int index, int dim = 0) {
    RadNameMap m = map_base(declared, RAD_MAP_CONCAT);
    m.n_src = (int)src.size();
    int i = 0;
    for (const char* s : src) { m.src[i] = s; m.src_index[i] = index; ++i; }
    m.concat_dim = dim;
    return rad_decl_name_map(b, &m);
}

/* A convert-time fusion: q/k/v into qkv, gate and up into gate_up. Only the plugin knows both
 * sides, and rad-convert reads the map from it (spec §2.4). One GEMM launch and one movement unit
 * instead of three, at the cost of placing q, k and v on different tiers. */
inline int map_concat(RadBuilder* b, const char* declared, std::initializer_list<const char*> src,
                      int dim = 0) {
    RadNameMap m = map_base(declared, RAD_MAP_CONCAT);
    m.n_src      = (int)src.size();
    int i = 0;
    for (const char* s : src) m.src[i++] = s;
    m.concat_dim = dim;
    return rad_decl_name_map(b, &m);
}

/* One logical tensor stored as n shards named `<prefix><i><suffix>`, i in [0, n). Qwen3.8-Flash-
 * Next's n-gram embedding is 320,001,536 rows in 128 tensors (`...ngram_embedding.shard_0.weight`
 * .. `shard_127.weight`), far more than RadNameMap's `src[8]`.
 *
 * Emits ceil(n/8) CONCAT maps, which the core joins in call order (merge rule in
 * core/build/rad_builder.cpp). Call order is the concatenation order and shard 0 must come first:
 * hashed row ids address the whole table, and a permuted table silently looks up wrong rows.
 *
 * Returns the first refusal, or RAD_OK. */
inline int map_concat_shards(RadBuilder* b, const char* declared, const char* prefix,
                             const char* suffix, int n, int dim = 0) {
    if (n <= 0) return RAD_E_INVAL;
    for (int i = 0; i < n; i += 8) {
        RadNameMap m = map_base(declared, RAD_MAP_CONCAT);
        m.concat_dim = dim;
        m.n_src = 0;
        char buf[8][256];
        for (int k = 0; k < 8 && i + k < n; ++k) {
            std::snprintf(buf[k], sizeof buf[k], "%s%d%s", prefix, i + k, suffix);
            m.src[m.n_src++] = buf[k];
        }
        const int rc = rad_decl_name_map(b, &m);
        if (rc != RAD_OK) return rc;
    }
    return RAD_OK;
}

/* ================================================================== buffer liveness */
/* States an op's buffer read and write sets. Weight operands are named at declare but buffer
 * operands are not, and a transient with no declared use must be assumed live for the whole
 * program. Returns the handle so a declaration stays one statement; a refusal from rad_op_reads
 * is recorded by the builder with the op named and surfaces in declare's final report. */
inline rad_op rw(RadBuilder* b, rad_op h,
                 std::initializer_list<rad_buf> reads,
                 std::initializer_list<rad_buf> writes) {
    if (h == RAD_NULL_HANDLE) return h;   /* an unresolved op has no liveness to declare */
    rad_op_reads (b, h, reads.begin(),  (int)reads.size());
    rad_op_writes(b, h, writes.begin(), (int)writes.size());
    return h;
}

/* ---- A weight read into the on-die cache beside a collective (libr4d `prefetch`, docs/OPS.md).
 * At decode heights a GEMM streams its weight once from memory, and a collective in front of it
 * leaves the card's memory idle for its round trip. prefetch_beside issues `op`, a `prefetch` of
 * the next GEMM's weight `w`, behind the GEMM before the collective, so the two do not share the
 * memory, and the next GEMM finds its leading bytes in the cache. Nothing it reads or writes is a
 * model value, so on the second lane it keeps that lane's rules with no buffer named
 * (core/runtime/ctx.h): it is no collective, keeps no counter and touches no arena byte.
 *
 * Only at decode and verify heights, up to kPrefetchRows rows: a DFlash2 verify's 8 a sequence, a
 * z-lab DFlash verify's 16, a plain decode's one. Prompt steps never take the reads: beside the
 * prefill all-reduces they made a fresh server's first greedy reply differ, the slot-reuse race
 * that libr4d's full fence on prefill-sized four-rank messages (kArFenceFromElems) also cures. */
constexpr int64_t kPrefetchRows = 16;
/* Where launches chain (rad_chain_on, the core's decision) the read goes on lane 0 itself, right
 * after the GEMM: its chained form (libr4d r4dchain_prefetch) waits for that GEMM and then reads
 * beside the collective after it without holding the collective, so there is no join between the
 * lanes and the read starts as the GEMM's last block ends rather than ~10 us later on lane 1 (on
 * 4x R9700, the 31B with DFlash2 decoded 6-8% faster than with the read on lane 1). A plain launch
 * would read in series on lane 0, so elsewhere the read goes on lane 1, behind a join from lane
 * 0. */
inline bool prefetch_lane0() { return rad_chain_on() != 0; }
inline void prefetch_beside(RadCtx* c, rad_op op, rad_weight w, int64_t T) {
    if (!op || T > kPrefetchRows) return;
    if (prefetch_lane0()) {
        RAD_ISSUE(c, op, RAD_W(w));
        return;
    }
    if (rad_lane_join(c, 0, 1) == RAD_OK && rad_lane(c, 1) == RAD_OK) RAD_ISSUE(c, op, RAD_W(w));
    rad_lane(c, 0);
}
/* The `prefetch` of `w`'s first `bytes` where a library serves it; 0 otherwise. A plugin declares
 * none where the deployment turns the reads off (RadBuildCtx::weight_prefetch_off). */
inline rad_op decl_prefetch(RadBuilder* b, rad_weight w, int64_t bytes) {
    if (!w || bytes <= 0 || !rad_op_declared(b, "prefetch")) return 0;
    return RAD_OP(b, "prefetch", RAD_PARAMS(RAD_INT("bytes", bytes)), RAD_WEIGHTS(w));
}

/* ================================================================== geometry */
/* Dimensions come from model metadata, so one plugin covers every size in a family. Sharded
 * dimensions are already divided by world_size: the plugin declares divided dimensions and emits
 * the collectives itself rather than relying on a sharding inference pass (spec §9). */
/* The paged KV block, in tokens. `attn_paged` declares `block_size` RAD_DERIVED (kernel-supplied)
 * and every libr4d row has C_EQ("block_size", 16), so pinning it changes nothing there. It must be
 * pinned for geometries that resolve to libref (e.g. Qwen3.5-0.8B, head_dim 256 at gqa 4), which
 * serves any block size and so derives none; the KV group could not be sized. */
enum { RAD_KV_BLOCK = 16 };

struct Geom {
    int64_t n_layer    = 0;
    int64_t n_embd     = 0;    /* the residual stream. Replicated on every rank. */
    int64_t head_dim   = 0;
    int64_t n_vocab    = 0;    /* per rank: lm_head is vocab-sharded (spec §9) */
    int64_t n_vocab_all= 0;
    int64_t vocab_off  = 0;    /* this rank's first vocab row, for the sampler's candidate merge */
    int64_t n_head     = 0;    /* per rank */
    int64_t n_head_kv  = 0;    /* per rank */
    int64_t n_head_all = 0;
    int64_t n_head_kv_all = 0;
    /* This rank's first row in the checkpoint's unsharded k/v projection, in container rows (the
     * unit of a LinearFP8 span, passed as `span_lo`). Nonzero when KV heads are replicated over
     * ranks (world > n_head_kv_all) or under a smaller attention world; zero on a plain split. */
    int64_t kv_off   = 0;
    /* The attention world. When world neither divides nor is a multiple of the KV head count
     * (three ranks over Flash-Next's two KV heads), attention runs on the largest multiple of the
     * head count below world (`attn_world`; split, or replicated if still above the count). The
     * remaining ranks are attention-zero: same heads and program as their peers, but they compute
     * no attention (a `fill` of the block's output and a zero KV group, rad_kv_group_zero) and
     * add zero to the reduce. `attn_rank` is the index inside the attention world (attention-zero
     * ranks clamp to the last). Only set for plugins that pass geom_from's `attn_world_ok`. */
    int     attn_world = 1;
    int     attn_rank  = 0;
    bool    attn_zero  = false;
    int64_t n_ff       = 0;    /* per rank */
    int64_t n_ff_all   = 0;
    int64_t max_tok    = 0;
    int64_t max_seqs   = 0;
    int64_t max_ctx    = 0;
    /* --kv-resident in tokens (RadBuildCtx::kv_resident_blocks), 0 when nothing streams: what
     * kv_resolve_declare states as the ring's depth. */
    int64_t kv_resident = 0;
    int     max_spec   = 0;
    /* Most rows the sampler can want in one step: one per sequence, or one per draft position per
     * sequence at a speculative verify. Bounds the logits buffer (at a 248K vocab, 63 MiB rather
     * than 8 GiB for max_tok rows). */
    int64_t max_logit_rows = 0;
    int     rank       = 0;
    int     world      = 1;
    /* 1 = exact all-reduce, 0 = a lossy wire is accepted. Passed to every `all_reduce` this
     * library emits; RadBuildCtx describes the trade-off. */
    int     wire_exact = 1;
    /* Messages below this many bytes use the exact all-reduce even with a lossy wire. Passed as
     * `all_reduce`'s `min_bytes`; 0 leaves the default to the kernel plugin. */
    int64_t wire_min_bytes = 0;
    float   eps        = 1e-6f;
    /* GemmaRMSNorm's `1 +`. Qwen3.5, Qwen3-Next and Gemma normalise with `x_hat * (1 + w)` and
     * store `w` near 0.1, so reading it as a plain gain shrinks every normed activation 5-10x. An
     * architecture property: `rmsnorm`, `rmsnorm_add`, `qk_norm_rope_gate` and
     * `rmsnorm_had_quant_i8` take a `wadd` parameter (libref/ref_registry.cpp) and every block
     * passes this; plain-norm families leave it 0.
     *
     * Not used by the gated delta net's `ssm_norm`: RMSNormGated is a plain RMS norm in these
     * families too (gains near 0.88) and `gdn_gated_rmsnorm` has no such parameter. */
    float   wadd       = 0.0f;
    /* The same for the per-head q/k norms, set separately: a different tensor (partly negative
     * gains in this family, which only `1 + w` explains) that a family could treat differently. */
    float   wadd_qk    = 0.0f;
    float   theta      = 1e6f;
    float   rope_scale = 1.0f;
    RadYarn yarn;
    /* M-RoPE: the checkpoint's `mrope_section`, if any. A multi-component rotary declares `rope`
     * with `rope_mode` and these sections and passes RadBatch::rope_pos; empty means
     * one-component NeoX over `positions`. Copied, because the container's strings may not
     * outlive declare. */
    bool    rope_mc    = false;
    char    rope_mode[8]      = "neox";
    char    rope_sections[24] = "";
    const char* dtype  = "bf16";   /* the OPERAND COMBINATION, spec §2.3's dtype key */
    uint32_t act_dtype = RAD_BF16; /* what activation buffers are made of */
    /* The KV cache dtype, separate from `dtype`: weights follow the checkpoint, the cache is a
     * deployment choice. Must be spelled as `rad_dtype_name` gives it, because the attention and
     * kv_store rows band on it verbatim. */
    const char* kv_dtype = "bf16";
    /* The linears are plain bf16 weights read by `gemm_nt`, in a model built from the fp8 blocks
     * (rad_fp8.h): a dense bf16 checkpoint converted without a quantising recipe. Producers in
     * front of a linear then quantise nothing, so an `ActFP8` is its bf16 plane alone. A model-wide
     * setting because producers and consumers must agree; the plugin reads it off a trunk weight's
     * encoding and every linear checks its own against it. */
    bool    w_bf16 = false;
    /* The linears are rotated int8 (rad_fp8.h's linear_is_w8h: IQ8R, or the 128-group form) with
     * one activation scale per `w_i8_group` columns, for a plugin whose producers are declared away
     * from its probe: each then declares an int8 code pair (ActFP8::declare_pair). Model-wide as
     * w_bf16 is. */
    bool    w_i8h = false;
    int64_t w_i8_group = 128;

    int64_t q_dim()   const { return n_head * head_dim; }
    int64_t kv_dim()  const { return n_head_kv * head_dim; }
    int64_t qkv_dim() const { return q_dim() + 2 * kv_dim(); }
    int64_t gqa()     const { return n_head_kv ? n_head / n_head_kv : 0; }
};

/* The KV cache dtype: RadBuildCtx::kv_dtype (--kv-cache-dtype), or `dflt` when it is
 * RAD_DT_INVALID (not given). Every architecture in this tree passes RAD_BF16 as the default. */
static inline uint32_t rad_kv_cache_dtype(const RadBuildCtx* c, uint32_t dflt) {
    return (c && c->kv_dtype != RAD_DT_INVALID) ? (uint32_t)c->kv_dtype : dflt;
}
/* Divide, refusing (with a message naming `what`) rather than rounding, before any weight is
 * mapped. */
inline int divide_or_fail(int64_t v, int world, const char* what, int64_t* out) {
    if (world <= 0 || v % world != 0) {
        fprintf(stderr, "radiance: %s = %lld is not divisible by world_size = %d\n",
                what, (long long)v, world);
        return RAD_E_INVAL;
    }
    *out = v / world;
    return RAD_OK;
}

/* The vocabulary split. The vocab need not divide by world (151669, 248320), so the last rank
 * holds fewer rows. The sampler merges candidate sets rather than gathering full logits (spec §9,
 * §13), so a ragged shard is free, while padding would need a fake token id masked everywhere.
 *
 * A vocabulary world divides keeps the even share. Otherwise each rank takes ceil(total / world)
 * rounded up to a 128-row block (RAD_FP8_BLOCK), the last rank short: a block-scaled fp8 lm_head
 * is stored in 128x128 tiles and a row span that cuts a tile is refused (rad_weight_shard_span).
 * Example: 248320 over 3 ranks gives 82816, 82816, 82688. core/engine.cpp restates this rule and
 * checks it against the declared width; keep the two in step. */
inline int64_t vocab_per_rank(int64_t total, int world) {
    if (total % world == 0) return total / world;
    const int64_t per = (total + world - 1) / world;
    return (per + 127) / 128 * 128;
}
inline int64_t shard_extent(int64_t total, int world, int rank) {
    const int64_t per = vocab_per_rank(total, world);
    const int64_t beg = per * rank;
    const int64_t end = beg + per < total ? beg + per : total;
    return end > beg ? end - beg : 0;
}
inline int64_t shard_begin(int64_t total, int world, int rank) {
    const int64_t per = vocab_per_rank(total, world);
    const int64_t beg = per * rank;
    return beg < total ? beg : total;
}

/* A container-header integer under its GGUF (`qwen35.ssm.state_size`) or HuggingFace
 * (`linear_key_head_dim`) spelling, so one plugin serves a .rad converted from either. `found`
 * reports whether the value was read or defaulted, so a caller can refuse a silent default. */
inline long long meta_int2(const RadModelMeta* m, const char* gguf_key, const char* hf_key,
                           long long dflt, bool* found = nullptr) {
    const long long sentinel = -0x7FFFFFFFll;
    long long v = rad_meta_geti(m, gguf_key, sentinel);
    if (v == sentinel) v = rad_meta_geti(m, hf_key, sentinel);
    if (found) *found = (v != sentinel);
    return v == sentinel ? dflt : v;
}

/* The key a text-model value is under: `<pre>text_config.<key>`, where a multimodal config nests
 * its language model, when the container has it, else `<pre><key>`. `pre` is "" for the model and
 * "draft." for a drafter's config. */
inline const char* meta_text_key(const RadModelMeta* m, const char* key, const char* pre,
                                 char (&buf)[192]) {
    std::snprintf(buf, sizeof buf, "%stext_config.%s", pre, key);
    if (!rad_meta_gets(m, buf, nullptr)) std::snprintf(buf, sizeof buf, "%s%s", pre, key);
    return buf;
}
inline long long meta_text_geti(const RadModelMeta* m, const char* key, long long dflt,
                                const char* pre = "") {
    char k[192];
    return rad_meta_geti(m, meta_text_key(m, key, pre, k), dflt);
}
inline double meta_text_getf(const RadModelMeta* m, const char* key, double dflt,
                             const char* pre = "") {
    char k[192];
    return rad_meta_getf(m, meta_text_key(m, key, pre, k), dflt);
}
inline const char* meta_text_gets(const RadModelMeta* m, const char* key, const char* dflt,
                                  const char* pre = "") {
    char k[192];
    return rad_meta_gets(m, meta_text_key(m, key, pre, k), dflt);
}

/* ================================================================== config arrays */
/* Calls fn(token, length) for each token of a flattened config array until fn returns false.
 * rad-convert writes an array of scalars space-separated (core/format/checkpoint.cpp,
 * flatten_json); commas, brackets and quotes also separate, for values written by hand. */
template <class F>
inline void for_each_token(const char* s, F&& fn) {
    if (!s) return;
    auto sep = [](char ch) { return ch == ' ' || ch == ',' || ch == '[' || ch == ']' || ch == '"'; };
    for (const char* p = s; *p;) {
        while (*p && sep(*p)) ++p;
        if (!*p) return;
        const char* t = p;
        while (*p && !sep(*p)) ++p;
        if (!fn(t, (size_t)(p - t))) return;
    }
}

inline bool token_is(const char* t, size_t n, const char* word) {
    return std::strlen(word) == n && std::strncmp(t, word, n) == 0;
}

/* "1 10 20 30 39" -> the trunk layers a drafter taps, at most `max`. Out-of-range entries are
 * dropped, not clamped: a drafter naming a layer this model lacks belongs to another target, and
 * clamping would hide that. Parsing stops at the first token that is not a number. */
inline int parse_layer_ids(const char* s, int64_t n_layer, int* out, int max) {
    int n = 0;
    for_each_token(s, [&](const char* t, size_t) {
        if (n >= max) return false;
        char* e = nullptr;
        const long v = std::strtol(t, &e, 10);
        if (e == t) return false;
        if (v >= 0 && v < n_layer) out[n++] = (int)v;
        return true;
    });
    return n;
}

/* A `layer_types` array as one flag per entry: 1 for any name in `one`, 0 for `zero`. Any other
 * name is refused, naming `arch`. */
inline int parse_layer_kinds(const char* arch, const char* s, std::initializer_list<const char*> one,
                             const char* zero, std::vector<int>* out) {
    out->clear();
    int rc = RAD_OK;
    for_each_token(s, [&](const char* t, size_t n) {
        int k = token_is(t, n, zero) ? 0 : -1;
        for (const char* w : one) if (token_is(t, n, w)) k = 1;
        if (k < 0) {
            fprintf(stderr, "radiance: %s: layer_types[%zu] is '%.*s'; known kinds are %s and %s\n",
                    arch, out->size(), (int)n, t, *one.begin(), zero);
            rc = RAD_E_UNSUPPORTED;
            return false;
        }
        out->push_back(k);
        return true;
    });
    return rc;
}

/* The hybrid trunk's interval rule (llama.cpp's): every `interval`-th layer is full attention, so
 * with interval 4 layers 3, 7, ... are full and the rest linear. */
inline bool is_full_attention(int64_t l, int64_t interval) {
    return interval > 0 && (l + 1) % interval == 0;
}

/* Checks a container's `layer_types` (`s`, null or empty when it states none, as a GGUF-converted
 * container does) against is_full_attention. The interval rule is not a property of the family,
 * and a model built with the wrong layers linear still generates fluent text, so a disagreement is
 * fatal. `full` lists the names of a full layer; entries past the trunk (an MTP block) are not
 * compared. */
inline int check_layer_schedule(const char* arch, const char* s, int64_t n_layer, int64_t interval,
                                std::initializer_list<const char*> full) {
    if (!s || !*s) return RAD_OK;
    std::vector<int> kind;
    RAD_ARCH_TRY(parse_layer_kinds(arch, s, full, "linear_attention", &kind));
    if ((int64_t)kind.size() < n_layer) {
        fprintf(stderr, "radiance: %s: layer_types names %zu layers and the model has %lld\n",
                arch, kind.size(), (long long)n_layer);
        return RAD_E_UNSUPPORTED;
    }
    for (int64_t l = 0; l < n_layer; ++l) {
        const bool rule = is_full_attention(l, interval);
        if ((kind[(size_t)l] != 0) != rule) {
            fprintf(stderr, "radiance: %s: layer_types makes layer %lld %s and "
                            "full_attention_interval %lld makes it %s; the layers are built by the "
                            "interval\n", arch, (long long)l, rule ? "linear" : "full",
                    (long long)interval, rule ? "full" : "linear");
            return RAD_E_UNSUPPORTED;
        }
    }
    return RAD_OK;
}

/* A rank's place in the remainder-first order. Head-sized work that does not divide evenly (the
 * extra delta-net head) goes first to attention-zero ranks, which pay only a fill for attention
 * (rad_block_attn_gated_fp8.h) and keep no attention cache, so the slowest rank is not also given
 * the extra head. An expert's extra block is not head-sized and stays on the rotation
 * (qwen4exp_fp8.cpp). Attention-zero ranks take places 0..n-1 regardless of `parity`; attention
 * ranks follow, rotated by `parity`. With no attention-zero rank the place is
 * (rank - parity) mod world. */
inline int64_t remainder_place(const Geom& g, int rank, int parity) {
    const int nz = g.world - g.attn_world;
    if (rank >= g.attn_world) return rank - g.attn_world;
    const int aw = g.attn_world;
    return nz + ((rank - parity) % aw + aw) % aw;
}

/* Fills `g` from model metadata and the build context.
 *
 * `arch_splits_ff`: the plugin splits the feed-forward width itself (qwen4exp cuts routed expert
 * width by whole fp8 blocks, by parity). A width world does not divide is then left whole in
 * g.n_ff for the plugin to slice; otherwise it is refused, since g.n_ff is the per-rank width.
 *
 * `attn_world_ok`: the plugin's attention block supports a smaller attention world
 * (Geom::attn_world): it declares the attention-zero fill and zero KV group and spans projections
 * by `attn_rank`. Only rad_block_attn_gated_fp8.h does. Otherwise a world the KV head count
 * neither divides nor is a multiple of is refused. */
inline int geom_from(Geom& g, const RadModelMeta* m, const RadBuildCtx* c,
                     const char* dtype, uint32_t act_dtype, bool arch_splits_ff = false,
                     bool attn_world_ok = false) {
    if (!m || !c) return RAD_E_INVAL;
    if (c->is_draft) {
        /* A draft scope would need a second set of per-rank handles and name-map namespace; an
         * unimplemented path never returns success (spec §17). */
        fprintf(stderr, "radiance: this plugin has no draft scope\n");
        return RAD_E_UNSUPPORTED;
    }
    if (c->rank < 0 || c->rank >= MAX_RANKS) return RAD_E_INVAL;

    g.n_layer   = m->n_layers;
    g.n_embd    = m->n_embd;
    g.head_dim  = m->head_dim ? m->head_dim : (m->n_head ? m->n_embd / m->n_head : 0);
    g.n_head_all    = m->n_head;
    g.n_head_kv_all = m->n_head_kv ? m->n_head_kv : m->n_head;
    g.n_ff_all      = m->n_ff;
    g.n_vocab_all   = m->n_vocab;
    g.max_tok   = c->max_tok;
    g.max_seqs  = c->max_seqs;
    g.max_ctx   = c->max_ctx;
    g.kv_resident = c->kv_resident_blocks;
    g.max_spec  = c->max_spec;
    g.max_logit_rows = c->max_seqs * (c->max_spec > 0 ? c->max_spec + 1 : 1);
    if (g.max_logit_rows > c->max_tok) g.max_logit_rows = c->max_tok;
    /* KL mode reads a row at every prompt position plus the sampler's own rows, which repeat a
     * scored one (RadBuildCtx::max_out_rows), so it can exceed the token count. */
    if (c->max_out_rows > g.max_logit_rows) g.max_logit_rows = c->max_out_rows;
    g.rank      = c->rank;
    g.world     = c->world_size ? c->world_size : 1;
    g.wire_exact= c->tp_wire_lossy ? 0 : 1;
    g.wire_min_bytes = c->tp_wire_min_bytes;
    g.eps       = m->rms_eps;
    g.theta     = m->rope_theta;
    g.rope_scale= 1.0f;   /* yarn_from reads the model's rotary scaling */
    g.dtype     = dtype;
    g.act_dtype = act_dtype;
    {
        const char* sec = rad_meta_gets(m, "rope_parameters.mrope_section",
                                        rad_meta_gets(m, "text_config.rope_parameters.mrope_section",
                                                      ""));
        if (sec && *sec) {
            const bool inter =
                rad_meta_geti(m, "rope_parameters.mrope_interleaved",
                              rad_meta_geti(m, "text_config.rope_parameters.mrope_interleaved",
                                            0)) != 0;
            g.rope_mc = true;
            std::snprintf(g.rope_mode, sizeof g.rope_mode, "%s", inter ? "imrope" : "mrope");
            std::snprintf(g.rope_sections, sizeof g.rope_sections, "%s", sec);
        }
    }

    if (g.n_layer <= 0 || g.n_embd <= 0 || g.head_dim <= 0 || g.n_head_all <= 0 ||
        g.n_ff_all <= 0 || g.n_vocab_all <= 0 || g.max_tok <= 0)
        return RAD_E_INVAL;
    /* eps and theta are required, not defaulted: a wrong default yields fluent wrong text. */
    if (!(g.eps > 0.0f) || !(g.theta > 0.0f)) {
        fprintf(stderr, "radiance: container gives rms_eps=%g rope_theta=%g; both are required\n",
                (double)g.eps, (double)g.theta);
        return RAD_E_INVAL;
    }

    int s;
    /* The attention world (see Geom::attn_world): the world itself when it is at most the KV head
     * count or a multiple of it, else the largest multiple below it, with the ranks past it
     * attention-zero. */
    g.attn_world = g.world;
    g.attn_rank  = g.rank;
    g.attn_zero  = false;
    if (attn_world_ok && g.world > g.n_head_kv_all && g.world % g.n_head_kv_all != 0) {
        g.attn_world = (int)(g.n_head_kv_all * (g.world / g.n_head_kv_all));
        g.attn_zero  = g.rank >= g.attn_world;
        g.attn_rank  = g.attn_zero ? g.attn_world - 1 : g.rank;
        fprintf(stderr, "radiance: world_size = %d over n_head_kv = %lld: attention on %d rank(s), "
                        "rank(s) %d..%d carry no attention and take the delta net's remainder "
                        "first\n", g.world, (long long)g.n_head_kv_all,
                g.attn_world, g.attn_world, g.world - 1);
    }
    const int aw = g.attn_world, ar = g.attn_rank;
    if ((s = divide_or_fail(g.n_head_all, aw, "n_head", &g.n_head)) < 0) return s;
    /* KV heads are replicated when there are fewer than ranks, which requires world to be a
     * multiple of the KV head count. With 24 query heads over 2 KV heads at world 4, ranks {0,1}
     * need only KV head 0 and ranks {2,3} only KV head 1. Each rank declares n_head_kv 1 and reads
     * its head's row-span of k/v at `kv_off`; kv_dim(), gqa() and the blocks then treat a
     * replicated rank like a split one. The replicating ranks hold identical KV weights, at no
     * extra VRAM per rank versus the TP2 split. */
    if (aw > g.n_head_kv_all) {
        /* Holds for the attention world by construction; checked for the plain world. */
        if (aw % g.n_head_kv_all != 0) {
            fprintf(stderr, "radiance: world_size = %d cannot replicate n_head_kv = %lld; "
                            "replication needs world to be a multiple of the KV head count\n",
                    aw, (long long)g.n_head_kv_all);
            return RAD_E_INVAL;
        }
        g.n_head_kv = 1;   /* aw / n_head_kv_all ranks share each KV head */
        g.kv_off    = (ar / (aw / g.n_head_kv_all)) * g.head_dim;
    } else if ((s = divide_or_fail(g.n_head_kv_all, aw, "n_head_kv", &g.n_head_kv)) < 0) {
        return s;
    } else {
        /* A split rank uses the plain row-shard, except under a smaller attention world, where
         * the even shard over `world` is the wrong cut and the block spans
         * [kv_off, kv_off + kv_dim) instead. */
        g.kv_off = aw == g.world ? 0 : ar * g.n_head_kv * g.head_dim;
    }
    if (arch_splits_ff && g.world > 1 && g.n_ff_all % g.world != 0) {
        g.n_ff = g.n_ff_all;   /* the plugin slices it (see the parameter's comment) */
    } else if ((s = divide_or_fail(g.n_ff_all, g.world, "n_ff", &g.n_ff)) < 0) {
        return s;
    }
    /* lm_head is row-sharded by vocabulary (see vocab_per_rank). r4d_sample_topk_f32 takes a
     * vocabulary offset and packs candidates as (u32 global id, f32 logit) pairs; the all-gather
     * places every rank's pairs side by side, the layout sample_merge_topk reads. */
    g.n_vocab   = shard_extent(g.n_vocab_all, g.world, g.rank);
    g.vocab_off = shard_begin (g.n_vocab_all, g.world, g.rank);
    if (g.n_vocab <= 0) return RAD_E_INVAL;
    return RAD_OK;
}

/* ================================================================== a vocab-sharded draft head */
/* Whether a drafter's own head can be vocab-sharded. Shared so the block declaring the op and the
 * plugin sizing its buffer agree; a disagreement is a read past the buffer's end.
 *
 * Requires an even split in multiples of 16: the packed w2a8 layout is tile-major at 16 output
 * rows a block, so an unaligned share splits a block across ranks, and the candidate merge's
 * global ids are `rank * n_vocab + local`, true only for a contiguous equal split. */
inline bool vocab_head_sharded(const Geom& g) {
    return g.world > 1 && g.n_vocab > 0 && (g.n_vocab % 16) == 0 &&
           g.n_vocab * (int64_t)g.world == g.n_vocab_all &&
           g.vocab_off == (int64_t)g.rank * g.n_vocab;
}
/* Rows of such a head on this rank, and so the width of its logit plane. */
inline int64_t vocab_head_rows(const Geom& g) {
    return vocab_head_sharded(g) ? g.n_vocab : g.n_vocab_all;
}

/* ================================================================== the KV cache operand */
/* The cache operand for attn_paged and kv_store. The pool base belongs to the block manager, not
 * the batch, so the operand names the group and layer and the core resolves it like a weight
 * handle. */
inline RadOperand kv_cache(rad_kvgroup group, int layer) { return RAD_KV(group, layer); }

/* One layer of a streamed group (RadKVStream): its rows of the host mirror and the ring's maps.
 * The mirror is layer-major over the group's bound layers in KVPoolBinding::layer_slot order, not
 * over model layers (Flash-Next's full-attention layers are every other layer, plus the MTP head),
 * so the row comes from rad_kv_layer_index. `host` is null when the group does not stream on this
 * rank or has no row for the layer. */
struct KVStreamLayer {
    const RadKVStream* st = nullptr;
    const uint8_t* host = nullptr;   /* [blocks, bbytes] mirror row */
    int32_t* res = nullptr;          /* [blocks] block -> ring slot, -1 when absent */
    int32_t* slot_block = nullptr;   /* [slots] ring slot -> block */
    int32_t* slot_state = nullptr;   /* [slots, 2] */
    int64_t  blocks = 0, bbytes = 0, slots = 0;
};
inline KVStreamLayer kv_stream_layer(RadCtx* c, rad_kvgroup group, int layer,
                                     const RadKVGroupBatch* kvb) {
    KVStreamLayer L;
    const RadKVStream* st = kvb ? kvb->stream : nullptr;
    if (!st || !st->host || st->block_bytes <= 0) return L;
    const int64_t li = rad_kv_layer_index(c, group, layer);
    if (li < 0) return L;
    L.st         = st;
    L.bbytes     = st->block_bytes;
    L.blocks     = st->host_layer_stride / L.bbytes;
    L.slots      = st->slots;
    L.host       = st->host + li * st->host_layer_stride;
    L.res        = st->res + li * L.blocks;
    L.slot_block = st->slot_block + li * L.slots;
    L.slot_state = st->slot_state + li * L.slots * 2;
    return L;
}

/* The prefill stage of a streamed group (the ordering rule is in core/kvstream/stage.h):
 *
 *   kv_stage_wait  -- immediately before the layer's store: the copy from the mirror must land
 *                     before the store writes this chunk's newer bytes into the stage;
 *   kv_stage_layer -- before the attention: true when the pass uses the stage, so the caller
 *                     reads block ids directly and must not resolve;
 *   kv_stage_next  -- after every attention launch of the layer: the next layer's copy starts
 *                     once the last read of the aliased stage is behind it.
 *
 * Both of the first two are rad_kv_stage_layer, which stages a layer once per pass. */
inline bool kv_stage_layer(RadCtx* c, rad_kvgroup group, const RadKVGroupBatch* kvb, int layer) {
    return rad_kv_stage_layer(c, group, kvb, layer) != 0;
}
inline bool kv_stage_wait(RadCtx* c, rad_kvgroup group, const RadKVGroupBatch* kvb, int layer) {
    return rad_kv_stage_layer(c, group, kvb, layer) != 0;
}
inline void kv_stage_next(RadCtx* c, rad_kvgroup group, int layer) {
    (void)rad_kv_stage_next(c, group, layer);
}

/* The `kv_resolve` a streamed group's block declares. None of its geometry is free:
 *
 *   blocks      the block-id space a selection can name: max_seqs sequences of the context;
 *   slots       the ring --kv-resident carves, in whole blocks, raised to one decode step's
 *               selection (max_seqs x (1 + n_spec) x sel_pitch) as kvstream_ring raises it, and
 *               at most `blocks`; the resolve is issued with the pool's depth, so declaring the
 *               context would size its scratch for slots it cannot reach;
 *   sel_pitch   the selection row's pitch, which the attention declares its rows with
 *               (`qsa_topk + 1` on the sparse path, the indexer's count on GLM).
 *
 * The kernel sizes its claim list from these (`M * sel_pitch`, r4d_kv_resolve.hip's
 * kv_claim_cap); ctl[12] reports the largest claim count a launch made, so a wrong declaration
 * shows in the engine's stream report. `sel` is read as well as written, so the arena keeps it
 * live from its producer to the attention. Returns 0 when the op is not declared or the group has
 * nothing to address. */
inline rad_op kv_resolve_declare(RadBuilder* b, const Geom& g, rad_kvgroup kv, rad_buf sel,
                                 rad_buf nsel, int64_t sel_pitch) {
    if (!rad_op_declared(b, "kv_resolve") || !sel || !nsel || sel_pitch <= 0) return 0;
    const int64_t bs     = rad_kv_block_size(b, kv);
    const int64_t bbytes = rad_kv_layer_bytes(b, kv);
    const int64_t blocks = bs > 0 ? g.max_seqs * ((g.max_ctx + bs - 1) / bs) : 0;
    if (bs <= 0 || bbytes <= 0 || blocks <= 0) return 0;
    const int64_t ring  = g.kv_resident > 0 ? (g.kv_resident + bs - 1) / bs : 0;
    const int64_t step  = g.max_seqs * (1 + (g.max_spec > 0 ? g.max_spec : 0)) * sel_pitch;
    const int64_t want  = ring > 0 && step > ring ? step : ring;
    const int64_t slots = want > 0 && want < blocks ? want : blocks;
    const rad_op op = RAD_OP(b, "kv_resolve",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("blocks", blocks),
                                        RAD_INT("slots", slots), RAD_INT("sel_pitch", sel_pitch),
                                        RAD_INT("block_bytes", bbytes)),
                             RAD_NOWEIGHTS);
    if (!op) return 0;
    const rad_buf rd[2] = { sel, nsel };
    rad_op_reads(b, op, rd, 2);
    rad_op_writes(b, op, &sel, 1);
    return op;
}

/* ================================================================== step-sized operands */
/* Every operand a step issues carries the step's row count, not the buffer's.
 *
 * A ranged parameter is collapsed to a band at resolve time (spec §2.2), so a kernel cannot read
 * the row count from `M` (it would get the band's upper bound); kernels read extents from the
 * operands (libref/ref_ops.h). An operand at its declared extent makes a one-row decode compute
 * over the whole arena buffer, reading uninitialised memory and overwriting the next op's input.
 *
 * `rows` is dim 0, tokens for every activation buffer here. Strides are products of the trailing
 * dimensions, so shortening dim 0 keeps them correct (core/runtime/issue.cpp).
 */
inline RadOperand brows(rad_buf h, int64_t rows) {
    RadOperand o = RAD_B(h);
    o.rows = rows;
    return o;
}

/* A column slice narrowed to this step's rows. */
inline RadOperand bcol(rad_buf h, int64_t first, int64_t width, int64_t rows) {
    RadOperand o = RAD_B_COL(h, first, width);
    o.rows = rows;
    return o;
}

/* A column slice of a range of rows. `offset` is a flat element offset and `cols` narrows the last
 * dimension keeping the declared width as stride, so first_row * pitch picks the row and
 * first_col the column (core/runtime/issue.cpp). Used by mixed steps, which issue the recurrent
 * path over the decode rows and the chunked path over the prefill rows, each over column views
 * of a fused projection. */
inline RadOperand bcol_at(rad_buf h, int64_t first_row, int64_t pitch,
                          int64_t first_col, int64_t width, int64_t rows) {
    RadOperand o = RAD_B(h);
    o.offset = first_row * pitch + first_col;
    o.rows   = rows;
    o.cols   = width;
    return o;
}

/* Whether every logits row this step samples belongs to a sequence that carries words for the
 * plugin (RadBatch::plugin_words). A head that answers from those words then has no use for the
 * lm_head's row: it is sampled once and discarded. One row a sampled sequence; a speculative
 * step samples more and answers false. */
inline bool plugin_words_cover_rows(const RadBatch* b) {
    if (b->n_out <= 0 || !b->plugin_cu_h) return false;
    int64_t n = 0;
    for (int64_t s = 0; s < b->n_seq; ++s) n += b->plugin_cu_h[s + 1] > b->plugin_cu_h[s] ? 1 : 0;
    return n == b->n_out;
}

/* Rows [first, first + n), seen by the kernel as [n, pitch]; RadOperand::offset is in elements.
 *
 * Lets one op be issued in slices: an op whose widest kernel stops at M=64 (the delta net's
 * 96-column a|b projection) would otherwise cap every op in the step at 64 tokens. Slicing it
 * costs a launch per 64 rows. */
inline RadOperand brow_slice(rad_buf h, int64_t first, int64_t n, int64_t pitch) {
    RadOperand o = RAD_B(h);
    o.offset = first * pitch;
    o.rows   = n;
    return o;
}

/* A dense two-dimensional batch field, [rows, cols] packed: the block table (RAD_P_T2 in
 * rad_runtime.h says why rank 1 is not enough). */
inline RadOperand praw2(const void* p, uint32_t dtype, int64_t rows, int64_t cols) {
    return RAD_P_T2(p, dtype, rows, cols);
}

/* A batch field. The core did not allocate it, so it has no declared shape and the row count
 * comes from here; without it the operand is rank 0 and a kernel sizing its loop by
 * numel(positions) runs one token. */
inline RadOperand praw(const void* p, uint32_t dtype, int64_t rows) {
    RadOperand o = RAD_P_T(p, dtype);
    o.rows = rows;
    return o;
}

/* kv_resolve over `rows` selection rows (kv_resolve_declare): makes the blocks they name resident
 * in the ring and rewrites `sel` from block ids to ring slots. */
inline void kv_resolve_issue(RadCtx* c, rad_op op, const KVStreamLayer& L, RadOperand sel,
                             RadOperand nsel, rad_kvgroup group, int layer, int64_t rows) {
    RAD_ISSUE_N(c, op, rows, sel, nsel, praw(L.res, RAD_I32, L.blocks),
                praw(L.slot_block, RAD_I32, L.slots), praw2(L.slot_state, RAD_I32, L.slots, 2),
                praw(L.st->ctl, RAD_I32, 16), praw2(L.host, RAD_U8, L.blocks, L.bbytes),
                kv_cache(group, layer));
}

/* Rotary positions of a pass, in the two forms consumers take.
 *
 * `rope_pos1`: one component per token, RadBatch::rope_pos's plane 0 if present, else
 * `positions`. A cos/sin table is filled and read at it. It is exact only when every token's
 * components agree (RadBatch::rope_mixed 0), so fused paths that read it are taken only then.
 *
 * `rope_posmc`: what a multi-component `rope` takes. The dense [3, T] planes for an M-RoPE model
 * whose batch carries them, else the one-component form, which every mode reads as NeoX. */
inline RadOperand rope_pos1(const RadBatch* b, int64_t T) {
    return praw(b->rope_pos ? b->rope_pos : b->positions, RAD_I32, T);
}
inline RadOperand rope_posmc(const Geom& g, const RadBatch* b, int64_t T) {
    if (g.rope_mc && b->rope_pos) return praw2(b->rope_pos, RAD_I32, 3, T);
    return praw(b->positions, RAD_I32, T);
}
/* True when some token's rotary components differ, which a one-component fused path cannot
 * rotate. */
inline bool rope_mixed(const RadBatch* b) { return b->rope_pos && b->rope_mixed; }

/* A rotary op declared with the geometry's YaRN coefficients appended. Without YaRN the
 * parameters are passed unchanged, so op identities and tune keys do not change. */
inline rad_op decl_rotary(RadBuilder* b, const Geom& g, const char* name,
                          const RadParam* params, int n, const rad_weight* weights, int nw) {
    if (g.yarn.factor == 0.0f) return rad_decl_op(b, name, params, n, weights, nw);
    std::vector<RadParam> p(params, params + n);
    p.push_back(RAD_F64("yarn_factor", g.yarn.factor));
    p.push_back(RAD_F64("yarn_low", g.yarn.low));
    p.push_back(RAD_F64("yarn_high", g.yarn.high));
    p.push_back(RAD_F64("yarn_magnitude", g.yarn.magnitude));
    return rad_decl_op(b, name, p.data(), (int)p.size(), weights, nw);
}

/* Reads a model's rotary scaling into `g.rope_scale` and `g.yarn`. Every architecture calls it
 * after geom_from, for the target ("") and for a drafter with its own config ("draft."). Only the
 * free-form keys are read; RadModelMeta::rope_scale is not, since a factor without its type would
 * be applied as a linear divide.
 *
 * The object read is the first of `text_config.rope_parameters.`, `rope_parameters.`,
 * `text_config.rope_scaling.`, `rope_scaling.` (a drafter: `draft.rope_parameters.`,
 * `draft.rope_scaling.`) that names a type in `rope_type` or `type`; a multimodal target's
 * text_config is authoritative. core/format/rope_override.h writes --rope-scaling into the same
 * object, found in the same order.
 *
 *   default, mrope   no scaling (`mrope` is the legacy Qwen2-VL spelling of default + sections)
 *   linear           positions divided by `factor`
 *   yarn             static YaRN: frequencies blended over the correction range that `factor`,
 *                    `original_max_position_embeddings`, `beta_fast`/`beta_slow` and `truncate`
 *                    give, cos/sin scaled by `attention_factor` (default 1 + 0.1 ln factor). The
 *                    factor is the config's, read as vLLM and transformers read it.
 *
 * Anything else -- llama3, longrope, dynamic -- is refused by name, as is YaRN's `mscale` form. A
 * refusal leaves the geometry as it was. */
inline int yarn_from(Geom& g, const RadModelMeta* meta, const char* prefix, int64_t rot) {
    const std::string root(prefix);
    const char* who = root.empty() ? "target" : "draft";
    const std::vector<std::string> homes = root.empty()
        ? std::vector<std::string>{"text_config.rope_parameters.", "rope_parameters.",
                                    "text_config.rope_scaling.", "rope_scaling."}
        : std::vector<std::string>{root + "rope_parameters.", root + "rope_scaling."};
    std::string home, type_key;
    const char* type = nullptr;
    for (const auto& h : homes) {
        for (const char* t : {"rope_type", "type"})
            if ((type = rad_meta_gets(meta, (h + t).c_str(), nullptr))) { type_key = h + t; break; }
        if (type) { home = h; break; }
    }
    if (!type || !std::strcmp(type, "default") || !std::strcmp(type, "mrope")) {
        g.yarn = {};
        g.rope_scale = 1.0f;
        return RAD_OK;
    }
    auto get = [&](const char* key, double fallback) {
        const char* text = rad_meta_gets(meta, (home + key).c_str(), nullptr);
        if (!text) return fallback;
        char* end = nullptr;
        const double value = std::strtod(text, &end);
        if (end == text) return (double)NAN;
        while (*end == ' ' || *end == '\t' || *end == '\n') ++end;
        return *end ? (double)NAN : value;
    };
    if (!std::strcmp(type, "linear")) {
        const double factor = get("factor", 0.0);
        const float scale = (float)factor;
        if (!std::isfinite(factor) || factor <= 0.0 || !std::isfinite(scale) || scale <= 0.0f) {
            fprintf(stderr, "radiance: %s linear rotary scaling (%s) needs a finite factor > 0\n",
                    who, home.c_str());
            return RAD_E_INVAL;
        }
        g.yarn = {};
        g.rope_scale = scale;
        return RAD_OK;
    }
    if (std::strcmp(type, "yarn")) {
        fprintf(stderr, "radiance: %s rotary type '%s' (%s) is not implemented; default, linear "
                        "and yarn are\n", who, type, type_key.c_str());
        return RAD_E_UNSUPPORTED;
    }
    /* `truncate` is a JSON boolean; the converter writes 1 or 0, and true/false are accepted. */
    double truncate = 1.0;
    if (const char* t = rad_meta_gets(meta, (home + "truncate").c_str(), nullptr))
        truncate = !std::strcmp(t, "true") ? 1.0 : !std::strcmp(t, "false") ? 0.0 : get("truncate", 1.0);
    const double factor = get("factor", 0.0), original = get("original_max_position_embeddings", 0.0);
    const double fast = get("beta_fast", 32.0), slow = get("beta_slow", 1.0);
    const double magnitude = get("attention_factor", factor > 1.0 ? 1.0 + 0.1 * std::log(factor) : 1.0);
    if (!std::isfinite(factor) || factor < 1.0 || !std::isfinite(original) || original < 1.0 ||
        original != std::floor(original) || (truncate != 0.0 && truncate != 1.0) ||
        !std::isfinite(fast) || !std::isfinite(slow) || slow <= 0.0 || fast < slow ||
        !std::isfinite(magnitude) || magnitude <= 0.0 || !std::isfinite(g.theta) || g.theta <= 1.0f ||
        rot < 2 || (rot & 1)) {
        fprintf(stderr, "radiance: %s YaRN (%s) needs a finite factor >= 1, an integral "
                        "original_max_position_embeddings > 0, beta_fast >= beta_slow > 0, "
                        "attention_factor > 0 and truncate true or false\n", who, home.c_str());
        return RAD_E_INVAL;
    }
    /* mscale variants are a different contract; never silently ignore them. */
    for (const char* k : {"mscale", "mscale_all_dim"})
        if (rad_meta_gets(meta, (home + k).c_str(), nullptr)) {
            fprintf(stderr, "radiance: %s YaRN %s%s is not implemented; attention_factor is\n",
                    who, home.c_str(), k);
            return RAD_E_UNSUPPORTED;
        }
    auto correction = [&](double beta) {
        return rot * std::log(original / (beta * 2.0 * 3.14159265358979323846)) /
               (2.0 * std::log((double)g.theta));
    };
    double low = correction(fast), high = correction(slow);
    if (truncate != 0.0) { low = std::floor(low); high = std::ceil(high); }
    const RadYarn y = {(float)factor, (float)std::max(0.0, low),
                       (float)std::min((double)rot - 1, high), (float)magnitude};
    if (!rad_yarn_valid(y)) {
        fprintf(stderr, "radiance: %s YaRN (%s) resolves to an empty correction range\n", who,
                home.c_str());
        return RAD_E_INVAL;
    }
    g.yarn = y;
    g.rope_scale = 1.0f; /* YaRN blends frequencies; it does not divide positions. */
    return RAD_OK;
}

/* `rope` declared for the geometry's rotary: M-RoPE's mode and sections when the model has them,
 * `mode` otherwise. `op` names the row: "rope" (every caller), or "rope_rows" for a model that
 * wants the per-(token, pair) angles form at prefill row counts. */
inline rad_op decl_rope(RadBuilder* b, const Geom& g, int64_t rows, int64_t head_dim,
                        int64_t n_head, int64_t n_head_kv, int64_t rot, const char* mode,
                        const char* op = "rope") {
    const RadParam ps[] = { RAD_RANGE("M", 1, rows), RAD_INT("head_dim", head_dim),
                            RAD_INT("n_head", n_head), RAD_INT("n_head_kv", n_head_kv),
                            RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                            RAD_STR("mode", g.rope_mc ? g.rope_mode : mode),
                            RAD_INT("rotary_dim", rot), RAD_STR("sections", g.rope_sections) };
    const int n = (int)(sizeof ps / sizeof ps[0]) - (g.rope_mc ? 0 : 1);
    return decl_rotary(b, g, op, ps, n, nullptr, 0);
}

/* How many of this step's sequences are decode rows, and where their tokens end.
 *
 * A mixed step carries the boundary in RadBatch::n_seq_decode / n_tok_decode. For a pure step the
 * phase decides (PREFILL: none, DECODE: all), so a caller that sets only the phase (a test, a
 * bench, an out-of-tree caller) still gets the right path; tests/arch_test.cpp covers this. The
 * core sets both fields on every step, so the two agree. */
inline void batch_split(const RadBatch* b, int64_t* n_seq_dec, int64_t* n_tok_dec) {
    switch (b->phase) {
        case RAD_PHASE_PREFILL: *n_seq_dec = 0;              *n_tok_dec = 0;              break;
        case RAD_PHASE_DECODE:  *n_seq_dec = b->n_seq;       *n_tok_dec = b->n_tok;       break;
        default:                *n_seq_dec = b->n_seq_decode; *n_tok_dec = b->n_tok_decode; break;
    }
}

/* The step batch's entry for one declared KV group (built once per step by the core, spec §8). */
inline const RadKVGroupBatch* kv_batch(const RadBatch* b, rad_kvgroup group) {
    for (int i = 0; i < b->n_kv_groups; ++i)
        if (b->kv[i].group == group) return &b->kv[i];
    return nullptr;
}

/* ================================================================== calibration in slices */
/* The layers [lo, hi) of n a calibration taps: the environment variable `var` as "lo:hi", else
 * all of them. A large model's Grams do not fit beside its weights at once, so it is calibrated
 * in slices of layers, the corpus pushed through once a slice. */
inline int calib_layers(const char* who, const char* var, int64_t n, int64_t* lo, int64_t* hi) {
    *lo = 0;
    *hi = n;
    const char* s = std::getenv(var);
    if (!s) return RAD_OK;
    long long l = 0, h = 0;
    int used = 0;
    if (std::sscanf(s, "%lld:%lld%n", &l, &h, &used) != 2 || s[used] != '\0' || l < 0 || h > n ||
        l >= h) {
        fprintf(stderr, "radiance: %s: %s '%s' is not lo:hi within 0:%lld\n", who, var, s,
                (long long)n);
        return RAD_E_INVAL;
    }
    *lo = l;
    *hi = h;
    return RAD_OK;
}

/* ================================================================== a block drafter */
/* A drafter's Geom: the target's (n_embd, vocabulary, deployment bounds, rank, wire) with the
 * layer, head, feed-forward, norm and rotary geometry of the container's `draft.` config, split
 * over the ranks, and a plain RMS norm. A drafter is a second model, so nothing is defaulted: a
 * config that does not describe one is refused, naming `what` (spec §17). So is a vocabulary
 * other than the target's, since a drafter reads the target's embedding and lm_head.
 *
 * The layer config sits at the top of `draft.` in z-lab's and SpecForge's checkpoints, and under
 * `draft.transformer_layer_config.` in vLLM's speculators format (RedHatAI's DFlash2 drafters),
 * whose top level holds the speculator's own keys instead (drafter_layer_prefix). */
inline const char* drafter_layer_prefix(const RadModelMeta* meta) {
    if (rad_meta_geti(meta, "draft.num_hidden_layers", 0) > 0) return "draft.";
    if (rad_meta_geti(meta, "draft.transformer_layer_config.num_hidden_layers", 0) > 0)
        return "draft.transformer_layer_config.";
    return "draft.";
}
/* `kv_repl_ok`: the drafter block reads its k and v at Geom::kv_off (rad_block_dflash2.h does), so
 * fewer KV heads than ranks are replicated as geom_from replicates the target's: world a multiple
 * of the head count, one KV head a rank, `kv_off` its first row in the unsharded k/v projection.
 * Otherwise such a world is refused, as before. */
inline int drafter_geom(Geom& dg, const Geom& target, const RadModelMeta* meta, const char* what,
                        bool kv_repl_ok = false) {
    dg = target;
    const char* pre = drafter_layer_prefix(meta);
    char key[160];
    auto k = [&](const char* name) { std::snprintf(key, sizeof key, "%s%s", pre, name); return key; };
    dg.n_layer       = rad_meta_geti(meta, k("num_hidden_layers"), 0);
    dg.head_dim      = rad_meta_geti(meta, k("head_dim"), 0);
    dg.n_head_all    = rad_meta_geti(meta, k("num_attention_heads"), 0);
    dg.n_head_kv_all = rad_meta_geti(meta, k("num_key_value_heads"), 0);
    dg.n_ff_all      = rad_meta_geti(meta, k("intermediate_size"), 0);
    dg.eps           = (float)rad_meta_getf(meta, k("rms_norm_eps"), 0.0);
    const double th  = rad_meta_getf(meta, k("rope_theta"), 0.0);
    dg.theta         = (float)rad_meta_getf(meta, k("rope_parameters.rope_theta"), th);
    dg.wadd = dg.wadd_qk = 0.0f;
    if (dg.n_layer <= 0 || dg.head_dim <= 0 || dg.n_head_all <= 0 || dg.n_head_kv_all <= 0 ||
        dg.n_ff_all <= 0 || !(dg.eps > 0.0f) || !(dg.theta > 0.0f)) {
        fprintf(stderr, "radiance: the container's `draft.` config does not describe a %s "
                        "drafter: layers %lld, heads %lld / kv %lld, head_dim %lld, ffn %lld, "
                        "rms_eps %g, rope_theta %g\n", what,
                (long long)dg.n_layer, (long long)dg.n_head_all, (long long)dg.n_head_kv_all,
                (long long)dg.head_dim, (long long)dg.n_ff_all, (double)dg.eps, (double)dg.theta);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(yarn_from(dg, meta, pre, dg.head_dim));
    RAD_ARCH_TRY(divide_or_fail(dg.n_head_all,    dg.world, "draft num_attention_heads", &dg.n_head));
    /* The target's kv_off is the target's; the drafter's is set here or zero. */
    dg.kv_off = 0;
    if (kv_repl_ok && dg.world > dg.n_head_kv_all && dg.world % dg.n_head_kv_all == 0) {
        dg.n_head_kv = 1;   /* world / n_head_kv_all ranks share each KV head */
        dg.kv_off    = (dg.rank / (dg.world / dg.n_head_kv_all)) * dg.head_dim;
    } else {
        RAD_ARCH_TRY(divide_or_fail(dg.n_head_kv_all, dg.world, "draft num_key_value_heads",
                                    &dg.n_head_kv));
    }
    RAD_ARCH_TRY(divide_or_fail(dg.n_ff_all,      dg.world, "draft intermediate_size",   &dg.n_ff));
    const long long dvocab = rad_meta_geti(meta, k("vocab_size"), dg.n_vocab_all);
    if (dvocab != dg.n_vocab_all) {
        fprintf(stderr, "radiance: the %s drafter claims a %lld-token vocabulary and the target "
                        "has %lld; it reads the target's embedding and lm_head\n",
                what, dvocab, (long long)dg.n_vocab_all);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* The drafter's own KV group, bound to its layers from `base_layer`: it differs from the trunk's
 * in layer count and head geometry. A `window` > 0 makes it RAD_KV_WINDOW, which recycles blocks
 * below the window. Its dtype is --kv-cache-dtype, like every attention cache, and is written to
 * dg.kv_dtype. */
inline int drafter_kv_group(RadBuilder* b, Names& nm, const RadBuildCtx* ctx, Geom& dg,
                            int64_t window, int base_layer, rad_kvgroup* out) {
    RadKVGroupDecl d{};
    d.kind      = window > 0 ? RAD_KV_WINDOW : RAD_KV_FULL;
    d.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    d.window    = window;
    d.n_head_kv = dg.n_head_kv;
    d.head_dim  = dg.head_dim;
    *out = rad_decl_kv_group(b, nm.f("kv_draft"), &d);
    for (int64_t l = 0; l < dg.n_layer; ++l)
        RAD_ARCH_TRY(rad_bind_layer_kv(b, base_layer + (int)l, *out));
    dg.kv_dtype = rad_dtype_name(d.dtype);
    return RAD_OK;
}

/* ================================================================== the plugin exports */
/* The C symbols spec §2.4 names, wired to a namespace. RAD_ARCH_NO_EXPORTS lets the unit test
 * include two plugin sources in one translation unit without the exports colliding, so the
 * declared graph is testable with no GPU and no model file.
 *
 * QUANT is the quantisation descriptor this plugin serves, "" for unquantised: the second half of
 * the selection key (spec §2.4, rad_builder.h). qwen35_bf16 and qwen35_fp8 both answer "qwen35"
 * to rad_arch_id and differ only here. */
#define RAD_ARCH_PLUGIN(NS, ID, QUANT, VERSION, DESC)                                          \
    extern "C" const char* rad_arch_id(void) { return ID; }                                    \
    extern "C" const char* rad_arch_quant(void) { return QUANT; }                              \
    extern "C" int rad_arch_declare(RadBuilder* b, const RadModelMeta* m,                      \
                                    const RadBuildCtx* c) { return NS::declare(b, m, c); }     \
    extern "C" void rad_arch_step(struct RadCtx* c, const struct RadBatch* batch) {            \
        NS::step(c, batch);                                                                    \
    }                                                                                          \
    extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }               \
    extern "C" uint64_t rad_plugin_abi_layout(void) { return rad_abi_layout(); }               \
    extern "C" const RadPluginInfo* rad_plugin_info(void) {                                    \
        /* The plugin's name (rad_abi.h: "qwen3_dense_bf16"), not the architecture id: plugin   \
         * names must be unique and one architecture can have several plugins. */               \
        static const RadPluginInfo info = { RAD_PLUGIN_ARCH, #NS, VERSION, DESC, "" };         \
        return &info;                                                                          \
    }

/* Optional pre-declare probe. Without it every RadArchProbe field stays zero. Kept out of
 * RAD_ARCH_PLUGIN so adding a RadArchProbe field never breaks an existing plugin's build. */
#define RAD_ARCH_PROBE(NS)                                                                     \
    extern "C" int rad_arch_probe(const RadModelMeta* m, RadArchProbe* out) {                  \
        return NS::probe(m, out);                                                              \
    }

/* Optional reply format, for a model whose reasoning and tool-call syntax the chat template
 * analysis misses or misreads. The reply parser, lazy call grammar and trigger words follow from
 * it. NS::chat_format returns static storage with struct_size = sizeof(RadChatFormat), or null to
 * let the template decide for this model. */
#define RAD_ARCH_CHAT_FORMAT(NS)                                                               \
    extern "C" const RadChatFormat* rad_arch_chat_format(const RadModelMeta* m) {              \
        return NS::chat_format(m);                                                             \
    }

/* The audio format, also opt-in: a model with an audio encoder states how its prompt carries a
 * clip and asks for a transcript (RadAudioFormat). NS::audio_format returns static storage with
 * struct_size = sizeof(RadAudioFormat), or null for a model that serves no speech route. */
#define RAD_ARCH_AUDIO_FORMAT(NS)                                                              \
    extern "C" const RadAudioFormat* rad_arch_audio_format(const RadModelMeta* m) {            \
        return NS::audio_format(m);                                                            \
    }

/* Optional: RAD_ARCH_SERVES_IMPORTS(0) keeps a family's second quantised plugin out of the pick for
 * an imported checkpoint (rad_builder.h, rad_arch_serves_imports). */
#define RAD_ARCH_SERVES_IMPORTS(V)                                                             \
    extern "C" int rad_arch_serves_imports(void) { return V; }

/* RAD_ARCH_MULTI_INSTANCE() states that the plugin keeps all per-instance state by slot, so several
 * engines in one process may run it at once (rad_builder.h, rad_arch_multi_instance). */
#define RAD_ARCH_MULTI_INSTANCE()                                                              \
    extern "C" int rad_arch_multi_instance(void) { return 1; }

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_ARCH_H */
