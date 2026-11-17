/* avx_harness.h -- plugin loading, operand sizing and drawing, and the geometry table.
 *
 * Shared by rad-avx-check and rad-avx-bench so that both run the same extents from one table
 * (tools/opshapes.h does the same for rad-kbench and rad-tune). Header-only for two translation
 * units, so everything is `static` or `inline`. Neither binary links a plugin: both dlopen by path,
 * so the results describe the .so the engine would load.
 */
#ifndef RAD_AVX_HARNESS_H
#define RAD_AVX_HARNESS_H

#include "rad_abi.h"
#include "rad_plugin.h"
#include "rad_sample.h"
#include "avx_common.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ================================================================== a loaded plugin */
struct Lib {
    void* h = nullptr;
    std::string path;
    const RadPluginInfo* info = nullptr;
    int (*kernel_count)() = nullptr;
    const RadKernelInfo* (*kernel_at)(int) = nullptr;
    int (*schema_count)() = nullptr;
    const RadOpSchema* (*schema_at)(int) = nullptr;
    int (*plugin_open)() = nullptr;
    /* libavx-only exports; null for other plugins, which tells a caller it cannot sweep ISA
     * levels or force its parallel regions. */
    int (*isa_force)(int) = nullptr;
    int (*isa_detect)() = nullptr;
    const char* (*isa_name)(int) = nullptr;
    int (*par_force)(int) = nullptr;

    bool open(const char* p) {
        path = p;
        h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", p, dlerror()); return false; }
        auto sym = [&](const char* n) { return dlsym(h, n); };
        auto abi = (uint32_t (*)())sym("rad_plugin_abi_version");
        auto inf = (const RadPluginInfo* (*)())sym("rad_plugin_info");
        kernel_count = (int (*)())sym("rad_kernel_count");
        kernel_at    = (const RadKernelInfo* (*)(int))sym("rad_kernel_at");
        schema_count = (int (*)())sym("rad_kernel_schema_count");
        schema_at    = (const RadOpSchema* (*)(int))sym("rad_kernel_schema_at");
        plugin_open  = (int (*)())sym("rad_plugin_open");
        isa_force    = (int (*)(int))sym("avx_isa_force");
        isa_detect   = (int (*)())sym("avx_isa_detect");
        isa_name     = (const char* (*)(int))sym("avx_isa_name");
        par_force    = (int (*)(int))sym("avx_par_force");
        if (!abi || !inf || !kernel_count || !kernel_at || !schema_count || !schema_at) {
            std::fprintf(stderr, "%s: not a kernel plugin (missing exports)\n", p);
            return false;
        }
        if (abi() != RAD_ABI_VERSION) {
            /* Same rule as the engine's loader: refuse any version but this one, so the tool never
             * measures a plugin the engine would not load. */
            std::fprintf(stderr, "%s: ABI %u, this tool is built for %u\n", p, abi(),
                         RAD_ABI_VERSION);
            return false;
        }
        info = inf();
        if (plugin_open && plugin_open() < 0) {
            std::fprintf(stderr, "%s: rad_plugin_open failed\n", p);
            return false;
        }
        return true;
    }

    const RadKernelInfo* row(const char* op) const {
        for (int i = 0, n = kernel_count(); i < n; ++i) {
            const RadKernelInfo* k = kernel_at(i);
            if (k && k->op && std::strcmp(k->op, op) == 0) return k;
        }
        return nullptr;
    }
    const RadOpSchema* schema(const char* op) const {
        for (int i = 0, n = schema_count(); i < n; ++i) {
            const RadOpSchema* s = schema_at(i);
            if (s && s->op && std::strcmp(s->op, op) == 0) return s;
        }
        return nullptr;
    }
    /* True when some row of the op runs on the host, i.e. can be launched with host pointers. */
    bool host_row(const char* op) const {
        for (int i = 0, n = kernel_count(); i < n; ++i) {
            const RadKernelInfo* k = kernel_at(i);
            if (k && k->op && std::strcmp(k->op, op) == 0 && k->domain == RAD_DOMAIN_HOST)
                return true;
        }
        return false;
    }

    /* The host row the selector would resolve this geometry to: every constraint holds, highest
     * priority, earlier declaration on a tie (core/plugin/select.cpp). A constraint on a derived
     * parameter the geometry does not name is skipped, and a "device." key never holds, as
     * without an engine nothing measured the machine. Tunables are not chosen; the kernel reads its
     * defaults. Null when no host row admits the geometry. */
    const RadKernelInfo* select(const char* op, const RadParam* p, int n_p) const {
        const RadOpSchema* sch = schema(op);
        const RadKernelInfo* best = nullptr;
        for (int i = 0, n = kernel_count(); i < n; ++i) {
            const RadKernelInfo* k = kernel_at(i);
            if (!k || !k->op || std::strcmp(k->op, op) != 0 || k->domain != RAD_DOMAIN_HOST)
                continue;
            bool ok = true;
            for (int c = 0; c < k->n_constraints && ok; ++c)
                ok = constraint_holds(k->constraints[c], sch, p, n_p);
            if (ok && (!best || k->priority > best->priority)) best = k;
        }
        return best;
    }

    static bool constraint_holds(const RadConstraint& c, const RadOpSchema* sch, const RadParam* p,
                                 int n_p) {
        static const size_t kFact = std::strlen(RAD_DEVICE_FACT_PREFIX);
        if (!c.key || std::strncmp(c.key, RAD_DEVICE_FACT_PREFIX, kFact) == 0) return false;
        const RadParam* v = rad_param_find(p, n_p, c.key);
        if (!v) {
            for (int i = 0; sch && i < sch->n_params; ++i)
                if (sch->params[i].key && !std::strcmp(sch->params[i].key, c.key))
                    return sch->params[i].required == RAD_DERIVED;
            return false;
        }
        char num[32];
        const char* sv = v->kind == RAD_P_STR ? v->sval : nullptr;
        if (c.op == RAD_C_IN) {
            if (!sv) {
                if (v->kind != RAD_P_INT) return false;
                std::snprintf(num, sizeof num, "%lld", v->ival);
                sv = num;
            }
            const size_t len = std::strlen(sv);
            for (const char* t = c.sval ? c.sval : ""; *t;) {
                while (*t == ' ') ++t;
                const char* e = t;
                while (*e && *e != ' ') ++e;
                if ((size_t)(e - t) == len && len && !std::strncmp(t, sv, len)) return true;
                t = e;
            }
            return false;
        }
        if (v->kind != RAD_P_INT) return false;
        switch (c.op) {
            case RAD_C_EQ:  return v->ival == c.ival;
            case RAD_C_LE:  return v->ival <= c.ival;
            case RAD_C_GE:  return v->ival >= c.ival;
            case RAD_C_DIV: return c.ival != 0 && v->ival % c.ival == 0;
            default:        return false;
        }
    }
};

/* ================================================================== guarded buffers
 *
 * Every buffer a kernel is handed has H_GUARD bytes of a known pattern on each side, checked after
 * the launch: a write past either end of an operand or of the scratch shows up as a changed guard
 * rather than as a corrupted neighbour that may never be read. */
enum { H_GUARD = 64 };

static inline uint8_t h_guard_byte(size_t i) { return (uint8_t)(0xA5u ^ (uint8_t)(i * 29u)); }

struct GBuf {
    std::vector<uint8_t> v;
    size_t n = 0;

    void assign(const uint8_t* src, size_t bytes) {
        n = bytes;
        v.resize(bytes + 2 * H_GUARD);
        for (size_t i = 0; i < H_GUARD; ++i) {
            v[i] = h_guard_byte(i);
            v[H_GUARD + bytes + i] = h_guard_byte(i + 1);
        }
        if (bytes) {
            if (src) std::memcpy(v.data() + H_GUARD, src, bytes);
            else     std::memset(v.data() + H_GUARD, 0, bytes);
        }
    }
    void assign(const std::vector<uint8_t>& src) { assign(src.data(), src.size()); }
    uint8_t* data() { return v.data() + H_GUARD; }
    const uint8_t* data() const { return v.data() + H_GUARD; }
    size_t size() const { return n; }
    bool empty() const { return n == 0; }
    /* 0 when both guards are intact, otherwise -1 for the leading guard and 1 for the trailing. */
    int guard_hit() const {
        if (v.empty()) return 0;
        for (size_t i = 0; i < H_GUARD; ++i)
            if (v[i] != h_guard_byte(i)) return -1;
        for (size_t i = 0; i < H_GUARD; ++i)
            if (v[H_GUARD + n + i] != h_guard_byte(i + 1)) return 1;
        return 0;
    }
};

/* ================================================================== the draw
 *
 * Values follow the fill laws in RadOpdDesc (rad_abi.h). A gated-delta-net `g` is an intra-chunk
 * cumulative sum of log-decays; drawn as normals, the oracle overflows to 2e38 and the comparison
 * tests the input rather than the kernel. The same holds for sigmoid gates, unit-triangular
 * factors, unit-norm keys and every index operand (a KV slot indexes n_blocks * block_size, a
 * block-table entry n_blocks); an index drawn out of range tests only out-of-range handling.
 */
static inline uint64_t hsplitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static inline void fill_normal(float* v, int64_t n, uint64_t seed, float sigma) {
    uint64_t s = seed ? seed : 1;
    for (int64_t i = 0; i < n; i += 2) {
        /* Box-Muller off two splitmix draws. u1 is nudged off zero so the log is finite. */
        const double u1 = ((hsplitmix(s) >> 40) + 1) * (1.0 / 16777216.0);
        const double u2 = ((hsplitmix(s) >> 40)) * (1.0 / 16777216.0);
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double th = 6.283185307179586 * u2;
        v[i] = (float)(r * std::cos(th)) * sigma;
        if (i + 1 < n) v[i + 1] = (float)(r * std::sin(th)) * sigma;
    }
}

struct Opd {
    RadOpdDesc d{};
    bool absent = false;
    int  role = RAD_OPD_IN;
    int64_t n = 0;
    std::vector<uint8_t> master;         /* the drawn bytes */
    GBuf a, b;                           /* one guarded copy per side */
};

static inline int64_t desc_numel(const RadOpdDesc& d) {
    int64_t n = 1;
    for (uint32_t i = 0; i < d.rank; ++i) n *= d.shape[i];
    return n;
}

static inline size_t dt_bytes(uint32_t dt, int64_t n) {
    return (size_t)((n * rad_dtype_bits(dt) + 7) / 8);
}

/* W4, W2 and MXFP4 have no store in the ABI's accessor, so the codes are written directly. Both
 * plugins decode them with libref's codes (libref/ref_common.h), so the bytes mean the same to
 * both sides, which is all the comparison needs. */
static inline void h_narrow(const float* src, uint32_t dt, int64_t n, void* dst) {
    if (dt == RAD_I4 || dt == RAD_FP4E2M1) {
        uint8_t* b = (uint8_t*)dst;
        std::memset(b, 0, dt_bytes(dt, n));
        for (int64_t i = 0; i < n; ++i) {
            int c = (int)std::lrintf(src[i] * 3.0f);
            c = c < -8 ? -8 : (c > 7 ? 7 : c);
            b[i >> 1] |= (uint8_t)(((uint8_t)((c + 8) & 0xf)) << (4 * (i & 1)));
        }
        return;
    }
    if (dt == RAD_E8M0) {
        /* An MX block's exponent byte: the power of two nearest the drawn magnitude, so a scale
         * drawn at weight size lands near 2^-6, not at the store's 0 (2^-127). */
        uint8_t* b = (uint8_t*)dst;
        for (int64_t i = 0; i < n; ++i) {
            const float a = std::fabs(src[i]);
            const long e = a > 0.0f && std::isfinite(a) ? 127 + std::lrint(std::log2(a)) : 127;
            b[i] = (uint8_t)(e < 1 ? 1 : (e > 254 ? 254 : e));
        }
        return;
    }
    if (dt == RAD_I2) {
        uint8_t* b = (uint8_t*)dst;
        std::memset(b, 0, dt_bytes(dt, n));
        for (int64_t i = 0; i < n; ++i) {
            int c = (int)std::lrintf(src[i]);
            c = c < -2 ? -2 : (c > 1 ? 1 : c);
            b[i >> 2] |= (uint8_t)(((uint8_t)((c + 2) & 0x3)) << (2 * (i & 3)));
        }
        return;
    }
    for (int64_t i = 0; i < n; ++i) rad_store_f32(dst, dt, i, src[i]);
}

/* Packed codes decode with libref's codes, through avx_common.h's ld_elem. */
static inline void h_widen(const void* src, uint32_t dt, int64_t n, float* dst) {
    for (int64_t i = 0; i < n; ++i) dst[i] = avx::ld_elem(src, dt, i);
}

static inline void h_draw(std::vector<Opd>& o, size_t k, int64_t M, uint64_t seed) {
    Opd& d = o[k];
    const int64_t n = d.n;
    d.master.assign(dt_bytes(d.d.dtype, n > 0 ? n : 0), 0);
    if (n <= 0 || d.absent) return;
    /* A weight table (a MoE layer's experts) is described stacked, [n_expert, ...], and drawn as
     * any weight; left undrawn, every expert would be zeros and the product would compare zeros. */
    const bool weight = d.role == RAD_OPD_WEIGHT || d.role == RAD_OPD_WTAB;
    if (d.role != RAD_OPD_IN && d.role != RAD_OPD_INOUT && !weight) return;

    std::vector<float> tmp((size_t)n, 0.0f);
    const bool is_index = d.d.fill == RAD_FILL_INDEX;
    const bool idx_cu   = (d.d.flags & RAD_OPD_F_IDX_CU) != 0;
    const bool idx_uniq = (d.d.flags & RAD_OPD_F_IDX_UNIQUE) != 0;
    const bool idx_pre  = (d.d.flags & RAD_OPD_F_IDX_PREFIX) != 0;

    /* Integer and packed codes are whole numbers, so a draw at weight scale would store every
     * code as 0 and the product would compare zeros with zeros. They are drawn to span their grid
     * instead: int8 at sigma 40, the 4- and 2-bit codes at unit sigma (h_narrow scales a 4-bit
     * draw by 3). */
    const bool codes = d.d.dtype == RAD_I8 || d.d.dtype == RAD_I4 || d.d.dtype == RAD_I2 ||
                       d.d.dtype == RAD_FP4E2M1;
    if (codes && !is_index) {
        fill_normal(tmp.data(), n, seed, d.d.dtype == RAD_I8 ? 40.0f : 1.0f);
    } else if (weight && !is_index) {
        /* Sigma 0.02, the size of real checkpoint weights. At unit variance an fp8 quantiser's
         * amax would be orders of magnitude away from anything a model produces. */
        fill_normal(tmp.data(), n, seed, 0.02f);
    } else if (is_index && idx_cu) {
        /* Cumulative sequence lengths over [0, T]: the batch's shape, not random. */
        const int64_t S = n > 1 ? n - 1 : 1;
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)((i * M) / S);
    } else if (is_index && d.d.idx_const >= 0) {
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)d.d.idx_const;
    } else if (is_index && idx_pre) {
        /* A routing histogram's offsets (moe_scatter's `expert_offset`): non-decreasing from 0 to
         * idx_max - 1. Drawn at random they are a malformed histogram, which a binary and a linear
         * search read differently with neither wrong. Sorted draws, pinned at both ends. */
        uint64_t s2 = seed;
        const int64_t top = d.d.idx_max > 0 ? d.d.idx_max - 1 : 0;
        for (int64_t i = 0; i < n; ++i)
            tmp[(size_t)i] = (float)(hsplitmix(s2) % (uint64_t)(top + 1));
        std::sort(tmp.begin(), tmp.end());
        tmp[0] = 0.0f;
        tmp[(size_t)(n - 1)] = (float)top;
    } else if (is_index && idx_uniq) {
        /* Distinct scatter destinations: two writes to one slot race, and the two plugins would
         * resolve the race differently. */
        uint64_t s2 = seed;
        const int64_t range = d.d.idx_max > 0 ? d.d.idx_max : n;
        std::vector<int32_t> pool((size_t)range);
        for (int64_t i = 0; i < range; ++i) pool[(size_t)i] = (int32_t)i;
        for (int64_t i = 0; i < n && i < range; ++i) {
            const int64_t j = i + (int64_t)(hsplitmix(s2) % (uint64_t)(range - i));
            std::swap(pool[(size_t)i], pool[(size_t)j]);
            tmp[(size_t)i] = (float)pool[(size_t)i];
        }
        for (int64_t i = range; i < n; ++i) tmp[(size_t)i] = -1.0f;
    } else if (is_index) {
        uint64_t s2 = seed;
        int64_t rows = 32000;
        if (k > 0 && o[k - 1].d.rank > 0) rows = o[k - 1].d.shape[0];
        if (d.d.idx_max > 0) rows = d.d.idx_max;
        if (rows <= 0) rows = 1;
        for (int64_t i = 0; i < n; ++i) tmp[(size_t)i] = (float)(hsplitmix(s2) % (uint64_t)rows);
    } else {
        fill_normal(tmp.data(), n, seed, 1.0f);
        if (d.d.fill == RAD_FILL_SIGMOID) {
            for (int64_t i = 0; i < n; ++i)
                tmp[(size_t)i] = 1.0f / (1.0f + sc_exp(-tmp[(size_t)i]));
        } else if (d.d.fill == RAD_FILL_TRI_LOWER) {
            /* Unit diagonal, small off-diagonal. `A` is the chunked delta rule's intra-chunk
             * matrix I + tril(diag(beta) K K^T, -1), which is unit lower triangular; that is what
             * keeps its inverse bounded. A random diagonal would make the comparison measure the
             * input's condition number. */
            const int64_t ch = d.d.fill_chunk > 0 ? d.d.fill_chunk : 1;
            const int64_t heads = d.d.rank >= 3 ? d.d.shape[1] : 1;
            const int64_t rows = (heads * ch) != 0 ? n / (heads * ch) : n;
            for (int64_t t = 0; t < rows; ++t)
                for (int64_t h = 0; h < heads; ++h)
                    for (int64_t j = 0; j < ch; ++j) {
                        const size_t i = (size_t)((t * heads + h) * ch + j);
                        const int64_t diag = t % ch;
                        if (j > diag)       tmp[i] = 0.0f;
                        else if (j == diag) tmp[i] = 1.0f;
                        else                tmp[i] *= 0.25f;
                    }
        } else if (d.d.fill == RAD_FILL_GATE_CUMSUM) {
            const int64_t heads = d.d.rank >= 2 ? d.d.shape[d.d.rank - 1] : 1;
            const int64_t rows = heads > 0 ? n / heads : n;
            const int64_t ch = d.d.fill_chunk > 0 ? d.d.fill_chunk : rows;
            for (int64_t h = 0; h < heads; ++h) {
                float acc = 0.0f;
                for (int64_t t = 0; t < rows; ++t) {
                    if (t % ch == 0) acc = 0.0f;
                    acc -= sc_log1p(sc_exp(tmp[(size_t)(t * heads + h)]));
                    tmp[(size_t)(t * heads + h)] = acc;
                }
            }
        } else if (d.d.fill == RAD_FILL_UNIT_ROW) {
            const int64_t w = d.d.rank ? d.d.shape[d.d.rank - 1] : n;
            if (w > 0)
                for (int64_t r = 0; r + w <= n; r += w) {
                    double ss = 0;
                    for (int64_t i = 0; i < w; ++i)
                        ss += (double)tmp[(size_t)(r + i)] * (double)tmp[(size_t)(r + i)];
                    const float inv = ss > 0 ? (float)(1.0 / std::sqrt(ss)) : 0.0f;
                    for (int64_t i = 0; i < w; ++i) tmp[(size_t)(r + i)] *= inv;
                }
        }
    }
    h_narrow(tmp.data(), d.d.dtype, n, d.master.data());
}

/* ================================================================== the geometry table
 *
 * The only op-specific knowledge in these tools, given as parameters; shapes come from libref's
 * `opd_shape`. Values are small but not degenerate: rows wider than a vector, more than one row, a
 * dividing group and, where allowed, a row length that is not a multiple of the vector width, to
 * exercise the tail.
 */
struct P { const char* k; int kind; long long i; const char* s; double d; };

static inline RadParam h_mk(const P& p) {
    RadParam r{};
    r.key = p.k; r.kind = p.kind; r.ival = p.i; r.sval = p.s; r.dval = p.d;
    return r;
}
#define HI(k, v) P{ k, RAD_P_INT, (long long)(v), nullptr, 0.0 }
#define HS(k, v) P{ k, RAD_P_STR, 0, v, 0.0 }
#define HF(k, v) P{ k, RAD_P_F64, 0, nullptr, (double)(v) }

struct Case {
    const char* op;
    std::vector<P> p;
    /* Bytes moved and flops done, for the bench's rate columns. 0 means no rate is reported (the
     * cost is not a simple function of the geometry). */
    double bytes = 0, flops = 0;
    /* Timed at a real model shape but not checked there; rad-avx-check skips it. A 640-wide row
     * exercises every code path a 2048-wide one does, but speed depends on the cache hierarchy
     * and needs the real shape. Same split as tools/kfixture.h. */
    bool speed_only = false;
};

/* Total token count of the batch, which the cumulative-length draw divides across sequences and
 * which no operand carries.
 *
 * It is not always the ranged parameter. An op banded on `M` is banded on the total
 * (`gdn_conv_prep`: activation [M, conv_dim], cu spans M). An op banded on `q_len` is banded on
 * the per-sequence length (`gdn_conv_update`: activation [n_seq * q_len, conv_dim], q_len candidate
 * rows per sequence), so the total is q_len * n_seq. With cu summing to q_len instead, libref
 * derives a doubled head count from numel(q) / (T * head_k) and refuses the case with RAD_E_SHAPE.
 * `attn_paged` is also banded on `q_len`; libref describes it at one sequence. */
static inline int64_t case_M(const Case& c) {
    int64_t n_seq = 1;
    for (const P& p : c.p) if (std::strcmp(p.k, "n_seq") == 0 && p.i > 0) n_seq = p.i;
    for (const P& p : c.p) if (std::strcmp(p.k, "M") == 0) return p.i;
    for (const P& p : c.p) if (std::strcmp(p.k, "q_len") == 0) return p.i * n_seq;
    for (const P& p : c.p) if (std::strcmp(p.k, "numel") == 0) return p.i;
    return 1;
}

static inline const char* case_dtype(const Case& c) {
    static const char* keys[] = { "dtype", "q_dtype", "kv_dtype", "to" };
    for (const char* k : keys)
        for (const P& p : c.p)
            if (std::strcmp(p.k, k) == 0 && p.kind == RAD_P_STR) return p.s;
    return "*";
}

/* A one-line rendering of a case, so a failure names a geometry a reader can reproduce. */
static inline std::string case_str(const Case& c) {
    std::string s;
    for (const P& p : c.p) {
        char buf[96];
        if (p.kind == RAD_P_STR)      std::snprintf(buf, sizeof buf, "%s=%s ", p.k, p.s ? p.s : "");
        else if (p.kind == RAD_P_F64) std::snprintf(buf, sizeof buf, "%s=%g ", p.k, p.d);
        else                          std::snprintf(buf, sizeof buf, "%s=%lld ", p.k, p.i);
        s += buf;
    }
    if (!s.empty()) s.pop_back();
    return s;
}

/* ================================================================== a prepared case
 * Operands sized by the oracle's description, drawn once and copied into one buffer per side, so
 * both plugins see byte-identical inputs. */
struct Prepared {
    bool ok = false;
    bool skipped = false;              /* the oracle declined this geometry */
    std::vector<RadParam> params;
    std::vector<Opd> opd;
    const RadOpSchema* schema = nullptr;
    std::vector<RadTensor> ta, tb;
    RadArgs aa{}, ab{};
    GBuf sa, sb;                       /* each side's scratch, when its kernel asks for one */
};

static inline void h_build_tensors(std::vector<Opd>& opd, bool side_a,
                                   std::vector<RadTensor>* out) {
    out->assign(opd.size(), RadTensor{});
    for (size_t i = 0; i < opd.size(); ++i) {
        RadTensor& x = (*out)[i];
        if (opd[i].absent) { x.data = nullptr; continue; }
        x.data = (side_a ? opd[i].a : opd[i].b).data();
        x.dtype = opd[i].d.dtype;
        x.rank = opd[i].d.rank;
        int64_t acc = 1;
        for (int k = (int)x.rank - 1; k >= 0; --k) {
            x.shape[k] = opd[i].d.shape[k];
            x.stride[k] = acc;
            acc *= x.shape[k];
        }
    }
}

/* The samplers' per-row `params` are a RadSampleParams struct (abi/rad_sample.h). libref describes
 * them with RAD_FILL_SAMPLE_PARAMS and the history with RAD_OPD_F_IDX_HISTORY, laws h_draw does not
 * implement: it draws such a plane as normals, whose integer fields and `flags` are bit patterns
 * no stage acts on, and such a history as uniform ids, which never repeat. Each row is written as
 * a request instead, rows differing in temperature, top-p, min-p and typical-p, with every stage
 * switched on except on the last row, which is greedy. The history's second half repeats its
 * first, so DRY finds repeats to penalise. rad-kbench draws the laws themselves. */
static inline void h_sampler_rows(Prepared& pr, const Case& c) {
    if (std::strncmp(c.op, "sample_", 7) != 0) return;
    Opd* params = nullptr;
    Opd* hist = nullptr;
    Opd* ngram = nullptr;
    for (size_t k = 0; k < pr.opd.size(); ++k) {
        const char* nm = pr.schema->operands[k].name;
        if (!nm || pr.opd[k].absent || pr.opd[k].d.rank != 2) continue;
        if (!std::strcmp(nm, "params")) params = &pr.opd[k];
        if (!std::strcmp(nm, "history") && pr.opd[k].d.dtype == RAD_I32) hist = &pr.opd[k];
        if (!std::strcmp(nm, "ngram") && pr.opd[k].d.shape[1] == 2) ngram = &pr.opd[k];
    }
    const int64_t words = (int64_t)(sizeof(RadSampleParams) / 4);
    if (!params || rad_dtype_bits(params->d.dtype) != 32 || params->d.shape[1] != words) return;
    const int64_t M = params->d.shape[0];
    const int64_t hw = hist ? hist->d.shape[1] : 0;
    for (int64_t r = 0; r < M; ++r) {
        RadSampleParams sp{};
        sp.seed = 0x5eed0000u + (uint64_t)r;
        sp.pos = 7 * (uint64_t)r;
        sp.temp = 0.6f + 0.2f * (float)(r % 4);
        sp.top_p = 0.9f - 0.1f * (float)(r % 3);
        sp.min_p = 0.02f + 0.03f * (float)(r % 3);
        sp.typical_p = 0.95f - 0.1f * (float)(r % 3);
        sp.rep_penalty = 1.1f;
        sp.freq_penalty = 0.2f;
        sp.pres_penalty = 0.3f;
        sp.xtc_probability = r % 2 ? 0.5f : 1.0f;
        sp.xtc_threshold = 0.05f;
        sp.dry_multiplier = 0.8f;
        sp.dry_base = 1.75f;
        sp.top_k = r % 2 ? 0 : 8;
        sp.penalty_last_n = r % 3 ? (int32_t)hw : -1;
        sp.dry_allowed_length = 2;
        sp.dry_penalty_last_n = -1;
        sp.hist_off = 0;
        sp.hist_len = (int32_t)hw;
        sp.mask_row = r % 2 ? (int32_t)r : -1;
        /* The n-gram ban's per-row pair: n 1 to 3 over windows short of the 64-token history, so
         * the window boundary and the n == 1 form (an empty prefix) both run. Only the op whose
         * schema has the `ngram` plane reads it. */
        if (ngram) {
            int32_t* v = (int32_t*)ngram->master.data();
            v[2 * r]     = 1 + (int32_t)(r % 3);
            v[2 * r + 1] = 12 + 8 * (int32_t)(r % 4);
        }
        sp.flags = RAD_SP_PENALTY | RAD_SP_XTC | RAD_SP_DRY | RAD_SP_LOGPROBS |
                   (sp.mask_row >= 0 ? RAD_SP_MASK : 0);
        if (r == M - 1 && M > 1) sp.flags = RAD_SP_GREEDY;
        std::memcpy(params->master.data() + (size_t)r * sizeof sp, &sp, sizeof sp);
    }
    if (hist)
        for (int64_t r = 0; r < hist->d.shape[0]; ++r) {
            int32_t* row = (int32_t*)hist->master.data() + r * hw;
            for (int64_t j = hw / 2; j < hw; ++j) row[j] = row[j - hw / 2];
        }
}

/* `describe` is the kernel whose opd_shape sizes the operands. A correctness check passes the
 * oracle's, since the reference semantics define the extents; a benchmark of a kernel with no
 * oracle passes its own (as tools/opshapes.h does for rad-kbench). */
static inline Prepared h_prepare(const Lib& oracle, const RadKernelInfo* describe, const Case& c,
                                 const RadKernelInfo* ka, const RadKernelInfo* kb) {
    Prepared pr;
    pr.schema = oracle.schema(c.op);
    if (!pr.schema || !describe || !describe->opd_shape) { pr.skipped = true; return pr; }

    pr.params.reserve(c.p.size());
    for (const P& p : c.p) pr.params.push_back(h_mk(p));

    pr.opd.assign((size_t)pr.schema->n_operands, Opd{});
    for (int i = 0; i < pr.schema->n_operands; ++i) {
        RadOpdDesc d{};
        d.idx_const = -1;      /* the ABI's "draw"; a hook that leaves it gets the default */
        if (describe->opd_shape(pr.params.data(), (int)pr.params.size(), i, &d) < 0) {
            pr.skipped = true;
            return pr;
        }
        Opd& o = pr.opd[(size_t)i];
        o.d = d;
        o.absent = (d.flags & RAD_OPD_F_ABSENT) != 0;
        o.role = pr.schema->operands[i].role;
        o.n = o.absent ? 0 : desc_numel(d);
        /* A rank the ABI cannot express is a hook bug; it must not become a zero-extent buffer
         * that every comparison passes vacuously. */
        if (!o.absent && (d.rank == 0 || d.rank > RAD_MAX_RANK)) { pr.skipped = true; return pr; }
    }

    uint64_t seed = 0x9E3779B9ull;
    for (const P& p : c.p) {
        seed = seed * 1000003ull + (uint64_t)p.i;
        if (p.s) for (const char* s = p.s; *s; ++s) seed = seed * 131ull + (unsigned char)*s;
    }
    for (size_t k = 0; k < pr.opd.size(); ++k) {
        uint64_t s = seed ^ (0x100000001b3ull * (k + 1));
        h_draw(pr.opd, k, case_M(c), s ? s : 1);
    }
    h_sampler_rows(pr, c);
    for (Opd& o : pr.opd) {
        o.a.assign(o.master);
        o.b.assign(o.master);
    }

    h_build_tensors(pr.opd, true, &pr.ta);
    h_build_tensors(pr.opd, false, &pr.tb);

    pr.aa.t = pr.ta.data(); pr.aa.n_t = (int)pr.ta.size();
    pr.aa.p = pr.params.data(); pr.aa.n_p = (int)pr.params.size();
    pr.aa.rank = 0; pr.aa.world_size = 1;
    pr.ab = pr.aa;
    pr.ab.t = pr.tb.data();

    /* Honour the scratch hook; a kernel that declares one and gets null would fault. */
    if (ka && ka->scratch) {
        const int64_t n = ka->scratch(&pr.aa);
        if (n > 0) {
            pr.sa.assign(nullptr, (size_t)n);
            pr.aa.scratch = pr.sa.data();
            pr.aa.scratch_bytes = n;
        }
    }
    if (kb && kb->scratch) {
        const int64_t n = kb->scratch(&pr.ab);
        if (n > 0) {
            pr.sb.assign(nullptr, (size_t)n);
            pr.ab.scratch = pr.sb.data();
            pr.ab.scratch_bytes = n;
        }
    }
    pr.ok = true;
    return pr;
}

/* Restore both sides to the drawn bytes. A loop that relaunches an op with an INOUT operand must
 * reset between iterations, or each iteration computes something different (twenty runs of
 * `rope` rotate the tensor twenty times over). */
static inline void h_reset(Prepared& pr) {
    for (Opd& o : pr.opd) {
        if (o.absent || o.n <= 0) continue;
        if (o.role == RAD_OPD_INOUT || o.role == RAD_OPD_OUT) {
            std::memcpy(o.a.data(), o.master.data(), o.master.size());
            std::memcpy(o.b.data(), o.master.data(), o.master.size());
        }
    }
}

std::vector<Case> build_cases();

#endif /* RAD_AVX_HARNESS_H */
