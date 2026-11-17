/* avx_common.h -- helpers every kernel in this plugin needs that the ABI does not provide.
 *
 * rad_plugin.h already has the parameter readers, the absent-operand rule, the row-pitch helper
 * and the narrowing conversions. This header adds three things:
 *
 *   Scratch arena. Every kernel converts rows to f32 before computing (avx_vec.h), so it needs row
 *   scratch. RadArgs::scratch does not fit: it is one buffer sized at declare, while the outer loop
 *   runs under OpenMP with a thread count unknown at declare. A thread_local arena grows to a
 *   high-water mark and is reused, so the hot path does not allocate (libref does the same).
 *
 *   Operand addressing. Every extent comes from a RadTensor, never from a parameter: a ranged
 *   parameter is collapsed to a band, so a loop bounded by `M` would run to the band's upper bound
 *   and walk off the arena (libref/ref_ops.h).
 *
 *   Threading. OpenMP over the outer dimension only, one thread owning whole output rows, no
 *   reduction() clause. Results are bit-identical across thread counts, so two runs of the same
 *   binary agree regardless of machine load and measurements stay comparable.
 */
#ifndef RAD_AVX_COMMON_H
#define RAD_AVX_COMMON_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>

/* Scalar transcendentals as libref evaluates them: in double, rounded once to float. glibc's float
 * routines differ between releases (2.41 made log1pf, tanhf and erff correctly rounded), so a float
 * call would agree with libref on one host and not on another; the double routines are stable,
 * and the rounded float is the same everywhere. The vector paths (avx_vec.h) are checked to a
 * tolerance and do not use these. */
static inline float sc_exp(float x)   { return (float)std::exp((double)x); }
static inline float sc_log(float x)   { return (float)std::log((double)x); }
static inline float sc_log1p(float x) { return (float)std::log1p((double)x); }
static inline float sc_tanh(float x)  { return (float)std::tanh((double)x); }
static inline float sc_erf(float x)   { return (float)std::erf((double)x); }
static inline float sc_sin(float x)   { return (float)std::sin((double)x); }
static inline float sc_cos(float x)   { return (float)std::cos((double)x); }
static inline float sc_pow(float b, float e) { return (float)std::pow((double)b, (double)e); }

#ifdef _OPENMP
#include <omp.h>
#include <sched.h>
#include <atomic>
#include <cstdio>
/* The checker's override, set through avx_par_force (avx_isa.h): 0, or the team size every
 * parallel region takes whatever its work threshold. Defined in avx_dispatch.cpp; one relaxed load
 * per region. */
extern std::atomic<int> avx_par_threads;
static inline bool avx_par_forced() {
    return avx_par_threads.load(std::memory_order_relaxed) > 0;
}
/* One team thread per physical core, not per logical CPU. These kernels are FMA- or
 * bandwidth-bound, so an SMT sibling adds little, and the engine runs its own threads (stream
 * workers, scheduler, HTTP server) beside the team. A team as wide as the logical CPU count makes
 * team barriers wait on descheduled threads: on a 16-core, 32-thread part a 576-token encoder
 * step takes 0.75 s at 32 threads and 0.09 s at 16. Cores are counted over the process affinity
 * mask, so a pinned or containerised engine uses what it was given. OMP_NUM_THREADS, when set,
 * wins, and a thread that asked for fewer (omp_set_num_threads) gets fewer. */
static inline int avx_team_size() {
    if (const int forced = avx_par_threads.load(std::memory_order_relaxed); forced > 0)
        return forced;
    /* The core count is per process and computed once; the OpenMP request is per calling thread
     * and read on every call (the benchmark lowers it for single-thread figures). */
    static const int cores = [] {
        if (const char* e = std::getenv("OMP_NUM_THREADS"); e && *e) return 0;
        cpu_set_t set;
        if (sched_getaffinity(0, sizeof set, &set) != 0) return 0;
        int n = 0;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &set)) continue;
            /* A CPU counts when it is the first of its core's siblings that is in the mask. */
            char path[96];
            std::snprintf(path, sizeof path,
                          "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
            int first = c;
            if (FILE* f = std::fopen(path, "r")) {
                int v = -1;
                if (std::fscanf(f, "%d", &v) == 1 && v >= 0 && v < CPU_SETSIZE && CPU_ISSET(v, &set))
                    first = v;
                std::fclose(f);
            }
            n += first == c;
        }
        return n;
    }();
    const int asked = omp_get_max_threads();
    return cores > 0 && cores < asked ? cores : asked;
}
#define AVX_PRAGMA(x) _Pragma(#x)
#define AVX_PARALLEL_FOR AVX_PRAGMA(omp parallel for schedule(static) num_threads(avx_team_size()))
/* Parallelise the outer loop only when the work pays for a fork/join, which costs a few
 * microseconds; `rad-avx-bench --threads` measures the crossover. `cond` is pasted into the
 * pragma text, so it must contain no comma. avx_par_force overrides it, so the checker runs every
 * parallel body with several threads at its small geometries. */
#define AVX_PARALLEL_FOR_IF(cond) \
    AVX_PRAGMA(omp parallel for schedule(static) if((cond) || avx_par_forced()) \
               num_threads(avx_team_size()))
#else
#define AVX_PARALLEL_FOR
#define AVX_PARALLEL_FOR_IF(cond)
#endif

/* Elements of output a launch must touch before a parallel region pays for itself. Used as
 * AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN). */
#define AVX_PAR_MIN 16384

namespace avx {

/* ================================================================== scratch
 * Aligned to 64 so a 512-bit aligned load is legal off it at every level. Returns a buffer of at
 * least `n` floats, valid until the next call on this thread that asks for more. */
float*   scratch_f32(size_t n);
/* Independent second and third buffers, for kernels with several rows live at once (gate and up,
 * a and b, x and residual). */
float*   scratch_f32_b(size_t n);
float*   scratch_f32_c(size_t n);
void*    scratch_raw(size_t bytes);

/* ================================================================== libref's addressing, exactly
 *
 * Copied from libref/ref_common.h and must stay identical. They are the addressing contract:
 * where element (r, i) lives, how a rank-3 tensor reads as [rows, cols], and what a strided view
 * means. Diverging here fails the oracle check in a way that looks like a numerical bug.
 *
 * `numel` returns 0 for a null-data tensor; libref's shape checks rely on that to refuse an absent
 * required operand. */
static inline int64_t numel(const RadTensor* t) {
    if (!t || !t->data || t->rank == 0) return 0;
    int64_t n = 1;
    for (uint32_t i = 0; i < t->rank; ++i) n *= t->shape[i];
    return n;
}

static inline int64_t laststride(const RadTensor* t) { return t->rank ? t->stride[t->rank - 1] : 1; }

/* Element offset of the `lin`-th element in row-major logical order. */
static inline int64_t offlin(const RadTensor* t, int64_t lin) {
    if (t->rank == 0) return lin;
    int64_t acc = 1;
    bool c = true;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) { c = false; break; }
        acc *= t->shape[i];
    }
    if (c) return lin;
    int64_t off = 0;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        int64_t s = t->shape[i];
        if (s <= 0) return 0;
        off += (lin % s) * t->stride[i];
        lin /= s;
    }
    return off;
}

/* Element offset of logical row `r` read as [rows, cols]. Leading axes use their own strides, so
 * a [T, H, V] tensor read as [T*H, V] is correct even when H's stride is not V (the
 * fused-projection column slice in docs/OPS.md). */
static inline int64_t rowoff(const RadTensor* t, int64_t r, int64_t cols) {
    if (t->rank >= 2 && t->shape[t->rank - 1] == cols) {
        int64_t off = 0;
        for (int i = (int)t->rank - 2; i >= 0; --i) {
            int64_t s = t->shape[i];
            if (s <= 0) return 0;
            off += (r % s) * t->stride[i];
            r /= s;
        }
        return off;
    }
    return r * cols;
}

static inline int64_t rows_of(const RadTensor* t, int64_t n) { return n > 0 ? numel(t) / n : 0; }

/* The operand's row count, with the ranged parameter as a ceiling (libref/ref_common.h:
 * band_rows). `M`, `numel` and `q_len` arrive as the band's upper bound, so the count comes from
 * the operand; a count above the band is a caller bug and yields 0. */
static inline int64_t band_rows(int64_t from_operand, int64_t band) {
    if (from_operand <= 0) return 0;
    return (band > 0 && from_operand > band) ? 0 : from_operand;
}

/* Block size implied by a scale grid of `blocks` pieces over `ext` elements (libref/ref_gemm.h:
 * grid_block): the power of two whose ceiling division gives `blocks`, not ext / blocks, because
 * the last block may be partial. */
static inline int64_t grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

/* Stride between rows of `cols` elements, at any rank (libref/ref_gemm.h's rule). Rank 2 does not
 * imply dense: attention receives a column slice of the fused QKV plane, [tokens, kv_heads *
 * head_dim] with the full projection width as row stride, so the stride must not be derived from
 * the shape. */
static inline int64_t row_stride(const RadTensor* t, int64_t cols) {
    if (!t || t->rank < 2) return cols;
    int64_t span = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        span *= t->shape[i];
        if (span == cols) return (i > 0) ? t->stride[i - 1] : cols;
        if (span > cols) break;
    }
    return t->stride[t->rank - 2];
}

/* One element as f32 and back:
 *  - Packed codes. rad_load_f32 returns 0 for W4, W2 and MXFP4 (an unreadable dtype is a
 *    refusal), so strided or odd-offset reads of packed weights decode here, with libref's codes
 *    (ref_common.h: w4_code, w2_code, mxfp4_code), the same ones row_to_f32 unpacks in bulk.
 *  - Integer and packed-code stores to the dtypes listed in st_elem round to the nearest code and
 *    saturate, NaN storing 0, as libref's st_dt does; every other dtype (u16, bool, the floats)
 *    is rad_store_f32's. */
static inline float ld_elem(const void* p, uint32_t dt, int64_t i) {
    switch (dt) {
        case RAD_I4: {
            const uint8_t b = ((const uint8_t*)p)[i >> 1];
            const uint8_t nb = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf);
            return (float)((int)(nb ^ 0x8u) - 8);
        }
        case RAD_I2: {
            const uint8_t b = ((const uint8_t*)p)[i >> 2];
            const uint8_t c = (uint8_t)((b >> (2 * (i & 3))) & 0x3u);
            return (float)((int)(c ^ 0x2u) - 2);
        }
        case RAD_FP4E2M1: {
            static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
            const uint8_t b = ((const uint8_t*)p)[i >> 1];
            const uint8_t nb = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf);
            return (nb & 8u) ? -kMag[nb & 7u] : kMag[nb & 7u];
        }
        default: return rad_load_f32(p, dt, i);
    }
}

/* Round to nearest-even and clamp to [lo, hi], in double so that every bound of every integer
 * width up to 32 bits is exact. NaN is 0. */
static inline double round_sat(float v, double lo, double hi) {
    if (std::isnan(v)) return 0.0;
    const double r = (double)std::nearbyintf(v);
    return r < lo ? lo : (r > hi ? hi : r);
}

/* A packed code's low `bits` into element `i` of a bit stream, lowest bits first. The atomics
 * touch only this element's bits, so rows written by different threads may share a byte. */
static inline void st_packed(void* p, int64_t i, int bits, unsigned c) {
    const int per = 8 / bits;
    uint8_t* b = (uint8_t*)p + i / per;
    const int sh = (int)(i % per) * bits;
    const uint8_t mask = (uint8_t)(((1u << bits) - 1u) << sh);
    __atomic_fetch_and(b, (uint8_t)~mask, __ATOMIC_RELAXED);
    __atomic_fetch_or(b, (uint8_t)((c << sh) & mask), __ATOMIC_RELAXED);
}

/* The e2m1 code nearest `v`, ties to the even code, saturating at 6; NaN is +0 (libref's). */
static inline unsigned e2m1_code(float v) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    if (v != v) return 0u;
    const unsigned sign = std::signbit(v) ? 8u : 0u;
    const float a = std::fabs(v);
    unsigned best = 0;
    for (unsigned c = 1; c < 8; ++c) {
        const float d = std::fabs(kMag[c] - a), db = std::fabs(kMag[best] - a);
        if (d < db || (d == db && (c & 1u) == 0u)) best = c;
    }
    return sign | best;
}

static inline void st_elem(void* p, uint32_t dt, int64_t i, float v) {
    switch (dt) {
        /* Packed codes: the nearest code, saturated, so a code ld_elem read stores back as itself
         * (a gather or scatter of packed rows moves them exactly). */
        case RAD_I4: st_packed(p, i, 4, (unsigned)(int)round_sat(v, -8.0, 7.0) & 0xFu); break;
        case RAD_I2: st_packed(p, i, 2, (unsigned)(int)round_sat(v, -2.0, 1.0) & 0x3u); break;
        case RAD_FP4E2M1: st_packed(p, i, 4, e2m1_code(v)); break;
        case RAD_I8:  ((int8_t*)p)[i]   = (int8_t)round_sat(v, -128.0, 127.0); break;
        case RAD_U8:  ((uint8_t*)p)[i]  = (uint8_t)round_sat(v, 0.0, 255.0); break;
        case RAD_I16: ((int16_t*)p)[i]  = (int16_t)round_sat(v, -32768.0, 32767.0); break;
        case RAD_I32: ((int32_t*)p)[i]  = (int32_t)round_sat(v, -2147483648.0, 2147483647.0); break;
        case RAD_U32: ((uint32_t*)p)[i] = (uint32_t)round_sat(v, 0.0, 4294967295.0); break;
        case RAD_I64: {
            /* 2^63 is the first float past the int64 range, so it is tested by comparison
             * rather than used as a clamp value. */
            const double r = round_sat(v, -9223372036854775808.0, 9223372036854775808.0);
            ((int64_t*)p)[i] = r >= 9223372036854775808.0 ? INT64_MAX : (int64_t)r;
            break;
        }
        default: rad_store_f32(p, dt, i, v); break;
    }
}

static inline float ldt(const RadTensor* t, int64_t off) { return ld_elem(t->data, t->dtype, off); }
static inline void  stt(const RadTensor* t, int64_t off, float v) {
    st_elem(t->data, t->dtype, off, v);
}

/* `v` rounded through a store and load of dtype `dt` (libref's round_dt). */
static inline float narrow1(uint32_t dt, float v) {
    uint32_t buf[2] = { 0, 0 };
    rad_store_f32(buf, dt, 0, v);
    return rad_load_f32(buf, dt, 0);
}

/* True when the row's elements are contiguous. A dense row converts with one call and is read as
 * a flat f32 array; any other row goes through the strided gather. Nearly every operand is dense;
 * docs/OPS.md names the exceptions. */
static inline bool dense_row(const RadTensor* t) { return laststride(t) == 1; }

/* The gemm_nt_q dtypes whose activation is quant_act_i8's, codes in the int8 WMMA's step order
 * (libref/ref_common.h). */
static inline bool i8_steps_dtype(const char* dt) {
    static const char* const kDtypes[] = { "w4a8", "w4a8_asym", "w8a8", "w2a8" };
    for (const char* d : kDtypes)
        if (dt && std::strcmp(dt, d) == 0) return true;
    return false;
}

/* ================================================================== parameters
 * Laxer than rad_param_geti, as in libref: a tool or test may call a launch directly with no
 * declare to enforce the type, so a range or a string spelling of an integer is read rather than
 * refused. */
static inline const RadParam* p_find(const RadArgs* a, const char* k) {
    return a ? rad_param_find(a->p, a->n_p, k) : nullptr;
}

static inline long long p_int(const RadArgs* a, const char* k, long long dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_INT)   return p->ival;
    if (p->kind == RAD_P_RANGE) return p->ihi;
    if (p->kind == RAD_P_STR && p->sval) return std::strtoll(p->sval, nullptr, 10);
    return dflt;
}

static inline const char* p_str(const RadArgs* a, const char* k, const char* dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_STR && p->sval) return p->sval;
    return dflt;
}

static inline float p_f32(const RadArgs* a, const char* k, float dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_F64)   return (float)p->dval;
    if (p->kind == RAD_P_STR && p->sval) return (float)std::strtod(p->sval, nullptr);
    if (p->kind == RAD_P_INT)   return (float)p->ival;
    if (p->kind == RAD_P_RANGE) return (float)p->ihi;
    return dflt;
}

/* An enumerated parameter spelled as a name or libr4d's integer; an unknown name is -1, for the
 * caller to refuse (as libref). */
static inline int p_enum(const RadArgs* a, const char* k, const char* const* names, int n,
                         int dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_STR && p->sval) {
        for (int i = 0; i < n; ++i)
            if (names[i] && std::strcmp(names[i], p->sval) == 0) return i;
        return -1;
    }
    if (p->kind == RAD_P_INT) return (int)p->ival;
    return dflt;
}

static inline int64_t ld_int_dt(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32:  return (int64_t)((const int32_t*)p)[i];
        case RAD_U32:  return (int64_t)((const uint32_t*)p)[i];
        case RAD_I64:  return ((const int64_t*)p)[i];
        case RAD_I16:  return (int64_t)((const int16_t*)p)[i];
        case RAD_I8:   return (int64_t)((const int8_t*)p)[i];
        case RAD_U8:   return (int64_t)((const uint8_t*)p)[i];
        case RAD_BOOL: return ((const uint8_t*)p)[i] ? 1 : 0;
        case RAD_F32:  return (int64_t)((const float*)p)[i];
        default:       return 0;
    }
}

static inline void st_int_dt(uint32_t dt, void* p, int64_t i, int64_t v) {
    switch (dt) {
        case RAD_I32: ((int32_t*)p)[i] = (int32_t)v; break;
        case RAD_U32: ((uint32_t*)p)[i] = (uint32_t)v; break;
        case RAD_I64: ((int64_t*)p)[i] = v; break;
        case RAD_I16: ((int16_t*)p)[i] = (int16_t)v; break;
        case RAD_I8:  ((int8_t*)p)[i]  = (int8_t)v; break;
        case RAD_U8:  ((uint8_t*)p)[i] = (uint8_t)v; break;
        default:      rad_store_f32(p, dt, i, (float)v); break;
    }
}

static inline int64_t ldi(const RadTensor* t, int64_t off) { return ld_int_dt(t->dtype, t->data, off); }

/* Address of the byte holding element i, for dtypes whose element is not a whole byte. Packed
 * codes are otherwise read through row_to_f32; `dequant` uses this to address a group start. */
static inline const void* byte_at(const void* base, uint32_t dt, int64_t i) {
    const uint8_t* b = (const uint8_t*)base;
    switch (dt) {
        case RAD_I4:    case RAD_FP4E2M1: return b + (i >> 1);
        case RAD_I2:                    return b + (i >> 2);
        default:                        return b + i * (size_t)(rad_dtype_bits(dt) / 8);
    }
}

}  /* namespace avx */
#endif /* RAD_AVX_COMMON_H */
