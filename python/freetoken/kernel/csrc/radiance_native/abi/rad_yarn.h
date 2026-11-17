#ifndef RAD_YARN_H
#define RAD_YARN_H

/* Static YaRN coefficients. The architecture resolves the correction bounds from the original
 * training length, never the deployment or table length. Kernels share the resolved coefficients
 * so direct and table-backed RoPE cannot disagree.
 *
 * C++ gets RadYarn with a member frequency() (callable from HIP device code) and default member
 * values; C gets the same four-float struct and rad_yarn_frequency(). The two definitions must
 * compute the same frequencies; tests/abi_layout_test.cpp compares them. */
#include "rad_plugin.h"

#ifdef __cplusplus
#include <cmath>

#if defined(__HIPCC__)
#define RAD_YARN_HD __host__ __device__
#else
#define RAD_YARN_HD
#endif

struct RadYarn {
    float factor = 0.0f;   /* zero: YaRN off, frequency() returns `inv` unchanged */
    float low = 0.0f;
    float high = 0.0f;
    float magnitude = 1.0f;

    RAD_YARN_HD float frequency(float inv, int i) const {
        if (factor == 0.0f) return inv;
        const float width = high == low ? 0.001f : high - low;
        const float ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - low) / width));
        /* Keep the blend order used by the reference YaRN implementation. */
        const float mask = 1.0f - ramp;
        return (inv / factor) * (1.0f - mask) + inv * mask;
    }
};

inline RadYarn rad_yarn_args(const RadArgs* a) {
    const float factor = (float)rad_args_getf_or(a, "yarn_factor", 0.0);
    /* Disabled YaRN ignores every auxiliary coefficient on every backend. */
    if (factor == 0.0f) return {};
    return { factor,
             (float)rad_args_getf_or(a, "yarn_low", 0.0),
             (float)rad_args_getf_or(a, "yarn_high", 0.0),
             (float)rad_args_getf_or(a, "yarn_magnitude", 1.0) };
}

inline bool rad_yarn_valid(const RadYarn& y) {
    return std::isfinite(y.factor) && (y.factor == 0.0f ||
        (y.factor >= 1.0f && std::isfinite(y.low) && std::isfinite(y.high) &&
         y.high >= y.low && std::isfinite(y.magnitude) && y.magnitude > 0.0f));
}

/* Llama 3's frequency scaling (transformers' rope_type "llama3"), the other static scaling a
 * checkpoint may name: a frequency whose wavelength 2 pi / inv is shorter than orig / high is
 * kept, one longer than orig / low is divided by `factor`, and those between are blended,
 * s = (orig / wavelength - low) / (high - low), (1 - s) inv / factor + s inv. f32 throughout, as
 * transformers computes it. `factor` zero is off. */
struct RadLlama3 {
    float factor = 0.0f;
    float low = 0.0f;    /* low_freq_factor */
    float high = 0.0f;   /* high_freq_factor */
    float orig = 0.0f;   /* original_max_position_embeddings */

    RAD_YARN_HD float frequency(float inv) const {
        if (factor == 0.0f) return inv;
        const float wavelen = 6.28318530717958647692f / inv;
        if (wavelen < orig / high) return inv;
        if (wavelen > orig / low) return inv / factor;
        const float sm = (orig / wavelen - low) / (high - low);
        return (1.0f - sm) * inv / factor + sm * inv;
    }
};

inline RadLlama3 rad_llama3_args(const RadArgs* a) {
    const float factor = (float)rad_args_getf_or(a, "llama3_factor", 0.0);
    if (factor == 0.0f) return {};
    return { factor, (float)rad_args_getf_or(a, "llama3_low", 0.0),
             (float)rad_args_getf_or(a, "llama3_high", 0.0),
             (float)rad_args_getf_or(a, "llama3_orig", 0.0) };
}

inline bool rad_llama3_valid(const RadLlama3& l) {
    return std::isfinite(l.factor) &&
           (l.factor == 0.0f || (l.factor >= 1.0f && l.low > 0.0f && l.high > l.low &&
                                 std::isfinite(l.high) && l.orig > 0.0f && std::isfinite(l.orig)));
}

#undef RAD_YARN_HD

#else  /* C */
#include <math.h>

typedef struct RadYarn {
    float factor;      /* zero: YaRN off, rad_yarn_frequency() returns `inv` unchanged */
    float low;
    float high;
    float magnitude;   /* 1 when off */
} RadYarn;

static inline float rad_yarn_frequency(const RadYarn* y, float inv, int i) {
    float width, ramp, mask;
    if (y->factor == 0.0f) return inv;
    width = y->high == y->low ? 0.001f : y->high - y->low;
    ramp = fminf(1.0f, fmaxf(0.0f, ((float)i - y->low) / width));
    mask = 1.0f - ramp;
    return (inv / y->factor) * (1.0f - mask) + inv * mask;
}

static inline RadYarn rad_yarn_args(const RadArgs* a) {
    RadYarn y = { 0.0f, 0.0f, 0.0f, 1.0f };
    y.factor = (float)rad_args_getf_or(a, "yarn_factor", 0.0);
    if (y.factor == 0.0f) return y;
    y.low       = (float)rad_args_getf_or(a, "yarn_low", 0.0);
    y.high      = (float)rad_args_getf_or(a, "yarn_high", 0.0);
    y.magnitude = (float)rad_args_getf_or(a, "yarn_magnitude", 1.0);
    return y;
}

typedef struct RadLlama3 {
    float factor;    /* zero: off */
    float low;
    float high;
    float orig;
} RadLlama3;

static inline float rad_llama3_frequency(const RadLlama3* l, float inv) {
    float wavelen, sm;
    if (l->factor == 0.0f) return inv;
    wavelen = 6.28318530717958647692f / inv;
    if (wavelen < l->orig / l->high) return inv;
    if (wavelen > l->orig / l->low) return inv / l->factor;
    sm = (l->orig / wavelen - l->low) / (l->high - l->low);
    return (1.0f - sm) * inv / l->factor + sm * inv;
}

static inline int rad_yarn_valid(const RadYarn* y) {
    return isfinite(y->factor) && (y->factor == 0.0f ||
        (y->factor >= 1.0f && isfinite(y->low) && isfinite(y->high) && y->high >= y->low &&
         isfinite(y->magnitude) && y->magnitude > 0.0f));
}
#endif

#endif
