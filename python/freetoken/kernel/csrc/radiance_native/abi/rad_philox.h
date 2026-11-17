#ifndef RAD_PHILOX_H
#define RAD_PHILOX_H

/* Counter-based Gaussian noise, the single definition every kernel library uses, so an image's
 * starting latent depends only on its seed: not on the batch it shares a pass with, the rank
 * count, or the order its rows are visited in.
 *
 * Element e of a seed's stream is Philox4x32-10 at counter (e / 4, 0) keyed by the 64-bit seed,
 * its four words made uniform in (0, 1) as ((w >> 8) + 0.5) * 2^-24 and paired by Box-Muller --
 * words 0 and 1 give elements 4g and 4g+1 (r cos t, r sin t), words 2 and 3 give 4g+2 and 4g+3.
 * f32 throughout. A library's libm may differ from another's in the last place of log, cos and
 * sin, so two libraries agree to a tolerance and each agrees with itself exactly. */
#include <stdint.h>
#include <math.h>

#if defined(__HIPCC__)
#define RAD_PHILOX_HD __host__ __device__ __forceinline__
#else
#define RAD_PHILOX_HD static inline
#endif

RAD_PHILOX_HD void rad_philox4x32_10(uint32_t c[4], uint32_t k0, uint32_t k1) {
    const uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u, W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    for (int r = 0; r < 10; ++r) {
        const uint64_t p0 = (uint64_t)M0 * c[0], p1 = (uint64_t)M1 * c[2];
        const uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
        const uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
        const uint32_t n0 = hi1 ^ c[1] ^ k0, n2 = hi0 ^ c[3] ^ k1;
        c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
        k0 += W0;
        k1 += W1;
    }
}

/* Element `e` of the standard-normal stream of `seed`. */
RAD_PHILOX_HD float rad_philox_normal(uint64_t seed, uint64_t e) {
    const uint64_t g = e >> 2;
    uint32_t c[4] = { (uint32_t)g, (uint32_t)(g >> 32), 0u, 0u };
    rad_philox4x32_10(c, (uint32_t)seed, (uint32_t)(seed >> 32));
    const int j = (int)(e & 3u);
    const uint32_t wa = c[(j >> 1) * 2], wb = c[(j >> 1) * 2 + 1];
    const float ua = ((float)(wa >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float ub = ((float)(wb >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float r = sqrtf(-2.0f * logf(ua));
    const float t = 6.28318530717958647692f * ub;
    return (j & 1) ? r * sinf(t) : r * cosf(t);
}

/* Element `e` of the standard-normal stream (seed, stream): the counter's upper half is `stream`,
 * so stream 0 is rad_philox_normal and each other value an independent stream of the same seed
 * (an audio window's noise is stream 1 + its index). */
RAD_PHILOX_HD float rad_philox_normal_s(uint64_t seed, uint64_t stream, uint64_t e) {
    const uint64_t g = e >> 2;
    uint32_t c[4] = { (uint32_t)g, (uint32_t)(g >> 32), (uint32_t)stream, (uint32_t)(stream >> 32) };
    rad_philox4x32_10(c, (uint32_t)seed, (uint32_t)(seed >> 32));
    const int j = (int)(e & 3u);
    const uint32_t wa = c[(j >> 1) * 2], wb = c[(j >> 1) * 2 + 1];
    const float ua = ((float)(wa >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float ub = ((float)(wb >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float r = sqrtf(-2.0f * logf(ua));
    const float t = 6.28318530717958647692f * ub;
    return (j & 1) ? r * sinf(t) : r * cosf(t);
}

/* Element `e` of the uniform stream (seed, stream), in (0, 1): word e % 4 of the counter
 * (e / 4, stream) made uniform as the normal stream's words are. A sampler draws its pick from
 * it. */
RAD_PHILOX_HD float rad_philox_uniform_s(uint64_t seed, uint64_t stream, uint64_t e) {
    const uint64_t g = e >> 2;
    uint32_t c[4] = { (uint32_t)g, (uint32_t)(g >> 32), (uint32_t)stream, (uint32_t)(stream >> 32) };
    rad_philox4x32_10(c, (uint32_t)seed, (uint32_t)(seed >> 32));
    return ((float)(c[e & 3u] >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

#endif /* RAD_PHILOX_H */
