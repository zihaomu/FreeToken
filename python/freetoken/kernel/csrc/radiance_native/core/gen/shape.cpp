/* shape.cpp -- the shape generator's host geometry (shape.h): the picture, the sampler's grid, the
 * field's levels, marching cubes and the mesh's files; and the ticket a caller waits on. */
#include "shape.h"
#include "rad_internal.h"

#include "util/json.h"
#include "util/rowpool.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>

namespace rad {
namespace gen {

ShapeGenInfo shape_info(const RadShapeGenDecl& d, const std::string& name) {
    ShapeGenInfo g;
    g.name = name;
    g.canvas = d.canvas;
    g.enc_size = d.enc_size;
    g.enc_patch = d.enc_patch;
    g.border_ratio = d.border_ratio;
    for (int c = 0; c < 3; ++c) { g.pix_mean[c] = d.pix_mean[c]; g.pix_std[c] = d.pix_std[c]; }
    g.default_steps = d.default_steps;
    g.max_steps = d.max_steps;
    g.default_guidance = d.default_guidance;
    g.bounds = d.bounds;
    g.level = d.level;
    g.default_resolution = d.default_resolution;
    g.min_resolution = d.min_resolution;
    g.max_resolution = d.max_resolution;
    g.max_lanes = d.max_lanes;
    return g;
}

/* ================================================================== the picture */
namespace {

/* The picture's passes below compute each output row from its own inputs only, in the same order
 * whichever thread runs it, so they spread over the row pool and stay bit-exact. */
constexpr int kRowWidth = 8;

/* An 8-bit image, `ch` channels interleaved. */
struct Img {
    int32_t w = 0, h = 0, ch = 0;
    std::vector<uint8_t> px;
    uint8_t*       at(int32_t y, int32_t x)       { return px.data() + ((size_t)y * w + x) * ch; }
    const uint8_t* at(int32_t y, int32_t x) const { return px.data() + ((size_t)y * w + x) * ch; }
};

/* OpenCV's saturate_cast<uchar> of a float: round half to even, clipped. */
inline uint8_t sat_u8(float v) {
    const float r = std::nearbyint(v);
    return (uint8_t)(r < 0.0f ? 0.0f : r > 255.0f ? 255.0f : r);
}

/* cv::resize's INTER_AREA when it shrinks both sides: computeResizeAreaTab's weights, a row's
 * horizontal sums, then the weighted rows, in float. */
struct AreaTap { int32_t di, si; float alpha; };
std::vector<AreaTap> area_tab(int32_t ssize, int32_t dsize, double scale) {
    std::vector<AreaTap> tab;
    for (int32_t dx = 0; dx < dsize; ++dx) {
        const double fsx1 = dx * scale, fsx2 = fsx1 + scale;
        const double cell = std::min(scale, ssize - fsx1);
        int32_t sx1 = (int32_t)std::ceil(fsx1), sx2 = (int32_t)std::floor(fsx2);
        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);
        if (sx1 - fsx1 > 1e-3) tab.push_back({ dx, sx1 - 1, (float)((sx1 - fsx1) / cell) });
        for (int32_t sx = sx1; sx < sx2; ++sx) tab.push_back({ dx, sx, (float)(1.0 / cell) });
        if (fsx2 - sx2 > 1e-3)
            tab.push_back({ dx, sx2, (float)(std::min(std::min(fsx2 - sx2, 1.0), cell) / cell) });
    }
    return tab;
}

void resize_area_down(const Img& s, int32_t dw, int32_t dh, Img* d) {
    const double sx = (double)s.w / dw, sy = (double)s.h / dh;
    const std::vector<AreaTap> xt = area_tab(s.w, dw, sx), yt = area_tab(s.h, dh, sy);
    const int32_t ch = s.ch;
    d->w = dw; d->h = dh; d->ch = ch;
    d->px.assign((size_t)dw * dh * ch, 0);
    /* Each destination row's first tap in yt, which lists them in row order. */
    std::vector<size_t> first((size_t)dh + 1, yt.size());
    for (size_t k = yt.size(); k-- > 0;) first[(size_t)yt[k].di] = k;
    for (int32_t dy = dh; dy-- > 0;) first[(size_t)dy] = std::min(first[(size_t)dy], first[(size_t)dy + 1]);
    row_pool_run(dh, kRowWidth, [&](int64_t dy) {
        thread_local std::vector<float> row, acc;
        row.assign((size_t)dw * ch, 0.0f);
        acc.assign((size_t)dw * ch, 0.0f);
        for (size_t k = first[(size_t)dy]; k < yt.size() && yt[k].di == dy; ++k) {
            std::fill(row.begin(), row.end(), 0.0f);
            const uint8_t* src = s.at(yt[k].si, 0);
            for (const AreaTap& t : xt)
                for (int32_t c = 0; c < ch; ++c)
                    row[(size_t)t.di * ch + c] += src[(size_t)t.si * ch + c] * t.alpha;
            for (size_t i = 0; i < acc.size(); ++i) acc[i] += row[i] * yt[k].alpha;
        }
        uint8_t* dst = d->at((int32_t)dy, 0);
        for (size_t i = 0; i < acc.size(); ++i) dst[i] = sat_u8(acc[i]);
    });
}

/* cv::resize's fixed-point separable filter for 8-bit images: each tap's float coefficient
 * rounded to 11 bits, the horizontal sums in int, the vertical sum shifted back by 22 with
 * rounding. `taps` coefficients a destination pixel, first at source index ofs[d] (indices
 * clamped to the image, OpenCV's BORDER_REPLICATE). */
struct Taps {
    int32_t ksize = 0;
    std::vector<int32_t> ofs;
    std::vector<int16_t> c;
};

Taps taps_linear_area(int32_t ssize, int32_t dsize) {
    /* INTER_AREA when it does not shrink both sides: linear taps in OpenCV's area mode. */
    const double scale = (double)ssize / dsize, inv = (double)dsize / ssize;
    Taps t;
    t.ksize = 2;
    for (int32_t dx = 0; dx < dsize; ++dx) {
        const int32_t sx = (int32_t)std::floor(dx * scale);
        float fx = (float)((dx + 1) - (sx + 1) * inv);
        fx = fx <= 0 ? 0.0f : fx - std::floor(fx);
        t.ofs.push_back(sx);
        t.c.push_back((int16_t)std::nearbyint((1.0f - fx) * 2048.0f));
        t.c.push_back((int16_t)std::nearbyint(fx * 2048.0f));
    }
    return t;
}

Taps taps_cubic(int32_t ssize, int32_t dsize) {
    const double scale = (double)ssize / dsize;
    const float A = -0.75f;
    Taps t;
    t.ksize = 4;
    for (int32_t dx = 0; dx < dsize; ++dx) {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        const int32_t sx = (int32_t)std::floor(fx);
        fx -= sx;
        float w[4];
        w[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
        w[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
        w[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
        w[3] = 1.0f - w[0] - w[1] - w[2];
        t.ofs.push_back(sx - 1);
        for (float v : w) t.c.push_back((int16_t)std::nearbyint(v * 2048.0f));
    }
    return t;
}

void resize_fixed(const Img& s, const Taps& tx, const Taps& ty, int32_t dw, int32_t dh, Img* d) {
    const int32_t ch = s.ch, kx = tx.ksize, ky = ty.ksize;
    /* Every source row through the horizontal taps, then each destination row from its rows. The
     * clamped source offsets are worked out once a tap, so the loops below only multiply and add,
     * in the same order as one pixel at a time. */
    std::vector<int32_t> xo((size_t)dw * kx);
    for (int32_t dx = 0; dx < dw; ++dx)
        for (int32_t k = 0; k < kx; ++k)
            xo[(size_t)dx * kx + k] = std::clamp(tx.ofs[(size_t)dx] + k, 0, s.w - 1) * ch;
    const size_t rw = (size_t)dw * ch;
    std::vector<int32_t> hrow((size_t)s.h * rw);
    row_pool_run(s.h, kRowWidth, [&](int64_t y) {
        const uint8_t* src = s.at((int32_t)y, 0);
        int32_t* h = hrow.data() + (size_t)y * rw;
        for (int32_t dx = 0; dx < dw; ++dx) {
            const int32_t* o = xo.data() + (size_t)dx * kx;
            const int16_t* cf = tx.c.data() + (size_t)dx * kx;
            for (int32_t c = 0; c < ch; ++c) {
                int32_t sum = 0;
                for (int32_t k = 0; k < kx; ++k) sum += src[o[k] + c] * cf[k];
                h[(size_t)dx * ch + c] = sum;
            }
        }
    });
    d->w = dw; d->h = dh; d->ch = ch;
    d->px.assign((size_t)dw * dh * ch, 0);
    row_pool_run(dh, kRowWidth, [&](int64_t dy) {
        thread_local std::vector<int64_t> acc;
        acc.assign(rw, 0);
        for (int32_t k = 0; k < ky; ++k) {
            const int32_t sy = std::clamp(ty.ofs[(size_t)dy] + k, 0, s.h - 1);
            const int32_t* h = hrow.data() + (size_t)sy * rw;
            simd::madd_i32_i64(acc.data(), h, ty.c[(size_t)dy * ky + k], rw);
        }
        uint8_t* out = d->px.data() + (size_t)dy * rw;
        for (size_t i = 0; i < rw; ++i)
            out[i] = (uint8_t)std::clamp<int64_t>((acc[i] + (1 << 21)) >> 22, 0, 255);
    });
}

/* cv2.resize(src, (dw, dh), INTER_AREA). */
void resize_area(const Img& s, int32_t dw, int32_t dh, Img* d) {
    if (dw == s.w && dh == s.h) { *d = s; return; }
    if (s.w >= dw && s.h >= dh) { resize_area_down(s, dw, dh, d); return; }
    resize_fixed(s, taps_linear_area(s.w, dw), taps_linear_area(s.h, dh), dw, dh, d);
}

/* cv2.resize(src, (dw, dh), INTER_CUBIC). */
void resize_cubic(const Img& s, int32_t dw, int32_t dh, Img* d) {
    if (dw == s.w && dh == s.h) { *d = s; return; }
    resize_fixed(s, taps_cubic(s.w, dw), taps_cubic(s.h, dh), dw, dh, d);
}

/* torchvision's antialiased bilinear resize along one axis (align_corners false): the triangle
 * filter around each output's centre, widened by the scale when shrinking, normalised over the
 * taps inside the image. */
struct AaTaps { std::vector<int32_t> lo; std::vector<std::vector<float>> w; };
AaTaps aa_taps(int32_t in, int32_t out) {
    const double scale = (double)in / out;
    const double support = scale >= 1.0 ? scale : 1.0;
    const double inv = scale >= 1.0 ? 1.0 / scale : 1.0;
    AaTaps t;
    for (int32_t i = 0; i < out; ++i) {
        const double center = scale * (i + 0.5);
        const int32_t lo = std::max((int32_t)(center - support + 0.5), 0);
        const int32_t hi = std::min((int32_t)(center + support + 0.5), in);
        std::vector<float> w;
        double total = 0.0;
        for (int32_t j = lo; j < hi; ++j) {
            const double x = std::fabs((j - center + 0.5) * inv);
            const double v = x < 1.0 ? 1.0 - x : 0.0;
            w.push_back((float)v);
            total += v;
        }
        if (total > 0.0) for (float& v : w) v = (float)(v / total);
        t.lo.push_back(lo);
        t.w.push_back(std::move(w));
    }
    return t;
}

}  /* namespace */

int shape_picture(const ShapeGenInfo& g, const mm::FrameRGBA& rgba, std::vector<float>* patches,
                  std::string* why) {
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    const int32_t H = rgba.h, W = rgba.w;
    if (H < 1 || W < 1 || rgba.rgba.size() != (size_t)H * W * 4) return no("the picture is empty");
    /* ---- ImageProcessorV2.recenter: the alpha's bounding box on a square of the longer side */
    /* Each row's first and last opaque column, then their extremes. */
    std::vector<int32_t> rx0((size_t)H), rx1((size_t)H);
    row_pool_run(H, kRowWidth, [&](int64_t y) {
        const uint8_t* r = rgba.rgba.data() + (size_t)y * W * 4;
        int32_t a = 0, b = W - 1;
        while (a < W && !r[(size_t)a * 4 + 3]) ++a;
        while (b > a && !r[(size_t)b * 4 + 3]) --b;
        rx0[(size_t)y] = a;
        rx1[(size_t)y] = a < W ? b : -1;
    });
    int32_t y0 = H, y1 = -1, x0 = W, x1 = -1;
    for (int32_t y = 0; y < H; ++y)
        if (rx1[(size_t)y] >= 0) {
            y0 = std::min(y0, y); y1 = std::max(y1, y);
            x0 = std::min(x0, rx0[(size_t)y]); x1 = std::max(x1, rx1[(size_t)y]);
        }
    if (y1 < 0) return no("the picture is fully transparent: nothing to make a shape of");
    const int32_t h = y1 - y0, w = x1 - x0;
    if (h == 0 || w == 0)
        return no("the picture's opaque part is a single row or column: nothing to make a shape "
                  "of");
    const int32_t size = std::max(H, W);
    const int32_t desired = (int32_t)(size * (1.0 - g.border_ratio));
    const double scale = (double)desired / std::max(h, w);
    const int32_t h2 = (int32_t)(h * scale), w2 = (int32_t)(w * scale);
    if (h2 < 1 || w2 < 1)
        return no("the picture's opaque part is too thin to scale onto the canvas");
    Img crop;
    crop.w = w; crop.h = h; crop.ch = 4;
    crop.px.resize((size_t)w * h * 4);
    for (int32_t y = 0; y < h; ++y)
        std::memcpy(crop.at(y, 0), rgba.rgba.data() + ((size_t)(y0 + y) * W + x0) * 4,
                    (size_t)w * 4);
    Img fit;
    resize_area(crop, w2, h2, &fit);
    const int32_t oy = (size - h2) / 2, ox = (size - w2) / 2;
    /* Flattened onto white in float, then truncated to 8 bits, as numpy's astype does. */
    Img sq;
    sq.w = size; sq.h = size; sq.ch = 3;
    sq.px.assign((size_t)size * size * 3, 255);
    row_pool_run(h2, kRowWidth, [&](int64_t y) {
        for (int32_t x = 0; x < w2; ++x) {
            const uint8_t* p = fit.at((int32_t)y, x);
            const float a = (float)p[3] / 255.0f;
            uint8_t* q = sq.at(oy + (int32_t)y, ox + x);
            for (int c = 0; c < 3; ++c) {
                const float v = (float)p[c] * a + 255.0f * (1.0f - a);
                q[c] = (uint8_t)std::clamp(v, 0.0f, 255.0f);
            }
        }
    });
    /* ---- the canvas, INTER_CUBIC */
    const int32_t C = (int32_t)g.canvas;
    Img can;
    resize_cubic(sq, C, C, &can);
    /* ---- the encoder's transform: 0..1, bilinear (antialiased) to enc_size, normalised */
    const int32_t E = (int32_t)g.enc_size, P = (int32_t)g.enc_patch, G = E / P;
    const AaTaps ta = aa_taps(C, E);
    std::vector<float> hor((size_t)C * E * 3), img((size_t)E * E * 3);
    row_pool_run(C, kRowWidth, [&](int64_t y) {
        for (int32_t x = 0; x < E; ++x)
            for (int c = 0; c < 3; ++c) {
                float s = 0.0f;
                const std::vector<float>& wt = ta.w[(size_t)x];
                for (size_t k = 0; k < wt.size(); ++k)
                    s += wt[k] * ((float)can.at((int32_t)y, ta.lo[(size_t)x] + (int32_t)k)[c] / 255.0f);
                hor[((size_t)y * E + x) * 3 + c] = s;
            }
    });
    row_pool_run(E, kRowWidth, [&](int64_t y) {
        for (int32_t x = 0; x < E; ++x)
            for (int c = 0; c < 3; ++c) {
                float s = 0.0f;
                const std::vector<float>& wt = ta.w[(size_t)y];
                for (size_t k = 0; k < wt.size(); ++k)
                    s += wt[k] * hor[(((size_t)ta.lo[(size_t)y] + k) * E + x) * 3 + c];
                img[((size_t)y * E + x) * 3 + c] =
                    (float)((s - g.pix_mean[c]) / g.pix_std[c]);
            }
    });
    /* ---- patches, row-major; a patch's values (channel, row, column) */
    const int64_t pd = 3 * (int64_t)P * P;
    patches->assign((size_t)G * G * pd, 0.0f);
    for (int32_t py = 0; py < G; ++py)
        for (int32_t px = 0; px < G; ++px) {
            float* o = patches->data() + ((size_t)py * G + px) * pd;
            for (int c = 0; c < 3; ++c)
                for (int32_t i = 0; i < P; ++i)
                    for (int32_t j = 0; j < P; ++j)
                        o[((size_t)c * P + i) * P + j] =
                            img[(((size_t)py * P + i) * E + px * P + j) * 3 + c];
        }
    return RAD_OK;
}

/* ================================================================== the sampler and the grid */
std::vector<float> shape_sigmas(int32_t steps) {
    std::vector<float> s;
    if (steps < 2) return s;
    const double step = 1.0 / (steps - 1);
    for (int32_t i = 0; i < steps; ++i) s.push_back(i == steps - 1 ? 1.0f : (float)(i * step));
    return s;
}

std::vector<int32_t> shape_levels(int32_t resolution) {
    std::vector<int32_t> r;
    if (resolution < 63) r.push_back(resolution);
    for (int32_t v = resolution; v >= 63; v /= 2) r.push_back(v);
    std::reverse(r.begin(), r.end());
    return r;
}

std::vector<float> shape_axis(double bounds, int32_t r) {
    std::vector<float> a((size_t)r + 1);
    const double step = 2.0 * bounds / r;
    for (int32_t i = 0; i <= r; ++i) a[(size_t)i] = (float)(i * step + -bounds);
    a[(size_t)r] = (float)bounds;
    return a;
}

namespace {

/* A 3 x 3 x 3 dilation of an n^3 mask (zero outside), as separable maxima along each axis. */
void dilate(std::vector<uint8_t>& m, int64_t n) {
    std::vector<uint8_t> t(m.size());
    /* One axis: each voxel ORs its two neighbours along it, where they exist. Along the
     * contiguous axis a line of n voxels at a time; along the others whole rows of `stride`
     * voxels, each with the rows either side of it. */
    auto pass = [&](const std::vector<uint8_t>& src, std::vector<uint8_t>& dst, int64_t stride) {
        const uint8_t* s = src.data();
        uint8_t* d = dst.data();
        const int64_t total = (int64_t)src.size(), plane = n * stride;
        for (int64_t base = 0; base < total; base += plane) {
            if (stride == 1) {
                simd::or_adjacent_u8(d + base, s + base, (size_t)n);
                continue;
            }
            for (int64_t c = 0; c < n; ++c) {
                const uint8_t* a = s + base + c * stride;
                simd::or3_u8(d + base + c * stride, a, c > 0 ? a - stride : nullptr,
                             c + 1 < n ? a + stride : nullptr, (size_t)stride);
            }
        }
    };
    pass(m, t, 1);
    pass(t, m, n);
    pass(m, t, n * n);
    m.swap(t);
}

}  /* namespace */

std::vector<uint32_t> shape_refine(const std::vector<float>& prev, int32_t rp, int32_t r,
                                   double level, bool last) {
    const int64_t np = (int64_t)rp + 1, n = (int64_t)r + 1;
    std::vector<uint8_t> near((size_t)(np * np * np), 0);
    const float lv = (float)level;
    /* A point is near the surface when it is within 0.95 of the level, or a neighbour along an
     * axis lies on the other side of it (neither a NaN, which is not sampled; past the grid's
     * edge the reference replicates the border, which never crosses). The sign of raw - lv is
     * the comparison of raw with lv, so each row's simd::level_bits_f32 bitmaps answer it 64
     * points at a time, three planes held at once. */
    const int64_t W = (np + 63) / 64, rowb = W * 4;
    std::vector<uint64_t> bits((size_t)(3 * np * rowb));
    auto plane = [&](int64_t i) { return bits.data() + (size_t)((i % 3) * np * rowb); };
    auto classify = [&](int64_t i) {
        uint64_t* pl = plane(i);
        for (int64_t j = 0; j < np; ++j) {
            uint64_t* row = pl + j * rowb;
            simd::level_bits_f32(&prev[(size_t)((i * np + j) * np)], (size_t)np, lv, 0.95f, row,
                                 row + W, row + 2 * W, row + 3 * W);
        }
    };
    auto diff = [&](const uint64_t* a, const uint64_t* b, int64_t w) {   /* rows of four maps */
        return ((a[w] ^ b[w]) | (a[W + w] ^ b[W + w])) & ~b[2 * W + w];
    };
    /* Row maps moved one point along k: `up` holds point k - 1 at bit k, `dn` point k + 1. */
    std::vector<uint64_t> up((size_t)rowb), dn((size_t)rowb);
    if (np > 0) classify(0);
    for (int64_t i = 0; i < np; ++i) {
        if (i + 1 < np) classify(i + 1);
        for (int64_t j = 0; j < np; ++j) {
            const uint64_t* c = plane(i) + j * rowb;
            for (int m = 0; m < 3; ++m)
                for (int64_t w = 0; w < W; ++w) {
                    const uint64_t* x = c + m * W;
                    up[(size_t)(m * W + w)] = (x[w] << 1) | (w > 0 ? x[w - 1] >> 63 : 0);
                    dn[(size_t)(m * W + w)] = (x[w] >> 1) | (w + 1 < W ? x[w + 1] << 63 : 0);
                }
            const uint64_t* nb[4] = { j > 0 ? c - rowb : nullptr, j + 1 < np ? c + rowb : nullptr,
                                      i > 0 ? plane(i - 1) + j * rowb : nullptr,
                                      i + 1 < np ? plane(i + 1) + j * rowb : nullptr };
            uint8_t* out = near.data() + (size_t)((i * np + j) * np);
            for (int64_t w = 0; w < W; ++w) {
                /* The row's points only; k - 1 exists from k = 1, k + 1 up to k = np - 2. */
                const int64_t k0 = w * 64, left = np - k0;
                const uint64_t inside = left >= 64 ? ~0ull : (1ull << left) - 1;
                const uint64_t has_dn = left > 64 ? ~0ull : (1ull << (left - 1)) - 1;
                uint64_t x = c[3 * W + w] | (diff(c, up.data(), w) & (w == 0 ? ~1ull : ~0ull)) |
                             (diff(c, dn.data(), w) & has_dn);
                for (const uint64_t* o : nb) if (o) x |= diff(c, o, w);
                x &= inside & ~c[2 * W + w];
                while (x) {
                    out[k0 + __builtin_ctzll(x)] = 1;
                    x &= x - 1;
                }
            }
        }
    }
    if (!last) dilate(near, np);
    std::vector<uint8_t> mark((size_t)(n * n * n), 0);
    auto upi = [&](int64_t c) { return (c * r + rp / 2) / rp; };
    /* The masks are sparse: eight bytes looked at a time. */
    auto each_set = [](const std::vector<uint8_t>& m, auto&& fn) {
        const size_t sz = m.size();
        size_t p = 0;
        for (; p + 8 <= sz; p += 8) {
            uint64_t x;
            std::memcpy(&x, m.data() + p, 8);
            while (x) {
                const int b = __builtin_ctzll(x) >> 3;
                fn((int64_t)p + b);
                x &= ~(0xffull << (8 * b));
            }
        }
        for (; p < sz; ++p) if (m[p]) fn((int64_t)p);
    };
    each_set(near, [&](int64_t p) {
        const int64_t i = p / (np * np), j = p / np % np, k = p % np;
        mark[(size_t)((upi(i) * n + upi(j)) * n + upi(k))] = 1;
    });
    dilate(mark, n);
    if (last) dilate(mark, n);
    std::vector<uint32_t> pts;
    each_set(mark, [&](int64_t p) { pts.push_back((uint32_t)p); });
    return pts;
}

/* ================================================================== marching cubes */
namespace {

/* Corner c of a cube is at (c & 1, c >> 1 & 1, c >> 2 & 1) along the grid's (i, j, k). Edge e is
 * along axis e / 4 from corner kEdge[e][0] to kEdge[e][1]. */
struct McTables {
    int32_t edge[12][2];
    /* Each case's triangles as edge triples, -1 ended. */
    int8_t  tri[256][31];
};

McTables build_tables() {
    McTables T{};
    int ne = 0;
    for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 8; ++c)
            if (!(c >> a & 1)) { T.edge[ne][0] = c; T.edge[ne][1] = c | (1 << a); ++ne; }
    auto edge_of = [&](int c0, int c1) {
        for (int e = 0; e < 12; ++e)
            if ((T.edge[e][0] == c0 && T.edge[e][1] == c1) ||
                (T.edge[e][0] == c1 && T.edge[e][1] == c0))
                return e;
        return -1;
    };
    auto pos = [](int c, double* p) { p[0] = c & 1; p[1] = c >> 1 & 1; p[2] = c >> 2 & 1; };
    for (int m = 0; m < 256; ++m) {
        for (int i = 0; i < 31; ++i) T.tri[m][i] = -1;
        if (m == 0 || m == 255) continue;
        auto in = [&](int c) { return (m >> c & 1) != 0; };
        /* Each face's segments between its crossing edges. A face with all four edges crossing
         * (its inside corners diagonal) separates the inside corners, the rule both cubes that
         * share the face apply, so the surface has no cracks. */
        std::vector<std::array<int, 2>> seg;
        for (int a = 0; a < 3; ++a)
            for (int s = 0; s < 2; ++s) {
                const int u = (a + 1) % 3, v = (a + 2) % 3;
                const int base = s << a;
                const int cyc[4] = { base, base | (1 << u), base | (1 << u) | (1 << v),
                                     base | (1 << v) };
                int fe[4];
                bool cr[4];
                int ncr = 0;
                for (int q = 0; q < 4; ++q) {
                    fe[q] = edge_of(cyc[q], cyc[(q + 1) % 4]);
                    cr[q] = in(cyc[q]) != in(cyc[(q + 1) % 4]);
                    ncr += cr[q];
                }
                if (ncr == 2) {
                    int a0 = -1, a1 = -1;
                    for (int q = 0; q < 4; ++q) {
                        if (!cr[q]) continue;
                        if (a0 < 0) a0 = fe[q]; else a1 = fe[q];
                    }
                    seg.push_back({ a0, a1 });
                } else if (ncr == 4) {
                    for (int q = 0; q < 4; ++q)
                        if (in(cyc[q])) seg.push_back({ fe[(q + 3) % 4], fe[q] });
                }
            }
        /* The segments close into loops; each crossing edge is in exactly two. */
        std::vector<bool> used(seg.size(), false);
        int nt = 0;
        for (size_t s0 = 0; s0 < seg.size(); ++s0) {
            if (used[s0]) continue;
            std::vector<int> loop = { seg[s0][0], seg[s0][1] };
            used[s0] = true;
            for (bool grew = true; grew;) {
                grew = false;
                for (size_t q = 0; q < seg.size(); ++q) {
                    if (used[q]) continue;
                    const int tail = loop.back();
                    int nx = -1;
                    if (seg[q][0] == tail) nx = seg[q][1];
                    else if (seg[q][1] == tail) nx = seg[q][0];
                    if (nx < 0) continue;
                    used[q] = true;
                    grew = true;
                    if (nx != loop.front()) loop.push_back(nx);
                }
            }
            /* Orientation: the loop's normal (Newell's, over the edges' midpoints) points from the
             * inside corners to the outside ones. */
            double nrm[3] = { 0, 0, 0 }, out[3] = { 0, 0, 0 };
            std::vector<std::array<double, 3>> mid;
            for (int e : loop) {
                double p0[3], p1[3];
                pos(T.edge[e][0], p0);
                pos(T.edge[e][1], p1);
                mid.push_back({ (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2 });
                const double sg = in(T.edge[e][0]) ? 1.0 : -1.0;
                for (int d = 0; d < 3; ++d) out[d] += sg * (p1[d] - p0[d]);
            }
            for (size_t q = 0; q < mid.size(); ++q) {
                const auto& c = mid[q];
                const auto& d = mid[(q + 1) % mid.size()];
                nrm[0] += (c[1] - d[1]) * (c[2] + d[2]);
                nrm[1] += (c[2] - d[2]) * (c[0] + d[0]);
                nrm[2] += (c[0] - d[0]) * (c[1] + d[1]);
            }
            if (nrm[0] * out[0] + nrm[1] * out[1] + nrm[2] * out[2] < 0)
                std::reverse(loop.begin(), loop.end());
            for (size_t q = 1; q + 1 < loop.size() && nt < 10; ++q) {
                T.tri[m][3 * nt] = (int8_t)loop[0];
                T.tri[m][3 * nt + 1] = (int8_t)loop[q];
                T.tri[m][3 * nt + 2] = (int8_t)loop[q + 1];
                ++nt;
            }
        }
    }
    return T;
}

const McTables& tables() {
    static const McTables t = build_tables();
    return t;
}

}  /* namespace */

void shape_mesh(const std::vector<float>& field, int32_t r, double bounds, double level,
                ShapeMesh* out) {
    out->v.clear();
    out->f.clear();
    const McTables& T = tables();
    const int64_t n = (int64_t)r + 1, nn = n * n;
    const float lv = (float)level;
    const double sc = 2.0 * bounds / (double)n;
    /* Each grid row's bitmaps (simd::level_bits_f32), for the two planes a slab of cubes lies
     * between: bit k of row j of plane p is grid point (i + p, j, k). */
    const int64_t W = (n + 63) / 64;
    std::vector<uint64_t> gt[2], nan[2], scratch((size_t)(2 * W));
    for (int p = 0; p < 2; ++p) { gt[p].assign((size_t)(n * W + 1), 0); nan[p].assign((size_t)(n * W + 1), 0); }
    auto classify = [&](int64_t i, int p) {
        for (int64_t j = 0; j < n; ++j)
            simd::level_bits_f32(&field[(size_t)((i * n + j) * n)], (size_t)n, lv, 0.0f,
                                 &gt[p][(size_t)(j * W)], scratch.data(), &nan[p][(size_t)(j * W)],
                                 scratch.data() + W);
    };
    /* An edge's vertex by its first grid point: the x edges between planes i and i + 1, and the
     * y and z edges of planes i and i + 1. Vertex indices only grow, so a slot holds a vertex of
     * the current slab when it is at least the vertex count the slab started at (`v_now`; for
     * the edges of plane i, which the slab before also made, `v_last`): no slot is cleared. */
    std::vector<int32_t> ex((size_t)nn, -1), ey[2], ez[2];
    for (int p = 0; p < 2; ++p) { ey[p].assign((size_t)nn, -1); ez[p].assign((size_t)nn, -1); }
    auto value = [&](int64_t i, int64_t j, int64_t k) {
        return field[(size_t)((i * n + j) * n + k)];
    };
    /* Bits [w * 64, w * 64 + 64) of a row, moved down by one: bit k is the row's bit k + 1. */
    auto next = [&](const uint64_t* row, int64_t w) {
        return (row[w] >> 1) | (w + 1 < W ? row[w + 1] << 63 : 0);
    };
    if (n >= 2) classify(0, 1);
    int32_t v_last = 0;
    for (int64_t i = 0; i + 1 < n; ++i) {
        std::swap(gt[0], gt[1]);
        std::swap(nan[0], nan[1]);
        classify(i + 1, 1);
        if (i > 0) {
            std::swap(ey[0], ey[1]);
            std::swap(ez[0], ez[1]);
        }
        const int32_t v_now = (int32_t)(out->v.size() / 3);
        auto fresh = [&](int32_t at, int a, int64_t bi) {
            return at >= (a == 0 || bi == 1 ? v_now : v_last) && at >= 0;
        };
        for (int64_t j = 0; j + 1 < n; ++j) {
            const uint64_t* g[4] = { &gt[0][(size_t)(j * W)], &gt[1][(size_t)(j * W)],
                                     &gt[0][(size_t)((j + 1) * W)], &gt[1][(size_t)((j + 1) * W)] };
            const uint64_t* q[4] = { &nan[0][(size_t)(j * W)], &nan[1][(size_t)(j * W)],
                                     &nan[0][(size_t)((j + 1) * W)], &nan[1][(size_t)((j + 1) * W)] };
            for (int64_t w = 0; w < W; ++w) {
                /* Corner c of the cube at k: row c & 3 (i, j offsets), at k + (c >> 2). */
                uint64_t cb[8], any = 0, all = ~0ull, bad = 0;
                for (int c = 0; c < 4; ++c) {
                    cb[c] = g[c][w];
                    cb[c + 4] = next(g[c], w);
                    bad |= q[c][w] | next(q[c], w);
                }
                for (int c = 0; c < 8; ++c) { any |= cb[c]; all &= cb[c]; }
                /* Cubes start at k < n - 1. */
                const int64_t k0 = w * 64;
                uint64_t live = any & ~all & ~bad;
                if (n - 1 - k0 < 64) live &= (n - 1 - k0 > 0) ? (1ull << (n - 1 - k0)) - 1 : 0;
                while (live) {
                    const int b = __builtin_ctzll(live);
                    live &= live - 1;
                    const int64_t k = k0 + b;
                    int m = 0;
                    for (int c = 0; c < 8; ++c) m |= (int)((cb[c] >> b) & 1u) << c;
                    float v[8];
                    for (int c = 0; c < 8; ++c) v[c] = value(i + (c & 1), j + (c >> 1 & 1), k + (c >> 2 & 1));
                    for (int t = 0; t < 30 && T.tri[m][t] >= 0; ++t) {
                        const int e = T.tri[m][t];
                        const int c0 = T.edge[e][0], c1 = T.edge[e][1], a = e / 4;
                        const int64_t bi = c0 & 1, bj = c0 >> 1 & 1, bk = c0 >> 2 & 1;
                        int32_t* slot = a == 0 ? &ex[(size_t)((j + bj) * n + k + bk)]
                                      : a == 1 ? &ey[bi][(size_t)(j * n + k + bk)]
                                               : &ez[bi][(size_t)((j + bj) * n + k)];
                        if (!fresh(*slot, a, bi)) {
                            const float t0 = v[c0], t1 = v[c1];
                            const float f = t1 != t0 ? (lv - t0) / (t1 - t0) : 0.5f;
                            float gp[3] = { (float)(i + bi), (float)(j + bj), (float)(k + bk) };
                            gp[a] += f;
                            *slot = (int32_t)(out->v.size() / 3);
                            for (int d = 0; d < 3; ++d) out->v.push_back((float)(gp[d] * sc - bounds));
                        }
                        out->f.push_back((uint32_t)*slot);
                    }
                }
            }
        }
        v_last = v_now;
    }
}

/* ================================================================== files */
namespace {

void put(std::string& s, const void* p, size_t n) { s.append((const char*)p, n); }
template <class T> void put_v(std::string& s, T v) { put(s, &v, sizeof v); }

/* Area-weighted vertex normals, unit length (zero for a vertex of only degenerate faces). */
std::vector<float> vertex_normals(const ShapeMesh& m) {
    std::vector<float> nrm(m.v.size(), 0.0f);
    for (size_t t = 0; t + 2 < m.f.size(); t += 3) {
        const float* a = &m.v[(size_t)m.f[t] * 3];
        const float* b = &m.v[(size_t)m.f[t + 1] * 3];
        const float* c = &m.v[(size_t)m.f[t + 2] * 3];
        const float u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        const float w[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        const float x[3] = { u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                             u[0] * w[1] - u[1] * w[0] };
        for (int q = 0; q < 3; ++q)
            for (int d = 0; d < 3; ++d) nrm[(size_t)m.f[t + q] * 3 + d] += x[d];
    }
    for (size_t i = 0; i + 2 < nrm.size(); i += 3) {
        const float l =
            std::sqrt(nrm[i] * nrm[i] + nrm[i + 1] * nrm[i + 1] + nrm[i + 2] * nrm[i + 2]);
        if (l > 0.0f) for (int d = 0; d < 3; ++d) nrm[i + d] /= l;
    }
    return nrm;
}

std::string glb(const ShapeMesh& m) {
    const size_t nv = m.v.size() / 3, ni = m.f.size();
    const std::vector<float> nrm = vertex_normals(m);
    float lo[3] = { INFINITY, INFINITY, INFINITY }, hi[3] = { -INFINITY, -INFINITY, -INFINITY };
    for (size_t i = 0; i < nv; ++i)
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], m.v[i * 3 + d]);
            hi[d] = std::max(hi[d], m.v[i * 3 + d]);
        }
    const size_t vb = nv * 12, ib = ni * 4;
    std::string bin;
    put(bin, m.v.data(), vb);
    put(bin, nrm.data(), vb);
    put(bin, m.f.data(), ib);
    while (bin.size() % 4) bin.push_back('\0');
    using json = Json;
    json j;
    j["asset"] = { { "version", "2.0" }, { "generator", "radiance" } };
    j["scene"] = 0;
    j["scenes"] = json::array({ { { "nodes", json::array({ 0 }) } } });
    j["nodes"] = json::array({ { { "mesh", 0 } } });
    const json prim = { { "attributes", { { "POSITION", 0 }, { "NORMAL", 1 } } },
                        { "indices", 2 }, { "mode", 4 } };
    j["meshes"] = json::array({ { { "primitives", json::array({ prim }) } } });
    j["buffers"] = json::array({ { { "byteLength", bin.size() } } });
    j["bufferViews"] = json::array({
        { { "buffer", 0 }, { "byteOffset", 0 }, { "byteLength", vb }, { "target", 34962 } },
        { { "buffer", 0 }, { "byteOffset", vb }, { "byteLength", vb }, { "target", 34962 } },
        { { "buffer", 0 }, { "byteOffset", 2 * vb }, { "byteLength", ib }, { "target", 34963 } } });
    j["accessors"] = json::array({
        { { "bufferView", 0 }, { "componentType", 5126 }, { "count", nv }, { "type", "VEC3" },
          { "min", { lo[0], lo[1], lo[2] } }, { "max", { hi[0], hi[1], hi[2] } } },
        { { "bufferView", 1 }, { "componentType", 5126 }, { "count", nv }, { "type", "VEC3" } },
        { { "bufferView", 2 }, { "componentType", 5125 }, { "count", ni },
          { "type", "SCALAR" } } });
    std::string js = j.dump();
    while (js.size() % 4) js.push_back(' ');
    std::string out;
    put_v<uint32_t>(out, 0x46546C67u);   /* "glTF" */
    put_v<uint32_t>(out, 2);
    put_v<uint32_t>(out, (uint32_t)(12 + 8 + js.size() + 8 + bin.size()));
    put_v<uint32_t>(out, (uint32_t)js.size());
    put_v<uint32_t>(out, 0x4E4F534Au);   /* "JSON" */
    out += js;
    put_v<uint32_t>(out, (uint32_t)bin.size());
    put_v<uint32_t>(out, 0x004E4942u);   /* "BIN\0" */
    out += bin;
    return out;
}

std::string obj(const ShapeMesh& m) {
    std::string s;
    char line[128];
    for (size_t i = 0; i + 2 < m.v.size(); i += 3) {
        const int n = std::snprintf(line, sizeof line, "v %.9g %.9g %.9g\n", (double)m.v[i],
                                    (double)m.v[i + 1], (double)m.v[i + 2]);
        s.append(line, (size_t)n);
    }
    for (size_t t = 0; t + 2 < m.f.size(); t += 3) {
        const int n = std::snprintf(line, sizeof line, "f %u %u %u\n", m.f[t] + 1, m.f[t + 1] + 1,
                                    m.f[t + 2] + 1);
        s.append(line, (size_t)n);
    }
    return s;
}

std::string ply(const ShapeMesh& m) {
    const size_t nv = m.v.size() / 3, nf = m.f.size() / 3;
    std::string s = "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(nv) +
                    "\nproperty float x\nproperty float y\nproperty float z\nelement face " +
                    std::to_string(nf) + "\nproperty list uchar int vertex_indices\nend_header\n";
    put(s, m.v.data(), m.v.size() * 4);
    for (size_t t = 0; t < nf; ++t) {
        put_v<uint8_t>(s, 3);
        for (int q = 0; q < 3; ++q) put_v<int32_t>(s, (int32_t)m.f[t * 3 + q]);
    }
    return s;
}

std::string stl(const ShapeMesh& m) {
    std::string s(80, '\0');
    std::memcpy(&s[0], "radiance", 8);
    const size_t nf = m.f.size() / 3;
    put_v<uint32_t>(s, (uint32_t)nf);
    for (size_t t = 0; t < nf; ++t) {
        const float* a = &m.v[(size_t)m.f[t * 3] * 3];
        const float* b = &m.v[(size_t)m.f[t * 3 + 1] * 3];
        const float* c = &m.v[(size_t)m.f[t * 3 + 2] * 3];
        const float u[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
        const float w[3] = { c[0] - a[0], c[1] - a[1], c[2] - a[2] };
        float x[3] = { u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                       u[0] * w[1] - u[1] * w[0] };
        const float l = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        if (l > 0.0f) for (float& q : x) q /= l;
        put(s, x, 12);
        put(s, a, 12);
        put(s, b, 12);
        put(s, c, 12);
        put_v<uint16_t>(s, 0);
    }
    return s;
}

}  /* namespace */

int shape_export(const ShapeMesh& m, const std::string& format, std::string* bytes,
                 std::string* mime, std::string* why) {
    if (format == "glb") { *bytes = glb(m); *mime = "model/gltf-binary"; return RAD_OK; }
    if (format == "obj") { *bytes = obj(m); *mime = "model/obj"; return RAD_OK; }
    if (format == "ply") { *bytes = ply(m); *mime = "application/x-ply"; return RAD_OK; }
    if (format == "stl") { *bytes = stl(m); *mime = "model/stl"; return RAD_OK; }
    if (why) *why = "type '" + format + "' is not one this server writes: glb, obj, ply or stl";
    return RAD_E_INVAL;
}

/* ================================================================== the ticket */
void ShapeTicket::progress(int32_t steps_done, int32_t points_done) {
    std::lock_guard<std::mutex> lk(mu_);
    steps_done_ = steps_done;
    points_done_ = points_done;
}

void ShapeTicket::set_field(std::vector<float>&& field, int32_t r) {
    std::lock_guard<std::mutex> lk(mu_);
    field_ = std::move(field);
    r_ = r;
}

void ShapeTicket::finish(int status, const std::string& why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (done_) return;
        done_ = true;
        status_ = status;
        why_ = why;
    }
    cv_.notify_all();
}

bool ShapeTicket::wait(int ms) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, std::chrono::milliseconds(ms), [this] { return done_; });
    return done_;
}

bool ShapeTicket::finished() const {
    std::lock_guard<std::mutex> lk(mu_);
    return done_;
}

int ShapeTicket::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
}

std::string ShapeTicket::why() const {
    std::lock_guard<std::mutex> lk(mu_);
    return why_;
}

std::vector<float> ShapeTicket::take_field(int32_t* r) {
    std::lock_guard<std::mutex> lk(mu_);
    if (r) *r = r_;
    return std::move(field_);
}

void ShapeTicket::progress_of(int32_t* steps_done, int32_t* points_done) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (steps_done) *steps_done = steps_done_;
    if (points_done) *points_done = points_done_;
}

}  /* namespace gen */
}  /* namespace rad */
