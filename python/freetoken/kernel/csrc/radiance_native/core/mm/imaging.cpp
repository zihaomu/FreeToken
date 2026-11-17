/* imaging.cpp -- PIL's Lanczos resize and alpha paste (imaging.h). The constants and the order of
 * every operation are Pillow's (src/libImaging/Resample.c, Convert.c, Paste.c). */
#include "mm/imaging.h"
#include "util/rowpool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

namespace rad {
namespace mm {

namespace {

constexpr int kPrecision = 32 - 8 - 2;

double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= M_PI;
    return std::sin(x) / x;
}

double lanczos(double x) { return (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3.0) : 0.0; }

/* PIL's BILINEAR filter: a triangle of support 1. */
double triangle(double x) {
    if (x < 0.0) x = -x;
    return x < 1.0 ? 1.0 - x : 0.0;
}

/* One axis: for each output index its first input index, its tap count, and `ksize` fixed-point
 * coefficients. */
struct Axis {
    int ksize = 0;
    std::vector<int> bounds;   /* [out, 2]: xmin, count */
    std::vector<int> kk;       /* [out, ksize] */
};

Axis coeffs(int in_size, int out_size, bool bilinear) {
    Axis a;
    const double scale = (double)in_size / (double)out_size;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = (bilinear ? 1.0 : 3.0) * filterscale;
    a.ksize = (int)std::ceil(support) * 2 + 1;
    a.bounds.assign((size_t)out_size * 2, 0);
    a.kk.assign((size_t)out_size * a.ksize, 0);
    std::vector<double> k((size_t)a.ksize);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        double ww = 0.0;
        const double ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        int x = 0;
        for (; x < xmax; ++x) {
            const double t = (x + xmin - center + 0.5) * ss;
            const double w = bilinear ? triangle(t) : lanczos(t);
            k[(size_t)x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x)
            if (ww != 0.0) k[(size_t)x] /= ww;
        for (; x < a.ksize; ++x) k[(size_t)x] = 0.0;
        a.bounds[(size_t)xx * 2] = xmin;
        a.bounds[(size_t)xx * 2 + 1] = xmax;
        for (x = 0; x < a.ksize; ++x) {
            const double v = k[(size_t)x] * (double)(1 << kPrecision);
            a.kk[(size_t)xx * a.ksize + x] = v < 0 ? (int)(-0.5 + v) : (int)(0.5 + v);
        }
    }
    return a;
}

inline uint8_t clip8(int v) {
    if (v >= (1 << kPrecision << 8)) return 255;
    if (v <= 0) return 0;
    return (uint8_t)(v >> kPrecision);
}

inline unsigned muldiv255(unsigned a, unsigned b) {
    const unsigned t = a * b + 128;
    return ((t >> 8) + t) >> 8;
}

/* Every pass below is row-independent: an output row reads only its own input rows and
 * writes only itself, each pixel accumulated in Pillow's tap order whichever thread runs
 * it, so splitting rows across threads is bit-exact (processor.cpp's run_rows states the
 * same for its resize, and mm_test's Pillow goldens pin this one's bytes). Rows under the
 * threshold run inline: handing them out costs more than the row. */
void run_rows(int rows, const std::function<void(int)>& row) {
    row_pool_run(rows, 8, [&row](int64_t y) { row((int)y); }, 32);
}

}  /* namespace */

static void resize_rgba(const FrameRGBA& src, int32_t w, int32_t h, bool bilinear, FrameRGBA* out);

void resize_lanczos_rgba(const FrameRGBA& src, int32_t w, int32_t h, FrameRGBA* out) {
    resize_rgba(src, w, h, false, out);
}

void resize_bilinear_rgba(const FrameRGBA& src, int32_t w, int32_t h, FrameRGBA* out) {
    resize_rgba(src, w, h, true, out);
}

static void resize_rgba(const FrameRGBA& src, int32_t w, int32_t h, bool bilinear, FrameRGBA* out) {
    const int iw = src.w, ih = src.h;
    /* Image.resize returns a copy at the same size. The premultiplied round trip below is not
     * lossless where alpha is below 255. */
    if (w == iw && h == ih) {
        *out = src;
        return;
    }
    /* RGBA -> RGBa: the colour premultiplied by the alpha */
    std::vector<uint8_t> pre((size_t)iw * ih * 4);
    run_rows(ih, [&](int y) {
        for (int x = 0; x < iw; ++x) {
            const size_t i = (size_t)y * iw + x;
            const unsigned al = src.rgba[i * 4 + 3];
            for (int c = 0; c < 3; ++c) pre[i * 4 + c] = (uint8_t)muldiv255(src.rgba[i * 4 + c], al);
            pre[i * 4 + 3] = (uint8_t)al;
        }
    });
    const bool horiz = w != iw, vert = h != ih;
    const Axis ax = coeffs(iw, w, bilinear), ay = coeffs(ih, h, bilinear);
    /* the rows the vertical pass reads */
    const int y_first = ay.bounds[0];
    const int y_last = ay.bounds[(size_t)(h - 1) * 2] + ay.bounds[(size_t)(h - 1) * 2 + 1];
    std::vector<uint8_t> tmp;
    const uint8_t* rows = pre.data();
    int rw = iw, row0 = 0;
    if (horiz) {
        const int th = vert ? y_last - y_first : ih;
        const int off = vert ? y_first : 0;
        tmp.assign((size_t)w * th * 4, 0);
        run_rows(th, [&](int yy) {
            for (int xx = 0; xx < w; ++xx) {
                const int xmin = ax.bounds[(size_t)xx * 2], xn = ax.bounds[(size_t)xx * 2 + 1];
                const int* k = &ax.kk[(size_t)xx * ax.ksize];
                int s[4] = { 1 << (kPrecision - 1), 1 << (kPrecision - 1), 1 << (kPrecision - 1),
                             1 << (kPrecision - 1) };
                const uint8_t* p = pre.data() + ((size_t)(yy + off) * iw + xmin) * 4;
                for (int x = 0; x < xn; ++x)
                    for (int c = 0; c < 4; ++c) s[c] += (int)p[(size_t)x * 4 + c] * k[x];
                for (int c = 0; c < 4; ++c) tmp[((size_t)yy * w + xx) * 4 + c] = clip8(s[c]);
            }
        });
        rows = tmp.data();
        rw = w;
        row0 = vert ? y_first : 0;
    }
    std::vector<uint8_t> res;
    if (vert) {
        res.assign((size_t)rw * h * 4, 0);
        run_rows(h, [&](int yy) {
            const int ymin = ay.bounds[(size_t)yy * 2] - row0, yn = ay.bounds[(size_t)yy * 2 + 1];
            const int* k = &ay.kk[(size_t)yy * ay.ksize];
            for (int xx = 0; xx < rw; ++xx) {
                int s[4] = { 1 << (kPrecision - 1), 1 << (kPrecision - 1), 1 << (kPrecision - 1),
                             1 << (kPrecision - 1) };
                for (int y = 0; y < yn; ++y) {
                    const uint8_t* p = rows + ((size_t)(y + ymin) * rw + xx) * 4;
                    for (int c = 0; c < 4; ++c) s[c] += (int)p[c] * k[y];
                }
                for (int c = 0; c < 4; ++c) res[((size_t)yy * rw + xx) * 4 + c] = clip8(s[c]);
            }
        });
    } else {
        res.assign(rows, rows + (size_t)rw * (horiz ? h : ih) * 4);
    }
    /* RGBa -> RGBA */
    out->w = w;
    out->h = h;
    out->rgba.resize((size_t)w * h * 4);
    run_rows(h, [&](int y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = (size_t)y * w + x;
            const unsigned al = res[i * 4 + 3];
            for (int c = 0; c < 3; ++c) {
                const unsigned v = res[i * 4 + c];
                out->rgba[i * 4 + c] =
                    (al == 255 || al == 0) ? (uint8_t)v : (uint8_t)std::min(255u, (255u * v) / al);
            }
            out->rgba[i * 4 + 3] = (uint8_t)al;
        }
    });
}

void resize_lanczos_rgb(const Frame& src, int32_t w, int32_t h, Frame* out) {
    /* At alpha 255 the premultiply and the divide back are the identity. */
    FrameRGBA pre;
    pre.w = src.w;
    pre.h = src.h;
    pre.rgba.resize((size_t)src.w * src.h * 4);
    for (size_t i = 0; i < (size_t)src.w * src.h; ++i) {
        for (int c = 0; c < 3; ++c) pre.rgba[i * 4 + c] = src.rgb[i * 3 + c];
        pre.rgba[i * 4 + 3] = 255;
    }
    FrameRGBA post;
    resize_lanczos_rgba(pre, w, h, &post);
    out->w = post.w;
    out->h = post.h;
    out->rgb.resize((size_t)post.w * post.h * 3);
    for (size_t i = 0; i < (size_t)post.w * post.h; ++i)
        for (int c = 0; c < 3; ++c) out->rgb[i * 3 + c] = post.rgba[i * 4 + c];
}

void composite_white(const FrameRGBA& src, Frame* out) {
    out->w = src.w;
    out->h = src.h;
    out->rgb.resize((size_t)src.w * src.h * 3);
    for (size_t i = 0; i < (size_t)src.w * src.h; ++i) {
        const unsigned a = src.rgba[i * 4 + 3];
        for (int c = 0; c < 3; ++c) {
            const unsigned t = src.rgba[i * 4 + c] * a + 255u * (255u - a) + 128u;
            out->rgb[i * 3 + c] = (uint8_t)(((t >> 8) + t) >> 8);
        }
    }
}

}  /* namespace mm */
}  /* namespace rad */
