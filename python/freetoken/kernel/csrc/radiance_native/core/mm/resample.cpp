#include "mm/resample.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>

namespace rad {
namespace mm {

namespace {

const int32_t kRates[] = { 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000 };
/* The sinc's zero crossings either side of an output sample, at the lower of the two rates. */
constexpr double kLobes = 16.0;
/* The cut-off as a fraction of the lower Nyquist frequency, and the Kaiser window's shape: about
 * 80 dB down past the transition band, which leaves nothing a 16-bit stream could show. */
constexpr double kRolloff = 0.92;
constexpr double kBeta = 8.0;
/* Input the output has moved past is dropped once there is this much of it, not sample by
 * sample: dropping is a move of everything after it. */
constexpr int64_t kDropAt = 8192;

double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 64; ++k) {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        if (term < 1e-15 * sum) break;
    }
    return sum;
}

}  /* namespace */

bool Resampler::takes(int32_t rate) {
    return std::find(std::begin(kRates), std::end(kRates), rate) != std::end(kRates);
}

std::string Resampler::rates() {
    std::string s;
    for (int32_t r : kRates) {
        if (!s.empty()) s += ", ";
        s += std::to_string(r);
    }
    return s;
}

Resampler::Resampler(int32_t from, int32_t to) {
    const int32_t g = std::gcd(from, to);
    up_ = to / g;
    down_ = from / g;
    if (up_ != down_) {
        /* The cut-off in cycles an input sample, and the taps' reach: kLobes zero crossings of a
         * sinc that narrow, which is wider in input samples the further the rate comes down. */
        const double fc = 0.5 * kRolloff * std::min(1.0, (double)up_ / down_);
        half_ = (int32_t)std::ceil(kLobes / (2.0 * fc));
        taps_.assign((size_t)up_ * 2 * half_, 0.0f);
        const double i0b = bessel_i0(kBeta);
        for (int32_t p = 0; p < up_; ++p) {
            /* Phase p is an output sample p/up_ of an input sample past the input it is centred
             * on; tap k reads the input half_ - 1 - k before that one. */
            const double frac = (double)p / up_;
            float* h = &taps_[(size_t)p * 2 * half_];
            double sum = 0.0;
            for (int32_t k = 0; k < 2 * half_; ++k) {
                const double t = (double)(k - half_ + 1) - frac;
                const double x = 2.0 * fc * t;
                const double sinc = x == 0.0 ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
                const double r = t / half_;
                const double w = std::fabs(r) >= 1.0
                                     ? 0.0 : bessel_i0(kBeta * std::sqrt(1.0 - r * r)) / i0b;
                h[k] = (float)(sinc * w);
                sum += h[k];
            }
            /* Unity gain at DC, phase by phase, so a constant comes out the constant it went in. */
            for (int32_t k = 0; k < 2 * half_; ++k) h[k] = (float)(h[k] / sum);
        }
    }
    reset();
}

void Resampler::reset() {
    /* The history before the first sample is silence. */
    in_.assign((size_t)std::max(half_ - 1, 0), 0.0f);
    base_ = -(int64_t)in_.size();
    next_ = 0;
    n_in_ = 0;
}

void Resampler::push(const float* in, size_t n, std::vector<float>* out) {
    if (up_ == down_) {
        out->insert(out->end(), in, in + n);
        return;
    }
    in_.insert(in_.end(), in, in + n);
    n_in_ += (int64_t)n;
    for (;;) {
        const int64_t pos = next_ * down_;
        const int64_t c = pos / up_;
        if (c + half_ >= n_in_) break;
        const float* h = &taps_[(size_t)(pos % up_) * 2 * half_];
        const float* x = &in_[(size_t)(c - half_ + 1 - base_)];
        float acc = 0.0f;
        for (int32_t k = 0; k < 2 * half_; ++k) acc += x[k] * h[k];
        out->push_back(acc);
        ++next_;
    }
    const int64_t keep = (next_ * down_) / up_ - half_ + 1;
    if (keep - base_ >= kDropAt) {
        in_.erase(in_.begin(), in_.begin() + (std::ptrdiff_t)(keep - base_));
        base_ = keep;
    }
}

void Resampler::flush(std::vector<float>* out) {
    if (up_ != down_) {
        /* Every output sample the input spans, the last ones read against silence. */
        const int64_t total = (n_in_ * up_ + down_ - 1) / down_;
        const std::vector<float> silence((size_t)(2 * half_ + down_), 0.0f);
        std::vector<float> rest;
        push(silence.data(), silence.size(), &rest);
        const int64_t want = total - (next_ - (int64_t)rest.size());
        rest.resize((size_t)std::max<int64_t>(0, std::min<int64_t>(want, (int64_t)rest.size())));
        out->insert(out->end(), rest.begin(), rest.end());
    }
    reset();
}

}  /* namespace mm */
}  /* namespace rad */
