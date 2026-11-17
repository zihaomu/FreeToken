/* resample.h -- a streaming resampler for mono float audio, between the rates a client sends
 * and the rate an audio encoder takes.
 *
 * A file is resampled by libswresample as it is decoded (media.cpp). A stream arrives a few
 * milliseconds at a time over a socket, and each piece must continue the last with no edge at the
 * seam: this keeps the input it still needs between calls, so a stream pushed in pieces comes out
 * sample for sample as it would if pushed whole.
 *
 * The filter is a Kaiser-windowed sinc evaluated at each output sample's position in the input: a
 * polyphase low-pass with one phase per distinct position, the rate step being a ratio of
 * integers. The cut-off sits under the lower of the two Nyquist frequencies, so a 24 kHz stream
 * taken to 16 kHz loses 8-12 kHz rather than folding it onto the band a speech model reads. Only
 * the common rates are taken: the phase table grows with the ratio's terms, and an arbitrary rate
 * can make it megabytes. */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rad {
namespace mm {

class Resampler {
public:
    /* Whether a stream at `rate` can be taken to a common rate: 8, 11.025, 12, 16, 22.05, 24,
     * 32, 44.1 or 48 kHz. */
    static bool takes(int32_t rate);
    /* The rates takes() accepts, for a refusal to name. */
    static std::string rates();

    /* From `from` Hz to `to` Hz, both rates takes() accepts. Equal rates pass samples through. */
    Resampler(int32_t from, int32_t to);

    /* Appends to `out` every output sample the input so far determines; the rest waits for more
     * input or flush(). */
    void push(const float* in, size_t n, std::vector<float>* out);
    /* Appends the remaining output, as if the input went on in silence, and starts afresh. */
    void flush(std::vector<float>* out);

private:
    void reset();

    int32_t up_ = 1, down_ = 1;   /* out/in as a ratio in lowest terms */
    int32_t half_ = 0;            /* taps either side of an output sample, in input samples */
    std::vector<float> taps_;     /* up_ phases of 2 * half_ taps */
    std::vector<float> in_;       /* input from index base_ on */
    int64_t base_ = 0;            /* the input index of in_[0]; negative while the history is the
                                   * silence before the stream */
    int64_t next_ = 0;            /* the next output sample's index */
    int64_t n_in_ = 0;            /* input samples pushed since the start */
};

}  /* namespace mm */
}  /* namespace rad */
