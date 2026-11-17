/* mp4.h -- a video and its soundtrack written as an MP4 file in memory: H.264 video in yuv420p
 * (BT.709, limited range) and AAC audio, the moov atom at the front so a player can start before
 * the whole file has arrived.
 *
 * The frames arrive in order, in any number of calls; the soundtrack may come before, between or
 * after them, whole or in pieces. Nothing is written until finish(), which encodes the soundtrack
 * and interleaves it with the video: a video is at most tens of seconds, and its frames are
 * encoded as they come so the RGB is never all held at once.
 *
 * The H.264 encoder is the best the linked libavcodec has: libx264 (constant quality), else
 * libopenh264, else whatever encoder it registers for H.264. Without libavcodec every call
 * returns RAD_E_UNSUPPORTED.
 */
#pragma once
#include "rad_core.h"

#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace mm {

class Mp4Writer {
public:
    Mp4Writer();
    ~Mp4Writer();
    Mp4Writer(const Mp4Writer&) = delete;
    Mp4Writer& operator=(const Mp4Writer&) = delete;

    /* A video of `width` x `height` (even) at fps_num / fps_den frames a second, with a soundtrack
     * of `channels` channels at `sample_rate` (0 channels: none). */
    int open(int32_t width, int32_t height, int64_t fps_num, int64_t fps_den, int64_t sample_rate,
             int64_t channels, std::string* why);
    /* `n` frames, each [height, width, 3] u8 RGB, after those already added. */
    int add_frames(const uint8_t* rgb, int32_t n, std::string* why);
    /* Samples interleaved over the channels, in [-1, 1], after those already added. */
    int add_audio(const float* samples, int64_t n_frames, std::string* why);
    /* Flushes both encoders and writes the file into `out`. */
    int finish(std::string* out, std::string* why);

    /* The H.264 encoder in use, for the log. */
    const std::string& video_encoder() const { return venc_name_; }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    std::string           venc_name_;
};

}  /* namespace mm */
}  /* namespace rad */
