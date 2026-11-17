/* media.h -- image, video and audio decoding, and picture encoding (spec §11).
 *
 * One decoder for everything: libavformat/libavcodec. An image is a one-frame stream, so PNG,
 * JPEG, WebP, GIF, BMP and TIFF take the same demuxer probe, decode loop and colour conversion
 * (libswscale) as a video. In a build without the libraries every call returns
 * RAD_E_UNSUPPORTED with the reason, and the server refuses the media part by name.
 *
 * Bytes come from the request only (`data:` URLs). Nothing here opens a file or a network
 * stream: the demuxer reads a memory buffer through its own I/O context, and the protocol
 * whitelist is empty.
 *
 * A Frame is packed 8-bit RGB, row-major, no padding. Alpha is dropped, not composited, as the
 * reference processor's PIL convert("RGB") does, so a transparent PNG reaches the encoder with
 * the colours under its alpha.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rad {
namespace mm {

struct Frame {
    int32_t              w = 0, h = 0;
    std::vector<uint8_t> rgb;          /* [h, w, 3] */
};

/* Packed 8-bit RGBA, row-major, no padding; what an image generator reads and writes. A source
 * without alpha decodes as opaque. */
struct FrameRGBA {
    int32_t              w = 0, h = 0;
    std::vector<uint8_t> rgba;         /* [h, w, 4] */
};

/* Whether this build has a decoder. */
bool decoder_available();

/* "libavformat <version>" when the decoder is built in, "absent (built without libav*)" when it
 * is not: one string for the startup report and /diagnostics, so a build that refuses every media
 * part shows it there. */
std::string decoder_version();

/* The first frame of an image (or of anything with a video stream). Refuses a source over
 * `max_pixels`, which bounds what a request can make the server allocate before any resize. */
int decode_image(const uint8_t* data, size_t n, int64_t max_pixels, Frame* out, std::string* why);
/* As decode_image, keeping the alpha channel. */
int decode_image_rgba(const uint8_t* data, size_t n, int64_t max_pixels, FrameRGBA* out,
                      std::string* why);

/* Encodes `px`, [h, w, ch] 8-bit with ch 3 (RGB) or 4 (RGBA), as an image file; `format` is png,
 * jpeg or webp. `quality` 0..100 applies to jpeg and lossy webp (0 or above 100 picks the
 * format's default); png is lossless and webp is lossless at 100. jpeg refuses alpha. Returns
 * RAD_E_UNSUPPORTED, with the reason, for a format this build cannot encode.
 *
 * Lossless webp is written by libwebp; FFmpeg's native VP8L decoder refuses some valid
 * libwebp streams ("Invalid data", reproducible with ffmpeg 8.0.1 on a low-colour picture),
 * so an FFmpeg-based consumer may not read every lossless file this writes. libwebp and
 * browsers do. */
int encode_image(const uint8_t* px, int32_t w, int32_t h, int32_t ch, const std::string& format,
                 int32_t quality, std::vector<uint8_t>* out, std::string* why);

/* Video sampling happens inside the decode: a minute of 1080p is 10 GB of RGB, so unsampled
 * frames are never converted or kept. The processor picks the frames from the stream's length
 * and rate, which only the decoder knows, so the choice is a callback: given the frame count and
 * rate, return the frame indices to keep, ascending. */
struct VideoInfo {
    int64_t n_frames = 0;    /* frames in the stream; counted by demuxing when the header omits it */
    double  fps = 0.0;       /* the stream's average rate */
    int32_t w = 0, h = 0;
};
typedef std::vector<int64_t> (*SampleFn)(const VideoInfo& info, const void* user);

struct Video {
    VideoInfo            info;
    std::vector<int64_t> index;        /* the frame each of `frames` is */
    std::vector<Frame>   frames;
};

int decode_video(const uint8_t* data, size_t n, SampleFn sample, const void* user,
                 int64_t max_pixels, Video* out, std::string* why);

/* The first audio stream as mono float samples at `rate` Hz. Resampled with libswresample at the
 * source's channel layout, then averaged across channels: the plain mean reference loaders
 * (librosa, torchaudio's to-mono) take, where swresample's downmix weights a stereo pair by
 * 0.707. A clip longer than `max_samples` is truncated, as the reference feature extractor does. */
int decode_audio(const uint8_t* data, size_t n, int32_t rate, int64_t max_samples,
                 std::vector<float>* out, std::string* why);

/* `in` encoded as a one-frame H.264 stream at constant quality `crf` (libx264, preset veryfast,
 * yuv420p) and decoded back, as a video model's reference pipeline does to a conditioning
 * picture so it carries the compression of the footage the model was trained on. An odd width
 * or height loses its last column or row first. Needs an H.264 encoder: libx264, else libopenh264. */
int h264_roundtrip(const Frame& in, int32_t crf, Frame* out, std::string* why);
/* Whether this build can run h264_roundtrip: its libavcodec has an H.264 encoder and decoder. */
bool h264_roundtrip_available();

}  /* namespace mm */
}  /* namespace rad */
