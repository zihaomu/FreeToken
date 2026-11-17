/* media.cpp -- image, video and audio decoding and image encoding through libavformat,
 * libavcodec, libswscale and libswresample. */
#include "mm/media.h"

#include "rad_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#ifdef RAD_HAVE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

namespace rad {
namespace mm {

#ifndef RAD_HAVE_FFMPEG

bool decoder_available() { return false; }

std::string decoder_version() { return "absent (built without libav*)"; }

static int no_decoder(std::string* why) {
    if (why) *why = "this build has no media decoder: it was configured without libavformat / "
                    "libavcodec / libswscale (RAD_WITH_FFMPEG)";
    return RAD_E_UNSUPPORTED;
}

int decode_image(const uint8_t*, size_t, int64_t, Frame*, std::string* why) {
    return no_decoder(why);
}
int decode_image_rgba(const uint8_t*, size_t, int64_t, FrameRGBA*, std::string* why) {
    return no_decoder(why);
}
int encode_image(const uint8_t*, int32_t, int32_t, int32_t, const std::string&, int32_t,
                 std::vector<uint8_t>*, std::string* why) {
    return no_decoder(why);
}
int decode_video(const uint8_t*, size_t, SampleFn, const void*, int64_t, Video*,
                 std::string* why) {
    return no_decoder(why);
}
int h264_roundtrip(const Frame&, int32_t, Frame*, std::string* why) { return no_decoder(why); }
bool h264_roundtrip_available() { return false; }
int decode_audio(const uint8_t*, size_t, int32_t, int64_t, std::vector<float>*,
                 std::string* why) {
    return no_decoder(why);
}

#else

bool decoder_available() { return true; }

std::string decoder_version() {
    const unsigned v = avformat_version();
    char b[48];
    std::snprintf(b, sizeof(b), "libavformat %u.%u.%u", AV_VERSION_MAJOR(v), AV_VERSION_MINOR(v),
                  AV_VERSION_MICRO(v));
    return b;
}

namespace {

/* The only demuxers a request may reach. A format that can name another file (a playlist, a
 * concat script, an image-sequence pattern) would let a request make the server open a path of
 * its choosing. The empty protocol whitelist also stops these from opening anything. */
constexpr const char* kFormats =
    "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts,flv,ogg,gif,apng,image2pipe,png_pipe,"
    "jpeg_pipe,jpegls_pipe,jpegxl_pipe,webp_pipe,bmp_pipe,tiff_pipe,qoi_pipe,pgm_pipe,ppm_pipe,"
    "pbm_pipe,pam_pipe,gif_pipe,avif,heif";

/* The size a PNG or JPEG states in its header, read from the bytes: such a picture needs no
 * stream probe. avformat_find_stream_info decodes the whole picture once to learn its size and the
 * decode then runs it again (a 1024x1024 PNG 8.8 -> 19.9 ms). False for any other format, or a
 * header it cannot read, which are probed as before. */
uint32_t be16(const uint8_t* p) { return (uint32_t)p[0] << 8 | p[1]; }
uint32_t be32(const uint8_t* p) { return be16(p) << 16 | be16(p + 2); }

bool still_size(const AVInputFormat* f, const uint8_t* p, size_t n, int* w, int* h) {
    if (!f || !f->name) return false;
    if (std::strcmp(f->name, "png_pipe") == 0) {
        /* Signature, then IHDR first: length 13, type, width, height. */
        static const uint8_t kSig[8] = { 0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a };
        if (n < 24 || std::memcmp(p, kSig, 8) != 0 || be32(p + 8) != 13 ||
            std::memcmp(p + 12, "IHDR", 4) != 0)
            return false;
        const uint32_t pw = be32(p + 16), ph = be32(p + 20);
        if (pw == 0 || ph == 0 || pw > INT32_MAX || ph > INT32_MAX) return false;
        *w = (int)pw;
        *h = (int)ph;
        return true;
    }
    if (std::strcmp(f->name, "jpeg_pipe") == 0) {
        /* Segments from SOI to the first start of frame (SOF0..SOF15 bar DHT, JPG and DAC):
         * precision, then height, then width. */
        if (n < 4 || p[0] != 0xff || p[1] != 0xd8) return false;
        size_t at = 2;
        while (at + 4 <= n) {
            if (p[at] != 0xff) return false;
            const uint8_t m = p[at + 1];
            if (m == 0xff) { ++at; continue; }                       /* fill byte */
            if (m == 0x01 || (m >= 0xd0 && m <= 0xd7)) { at += 2; continue; }   /* no length */
            if (m == 0xd9 || m == 0xda) return false;               /* EOI or scan before a frame */
            const uint32_t len = be16(p + at + 2);
            if (len < 2) return false;
            if (m >= 0xc0 && m <= 0xcf && m != 0xc4 && m != 0xc8 && m != 0xcc) {
                if (len < 7 || at + 9 > n) return false;
                const uint32_t ph = be16(p + at + 5), pw = be16(p + at + 7);
                if (pw == 0 || ph == 0) return false;
                *w = (int)pw;
                *h = (int)ph;
                return true;
            }
            at += 2 + len;
        }
        return false;
    }
    return false;
}

struct MemIO {
    const uint8_t* p = nullptr;
    size_t         n = 0;
    size_t         at = 0;
};

int mem_read(void* opaque, uint8_t* buf, int size) {
    MemIO* m = static_cast<MemIO*>(opaque);
    const size_t left = m->n - m->at;
    if (left == 0) return AVERROR_EOF;
    const size_t k = std::min(left, (size_t)size);
    std::memcpy(buf, m->p + m->at, k);
    m->at += k;
    return (int)k;
}

int64_t mem_seek(void* opaque, int64_t off, int whence) {
    MemIO* m = static_cast<MemIO*>(opaque);
    if (whence & AVSEEK_SIZE) return (int64_t)m->n;
    int64_t base = 0;
    switch (whence & ~AVSEEK_FORCE) {
        case SEEK_SET: base = 0; break;
        case SEEK_CUR: base = (int64_t)m->at; break;
        case SEEK_END: base = (int64_t)m->n; break;
        default: return AVERROR(EINVAL);
    }
    const int64_t to = base + off;
    if (to < 0 || to > (int64_t)m->n) return AVERROR(EINVAL);
    m->at = (size_t)to;
    return to;
}

std::string averr(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(e, b, sizeof b);
    return b;
}

/* One opened input over a memory buffer: the demuxer, its best video stream and a decoder for it.
 * Every libav object is released in the destructor, so an early return anywhere leaks nothing. */
struct Input {
    MemIO            io;
    AVIOContext*     avio = nullptr;
    AVFormatContext* fmt = nullptr;
    AVCodecContext*  dec = nullptr;
    int              stream = -1;
    /* A PNG or JPEG opened for decoding: the stream was not probed, and these are the size its
     * header states (still_size). */
    bool             still = false;
    int              still_w = 0, still_h = 0;

    ~Input() {
        if (dec) avcodec_free_context(&dec);
        if (fmt) avformat_close_input(&fmt);
        if (avio) {
            av_freep(&avio->buffer);
            avio_context_free(&avio);
        }
    }

    /* `max_pixels` > 0 makes the decoder refuse a larger picture itself. */
    int open(const uint8_t* data, size_t n, bool want_decoder, std::string* why,
             int64_t max_pixels = 0) {
        io.p = data; io.n = n; io.at = 0;
        constexpr int kBuf = 64 * 1024;
        uint8_t* buf = static_cast<uint8_t*>(av_malloc(kBuf));
        if (!buf) { *why = "out of memory"; return RAD_E_NOMEM; }
        avio = avio_alloc_context(buf, kBuf, 0, &io, mem_read, nullptr, mem_seek);
        if (!avio) { av_free(buf); *why = "out of memory"; return RAD_E_NOMEM; }
        fmt = avformat_alloc_context();
        if (!fmt) { *why = "out of memory"; return RAD_E_NOMEM; }
        fmt->pb = avio;
        fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "format_whitelist", kFormats, 0);
        av_dict_set(&opts, "protocol_whitelist", "", 0);
        int e = avformat_open_input(&fmt, nullptr, nullptr, &opts);
        av_dict_free(&opts);
        if (e < 0) {
            fmt = nullptr;   /* freed by a failed open */
            *why = "not a recognised image or video format (" + averr(e) + ")";
            return RAD_E_INVAL;
        }
        still = want_decoder && still_size(fmt->iformat, data, n, &still_w, &still_h);
        if (!still) {
            e = avformat_find_stream_info(fmt, nullptr);
            if (e < 0) { *why = "could not read the stream headers (" + averr(e) + ")"; return RAD_E_INVAL; }
        }
        const AVCodec* codec = nullptr;
        stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
        if (stream < 0 || !codec) { *why = "no picture in it: the media holds no video stream"; return RAD_E_INVAL; }
        if (!want_decoder) return RAD_OK;
        dec = avcodec_alloc_context3(codec);
        if (!dec) { *why = "out of memory"; return RAD_E_NOMEM; }
        e = avcodec_parameters_to_context(dec, fmt->streams[stream]->codecpar);
        if (e < 0) { *why = "decoder setup failed (" + averr(e) + ")"; return RAD_E_INVAL; }
        /* Decodes run on server workers, one per request in flight; a few threads each avoids
         * oversubscribing the machine. One picture gains nothing from frame threads, whose start
         * and exit are all they would add. */
        dec->thread_count = still ? 1 : 4;
        if (max_pixels > 0) dec->max_pixels = max_pixels;
        e = avcodec_open2(dec, codec, nullptr);
        if (e < 0) { *why = std::string("no decoder for ") + codec->name + " (" + averr(e) + ")"; return RAD_E_UNSUPPORTED; }
        return RAD_OK;
    }
};

/* One decoded frame to packed `fmt` (RGB24 or RGBA, `ch` bytes a pixel) at its own size, rows
 * tight. The YUV coefficients and range are the frame's own, so a BT.709 video and a full-range
 * JPEG keep the colours they were encoded with.
 *
 * The scaler flags match the reference pipeline: transformers decodes video through PyAV
 * (`to_ndarray(format="rgb24")`), which converts with plain SWS_BILINEAR. The flags decide chroma
 * upsampling, which can differ by up to 70 levels at a sharp colour edge; with these flags every
 * patch lands within one level of the reference. */
int to_packed(const AVFrame* f, SwsContext** sws, AVPixelFormat fmt, size_t ch,
              std::vector<uint8_t>* out, std::string* why) {
    const AVPixelFormat src = (AVPixelFormat)f->format;
    *sws = sws_getCachedContext(*sws, f->width, f->height, src, f->width, f->height, fmt,
                                SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!*sws) { *why = "no colour conversion from this pixel format"; return RAD_E_UNSUPPORTED; }
    int cs = f->colorspace;
    if (cs == AVCOL_SPC_UNSPECIFIED || cs == AVCOL_SPC_RGB) cs = AVCOL_SPC_BT470BG;
    const int src_full = f->color_range == AVCOL_RANGE_JPEG ||
                         src == AV_PIX_FMT_YUVJ420P || src == AV_PIX_FMT_YUVJ422P ||
                         src == AV_PIX_FMT_YUVJ444P || src == AV_PIX_FMT_YUVJ440P;
    sws_setColorspaceDetails(*sws, sws_getCoefficients(cs), src_full,
                             sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16, 1 << 16);
    /* Convert into an FFmpeg-allocated frame, then copy the rows out. The scaler's SIMD paths
     * store whole vectors and overrun a tight row (a 4:2:0 or 4:2:2 picture 1788 wide writes
     * 12 bytes past 1788 x 3), which corrupts the heap in a buffer of exactly w x h x 3.
     * av_frame_get_buffer pads every line and the end as the scaler expects. */
    AVFrame* dst = av_frame_alloc();
    if (!dst) { *why = "out of memory"; return RAD_E_NOMEM; }
    dst->format = fmt;
    dst->width = f->width;
    dst->height = f->height;
    if (av_frame_get_buffer(dst, 0) < 0) {
        av_frame_free(&dst);
        *why = "out of memory";
        return RAD_E_NOMEM;
    }
    const int e = sws_scale(*sws, f->data, f->linesize, 0, f->height, dst->data, dst->linesize);
    if (e != f->height) {
        av_frame_free(&dst);
        *why = "colour conversion failed";
        return RAD_E_INVAL;
    }
    const size_t row = (size_t)f->width * ch;
    out->resize(row * (size_t)f->height);
    for (int y = 0; y < f->height; ++y)
        std::memcpy(out->data() + (size_t)y * row, dst->data[0] + (ptrdiff_t)y * dst->linesize[0],
                    row);
    av_frame_free(&dst);
    return RAD_OK;
}

int to_rgb(const AVFrame* f, SwsContext** sws, Frame* out, std::string* why) {
    out->w = f->width;
    out->h = f->height;
    return to_packed(f, sws, AV_PIX_FMT_RGB24, 3, &out->rgb, why);
}

/* As to_rgb, to packed RGBA, for a picture whose alpha is kept. */
int to_rgba(const AVFrame* f, SwsContext** sws, FrameRGBA* out, std::string* why) {
    out->w = f->width;
    out->h = f->height;
    return to_packed(f, sws, AV_PIX_FMT_RGBA, 4, &out->rgba, why);
}

/* The decode loop for images and video: every frame in presentation order goes to
 * `keep(i, frame)`, which returns < 0 to fail, > 0 to stop and 0 to continue.
 *
 * An empty packet repeats the previous frame (Ogg Theora stores an unchanged frame this way).
 * The decoder would read it as end of stream, so it is passed to `keep` as another copy of the
 * last frame instead, keeping indices and length consistent with the stated frame count. */
template <class Keep>
int decode_frames(Input& in, Keep&& keep, std::string* why) {
    AVPacket* pkt = av_packet_alloc();
    AVFrame*  frm = av_frame_alloc();
    AVFrame*  last = av_frame_alloc();
    if (!pkt || !frm || !last) {
        av_packet_free(&pkt);
        av_frame_free(&frm);
        av_frame_free(&last);
        *why = "out of memory";
        return RAD_E_NOMEM;
    }
    int64_t idx = 0;
    int st = RAD_OK;
    bool done = false, flushing = false;
    while (!done && st >= 0) {
        if (!flushing) {
            const int r = av_read_frame(in.fmt, pkt);
            if (r == AVERROR_EOF) {
                flushing = true;
                avcodec_send_packet(in.dec, nullptr);
            } else if (r < 0) {
                *why = "demuxing failed (" + averr(r) + ")";
                st = RAD_E_INVAL;
                break;
            } else {
                if (pkt->stream_index != in.stream) { av_packet_unref(pkt); continue; }
                if (pkt->size == 0) {
                    av_packet_unref(pkt);
                    if (!last->buf[0]) continue;
                    const int k = keep(idx++, last);
                    if (k < 0) { st = k; break; }
                    if (k > 0) done = true;
                    continue;
                }
                const int s = avcodec_send_packet(in.dec, pkt);
                av_packet_unref(pkt);
                if (s < 0 && s != AVERROR(EAGAIN)) {
                    *why = "decoding failed (" + averr(s) + ")";
                    st = RAD_E_INVAL;
                    break;
                }
            }
        }
        for (;;) {
            const int r = avcodec_receive_frame(in.dec, frm);
            if (r == AVERROR(EAGAIN)) break;
            if (r == AVERROR_EOF) { done = true; break; }
            if (r < 0) {
                *why = "decoding failed (" + averr(r) + ")";
                st = RAD_E_INVAL;
                break;
            }
            const int k = keep(idx++, frm);
            av_frame_unref(last);
            av_frame_move_ref(last, frm);
            if (k < 0) { st = k; break; }
            if (k > 0) { done = true; break; }
        }
    }
    av_packet_free(&pkt);
    av_frame_free(&frm);
    av_frame_free(&last);
    return st;
}

}  /* namespace */

int decode_image(const uint8_t* data, size_t n, int64_t max_pixels, Frame* out, std::string* why) {
    std::string w;
    if (!why) why = &w;
    if (!data || n == 0) { *why = "empty"; return RAD_E_INVAL; }
    Input in;
    RAD_TRY(in.open(data, n, true, why, max_pixels));
    const AVCodecParameters* cp = in.fmt->streams[in.stream]->codecpar;
    const int hw = in.still ? in.still_w : cp->width, hh = in.still ? in.still_h : cp->height;
    if (hw <= 0 || hh <= 0) { *why = "the image has no size"; return RAD_E_INVAL; }
    if ((int64_t)hw * hh > max_pixels) {
        *why = "the image is " + std::to_string(hw) + "x" + std::to_string(hh) +
               ", over the server's " + std::to_string(max_pixels) + "-pixel limit for a source";
        return RAD_E_INVAL;
    }
    SwsContext* sws = nullptr;
    bool got = false;
    const int s = decode_frames(in, [&](int64_t, const AVFrame* f) -> int {
        /* The decoded frame, not the header's size, is what gets converted: a container whose
         * bitstream disagrees with its declared dimensions must not slip past the cap. */
        if ((int64_t)f->width * f->height > max_pixels) {
            *why = "the decoded image is larger than the server's pixel limit";
            return RAD_E_INVAL;
        }
        const int c = to_rgb(f, &sws, out, why);
        if (c < 0) return c;
        got = true;
        return 1;
    }, why);
    sws_freeContext(sws);
    if (s < 0) return s;
    if (!got) { *why = "the image decoded to no frame"; return RAD_E_INVAL; }
    return RAD_OK;
}

int decode_image_rgba(const uint8_t* data, size_t n, int64_t max_pixels, FrameRGBA* out,
                      std::string* why) {
    std::string w;
    if (!why) why = &w;
    if (!data || n == 0) { *why = "empty"; return RAD_E_INVAL; }
    Input in;
    RAD_TRY(in.open(data, n, true, why, max_pixels));
    const AVCodecParameters* cp = in.fmt->streams[in.stream]->codecpar;
    const int hw = in.still ? in.still_w : cp->width, hh = in.still ? in.still_h : cp->height;
    if (hw <= 0 || hh <= 0) { *why = "the image has no size"; return RAD_E_INVAL; }
    if ((int64_t)hw * hh > max_pixels) {
        *why = "the image is " + std::to_string(hw) + "x" + std::to_string(hh) +
               ", over the server's " + std::to_string(max_pixels) + "-pixel limit for a source";
        return RAD_E_INVAL;
    }
    SwsContext* sws = nullptr;
    bool got = false;
    const int s = decode_frames(in, [&](int64_t, const AVFrame* f) -> int {
        /* Re-check the decoded frame against the cap, as decode_image and the video path do:
         * the bitstream, not the header, decides what the converter allocates. */
        if ((int64_t)f->width * f->height > max_pixels) {
            *why = "the decoded image is larger than the server's pixel limit";
            return RAD_E_INVAL;
        }
        const int c = to_rgba(f, &sws, out, why);
        if (c < 0) return c;
        got = true;
        return 1;
    }, why);
    sws_freeContext(sws);
    if (s < 0) return s;
    if (!got) { *why = "the image decoded to no frame"; return RAD_E_INVAL; }
    return RAD_OK;
}

/* One picture through one encoder: convert to the encoder's pixel format (swscale; full-range
 * BT.601 for jpeg's YUV, as PIL writes it) and send one frame. png, mjpeg and libwebp each write
 * a complete image file per packet, so the packet that comes back is the file. */
int encode_image(const uint8_t* px, int32_t w, int32_t h, int32_t ch, const std::string& format,
                 int32_t quality, std::vector<uint8_t>* out, std::string* why) {
    std::string wy;
    if (!why) why = &wy;
    if (!px || w <= 0 || h <= 0 || (ch != 3 && ch != 4) || !out) {
        *why = "no picture to encode";
        return RAD_E_INVAL;
    }
    const AVCodec* codec = nullptr;
    AVPixelFormat pf = AV_PIX_FMT_NONE;
    if (format == "png") {
        codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
        pf = ch == 4 ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24;
    } else if (format == "jpeg") {
        if (ch == 4) { *why = "jpeg carries no alpha"; return RAD_E_INVAL; }
        codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
        pf = AV_PIX_FMT_YUVJ420P;
    } else if (format == "webp") {
        codec = avcodec_find_encoder_by_name("libwebp");
        pf = AV_PIX_FMT_BGRA;
    } else {
        *why = "'" + format + "' is not an image format this server writes (png, jpeg, webp)";
        return RAD_E_INVAL;
    }
    if (!codec) {
        *why = "this build has no " + format + " encoder";
        return RAD_E_UNSUPPORTED;
    }
    AVCodecContext* cx = avcodec_alloc_context3(codec);
    AVFrame* fr = av_frame_alloc();
    AVPacket* pk = av_packet_alloc();
    SwsContext* sws = nullptr;
    int st = RAD_OK;
    auto fail = [&](const std::string& m, int code) { *why = m; st = code; };
    do {
        if (!cx || !fr || !pk) { fail("out of memory", RAD_E_NOMEM); break; }
        cx->width = w;
        cx->height = h;
        cx->pix_fmt = pf;
        cx->time_base = AVRational{ 1, 25 };
        cx->color_range = format == "jpeg" ? AVCOL_RANGE_JPEG : AVCOL_RANGE_UNSPECIFIED;
        if (format == "jpeg") {
            const int q = quality > 0 && quality <= 100 ? quality : 90;
            /* qscale 2 (best) .. 31 (worst), linear in the quality */
            const int qs = 2 + (100 - q) * 29 / 100;
            cx->flags |= AV_CODEC_FLAG_QSCALE;
            cx->global_quality = FF_QP2LAMBDA * qs;
        }
        /* zlib level 1 with per-row prediction: a 2048^2 picture takes 0.16 s for 5.9 MB, against
         * 1.0 s for 5.3 MB at the default level 6, on one core. */
        if (format == "png") {
            cx->compression_level = 1;
            av_opt_set(cx->priv_data, "pred", "mixed", 0);
        }
        if (format == "webp") {
            const int q = quality > 0 && quality <= 100 ? quality : 90;
            if (q >= 100) av_opt_set_int(cx->priv_data, "lossless", 1, 0);
            else          av_opt_set_double(cx->priv_data, "quality", (double)q, 0);
        }
        int r = avcodec_open2(cx, codec, nullptr);
        if (r < 0) { fail("the " + format + " encoder would not open (" + averr(r) + ")", RAD_E_INVAL); break; }
        fr->format = pf;
        fr->width = w;
        fr->height = h;
        if (av_frame_get_buffer(fr, 0) < 0) { fail("out of memory", RAD_E_NOMEM); break; }
        const AVPixelFormat src = ch == 4 ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24;
        sws = sws_getContext(w, h, src, w, h, pf, SWS_BILINEAR | SWS_ACCURATE_RND, nullptr,
                             nullptr, nullptr);
        if (!sws) { fail("no colour conversion to the encoder's format", RAD_E_UNSUPPORTED); break; }
        if (format == "jpeg")
            sws_setColorspaceDetails(sws, sws_getCoefficients(SWS_CS_DEFAULT), 1,
                                     sws_getCoefficients(SWS_CS_ITU601), 1, 0, 1 << 16, 1 << 16);
        const uint8_t* sp[1] = { px };
        const int sl[1] = { w * ch };
        if (sws_scale(sws, sp, sl, 0, h, fr->data, fr->linesize) != h) {
            fail("colour conversion failed", RAD_E_INVAL);
            break;
        }
        fr->pts = 0;
        r = avcodec_send_frame(cx, fr);
        if (r >= 0) r = avcodec_send_frame(cx, nullptr);
        if (r < 0) { fail("encoding failed (" + averr(r) + ")", RAD_E_INVAL); break; }
        out->clear();
        for (;;) {
            r = avcodec_receive_packet(cx, pk);
            if (r == AVERROR_EOF || r == AVERROR(EAGAIN)) break;
            if (r < 0) { fail("encoding failed (" + averr(r) + ")", RAD_E_INVAL); break; }
            out->insert(out->end(), pk->data, pk->data + pk->size);
            av_packet_unref(pk);
        }
        if (st == RAD_OK && out->empty()) fail("the encoder wrote nothing", RAD_E_INVAL);
    } while (false);
    sws_freeContext(sws);
    av_packet_free(&pk);
    av_frame_free(&fr);
    avcodec_free_context(&cx);
    return st;
}

int decode_video(const uint8_t* data, size_t n, SampleFn sample, const void* user,
                 int64_t max_pixels, Video* out, std::string* why) {
    std::string w;
    if (!why) why = &w;
    if (!data || n == 0) { *why = "empty"; return RAD_E_INVAL; }

    /* Frame count first, since the sample choice depends on it. A container that does not state
     * it is demuxed once without decoding to count packets (one per frame for the codecs requests
     * send), at the cost of reading the bytes twice. */
    VideoInfo info;
    {
        Input probe;
        RAD_TRY(probe.open(data, n, false, why));
        const AVStream* st = probe.fmt->streams[probe.stream];
        info.w = st->codecpar->width;
        info.h = st->codecpar->height;
        AVRational r = st->avg_frame_rate;
        if (r.num <= 0 || r.den <= 0) r = st->r_frame_rate;
        info.fps = (r.num > 0 && r.den > 0) ? av_q2d(r) : 0.0;
        info.n_frames = st->nb_frames;
        if (info.n_frames <= 0) {
            AVPacket* pkt = av_packet_alloc();
            if (!pkt) { *why = "out of memory"; return RAD_E_NOMEM; }
            int64_t c = 0;
            while (av_read_frame(probe.fmt, pkt) >= 0) {
                if (pkt->stream_index == probe.stream) ++c;
                av_packet_unref(pkt);
            }
            av_packet_free(&pkt);
            info.n_frames = c;
        }
    }
    if (info.w <= 0 || info.h <= 0) { *why = "the video has no size"; return RAD_E_INVAL; }
    if ((int64_t)info.w * info.h > max_pixels) {
        *why = "the video is " + std::to_string(info.w) + "x" + std::to_string(info.h) +
               ", over the server's " + std::to_string(max_pixels) + "-pixel limit for a frame";
        return RAD_E_INVAL;
    }
    if (info.n_frames <= 0) { *why = "the video holds no frames"; return RAD_E_INVAL; }

    out->info = info;
    out->index = sample(info, user);
    if (out->index.empty()) { *why = "no frames were sampled"; return RAD_E_INVAL; }
    out->frames.assign(out->index.size(), Frame{});

    Input in;
    RAD_TRY(in.open(data, n, true, why));
    SwsContext* sws = nullptr;
    size_t next = 0;
    int64_t last_kept = -1;
    const int s = decode_frames(in, [&](int64_t i, const AVFrame* f) -> int {
        if ((int64_t)f->width * f->height > max_pixels) {
            *why = "a frame is larger than the server's pixel limit";
            return RAD_E_INVAL;
        }
        /* A frame may be sampled twice when the stream is shorter than its header says; each
         * slot naming this index takes a copy. */
        bool converted = false;
        while (next < out->index.size() && out->index[next] == i) {
            if (!converted) {
                const int c = to_rgb(f, &sws, &out->frames[next], why);
                if (c < 0) return c;
                converted = true;
            } else {
                out->frames[next] = out->frames[next - 1];
            }
            last_kept = (int64_t)next;
            ++next;
        }
        return next >= out->index.size() ? 1 : 0;
    }, why);
    sws_freeContext(sws);
    if (s < 0) return s;
    if (last_kept < 0) { *why = "the video decoded to no frame"; return RAD_E_INVAL; }
    /* A stream shorter than its header leaves the last sampled indices unreached. They take the
     * last decoded frame, as a reader that clamps an index past the end would. */
    for (size_t k = next; k < out->index.size(); ++k) out->frames[k] = out->frames[(size_t)last_kept];
    return RAD_OK;
}

/* ================================================================== audio */

namespace {

/* The only audio demuxers a request may reach, chosen as kFormats is: formats that hold one
 * stream of samples and cannot name another file. */
constexpr const char* kAudioFormats = "wav,w64,mp3,flac,ogg,mov,mp4,m4a,matroska,webm,aac,aiff";

struct AudioIn {
    MemIO            io;
    AVIOContext*     avio = nullptr;
    AVFormatContext* fmt = nullptr;
    AVCodecContext*  dec = nullptr;
    SwrContext*      swr = nullptr;
    int              stream = -1;

    ~AudioIn() {
        if (swr) swr_free(&swr);
        if (dec) avcodec_free_context(&dec);
        if (fmt) avformat_close_input(&fmt);
        if (avio) {
            av_freep(&avio->buffer);
            avio_context_free(&avio);
        }
    }

    int open(const uint8_t* data, size_t n, std::string* why) {
        io.p = data; io.n = n; io.at = 0;
        constexpr int kBuf = 64 * 1024;
        uint8_t* buf = static_cast<uint8_t*>(av_malloc(kBuf));
        if (!buf) { *why = "out of memory"; return RAD_E_NOMEM; }
        avio = avio_alloc_context(buf, kBuf, 0, &io, mem_read, nullptr, mem_seek);
        if (!avio) { av_free(buf); *why = "out of memory"; return RAD_E_NOMEM; }
        fmt = avformat_alloc_context();
        if (!fmt) { *why = "out of memory"; return RAD_E_NOMEM; }
        fmt->pb = avio;
        fmt->flags |= AVFMT_FLAG_CUSTOM_IO;
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "format_whitelist", kAudioFormats, 0);
        av_dict_set(&opts, "protocol_whitelist", "", 0);
        int e = avformat_open_input(&fmt, nullptr, nullptr, &opts);
        av_dict_free(&opts);
        if (e < 0) {
            fmt = nullptr;
            *why = "not a recognised audio format (" + averr(e) + ")";
            return RAD_E_INVAL;
        }
        e = avformat_find_stream_info(fmt, nullptr);
        if (e < 0) { *why = "could not read the stream headers (" + averr(e) + ")"; return RAD_E_INVAL; }
        const AVCodec* codec = nullptr;
        stream = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, &codec, 0);
        if (stream < 0 || !codec) { *why = "no sound in it: the media holds no audio stream"; return RAD_E_INVAL; }
        dec = avcodec_alloc_context3(codec);
        if (!dec) { *why = "out of memory"; return RAD_E_NOMEM; }
        e = avcodec_parameters_to_context(dec, fmt->streams[stream]->codecpar);
        if (e < 0) { *why = "decoder setup failed (" + averr(e) + ")"; return RAD_E_INVAL; }
        dec->thread_count = 1;
        e = avcodec_open2(dec, codec, nullptr);
        if (e < 0) { *why = std::string("no decoder for ") + codec->name + " (" + averr(e) + ")"; return RAD_E_UNSUPPORTED; }
        return RAD_OK;
    }
};

}  /* namespace */

int decode_audio(const uint8_t* data, size_t n, int32_t rate, int64_t max_samples,
                 std::vector<float>* out, std::string* why) {
    std::string w;
    if (!why) why = &w;
    out->clear();
    if (rate <= 0 || max_samples <= 0) { *why = "no sample rate or length to decode to"; return RAD_E_INVAL; }
    AudioIn in;
    RAD_TRY(in.open(data, n, why));
    AVCodecContext* dec = in.dec;
    const int channels = dec->ch_layout.nb_channels > 0 ? dec->ch_layout.nb_channels : 1;
    if (dec->sample_rate <= 0) { *why = "the audio stream states no sample rate"; return RAD_E_INVAL; }

    /* Resampled at the source's own layout, float interleaved; the mono mean is taken after. A
     * layout the stream does not state is taken as `channels` unordered channels. */
    AVChannelLayout layout{};
    if (av_channel_layout_check(&dec->ch_layout)) av_channel_layout_copy(&layout, &dec->ch_layout);
    else av_channel_layout_default(&layout, channels);
    int e = swr_alloc_set_opts2(&in.swr, &layout, AV_SAMPLE_FMT_FLT, rate, &layout,
                                (AVSampleFormat)dec->sample_fmt, dec->sample_rate, 0, nullptr);
    av_channel_layout_uninit(&layout);
    if (e < 0 || !in.swr || swr_init(in.swr) < 0) {
        *why = "no resampler for this audio (" + averr(e) + ")";
        return RAD_E_UNSUPPORTED;
    }

    std::vector<float> buf;
    auto take = [&](const uint8_t** src, int n_in) -> int {
        const int cap = swr_get_out_samples(in.swr, n_in);
        if (cap <= 0) return 0;
        buf.resize((size_t)cap * (size_t)channels);
        uint8_t* dst = reinterpret_cast<uint8_t*>(buf.data());
        const int got = swr_convert(in.swr, &dst, cap, src, n_in);
        if (got < 0) { *why = "resampling failed (" + averr(got) + ")"; return RAD_E_INVAL; }
        for (int i = 0; i < got && (int64_t)out->size() < max_samples; ++i) {
            float s = 0.0f;
            for (int c = 0; c < channels; ++c) s += buf[(size_t)i * channels + c];
            out->push_back(channels == 1 ? s : s / (float)channels);
        }
        return 0;
    };

    AVPacket* pkt = av_packet_alloc();
    AVFrame* f = av_frame_alloc();
    if (!pkt || !f) {
        av_packet_free(&pkt);
        av_frame_free(&f);
        *why = "out of memory";
        return RAD_E_NOMEM;
    }
    int st = RAD_OK;
    bool done = false;
    while (!done && (int64_t)out->size() < max_samples) {
        e = av_read_frame(in.fmt, pkt);
        if (e < 0) {
            avcodec_send_packet(dec, nullptr);   /* drain */
            done = true;
        } else {
            if (pkt->stream_index == in.stream) e = avcodec_send_packet(dec, pkt);
            av_packet_unref(pkt);
            if (e < 0 && e != AVERROR(EAGAIN)) { *why = "the audio would not decode (" + averr(e) + ")"; st = RAD_E_INVAL; break; }
        }
        while ((e = avcodec_receive_frame(dec, f)) >= 0) {
            if ((st = take(const_cast<const uint8_t**>(f->extended_data), f->nb_samples)) < 0) break;
            av_frame_unref(f);
        }
        if (st < 0) break;
        if (e != AVERROR(EAGAIN) && e != AVERROR_EOF) { *why = "the audio would not decode (" + averr(e) + ")"; st = RAD_E_INVAL; break; }
    }
    /* Flush what the resampler still holds. */
    if (st == RAD_OK && (int64_t)out->size() < max_samples) st = take(nullptr, 0);
    av_packet_free(&pkt);
    av_frame_free(&f);
    if (st < 0) return st;
    if (out->empty()) { *why = "the audio decoded to no samples"; return RAD_E_INVAL; }
    return RAD_OK;
}

namespace {

/* The H.264 encoder the round trip uses: libx264, the reference's, else libopenh264 (the published
 * image carries libopenh264 and no libx264). Not any H.264 encoder: the hardware ones need a device
 * context this path does not set up. */
const AVCodec* h264_encoder() {
    for (const char* name : { "libx264", "libopenh264" })
        if (const AVCodec* c = avcodec_find_encoder_by_name(name)) return c;
    return nullptr;
}

}  /* namespace */

bool h264_roundtrip_available() {
    return h264_encoder() && avcodec_find_decoder(AV_CODEC_ID_H264);
}

/* PyAV's single-frame round trip: VideoFrame.from_ndarray(rgb24).reformat("yuv420p") (plain
 * SWS_BILINEAR, BT.601 coefficients, limited-range output), libx264 with the given crf, then the
 * decoded frame's to_ndarray("rgb24"), which to_rgb reproduces. Without libx264, libopenh264 takes
 * the frame at a quantiser held at the crf, about the compression libx264's crf gives a single intra
 * frame, so the first frame is close to the reference's but not exact. libopenh264 takes no side
 * under 16, so a smaller picture is encoded with its last row and column repeated out to 16 and
 * the decoded frame cut back. */
int h264_roundtrip(const Frame& in, int32_t crf, Frame* out, std::string* why) {
    std::string w;
    if (!why) why = &w;
    const int W = in.w & ~1, H = in.h & ~1;
    if (W <= 0 || H <= 0) { *why = "the picture is smaller than 2x2"; return RAD_E_INVAL; }
    const AVCodec* enc = h264_encoder();
    const AVCodec* dcd = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!enc) { *why = "this libavcodec has no H.264 encoder"; return RAD_E_UNSUPPORTED; }
    if (!dcd) { *why = "this libavcodec has no H.264 decoder"; return RAD_E_UNSUPPORTED; }
    const bool x264 = !std::strcmp(enc->name, "libx264");
    const int EW = x264 ? W : std::max(W, 16), EH = x264 ? H : std::max(H, 16);
    AVCodecContext* ec = avcodec_alloc_context3(enc);
    AVCodecContext* dc = avcodec_alloc_context3(dcd);
    AVFrame* yuv = av_frame_alloc();
    AVFrame* dec = av_frame_alloc();
    AVPacket* pkt = av_packet_alloc();
    SwsContext* sws = nullptr;
    int st = RAD_OK;
    auto fail = [&](const char* what, int e) {
        *why = std::string(what) + (e < 0 ? " (" + averr(e) + ")" : std::string());
        st = e == AVERROR(ENOMEM) ? RAD_E_NOMEM : RAD_E_INVAL;
    };
    bool got = false;
    if (!ec || !dc || !yuv || !dec || !pkt) {
        *why = "out of memory";
        st = RAD_E_NOMEM;
    } else {
        ec->width = EW;
        ec->height = EH;
        ec->pix_fmt = AV_PIX_FMT_YUV420P;
        ec->time_base = AVRational{ 1, 1 };
        ec->framerate = AVRational{ 1, 1 };
        /* PyAV's codec defaults: as many threads as the machine has, splitting each frame into
         * that many slices, which shapes the bitstream. */
        ec->thread_count = 0;
        ec->thread_type = FF_THREAD_SLICE;
        if (x264) {
            av_opt_set(ec->priv_data, "preset", "veryfast", 0);
            av_opt_set(ec->priv_data, "crf", std::to_string(crf).c_str(), 0);
        } else {
            /* libopenh264's rate control held between qmin and qmax: its "off" mode encodes at the
             * layer quantiser FFmpeg leaves at the library's default and ignores both. */
            ec->qmin = ec->qmax = std::clamp(crf, 1, 51);
            av_opt_set(ec->priv_data, "rc_mode", "quality", 0);
        }
        int e = avcodec_open2(ec, enc, nullptr);
        if (e < 0) fail(x264 ? "opening libx264" : "opening the H.264 encoder", e);
        if (st == RAD_OK && (e = avcodec_open2(dc, dcd, nullptr)) < 0) fail("opening the H.264 decoder", e);
        if (st == RAD_OK) {
            yuv->format = AV_PIX_FMT_YUV420P;
            yuv->width = EW;
            yuv->height = EH;
            if ((e = av_frame_get_buffer(yuv, 0)) < 0) fail("allocating a frame", e);
        }
        if (st == RAD_OK) {
            sws = sws_getContext(EW, EH, AV_PIX_FMT_RGB24, EW, EH, AV_PIX_FMT_YUV420P, SWS_BILINEAR,
                                 nullptr, nullptr, nullptr);
            if (!sws) fail("no RGB to yuv420p conversion", 0);
        }
        if (st == RAD_OK) {
            /* The source rows into a padded RGB frame: the scaler reads whole vectors past a tight
             * row's end. */
            AVFrame* rgb = av_frame_alloc();
            if (!rgb) fail("out of memory", AVERROR(ENOMEM));
            else {
                rgb->format = AV_PIX_FMT_RGB24;
                rgb->width = EW;
                rgb->height = EH;
                if ((e = av_frame_get_buffer(rgb, 0)) < 0) fail("allocating a frame", e);
                else {
                    for (int y = 0; y < EH; ++y) {
                        uint8_t* row = rgb->data[0] + (ptrdiff_t)y * rgb->linesize[0];
                        std::memcpy(row, in.rgb.data() + (size_t)std::min(y, H - 1) * in.w * 3,
                                    (size_t)W * 3);
                        for (int x = W; x < EW; ++x) std::memcpy(row + x * 3, row + (W - 1) * 3, 3);
                    }
                    if (sws_scale(sws, rgb->data, rgb->linesize, 0, EH, yuv->data, yuv->linesize) != EH)
                        fail("colour conversion failed", 0);
                }
                av_frame_free(&rgb);
            }
        }
        /* Encode the frame and flush; every packet goes straight to the decoder. */
        for (int pass = 0; st == RAD_OK && pass < 2; ++pass) {
            yuv->pts = 0;
            if ((e = avcodec_send_frame(ec, pass == 0 ? yuv : nullptr)) < 0) { fail("encoding", e); break; }
            while (st == RAD_OK && (e = avcodec_receive_packet(ec, pkt)) >= 0) {
                if ((e = avcodec_send_packet(dc, pkt)) < 0) fail("decoding", e);
                av_packet_unref(pkt);
            }
            if (st == RAD_OK && e != AVERROR(EAGAIN) && e != AVERROR_EOF) fail("encoding", e);
        }
        if (st == RAD_OK) avcodec_send_packet(dc, nullptr);
        while (st == RAD_OK && (e = avcodec_receive_frame(dc, dec)) >= 0) {
            if (!got) {
                SwsContext* back = nullptr;
                st = to_rgb(dec, &back, out, why);
                sws_freeContext(back);
                got = st == RAD_OK;
                if (got && (out->w != W || out->h != H)) {
                    if (out->w < W || out->h < H) {
                        fail("the round trip decoded to a smaller frame", 0);
                        got = false;
                    } else {
                        for (int y = 0; y < H; ++y)
                            std::memmove(out->rgb.data() + (size_t)y * W * 3,
                                         out->rgb.data() + (size_t)y * out->w * 3, (size_t)W * 3);
                        out->rgb.resize((size_t)W * H * 3);
                        out->w = W;
                        out->h = H;
                    }
                }
            }
            av_frame_unref(dec);
        }
        if (st == RAD_OK && !got) fail("the round trip decoded to no frame", 0);
    }
    sws_freeContext(sws);
    av_packet_free(&pkt);
    av_frame_free(&yuv);
    av_frame_free(&dec);
    avcodec_free_context(&ec);
    avcodec_free_context(&dc);
    return st;
}

#endif  /* RAD_HAVE_FFMPEG */

}  /* namespace mm */
}  /* namespace rad */
