/* video.h -- video generation: the request and the channel its video comes back on.
 *
 * A video generator (abi/rad_builder.h, rad_declare_video_generator) makes a video and its
 * soundtrack in one denoising loop over a packed sequence, then decodes the soundtrack and the
 * frames, a chunk of frames at a time. core/gen/video_driver.h turns a request into those passes;
 * this is what a caller hands it and what it hands back: progress as the steps run, the
 * soundtrack, the frames chunk by chunk in order, then the end.
 */
#pragma once
#include "rad_core.h"
#include "mm/processor.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace gen {

/* One thing a request has made. */
struct VideoEvent {
    enum Kind { STEP = 1, AUDIO = 2, FRAMES = 3, LENGTH = 4 };
    int                  kind = 0;
    int32_t              steps_done = 0;   /* STEP: denoising steps finished */
    /* AUDIO: the whole soundtrack, interleaved, in [-1, 1]. */
    std::vector<float>   samples;
    /* FRAMES: frames [first, first + n) of the video, each [height, width, 3] u8 RGB. LENGTH:
     * the `n` frames the model chose for a request that named no length, before any step. */
    int32_t              first = 0, n = 0;
    std::vector<uint8_t> rgb;
};

/* As Ticket (gen.h) is for an image: the driver thread writes, the caller reads. */
class VideoTicket {
public:
    /* driver side */
    void push(VideoEvent&& e);
    void finish(int status, const std::string& why = std::string());

    /* caller side: the next event, or false once the request has ended and every event was
     * taken. Waits at most `ms` milliseconds; `*timed_out` marks a false that was a timeout. */
    bool next(VideoEvent* out, int ms, bool* timed_out);
    bool finished() const;
    int  status() const;
    std::string why() const;

    std::atomic<bool> cancel{false};

    /* What the request makes, written by the driver at submit: the frames and their size, the
     * steps, and the soundtrack's rate and channels. A request whose length the model predicts
     * has its `frames`, `steps` and `rows` rewritten once, before its LENGTH event. */
    int32_t frames = 0, width = 0, height = 0, steps = 0;
    int64_t sample_rate = 0, channels = 0;
    double  fps = 0.0;
    /* With latent upscaling, the size the first evaluations run at and the evaluations after the
     * upscale; zero without. */
    int32_t small_width = 0, small_height = 0, refine = 0;
    /* Usage, written by the driver before finish(): prompt tokens, rows a step and per-phase
     * times (t_small_ns the last small step done, t_upscaled_ns the upscale; zero without). */
    int64_t in_tokens = 0, rows = 0;
    int64_t t_submit_ns = 0, t_cond_ns = 0, t_small_ns = 0, t_upscaled_ns = 0, t_denoised_ns = 0,
            t_done_ns = 0;

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<VideoEvent>  q_;
    bool                    done_ = false;
    int                     status_ = RAD_OK;
    std::string             why_;
};

/* A reference medium or a keyframe of a request: a picture, a video, a video's own soundtrack, a
 * standalone soundtrack, or the first or last frame. `index` is its number within its kind from
 * 1 (VIDEO_AUDIO i is VIDEO i's soundtrack); a keyframe's is its <Picture N> in the prompt, 1 the
 * first and 2 the last. */
struct VideoRefMedia {
    enum Kind : int32_t {
        IMAGE = 0, VIDEO = 1, VIDEO_AUDIO = 2, AUDIO = 3, FIRST_FRAME = 4, LAST_FRAME = 5
    };
    Kind        kind = IMAGE;
    int32_t     index = 0;
    std::string name, filename, content_type;   /* the form part's */
    std::string data;                           /* the file's bytes */

    /* The media as its encoder takes it, decoded by the server, before the driver pads it to the
     * encoder's clips and hops. A picture's or video's frames, f32 [enc_frames, enc_h, enc_w, 3] RGB
     * in 0..255: a reference's sides multiples of the VAE's ratio within max_enc_side, aspect
     * kept, a keyframe at the video's size. A soundtrack, f32 [enc_channels, enc_n] at the
     * model's rate, a mono one copied to every channel. */
    std::vector<float> enc_pixels;
    int32_t            enc_frames = 0, enc_h = 0, enc_w = 0;
    std::vector<float> enc_samples;
    int64_t            enc_n = 0;
    int32_t            enc_channels = 0;
    /* A keyframe of a latent-upscaled request again at the small size, prepared as at the full
     * one: the steps before the upscale pin it on their own grid. Empty otherwise. */
    std::vector<float> small_pixels;
    int32_t            small_h = 0, small_w = 0;
};

struct VideoRequest {
    std::vector<int32_t> tokens;   /* the prompt, rendered and tokenised */
    int32_t  width = 0, height = 0;   /* pixels, within the declaration's limits */
    int32_t  chunks = 0;           /* the length in chunks (RadVideoGenDecl) */
    int32_t  steps = 0;            /* sigma grid points; 0: the model's default */
    /* RAD_SAMPLER_* (abi/rad_runtime.h). */
    int32_t  sampler = 0;
    /* Each stream's schedule shift; 0: the declaration's. */
    double   shift_video = 0.0;
    double   shift_audio = 0.0;
    /* Latent upscaling (abi/rad_builder.h, RadVideoGenDecl): the factor, 0 for none, and the
     * evaluations at the full size after it, 0 for the model's default. */
    double   upscale = 0.0;
    int32_t  refine = 0;
    uint64_t seed = 0;
    /* The reference media and keyframes, sorted by (kind, index). */
    std::vector<VideoRefMedia> refs;
    /* The pictures the prompt shows the text encoder's vision tower, placed in `tokens`; null for
     * none. */
    std::shared_ptr<mm::PromptMedia> media;
    /* Guidance (RadVideoGenDecl::guidance): the eight terms, NaN for the model's default, and the
     * negative prompt, rendered and tokenised; `has_neg` false for the model's default. */
    double   guidance[8] = { NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN };
    std::vector<int32_t> neg_tokens;
    bool     has_neg = false;
    /* A first frame (RadVideoGenDecl::image_cond): [height, width, 3] u8 RGB at the size the
     * first evaluations run at (video_small_size under upscaling, else the video's), already
     * prepared as the model's pipeline prepares it; empty for none. */
    std::vector<uint8_t> image;
    /* The decoder (RadVideoGenDecl::decoders) by its index; 0 the model's default. */
    int32_t  decoder = 0;
    /* The runtime LoRA slot the request's conditioning and denoising passes run with
     * (Engine::gen_lora_acquire), -1 for none, and the hold that keeps it until the request is
     * done. */
    int32_t  lora_slot = -1;
    std::shared_ptr<void> lora_hold;
    std::shared_ptr<VideoTicket> ticket;
};

/* A request's shape under a declaration: frames, latent grid, soundtrack latents and the rows of
 * a step. Shared by the driver and the server, which checks a request before preparing it. */
struct VideoShape {
    int32_t frames = 0, lat_frames = 0, lat_h = 0, lat_w = 0, n_audio = 0;
    int64_t video_rows = 0, audio_rows = 0;
};
VideoShape video_shape(const RadVideoGenDecl& d, int32_t width, int32_t height, int32_t chunks);

/* The small size a request's first evaluations run at under latent upscaling by `factor`: each
 * side divided by it and rounded to the nearest multiple of the size multiple, at least one
 * multiple and at most the side. */
void video_small_size(const RadVideoGenDecl& d, int32_t width, int32_t height, double factor,
                      int32_t* small_w, int32_t* small_h);

/* The two schedules a request of `steps` runs (RadVideoGenDecl, "The sampler": grid points under
 * form 0, evaluations under form 1): each the f32 sigmas, last 0, and the model evaluations it
 * takes, the shorter of the two. With a trained schedule any other step count than the declared
 * one gives no steps. `video_tokens` are the video's latent tokens, which form 1's dynamic shift
 * reads. With `refine` (form 1 and refinement rungs) the rungs are followed by the refinement
 * rungs instead of a 0, the evaluations their sum. `shift_video` and `shift_audio` replace the
 * declaration's shifts of form 0 when above 0. */
struct VideoSchedule {
    std::vector<float> sig_v, sig_a;
    int32_t            steps = 0;
};
VideoSchedule video_schedule(const RadVideoGenDecl& d, int32_t steps, int64_t video_tokens = 0,
                             bool refine = false, double shift_video = 0.0,
                             double shift_audio = 0.0);

}  /* namespace gen */
}  /* namespace rad */
