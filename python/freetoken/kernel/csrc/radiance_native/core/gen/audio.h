/* audio.h -- audio generation: the request and the channel its audio comes back on.
 *
 * An audio generator (abi/rad_builder.h, rad_declare_audio_generator) makes a piece in two stages,
 * a language model writing it frame by frame and a denoiser turning runs of frames (windows) into
 * audio, which a vocoder plays. core/gen/audio_driver.h turns a request into those passes; this is
 * what a caller hands it and what it hands back: a window's samples as each is vocoded, then the
 * end.
 */
#pragma once
#include "rad_core.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace gen {

/* One window's playable samples, interleaved, in [-1, 1]. */
struct AudioChunk {
    int32_t            window = 0;
    std::vector<float> samples;
};

/* As Ticket (gen.h) is for an image: the driver thread writes, the caller reads. */
class AudioTicket {
public:
    /* driver side */
    void push(AudioChunk&& c);
    void finish(int status, const std::string& why = std::string());

    /* caller side: the next chunk, or false once the request has ended and every chunk was
     * taken. Waits at most `ms` milliseconds; `*timed_out` marks a false that was a timeout. */
    bool next(AudioChunk* out, int ms, bool* timed_out);
    bool finished() const;
    int  status() const;
    std::string why() const;

    std::atomic<bool> cancel{false};

    /* Usage for the response, written by the driver before finish(): prompt tokens, frames made,
     * windows played and per-phase times. */
    int64_t in_tokens = 0, frames = 0, windows = 0;
    int64_t t_submit_ns = 0, t_prompt_ns = 0, t_frames_ns = 0, t_first_audio_ns = 0,
            t_done_ns = 0;
    /* Why the piece ended: the model ended it, or it reached the request's length. */
    bool    model_ended = false;

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<AudioChunk>  q_;
    bool                    done_ = false;
    int                     status_ = RAD_OK;
    std::string             why_;
};

struct AudioRequest {
    /* The conditional prompt, and the unconditional one when the declaration renders it from a
     * template of its own (RadAudioGenDecl::neg_tmpl); otherwise it is derived from the
     * conditional prompt as the declaration says (neg_token), and `neg_tokens` is empty. */
    std::vector<int32_t> tokens;
    std::vector<int32_t> neg_tokens;
    /* References (RadAudioGenDecl::tmpl_ref): the clips' samples end to end, each padded to whole
     * frames; their bounds in samples, [clips + 1]; and for each row of `tokens` the frame of the
     * clips it is, -1 for a token. All empty without references. */
    std::vector<float>   ref_audio;
    std::vector<int32_t> ref_cu;
    std::vector<int32_t> ref_frame;
    /* Reference-direction guidance: the negative prompt's own map into the same clips. */
    std::vector<int32_t> neg_ref_frame;
    int64_t  max_frames = 0;       /* the most frames to make, >= 1 */
    int32_t  steps = 0;            /* 0: the model's default */
    float    cfg = 0.0f;           /* 0: the model's default */
    uint64_t seed = 0;
    std::shared_ptr<AudioTicket> ticket;
};

}  /* namespace gen */
}  /* namespace rad */
