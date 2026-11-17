/* gen.h -- image generation: the request, what comes back, and the driver that turns a request
 * into the passes an image generator declares (rad_declare_generator, abi/rad_builder.h).
 *
 * A generator is not driven by token steps: an image is tens of denoising passes over thousands of
 * latent rows, all read whole, none sampled. Token positions, prefix reuse, preemption and
 * speculation have no meaning here, so the generator has its own driver and shares only what lies
 * below the step: the ranks, the plugin's step function and the KV manager holding conditioning.
 *
 * Passes, as the driver runs them:
 *
 *   COND     once per conditioning sequence (the prompt, plus the negative prompt with guidance):
 *            the prompt and its reference pictures encoded into the generator's KV group. A
 *            request is admitted once its lanes and its KV are free.
 *   DENOISE  one step for each in-flight image that fits the pass, any mix of requests, sizes and
 *            step counts; each image carries its own sigma. With guidance an image is two entries,
 *            positive and negative conditioning, which the plugin combines.
 *   DECODE   one image's latent to pixels, read back and handed to its request.
 *
 * Images run in lanes, the plugin's own latent state (`max_lanes` of them): a request of n images
 * takes n lanes and shares one conditioning.
 */
#pragma once
#include "rad_core.h"
#include "mm/processor.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
struct IKVManager;
namespace gen {

/* A picture as the generator's own encoder reads it: RGBA (or `ch` channels), row-major, each value
 * bf16 in [-1, 1]. */
struct Pixels {
    int32_t w = 0, h = 0, ch = 4;
    std::vector<uint16_t> bf16;
};

/* One conditioning sequence: the prompt tokens with their media placed, and how many leading
 * tokens the generator does not see (RadGeneratorDecl::drop_prefix). */
struct Branch {
    std::vector<int32_t>             tokens;
    std::shared_ptr<mm::PromptMedia> media;    /* null for a prompt with no pictures */
    int32_t                          n_drop = 0;
    /* The user's prompt through the model's second tokenizer (RadGeneratorDecl::aux_tokens);
     * empty otherwise. */
    std::vector<int32_t>             aux;
};

/* One finished image, or a preview of one. */
struct Image {
    int32_t              index = 0;     /* which of the request's n */
    int32_t              w = 0, h = 0, ch = 4;
    std::vector<uint8_t> px;            /* [h, w, ch] */
    bool                 preview = false;
    int32_t              preview_no = 0;
    uint64_t             seed = 0;
};

/* The channel back to the caller: images as they finish, then the end. The driver thread writes,
 * the caller reads; a mutex suffices for a handful of megabyte images per request. `cancel` is set
 * by the caller and read by the driver at its next pass. */
class Ticket {
public:
    /* driver side */
    void push(Image&& im);
    void finish(int status, const std::string& why = std::string());

    /* caller side: the next image, or false once the request has ended and every image was taken.
     * Waits at most `ms` milliseconds; `*timed_out` marks a false that was a timeout, not the end. */
    bool next(Image* out, int ms, bool* timed_out);
    bool finished() const;
    int  status() const;
    std::string why() const;

    std::atomic<bool> cancel{false};

    /* Usage for the response: conditioning tokens, latent rows made, and per-phase times.
     * Written by the driver before finish(). */
    int64_t in_tokens = 0, out_tokens = 0;
    int64_t t_submit_ns = 0, t_cond_ns = 0, t_first_step_ns = 0, t_denoised_ns = 0, t_done_ns = 0;
    int32_t steps = 0;

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::deque<Image>       q_;
    bool                    done_ = false;
    int                     status_ = RAD_OK;
    std::string             why_;
};

struct Request {
    Branch   pos, neg;
    bool     has_neg = false;
    float    cfg = 1.0f;
    int32_t  width = 0, height = 0;     /* pixels, whole multiples of the declared multiple */
    int32_t  steps = 0;                 /* 0: the model's default (or its fixed schedule) */
    /* A discrete-time model's sampler and scheduler (RAD_GEN_SAMPLER_*, RAD_GEN_SCHED_*). */
    int32_t  sampler = 0, scheduler = 0;
    int32_t  n = 1;
    uint64_t seed = 0;                  /* image i draws its noise from seed + i */
    int32_t  previews = 0;              /* partial images per image, 0..3 */
    /* The reference pictures' encoder view, one per media item of `pos`, in order. */
    std::vector<Pixels> refs;
    /* Inpainting (an edit with a mask): the source picture at the output size, and per latent
     * token how much is repainted (1) versus kept (0). */
    bool                 inpaint = false;
    Pixels               inp_src;
    std::vector<float>   inp_mask;
    /* The runtime LoRA slot every pass of the request runs with (Engine::gen_lora_acquire), -1
     * for none, and the hold that keeps the slot's adapters in it until the request is done. */
    int32_t              lora_slot = -1;
    std::shared_ptr<void> lora_hold;
    /* RAD_GEN_COND_* for the plugin's COND passes (`rebalance`: Krea 2's tap gains). */
    int32_t              cond_opts = 0;
    std::shared_ptr<Ticket> ticket;
};

/* ================================================================== the sampler's schedule */
/* The n + 1 sigmas a request steps through, the last 0: the decl's fixed schedule, or `steps`
 * (0: the default) shifted as the decl says. `tokens` is the image's latent token count, read by a
 * dynamic shift. f32, matching the reference. */
std::vector<float> schedule(const RadGeneratorDecl& d, const std::vector<double>& fixed,
                            int32_t steps, int64_t tokens);

/* ================================================================== a discrete-time schedule */
/* The sigma table of a model trained on discrete timesteps (RadGeneratorDecl::train_steps), as
 * ComfyUI's ModelSamplingDiscrete builds it: betas in f64, their cumulative product, sigma_t =
 * sqrt((1 - abar_t) / abar_t), then f32, and the f32 logs. */
struct DiscreteTable {
    std::vector<float> sigmas;      /* [train_steps], ascending: sigmas[t] is timestep t's */
    std::vector<float> log_sigmas;
    float sigma_min() const { return sigmas.front(); }
    float sigma_max() const { return sigmas.back(); }
    /* sigma at a fractional timestep: the logs interpolated linearly, t clamped to the table. */
    float sigma_at(double t) const;
    /* The timestep a sigma is conditioned at: the table's nearest in log sigma. */
    int32_t timestep(float sigma) const;
};
/* RAD_E_INVAL with `why` for a declaration whose schedule cannot be built. */
int discrete_table(const RadGeneratorDecl& d, DiscreteTable* out, std::string* why);
/* The n + 1 sigmas `scheduler` places for `steps` (0: the default) steps, the last 0. */
std::vector<float> schedule_discrete(const RadGeneratorDecl& d, const DiscreteTable& t,
                                     int32_t scheduler, int32_t steps);
/* The names a request spells samplers and schedulers by, and their ids (-1: none). */
const char* sampler_name(int32_t id);
const char* scheduler_name(int32_t id);
int32_t     sampler_id(const std::string& name);
int32_t     scheduler_id(const std::string& name);

/* ================================================================== the generator's sequence */
/* One conditioning sequence as the generator sees it (RadGeneratorDecl::layout): the prompt row or
 * reference latent row behind each row, rotary positions, segments, and the output image's frame. */
struct Layout {
    int64_t              rows = 0;
    std::vector<int32_t> src;          /* [rows] */
    std::vector<int32_t> rope;         /* [3, rows] component-major */
    std::vector<int32_t> seg;          /* [n_seg + 1] */
    std::vector<int32_t> bidi;         /* [n_seg] */
    int32_t              next_pos = 0; /* the rotary count after the last row: the image's frame */
    /* JOINT_STEP only: the text's rows (padding included), the padding rows among them, and the
     * f32 positions, [4, rows] component-major (`rope` is then empty). */
    int64_t              text = 0;
    int32_t              n_pad = 0;
    std::vector<float>   rope_f;
};
/* `ref_hw` is each reference's latent grid (h, w), in media order. RAD_E_INVAL with `why` when the
 * prompt does not start with its drop prefix or a picture's slots do not tile its latent grid. */
int build_layout(const Branch& b, int64_t slot_rows, const std::vector<std::pair<int32_t, int32_t>>& ref_hw,
                 Layout* out, std::string* why);
/* The rotary positions of an h x w image at frame `pos`: [3, h * w] component-major. */
void image_rope(int32_t pos, int32_t h, int32_t w, int32_t* planes, int64_t n_total);

/* A conditioning in the JOINT_STEP layout (RadGeneratorDecl): the prompt's rows after its drop
 * prefix, padding to a multiple of `pad_multiple` (0: none), then each reference picture's latent
 * tokens. `src` is the prompt row behind a text row -- a padding row's is past the prompt's last,
 * where the COND pass appends the padding tokens -- or -(1 + i) for reference latent row i. One
 * text segment (causal) and a segment per picture (bidirectional). */
int build_joint_layout(const Branch& b, int32_t pad_multiple, double hw_scale, bool text_l,
                       const std::vector<std::pair<int32_t, int32_t>>& ref_hw, Layout* out,
                       std::string* why);
/* The f32 positions (0, s h, s w, l) of an h x w grid: [4, h * w] component-major. */
void grid_rope_f(double hw_scale, float l, int32_t h, int32_t w, float* planes, int64_t n_total);

/* A prompt's pictures for the vision tower of a conditioning pass (RadBatch::enc_*, mm_rows), the
 * image and the video drivers': every item's patches in prompt order at `patch_dim` bf16 each, their
 * coordinates [4, patches], the segment boundaries and the prompt rows the tower's output
 * replaces. */
struct StagedMedia {
    int64_t n_patch = 0, n_seg = 0, n_rows = 0;
    int32_t max_seg = 0;
};
StagedMedia stage_media(const mm::PromptMedia& pm, int64_t patch_dim, uint16_t* pixels,
                        int32_t* coords, int32_t* cu, int32_t* rows);

}  /* namespace gen */
}  /* namespace rad */
