/* video_driver.cpp -- the video generator's pass driver (video_driver.h), the ticket a caller
 * waits on, and a request's shape and schedule (video.h). */
#include "video_driver.h"
#include "gen.h"
#include "rad_internal.h"
#include "rad_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace rad {
namespace gen {

/* ================================================================== the ticket */
void VideoTicket::push(VideoEvent&& e) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back(std::move(e));
    }
    cv_.notify_all();
}

void VideoTicket::finish(int status, const std::string& why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (done_) return;
        done_ = true;
        status_ = status;
        why_ = why;
    }
    cv_.notify_all();
}

bool VideoTicket::next(VideoEvent* out, int ms, bool* timed_out) {
    std::unique_lock<std::mutex> lk(mu_);
    if (timed_out) *timed_out = false;
    if (q_.empty() && !done_)
        cv_.wait_for(lk, std::chrono::milliseconds(ms), [this] { return !q_.empty() || done_; });
    if (!q_.empty()) {
        *out = std::move(q_.front());
        q_.pop_front();
        return true;
    }
    if (!done_ && timed_out) *timed_out = true;
    return false;
}

bool VideoTicket::finished() const {
    std::lock_guard<std::mutex> lk(mu_);
    return done_ && q_.empty();
}

int VideoTicket::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
}

std::string VideoTicket::why() const {
    std::lock_guard<std::mutex> lk(mu_);
    return why_;
}

/* ================================================================== shape and schedule */
VideoShape video_shape(const RadVideoGenDecl& d, int32_t width, int32_t height, int32_t chunks) {
    VideoShape s;
    s.frames = (int32_t)(d.frames_per_chunk * chunks + d.frames_tail);
    s.lat_frames = (int32_t)(d.latents_per_chunk * chunks + d.latents_tail);
    s.lat_h = (int32_t)(height / d.latent_scale);
    s.lat_w = (int32_t)(width / d.latent_scale);
    /* round(frames / fps * audio_rate), ties to even as the reference's float round; with an
     * integer numerator and denominator a tie is exact. */
    const int64_t num = (int64_t)s.frames * d.fps_den * d.audio_rate, den = d.fps_num;
    int64_t q = num / den;
    const int64_t r2 = 2 * (num % den);
    if (r2 > den || (r2 == den && (q & 1))) ++q;
    s.n_audio = (int32_t)q;
    s.video_rows = (int64_t)s.lat_frames * (s.lat_h / d.patch) * (s.lat_w / d.patch);
    s.audio_rows = (d.audio_streams > 0 ? d.audio_streams : d.audio_channels) * (int64_t)s.n_audio;
    return s;
}

void video_small_size(const RadVideoGenDecl& d, int32_t width, int32_t height, double factor,
                      int32_t* small_w, int32_t* small_h) {
    const int64_t m = d.size_multiple;
    auto side = [&](int32_t full) {
        const int64_t k = (int64_t)std::floor((double)full / factor / (double)m + 0.5);
        return (int32_t)(std::min<int64_t>(std::max<int64_t>(k, 1), full / m) * m);
    };
    *small_w = side(width);
    *small_h = side(height);
}

/* The shift of one base time, in f32 as the reference's tensor arithmetic. */
static float shift_sigma(double shift, float b) {
    const float sh = (float)shift, sh1 = (float)(shift - 1.0);
    return (sh * b) / (1.0f + sh1 * b);
}

/* torch.linspace(1, 0, n) in f32, which computes the first half up from the start and the
 * second half down from the end, then the exponential shift, consecutive duplicates dropped. */
static std::vector<float> shifted_sigmas(double shift, int64_t n) {
    std::vector<float> s;
    if (n < 2) return s;
    const float start = 1.0f, end = 0.0f;
    const float step = (end - start) / (float)(n - 1);
    const int64_t half = n / 2;
    for (int64_t i = 0; i < n; ++i) {
        const float b = i < half ? start + step * (float)i : end - step * (float)(n - i - 1);
        const float v = shift_sigma(shift, b);
        if (s.empty() || v != s.back()) s.push_back(v);
    }
    return s;
}

/* A trained schedule: each rung taken to f32 and shifted, then the terminal 0. */
static std::vector<float> rung_sigmas(double shift, const double* rungs, int32_t n) {
    std::vector<float> s;
    for (int32_t i = 0; i < n; ++i) s.push_back(shift_sigma(shift, (float)rungs[i]));
    s.push_back(0.0f);
    return s;
}

/* Form 1 (RadVideoGenDecl::sched_kind): FlowMatchEulerDiscreteScheduler.set_timesteps with
 * LTX-2's sigmas. numpy's linspace in f64 (i * step + start, the last exactly the stop) taken to
 * f32; the shift and the terminal stretch in f32, as numpy computes on an f32 array with Python
 * scalars. */
static std::vector<float> flow_sigmas(const RadVideoGenDecl& d, int32_t n, int64_t tokens) {
    std::vector<float> s((size_t)n);
    const double stop = 1.0 / n, step = n > 1 ? (stop - 1.0) / (n - 1) : 0.0;
    for (int32_t i = 0; i < n; ++i) s[(size_t)i] = (float)(i == n - 1 && n > 1 ? stop : i * step + 1.0);
    if (d.dyn_shift) {
        const double m = (d.max_shift - d.base_shift) / (double)(d.max_seq - d.base_seq);
        const double mu = (double)tokens * m + (d.base_shift - m * (double)d.base_seq);
        const float e = (float)std::exp(mu);
        for (float& v : s) v = e / (e + (1.0f / v - 1.0f));
    }
    if (d.shift_terminal > 0.0) {
        const float sf = (1.0f - s.back()) / (float)(1.0 - d.shift_terminal);
        for (float& v : s) v = 1.0f - (1.0f - v) / sf;
    }
    s.push_back(0.0f);
    return s;
}

VideoSchedule video_schedule(const RadVideoGenDecl& d, int32_t steps, int64_t video_tokens,
                             bool refine, double shift_video, double shift_audio) {
    /* A request's shift replaces the declaration's, form 0 only. */
    const double sv = shift_video > 0.0 ? shift_video : d.shift_video;
    const double sa = shift_audio > 0.0 ? shift_audio : d.shift_audio;
    VideoSchedule v;
    if (d.sched_kind == 1) {
        if (d.n_rungs > 0) {
            if (steps != d.n_rungs) return v;
            for (int32_t i = 0; i < d.n_rungs; ++i) v.sig_v.push_back((float)d.rungs[i]);
            if (refine)
                for (int32_t i = 0; i < d.n_refine_rungs; ++i)
                    v.sig_v.push_back((float)d.refine_rungs[i]);
            v.sig_v.push_back(0.0f);
        } else {
            if (steps < 1) return v;
            v.sig_v = flow_sigmas(d, steps, video_tokens);
        }
        v.sig_a = v.sig_v;
        v.steps = (int32_t)v.sig_v.size() - 1;
        return v;
    }
    if (d.n_rungs > 0) {
        if (steps != d.n_rungs + 1) return v;
        v.sig_v = rung_sigmas(sv, d.rungs, d.n_rungs);
        v.sig_a = rung_sigmas(sa, d.rungs, d.n_rungs);
        v.steps = d.n_rungs;
        return v;
    }
    v.sig_v = shifted_sigmas(sv, steps);
    v.sig_a = shifted_sigmas(sa, steps);
    const size_t n = std::min(v.sig_v.size(), v.sig_a.size());
    v.steps = n > 0 ? (int32_t)(n - 1) : 0;
    return v;
}

/* ================================================================== sizing */
namespace {

constexpr int64_t kAlign = 256;

/* The segment boundaries a step with references may have: text, keyframes, a segment or two a
 * reference at nine pictures, three videos and three soundtracks, the two latents, the end. */
constexpr int64_t kSegCap = 64;
/* An encode pass's geometry, rad_arch_encode_geom's int32s at most. */
constexpr int64_t kEncGeomCap = 32;

/* Region offsets and sizes, in the enum's order (video_driver.h), and the total; shared by the
 * allocation and the VRAM budget. `ep` and `pd` are the vision tower's patches a pass and patch
 * width, 0 without one. */
int64_t carve(const RadVideoGenDecl& d, int64_t max_tok, int64_t ep, int64_t pd,
              int64_t (*bytes)[2], int n) {
    /* A guided model's VCOND carries the negative prompt after the positive. */
    const int64_t prompts = d.guidance ? 2 : 1;
    /* A schedule's steps: a request's, at most max_steps, plus the refinement rungs an upscaling
     * request appends after them (video_schedule). */
    const int64_t sched_cap = (int64_t)d.max_steps + std::max<int32_t>(d.n_refine_rungs, 0);
    const int64_t want[] = {
        prompts * d.max_text * 4,   /* R_TOK */
        prompts * d.max_text * 4,   /* R_POS */
        3 * 4,                   /* R_CU */
        3 * max_tok * 4,         /* R_ROPE: a step's positions, or the prompt's M-RoPE */
        RAD_VGEOM_N * 4,         /* R_GEOM */
        16 * 4,                  /* R_SCHED */
        sched_cap * 4,           /* R_TV */
        sched_cap * 4,           /* R_TA */
        4 * 4,                   /* R_SEED */
        4,                       /* R_LANE */
        4,                       /* R_FIRST */
        kSegCap * 4,             /* R_SEG */
        2 * kSegCap * 4,         /* R_MOD: the boundaries, then a cell a group */
        4 * 4,                   /* R_LAT */
        4,                       /* R_UP */
        d.max_text * 4,          /* R_MMROW: a row a placeholder token */
        ep * pd * 2,             /* R_ENCPIX: [ep, pd] bf16 */
        4 * ep * 4,              /* R_ENCCRD: [4, ep] i32 */
        (ep + 2) * 4,            /* R_ENCCU */
        d.max_enc_frames * d.max_enc_side * d.max_enc_side * 3 * 4,   /* R_ENCPX: f32 */
        d.max_enc_samples * d.audio_channels * 4,                     /* R_ENCSM: f32 */
        kEncGeomCap * 4,         /* R_ENCG */
        d.image_cond ? d.max_area * 3 * 2 : 0,   /* R_IMG: a first frame, bf16 */
        8 * 4,                   /* R_GUIDE */
    };
    int64_t off = 0;
    for (int r = 0; r < n; ++r) {
        bytes[r][0] = off;
        bytes[r][1] = want[r];
        off = align_up(off + want[r], kAlign);
    }
    return off;
}

int64_t frames_bytes_of(const RadVideoGenDecl& d) {
    return (d.frames_per_chunk + d.frames_tail) * d.max_area * 3;
}

/* The samples a channel of a soundtrack under `frames` frames: its latents rounded up, which
 * bounds the rounding video_shape does. */
int64_t samples_of(const RadVideoGenDecl& d, int32_t frames) {
    const int64_t num = (int64_t)frames * d.fps_den * d.audio_rate;
    return (num / d.fps_num + 1) * (d.sample_rate / d.audio_rate);
}

/* The vision tower's patches a pass and patch width, 0 without a tower. */
void encoder_bounds(const Program& prog, int64_t* ep, int64_t* pd) {
    *ep = prog.encoder.modalities != 0 ? prog.ctx.max_enc_patches : 0;
    *pd = prog.encoder.modalities != 0 ? prog.encoder.patch_dim : 0;
}
}  /* namespace */

int64_t VideoDriver::device_bytes(const Program& prog) {
    if (!prog.is_video_generator()) return 0;
    int64_t ep = 0, pd = 0;
    encoder_bounds(prog, &ep, &pd);
    int64_t b[R_N][2];
    return carve(prog.video_gen, prog.ctx.max_tok, ep, pd, b, R_N);
}

int VideoDriver::init(const Program& prog, PositionsFn positions,
                      const std::vector<RankIO>& ranks, PackFn pack, EncodeGeomFn enc_geom) {
    fini();
    if (!prog.is_video_generator()) return RAD_E_STATE;
    decl_  = prog.video_gen;
    rungs_ = prog.vgen_str.rungs;
    decl_.rungs = rungs_.empty() ? nullptr : rungs_.data();
    name_  = prog.vgen_str.name;
    ranks_ = ranks;
    positions_ = positions;
    pack_ = pack;
    enc_geom_ = enc_geom;
    max_tok_ = prog.ctx.max_tok;
    encoder_bounds(prog, &max_enc_, &patch_dim_);
    if (!positions_) {
        RAD_ERR("video generator '%s': the plugin exports no rad_arch_video_positions, which a "
                "video generator must", name_.c_str());
        return RAD_E_UNSUPPORTED;
    }
    decoder_names_ = prog.vgen_str.decoders;
    decoder_area_ = prog.vgen_str.decoder_max_area;
    refine_rungs_ = prog.vgen_str.refine_rungs;
    decl_.refine_rungs = refine_rungs_.empty() ? nullptr : refine_rungs_.data();
    default_negative_ = prog.vgen_str.default_negative;
    decl_.default_negative = default_negative_.c_str();
    const int64_t prompts = decl_.guidance ? 2 : 1;
    if (max_tok_ < prompts * decl_.max_text) {
        RAD_ERR("video generator '%s': a pass carries %lld rows and %s may be %lld tokens",
                name_.c_str(), (long long)max_tok_, prompts == 2 ? "a prompt and its negative" :
                "a prompt", (long long)(prompts * decl_.max_text));
        return RAD_E_UNSUPPORTED;
    }
    int64_t b[R_N][2];
    bytes_ = carve(decl_, max_tok_, max_enc_, patch_dim_, b, R_N);
    for (int r = 0; r < R_N; ++r) { reg_[r].off = b[r][0]; reg_[r].bytes = b[r][1]; reg_[r].used = 0; }

    frames_bytes_ = frames_bytes_of(decl_);
    audio_bytes_ = samples_of(decl_, (int32_t)(decl_.frames_per_chunk * decl_.max_chunks +
                                               decl_.frames_tail)) *
                   decl_.audio_channels * 4;
    host_ = static_cast<char*>(rad_dev_alloc(bytes_, RAD_MEM_HOST_PINNED));
    frames_host_ = static_cast<uint8_t*>(rad_dev_alloc(frames_bytes_, RAD_MEM_HOST_PINNED));
    audio_host_ = static_cast<float*>(rad_dev_alloc(audio_bytes_, RAD_MEM_HOST_PINNED));
    duration_host_ = static_cast<float*>(rad_dev_alloc(64, RAD_MEM_HOST_PINNED));
    if (!host_ || !frames_host_ || !audio_host_ || !duration_host_) {
        RAD_ERR("video generator: no pinned host memory for the pass staging and outputs (%lld + "
                "%lld + %lld bytes)", (long long)bytes_, (long long)frames_bytes_,
                (long long)audio_bytes_);
        return RAD_E_NOMEM;
    }
    std::memset(host_, 0, (size_t)bytes_);
    dev_.assign(ranks_.size(), nullptr);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r].device));
        dev_[r] = rad_dev_alloc(bytes_, RAD_MEM_DEVICE);
        if (!dev_[r]) {
            RAD_ERR("video generator: rank %zu has no room for the pass staging (%lld bytes)", r,
                    (long long)bytes_);
            return RAD_E_NOMEM;
        }
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0].device));
    out_.assign(ranks_.size(), RadBatch{});
    vout_.assign(ranks_.size(), RadVideoGenBatch{});
    free_lanes_.clear();
    for (int64_t i = decl_.max_lanes - 1; i >= 0; --i) free_lanes_.push_back((int32_t)i);
    return RAD_OK;
}

void VideoDriver::fini() {
    for (size_t r = 0; r < dev_.size(); ++r)
        if (dev_[r]) {
            if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
            rad_dev_free(dev_[r], RAD_MEM_DEVICE);
        }
    dev_.clear();
    if (!ranks_.empty()) rad_dev_set(ranks_[0].device);
    if (host_) rad_dev_free(host_, RAD_MEM_HOST_PINNED);
    if (frames_host_) rad_dev_free(frames_host_, RAD_MEM_HOST_PINNED);
    if (audio_host_) rad_dev_free(audio_host_, RAD_MEM_HOST_PINNED);
    if (duration_host_) rad_dev_free(duration_host_, RAD_MEM_HOST_PINNED);
    duration_host_ = nullptr;
    host_ = nullptr;
    frames_host_ = nullptr;
    audio_host_ = nullptr;
}

/* ================================================================== requests */
/* A length head's prediction (RadVideoGenDecl::duration_out) in chunks: diffusers'
 * LTX2DurationHead.predict_num_frames on the chunk grid -- the seconds clamped to the declared
 * bounds, rounded to frames, floored to the grid and lifted one chunk when that falls under the
 * minimum -- then held to min_chunks..max_chunks. `note` says what was clamped. */
int32_t VideoDriver::predicted_chunks(double log_seconds, std::string* note) const {
    const RadVideoGenDecl& d = decl_;
    const double fps = (double)d.fps_num / (double)d.fps_den;
    float e = std::exp((float)log_seconds);
    if (d.duration_dtype == RAD_BF16) e = rad_bf16_to_f32(rad_f32_to_bf16(e));
    const double sec = e;
    auto round_even = [](double v) { return (int64_t)std::nearbyint(v); };
    const int64_t lo = std::max<int64_t>(1, round_even(d.duration_min * fps));
    const int64_t hi = round_even(d.duration_max * fps);
    const int64_t want = std::isfinite(sec) ? round_even(sec * fps) : hi;
    const int64_t frames = std::max(lo, std::min(want, hi));
    int64_t n = std::max<int64_t>(0, (frames - d.frames_tail) / d.frames_per_chunk);
    if (d.frames_per_chunk * n + d.frames_tail < lo &&
        d.frames_per_chunk * (n + 1) + d.frames_tail <= hi)
        ++n;
    std::string what = fmt(": %.3f s predicted", sec);
    if (!(sec >= d.duration_min && sec <= d.duration_max))
        what += fmt(", held to %g..%g s", d.duration_min, d.duration_max);
    if (n < d.min_chunks || n > d.max_chunks) {
        n = std::max<int64_t>(d.min_chunks, std::min<int64_t>(n, d.max_chunks));
        what += fmt(", within the %lld..%lld frames this model makes",
                    (long long)(d.frames_per_chunk * d.min_chunks + d.frames_tail),
                    (long long)(d.frames_per_chunk * d.max_chunks + d.frames_tail));
    }
    if (note) *note = what;
    return (int32_t)n;
}

/* The encode passes and the reference table (RadVideoGenBatch::ref_tab) of a request's media, in
 * the order of `refs`. A video's soundtrack is in the video's row and encoded under its number.
 * The latent counts are rad_arch_encode_geom's, which refuses here the media an encode pass would
 * fail the engine on. */
int VideoDriver::plan_refs(Job& j, const VideoRequest& r, int32_t frames, std::string* why) const {
    const RadVideoGenDecl& d = decl_;
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    j.ref_tab.clear();
    j.enc_q.clear();
    j.enc_pre = 0;
    if (r.refs.empty()) return RAD_OK;
    if (!pack_ || !enc_geom_)
        return no("this model conditions on no reference media");
    int32_t eg[kEncGeomCap];
    /* A picture's or video's latent frames or a soundtrack's latents, -1 for media refused. */
    auto latents = [&](const VideoRefMedia& m) -> int64_t {
        const bool audio = m.kind == VideoRefMedia::AUDIO || m.kind == VideoRefMedia::VIDEO_AUDIO;
        if (audio ? d.max_enc_samples <= 0 : d.max_enc_side <= 0) return -1;
        if (audio ? m.enc_samples.empty() : m.enc_pixels.empty()) return -1;
        const int64_t in[4] = { audio ? m.enc_n : m.enc_frames, audio ? 0 : m.enc_h,
                                audio ? 0 : m.enc_w, audio ? m.enc_channels : 3 };
        const int32_t kind = audio ? RAD_GEN_PASS_AENCODE : RAD_GEN_PASS_VENCODE;
        if (enc_geom_(kind, in, eg, kEncGeomCap) < 0) return -1;
        return audio ? eg[12] : eg[24];
    };
    auto refused = [&](const VideoRefMedia& m) {
        return no("'" + m.name + "' is not media this model's encoders take (" +
                  (m.kind == VideoRefMedia::AUDIO || m.kind == VideoRefMedia::VIDEO_AUDIO
                       ? std::to_string(m.enc_n) + " samples"
                       : std::to_string(m.enc_w) + "x" + std::to_string(m.enc_h) + ", " +
                             std::to_string(m.enc_frames) + " frames") + ")");
    };
    auto grid = [&](const VideoRefMedia& m, int32_t* lh, int32_t* lw) {
        if (d.enc_ratio < 1 || m.enc_h % d.enc_ratio || m.enc_w % d.enc_ratio) return false;
        *lh = m.enc_h / d.enc_ratio;
        *lw = m.enc_w / d.enc_ratio;
        return true;
    };
    std::vector<int32_t> keys;
    j.ref_tab.assign(2, 0);
    for (size_t i = 0; i < r.refs.size(); ++i) {
        const VideoRefMedia& m = r.refs[i];
        if (m.kind == VideoRefMedia::VIDEO_AUDIO) continue;
        const int64_t t = latents(m);
        int32_t lh = 0, lw = 0;
        if (t < 1 || (m.kind != VideoRefMedia::AUDIO && !grid(m, &lh, &lw))) return refused(m);
        if (m.kind == VideoRefMedia::FIRST_FRAME || m.kind == VideoRefMedia::LAST_FRAME) {
            if (lh != j.geom.lat_h || lw != j.geom.lat_w)
                return no("'" + m.name + "' is not the video's size");
            const int32_t n = (int32_t)keys.size() / 3 + 1;
            j.enc_q.push_back({ RAD_ENC_KEYFRAME, n, (int32_t)i });
            keys.insert(keys.end(), { m.kind == VideoRefMedia::FIRST_FRAME ? 0 : frames - 1,
                                      (int32_t)t, 0 });
            continue;
        }
        int32_t kind = RAD_VREF_PICTURE, enc = RAD_ENC_IMAGE, sound = 0;
        int64_t cap = d.max_ref_images;
        if (m.kind == VideoRefMedia::AUDIO) {
            kind = RAD_VREF_AUDIO;
            enc = RAD_ENC_AUDIO;
            cap = d.max_ref_audios;
        } else if (m.kind == VideoRefMedia::VIDEO) {
            kind = RAD_VREF_VIDEO;
            enc = RAD_ENC_VIDEO;
            cap = d.max_ref_videos;
            for (size_t k = 0; k < r.refs.size(); ++k) {
                const VideoRefMedia& a = r.refs[k];
                if (a.kind != VideoRefMedia::VIDEO_AUDIO || a.index != m.index) continue;
                const int64_t at = latents(a);
                if (at < 1) return refused(a);
                sound = (int32_t)at;
                j.enc_q.push_back({ RAD_ENC_VAUDIO, m.index, (int32_t)k });
            }
        }
        if (m.index < 1 || m.index > cap)
            return no("'" + m.name + "' is reference " + std::to_string(m.index) + " of its kind "
                      "and this model takes " + std::to_string(cap));
        j.enc_q.push_back({ enc, m.index, (int32_t)i });
        ++j.ref_tab[0];
        j.ref_tab.insert(j.ref_tab.end(),
                         { kind, m.index, lh, lw, kind == RAD_VREF_AUDIO ? 0 : (int32_t)t,
                           kind == RAD_VREF_AUDIO ? (int32_t)t : sound });
    }
    j.ref_tab[1] = (int32_t)keys.size() / 3;
    j.ref_tab.insert(j.ref_tab.end(), keys.begin(), keys.end());
    j.enc_pre = j.enc_q.size();
    /* The prompt's vision pads: each picture's tokens, which ComfyUI modulates as video. */
    const size_t at = j.ref_tab.size();
    j.ref_tab.push_back(0);
    if (r.media)
        for (const mm::Placed& pl : r.media->items) {
            if (!pl.item || pl.item->kind != mm::Kind::Image) continue;
            const int64_t n = (int64_t)pl.item->expand.size();
            if (pl.tok < 0 || n < 1 || pl.tok + n > j.geom.n_text) continue;
            ++j.ref_tab[at];
            j.ref_tab.insert(j.ref_tab.end(), { pl.tok, (int32_t)n });
        }
    return RAD_OK;
}

/* A step's packed sequence with the request's references in, at geometry `g`. */
int VideoDriver::lay_out(const RadVideoGeom& g, const std::vector<int32_t>& tab, Pack* out,
                         std::string* why) const {
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    int32_t n_seg = 0, n_mod = 0;
    const int64_t rows = pack_(&g, tab.data(), nullptr, &n_seg, nullptr, &n_mod, nullptr, 0);
    if (rows <= 0 || n_seg < 2 || n_mod < 3 || n_mod % 2 == 0 || n_seg > kSegCap ||
        n_mod > 2 * kSegCap)
        return no("the model cannot lay these references out");
    if (rows > max_tok_)
        return no("a step with these references is " + std::to_string(rows) + " rows and a pass "
                  "carries " + std::to_string(max_tok_) + "; ask for a shorter or smaller video, "
                  "or fewer references");
    out->rows = rows;
    out->seg.assign((size_t)n_seg, 0);
    out->mod.assign((size_t)n_mod, 0);
    out->pos.assign((size_t)(3 * rows), 0.0f);
    if (pack_(&g, tab.data(), out->seg.data(), &n_seg, out->mod.data(), &n_mod, out->pos.data(),
              3 * rows) != rows)
        return no("the model cannot lay these references out");
    const int32_t cells = 1 + *std::max_element(out->mod.begin() + (n_mod + 1) / 2, out->mod.end());
    if (decl_.max_mod_cells > 0 && cells > decl_.max_mod_cells)
        return no("these references read " + std::to_string(cells) + " modulations and this "
                  "model's table holds " + std::to_string(decl_.max_mod_cells));
    return RAD_OK;
}

/* An upscaled request's keyframes, pinned on the grid its steps run on: encoded at the small size
 * before the prompt, then at the full size into the same slots once the upscale is done. */
int VideoDriver::plan_small_keys(Job& j, const VideoRequest& r, std::string* why) const {
    const RadVideoGenDecl& d = decl_;
    std::vector<Job::EncItem> q;
    int32_t eg[kEncGeomCap];
    for (const Job::EncItem& it : j.enc_q) {
        if (it.kind != RAD_ENC_KEYFRAME) continue;
        const VideoRefMedia& m = r.refs[(size_t)it.ref];
        const int64_t in[4] = { m.enc_frames, m.small_h, m.small_w, 3 };
        if (m.small_pixels.size() != (size_t)m.enc_frames * m.small_h * m.small_w * 3 ||
            d.enc_ratio < 1 || m.small_h % d.enc_ratio || m.small_w % d.enc_ratio ||
            m.small_h / d.enc_ratio != j.geom_small.lat_h ||
            m.small_w / d.enc_ratio != j.geom_small.lat_w ||
            enc_geom_(RAD_GEN_PASS_VENCODE, in, eg, kEncGeomCap) < 0) {
            if (why) *why = "'" + m.name + "' has no picture at the latent upscale's small size "
                            "the model's encoder takes";
            return RAD_E_INVAL;
        }
        q.push_back({ it.kind, it.slot, it.ref, true });
    }
    j.enc_pre = q.size();
    q.insert(q.end(), j.enc_q.begin(), j.enc_q.end());
    j.enc_q = std::move(q);
    return RAD_OK;
}

/* The request's shape at `chunks` long: its geometry, rows and positions, its schedule, the
 * upscale's split and what its ticket says it makes. Run at submit, and again for a request whose
 * length the model predicts once its VCOND pass has written the prediction. */
int VideoDriver::shape_job(Job& j, const VideoRequest& r, int32_t chunks, int32_t steps,
                           std::string* why) {
    const RadVideoGenDecl& d = decl_;
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    const int64_t T = (int64_t)r.tokens.size();
    const VideoShape s = video_shape(d, r.width, r.height, chunks);
    j.geom.n_text = (int32_t)(d.step_text_rows > 0 ? d.step_text_rows : T);
    j.geom.frames = s.frames;
    j.geom.lat_frames = s.lat_frames;
    j.geom.lat_h = s.lat_h;
    j.geom.lat_w = s.lat_w;
    j.geom.n_audio = s.n_audio;
    j.rows = j.geom.n_text + s.audio_rows + s.video_rows;
    if (j.rows > max_tok_)
        return no("a step of this video is " + std::to_string(j.rows) + " rows and a pass "
                  "carries " + std::to_string(max_tok_) + "; ask for a shorter or smaller video");
    const int64_t nr = positions_(RAD_VPOS_STEP, &j.geom, nullptr, 0);
    const int64_t nd = positions_(RAD_VPOS_DECODE, &j.geom, nullptr, 0);
    if (nr != j.rows || nd <= 0 || nd > max_tok_)
        return no("the model lays this video out in " + std::to_string(nr) + " step rows and " +
                  std::to_string(nd) + " decoder rows, which this server cannot run");
    j.dec_rows = nd;
    j.pos.assign((size_t)(3 * nr), 0.0f);
    if (positions_(RAD_VPOS_STEP, &j.geom, j.pos.data(), nr) != nr)
        return no("the model could not lay this video out");
    RAD_TRY(plan_refs(j, r, s.frames, why));
    j.pack = Pack{};
    j.pack_small = Pack{};
    if (!j.ref_tab.empty()) RAD_TRY(lay_out(j.geom, j.ref_tab, &j.pack, why));
    /* The schedule follows the upscale's split, which needs the small size first. */
    const bool upscaling = r.upscale > 1.0;
    const bool own_refine = upscaling && d.n_refine_rungs > 0;
    int64_t first_tokens = s.video_rows;
    if (upscaling) {
        int32_t sw = 0, sh = 0;
        video_small_size(d, r.width, r.height, r.upscale, &sw, &sh);
        first_tokens = video_shape(d, sw, sh, chunks).video_rows;
    }
    j.sched = video_schedule(d, steps, first_tokens, own_refine, r.shift_video, r.shift_audio);
    if (j.sched.steps < 1) return no("the schedule has no steps");
    int32_t first_w = r.width, first_h = r.height;
    if (upscaling) {
        if (!(d.upscale_max > 1.0)) return no("this model has no latent upscaler");
        if (d.upscale_exact) {
            const int64_t m = (int64_t)std::llround(d.size_multiple * d.upscale_max);
            if (r.upscale != d.upscale_max || r.width % m || r.height % m)
                return no("this model's latent upscaler enlarges by exactly " +
                          fmt("%g", d.upscale_max) + ": latent_upscale must be that, and both "
                          "sides multiples of " + std::to_string(m));
        }
        if (r.upscale > d.upscale_max)
            return no("latent_upscale is at most " + fmt("%g", d.upscale_max) + " for this model");
        int32_t sw = 0, sh = 0;
        video_small_size(d, r.width, r.height, r.upscale, &sw, &sh);
        if (sw == r.width && sh == r.height)
            return no("latent_upscale " + fmt("%g", r.upscale) + " leaves " +
                      std::to_string(r.width) + "x" + std::to_string(r.height) + " as it is: "
                      "the small size is each side over the factor, rounded to a multiple of " +
                      std::to_string(d.size_multiple));
        const int32_t evals = j.sched.steps;
        int32_t refine = r.refine;
        if (own_refine) {
            if (refine != 0 && refine != d.n_refine_rungs)
                return no("upscale_refine_steps must be " + std::to_string(d.n_refine_rungs) +
                          ": this model refines the upscaled video over the schedule it was "
                          "trained on");
            refine = d.n_refine_rungs;
        }
        if (refine == 0) refine = std::min<int32_t>(d.upscale_refine, evals - 1);
        if (refine < 1 || refine >= evals)
            return no("upscale_refine_steps must be 1.." + std::to_string(evals - 1) + ": the "
                      "schedule takes " + std::to_string(evals) + " evaluations and the small "
                      "size at least one of them");
        const VideoShape ss = video_shape(d, sw, sh, chunks);
        j.geom_small = j.geom;
        j.geom_small.lat_h = ss.lat_h;
        j.geom_small.lat_w = ss.lat_w;
        j.rows_small = j.geom.n_text + ss.audio_rows + ss.video_rows;
        const int64_t ns = positions_(RAD_VPOS_STEP, &j.geom_small, nullptr, 0);
        if (ns != j.rows_small)
            return no("the model lays the small size " + std::to_string(sw) + "x" +
                      std::to_string(sh) + " out in " + std::to_string(ns) + " step rows, which "
                      "this server cannot run");
        j.pos_small.assign((size_t)(3 * ns), 0.0f);
        if (positions_(RAD_VPOS_STEP, &j.geom_small, j.pos_small.data(), ns) != ns)
            return no("the model could not lay the small size out");
        if (!j.ref_tab.empty()) RAD_TRY(lay_out(j.geom_small, j.ref_tab, &j.pack_small, why));
        if (!j.ref_tab.empty() && j.ref_tab[1] > 0) RAD_TRY(plan_small_keys(j, r, why));
        j.split = evals - refine;
        j.factor = (float)(((double)r.width / sw + (double)r.height / sh) / 2.0);
        r.ticket->small_width = sw;
        r.ticket->small_height = sh;
        r.ticket->refine = refine;
        first_w = sw;
        first_h = sh;
    }
    if (!r.image.empty()) {
        if (!d.image_cond) return no("this model cannot start from a picture");
        if (r.image.size() != (size_t)first_w * first_h * 3)
            return no("the first frame is not " + std::to_string(first_w) + "x" +
                      std::to_string(first_h) + " RGB");
    }
    j.n_chunks = chunks;
    j.n_samples = (int64_t)s.n_audio * (d.sample_rate / d.audio_rate);
    VideoTicket& t = *r.ticket;
    t.frames = s.frames;
    t.steps = j.sched.steps;
    t.rows = j.pack.rows > 0 ? j.pack.rows : j.rows;
    return RAD_OK;
}

int VideoDriver::submit(std::unique_ptr<VideoRequest> r, std::string* why) {
    if (!r || !r->ticket) return RAD_E_INVAL;
    const RadVideoGenDecl& d = decl_;
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    const int64_t T = (int64_t)r->tokens.size();
    if (T < 1 || T > d.max_text)
        return no("the prompt is " + std::to_string(T) + " tokens; this model takes 1.." +
                  std::to_string(d.max_text));
    /* A reference picture is shown to the text encoder too: `media` places it in the prompt. */
    for (const VideoRefMedia& m : r->refs)
        if (m.kind == VideoRefMedia::IMAGE && !r->media)
            return no("'" + m.name + "' is a reference picture and the prompt does not show it");
    if (r->media) {
        int64_t n_patch = 0;
        for (const mm::Placed& p : r->media->items) {
            if ((int64_t)p.tok + (int64_t)p.item->expand.size() > T)
                return no("a reference's placeholder run is placed past the end of the prompt");
            n_patch += p.item->n_patch;
        }
        if (n_patch > max_enc_)
            return no("the pictures are " + std::to_string(n_patch) + " patches; this model "
                      "encodes at most " + std::to_string(max_enc_) + " in one pass");
    }
    if (r->chunks != 0 && (r->chunks < d.min_chunks || r->chunks > d.max_chunks))
        return no("the length must be " + std::to_string(d.min_chunks) + ".." +
                  std::to_string(d.max_chunks) + " chunks");
    const int64_t W = r->width, H = r->height;
    if (W % d.size_multiple || H % d.size_multiple || W < d.min_side || H < d.min_side ||
        W * H > d.max_area || std::max(W, H) > d.max_aspect * std::min(W, H))
        return no("the size " + std::to_string(W) + "x" + std::to_string(H) + " is not one this "
                  "model makes: sides multiples of " + std::to_string(d.size_multiple) + " from " +
                  std::to_string(d.min_side) + ", at most " + std::to_string(d.max_area) +
                  " pixels, the long side at most " + std::to_string(d.max_aspect) +
                  " times the short");
    const int32_t steps = r->steps > 0 ? r->steps : (int32_t)d.default_steps;
    const bool evals = d.sched_kind == 1;   /* form 1 counts evaluations, form 0 grid points */
    const int32_t rung_steps = evals ? d.n_rungs : d.n_rungs + 1;
    if (d.n_rungs > 0 && steps != rung_steps)
        return no("num_inference_steps must be " + std::to_string(rung_steps) + ": this model "
                  "runs the schedule it was trained on, " + std::to_string(d.n_rungs) +
                  " evaluations");
    if (steps < (evals ? 1 : 2) || steps > d.max_steps)
        return no("num_inference_steps must be " + std::to_string(evals ? 1 : 2) + ".." +
                  std::to_string(d.max_steps) + (evals ? " (model evaluations)" :
                  " (sigma grid points, the last one 0)"));

    auto j = std::make_unique<Job>();
    /* Guidance: the request's terms over the model's defaults; a negative prompt only when a
     * scale asks for the unconditional branch. */
    bool named = r->has_neg;
    for (int i = 0; i < 8; ++i) {
        named = named || !std::isnan(r->guidance[i]);
        j->guidance[i] = (float)(std::isnan(r->guidance[i]) ? d.guidance_default[i] : r->guidance[i]);
        if (!(j->guidance[i] >= 0.0f) || !std::isfinite(j->guidance[i]))
            return no("guidance terms must be finite and not negative");
    }
    if (!d.guidance && named)
        return no("this model takes no guidance or negative prompt");
    j->n_neg = 0;
    if (d.guidance && (j->guidance[0] != 1.0f || j->guidance[4] != 1.0f)) {
        const int64_t N = (int64_t)r->neg_tokens.size();
        if (N < 1 || N > d.max_text)
            return no("the negative prompt is " + std::to_string(N) + " tokens; this model takes "
                      "1.." + std::to_string(d.max_text));
        j->n_neg = (int32_t)N;
    }
    if (r->decoder < 0 || (r->decoder > 0 && (size_t)r->decoder >= decoder_area_.size()))
        return no("this model has no decoder " + std::to_string(r->decoder));
    if (r->decoder > 0 && W * H > decoder_area_[(size_t)r->decoder])
        return no("the " + decoder_names_[(size_t)r->decoder] + " decoder makes frames of " +
                  std::to_string(decoder_area_[(size_t)r->decoder]) + " pixels at most");
    j->auto_len = r->chunks == 0 && d.duration_out != RAD_NULL_HANDLE;
    const int32_t chunks = r->chunks == 0 ? (int32_t)d.default_chunks : r->chunks;
    if (shape_job(*j, *r, chunks, steps, why) < 0) return RAD_E_INVAL;

    VideoTicket& t = *r->ticket;
    t.width = r->width;
    t.height = r->height;
    t.sample_rate = d.sample_rate;
    t.channels = d.audio_channels;
    t.fps = (double)d.fps_num / (double)d.fps_den;
    t.in_tokens = T;
    t.t_submit_ns = (int64_t)rad_mono_ns();
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) {
        if (why) *why = closed_why_;
        return closed_status_;
    }
    j->id = next_id_++;
    j->req = std::move(r);
    jobs_.push_back(std::move(j));
    return RAD_OK;
}

void VideoDriver::shutdown(const char* why, int status) {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    closed_why_ = why ? why : "the engine stopped";
    closed_status_ = status;
    for (auto& j : jobs_) j->req->ticket->finish(status, closed_why_);
}

int64_t VideoDriver::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (!j->admitted) ++n;
    return n;
}

int64_t VideoDriver::running() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (j->admitted) ++n;
    return n;
}

VideoDriver::Job* VideoDriver::job_locked(uint64_t id) {
    for (auto& j : jobs_) if (j->id == id) return j.get();
    return nullptr;
}

void VideoDriver::release_locked(Job& j) {
    if (j.lane >= 0) {
        free_lanes_.push_back(j.lane);
        j.lane = -1;
    }
}

void VideoDriver::fail_locked(Job& j, int status, const std::string& why) {
    j.failed = true;
    j.req->ticket->finish(status, why);
}

/* In arrival order: a request needs only a lane. */
void VideoDriver::admit_locked() {
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted || j.failed) continue;
        if (free_lanes_.empty()) break;
        j.lane = free_lanes_.back();
        free_lanes_.pop_back();
        j.admitted = true;
    }
}

/* ================================================================== staging */
int VideoDriver::upload() {
    for (size_t r = 0; r < ranks_.size(); ++r)
        for (int g = 0; g < R_N; ++g) {
            if (reg_[g].used <= 0) continue;
            RAD_TRY(rad_memcpy_async((char*)dev_[r] + reg_[g].off, host_ + reg_[g].off,
                                     reg_[g].used, ranks_[r].stream));
        }
    return RAD_OK;
}

/* The geometry a pass's ops read off the device (RAD_VGEOM_*) for a pass at geometry `g` of
 * `rows` step rows, with chunk `chunk`'s span when it is a decode's and both grids when the job
 * upscales. */
void VideoDriver::stage_geom(const Job& j, const RadVideoGeom& g0, int64_t rows, int32_t chunk,
                             const Pack* pack) {
    int32_t* g = hptr<int32_t>(R_GEOM);
    std::memset(g, 0, RAD_VGEOM_N * 4);
    g[RAD_VGEOM_LAT_FRAMES] = g0.lat_frames;
    g[RAD_VGEOM_LAT_H] = g0.lat_h;
    g[RAD_VGEOM_LAT_W] = g0.lat_w;
    g[RAD_VGEOM_N_TEXT] = g0.n_text;
    g[RAD_VGEOM_N_AUDIO] = g0.n_audio;
    g[RAD_VGEOM_CHUNK_LAT0] = (int32_t)(decl_.latents_per_chunk * chunk);
    g[RAD_VGEOM_CHUNK_FRAMES] = (int32_t)decl_.dec_latents;
    g[RAD_VGEOM_CHUNK_H] = g0.lat_h;
    g[RAD_VGEOM_CHUNK_W] = g0.lat_w;
    g[RAD_VGEOM_CHUNK_FIRST] = chunk == 0;
    g[RAD_VGEOM_CHUNK_LAST] = chunk == j.n_chunks - 1;
    g[RAD_VGEOM_PATCH_FRAMES] = (int32_t)decl_.dec_latents;
    g[RAD_VGEOM_PATCH_H] = (int32_t)(g0.lat_h / decl_.patch);
    g[RAD_VGEOM_PATCH_W] = (int32_t)(g0.lat_w / decl_.patch);
    if (j.split > 0) {
        const RadVideoGeom& a = j.geom_small;
        const RadVideoGeom& b = j.geom;
        const int32_t p = (int32_t)decl_.patch;
        const int32_t up[] = { a.lat_frames, a.lat_h, a.lat_w, b.lat_h, b.lat_w,
                               b.lat_frames, b.lat_h, b.lat_w,
                               a.lat_frames, a.lat_h / p, a.lat_w / p,
                               b.lat_frames, b.lat_h / p, b.lat_w / p };
        std::memcpy(g + RAD_VGEOM_UP_SRC, up, sizeof up);
    }
    g[RAD_VGEOM_AUDIO_VOL] = g0.n_audio;
    g[RAD_VGEOM_AUDIO_VOL + 1] = 1;
    g[RAD_VGEOM_AUDIO_VOL + 2] = 1;
    const int32_t ppf = (int32_t)((g0.lat_h / decl_.patch) * (g0.lat_w / decl_.patch));
    g[RAD_VGEOM_FIRST_CU] = 0;
    g[RAD_VGEOM_FIRST_CU + 1] = ppf;
    g[RAD_VGEOM_FIRST_CU + 2] = ppf * g0.lat_frames;
    reg_[R_GEOM].used = RAD_VGEOM_N * 4;
    const int64_t audio_rows =
        (decl_.audio_streams > 0 ? decl_.audio_streams : decl_.audio_channels) * (int64_t)g0.n_audio;
    const int64_t video_rows = rows - g0.n_text - audio_rows;
    *hptr<int32_t>(R_LANE) = j.lane;
    *hptr<int32_t>(R_FIRST) = j.step == 0;
    /* A step with references takes the plugin's segments and modulation groups; every other pass
     * the plain sequence's segments, which are its modulation groups as well. */
    const std::vector<int32_t> plain = { 0, g0.n_text, (int32_t)(g0.n_text + audio_rows),
                                         (int32_t)rows };
    const std::vector<int32_t>& seg = pack ? pack->seg : plain;
    const std::vector<int32_t>& mod = pack ? pack->mod : plain;
    std::memcpy(hptr<int32_t>(R_SEG), seg.data(), seg.size() * 4);
    std::memcpy(hptr<int32_t>(R_MOD), mod.data(), mod.size() * 4);
    reg_[R_SEG].used = (int64_t)seg.size() * 4;
    reg_[R_MOD].used = (int64_t)mod.size() * 4;
    int32_t* lat = hptr<int32_t>(R_LAT);
    lat[0] = 0;
    lat[1] = (int32_t)video_rows;
    lat[2] = 0;
    lat[3] = (int32_t)audio_rows;
    std::memcpy(hptr<float>(R_GUIDE), j.guidance, sizeof j.guidance);
    reg_[R_GUIDE].used = sizeof j.guidance;
    reg_[R_LANE].used = 4;
    reg_[R_FIRST].used = 4;
    reg_[R_LAT].used = 4 * 4;
}

void VideoDriver::base_batches(const Job& j, const RadVideoGeom& g, int64_t n_tok, int step) {
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        RadVideoGenBatch& V = vout_[r];
        const int ri = (int)r;
        B = RadBatch{};
        B.step = step;
        B.phase = RAD_PHASE_PREFILL;
        B.n_tok = n_tok;
        B.n_seq = 1;
        B.max_q_len = (int32_t)n_tok;
        B.max_q_len_prefill = (int32_t)n_tok;
        B.token_ids = dptr<int32_t>(ri, R_TOK);
        B.positions = dptr<int32_t>(ri, R_POS);
        B.cu_seqlens = dptr<int32_t>(ri, R_CU);
        V = RadVideoGenBatch{};
        V.lane = j.lane;
        V.geom = g;
        V.geom_from = j.geom_small;
        V.geom_d = dptr<int32_t>(ri, R_GEOM);
        V.lane_d = dptr<int32_t>(ri, R_LANE);
        V.first_d = dptr<int32_t>(ri, R_FIRST);
        V.seg_cu = dptr<int32_t>(ri, R_SEG);
        V.mod_cu = dptr<int32_t>(ri, R_MOD);
        V.lat_cu = dptr<int32_t>(ri, R_LAT);
        V.n_steps = j.sched.steps;
        V.t_video = dptr<float>(ri, R_TV);
        V.t_audio = dptr<float>(ri, R_TA);
        V.step = j.step;
        V.sampler = j.req->sampler;
        V.sched = dptr<float>(ri, R_SCHED);
        V.pos = dptr<float>(ri, R_ROPE);
        V.seed = dptr<uint32_t>(ri, R_SEED);
        V.chunk = j.chunk;
        V.n_chunks = j.n_chunks;
        V.dec_rows = j.dec_rows;
        V.n_samples = j.n_samples;
        V.upscale = dptr<float>(ri, R_UP);
        V.ref_tab = j.ref_tab.empty() ? nullptr : j.ref_tab.data();
        V.guidance_h = j.guidance;
        V.guidance = dptr<float>(ri, R_GUIDE);
        V.has_image = !j.req->image.empty();
        V.decoder = j.req->decoder;
        B.vgen_b = &V;
    }
}

/* ================================================================== the passes */
const RadBatch* VideoDriver::next(int step) {
    std::lock_guard<std::mutex> lk(mu_);
    pass_ = Pass{};
    rb_.clear();
    if (closed_) return nullptr;
    /* Retire finished, failed and cancelled jobs, returning their lanes before admitting. Safe
     * because next() runs between passes, with nothing in flight. */
    for (auto it = jobs_.begin(); it != jobs_.end();) {
        Job& j = **it;
        const bool cancelled = j.req->ticket->cancel.load(std::memory_order_acquire);
        if (!(j.failed || cancelled || j.finished)) { ++it; continue; }
        if (cancelled && !j.failed && !j.finished)
            j.req->ticket->finish(RAD_E_CANCELLED, "the request was cancelled");
        release_locked(j);
        it = jobs_.erase(it);
    }
    admit_locked();

    /* 1. The decodes of a denoised lane: the soundtrack, then the chunks in order. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed || j.step < j.sched.steps) continue;
        if (!j.audio_done) return build_audio(j, step);
        if (j.chunk < j.n_chunks) return build_chunk(j, step);
    }
    /* 2. The encodes of an admitted request's media, before its prompt. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed || j.cond_done) continue;
        if (j.enc_next < j.enc_pre) return build_encode(j, step);
    }
    /* 3. A prompt. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted && !j.failed && !j.cond_done) return build_cond(j, step);
    }
    /* 4. The oldest lane's next step, or its upscale once its small steps are done, and after it
     * the keyframes again at the full size. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed || !j.cond_done || j.step >= j.sched.steps) continue;
        if (small_now(j) && j.step == j.split) return build_upscale(j, step);
        if (!small_now(j) && j.enc_next < j.enc_q.size()) return build_encode(j, step);
        return build_step(j, step);
    }
    return nullptr;
}

const RadBatch* VideoDriver::batch_for(int rank) const {
    if (rank < 0 || (size_t)rank >= out_.size() || pass_.kind == 0) return nullptr;
    return &out_[(size_t)rank];
}

const RadBatch* VideoDriver::build_cond(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    /* The prompt, then the negative prompt when the request is guided against one, each its own
     * sequence from position 0. */
    const int64_t T = (int64_t)j.req->tokens.size(), N = j.n_neg;
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    std::memcpy(tok, j.req->tokens.data(), (size_t)T * 4);
    if (N > 0) std::memcpy(tok + T, j.req->neg_tokens.data(), (size_t)N * 4);
    /* A left-padded encoder's prompt ends at text_left_pad - 1. */
    const int64_t lp = decl_.text_left_pad;
    for (int64_t i = 0; i < T; ++i) pos[i] = (int32_t)(lp > 0 ? lp - T + i : i);
    for (int64_t i = 0; i < N; ++i) pos[T + i] = (int32_t)(lp > 0 ? lp - N + i : i);
    int32_t* cu = hptr<int32_t>(R_CU);
    cu[0] = 0;
    cu[1] = (int32_t)T;
    cu[2] = (int32_t)(T + N);
    /* The first frame as the reference's image processor hands it to the VAE: x / 255 * 2 - 1 in
     * f32, then bf16. */
    const std::vector<uint8_t>& img = j.req->image;
    if (!img.empty()) {
        uint16_t* px = hptr<uint16_t>(R_IMG);
        for (size_t i = 0; i < img.size(); ++i)
            px[i] = rad_f32_to_bf16(2.0f * ((float)img[i] / 255.0f) - 1.0f);
        reg_[R_IMG].used = (int64_t)img.size() * 2;
    }
    if ((int64_t)j.sched.steps * 4 > reg_[R_TV].bytes) {
        fail_locked(j, RAD_E_STATE, "the schedule's " + std::to_string(j.sched.steps) +
                                        " steps are more than the timestep regions hold");
        return nullptr;
    }
    float* tv = hptr<float>(R_TV);
    float* ta = hptr<float>(R_TA);
    for (int32_t i = 0; i < j.sched.steps; ++i) {
        tv[i] = 1.0f - j.sched.sig_v[(size_t)i];
        ta[i] = 1.0f - j.sched.sig_a[(size_t)i];
    }
    reg_[R_TOK].used = (T + N) * 4;
    reg_[R_POS].used = (T + N) * 4;
    reg_[R_CU].used = (N > 0 ? 3 : 2) * 4;
    reg_[R_TV].used = j.sched.steps * 4;
    reg_[R_TA].used = j.sched.steps * 4;
    const RadVideoGeom& g = small_now(j) ? j.geom_small : j.geom;
    stage_geom(j, g, small_now(j) ? j.rows_small : j.rows, 0);

    /* The prompt's pictures for the vision tower, and its M-RoPE in R_ROPE. */
    StagedMedia sm;
    if (j.req->media) {
        const mm::PromptMedia& pm = *j.req->media;
        int32_t* rope = hptr<int32_t>(R_ROPE);
        for (int64_t i = 0; i < T; ++i) {
            int32_t p3[3] = { (int32_t)i, (int32_t)i, (int32_t)i };
            pm.rope.at(i, p3);
            rope[i] = p3[0];
            rope[T + i] = p3[1];
            rope[2 * T + i] = p3[2];
        }
        reg_[R_ROPE].used = 3 * T * 4;
        sm = stage_media(pm, patch_dim_, hptr<uint16_t>(R_ENCPIX), hptr<int32_t>(R_ENCCRD),
                         hptr<int32_t>(R_ENCCU), hptr<int32_t>(R_MMROW));
        reg_[R_ENCPIX].used = sm.n_patch * patch_dim_ * 2;
        reg_[R_ENCCRD].used = 4 * sm.n_patch * 4;
        reg_[R_ENCCU].used = (sm.n_seg + 1) * 4;
        reg_[R_MMROW].used = sm.n_rows * 4;
    }
    if (upload() < 0) return nullptr;
    base_batches(j, g, T + N, step);
    if (j.req->media)
        for (size_t r = 0; r < ranks_.size(); ++r) {
            RadBatch& B = out_[r];
            const int ri = (int)r;
            B.rope_pos = dptr<int32_t>(ri, R_ROPE);
            B.rope_mixed = 1;
            B.n_mm_rows = sm.n_rows;
            B.mm_rows = sm.n_rows ? dptr<int32_t>(ri, R_MMROW) : nullptr;
            B.enc = 0;
            B.enc_n_patch = sm.n_patch;
            B.enc_pixels = sm.n_patch ? dptr<uint16_t>(ri, R_ENCPIX) : nullptr;
            B.enc_coord = sm.n_patch ? dptr<int32_t>(ri, R_ENCCRD) : nullptr;
            B.enc_n_seg = sm.n_seg;
            B.enc_cu = sm.n_seg ? dptr<int32_t>(ri, R_ENCCU) : nullptr;
            B.enc_max_seg = sm.max_seg;
        }
    for (size_t r = 0; r < out_.size(); ++r) {
        RadBatch& B = out_[r];
        B.gen = RAD_GEN_PASS_VCOND;
        B.n_seq = N > 0 ? 2 : 1;
        B.max_q_len = B.max_q_len_prefill = (int32_t)std::max(T, N);
        vout_[r].n_neg = (int32_t)N;
        if (!img.empty()) vout_[r].image = dptr<uint16_t>((int)r, R_IMG);
    }
    if (j.auto_len) rb_.assign(1, Readback{ decl_.duration_out, 4, duration_host_ });
    pass_.kind = RAD_GEN_PASS_VCOND;
    pass_.job = j.id;
    pass_.lora_slot = j.req->lora_slot;
    return &out_[0];
}

/* Step i's timesteps, sigmas and res_multistep scalars (RadVideoGenBatch::sched). With
 * `to_clean` the next sigmas are 0, so the step lands on its clean estimate. res_multistep is
 * k-diffusion's at eta 0 (DPM-Solver++ 2M): with h = log(sigma / sigma_next) and
 * rs = log(sigma_prev / sigma) / h,
 *
 *   hb1 = (1 - r) (1 + 1 / (2 rs)),  hb2 = -(1 - r) / (2 rs),  order 2,
 *
 * and order 1 (Euler) on a step with no last estimate on its grid: the first, the first after an
 * upscale (i == split) and one to sigma 0. */
void VideoDriver::stage_sched(const Job& j, int32_t i, bool to_clean) {
    float* s = hptr<float>(R_SCHED);
    s[0] = 1.0f - j.sched.sig_v[(size_t)i];
    s[1] = j.sched.sig_v[(size_t)i];
    s[2] = to_clean ? 0.0f : j.sched.sig_v[(size_t)i + 1];
    s[3] = 1.0f - j.sched.sig_a[(size_t)i];
    s[4] = j.sched.sig_a[(size_t)i];
    s[5] = to_clean ? 0.0f : j.sched.sig_a[(size_t)i + 1];
    /* A stream's five: sigma, r, hb1, hb2, order. */
    auto coeffs = [&](const std::vector<float>& sig, float* out) {
        const float sg = sig[(size_t)i];
        const float sgn = to_clean ? 0.0f : sig[(size_t)i + 1];
        const float r = sg > 0.0f ? sgn / sg : 0.0f;
        float hb1 = 1.0f - r, hb2 = 0.0f, order = 1.0f;
        const bool two = sgn > 0.0f && i > 0 && !(j.split > 0 && i == j.split);
        if (two) {
            const float h = std::log(sg / sgn);
            const float h_last = std::log(sig[(size_t)i - 1] / sg);
            const float rs = h_last / h;
            hb1 = (1.0f - r) * (1.0f + 1.0f / (2.0f * rs));
            hb2 = -(1.0f - r) / (2.0f * rs);
            order = 2.0f;
        }
        out[0] = sg;
        out[1] = r;
        out[2] = hb1;
        out[3] = hb2;
        out[4] = order;
    };
    coeffs(j.sched.sig_v, s + 6);
    coeffs(j.sched.sig_a, s + 11);
    reg_[R_SCHED].used = 16 * 4;
}

/* The video's noise seed is the request's; the soundtrack's the same with its top bit flipped, so
 * the two streams never coincide. */
void VideoDriver::stage_seed(const Job& j) {
    uint32_t* sd = hptr<uint32_t>(R_SEED);
    const uint64_t seed = j.req->seed;
    sd[0] = (uint32_t)seed;
    sd[1] = (uint32_t)(seed >> 32);
    sd[2] = (uint32_t)seed;
    sd[3] = (uint32_t)(seed >> 32) ^ 0x80000000u;
    reg_[R_SEED].used = 4 * 4;
}

const RadBatch* VideoDriver::build_step(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const bool small = small_now(j);
    const RadVideoGeom& g = small ? j.geom_small : j.geom;
    /* With references the step is the plugin's packed sequence at this size. */
    const Pack* pack = small ? &j.pack_small : &j.pack;
    if (pack->rows == 0) pack = nullptr;
    const int64_t plain = small ? j.rows_small : j.rows, rows = pack ? pack->rows : plain;
    const std::vector<float>& pos = pack ? pack->pos : small ? j.pos_small : j.pos;
    stage_sched(j, j.step, small && j.step == j.split - 1);
    std::memcpy(hptr<float>(R_ROPE), pos.data(), pos.size() * 4);
    reg_[R_ROPE].used = (int64_t)pos.size() * 4;
    stage_seed(j);
    /* The packed sequence is one attention document. */
    int32_t* cu = hptr<int32_t>(R_CU);
    cu[0] = 0;
    cu[1] = (int32_t)rows;
    reg_[R_CU].used = 2 * 4;
    stage_geom(j, g, plain, 0, pack);
    if (upload() < 0) return nullptr;
    base_batches(j, g, rows, step);
    for (auto& B : out_) B.gen = RAD_GEN_PASS_VSTEP;
    pass_.kind = RAD_GEN_PASS_VSTEP;
    pass_.job = j.id;
    pass_.lora_slot = j.req->lora_slot;
    return &out_[0];
}

/* The small latent to the full size, both latents re-noised to the first full-size step's
 * sigmas. */
const RadBatch* VideoDriver::build_upscale(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    stage_sched(j, j.split, false);
    stage_seed(j);
    *hptr<float>(R_UP) = j.factor;
    reg_[R_UP].used = 4;
    stage_geom(j, j.geom, j.rows, 0);
    if (upload() < 0) return nullptr;
    base_batches(j, j.geom, 0, step);
    for (auto& B : out_) B.gen = RAD_GEN_PASS_VUPSCALE;
    pass_.kind = RAD_GEN_PASS_VUPSCALE;
    pass_.job = j.id;
    return &out_[0];
}

const RadBatch* VideoDriver::build_audio(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    stage_geom(j, j.geom, j.rows, 0);
    if (upload() < 0) return nullptr;
    base_batches(j, j.geom, 0, step);
    for (auto& B : out_) B.gen = RAD_GEN_PASS_ADECODE;
    rb_.assign(1, Readback{ decl_.audio_out, j.n_samples * decl_.audio_channels * 4, audio_host_ });
    pass_.kind = RAD_GEN_PASS_ADECODE;
    pass_.job = j.id;
    return &out_[0];
}

const RadBatch* VideoDriver::build_chunk(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    stage_geom(j, j.geom, j.rows, j.chunk);
    stage_seed(j);   /* a decoder that denoises draws from the request's seed */
    if (upload() < 0) return nullptr;
    base_batches(j, j.geom, j.dec_rows, step);
    const bool last = j.chunk == j.n_chunks - 1;
    const int32_t n = (int32_t)(decl_.frames_per_chunk + (last ? decl_.frames_tail : 0));
    for (size_t r = 0; r < out_.size(); ++r) {
        out_[r].gen = RAD_GEN_PASS_VDECODE;
        vout_[r].n_frames_out = n;
    }
    const int64_t bytes = (int64_t)n * j.req->height * j.req->width * 3;
    rb_.assign(1, Readback{ decl_.frames_out, bytes, frames_host_ });
    pass_.kind = RAD_GEN_PASS_VDECODE;
    pass_.job = j.id;
    pass_.n_frames = n;
    return &out_[0];
}

/* The job's next media through its encoder: its frames padded to whole clips by repeating the
 * last, or its samples padded with zeros to the hop, as the plugin's geometry says. */
const RadBatch* VideoDriver::build_encode(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const Job::EncItem& it = j.enc_q[j.enc_next];
    const VideoRefMedia& m = j.req->refs[(size_t)it.ref];
    const bool audio = it.kind == RAD_ENC_VAUDIO || it.kind == RAD_ENC_AUDIO;
    /* A keyframe at an upscale's small size is the media's second picture. */
    const std::vector<float>& pix = it.small ? m.small_pixels : m.enc_pixels;
    const int32_t eh = it.small ? m.small_h : m.enc_h, ew = it.small ? m.small_w : m.enc_w;
    int32_t* gd = hptr<int32_t>(R_ENCG);
    const int64_t in[4] = { audio ? m.enc_n : m.enc_frames, audio ? 0 : eh, audio ? 0 : ew,
                            audio ? m.enc_channels : 3 };
    const int64_t ng = enc_geom_(audio ? RAD_GEN_PASS_AENCODE : RAD_GEN_PASS_VENCODE, in, gd,
                                 kEncGeomCap);
    int64_t stage_frames = 0, stage_samples = 0;
    if (audio) {
        const int64_t n = m.enc_n, ch = m.enc_channels;
        stage_samples = ng > 0 ? gd[0] : 0;
        if (stage_samples < n || stage_samples * ch * 4 > reg_[R_ENCSM].bytes) {
            fail_locked(j, RAD_E_STATE, "'" + m.name + "' does not fit an encode pass");
            return nullptr;
        }
        float* sm = hptr<float>(R_ENCSM);
        std::memset(sm, 0, (size_t)(stage_samples * ch) * 4);
        for (int64_t c = 0; c < ch; ++c)
            std::memcpy(sm + c * stage_samples, m.enc_samples.data() + c * n, (size_t)n * 4);
        reg_[R_ENCSM].used = stage_samples * ch * 4;
    } else {
        const int64_t nf = m.enc_frames, area = (int64_t)eh * ew * 3;
        stage_frames = ng > 23 ? gd[23] : 0;
        if (nf < 1 || stage_frames < nf || stage_frames * area * 4 > reg_[R_ENCPX].bytes) {
            fail_locked(j, RAD_E_STATE, "'" + m.name + "' does not fit an encode pass");
            return nullptr;
        }
        float* px = hptr<float>(R_ENCPX);
        std::memcpy(px, pix.data(), (size_t)(nf * area) * 4);
        for (int64_t f = nf; f < stage_frames; ++f)
            std::memcpy(px + f * area, px + (nf - 1) * area, (size_t)area * 4);
        reg_[R_ENCPX].used = stage_frames * area * 4;
    }
    reg_[R_ENCG].used = ng * 4;
    stage_seed(j);   /* the condition noise draws from the request's seed */
    stage_geom(j, j.geom, j.rows, 0);
    if (upload() < 0) return nullptr;
    base_batches(j, j.geom, 0, step);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        const int ri = (int)r;
        RadVideoGenBatch& V = vout_[(size_t)r];
        V.enc_geom_d = dptr<int32_t>(ri, R_ENCG);
        V.enc_kind = it.kind;
        V.enc_slot = it.slot;
        if (audio) {
            V.enc_samples = dptr<float>(ri, R_ENCSM);
            V.enc_n_samples = stage_samples;
            V.enc_channels = m.enc_channels;
        } else {
            V.enc_pixels = dptr<float>(ri, R_ENCPX);
            V.enc_frames = (int32_t)stage_frames;
            V.enc_height = eh;
            V.enc_width = ew;
        }
    }
    const int32_t kind = audio ? RAD_GEN_PASS_AENCODE : RAD_GEN_PASS_VENCODE;
    for (auto& B : out_) B.gen = kind;
    pass_.kind = kind;
    pass_.job = j.id;
    return &out_[0];
}

int VideoDriver::done() {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t now = (int64_t)rad_mono_ns();
    Job* j = job_locked(pass_.job);
    switch (j ? pass_.kind : 0) {
    case RAD_GEN_PASS_VCOND:
        j->cond_done = true;
        j->req->ticket->t_cond_ns = now;
        if (j->auto_len) {
            j->auto_len = false;
            std::string note, why;
            int32_t n = predicted_chunks((double)duration_host_[0], &note);
            /* Shaped as at submit, which passed at default_chunks; a longer video than a pass
             * can carry is shortened until it fits. */
            const int32_t steps = j->req->steps > 0 ? j->req->steps : (int32_t)decl_.default_steps;
            const int32_t at_submit = j->n_chunks;
            while (n > at_submit && shape_job(*j, *j->req, n, steps, &why) < 0) --n;
            if (n <= at_submit && shape_job(*j, *j->req, n, steps, &why) < 0) {
                fail_locked(*j, RAD_E_INVAL, why);
                break;
            }
            RAD_INFO("video: the model gives the prompt %d frames (%.2f s)%s", j->geom.frames,
                     (double)j->geom.frames * decl_.fps_den / decl_.fps_num, note.c_str());
            VideoEvent e;
            e.kind = VideoEvent::LENGTH;
            e.n = j->geom.frames;
            j->req->ticket->push(std::move(e));
        }
        break;
    case RAD_GEN_PASS_VENCODE:
    case RAD_GEN_PASS_AENCODE:
        ++j->enc_next;
        break;
    case RAD_GEN_PASS_VSTEP: {
        ++j->step;
        VideoEvent e;
        e.kind = VideoEvent::STEP;
        e.steps_done = j->step;
        j->req->ticket->push(std::move(e));
        if (j->split > 0 && j->step == j->split) j->req->ticket->t_small_ns = now;
        if (j->step >= j->sched.steps) j->req->ticket->t_denoised_ns = now;
        break;
    }
    case RAD_GEN_PASS_VUPSCALE:
        j->upscaled = true;
        j->req->ticket->t_upscaled_ns = now;
        break;
    case RAD_GEN_PASS_ADECODE: {
        VideoEvent e;
        e.kind = VideoEvent::AUDIO;
        e.samples.assign(audio_host_, audio_host_ + j->n_samples * decl_.audio_channels);
        j->req->ticket->push(std::move(e));
        j->audio_done = true;
        break;
    }
    case RAD_GEN_PASS_VDECODE: {
        VideoEvent e;
        e.kind = VideoEvent::FRAMES;
        e.first = j->frames_out;
        e.n = pass_.n_frames;
        const size_t bytes = (size_t)pass_.n_frames * j->req->height * j->req->width * 3;
        e.rgb.assign(frames_host_, frames_host_ + bytes);
        j->req->ticket->push(std::move(e));
        j->frames_out += pass_.n_frames;
        if (++j->chunk == j->n_chunks) {
            j->finished = true;
            j->req->ticket->t_done_ns = now;
            j->req->ticket->finish(RAD_OK);
        }
        break;
    }
    default:
        break;
    }
    pass_ = Pass{};
    rb_.clear();
    return RAD_OK;
}

}  /* namespace gen */
}  /* namespace rad */
