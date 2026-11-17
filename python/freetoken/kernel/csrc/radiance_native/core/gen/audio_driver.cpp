/* audio_driver.cpp -- the audio generator's pass driver (audio_driver.h), and the ticket a caller
 * waits on (audio.h). */
#include "audio_driver.h"
#include "sched/mem_iface.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace rad {
namespace gen {

/* ================================================================== the ticket */
void AudioTicket::push(AudioChunk&& c) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back(std::move(c));
    }
    cv_.notify_all();
}

void AudioTicket::finish(int status, const std::string& why) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (done_) return;
        done_ = true;
        status_ = status;
        why_ = why;
    }
    cv_.notify_all();
}

bool AudioTicket::next(AudioChunk* out, int ms, bool* timed_out) {
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

bool AudioTicket::finished() const {
    std::lock_guard<std::mutex> lk(mu_);
    return done_ && q_.empty();
}

int AudioTicket::status() const {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
}

std::string AudioTicket::why() const {
    std::lock_guard<std::mutex> lk(mu_);
    return why_;
}

/* ================================================================== sizing */
namespace {

constexpr int64_t kAlign = 256;

struct Sizes {
    int64_t T = 0, S = 0, L = 0, NB = 0;
    int64_t NG = 0;    /* stateful KV groups */
    int64_t RS = 0, RN = 0;   /* references: samples and clips a request */
};

/* The paged full-attention group, and how many stateful groups sit beside it. */
int32_t full_group(const Program& prog, int64_t* n_state) {
    int32_t full = 0;
    int64_t ns = 0;
    for (size_t g = 1; g < prog.kv_groups.size(); ++g) {
        const int kind = prog.kv_groups[g].decl.kind;
        if (kind == RAD_KV_FULL && !full) full = (int32_t)g;
        else if (kind == RAD_KV_LINEAR || kind == RAD_KV_CONV) ++ns;
    }
    if (n_state) *n_state = ns;
    return full;
}

Sizes sizes_of(const Program& prog, int64_t block_size) {
    const RadAudioGenDecl& d = prog.audio_gen;
    Sizes z;
    z.T  = prog.ctx.max_tok;
    z.S  = prog.ctx.max_seqs;
    z.L  = d.max_lanes;
    z.NB = block_size > 0 ? (AudioDriver::max_seq_rows(d) + block_size - 1) / block_size : 0;
    full_group(prog, &z.NG);
    z.RS = d.ref_max_samples;
    z.RN = d.ref_max;
    return z;
}

/* Region offsets and sizes, in order, and the total; shared by the allocation and the VRAM budget.
 * The order is the enum's in audio_driver.h. */
int64_t carve(const Sizes& z, int64_t (*bytes)[2], int n) {
    const int64_t I = 4;
    const int64_t SL = std::max(z.S, z.L);
    const int64_t want[] = {
        z.T * I,             /* R_TOK */
        z.T * I,             /* R_POS */
        (z.S + 1) * I,       /* R_CU */
        z.T * I,             /* R_SLOT */
        z.S * z.NB * I,      /* R_BT */
        z.S * I,             /* R_USED */
        z.L * I,             /* R_LANE */
        z.L * I,             /* R_LAST */
        z.L * I,             /* R_FRAME */
        2 * SL * I,          /* R_SEED */
        z.S * 4,             /* R_T */
        z.S * 4,             /* R_TN */
        z.S * 4,             /* R_CFG */
        z.S * I,             /* R_PAIR */
        z.S * I,             /* R_WLANE */
        z.S * I,             /* R_FLAGS */
        z.S * I,             /* R_WIDX */
        z.S * I,             /* R_WF0 */
        z.S * I,             /* R_WNF */
        (z.S + 1) * I,       /* R_LCU */
        z.T * I,             /* R_LSRC */
        z.S * I,             /* R_ESLOT */
        z.S * I,             /* R_LROW */
        z.S * I,             /* R_PREFIX */
        z.S * I,             /* R_CTX */
        z.S * I,             /* R_ACC */
        z.NG * z.S * I,      /* R_SBT */
        z.NG * z.S * AudioDriver::kMaxStateWidth * I,   /* R_SIDX */
        z.S * I,             /* R_PSLOT */
        z.S * I,             /* R_PPOS */
        z.S * I,             /* R_PUSED */
        z.RS * 4,            /* R_RAUD */
        z.RN > 0 ? (z.RN + 1) * I : 0,   /* R_RCU */
        z.RN > 0 ? z.T * I : 0,          /* R_RFRM */
    };
    int64_t off = 0;
    for (int r = 0; r < n; ++r) {
        bytes[r][0] = off;
        bytes[r][1] = want[r];
        off = align_up(off + want[r], kAlign);
    }
    return off;
}

}  /* namespace */

int64_t AudioDriver::device_bytes(const Program& prog) {
    if (!prog.is_audio_generator()) return 0;
    const int32_t fg = full_group(prog, nullptr);
    if (!fg) return 0;
    const Sizes z = sizes_of(prog, prog.kv_groups[(size_t)fg].block_size);
    int64_t b[R_N][2];
    return carve(z, b, R_N);
}

int AudioDriver::init(const Program& prog, IKVManager* kv, const std::vector<RankIO>& ranks) {
    fini();
    if (!prog.is_audio_generator()) return RAD_E_STATE;
    decl_  = prog.audio_gen;
    name_  = prog.agen_str.name;
    kv_    = kv;
    ranks_ = ranks;
    /* The language model's KV, every branch of every lane: one paged full-attention group, and
     * a hybrid model's linear-attention state and convolution groups beside it. */
    state_groups_.clear();
    int n_full = 0;
    for (size_t g = 1; g < prog.kv_groups.size(); ++g) {
        const int kind = prog.kv_groups[g].decl.kind;
        if (kind == RAD_KV_FULL) {
            ++n_full;
        } else if (kind == RAD_KV_LINEAR || kind == RAD_KV_CONV) {
            StateGroup sg;
            sg.g = (int32_t)g;
            sg.width = std::max<int64_t>(1, kv->state_index_width((int32_t)g));
            if (sg.width > kMaxStateWidth) {
                RAD_ERR("audio generator: KV group %zu asks for %lld state slots a sequence; the "
                        "driver stages at most %lld", g, (long long)sg.width,
                        (long long)kMaxStateWidth);
                return RAD_E_UNSUPPORTED;
            }
            state_groups_.push_back(sg);
        } else {
            n_full = -1;
            break;
        }
    }
    kv_group_ = full_group(prog, nullptr);
    if (n_full != 1 || !kv_group_ || prog.kv_groups[(size_t)kv_group_].block_size <= 0) {
        RAD_ERR("audio generator: the program declares %zu KV group(s); the driver stages one "
                "paged full-attention group, the language model's, and the state and "
                "convolution groups of a hybrid one beside it", prog.kv_groups.size() - 1);
        return RAD_E_UNSUPPORTED;
    }
    block_size_ = prog.kv_groups[(size_t)kv_group_].block_size;
    nb_ = decl_.branches == 1 ? 1 : 2;
    const Sizes z = sizes_of(prog, block_size_);
    max_tok_    = z.T;
    max_seqs_   = z.S;
    max_blocks_ = z.NB;
    neg_own_ = !prog.agen_str.neg_tmpl.empty();
    /* A FRAME pass carries both branches of every lane; a window's step at least one window, a
     * chunk's vocode its frames and their context, and a whole prompt both its branches. */
    const int64_t step_rows = chunked() ? decl_.voc_context + decl_.voc_chunk
                                        : 2 * (decl_.win_lead + win_latents(decl_.win_frames));
    const int64_t prompt_rows = decl_.prompt_whole ? nb_ * decl_.max_prompt : nb_;
    if (max_seqs_ < nb_ * decl_.max_lanes || max_tok_ < nb_ * decl_.max_lanes ||
        max_tok_ < step_rows || max_tok_ < prompt_rows) {
        RAD_ERR("audio generator: a pass carries %lld rows of %lld sequences; '%s' needs %lld "
                "sequences for its %lld lanes, %lld rows for a %s and %lld for a whole prompt",
                (long long)max_tok_, (long long)max_seqs_, name_.c_str(),
                (long long)(nb_ * decl_.max_lanes), (long long)decl_.max_lanes,
                (long long)step_rows, chunked() ? "chunk's vocode" : "window's step",
                (long long)prompt_rows);
        return RAD_E_UNSUPPORTED;
    }
    int64_t b[R_N][2];
    bytes_ = carve(z, b, R_N);
    for (int r = 0; r < R_N; ++r) { reg_[r].off = b[r][0]; reg_[r].bytes = b[r][1]; reg_[r].used = 0; }

    host_ = static_cast<char*>(rad_dev_alloc(bytes_, RAD_MEM_HOST_PINNED));
    status_host_ = static_cast<int32_t*>(rad_dev_alloc(decl_.max_lanes * 4, RAD_MEM_HOST_PINNED));
    audio_bytes_ = (chunked() ? decl_.voc_chunk : win_latents(decl_.win_frames)) *
                   decl_.samples_per_latent * decl_.channels * 4;
    audio_host_ = static_cast<float*>(rad_dev_alloc(audio_bytes_, RAD_MEM_HOST_PINNED));
    if (!host_ || !status_host_ || !audio_host_) {
        RAD_ERR("audio generator: no pinned host memory for the pass staging (%lld + %lld bytes)",
                (long long)bytes_, (long long)audio_bytes_);
        return RAD_E_NOMEM;
    }
    std::memset(host_, 0, (size_t)bytes_);
    dev_.assign(ranks_.size(), nullptr);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r].device));
        dev_[r] = rad_dev_alloc(bytes_, RAD_MEM_DEVICE);
        if (!dev_[r]) {
            RAD_ERR("audio generator: rank %zu has no room for the pass staging (%lld bytes)", r,
                    (long long)bytes_);
            return RAD_E_NOMEM;
        }
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0].device));
    out_.assign(ranks_.size(), RadBatch{});
    aout_.assign(ranks_.size(), RadAudioGenBatch{});
    kout_.assign(ranks_.size() * (1 + state_groups_.size()), RadKVGroupBatch{});
    free_lanes_.clear();
    for (int64_t i = decl_.max_lanes - 1; i >= 0; --i) free_lanes_.push_back((int32_t)i);
    return RAD_OK;
}

void AudioDriver::fini() {
    for (size_t r = 0; r < dev_.size(); ++r)
        if (dev_[r]) {
            if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
            rad_dev_free(dev_[r], RAD_MEM_DEVICE);
        }
    dev_.clear();
    if (!ranks_.empty()) rad_dev_set(ranks_[0].device);
    if (host_) rad_dev_free(host_, RAD_MEM_HOST_PINNED);
    if (status_host_) rad_dev_free(status_host_, RAD_MEM_HOST_PINNED);
    if (audio_host_) rad_dev_free(audio_host_, RAD_MEM_HOST_PINNED);
    host_ = nullptr;
    status_host_ = nullptr;
    audio_host_ = nullptr;
}

/* ================================================================== requests */
int AudioDriver::submit(std::unique_ptr<AudioRequest> r, std::string* why) {
    if (!r || !r->ticket) return RAD_E_INVAL;
    const RadAudioGenDecl& d = decl_;
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    const int64_t T = (int64_t)r->tokens.size(), Tn = (int64_t)r->neg_tokens.size();
    if (nb_ == 1 && r->cfg != 0.0f && r->cfg != 1.0f)
        return no("this model runs without guidance; guidance_scale must be 1");
    if (nb_ == 2 && !neg_own_ && T <= d.neg_keep_head + d.neg_keep_tail)
        return no("the prompt is " + std::to_string(T) + " tokens, too short to hold its own "
                  "markers");
    if (T < 1 || std::max(T, Tn) > d.max_prompt)
        return no("the prompt is " + std::to_string(std::max(T, Tn)) + " tokens; this model takes "
                  "1.." + std::to_string(d.max_prompt));
    if (r->max_frames < 1 || r->max_frames > d.max_frames)
        return no("the length must be 1.." + std::to_string(d.max_frames) + " frames (" +
                  std::to_string(d.max_frames * d.frame_rate_den / d.frame_rate_num) +
                  " seconds)");
    if (r->steps < 0 || r->steps > 1000) return no("num_inference_steps must be 1..1000");
    if (r->cfg != 0.0f && !(r->cfg >= 1.0f)) return no("guidance_scale must be >= 1");
    /* References: the clips the declaration allows, each a whole number of frames, and a frame
     * for every row that names one. */
    if (!r->ref_cu.empty() || !r->ref_frame.empty() || !r->ref_audio.empty() ||
        !r->neg_ref_frame.empty()) {
        if (nb_ == 2 && r->cfg != 0.0f && r->cfg != 1.0f && r->neg_ref_frame.empty())
            return no("voice cloning requires guidance_scale 1; reference CFG is not served");
        const int64_t n_ref = (int64_t)r->ref_cu.size() - 1, hop = d.ref_hop;
        bool ok = d.ref_max > 0 && n_ref >= 1 && n_ref <= d.ref_max && r->ref_cu[0] == 0 &&
                  r->ref_cu.back() == (int64_t)r->ref_audio.size() &&
                  (int64_t)r->ref_audio.size() <= d.ref_max_samples &&
                  (int64_t)r->ref_frame.size() == T;
        for (int64_t i = 0; ok && i < n_ref; ++i)
            ok = r->ref_cu[(size_t)i + 1] > r->ref_cu[(size_t)i] && r->ref_cu[(size_t)i + 1] % hop == 0;
        const int64_t frames = ok ? r->ref_cu.back() / hop : 0;
        for (int64_t t = 0; ok && t < T; ++t)
            ok = r->ref_frame[(size_t)t] >= -1 && r->ref_frame[(size_t)t] < frames;
        if (ok && !r->neg_ref_frame.empty()) {
            ok = nb_ == 2 && neg_own_ && (int64_t)r->neg_ref_frame.size() == Tn;
            for (int32_t f : r->neg_ref_frame) ok = ok && f >= -1 && f < frames;
        }
        if (!ok)
            return no(d.ref_max > 0 ? "the references do not fit this model: at most " +
                                          std::to_string(d.ref_max) + " clips of whole frames, " +
                                          std::to_string(d.ref_max_samples) + " samples together"
                                    : std::string("this model takes no reference audio"));
    }
    if (!kv_->can_ever_fit(std::max(T, Tn) + r->max_frames + 1))
        return no("a piece of " + std::to_string(r->max_frames) + " frames after a " +
                  std::to_string(T) + "-token prompt does not fit the KV cache even when it is "
                  "empty; ask for a shorter piece");
    r->ticket->t_submit_ns = (int64_t)rad_mono_ns();
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) {
        if (why) *why = closed_why_;
        return closed_status_;
    }
    auto j = std::make_unique<Job>();
    j->id = next_id_++;
    j->T = T;
    j->req = std::move(r);
    jobs_.push_back(std::move(j));
    return RAD_OK;
}

void AudioDriver::shutdown(const char* why, int status) {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    closed_why_ = why ? why : "the engine stopped";
    closed_status_ = status;
    for (auto& j : jobs_) j->req->ticket->finish(status, closed_why_);
}

int64_t AudioDriver::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (!j->admitted) ++n;
    return n;
}

int64_t AudioDriver::running() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (j->admitted) ++n;
    return n;
}

AudioDriver::Job* AudioDriver::job_locked(uint64_t id) {
    for (auto& j : jobs_) if (j->id == id) return j.get();
    return nullptr;
}

void AudioDriver::release_locked(Job& j) {
    for (int b = 0; b < 2; ++b)
        if (j.seq[b]) {
            kv_->free_sequence(j.seq[b]);
            j.seq[b] = 0;
        }
    j.have_kv = false;
    if (j.lane >= 0) {
        free_lanes_.push_back(j.lane);
        j.lane = -1;
    }
}

void AudioDriver::fail_locked(Job& j, int status, const std::string& why) {
    j.failed = true;
    j.req->ticket->finish(status, why);
}

/* In arrival order; a request that does not fit stops the ones behind it. */
int AudioDriver::admit_locked() {
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted || j.failed) continue;
        if (free_lanes_.empty()) break;
        const AudioRequest& r = *j.req;
        const float cfg = r.cfg > 0.0f ? r.cfg : (float)decl_.default_cfg;
        /* The unconditional prompt: the request's own (the conditional one at guidance 1, which
         * makes the guided logits the conditional ones), or derived from the conditional one. */
        std::vector<int32_t> neg;
        if (nb_ > 1) {
            neg = r.tokens;
            if (neg_own_ && cfg != 1.0f && !r.neg_tokens.empty()) {
                neg = r.neg_tokens;
            } else if (!neg_own_) {
                for (int64_t i = decl_.neg_keep_head; i < j.T - decl_.neg_keep_tail; ++i)
                    neg[(size_t)i] = decl_.neg_token;
            }
        }
        /* Every branch for the whole piece, so nothing in flight ever waits for KV. */
        bool ok = true;
        for (int b = 0; b < nb_ && ok; ++b) {
            const int64_t rows = (b ? (int64_t)neg.size() : j.T) + r.max_frames + 1;
            j.seq[b] = ((uint64_t)1 << 61) | next_seq_++;
            if (kv_->add_sequence(j.seq[b]) < 0 || kv_->ensure(j.seq[b], rows) < 0) ok = false;
        }
        if (!ok) {
            for (int b = 0; b < 2; ++b)
                if (j.seq[b]) {
                    if (kv_->has_sequence(j.seq[b])) kv_->free_sequence(j.seq[b]);
                    j.seq[b] = 0;
                }
            /* With nothing running to free blocks, waiting would never end. */
            bool running = false;
            for (const auto& o : jobs_) running = running || o->admitted;
            if (running) break;
            fail_locked(j, RAD_E_FULL, "the KV cache cannot hold this piece, and nothing is running "
                                       "that would free it; ask for a shorter piece");
            continue;
        }
        j.have_kv = true;
        j.lane = free_lanes_.back();
        free_lanes_.pop_back();
        j.neg = std::move(neg);
        j.Tn = (int64_t)j.neg.size();
        j.steps = r.steps > 0 ? r.steps : (int32_t)decl_.default_steps;
        j.cfg = cfg;
        j.req->ticket->in_tokens = j.T;
        j.admitted = true;
    }
    return RAD_OK;
}

/* ================================================================== windows */
/* Window k is frames [k hop, k hop + win_frames), cut short at the piece's end. Window 0 always
 * exists; window k >= 1 exists while frames remain past its overlap with window k - 1. */
int64_t AudioDriver::window_frames(const Job& j, int32_t k) const {
    const int64_t f0 = win_frame0(k);
    return j.ar_done ? std::min<int64_t>(decl_.win_frames, j.kept - f0) : decl_.win_frames;
}

bool AudioDriver::window_ready(const Job& j) const {
    if (j.voc_due >= 0) return false;
    if (j.win_step > 0) return true;
    const int64_t k = j.win;
    if (j.ar_done)
        return k == 0 ? j.kept >= 1
                      : j.kept > k * decl_.win_hop + (decl_.win_frames - decl_.win_hop);
    return j.kept >= k * decl_.win_hop + decl_.win_frames;
}

bool AudioDriver::chunk_due(const Job& j) const {
    const int64_t ready = j.kept - j.played;
    return !j.finished && (ready >= decl_.voc_chunk || (j.ar_done && ready > 0));
}

int AudioDriver::window_last(const Job& j, int32_t k) const {
    const bool next_exists =
        j.kept > (int64_t)(k + 1) * decl_.win_hop + (decl_.win_frames - decl_.win_hop);
    if (next_exists) return 0;
    return j.ar_done ? 1 : -1;
}

/* ================================================================== staging */
int AudioDriver::upload() {
    for (size_t r = 0; r < ranks_.size(); ++r)
        for (int g = 0; g < R_N; ++g) {
            if (reg_[g].used <= 0) continue;
            RAD_TRY(rad_memcpy_async((char*)dev_[r] + reg_[g].off, host_ + reg_[g].off,
                                     reg_[g].used, ranks_[r].stream));
        }
    return RAD_OK;
}

void AudioDriver::base_batches(int64_t n_tok, int64_t n_seq, int32_t max_q, int32_t max_ctx,
                               int step, bool kv, int64_t bt_pitch) {
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        RadAudioGenBatch& A = aout_[r];
        RadKVGroupBatch& K = kout_[r * (1 + state_groups_.size())];
        const int ri = (int)r;
        B = RadBatch{};
        B.step = step;
        B.phase = RAD_PHASE_PREFILL;
        B.n_tok = n_tok;
        B.n_seq = n_seq;
        B.max_q_len = max_q;
        B.max_ctx_len = max_ctx;
        B.max_q_len_prefill = max_q;
        B.token_ids = dptr<int32_t>(ri, R_TOK);
        B.positions = dptr<int32_t>(ri, R_POS);
        B.cu_seqlens = dptr<int32_t>(ri, R_CU);
        if (kv) {
            K = RadKVGroupBatch{};
            K.group = (rad_kvgroup)kv_group_;
            K.slot_mapping = dptr<int32_t>(ri, R_SLOT);
            K.block_table = dptr<int32_t>(ri, R_BT);
            K.block_table_pitch = bt_pitch;
            K.seqused = dptr<int32_t>(ri, R_USED);
            K.block_size = block_size_;
            K.max_blocks = max_blocks_;
            /* A stateful group's block-table row is the sequence's slot, as the text scheduler
             * stages it, so a kernel reading tables needs no second path. */
            for (size_t gi = 0; gi < state_groups_.size(); ++gi) {
                RadKVGroupBatch& Ks = (&K)[1 + gi];
                Ks = RadKVGroupBatch{};
                Ks.group = (rad_kvgroup)state_groups_[gi].g;
                Ks.block_table = dptr<int32_t>(ri, R_SBT) + gi * max_seqs_;
                Ks.block_table_pitch = 1;
                Ks.seqused = dptr<int32_t>(ri, R_USED);
                Ks.state_index = dptr<int32_t>(ri, R_SIDX) + gi * max_seqs_ * kMaxStateWidth;
                Ks.state_index_pitch = state_groups_[gi].width;
                Ks.max_blocks = 1;
            }
            B.n_kv_groups = (int)(1 + state_groups_.size());
            B.kv = &K;
            B.ctx_lens = dptr<int32_t>(ri, R_CTX);
            B.num_accepted = dptr<int32_t>(ri, R_ACC);
        }
        A = RadAudioGenBatch{};
        A.lane = dptr<int32_t>(ri, R_LANE);
        A.last = dptr<int32_t>(ri, R_LAST);
        A.frame = dptr<int32_t>(ri, R_FRAME);
        A.seed = dptr<uint32_t>(ri, R_SEED);
        A.t = dptr<float>(ri, R_T);
        A.t_next = dptr<float>(ri, R_TN);
        A.cfg = dptr<float>(ri, R_CFG);
        A.pair = dptr<int32_t>(ri, R_PAIR);
        A.win_lane = dptr<int32_t>(ri, R_WLANE);
        A.flags = dptr<int32_t>(ri, R_FLAGS);
        A.win_index = dptr<int32_t>(ri, R_WIDX);
        A.win_frame0 = dptr<int32_t>(ri, R_WF0);
        A.win_frames = dptr<int32_t>(ri, R_WNF);
        A.lat_cu = dptr<int32_t>(ri, R_LCU);
        A.lat_src = dptr<int32_t>(ri, R_LSRC);
        A.slot = dptr<int32_t>(ri, R_ESLOT);
        A.last_row = dptr<int32_t>(ri, R_LROW);
        A.prefix = dptr<int32_t>(ri, R_PREFIX);
        A.prev_slot = dptr<int32_t>(ri, R_PSLOT);
        A.prev_pos = dptr<int32_t>(ri, R_PPOS);
        A.prev_used = dptr<int32_t>(ri, R_PUSED);
        B.agen_b = &A;
    }
}

void AudioDriver::stage_state(const std::vector<uint64_t>& seqs, const std::vector<int32_t>& ctx) {
    const int64_t n = (int64_t)seqs.size();
    int32_t* cl = hptr<int32_t>(R_CTX);
    int32_t* acc = hptr<int32_t>(R_ACC);
    for (int64_t e = 0; e < n; ++e) {
        cl[e] = ctx[(size_t)e];
        /* One-based, as num_accepted always is: a pass commits every row it runs. */
        acc[e] = 1;
    }
    reg_[R_CTX].used = reg_[R_ACC].used = n * 4;
    for (size_t gi = 0; gi < state_groups_.size(); ++gi) {
        const StateGroup& sg = state_groups_[gi];
        int32_t* bt = hptr<int32_t>(R_SBT) + gi * max_seqs_;
        int32_t* si = hptr<int32_t>(R_SIDX) + gi * max_seqs_ * kMaxStateWidth;
        for (int64_t e = 0; e < n; ++e) {
            bt[e] = kv_->state_slot(seqs[(size_t)e], sg.g);
            kv_->state_index_row(seqs[(size_t)e], sg.g, si + e * sg.width, (int)sg.width);
        }
    }
    if (!state_groups_.empty()) {
        reg_[R_SBT].used = (int64_t)state_groups_.size() * max_seqs_ * 4;
        reg_[R_SIDX].used = (int64_t)state_groups_.size() * max_seqs_ * kMaxStateWidth * 4;
    }
}

/* The KV slot of position `p` in a sequence with this block table. */
static int32_t slot_of(const std::vector<int32_t>& bt, int64_t p, int64_t bs) {
    const int64_t bi = p / bs;
    return bi < (int64_t)bt.size() ? (int32_t)(bt[(size_t)bi] * bs + p % bs) : -1;
}

/* ================================================================== the passes */
const RadBatch* AudioDriver::next(int step) {
    std::lock_guard<std::mutex> lk(mu_);
    pass_ = Pass{};
    rb_.clear();
    if (closed_) return nullptr;
    /* Retire finished, failed and cancelled jobs, returning their KV and lanes before admitting.
     * Safe because next() runs between passes, with nothing in flight. */
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

    /* 1. A vocode: a chunk of frames that is ready, or a finished window, once it is known
     * whether a window follows it. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed) continue;
        if (chunked()) {
            if (chunk_due(j)) return build_chunk(j, step);
            continue;
        }
        if (j.voc_due >= 0 && window_last(j, j.voc_due) >= 0) return build_vocode(j, step);
    }
    /* 2. A prompt chunk. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted && !j.failed && j.prompt_done < j.T) return build_prompt(j, step);
    }
    /* 3. A frame of every lane in its autoregressive stage, or a step of the ready windows. */
    std::vector<Job*> fr, win;
    int64_t win_rows = 0, win_ent = 0;
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed || j.prompt_done < j.T) continue;
        if (!j.ar_done) fr.push_back(&j);
        if (!chunked() && window_ready(j)) {
            const int64_t nb = j.cfg > 1.0f ? 2 : 1;
            const int64_t rows =
                nb * (decl_.win_lead + win_latents(window_frames(j, j.win)));
            if (win_rows + rows <= max_tok_ && win_ent + nb <= max_seqs_) {
                win_rows += rows;
                win_ent += nb;
                win.push_back(&j);
            }
        }
    }
    if (!fr.empty() && (win.empty() || frame_turn_)) {
        frame_turn_ = false;
        return build_frame(fr, step);
    }
    if (!win.empty()) {
        frame_turn_ = true;
        return build_window(win, step);
    }
    return nullptr;
}

const RadBatch* AudioDriver::batch_for(int rank) const {
    if (rank < 0 || (size_t)rank >= out_.size() || pass_.kind == 0) return nullptr;
    return &out_[(size_t)rank];
}

const RadBatch* AudioDriver::build_prompt(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    /* Each branch's rows: both whole, or the same chunk of both (their lengths are then equal). */
    const int64_t p0 = j.prompt_done;
    int64_t cb[2] = { 0, 0 };
    if (decl_.prompt_whole) {
        cb[0] = j.T;
        cb[1] = nb_ > 1 ? j.Tn : 0;
    } else {
        cb[0] = std::min<int64_t>(j.T - j.prompt_done, max_tok_ / nb_);
        cb[1] = nb_ > 1 ? cb[0] : 0;
    }
    const int64_t cmax = std::max(cb[0], cb[1]);
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    int32_t* slot = hptr<int32_t>(R_SLOT);
    int32_t* cu = hptr<int32_t>(R_CU);
    int64_t pitch = 1;
    for (int b = 0; b < nb_; ++b)
        pitch = std::max<int64_t>(pitch, (int64_t)kv_->block_table(j.seq[b], kv_group_).size());
    int32_t* btab = hptr<int32_t>(R_BT);
    int32_t* used = hptr<int32_t>(R_USED);
    cu[0] = 0;
    for (int b = 0; b < nb_; ++b) {
        const std::vector<int32_t>& src = b ? j.neg : j.req->tokens;
        const std::vector<int32_t>& bt = kv_->block_table(j.seq[b], kv_group_);
        for (int64_t i = 0; i < cb[b]; ++i) {
            const int64_t e = cu[b] + i;
            tok[e] = src[(size_t)(p0 + i)];
            pos[e] = (int32_t)(p0 + i);
            slot[e] = slot_of(bt, p0 + i, block_size_);
        }
        cu[b + 1] = (int32_t)(cu[b] + cb[b]);
        for (int64_t k = 0; k < pitch; ++k)
            btab[b * pitch + k] = k < (int64_t)bt.size() ? bt[(size_t)k] : -1;
        used[b] = (int32_t)(p0 + cb[b]);
    }
    const int64_t rows = cu[nb_];
    const bool last = p0 + cb[0] == j.T;
    std::vector<uint64_t> seqs;
    std::vector<int32_t> ctx;
    for (int b = 0; b < nb_; ++b) {
        hptr<int32_t>(R_ESLOT)[b] = nb_ * j.lane + b;
        hptr<int32_t>(R_LROW)[b] = cu[b + 1] - 1;
        hptr<int32_t>(R_PREFIX)[b] = (int32_t)p0;
        seqs.push_back(j.seq[b]);
        ctx.push_back((int32_t)p0);
    }
    stage_state(seqs, ctx);
    reg_[R_ESLOT].used = reg_[R_LROW].used = reg_[R_PREFIX].used = nb_ * 4;
    hptr<int32_t>(R_LANE)[0] = j.lane;
    hptr<int32_t>(R_LAST)[0] = last ? 1 : 0;
    hptr<uint32_t>(R_SEED)[0] = (uint32_t)(j.req->seed & 0xffffffffu);
    hptr<uint32_t>(R_SEED)[1] = (uint32_t)(j.req->seed >> 32);
    reg_[R_TOK].used = reg_[R_POS].used = reg_[R_SLOT].used = rows * 4;
    reg_[R_CU].used = (nb_ + 1) * 4;
    reg_[R_BT].used = nb_ * pitch * 4;
    reg_[R_USED].used = nb_ * 4;
    reg_[R_LANE].used = reg_[R_LAST].used = 4;
    reg_[R_SEED].used = 8;
    /* The references: their clips, bounds and each row's frame (a whole prompt, so the rows are
     * the prompt's own). */
    const AudioRequest& q = *j.req;
    const int64_t n_ref = q.ref_cu.empty() ? 0 : (int64_t)q.ref_cu.size() - 1;
    h_rcu_.clear();
    h_rfrm_.clear();
    if (n_ref > 0) {
        std::memcpy(hptr<float>(R_RAUD), q.ref_audio.data(), q.ref_audio.size() * 4);
        std::memcpy(hptr<int32_t>(R_RCU), q.ref_cu.data(), q.ref_cu.size() * 4);
        reg_[R_RAUD].used = (int64_t)q.ref_audio.size() * 4;
        reg_[R_RCU].used = (int64_t)q.ref_cu.size() * 4;
        h_rcu_ = q.ref_cu;
        h_rfrm_.assign(q.ref_frame.begin() + p0, q.ref_frame.begin() + p0 + cb[0]);
        /* At guidance 1 the driver runs the conditional prompt in both branches. Its audio
         * rows must be duplicated too, not silently turned into text in the second branch. */
        if (nb_ > 1) {
            if (j.cfg == 1.0f)
                h_rfrm_.insert(h_rfrm_.end(), q.ref_frame.begin() + p0,
                               q.ref_frame.begin() + p0 + cb[1]);
            else if (!q.neg_ref_frame.empty())
                h_rfrm_.insert(h_rfrm_.end(), q.neg_ref_frame.begin() + p0,
                               q.neg_ref_frame.begin() + p0 + cb[1]);
            else
                h_rfrm_.resize((size_t)rows, -1);
        }
        std::memcpy(hptr<int32_t>(R_RFRM), h_rfrm_.data(), h_rfrm_.size() * 4);
        reg_[R_RFRM].used = (int64_t)h_rfrm_.size() * 4;
    }
    if (upload() < 0) return nullptr;
    base_batches(rows, nb_, (int32_t)cmax, (int32_t)(p0 + cmax), step, true, pitch);
    h_lane_.assign(1, j.lane);
    h_last_.assign(1, last ? 1 : 0);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        out_[r].gen = RAD_GEN_PASS_PROMPT;
        RadAudioGenBatch& A = aout_[r];
        A.prompt_cu_h = cu;
        A.n_lane = 1;
        A.lane_h = h_lane_.data();
        A.last_h = h_last_.data();
        if (n_ref > 0) {
            const int ri = (int)r;
            A.n_ref = n_ref;
            A.ref_samples = (int64_t)q.ref_audio.size();
            A.ref_audio = dptr<float>(ri, R_RAUD);
            A.ref_cu = dptr<int32_t>(ri, R_RCU);
            A.ref_cu_h = h_rcu_.data();
            A.ref_frame = dptr<int32_t>(ri, R_RFRM);
            A.ref_frame_h = h_rfrm_.data();
        }
    }
    pass_.kind = RAD_GEN_PASS_PROMPT;
    pass_.jobs.assign(1, j.id);
    pass_.rows.assign(1, cb[0]);
    return &out_[0];
}

const RadBatch* AudioDriver::build_frame(const std::vector<Job*>& js, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const int64_t L = (int64_t)js.size(), S = nb_ * L;
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    int32_t* slot = hptr<int32_t>(R_SLOT);
    int32_t* cu = hptr<int32_t>(R_CU);
    int32_t* btab = hptr<int32_t>(R_BT);
    int32_t* used = hptr<int32_t>(R_USED);
    int32_t* lane = hptr<int32_t>(R_LANE);
    int32_t* frame = hptr<int32_t>(R_FRAME);
    uint32_t* seed = hptr<uint32_t>(R_SEED);
    int64_t pitch = 1;
    int32_t max_ctx = 0;
    for (const Job* j : js)
        for (int b = 0; b < nb_; ++b)
            pitch = std::max<int64_t>(pitch, (int64_t)kv_->block_table(j->seq[b], kv_group_).size());
    h_lane_.assign((size_t)L, 0);
    h_frame_.assign((size_t)L, 0);
    pass_.jobs.clear();
    std::vector<uint64_t> seqs;
    std::vector<int32_t> ctx;
    for (int64_t i = 0; i < L; ++i) {
        const Job& j = *js[(size_t)i];
        for (int b = 0; b < nb_; ++b) {
            /* The frame's row follows the branch's prompt and its earlier frames. */
            const int64_t p = (b ? j.Tn : j.T) + j.frame;
            const int64_t e = nb_ * i + b;
            const std::vector<int32_t>& bt = kv_->block_table(j.seq[b], kv_group_);
            tok[e] = 0;
            pos[e] = (int32_t)p;
            slot[e] = slot_of(bt, p, block_size_);
            cu[e] = (int32_t)e;
            for (int64_t k = 0; k < pitch; ++k)
                btab[e * pitch + k] = k < (int64_t)bt.size() ? bt[(size_t)k] : -1;
            used[e] = (int32_t)(p + 1);
            /* The row before, whose hidden state the frame is sampled from: the prompt's last
             * row for the first frame. */
            hptr<int32_t>(R_PSLOT)[e] = slot_of(bt, p - 1, block_size_);
            hptr<int32_t>(R_PPOS)[e] = (int32_t)(p - 1);
            hptr<int32_t>(R_PUSED)[e] = (int32_t)p;
            hptr<int32_t>(R_ESLOT)[e] = nb_ * j.lane + b;
            hptr<int32_t>(R_LROW)[e] = (int32_t)e;
            max_ctx = std::max<int32_t>(max_ctx, (int32_t)(p + 1));
            seqs.push_back(j.seq[b]);
            ctx.push_back((int32_t)p);
        }
        hptr<float>(R_CFG)[i] = j.cfg;
        lane[i] = h_lane_[(size_t)i] = j.lane;
        frame[i] = h_frame_[(size_t)i] = j.frame;
        seed[2 * i] = (uint32_t)(j.req->seed & 0xffffffffu);
        seed[2 * i + 1] = (uint32_t)(j.req->seed >> 32);
        pass_.jobs.push_back(j.id);
    }
    cu[S] = (int32_t)S;
    stage_state(seqs, ctx);
    reg_[R_TOK].used = reg_[R_POS].used = reg_[R_SLOT].used = S * 4;
    reg_[R_CU].used = (S + 1) * 4;
    reg_[R_BT].used = S * pitch * 4;
    reg_[R_USED].used = S * 4;
    reg_[R_LANE].used = reg_[R_FRAME].used = reg_[R_CFG].used = L * 4;
    reg_[R_SEED].used = 2 * L * 4;
    reg_[R_ESLOT].used = reg_[R_LROW].used = S * 4;
    reg_[R_PSLOT].used = reg_[R_PPOS].used = reg_[R_PUSED].used = S * 4;
    if (upload() < 0) return nullptr;
    base_batches(S, S, 1, max_ctx, step, true, pitch);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        B.gen = RAD_GEN_PASS_FRAME;
        B.phase = RAD_PHASE_DECODE;
        B.n_seq_decode = S;
        B.n_tok_decode = S;
        B.max_q_len_decode = 1;
        B.max_q_len_prefill = 0;
        RadAudioGenBatch& A = aout_[r];
        A.n_lane = L;
        A.lane_h = h_lane_.data();
        A.frame_h = h_frame_.data();
    }
    rb_.assign(1, Readback{ decl_.status, decl_.max_lanes * 4, status_host_ });
    pass_.kind = RAD_GEN_PASS_FRAME;
    return &out_[0];
}

const RadBatch* AudioDriver::build_window(const std::vector<Job*>& js, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    int32_t* cu = hptr<int32_t>(R_CU);
    float* t = hptr<float>(R_T);
    float* tn = hptr<float>(R_TN);
    float* cfg = hptr<float>(R_CFG);
    int32_t* pair = hptr<int32_t>(R_PAIR);
    int32_t* wlane = hptr<int32_t>(R_WLANE);
    int32_t* flags = hptr<int32_t>(R_FLAGS);
    int32_t* widx = hptr<int32_t>(R_WIDX);
    int32_t* wf0 = hptr<int32_t>(R_WF0);
    int32_t* wnf = hptr<int32_t>(R_WNF);
    int32_t* lcu = hptr<int32_t>(R_LCU);
    int32_t* lsrc = hptr<int32_t>(R_LSRC);
    uint32_t* seed = hptr<uint32_t>(R_SEED);
    h_wlane_.clear();
    h_flags_.clear();
    h_widx_.clear();
    h_wf0_.clear();
    h_wnf_.clear();
    h_lcu_.assign(1, 0);
    pass_.jobs.clear();
    int64_t S = 0, row = 0, lrow = 0;
    int32_t max_q = 0;
    cu[0] = 0;
    lcu[0] = 0;
    for (Job* jp : js) {
        Job& j = *jp;
        const int64_t nf = window_frames(j, j.win);
        const int64_t nl = win_latents(nf);
        const int64_t len = decl_.win_lead + nl;
        const int nb = j.cfg > 1.0f ? 2 : 1;
        int32_t f = 0;
        if (j.win_step == 0) f |= RAD_AGEN_F_FIRST;
        if (j.win_step + 1 == j.steps) f |= RAD_AGEN_F_LAST;
        if (j.win > 0) f |= RAD_AGEN_F_JOIN;
        for (int b = 0; b < nb; ++b) {
            const int64_t e = S++;
            for (int64_t k = 0; k < len; ++k) {
                tok[row + k] = 0;
                pos[row + k] = (int32_t)k;
            }
            for (int64_t k = 0; k < nl; ++k) lsrc[lrow + k] = (int32_t)(row + decl_.win_lead + k);
            row += len;
            lrow += nl;
            cu[e + 1] = (int32_t)row;
            lcu[e + 1] = (int32_t)lrow;
            t[e] = (float)j.win_step / (float)j.steps;
            tn[e] = (float)(j.win_step + 1) / (float)j.steps;
            cfg[e] = b ? 0.0f : (nb > 1 ? j.cfg : 1.0f);
            pair[e] = b ? -2 : (nb > 1 ? (int32_t)(e + 1) : -1);
            wlane[e] = j.lane;
            flags[e] = f;
            widx[e] = j.win;
            wf0[e] = (int32_t)win_frame0(j.win);
            wnf[e] = (int32_t)nf;
            seed[2 * e] = (uint32_t)(j.req->seed & 0xffffffffu);
            seed[2 * e + 1] = (uint32_t)(j.req->seed >> 32);
            h_wlane_.push_back(j.lane);
            h_flags_.push_back(f);
            h_widx_.push_back(j.win);
            h_wf0_.push_back(wf0[e]);
            h_wnf_.push_back((int32_t)nf);
            h_lcu_.push_back((int32_t)lrow);
            max_q = std::max<int32_t>(max_q, (int32_t)len);
        }
        pass_.jobs.push_back(j.id);
    }
    reg_[R_TOK].used = reg_[R_POS].used = row * 4;
    reg_[R_CU].used = reg_[R_LCU].used = (S + 1) * 4;
    reg_[R_T].used = reg_[R_TN].used = reg_[R_CFG].used = S * 4;
    reg_[R_PAIR].used = reg_[R_WLANE].used = reg_[R_FLAGS].used = S * 4;
    reg_[R_WIDX].used = reg_[R_WF0].used = reg_[R_WNF].used = S * 4;
    reg_[R_SEED].used = 2 * S * 4;
    reg_[R_LSRC].used = lrow * 4;
    if (upload() < 0) return nullptr;
    base_batches(row, S, max_q, 0, step, false, 0);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        out_[r].gen = RAD_GEN_PASS_WINDOW;
        RadAudioGenBatch& A = aout_[r];
        A.win_lane_h = h_wlane_.data();
        A.flags_h = h_flags_.data();
        A.win_index_h = h_widx_.data();
        A.win_frame0_h = h_wf0_.data();
        A.win_frames_h = h_wnf_.data();
        A.lat_rows = lrow;
        A.lat_cu_h = h_lcu_.data();
    }
    pass_.kind = RAD_GEN_PASS_WINDOW;
    return &out_[0];
}

const RadBatch* AudioDriver::build_vocode(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const int32_t k = j.voc_due;
    const int64_t nl = win_latents(window_frames(j, k));
    int64_t left = k > 0 ? decl_.win_crop_left : 0;
    int64_t right = window_last(j, k) == 1 ? 0 : decl_.win_crop_right;
    left = std::min<int64_t>(left, nl);
    right = std::min<int64_t>(right, nl - left);
    int32_t* cu = hptr<int32_t>(R_CU);
    cu[0] = 0;
    cu[1] = (int32_t)nl;
    reg_[R_CU].used = 2 * 4;
    if (upload() < 0) return nullptr;
    base_batches(nl, 1, (int32_t)nl, 0, step, false, 0);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        out_[r].gen = RAD_GEN_PASS_VOCODE;
        RadAudioGenBatch& A = aout_[r];
        A.voc_lane = j.lane;
        A.voc_window = k;
        A.voc_latents = (int32_t)nl;
        A.voc_left = (int32_t)left;
        A.voc_right = (int32_t)right;
    }
    const int64_t samples = (nl - left - right) * decl_.samples_per_latent;
    rb_.assign(1, Readback{ decl_.audio_out, samples * decl_.channels * 4, audio_host_ });
    pass_.kind = RAD_GEN_PASS_VOCODE;
    pass_.voc_job = j.id;
    pass_.voc_window = k;
    pass_.voc_samples = samples;
    return &out_[0];
}

/* A chunk of a lane's kept frames, after the frames it reads again for context. Its rows are
 * the frames, at their positions in the piece. */
const RadBatch* AudioDriver::build_chunk(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const int64_t n = std::min<int64_t>(decl_.voc_chunk, j.kept - j.played);
    const int64_t ctx = std::min<int64_t>(decl_.voc_context, j.played);
    const int64_t f0 = j.played - ctx, nl = ctx + n;
    int32_t* cu = hptr<int32_t>(R_CU);
    int32_t* pos = hptr<int32_t>(R_POS);
    cu[0] = 0;
    cu[1] = (int32_t)nl;
    for (int64_t i = 0; i < nl; ++i) pos[i] = (int32_t)(f0 + i);
    reg_[R_CU].used = 2 * 4;
    reg_[R_POS].used = nl * 4;
    if (upload() < 0) return nullptr;
    base_batches(nl, 1, (int32_t)nl, 0, step, false, 0);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        out_[r].gen = RAD_GEN_PASS_VOCODE;
        RadAudioGenBatch& A = aout_[r];
        A.voc_lane = j.lane;
        A.voc_window = j.windows_played;
        A.voc_latents = (int32_t)nl;
        A.voc_left = (int32_t)ctx;
        A.voc_right = 0;
        A.voc_frame0 = f0;
    }
    const int64_t samples = n * decl_.samples_per_latent;
    rb_.assign(1, Readback{ decl_.audio_out, samples * decl_.channels * 4, audio_host_ });
    pass_.kind = RAD_GEN_PASS_VOCODE;
    pass_.voc_job = j.id;
    pass_.voc_window = j.windows_played;
    pass_.voc_samples = samples;
    pass_.voc_frames = n;
    return &out_[0];
}

int AudioDriver::done() {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t now = (int64_t)rad_mono_ns();
    switch (pass_.kind) {
    case RAD_GEN_PASS_PROMPT:
        for (size_t i = 0; i < pass_.jobs.size(); ++i) {
            Job* j = job_locked(pass_.jobs[i]);
            if (!j) continue;
            j->prompt_done += pass_.rows[i];
            if (j->prompt_done >= j->T) j->req->ticket->t_prompt_ns = now;
        }
        break;
    case RAD_GEN_PASS_FRAME:
        for (uint64_t id : pass_.jobs) {
            Job* j = job_locked(id);
            if (!j || j->failed) continue;
            const int32_t st = status_host_[j->lane];
            if (st < 0) {
                j->ar_done = true;
                j->req->ticket->model_ended = true;
                if (j->kept == 0) {
                    fail_locked(*j, RAD_E_INVAL, "the model ended the piece before its first "
                                "frame; try other text or another seed");
                    continue;
                }
            } else {
                if (j->frame >= decl_.first_frame) j->kept = j->frame - decl_.first_frame + 1;
                ++j->frame;
                if (j->kept >= j->req->max_frames) j->ar_done = true;
            }
            if (j->ar_done) {
                j->req->ticket->frames = j->kept;
                j->req->ticket->t_frames_ns = now;
                /* Every frame already played: nothing more is owed. */
                if (chunked() && j->played == j->kept) {
                    j->finished = true;
                    j->req->ticket->windows = j->windows_played;
                    j->req->ticket->t_done_ns = now;
                    j->req->ticket->finish(RAD_OK);
                }
            }
        }
        break;
    case RAD_GEN_PASS_WINDOW:
        for (uint64_t id : pass_.jobs) {
            Job* j = job_locked(id);
            if (!j) continue;
            if (++j->win_step >= j->steps) {
                j->voc_due = j->win;
                j->win_step = 0;
            }
        }
        break;
    case RAD_GEN_PASS_VOCODE: {
        Job* j = job_locked(pass_.voc_job);
        if (!j) break;
        AudioChunk c;
        c.window = pass_.voc_window;
        c.samples.assign(audio_host_, audio_host_ + pass_.voc_samples * decl_.channels);
        j->req->ticket->push(std::move(c));
        if (j->windows_played == 0) j->req->ticket->t_first_audio_ns = now;
        ++j->windows_played;
        bool last;
        if (chunked()) {
            j->played += pass_.voc_frames;
            last = j->ar_done && j->played == j->kept;
        } else {
            last = window_last(*j, j->voc_due) == 1;
            j->voc_due = -1;
            ++j->win;
        }
        if (last) {
            j->finished = true;
            j->req->ticket->windows = j->windows_played;
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
