/* audio_driver.h -- the audio generator's pass driver (audio.h, abi/rad_builder.h's
 * RadAudioGenDecl describes the passes).
 *
 * Driven by the engine thread like the image driver (driver.h): `next()` picks and stages a pass,
 * `done()` folds its result back once every rank has run it. Requests arrive from HTTP threads
 * through `submit()` and leave through their AudioTicket.
 *
 * A request holds a lane and two KV sequences, its conditional and unconditional branches (one,
 * for a model of one branch: RadAudioGenDecl::branches), from admission to its last window; they
 * are reserved for the whole piece at admission (prompt plus every frame the request may make), so
 * a request in flight never waits for KV.
 *
 * The language model's KV is one paged full-attention group, beside which a hybrid model keeps
 * the linear-attention state and convolution groups of its recurrent layers (RAD_KV_LINEAR,
 * RAD_KV_CONV): one slot a sequence each, staged as their `state_index` rows, zeroed by the engine
 * before the pass that first reads them.
 *
 * Pass priority, one pass at a time:
 *
 *   1. VOCODE a finished window is owed (it frees the lane's latent for the next window and gives
 *      the caller audio), or, for a vocoder that plays frames as they are made
 *      (RadAudioGenDecl::voc_chunk), a chunk of frames is ready: `voc_chunk` of them, or the
 *      piece's last.
 *   2. PROMPT an admitted request's prompt is not all in the KV: the next chunk of it, both
 *      branches.
 *   3. FRAME and WINDOW, taking turns when both have work: one frame of every request in its
 *      autoregressive stage, or one denoising step of the windows that are ready. A FRAME pass
 *      carries every lane at once, since its cost is reading the language model's weights, which
 *      the lanes share; a window's step is compute and carries what fits the pass's rows.
 *
 * A window is ready once the language model has made all its frames, or ended the piece. Its
 * vocoding waits until it is known whether a window follows it (the crop at its end depends on
 * that): once the frames of the next window have begun, or the piece has ended.
 */
#pragma once
#include "audio.h"
#include "pass.h"
#include "device/device.h"

#include <mutex>

namespace rad {
struct IKVManager;
namespace gen {

class AudioDriver final : public IPassDriver {
public:
    struct RankIO {
        int       device = 0;
        RadStream stream = nullptr;
    };

    /* `kv` is rank 0's KV manager: one block table serves every rank. */
    int  init(const Program& prog, IKVManager* kv, const std::vector<RankIO>& ranks);
    void fini();
    ~AudioDriver() override { fini(); }

    /* The bytes init() allocates on each rank's card, for the VRAM budget. */
    static int64_t device_bytes(const Program& prog);
    /* The longest sequence a branch holds: the whole prompt and every frame. The KV group is
     * sized from it. */
    static int64_t max_seq_rows(const RadAudioGenDecl& d) { return d.max_prompt + d.max_frames + 1; }
    /* State slots a state_index row holds at most: the state, and the scratch a kernel serving
     * speculation asks for (RadKVGroupBatch::state_index_pitch), which a generator never runs. */
    static constexpr int64_t kMaxStateWidth = 8;

    /* ---------------------------------------------------------- HTTP threads */
    /* Queue a request, refused at once with `why` when it can never run. */
    int  submit(std::unique_ptr<AudioRequest> r, std::string* why);
    const RadAudioGenDecl& decl() const { return decl_; }
    void shutdown(const char* why, int status) override;

    /* ---------------------------------------------------------- the engine thread */
    const RadBatch* next(int step) override;
    const RadBatch* batch_for(int rank) const override;
    const std::vector<Readback>& readbacks() const override { return rb_; }
    int  done() override;

    int64_t queued() const override;
    int64_t running() const override;

private:
    struct Job {
        uint64_t id = 0;
        std::unique_ptr<AudioRequest> req;
        int32_t  lane = -1;
        uint64_t seq[2] = { 0, 0 };
        bool     have_kv = false;
        bool     admitted = false;
        bool     failed = false;
        int64_t  T = 0;               /* prompt tokens */
        std::vector<int32_t> neg;     /* the unconditional prompt */
        int64_t  Tn = 0;              /* its tokens */
        int64_t  prompt_done = 0;     /* prompt rows in the KV */
        /* The autoregressive stage: the next frame to make, the frames kept, and whether it is
         * over (the model ended the piece, or it reached max_frames). */
        int32_t  frame = 0;
        int64_t  kept = 0;
        bool     ar_done = false;
        /* The windows: the one being stepped (or next to start), its step, and the one owed a
         * vocode pass (-1 none). */
        int32_t  win = 0;
        int32_t  win_step = 0;
        int32_t  steps = 0;
        float    cfg = 1.0f;
        int32_t  voc_due = -1;
        int32_t  windows_played = 0;
        /* A vocoder playing frames as they are made: the kept frames played so far. */
        int64_t  played = 0;
        bool     finished = false;
    };
    /* The pass in flight, for done(). */
    struct Pass {
        int      kind = 0;
        std::vector<uint64_t> jobs;   /* PROMPT, FRAME, WINDOW: one per lane or window */
        std::vector<int64_t>  rows;   /* PROMPT: the chunk each job's branches advanced */
        uint64_t voc_job = 0;
        int32_t  voc_window = -1;
        int64_t  voc_samples = 0;
        int64_t  voc_frames = 0;      /* a chunk's frames played */
    };

    /* Staging regions: their offsets in the host span and every rank's slab. */
    struct Region { int64_t off = 0, bytes = 0, used = 0; };
    enum { R_TOK, R_POS, R_CU, R_SLOT, R_BT, R_USED, R_LANE, R_LAST, R_FRAME, R_SEED,
           R_T, R_TN, R_CFG, R_PAIR, R_WLANE, R_FLAGS, R_WIDX, R_WF0, R_WNF, R_LCU, R_LSRC,
           R_ESLOT, R_LROW, R_PREFIX, R_CTX, R_ACC, R_SBT, R_SIDX, R_PSLOT, R_PPOS, R_PUSED,
           R_RAUD, R_RCU, R_RFRM, R_N };
    /* A hybrid model's stateful groups (linear-attention state, convolution history), each staged
     * as one block-table word a sequence (its slot) and its state_index row. */
    struct StateGroup { int32_t g = 0; int64_t width = 1; };

    template <class T> T* hptr(int r) const { return reinterpret_cast<T*>(host_ + reg_[r].off); }
    template <class T> T* dptr(int rank, int r) const {
        return reinterpret_cast<T*>((char*)dev_[(size_t)rank] + reg_[r].off);
    }
    int  upload();
    void base_batches(int64_t n_tok, int64_t n_seq, int32_t max_q, int32_t max_ctx, int step,
                      bool kv, int64_t bt_pitch);
    /* The stateful groups' rows for the entries of a PROMPT or FRAME pass, entry e being `seq[e]`,
     * with each entry's context before the pass (ctx_lens). */
    void stage_state(const std::vector<uint64_t>& seqs, const std::vector<int32_t>& ctx);

    int  admit_locked();
    void release_locked(Job& j);
    void fail_locked(Job& j, int status, const std::string& why);
    Job* job_locked(uint64_t id);

    /* Window geometry: window k's first frame and frames once the piece has `n` frames (or
     * at least `n`, with `ended` false), whether it exists, and whether it is the last. */
    int64_t win_frame0(int32_t k) const { return (int64_t)k * decl_.win_hop; }
    int64_t win_latents(int64_t frames) const { return frames * decl_.lat_num / decl_.lat_den; }
    bool    window_ready(const Job& j) const;
    /* A vocoder playing frames as they are made: whether a chunk of `j`'s frames is due. */
    bool    chunk_due(const Job& j) const;
    bool    chunked() const { return decl_.voc_chunk > 0; }
    int     window_last(const Job& j, int32_t k) const;   /* 1 last, 0 not, -1 not yet known */
    int64_t window_frames(const Job& j, int32_t k) const;

    const RadBatch* build_prompt(Job& j, int step);
    const RadBatch* build_frame(const std::vector<Job*>& js, int step);
    const RadBatch* build_window(const std::vector<Job*>& js, int step);
    const RadBatch* build_vocode(Job& j, int step);
    const RadBatch* build_chunk(Job& j, int step);

    RadAudioGenDecl     decl_{};
    std::string         name_;
    int64_t             max_tok_ = 0, max_seqs_ = 0;
    int64_t             block_size_ = 0, max_blocks_ = 0;
    int32_t             kv_group_ = 0;
    std::vector<StateGroup> state_groups_;
    int                 nb_ = 2;            /* sequences a request: RadAudioGenDecl::branches */
    bool                neg_own_ = false;   /* the unconditional prompt is the request's own */
    std::vector<RankIO> ranks_;
    IKVManager*         kv_ = nullptr;

    char*               host_ = nullptr;   /* pinned, same layout as each slab */
    int64_t             bytes_ = 0;
    std::vector<void*>  dev_;
    Region              reg_[R_N];
    int32_t*            status_host_ = nullptr;
    float*              audio_host_ = nullptr;
    int64_t             audio_bytes_ = 0;
    std::vector<Readback> rb_;

    std::vector<RadBatch>         out_;      /* one per rank */
    std::vector<RadAudioGenBatch> aout_;
    std::vector<RadKVGroupBatch>  kout_;    /* [rank][1 + state groups] */
    Pass                          pass_;
    bool                          frame_turn_ = true;   /* FRAME goes first when both are due */
    /* Host arrays a pass's plugin reads on the host (the _h fields), for the life of the pass. */
    std::vector<int32_t>          h_lane_, h_last_, h_frame_, h_wlane_, h_flags_, h_widx_,
                                  h_wf0_, h_wnf_, h_lcu_, h_rcu_, h_rfrm_;

    mutable std::mutex              mu_;
    std::deque<std::unique_ptr<Job>> jobs_;  /* arrival order */
    std::vector<int32_t>            free_lanes_;
    uint64_t                        next_id_ = 1;
    uint64_t                        next_seq_ = 1;
    bool                            closed_ = false;
    std::string                     closed_why_;
    int                             closed_status_ = RAD_E_STATE;
};

}  /* namespace gen */
}  /* namespace rad */
