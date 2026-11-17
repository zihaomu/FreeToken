/* driver.h -- the image generator's pass driver (gen.h describes the passes).
 *
 * Driven by the engine thread: `next()` picks and stages a pass, `done()` folds its result back
 * once every rank has run it. Requests arrive from HTTP threads through `submit()` and leave
 * through their Ticket; the job queue is the only shared state, behind one mutex.
 *
 * Pass priority, one pass at a time:
 *
 *   1. DECODE an image is owed: a finished image, or a preview its last step marked. Goes first
 *      because a finished image frees its lane and answers its caller, and a preview must be read
 *      before the lane's next step overwrites the preview latent.
 *   2. COND an admitted request is owed, so its images can join the next denoise pass.
 *   3. DENOISE over as many in-flight images as fit the step's rows, oldest request first, each
 *      at its own step.
 *
 * A request is admitted once it can get lanes for every image and KV for every conditioning
 * sequence. Admission is in order: a request that does not fit blocks those behind it, so a large
 * request is not starved by a stream of small ones.
 *
 * Staging is one device region per rank plus one pinned host region with the same layout. Every
 * pass is synchronised before the next is built (Engine::gen_loop), so one host region suffices:
 * a pass takes tens of milliseconds, the host work between passes microseconds.
 */
#pragma once
#include "gen.h"
#include "pass.h"
#include "device/device.h"

#include <mutex>
#include <unordered_map>

namespace rad {
namespace gen {

class Driver final : public IPassDriver {
public:
    struct RankIO {
        int       device = 0;
        RadStream stream = nullptr;
    };

    /* `kv` is rank 0's KV manager: one block table serves every rank. */
    int  init(const Program& prog, IKVManager* kv, const std::vector<RankIO>& ranks);
    void fini();
    ~Driver() { fini(); }

    /* The bytes init() allocates on each rank's card, for the VRAM budget. */
    static int64_t device_bytes(const Program& prog);

    /* ---------------------------------------------------------- HTTP threads */
    /* Queue a request. Refused at once, with `why`, when it can never run: a size past the
     * largest image, more pictures than the model takes, a conditioning longer than a pass. */
    int  submit(std::unique_ptr<Request> r, std::string* why);
    /* The model's limits, for the server to check before the expensive part of preparing a
     * request. */
    const RadGeneratorDecl&    decl() const { return decl_; }
    int64_t max_cond_tokens() const { return max_tok_; }
    void shutdown(const char* why, int status) override;

    /* ---------------------------------------------------------- the engine thread */
    const RadBatch* next(int step) override;
    const RadBatch* batch_for(int rank) const override;
    int32_t lora_slot() const override { return pass_.lora_slot; }
    /* A decode pass's pixels: `image_out` into the pinned image buffer. */
    const std::vector<Readback>& readbacks() const override { return rb_; }
    int  done() override;
    bool busy() const;

    int64_t queued() const override;
    int64_t running() const override;

private:
    struct Cond {
        Layout   lay;
        uint64_t seq = 0;          /* the KV manager's key */
        bool     have_kv = false;
        bool     encoded = false;
        std::vector<int32_t> rope_img;   /* [3, image tokens]: the image's rows at this sequence */
        std::vector<float>   rope_img_f; /* JOINT_STEP: [4, image tokens] */
    };
    struct Job {
        uint64_t id = 0;
        std::unique_ptr<Request> req;
        int      n_branch = 1;
        Cond     cond[2];
        int32_t  h_lat = 0, w_lat = 0;
        int64_t  tokens = 0;
        std::vector<float> sigmas;
        /* A discrete-time model's: each step's timestep, beside `sigmas`. */
        std::vector<float> tsteps;
        std::vector<int32_t> lanes;
        std::vector<std::pair<int32_t, int32_t>> ref_hw;
        int32_t  images_done = 0;
        bool     admitted = false;
        bool     failed = false;
    };
    struct Lane {
        bool     busy = false;      /* held by request job_id */
        uint64_t job_id = 0;
        int32_t  index = 0;
        int32_t  step = 0;
        uint64_t seed = 0;
        std::vector<int32_t> preview_at;   /* the steps after which a preview is decoded */
        int32_t  previews_done = 0;
        bool     preview_due = false;
        bool     final_due = false;
        bool     finished = false;
    };
    /* The pass in flight, for done(). */
    struct Pass {
        int      kind = 0;
        uint64_t job = 0;
        int      branch = 0;
        int32_t  lane = -1;
        bool     preview = false;
        std::vector<int32_t> lanes;   /* DENOISE: the lanes it stepped, one per image */
        std::vector<char>    marks;   /* ... and whether each was marked for a preview */
        int32_t  lora_slot = -1;      /* every image of a pass runs with the same adapters */
    };

    /* Staging regions: their offsets in the host span and every rank's slab. */
    struct Region { int64_t off = 0, bytes = 0, used = 0; };
    enum { R_TOK, R_POS, R_ROPE, R_CU, R_MMROW, R_ENCPIX, R_ENCCRD, R_ENCCU, R_SRC, R_GROPE,
           R_SEG, R_SLOT, R_PX, R_MASK, R_SIG, R_SIGN, R_CFG, R_PAIR, R_LANE, R_FLAGS, R_SEED,
           R_BT, R_USED, R_ROPEF, R_ICU, R_IROWS, R_MCU, R_VCU, R_MROWS, R_MSRC, R_MCCU, R_MROPE,
           R_AUX, R_SIGP, R_TSTEP, R_SAMP, R_STEPI, R_SIZE, R_N };

    template <class T> T* hptr(int r) const { return reinterpret_cast<T*>(host_ + reg_[r].off); }
    template <class T> T* dptr(int rank, int r) const {
        return reinterpret_cast<T*>((char*)dev_[(size_t)rank] + reg_[r].off);
    }
    int  upload();
    void finish_batches(int64_t n_tok, int64_t n_seq, int32_t max_q, int step);

    int  admit_locked();
    void release_kv_locked(Job& j);
    void release_locked(Job& j);
    void fail_locked(Job& j, int status, const std::string& why);
    Job* job_locked(uint64_t id);

    /* The JOINT_STEP layout: the conditioning rides in every denoising entry, and no KV. */
    bool joint() const { return decl_.layout == RAD_GEN_LAYOUT_JOINT_STEP; }
    /* An entry's rows in a denoising pass. */
    int64_t entry_rows(const Job& j, int branch) const {
        return j.tokens + (joint() ? j.cond[branch].lay.rows : 0);
    }

    const RadBatch* build_cond(Job& j, int branch, int step);
    const RadBatch* build_denoise(const std::vector<int32_t>& lanes, int step);
    const RadBatch* build_decode(int32_t lane, bool preview, int step);

    RadGeneratorDecl      decl_{};
    std::vector<double>   fixed_;
    /* A discrete-time model's sigma table (RadGeneratorDecl::train_steps), empty otherwise. */
    DiscreteTable         table_;
    bool discrete() const { return decl_.train_steps > 0; }
    int64_t               max_tok_ = 0, max_seqs_ = 0, max_enc_ = 0, patch_dim_ = 0;
    int64_t               max_rows_ = 0;     /* the longest conditioning sequence */
    int64_t               max_px_ = 0;       /* pixels the staging holds */
    int64_t               block_size_ = 0, max_blocks_ = 0;
    int32_t               kv_group_ = 0;
    std::vector<RankIO>   ranks_;
    IKVManager*           kv_ = nullptr;

    char*                 host_ = nullptr;   /* pinned, same layout as each slab */
    int64_t               bytes_ = 0;
    std::vector<void*>    dev_;
    Region                reg_[R_N];
    void*                 img_host_ = nullptr;
    int64_t               img_bytes_ = 0;
    int64_t               dec_bytes_ = 0;
    std::vector<Readback> rb_;

    std::vector<RadBatch>        out_;     /* one per rank */
    std::vector<RadGenBatch>     gout_;
    std::vector<RadKVGroupBatch> kout_;
    Pass                         pass_;
    /* Host arrays a COND pass's plugin reads on the host (RadGenBatch's _h fields). */
    std::vector<int32_t>         host_bidi_, host_px_, host_lanes_;
    /* A DENOISE pass's per-entry host array (RadGenBatch::entry_h). */
    std::vector<int32_t>         host_entry_;
    int64_t                      host_nmode_[3] = { 0, 0, 0 };

    mutable std::mutex           mu_;
    std::deque<std::unique_ptr<Job>> jobs_;  /* arrival order */
    std::vector<Lane>            lanes_;
    std::vector<int32_t>         free_lanes_;
    uint64_t                     next_id_ = 1;
    uint64_t                     next_seq_ = 1;
    bool                         closed_ = false;
    std::string                  closed_why_;
    int                          closed_status_ = RAD_E_STATE;
};

}  /* namespace gen */
}  /* namespace rad */
