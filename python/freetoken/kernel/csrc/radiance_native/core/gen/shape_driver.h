/* shape_driver.h -- the shape generator's pass driver (shape.h; abi/rad_builder.h's
 * RadShapeGenDecl describes the passes).
 *
 * Driven by the engine thread like the other drivers (pass.h): `next()` picks and stages a pass,
 * `done()` folds its result back once every rank has run it. Requests arrive from HTTP threads
 * through `submit()` and leave through their ShapeTicket.
 *
 * A request holds a lane from admission to its last query. Its picture is prepared and its
 * schedule fixed at submit; between passes the driver only stages, except when a level of the
 * grid is complete, when it works out the points of the next (shape_refine).
 *
 * Pass priority, one pass at a time:
 *
 *   1. SLATENT, then the SQUERY passes level by level, of a lane whose denoising is done: they
 *      finish a request and free its lane.
 *   2. SCOND of an admitted request, so its steps can follow the current request's without a gap.
 *   3. SSTEP of the oldest lane with steps left. Requests are denoised one after another, not in
 *      turns: a pass is one request's, so taking turns would only delay the first.
 */
#pragma once
#include "shape.h"
#include "pass.h"
#include "device/device.h"

#include <deque>
#include <mutex>

namespace rad {
namespace gen {

class ShapeDriver final : public IPassDriver {
public:
    struct RankIO {
        int       device = 0;
        RadStream stream = nullptr;
    };

    int  init(const Program& prog, const std::vector<RankIO>& ranks);
    void fini();
    ~ShapeDriver() override { fini(); }

    /* The bytes init() allocates on each rank's card, for the VRAM budget. */
    static int64_t device_bytes(const Program& prog);

    /* ---------------------------------------------------------- HTTP threads */
    /* Queue a request, refused at once with `why` when it can never run. */
    int  submit(std::unique_ptr<ShapeRequest> r, std::string* why);
    const ShapeGenInfo& info() const { return info_; }
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
        std::unique_ptr<ShapeRequest> req;
        std::vector<float> sigmas;       /* the request's grid points */
        float    cfg = 0.0f;
        int32_t  evals = 0;              /* steps the request takes: grid points less one */
        int32_t  lane = -1;
        bool     admitted = false;
        bool     cond_done = false;
        int32_t  step = 0;               /* steps done */
        bool     latent_done = false;
        /* The grid: its levels, the one being sampled, its field (NaN where not sampled) and
         * the points it samples, all of them when `all` (the coarsest level, and dense
         * sampling); `next` is the first not yet queried. */
        std::vector<int32_t>  levels;
        size_t                level = 0;
        std::vector<float>    field;
        std::vector<float>    axis;
        std::vector<uint32_t> pts;
        bool                  all = false;
        int64_t               n_pts = 0, next = 0;
        int64_t               points = 0;     /* every level's, for the usage */
        bool     failed = false;
        bool     finished = false;
    };
    struct Pass {
        int      kind = 0;
        uint64_t job = 0;
        int64_t  n_query = 0;
    };
    struct Region { int64_t off = 0, bytes = 0, used = 0; };
    enum { R_PATCH, R_SIGMA, R_SNEXT, R_CFG, R_PAIR, R_LANES, R_FLAGS, R_SEED, R_LATCU, R_QUERY,
           R_QCU, R_N };

    template <class T> T* hptr(int r) const { return reinterpret_cast<T*>(host_ + reg_[r].off); }
    template <class T> T* dptr(int rank, int r) const {
        return reinterpret_cast<T*>((char*)dev_[(size_t)rank] + reg_[r].off);
    }
    int  upload();
    /* What every pass carries: the lane twice (the step's two entries) and the latents' bounds. */
    void stage_common(const Job& j);
    void base_batches(const Job& j, int64_t n_tok, int kind, int step);
    void start_level(Job& j, size_t level, std::vector<uint32_t>&& pts, bool all);

    void admit_locked();
    void release_locked(Job& j);
    void fail_locked(Job& j, int status, const std::string& why);
    Job* job_locked(uint64_t id);

    const RadBatch* build_cond(Job& j, int step);
    const RadBatch* build_step(Job& j, int step);
    const RadBatch* build_latent(Job& j, int step);
    const RadBatch* build_query(Job& j, int step);

    RadShapeGenDecl     decl_{};
    ShapeGenInfo        info_;
    int64_t             n_patch_ = 0, patch_dim_ = 0;
    std::vector<RankIO> ranks_;

    char*               host_ = nullptr;   /* pinned, same layout as each slab */
    int64_t             bytes_ = 0;
    std::vector<void*>  dev_;
    Region              reg_[R_N];
    float*              field_host_ = nullptr;
    std::vector<Readback> rb_;

    std::vector<RadBatch>         out_;      /* one per rank */
    std::vector<RadShapeGenBatch> sout_;
    Pass                          pass_;

    mutable std::mutex               mu_;
    std::deque<std::unique_ptr<Job>> jobs_;  /* arrival order */
    std::vector<int32_t>             free_lanes_;
    uint64_t                         next_id_ = 1;
    bool                             closed_ = false;
    std::string                      closed_why_;
    int                              closed_status_ = RAD_E_STATE;
};

}  /* namespace gen */
}  /* namespace rad */
