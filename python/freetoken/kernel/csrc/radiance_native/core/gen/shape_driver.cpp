/* shape_driver.cpp -- the shape generator's pass driver (shape_driver.h). */
#include "shape_driver.h"
#include "rad_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rad {
namespace gen {

namespace {

constexpr int64_t kAlign = 256;

/* Region offsets and sizes, in the enum's order (shape_driver.h), and the total; shared by the
 * allocation and the VRAM budget. */
int64_t carve(const RadShapeGenDecl& d, int64_t (*bytes)[2], int n) {
    const int64_t side = d.enc_size / d.enc_patch;
    const int64_t want[] = {
        side * side * 3 * d.enc_patch * d.enc_patch * 4,   /* R_PATCH: f32 */
        2 * 4,                 /* R_SIGMA */
        2 * 4,                 /* R_SNEXT */
        2 * 4,                 /* R_CFG */
        2 * 4,                 /* R_PAIR */
        2 * 4,                 /* R_LANES */
        2 * 4,                 /* R_FLAGS */
        2 * 2 * 4,             /* R_SEED */
        3 * 4,                 /* R_LATCU */
        d.max_query * 3 * 4,   /* R_QUERY: f32 */
        2 * 4,                 /* R_QCU */
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

int64_t ShapeDriver::device_bytes(const Program& prog) {
    if (!prog.is_shape_generator()) return 0;
    int64_t b[R_N][2];
    return carve(prog.shape_gen, b, R_N);
}

int ShapeDriver::init(const Program& prog, const std::vector<RankIO>& ranks) {
    fini();
    if (!prog.is_shape_generator()) return RAD_E_STATE;
    decl_ = prog.shape_gen;
    info_ = shape_info(decl_, prog.sgen_name);
    ranks_ = ranks;
    n_patch_ = (decl_.enc_size / decl_.enc_patch) * (decl_.enc_size / decl_.enc_patch);
    patch_dim_ = 3 * decl_.enc_patch * decl_.enc_patch;
    int64_t b[R_N][2];
    bytes_ = carve(decl_, b, R_N);
    for (int r = 0; r < R_N; ++r) {
        reg_[r].off = b[r][0];
        reg_[r].bytes = b[r][1];
        reg_[r].used = 0;
    }
    host_ = static_cast<char*>(rad_dev_alloc(bytes_, RAD_MEM_HOST_PINNED));
    field_host_ = static_cast<float*>(rad_dev_alloc(decl_.max_query * 4, RAD_MEM_HOST_PINNED));
    if (!host_ || !field_host_) {
        RAD_ERR("shape generator: no pinned host memory for the pass staging and the field "
                "(%lld + %lld bytes)", (long long)bytes_, (long long)(decl_.max_query * 4));
        return RAD_E_NOMEM;
    }
    std::memset(host_, 0, (size_t)bytes_);
    dev_.assign(ranks_.size(), nullptr);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r].device));
        dev_[r] = rad_dev_alloc(bytes_, RAD_MEM_DEVICE);
        if (!dev_[r]) {
            RAD_ERR("shape generator: rank %zu has no room for the pass staging (%lld bytes)", r,
                    (long long)bytes_);
            return RAD_E_NOMEM;
        }
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0].device));
    out_.assign(ranks_.size(), RadBatch{});
    sout_.assign(ranks_.size(), RadShapeGenBatch{});
    free_lanes_.clear();
    for (int64_t i = decl_.max_lanes - 1; i >= 0; --i) free_lanes_.push_back((int32_t)i);
    return RAD_OK;
}

void ShapeDriver::fini() {
    for (size_t r = 0; r < dev_.size(); ++r)
        if (dev_[r]) {
            if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
            rad_dev_free(dev_[r], RAD_MEM_DEVICE);
        }
    dev_.clear();
    if (!ranks_.empty()) rad_dev_set(ranks_[0].device);
    if (host_) rad_dev_free(host_, RAD_MEM_HOST_PINNED);
    if (field_host_) rad_dev_free(field_host_, RAD_MEM_HOST_PINNED);
    host_ = nullptr;
    field_host_ = nullptr;
}

/* ================================================================== requests */
int ShapeDriver::submit(std::unique_ptr<ShapeRequest> r, std::string* why) {
    if (!r || !r->ticket) return RAD_E_INVAL;
    const RadShapeGenDecl& d = decl_;
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    if ((int64_t)r->patches.size() != n_patch_ * patch_dim_)
        return no("the picture is not prepared for this model's image encoder");
    const int32_t steps = r->steps > 0 ? r->steps : (int32_t)d.default_steps;
    if (steps < 2 || steps > d.max_steps)
        return no("num_inference_steps must be 2.." + std::to_string(d.max_steps) +
                  " (sigma grid points, the last one the clean shape)");
    const double cfg = std::isnan(r->guidance) ? d.default_guidance : r->guidance;
    if (!(cfg >= 0.0) || !std::isfinite(cfg))
        return no("guidance_scale must be finite and not negative");
    const int32_t res = r->resolution > 0 ? r->resolution : (int32_t)d.default_resolution;
    if (res < d.min_resolution || res > d.max_resolution)
        return no("octree_resolution must be " + std::to_string(d.min_resolution) + ".." +
                  std::to_string(d.max_resolution));

    auto j = std::make_unique<Job>();
    j->sigmas = shape_sigmas(steps);
    j->evals = steps - 1;
    j->cfg = (float)cfg;
    j->levels = r->dense ? std::vector<int32_t>{ res } : shape_levels(res);
    ShapeTicket& t = *r->ticket;
    t.steps = j->evals;
    t.resolution = res;
    t.guidance = cfg;
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

void ShapeDriver::shutdown(const char* why, int status) {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    closed_why_ = why ? why : "the engine stopped";
    closed_status_ = status;
    for (auto& j : jobs_) j->req->ticket->finish(status, closed_why_);
}

int64_t ShapeDriver::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (!j->admitted) ++n;
    return n;
}

int64_t ShapeDriver::running() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (j->admitted) ++n;
    return n;
}

ShapeDriver::Job* ShapeDriver::job_locked(uint64_t id) {
    for (auto& j : jobs_) if (j->id == id) return j.get();
    return nullptr;
}

void ShapeDriver::release_locked(Job& j) {
    if (j.lane >= 0) {
        free_lanes_.push_back(j.lane);
        j.lane = -1;
    }
}

void ShapeDriver::fail_locked(Job& j, int status, const std::string& why) {
    j.failed = true;
    j.req->ticket->finish(status, why);
}

/* In arrival order: a request needs only a lane. */
void ShapeDriver::admit_locked() {
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted || j.failed) continue;
        if (free_lanes_.empty()) break;
        j.lane = free_lanes_.back();
        free_lanes_.pop_back();
        j.admitted = true;
    }
}

/* A level of the grid begins: its field unsampled, its points `pts` (every point when `all`). */
void ShapeDriver::start_level(Job& j, size_t level, std::vector<uint32_t>&& pts, bool all) {
    const int32_t r = j.levels[level];
    const int64_t n = (int64_t)r + 1;
    j.level = level;
    j.field.assign((size_t)(n * n * n), NAN);
    j.axis = shape_axis(decl_.bounds, r);
    j.all = all;
    j.pts = std::move(pts);
    j.n_pts = all ? n * n * n : (int64_t)j.pts.size();
    j.next = 0;
}

/* ================================================================== staging */
int ShapeDriver::upload() {
    for (size_t r = 0; r < ranks_.size(); ++r)
        for (int g = 0; g < R_N; ++g) {
            if (reg_[g].used <= 0) continue;
            RAD_TRY(rad_memcpy_async((char*)dev_[r] + reg_[g].off, host_ + reg_[g].off,
                                     reg_[g].used, ranks_[r].stream));
        }
    return RAD_OK;
}

void ShapeDriver::stage_common(const Job& j) {
    const int32_t lanes[2] = { j.lane, j.lane };
    const int32_t cu[3] = { 0, (int32_t)decl_.latent_rows, (int32_t)(2 * decl_.latent_rows) };
    std::memcpy(hptr<int32_t>(R_LANES), lanes, sizeof lanes);
    std::memcpy(hptr<int32_t>(R_LATCU), cu, sizeof cu);
    reg_[R_LANES].used = sizeof lanes;
    reg_[R_LATCU].used = sizeof cu;
}

void ShapeDriver::base_batches(const Job& j, int64_t n_tok, int kind, int step) {
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        RadShapeGenBatch& S = sout_[r];
        const int ri = (int)r;
        B = RadBatch{};
        B.step = step;
        B.phase = RAD_PHASE_PREFILL;
        B.n_tok = n_tok;
        B.n_seq = 1;
        B.max_q_len = (int32_t)n_tok;
        B.max_q_len_prefill = (int32_t)n_tok;
        B.gen = kind;
        S = RadShapeGenBatch{};
        S.lane = j.lane;
        S.patches = dptr<float>(ri, R_PATCH);
        S.step = j.step;
        S.n_steps = j.evals;
        S.sigma = dptr<float>(ri, R_SIGMA);
        S.sigma_next = dptr<float>(ri, R_SNEXT);
        S.cfg = dptr<float>(ri, R_CFG);
        S.pair = dptr<int32_t>(ri, R_PAIR);
        S.lanes = dptr<int32_t>(ri, R_LANES);
        S.flags = dptr<int32_t>(ri, R_FLAGS);
        S.seed = dptr<uint32_t>(ri, R_SEED);
        S.lat_cu = dptr<int32_t>(ri, R_LATCU);
        S.query = dptr<float>(ri, R_QUERY);
        S.query_cu = dptr<int32_t>(ri, R_QCU);
        B.sgen_b = &S;
    }
    pass_.kind = kind;
    pass_.job = j.id;
}

/* ================================================================== the passes */
const RadBatch* ShapeDriver::next(int step) {
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

    /* 1. The decoding of a denoised lane: its latent, then its grid. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed || !j.cond_done || j.step < j.evals) continue;
        return j.latent_done ? build_query(j, step) : build_latent(j, step);
    }
    /* 2. A picture. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted && !j.failed && !j.cond_done) return build_cond(j, step);
    }
    /* 3. The oldest lane's next step. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted && !j.failed && j.cond_done && j.step < j.evals) return build_step(j, step);
    }
    return nullptr;
}

const RadBatch* ShapeDriver::batch_for(int rank) const {
    if (rank < 0 || (size_t)rank >= out_.size() || pass_.kind == 0) return nullptr;
    return &out_[(size_t)rank];
}

const RadBatch* ShapeDriver::build_cond(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    std::memcpy(hptr<float>(R_PATCH), j.req->patches.data(), j.req->patches.size() * 4);
    reg_[R_PATCH].used = (int64_t)j.req->patches.size() * 4;
    stage_common(j);
    if (upload() < 0) return nullptr;
    base_batches(j, n_patch_, RAD_GEN_PASS_SCOND, step);
    return &out_[0];
}

/* The two entries of the step: the picture's latent guided against the unconditioned one, which
 * reads the same lane and moves nothing. */
const RadBatch* ShapeDriver::build_step(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const float s = j.sigmas[(size_t)j.step], sn = j.sigmas[(size_t)j.step + 1];
    const float sig[2] = { s, s }, nxt[2] = { sn, sn }, cfg[2] = { j.cfg, 1.0f };
    const int32_t pair[2] = { 1, -2 };
    const int32_t flags[2] = { j.step == 0 ? (int32_t)RAD_GEN_F_FIRST : 0, 0 };
    const uint32_t lo = (uint32_t)j.req->seed, hi = (uint32_t)(j.req->seed >> 32);
    const uint32_t seed[4] = { lo, hi, lo, hi };
    auto stage = [&](int r, const void* p, int64_t n) {
        std::memcpy(host_ + reg_[r].off, p, (size_t)n);
        reg_[r].used = n;
    };
    stage(R_SIGMA, sig, sizeof sig);
    stage(R_SNEXT, nxt, sizeof nxt);
    stage(R_CFG, cfg, sizeof cfg);
    stage(R_PAIR, pair, sizeof pair);
    stage(R_FLAGS, flags, sizeof flags);
    stage(R_SEED, seed, sizeof seed);
    stage_common(j);
    if (upload() < 0) return nullptr;
    base_batches(j, decl_.step_rows, RAD_GEN_PASS_SSTEP, step);
    return &out_[0];
}

const RadBatch* ShapeDriver::build_latent(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    stage_common(j);
    if (upload() < 0) return nullptr;
    base_batches(j, decl_.latent_rows, RAD_GEN_PASS_SLATENT, step);
    return &out_[0];
}

/* The level's next run of points, at most max_query, as x, y, z. */
const RadBatch* ShapeDriver::build_query(Job& j, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const int64_t n = std::min<int64_t>(decl_.max_query, j.n_pts - j.next);
    const int64_t side = (int64_t)j.levels[j.level] + 1;
    float* q = hptr<float>(R_QUERY);
    const float* ax = j.axis.data();
    if (j.all) {
        /* Consecutive points: the grid coordinates count up like an odometer, with no division
         * past the first point's. */
        int64_t z = j.next / (side * side), y = j.next / side % side, x = j.next % side;
        for (int64_t i = 0; i < n; ++i) {
            q[3 * i]     = ax[z];
            q[3 * i + 1] = ax[y];
            q[3 * i + 2] = ax[x];
            if (++x == side) {
                x = 0;
                if (++y == side) { y = 0; ++z; }
            }
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const int64_t p = (int64_t)j.pts[(size_t)(j.next + i)];
            q[3 * i]     = ax[p / (side * side)];
            q[3 * i + 1] = ax[p / side % side];
            q[3 * i + 2] = ax[p % side];
        }
    }
    reg_[R_QUERY].used = n * 3 * 4;
    const int32_t qcu[2] = { 0, (int32_t)n };
    std::memcpy(hptr<int32_t>(R_QCU), qcu, sizeof qcu);
    reg_[R_QCU].used = sizeof qcu;
    stage_common(j);
    if (upload() < 0) return nullptr;
    base_batches(j, n, RAD_GEN_PASS_SQUERY, step);
    for (auto& S : sout_) S.n_query = n;
    pass_.n_query = n;
    rb_.assign(1, Readback{ decl_.field_out, n * 4, field_host_ });
    return &out_[0];
}

int ShapeDriver::done() {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t now = (int64_t)rad_mono_ns();
    Job* j = job_locked(pass_.job);
    switch (j ? pass_.kind : 0) {
    case RAD_GEN_PASS_SCOND:
        j->cond_done = true;
        j->req->ticket->t_cond_ns = now;
        /* The patches are on the card for good; the request keeps no copy. */
        std::vector<float>().swap(j->req->patches);
        break;
    case RAD_GEN_PASS_SSTEP:
        ++j->step;
        j->req->ticket->progress(j->step, 0);
        if (j->step >= j->evals) j->req->ticket->t_denoised_ns = now;
        break;
    case RAD_GEN_PASS_SLATENT:
        j->latent_done = true;
        start_level(*j, 0, {}, true);
        break;
    case RAD_GEN_PASS_SQUERY: {
        const int64_t n = pass_.n_query;
        if (j->all) {
            std::memcpy(j->field.data() + j->next, field_host_, (size_t)n * sizeof(float));
        } else {
            for (int64_t i = 0; i < n; ++i)
                j->field[(size_t)j->pts[(size_t)(j->next + i)]] = field_host_[i];
        }
        j->next += n;
        j->points += n;
        j->req->ticket->progress(j->step, (int32_t)std::min<int64_t>(j->points, INT32_MAX));
        if (j->next < j->n_pts) break;
        /* The level is complete: the next one's points, or the end (also when no point of this
         * level is near the surface, which leaves the finer levels nothing to sample). */
        if (j->level + 1 < j->levels.size()) {
            const int32_t rp = j->levels[j->level], r = j->levels[j->level + 1];
            const bool last = j->level + 2 == j->levels.size();
            std::vector<uint32_t> pts = shape_refine(j->field, rp, r, decl_.level, last);
            if (!pts.empty()) {
                start_level(*j, j->level + 1, std::move(pts), false);
                break;
            }
        }
        ShapeTicket& t = *j->req->ticket;
        t.points = j->points;
        t.t_done_ns = now;
        t.set_field(std::move(j->field), j->levels[j->level]);
        j->finished = true;
        t.finish(RAD_OK);
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
