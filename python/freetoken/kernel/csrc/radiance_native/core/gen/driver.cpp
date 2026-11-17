/* driver.cpp -- the image generator's pass driver (driver.h). */
#include "driver.h"
#include "sched/mem_iface.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rad {
namespace gen {

namespace {

constexpr int64_t kAlign = 256;

/* Upper bound on a reference picture's latent tokens: its area plus slack for rounding its sides
 * to the size multiple. */
int64_t ref_tokens_max(const RadGeneratorDecl& d) {
    const int64_t s2 = d.latent_scale * d.latent_scale;
    return (d.ref_area + s2 - 1) / s2 * 11 / 10 + 64;
}

struct Sizes {
    int64_t T = 0, S = 0, EP = 0, PD = 0, R = 0, P = 0, PX = 0, NT = 0, NB = 0, AX = 0;
    bool    joint = false;
};

Sizes sizes_of(const Program& prog, int64_t block_size) {
    const RadGeneratorDecl& d = prog.generator;
    Sizes z;
    z.T  = prog.ctx.max_tok;
    z.S  = prog.ctx.max_seqs;
    z.EP = prog.encoder.modalities ? prog.ctx.max_enc_patches : 0;
    z.PD = prog.encoder.modalities ? prog.encoder.patch_dim : 0;
    z.R  = d.max_refs;
    z.joint = d.layout == RAD_GEN_LAYOUT_JOINT_STEP;
    /* The longest conditioning: in the joint layout, the declared bound (its rows ride in the
     * step, which the step size bounds anyway). */
    z.P  = z.joint ? std::min(d.cond_rows, z.T) : z.T + z.R * ref_tokens_max(d);
    const int64_t s2 = d.latent_scale * d.latent_scale;
    z.PX = z.R * (ref_tokens_max(d) * s2) + d.max_tokens * s2;
    z.NT = d.max_tokens;
    z.NB = block_size > 0 ? (z.P + block_size - 1) / block_size : 0;
    z.AX = d.aux_tokens ? d.aux_max_tokens : 0;
    return z;
}

/* Region offsets and sizes, in order, and the total. Shared by the allocation and the VRAM budget
 * so the two always agree. */
int64_t carve(const Sizes& z, int64_t pixel_ch, int64_t (*bytes)[2], int n) {
    const int64_t I = 4, H = 2;
    const int64_t rope_rows = std::max(z.P, z.T);
    const int64_t J = z.joint ? 1 : 0;   /* the joint layout's regions, empty otherwise */
    const int64_t want[][2] = {
        { 0, z.T * I },                 /* R_TOK */
        { 0, z.T * I },                 /* R_POS */
        { 0, 3 * z.T * I },             /* R_ROPE */
        { 0, (z.S + 1) * I },           /* R_CU */
        { 0, z.T * I },                 /* R_MMROW */
        { 0, z.EP * z.PD * H },         /* R_ENCPIX */
        { 0, 4 * z.EP * I },            /* R_ENCCRD */
        { 0, (z.R + 2) * I },           /* R_ENCCU */
        { 0, rope_rows * I },           /* R_SRC: a COND's rows, or a joint DENOISE's */
        { 0, 3 * rope_rows * I },       /* R_GROPE */
        { 0, (12 * z.R + 24) * I },     /* R_SEG: offsets; per segment a {0, length} pair and a start; per picture a {0, tokens} pair */
        { 0, z.P * I },                 /* R_SLOT */
        { 0, z.PX * pixel_ch * H },     /* R_PX */
        { 0, z.NT * 4 },                /* R_MASK */
        { 0, z.S * 4 },                 /* R_SIG */
        { 0, z.S * 4 },                 /* R_SIGN */
        { 0, z.S * 4 },                 /* R_CFG */
        { 0, z.S * I },                 /* R_PAIR */
        { 0, z.S * I },                 /* R_LANE */
        { 0, z.S * I },                 /* R_FLAGS */
        { 0, 2 * z.S * 4 },             /* R_SEED */
        { 0, z.S * z.NB * I },          /* R_BT */
        { 0, z.S * I },                 /* R_USED */
        { 0, J * 4 * rope_rows * 4 },   /* R_ROPEF */
        { 0, J * (z.S + 1) * I },       /* R_ICU */
        { 0, J * z.T * I },             /* R_IROWS */
        { 0, J * (3 * z.S + 1) * I },   /* R_MCU */
        { 0, J * 2 * I },               /* R_VCU */
        { 0, J * z.T * I },             /* R_MROWS */
        { 0, J * z.T * I },             /* R_MSRC */
        { 0, J * 3 * (z.S + 1) * I },   /* R_MCCU */
        { 0, J * 4 * z.T * 4 },         /* R_MROPE */
        { 0, (z.AX ? z.AX + 2 : 0) * I },  /* R_AUX: a COND's {0, n} and second-tokenizer ids */
        { 0, z.S * 4 },                 /* R_SIGP */
        { 0, z.S * 4 },                 /* R_TSTEP */
        { 0, z.S * I },                 /* R_SAMP */
        { 0, z.S * I },                 /* R_STEPI */
        { 0, 2 * z.S * 4 },             /* R_SIZE */
    };
    int64_t off = 0;
    for (int r = 0; r < n; ++r) {
        bytes[r][0] = off;
        bytes[r][1] = want[r][1];
        off = align_up(off + want[r][1], kAlign);
    }
    return off;
}

}  /* namespace */

int64_t Driver::device_bytes(const Program& prog) {
    if (!prog.is_generator()) return 0;
    const bool joint = prog.generator.layout == RAD_GEN_LAYOUT_JOINT_STEP;
    if (!joint && prog.kv_groups.size() < 2) return 0;
    const Sizes z = sizes_of(prog, joint ? 0 : prog.kv_groups[1].block_size);
    int64_t b[R_N][2];
    return carve(z, prog.generator.pixel_channels, b, R_N);
}

int Driver::init(const Program& prog, IKVManager* kv, const std::vector<RankIO>& ranks) {
    fini();
    if (!prog.is_generator()) return RAD_E_STATE;
    decl_   = prog.generator;
    fixed_  = prog.gen_str.sigmas;
    kv_     = kv;
    ranks_  = ranks;
    table_  = DiscreteTable{};
    if (decl_.cond_opts & ~(uint32_t)(RAD_GEN_COND_REBALANCE | RAD_GEN_COND_REBALANCE_PRESET)) {
        RAD_ERR("image generator: '%s' declares conditioning options 0x%x, and this engine knows "
                "only RAD_GEN_COND_REBALANCE (0x%x) and its preset (0x%x)",
                decl_.name ? decl_.name : "", (unsigned)decl_.cond_opts, (unsigned)RAD_GEN_COND_REBALANCE,
                (unsigned)RAD_GEN_COND_REBALANCE_PRESET);
        return RAD_E_UNSUPPORTED;
    }
    if (discrete()) {
        std::string why;
        if (discrete_table(decl_, &table_, &why) < 0) {
            RAD_ERR("image generator: %s", why.c_str());
            return RAD_E_INVAL;
        }
    }
    /* Exactly one paged full-attention KV group, holding the conditioning; none in the joint
     * layout, where the conditioning rides in the steps. Group handles start at 1 (group 0 is the
     * null handle); the KV manager and a batch both index by handle. */
    if (joint()) {
        if (prog.kv_groups.size() > 1) {
            RAD_ERR("image generator: the joint layout keeps its conditioning in the plugin's "
                    "state, and the program declares %zu KV group(s)", prog.kv_groups.size() - 1);
            return RAD_E_UNSUPPORTED;
        }
        kv_group_ = 0;
        block_size_ = 0;
    } else if (prog.kv_groups.size() != 2 || prog.kv_groups[1].decl.kind != RAD_KV_FULL ||
               prog.kv_groups[1].block_size <= 0) {
        RAD_ERR("image generator: the program declares %zu KV group(s); the driver stages exactly "
                "one full-attention group, the conditioning's", prog.kv_groups.size() - 1);
        return RAD_E_UNSUPPORTED;
    } else {
        kv_group_   = 1;
        block_size_ = prog.kv_groups[1].block_size;
    }
    const Sizes z = sizes_of(prog, block_size_);
    max_tok_   = z.T;
    max_seqs_  = z.S;
    max_enc_   = z.EP;
    patch_dim_ = z.PD;
    max_rows_  = z.P;
    max_px_    = z.PX;
    max_blocks_ = z.NB;
    int64_t b[R_N][2];
    bytes_ = carve(z, decl_.pixel_channels, b, R_N);
    for (int r = 0; r < R_N; ++r) { reg_[r].off = b[r][0]; reg_[r].bytes = b[r][1]; reg_[r].used = 0; }

    host_ = static_cast<char*>(rad_dev_alloc(bytes_, RAD_MEM_HOST_PINNED));
    img_bytes_ = decl_.max_tokens * decl_.latent_scale * decl_.latent_scale * decl_.pixel_channels;
    img_host_ = rad_dev_alloc(img_bytes_, RAD_MEM_HOST_PINNED);
    if (!host_ || !img_host_) {
        RAD_ERR("image generator: no pinned host memory for the pass staging (%lld + %lld bytes)",
                (long long)bytes_, (long long)img_bytes_);
        return RAD_E_NOMEM;
    }
    std::memset(host_, 0, (size_t)bytes_);
    dev_.assign(ranks_.size(), nullptr);
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r].device));
        dev_[r] = rad_dev_alloc(bytes_, RAD_MEM_DEVICE);
        if (!dev_[r]) {
            RAD_ERR("image generator: rank %zu has no room for the pass staging (%lld bytes)", r,
                    (long long)bytes_);
            return RAD_E_NOMEM;
        }
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0].device));
    out_.assign(ranks_.size(), RadBatch{});
    gout_.assign(ranks_.size(), RadGenBatch{});
    kout_.assign(ranks_.size(), RadKVGroupBatch{});
    lanes_.assign((size_t)decl_.max_lanes, Lane{});
    free_lanes_.clear();
    for (int64_t i = decl_.max_lanes - 1; i >= 0; --i) free_lanes_.push_back((int32_t)i);
    return RAD_OK;
}

void Driver::fini() {
    for (size_t r = 0; r < dev_.size(); ++r)
        if (dev_[r]) {
            if (r < ranks_.size()) rad_dev_set(ranks_[r].device);
            rad_dev_free(dev_[r], RAD_MEM_DEVICE);
        }
    dev_.clear();
    if (!ranks_.empty()) rad_dev_set(ranks_[0].device);
    if (host_) rad_dev_free(host_, RAD_MEM_HOST_PINNED);
    if (img_host_) rad_dev_free(img_host_, RAD_MEM_HOST_PINNED);
    host_ = nullptr;
    img_host_ = nullptr;
}

/* ================================================================== requests */
int Driver::submit(std::unique_ptr<Request> r, std::string* why) {
    if (!r || !r->ticket) return RAD_E_INVAL;
    const RadGeneratorDecl& d = decl_;
    auto no = [&](const std::string& w) {
        if (why) *why = w;
        return RAD_E_INVAL;
    };
    if (r->width <= 0 || r->height <= 0 || r->width % d.size_multiple || r->height % d.size_multiple)
        return no("the size must be whole multiples of " + std::to_string(d.size_multiple) +
                  " pixels");
    const int64_t tokens = (int64_t)(r->width / d.latent_scale) * (r->height / d.latent_scale);
    const int nb = r->has_neg ? 2 : 1;
    if (tokens > d.max_tokens)
        return no("the image is " + std::to_string(r->width) + "x" + std::to_string(r->height) +
                  "; the largest this model makes is " + std::to_string(d.max_tokens) +
                  " latent tokens (" + std::to_string(d.max_tokens * d.latent_scale * d.latent_scale) +
                  " pixels)");
    if (tokens * nb > max_tok_)
        return no("an image of this size" + std::string(nb > 1 ? " with guidance" : "") +
                  " needs " + std::to_string(tokens * nb) + " rows a step and a step carries " +
                  std::to_string(max_tok_) + " (--max-num-batched-tokens)");
    if (r->n < 1 || r->n > d.max_lanes)
        return no("n must be 1.." + std::to_string(d.max_lanes));
    if ((int64_t)r->refs.size() > d.max_refs)
        return no("at most " + std::to_string(d.max_refs) + " pictures a request");
    for (const Branch* b : { &r->pos, &r->neg }) {
        if (b == &r->neg && !r->has_neg) continue;
        if (d.aux_tokens && (b->aux.empty() || (int64_t)b->aux.size() > d.aux_max_tokens))
            return no("the prompt is " + std::to_string(b->aux.size()) + " tokens to the "
                      "model's second tokenizer, which reads 1.." +
                      std::to_string(d.aux_max_tokens));
        if ((int64_t)b->tokens.size() > max_tok_)
            return no("the prompt is " + std::to_string(b->tokens.size()) + " tokens with its "
                      "pictures; a conditioning pass carries " + std::to_string(max_tok_));
        int64_t np = 0;
        if (b->media)
            for (const mm::Placed& p : b->media->items) np += p.item ? p.item->n_patch : 0;
        if (np > max_enc_)
            return no("the pictures are " + std::to_string(np) + " patches together and a "
                      "conditioning pass encodes " + std::to_string(max_enc_) +
                      " (--mm-max-patches)");
    }
    int64_t px = 0;
    for (const Pixels& p : r->refs) px += (int64_t)p.w * p.h;
    if (r->inpaint) px += (int64_t)r->inp_src.w * r->inp_src.h;
    if (px > max_px_) return no("the pictures are larger than the staging holds");
    if (r->inpaint && ((int64_t)r->inp_mask.size() != tokens || r->inp_src.w != r->width ||
                       r->inp_src.h != r->height))
        return no("the mask's picture is not at the size being made");
    if (r->previews < 0 || r->previews > 3) return no("partial_images must be 0..3");
    if (discrete() && (r->sampler < 0 || r->sampler >= RAD_GEN_SAMPLER_COUNT ||
                       !(decl_.samplers & (1u << r->sampler)) || r->scheduler < 0 ||
                       r->scheduler >= RAD_GEN_SCHED_COUNT))
        return no("the sampler or scheduler is not one this model serves");
    if ((uint32_t)r->cond_opts & ~decl_.cond_opts)
        return no("a conditioning option (rebalance) this model does not serve");
    r->ticket->t_submit_ns = (int64_t)rad_mono_ns();
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) {
        if (why) *why = closed_why_;
        return closed_status_;
    }
    auto j = std::make_unique<Job>();
    j->id = next_id_++;
    j->n_branch = nb;
    j->tokens = tokens;
    j->h_lat = (int32_t)(r->height / d.latent_scale);
    j->w_lat = (int32_t)(r->width / d.latent_scale);
    j->req = std::move(r);
    jobs_.push_back(std::move(j));
    return RAD_OK;
}

void Driver::shutdown(const char* why, int status) {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
    closed_why_ = why ? why : "the engine stopped";
    closed_status_ = status;
    for (auto& j : jobs_) j->req->ticket->finish(status, closed_why_);
}

int64_t Driver::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (!j->admitted) ++n;
    return n;
}

int64_t Driver::running() const {
    std::lock_guard<std::mutex> lk(mu_);
    int64_t n = 0;
    for (const auto& j : jobs_) if (j->admitted) ++n;
    return n;
}

bool Driver::busy() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !jobs_.empty();
}

Driver::Job* Driver::job_locked(uint64_t id) {
    for (auto& j : jobs_) if (j->id == id) return j.get();
    return nullptr;
}

void Driver::release_kv_locked(Job& j) {
    for (int b = 0; b < 2; ++b)
        if (j.cond[b].have_kv) {
            kv_->free_sequence(j.cond[b].seq);
            j.cond[b].have_kv = false;
        }
}

void Driver::release_locked(Job& j) {
    release_kv_locked(j);
    for (int32_t l : j.lanes) {
        lanes_[(size_t)l] = Lane{};
        free_lanes_.push_back(l);
    }
    j.lanes.clear();
}

void Driver::fail_locked(Job& j, int status, const std::string& why) {
    j.failed = true;
    j.req->ticket->finish(status, why);
}

/* In arrival order; a request that does not fit stops the ones behind it. */
int Driver::admit_locked() {
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (j.admitted || j.failed) continue;
        Request& r = *j.req;
        if ((int64_t)free_lanes_.size() < r.n) break;
        /* Build the layouts once; their length sizes the KV. */
        if (j.cond[0].lay.rows == 0) {
            j.ref_hw.clear();
            for (const Pixels& p : r.refs)
                j.ref_hw.push_back({ (int32_t)(p.h / decl_.latent_scale),
                                     (int32_t)(p.w / decl_.latent_scale) });
            int64_t step_rows = 0;
            for (int b = 0; b < j.n_branch; ++b) {
                std::string why;
                const int s = joint()
                    ? build_joint_layout(b ? r.neg : r.pos, decl_.pad_multiple,
                                         decl_.rope_hw_scale, decl_.rope_text_l != 0, j.ref_hw,
                                         &j.cond[b].lay, &why)
                    : build_layout(b ? r.neg : r.pos, decl_.slot_rows, j.ref_hw, &j.cond[b].lay,
                                   &why);
                if (s < 0) { fail_locked(j, s, why); break; }
                if (j.cond[b].lay.rows > max_rows_) {
                    fail_locked(j, RAD_E_INVAL, "the conditioning is " +
                                std::to_string(j.cond[b].lay.rows) + " rows (the prompt" +
                                std::string(r.refs.empty() ? "" : " and its pictures") +
                                "); this model conditions on at most " +
                                std::to_string(max_rows_));
                    break;
                }
                step_rows += entry_rows(j, b);
            }
            if (j.failed) continue;
            if (step_rows > max_tok_) {
                fail_locked(j, RAD_E_INVAL, "an image with its conditioning needs " +
                            std::to_string(step_rows) + " rows a step and a step carries " +
                            std::to_string(max_tok_) + " (--max-num-batched-tokens)");
                continue;
            }
        }
        bool ok = true;
        for (int b = 0; b < j.n_branch && ok && !joint(); ++b) {
            Cond& c = j.cond[b];
            if (c.have_kv) continue;
            if (!kv_->can_ever_fit(c.lay.rows)) {
                fail_locked(j, RAD_E_INVAL, "the conditioning's " + std::to_string(c.lay.rows) +
                            " rows do not fit the KV cache even when it is empty");
                ok = false;
                break;
            }
            c.seq = ((uint64_t)1 << 62) | next_seq_++;
            if (kv_->add_sequence(c.seq) < 0) { ok = false; break; }
            const int s = kv_->ensure(c.seq, c.lay.rows);
            if (s < 0) {
                kv_->free_sequence(c.seq);
                ok = false;
                break;
            }
            c.have_kv = true;
        }
        if (j.failed) continue;
        if (!ok) {
            /* KV is short: release what this request took and wait, unless nothing is running
             * that would free any. */
            release_kv_locked(j);
            bool running = false;
            for (const auto& o : jobs_) running = running || o->admitted;
            if (running) break;
            fail_locked(j, RAD_E_FULL, "the KV cache cannot hold this prompt, and nothing is running "
                                       "that would free it; try a shorter prompt");
            continue;
        }
        if (discrete()) {
            j.sigmas = schedule_discrete(decl_, table_, r.scheduler, r.steps);
            j.tsteps.clear();
            for (float sg : j.sigmas) j.tsteps.push_back(sg > 0 ? (float)table_.timestep(sg) : 0.0f);
        } else {
            j.sigmas = schedule(decl_, fixed_, r.steps, j.tokens);
        }
        const int32_t n_steps = (int32_t)j.sigmas.size() - 1;
        for (int b = 0; b < j.n_branch; ++b) {
            Cond& c = j.cond[b];
            if (joint()) {
                c.rope_img_f.assign((size_t)(4 * j.tokens), 0.0f);
                grid_rope_f(decl_.rope_hw_scale, 0.0f, j.h_lat, j.w_lat, c.rope_img_f.data(),
                            j.tokens);
            } else {
                c.rope_img.assign((size_t)(3 * j.tokens), 0);
                image_rope(c.lay.next_pos, j.h_lat, j.w_lat, c.rope_img.data(), j.tokens);
            }
        }
        for (int32_t i = 0; i < r.n; ++i) {
            const int32_t l = free_lanes_.back();
            free_lanes_.pop_back();
            Lane& ln = lanes_[(size_t)l];
            ln = Lane{};
            ln.busy = true;
            ln.job_id = j.id;
            ln.index = i;
            ln.seed = r.seed + (uint64_t)i;
            for (int32_t k = 1; k <= r.previews; ++k) {
                const int32_t at = (int32_t)((int64_t)n_steps * k / (r.previews + 1));
                if (at >= 1 && at < n_steps &&
                    (ln.preview_at.empty() || ln.preview_at.back() != at))
                    ln.preview_at.push_back(at);
            }
            j.lanes.push_back(l);
        }
        r.ticket->steps = n_steps;
        int64_t in_tok = (int64_t)r.pos.tokens.size();
        if (r.has_neg) in_tok += (int64_t)r.neg.tokens.size();
        r.ticket->in_tokens = in_tok;
        r.ticket->out_tokens = j.tokens * r.n;
        j.admitted = true;
    }
    return RAD_OK;
}

/* ================================================================== staging */
int Driver::upload() {
    for (size_t r = 0; r < ranks_.size(); ++r)
        for (int g = 0; g < R_N; ++g) {
            if (reg_[g].used <= 0) continue;
            RAD_TRY(rad_memcpy_async((char*)dev_[r] + reg_[g].off, host_ + reg_[g].off,
                                     reg_[g].used, ranks_[r].stream));
        }
    return RAD_OK;
}

void Driver::finish_batches(int64_t n_tok, int64_t n_seq, int32_t max_q, int step) {
    for (size_t r = 0; r < ranks_.size(); ++r) {
        RadBatch& B = out_[r];
        RadGenBatch& G = gout_[r];
        RadKVGroupBatch& K = kout_[r];
        const int ri = (int)r;
        B.step = step;
        B.phase = RAD_PHASE_PREFILL;
        B.n_tok = n_tok;
        B.n_seq = n_seq;
        B.max_q_len = max_q;
        B.token_ids = dptr<int32_t>(ri, R_TOK);
        B.positions = dptr<int32_t>(ri, R_POS);
        B.cu_seqlens = dptr<int32_t>(ri, R_CU);
        B.q_lens = nullptr;
        B.ctx_lens = nullptr;
        B.n_seq_decode = 0;
        B.n_tok_decode = 0;
        B.max_q_len_prefill = max_q;
        /* No rotary table and no encoder input; build_cond sets its own. */
        B.rope_pos = nullptr;
        B.rope_mixed = 0;
        B.n_mm_rows = 0;
        B.mm_rows = nullptr;
        B.enc_n_patch = 0;
        B.enc_pixels = nullptr;
        B.enc_coord = nullptr;
        B.enc_n_seg = 0;
        B.enc_cu = nullptr;
        K.group = (rad_kvgroup)kv_group_;
        K.slot_mapping = reg_[R_SLOT].used > 0 ? dptr<int32_t>(ri, R_SLOT) : nullptr;
        K.block_table = dptr<int32_t>(ri, R_BT);
        K.seqused = dptr<int32_t>(ri, R_USED);
        K.block_size = block_size_;
        K.max_blocks = max_blocks_;
        B.n_kv_groups = joint() ? 0 : 1;
        B.kv = joint() ? nullptr : &K;
        B.gen_b = &G;
        G.src = dptr<int32_t>(ri, R_SRC);
        G.rope = dptr<int32_t>(ri, R_GROPE);
        G.seg = dptr<int32_t>(ri, R_SEG);
        G.seg_h = hptr<int32_t>(R_SEG);
        G.slot_mapping = dptr<int32_t>(ri, R_SLOT);
        G.px = dptr<uint16_t>(ri, R_PX);
        G.inp_mask = dptr<float>(ri, R_MASK);
        G.sigma = dptr<float>(ri, R_SIG);
        G.sigma_next = dptr<float>(ri, R_SIGN);
        G.cfg = dptr<float>(ri, R_CFG);
        G.pair = dptr<int32_t>(ri, R_PAIR);
        G.lane = dptr<int32_t>(ri, R_LANE);
        G.lane_dh = hptr<int32_t>(R_LANE);
        G.flags = dptr<int32_t>(ri, R_FLAGS);
        G.seed = dptr<uint32_t>(ri, R_SEED);
        /* The joint layout's fields, which only the passes that use them set. */
        G.branch = 0;
        G.n_pad = 0;
        G.valid_cu = nullptr;
        G.n_img = 0;
        G.img_cu = nullptr;
        G.img_rows = nullptr;
        G.mode_cu = nullptr;
        G.rope_f = nullptr;
        G.mode_rows = nullptr;
        G.mode_src = nullptr;
        G.mode_ccu = nullptr;
        G.n_mode_h = nullptr;
        G.mode_rope_f = nullptr;
        G.n_aux = 0;
        G.aux_ids = nullptr;
        G.aux_cu = nullptr;
        G.sigma_prev = nullptr;
        G.timestep = nullptr;
        G.sampler = nullptr;
        G.step = nullptr;
        G.size_px = nullptr;
        G.entry_h = nullptr;
        G.tokens_h = nullptr;
        G.cond_opts = 0;
    }
}

/* ================================================================== the passes */
const RadBatch* Driver::next(int step) {
    std::lock_guard<std::mutex> lk(mu_);
    pass_ = Pass{};
    dec_bytes_ = 0;
    rb_.clear();
    if (closed_) return nullptr;
    /* Retire finished, failed and cancelled jobs, returning their KV and lanes before admitting.
     * Safe because next() runs between passes, with nothing in flight. */
    for (auto it = jobs_.begin(); it != jobs_.end();) {
        Job& j = **it;
        const bool cancelled = j.req->ticket->cancel.load(std::memory_order_acquire);
        const bool gone = j.failed || cancelled || (j.admitted && j.images_done >= j.req->n);
        if (!gone) { ++it; continue; }
        if (cancelled && !j.failed && j.images_done < j.req->n)
            j.req->ticket->finish(RAD_E_CANCELLED, "the request was cancelled");
        release_locked(j);
        it = jobs_.erase(it);
    }
    admit_locked();

    /* 1. A decode an image is owed. */
    for (int32_t l = 0; l < (int32_t)lanes_.size(); ++l) {
        Lane& ln = lanes_[(size_t)l];
        if (!ln.busy || ln.finished) continue;
        if (ln.preview_due) return build_decode(l, true, step);
        if (ln.final_due) return build_decode(l, false, step);
    }
    /* 2. A conditioning an admitted request is owed. */
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed) continue;
        for (int b = 0; b < j.n_branch; ++b)
            if (!j.cond[b].encoded) return build_cond(j, b, step);
    }
    /* 3. A denoise step over what is in flight, oldest request first, with the oldest one's
     * adapters: a pass runs one runtime LoRA slot in every row. */
    std::vector<int32_t> take;
    int64_t rows = 0, entries = 0;
    int32_t lora = -2;
    for (auto& jp : jobs_) {
        Job& j = *jp;
        if (!j.admitted || j.failed) continue;
        bool ready = true;
        for (int b = 0; b < j.n_branch; ++b) ready = ready && j.cond[b].encoded;
        if (!ready) continue;
        if (lora != -2 && j.req->lora_slot != lora) continue;
        const int32_t n_steps = (int32_t)j.sigmas.size() - 1;
        for (int32_t l : j.lanes) {
            const Lane& ln = lanes_[(size_t)l];
            if (ln.finished || ln.final_due || ln.preview_due || ln.step >= n_steps) continue;
            int64_t need = 0;
            for (int b = 0; b < j.n_branch; ++b) need += entry_rows(j, b);
            if (rows + need > max_tok_ || entries + j.n_branch > max_seqs_) break;
            rows += need;
            entries += j.n_branch;
            take.push_back(l);
            lora = j.req->lora_slot;
        }
    }
    if (take.empty()) return nullptr;
    const RadBatch* b = build_denoise(take, step);
    pass_.lora_slot = lora;
    return b;
}

const RadBatch* Driver::batch_for(int rank) const {
    if (rank < 0 || (size_t)rank >= out_.size() || pass_.kind == 0) return nullptr;
    return &out_[(size_t)rank];
}

const RadBatch* Driver::build_cond(Job& j, int branch, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const Request& r = *j.req;
    const Branch& br = branch ? r.neg : r.pos;
    const Cond& c = j.cond[branch];
    /* The joint layout's padding tokens follow the prompt's; each sits at position 1 and sees
     * only the prompt (`valid_cu`), as a masked padding token does in the reference. */
    const int64_t T_valid = (int64_t)br.tokens.size();
    const int64_t T = T_valid + (joint() ? c.lay.n_pad : 0), P = c.lay.rows;

    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* pos = hptr<int32_t>(R_POS);
    int32_t* rope = hptr<int32_t>(R_ROPE);
    std::memcpy(tok, br.tokens.data(), (size_t)T_valid * 4);
    for (int64_t i = T_valid; i < T; ++i) tok[i] = decl_.pad_token;
    for (int64_t i = 0; i < T; ++i) pos[i] = i < T_valid ? (int32_t)i : 1;
    for (int64_t i = 0; i < T; ++i) {
        int32_t p3[3] = { pos[i], pos[i], pos[i] };
        if (br.media) br.media->rope.at(i, p3);
        rope[i] = p3[0];
        rope[T + i] = p3[1];
        rope[2 * T + i] = p3[2];
    }
    int32_t* cu = hptr<int32_t>(R_CU);
    cu[0] = 0;
    cu[1] = (int32_t)T;
    reg_[R_TOK].used = reg_[R_POS].used = T * 4;
    reg_[R_ROPE].used = 3 * T * 4;
    reg_[R_CU].used = 2 * 4;

    /* Pictures for the conditioning network's encoder: every item's patches in prompt order, its
     * segments, and the prompt rows the encoder's output replaces. */
    StagedMedia sm;
    if (br.media) {
        sm = stage_media(*br.media, patch_dim_, hptr<uint16_t>(R_ENCPIX), hptr<int32_t>(R_ENCCRD),
                         hptr<int32_t>(R_ENCCU), hptr<int32_t>(R_MMROW));
        reg_[R_ENCPIX].used = sm.n_patch * patch_dim_ * 2;
        reg_[R_ENCCRD].used = 4 * sm.n_patch * 4;
        reg_[R_ENCCU].used = (sm.n_seg + 1) * 4;
        reg_[R_MMROW].used = sm.n_rows * 4;
    }

    /* The generator's sequence, run by the plugin a segment at a time: per segment, its rotary
     * positions as a dense [3, length] block, a {0, length} cu pair and its first row. */
    std::memcpy(hptr<int32_t>(R_SRC), c.lay.src.data(), (size_t)P * 4);
    const int64_t nsg = (int64_t)c.lay.bidi.size();
    int32_t* grope = hptr<int32_t>(R_GROPE);
    int32_t* segp = hptr<int32_t>(R_SEG);
    std::memcpy(segp, c.lay.seg.data(), (size_t)(nsg + 1) * 4);
    for (int64_t s = 0; s < nsg; ++s) {
        const int32_t a = c.lay.seg[(size_t)s], len = c.lay.seg[(size_t)s + 1] - a;
        /* The joint layout's positions are f32, staged whole below. */
        for (int cc = 0; cc < 3 && !joint(); ++cc)
            std::memcpy(grope + 3 * (int64_t)a + cc * (int64_t)len,
                        &c.lay.rope[(size_t)(cc * P + a)], (size_t)len * 4);
        segp[nsg + 1 + 2 * s] = 0;
        segp[nsg + 1 + 2 * s + 1] = len;
        segp[nsg + 1 + 2 * nsg + s] = a;
    }
    reg_[R_SRC].used = P * 4;
    reg_[R_GROPE].used = joint() ? 0 : 3 * P * 4;
    reg_[R_SEG].used = (4 * nsg + 1) * 4;
    static const std::vector<int32_t> no_bt;
    const std::vector<int32_t>& bt = joint() ? no_bt : kv_->block_table(c.seq, kv_group_);
    if (joint()) {
        std::memcpy(hptr<float>(R_ROPEF), c.lay.rope_f.data(), (size_t)(4 * P) * 4);
        reg_[R_ROPEF].used = 4 * P * 4;
        int32_t* vcu = hptr<int32_t>(R_VCU);
        vcu[0] = 0;
        vcu[1] = (int32_t)T_valid;
        reg_[R_VCU].used = 2 * 4;
    } else {
        int32_t* slot = hptr<int32_t>(R_SLOT);
        for (int64_t p = 0; p < P; ++p) {
            const int64_t bi = p / block_size_;
            slot[p] = bi < (int64_t)bt.size()
                          ? (int32_t)(bt[(size_t)bi] * block_size_ + p % block_size_) : -1;
        }
        reg_[R_SLOT].used = P * 4;
        int32_t* btab = hptr<int32_t>(R_BT);
        for (size_t i = 0; i < bt.size(); ++i) btab[i] = bt[i];
        reg_[R_BT].used = (int64_t)bt.size() * 4;
        hptr<int32_t>(R_USED)[0] = (int32_t)P;
        reg_[R_USED].used = 4;
    }

    /* Pictures again for the generator's own encoder, plus the inpainting source when masked.
     * Only the positive conditioning's pass writes the lanes. */
    const int64_t ch = decl_.pixel_channels;
    uint16_t* px = hptr<uint16_t>(R_PX);
    std::vector<int32_t> pxh;
    int64_t at = 0;
    for (const Pixels& p : r.refs) {
        std::memcpy(px + at * ch, p.bf16.data(), (size_t)((int64_t)p.w * p.h * ch) * 2);
        pxh.insert(pxh.end(), { (int32_t)at, p.h, p.w });
        at += (int64_t)p.w * p.h;
    }
    const bool inp = r.inpaint && branch == 0;
    if (inp) {
        const Pixels& p = r.inp_src;
        std::memcpy(px + at * ch, p.bf16.data(), (size_t)((int64_t)p.w * p.h * ch) * 2);
        pxh.insert(pxh.end(), { (int32_t)at, p.h, p.w });
        at += (int64_t)p.w * p.h;
        std::memcpy(hptr<float>(R_MASK), r.inp_mask.data(), r.inp_mask.size() * 4);
        reg_[R_MASK].used = (int64_t)r.inp_mask.size() * 4;
    }
    reg_[R_PX].used = at * ch * 2;
    /* Each picture's latent tokens as a cu pair, after the segments' arrays. */
    const int64_t npx = (int64_t)pxh.size() / 3;
    int32_t* pxcu = hptr<int32_t>(R_SEG) + 4 * nsg + 1;
    for (int64_t b = 0; b < npx; ++b) {
        pxcu[2 * b] = 0;
        pxcu[2 * b + 1] = (pxh[(size_t)(3 * b + 1)] / (int32_t)decl_.latent_scale) *
                          (pxh[(size_t)(3 * b + 2)] / (int32_t)decl_.latent_scale);
    }
    reg_[R_SEG].used = (4 * nsg + 1 + 2 * npx) * 4;
    /* The conditioning is modulated at t = 0. */
    hptr<float>(R_SIG)[0] = 0.0f;
    reg_[R_SIG].used = 4;
    /* The prompt through the second tokenizer, for a model that reads it. */
    const int64_t n_aux = decl_.aux_tokens ? (int64_t)br.aux.size() : 0;
    if (n_aux) {
        int32_t* ax = hptr<int32_t>(R_AUX);
        ax[0] = 0;
        ax[1] = (int32_t)n_aux;
        std::memcpy(ax + 2, br.aux.data(), (size_t)n_aux * 4);
        reg_[R_AUX].used = (n_aux + 2) * 4;
    }
    /* Host-side arrays the plugin reads (segment kinds, pixel blocks, lanes), owned by the driver
     * for the life of the pass. */
    host_bidi_ = c.lay.bidi;
    host_px_ = pxh;
    /* The joint layout's conditioning goes to every lane of the request. */
    host_lanes_ = inp || joint() ? j.lanes : std::vector<int32_t>{};
    /* The lanes on the device too: an op copies the inpainting latent and mask into each. */
    if (!host_lanes_.empty()) {
        std::memcpy(hptr<int32_t>(R_LANE), host_lanes_.data(), host_lanes_.size() * 4);
        reg_[R_LANE].used = (int64_t)host_lanes_.size() * 4;
    }

    if (upload() < 0) return nullptr;
    finish_batches(T, 1, (int32_t)T, step);
    for (size_t rr = 0; rr < ranks_.size(); ++rr) {
        RadBatch& B = out_[rr];
        RadGenBatch& G = gout_[rr];
        const int ri = (int)rr;
        B.max_ctx_len = (int32_t)P;
        B.rope_pos = dptr<int32_t>(ri, R_ROPE);
        B.rope_mixed = br.media ? 1 : 0;
        B.n_mm_rows = sm.n_rows;
        B.mm_rows = sm.n_rows ? dptr<int32_t>(ri, R_MMROW) : nullptr;
        B.mm_embd = nullptr;
        B.enc = 0;
        B.enc_n_patch = sm.n_patch;
        B.enc_pixels = sm.n_patch ? dptr<uint16_t>(ri, R_ENCPIX) : nullptr;
        B.enc_coord = sm.n_patch ? dptr<int32_t>(ri, R_ENCCRD) : nullptr;
        B.enc_n_seg = sm.n_seg;
        B.enc_cu = sm.n_seg ? dptr<int32_t>(ri, R_ENCCU) : nullptr;
        B.enc_max_seg = sm.max_seg;
        B.gen = RAD_GEN_PASS_COND;
        kout_[rr].block_table_pitch = (int64_t)bt.size();
        G = RadGenBatch{};
        G.rows = P;
        G.src = dptr<int32_t>(ri, R_SRC);
        G.rope = dptr<int32_t>(ri, R_GROPE);
        G.n_seg = nsg;
        G.seg = dptr<int32_t>(ri, R_SEG);
        G.seg_h = hptr<int32_t>(R_SEG);
        G.seg_bidi_h = host_bidi_.data();
        G.seg_cu = dptr<int32_t>(ri, R_SEG) + nsg + 1;
        G.seg_start = dptr<int32_t>(ri, R_SEG) + nsg + 1 + 2 * nsg;
        G.slot_mapping = dptr<int32_t>(ri, R_SLOT);
        G.n_px = (int64_t)pxh.size() / 3;
        G.px = at ? dptr<uint16_t>(ri, R_PX) : nullptr;
        G.px_h = host_px_.data();
        G.px_cu = dptr<int32_t>(ri, R_SEG) + 4 * nsg + 1;
        G.sigma = dptr<float>(ri, R_SIG);
        G.inpaint = inp ? 1 : 0;
        G.inp_mask = inp ? dptr<float>(ri, R_MASK) : nullptr;
        G.n_lane = (int64_t)host_lanes_.size();
        G.lane_h = host_lanes_.data();
        G.lane = G.n_lane ? dptr<int32_t>(ri, R_LANE) : nullptr;
        G.n_aux = n_aux;
        G.aux_cu = n_aux ? dptr<int32_t>(ri, R_AUX) : nullptr;
        G.aux_ids = n_aux ? dptr<int32_t>(ri, R_AUX) + 2 : nullptr;
        G.tokens_h = hptr<int32_t>(R_TOK);
        G.cond_opts = j.req->cond_opts;
        if (joint()) {
            B.n_kv_groups = 0;
            B.kv = nullptr;
            G.slot_mapping = nullptr;
            G.branch = branch;
            G.n_pad = c.lay.n_pad;
            G.valid_cu = dptr<int32_t>(ri, R_VCU);
            G.rope = nullptr;
            G.rope_f = dptr<float>(ri, R_ROPEF);
        }
    }
    pass_.kind = RAD_GEN_PASS_COND;
    pass_.job = j.id;
    pass_.lora_slot = j.req->lora_slot;
    pass_.branch = branch;
    return &out_[0];
}

const RadBatch* Driver::build_denoise(const std::vector<int32_t>& take, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    /* Two passes: collect the entries and their rows, then fill the per-entry arrays. */
    struct E { const Job* j; int32_t lane; int branch; };
    std::vector<E> es;
    for (int32_t l : take) {
        const Job* j = job_locked(lanes_[(size_t)l].job_id);
        if (!j) continue;
        for (int b = 0; b < j->n_branch; ++b) es.push_back({ j, l, b });
    }
    const int64_t S = (int64_t)es.size();
    int64_t n_tok = 0, n_img = 0;
    int32_t max_q = 0, max_ctx = 0;
    int64_t pitch = 1;
    for (const E& e : es) {
        const int64_t rows = entry_rows(*e.j, e.branch);
        n_tok += rows;
        n_img += e.j->tokens;
        max_q = std::max<int32_t>(max_q, (int32_t)rows);
        max_ctx = std::max<int32_t>(max_ctx, (int32_t)e.j->cond[e.branch].lay.rows);
        if (!joint())
            pitch = std::max<int64_t>(pitch, (int64_t)kv_->block_table(e.j->cond[e.branch].seq,
                                                                         kv_group_).size());
    }
    /* The joint layout's per-row arrays: each row's source, the compact image order's bounds and
     * rows, each entry's three modes, and the f32 positions. */
    int32_t* src = hptr<int32_t>(R_SRC);
    int32_t* icu = hptr<int32_t>(R_ICU);
    int32_t* irows = hptr<int32_t>(R_IROWS);
    int32_t* mcu = hptr<int32_t>(R_MCU);
    float* ropef = hptr<float>(R_ROPEF);
    int64_t img = 0;
    if (joint()) icu[0] = 0;
    int32_t* cu = hptr<int32_t>(R_CU);
    int32_t* pos = hptr<int32_t>(R_POS);
    int32_t* tok = hptr<int32_t>(R_TOK);
    int32_t* grope = hptr<int32_t>(R_GROPE);
    float* sig = hptr<float>(R_SIG);
    float* sign = hptr<float>(R_SIGN);
    float* cfg = hptr<float>(R_CFG);
    int32_t* pair = hptr<int32_t>(R_PAIR);
    int32_t* lane = hptr<int32_t>(R_LANE);
    int32_t* flags = hptr<int32_t>(R_FLAGS);
    uint32_t* seed = hptr<uint32_t>(R_SEED);
    int32_t* btab = hptr<int32_t>(R_BT);
    int32_t* used = hptr<int32_t>(R_USED);
    float* sigp = hptr<float>(R_SIGP);
    float* tstep = hptr<float>(R_TSTEP);
    int32_t* samp = hptr<int32_t>(R_SAMP);
    int32_t* stepi = hptr<int32_t>(R_STEPI);
    float* size = hptr<float>(R_SIZE);
    host_entry_.assign((size_t)(4 * S), 0);
    int64_t row = 0;
    cu[0] = 0;
    pass_.lanes.clear();
    pass_.marks.clear();
    for (int64_t i = 0; i < S; ++i) {
        const E& e = es[(size_t)i];
        const Job& j = *e.j;
        const Lane& ln = lanes_[(size_t)e.lane];
        const Cond& c = j.cond[e.branch];
        const int64_t N = j.tokens;
        if (joint()) {
            /* [conditioning | image]: the conditioning's rows from the plugin's copy at this lane
             * and branch, the image's from the compact order. */
            const int64_t P = c.lay.rows;
            const int64_t base = (2 * (int64_t)e.lane + e.branch) * decl_.cond_rows;
            for (int64_t k = 0; k < P; ++k) {
                src[row + k] = (int32_t)(base + k);
                pos[row + k] = (int32_t)k;
                tok[row + k] = 0;
            }
            for (int cc = 0; cc < 4; ++cc)
                std::memcpy(ropef + cc * n_tok + row, c.lay.rope_f.data() + cc * P, (size_t)P * 4);
            mcu[3 * i] = (int32_t)row;
            mcu[3 * i + 1] = (int32_t)(row + c.lay.text);
            mcu[3 * i + 2] = (int32_t)(row + P);
            row += P;
            for (int64_t k = 0; k < N; ++k) {
                src[row + k] = (int32_t)(-(1 + img + k));
                irows[img + k] = (int32_t)(row + k);
                pos[row + k] = (int32_t)k;
                tok[row + k] = 0;
            }
            for (int cc = 0; cc < 4; ++cc)
                std::memcpy(ropef + cc * n_tok + row, c.rope_img_f.data() + cc * N, (size_t)N * 4);
            img += N;
            icu[i + 1] = (int32_t)img;
        } else {
            for (int64_t k = 0; k < N; ++k) { pos[row + k] = (int32_t)k; tok[row + k] = 0; }
            for (int cc = 0; cc < 3; ++cc)
                std::memcpy(grope + cc * n_tok + row, c.rope_img.data() + cc * N, (size_t)N * 4);
        }
        row += N;
        cu[i + 1] = (int32_t)row;
        sig[i] = j.sigmas[(size_t)ln.step];
        sign[i] = j.sigmas[(size_t)ln.step + 1];
        sigp[i] = ln.step > 0 ? j.sigmas[(size_t)ln.step - 1] : 0.0f;
        tstep[i] = discrete() ? j.tsteps[(size_t)ln.step] : 0.0f;
        samp[i] = j.req->sampler;
        stepi[i] = ln.step;
        size[2 * i] = (float)j.req->height;
        size[2 * i + 1] = (float)j.req->width;
        host_entry_[(size_t)(4 * i)] = joint() ? (int32_t)c.lay.rows : 0;
        host_entry_[(size_t)(4 * i + 1)] = j.req->height;
        host_entry_[(size_t)(4 * i + 2)] = j.req->width;
        host_entry_[(size_t)(4 * i + 3)] = e.branch;
        const bool neg = e.branch == 1;
        cfg[i] = neg ? 0.0f : (j.n_branch > 1 ? j.req->cfg : 1.0f);
        pair[i] = neg ? -2 : (j.n_branch > 1 ? (int32_t)(i + 1) : -1);
        lane[i] = e.lane;
        int32_t f = 0;
        bool mark = false;
        if (!neg) {
            if (ln.step == 0) f |= RAD_GEN_F_FIRST;
            if (j.req->inpaint) f |= RAD_GEN_F_INPAINT;
            if (ln.previews_done < (int32_t)ln.preview_at.size() &&
                ln.step + 1 == ln.preview_at[(size_t)ln.previews_done]) {
                f |= RAD_GEN_F_PREVIEW;
                mark = true;
            }
            pass_.lanes.push_back(e.lane);
            pass_.marks.push_back(mark ? 1 : 0);
        }
        flags[i] = f;
        seed[2 * i] = (uint32_t)(ln.seed & 0xffffffffu);
        seed[2 * i + 1] = (uint32_t)(ln.seed >> 32);
        if (joint()) continue;
        const std::vector<int32_t>& bt = kv_->block_table(c.seq, kv_group_);
        for (int64_t b = 0; b < pitch; ++b)
            btab[i * pitch + b] = b < (int64_t)bt.size() ? bt[(size_t)b] : -1;
        used[i] = (int32_t)c.lay.rows;
    }
    reg_[R_CU].used = (S + 1) * 4;
    reg_[R_POS].used = reg_[R_TOK].used = n_tok * 4;
    if (joint()) {
        mcu[3 * S] = (int32_t)n_tok;
        /* The mode-major order: every entry's text, then pictures, then image. */
        int32_t* mrows = hptr<int32_t>(R_MROWS);
        int32_t* msrc = hptr<int32_t>(R_MSRC);
        int32_t* mccu = hptr<int32_t>(R_MCCU);
        float* mrope = hptr<float>(R_MROPE);
        int64_t k = 0;
        for (int mode = 0; mode < 3; ++mode) {
            const int64_t k0 = k;
            int64_t nm = 0;
            for (int64_t i = 0; i < S; ++i) nm += mcu[3 * i + mode + 1] - mcu[3 * i + mode];
            for (int64_t i = 0; i < S; ++i) {
                mccu[mode * (S + 1) + i] = (int32_t)(k - k0);
                for (int32_t r = mcu[3 * i + mode]; r < mcu[3 * i + mode + 1]; ++r, ++k) {
                    mrows[k] = r;
                    msrc[k] = src[r];
                    for (int cc = 0; cc < 4; ++cc)
                        mrope[4 * k0 + cc * nm + (k - k0)] = ropef[cc * n_tok + r];
                }
            }
            mccu[mode * (S + 1) + S] = (int32_t)(k - k0);
            host_nmode_[mode] = k - k0;
        }
        reg_[R_MROWS].used = reg_[R_MSRC].used = n_tok * 4;
        reg_[R_MCCU].used = 3 * (S + 1) * 4;
        reg_[R_MROPE].used = 4 * n_tok * 4;
        reg_[R_SRC].used = n_tok * 4;
        reg_[R_IROWS].used = n_img * 4;
        reg_[R_ICU].used = (S + 1) * 4;
        reg_[R_MCU].used = (3 * S + 1) * 4;
        reg_[R_ROPEF].used = 4 * n_tok * 4;
    } else {
        reg_[R_GROPE].used = 3 * n_tok * 4;
    }
    reg_[R_SIG].used = reg_[R_SIGN].used = reg_[R_CFG].used = S * 4;
    reg_[R_SIGP].used = reg_[R_TSTEP].used = reg_[R_SAMP].used = reg_[R_STEPI].used = S * 4;
    reg_[R_SIZE].used = 2 * S * 4;
    reg_[R_PAIR].used = reg_[R_LANE].used = reg_[R_FLAGS].used = S * 4;
    reg_[R_SEED].used = 2 * S * 4;
    reg_[R_BT].used = joint() ? 0 : S * pitch * 4;
    reg_[R_USED].used = joint() ? 0 : S * 4;
    if (upload() < 0) return nullptr;
    finish_batches(n_tok, S, max_q, step);
    for (size_t rr = 0; rr < ranks_.size(); ++rr) {
        RadBatch& B = out_[rr];
        const int ri = (int)rr;
        B.max_ctx_len = max_ctx;
        B.gen = RAD_GEN_PASS_DENOISE;
        kout_[rr].block_table_pitch = pitch;
        kout_[rr].slot_mapping = nullptr;
        RadGenBatch& G = gout_[rr];
        G.rows = 0;
        G.n_seg = 0;
        G.seg_bidi_h = nullptr;
        G.n_px = 0;
        G.px_h = nullptr;
        G.inpaint = 0;
        G.n_lane = 0;
        G.lane_h = nullptr;
        G.rope = joint() ? nullptr : dptr<int32_t>(ri, R_GROPE);
        G.lane_dh = hptr<int32_t>(R_LANE);
        G.sigma_prev = dptr<float>(ri, R_SIGP);
        G.timestep = dptr<float>(ri, R_TSTEP);
        G.sampler = dptr<int32_t>(ri, R_SAMP);
        G.step = dptr<int32_t>(ri, R_STEPI);
        G.size_px = dptr<float>(ri, R_SIZE);
        G.entry_h = host_entry_.data();
        if (joint()) {
            G.src = dptr<int32_t>(ri, R_SRC);
            G.n_img = n_img;
            G.img_cu = dptr<int32_t>(ri, R_ICU);
            G.img_rows = dptr<int32_t>(ri, R_IROWS);
            G.mode_cu = dptr<int32_t>(ri, R_MCU);
            G.rope_f = dptr<float>(ri, R_ROPEF);
            G.slot_mapping = nullptr;
            G.mode_rows = dptr<int32_t>(ri, R_MROWS);
            G.mode_src = dptr<int32_t>(ri, R_MSRC);
            G.mode_ccu = dptr<int32_t>(ri, R_MCCU);
            G.n_mode_h = host_nmode_;
            G.mode_rope_f = dptr<float>(ri, R_MROPE);
        }
    }
    pass_.kind = RAD_GEN_PASS_DENOISE;
    return &out_[0];
}

const RadBatch* Driver::build_decode(int32_t l, bool preview, int step) {
    for (int g = 0; g < R_N; ++g) reg_[g].used = 0;
    const Lane& ln = lanes_[(size_t)l];
    const Job* j = job_locked(ln.job_id);
    if (!j) return nullptr;
    int32_t* cu = hptr<int32_t>(R_CU);
    cu[0] = 0;
    cu[1] = (int32_t)j->tokens;
    reg_[R_CU].used = 2 * 4;
    if (upload() < 0) return nullptr;
    finish_batches(j->tokens, 1, (int32_t)j->tokens, step);
    for (size_t rr = 0; rr < ranks_.size(); ++rr) {
        RadBatch& B = out_[rr];
        B.max_ctx_len = 0;
        B.gen = RAD_GEN_PASS_DECODE;
        B.n_kv_groups = 0;
        B.kv = nullptr;
        RadGenBatch& G = gout_[rr];
        G = RadGenBatch{};
        G.dec_lane = l;
        G.dec_height = j->req->height;
        G.dec_width = j->req->width;
        G.dec_preview = preview ? 1 : 0;
    }
    dec_bytes_ = (int64_t)j->req->width * j->req->height * decl_.pixel_channels;
    rb_.assign(1, Readback{ decl_.image_out, dec_bytes_, img_host_ });
    pass_.kind = RAD_GEN_PASS_DECODE;
    pass_.job = j->id;
    pass_.lane = l;
    pass_.preview = preview;
    return &out_[0];
}

int Driver::done() {
    std::lock_guard<std::mutex> lk(mu_);
    const uint64_t now = rad_mono_ns();
    switch (pass_.kind) {
    case RAD_GEN_PASS_COND: {
        Job* j = job_locked(pass_.job);
        if (!j) break;
        j->cond[pass_.branch].encoded = true;
        bool all = true;
        for (int b = 0; b < j->n_branch; ++b) all = all && j->cond[b].encoded;
        if (all) j->req->ticket->t_cond_ns = (int64_t)now;
        break;
    }
    case RAD_GEN_PASS_DENOISE:
        for (size_t i = 0; i < pass_.lanes.size(); ++i) {
            Lane& ln = lanes_[(size_t)pass_.lanes[i]];
            Job* j = job_locked(ln.job_id);
            if (!j) continue;
            if (ln.step == 0 && j->req->ticket->t_first_step_ns == 0)
                j->req->ticket->t_first_step_ns = (int64_t)now;
            ++ln.step;
            const int32_t n_steps = (int32_t)j->sigmas.size() - 1;
            if (pass_.marks[i]) ln.preview_due = true;
            if (ln.step >= n_steps) {
                ln.final_due = true;
                j->req->ticket->t_denoised_ns = (int64_t)now;
            }
        }
        break;
    case RAD_GEN_PASS_DECODE: {
        Lane& ln = lanes_[(size_t)pass_.lane];
        Job* j = job_locked(pass_.job);
        if (!j) break;
        Image im;
        im.index = ln.index;
        im.w = j->req->width;
        im.h = j->req->height;
        im.ch = (int32_t)decl_.pixel_channels;
        im.seed = ln.seed;
        im.px.assign((const uint8_t*)img_host_, (const uint8_t*)img_host_ + dec_bytes_);
        if (pass_.preview) {
            im.preview = true;
            im.preview_no = ln.previews_done;
            ln.preview_due = false;
            ++ln.previews_done;
        } else {
            ln.final_due = false;
            ln.finished = true;
            ++j->images_done;
        }
        j->req->ticket->push(std::move(im));
        if (j->images_done >= j->req->n) {
            j->req->ticket->t_done_ns = (int64_t)now;
            j->req->ticket->finish(RAD_OK);
        }
        break;
    }
    default:
        break;
    }
    pass_ = Pass{};
    dec_bytes_ = 0;
    rb_.clear();
    return RAD_OK;
}

}  /* namespace gen */
}  /* namespace rad */
