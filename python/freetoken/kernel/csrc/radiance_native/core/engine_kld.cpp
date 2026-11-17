/* engine_kld.cpp -- the KL mode on the Engine: the driver that replaces the HTTP server, and the
 * copy of a step's scored rows off the cards. The arithmetic and files are core/kld.h's. */
#include "engine_priv.h"
#include "kld.h"

#include "device/device.h"
#include "server/sink.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace rad {

/* Every request of a corpus run through the scheduler, each reaped once its sink is done; the
 * documents that ended in an error go to *bad. With `fit_pool`, a request is submitted only while
 * the whole lengths of those in flight, its own included, fit the KV pool at once, so none is ever
 * preempted: a 256K-token document runs beside short ones only as far as the pool holds both.
 * Without it, all are submitted under one hold of the queue, so batching depends only on the
 * corpus and the flags and a rerun batches identically. `progress` runs every `every_s` seconds
 * with the documents done and the seconds since the start. */
int Engine::run_docs(std::vector<std::unique_ptr<Request>>& reqs,
                     std::vector<std::unique_ptr<server::Sink>>& sinks, bool fit_pool,
                     const char* tag, int every_s,
                     const std::function<void(size_t, double)>& progress, int* bad) {
    const size_t n = reqs.size();
    const KVManager& kv = ranks_[0]->kv;
    /* What a document holds of the pool at most: its tokens, plus a block lost to rounding and
     * one to a copy-on-write in each group, bounded generously. */
    constexpr int64_t kSlack = 1024;
    std::vector<int64_t> len(n);
    for (size_t d = 0; d < n; ++d) len[d] = (int64_t)reqs[d]->prompt.size() + kSlack;
    std::vector<char> live(n, 0);
    size_t next = 0, n_live = 0, finished = 0;
    int64_t held = 0;
    if (!fit_pool) {
        std::vector<int> rc(n, RAD_OK);
        sched_.add_all(reqs.data(), n, rc.data());
        for (size_t d = 0; d < n; ++d) {
            if (rc[d] >= 0) continue;
            RAD_ERR("%s: the scheduler refused document %zu (%zu tokens): %s", tag, d,
                    (size_t)(len[d] - kSlack), rad_strerror(rc[d]));
            return rc[d];
        }
        std::fill(live.begin(), live.end(), 1);
        next = n_live = n;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto last = t0;
    while (finished < n) {
        while (next < n && (n_live == 0 || kv.can_ever_fit(held + len[next]))) {
            int rc = RAD_OK;
            sched_.add_all(&reqs[next], 1, &rc);
            if (rc < 0) {
                RAD_ERR("%s: the scheduler refused document %zu (%zu tokens): %s", tag, next,
                        (size_t)(len[next] - kSlack), rad_strerror(rc));
                return rc;
            }
            live[next] = 1;
            ++n_live;
            held += len[next];
            ++next;
        }
        const int ls = loop_status_.load(std::memory_order_acquire);
        if (ls < 0) return ls;
        size_t oldest = 0;
        while (oldest < next && !live[oldest]) ++oldest;
        sinks[oldest]->wait(200);
        for (size_t d = oldest; d < next; ++d) {
            if (!live[d] || !sinks[d]->done()) continue;
            const server::Finish f = sinks[d]->finish_reason();
            if (f == server::Finish::Error || f == server::Finish::Cancelled) {
                RAD_ERR("%s: document %zu did not finish: %s", tag, d,
                        rad_strerror(sinks[d]->status()));
                ++*bad;
            }
            sched_.reap(sinks[d]->id);
            sinks[d].reset();
            live[d] = 0;
            --n_live;
            held -= len[d];
            ++finished;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last >= std::chrono::seconds(every_s)) {
            last = now;
            progress(finished, std::chrono::duration<double>(now - t0).count());
        }
    }
    return RAD_OK;
}

int Engine::run_kld() {
    /* Each rank's logits buffer must hold a step's tokens after the sampler's rows
     * (RadBuildCtx::max_out_rows); a plugin that ignored the field is refused here rather than
     * overrun. */
    const int64_t need = cfg_.max_tok + cfg_.max_seqs;
    const int64_t nv = meta_.n_vocab;
    kld_vocab0_.clear();
    kld_width_.clear();
    for (const auto& rk : ranks_) {
        const Program& p = rk->program;
        if (p.logits_buf == RAD_NULL_HANDLE || (size_t)p.logits_buf >= p.buffers.size()) {
            RAD_ERR("kld: rank %d declared no logits buffer", rk->index);
            return RAD_E_UNSUPPORTED;
        }
        const RadBufDecl& d = p.buffers[p.logits_buf].decl;
        if (d.dtype != RAD_F32 || d.rank != 2 || d.shape[0] < need) {
            RAD_ERR("kld: the '%s' plugin's logits buffer is %lld rows of %s; the KL mode needs "
                    "f32 rows for a whole step, %lld (it reads RadBuildCtx::max_out_rows)",
                    plugins_.arch_plugin.c_str(), (long long)d.shape[0],
                    d.dtype == RAD_F32 ? "f32" : "another dtype", (long long)need);
            return RAD_E_UNSUPPORTED;
        }
        /* The columns a rank owns: the sampler's split, which declare_sampler checked, or the
         * whole vocabulary on every rank (then read from rank 0 only). */
        const int64_t w = d.shape[1];
        if (w >= nv) {
            if (!kld_width_.empty()) continue;
            kld_vocab0_.push_back(0);
            kld_width_.push_back(nv);
            continue;
        }
        kld_vocab0_.push_back(rk->sampler.config().vocab_off);
        kld_width_.push_back(w);
    }
    int64_t cover = 0;
    for (int64_t w : kld_width_) cover += w;
    if (cover != nv) {
        RAD_ERR("kld: the ranks' logits columns add up to %lld of a %lld-token vocabulary",
                (long long)cover, (long long)nv);
        return RAD_E_STATE;
    }
    kld_host_.assign(kld_width_.size(), nullptr);
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        kld_host_[r] = static_cast<float*>(
            rad_dev_alloc(cfg_.max_tok * kld_width_[r] * (int64_t)sizeof(float), RAD_MEM_HOST_PINNED));
        if (!kld_host_[r]) {
            RAD_ERR("kld: no pinned host memory for a step's logits (%lld MiB a rank); lower "
                    "--max-num-batched-tokens", (long long)(cfg_.max_tok * kld_width_[r] * 4 >> 20));
            return RAD_E_NOMEM;
        }
    }

    auto kld = std::make_unique<Kld>();
    RAD_TRY(kld->open(cfg_, vocab_, nv, cfg_.model));
    /* Published before add_all takes the scheduler's lock; the step loop reads it only on steps
     * planned under that lock afterwards. */
    kld_ = std::move(kld);

    const size_t n = kld_->n_docs();
    /* A document the KV pool cannot hold even alone would wait at the head of the queue forever:
     * admission takes only a chunk the pool can hold, and nothing ever frees enough. Refused up
     * front, naming every such document. The pool is the least that a full paged group holds. */
    {
        const KVManager& kv = ranks_[0]->kv;
        int64_t pool = INT64_MAX;
        for (int32_t g = 0; g < kv.n_groups(); ++g) {
            const KVGroupPlan* p = kv.plan(g);
            if (p && p->paged() && p->window_blocks == 0)
                pool = std::min<int64_t>(pool, p->n_blocks * p->block_size);
        }
        int too_long = 0;
        for (size_t d = 0; d < n; ++d) {
            const int64_t len = (int64_t)kld_->tokens(d).size();
            if (kv.can_ever_fit(len)) continue;
            RAD_ERR("kld: document %zu is %lld tokens and the KV pool holds %lld, so it can never "
                    "be scored; give the cache more room or shorten the document", d,
                    (long long)len, (long long)pool);
            ++too_long;
        }
        if (too_long) return RAD_E_FULL;
    }

    /* --kld-lora: every document runs through one runtime LoRA adapter, held for the whole run --
     * how an adapter is scored against a reference recorded from its merged or HF counterpart. */
    int32_t  lora_slot = -1;
    uint64_t lora_uid = 0;
    std::shared_ptr<void> lora_hold;
    if (!cfg_.kld_lora.empty() &&
        lora_acquire(cfg_.kld_lora, &lora_slot, &lora_uid, &lora_hold) != RAD_OK) {
        RAD_ERR("kld: --kld-lora '%s' is not a loaded adapter (--lora NAME=PATH)", cfg_.kld_lora.c_str());
        return RAD_E_NOTFOUND;
    }
    if (lora_slot >= 0)
        RAD_INFO("kld: scoring through the runtime LoRA adapter '%s' (slot %d)", cfg_.kld_lora.c_str(),
                 lora_slot);

    std::vector<std::unique_ptr<server::Sink>> sinks(n);
    std::vector<std::unique_ptr<Request>> reqs(n);
    for (size_t d = 0; d < n; ++d) {
        sinks[d] = std::make_unique<server::Sink>(64);
        auto r = std::make_unique<Request>();
        r->id         = d + 1;
        r->prompt     = kld_->tokens(d);
        r->max_tokens = 1;
        r->ignore_eos = true;
        r->sp.temp    = 0.0f;
        r->score      = true;
        /* A loaded control vector scores at the deployment's default (--control-vector on|off). */
        r->sp.use_control_vector = cvec_default_on();
        /* --kld-lora-sources: the other documents run as the base, in the same steps. */
        const bool via = cfg_.kld_lora_sources.empty() ||
                         kld_->source(d).rfind(cfg_.kld_lora_sources, 0) == 0;
        r->lora_slot  = via ? lora_slot : -1;
        r->lora_uid   = via ? lora_uid : 0;
        r->sink       = sinks[d].get();
        sinks[d]->id  = r->id;
        reqs[d] = std::move(r);
    }
    /* A recording that also records the calibration statistics (--calib-dir) must not preempt:
     * the re-prefilled tokens would count twice. */
    const auto t0 = std::chrono::steady_clock::now();
    int bad = 0;
    RAD_TRY(run_docs(reqs, sinks, !cfg_.calib_dir.empty(), "kld", 15, [&](size_t done, double s) {
        RAD_INFO("kld: %lld / %lld positions, %zu / %zu documents, %.0f s",
                 (long long)kld_->rows_done(), (long long)kld_->n_rows(), done, n, s);
    }, &bad));
    if (bad) return RAD_E_STATE;
    RAD_INFO("kld: %zu documents in %.0f s, %lld preemption(s)", n,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
             (long long)sched_.metrics().preemptions);
    if (!cfg_.calib_dir.empty() && sched_.metrics().preemptions) {
        RAD_ERR("kld: the calibration statistics counted the tokens of %lld preempted request(s) "
                "twice; the reference is written, the statistics are not to be used",
                (long long)sched_.metrics().preemptions);
        RAD_TRY(kld_->finish());
        return RAD_E_STATE;
    }
    return kld_->finish();
}

/* --calib-corpus: every document through prefill once as a plain request, nothing scored. The
 * plugin records its statistics while the steps run and writes them once it has seen the corpus's
 * tokens (RadBuildCtx::calib_tokens). A preempted request is prefilled again, which would count
 * its tokens twice, so the documents are fed so that it cannot (run_docs), and a run that
 * preempts all the same is reported as failed. */
int Engine::run_calib() {
    if (!calib_) return RAD_E_STATE;
    const size_t n = calib_->n_docs();
    {
        const KVManager& kv = ranks_[0]->kv;
        int too_long = 0;
        for (size_t d = 0; d < n; ++d) {
            const int64_t len = (int64_t)calib_->tokens(d).size();
            if (kv.can_ever_fit(len)) continue;
            RAD_ERR("calib: document %zu is %lld tokens, more than the KV pool can ever hold", d,
                    (long long)len);
            ++too_long;
        }
        if (too_long) return RAD_E_FULL;
    }
    std::vector<std::unique_ptr<server::Sink>> sinks(n);
    std::vector<std::unique_ptr<Request>> reqs(n);
    for (size_t d = 0; d < n; ++d) {
        sinks[d] = std::make_unique<server::Sink>(64);
        auto r = std::make_unique<Request>();
        r->id         = d + 1;
        r->prompt     = calib_->tokens(d);
        r->max_tokens = 1;           /* sampled from the last prefill row, never fed back */
        r->ignore_eos = true;
        r->sp.temp    = 0.0f;
        r->sink       = sinks[d].get();
        sinks[d]->id  = r->id;
        reqs[d] = std::move(r);
    }
    const auto t0 = std::chrono::steady_clock::now();
    int bad = 0;
    RAD_TRY(run_docs(reqs, sinks, true, "calib", 30, [&](size_t done, double s) {
        RAD_INFO("calib: %zu / %zu documents, %.0f s", done, n, s);
    }, &bad));
    const long long pre = (long long)sched_.metrics().preemptions;
    RAD_INFO("calib: %zu documents, %lld tokens in %.0f s, %lld preemption(s)", n,
             (long long)calib_->n_tokens(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), pre);
    if (pre) {
        RAD_ERR("calib: %lld request(s) were preempted and prefilled again, so their tokens were "
                "counted twice; rerun with fewer --max-num-seqs or a larger KV pool", pre);
        return RAD_E_STATE;
    }
    return bad ? RAD_E_STATE : RAD_OK;
}

/* Copy a step's scored rows off the cards. They follow the sampler's rows in the logits buffer in
 * the batch builder's entry order (core/sched/batch.cpp), so the walk below repeats that loop and
 * checks the count. Each rank copies on its own stream behind the step; KL mode does not
 * pipeline, so the next step waits for this. */
int Engine::score_rows(const RadBatch* b, const StepOut& so) {
    const int64_t n = b->n_out - so.n_rows;
    if (n <= 0) return RAD_OK;
    const StepPlan& plan = sched_.plan();
    std::vector<KldSpan> spans;
    int64_t row = 0;
    for (const StepEntry& e : plan.e) {
        if (!(e.req && e.req->score && e.is_prefill && e.hidden_row < 0)) continue;
        spans.push_back({ e.req->id, (int64_t)e.ctx_len, (int64_t)e.n_tokens, row });
        row += e.n_tokens;
    }
    if (row != n || n > cfg_.max_tok) {
        RAD_ERR("kld: step %d names %lld scored rows and its plan %lld", b->step, (long long)n,
                (long long)row);
        return RAD_E_STATE;
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        Rank& rk = *ranks_[r];
        RAD_TRY(rad_dev_set(rk.device));
        const auto* src = static_cast<const float*>(rk.ctx.buf_ptr(rk.program.logits_buf));
        if (!src) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(kld_host_[r], src + (int64_t)so.n_rows * kld_width_[r],
                                 n * kld_width_[r] * (int64_t)sizeof(float), rk.ctx.stream()));
    }
    for (size_t r = 0; r < kld_width_.size(); ++r) {
        RAD_TRY(rad_dev_set(ranks_[r]->device));
        RAD_TRY(rad_stream_sync(ranks_[r]->ctx.stream()));
    }
    RAD_TRY(rad_dev_set(ranks_[0]->device));
    std::vector<KldShard> shards(kld_width_.size());
    for (size_t r = 0; r < shards.size(); ++r)
        shards[r] = { kld_host_[r], kld_vocab0_[r], kld_width_[r] };
    return kld_->score(spans, shards);
}

}  /* namespace rad */
