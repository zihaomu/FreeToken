/* engine_draft.cpp -- the two drafter drivers (spec §10).
 *
 * A draft head runs as extra passes after the trunk step. SERIAL runs `depth` dependent passes;
 * BLOCK runs a context pass, then one pass that fills the whole block. The architecture plugin
 * declares which (rad_declare_drafter); nothing here depends on the model.
 */
#include "engine_priv.h"

#include "sample/mtp_couple_evidence.h"
#include "sched/mtp_couple_bind.h"
#include "sched/mtp_couple_plan.h"
#include "sched/scheduler.h"
#include "device/device.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the card's step state */
/* Every rank's zeroed StepSlots and sorted end-of-generation ids, and the pinned host copy of rank
 * 0's slots the step reads back. Allocated once at start, so the step path never allocates. */
int Engine::step_state_init() {
    std::vector<int32_t> eog;
    if (vocab_) eog = vocab_->eog();
    std::sort(eog.begin(), eog.end());
    eog.erase(std::unique(eog.begin(), eog.end()), eog.end());
    /* Scheduler::ends_every_request reads the same ids. */
    sched_.set_eog(eog);
    const int64_t n_slot = cfg_.max_seqs > 0 ? cfg_.max_seqs : 1;
    const int64_t slot_bytes = n_slot * (int64_t)sizeof(StepSlot);
    for (auto& rk : ranks_) {
        RAD_TRY(rad_dev_set(rk->device));
        rk->step_slots = static_cast<StepSlot*>(rad_dev_alloc(slot_bytes, RAD_MEM_DEVICE));
        rk->n_eog = (int32_t)eog.size();
        rk->eog = static_cast<int32_t*>(
            rad_dev_alloc(std::max<int64_t>(1, (int64_t)eog.size()) * 4, RAD_MEM_DEVICE));
        if (!rk->step_slots || !rk->eog) {
            RAD_ERR("step state: no device memory for %lld slot(s) on card %d",
                    (long long)n_slot, rk->device);
            return RAD_E_NOMEM;
        }
        RAD_TRY(rad_memset_async(rk->step_slots, 0, slot_bytes, rk->ctx.stream()));
        if (!eog.empty())
            RAD_TRY(rad_memcpy_async(rk->eog, eog.data(), (int64_t)eog.size() * 4,
                                     rk->ctx.stream()));
        RAD_TRY(rad_stream_sync(rk->ctx.stream()));
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0]->device));
    slots_host_ = static_cast<StepSlot*>(rad_dev_alloc(slot_bytes, RAD_MEM_HOST_PINNED));
    if (!slots_host_) return RAD_E_NOMEM;

    /* --draft-coupling: the head's two planes, found by name in the declared program, and the
     * pinned staging beside them. A program without them is refused rather than served
     * uncoupled. */
    mtp_couple_ = cfg_.draft_coupling;
    if (mtp_couple_) {
        /* Coupling is defined on a linear chain of drafts: a serial head's rounds, or a block
         * drafter whose walk samples each position in turn (DFlash2, dflash_couple_propose), its
         * position l standing for round l + 1. Either way one draw a drafting row and one state a
         * (round, row). */
        if ((drafter_.kind != RAD_DRAFT_SERIAL && drafter_.kind != RAD_DRAFT_BLOCK) ||
            drafter_.depth <= 0) {
            RAD_ERR("--draft-coupling on requires a linear draft chain (a serial head or a block "
                    "drafter with a coupled walk, depth > 0); this model declares drafter '%s' "
                    "kind %d depth %d, which has no per-round draw to couple",
                    drafter_.name ? drafter_.name : "(none)", drafter_.kind, drafter_.depth);
            return RAD_E_UNSUPPORTED;
        }
        const Program& prog = ranks_[0]->program;
        for (size_t i = 0; i < prog.buffers.size(); ++i) {
            const std::string& name = prog.buffers[i].name;
            auto ends = [&](const char* suffix) {
                const size_t n = std::strlen(suffix);
                return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
            };
            if (ends("mtp_couple_draws")) couple_draws_buf_ = (rad_buf)i;
            if (ends("mtp_couple_state")) couple_state_buf_ = (rad_buf)i;
        }
        if (!couple_draws_buf_ || !couple_state_buf_) {
            RAD_ERR("--draft-coupling on, but the architecture declared no coupled MTP planes; "
                    "this model's drafter cannot couple");
            return RAD_E_STATE;
        }
        const RadBufDecl& sd = prog.buffers[couple_state_buf_].decl;
        if (sd.rank != 3 || sd.shape[2] != (int64_t)RAD_MTP_COUPLE_STATE_WORDS ||
            sd.shape[0] < (int64_t)drafter_.depth) {
            RAD_ERR("--draft-coupling: 'mtp_couple_state' is not the [rounds, rows, %d]-word "
                    "plane the coupling reads", (int)RAD_MTP_COUPLE_STATE_WORDS);
            return RAD_E_STATE;
        }
        couple_rows_ = sd.shape[1];
        /* One entry per card slot, indexed by StepEntry::card as slots_host_ is. */
        couple_by_card_.assign((size_t)n_slot, RadMtpCoupleDraw{});
        couple_draw_stage_ = static_cast<RadMtpCoupleDraw*>(
            rad_dev_alloc(n_slot * (int64_t)sizeof(RadMtpCoupleDraw), RAD_MEM_HOST_PINNED));
        couple_readback_ = static_cast<RadMtpCoupleState*>(rad_dev_alloc(
            (int64_t)drafter_.depth * couple_rows_ * (int64_t)sizeof(RadMtpCoupleState),
            RAD_MEM_HOST_PINNED));
        if (!couple_draw_stage_ || !couple_readback_) return RAD_E_NOMEM;
        if (drafter_.kind == RAD_DRAFT_BLOCK)
            RAD_INFO("DFlash coupled LIVE: %d position(s), x~Q along the candidate walk, maximal "
                     "coupling at the verify; fixed-seed tokens are NOT preserved", drafter_.depth);
        else
            RAD_INFO("MTP coupled LIVE: %d linear round(s), x~Q from the final reranked top8, "
                     "maximal coupling at the verify; fixed-seed tokens are NOT preserved",
                     drafter_.depth);
    }

    RAD_TRY(rad_event_create_as(&step_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
    RAD_TRY(rad_event_create_as(&tok_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
    return RAD_OK;
}

void Engine::step_state_fini() {
    for (auto& rk : ranks_) {
        (void)rad_dev_set(rk->device);
        if (rk->step_slots) rad_dev_free(rk->step_slots, RAD_MEM_DEVICE);
        if (rk->eog) rad_dev_free(rk->eog, RAD_MEM_DEVICE);
        rk->step_slots = nullptr;
        rk->eog = nullptr;
    }
    if (slots_host_) rad_dev_free(slots_host_, RAD_MEM_HOST_PINNED);
    slots_host_ = nullptr;
    if (couple_draw_stage_) rad_dev_free(couple_draw_stage_, RAD_MEM_HOST_PINNED);
    if (couple_readback_) rad_dev_free(couple_readback_, RAD_MEM_HOST_PINNED);
    couple_draw_stage_ = nullptr;
    couple_readback_ = nullptr;
    if (step_end_) rad_event_destroy(step_end_);
    step_end_ = nullptr;
    if (tok_end_) rad_event_destroy(tok_end_);
    tok_end_ = nullptr;
}

/* ------------------------------------------------------------------ the serial driver
 *
 * RAD_DRAFT_SERIAL: one head pass per drafted token after the trunk step (MTP, EAGLE, Medusa).
 * Round 1 consumes the sampled token and the hidden state of the position it came from, which
 * depends on how many drafts were accepted; it runs in the history pass, whose rows include that
 * position (RadBatch::draft_out_ids).
 *
 * The card, not the host, resolves those positions (sched/advance.h): behind the trunk's sampler
 * every rank runs the acceptance into its StepSlots, and an advance before each draft pass fills
 * in its positions and tokens. So the host issues the whole step (trunk, history, every round)
 * without waiting and reads back once after the last round, instead of idling the card for a
 * readback, commit and batch build per step.
 *
 * Each pass is an ordinary RadBatch through the same barrier and sampler chain; `draft_pass`
 * tells the plugin to run the head. Tokens go out through set_draft(), like prompt lookup's; a
 * sequence the head declined keeps n_draft 0, leaving it to the n-gram drafter.
 */
bool Engine::serial_drafting() const {
    return drafter_.kind == RAD_DRAFT_SERIAL && drafter_.depth > 0;
}

/* One advance on every rank against the scheduler's latest build, over its first `rows` rows
 * when given, else all. */
int Engine::advance_all(int mode, int round, bool from_proposal, int rows) {
    for (size_t i = 0; i < ranks_.size(); ++i) {
        Rank& rk = *ranks_[i];
        AdvanceArgs a = sched_.advance_args((int)i);
        if (rows >= 0 && rows < a.n_seq) a.n_seq = rows;
        a.mode  = mode;
        a.round = round;
        a.st    = rk.step_slots;
        a.eog   = rk.eog;
        a.n_eog = rk.n_eog;
        if (from_proposal) {
            a.src = static_cast<const int32_t*>(drafter_.proposal ? rk.ctx.buf_ptr(drafter_.proposal)
                                                                  : rk.ctx.buf_ptr(rk.tok_buf));
            a.src_pitch = (drafter_.proposal && drafter_.proposal_pitch > 0)
                              ? drafter_.proposal_pitch : 1;
            /* A tree drafter's parents, at the proposal's pitch (rad_drafter_tree). */
            if (tree_ && mode == ADV_BTAKE) {
                a.par = static_cast<const int32_t*>(rk.ctx.buf_ptr(tree_parents_));
                if (!a.par) {
                    RAD_ERR("drafter '%s': rank %zu has no parents buffer", drafter_.name, i);
                    return RAD_E_STATE;
                }
            }
        } else {
            a.src = static_cast<const int32_t*>(rk.ctx.buf_ptr(rk.tok_buf));
        }
        if (!a.src || !a.st) {
            RAD_ERR("drafter '%s': rank %zu has no %s to advance from", drafter_.name, i,
                    a.st ? "proposal buffer" : "step state");
            return RAD_E_STATE;
        }
        RAD_TRY(step_advance(a, rk.ctx.stream()));
    }
    return RAD_OK;
}

/* ADV_COMPACT on every rank, against the scheduler's latest build (the tree step just accepted):
 * the accepted paths' K and V into place in every paged group. Each rank's pools are its own. */
int Engine::tree_compact(int rows) {
    for (size_t i = 0; i < ranks_.size(); ++i) {
        Rank& rk = *ranks_[i];
        AdvanceArgs a = sched_.advance_args((int)i);
        if (rows >= 0 && rows < a.n_seq) a.n_seq = rows;
        a.mode = 0;
        a.st   = rk.step_slots;
        const Program& p = rk.program;
        for (size_t g = 1; g < p.kv_groups.size() && (int)g < a.n_groups; ++g) {
            const int kind = p.kv_groups[g].decl.kind;
            if (kind != RAD_KV_FULL && kind != RAD_KV_WINDOW) continue;
            const KVPoolBinding* b = rk.ctx.kv_binding((uint32_t)g);
            if (!b || !b->base || b->rank != 4) continue;   /* a zero group on this rank */
            const int bits = rad_dtype_bits(b->dtype);
            const int64_t row = bits > 0 ? b->shape[3] * bits / 8 : 0;
            int32_t layers = 0;
            for (int32_t s : b->layer_slot) layers += s >= 0;
            if (row <= 0 || row % 4 || (bits != 8 && bits != 16 && bits != 32) ||
                (b->bytes && (int64_t)layers * b->layer_stride > b->bytes)) {
                RAD_ERR("speculation: KV group %zu's cache (%s, %lld-byte rows) is not one a tree's "
                        "acceptance can move", g, rad_dtype_name(b->dtype), (long long)row);
                return RAD_E_UNSUPPORTED;
            }
            if (a.n_pool >= kAdvMaxPools) {
                RAD_ERR("speculation: more than %d paged KV groups for a tree's acceptance",
                        (int)kAdvMaxPools);
                return RAD_E_UNSUPPORTED;
            }
            AdvPool& q = a.pool[a.n_pool++];
            q.base         = static_cast<char*>(b->base);
            q.layer_stride = b->layer_stride;
            q.n_layers     = layers;
            q.n_head       = (int32_t)b->shape[1];
            q.row_bytes    = (int32_t)row;
            q.grp          = (int32_t)g;
        }
        RAD_TRY(step_compact(a, rk.ctx.stream()));
    }
    return RAD_OK;
}

/* Issue the step's draft passes behind it, without waiting. After the trunk's ranks join: the
 * acceptance on every rank, the sampled-id readback (before a round can overwrite them), the
 * history pass and every round, then the final proposal into the slots and their readback.
 * Nothing synchronises here: collect_step waits for the ids alone, and the step loop for the
 * passes after the commit, before finish_serial_drafter reads the slots. */
int Engine::issue_serial_drafter(int step, StepOut& so) {
    serial_rounds_ = 0;
    serial_rows_   = 0;
    serial_done_   = 0;
    /* A step issued ahead continues from the card's acceptance, so run it even without drafts. */
    if (!serial_drafting()) {
        if (pipelining_) RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
        return RAD_OK;
    }

    RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
    RAD_TRY(collect_issue(so, /*fence=*/true));

    int rounds = drafter_.depth;
    serial_row_.resize((size_t)cfg_.max_seqs);
    /* `n` counts rows that can draft. It can be zero while the history pass still has work (a
     * mid-prompt sequence has hidden states the head needs but no token to propose from). */
    const int n = sched_.serial_begin(&rounds, serial_row_.data(), (int)serial_row_.size());
    if (rounds <= 0) return RAD_OK;
    serial_rounds_ = rounds;
    serial_rows_   = n;
    /* Before any serial pass reads a draw; round 1 is the history pass below. */
    if (mtp_couple_ && n > 0) RAD_TRY(stage_couple_draws(serial_row_.data(), n));

    /* RADIANCE_DEBUG_STAGE: running means of a draft pass's host time: batch + advance, greedy
     * sampler chain, barrier release and issue. They cost the step nothing until their sum exceeds
     * the previous pass's device time. */
    static const bool stg = std::getenv("RADIANCE_DEBUG_STAGE") != nullptr;
    static double a_batch = 0, a_samp = 0, a_rel = 0, a_iss = 0;
    static long   n_pass = 0;

    /* Pass 0, then rounds 2..rounds: round 1 has no pass of its own. */
    for (int rd = 0; rd <= rounds; rd += rd == 0 ? 2 : 1) {
        if (rd > 0 && n <= 0) break;
        const auto tA = std::chrono::steady_clock::now();
        const RadBatch* db = sched_.serial_batch(rd, step);
        if (!db) { if (rd == 0) continue; break; }
        /* Pass 0 is the history pass, writing the head's own K and V (nothing else does). It is
         * also draft round 1 for drafting rows, whose inputs are the last row of their history run
         * (RadBatch::draft_out_ids), so it ends in the lm_head over those rows. Round r >= 2 embeds
         * round r-1's proposal, which the advance takes from the proposal buffer on the card. */
        const int drafted = rd == 0 ? (int)db->n_draft_out : n;
        RAD_TRY(advance_all(rd == 0 ? ADV_HIST : ADV_ROUND, rd, rd > 0));
        const auto tB = std::chrono::steady_clock::now();

        /* Greedy, not the request's sampler: the target verifies against its own draw
         * (core/sample/accept.h), so sampling a draft would spend the request's RNG stream and
         * lower acceptance. A head with its own proposal buffer samples nothing (a zero-row
         * chain). */
        for (auto& rk : ranks_) {
            int s = rk->sampler.build_greedy(drafter_.proposal ? 0 : drafted);
            if (s < 0) {
                RAD_ERR("drafter '%s': build_greedy: %s", drafter_.name, rad_strerror(s));
                return s;
            }
        }

        const auto tC = std::chrono::steady_clock::now();
        issue_what_.store(kIssueAll, std::memory_order_release);
        step_batch_.store(db, std::memory_order_release);
        /* Timed separately: the first crossing releases the parked ranks (the wake), the second
         * waits for them to finish issuing (the issue). */
        barrier_->arrive_and_wait();
        const auto tC2 = std::chrono::steady_clock::now();
        barrier_->arrive_and_wait();
        const auto tD = std::chrono::steady_clock::now();

        int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("drafter '%s': pass %d of step %d failed: %s -- the engine cannot continue",
                    drafter_.name, rd, step, rad_strerror(st));
            return st;
        }
        if (stg) {
            using us = std::chrono::duration<double, std::micro>;
            a_batch += us(tB - tA).count();
            a_samp  += us(tC - tB).count();
            a_rel   += us(tC2 - tC).count();
            a_iss   += us(tD - tC2).count();
            if (++n_pass % 400 == 0)
                RAD_INFO("draft: %ld passes; batch %.1f, sampler %.1f, release %.1f, issue %.1f us "
                         "a pass", n_pass, a_batch / (double)n_pass, a_samp / (double)n_pass,
                         a_rel / (double)n_pass, a_iss / (double)n_pass);
        }
        if (rd > 0) serial_done_ = rd;
        else if (drafted > 0) serial_done_ = 1;
    }
    if (serial_done_ <= 0 || n <= 0) return RAD_OK;

    /* The last round's proposal into the slots, then rank 0's slots back to the host. Only the
     * drafting rows: at depth 1 the last pass is the history pass, whose later rows only carried
     * history (Scheduler::serial_batch). */
    RAD_TRY(advance_all(ADV_TAKE, serial_done_, true, n));
    Rank& r0 = *ranks_[0];
    if (mtp_couple_) {
        /* Rounds 1..done of the drafting rows, on the same stream as the slots below. */
        const void* state = r0.ctx.buf_ptr(couple_state_buf_);
        if (!state || serial_done_ > (int)drafter_.depth) return RAD_E_STATE;
        const int64_t pitch = couple_rows_ * (int64_t)sizeof(RadMtpCoupleState);
        RAD_TRY(rad_memcpy_2d_async(couple_readback_, pitch, state, pitch,
                                    (int64_t)n * (int64_t)sizeof(RadMtpCoupleState), serial_done_,
                                    r0.ctx.stream()));
    }
    RAD_TRY(rad_memcpy_async(slots_host_, r0.step_slots,
                             (int64_t)cfg_.max_seqs * (int64_t)sizeof(StepSlot), r0.ctx.stream()));
    return RAD_OK;
}

/* After commit: check the host's acceptance against the card's, publish the proposals through
 * set_draft, and move the history watermark. The card's acceptance positioned every draft pass,
 * so a disagreement is a defect and the step is refused. */
int Engine::finish_serial_drafter(const StepOut& so) {
    if (!serial_drafting() || serial_rounds_ <= 0) return RAD_OK;
    const StepPlan& plan = sched_.plan();
    if (serial_done_ > 0 && serial_rows_ > 0) {
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (so.first_row[i] < 0) continue;
            const StepSlot& S = slots_host_[e.card];   /* in range: the builder refuses others */
            if (S.acc != so.accepted[i] || S.tok[0] != so.token[i]) {
                RAD_ERR("drafter '%s': the card accepted %d draft(s) ending in token %d for "
                        "sequence %llu and the host %d ending in %d; the draft passes ran from "
                        "the card's", drafter_.name, S.acc, S.tok[0],
                        (unsigned long long)e.seq, so.accepted[i], so.token[i]);
                return RAD_E_STATE;
            }
        }
    }
    sched_.serial_end(serial_done_);
    if (serial_done_ <= 0 || serial_rows_ <= 0) return RAD_OK;

    Rank& r0 = *ranks_[0];
    const int rounds = serial_rounds_, done = serial_done_;
    draft_out_.assign((size_t)serial_rows_ * (size_t)rounds, 0);
    for (int i = 0; i < serial_rows_; ++i) {
        const StepSlot& S = slots_host_[serial_row_[(size_t)i].card];
        for (int rd = 1; rd <= done; ++rd)
            draft_out_[(size_t)i * (size_t)rounds + (size_t)(rd - 1)] = S.tok[rd];
    }
    for (int i = 0; i < serial_rows_; ++i) {
        /* No draft for a request the sampler will not speculate on (a constraining grammar that
         * cannot snapshot its state); the sampler would refuse the next step otherwise. */
        if (!r0.sampler.allows_speculation(serial_row_[(size_t)i].seq)) continue;
        const uint64_t seq = serial_row_[(size_t)i].seq;
        const int rc = sched_.set_draft(seq, &draft_out_[(size_t)i * (size_t)rounds], done);
        if (mtp_couple_)
            RAD_TRY(bind_couple(seq, i, done, rc == RAD_OK, serial_row_[(size_t)i].card));
    }
    return RAD_OK;
}

/* --draft-coupling, after the commit: bind drafting row `i`'s `done` states (its readback column)
 * to the positions the next build samples, and hand them to every rank's sampler.
 *
 * Bound only for a request still running: one this commit finished, or whose draft the scheduler
 * refused (`drafted` false), binds nothing, and set_mtp_couple(..., 0) clears an earlier step's
 * states (RAD_E_NOTFOUND for a request the sampler already dropped).
 *
 * The position is the committed output length after this commit, the n_out the next build_step
 * reads. It comes from the scheduler's commit snapshot, not the plan's Request pointer, which
 * reap() may already have freed (sched/mtp_couple_bind.h). */
int Engine::bind_couple(uint64_t seq, int i, int done, bool drafted, int32_t card) {
    MtpCoupleBindPoint bp;
    bp.found = sched_.committed_seq(seq, &bp.n_out, &bp.live);
    if (!bp.found)
        RAD_WARN("coupled draft: row for sequence %llu is not in the committed step; clearing its "
                 "coupling", (unsigned long long)seq);
    int n_bound = 0, n_malformed = 0;
    std::string why;
    const int bs = mtp_couple_bind_rounds(bp, couple_draw_stage_[(size_t)i],
                                          &couple_readback_[(size_t)i], (int)couple_rows_, done,
                                          slots_host_[card], meta_.n_vocab, couple_bound_,
                                          &n_bound, &why, &n_malformed);
    if (bs < 0) { RAD_ERR("%s", why.c_str()); return bs; }
    if (n_malformed > 0 && !couple_malformed_note_) {
        couple_malformed_note_ = true;
        RAD_WARN("coupled draft: sequence %llu: %d draft position(s) had a shortlist with no valid "
                 "Q (non-finite score, duplicate id or no live entry); those keep the greedy "
                 "proposal and the ordinary verify (logged once)",
                 (unsigned long long)seq, n_malformed);
    }
    for (auto& rk : ranks_) {
        const int st = rk->sampler.set_mtp_couple(seq, couple_bound_, drafted ? n_bound : 0);
        if (st < 0 && st != RAD_E_NOTFOUND) return st;
    }
    return RAD_OK;
}

/* One line, once, at the first coupled verify that completed. It reads the snapshot build_step
 * froze for the step whose wait just returned (sample/mtp_couple_evidence.h), which the
 * drafter's build_greedy does not clear, and checks that run() issued the verify for that build.
 * Sequences are re-checked as Running after the wait. Request numbers, the row count and the
 * kernel name only; a host-domain row is labelled as such. */
void Engine::couple_verify_evidence(int step) {
    if (!mtp_couple_ || couple_evidence_) return;
    Sampler& smp = ranks_[0]->sampler;
    const rad_op op = smp.staged_couple_op();
    const MtpCoupleEvidence ev = mtp_couple_evidence(
        smp.couple_build(), step, ranks_[0]->ctx.op_domain(op),
        [this](uint64_t seq) { return sched_.running(seq); });
    if (ev.rows <= 0 || smp.staged_couple_issued_step() != step) return;
    const char* kernel = ranks_[0]->ctx.ip_kernel_name(op);
    if (ev.device) {
        couple_evidence_ = true;
        RAD_INFO("MTP coupled verify completed: step %d, %lld bound row(s) over request(s) [%s], "
                 "kernel %s, domain device", step, (long long)ev.rows, ev.seqs.c_str(), kernel);
    } else if (!couple_host_note_) {
        couple_host_note_ = true;
        RAD_INFO("MTP coupled verify completed on the host domain: step %d, %lld bound row(s) over "
                 "request(s) [%s], kernel %s -- not device evidence", step, (long long)ev.rows,
                 ev.seqs.c_str(), kernel);
    }
}

/* --draft-coupling: the draw for every drafting row, before any draft pass reads it.
 *
 * The coordinate is the pre-commit output size, which strictly grows between two steps that draft
 * one request, so (base, round) never repeats a draft stream. The card is the slot the advance
 * writes; the propose op keeps the greedy proposal for a row whose draw names another card. The
 * rule is sched/mtp_couple_plan.h. */
template <class Row>
int Engine::stage_couple_draws(const Row* rows, int n) {
    std::string why;
    const int stg = mtp_couple_plan_stage(sched_.plan(),
                                          (int)(cfg_.max_seqs > 0 ? cfg_.max_seqs : 1),
                                          rows, n, &couple_by_card_, couple_draw_stage_, &why);
    if (stg < 0) {
        RAD_ERR("MTP coupled: %s", why.c_str());
        return stg;
    }
    for (auto& rk : ranks_) {
        void* dst = rk->ctx.buf_ptr(couple_draws_buf_);
        if (!dst) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(dst, couple_draw_stage_,
                                 n * (int64_t)sizeof(RadMtpCoupleDraw), rk->ctx.stream()));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the block driver
 *
 * RAD_DRAFT_BLOCK (block diffusion, e.g. DFlash2): one backbone pass over `block` positions (an
 * anchor and `block - 1` masked slots) proposes `depth` tokens at once, with no rounds. `block` is
 * the drafter's own: depth + 1 when the anchor row predicts nothing, depth when it does
 * (rad_builder.h). Two passes:
 *
 *   Context pass: mirrors the trunk step just run. The drafter projects the trunk's tapped hidden
 *   states and stores K and V for those positions in its own cache, over exactly the trunk's rows
 *   in order.
 *
 *   Query pass: the drafter's layers over the block fill every masked position. Its own selector
 *   writes the tokens to the plugin's `proposal` buffer, which is why RAD_DRAFT_BLOCK requires
 *   one.
 *
 * Proposals go out through set_draft() like every other.
 *
 * Both passes are queued behind the step before its readback, from the card's acceptance, as the
 * serial driver's are (sched/advance.h), so the next step can be issued ahead of the commit.
 */

/* One drafter pass through the barrier. Neither pass samples (no logits in the context pass; the
 * query pass has its own selector), so the sampler chain is zero rows. */
int Engine::block_pass(const RadBatch* db, const char* what, int step) {
    for (auto& rk : ranks_) {
        int s = rk->sampler.build_greedy(0);
        if (s < 0) {
            RAD_ERR("drafter '%s': build_greedy: %s", drafter_.name, rad_strerror(s));
            return s;
        }
    }
    issue_what_.store(kIssueAll, std::memory_order_release);
    step_batch_.store(db, std::memory_order_release);
    barrier_->arrive_and_wait();
    barrier_->arrive_and_wait();
    const int s = step_status_.load(std::memory_order_acquire);
    if (s < 0)
        RAD_ERR("drafter '%s': %s pass of step %d failed: %s -- the engine cannot continue",
                drafter_.name, what, step, rad_strerror(s));
    /* Probe after each pass; by the step loop's dump these buffers hold stale values. */
    if (s >= 0) dump_buffers(ranks_[0]->ctx, step, what);
    return s;
}

bool Engine::block_drafting() const {
    return drafter_.kind == RAD_DRAFT_BLOCK && drafter_.depth > 0;
}

/* Both passes behind the step, without waiting. After the trunk's ranks join, the acceptance runs
 * on every rank (ADV_ACCEPT), leaving each sequence's anchor and committed end in its StepSlot. The context pass mirrors the trunk step, whose positions the host knows.
 * The query pass's rows are device rows that the card positions at the committed end (ADV_BLOCK),
 * and its proposals go into the slots behind the anchor (ADV_BTAKE): exactly the token row the
 * next verify reads (ADV_TRUNK), so the step issued ahead verifies them with no host round trip.
 * collect_step waits for the sampled ids alone, so the commit streams the tokens while the card
 * runs both passes; the step loop waits for them (step_end_) before finish_block_drafter reads the
 * slots. */
void Engine::block_shape(int* block, int* steps) {
    *steps = drafter_.depth;
    /* The row count is the drafter's (defaulted and checked by declare_drafter). */
    *block = drafter_.block > 0 ? (int)drafter_.block : drafter_.depth + 1;
    /* Its short block under load (rad_drafter_short_block): the same anchor convention, so as many
     * fewer proposals as rows. The load is the step just run's, which the next step, the one these
     * proposals are for, mostly repeats (Scheduler::short_block_on). */
    if (drafter_short_ <= 0) return;
    short_on_ = Scheduler::short_block_on(short_on_, sched_.decoded_seqs(), short_from_);
    if (short_on_) {
        /* A tree (rad_drafter_tree) grows a tree of its own over the shorter pass, at most its
         * draft positions of nodes (DFlash2 attends across its block, so the short pass is not the
         * full pass's prefix); the verify takes that many. */
        if (tree_) *steps = std::min(*steps, (int)drafter_short_ - 1);
        else       *steps -= *block - (int)drafter_short_;
        *block  = (int)drafter_short_;
    }
}

int Engine::issue_block_drafter(int step, StepOut& so) {
    block_rows_ = 0;
    if (!block_drafting()) return RAD_OK;

    RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
    /* A tree's accepted path into place in the trunk's cache, against the same build (the step
     * just accepted), before any later pass reads it. */
    if (tree_) RAD_TRY(tree_compact(-1));
    /* The sampled ids before either pass, as the serial driver reads them. */
    RAD_TRY(collect_issue(so, /*fence=*/true));

    /* The short block too (block_shape): its query pass is a shorter batch the card positions
     * the same way, its proposals the front of each row at the drafter's pitch (ADV_BTAKE), and
     * the step issued ahead verifies that many (issue_ahead). */
    int block = 0, steps = 0;
    block_shape(&block, &steps);
    block_steps_ = steps;
    block_row_.resize((size_t)cfg_.max_seqs);
    /* The head reads the block's draft rows. A tree (rad_drafter_tree; DFlash2's convention, the
     * anchor predicts nothing) reads all block - 1 of them whatever its node count. */
    const int head = tree_ ? block - 1 : steps;
    const int n = sched_.block_begin(block, head, (int)drafter_.window, drafter_.mask_token,
                                     block_row_.data(), (int)block_row_.size(), /*on_card=*/true);

    /* --draft-coupling: the draws before the query pass reads them (stage_couple_draws). */
    if (mtp_couple_ && n > 0) RAD_TRY(stage_couple_draws(block_row_.data(), n));

    /* The context pass runs even when nothing drafts: a mid-prompt sequence has hidden states the
     * drafter will need, and skipping would leave a permanent hole in its cache. A tree verify's
     * context rows store only the accepted path, at its committed positions (ADV_CTXTREE). */
    if (const RadBatch* cb = sched_.block_context(step)) {
        if (tree_) RAD_TRY(advance_all(ADV_CTXTREE, 0, false));
        RAD_TRY(block_pass(cb, "context", step));
    }
    /* A snapshot carrying the drafter's window is saved now, with the step's context rows stored
     * and before the query pass and the step issued ahead (save_checkpoints). */
    if (drafter_snap_) RAD_TRY(save_checkpoints());
    if (n <= 0) return RAD_OK;
    const RadBatch* qb = sched_.block_query(step);
    if (!qb) return RAD_OK;
    RAD_TRY(advance_all(ADV_BLOCK, 0, false));
    RAD_TRY(block_pass(qb, "query", step));
    RAD_TRY(advance_all(ADV_BTAKE, steps, true, n));
    Rank& r0 = *ranks_[0];
    if (mtp_couple_) {
        /* Positions 1..steps of the drafting rows, on the stream of the slots below. */
        const void* state = r0.ctx.buf_ptr(couple_state_buf_);
        if (!state || steps > (int)drafter_.depth) return RAD_E_STATE;
        const int64_t pitch = couple_rows_ * (int64_t)sizeof(RadMtpCoupleState);
        RAD_TRY(rad_memcpy_2d_async(couple_readback_, pitch, state, pitch,
                                    (int64_t)n * (int64_t)sizeof(RadMtpCoupleState), steps,
                                    r0.ctx.stream()));
    }
    RAD_TRY(rad_memcpy_async(slots_host_, r0.step_slots,
                             (int64_t)cfg_.max_seqs * (int64_t)sizeof(StepSlot), r0.ctx.stream()));
    block_rows_ = n;
    return RAD_OK;
}

/* After commit: check the host's acceptance against the card's, which positioned the query pass
 * (a disagreement is a defect and the step is refused, as in finish_serial_drafter), move the
 * drafter's filled range, and publish each row's proposal from the slots. */
int Engine::finish_block_drafter(const StepOut& so) {
    if (!block_drafting()) return RAD_OK;
    if (block_rows_ > 0) {
        const StepPlan& plan = sched_.plan();
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (so.first_row[i] < 0) continue;
            const StepSlot& S = slots_host_[e.card];   /* in range: the builder refuses others */
            if (S.acc != so.accepted[i] || S.tok[0] != so.token[i]) {
                RAD_ERR("drafter '%s': the card accepted %d draft(s) ending in token %d for "
                        "sequence %llu and the host %d ending in %d; the query pass ran from the "
                        "card's", drafter_.name, S.acc, S.tok[0], (unsigned long long)e.seq,
                        so.accepted[i], so.token[i]);
                sched_.block_end();
                return RAD_E_STATE;
            }
        }
    }
    sched_.block_end();
    if (block_rows_ <= 0) return RAD_OK;

    Rank& r0 = *ranks_[0];
    const int steps = block_steps_;
    draft_out_.assign((size_t)block_rows_ * (size_t)steps, 0);
    for (int i = 0; i < block_rows_; ++i) {
        const StepSlot& S = slots_host_[block_row_[(size_t)i].card];
        for (int j = 0; j < steps; ++j)
            draft_out_[(size_t)i * (size_t)steps + (size_t)j] = S.tok[1 + j];
    }
    /* -v: what each block pass proposed, a line a sequence, so two runs can be compared draft by
     * draft where acceptance alone cannot tell them apart (a drafter whose context is wrong still
     * drafts, and its rejected drafts change no token). */
    const bool show = (int)log_level() >= (int)Log::Debug;
    for (int i = 0; i < block_rows_; ++i) {
        const int32_t* d = &draft_out_[(size_t)i * (size_t)steps];
        if (show) {
            std::string t;
            for (int j = 0; j < steps; ++j) t += fmt(" %d", (int)d[j]);
            RAD_DEBUG("drafter '%s': sequence %llu proposes%s", drafter_.name,
                      (unsigned long long)block_row_[(size_t)i].seq, t.c_str());
        }
        /* As in finish_serial_drafter: the step issued ahead only carries a request whose answer
         * is fixed (Scheduler::step_next_locked), so this agrees with the count it verifies. A
         * tree's parents come from the same slot, as ADV_BTAKE took them. */
        if (!r0.sampler.allows_speculation(block_row_[(size_t)i].seq)) continue;
        const uint64_t seq = block_row_[(size_t)i].seq;
        const StepSlot& S = slots_host_[block_row_[(size_t)i].card];
        const int rc = sched_.set_draft(seq, d, steps, tree_ ? S.par : nullptr);
        if (mtp_couple_)
            RAD_TRY(bind_couple(seq, i, steps, rc == RAD_OK, block_row_[(size_t)i].card));
    }
    return RAD_OK;
}

}  /* namespace rad */
