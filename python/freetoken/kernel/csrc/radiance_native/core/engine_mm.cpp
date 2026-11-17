/* engine_mm.cpp -- the encoder driver (spec §11).
 *
 * The scheduler holds back a request whose media is not encoded (Scheduler::media_ready) and hands
 * its items here. They are encoded between steps, never inside one: a pass runs with no step in
 * flight, is read back, and the item is marked done, so admission finds its rows in host memory and
 * the batch builder copies them in with the chunk that reaches them. The step path never waits on
 * the tower.
 *
 * A pass holds whole segments: an image is one segment, a video one per temporal group, and the
 * encoder attends within a segment (the processor keeps each within a pass). Several items share a
 * pass when they fit. An audio clip is one segment of the audio encoder, whose passes are separate.
 *
 * Every rank runs the whole tower from its own copy of the patches: every rank embeds every token,
 * so this costs compute per card instead of a transfer between them.
 */
#include "engine_priv.h"

#include "mm/processor.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>

namespace rad {

int Engine::run_encodes() {
    std::vector<std::shared_ptr<mm::Item>> items;
    sched_.take_encodes(&items);
    if (items.empty()) return RAD_OK;
    const auto t0 = std::chrono::steady_clock::now();

    /* Diagnostic: RADIANCE_DEBUG_MM_DUMP=<dir> writes each item's patches (`.pix.bf16`, processor
     * output in merge-block order) and rows (`.bf16`) as raw bf16, named by content key, for
     * comparison against a reference encoder. */
    static const char* const dump_dir = std::getenv("RADIANCE_DEBUG_MM_DUMP");
    auto dump = [&](const mm::Item& it, const char* ext, const uint16_t* p, size_t n) {
        char path[512];
        std::snprintf(path, sizeof path, "%s/mm_%016llx%016llx_%dx%dx%d%s", dump_dir,
                      (unsigned long long)it.key_hi, (unsigned long long)it.key_lo, it.grid_t,
                      it.grid_h, it.grid_w, ext);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(p, sizeof(uint16_t), n, f);
            std::fclose(f);
            RAD_INFO("encoder: wrote %s", path);
        }
    };

    Rank& r0 = *ranks_[0];
    const Program& prog = r0.program;

    /* Diagnostic: RADIANCE_DEBUG_MM_BUFFERS names encoder buffers (comma separated, with an
     * optional `:rows` cap) and each pass writes them, as raw bytes, to `<dump dir>/buf.<name>`.
     * The pass has finished and been read back, so the copy sees each buffer's last writer; the
     * tower's per-stage intermediates are tapped inside the pass instead (rad_block_sam.h). */
    struct DumpBuf { std::string name; int64_t rows; };
    std::vector<DumpBuf> dump_bufs;
    static const char* const dump_list = std::getenv("RADIANCE_DEBUG_MM_BUFFERS");
    if (dump_dir && *dump_dir && dump_list) {
        for (const char* s = dump_list; *s; ) {
            const char* e = s;
            while (*e && *e != ',') ++e;
            std::string want(s, (size_t)(e - s));
            s = *e ? e + 1 : e;
            int64_t rows = 0;
            if (const size_t colon = want.find(':'); colon != std::string::npos) {
                rows = std::strtoll(want.c_str() + colon + 1, nullptr, 10);
                want.resize(colon);
            }
            if (!want.empty()) dump_bufs.push_back({ want, rows });
        }
    }
    auto answers_to = [](const std::string& name, const std::string& want) {
        if (name.size() < want.size() ||
            name.compare(name.size() - want.size(), want.size(), want) != 0) return false;
        return name.size() == want.size() || name[name.size() - want.size() - 1] == '.';
    };
    for (auto& ip : items) {
        if (ip->encoded) continue;
        const RadEncoderDecl& e = ip->kind == mm::Kind::Audio ? prog.audio_encoder : prog.encoder;
        ip->embd.assign((size_t)(ip->n_rows() * e.n_embd), 0);
        if (dump_dir && *dump_dir) dump(*ip, ".pix.bf16", ip->pixels.data(), ip->pixels.size());
    }

    /* Passes per encoder: audio through the audio encoder, pictures and video through the vision
     * encoder, each with its own pass size and output buffer. */
    for (int which = 0; which < 2; ++which) {
        const bool audio = which == 1;
        const RadEncoderDecl& enc = audio ? prog.audio_encoder : prog.encoder;
        const std::string& enc_name = audio ? prog.audio_encoder_name : prog.encoder_name;
        std::vector<mm::Item*> mine;
        for (auto& ip : items)
            if (!ip->encoded && (ip->kind == mm::Kind::Audio) == audio) mine.push_back(ip.get());
        if (mine.empty()) continue;
        if (enc.modalities == 0) {
            RAD_ERR("encoder: a %s item reached the encoder and this program declares no %s "
                    "encoder", audio ? "audio" : "picture", audio ? "audio" : "image");
            return RAD_E_STATE;
        }
        const int64_t max_p = enc.max_patches;
        const int64_t n_embd = enc.n_embd;

        size_t i = 0;
        int32_t seg = 0;
        /* The pass's grid, when the encoder declares that a pass is one geometry
         * (RadEncoderDecl::one_grid). */
        const bool one_grid = enc.one_grid != 0;
        int32_t seg_grid = 0;
        while (i < mine.size()) {
            std::vector<BatchBuilder::EncoderPart> parts;
            int64_t np = 0;
            while (i < mine.size()) {
                mm::Item& it = *mine[i];
                const int32_t first = seg;
                for (; seg < it.n_seg; ++seg) {
                    const int32_t g = it.seg_h(seg);
                    /* The one-output-row view separator (core/mm marks it explicitly) belongs to
                     * the pass of the view before it, so it neither breaks at a "grid change" nor
                     * sets the pass's grid. A square segment that is a real view (a chunked line
                     * of one tile) keeps the ordinary geometry rule below. */
                    const bool sep = it.seg_is_sep(seg);
                    const int64_t seg_p = it.patch_at(seg + 1) - it.patch_at(seg);
                    if (seg_p > max_p) {
                        RAD_ERR("encoder: a segment of %lld patches does not fit a pass of %lld",
                                (long long)seg_p, (long long)max_p);
                        return RAD_E_STATE;
                    }
                    if (one_grid && !sep && np > 0 && seg_grid != 0 && g != seg_grid) break;
                    if (np + seg_p > max_p) break;
                    np += seg_p;
                    if (!sep) seg_grid = g;
                }
                if (seg == first) {
                    /* Nothing of this item packed into the pass: its first segment did not fit
                     * a pass that already has content (a segment larger than the whole pass was
                     * refused above). `seg` addresses a real segment only below `it.n_seg`;
                     * a degenerate zero-segment item leaves seg_grid alone. */
                    if (seg_grid == 0 && seg < it.n_seg) seg_grid = it.seg_h(seg);
                    break;
                }
                parts.push_back({ &it, first, seg - first });
                if (seg == it.n_seg) { ++i; seg = 0; }
            }
            if (parts.empty()) break;

            const RadBatch* b = sched_.build_encoder(parts, enc_step_++, enc);
            if (!b) return RAD_E_STATE;
            RAD_TRY(prepare_release(b));
            /* Read the output address after prepare_release: a pass runs at the arena's full plan
             * (Engine::arena_step), and a buffer's offset differs per level. */
            void* out = r0.ctx.buf_ptr(enc.out);
            if (!out) {
                RAD_ERR("encoder '%s': its output buffer was declared but never planned",
                        enc_name.c_str());
                return RAD_E_STATE;
            }
            /* kIssueModel: the pass only, with no sampler and no tiering tick. */
            issue_what_.store(kIssueModel, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            barrier_->arrive_and_wait();
            barrier_->arrive_and_wait();
            const int st = step_status_.load(std::memory_order_acquire);
            if (st < 0) {
                RAD_ERR("encoder pass failed: %s -- the engine cannot continue", rad_strerror(st));
                return st;
            }
            /* The rows, in pass order: each part's segments are consecutive rows of its item. */
            int64_t row = 0;
            for (const BatchBuilder::EncoderPart& p : parts) {
                mm::Item& it = *const_cast<mm::Item*>(p.item);
                const int64_t first = it.row_at(p.seg0);
                const int64_t rows = it.row_at(p.seg0 + p.n_seg) - first;
                const int64_t bytes = rows * n_embd * (int64_t)sizeof(uint16_t);
                RAD_TRY(rad_memcpy_async(it.embd.data() + first * n_embd,
                                         static_cast<const char*>(out) +
                                             row * n_embd * (int64_t)sizeof(uint16_t),
                                         bytes, r0.ctx.stream()));
                row += rows;
            }
            /* Sync every rank's stream: the next pass rewrites every rank's staging and arena. */
            for (auto& rk : ranks_) RAD_TRY(rad_stream_sync(rk->ctx.stream()));

            if (dump_dir && *dump_dir && !dump_bufs.empty()) {
                for (const DumpBuf& d : dump_bufs) {
                    for (size_t bi = 1; bi < prog.buffers.size(); ++bi) {
                        const BufferInfo& info = prog.buffers[bi];
                        if (!answers_to(info.name, d.name) || info.bytes <= 0) continue;
                        const int64_t nrow = info.decl.rank >= 1 ? info.decl.shape[0] : 1;
                        const int64_t rowb = nrow > 0 ? info.bytes / nrow : info.bytes;
                        const int64_t rows = d.rows > 0 ? std::min<int64_t>(d.rows, nrow) : nrow;
                        void* ptr = r0.ctx.buf_ptr((rad_buf)bi);
                        std::vector<char> raw((size_t)(rows * rowb));
                        if (!ptr || rad_memcpy_async(raw.data(), ptr, raw.size(),
                                                     r0.ctx.stream()) < 0)
                            continue;
                        rad_stream_sync(r0.ctx.stream());
                        char path[512];
                        std::snprintf(path, sizeof path, "%s/mm_%016llx%016llx_buf.%s.raw",
                                      dump_dir, (unsigned long long)parts.front().item->key_hi,
                                      (unsigned long long)parts.front().item->key_lo,
                                      info.name.c_str());
                        if (FILE* f = std::fopen(path, "wb")) {
                            std::fwrite(raw.data(), 1, raw.size(), f);
                            std::fclose(f);
                            RAD_INFO("encoder: wrote %s (%lld rows of %lld bytes)", path,
                                     (long long)rows, (long long)rowb);
                        }
                    }
                }
            }
        }
    }

    for (auto& ip : items) {
        mm::Item& it = *ip;
        if (it.encoded) continue;
        it.encoded = true;
        it.pixels.clear();
        it.pixels.shrink_to_fit();
        if (dump_dir && *dump_dir) dump(it, ".bf16", it.embd.data(), it.embd.size());
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    RAD_DEBUG("encoder: %zu item(s) encoded in %.1f ms", items.size(), ms);
    return RAD_OK;
}

}  /* namespace rad */
