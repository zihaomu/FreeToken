/* item.cpp -- the parts of Item the step path uses. The batch builder links only this unit from
 * core/mm, so a test that fakes the device layer does not pull in the processor and, through its
 * metadata reader, the real device layer. */
#include "mm/processor.h"

#include <algorithm>

namespace rad {
namespace mm {

void Item::coords(int64_t first, int64_t count, int32_t* planes) const {
    /* An audio clip is 1-D (RadBatch::enc): frame index, whether the frame takes part in the
     * convolution, the clip's real frame count and its staged frame count. A clip staged in
     * chunks (AudioFamily::Qwen3ASR) reads each chunk's first `chunk_read` frames; the rest of the
     * chunk is alignment the encoder reads as zero. */
    if (kind == Kind::Audio) {
        for (int64_t k = 0; k < count; ++k) {
            const int64_t f = first + k;
            planes[k]             = (int32_t)f;
            planes[count + k]     = chunk_staged > 0 ? (f % chunk_staged < chunk_read ? 1 : 0)
                                                     : (f < valid ? 1 : 0);
            planes[2 * count + k] = valid;
            planes[3 * count + k] = grid_h;
        }
        return;
    }
    /* Merge-block order inverted: a patch's index within its segment is
     * ((block_row * blocks_w + block_col) * merge + in_row) * merge + in_col. */
    const int32_t merge = grid_h / rows_h;
    int32_t s = 0;
    while (s + 1 < n_seg && patch_at(s + 1) <= first) ++s;
    for (int64_t k = 0; k < count; ++k) {
        while (s + 1 < n_seg && patch_at(s + 1) <= first + k) ++s;
        int64_t within = first + k - patch_at(s);
        if (within < seg_prefix) {
            /* A row the tower fills itself: row -1, the column naming which. */
            planes[k]             = -1;
            planes[count + k]     = (int32_t)within;
            planes[2 * count + k] = seg_h(s);
            planes[3 * count + k] = seg_w(s);
            continue;
        }
        within -= seg_prefix;
        const int64_t bw = seg_w(s) / merge;
        int64_t row, col;
        if (patch_raster) {
            /* Raster order is its own inverse: patch i is row i / w, column i % w. */
            row = within / seg_w(s);
            col = within % seg_w(s);
        } else {
            const int64_t in_col = within % merge;
            const int64_t in_row = (within / merge) % merge;
            const int64_t blk = within / (merge * merge);
            row = (blk / bw) * merge + in_row;
            col = (blk % bw) * merge + in_col;
        }
        planes[k]             = (int32_t)row;
        planes[count + k]     = (int32_t)col;
        planes[2 * count + k] = seg_h(s);
        planes[3 * count + k] = seg_w(s);
    }
}

/* The rotary layout of one prompt: a run a segment, the text in front of a run counting up and a
 * grid advancing the rotary by its longer side. Here, beside the accessors, so that code without
 * the processor has it. */
void PromptMedia::layout() {
    rope.runs.clear();
    int64_t pos = 0, tok = 0;
    for (size_t pi = 0; pi < items.size(); ++pi) {
        const Placed& p = items[pi];
        const Item& it = *p.item;
        if (!it.row_tokens.empty()) {
            /* Explicit image placeholders form flat runs separated by ordinary vocabulary
             * tokens. The scheduler already maps each run through row0; separators get no row. */
            for (size_t first = 0; first < it.row_tokens.size();) {
                size_t end = first + 1;
                while (end < it.row_tokens.size() &&
                       it.row_tokens[end] == it.row_tokens[end - 1] + 1)
                    ++end;
                const int64_t run_tok = (int64_t)p.tok + it.row_tokens[first];
                pos += run_tok - tok;
                RopeLayout::Run r;
                r.tok = (int32_t)run_tok;
                r.n = (int32_t)(end - first);
                r.w = r.n;
                r.base = (int32_t)pos;
                r.flat = true;
                r.item = (int32_t)pi;
                r.row0 = (int32_t)first;
                pos += r.n;
                tok = run_tok + r.n;
                r.shift = (int32_t)(pos - tok);
                rope.runs.push_back(r);
                first = end;
            }
            continue;
        }
        const int32_t m = it.rows_h > 0 ? it.grid_h / it.rows_h : 1;
        for (int32_t s = 0; s < it.n_seg; ++s) {
            const int64_t run_tok = (int64_t)p.tok + it.seg_at[(size_t)s];
            pos += run_tok - tok;            /* the text in front of the run counts up */
            const int32_t rh = it.segments.empty() ? it.rows_h : it.seg_h(s) / m;
            const int32_t rw = it.segments.empty() ? it.rows_w : it.seg_w(s) / m;
            RopeLayout::Run r;
            r.tok = (int32_t)run_tok;
            r.n = (int32_t)(it.row_at(s + 1) - it.row_at(s));
            r.w = rw;
            r.base = (int32_t)pos;
            r.flat = it.rope_1d;
            /* A grid advances the rotary by its longer side; a 1-D run by its length. */
            pos += it.rope_1d ? r.n : std::max(rh, rw);
            tok = run_tok + r.n;
            r.shift = (int32_t)(pos - tok);
            r.item = (int32_t)pi;
            r.row0 = (int32_t)it.row_at(s);
            rope.runs.push_back(r);
        }
    }
}

}  /* namespace mm */
}  /* namespace rad */
