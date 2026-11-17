/* share.cpp -- which bytes of a container entry one rank's declaration takes. See share.h. */
#include "share.h"
#include "encoding.h"

namespace rad {

int64_t PlaneShare::dense_bytes() const { return rows * rad_enc_row_bytes(dtype, cols); }

int weight_share(const WeightInfo& w, const RadEncoding& enc, uint32_t rank, const int64_t* shape,
                 int world, int index, PlaneShare out[RAD_ENC_MAX_PLANES], std::string* why) {
    auto refuse = [&](const std::string& m) {
        if (why) *why = m;
        return RAD_E_SHAPE;
    };
    if (w.n_sel < 1 || w.n_sel > RAD_ENC_MAX_PLANES) return refuse("no planes selected");
    int64_t R = 0, C = 0;
    rad_enc_view(rank, shape, &R, &C);

    /* The shard is cut in the view the declaration uses: one plane's own extents, or the logical
     * ones for several planes. That is where the plugin declared its share and stacked parts; a
     * block-fp8 scale plane's parts are 1/128 of its weight's. */
    const bool single = w.n_sel == 1;
    const bool whole = world <= 1 || w.decl.shard == RAD_SHARD_NONE;
    /* A reshape: one whole plane of single elements (a byte or more each) declared under another
     * shape with the same element count, which decl_weight admits. The row-major bytes are
     * identical, so the entry takes the declared shape. A shard would be a different cut and a
     * sub-byte row pads, so neither is a reshape. */
    if (single && whole) {
        const RadEncPlane& p0 = enc.plane[w.sel[0]];
        if (p0.kind == RAD_PLANE_TILED && p0.block[0] == 1 && p0.block[1] == 1 &&
            rad_dtype_bits(p0.dtype) >= 8 && R * C == w.sel_rows[0] * w.sel_cols[0]) {
            R = w.sel_rows[0];
            C = w.sel_cols[0];
        }
    }
    int64_t VR = R, VC = C;
    if (single) rad_enc_plane_dims(&enc.plane[w.sel[0]], R, C, &VR, &VC);

    struct VRect { int64_t r0, r1, c0, c1; };
    std::vector<VRect> view;
    const bool span  = w.shard_hi > 0 || w.n_part_spans > 0;
    if (whole) {
        view.push_back({ 0, VR, 0, VC });
    } else {
        /* A sharded weight must be a matrix; anything else is refused rather than guessing which
         * dimension a rank owns, with two exceptions. A rank-1 weight sharded by ROW is a
         * contiguous slice of a per-head vector, which is a column cut in the [1, n] view. A
         * weight whose inner dimensions are all 1 is the matrix of its first and last (a
         * depthwise conv kernel [C, 1, K] is [C, K]). */
        uint32_t mat = rank;
        for (uint32_t i = 1; i + 1 < rank; ++i)
            if (shape[i] == 1) --mat;
        if (mat > 2 || (rank == 1 && w.decl.shard == RAD_SHARD_COL))
            return refuse(fmt("a %s shard of a rank-%u weight is not defined; declare it "
                              "RAD_SHARD_NONE", w.decl.shard == RAD_SHARD_ROW ? "ROW" : "COL",
                              rank));
        const bool cols = w.decl.shard == RAD_SHARD_COL || rank == 1;
        const int np1 = rank == 1 && w.decl.n_row_parts > 1 ? w.decl.n_row_parts : 1;
        if (cols && np1 > 1) {
            /* A rank-1 weight of stacked parts (a fused projection's bias, [value | gate]) is cut
             * as the matrix it goes with: this rank's slice of each part, never a contiguous
             * slice of the stack. Per-part spans and declared parts read as the row cut's do. */
            if (w.n_part_spans > 0) {
                if (w.n_part_spans != np1)
                    return refuse(fmt("%d per-part spans for %d stacked parts", w.n_part_spans, np1));
                int64_t sum = 0, soff = 0;
                for (int g = 0; g < np1; ++g) sum += w.part_total[g];
                if (sum != VC)
                    return refuse(fmt("the %d stacked parts' stated elements sum to %lld and the "
                                      "entry has %lld", np1, (long long)sum, (long long)VC));
                for (int g = 0; g < np1; ++g) {
                    view.push_back({ 0, VR, soff + w.part_lo[g], soff + w.part_hi[g] });
                    soff += w.part_total[g];
                }
            } else {
                if (span)
                    return refuse("a single span over a rank-1 weight of stacked parts is not "
                                  "defined; state one span a part");
                if (VC % world) return refuse(fmt("%lld elements do not divide by %d ranks",
                                                  (long long)VC, world));
                int64_t sum = 0, soff = 0;
                for (int g = 0; g < np1; ++g) sum += w.decl.row_parts[g];
                if (sum != VC / world)
                    return refuse(fmt("%d declared parts sum to %lld, and the rank's share of %lld "
                                      "elements at %d ranks is %lld", np1, (long long)sum,
                                      (long long)VC, world, (long long)(VC / world)));
                for (int g = 0; g < np1; ++g) {
                    const int64_t take = w.decl.row_parts[g];
                    view.push_back({ 0, VR, soff + index * take, soff + (index + 1) * take });
                    soff += take * world;
                }
            }
        } else if (cols) {
            int64_t c0 = w.shard_lo, c1 = w.shard_hi;
            if (!span) {
                if (VC % world) return refuse(fmt("%lld columns do not divide by %d ranks",
                                                  (long long)VC, world));
                c0 = (int64_t)index * (VC / world);
                c1 = c0 + VC / world;
            }
            if (c1 > VC) return refuse(fmt("the column span [%lld, %lld) is outside %lld",
                                           (long long)c0, (long long)c1, (long long)VC));
            view.push_back({ 0, VR, c0, c1 });
        } else {
            /* Contiguous within each stacked part. The weight may be several projections
             * concatenated along dim 0 ([gate|up], the delta net's [q|k|v]), and a rank owns a
             * slice of each, not a contiguous slice of the stack. Declared parts are this rank's
             * rows, so a part has `world` times as many rows in the entry. */
            const int np = w.decl.n_row_parts > 1 ? w.decl.n_row_parts : 1;
            if (w.n_part_spans > 0) {
                /* Per-part spans: the plugin states each stacked part's rows in the container
                 * and this rank's [lo, hi) of it, so parts may differ in size. */
                if (w.n_part_spans != np)
                    return refuse(fmt("%d per-part spans for %d stacked parts", w.n_part_spans, np));
                int64_t sum = 0, soff = 0;
                for (int g = 0; g < np; ++g) sum += w.part_total[g];
                if (sum != VR)
                    return refuse(fmt("the %d stacked parts' stated rows sum to %lld and the entry "
                                      "has %lld", np, (long long)sum, (long long)VR));
                for (int g = 0; g < np; ++g) {
                    view.push_back({ soff + w.part_lo[g], soff + w.part_hi[g], 0, VC });
                    soff += w.part_total[g];
                }
            } else if (span) {
                if (VR % np) return refuse(fmt("%lld rows do not divide into %d parts",
                                               (long long)VR, np));
                const int64_t cpart = VR / np;
                if (w.shard_hi > cpart)
                    return refuse(fmt("the row span [%lld, %lld) of each of %d part(s) does not "
                                      "fit the entry's %lld rows", (long long)w.shard_lo,
                                      (long long)w.shard_hi, np, (long long)VR));
                for (int g = 0; g < np; ++g)
                    view.push_back({ g * cpart + w.shard_lo, g * cpart + w.shard_hi, 0, VC });
            } else {
                if (VR % world) return refuse(fmt("%lld rows do not divide by %d ranks",
                                                  (long long)VR, world));
                const int64_t per = VR / world;
                if (np == 1) {
                    view.push_back({ index * per, (index + 1) * per, 0, VC });
                } else {
                    int64_t sum = 0;
                    for (int g = 0; g < np; ++g) sum += w.decl.row_parts[g];
                    if (sum != per)
                        return refuse(fmt("%d declared row parts sum to %lld, and the rank's share "
                                          "of %lld rows at %d ranks is %lld", np, (long long)sum,
                                          (long long)VR, world, (long long)per));
                    int64_t soff = 0;
                    for (int g = 0; g < np; ++g) {
                        const int64_t take = w.decl.row_parts[g];
                        view.push_back({ soff + index * take, soff + (index + 1) * take, 0, VC });
                        soff += take * world;
                    }
                }
            }
        }
    }

    /* The view's rectangles come from plugin-stated spans and parts, sum-checked only. A single
     * selected plane takes them as its own rectangles below, with no enc_plane_rect to bound
     * them, so every one is held to the entry here. */
    for (int g = 0; !whole && g < w.n_part_spans; ++g)
        if (w.part_lo[g] < 0 || w.part_lo[g] > w.part_hi[g] || w.part_hi[g] > w.part_total[g])
            return refuse(fmt("part %d's span [%lld, %lld) is outside its %lld", g,
                              (long long)w.part_lo[g], (long long)w.part_hi[g],
                              (long long)w.part_total[g]));
    for (const VRect& q : view)
        if (q.r0 < 0 || q.r0 > q.r1 || q.r1 > VR || q.c0 < 0 || q.c0 > q.c1 || q.c1 > VC)
            return refuse(fmt("the slice [%lld, %lld) x [%lld, %lld) is outside the entry's "
                              "%lld x %lld", (long long)q.r0, (long long)q.r1, (long long)q.c0,
                              (long long)q.c1, (long long)VR, (long long)VC));

    for (int k = 0; k < w.n_sel; ++k) {
        const RadEncPlane& p = enc.plane[w.sel[k]];
        PlaneShare& s = out[k];
        s = PlaneShare{};
        s.dtype = p.dtype;
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&p, R, C, &pr, &pc);
        s.row_bytes = rad_enc_row_bytes(p.dtype, pc);
        const int bits = rad_dtype_bits(p.dtype) > 0 ? rad_dtype_bits(p.dtype)
                                                     : 8 * (int)rad_dtype_bytes(p.dtype, 1);
        PlaneRect last{ -1, -1, -1, -1 };
        for (const VRect& q : view) {
            PlaneRect pq;
            if (single) {
                pq = PlaneRect{ q.r0, q.r1 - q.r0, q.c0, q.c1 - q.c0 };
            } else {
                std::string m;
                if (enc_plane_rect(p, R, C, q.r0, q.r1, q.c0, q.c1, &pq, &m) != RAD_OK)
                    return refuse(m);
            }
            /* A plane every slice carries whole (a table, a per-tensor scale) yields the same
             * rectangle for every stacked part; take it once. */
            if (pq.row0 == last.row0 && pq.rows == last.rows && pq.col0 == last.col0 &&
                pq.cols == last.cols)
                continue;
            last = pq;
            /* A share is copied as bytes, so both column ends must fall on a byte boundary. */
            if ((pq.col0 * bits) % 8 || ((pq.col0 + pq.cols) * bits % 8 && pq.col0 + pq.cols != pc))
                return refuse(fmt("columns [%lld, %lld) of the %s '%s' plane do not fall on whole "
                                  "bytes", (long long)pq.col0, (long long)(pq.col0 + pq.cols),
                                  rad_dtype_name(p.dtype), p.role));
            /* A vector's stacked parts lie end to end as one row, so they may differ in width (a
             * Mamba convolution's bias is [x | B | C]); a matrix's parts stack as rows. */
            const bool vec = rank == 1 && pq.rows == 1;
            if (!vec && !s.rects.empty() && pq.cols != s.cols)
                return refuse("the share's parts differ in width");
            PlaneShare::Rect r;
            r.row0  = pq.row0;
            r.rows  = pq.rows;
            r.byte0 = pq.col0 * bits / 8;
            r.bytes = rad_enc_row_bytes(p.dtype, pq.cols);
            s.rects.push_back(r);
            if (vec) {
                s.rows = 1;
                s.cols += pq.cols;
            } else {
                s.rows += pq.rows;
                s.cols = pq.cols;
            }
        }
        /* Laid end to end, a vector's parts are the declared vector, so only the element count
         * has to agree. */
        const bool fits = rank == 1 ? s.rows * s.cols == w.sel_rows[k] * w.sel_cols[k]
                                    : s.rows == w.sel_rows[k] && s.cols == w.sel_cols[k];
        if (!fits)
            return refuse(fmt("this rank's share of the '%s' plane is [%lld, %lld] and the "
                              "declaration says [%lld, %lld]", p.role, (long long)s.rows,
                              (long long)s.cols, (long long)w.sel_rows[k],
                              (long long)w.sel_cols[k]));
    }
    return RAD_OK;
}

}  /* namespace rad */
