/* lora/declare.cpp -- see declare.h. */
#include "lora/declare.h"
#include "build/rad_build.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>

namespace rad {

namespace {

std::string join(const lora::TargetSet& t) {
    std::string s;
    for (const std::string& x : t) s += (s.empty() ? "" : ",") + x;
    return s.empty() ? std::string("(none)") : s;
}

int opd_index(const OpInfo& o, const char* name) {
    if (!o.schema || !o.schema->operands) return -1;
    for (int k = 0; k < o.schema->n_operands; ++k)
        if (o.schema->operands[k].name && std::strcmp(o.schema->operands[k].name, name) == 0) return k;
    return -1;
}

bool is_coded(uint32_t dt) {
    return dt == RAD_F8E4M3 || dt == RAD_F8E5M2 || dt == RAD_I8;
}

/* The block of a scale stream of `blocks` groups over `ext` elements, as the kernel libraries
 * read one (libref/ref_gemm.h's grid_block): the power of two whose ceiling division gives
 * `blocks`, else the plain ceiling; one group is the whole extent. */
int64_t grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

/* The scale grid of a coded activation the GEMM `reads`: the read set's one f32 buffer beside
 * the codes (reads[0]), [rows, groups] or one scale a row, over the codes' full width. The
 * GEMM's own `group` is the weight's (an MXFP4 weight's 32-wide exponent block reads an
 * activation scaled per 128), and a GEMM may read a column span of the activation, so neither
 * its `group` nor its K gives the activation's grid. 0 when the stream is not one f32 buffer. */
int64_t activation_grid(const Program& P, const std::vector<rad_buf>& reads) {
    const RadBufDecl& a = P.buffers[reads[0]].decl;
    const RadBufDecl* s = nullptr;
    for (size_t i = 1; i < reads.size(); ++i) {
        const RadBufDecl& d = P.buffers[reads[i]].decl;
        if (d.dtype != RAD_F32) continue;
        if (s) return 0;
        s = &d;
    }
    if (!s || a.rank < 1 || s->rank < 1) return 0;
    const int64_t width = a.shape[a.rank - 1];
    const int64_t groups = s->rank >= 2 ? s->shape[s->rank - 1] : 1;
    return groups > 0 ? grid_block(width, groups) : 0;
}

rad_buf decl_persist(Builder& b, const std::string& name, uint32_t dtype, uint32_t rank,
                     const int64_t* shape) {
    RadBufDecl d{};
    d.dtype = dtype;
    d.rank = rank;
    for (uint32_t i = 0; i < rank; ++i) d.shape[i] = shape[i];
    d.kind = RAD_BUF_PERSIST;
    d.domain = RAD_DOMAIN_DEVICE;
    return b.decl_buffer(name.c_str(), &d);
}

}  /* namespace */

/* A generator pass's adapter ops run over at most this many rows at a time: large enough that the
 * kernels' row tiles are all busy, small enough that `t` for the largest rank is a few MiB. */
constexpr int64_t kGenChunk = 8192;

int lora_declare(Builder& b, const LoraDeclare& d) {
    if (!d.cfg || !d.cfg->enabled()) return RAD_OK;
    Program& P = b.program();
    /* Adapters are chosen per request and applied on the token steps of a model that writes
     * logits, or on every pass of a generator that names its adapter keys (rad_lora_root); an
     * embedding model's requests carry no adapter. */
    const bool gen = P.is_pass_driven();
    if (gen ? P.lora_roots.empty()
            : (P.pooled_buf != RAD_NULL_HANDLE || P.logits_buf == RAD_NULL_HANDLE)) {
        RAD_ERR("lora: runtime LoRA adapters serve a model that generates text, or a generator "
                "that declares how adapter files name its modules, and this one %s",
                gen                                 ? "is a generator that declares no adapter "
                                                      "key names (rad_lora_root)"
                : P.pooled_buf != RAD_NULL_HANDLE ? "makes embeddings"
                                                    : "declares no logits");
        return RAD_E_UNSUPPORTED;
    }
    if (!gen && d.targets.count("*")) {
        RAD_ERR("lora: an adapter without adapter_config.json names no target modules, and a "
                "text model has no default set; give them with --lora-target-modules");
        return RAD_E_UNSUPPORTED;
    }
    const int     S = d.cfg->slots();
    const int     max_rank = d.cfg->rank(gen);
    const int64_t max_tok = P.ctx.max_tok;
    if (S <= 0 || max_rank <= 0 || (!gen && max_tok <= 0)) {
        RAD_ERR("lora: %d slots of rank %d over steps of %lld tokens is not a configuration", S,
                max_rank, (long long)max_tok);
        return RAD_E_INVAL;
    }

    /* Every reader of a targeted weight carries the delta, or startup refuses: a weight some op
     * multiplies without the adapter would be the base in one place and the adapter in another. */
    std::vector<lora::UnsupportedUse> bad;
    int s = lora::find_unsupported_uses(P, d.targets, &bad);
    if (s < 0) return s;
    if (!bad.empty()) {
        for (const lora::UnsupportedUse& u : bad)
            RAD_ERR("lora: weight '%s' (%s) is read by op %u '%s', which cannot carry an adapter: %s",
                    u.weight.c_str(), u.module.c_str(), u.op_index, u.op.c_str(), u.why.c_str());
        RAD_ERR("lora: narrow --lora-target-modules (now %s) to modules every reader of which is a "
                "plain GEMM", join(d.targets).c_str());
        return RAD_E_UNSUPPORTED;
    }

    std::vector<lora::Site>  sites;
    std::vector<std::string> pool_weights;
    std::string              err;
    s = lora::build_sites(P, d.targets, max_rank, &sites, &err, &pool_weights);
    if (s < 0) {
        RAD_ERR("lora: %s", err.c_str());
        return s;
    }

    /* ---- the pools: one per targeted weight, [S, R, K] / [S, N, R] bf16 and [S, N] f32 */
    P.lora_pools.clear();
    P.lora_pools.resize(pool_weights.size());
    for (size_t p = 0; p < pool_weights.size(); ++p) {
        const lora::Site* first = nullptr;
        for (const lora::Site& x : sites)
            if (x.pool == (int32_t)p) { first = &x; break; }
        if (!first) {
            RAD_ERR("lora: pool %zu ('%s') has no site", p, pool_weights[p].c_str());
            return RAD_E_STATE;
        }
        for (const lora::Site& x : sites)
            if (x.pool == (int32_t)p && (x.R != first->R || x.K_r != first->K_r || x.N_r != first->N_r)) {
                RAD_ERR("lora: two GEMMs over '%s' disagree about its geometry (R %lld/%lld, K %lld/%lld, "
                        "N %lld/%lld)", pool_weights[p].c_str(), (long long)x.R, (long long)first->R,
                        (long long)x.K_r, (long long)first->K_r, (long long)x.N_r,
                        (long long)first->N_r);
                return RAD_E_STATE;
            }
        LoraPool& lp = P.lora_pools[p];
        lp.weight = pool_weights[p];
        lp.R = first->R; lp.K = first->K_r; lp.N = first->N_r; lp.S = S;
        const int64_t sa[3] = { S, lp.R, lp.K }, sb[3] = { S, lp.N, lp.R }, ss[2] = { S, lp.N };
        lp.a     = decl_persist(b, "lora." + lp.weight + ".a", RAD_BF16, 3, sa);
        lp.b     = decl_persist(b, "lora." + lp.weight + ".b", RAD_BF16, 3, sb);
        lp.scale = decl_persist(b, "lora." + lp.weight + ".scale", RAD_F32, 2, ss);
        if (!lp.a || !lp.b || !lp.scale) {
            RAD_ERR("lora: could not declare the pools of '%s'", lp.weight.c_str());
            return RAD_E_STATE;
        }
    }

    /* ---- a token step's rows are its tokens; a generator pass's are each GEMM's own, issued in
     * chunks of at most kGenChunk rows (Ctx::issue_lora), so `t` holds one chunk. */
    int64_t rows = max_tok;
    if (gen) {
        rows = 0;
        for (const lora::Site& x : sites)
            if (x.gemm > 0 && x.gemm < P.ops.size()) rows = std::max(rows, P.ops[x.gemm].range_hi);
        rows = std::min<int64_t>(rows, kGenChunk);
    }

    /* On the real declare, what the adapters cost: the pools are persistent buffers, so the
     * arena plan (and with it what is left for weights and the KV cache) holds them. */
    if (d.sites_out) {
        int64_t slot_bytes = 0;
        for (const LoraPool& p : P.lora_pools) slot_bytes += 2 * p.R * p.K + 2 * p.N * p.R + 4 * p.N;
        RAD_INFO("lora: %zu weights carry adapters, %d slots of rank %d, %s a slot on this rank, "
                 "%s in all (--max-loras, --max-lora-rank)", P.lora_pools.size(), S, max_rank,
                 humanb(slot_bytes).c_str(), humanb(slot_bytes * S).c_str());
    }

    /* ---- t = A.x, one f32 plane per capacity R and lane. Persistent, not transient: the ops that
     * use it run at the GEMM's position, but the buffer plan would place a transient at the
     * appended ops' own position, the end of the program, and share its bytes with buffers live
     * mid-program. */
    std::map<int64_t, int32_t> t_by_r;
    P.lora_t.clear();
    for (const lora::Site& x : sites) {
        if (t_by_r.count(x.R)) continue;
        LoraScratch sc;
        sc.R = x.R;
        const int64_t sh[2] = { rows, x.R };
        for (int l = 0; l < 2; ++l) {
            sc.lane[l] = decl_persist(b, fmt("lora.t.r%lld.lane%d", (long long)x.R, l), RAD_F32, 2, sh);
            if (!sc.lane[l]) return RAD_E_STATE;
        }
        t_by_r[x.R] = (int32_t)P.lora_t.size();
        P.lora_t.push_back(sc);
    }

    /* ---- a generator pass's slot operand: elements [s rows, (s + 1) rows) of an [S rows] i32
     * buffer, every one s (filled when the engine starts adapters) */
    P.lora_rows = 0;
    P.lora_rows_n = 0;
    if (gen && rows > 0) {
        const int64_t sh[1] = { S * rows };
        P.lora_rows = decl_persist(b, "lora.slot_rows", RAD_I32, 1, sh);
        if (!P.lora_rows) return RAD_E_STATE;
        P.lora_rows_n = rows;
    }

    /* ---- the two ops behind each targeted GEMM */
    P.lora_sites.clear();
    int n_refused = 0;
    std::vector<char> dropped(sites.size(), 0);
    for (lora::Site& x : sites) {
        const uint32_t gh = x.gemm;
        if (gh == 0 || gh >= P.ops.size()) return RAD_E_STATE;
        if (!b.op_resolved(gh)) continue;   /* never handed to the plugin, so never issued */
        /* Copied out first: decl_op below appends to P.ops, which may move every OpInfo. */
        const OpInfo&  g = P.ops[gh];
        const std::string gop = g.op;
        const std::string rkey = g.ranged_key;
        const int64_t  lo = g.range_lo, hi = g.range_hi;
        long long gk = 0, gn = 0;
        const bool     has_k = g.base.get_i("K", &gk), has_n = g.base.get_i("N", &gn);
        const int      oa = opd_index(g, "a"), oas = opd_index(g, "a_scale"), oy = opd_index(g, "y");
        const std::vector<rad_buf> reads = b.op_read_set(gh), writes = b.op_write_set(gh);

        /* A text model's targeted GEMM that cannot carry the ops is refused; a generator's, whose
         * every plain GEMM is a candidate, stays the base and its modules are reported unplaced. */
        auto refuse = [&](const std::string& why) {
            if (gen) {
                RAD_WARN("lora: op %u '%s' over '%s' carries no adapter: %s", gh, gop.c_str(),
                         x.weight.c_str(), why.c_str());
                dropped[(size_t)(&x - sites.data())] = 1;
                return;
            }
            RAD_ERR("lora: op %u '%s' over '%s': %s", gh, gop.c_str(), x.weight.c_str(), why.c_str());
            ++n_refused;
        };
        if (rkey != "M") { refuse("its row count is not the ranged parameter M"); continue; }
        if (!has_k || !has_n || gk != x.K_r || gn != x.N_r) {
            refuse(fmt("declared K %lld, N %lld; the weight's per-rank slice is K %lld, N %lld", gk, gn,
                       (long long)x.K_r, (long long)x.N_r));
            continue;
        }
        if (oa < 0 || oy < 0) { refuse("its schema names no `a` or no `y` operand"); continue; }
        if (reads.empty() || writes.empty()) {
            refuse("it declared no read or write set (rad_op_reads / rad_op_writes), so the dtypes "
                   "of its activation and output are not known at declare");
            continue;
        }
        const uint32_t a_dt = P.buffers[reads[0]].decl.dtype;
        const uint32_t y_dt = P.buffers[writes[0]].decl.dtype;
        const bool coded = is_coded(a_dt);
        if (coded && oas < 0) { refuse("a coded activation and no `a_scale` operand"); continue; }
        const int64_t a_grid = coded ? activation_grid(P, reads) : 0;
        if (coded && a_grid <= 0) {
            refuse("a coded activation whose read set holds no single f32 scale stream beside the "
                   "codes, so the grid lora_shrink reads it at is not known at declare");
            continue;
        }

        RadParam sp[6];
        int n_sp = 0;
        sp[n_sp++] = RAD_RANGE("M", lo, hi);
        sp[n_sp++] = RAD_INT("K", x.K_r);
        sp[n_sp++] = RAD_INT("R", x.R);
        sp[n_sp++] = RAD_INT("S", S);
        if (coded) sp[n_sp++] = RAD_INT("group", a_grid);
        sp[n_sp++] = RAD_STR("dtype", rad_dtype_name(a_dt));
        const rad_op hs = b.decl_op("lora_shrink", sp, n_sp, nullptr, 0);

        RadParam ep[5] = { RAD_RANGE("M", lo, hi), RAD_INT("N", x.N_r), RAD_INT("R", x.R),
                           RAD_INT("S", S), RAD_STR("dtype", rad_dtype_name(y_dt)) };
        /* A generator's passes are many rows of one adapter: its expand runs on the WMMA
         * (lora_expand_mm) where a library has it. A token step keeps the exact row, whose bytes
         * do not depend on the batch. */
        const char* expand_op = gen && b.op_declared("lora_expand_mm") ? "lora_expand_mm"
                                                                        : "lora_expand";
        const rad_op he = b.decl_op(expand_op, ep, 5, nullptr, 0);
        if (!hs || !he) {
            refuse(fmt("no kernel resolves lora_%s for an activation of %s and an output of %s",
                       !hs ? "shrink" : "expand", rad_dtype_name(a_dt), rad_dtype_name(y_dt)));
            continue;
        }

        LoraIssueSite is;
        /* A generator's slot packs its ranks from 0 (lora::pack_modules, compact): pairs that read
         * the first half and quarter of R, so a rank-16 adapter in a slot sized 64 a module costs
         * a quarter. Multiples of 16, the WMMA's step. */
        if (gen) {
            int64_t prev = x.R;
            for (int v = 0; v < 2; ++v) {
                const int64_t rv = ((x.R >> (v + 1)) + 15) / 16 * 16;
                if (rv < 16 || rv >= prev) break;
                RadParam vs[6];
                for (int i = 0; i < n_sp; ++i) vs[i] = sp[i];
                vs[2] = RAD_INT("R", rv);
                ep[2] = RAD_INT("R", rv);
                const rad_op vs_op = b.decl_op("lora_shrink", vs, n_sp, nullptr, 0);
                const rad_op ve_op = b.decl_op(expand_op, ep, 5, nullptr, 0);
                if (!vs_op || !ve_op) break;
                P.ops[vs_op].anchor = (int32_t)gh;
                P.ops[ve_op].anchor = (int32_t)gh;
                is.shrink_v[v] = vs_op; is.expand_v[v] = ve_op; is.R_v[v] = rv;
                is.n_v = (int8_t)(v + 1);
                prev = rv;
            }
        }
        is.gemm = gh; is.shrink = hs; is.expand = he;
        is.opd_a = (int8_t)oa;
        is.opd_a_scale = (int8_t)(coded ? oas : -1);
        is.opd_y = (int8_t)oy;
        is.a_dtype = a_dt; is.y_dtype = y_dt;
        is.pool = x.pool;
        is.t = t_by_r[x.R];
        is.R = x.R; is.K = x.K_r; is.N = x.N_r;
        P.ops[hs].anchor = (int32_t)gh;
        P.ops[he].anchor = (int32_t)gh;
        P.ops[gh].lora_site = (int32_t)P.lora_sites.size();
        P.lora_sites.push_back(is);
    }
    if (n_refused) return RAD_E_UNSUPPORTED;

    if (gen) P.lora_rused.assign(P.lora_pools.size() * (size_t)S, 0);
    if (d.sites_out) {
        std::vector<lora::Site> kept;
        for (size_t i = 0; i < sites.size(); ++i)
            if (!dropped[i]) kept.push_back(std::move(sites[i]));
        *d.sites_out = std::move(kept);
    }
    return RAD_OK;
}

}  /* namespace rad */
