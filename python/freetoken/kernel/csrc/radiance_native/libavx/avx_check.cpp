/* avx_check.cpp -- rad-avx-check: a host kernel library against libref, every op, every ISA level,
 * no engine.
 *
 * rad-kbench needs rad_core, a Geometry, the plugin loader and sometimes a recorded fixture, i.e.
 * building the engine. This is the same comparison with only two dlopen calls, a geometry table and
 * libref's operand descriptions. Neither plugin is linked, so the .so under test is the one the
 * engine would load.
 *
 * Checks, in order:
 *
 *   1. Schema agreement. The loader refuses a whole plugin on any schema disagreement, which would
 *      make every row unreachable. libavx's schema table is a hand-kept copy of libref's
 *      (avx_registry.cpp), checked here with the loader's rule: parameter key/type/requiredness
 *      and operand role/optionality, by position; names are not compared.
 *
 *   2. Coverage. libavx must implement every op libref implements. An absent row and a row that
 *      returns RAD_E_UNSUPPORTED are reported separately and neither passes; this catches kernels
 *      built but never wired.
 *
 *   3. The numbers. Both plugins get byte-identical inputs and every OUT and INOUT operand is
 *      compared. The finiteness pattern is compared first (a NaN on one side only is a failure;
 *      `rel > tol` is false for NaN), and ids exactly. Every buffer carries guard bytes on both
 *      sides, and every input must be unchanged after the launch. A case libref cannot describe
 *      or run fails: the table is then wrong. Hand-built cases cover operand shapes a description
 *      cannot draw (avx_check_hand.cpp).
 *
 *   4. Threads. No check geometry reaches the kernels' parallel thresholds, so every case runs a
 *      second time with each parallel region forced to several threads (avx_par_force) and must
 *      reproduce the serial run byte for byte.
 *
 *   5. Every ISA level, each forced in turn through avx_isa_force (an exported symbol, not an env
 *      variable). A level the CPU cannot execute is reported as not executable and not counted.
 *
 * Another host library is checked the same way at the one level it selects for itself, with the
 * row the selector would resolve each geometry to; the ops it does not implement are listed rather
 * than failed (libavx/README.md, "Checking another host library").
 */
#include "avx_check.h"
#include "avx_isa.h"

#include <cinttypes>

/* ================================================================== comparison */
Err compare(const float* got, const float* ref, int64_t n) {
    Err e;
    e.n = n;
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        const bool fg = std::isfinite(got[i]), fr = std::isfinite(ref[i]);
        if (fg != fr) { ++e.nonfinite_gap; continue; }
        if (!fg) continue;
        const double d = (double)got[i] - (double)ref[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d));
        const double sc = std::fabs((double)ref[i]);
        if (sc > 1e-30) e.max_rel = std::max(e.max_rel, std::fabs(d) / sc);
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    e.rel_l2 = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    return e;
}

bool same_values(uint32_t dt, const void* ref, const void* got, int64_t n, double tol,
                 double* rel, std::string* why) {
    char buf[256];
    if (dt == RAD_I32 || dt == RAD_U32 || dt == RAD_I64) {
        int64_t bad = 0, first = -1;
        for (int64_t i = 0; i < n; ++i)
            if (avx::ld_int_dt(dt, ref, i) != avx::ld_int_dt(dt, got, i)) {
                if (first < 0) first = i;
                ++bad;
            }
        *rel = bad ? 1.0 : 0.0;
        if (!bad) return true;
        std::snprintf(buf, sizeof buf, "%lld of %lld ids differ, first at %lld: %lld where libref "
                      "has %lld", (long long)bad, (long long)n, (long long)first,
                      (long long)avx::ld_int_dt(dt, got, first),
                      (long long)avx::ld_int_dt(dt, ref, first));
        *why = buf;
        return false;
    }
    std::vector<float> fr((size_t)n), fa((size_t)n);
    h_widen(ref, dt, n, fr.data());
    h_widen(got, dt, n, fa.data());
    const Err e = compare(fa.data(), fr.data(), n);
    *rel = e.rel_l2;
    if (e.nonfinite_gap == 0 && e.rel_l2 <= tol) return true;
    std::snprintf(buf, sizeof buf, "rel_l2=%.3e max_abs=%.3e nonfinite_gap=%lld tol=%.1e",
                  e.rel_l2, e.max_abs, (long long)e.nonfinite_gap, tol);
    *why = buf;
    return false;
}

bool parallel_rerun(const Lib& lib, int threads, const std::vector<uint8_t*>& outs,
                    const std::vector<size_t>& bytes, const std::function<void()>& reset,
                    const std::function<int()>& launch, std::string* why) {
    std::vector<std::vector<uint8_t>> serial(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) serial[i].assign(outs[i], outs[i] + bytes[i]);
    reset();
    lib.par_force(threads);
    const int rc = launch();
    lib.par_force(0);
    char buf[160];
    if (rc < 0) {
        std::snprintf(buf, sizeof buf, "on %d threads it returned %d", threads, rc);
        *why = buf;
        return false;
    }
    for (size_t i = 0; i < outs.size(); ++i) {
        if (std::memcmp(outs[i], serial[i].data(), bytes[i]) == 0) continue;
        size_t at = 0;
        while (at < bytes[i] && outs[i][at] == serial[i][at]) ++at;
        std::snprintf(buf, sizeof buf, "on %d threads output buffer %zu differs from the serial "
                      "run from byte %zu of %zu", threads, i, at, bytes[i]);
        *why = buf;
        return false;
    }
    return true;
}

/* rad-kbench's tolerance table: rel_l2 bounds for the same arithmetic in a different summation
 * order, which is what a vectorised reduction is. A wrong kernel (e.g. a transposed operand) lands
 * far outside these. First match wins. */
struct Tol { const char* op; const char* dtype; double rel_l2; };
static const Tol kTol[] = {
    { "gemm_nt",      "bf16",     8e-3 },
    { "gemm_nt",      "f16",      4e-3 },
    { "gemm_nt",      "f32",      2e-5 },
    { "gemm_nt",      "*",        3e-2 },
    { "logits_gemm",  "bf16",     8e-3 },
    { "logits_gemm",  "f32",      2e-5 },
    /* The plain grouped GEMM is gemm_nt's arithmetic per expert, held to its bounds. */
    { "moe_gemm",     "f32",      2e-5 },
    { "moe_gemm",     "bf16",     8e-3 },
    { "moe_gemm",     "*",        3e-2 },
    { "lora_shrink",  "*",        2e-5 },
    { "attn_",        "bf16",     2e-2 },
    { "attn_",        "f32",      1e-4 },
    { "attn_",        "*",        3e-2 },
    { "rmsnorm",      "bf16",     4e-3 },
    { "rmsnorm",      "f32",      1e-6 },
    { "layernorm",    "f32",      1e-6 },
    { "softmax",      "f32",      1e-6 },
    { "quant_act",    "*",        1e-6 },
    { "dequant",      "*",        1e-6 },
    { "cast",         "*",        1e-6 },
    /* An fp8 cache stores x / scale narrowed once; both libraries narrow the same quotient. */
    { "kv_store",     "fp8_e4m3", 1e-6 },
    { "all_reduce",   "f32",      1e-6 },
    { "all_reduce",   "*",        4e-3 },
    { "all_gather",   "*",        0.0  },
    { "*",            "f32",      1e-5 },
    { "*",            "bf16",     6e-3 },
    { "*",            "f16",      3e-3 },
    { "*",            "*",        2e-2 },
};

static double tolerance_for(const std::string& op, const char* dtype) {
    const char* dt = dtype ? dtype : "*";
    for (const Tol& t : kTol) {
        const bool om = std::strcmp(t.op, "*") == 0 || op.rfind(t.op, 0) == 0;
        const bool dm = std::strcmp(t.dtype, "*") == 0 || std::strcmp(t.dtype, dt) == 0;
        if (om && dm) return t.rel_l2;
    }
    return 2e-2;
}

struct Stats {
    int checked = 0, passed = 0, failed = 0, skipped = 0, declined = 0, missing = 0;
    int unselected = 0, rerun = 0;
    double worst = 0;
    std::string worst_where;
};

/* After a launch on one side: both guards of every operand and of the scratch intact, and every
 * input operand as drawn. Prints each violation and returns false. */
static bool intact(const Prepared& pr, bool side_a, const char* who, const Case& c) {
    bool ok = true;
    auto name = [&](size_t i) {
        return pr.schema->operands[i].name ? pr.schema->operands[i].name : "?";
    };
    for (size_t i = 0; i < pr.opd.size(); ++i) {
        const Opd& o = pr.opd[i];
        if (o.absent) continue;
        const GBuf& g = side_a ? o.a : o.b;
        if (const int hit = g.guard_hit()) {
            std::printf("  FAIL %-20s %s\n       %s wrote %s operand %d (%s)\n", c.op,
                        case_str(c).c_str(), who, hit < 0 ? "before the start of" : "past the end of",
                        (int)i, name(i));
            ok = false;
        }
        const bool input = o.role == RAD_OPD_IN || o.role == RAD_OPD_WEIGHT ||
                           o.role == RAD_OPD_WTAB;
        if (input && g.size() && std::memcmp(g.data(), o.master.data(), g.size()) != 0) {
            std::printf("  FAIL %-20s %s\n       %s changed its input operand %d (%s)\n", c.op,
                        case_str(c).c_str(), who, (int)i, name(i));
            ok = false;
        }
    }
    if ((side_a ? pr.sa : pr.sb).guard_hit()) {
        std::printf("  FAIL %-20s %s\n       %s wrote outside its scratch\n", c.op,
                    case_str(c).c_str(), who);
        ok = false;
    }
    return ok;
}

static void run_case(const Check& ck, const Case& c, Stats* st) {
    const Lib& ref = *ck.ref;
    const Lib& lib = *ck.lib;
    const char* who = lib.info->name;
    auto skip = [&](const char* why) {
        ++st->skipped;
        std::printf("  SKIP %-20s %s\n       %s\n", c.op, case_str(c).c_str(), why);
    };
    const RadKernelInfo* kr = ref.row(c.op);
    if (!kr) { skip("libref has no row for the op"); return; }
    std::vector<RadParam> params;
    for (const P& p : c.p) params.push_back(h_mk(p));
    const RadKernelInfo* ka = lib.select(c.op, params.data(), (int)params.size());
    if (!ka) {
        if (lib.host_row(c.op)) {
            ++st->unselected;
            if (ck.verbose)
                std::printf("  %-22s %-46s no row's constraints admit it\n", c.op,
                            case_str(c).c_str());
        } else {
            ++st->missing;
        }
        return;
    }

    /* Side A is libref, side B the library under test; operands are sized by the oracle's
     * description. A case libref cannot describe or run fails as a skip: the table is wrong. */
    Prepared pr = h_prepare(ref, kr, c, kr, ka);
    if (pr.skipped || !pr.ok) { skip("libref's opd_shape declines this geometry"); return; }

    const int rr = kr->launch(&pr.aa, nullptr);
    const int ra = ka->launch(&pr.ab, nullptr);

    /* Check the oracle first: if libref declines, there is nothing to compare, even if the library
     * declined too. A decline only counts when libref answered. */
    if (rr < 0) {
        char why[64];
        std::snprintf(why, sizeof why, "libref's launch returned %d", rr);
        skip(why);
        return;
    }
    if (ra == RAD_E_UNSUPPORTED) {
        ++st->declined;
        std::printf("  %-22s %-46s DECLINED\n", c.op, case_str(c).c_str());
        return;
    }
    ++st->checked;
    if (ra < 0) {
        std::printf("  FAIL %-20s %s\n       %s returned %d where libref returned %d\n", c.op,
                    case_str(c).c_str(), who, ra, rr);
        ++st->failed;
        return;
    }

    bool ok = intact(pr, true, "libref", c);
    ok = intact(pr, false, who, c) && ok;
    double worst = 0;
    const double tol = ck.tol_override >= 0 ? ck.tol_override : tolerance_for(c.op, case_dtype(c));
    for (size_t i = 0; i < pr.opd.size(); ++i) {
        const Opd& o = pr.opd[i];
        if (o.absent || o.n <= 0) continue;
        if (o.role != RAD_OPD_OUT && o.role != RAD_OPD_INOUT) continue;
        double rel = 0;
        std::string why;
        if (!same_values(o.d.dtype, o.a.data(), o.b.data(), o.n, tol, &rel, &why)) {
            ok = false;
            std::printf("  FAIL %-20s %s\n       operand %d (%s): %s\n", c.op, case_str(c).c_str(),
                        (int)i, pr.schema->operands[i].name ? pr.schema->operands[i].name : "?",
                        why.c_str());
        }
        worst = std::max(worst, rel);
    }
    /* A launch that left every output's values as drawn compares the draw with itself: the case
     * proves nothing about the arithmetic (all-zero weights, or a stage its parameters switch
     * off). -0 counts as 0, so a product of zeros does not pass for work. */
    bool wrote = false;
    for (const Opd& o : pr.opd) {
        if (o.absent || o.n <= 0 || (o.role != RAD_OPD_OUT && o.role != RAD_OPD_INOUT)) continue;
        std::vector<float> now((size_t)o.n), was((size_t)o.n);
        h_widen(o.a.data(), o.d.dtype, o.n, now.data());
        h_widen(o.master.data(), o.d.dtype, o.n, was.data());
        for (int64_t i = 0; i < o.n && !wrote; ++i)
            wrote = !(now[(size_t)i] == was[(size_t)i]);
    }
    if (!wrote) {
        ok = false;
        std::printf("  FAIL %-20s %s\n       libref's launch changed no output byte, so the "
                    "comparison is vacuous\n", c.op, case_str(c).c_str());
    }

    /* The same launch with every parallel region on several threads, against the serial bytes. */
    if (ok && ck.threads > 0 && lib.par_force) {
        std::vector<uint8_t*> outs;
        std::vector<size_t> bytes;
        for (Opd& o : pr.opd)
            if (!o.absent && o.n > 0 && (o.role == RAD_OPD_OUT || o.role == RAD_OPD_INOUT)) {
                outs.push_back(o.b.data());
                bytes.push_back(o.b.size());
            }
        auto reset = [&] {
            for (Opd& o : pr.opd)
                if (!o.absent && o.n > 0 && (o.role == RAD_OPD_OUT || o.role == RAD_OPD_INOUT))
                    std::memcpy(o.b.data(), o.master.data(), o.master.size());
            if (!pr.sb.empty()) std::memset(pr.sb.data(), 0, pr.sb.size());
        };
        std::string why;
        if (!parallel_rerun(lib, ck.threads, outs, bytes, reset,
                            [&] { return ka->launch(&pr.ab, nullptr); }, &why)) {
            ok = false;
            std::printf("  FAIL %-20s %s\n       %s\n", c.op, case_str(c).c_str(), why.c_str());
        }
        ok = intact(pr, false, who, c) && ok;
        ++st->rerun;
    }

    if (ok) ++st->passed; else ++st->failed;
    if (worst > st->worst) {
        st->worst = worst;
        st->worst_where = std::string(c.op) + " " + case_str(c);
    }
    if (ck.verbose)
        std::printf("  %-22s %-46s rel_l2=%.3e tol=%.1e %s\n", c.op, case_str(c).c_str(), worst,
                    tol, ok ? "ok" : "FAIL");
}

/* ================================================================== zero rows
 *
 * docs/OPS.md: an op issued over zero rows returns RAD_E_SHAPE and writes nothing. Each op's first
 * case is described at its row count and at twice it; every extent that changes is a row extent
 * and is set to zero, and both plugins are launched with the row parameter at 0 (a cumulative
 * length draws as all zeros, every sequence empty). libref answering anything else means the
 * construction does not fit the op, which is reported and not counted. */
static bool zero_row_cases(const Check& ck, const std::vector<Case>& cases) {
    const Lib& ref = *ck.ref;
    const Lib& lib = *ck.lib;
    std::vector<std::string> seen;
    int checked = 0, failed = 0, unfit = 0;
    for (const Case& c : cases) {
        if (c.speed_only || (!ck.only_op.empty() && ck.only_op != c.op)) continue;
        if (std::find(seen.begin(), seen.end(), c.op) != seen.end()) continue;
        seen.push_back(c.op);
        const char* key = nullptr;
        for (const char* k : { "M", "q_len", "numel" })
            for (const P& p : c.p)
                if (!key && p.kind == RAD_P_INT && !std::strcmp(p.k, k)) key = k;
        const RadKernelInfo* kr = ref.row(c.op);
        if (!key || !kr || !kr->opd_shape) continue;
        auto with = [&](long long r) {
            std::vector<RadParam> v;
            for (const P& p : c.p) {
                v.push_back(h_mk(p));
                if (!std::strcmp(p.k, key)) v.back().ival = r == 2 ? 2 * p.i : 0;
            }
            return v;
        };
        long long r1 = 0;
        for (const P& p : c.p) if (!std::strcmp(p.k, key)) r1 = p.i;
        std::vector<RadParam> p1, p2 = with(2), p0 = with(0);
        for (const P& p : c.p) p1.push_back(h_mk(p));
        const RadKernelInfo* ka = lib.select(c.op, p0.data(), (int)p0.size());
        const RadOpSchema* sch = ref.schema(c.op);
        if (!ka || !sch || r1 <= 0) continue;

        Prepared pr;
        pr.schema = sch;
        pr.opd.assign((size_t)sch->n_operands, Opd{});
        bool described = true;
        for (int i = 0; i < sch->n_operands && described; ++i) {
            RadOpdDesc d1{}, d2{};
            d1.idx_const = d2.idx_const = -1;
            described = kr->opd_shape(p1.data(), (int)p1.size(), i, &d1) >= 0 &&
                        kr->opd_shape(p2.data(), (int)p2.size(), i, &d2) >= 0;
            Opd& o = pr.opd[(size_t)i];
            o.d = d1;
            for (uint32_t k = 0; k < d1.rank; ++k)
                if (d1.shape[k] != d2.shape[k]) o.d.shape[k] = 0;
            o.absent = (d1.flags & RAD_OPD_F_ABSENT) != 0;
            o.role = sch->operands[i].role;
            o.n = o.absent ? 0 : desc_numel(o.d);
        }
        if (!described) continue;
        for (size_t k = 0; k < pr.opd.size(); ++k) h_draw(pr.opd, k, 0, 0x2e70u + k);
        for (Opd& o : pr.opd) {
            o.a.assign(o.master);
            o.b.assign(o.master);
        }
        pr.params = p0;
        h_build_tensors(pr.opd, true, &pr.ta);
        h_build_tensors(pr.opd, false, &pr.tb);
        pr.aa.t = pr.ta.data(); pr.aa.n_t = (int)pr.ta.size();
        pr.aa.p = pr.params.data(); pr.aa.n_p = (int)pr.params.size();
        pr.aa.world_size = 1;
        pr.ab = pr.aa;
        pr.ab.t = pr.tb.data();

        const int rr = kr->launch(&pr.aa, nullptr);
        if (rr != RAD_E_SHAPE) {
            ++unfit;
            if (ck.verbose)
                std::printf("  %-22s zero rows: libref returned %d; not counted\n", c.op, rr);
            continue;
        }
        const int ra = ka->launch(&pr.ab, nullptr);
        bool untouched = true;
        for (const Opd& o : pr.opd)
            untouched = untouched && o.b.guard_hit() == 0 &&
                        std::memcmp(o.b.data(), o.master.data(), o.master.size()) == 0;
        ++checked;
        if (ra != RAD_E_SHAPE || !untouched) {
            ++failed;
            std::printf("  FAIL %-20s zero rows (%s = 0)\n       %s returned %d%s; libref "
                        "refuses with RAD_E_SHAPE and writes nothing\n", c.op, key,
                        lib.info->name, ra, untouched ? "" : " and wrote");
        } else if (ck.verbose) {
            std::printf("  %-22s zero rows refused, nothing written\n", c.op);
        }
    }
    if (checked || unfit)
        std::printf("  zero rows: %d ops checked, %d failed%s\n", checked, failed,
                    unfit ? (", " + std::to_string(unfit) + " not reducible to zero rows").c_str()
                          : "");
    return failed == 0;
}

/* ================================================================== exhaustive conversion
 *
 * Drawn normals reach almost none of a narrow float's special patterns (NaN, infinity,
 * subnormals, negative zero), yet conversions branch on exactly those (e.g. cvt_fp8e4m3_to_f32's
 * subnormal and NaN handling). So every bit pattern of each 8- and 16-bit format is widened to f32
 * through both plugins' `cast` and compared bit for bit, NaNs as NaNs. The narrowing direction
 * cannot cover 2^32 inputs. It is fed every representable value (the widened patterns), then
 * every midpoint between two adjacent values with 1 and 2 f32 ulps to either side, where
 * round-to-nearest-even decides and a rounding error lands, then the overflow edges and every
 * class of NaN. */
static bool exhaustive_cast(const Check& ck) {
    const Lib& ref = *ck.ref;
    const Lib& lib = *ck.lib;
    const RadKernelInfo* kr = ref.row("cast");
    if (!kr || !lib.host_row("cast")) {
        std::printf("EXACT   cast: no row in %s\n", kr ? lib.info->name : "libref");
        return kr && !ck.own;
    }

    /* One cast of `n` elements through both plugins; the outputs compared byte for byte, element
     * by element. Returns false having printed the first difference. */
    auto both = [&](const char* label, uint32_t from, const char* fname, uint32_t to,
                    const char* tname, const void* src, int64_t n, std::vector<uint8_t>* out_a) {
        const size_t ob = dt_bytes(to, n), eb = (size_t)rad_dtype_bits(to) / 8;
        std::vector<uint8_t> ya(ob, 0), yb(ob, 0);
        RadParam p[4];
        p[0] = RadParam{ "M", RAD_P_INT, 1, 0, nullptr, 0 };
        p[1] = RadParam{ "n", RAD_P_INT, (long long)n, 0, nullptr, 0 };
        p[2] = RadParam{ "from", RAD_P_STR, 0, 0, fname, 0 };
        p[3] = RadParam{ "to", RAD_P_STR, 0, 0, tname, 0 };
        const RadKernelInfo* kb = lib.select("cast", p, 4);
        if (!kb) {
            std::printf("EXACT   cast %-16s %-10s no row of %s admits it\n", label, fname,
                        lib.info->name);
            return !ck.own;
        }
        auto mk_t = [&](const void* data, uint32_t dt) {
            RadTensor t{};
            t.data = const_cast<void*>(data); t.dtype = dt; t.rank = 2;
            t.shape[0] = 1; t.shape[1] = n;
            t.stride[0] = n; t.stride[1] = 1;
            return t;
        };
        RadTensor ta[2] = { mk_t(src, from), mk_t(ya.data(), to) };
        RadTensor tb[2] = { mk_t(src, from), mk_t(yb.data(), to) };
        RadArgs aa{}, ab{};
        aa.t = ta; aa.n_t = 2; aa.p = p; aa.n_p = 4; aa.world_size = 1;
        ab = aa; ab.t = tb;
        const int rr = kr->launch(&aa, nullptr), ra = kb->launch(&ab, nullptr);
        if (rr < 0 || ra < 0) {
            std::printf("EXACT   cast %-16s refused: libref %d, %s %d\n", label, rr,
                        lib.info->name, ra);
            return false;
        }
        /* Widened NaNs need only be NaNs: F16C quiets a signalling one, libref keeps it. */
        int64_t bad = 0, first = -1;
        for (int64_t i = 0; i < n; ++i) {
            if (std::memcmp(ya.data() + i * eb, yb.data() + i * eb, eb) == 0) continue;
            if (to == RAD_F32 && std::isnan(((const float*)ya.data())[i]) &&
                std::isnan(((const float*)yb.data())[i]))
                continue;
            if (first < 0) first = i;
            ++bad;
        }
        if (out_a) *out_a = ya;
        if (!bad) {
            if (ck.verbose)
                std::printf("  cast %-16s all %lld inputs identical\n", label, (long long)n);
            return true;
        }
        uint32_t in_bits = 0, got = 0, want = 0;
        std::memcpy(&in_bits, (const uint8_t*)src + first * (rad_dtype_bits(from) / 8),
                    (size_t)rad_dtype_bits(from) / 8);
        std::memcpy(&want, ya.data() + first * eb, eb);
        std::memcpy(&got, yb.data() + first * eb, eb);
        std::printf("EXACT   cast %-16s %lld of %lld inputs differ, first input 0x%x: libref "
                    "0x%x, %s 0x%x\n", label, (long long)bad, (long long)n, in_bits, want,
                    lib.info->name, got);
        return false;
    };

    struct Fmt { const char* name; uint32_t dt; int bits; };
    static const Fmt kFmt[] = {
        { "fp8_e4m3", RAD_F8E4M3, 8 }, { "fp8_e5m2", RAD_F8E5M2, 8 },
        { "bf16", RAD_BF16, 16 },      { "f16", RAD_F16, 16 },
        { "i8", RAD_I8, 8 },
    };
    bool ok = true;
    for (const Fmt& f : kFmt) {
        const int64_t n = (int64_t)1 << f.bits;
        std::vector<uint8_t> src((size_t)((n * f.bits) / 8));
        if (f.bits == 8) for (int64_t i = 0; i < n; ++i) src[(size_t)i] = (uint8_t)i;
        else for (int64_t i = 0; i < n; ++i) ((uint16_t*)src.data())[i] = (uint16_t)i;
        const std::string wide = std::string(f.name) + " -> f32";
        const std::string narrow = std::string("f32 -> ") + f.name;

        /* Every pattern widened; then every value narrowed back. */
        std::vector<uint8_t> wbytes;
        if (!both(wide.c_str(), f.dt, f.name, RAD_F32, "f32", src.data(), n, &wbytes)) {
            ok = false;
            continue;
        }
        std::vector<float> vals((size_t)n);
        std::memcpy(vals.data(), wbytes.data(), (size_t)n * sizeof(float));
        if (!both(narrow.c_str(), RAD_F32, "f32", f.dt, f.name, vals.data(), n, nullptr))
            ok = false;

        /* The midpoints, their neighbours, the overflow edges and the NaNs. */
        std::vector<float> fin;
        for (float v : vals) if (std::isfinite(v)) fin.push_back(v);
        std::sort(fin.begin(), fin.end());
        fin.erase(std::unique(fin.begin(), fin.end()), fin.end());
        std::vector<float> in;
        for (size_t i = 0; i + 1 < fin.size(); ++i) {
            const float m = (float)(((double)fin[i] + (double)fin[i + 1]) / 2.0);
            float lo = m, hi = m;
            in.push_back(m);
            for (int u = 0; u < 2; ++u) {
                lo = std::nextafter(lo, -INFINITY);
                hi = std::nextafter(hi, INFINITY);
                in.push_back(lo);
                in.push_back(hi);
            }
        }
        static const float kEdges[] = { 127.5f, 128.0f, 300.0f, 448.0f, 464.0f, 480.0f, 57344.0f,
                                        61440.0f, 65504.0f, 65520.0f, 1e10f, 3.4028235e38f,
                                        INFINITY };
        for (float e : kEdges) { in.push_back(e); in.push_back(-e); }
        static const uint32_t kNaN[] = { 0x7fc00000u, 0xffc00000u, 0x7fffffffu, 0xffffffffu,
                                         0x7ff00000u, 0x7f800001u, 0xff800001u, 0x7fc00001u };
        for (uint32_t b : kNaN) {
            float v;
            std::memcpy(&v, &b, 4);
            in.push_back(v);
        }
        const std::string mid = narrow + " mid";
        if (!both(mid.c_str(), RAD_F32, "f32", f.dt, f.name, in.data(), (int64_t)in.size(),
                  nullptr))
            ok = false;
    }
    return ok;
}

/* ================================================================== schema agreement
 * A copy of core/plugin/loader.cpp: schema_same, so drift is caught here rather than at engine
 * start. */
static bool schema_same(const RadOpSchema& a, const RadOpSchema& b, std::string* why) {
    char buf[256];
    if (a.n_params != b.n_params) {
        std::snprintf(buf, sizeof buf, "%d parameters vs %d", a.n_params, b.n_params);
        *why = buf; return false;
    }
    for (int i = 0; i < a.n_params; ++i) {
        const RadParamSpec& x = a.params[i];
        const RadParamSpec& y = b.params[i];
        const char* xk = x.key ? x.key : "";
        const char* yk = y.key ? y.key : "";
        if (std::strcmp(xk, yk)) {
            std::snprintf(buf, sizeof buf, "parameter %d is '%s' vs '%s'", i, xk, yk);
            *why = buf; return false;
        }
        if (x.type != y.type) {
            std::snprintf(buf, sizeof buf, "parameter '%s' has type %d vs %d", xk, x.type, y.type);
            *why = buf; return false;
        }
        if (x.required != y.required) {
            std::snprintf(buf, sizeof buf, "parameter '%s' is %s vs %s", xk,
                          x.required ? "required" : "optional",
                          y.required ? "required" : "optional");
            *why = buf; return false;
        }
    }
    if (a.n_operands != b.n_operands) {
        std::snprintf(buf, sizeof buf, "%d operands vs %d", a.n_operands, b.n_operands);
        *why = buf; return false;
    }
    for (int i = 0; i < a.n_operands; ++i)
        if (a.operands[i].role != b.operands[i].role ||
            a.operands[i].optional != b.operands[i].optional) {
            std::snprintf(buf, sizeof buf, "operand %d ('%s') has a different role/optionality", i,
                          a.operands[i].name ? a.operands[i].name : "?");
            *why = buf; return false;
        }
    return true;
}

/* The directory holding this binary, on the build's filesystem (which reports its direct-I/O
 * alignment on ext4 or XFS). */
static std::string self_dir(const char* argv0) {
    const std::string self = argv0;
    const size_t sl = self.rfind('/');
    return sl == std::string::npos ? "." : self.substr(0, sl);
}

int main(int argc, char** argv) {
    std::string ref_path = "libref.so", avx_path = "libavx.so";
    int only_level = -1;
    Check ck;
    ck.threads = 4;

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if      (s == "--ref")     ref_path = next();
        else if (s == "--avx")     avx_path = next();
        else if (s == "--op")      ck.only_op = next();
        else if (s == "--level")   only_level = std::atoi(next());
        else if (s == "--tol")     ck.tol_override = std::atof(next());
        else if (s == "--threads") ck.threads = std::atoi(next());
        else if (s == "-v" || s == "--verbose") ck.verbose = true;
        else if (s == "-h" || s == "--help") {
            std::printf(
                "rad-avx-check -- a host kernel library against libref, every op, every ISA "
                "level.\n\n"
                "  --ref PATH     the oracle plugin      (default libref.so)\n"
                "  --avx PATH     the plugin under test  (default libavx.so); any host library\n"
                "  --op NAME      only this op\n"
                "  --level N      only this ISA level (0 scalar 1 avx 2 avx2 3 avx512)\n"
                "  --threads N    rerun every case with each parallel region on N threads and\n"
                "                 require the same bytes (default 4, 0 to skip; libavx only)\n"
                "  --tol X        override every tolerance (for bisecting, not for passing)\n"
                "  -v             print every case, not only the failures\n\n"
                "A library not named libavx is checked as a host library: the libref ops it\n"
                "does not serve are listed rather than failed (libavx/README.md).\n");
            return 0;
        }
    }

    Lib ref, lib;
    if (!ref.open(ref_path.c_str()) || !lib.open(avx_path.c_str())) return 2;
    ck.ref = &ref;
    ck.lib = &lib;
    ck.own = std::strcmp(lib.info->name, "libavx") == 0;
    const char* who = lib.info->name;
    std::printf("oracle : %s %s  (%s)\n", ref.info->name, ref.info->version, ref_path.c_str());
    std::printf("under  : %s %s  (%s)\n", who, lib.info->version, avx_path.c_str());
    if (!ck.own)
        std::printf("         checked as a host library: the ops it lacks are listed, not "
                    "failed\n");

    /* ---- 1. schema agreement */
    int sch_bad = 0, sch_extra = 0;
    for (int i = 0, n = lib.schema_count(); i < n; ++i) {
        const RadOpSchema* sa = lib.schema_at(i);
        if (!sa || !sa->op) continue;
        const RadOpSchema* sr = ref.schema(sa->op);
        if (!sr) {
            if (ck.own) {
                std::printf("SCHEMA  %-24s declared by %s and not by libref\n", sa->op, who);
                ++sch_bad;
            } else {
                if (ck.verbose)
                    std::printf("SCHEMA  %-24s not in libref: no oracle, not checked\n", sa->op);
                ++sch_extra;
            }
            continue;
        }
        std::string why;
        if (!schema_same(*sr, *sa, &why)) {
            std::printf("SCHEMA  %-24s DISAGREES: %s\n", sa->op, why.c_str());
            ++sch_bad;
        }
    }
    std::printf("schemas: %d declared, %d disagree with libref", lib.schema_count(), sch_bad);
    if (sch_extra) std::printf(", %d not in libref (no oracle)", sch_extra);
    std::printf("\n");
    if (sch_bad) {
        std::printf("\nThe loader refuses a plugin on any schema disagreement, so every row of %s "
                    "would be\nunreachable. Fix the schemas first.\n", who);
        return 1;
    }

    /* ---- 2. coverage
     *
     * Any op libref implements and libavx lacks is a failure. For another library every gap is
     * listed. */
    int missing = 0, device_only = 0;
    std::vector<std::string> gaps;
    for (int i = 0, n = ref.kernel_count(); i < n; ++i) {
        const RadKernelInfo* kr = ref.kernel_at(i);
        if (!kr || !kr->op) continue;
        if (lib.host_row(kr->op)) continue;
        if (lib.row(kr->op)) ++device_only;
        if (ck.own) std::printf("COVER   %-24s libref implements it, %s does not\n", kr->op, who);
        ++missing;
        gaps.push_back(kr->op);
    }
    if (ck.own) {
        std::printf("coverage: libref %d rows, %s %d rows, %d op(s) missing\n",
                    ref.kernel_count(), who, lib.kernel_count(), missing);
    } else {
        std::printf("coverage: libref %d rows, %s %d rows; %d of libref's ops have no host row "
                    "here (%d a device row only)", ref.kernel_count(), who, lib.kernel_count(),
                    missing, device_only);
        if (!gaps.empty()) {
            std::printf(", not checked:\n ");
            for (const std::string& g : gaps) std::printf(" %s", g.c_str());
        }
        std::printf("\n");
    }

    /* ---- 3/4/5. the numbers, per ISA level */
    const std::vector<Case> cases = build_cases();
    int lo = 0, hi = 0, top = 0;
    if (!lib.isa_force) {
        std::printf("\nNote: %s exports no avx_isa_force, so only the level it selects for itself "
                    "is checked.\n", avx_path.c_str());
    } else {
        std::printf("cpu    : highest executable level is %s\n", lib.isa_name(lib.isa_detect()));
        lo = only_level >= 0 ? only_level : 0;
        hi = only_level >= 0 ? only_level : AVX_N_LEVELS - 1;
        top = std::min(hi, lib.isa_detect());
    }
    if (ck.threads > 0 && !lib.par_force) {
        std::printf("Note: %s exports no avx_par_force, so its parallel regions are not forced.\n",
                    avx_path.c_str());
        ck.threads = 0;
    } else if (ck.threads > 0) {
        const int forced = lib.par_force(ck.threads);
        lib.par_force(0);
        if (!forced) {
            std::printf("Note: %s was built without OpenMP: no parallel region to force.\n",
                        avx_path.c_str());
            ck.threads = 0;
        }
    }

    int rc = ck.own && missing ? 1 : 0;
    const bool gather = ck.own && (ck.only_op.empty() || ck.only_op == "embed_lookup_q");
    /* Run before any gather: the case forks, best done before the gather's threads exist. */
    if (gather) {
        std::printf("\n");
        if (!ngram_init_case(lib)) rc = 1;
    }
    std::vector<std::string> declined_ops;
    for (int lvl = lo; lvl <= hi; ++lvl) {
        if (lib.isa_force) {
            const int actual = lib.isa_force(lvl);
            if (actual != lvl) {
                /* Clamped: the level cannot run here, so report it and count nothing. */
                std::printf("\n== level %d (%s): not executable on this CPU (clamped to %s), not "
                            "counted\n", lvl, lib.isa_name(lvl), lib.isa_name(actual));
                continue;
            }
            std::printf("\n== level %d (%s) ==\n", lvl, lib.isa_name(lvl));
        } else {
            std::printf("\n== %s ==\n", who);
        }
        /* Exhaustive conversion check first, per level (the vector bodies differ per level). */
        if (ck.only_op.empty() || ck.only_op == "cast")
            if (!exhaustive_cast(ck)) rc = 1;
        Stats st;
        for (const Case& c : cases) {
            if (!ck.only_op.empty() && ck.only_op != c.op) continue;
            /* Speed-only cases are not checked (avx_harness.h). */
            if (c.speed_only) continue;
            const int before = st.declined;
            run_case(ck, c, &st);
            if (st.declined != before &&
                std::find(declined_ops.begin(), declined_ops.end(), c.op) == declined_ops.end())
                declined_ops.push_back(c.op);
        }
        std::printf("  %d checked, %d passed, %d failed, %d declined, %d skipped, %d missing.",
                    st.checked, st.passed, st.failed, st.declined, st.skipped, st.missing);
        if (st.unselected) std::printf("  %d admitted by no row's constraints.", st.unselected);
        std::printf("  worst rel_l2 = %.3e", st.worst);
        if (!st.worst_where.empty()) std::printf("\n  worst case: %s", st.worst_where.c_str());
        if (st.rerun) std::printf("\n  %d rerun on %d threads, byte for byte", st.rerun, ck.threads);
        std::printf("\n");
        if (st.failed || st.declined || st.skipped || (ck.own && st.missing)) rc = 1;
        if (!hand_cases(ck)) rc = 1;
        if (!zero_row_cases(ck, cases)) rc = 1;
        if (gather) {
            /* The E4M3 table of quantised containers and the plain bf16 one of an unconverted
             * checkpoint: one path, two row sizes. The reads, the cache and the reader thread are
             * compiled once, at the baseline, so the two ranks race through 1000 calls only at the
             * highest level; the others run 50, enough to check their decode. Ids differ per level,
             * so each level also misses the cache. */
            const std::string at = self_dir(argv[0]);
            const int64_t calls = lvl == top ? 1000 : 50;
            const uint64_t seed = 0x9e3779b97f4a7c15ull + (uint64_t)lvl;
            if (!row_gather_case(lib, at, calls, seed, ck.verbose, false)) rc = 1;
            if (!row_gather_case(lib, at, calls, seed, ck.verbose, true)) rc = 1;
            /* And in TMPDIR, usually tmpfs: since Linux 6.6 tmpfs does direct reads but reports
             * no alignment (like btrfs), so the gather uses the block size. Only the read differs
             * there, so it runs once. */
            if (lvl == top && !row_ahead_case(lib, at, ck.verbose)) rc = 1;
            if (lvl == top) {
                const char* tmp = std::getenv("TMPDIR");
                if (!row_gather_case(lib, tmp && *tmp ? tmp : "/tmp", calls, seed, ck.verbose,
                                     false))
                    rc = 1;
            }
        }
    }
    if (!declined_ops.empty()) {
        std::printf("\nDeclined (RAD_E_UNSUPPORTED from a row that claims the op), %d op(s):\n ",
                    (int)declined_ops.size());
        for (const std::string& s : declined_ops) std::printf(" %s", s.c_str());
        std::printf("\nThe selector resolves to such a row, so the step fails at issue, not at "
                    "declare.\n");
    }
    std::printf("\n%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
