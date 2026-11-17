/* rad_calib_full.h -- the whole Hessian of each linear's input in a dense bf16 trunk, under tensor
 * parallelism, recorded over a --calib-corpus run into --calib-dir (RadBuildCtx::calib_*).
 *
 * GPTQ needs H = sum over tokens of x x^T for the whole input x of a linear. A linear split by its
 * input columns across ranks (o_proj, the delta net's out_proj, down_proj) sees only its rank's
 * columns, and a Gram of those is one diagonal block of H: the products across ranks are never
 * formed on any card. So each such input is gathered to its whole width first (`all_gather`,
 * row-interleaved: rank r's columns land at r * width / world), and the Gram is taken of the whole
 * row. An input every rank holds whole (a norm's output) is read where it is.
 *
 * One rank records each layer (layer % world), so the Gram work and the host memory split
 * evenly. Its f32 scratch receives one step's sum for one input (`gram_accum` with `upper` 1: H is
 * symmetric and only k2 >= k1 is kept), whose upper triangle is copied to a pinned square and which
 * is zeroed for the next input, all in stream order. A worker thread waits on the copy's event and adds the square to a
 * host accumulator holding the upper triangle packed row by row (k(k+1)/2 f32), while the rank goes
 * on issuing the step: two squares alternate, so the rank waits only when the adds fall two Grams
 * behind. Syncing and adding on the rank's own thread instead left the card idle for every add, 16
 * of 1.2 GB a step on each rank of Qwen3.8-27B, and the ranks wait on each other at every
 * all-reduce. The host holds every recorded input of the rank: 786 MB a layer at Qwen3.8-27B,
 * which is what --calib-layers bounds. A step's sum is f32 over at most a step's
 * rows, and the host adds a few thousand of them; the consumer damps H by a percent of its trace,
 * far above that rounding.
 *
 * Files, written once the plugin has seen RadBuildCtx::calib_tokens rows (the whole corpus): one
 * safetensors file per input, <dir>/<stem>.safetensors, holding `hessian_upper`, f32
 * [k * (k + 1) / 2], row i's entries k2 = i .. k-1 after row i-1's, and metadata naming `k`, the
 * rows summed and the checkpoint linears that read the input. H is the raw sum; divide by `rows`
 * for the mean second moment.
 *
 * An int8 trunk (site_i8, one rank only) is recorded from the codes and scales its GEMMs actually
 * read, so H includes what the model's own quantisation did to the input -- the input a quantised
 * model feeds each layer, for GPTQ against the error it inherits. Codes a rotating producer wrote
 * are in the rotated domain; the file's metadata then names it (`domain`: the Hadamard width), as
 * RADGRAM2's header does, and the reader converts. */
#pragma once

#include <arch/rad_arch.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rad::arch {

struct CalibFull {
    struct Site {
        std::string stem;              /* the file's name, without .safetensors */
        std::string linears;           /* the checkpoint linears that read this input, comma-separated */
        rad_buf     local = 0;         /* this rank's part of the input (the whole input when replicated) */
        int64_t     K = 0;             /* the whole input's width */
        int64_t     k_local = 0;       /* this rank's columns; K when replicated */
        rad_op      op_gather = 0;     /* the columns of every rank into `gathered`; 0 when replicated */
        rad_op      op_gram = 0;       /* declared on the owner only */
        bool        owner = false;
        rad_buf     q8 = 0, s8 = 0;    /* site_i8: the int8 codes and their f32 (row, 128) scales */
        uint32_t    domain = 0;        /* site_i8: the codes' rotation width, 0 for none */
        std::vector<float> tri;        /* the owner's accumulator, upper triangle packed */
    };

    bool        on = false;
    const char* dir = nullptr;
    int         rank = 0, world = 1;
    int64_t     lo = 0, hi = 0;        /* layers recorded */
    long long   target = 0, seen = 0;
    bool        written = false;
    rad_buf     gathered = 0;          /* [max_tok, widest split input] bf16 */
    int64_t     gather_w = 0;
    rad_buf     scratch = 0;           /* [kmax, kmax] f32 on a rank that owns any input */
    int64_t     kmax = 0;
    std::vector<Site> sites;

    /* The host side of the taps (see the file comment). Created at the first tap and released once
     * the files are written; a run that ends before then leaks it, since its worker may still wait
     * on a card that is being torn down. */
    struct Async {
        std::mutex              mu;
        std::condition_variable cv;
        std::thread             worker;
        float*                  square[2] = { nullptr, nullptr };   /* pinned, kmax * kmax f32 */
        RadEvent                copied[2] = { nullptr, nullptr };
        int                     site[2]   = { -1, -1 };   /* the site square s holds; -1 when free */
        std::deque<int>         queue;                    /* squares copied and not yet added */
        int                     next = 0;
        bool                    stop = false, failed = false;
    };
    Async* async = nullptr;

    /* Whether this run records layer l. */
    bool records(int64_t l) const { return on && l >= lo && l < hi; }
    int  owner_of(int64_t l) const { return (int)(l % world); }

    /* From the build context: nothing unless --calib-dir is set. `widest_split` and `widest` are
     * the largest split input and the largest input of any recorded site, which size the gather
     * buffer and the scratch. */
    int begin(RadBuilder* b, Names& nm, const Geom& g, const RadBuildCtx* ctx, int64_t n_layer,
              int64_t widest_split, int64_t widest) {
        if (!ctx->calib_dir || !*ctx->calib_dir) return RAD_OK;
        on     = true;
        dir    = ctx->calib_dir;
        rank   = g.rank;
        world  = g.world;
        lo     = ctx->calib_layer_hi ? ctx->calib_layer_lo : 0;
        hi     = ctx->calib_layer_hi ? std::min<int64_t>(ctx->calib_layer_hi, n_layer) : n_layer;
        target = ctx->calib_tokens;
        if (lo >= hi) {
            std::fprintf(stderr, "radiance: --calib-layers %lld:%lld names no layer of %lld\n",
                         (long long)lo, (long long)hi, (long long)n_layer);
            return RAD_E_INVAL;
        }
        if (target <= 0) {
            std::fprintf(stderr, "radiance: --calib-dir records over a --calib-corpus run, which "
                                 "states the corpus's size\n");
            return RAD_E_INVAL;
        }
        if (world > 1) {
            gather_w = widest_split;
            gathered = decl_b(b, nm.f("calib.gathered"), RAD_BF16, { g.max_tok, gather_w });
            if (!gathered) return RAD_E_INVAL;
        }
        kmax = widest;
        scratch = decl_b(b, nm.f("calib.scratch"), RAD_F32, { kmax, kmax }, RAD_BUF_PERSIST);
        return scratch ? RAD_OK : RAD_E_INVAL;
    }

    /* One input of layer `layer` (-1: the lm_head's, recorded by rank 0). `local` is the rank's
     * part, `k_local` its width; the input is split across ranks when k_local * world == K. */
    int site(RadBuilder* b, const Geom& g, int64_t layer, std::string stem, std::string linears,
             rad_buf local, int64_t k_local, int64_t K) {
        Site s;
        s.stem    = std::move(stem);
        s.linears = std::move(linears);
        s.local   = local;
        s.K       = K;
        s.k_local = k_local;
        s.owner   = (layer < 0 ? 0 : owner_of(layer)) == rank;
        const bool split = k_local != K;
        if (split) {
            if (k_local * world != K || !gathered || K > gather_w) return RAD_E_INVAL;
            s.op_gather = rw(b, RAD_OP(b, "all_gather",
                                 RAD_PARAMS(RAD_INT("world_size", world),
                                            RAD_RANGE("numel", k_local, g.max_tok * k_local),
                                            RAD_STR("dtype", "bf16"), RAD_INT("row", k_local)),
                                 RAD_NOWEIGHTS),
                             { local }, { gathered });
            if (!s.op_gather) return RAD_E_INVAL;
        }
        if (s.owner) {
            if (K > kmax) return RAD_E_INVAL;
            s.op_gram = rw(b, RAD_OP(b, "gram_accum",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                          RAD_INT("group", 128), RAD_STR("dtype", "bf16"),   /* group: unread at bf16 */
                                          RAD_INT("upper", 1)),
                               RAD_NOWEIGHTS),
                           { split ? gathered : local, scratch }, { scratch });
            if (!s.op_gram) return RAD_E_INVAL;
        }
        sites.push_back(std::move(s));
        return RAD_OK;
    }

    /* One input of layer `layer` from an int8 trunk's codes `q8` and scales `s8` (see the file
     * comment): the whole input on one rank, `domain` the Hadamard width the codes were rotated by
     * (0 for none). */
    int site_i8(RadBuilder* b, const Geom& g, int64_t layer, std::string stem, std::string linears,
                rad_buf q8, rad_buf s8, int64_t K, uint32_t domain) {
        if (world > 1 || !q8 || !s8 || K > kmax) return RAD_E_INVAL;
        Site s;
        s.stem    = std::move(stem);
        s.linears = std::move(linears);
        s.K = s.k_local = K;
        s.owner   = true;
        s.q8 = q8;
        s.s8 = s8;
        s.domain  = domain;
        (void)layer;
        s.op_gram = rw(b, RAD_OP(b, "gram_accum",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                      RAD_INT("group", 128), RAD_STR("dtype", "i8a8"),
                                      RAD_INT("upper", 1)),
                           RAD_NOWEIGHTS),
                       { q8, s8, scratch }, { scratch });
        if (!s.op_gram) return RAD_E_INVAL;
        sites.push_back(std::move(s));
        return RAD_OK;
    }

    /* Site `i` over a step's T rows: gather (every rank), then on the owner the Gram into the
     * scratch, copied to a free square, the scratch zeroed, and the square handed to the worker. */
    void tap(RadCtx* c, size_t i, int64_t T) {
        Site& s = sites[i];
        RadOperand a = brows(s.local, T);
        if (s.op_gather) {
            RAD_ISSUE_N(c, s.op_gather, T * s.k_local, brows(s.local, T), bcol(gathered, 0, s.K, T));
            a = bcol(gathered, 0, s.K, T);
        }
        if (!s.owner) return;
        float* dev = static_cast<float*>(rad_buf_ptr(c, scratch));
        if (!dev || (!async && !start())) {
            rad_step_fail(c, "calibration: no scratch or pinned squares");
            return;
        }
        /* Allocated at the first step, not at declare: the engine may declare more than once. */
        if (s.tri.empty()) s.tri.assign((size_t)(s.K * (s.K + 1) / 2), 0.0f);
        Async& q = *async;
        const int k = q.next;
        q.next ^= 1;
        {
            std::unique_lock<std::mutex> lk(q.mu);
            q.cv.wait(lk, [&] { return q.site[k] < 0 || q.failed; });
            if (q.failed) {
                lk.unlock();
                rad_step_fail(c, "calibration: waiting on a Gram's copy failed");
                return;
            }
        }
        const int64_t K = s.K;
        if (s.q8)
            RAD_ISSUE_N(c, s.op_gram, T, brows(s.q8, T), brows(s.s8, T), praw2(dev, RAD_F32, K, K));
        else
            RAD_ISSUE_N(c, s.op_gram, T, a, RAD_NONE, praw2(dev, RAD_F32, K, K));
        /* `upper` 1 writes no entry below the diagonal, so only the upper triangle comes back:
         * band by band of kBand rows, each from its first row's diagonal on, about half the
         * square's bytes. The link is what a step's taps wait on, 25 GB a step on each rank of
         * Qwen3.8-27B as whole squares. */
        constexpr int64_t kBand = 512;
        bool ok = true;
        for (int64_t r = 0; r < K && ok; r += kBand)
            ok = rad_memcpy_2d_async(q.square[k] + r * K + r, K * 4, dev + r * K + r, K * 4,
                                     (K - r) * 4, std::min(kBand, K - r), rad_stream(c)) >= 0;
        if (!ok || rad_memset_async(dev, 0, K * K * 4, rad_stream(c)) < 0 ||
            rad_event_record(q.copied[k], rad_stream(c)) < 0) {
            rad_step_fail(c, "calibration: copying a Gram back failed");
            return;
        }
        {
            std::lock_guard<std::mutex> lk(q.mu);
            q.site[k] = (int)i;
            q.queue.push_back(k);
        }
        q.cv.notify_all();
    }

    /* After a step: once the corpus is through, every owned input to its file. */
    void after_step(RadCtx* c, int64_t T) {
        if (!on || written) return;
        seen += T;
        if (seen < target) return;
        written = true;
        int n = 0, bad = finish() ? 0 : 1;
        for (const Site& s : sites) {
            if (!s.owner || s.tri.empty()) continue;
            ++n;
            if (!write(s)) ++bad;
        }
        std::fprintf(stderr, "radiance: calibration: rank %d wrote %d Hessian(s) of layers %lld:%lld "
                             "after %lld rows to %s%s\n", rad_rank(c), n, (long long)lo, (long long)hi,
                     seen, dir, bad ? ", with errors" : "");
        if (seen != target)
            std::fprintf(stderr, "radiance: calibration: %lld rows seen and the corpus has %lld\n",
                         seen, target);
    }

private:
    bool start() {
        Async* q = new Async;
        bool ok = true;
        for (int k = 0; k < 2; ++k) {
            q->square[k] = static_cast<float*>(rad_dev_alloc(kmax * kmax * 4, RAD_MEM_HOST_PINNED));
            ok = ok && q->square[k] && rad_event_create(&q->copied[k]) >= 0;
        }
        if (!ok) {
            std::fprintf(stderr, "radiance: calibration could not pin 2 x %lld MiB of host squares "
                                 "on rank %d\n", (long long)((kmax * kmax * 4) >> 20), rank);
            release(q);
            return false;
        }
        async = q;
        q->worker = std::thread([this] { work(); });
        return true;
    }

    /* The worker: each copied square, in the order the rank issued them, into its site. */
    void work() {
        Async& q = *async;
        for (;;) {
            int k;
            {
                std::unique_lock<std::mutex> lk(q.mu);
                q.cv.wait(lk, [&] { return q.stop || !q.queue.empty(); });
                if (q.queue.empty()) return;
                k = q.queue.front();
                q.queue.pop_front();
            }
            Site& s = sites[(size_t)q.site[k]];
            const bool ok = rad_event_sync(q.copied[k]) >= 0;
            if (ok) add_upper(s.tri.data(), q.square[k], s.K);
            {
                std::lock_guard<std::mutex> lk(q.mu);
                q.site[k] = -1;
                if (!ok) q.failed = true;
            }
            q.cv.notify_all();
        }
    }

    /* Every square added and the worker gone; false if a copy failed. */
    bool finish() {
        if (!async) return true;
        {
            std::lock_guard<std::mutex> lk(async->mu);
            async->stop = true;
        }
        async->cv.notify_all();
        async->worker.join();
        const bool ok = !async->failed;
        if (!ok) std::fprintf(stderr, "radiance: calibration: a Gram's copy failed on rank %d\n", rank);
        release(async);
        async = nullptr;
        return ok;
    }

    static void release(Async* q) {
        for (int k = 0; k < 2; ++k) {
            if (q->square[k]) rad_dev_free(q->square[k], RAD_MEM_HOST_PINNED);
            if (q->copied[k]) rad_event_destroy(q->copied[k]);
        }
        delete q;
    }

    /* acc's upper triangle, packed row by row, += the upper triangle of the k x k row-major `full`. */
    static void add_upper(float* acc, const float* full, int64_t k) {
        const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nt; ++t)
            th.emplace_back([=] {
                for (int64_t i = t; i < k; i += nt) {
                    float* d = acc + (i * k - i * (i - 1) / 2);
                    const float* s = full + i * k + i;
                    for (int64_t j = 0; j < k - i; ++j) d[j] += s[j];
                }
            });
        for (std::thread& x : th) x.join();
    }

    bool write(const Site& s) const {
        char path[1024];
        std::snprintf(path, sizeof path, "%s/%s.safetensors", dir, s.stem.c_str());
        const uint64_t bytes = (uint64_t)s.tri.size() * 4;
        std::string hdr = "{\"__metadata__\":{\"k\":\"" + std::to_string(s.K) + "\",\"rows\":\"" +
                          std::to_string(seen) + "\",\"layout\":\"upper triangle, row-major, packed\","
                          "\"linears\":\"" + s.linears + "\"" +
                          (s.domain ? ",\"domain\":\"" + std::to_string(s.domain) + "\"" : std::string()) +
                          "},\"hessian_upper\":{\"dtype\":\"F32\","
                          "\"shape\":[" + std::to_string(s.tri.size()) + "],\"data_offsets\":[0," +
                          std::to_string(bytes) + "]}}";
        while (hdr.size() % 8) hdr += ' ';
        const uint64_t hl = hdr.size();
        FILE* f = std::fopen(path, "wb");
        if (!f) { std::fprintf(stderr, "radiance: calibration cannot write %s\n", path); return false; }
        bool ok = std::fwrite(&hl, 8, 1, f) == 1 && std::fwrite(hdr.data(), 1, hl, f) == hl &&
                  std::fwrite(s.tri.data(), 4, s.tri.size(), f) == s.tri.size();
        ok = std::fclose(f) == 0 && ok;
        if (!ok) std::fprintf(stderr, "radiance: calibration write to %s was short\n", path);
        return ok;
    }
};

}  /* namespace rad::arch */
