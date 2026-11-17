/* lq_hess.cpp -- reading the calibration Hessians, and D . H. See lq_hess.h. */
#include "lq_hess.h"
#include "lq_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>

namespace lq {

namespace {

/* "RADGRAM2", k, rank, rows, domain, a pad word, then k*k f32 row-major: the file the engine's
 * calibration tap writes (arch/common/rad_arch.h, calib_write). The header is 32 bytes. */
constexpr int kHdr = 32;

/* One file per rank, ranks 0 .. kMaxRanks - 1. */
constexpr int kMaxRanks = 16;

std::string gram_path(const char* dir, int rank, const std::string& stem) {
    return std::string(dir) + "/gram.r" + std::to_string(rank) + "." + stem + ".bin";
}

/* The unnormalised Hadamard of width `n` over every n-wide block of a length-k row, in place. */
void fwht_row(double* v, int64_t k, int64_t n) {
    for (int64_t b0 = 0; b0 < k; b0 += n) fwht_inplace(v + b0, n);
}

/* H -> F H F for the block-diagonal Hadamard F of width `n`: every row, then every column. F is
 * symmetric, so the column pass is the row pass over the transpose; the result stays bit-exactly
 * symmetric only because both passes run the same butterflies in the same order. The column pass
 * runs fwht_inplace's butterflies on whole rows of a block of `n` rows, so each column sees the
 * same operations in the same order as the transform of that column alone; rows, and blocks of
 * rows, are independent, so threads change nothing. */
void rotate_gram(std::vector<double>& h, int64_t k, int64_t n) {
    double* d = h.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < k; ++r) fwht_row(d + r * k, k, n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t b0 = 0; b0 < k; b0 += n)
        for (int64_t s = 1; s < n; s <<= 1)
            for (int64_t i = b0; i < b0 + n; i += s << 1)
                for (int64_t j = i; j < i + s; ++j) {
                    double* x = d + j * k;
                    double* y = d + (j + s) * k;
                    for (int64_t c = 0; c < k; ++c) {
                        const double a = x[c], b = y[c];
                        x[c] = a + b;
                        y[c] = a - b;
                    }
                }
}

bool read_gram(const std::string& path, int64_t k, int64_t domain, std::vector<double>* acc) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char hdr[kHdr];
    if (std::fread(hdr, 1, kHdr, f) != (size_t)kHdr || std::memcmp(hdr, "RADGRAM2", 8) != 0) {
        std::fprintf(stderr, "libquant: %s is not a RADGRAM2 file\n", path.c_str());
        std::fclose(f);
        return false;
    }
    uint32_t kk = 0, dom = 0;
    std::memcpy(&kk, hdr + 8, 4);
    std::memcpy(&dom, hdr + 24, 4);
    if ((int64_t)kk != k) {
        std::fprintf(stderr, "libquant: %s holds a %u-wide Hessian and this weight's K is %lld\n",
                     path.c_str(), kk, (long long)k);
        std::fclose(f);
        return false;
    }
    /* Domains convert only through the model's own (domain 0): a file rotated at one width is
     * rotated back before rotating to another. Both widths must tile K. */
    const int64_t from = (int64_t)dom;
    auto tiles = [k](int64_t n) { return n == 0 || (n >= 2 && (n & (n - 1)) == 0 && k % n == 0); };
    if (!tiles(from) || !tiles(domain)) {
        std::fprintf(stderr, "libquant: %s is in domain %lld and the weight wants %lld, which do "
                             "not both tile K=%lld\n", path.c_str(), (long long)from,
                     (long long)domain, (long long)k);
        std::fclose(f);
        return false;
    }
    std::vector<float> buf((size_t)(k * k));
    const bool okr = std::fread(buf.data(), 4, buf.size(), f) == buf.size();
    std::fclose(f);
    if (!okr) {
        std::fprintf(stderr, "libquant: %s is short\n", path.c_str());
        return false;
    }
    std::vector<double> h(buf.begin(), buf.end());
    if (from != domain) {
        if (from != 0) {
            rotate_gram(h, k, from);
            const double inv = 1.0 / ((double)from * (double)from);
            for (double& x : h) x *= inv;
        }
        if (domain != 0) rotate_gram(h, k, domain);
        /* Round to f32, as a tap in that domain would have stored it, so a converted Hessian
         * matches one summed there and a bf16 model's calibration and a rotated container's
         * quantise a weight identically. */
        for (double& x : h) x = (double)(float)x;
    }
    if (acc->empty()) acc->assign(h.size(), 0.0);
    for (size_t i = 0; i < h.size(); ++i) (*acc)[i] += h[i];
    return true;
}

/* The total rows summed by every rank's file for `stem`, or -1 when `dir` holds none. */
int64_t calib_rows(const char* dir, const std::string& stem) {
    int64_t total = -1;
    for (int r = 0; r < kMaxRanks; ++r) {
        FILE* f = std::fopen(gram_path(dir, r, stem).c_str(), "rb");
        if (!f) continue;
        char hdr[kHdr];
        if (std::fread(hdr, 1, kHdr, f) == (size_t)kHdr && std::memcmp(hdr, "RADGRAM2", 8) == 0) {
            uint64_t n = 0;
            std::memcpy(&n, hdr + 16, 8);
            total = (total < 0 ? 0 : total) + (int64_t)n;
        }
        std::fclose(f);
    }
    return total;
}

/* The name less a `.weight` or `.scale` tail. */
std::string own_stem(const char* name) {
    std::string s = name ? name : "";
    const char* tails[] = { ".weight", ".scale" };
    for (const char* t : tails) {
        const size_t n = std::strlen(t);
        if (s.size() > n && s.compare(s.size() - n, n, t) == 0) { s.resize(s.size() - n); break; }
    }
    return s;
}

/* A few layers' Grams, the oldest dropped first: a MoE layer's experts read their layer's again
 * and again, a dense model reads each once, and every layer's kept at once outgrows the host
 * (Qwen3.8-27B's are ~100 GB; a GPTQ convert with descent was OOM-killed at layer 50). */
constexpr size_t kGrams = 4;

struct GramCache {
    std::mutex mu;
    std::map<std::string, Gram> g;   /* null = looked and found nothing */
    std::deque<std::string> order;   /* g's keys, oldest first */
};

GramCache& gram_cache() {
    static GramCache c;
    return c;
}

/* A panel is CB columns of H over all K rows, packed into a K x CB buffer so each k step reads one
 * contiguous line instead of striding K floats; this keeps the product compute-bound (3-5x faster
 * than unpacked). A row block is RB rows of D held as RB*CB register accumulators: 8 x 32 under
 * AVX-512 and 4 x 16 under AVX2 use sixteen and eight vector registers. Each accumulator sums k in
 * ascending order and is stored once, so panel, block and instruction set change the schedule, not
 * any sum. Both functions are forced inline so each compiles under its caller's target ISA. */
template <int RB, int CB>
__attribute__((always_inline)) inline void panel(const float* D, int64_t K, const float* P,
                                                 float* G) {
    float acc[RB][CB];
    for (int r = 0; r < RB; ++r)
        for (int c = 0; c < CB; ++c) acc[r][c] = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const float* h = P + k * CB;
        for (int r = 0; r < RB; ++r) {
            const float d = D[r * K + k];
#pragma omp simd
            for (int c = 0; c < CB; ++c) acc[r][c] += d * h[c];
        }
    }
    for (int r = 0; r < RB; ++r)
        for (int c = 0; c < CB; ++c) G[r * K + c] = acc[r][c];
}

/* A task: one panel of H packed, then a run of row blocks against it. */
template <int RB, int CB>
__attribute__((always_inline)) inline void task(const float* D, int64_t K, const float* H,
                                                int64_t j0, float* G, int64_t b0, int64_t b1,
                                                float* pack) {
    for (int64_t k = 0; k < K; ++k) std::memcpy(pack + k * CB, H + k * K + j0, CB * sizeof(float));
    for (int64_t b = b0; b < b1; ++b) panel<RB, CB>(D + b * RB * K, K, pack, G + b * RB * K + j0);
}

/* The edges the blocks leave, one element at a time in the same k order. */
void edge(const float* D, int64_t K, const float* H, int64_t r0, int64_t r1, int64_t j0,
          int64_t j1, float* G) {
    for (int64_t r = r0; r < r1; ++r)
        for (int64_t j = j0; j < j1; ++j) {
            float s = 0.0f;
            for (int64_t k = 0; k < K; ++k) s += D[r * K + k] * H[k * K + j];
            G[r * K + j] = s;
        }
}

/* The blocked part at one shape, threaded over (panel, run of row blocks). Returns the rows and
 * columns it covered. */
template <int RB, int CB>
void blocked(const float* D, int64_t R, int64_t K, const float* H, float* G,
             void (*run)(const float*, int64_t, const float*, int64_t, float*, int64_t, int64_t,
                         float*),
             int64_t* rf, int64_t* cf) {
    const int64_t nrb = R / RB, ncb = K / CB;
    *rf = nrb * RB;
    *cf = ncb * CB;
    if (nrb == 0 || ncb == 0) { *rf = *cf = 0; return; }
    const int64_t chunk = 64;                       /* row blocks a task */
    const int64_t nch = (nrb + chunk - 1) / chunk;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<float> pack((size_t)(K * CB));
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(dynamic, 1)
#endif
        for (int64_t cb = 0; cb < ncb; ++cb)
            for (int64_t ch = 0; ch < nch; ++ch)
                run(D, K, H, cb * CB, G, ch * chunk, std::min(nrb, (ch + 1) * chunk), pack.data());
    }
}

__attribute__((target("avx512f")))
void run_avx512(const float* D, int64_t K, const float* H, int64_t j0, float* G, int64_t b0,
                int64_t b1, float* pack) {
    task<8, 32>(D, K, H, j0, G, b0, b1, pack);
}
__attribute__((target("avx2")))
void run_avx2(const float* D, int64_t K, const float* H, int64_t j0, float* G, int64_t b0,
              int64_t b1, float* pack) {
    task<4, 16>(D, K, H, j0, G, b0, b1, pack);
}
void run_base(const float* D, int64_t K, const float* H, int64_t j0, float* G, int64_t b0,
              int64_t b1, float* pack) {
    task<4, 8>(D, K, H, j0, G, b0, b1, pack);
}

}  /* namespace */

std::string calib_stem(const char* name) {
    std::string s = own_stem(name);
    const size_t dot = s.rfind('.');
    if (dot != std::string::npos && dot + 1 < s.size()) {
        bool all_digits = true;
        for (size_t i = dot + 1; i < s.size(); ++i)
            if (s[i] < '0' || s[i] > '9') { all_digits = false; break; }
        if (all_digits) s.resize(dot);
    }
    /* A vocabulary-weighted lm_head's width classes (output.k<B>) are rows of one weight: they
     * read the head's input Gram. */
    if (s.size() > 8 && s.compare(0, 8, "output.k") == 0 &&
        s.find_first_not_of("0123456789", 8) == std::string::npos)
        s.resize(6);
    return s;
}

/* The rank parts of `stem` (see calib_sum), block diagonal into `acc`; the number read. */
int calib_sum_parts(const char* dir, const std::string& stem, int64_t K, int64_t domain,
                    std::vector<double>* acc) {
    std::vector<std::vector<float>> parts;
    int64_t total = 0;
    for (int r = 0; r < kMaxRanks; ++r) {
        const std::string path = gram_path(dir, r, stem + ".part");
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) break;
        char hdr[kHdr];
        uint32_t kk = 0, dom = 0;
        bool ok = std::fread(hdr, 1, kHdr, f) == (size_t)kHdr && std::memcmp(hdr, "RADGRAM2", 8) == 0;
        if (ok) { std::memcpy(&kk, hdr + 8, 4); std::memcpy(&dom, hdr + 24, 4); }
        ok = ok && dom == 0 && kk > 0 && total + kk <= K;
        std::vector<float> h;
        if (ok) {
            h.resize((size_t)kk * kk);
            ok = std::fread(h.data(), 4, h.size(), f) == h.size();
        }
        std::fclose(f);
        if (!ok) {
            std::fprintf(stderr, "libquant: %s is not a usable part of the %lld-wide '%s' (a domain-0 "
                                 "RADGRAM2 within K)\n", path.c_str(), (long long)K, stem.c_str());
            return 0;
        }
        total += kk;
        parts.push_back(std::move(h));
    }
    if (parts.empty()) return 0;
    if (total != K) {
        std::fprintf(stderr, "libquant: the %zu part(s) of '%s' cover %lld of K=%lld columns\n",
                     parts.size(), stem.c_str(), (long long)total, (long long)K);
        return 0;
    }
    std::vector<double> h((size_t)(K * K), 0.0);
    int64_t off = 0;
    for (const auto& p : parts) {
        const int64_t k = (int64_t)std::llround(std::sqrt((double)p.size()));
        for (int64_t i = 0; i < k; ++i)
            for (int64_t j = 0; j < k; ++j)
                h[(size_t)((off + i) * K + off + j)] = p[(size_t)(i * k + j)];
        off += k;
    }
    if (domain != 0) {
        rotate_gram(h, K, domain);
        for (double& x : h) x = (double)(float)x;
    }
    if (acc->empty()) acc->assign(h.size(), 0.0);
    for (size_t i = 0; i < h.size(); ++i) (*acc)[i] += h[i];
    return (int)parts.size();
}

int calib_sum(const char* dir, const std::string& stem, int64_t K, int64_t domain,
              std::vector<double>* acc) {
    /* Sum every rank's file. Under expert parallelism a rank's `down` Hessian covers only tokens
     * routed to its own experts, so no single file holds the layer's distribution. The gate_up
     * files are duplicated across ranks; summing them scales H by the rank count, which every
     * algorithm here is invariant to. */
    int found = 0;
    for (int r = 0; r < kMaxRanks; ++r)
        if (read_gram(gram_path(dir, r, stem), K, domain, acc)) ++found;
    if (!found && (domain == 0 || (domain >= 2 && (domain & (domain - 1)) == 0 && K % domain == 0)))
        found = calib_sum_parts(dir, stem, K, domain, acc);
    return found;
}

std::string calib_resolve(const char* dir, const char* name, int64_t K) {
    const std::string pooled = calib_stem(name);
    const std::string own = own_stem(name);
    if (!dir || !*dir || own == pooled) return pooled;
    const int64_t rows = calib_rows(dir, own);
    if (rows >= kOwnRowsPerK * K) return own;
    if (rows >= 0)
        std::fprintf(stderr, "libquant: '%s' was routed %lld calibration rows, fewer than %lld; it "
                             "reads the layer's Hessian '%s'\n", own.c_str(), (long long)rows,
                     (long long)(kOwnRowsPerK * K), pooled.c_str());
    return pooled;
}

Gram calib_gram(const char* dir, const char* name, int64_t K, int64_t domain, const char* who) {
    if (!dir || !*dir || !name || !*name || K <= 0) return nullptr;
    const std::string stem = calib_resolve(dir, name, K);
    int found = 0;
    auto read = [&]() -> Gram {
        std::vector<double> acc;
        found = calib_sum(dir, stem, K, domain, &acc);
        if (!found) {
            std::fprintf(stderr, "libquant: no calibration for '%s' in %s (looked for "
                                 "gram.r*.%s.bin); %s rounds it to nearest\n", name, dir,
                         stem.c_str(), who);
            return nullptr;
        }
        auto h = std::make_shared<std::vector<float>>(acc.size());
        for (size_t i = 0; i < acc.size(); ++i) (*h)[i] = (float)acc[i];
        return h;
    };
    if (stem != calib_stem(name)) return read();

    const std::string key = std::string(dir) + "|" + stem + "|" + std::to_string(K) + "|" +
                            std::to_string(domain);
    GramCache& c = gram_cache();
    std::lock_guard<std::mutex> lk(c.mu);
    auto it = c.g.find(key);
    if (it != c.g.end()) return it->second;
    Gram h = read();
    if (h)
        std::fprintf(stderr, "libquant: %s reads '%s' against %d calibration file(s), K=%lld\n",
                     who, stem.c_str(), found, (long long)K);
    while (c.order.size() >= kGrams) {
        c.g.erase(c.order.front());
        c.order.pop_front();
    }
    c.g.emplace(key, h);
    c.order.push_back(key);
    return h;
}

void mat_hess(const float* D, int64_t R, int64_t K, const float* H, float* G) {
    if (R <= 0 || K <= 0) return;
    int64_t rf = 0, cf = 0;
    if (__builtin_cpu_supports("avx512f")) blocked<8, 32>(D, R, K, H, G, run_avx512, &rf, &cf);
    else if (__builtin_cpu_supports("avx2")) blocked<4, 16>(D, R, K, H, G, run_avx2, &rf, &cf);
    else blocked<4, 8>(D, R, K, H, G, run_base, &rf, &cf);
    /* The column edge of the blocked rows, then the row edge across every column. */
    if (cf < K && rf > 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t r = 0; r < rf; ++r) edge(D, K, H, r, r + 1, cf, K, G);
    }
    if (rf < R) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t j = 0; j < K; ++j) edge(D, K, H, rf, R, j, j + 1, G);
    }
}

void row_dots(const float* D, const float* G, int64_t R, int64_t K, double* out) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < R; ++r) {
        double s = 0.0;
        for (int64_t k = 0; k < K; ++k) s += (double)D[r * K + k] * (double)G[r * K + k];
        out[r] = s;
    }
}

}  /* namespace lq */
