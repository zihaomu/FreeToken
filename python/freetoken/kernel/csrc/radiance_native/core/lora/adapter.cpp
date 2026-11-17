/* lora/adapter.cpp -- see adapter.h. Host code only. */
#include "adapter.h"

#include "rad_plugin.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <thread>
#include <unordered_map>

namespace rad {
namespace lora {

namespace {

#define TRY(x) do { const int s_ = (x); if (s_ < 0) return s_; } while (0)

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

int fail(std::string* err, int code, const std::string& m) {
    if (err) *err = m;
    return code;
}

std::string shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

/* ---------------------------------------------------------------- tensors as f32 */
bool float_dtype(uint32_t dt) { return dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F32; }

float elem(const StTensor& t, int64_t i) {
    switch (t.dtype) {
        case RAD_BF16: { uint16_t v; std::memcpy(&v, t.data + 2 * i, 2); return rad_bf16_to_f32(v); }
        case RAD_F16:  { uint16_t v; std::memcpy(&v, t.data + 2 * i, 2); return rad_f16_to_f32(v); }
        default:       { float v; std::memcpy(&v, t.data + 4 * i, 4); return v; }
    }
}

/* A tensor as a row-major matrix [shape[0], product of the rest]. */
int as_matrix(const StTensor* t, const char* what, int64_t* rows, int64_t* cols,
              std::vector<float>* out, std::string* err) {
    if (!t) return fail(err, RAD_E_FORMAT, fmt("no %s tensor", what));
    if (!float_dtype(t->dtype))
        return fail(err, RAD_E_DTYPE, fmt("%s is %s; expected BF16, F16 or F32", t->name.c_str(),
                                          t->dtype_str.c_str()));
    if (t->shape.size() < 2 || !t->data)
        return fail(err, RAD_E_SHAPE, fmt("%s has shape %s; expected a matrix", t->name.c_str(),
                                          shape_str(t->shape).c_str()));
    *rows = t->shape[0];
    *cols = t->numel() / (t->shape[0] ? t->shape[0] : 1);
    if (out) {
        out->resize((size_t)(*rows * *cols));
        for (int64_t i = 0; i < *rows * *cols; ++i) (*out)[(size_t)i] = elem(*t, i);
    }
    return RAD_OK;
}

/* The trailing dims past the first two, all 1 (an up projection of a convolution is 1 x 1). */
bool trailing_ones(const StTensor* t) {
    for (size_t i = 2; i < t->shape.size(); ++i) if (t->shape[i] != 1) return false;
    return true;
}

/* One number: shape [] (numel() counts it as no element) or [1], a float or (as some trainers
 * write an alpha) an integer or F64. */
int scalar(const StTensor* t, double* v, std::string* err) {
    const int64_t n = t->shape.empty() ? 1 : t->numel();
    if (n != 1 || !t->data)
        return fail(err, RAD_E_FORMAT, fmt("%s is not one number", t->name.c_str()));
    if (float_dtype(t->dtype)) {
        *v = elem(*t, 0);
    } else if (t->dtype == RAD_I64 && t->bytes >= 8) {
        int64_t x; std::memcpy(&x, t->data, 8); *v = (double)x;
    } else if (t->dtype == RAD_I32 && t->bytes >= 4) {
        int32_t x; std::memcpy(&x, t->data, 4); *v = (double)x;
    } else if (t->dtype_str == "F64" && t->bytes >= 8) {
        std::memcpy(v, t->data, 8);
    } else {
        return fail(err, RAD_E_FORMAT, fmt("%s is %s, not a number this reader knows",
                                           t->name.c_str(), t->dtype_str.c_str()));
    }
    return std::isfinite(*v) ? RAD_OK : fail(err, RAD_E_FORMAT, t->name + " is not finite");
}

/* ---------------------------------------------------------------- dense f32 kernels */
/* C[m, n] (+)= A[m, k] B[k, n], row-major. */
void mm(const float* A, const float* B, float* C, int64_t m, int64_t k, int64_t n, bool add = false) {
    if (!add) std::fill(C, C + m * n, 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        float* c = C + i * n;
        for (int64_t p = 0; p < k; ++p) {
            const float a = A[i * k + p];
            if (a == 0.0f) continue;
            const float* b = B + p * n;
            for (int64_t j = 0; j < n; ++j) c[j] += a * b[j];
        }
    }
}
/* C[m, n] (+)= A[k, m]^T B[k, n]. */
void mm_tn(const float* A, const float* B, float* C, int64_t m, int64_t k, int64_t n, bool add = false) {
    if (!add) std::fill(C, C + m * n, 0.0f);
    for (int64_t p = 0; p < k; ++p) {
        const float* a = A + p * m;
        const float* b = B + p * n;
        for (int64_t i = 0; i < m; ++i) {
            const float x = a[i];
            if (x == 0.0f) continue;
            float* c = C + i * n;
            for (int64_t j = 0; j < n; ++j) c[j] += x * b[j];
        }
    }
}

/* Orthonormalise the l columns of Y [n, l] in place (modified Gram-Schmidt, twice). A column
 * that vanishes is left zero. */
void orth(float* Y, int64_t n, int64_t l) {
    std::vector<double> col((size_t)n);
    for (int64_t j = 0; j < l; ++j) {
        for (int pass = 0; pass < 2; ++pass)
            for (int64_t i = 0; i < j; ++i) {
                double d = 0.0;
                for (int64_t r = 0; r < n; ++r) d += (double)Y[r * l + i] * Y[r * l + j];
                for (int64_t r = 0; r < n; ++r) Y[r * l + j] -= (float)(d * Y[r * l + i]);
            }
        double nn = 0.0;
        for (int64_t r = 0; r < n; ++r) nn += (double)Y[r * l + j] * Y[r * l + j];
        const double inv = nn > 1e-30 ? 1.0 / std::sqrt(nn) : 0.0;
        for (int64_t r = 0; r < n; ++r) Y[r * l + j] = (float)(Y[r * l + j] * inv);
    }
}

/* Eigen-decomposition of a symmetric l x l matrix (cyclic Jacobi): eigenvalues in `w`, sorted
 * descending, and eigenvectors as the columns of V [l, l]. */
void sym_eig(std::vector<double> G, int64_t l, std::vector<double>* w, std::vector<double>* V) {
    V->assign((size_t)(l * l), 0.0);
    for (int64_t i = 0; i < l; ++i) (*V)[(size_t)(i * l + i)] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0, tot = 0.0;
        for (int64_t i = 0; i < l; ++i)
            for (int64_t j = 0; j < l; ++j) {
                const double g = G[(size_t)(i * l + j)];
                tot += g * g;
                if (i != j) off += g * g;
            }
        if (off <= 1e-24 * (tot > 0 ? tot : 1.0)) break;
        for (int64_t p = 0; p < l; ++p)
            for (int64_t q = p + 1; q < l; ++q) {
                const double gpq = G[(size_t)(p * l + q)];
                if (std::fabs(gpq) < 1e-300) continue;
                const double gpp = G[(size_t)(p * l + p)], gqq = G[(size_t)(q * l + q)];
                const double th = (gqq - gpp) / (2.0 * gpq);
                const double t = (th >= 0 ? 1.0 : -1.0) / (std::fabs(th) + std::sqrt(th * th + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int64_t k = 0; k < l; ++k) {
                    const double gkp = G[(size_t)(k * l + p)], gkq = G[(size_t)(k * l + q)];
                    G[(size_t)(k * l + p)] = c * gkp - s * gkq;
                    G[(size_t)(k * l + q)] = s * gkp + c * gkq;
                }
                for (int64_t k = 0; k < l; ++k) {
                    const double gpk = G[(size_t)(p * l + k)], gqk = G[(size_t)(q * l + k)];
                    G[(size_t)(p * l + k)] = c * gpk - s * gqk;
                    G[(size_t)(q * l + k)] = s * gpk + c * gqk;
                }
                for (int64_t k = 0; k < l; ++k) {
                    const double vkp = (*V)[(size_t)(k * l + p)], vkq = (*V)[(size_t)(k * l + q)];
                    (*V)[(size_t)(k * l + p)] = c * vkp - s * vkq;
                    (*V)[(size_t)(k * l + q)] = s * vkp + c * vkq;
                }
            }
    }
    std::vector<int64_t> ord((size_t)l);
    for (int64_t i = 0; i < l; ++i) ord[(size_t)i] = i;
    std::sort(ord.begin(), ord.end(), [&](int64_t a, int64_t b) {
        return G[(size_t)(a * l + a)] > G[(size_t)(b * l + b)];
    });
    std::vector<double> V2((size_t)(l * l));
    w->resize((size_t)l);
    for (int64_t c = 0; c < l; ++c) {
        (*w)[(size_t)c] = G[(size_t)(ord[(size_t)c] * l + ord[(size_t)c])];
        for (int64_t r = 0; r < l; ++r) V2[(size_t)(r * l + c)] = (*V)[(size_t)(r * l + ord[(size_t)c])];
    }
    *V = std::move(V2);
}

/* ||B A||_F^2 for B [n, r], A [r, k]: trace((B^T B)(A A^T)). */
double lowrank_energy(const std::vector<float>& B, const std::vector<float>& A, int64_t n, int64_t r,
                      int64_t k) {
    std::vector<double> bb((size_t)(r * r), 0.0), aa((size_t)(r * r), 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t p = 0; p < r; ++p) {
            const double x = B[(size_t)(i * r + p)];
            if (x == 0.0) continue;
            for (int64_t q = 0; q < r; ++q) bb[(size_t)(p * r + q)] += x * B[(size_t)(i * r + q)];
        }
    for (int64_t p = 0; p < r; ++p)
        for (int64_t q = p; q < r; ++q) {
            double d = 0.0;
            for (int64_t j = 0; j < k; ++j) d += (double)A[(size_t)(p * k + j)] * A[(size_t)(q * k + j)];
            aa[(size_t)(p * r + q)] = aa[(size_t)(q * r + p)] = d;
        }
    double e = 0.0;
    for (size_t i = 0; i < bb.size(); ++i) e += bb[i] * aa[i];
    return e;
}

/* ---------------------------------------------------------------- the key suffixes */
struct Suffix {
    const char* s;
    int kind;      /* FileModule::Kind, or -1 alpha, -2 refused, -3 skipped */
    int slot;
    const char* why;
};
const Suffix kSuffix[] = {
    { ".lora_down.weight", FileModule::LORA, 0, nullptr },
    { ".lora_up.weight", FileModule::LORA, 1, nullptr },
    { ".lora_A.default.weight", FileModule::LORA, 0, nullptr },
    { ".lora_B.default.weight", FileModule::LORA, 1, nullptr },
    { ".lora_A.weight", FileModule::LORA, 0, nullptr },
    { ".lora_B.weight", FileModule::LORA, 1, nullptr },
    { ".lora_linear_layer.down.weight", FileModule::LORA, 0, nullptr },
    { ".lora_linear_layer.up.weight", FileModule::LORA, 1, nullptr },
    { ".lora.down.weight", FileModule::LORA, 0, nullptr },
    { ".lora.up.weight", FileModule::LORA, 1, nullptr },
    { ".hada_w1_a", FileModule::LOHA, 0, nullptr },
    { ".hada_w1_b", FileModule::LOHA, 1, nullptr },
    { ".hada_w2_a", FileModule::LOHA, 2, nullptr },
    { ".hada_w2_b", FileModule::LOHA, 3, nullptr },
    { ".lokr_w1", FileModule::LOKR, 0, nullptr },
    { ".lokr_w1_a", FileModule::LOKR, 1, nullptr },
    { ".lokr_w1_b", FileModule::LOKR, 2, nullptr },
    { ".lokr_w2", FileModule::LOKR, 3, nullptr },
    { ".lokr_w2_a", FileModule::LOKR, 4, nullptr },
    { ".lokr_w2_b", FileModule::LOKR, 5, nullptr },
    { ".diff", FileModule::FULL, 0, nullptr },
    { ".alpha", -1, 0, nullptr },
    { ".lora_alpha", -1, 0, nullptr },
    { ".dora_scale", -2, 0, "a DoRA magnitude, which rescales the base weight's columns" },
    { ".lora_magnitude_vector.default.weight", -2, 0, "a DoRA magnitude, which rescales the base "
                                                      "weight's columns" },
    { ".lora_magnitude_vector.weight", -2, 0, "a DoRA magnitude, which rescales the base weight's "
                                              "columns" },
    { ".lora_magnitude_vector", -2, 0, "a DoRA magnitude, which rescales the base weight's columns" },
    { ".hada_t1", -2, 0, "a LoHa Tucker core" },
    { ".hada_t2", -2, 0, "a LoHa Tucker core" },
    { ".lokr_t2", -2, 0, "a LoKr Tucker core" },
    { ".lora_mid.weight", -2, 0, "a LoCon Tucker core" },
    { ".oft_blocks", -2, 0, "an OFT rotation" },
    { ".oft_diag", -2, 0, "an OFT rotation" },
    { ".oft_R.weight", -2, 0, "an OFT rotation" },
    { ".boft_R", -2, 0, "a BOFT rotation" },
    { ".diff_b", -3, 0, "a bias delta" },
    { ".w_norm", -3, 0, "a norm's weight delta" },
    { ".b_norm", -3, 0, "a norm's bias delta" },
    { ".lora_B.bias", -3, 0, "a bias on lora_B" },
    { ".lora_up.bias", -3, 0, "a bias on lora_up" },
    { ".set_weight", -2, 0, "a replacement weight" },
};

}  /* namespace */

/* ================================================================== the file */
int AdapterFile::open(const std::string& path, std::string* err) {
    mods_.clear();
    skipped_.clear();
    if (st_.open(path) != RAD_OK) return fail(err, RAD_E_IO, path + ": cannot open as safetensors");
    std::map<std::string, FileModule> by;
    std::map<std::string, const StTensor*> alphas;
    for (const StTensor& t : st_.tensors()) {
        const Suffix* hit = nullptr;
        for (const Suffix& s : kSuffix)
            if (t.name.ends_with(s.s) && (!hit || std::strlen(s.s) > std::strlen(hit->s))) hit = &s;
        if (!hit) {
            skipped_.push_back({ t.name, "not an adapter tensor" });
            continue;
        }
        const std::string key = t.name.substr(0, t.name.size() - std::strlen(hit->s));
        if (hit->kind == -2)
            return fail(err, RAD_E_UNSUPPORTED, fmt("%s: %s is %s; that kind of adapter is not "
                                                    "served", path.c_str(), t.name.c_str(), hit->why));
        if (hit->kind == -3) { skipped_.push_back({ t.name, hit->why }); continue; }
        if (hit->kind == -1) { alphas[key] = &t; continue; }
        FileModule& m = by[key];
        if (m.key.empty()) { m.key = key; m.kind = (FileModule::Kind)hit->kind; }
        else if (m.kind != hit->kind)
            return fail(err, RAD_E_FORMAT, fmt("%s: module %s mixes adapter kinds", path.c_str(),
                                               key.c_str()));
        if (m.t[hit->slot])
            return fail(err, RAD_E_FORMAT, fmt("%s: module %s has two %s tensors", path.c_str(),
                                               key.c_str(), hit->s));
        m.t[hit->slot] = &t;
    }
    for (auto& [k, a] : alphas) {
        auto it = by.find(k);
        if (it == by.end()) { skipped_.push_back({ a->name, "an alpha of no module" }); continue; }
        it->second.alpha = a;
    }
    for (auto& [k, m] : by) {
        const char* missing = nullptr;
        switch (m.kind) {
            case FileModule::LORA:
                if (!m.t[0] || !m.t[1]) missing = m.t[0] ? "up/B" : "down/A";
                break;
            case FileModule::LOHA:
                if (!m.t[0] || !m.t[1] || !m.t[2] || !m.t[3]) missing = "a hada_w factor";
                break;
            case FileModule::LOKR:
                if (!m.t[0] && !(m.t[1] && m.t[2])) missing = "lokr_w1 (or lokr_w1_a and _b)";
                else if (!m.t[3] && !(m.t[4] && m.t[5])) missing = "lokr_w2 (or lokr_w2_a and _b)";
                break;
            case FileModule::FULL: break;
        }
        if (missing)
            return fail(err, RAD_E_FORMAT, fmt("%s: module %s has no %s", path.c_str(), k.c_str(),
                                               missing));
        mods_.push_back(m);
    }
    if (mods_.empty()) return fail(err, RAD_E_FORMAT, path + ": no adapter modules in the file");
    return RAD_OK;
}

const char* AdapterFile::kind_name() const {
    int n[4] = { 0, 0, 0, 0 };
    for (const FileModule& m : mods_) ++n[m.kind];
    const int k = (int)(std::max_element(n, n + 4) - n);
    static const char* const kName[] = { "LoRA", "LoHa", "LoKr", "full" };
    return kName[k];
}

/* ================================================================== deltas */
int module_delta(const FileModule& m, double w, Delta* out, std::string* err) {
    *out = Delta{};
    double alpha = 0.0;
    const bool has_alpha = m.alpha != nullptr;
    if (has_alpha) TRY(scalar(m.alpha, &alpha, err));

    if (m.kind == FileModule::LORA) {
        int64_t r = 0, K = 0, N = 0, r2 = 0;
        std::vector<float> A, B;
        TRY(as_matrix(m.t[0], "down", &r, &K, &A, err));
        TRY(as_matrix(m.t[1], "up", &N, &r2, &B, err));
        if (!trailing_ones(m.t[1]) || r2 != r)
            return fail(err, RAD_E_SHAPE, fmt("module %s: down %s and up %s are not one rank-%lld "
                                              "pair", m.key.c_str(), shape_str(m.t[0]->shape).c_str(),
                                              shape_str(m.t[1]->shape).c_str(), (long long)r));
        const double s = w * (has_alpha && r > 0 ? alpha / (double)r : 1.0);
        for (float& x : B) x = (float)(x * s);
        out->N = N; out->K = K; out->r = r; out->rank_bound = r; out->low_rank = true;
        out->A = std::move(A); out->B = std::move(B);
        out->energy = lowrank_energy(out->B, out->A, N, r, K);
        return RAD_OK;
    }

    if (m.kind == FileModule::LOHA) {
        int64_t N1 = 0, r1 = 0, r1b = 0, K1 = 0, N2 = 0, r2 = 0, r2b = 0, K2 = 0;
        std::vector<float> w1a, w1b, w2a, w2b;
        TRY(as_matrix(m.t[0], "hada_w1_a", &N1, &r1, &w1a, err));
        TRY(as_matrix(m.t[1], "hada_w1_b", &r1b, &K1, &w1b, err));
        TRY(as_matrix(m.t[2], "hada_w2_a", &N2, &r2, &w2a, err));
        TRY(as_matrix(m.t[3], "hada_w2_b", &r2b, &K2, &w2b, err));
        if (r1 != r1b || r2 != r2b || N1 != N2 || K1 != K2)
            return fail(err, RAD_E_SHAPE, fmt("module %s: LoHa factors %s %s %s %s do not pair",
                                              m.key.c_str(), shape_str(m.t[0]->shape).c_str(),
                                              shape_str(m.t[1]->shape).c_str(),
                                              shape_str(m.t[2]->shape).c_str(),
                                              shape_str(m.t[3]->shape).c_str()));
        /* (w1a w1b) o (w2a w2b) = sum over (i, j) of (w1a[:, i] o w2a[:, j]) (w1b[i] o w2b[j]):
         * exactly rank r1 r2. */
        const double s = w * (has_alpha && r1 > 0 ? alpha / (double)r1 : 1.0);
        const int64_t R = r1 * r2, N = N1, K = K1;
        out->B.assign((size_t)(N * R), 0.0f);
        out->A.assign((size_t)(R * K), 0.0f);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t i = 0; i < r1; ++i)
                for (int64_t j = 0; j < r2; ++j)
                    out->B[(size_t)(n * R + i * r2 + j)] =
                        (float)(s * w1a[(size_t)(n * r1 + i)] * w2a[(size_t)(n * r2 + j)]);
        for (int64_t i = 0; i < r1; ++i)
            for (int64_t j = 0; j < r2; ++j)
                for (int64_t k = 0; k < K; ++k)
                    out->A[(size_t)((i * r2 + j) * K + k)] = w1b[(size_t)(i * K + k)] * w2b[(size_t)(j * K + k)];
        out->N = N; out->K = K; out->r = R; out->rank_bound = R; out->low_rank = true;
        out->energy = lowrank_energy(out->B, out->A, N, R, K);
        return RAD_OK;
    }

    if (m.kind == FileModule::LOKR) {
        /* kron(w1 [a, b], w2 [c, e]) is [a c, b e]: element (i1 c + i2, j1 e + j2) is
         * w1[i1, j1] w2[i2, j2]. A convolution's w2 [c, d, kh, kw] flattens to e = d kh kw, which
         * is the flattened kron of the 4-D tensors. */
        int64_t a = 0, b = 0, dim = 0;
        std::vector<float> W1;
        if (m.t[0]) {
            TRY(as_matrix(m.t[0], "lokr_w1", &a, &b, &W1, err));
        } else {
            int64_t r = 0, r2 = 0;
            std::vector<float> x, y;
            TRY(as_matrix(m.t[1], "lokr_w1_a", &a, &r, &x, err));
            TRY(as_matrix(m.t[2], "lokr_w1_b", &r2, &b, &y, err));
            if (r != r2) return fail(err, RAD_E_SHAPE, "module " + m.key + ": lokr_w1_a/_b ranks differ");
            W1.resize((size_t)(a * b));
            mm(x.data(), y.data(), W1.data(), a, r, b);
            dim = r;
        }
        int64_t c = 0, e = 0, r2 = 0;
        auto W2 = std::make_shared<std::vector<float>>();
        auto W2a = std::make_shared<std::vector<float>>();
        auto W2b = std::make_shared<std::vector<float>>();
        if (m.t[3]) {
            TRY(as_matrix(m.t[3], "lokr_w2", &c, &e, W2.get(), err));
        } else {
            int64_t rr = 0;
            TRY(as_matrix(m.t[4], "lokr_w2_a", &c, &r2, W2a.get(), err));
            TRY(as_matrix(m.t[5], "lokr_w2_b", &rr, &e, W2b.get(), err));
            if (rr != r2) return fail(err, RAD_E_SHAPE, "module " + m.key + ": lokr_w2_a/_b ranks differ");
            dim = r2;
        }
        const double s = w * (has_alpha && dim > 0 ? alpha / (double)dim : 1.0);
        for (float& x : W1) x = (float)(x * s);
        auto w1 = std::make_shared<std::vector<float>>(std::move(W1));
        const int64_t N = a * c, K = b * e;
        out->N = N; out->K = K;
        out->rank_bound = std::min(a, b) * (r2 ? r2 : std::min(c, e));
        const bool full2 = !W2->empty();
        /* ||kron(w1, w2)|| = ||w1|| ||w2|| */
        {
            double e1 = 0.0;
            for (float x : *w1) e1 += (double)x * x;
            double e2 = 0.0;
            if (full2) for (float x : *W2) e2 += (double)x * x;
            else e2 = lowrank_energy(*W2a, *W2b, c, r2, e);
            out->energy = e1 * e2;
        }
        /* Column x of X [K, l], as the matrix [b, e]: kron(w1, w2) x = w1 X w2^T, read as [a, c]. */
        out->apply = [=](const float* X, int64_t l, float* Y) {
            std::vector<float> xm((size_t)(b * e)), t((size_t)(b * c)), y((size_t)(a * c)), u;
            for (int64_t col = 0; col < l; ++col) {
                for (int64_t k = 0; k < K; ++k) xm[(size_t)k] = X[k * l + col];
                if (full2) {
                    /* t[j1, i2] = sum_j2 xm[j1, j2] w2[i2, j2] */
                    for (int64_t j1 = 0; j1 < b; ++j1)
                        for (int64_t i2 = 0; i2 < c; ++i2) {
                            float d = 0.0f;
                            const float* xr = xm.data() + j1 * e;
                            const float* wr = W2->data() + i2 * e;
                            for (int64_t j2 = 0; j2 < e; ++j2) d += xr[j2] * wr[j2];
                            t[(size_t)(j1 * c + i2)] = d;
                        }
                } else {
                    u.assign((size_t)(b * r2), 0.0f);
                    for (int64_t j1 = 0; j1 < b; ++j1)
                        for (int64_t p = 0; p < r2; ++p) {
                            float d = 0.0f;
                            const float* xr = xm.data() + j1 * e;
                            const float* wr = W2b->data() + p * e;
                            for (int64_t j2 = 0; j2 < e; ++j2) d += xr[j2] * wr[j2];
                            u[(size_t)(j1 * r2 + p)] = d;
                        }
                    for (int64_t j1 = 0; j1 < b; ++j1)
                        for (int64_t i2 = 0; i2 < c; ++i2) {
                            float d = 0.0f;
                            for (int64_t p = 0; p < r2; ++p) d += u[(size_t)(j1 * r2 + p)] * (*W2a)[(size_t)(i2 * r2 + p)];
                            t[(size_t)(j1 * c + i2)] = d;
                        }
                }
                mm(w1->data(), t.data(), y.data(), a, b, c);
                for (int64_t n = 0; n < N; ++n) Y[n * l + col] = y[(size_t)n];
            }
        };
        /* The transpose: kron(w1, w2)^T y = w1^T Y w2, Y the column as [a, c], read as [b, e]. */
        out->apply_t = [=](const float* Yin, int64_t l, float* X) {
            std::vector<float> ym((size_t)(a * c)), t((size_t)(b * c)), x((size_t)(b * e)), u;
            for (int64_t col = 0; col < l; ++col) {
                for (int64_t n = 0; n < N; ++n) ym[(size_t)n] = Yin[n * l + col];
                mm_tn(w1->data(), ym.data(), t.data(), b, a, c);           /* [b, c] */
                if (full2) {
                    mm(t.data(), W2->data(), x.data(), b, c, e);
                } else {
                    u.assign((size_t)(b * r2), 0.0f);
                    mm(t.data(), W2a->data(), u.data(), b, c, r2);
                    mm(u.data(), W2b->data(), x.data(), b, r2, e);
                }
                for (int64_t k = 0; k < K; ++k) X[k * l + col] = x[(size_t)k];
            }
        };
        return RAD_OK;
    }

    /* FULL */
    int64_t N = 0, K = 0;
    auto D = std::make_shared<std::vector<float>>();
    TRY(as_matrix(m.t[0], "diff", &N, &K, D.get(), err));
    for (float& x : *D) x = (float)(x * w);
    out->N = N; out->K = K; out->rank_bound = std::min(N, K);
    out->energy = 0.0;
    for (float x : *D) out->energy += (double)x * x;
    out->apply = [D, N, K](const float* X, int64_t l, float* Y) { mm(D->data(), X, Y, N, K, l); };
    out->apply_t = [D, N, K](const float* Yin, int64_t l, float* X) { mm_tn(D->data(), Yin, X, K, N, l); };
    return RAD_OK;
}

void truncate(const Delta& d, int64_t r, std::vector<float>* B, std::vector<float>* A, double* kept) {
    const int64_t N = d.N, K = d.K;
    r = std::max<int64_t>(0, std::min({ r, N, K, d.rank_bound > 0 ? d.rank_bound : r }));
    if (kept) *kept = 1.0;
    if (d.low_rank && d.r <= r) { *B = d.B; *A = d.A; return; }
    auto fwd = [&](const float* X, int64_t l, float* Y) {
        if (!d.low_rank) { d.apply(X, l, Y); return; }
        std::vector<float> t((size_t)(d.r * l));
        mm(d.A.data(), X, t.data(), d.r, K, l);
        mm(d.B.data(), t.data(), Y, N, d.r, l);
    };
    auto bwd = [&](const float* Y, int64_t l, float* X) {
        if (!d.low_rank) { d.apply_t(Y, l, X); return; }
        std::vector<float> t((size_t)(d.r * l));
        mm_tn(d.B.data(), Y, t.data(), d.r, N, l);
        mm_tn(d.A.data(), t.data(), X, K, d.r, l);
    };
    const int64_t l = std::min({ r + 8, N, K });
    /* A fixed sketch: the same adapter composes to the same bytes on every rank and every load. */
    std::vector<float> Om((size_t)(K * l));
    uint64_t st = 0x2545F4914F6CDD1Dull ^ (uint64_t)(N * 1315423911ull + K);
    for (float& x : Om) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        x = (float)((double)(st >> 11) / (double)(1ull << 53) - 0.5);
    }
    std::vector<float> Y((size_t)(N * l)), Z((size_t)(K * l));
    fwd(Om.data(), l, Y.data());
    orth(Y.data(), N, l);
    for (int it = 0; it < 2; ++it) {
        bwd(Y.data(), l, Z.data());
        orth(Z.data(), K, l);
        fwd(Z.data(), l, Y.data());
        orth(Y.data(), N, l);
    }
    /* C^T = D^T Q [K, l]; the SVD of C = Q^T D from the eigen-decomposition of C C^T. */
    bwd(Y.data(), l, Z.data());
    std::vector<double> G((size_t)(l * l), 0.0);
    for (int64_t k = 0; k < K; ++k)
        for (int64_t i = 0; i < l; ++i) {
            const double x = Z[(size_t)(k * l + i)];
            if (x == 0.0) continue;
            for (int64_t j = i; j < l; ++j) G[(size_t)(i * l + j)] += x * Z[(size_t)(k * l + j)];
        }
    for (int64_t i = 0; i < l; ++i)
        for (int64_t j = 0; j < i; ++j) G[(size_t)(i * l + j)] = G[(size_t)(j * l + i)];
    std::vector<double> w, V;
    sym_eig(G, l, &w, &V);
    double tot = 0.0, top = 0.0;
    for (int64_t i = 0; i < l; ++i) tot += std::max(0.0, w[(size_t)i]);
    for (int64_t i = 0; i < r; ++i) top += std::max(0.0, w[(size_t)i]);
    if (d.energy > 0.0) tot = std::max(tot, d.energy);
    if (kept) *kept = tot > 0.0 ? std::min(1.0, top / tot) : 1.0;
    B->assign((size_t)(N * r), 0.0f);
    A->assign((size_t)(r * K), 0.0f);
    for (int64_t i = 0; i < r; ++i) {
        const double sg = std::sqrt(std::max(0.0, w[(size_t)i]));
        if (sg <= 1e-20) continue;
        /* B[:, i] = Q V[:, i] sigma_i; A[i] = V[:, i]^T C / sigma_i. */
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t j = 0; j < l; ++j) acc += (double)Y[(size_t)(n * l + j)] * V[(size_t)(j * l + i)];
            (*B)[(size_t)(n * r + i)] = (float)(acc * sg);
        }
        for (int64_t k = 0; k < K; ++k) {
            double acc = 0.0;
            for (int64_t j = 0; j < l; ++j) acc += V[(size_t)(j * l + i)] * Z[(size_t)(k * l + j)];
            (*A)[(size_t)(i * K + k)] = (float)(acc / sg);
        }
    }
}

/* ================================================================== the key map */
namespace {

std::string underscored(std::string s) {
    for (char& c : s) if (c == '.') c = '_';
    return s;
}

std::regex pattern(const std::string& from, bool kohya) {
    std::string re = "^";
    for (char c : kohya ? underscored(from) : from) {
        if (c == '#') re += "([0-9]+)";
        else if (std::strchr(".^$|()[]{}*+?\\", c)) { re += '\\'; re += c; }
        else re += c;
    }
    return std::regex(re + "$", std::regex::ECMAScript | std::regex::optimize);
}

std::string module_of(const std::string& src) {
    return src.ends_with(".weight") ? src.substr(0, src.size() - 7) : std::string();
}

}  /* namespace */

KeyMap::KeyMap(const Program& prog, const std::vector<Site>& sites) : prog_(&prog) {
    for (const Site& s : sites)
        for (size_t i = 0; i < s.sources.size(); ++i) {
            const SiteSource& src = s.sources[i];
            if (!src.targeted) continue;
            const size_t at = src.module.find('@');
            if (at != std::string::npos) {
                /* A row piece: the module stays adaptable, its rows open (the adapter's say). */
                Piece p;
                p.name = src.module;
                p.lo = std::strtoll(src.module.c_str() + at + 1, nullptr, 10);
                const size_t plus = src.module.find('+', at);
                p.n = plus == std::string::npos ? 0 : std::strtoll(src.module.c_str() + plus + 1, nullptr, 10);
                const std::string real = src.module.substr(0, at);
                std::vector<Piece>& ps = pieces_[real];
                bool dup = false;
                for (const Piece& q : ps) dup = dup || q.name == p.name;
                if (!dup) ps.push_back(p);
                adaptable_[real] = { -1, s.K_full > 0 ? s.K_full : -1 };
                continue;
            }
            whole_.insert(src.module);
            int64_t rows = -1;
            if (src.part_lo >= 0) {
                rows = 0;
                for (int g = src.part_lo; g < src.part_hi; ++g) {
                    const int64_t t = s.part_total[(size_t)g];
                    if (t <= 0) { rows = -1; break; }
                    rows += t;
                }
            } else if (s.sources.size() == 1 && s.part_total.size() == 1 && s.part_total[0] > 0) {
                rows = s.part_total[0];
            }
            adaptable_[src.module] = { rows, s.K_full > 0 ? s.K_full : -1 };
        }
    for (const LoraRoot& r : prog.lora_roots) {
        Root R;
        R.native = r.native;
        R.spellings = r.spellings;
        R.spellings.push_back(r.native);
        std::sort(R.spellings.begin(), R.spellings.end(),
                  [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
        R.spellings.erase(std::unique(R.spellings.begin(), R.spellings.end()), R.spellings.end());
        roots_.push_back(std::move(R));
    }
    for (const RadNameMap& m : prog.name_map)
        for (int k = 0; k < m.n_src && k < 8; ++k) {
            if (!m.src[k]) continue;
            const std::string mod = module_of(m.src[k]);
            if (mod.empty()) continue;
            for (Root& R : roots_) {
                if (mod.compare(0, R.native.size(), R.native) != 0) continue;
                if (!adaptable_.count(mod) && !known_.count(mod))
                    known_[mod] = "the base reads it with an op no adapter path follows (a fused "
                                  "activation, a rotated input, or not a GEMM)";
                const std::string u = underscored(mod.substr(R.native.size()));
                auto it = R.kohya.find(u);
                if (it == R.kohya.end()) R.kohya[u] = mod;
                else if (it->second != mod) it->second.clear();   /* ambiguous */
            }
        }
    for (const LoraAlias& a : prog.lora_aliases) {
        for (size_t ri = 0; ri < roots_.size(); ++ri) {
            if (roots_[ri].native != a.root) continue;
            Alias x;
            x.root = ri;
            x.from = a.from; x.to = a.to;
            x.from_lo = a.from_lo; x.from_hi = a.from_hi; x.to_lo = a.to_lo;
            x.dotted = pattern(a.from, false);
            x.kohya = pattern(a.from, true);
            aliases_.push_back(std::move(x));
        }
    }
}

int64_t KeyMap::rows(const std::string& module) const {
    auto it = adaptable_.find(module);
    return it == adaptable_.end() ? -1 : it->second.first;
}
const std::vector<KeyMap::Piece>& KeyMap::pieces(const std::string& module) const {
    static const std::vector<Piece> none;
    auto it = pieces_.find(module);
    return it == pieces_.end() ? none : it->second;
}
int64_t KeyMap::cols(const std::string& module) const {
    auto it = adaptable_.find(module);
    return it == adaptable_.end() ? -1 : it->second.second;
}

bool KeyMap::lookup(const Root& r, const std::string& rel, bool kohya, std::vector<Target>* out,
                    std::string* why) const {
    auto take = [&](const std::string& mod, int64_t lo, int64_t hi, int64_t to) {
        if (adaptable_.count(mod)) {
            Target t;
            t.module = mod; t.from_lo = lo; t.from_hi = hi; t.to_lo = to;
            out->push_back(t);
            return true;
        }
        auto k = known_.find(mod);
        if (k != known_.end() && why) *why = mod + ": " + k->second;
        return false;
    };
    if (kohya) {
        auto it = r.kohya.find(rel);
        if (it != r.kohya.end()) {
            if (it->second.empty()) { if (why) *why = "its kohya name matches two modules of the base"; }
            else if (take(it->second, 0, 0, 0)) return true;
        }
    } else if (take(r.native + rel, 0, 0, 0)) {
        return true;
    }
    const size_t ri = (size_t)(&r - roots_.data());
    for (const Alias& a : aliases_) {
        if (a.root != ri) continue;
        std::smatch m;
        if (!std::regex_match(rel, m, kohya ? a.kohya : a.dotted)) continue;
        std::string to;
        size_t g = 1;
        for (char c : a.to) {
            if (c == '#' && g < m.size()) to += m[g++].str();
            else to += c;
        }
        take(r.native + to, a.from_lo, a.from_hi, a.to_lo);
    }
    return !out->empty();
}

std::vector<Target> KeyMap::resolve(const std::string& key, std::string* why) const {
    struct Cand { size_t root; size_t len; bool kohya; };
    std::vector<Cand> cs;
    for (size_t ri = 0; ri < roots_.size(); ++ri)
        for (const std::string& sp : roots_[ri].spellings)
            if (key.compare(0, sp.size(), sp) == 0)
                cs.push_back({ ri, sp.size(), !sp.empty() && sp.back() == '_' });
    std::stable_sort(cs.begin(), cs.end(), [](const Cand& a, const Cand& b) { return a.len > b.len; });
    std::vector<Target> out;
    std::string first_why;
    for (const Cand& c : cs) {
        std::string w;
        if (lookup(roots_[c.root], key.substr(c.len), c.kohya, &out, &w)) return out;
        if (first_why.empty()) first_why = w;
    }
    if (why) *why = !first_why.empty() ? first_why
                    : cs.empty() ? std::string("its prefix names no network of this model")
                                 : std::string("no module of this model is named so");
    return out;
}

/* ================================================================== composition */
namespace {

struct Part {
    std::shared_ptr<Delta> d;
    int64_t from_lo = 0, n = 0, to_lo = 0;
};

}  /* namespace */

int compose(const KeyMap& km, const std::vector<Strength>& ads, int max_rank,
            std::map<std::string, Composite>* out, ComposeReport* rep, std::string* err) {
    out->clear();
    *rep = ComposeReport{};
    std::map<std::string, std::vector<Part>> by;
    for (const Strength& s : ads) {
        if (!s.file) return fail(err, RAD_E_INVAL, "adapter '" + s.name + "' is not open");
        for (const auto& [t, why] : s.file->skipped()) rep->skipped.push_back(s.name + ": " + t + ": " + why);
        if (s.weight == 0.0) continue;
        for (const FileModule& fm : s.file->modules()) {
            std::string why;
            const std::vector<Target> ts = km.resolve(fm.key, &why);
            if (ts.empty()) { rep->unplaced.push_back(s.name + ": " + fm.key + ": " + why); continue; }
            auto d = std::make_shared<Delta>();
            std::string e;
            if (module_delta(fm, s.weight, d.get(), &e) < 0)
                return fail(err, RAD_E_FORMAT, s.name + ": " + e);
            for (const Target& t : ts) {
                Part p;
                p.d = d;
                p.from_lo = t.from_lo;
                p.n = (t.from_hi > 0 ? t.from_hi : d->N) - t.from_lo;
                p.to_lo = t.to_lo;
                if (p.n <= 0 || t.from_lo + p.n > d->N)
                    return fail(err, RAD_E_SHAPE, fmt("%s: %s has %lld output rows; the base's alias "
                                                      "takes rows [%lld, %lld) of it for %s",
                                                      s.name.c_str(), fm.key.c_str(), (long long)d->N,
                                                      (long long)t.from_lo, (long long)(t.from_lo + p.n),
                                                      t.module.c_str()));
                by[t.module].push_back(p);
            }
        }
    }

    std::vector<std::string> names;
    for (auto& [k, v] : by) { names.push_back(k); (*out)[k]; }
    std::vector<std::string> errs(names.size());
    std::vector<double> kept(names.size(), 1.0);
    std::vector<char> exact(names.size(), 1);
    std::atomic<size_t> next{ 0 };
    auto work = [&]() {
        for (size_t i; (i = next.fetch_add(1)) < names.size();) {
            const std::string& mod = names[i];
            const std::vector<Part>& ps = by.find(mod)->second;
            Composite& c = out->find(mod)->second;
            int64_t rows = km.rows(mod), K = km.cols(mod);
            int64_t extent = 0;
            for (const Part& p : ps) extent = std::max(extent, p.to_lo + p.n);
            for (const KeyMap::Piece& pc : km.pieces(mod)) extent = std::max(extent, pc.lo + pc.n);
            if (rows < 0) rows = extent;
            if (K < 0) K = ps[0].d->K;
            bool lr = true;
            int64_t R = 0, bound = 0;
            for (const Part& p : ps) {
                if (p.d->K != K) {
                    errs[i] = fmt("%s: an adapter's delta has %lld input columns, the base's module "
                                  "%lld", mod.c_str(), (long long)p.d->K, (long long)K);
                    break;
                }
                if (p.to_lo + p.n > rows) {
                    errs[i] = fmt("%s: an adapter's delta reaches row %lld of a module of %lld rows",
                                  mod.c_str(), (long long)(p.to_lo + p.n), (long long)rows);
                    break;
                }
                lr = lr && p.d->low_rank;
                R += p.d->low_rank ? p.d->r : 0;
                bound += p.d->rank_bound;
            }
            if (!errs[i].empty()) continue;
            c.rows = rows; c.K = K;
            if (lr && R <= max_rank) {
                /* Side by side: every delta its own block of ranks. */
                c.r = R;
                c.B.assign((size_t)(rows * R), 0.0f);
                c.A.assign((size_t)(R * K), 0.0f);
                int64_t o = 0;
                for (const Part& p : ps) {
                    const Delta& d = *p.d;
                    for (int64_t row = 0; row < p.n; ++row)
                        std::memcpy(&c.B[(size_t)((p.to_lo + row) * R + o)],
                                    &d.B[(size_t)((p.from_lo + row) * d.r)], (size_t)d.r * 4);
                    std::memcpy(&c.A[(size_t)(o * K)], d.A.data(), (size_t)(d.r * K) * 4);
                    o += d.r;
                }
                continue;
            }
            /* The sum, as one operator over the module's rows, truncated to the slot's rank. */
            exact[i] = 0;
            Delta sum;
            sum.N = rows; sum.K = K; sum.rank_bound = bound;
            /* One whole delta's energy is known; a sum's or a row slice's is what the sketch
             * holds, exact when its rank is within the sketch's. */
            sum.energy = ps.size() == 1 && ps[0].from_lo == 0 && ps[0].n == ps[0].d->N
                             ? ps[0].d->energy : -1.0;
            sum.apply = [&ps, rows](const float* X, int64_t l, float* Y) {
                std::fill(Y, Y + rows * l, 0.0f);
                for (const Part& p : ps) {
                    const Delta& d = *p.d;
                    std::vector<float> y((size_t)(d.N * l));
                    if (d.low_rank) {
                        std::vector<float> t((size_t)(d.r * l));
                        mm(d.A.data(), X, t.data(), d.r, d.K, l);
                        mm(d.B.data(), t.data(), y.data(), d.N, d.r, l);
                    } else {
                        d.apply(X, l, y.data());
                    }
                    for (int64_t row = 0; row < p.n; ++row)
                        for (int64_t j = 0; j < l; ++j)
                            Y[(p.to_lo + row) * l + j] += y[(size_t)((p.from_lo + row) * l + j)];
                }
            };
            sum.apply_t = [&ps, K](const float* Yin, int64_t l, float* X) {
                std::fill(X, X + K * l, 0.0f);
                std::vector<float> x((size_t)(K * l));
                for (const Part& p : ps) {
                    const Delta& d = *p.d;
                    std::vector<float> y((size_t)(d.N * l), 0.0f);
                    for (int64_t row = 0; row < p.n; ++row)
                        for (int64_t j = 0; j < l; ++j)
                            y[(size_t)((p.from_lo + row) * l + j)] = Yin[(p.to_lo + row) * l + j];
                    if (d.low_rank) {
                        std::vector<float> t((size_t)(d.r * l));
                        mm_tn(d.B.data(), y.data(), t.data(), d.r, d.N, l);
                        mm_tn(d.A.data(), t.data(), x.data(), d.K, d.r, l);
                    } else {
                        d.apply_t(y.data(), l, x.data());
                    }
                    for (int64_t k = 0; k < K * l; ++k) X[k] += x[(size_t)k];
                }
            };
            truncate(sum, max_rank, &c.B, &c.A, &kept[i]);
            c.r = (int64_t)(c.A.size() / (size_t)(K > 0 ? K : 1));
            /* Kept energy is exact when the sum is low-rank: ||B A||^2 of the parts side by side. */
            if (lr) {
                std::vector<float> Bc((size_t)(rows * R), 0.0f), Ac((size_t)(R * K), 0.0f);
                int64_t o = 0;
                for (const Part& p : ps) {
                    const Delta& d = *p.d;
                    for (int64_t row = 0; row < p.n; ++row)
                        std::memcpy(&Bc[(size_t)((p.to_lo + row) * R + o)],
                                    &d.B[(size_t)((p.from_lo + row) * d.r)], (size_t)d.r * 4);
                    std::memcpy(&Ac[(size_t)(o * K)], d.A.data(), (size_t)(d.r * K) * 4);
                    o += d.r;
                }
                const double tot = lowrank_energy(Bc, Ac, rows, R, K);
                const double got = lowrank_energy(c.B, c.A, rows, c.r, K);
                kept[i] = tot > 0.0 ? std::min(1.0, got / tot) : 1.0;
            }
        }
    };
    const unsigned hw = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned t = 1; t < hw && t < names.size(); ++t) th.emplace_back(work);
    work();
    for (std::thread& t : th) t.join();
    for (size_t i = 0; i < names.size(); ++i) {
        if (!errs[i].empty()) return fail(err, RAD_E_SHAPE, errs[i]);
        ++rep->modules;
        /* Held exactly: side by side, or an operator whose rank fits (LoKr, LoHa within R). */
        if (exact[i] || kept[i] >= 0.9999) ++rep->exact;
        else if (kept[i] < rep->worst_kept) { rep->worst_kept = kept[i]; rep->worst_module = names[i]; }
    }
    return RAD_OK;
}

std::map<std::string, AdModule> as_modules(std::map<std::string, Composite>& c, const KeyMap& km) {
    std::map<std::string, AdModule> m;
    for (auto& [name, x] : c) {
        if (x.r <= 0) continue;
        const std::vector<KeyMap::Piece>& pcs = km.pieces(name);
        x.tPieces.assign(pcs.size(), StTensor{});
        x.tA = StTensor{};
        x.tA.name = name + ".A";
        x.tA.dtype = RAD_F32;
        x.tA.dtype_str = "F32";
        x.tA.shape = { x.r, x.K };
        x.tA.bytes = x.A.size() * 4;
        x.tA.data = (const uint8_t*)x.A.data();
        x.tB = StTensor{};
        x.tB.name = name + ".B";
        x.tB.dtype = RAD_F32;
        x.tB.dtype_str = "F32";
        x.tB.shape = { x.rows, x.r };
        x.tB.bytes = x.B.size() * 4;
        x.tB.data = (const uint8_t*)x.B.data();
        AdModule a;
        a.name = name;
        a.A = &x.tA;
        a.B = &x.tB;
        a.rank = x.r;
        a.alpha = (double)x.r;
        a.scale = 1.0f;
        if (pcs.empty() || km.whole(name)) m[name] = a;
        for (size_t i = 0; i < pcs.size(); ++i) {
            StTensor& t = x.tPieces[i];
            t = x.tB;
            t.name = pcs[i].name + ".B";
            t.shape = { pcs[i].n, x.r };
            t.bytes = (uint64_t)(pcs[i].n * x.r) * 4;
            t.data = (const uint8_t*)(x.B.data() + pcs[i].lo * x.r);
            AdModule p = a;
            p.name = pcs[i].name;
            p.B = &t;
            m[p.name] = p;
        }
    }
    return m;
}

}  /* namespace lora */
}  /* namespace rad */
