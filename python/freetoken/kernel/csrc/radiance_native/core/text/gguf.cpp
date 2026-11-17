#include "text/gguf.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rad {

/* ------------------------------------------------------------------ dtype geometry */
/* Produced by running ggml_blck_size() and ggml_type_size() over the whole enum at llama.cpp
 * 06938ac12, so these are the values the producer used, not the documented ones. Rows 42..44 are
 * local to that checkout and kept so files it wrote stay readable; upstream stops at 41. */
static const GgufDtype k_dtypes[] = {
    {   1,   4, "f32"     }, /*  0 */
    {   1,   2, "f16"     }, /*  1 */
    {  32,  18, "q4_0"    }, /*  2 */
    {  32,  20, "q4_1"    }, /*  3 */
    {   0,   0, nullptr   }, /*  4 removed */
    {   0,   0, nullptr   }, /*  5 removed */
    {  32,  22, "q5_0"    }, /*  6 */
    {  32,  24, "q5_1"    }, /*  7 */
    {  32,  34, "q8_0"    }, /*  8 */
    {  32,  36, "q8_1"    }, /*  9 */
    { 256,  84, "q2_K"    }, /* 10 */
    { 256, 110, "q3_K"    }, /* 11 */
    { 256, 144, "q4_K"    }, /* 12 */
    { 256, 176, "q5_K"    }, /* 13 */
    { 256, 210, "q6_K"    }, /* 14 */
    { 256, 292, "q8_K"    }, /* 15 */
    { 256,  66, "iq2_xxs" }, /* 16 */
    { 256,  74, "iq2_xs"  }, /* 17 */
    { 256,  98, "iq3_xxs" }, /* 18 */
    { 256,  50, "iq1_s"   }, /* 19 */
    {  32,  18, "iq4_nl"  }, /* 20 */
    { 256, 110, "iq3_s"   }, /* 21 */
    { 256,  82, "iq2_s"   }, /* 22 */
    { 256, 136, "iq4_xs"  }, /* 23 */
    {   1,   1, "i8"      }, /* 24 */
    {   1,   2, "i16"     }, /* 25 */
    {   1,   4, "i32"     }, /* 26 */
    {   1,   8, "i64"     }, /* 27 */
    {   1,   8, "f64"     }, /* 28 */
    { 256,  56, "iq1_m"   }, /* 29 */
    {   1,   2, "bf16"    }, /* 30 */
    {   0,   0, nullptr   }, /* 31 removed */
    {   0,   0, nullptr   }, /* 32 removed */
    {   0,   0, nullptr   }, /* 33 removed */
    { 256,  54, "tq1_0"   }, /* 34 */
    { 256,  66, "tq2_0"   }, /* 35 */
    {   0,   0, nullptr   }, /* 36 removed */
    {   0,   0, nullptr   }, /* 37 removed */
    {   0,   0, nullptr   }, /* 38 removed */
    {  32,  17, "mxfp4"   }, /* 39 */
    {  64,  36, "nvfp4"   }, /* 40 */
    { 128,  18, "q1_0"    }, /* 41 */
    {  32,  34, "rad4_q8" }, /* 42 */
    {  32,  20, "rad4_q4" }, /* 43 */
    { 256, 144, "rad4_q4k"}, /* 44 */
};
static const size_t k_n_dtypes = sizeof(k_dtypes) / sizeof(k_dtypes[0]);

const GgufDtype* gguf_dtype(uint32_t t) {
    if (t >= k_n_dtypes || k_dtypes[t].blck == 0) return nullptr;
    return &k_dtypes[t];
}

uint32_t gguf_to_rad_dtype(uint32_t t) {
    /* Block-quantised types are deliberately absent: a q4_K block reinterpreted as RAD_I4 would
     * load and compute, and be wrong. */
    switch (t) {
        case 0:  return RAD_F32;
        case 1:  return RAD_F16;
        case 24: return RAD_I8;
        case 25: return RAD_I16;
        case 26: return RAD_I32;
        case 27: return RAD_I64;
        case 30: return RAD_BF16;
        default: return RAD_DT_INVALID;
    }
}

/* ------------------------------------------------------------------ cursor */
/* A bounds-checked forward cursor over the mmap. Reads return false at the end of the file
 * instead of faulting, so a truncated GGUF gets an error message, not a SIGSEGV. */
namespace {
struct Cursor {
    const uint8_t* p = nullptr;
    uint64_t off = 0, end = 0;

    bool raw(void* dst, size_t n) {
        if (n > end - off) return false;
        std::memcpy(dst, p + off, n);
        off += n;
        return true;
    }
    template <class T> bool pod(T* v) { return raw(v, sizeof(T)); }

    bool str(std::string* out) {
        uint64_t n = 0;
        if (!pod(&n)) return false;
        if (n > end - off) return false;
        out->assign(reinterpret_cast<const char*>(p + off), (size_t)n);
        off += n;
        return true;
    }
    bool skip(uint64_t n) {
        if (n > end - off) return false;
        off += n;
        return true;
    }
};

size_t gguf_scalar_bytes(uint32_t t) {
    switch (t) {
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: return 1;
        case GGUF_U16: case GGUF_I16:               return 2;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: return 4;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: return 8;
        default: return 0;   /* STR and ARR are not fixed-width */
    }
}

/* Smallest encodings of a KV (key length, type, one value byte) and a tensor record (name length,
 * dimension count, type, offset). The header's counts are checked against them. */
constexpr uint64_t kMinKvBytes     = sizeof(uint64_t) + sizeof(uint32_t) + 1;
constexpr uint64_t kMinTensorBytes = sizeof(uint64_t) + 2 * sizeof(uint32_t) + sizeof(uint64_t);

/* Caps a reservation sized by a count from the file. Past the cap the table grows as entries are
 * read, so memory follows the file's content, not its claims. */
size_t reserve_hint(uint64_t n) { return (size_t)std::min<uint64_t>(n, 1u << 16); }

/* Widen one fixed-width scalar into the KV's integer or float lane. */
bool read_scalar(Cursor& c, uint32_t t, GgufKV& kv) {
    switch (t) {
        case GGUF_U8:   { uint8_t  v; if (!c.pod(&v)) return false; kv.i.push_back(v); return true; }
        case GGUF_I8:   { int8_t   v; if (!c.pod(&v)) return false; kv.i.push_back(v); return true; }
        case GGUF_BOOL: { int8_t   v; if (!c.pod(&v)) return false; kv.i.push_back(v ? 1 : 0); return true; }
        case GGUF_U16:  { uint16_t v; if (!c.pod(&v)) return false; kv.i.push_back(v); return true; }
        case GGUF_I16:  { int16_t  v; if (!c.pod(&v)) return false; kv.i.push_back(v); return true; }
        case GGUF_U32:  { uint32_t v; if (!c.pod(&v)) return false; kv.i.push_back(v); return true; }
        case GGUF_I32:  { int32_t  v; if (!c.pod(&v)) return false; kv.i.push_back(v); return true; }
        case GGUF_U64:  { uint64_t v; if (!c.pod(&v)) return false; kv.i.push_back((int64_t)v); return true; }
        case GGUF_I64:  { int64_t  v; if (!c.pod(&v)) return false; kv.i.push_back(v); return true; }
        case GGUF_F32:  { float    v; if (!c.pod(&v)) return false; kv.f.push_back(v); return true; }
        case GGUF_F64:  { double   v; if (!c.pod(&v)) return false; kv.f.push_back(v); return true; }
        case GGUF_STR:  { std::string s; if (!c.str(&s)) return false; kv.s.push_back(std::move(s)); return true; }
        default: return false;
    }
}
}  /* namespace */

/* ------------------------------------------------------------------ open */
GgufFile::~GgufFile() { close(); }

void GgufFile::close() {
    if (base_) {
        ::munmap(const_cast<uint8_t*>(base_), (size_t)file_bytes_);
        base_ = nullptr;
    }
    kv_.clear();
    tensors_.clear();
    kv_index_.clear();
    tensor_index_.clear();
    file_bytes_ = 0;
    data_off_ = 0;
    version_ = 0;
    alignment_ = 32;
    path_.clear();
}

int GgufFile::open(const char* path) {
    close();
    path_ = path ? path : "";

    int fd = ::open(path_.c_str(), O_RDONLY);
    if (fd < 0) {
        RAD_ERR("gguf: open %s: %s", path_.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        ::close(fd);
        RAD_ERR("gguf: stat %s: %s", path_.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    void* m = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);          /* the mapping keeps its own reference; the fd is not needed */
    if (m == MAP_FAILED) {
        RAD_ERR("gguf: mmap %s (%lld bytes): %s", path_.c_str(),
                (long long)st.st_size, std::strerror(errno));
        return RAD_E_IO;
    }
    base_ = static_cast<const uint8_t*>(m);
    file_bytes_ = (uint64_t)st.st_size;
    /* Every refusal from here on unmaps the file. */
    auto fail = [this] {
        close();
        return RAD_E_FORMAT;
    };

    Cursor c{ base_, 0, file_bytes_ };

    char magic[4] = { 0, 0, 0, 0 };
    uint64_t n_tensors = 0, n_kv = 0;
    if (!c.raw(magic, 4) || !c.pod(&version_) || !c.pod(&n_tensors) || !c.pod(&n_kv)) {
        RAD_ERR("gguf: %s is too short to hold a header", path_.c_str());
        return fail();
    }
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        RAD_ERR("gguf: %s is not a GGUF file (magic %02x%02x%02x%02x)", path_.c_str(),
                (unsigned)(uint8_t)magic[0], (unsigned)(uint8_t)magic[1],
                (unsigned)(uint8_t)magic[2], (unsigned)(uint8_t)magic[3]);
        return fail();
    }
    /* v1 used 32-bit counts and is no longer produced. v2 and v3 differ only in the signedness
     * of a tensor dimension, which is read as int64 either way. */
    if (version_ < 2 || version_ > 3) {
        RAD_ERR("gguf: %s is version %u; this reader handles 2 and 3", path_.c_str(), version_);
        return fail();
    }
    /* Both counts bound loops that fill heap tables, so refuse them unless the rest of the file
     * can hold every entry at its smallest encoding. */
    const uint64_t rest = c.end - c.off;
    if (n_kv > rest / kMinKvBytes || n_tensors > rest / kMinTensorBytes) {
        RAD_ERR("gguf: %s declares %llu KVs and %llu tensors in %llu bytes", path_.c_str(),
                (unsigned long long)n_kv, (unsigned long long)n_tensors,
                (unsigned long long)file_bytes_);
        return fail();
    }

    kv_.reserve(reserve_hint(n_kv));
    for (uint64_t k = 0; k < n_kv; ++k) {
        GgufKV e;
        uint32_t t = 0;
        if (!c.str(&e.key) || !c.pod(&t)) {
            RAD_ERR("gguf: %s: truncated in KV %llu", path_.c_str(), (unsigned long long)k);
            return fail();
        }
        uint64_t n = 1;
        if (t == GGUF_ARR) {
            e.is_array = true;
            if (!c.pod(&t) || !c.pod(&n)) {
                RAD_ERR("gguf: %s: truncated array header for '%s'", path_.c_str(), e.key.c_str());
                return fail();
            }
            if (t == GGUF_ARR) {
                RAD_ERR("gguf: %s: '%s' is an array of arrays, which GGUF does not define",
                        path_.c_str(), e.key.c_str());
                return fail();
            }
            /* Each element takes at least its width (a string, its length prefix), so a bogus
             * count is refused against the remaining bytes before anything is reserved for it.
             * An undefined element type is refused here for the same reason. */
            const size_t w = t == GGUF_STR ? sizeof(uint64_t) : gguf_scalar_bytes(t);
            if (!w) {
                RAD_ERR("gguf: %s: '%s' is an array of type %u, which GGUF does not define",
                        path_.c_str(), e.key.c_str(), t);
                return fail();
            }
            if (n > (c.end - c.off) / w) {
                RAD_ERR("gguf: %s: '%s' claims %llu elements, past the end of the file",
                        path_.c_str(), e.key.c_str(), (unsigned long long)n);
                return fail();
            }
        }
        e.type = t;
        if (t == GGUF_STR) e.s.reserve(reserve_hint(n));
        else if (t == GGUF_F32 || t == GGUF_F64) e.f.reserve(reserve_hint(n));
        else e.i.reserve(reserve_hint(n));

        for (uint64_t j = 0; j < n; ++j) {
            if (!read_scalar(c, t, e)) {
                RAD_ERR("gguf: %s: truncated value %llu of '%s' (type %u)", path_.c_str(),
                        (unsigned long long)j, e.key.c_str(), t);
                return fail();
            }
        }
        /* The first of duplicate keys wins, as in other readers. GGUF does not forbid
         * duplicates and a rewriter can leave one behind. */
        if (!kv_index_.count(e.key)) kv_index_[e.key] = kv_.size();
        kv_.push_back(std::move(e));
    }

    /* general.alignment moves the data section, so it is read before the directory's offsets.
     * The format's default is 32. A value that is not one integer power of two is refused, as
     * ggml's reader refuses it: no writer places the data section by it, and falling back to
     * the default would read every tensor from the wrong bytes. */
    if (const GgufKV* a = find("general.alignment")) {
        const int64_t al = !a->is_array && a->type != GGUF_BOOL && a->i.size() == 1 ? a->i[0] : 0;
        if (al <= 0 || (al & (al - 1)) != 0) {
            RAD_ERR("gguf: %s: general.alignment is not a power of two", path_.c_str());
            return fail();
        }
        alignment_ = (uint64_t)al;
    }

    tensors_.reserve(reserve_hint(n_tensors));
    for (uint64_t t = 0; t < n_tensors; ++t) {
        GgufTensor ti;
        if (!c.str(&ti.name) || !c.pod(&ti.n_dims)) {
            RAD_ERR("gguf: %s: truncated in tensor %llu", path_.c_str(), (unsigned long long)t);
            return fail();
        }
        if (ti.n_dims > 4) {
            RAD_ERR("gguf: %s: tensor '%s' has %u dimensions; GGUF allows 4",
                    path_.c_str(), ti.name.c_str(), ti.n_dims);
            return fail();
        }
        for (uint32_t d = 0; d < ti.n_dims; ++d) {
            if (!c.pod(&ti.ne[d]) || ti.ne[d] < 0) {
                RAD_ERR("gguf: %s: tensor '%s' has a bad dimension %u", path_.c_str(),
                        ti.name.c_str(), d);
                return fail();
            }
        }
        if (!c.pod(&ti.ggml_type) || !c.pod(&ti.offset)) {
            RAD_ERR("gguf: %s: truncated type/offset for '%s'", path_.c_str(), ti.name.c_str());
            return fail();
        }
        const GgufDtype* dt = gguf_dtype(ti.ggml_type);
        if (!dt) {
            RAD_ERR("gguf: %s: tensor '%s' has dtype %u, which this build does not know",
                    path_.c_str(), ti.name.c_str(), ti.ggml_type);
            return fail();
        }
        if (ti.ne[0] % dt->blck != 0) {
            RAD_ERR("gguf: %s: tensor '%s' row is %lld elements, not a multiple of the %s block "
                    "of %d", path_.c_str(), ti.name.c_str(), (long long)ti.ne[0], dt->name,
                    dt->blck);
            return fail();
        }
        /* The product of four file-supplied dimensions and its byte size can each wrap to a
         * small value that passes the bounds check below. Check both, and that the element
         * count fits int64, the type consumers of `ne` compute it in. */
        uint64_t nelem = 1;
        bool     wraps = false;
        for (int d = 0; d < 4; ++d)
            wraps |= __builtin_mul_overflow(nelem, (uint64_t)ti.ne[d], &nelem);
        wraps |= nelem > (uint64_t)INT64_MAX;
        wraps |= __builtin_mul_overflow(nelem / (uint64_t)dt->blck, (uint64_t)dt->block_bytes,
                                        &ti.bytes);
        if (wraps) {
            RAD_ERR("gguf: %s: tensor '%s' has dimensions %lld x %lld x %lld x %lld, whose size "
                    "does not fit in 64 bits", path_.c_str(), ti.name.c_str(),
                    (long long)ti.ne[0], (long long)ti.ne[1], (long long)ti.ne[2],
                    (long long)ti.ne[3]);
            return fail();
        }
        /* A second entry under one name could never be looked up. */
        if (tensor_index_.count(ti.name)) {
            RAD_ERR("gguf: %s: tensor '%s' is listed twice", path_.c_str(), ti.name.c_str());
            return fail();
        }
        tensor_index_[ti.name] = tensors_.size();
        tensors_.push_back(std::move(ti));
    }

    data_off_ = align_up((int64_t)c.off, (int64_t)alignment_);
    if (data_off_ > file_bytes_) {
        RAD_ERR("gguf: %s: data section starts at %llu, past the end of a %llu-byte file",
                path_.c_str(), (unsigned long long)data_off_, (unsigned long long)file_bytes_);
        return fail();
    }
    /* Every tensor must lie inside the file, so tensor_data() can hand out raw pointers, and
     * start on the file's alignment, as ggml's reader requires, so those pointers are aligned
     * for the vector loads of whatever reads them. */
    for (const GgufTensor& ti : tensors_) {
        if (ti.offset % alignment_ != 0) {
            RAD_ERR("gguf: %s: tensor '%s' starts at +%llu, off the file's %llu-byte alignment",
                    path_.c_str(), ti.name.c_str(), (unsigned long long)ti.offset,
                    (unsigned long long)alignment_);
            return fail();
        }
        if (ti.offset > file_bytes_ - data_off_ ||
            ti.bytes  > file_bytes_ - data_off_ - ti.offset) {
            RAD_ERR("gguf: %s: tensor '%s' (%llu bytes at +%llu) leaves the file",
                    path_.c_str(), ti.name.c_str(),
                    (unsigned long long)ti.bytes, (unsigned long long)ti.offset);
            return fail();
        }
    }

    RAD_DEBUG("gguf: %s v%u, %zu KVs, %zu tensors, data at +%llu, align %llu",
              path_.c_str(), version_, kv_.size(), tensors_.size(),
              (unsigned long long)data_off_, (unsigned long long)alignment_);
    return RAD_OK;
}

/* ------------------------------------------------------------------ lookup */
const GgufKV* GgufFile::find(std::string_view key) const {
    auto it = kv_index_.find(std::string(key));
    return it == kv_index_.end() ? nullptr : &kv_[it->second];
}

const GgufTensor* GgufFile::tensor(std::string_view name) const {
    auto it = tensor_index_.find(std::string(name));
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

bool GgufFile::get_i64(std::string_view key, int64_t* out) const {
    const GgufKV* e = find(key);
    if (!e || e->i.empty()) return false;
    *out = e->i[0];
    return true;
}

bool GgufFile::get_f64(std::string_view key, double* out) const {
    const GgufKV* e = find(key);
    if (!e || e->f.empty()) return false;
    *out = e->f[0];
    return true;
}

bool GgufFile::get_str(std::string_view key, std::string* out) const {
    const GgufKV* e = find(key);
    if (!e || e->s.empty()) return false;
    *out = e->s[0];
    return true;
}

bool GgufFile::get_strs(std::string_view key, std::vector<std::string>* out) const {
    const GgufKV* e = find(key);
    if (!e || e->type != GGUF_STR) return false;
    *out = e->s;
    return true;
}

bool GgufFile::get_i64s(std::string_view key, std::vector<int64_t>* out) const {
    const GgufKV* e = find(key);
    if (!e || e->i.empty()) return false;
    *out = e->i;
    return true;
}

bool GgufFile::get_f32s(std::string_view key, std::vector<float>* out) const {
    const GgufKV* e = find(key);
    if (!e || e->f.empty()) return false;
    out->assign(e->f.begin(), e->f.end());
    return true;
}

const void* GgufFile::tensor_data(const GgufTensor& t) const {
    if (!base_) return nullptr;
    if (t.offset > file_bytes_ - data_off_ || t.bytes > file_bytes_ - data_off_ - t.offset)
        return nullptr;
    return base_ + data_off_ + t.offset;
}

}  /* namespace rad */
