/* safetensors.cpp -- the reader and its small JSON scanner. Not a general JSON parser: the header
 * is one flat object of objects with three known keys, so strings, integer arrays and nested braces
 * suffice. Anything else is an error naming the byte offset.
 */
#include "safetensors.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rad {
namespace {

/* ------------------------------------------------------------------ a scanner, not a parser */
struct Scan {
    const char* p;
    const char* end;
    std::string err;

    void ws() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p; }
    bool eat(char c) { ws(); if (p < end && *p == c) { ++p; return true; } return false; }
    bool peek(char c) { ws(); return p < end && *p == c; }
    bool fail(const char* what, const char* base) {
        if (err.empty()) err = fmt("%s at byte %lld of the header", what, (long long)(p - base));
        return false;
    }

    /* A JSON string with the escapes that occur in these files; \u is decoded to UTF-8 because
     * tensor names are compared. */
    bool str(std::string* out, const char* base) {
        ws();
        if (p >= end || *p != '"') return fail("expected a string", base);
        ++p;
        out->clear();
        while (p < end && *p != '"') {
            if (*p != '\\') { out->push_back(*p++); continue; }
            if (++p >= end) return fail("string ends inside an escape", base);
            switch (*p++) {
                case '"':  out->push_back('"');  break;
                case '\\': out->push_back('\\'); break;
                case '/':  out->push_back('/');  break;
                case 'b':  out->push_back('\b'); break;
                case 'f':  out->push_back('\f'); break;
                case 'n':  out->push_back('\n'); break;
                case 'r':  out->push_back('\r'); break;
                case 't':  out->push_back('\t'); break;
                case 'u': {
                    if (end - p < 4) return fail("truncated \\u escape", base);
                    char hex[5] = { p[0], p[1], p[2], p[3], 0 };
                    unsigned cp = (unsigned)std::strtoul(hex, nullptr, 16);
                    p += 4;
                    /* Join a surrogate pair (\uD800-\uDBFF \uDC00-\uDFFF) into one codepoint;
                     * encoding the halves separately is CESU-8, which compares unequal to the same
                     * name in UTF-8. An unpaired surrogate takes the three-byte path rather than
                     * being rejected: names are matched, not validated. */
                    if (cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' &&
                        p[1] == 'u') {
                        char lo_hex[5] = { p[2], p[3], p[4], p[5], 0 };
                        const unsigned lo = (unsigned)std::strtoul(lo_hex, nullptr, 16);
                        if (lo >= 0xDC00 && lo < 0xE000) {
                            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                            p += 6;
                        }
                    }
                    if (cp < 0x80) out->push_back((char)cp);
                    else if (cp < 0x800) {
                        out->push_back((char)(0xC0 | (cp >> 6)));
                        out->push_back((char)(0x80 | (cp & 0x3F)));
                    } else if (cp < 0x10000) {
                        out->push_back((char)(0xE0 | (cp >> 12)));
                        out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                        out->push_back((char)(0x80 | (cp & 0x3F)));
                    } else {
                        out->push_back((char)(0xF0 | (cp >> 18)));
                        out->push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
                        out->push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                        out->push_back((char)(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default: return fail("unknown escape", base);
            }
        }
        if (p >= end) return fail("unterminated string", base);
        ++p;
        return true;
    }

    bool i64(int64_t* out, const char* base) {
        ws();
        const char* s = p;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        if (p >= end || *p < '0' || *p > '9') return fail("expected an integer", base);
        while (p < end && *p >= '0' && *p <= '9') ++p;
        *out = std::strtoll(std::string(s, p - s).c_str(), nullptr, 10);
        return true;
    }

    /* Skip one value of any type, so unknown keys in a tensor object are tolerated. */
    bool skip(const char* base) {
        ws();
        if (p >= end) return fail("truncated", base);
        if (*p == '"') { std::string t; return str(&t, base); }
        if (*p == '{' || *p == '[') {
            const char open = *p, close = (open == '{') ? '}' : ']';
            int depth = 0;
            while (p < end) {
                if (*p == '"') { std::string t; if (!str(&t, base)) return false; continue; }
                if (*p == open) ++depth;
                else if (*p == close && --depth == 0) { ++p; return true; }
                ++p;
            }
            return fail("unbalanced brackets", base);
        }
        while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
        return true;
    }
};

/* safetensors dtype names to RAD dtypes. */
uint32_t st_dtype(const std::string& s) {
    if (s == "BF16") return RAD_BF16;
    if (s == "F16")  return RAD_F16;
    if (s == "F32")  return RAD_F32;
    if (s == "I8")   return RAD_I8;
    if (s == "U8")   return RAD_U8;
    if (s == "I16")  return RAD_I16;
    if (s == "I32")  return RAD_I32;
    if (s == "I64")  return RAD_I64;
    if (s == "U32")  return RAD_U32;
    if (s == "BOOL") return RAD_BOOL;
    if (s == "F8_E4M3") return RAD_F8E4M3;
    if (s == "F8_E5M2") return RAD_F8E5M2;
    /* F64 and F8_E8M0 have no RAD_ equivalent; refused by name at use rather than approximated. */
    return RAD_DT_INVALID;
}

}  /* namespace */

SafeTensorsFile::~SafeTensorsFile() { close(); }

void SafeTensorsFile::close() {
    if (map_) ::munmap(map_, (size_t)map_bytes_);
    if (fd_ >= 0) ::close(fd_);
    map_ = nullptr; map_bytes_ = 0; fd_ = -1;
    tensors_.clear(); index_.clear(); metadata_.clear(); path_.clear();
}

int SafeTensorsFile::open(const std::string& path) {
    close();
    path_ = path;

    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { RAD_ERR("%s: %s", path.c_str(), std::strerror(errno)); return RAD_E_IO; }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) { RAD_ERR("%s: %s", path.c_str(), std::strerror(errno));
                                  close(); return RAD_E_IO; }
    map_bytes_ = (uint64_t)st.st_size;
    if (map_bytes_ < 8) {
        RAD_ERR("%s: %llu bytes -- too short to be safetensors", path.c_str(),
                (unsigned long long)map_bytes_);
        close();
        return RAD_E_FORMAT;
    }

    void* m = ::mmap(nullptr, (size_t)map_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { RAD_ERR("%s: mmap: %s", path.c_str(), std::strerror(errno));
                           close(); return RAD_E_IO; }
    map_ = (uint8_t*)m;

    uint64_t hlen = 0;
    std::memcpy(&hlen, map_, 8);
    if (hlen == 0 || hlen > map_bytes_ - 8) {
        RAD_ERR("%s: header length %llu does not fit in a %llu-byte file", path.c_str(),
                (unsigned long long)hlen, (unsigned long long)map_bytes_);
        close();
        return RAD_E_FORMAT;
    }
    const uint8_t* data_base = map_ + 8 + hlen;
    const uint64_t data_bytes = map_bytes_ - 8 - hlen;

    const char* base = (const char*)map_ + 8;
    Scan sc{ base, base + hlen, {} };

    if (!sc.eat('{')) { RAD_ERR("%s: the header is not a JSON object", path.c_str());
                        close(); return RAD_E_FORMAT; }

    while (!sc.peek('}')) {
        std::string name;
        if (!sc.str(&name, base) || !sc.eat(':')) {
            RAD_ERR("%s: %s", path.c_str(), sc.err.empty() ? "malformed header" : sc.err.c_str());
            close();
            return RAD_E_FORMAT;
        }
        if (name == "__metadata__") {
            const char* s = sc.p;
            if (!sc.skip(base)) { RAD_ERR("%s: %s", path.c_str(), sc.err.c_str());
                                  close(); return RAD_E_FORMAT; }
            metadata_.assign(s, sc.p - s);
            if (!sc.eat(',')) break;
            continue;
        }

        if (!sc.eat('{')) { RAD_ERR("%s: '%s' is not an object", path.c_str(), name.c_str());
                            close(); return RAD_E_FORMAT; }

        StTensor t;
        t.name = name;
        int64_t off_begin = -1, off_end = -1;

        while (!sc.peek('}')) {
            std::string key;
            if (!sc.str(&key, base) || !sc.eat(':')) {
                RAD_ERR("%s: malformed entry for '%s'", path.c_str(), name.c_str());
                close();
                return RAD_E_FORMAT;
            }
            if (key == "dtype") {
                if (!sc.str(&t.dtype_str, base)) { RAD_ERR("%s: %s", path.c_str(), sc.err.c_str());
                                                   close(); return RAD_E_FORMAT; }
                t.dtype = st_dtype(t.dtype_str);
            } else if (key == "shape") {
                if (!sc.eat('[')) { RAD_ERR("%s: '%s'.shape is not an array", path.c_str(),
                                            name.c_str()); close(); return RAD_E_FORMAT; }
                while (!sc.peek(']')) {
                    int64_t d;
                    if (!sc.i64(&d, base)) { RAD_ERR("%s: %s", path.c_str(), sc.err.c_str());
                                             close(); return RAD_E_FORMAT; }
                    t.shape.push_back(d);
                    if (!sc.eat(',')) break;
                }
                if (!sc.eat(']')) { RAD_ERR("%s: '%s'.shape is unterminated", path.c_str(),
                                            name.c_str()); close(); return RAD_E_FORMAT; }
            } else if (key == "data_offsets") {
                if (!sc.eat('[') || !sc.i64(&off_begin, base) || !sc.eat(',') ||
                    !sc.i64(&off_end, base) || !sc.eat(']')) {
                    RAD_ERR("%s: '%s'.data_offsets is not [begin, end]", path.c_str(),
                            name.c_str());
                    close();
                    return RAD_E_FORMAT;
                }
            } else if (!sc.skip(base)) {
                RAD_ERR("%s: %s", path.c_str(), sc.err.c_str());
                close();
                return RAD_E_FORMAT;
            }
            if (!sc.eat(',')) break;
        }
        if (!sc.eat('}')) { RAD_ERR("%s: entry for '%s' is unterminated", path.c_str(),
                                    name.c_str()); close(); return RAD_E_FORMAT; }

        if (off_begin < 0 || off_end < off_begin || (uint64_t)off_end > data_bytes) {
            RAD_ERR("%s: '%s' spans [%lld, %lld) of a %llu-byte data section -- truncated",
                    path.c_str(), name.c_str(), (long long)off_begin, (long long)off_end,
                    (unsigned long long)data_bytes);
            close();
            return RAD_E_FORMAT;
        }
        t.offset = (uint64_t)off_begin;
        t.bytes  = (uint64_t)(off_end - off_begin);
        t.data   = data_base + t.offset;

        /* The shape must match the span: readers walk numel() elements from `data`, not
         * data_offsets, so an oversized shape reads the next tensor or past the mapping. Extents
         * are untrusted, so bound each multiply against wrap. An empty shape is a scalar. Unknown
         * dtypes are not sized here; consumers refuse them by name. */
        int64_t elems = 1;
        bool    shape_ok = true;
        for (int64_t d : t.shape) if (d < 0) shape_ok = false; else if (d == 0) elems = 0;
        for (int64_t d : t.shape) {
            if (!shape_ok || elems == 0) break;
            if (elems > INT64_MAX / d) { shape_ok = false; break; }
            elems *= d;
        }
        const int64_t want = shape_ok && t.dtype != RAD_DT_INVALID
                                 ? (elems > INT64_MAX / 64 ? -1 : rad_dtype_bytes(t.dtype, elems))
                                 : 0;
        if (!shape_ok || want < 0 ||
            (t.dtype != RAD_DT_INVALID && (uint64_t)want != t.bytes)) {
            std::string sh;
            for (size_t k = 0; k < t.shape.size(); ++k)
                sh += (k ? ", " : "") + std::to_string(t.shape[k]);
            RAD_ERR("%s: '%s' is %s [%s], which is not the %llu bytes its data_offsets span. The "
                    "header disagrees with its own data; the file is damaged or was written by a "
                    "tool that does not follow the format -- fetch or export it again.",
                    path.c_str(), name.c_str(), t.dtype_str.c_str(), sh.c_str(),
                    (unsigned long long)t.bytes);
            close();
            return RAD_E_FORMAT;
        }

        /* A JSON object's keys are unique; a header naming one tensor twice has two answers to
         * find() and is refused rather than resolved to the first. */
        if (!index_.emplace(t.name, tensors_.size()).second) {
            RAD_ERR("%s: the header names '%s' twice", path.c_str(), name.c_str());
            close();
            return RAD_E_FORMAT;
        }
        tensors_.push_back(std::move(t));

        if (!sc.eat(',')) break;
    }
    if (!sc.eat('}')) {
        RAD_ERR("%s: the header object is unterminated", path.c_str());
        close();
        return RAD_E_FORMAT;
    }

    RAD_DEBUG("%s: %zu tensor(s), %s of data", path.c_str(), tensors_.size(),
              humanb((int64_t)data_bytes).c_str());
    return RAD_OK;
}

const StTensor* SafeTensorsFile::find(const std::string& name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

int st_load_index(const std::string& path, std::unordered_map<std::string, std::string>* out) {
    if (!out) return RAD_E_INVAL;
    out->clear();

    FILE* f = ::fopen(path.c_str(), "rb");
    if (!f) {
        if (errno == ENOENT) return RAD_E_NOTFOUND;
        RAD_ERR("%s: %s", path.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    std::string buf;
    char chunk[65536];
    size_t n;
    while ((n = ::fread(chunk, 1, sizeof chunk, f)) > 0) buf.append(chunk, n);
    ::fclose(f);

    /* One key matters: "weight_map". Find it, then read the flat string->string object. */
    size_t at = buf.find("\"weight_map\"");
    if (at == std::string::npos) {
        RAD_ERR("%s: no \"weight_map\" object", path.c_str());
        return RAD_E_FORMAT;
    }
    at = buf.find('{', at);
    if (at == std::string::npos) { RAD_ERR("%s: weight_map is not an object", path.c_str());
                                   return RAD_E_FORMAT; }

    const char* base = buf.data();
    Scan sc{ buf.data() + at, buf.data() + buf.size(), {} };
    if (!sc.eat('{')) return RAD_E_FORMAT;
    while (!sc.peek('}')) {
        std::string k, v;
        if (!sc.str(&k, base) || !sc.eat(':') || !sc.str(&v, base)) {
            RAD_ERR("%s: %s", path.c_str(), sc.err.empty() ? "malformed weight_map"
                                                           : sc.err.c_str());
            return RAD_E_FORMAT;
        }
        out->emplace(std::move(k), std::move(v));
        if (!sc.eat(',')) break;
    }
    RAD_DEBUG("%s: %zu tensor(s) across the shards", path.c_str(), out->size());
    return RAD_OK;
}

}  /* namespace rad */
