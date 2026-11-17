/* gguf.h -- a reader for the GGUF container.
 *
 * A rewrite of `ggml/src/gguf.cpp` (spec §12), not a lift: that file is also a writer and a
 * ggml_context builder (it fills `struct ggml_tensor` and needs `ggml.h`), and §12 excludes ggml.
 * The format is a header, a KV table, a tensor directory and a padded data blob. `vendor/UPSTREAM`
 * records the file it was written against.
 *
 * The only thing taken from upstream is data: the block geometry of every ggml dtype, needed to
 * compute a tensor's byte size (GgufDtype).
 *
 * The file is mmapped whole and never copied. rad-convert walks the directory, reads tensors
 * through `tensor_data()` and writes the .rad; nothing here interprets tensor content.
 */
#pragma once
#include "rad_internal.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rad {

/* GGUF's value type numbering (wire values, fixed by the format). */
enum GgufType {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4, GGUF_I32 = 5,
    GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STR = 8, GGUF_ARR = 9, GGUF_U64 = 10,
    GGUF_I64 = 11, GGUF_F64 = 12
};

/* One metadata entry. A scalar is stored as a one-element vector, so callers read scalars and
 * arrays the same way without branching on `is_array`. */
struct GgufKV {
    std::string key;
    uint32_t    type = GGUF_U8;   /* for an array, the element type; never GGUF_ARR */
    bool        is_array = false;

    /* By `type`: integers and bools widen into `i`, floats into `f`, strings into `s`. The
     * other two stay empty. */
    std::vector<int64_t>     i;
    std::vector<double>      f;
    std::vector<std::string> s;
};

/* One tensor directory row. `offset` is relative to the data section; the file offset is
 * `data_off() + offset` (see `tensor_data()`). */
struct GgufTensor {
    std::string name;
    uint32_t    ggml_type = 0;
    uint32_t    n_dims = 0;
    int64_t     ne[4] = { 1, 1, 1, 1 };   /* GGUF is fixed at 4 dims, unused ones are 1 */
    uint64_t    offset = 0;
    uint64_t    bytes = 0;                /* computed from ne and the block geometry */
};

/* Block geometry for a ggml dtype, transcribed from ggml.c's `type_traits` at llama.cpp
 * 06938ac12 (vendor/UPSTREAM). `blck` 0 marks a removed or unknown type; open() refuses a tensor
 * of that type rather than mis-size it. */
struct GgufDtype {
    int         blck;        /* logical elements per stored block */
    size_t      block_bytes; /* bytes per stored block */
    const char* name;
};
const GgufDtype* gguf_dtype(uint32_t ggml_type);   /* null if out of range */

/* Map a ggml dtype onto RAD_* where the bytes agree exactly. RAD_DT_INVALID for block-quantised
 * types (k-quants, iq-quants), which rad-convert must dequantise or refuse, never reinterpret. */
uint32_t gguf_to_rad_dtype(uint32_t ggml_type);

class GgufFile {
public:
    GgufFile() = default;
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    /* RAD_OK, RAD_E_IO if the file will not open or map, or RAD_E_FORMAT if the magic, version
     * or directory is invalid. */
    int  open(const char* path);
    void close();
    bool is_open() const { return base_ != nullptr; }

    uint32_t version()   const { return version_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t data_off()  const { return data_off_; }     /* file offset of the tensor blob */
    uint64_t file_bytes() const { return file_bytes_; }
    const void* base()   const { return base_; }         /* the mmap, or null */
    const std::string& path() const { return path_; }

    const std::vector<GgufKV>&     kv()      const { return kv_; }
    const std::vector<GgufTensor>& tensors() const { return tensors_; }

    const GgufKV*     find(std::string_view key)  const;
    const GgufTensor* tensor(std::string_view name) const;

    /* Typed reads. Each returns false and leaves *out untouched when the key is absent or holds
     * the wrong kind of value; a caller with a default treats both the same. */
    bool get_i64(std::string_view key, int64_t* out) const;
    bool get_f64(std::string_view key, double* out) const;
    bool get_str(std::string_view key, std::string* out) const;
    bool get_strs(std::string_view key, std::vector<std::string>* out) const;
    bool get_i64s(std::string_view key, std::vector<int64_t>* out) const;
    bool get_f32s(std::string_view key, std::vector<float>* out) const;

    /* A pointer into the mmap, or null if the tensor's extent leaves the file (a corrupt
     * container, which open() already refuses). */
    const void* tensor_data(const GgufTensor& t) const;

private:
    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t  file_bytes_ = 0;
    uint32_t  version_ = 0;
    uint64_t  alignment_ = 32;
    uint64_t  data_off_ = 0;

    std::vector<GgufKV>     kv_;
    std::vector<GgufTensor> tensors_;
    std::unordered_map<std::string, size_t> kv_index_;
    std::unordered_map<std::string, size_t> tensor_index_;
};

}  /* namespace rad */
