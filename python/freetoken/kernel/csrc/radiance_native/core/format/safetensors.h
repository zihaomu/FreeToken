/* safetensors.h -- an 8-byte header length, a JSON object mapping tensor name to {dtype, shape,
 * data_offsets}, and raw little-endian bytes. A small fixed-shape parser rather than a dependency,
 * since the converter must run wherever a checkpoint lands.
 *
 * Shards are mmapped, not copied.
 */
#pragma once
#include "../rad_internal.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

struct StTensor {
    std::string          name;
    uint32_t             dtype = RAD_DT_INVALID;   /* RAD_* ; RAD_DT_INVALID if unrecognised */
    std::string          dtype_str;                /* as written, for the diagnostic */
    std::vector<int64_t> shape;
    uint64_t             offset = 0;               /* relative to the data section */
    uint64_t             bytes  = 0;
    const uint8_t*       data   = nullptr;

    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

class SafeTensorsFile {
public:
    SafeTensorsFile() = default;
    ~SafeTensorsFile();
    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    int  open(const std::string& path);
    void close();

    const StTensor*              find(const std::string& name) const;
    const std::vector<StTensor>& tensors()  const { return tensors_; }
    const std::string&           metadata() const { return metadata_; }
    const std::string&           path()     const { return path_; }

private:
    int         fd_ = -1;
    uint8_t*    map_ = nullptr;
    uint64_t    map_bytes_ = 0;
    std::string path_, metadata_;
    std::vector<StTensor> tensors_;
    std::unordered_map<std::string, size_t> index_;
};

/* model.safetensors.index.json -> { tensor name: shard filename }. A single-file checkpoint has no
 * index; the caller then opens model.safetensors. */
int st_load_index(const std::string& index_path,
                  std::unordered_map<std::string, std::string>* out);

}  /* namespace rad */
