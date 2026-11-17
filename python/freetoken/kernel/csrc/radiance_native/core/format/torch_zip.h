/* torch_zip.h -- a PyTorch checkpoint (torch.save's zip format) as tensors, read without running
 * any of its pickle.
 *
 * A .pth from torch.save is a zip archive of stored entries: `<archive>/data.pkl`, a pickle of the
 * state dict whose tensors name their bytes by a persistent id, `<archive>/data/<key>` holding
 * each storage's raw little-endian bytes, and `<archive>/byteorder`. Fish Audio's codec ships this
 * way (drama-3's codec.pth), with no safetensors copy.
 *
 * Unpickling would run whatever the file names. This reads the pickle as data instead: a small
 * machine over the opcodes a weights-only file uses, whose only callables are the ones torch's own
 * `weights_only` loader allows a state dict (collections.OrderedDict,
 * torch._utils._rebuild_tensor_v2 and _rebuild_parameter) and whose only classes are the typed
 * storages. Any other global, any other opcode, a compressed or encrypted entry, or a storage that
 * does not hold its tensor is refused by name. Every read is bounds-checked and the machine's
 * stack, memo and object counts are capped.
 *
 * Storages are mmapped, not copied; a tensor must be contiguous in its storage. Nested dicts
 * flatten to dotted names, and values that are not tensors (an epoch count) are skipped.
 */
#pragma once
#include "../rad_internal.h"

#include <string>
#include <vector>

namespace rad {

struct TorchTensor {
    std::string          name;
    uint32_t             dtype = RAD_DT_INVALID;
    std::string          storage;               /* the torch storage class, for diagnostics */
    std::vector<int64_t> shape;                 /* row-major */
    const uint8_t*       data = nullptr;
    uint64_t             bytes = 0;
};

class TorchZipFile {
public:
    TorchZipFile() = default;
    ~TorchZipFile();
    TorchZipFile(const TorchZipFile&) = delete;
    TorchZipFile& operator=(const TorchZipFile&) = delete;

    /* RAD_OK, or a negative RAD_E_* with `why` saying what was refused. */
    int  open(const std::string& path, std::string* why);
    /* The same over bytes the caller keeps alive (a test's archive in memory). */
    int  open_bytes(const uint8_t* data, uint64_t bytes, std::string* why);
    void close();

    const std::vector<TorchTensor>& tensors() const { return tensors_; }
    const std::string&              path() const { return path_; }

private:
    int parse(std::string* why);

    int            fd_ = -1;
    uint8_t*       map_ = nullptr;
    const uint8_t* base_ = nullptr;
    uint64_t       bytes_ = 0;
    std::string    path_;
    std::vector<TorchTensor> tensors_;
};

}  /* namespace rad */
