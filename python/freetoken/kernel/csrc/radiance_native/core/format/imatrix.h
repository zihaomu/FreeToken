/* imatrix.h -- llama.cpp's importance matrix, in both formats it has written.
 *
 * An imatrix is a per-column activation second moment over a calibration set. Two unrelated uses:
 *
 *   - the quantiser weights its per-block error by it, so bits go where activations are
 *     (spec §4.5);
 *   - the converter derives the expert-popularity profile from its counts, the planner's warm
 *     start (spec §5.3). Only a warm start: the heat engine re-ranks from real routing within a
 *     few dispatches.
 *
 * Formats:
 *
 *   legacy  flat little-endian binary: int32 n_entries, then per entry int32 name_len,
 *           char[name_len], int32 ncall, int32 nval, float[nval]. Whether values are already
 *           divided by ncall varies by writer version.
 *   GGUF    a GGUF v3 container with tensors `<name>.in_sum2` (f32 [cols] or [cols,n_expert]) and
 *           `<name>.counts` (f32 [1] or [1,n_expert]), plus `imatrix.chunk_count`,
 *           `imatrix.chunk_size` and `imatrix.datasets`. Only this format has per-expert data.
 *
 * The GGUF walker is local rather than core/text's: it is small, and keeps the quantiser's input
 * independent of the tokeniser's reader.
 */
#pragma once
#include "../rad_internal.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* For a GGUF imatrix, `sum2` and `counts` point into the mapping at general.alignment (possibly 1),
 * so importance() and count() read them with memcpy, never in place. */
struct ImatrixEntry {
    std::string  name;              /* the checkpoint tensor name, suffix stripped */
    const float* sum2   = nullptr;  /* [n_expert][cols], expert-major and contiguous */
    const float* counts = nullptr;  /* [n_expert], or null in the legacy format */
    int64_t      cols     = 0;
    int64_t      n_expert = 1;      /* 1 for a dense tensor */
    int64_t      ncall    = 0;      /* legacy only: calls folded into sum2 */
};

class ImatrixFile {
public:
    ImatrixFile() = default;
    ~ImatrixFile();
    ImatrixFile(const ImatrixFile&) = delete;
    ImatrixFile& operator=(const ImatrixFile&) = delete;

    /* Sniffs the magic and picks a reader. RAD_E_FORMAT if it is neither. */
    int  open(const std::string& path);
    void close();

    const ImatrixEntry*              find(const std::string& name) const;
    const std::vector<ImatrixEntry>& entries() const { return entries_; }

    const std::string& dataset()     const { return dataset_; }
    int64_t            chunk_count() const { return chunk_count_; }
    int64_t            chunk_size()  const { return chunk_size_; }
    bool               is_gguf()     const { return gguf_; }
    const std::string& path()        const { return path_; }

    /* Column importance for one expert, normalised to mean 1 so weighted and unweighted errors are
     * comparable. False if the tensor is absent, the expert out of range, or no values. */
    bool importance(const std::string& name, int64_t expert, std::vector<float>* out) const;

    /* importance() for a weight as the converter names it. The imatrix uses GGUF names without the
     * `.weight` suffix and keeps a MoE layer's experts as one entry, so `blk.5.ffn_down.weight` is
     * entry `blk.5.ffn_down` (expert 0) and an expert slice `blk.5.ffn_down_exps.9.weight` is
     * expert 9 of `blk.5.ffn_down_exps`. False when no entry matches or its width is not `cols`. */
    bool importance_for(const std::string& weight, int64_t cols, std::vector<float>* out) const;

    /* Raw activation count for one expert (the popularity signal). 0 if absent. */
    double count(const std::string& name, int64_t expert) const;

private:
    int read_legacy(const uint8_t* p, uint64_t n);
    int read_gguf(const uint8_t* p, uint64_t n);

    int         fd_ = -1;
    uint8_t*    map_ = nullptr;
    uint64_t    map_bytes_ = 0;
    bool        gguf_ = false;
    std::string path_, dataset_;
    int64_t     chunk_count_ = 0, chunk_size_ = 0;

    /* Legacy floats are unaligned in the file, so they are copied out; GGUF payloads are pointed
     * at. */
    std::vector<std::vector<float>>         owned_;
    std::vector<ImatrixEntry>               entries_;
    std::unordered_map<std::string, size_t> index_;
};

}  /* namespace rad */
