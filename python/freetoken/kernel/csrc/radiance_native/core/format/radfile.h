/* radfile.h -- the .rad reader and writer, implementing exactly the format frozen in
 * abi/rad_format.h. The writer and reader live together so they cannot disagree.
 *
 * The reader mmaps and builds an index. Every invariant is checked at open, not at use:
 *
 *   - every offset and count is bounds-checked against the mapped length, so a truncated or
 *     corrupt file is RAD_E_FORMAT with a message, not a later SIGSEGV;
 *   - every rad_stroff lands inside a NUL-terminated string blob, so str() is a plain pointer;
 *   - the expert directory's fixed stride is re-derived and verified, so `base + id * stride` on
 *     the critical path is safe.
 *
 * Consequently a container whose experts within a layer differ in shape or format is refused:
 * addressing by arithmetic is the format's purpose. So is one that names a weight twice, since
 * find() could return only one of them.
 */
#pragma once
#include "../rad_internal.h"

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rad {

/* ================================================================== vocab */
/* The vocab's mutable side (VocabBuild, the tokenizer.json interpreter, the serialiser) belongs to
 * core/text (spec §12). The container's half of the contract:
 *
 *   RadWriter::intern_string()     the vocab's strings go into the .rad's single string blob;
 *   RadWriter::set_vocab_section() the writer places and rebases the serialised bytes like every
 *                                  other offset.
 *
 * core/text's `StringSink` joins them; rad-convert supplies the adapter.
 *
 * The read-back view below holds pointers into the mapping, bounds-checked at open;
 * vocab_deserialize() copies out of it. Not named VocabView, which core/sample/vocab_view.h
 * declares in this namespace (an ODR clash in any unit including both). */
struct VocabSection {
    const RadVocabHeader* h        = nullptr;
    const rad_stroff*     tok_text = nullptr;   /* [n_tokens] */
    const float*          tok_score= nullptr;   /* [n_tokens] or null */
    const uint8_t*        tok_type = nullptr;   /* [n_tokens] or null */
    const uint32_t*       merges   = nullptr;   /* [n_merges][2] or null */
    const RadVocabStep*   steps    = nullptr;   /* [n_steps] or null */
    explicit operator bool() const { return h != nullptr; }
};

/* ================================================================== reader */
class RadFile {
public:
    RadFile() = default;
    ~RadFile();
    RadFile(const RadFile&) = delete;
    RadFile& operator=(const RadFile&) = delete;

    /* mmap, validate, index. RAD_OK, or RAD_E_IO / RAD_E_FORMAT with the failing field logged. */
    int  open(const char* path);
    void close();
    bool is_open() const { return base_ != nullptr; }

    const char*          path()   const { return path_.c_str(); }
    const RadFileHeader& header() const { return *hdr_; }
    const uint8_t*       base()   const { return base_; }
    int64_t              size()   const { return size_; }

    /* Drop the blob from the page cache and this process's page tables, so a container larger
     * than RAM does not leave the cache full of pages nothing will read. Only the blob: the tables
     * are small and read constantly. The ranges named to keep_cached() are left alone, on whole
     * pages. Advisory; the kernel may decline either way. */
    void drop_blob_cache() const;
    /* Names [p, p + n) as read through the mapping for the engine's life (the mapped tier: an
     * n-gram table), so no drop_blob_cache() frees its pages: a gather from dropped pages starts
     * cold, from the drive. An address outside the container is ignored (a copy in RAM, or a
     * table read from its own file). Returns the bytes now kept. */
    int64_t keep_cached(const void* p, int64_t n);

    /* Drop one range of the blob behind a single forward cursor. A buffered front-to-back read of
     * a container larger than RAM otherwise builds reclaim pressure that slows the read itself.
     * Correct only for one cursor moving in one direction with no revisits; with several, one
     * would drop what another is about to read. */
    void drop_range(int64_t off, int64_t bytes) const;

    /* pread `bytes` at `off` into `dst`, bypassing the mapping. For loads: an async copy from
     * pageable memory goes through the HIP runtime's own staging, which degrades as a load
     * proceeds (a small hipMemcpyAsync can block for seconds even with resident pages); reading
     * into the loader's pinned buffer makes the copy a real DMA. RAD_OK, or RAD_E_IO with the
     * reason logged; short reads and EINTR are retried. */
    int read_at(void* dst, int64_t bytes, int64_t off) const;

    /* Strings. Offset 0 is the empty string; every offset was bounds-checked at open. */
    const char* str(rad_stroff off) const {
        return (const char*)(base_ + hdr_->str_off + (off < str_bytes_ ? off : 0));
    }

    /* ------------------------------------------------------------ metadata */
    int64_t          meta_count() const { return (int64_t)hdr_->meta_count; }
    const RadFileKV& meta_at(int64_t i) const { return meta_[i]; }
    bool             get_i(const char* key, int64_t* out) const;
    bool             get_f(const char* key, double* out) const;
    const char*      get_s(const char* key, const char* dflt = nullptr) const;

    /* ------------------------------------------------------------ weight directory */
    int64_t             entry_count() const { return (int64_t)hdr_->dir_count; }
    const RadFileEntry& entry(int64_t i) const { return dir_[i]; }
    const RadFileEntry* find(std::string_view name) const;
    const uint8_t*      data(const RadFileEntry& e) const { return base_ + e.offset; }

    /* An entry's encoding and its planes in encoding order; each plane's size was checked at
     * open. */
    const RadEncoding&  encoding(const RadFileEntry& e) const { return encs_[e.enc]; }
    const RadFilePlane& plane(const RadFileEntry& e, int k) const {
        return planes_[e.plane_first + (uint64_t)k];
    }
    const uint8_t*      data(const RadFilePlane& p) const { return base_ + p.offset; }
    int64_t             encoding_count() const { return (int64_t)hdr_->enc_count; }
    const RadEncoding&  encoding_at(int64_t i) const { return encs_[i]; }

    /* ------------------------------------------------------------ fixed-stride addressing */
    /* One layer's experts, all of one shape and format, so addr = base + id * stride with no table
     * walk on the critical path (spec §4.2). Verified at open. */
    struct ExpertGroup {
        int32_t  layer      = -1;
        int32_t  n_expert   = 0;
        int32_t  n_slot     = 0;      /* gate/up/down = 3 */
        int64_t  first_index= 0;      /* directory index of (expert 0, slot 0) */
        uint64_t base       = 0;      /* file offset of expert 0's first byte */
        uint64_t stride     = 0;      /* bytes between consecutive experts, aligned */
        uint64_t unit_bytes = 0;      /* payload of one expert: every slot's every plane */
        /* Per-slot offset from an expert's base, the same for every expert. */
        uint64_t slot_off[8] = {0};
        uint64_t slot_bytes[8] = {0};
    };

    int64_t            group_count() const { return (int64_t)groups_.size(); }
    const ExpertGroup* group_for_layer(int32_t layer) const;

    /* No bounds check on the hot path: the caller has the group's n_expert and n_slot. */
    const uint8_t* expert_ptr(const ExpertGroup& g, int32_t expert, int32_t slot) const {
        return base_ + g.base + (uint64_t)expert * g.stride + g.slot_off[slot];
    }
    const RadFileEntry& expert_entry(const ExpertGroup& g, int32_t expert, int32_t slot) const {
        return dir_[g.first_index + (int64_t)expert * g.n_slot + slot];
    }

    /* ------------------------------------------------------------ vocab and profile */
    VocabSection              vocab() const { return vocab_; }
    /* The second vocab section, or an empty one (RAD_HDR_AUX_VOCAB_OFF in rad_format.h). */
    VocabSection              aux_vocab() const { return aux_vocab_; }
    int64_t                   profile_count() const { return (int64_t)hdr_->prof_count; }
    const RadFileProfile*     profile() const { return prof_; }

private:
    int  validate();
    int  index_experts();

    std::string path_;
    uint8_t*    base_ = nullptr;
    int64_t     size_ = 0;
    std::vector<std::pair<int64_t, int64_t>> keep_;   /* keep_cached: (blob offset, bytes) */
    int         fd_   = -1;

    const RadFileHeader*  hdr_  = nullptr;
    const RadFileKV*      meta_ = nullptr;
    const RadFileEntry*   dir_  = nullptr;
    const RadFilePlane*   planes_ = nullptr;
    const RadEncoding*    encs_ = nullptr;
    const RadFileProfile* prof_ = nullptr;
    uint64_t              str_bytes_ = 1;

    VocabSection                                  vocab_, aux_vocab_;
    std::vector<ExpertGroup>                      groups_;
    std::unordered_map<std::string_view, int64_t> by_name_;
};

/* The blob range [*off, *off + *len) a pointer covers, when it lies wholly inside the container
 * [base, base + size): what a load records for the mapped tier. False when the address is not in
 * the blob (a copy in RAM under `ngram-placement: ram`, or a table read from its own file under
 * `disk=DIR`), where keeping it would name bytes that belong to something else. */
bool rad_blob_range(const uint8_t* base, int64_t size, const void* p, int64_t n,
                    int64_t* off, int64_t* len);

/* The ranges of [tables, size) a page-cache drop must free when `keep` stays: the complement of
 * the keep ranges as (offset, bytes), the same convention `keep` uses. Page-rounded the way the
 * drop needs it -- a gap's start rounds up and its end down, so a page holding kept bytes is never
 * dropped. Unsorted, overlapping and out-of-range input is merged and clamped; a keep list covering
 * the whole blob gives no ranges. */
std::vector<std::pair<int64_t, int64_t>> rad_keep_complement(
    int64_t tables, int64_t size, std::vector<std::pair<int64_t, int64_t>> keep);

/* ================================================================== writer */
/* Streaming writer: the blob is never held in RAM (a checkpoint may be 100 GB). The tables
 * (strings, directory, planes, encodings, vocab, profile) are in memory; ~50k directory rows for a
 * 512-expert 48-layer model is megabytes.
 *
 * A weight is reserved, then written: add_weight() places the entry and every plane and returns
 * its index; write_plane() fills any part of any plane in any order. A quantiser produces rows of
 * every plane at once, and a 51 GB table is produced block by block. finish() refuses a plane left
 * short.
 *
 * Callers must add weights in movement-unit order: an expert's slots consecutively, and a layer's
 * experts consecutively with identical shapes. The writer verifies this and refuses with
 * RAD_E_INVAL naming the offending pair. rad-convert sorts its manifest before writing.
 */
class RadWriter {
public:
    RadWriter() = default;
    ~RadWriter();
    RadWriter(const RadWriter&) = delete;
    RadWriter& operator=(const RadWriter&) = delete;

    int  begin(const char* path);
    void abort();                       /* discard, unlink both the target and the spill -- or, in
                                         * append mode, cut the file back to its old length */

    void set_arch(std::string_view s)       { arch_ = s; }
    void set_model_name(std::string_view s) { model_ = s; }
    void set_quant(std::string_view s)      { quant_ = s; }
    void set_recipe(std::string_view s)     { recipe_ = s; }
    void set_created_by(std::string_view s) { created_by_ = s; }

    void meta_i(std::string_view k, int64_t v);
    void meta_f(std::string_view k, double v);
    void meta_s(std::string_view k, std::string_view v);

    struct Weight {
        std::string name;                /* the logical weight's name */
        std::string quantizer;           /* "" for the checkpoint's own planes */
        std::string options;             /* "k=v,k=v", sorted by key */
        RadEncoding enc{};
        uint32_t    rank  = 0;
        int64_t     shape[RAD_MAX_RANK] = {0};   /* logical */
        int32_t     layer = -1, expert = -1, slot = 0;
    };

    /* Append in place: extend container `old` (opened at `path`) with new weights without copying
     * the existing ones. Offsets are absolute, so new bytes go past the end, followed by fresh
     * tables naming both the old weights (at their offsets, via add_existing()) and the new; the
     * header is rewritten last. Until then the file is its old self with an unreachable tail;
     * afterwards the old tables are dead bytes. finish() leaves `<path>.pre-append` (the old
     * header and length) to restore it.
     *
     * A layer's experts sit at a fixed stride, so a layer whose experts are kept (add_existing)
     * takes no further one. A layer none of whose experts is kept takes its whole table, written
     * as in a new file: experts 0..n-1 in order, each one's slots in order -- a layer `old` holds
     * none of, or one whose old table the caller leaves behind as dead bytes. An old weight the
     * caller neither keeps nor writes again is dead bytes too. The caller must be the file's only
     * user, since a process mapping it would see the header change under it. */
    int  begin_append(const char* path, const RadFile& old);
    /* A weight `old` already holds, recorded at the offsets it has there. Its encoding and planes
     * are the old entry's; its name, provenance and movement group are `w`'s. */
    int64_t add_existing(const Weight& w, const RadFile& old, const RadFileEntry& e);

    /* Reserve the weight and every plane of its encoding. Returns the directory index, or a
     * negative status: an invalid encoding, a shape it cannot cover, a movement-unit order the
     * stride cannot hold, a name the container already holds. */
    int64_t add_weight(const Weight& w);
    /* Reserve and write in one call: `planes[k]` is plane k's whole bytes. */
    int64_t add_weight(const Weight& w, const void* const* planes);

    /* `bytes` of plane `k` of weight `idx`, at byte `off` within the plane. */
    int write_plane(int64_t idx, int k, int64_t off, const void* data, int64_t bytes);
    int64_t plane_bytes(int64_t idx, int k) const;

    /* Intern into the .rad's single string blob. Public for core/text's vocab writer (via
     * StringSink). */
    rad_stroff intern_string(std::string_view s) { return intern(s); }

    /* The serialised vocab section. Its internal offsets must be section-relative (serialise with
     * section_file_off 0); finish() rebases them, as it does the directory. */
    int  set_vocab_section(const void* blob, int64_t bytes);
    /* The second vocab section (RAD_HDR_AUX_VOCAB_OFF), serialised the same way. */
    int  set_aux_vocab_section(const void* blob, int64_t bytes);

    void add_profile(int32_t layer, int32_t expert, float share);

    /* Fix the layout once every weight, the metadata, the vocab section and the profile are in, so
     * the blob's offset is known: afterwards plane writes land in the target directly and
     * finish() writes the tables in front, with no spill and no copy. Changing a table afterwards
     * fails finish(). An unsealed writer spills the blob and splices it in at finish(). */
    int  seal();

    /* Writes the header and every table, then (if unsealed) splices the spilled blob in at a
     * 2 MiB boundary. */
    int  finish();

    int64_t data_bytes() const { return blob_bytes_; }
    int64_t weight_count() const { return (int64_t)dir_.size(); }

private:
    rad_stroff intern(std::string_view s);
    void       layout(RadFileHeader* h);
    uint32_t   intern_encoding(const RadEncoding& e);
    int        check_unit_invariant(const Weight& w, uint64_t off, uint64_t unit_bytes);

    std::string path_, spill_path_;
    int         fd_out_ = -1, fd_spill_ = -1;
    int64_t     spill_base_ = 0;    /* where blob offset 0 is in fd_spill_: 0, append_base_ or
                                     * the sealed data_off */
    bool        began_ = false;
    bool        sealed_ = false, late_ = false;
    RadFileHeader hdr_{};           /* the sealed layout */

    /* Append mode (begin_append). New bytes go at `append_base_`, the old length rounded up to a
     * pool boundary; add_existing() entries are already absolute and flagged so finish() does not
     * rebase them. */
    int  finish_append();
    int  check_complete() const;
    bool     append_ = false;
    int64_t  old_size_ = 0, append_base_ = 0;
    uint64_t old_data_off_ = 0;
    RadFileHeader old_hdr_{};
    std::vector<uint8_t> dir_abs_;
    /* While extending in place: the layers an expert was kept in (add_existing) and the layers a
     * new expert was reserved in. A layer is in one of them only. */
    std::unordered_set<int32_t> kept_expert_layers_, new_expert_layers_;

    std::string arch_, model_, quant_, recipe_, created_by_;

    std::string                                   strblob_{std::string(1, '\0')};
    std::unordered_map<std::string, rad_stroff>   strmap_;
    std::vector<RadFileKV>                        meta_;
    std::vector<RadFileEntry>                     dir_;
    std::vector<RadFilePlane>                     planes_;
    std::vector<int64_t>                          written_;   /* parallel to planes_ */
    std::vector<RadEncoding>                      encs_;
    std::vector<RadFileProfile>                   prof_;
    /* Every weight name added, since a name is looked up and must find one entry. */
    std::unordered_set<std::string>               names_;

    /* The vocab section, assembled in memory and appended verbatim. */
    std::vector<uint8_t> vocab_blob_;
    bool                 have_vocab_ = false;
    std::vector<uint8_t> aux_vocab_blob_;
    bool                 have_aux_vocab_ = false;

    int64_t blob_bytes_ = 0;    /* the blob's reserved length */

    /* Movement-unit bookkeeping, so the writer verifies the fixed stride. */
    int32_t  cur_layer_ = -2, cur_expert_ = -2;
    int32_t  grp_layer_ = -2;
    int32_t  grp_n_slot_ = 0, grp_slot_seen_ = 0, grp_n_expert_ = 0;
    uint64_t grp_first_off_ = 0, grp_expert_off_ = 0, grp_stride_ = 0;
    /* Expert 0's per-slot payload, which enforces ExpertGroup's fixed slot_off[]/slot_bytes[]: an
     * expert whose slot differs in size is refused at its own add_weight. The offset check alone
     * misses an oversized last expert, which starts at the right base and overruns. */
    std::vector<uint64_t> grp_slot_bytes_;
};

/* ================================================================== helpers for the tools */
/* "layer 12 / expert 7" etc., so every diagnostic names a unit the same way. */
std::string rad_unit_name(int32_t layer, int32_t expert);

}  /* namespace rad */
