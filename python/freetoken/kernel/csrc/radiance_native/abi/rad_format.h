/* rad_format.h -- the .rad container (spec.md §4.2).
 *
 *   canonical, not kernel-shaped    -- an entry is a logical weight holding the planes of its
 *                                      encoding (rad_encoding.h), row-major; no kernel library
 *                                      is named anywhere in the file, and the kernel that reads a
 *                                      weight rearranges it at load (spec §4.3)
 *   grouped by movement unit        -- an expert's entries are adjacent, and addressing within a
 *                                      layer is base + id*stride
 *   aligned                         -- pool 2 MiB, movement unit 4 KiB, plane 256 B
 *   self-describing                 -- each entry carries its encoding, its logical shape and the
 *                                      quantiser and options that made it
 *
 * Little-endian only. Every offset is from the start of the file. Every string is an offset into
 * the string blob, so the directory is fixed-stride and mmap-walkable without a parse.
 *
 * There is one format version. A file in any other is refused with an instruction to convert
 * again, so there is a single reader with a single set of invariants. */
#ifndef RAD_FORMAT_H
#define RAD_FORMAT_H

#include "rad_types.h"
#include "rad_encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RAD_MAGIC        0x31444152u   /* "RAD1" little-endian: the family; the version follows */
#define RAD_FORMAT_VER   2u

#define RAD_ALIGN_POOL   (2u * 1024u * 1024u)
#define RAD_ALIGN_UNIT   4096u
#define RAD_ALIGN_SUB    256u

typedef uint64_t rad_stroff;   /* offset into the string blob; 0 is the empty string */

/* ------------------------------------------------------------------ header */
typedef struct RadFileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t file_bytes;

    rad_stroff arch_id;
    rad_stroff model_name;
    rad_stroff quant;           /* the source checkpoint's quantisation descriptor, "" for none:
                                 * with arch_id, what selects the architecture plugin (spec §2.4) */
    rad_stroff recipe;          /* the recipe it was converted with, as text; "" for none */
    rad_stroff created_by;      /* rad-convert version + the source checkpoint's identity */

    uint64_t str_off,   str_bytes;
    uint64_t meta_off,  meta_count;    /* RadFileKV[] */
    uint64_t dir_off,   dir_count;     /* RadFileEntry[] */
    uint64_t plane_off, plane_count;   /* RadFilePlane[] */
    uint64_t enc_off,   enc_count;     /* RadEncoding[], each distinct one once */
    uint64_t vocab_off, vocab_bytes;   /* RadVocabHeader + its payload */
    uint64_t prof_off,  prof_count;    /* RadFileProfile[] -- expert popularity */
    uint64_t data_off,  data_bytes;    /* the blob; RAD_ALIGN_POOL-aligned */

    /* reserved[RAD_HDR_AUX_VOCAB_OFF] and reserved[RAD_HDR_AUX_VOCAB_BYTES] place a second vocab
     * section, laid out as the first (RadVocabHeader + its payload, its strings in the one string
     * blob): a tokenizer a generator reads its prompt through beside the model's own
     * (RadGeneratorDecl::aux_tokens). Both zero when there is none, as in a file written before
     * the section existed; the rest are zero. */
    uint64_t reserved[8];
} RadFileHeader;

enum { RAD_HDR_AUX_VOCAB_OFF = 0, RAD_HDR_AUX_VOCAB_BYTES = 1 };

/* ------------------------------------------------------------------ metadata */
typedef struct RadFileKV {
    rad_stroff key;
    uint32_t   type;      /* RAD_P_INT | RAD_P_STR, or RAD_KV_F64 */
    uint32_t   pad;
    union {
        int64_t    i;
        double     f;
        rad_stroff s;
    } v;
} RadFileKV;

enum { RAD_KV_F64 = 100 };

/* ------------------------------------------------------------------ weight directory */
/* One logical weight: what a declaration's `source` names (RadWeightDecl). Fixed stride, so an
 * expert's address inside a layer is base + id*stride and no table is walked on the critical
 * path. */
typedef struct RadFileEntry {
    rad_stroff name;          /* the logical weight's name */
    rad_stroff quantizer;     /* the quantiser that wrote the planes; "" for the checkpoint's own */
    rad_stroff options;       /* the options it ran with, "k=v,k=v" sorted by key */

    uint32_t   enc;           /* index into the encoding table */
    uint32_t   rank;
    int64_t    shape[RAD_MAX_RANK];   /* the logical shape */

    int32_t    layer;         /* RadWeightGroup, flattened: the movement unit */
    int32_t    expert;
    int32_t    slot;
    uint32_t   n_planes;      /* the encoding's, in its order */

    uint64_t   plane_first;   /* index into the plane table */
    uint64_t   offset;        /* the entry's first byte, from file start, RAD_ALIGN_UNIT-aligned */
    uint64_t   bytes;         /* to the end of its last plane */
    uint64_t   reserved[2];
} RadFileEntry;

/* One plane of an entry: dense, row-major, every row starting on a byte, exactly
 * rad_enc_plane_bytes() long for the entry's logical shape. */
typedef struct RadFilePlane {
    uint64_t   offset;        /* from file start, RAD_ALIGN_SUB-aligned */
    uint64_t   bytes;
} RadFilePlane;

/* ------------------------------------------------------------------ expert profile */
/* Derived from the imatrix's activation counts. A warm start, not a policy (spec §5.3): a
 * calibration set routes differently from decoding, so this gives only a small residency hit-rate
 * gain over starting cold. */
typedef struct RadFileProfile {
    int32_t layer;
    int32_t expert;
    float   share;      /* fraction of tokens routed here in the calibration set */
    float   reserved;
} RadFileProfile;

/* ------------------------------------------------------------------ vocab */
/* rad-convert interprets tokenizer.json and bakes the result here, rather than hashing the file
 * against a hardcoded enum as llama.cpp does (spec §12). */
enum { RAD_TOK_BPE = 1, RAD_TOK_UNIGRAM = 2, RAD_TOK_WORDPIECE = 3, RAD_TOK_RWKV = 4 };

/* Normaliser / pre-tokeniser / decoder steps, in declared order. */
enum {
    RAD_NORM_NFC = 1, RAD_NORM_NFKC = 2, RAD_NORM_NFD = 3, RAD_NORM_NFKD = 4,
    RAD_NORM_LOWERCASE = 5, RAD_NORM_STRIP = 6, RAD_NORM_REPLACE = 7, RAD_NORM_PREPEND = 8,
    RAD_NORM_PRECOMPILED = 9,
    RAD_PRE_BYTELEVEL = 32, RAD_PRE_SPLIT = 33, RAD_PRE_METASPACE = 34,
    RAD_PRE_WHITESPACE = 35, RAD_PRE_PUNCTUATION = 36, RAD_PRE_DIGITS = 37,
    RAD_PRE_BYTE_FALLBACK = 38,
    /* BPE's end-of-word suffix (arg0, CLIP's "</w>"): a word's last symbol carries it, so the
     * merges and the vocabulary tell a word's end from its middle. */
    RAD_PRE_END_OF_WORD = 39,
    /* An added token matched in the normalised text rather than the raw one (tokenizers'
     * `normalized`): arg0 its spelling, iarg its id. Cut out after the normalisers and before
     * the other pre-tokenisers. */
    RAD_PRE_ADDED_NORMALIZED = 40,
    RAD_DEC_BYTELEVEL = 64, RAD_DEC_METASPACE = 65, RAD_DEC_REPLACE = 66,
    RAD_DEC_STRIP = 67, RAD_DEC_FUSE = 68, RAD_DEC_BYTE_FALLBACK = 69
};

typedef struct RadVocabStep {
    uint32_t   kind;         /* RAD_NORM_* | RAD_PRE_* | RAD_DEC_* */
    uint32_t   flags;        /* step-specific: invert, add_prefix_space, use_regex, ... */
    rad_stroff arg0;         /* a pattern, a replacement, a prepend string */
    rad_stroff arg1;
    int64_t    iarg;
} RadVocabStep;

typedef struct RadVocabHeader {
    uint32_t kind;              /* RAD_TOK_* */
    uint32_t n_tokens;
    uint32_t n_merges;
    uint32_t n_steps;
    uint64_t tok_text_off;      /* rad_stroff[n_tokens] */
    uint64_t tok_score_off;     /* float[n_tokens], unigram only */
    uint64_t tok_type_off;      /* uint8[n_tokens]: normal/unknown/control/user/byte */
    uint64_t merge_off;         /* uint32[n_merges][2] */
    uint64_t step_off;          /* RadVocabStep[n_steps] */
    int32_t  bos, eos, eot, pad_id, unk, sep;
    uint32_t add_bos, add_eos;
    rad_stroff chat_template;
    uint64_t flags;             /* RAD_VOCAB_*; zero in a file written before the field existed */
    uint64_t reserved[3];
} RadVocabHeader;

/* BPE: a pre-token that is already a vocabulary entry is emitted whole instead of merged
 * (tokenizers' `ignore_merges`). LFM2's vocabulary sets it. */
enum { RAD_VOCAB_IGNORE_MERGES = 1u << 0 };

enum { RAD_TT_NORMAL = 1, RAD_TT_UNKNOWN = 2, RAD_TT_CONTROL = 3,
       RAD_TT_USER_DEFINED = 4, RAD_TT_BYTE = 6 };

#ifdef __cplusplus
}
#endif
#endif /* RAD_FORMAT_H */
