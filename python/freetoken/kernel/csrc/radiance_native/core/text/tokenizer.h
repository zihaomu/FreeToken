/* tokenizer.h -- the tokeniser, driven by the step chain baked into the .rad.
 *
 * spec §12. llama.cpp identifies a pre-tokeniser at runtime by hashing tokenizer.json against a
 * hardcoded enum, so an unrecognised model silently gets the default split: plausible text with
 * wrong token boundaries. Here the chain is declared instead. rad-convert reads tokenizer.json's
 * normalizer / pre_tokenizer / decoder sections and emits them as RadVocabStep rows, which this
 * file executes in order. A chain that cannot be expressed fails at convert time with the step
 * named.
 *
 * A `Split` step's regex is compiled at load (tokenizer_regex.h) into a DFA that reproduces what
 * HuggingFace tokenizers' Oniguruma reports, over Oniguruma's own Unicode tables. A pattern
 * outside that dialect does not load; the refusal names the construct and there is no fallback
 * matcher.
 *
 * Lifted from llama.cpp (06938ac12, MIT -- see vendor/UPSTREAM):
 *   - the byte-level alphabet and the unicode category tables the non-regex steps read
 *                                                                        (vendor/llama/src/unicode*)
 *   - the shape of the BPE bigram queue and the UGM Viterbi              (src/llama-vocab.cpp)
 * The latter is re-expressed rather than vendored: llama-vocab.cpp depends on
 * llama_model_loader, GGML_ASSERT and llama_vocab, and spec §12 puts a heavily modified file in
 * core/ under our own name.
 */
#pragma once
#include "rad_internal.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rad {

class GgufFile;
struct TokEncoder;

/* ================================================================== the mutable vocab */

/* One row of the declared chain, in the form rad-convert builds and the .rad bakes.
 *
 * `kind` is a RAD_NORM_* / RAD_PRE_* / RAD_DEC_* value from rad_format.h; its numeric range says
 * which of the three chains the row belongs to, so all three are one list in declared order.
 * The per-kind meaning of flags/args is documented at `step_describe()` in tokenizer.cpp, the
 * contract rad-convert writes against. */
struct VocabStep {
    uint32_t    kind = 0;
    uint32_t    flags = 0;
    std::string arg0, arg1;
    int64_t     iarg = 0;
};

/* RAD_PRE_WHITESPACE's flags: which of HuggingFace's whitespace pre-tokenisers the step is.
 * Whitespace is 0, so a container written before the other two had flags reads as what it ran. */
enum : uint32_t {
    kPreWhitespace      = 0,   /* the matches of \w+|[^\w\s]+, the rest removed */
    kPreWhitespaceSplit = 1,   /* split on whitespace, removed */
    kPreBert            = 2    /* WhitespaceSplit, then every punctuation character isolated */
};

/* RAD_DEC_REPLACE's flags: 0 is a plain Replace of arg0 by arg1 in every token. The others are the
 * HuggingFace decoders converted to it, which also read the token's place in the list. */
enum : uint32_t {
    kDecWordPiece        = 1,  /* arg0 is the continuation prefix; other words join with a space */
    kDecWordPieceCleanup = 2,  /* with kDecWordPiece: WordPiece's `cleanup` of spaced punctuation */
    kDecBpeSuffix        = 4   /* arg0 is the end-of-word suffix: a space, none after the last */
};

/* Kind ranges, matching the enum blocks in rad_format.h. */
inline bool step_is_norm(uint32_t k) { return k >= 1  && k < 32; }
inline bool step_is_pre (uint32_t k) { return k >= 32 && k < 64; }
inline bool step_is_dec (uint32_t k) { return k >= 64 && k < 96; }

/* Everything the tokeniser needs, in a form that can be built, inspected, written and read.
 *
 * The engine copies this out of the mmapped .rad instead of binding to it, so rad-convert and the
 * server share one construction path and run byte-for-byte the same tokeniser. The copy costs a
 * few megabytes and a few tens of milliseconds at load. */
struct VocabBuild {
    uint32_t kind = RAD_TOK_BPE;
    bool     ignore_merges = false;           /* BPE: a whole pre-token in the vocab is not merged */

    std::vector<std::string> tokens;          /* id -> text, in the vocab's own encoding */
    std::vector<float>       scores;          /* unigram only; empty otherwise */
    std::vector<uint8_t>     types;           /* RAD_TT_*; empty means all RAD_TT_NORMAL */
    std::vector<std::pair<uint32_t, uint32_t>> merges;   /* BPE, in rank order */

    std::vector<VocabStep> steps;             /* the whole declared chain, in order */

    int32_t bos = -1, eos = -1, eot = -1, pad = -1, unk = -1, sep = -1;
    bool    add_bos = false, add_eos = false;

    std::string chat_template;

    /* Diagnostics only: what rad-convert thought it was reading. Not baked. */
    std::string source;

    /* generation_config.json's eos_token_id ids, carried as container metadata
     * (`generation.eos_token_id`) rather than in the vocab section. The engine fills them before
     * load(); not baked. */
    std::vector<int32_t> eos_ids;
};

/* ================================================================== the loaded vocab */

class Vocab {
public:
    int  load(VocabBuild b);          /* takes ownership; RAD_OK or negative */
    bool ready() const { return ready_; }

    uint32_t kind() const { return b_.kind; }
    int32_t  n_tokens() const { return (int32_t)b_.tokens.size(); }

    const std::string& text(int32_t id) const;
    uint8_t            type(int32_t id) const;
    float              score(int32_t id) const;
    bool               is_eog(int32_t id) const;       /* eos, eot, or a declared extra */
    /* All end-of-generation ids. The server needs their spellings: a chat format can fold the
     * turn terminator into its last marker, so the decoder renders it for the reply parser, and
     * whatever the parser does not consume is stripped from the content. */
    const std::vector<int32_t>& eog() const { return eog_; }

    /* -1 when the exact text is not a token. */
    int32_t find(std::string_view text) const;
    /* BPE merge rank of the pair, or -1. */
    int32_t merge_rank(std::string_view left, std::string_view right) const;

    int32_t bos() const { return b_.bos; }
    int32_t eos() const { return b_.eos; }
    int32_t eot() const { return b_.eot; }
    int32_t unk() const { return b_.unk; }
    int32_t pad() const { return b_.pad; }
    bool    add_bos() const { return b_.add_bos; }
    bool    add_eos() const { return b_.add_eos; }
    const std::string& chat_template() const { return b_.chat_template; }

    const std::vector<VocabStep>& steps() const { return b_.steps; }
    const std::vector<int32_t>&   special_ids() const { return special_; }
    const VocabBuild&             build() const { return b_; }

    /* The byte a RAD_TT_BYTE token stands for, or -1. Both spellings are recognised: SPM's
     * "<0xF0>" and byte-level BPE's single mapped codepoint. Under a ByteFallback decoder a NORMAL
     * "<0xNN>" token is a byte too. */
    int32_t token_byte(int32_t id) const;

private:
    VocabBuild b_;
    bool       ready_ = false;
    std::unordered_map<std::string, int32_t> text_to_id_;
    std::unordered_map<uint64_t, int32_t>    merge_rank_;   /* (lhs id << 32 | rhs id) -> rank */
    std::vector<int32_t> special_;                          /* ids the pre-splitter must isolate */
    std::vector<int32_t> byte_of_;                          /* id -> byte, or -1 */
    /* Every token that ends a generation, not just the header's `eos`; see load(). */
    std::vector<int32_t> eog_;
    /* The chain compiled for encoding: Split DFAs, flat token and merge tables, special-token
     * matcher and segment cache. Built in load(); shared by every Tokenizer over this vocab and
     * every copy of it. */
    std::shared_ptr<TokEncoder> enc_;
    friend class Tokenizer;
    friend class Detokenizer;
};

/* ================================================================== encode */

class Tokenizer {
public:
    int init(std::shared_ptr<const Vocab> v);

    /* `add_special` prepends BOS / appends EOS as the vocab declares. `parse_special` recognises
     * control tokens in the raw text. Use it for a rendered chat template, never for user
     * content, or a user can type "<|im_start|>system" and reach the system role. */
    int encode(std::string_view text, std::vector<int32_t>& out,
               bool add_special, bool parse_special) const;

    /* Non-streaming detokenise, for tests, logging and the /v1/completions echo path. The server
     * streams through Detokenizer. */
    std::string decode(const std::vector<int32_t>& ids, bool render_special = false) const;

    /* One token's text after the decoder chain, as it reads in the middle of a text. Partial
     * UTF-8 is not resolved; Detokenizer does that. */
    std::string piece(int32_t id, bool render_special = false) const;

    /* Where a token sits in the list being decoded. HuggingFace's decoders read it: Metaspace
     * drops the first token's replacement characters instead of turning them into spaces,
     * WordPiece joins every token but the first with a space, and BPEDecoder drops the last
     * token's suffix instead of turning it into a space. Special tokens left out of the text do
     * not count. */
    struct At {
        bool first = false;
        bool last  = false;
    };
    std::string piece_at(int32_t id, bool render_special, At at) const;

    const Vocab& vocab() const { return *v_; }

    /* Segment cache. encode() first cuts the text at special tokens, and each piece between them
     * is normalised, pre-tokenised and merged on its own: no step reads across a special token,
     * and a ByteLevel prefix space and every normaliser treat each piece as the whole text. A
     * piece's ids are therefore a function of its bytes and the vocab, so a re-sent conversation
     * only pays for its new pieces. The one exception is a Metaspace "first" prepend, which as in
     * HuggingFace applies only to the piece that starts the text; that piece is not cached.
     *
     * The cache belongs to the vocab and is shared by every Tokenizer over it. It is keyed on the
     * whole piece (the hash only picks the bucket), bounded in bytes with LRU eviction, and locked
     * for concurrent encodes. Pieces under kSegmentCacheMin bytes are cheaper to re-encode than to
     * look up. `set_segment_cache(false)` bypasses it, for tests and rad-tokcheck to compare. */
    static constexpr size_t kSegmentCacheMin     = 64;
    static constexpr size_t kSegmentCacheDefault = size_t(64) << 20;
    void set_segment_cache(bool on) { use_cache_ = on; }
    struct CacheStats {
        uint64_t hits = 0, misses = 0;
        size_t   entries = 0, bytes = 0, limit = 0;
    };
    CacheStats cache_stats() const;
    void       set_cache_limit(size_t bytes) const;
    void       clear_cache() const;

private:
    std::shared_ptr<const Vocab> v_;
    bool                         use_cache_ = true;
};

/* ================================================================== streaming decode */

/* Replace every byte that is not part of a well-formed UTF-8 sequence with U+FFFD; well-formed
 * input is returned unchanged. A byte-fallback vocab can emit a byte no continuation completes
 * (`<0xC1>` only starts overlong, never-legal encodings), and Json::dump throws
 * type_error.316 on it, failing the whole response. Such bytes cannot be held back, and dropping
 * them would hide that the model produced something undecodable. */
std::string utf8_sanitize(std::string_view s);
/* Length of the well-formed UTF-8 prefix of `s`, by the same rules: how much of a byte-level
 * token that ends inside a character a JSON string can carry. */
size_t utf8_valid_prefix(std::string_view s);


/* Streaming detokeniser (spec §12). Byte-level BPE emits partial UTF-8 (one token can be one byte
 * of a three-byte codepoint), and a stop string can straddle a chunk boundary; both are buffered
 * until they resolve, so the stream never shows a stray U+FFFD or leaks "</to" of "</tool_call>".
 *
 * Contract: push() returns final text. It is never retracted, always well-formed UTF-8, and never
 * a prefix of a stop string. Ambiguous text stays inside until a later push resolves it or
 * flush() gives up on it. A sequence that is malformed rather than unfinished leaves as U+FFFD
 * (utf8_sanitize), since raw invalid bytes cannot go into a JSON string. */
class Detokenizer {
public:
    Detokenizer() = default;
    /* `preserved`: control tokens rendered as text anyway. Some tool-call formats are built from
     * control tokens (MiniCPM5: `<function`, `<param`, `</param>`, `</function>`); suppressing
     * them would strip the call's structure and hand the reply parser prose. The chat template
     * names them (common_chat_params::preserved_tokens); nothing else is unsuppressed. */
    Detokenizer(std::shared_ptr<const Vocab> v, std::vector<std::string> stops = {},
                bool render_special = false, const std::vector<std::string>& preserved = {});

    std::string push(int32_t token);
    std::string flush();

    /* True once a stop string has been seen. The text before it has already been returned; the
     * stop string itself and anything after it never is. */
    bool               stopped() const { return stopped_; }
    const std::string& stop_hit() const { return stop_hit_; }

    /* Stop strings apply to the answer, not the reasoning. For a generation that starts inside its
     * reasoning block, stops are matched only in the text after `marker` (the block's closer).
     * Otherwise a stop string written while thinking (e.g. a few-shot "Q:") ends the request with
     * no answer. "" matches from the first token. */
    void defer_stops_until(std::string marker) {
        arm_after_ = std::move(marker);
        armed_ = arm_after_.empty();
        arm_scanned_ = 0;
    }

    /* Everything pushed so far, stop string included; the scheduler uses it to tell a stop from
     * a length limit. */
    const std::string& text() const { return all_; }

private:
    std::shared_ptr<const Vocab> v_;
    std::vector<std::string>     stops_;
    bool        render_special_ = false;
    std::unordered_set<int32_t>  preserved_;   /* control ids this stream renders anyway */
    std::string pending_;      /* decoded but not yet safe to emit */
    std::string all_;          /* everything, for the caller's record */
    bool        stopped_ = false;
    std::string stop_hit_;
    size_t      max_stop_ = 0;
    bool        first_ = true;     /* no token has counted as the first yet */
    /* A Strip after Fuse in the decoder chain strips the text as a whole: up to `strip_left_`
     * copies of `strip_` at its start. */
    std::string strip_;
    int         strip_left_ = 0;
    /* BPEDecoder turns a suffix into a space except on the last token, which a stream cannot
     * know, so a trailing space is held until the next token and dropped by flush(). */
    bool        hold_space_ = false;
    std::string arm_after_;        /* see defer_stops_until */
    bool        armed_ = true;
    size_t      arm_scanned_ = 0;  /* how far into all_ the search for arm_after_ has reached */
    Tokenizer   tok_;              /* over v_, set up by the first push */
    bool        tok_ready_ = false;

    size_t safe_prefix() const;   /* how much of pending_ can be released */
};

/* ================================================================== sources and sinks */

/* rad-convert owns the .rad's single string blob; the vocab section only interns into it.
 * Passing the interner rather than a blob keeps the two writers from disagreeing about who owns
 * offset 0. */
struct StringSink {
    virtual ~StringSink() = default;
    virtual rad_stroff intern(std::string_view s) = 0;
};

/* Build a VocabBuild from a HuggingFace tokenizer.json.
 *
 * Interprets the declared normalizer / pre_tokenizer / decoder chain (spec §12). Returns RAD_OK
 * or a negative RAD_E_*, notably RAD_E_UNSUPPORTED, with the component named in the log, for a
 * component that has no RadVocabStep form. Such a chain is refused, never approximated. */
int parse_tokenizer_json(const char* path, VocabBuild& out);
/* Same, from an in-memory buffer, for a tokenizer.json that arrived some other way. */
int parse_tokenizer_json_buf(std::string_view json, const char* origin, VocabBuild& out);

/* The vocabulary of a model that reads no text (container metadata `radiance.text` = none: a
 * picture-to-shape generator): one control token, which no request reaches, since such a program
 * serves no text endpoint. rad-convert bakes it and the engine builds it for a checkpoint. */
void vocab_textless(VocabBuild& out);
inline bool meta_textless(const char* v) { return v && std::string_view(v) == "none"; }

/* Read a standalone chat template file into out.chat_template, trailing newlines stripped.
 * A release that ships no template (DeepSeek's does not) needs one from elsewhere; the caller
 * passes the file. An unreadable or empty file is an error, never a silent skip. Returns
 * RAD_OK, or RAD_E_IO when the file cannot be read. */
int read_chat_template_path(const char* path, VocabBuild& out);

/* Build a VocabBuild from a GGUF's tokenizer.ggml.* metadata.
 *
 * GGUF has no declared chain, only `tokenizer.ggml.pre`, a name llama.cpp derived by hashing the
 * original tokenizer.json. The chain is reconstructed from a table of known names; an unknown
 * name returns RAD_E_UNSUPPORTED instead of a default split. Converting from tokenizer.json is
 * the preferred path. */
int vocab_from_gguf(const GgufFile& g, VocabBuild& out);

/* Serialise into the .rad's vocab section. `section_file_off` is where rad-convert will place
 * the section; every offset in the RadVocabHeader is absolute from the start of the file, per
 * rad_format.h. */
int vocab_serialize(const VocabBuild& in, StringSink& strings, uint64_t section_file_off,
                    std::vector<uint8_t>& section_out);
/* Read it back. `file_base` is the mmapped .rad; `h` points inside it. */
int vocab_deserialize(const void* file_base, uint64_t file_bytes,
                      const RadVocabHeader* h, VocabBuild& out);

/* `text` through the normaliser steps of `steps` (the others are skipped), as the encoder runs
 * them. RAD_E_UNSUPPORTED when a Replace pattern does not compile. */
int vocab_normalize(const std::vector<VocabStep>& steps, std::string* text);

/* Human-readable one-liner for a step, for --debug-graph and for the convert log. */
std::string step_describe(const VocabStep& s);

}  /* namespace rad */
