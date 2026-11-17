/* tokenizer_grapheme.h -- extended grapheme clusters (UAX #29) and SentencePiece's precompiled
 * normaliser, which walks its text by them.
 *
 * HuggingFace tokenizers' `Precompiled` normaliser (a SentencePiece model's `precompiled_charsmap`,
 * T5's among them) takes the text one extended grapheme cluster at a time: a cluster shorter than
 * six bytes is replaced whole by the mapping of its shortest mapped prefix, if it has one;
 * otherwise each of its characters is replaced by its own mapping or kept. The cluster
 * boundaries are therefore part of the result, and they are Unicode 16.0.0's, the version the
 * reference's segmentation crate implements (tools/gen_grapheme_ucd.py writes the tables).
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rad {

namespace ucd {
/* A run of codepoints [first, last] holding one value of a property. */
struct GbRange { uint32_t first, last; uint32_t v; };
/* Grapheme_Cluster_Break, values in tools/gen_grapheme_ucd.py's GCB order (0 is Other). */
extern const GbRange k_gcb[];
extern const size_t  k_gcb_n;
/* Extended_Pictographic (v 1). */
extern const GbRange k_ext_pict[];
extern const size_t  k_ext_pict_n;
/* Indic_Conjunct_Break: 1 Linker, 2 Consonant, 3 Extend (absent: None). */
extern const GbRange k_incb[];
extern const size_t  k_incb_n;
}  /* namespace ucd */

/* The end of the extended grapheme cluster that starts at `cps[i]`: the index one past its last
 * codepoint. `i < n`. */
size_t grapheme_end(const uint32_t* cps, size_t n, size_t i);

/* SentencePiece's precompiled character map: a double-array trie over UTF-8 source strings, each
 * leaf the offset of its NUL-terminated replacement in a string blob. */
class PrecompiledMap {
public:
    /* `charsmap_b64` is tokenizer.json's `precompiled_charsmap` as it stands, base64. RAD_OK, or
     * RAD_E_FORMAT with `why` for a map that does not decode or whose trie points outside it. */
    int load(std::string_view charsmap_b64, std::string* why);
    bool ready() const { return !trie_.empty(); }
    /* The normalised text, as HuggingFace's Precompiled normaliser writes it. */
    std::string normalize(std::string_view text) const;

private:
    /* The replacement of the shortest mapped prefix of `s`, or false when no prefix is mapped. */
    bool transform(std::string_view s, std::string_view* out) const;
    std::vector<uint32_t> trie_;
    std::string           blob_;
};

}  /* namespace rad */
