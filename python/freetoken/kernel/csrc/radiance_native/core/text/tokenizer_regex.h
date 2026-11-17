/* tokenizer_regex.h -- the pre-tokeniser pattern compiler.
 *
 * A tokenizer.json `Split` carries a regex in Oniguruma's Ruby syntax, and HuggingFace tokenizers
 * runs it through Oniguruma's backtracking matcher. This compiles the same pattern into a DFA over
 * codepoint classes that reports the same matches:
 *
 *   - Unicode properties are Oniguruma's own (tokenizer_ucd.h), so `\p{L}`, `\s`, `\w` and the
 *     `(?i)` orbits mean what they mean there.
 *   - Alternation is leftmost-first, not leftmost-longest: the DFA carries its NFA threads in
 *     priority order and a match cuts every lower thread, giving backtracking order without
 *     backtracking (the RE2 / regex-automata construction). Lazy quantifiers reverse the priority.
 *   - One-codepoint lookaround (`\s+(?!\S)`, `(?=[\p{L}])`, `(?<=x)`), `^`, `$`, `\A`, `\z`, `\b`
 *     and `\B` are assertions on the neighbouring codepoints. The state carries what it needs of
 *     the previous codepoint, and a lookahead is decided on the transition that reads the next
 *     one, so a match is reported one transition late.
 *   - find_iter follows Oniguruma's iteration, empty matches included: each match is the leftmost
 *     one, and an empty match at the end of the previous match is skipped by one codepoint, as
 *     the `onig` crate does.
 *
 * Anything else is refused at compile time with the construct named: backreferences, atomic
 * groups, possessive quantifiers except where the greedy reading provably matches the same (one
 * class at the top level, followed only by classes it shares no codepoint with, as in Ling-3.0's
 * pattern), lookaround wider than one codepoint, unbounded repetition of
 * something that can match empty, `(?i)` over a character that folds to several (U+00DF vs "ss"),
 * and properties the tables do not carry. There is no fallback matcher; a pattern that does not
 * compile fails the vocabulary load.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rad {

/* [begin, end) in bytes. */
struct TokSpan { uint32_t begin, end; };

class TokRegex {
public:
    ~TokRegex();

    /* RAD_OK, or RAD_E_UNSUPPORTED with `why` saying which construct at which byte. */
    static int compile(std::string_view pattern, std::unique_ptr<TokRegex>* out, std::string* why);

    /* Every match Oniguruma's find_iter reports over `text`, appended in order. `text` must be
     * canonical UTF-8 (utf8_is_canonical); the tokeniser canonicalises before it splits. */
    void find_iter(std::string_view text, std::vector<TokSpan>& out) const;

    /* The end of the leftmost-first match anchored at byte `pos`, or -1 when there is none. */
    int64_t match_at(std::string_view text, size_t pos) const;

    const std::string& pattern() const { return pattern_; }
    size_t n_states() const;
    size_t n_classes() const;

private:
    TokRegex() = default;
    struct Dfa;
    std::string          pattern_;
    std::unique_ptr<Dfa> dfa_;
};

/* HuggingFace's SplitDelimiterBehavior, in the order VocabStep::iarg carries it. */
enum TokSplitBehavior {
    TOK_SPLIT_REMOVED = 0, TOK_SPLIT_ISOLATED = 1, TOK_SPLIT_MERGED_PREV = 2,
    TOK_SPLIT_MERGED_NEXT = 3, TOK_SPLIT_CONTIGUOUS = 4,
};

/* NormalizedString::split: the non-empty pieces of `text` that `behavior` keeps, appended to
 * `out` as spans of `text`. `scratch` is reused between calls to avoid allocation. */
void tok_split(const TokRegex& re, std::string_view text, int behavior, bool invert,
               std::vector<TokSpan>& out, std::vector<TokSpan>& scratch);

/* NormalizedString::replace with a regex: every match replaced by `content`, verbatim. */
void tok_replace(const TokRegex& re, std::string_view text, std::string_view content,
                 std::string& out);

/* True when `s` decodes to codepoints and re-encodes to the same bytes under the tokeniser's
 * decoder: no invalid or truncated sequence, no overlong form, nothing above U+10FFFF. */
bool utf8_is_canonical(std::string_view s);

/* Append that round trip of `s` to `out`: each byte that does not start a decodable sequence
 * becomes U+FFFD, and an overlong form becomes its shortest one. These are the rules of
 * unicode_cpts_from_utf8, the decoder the normalisers and other pre-tokenisers use, so a Split
 * sees the same text they do. */
void utf8_canonicalize(std::string_view s, std::string& out);

}  /* namespace rad */
