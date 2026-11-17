/* tokenizer_ucd.h -- the Unicode properties a pre-tokeniser pattern can name.
 *
 * The tables are generated (tools/gen_tokenizer_ucd.cpp) by querying Oniguruma, the engine
 * HuggingFace tokenizers matches Split patterns with. A tokenizer.json pattern means what it means
 * there, so Oniguruma is the only definition used: its general categories at its Unicode version,
 * its `\s`, `\d` and `\w`, its scripts, and its case-folding orbits for `(?i)`.
 */
#pragma once
#include <cstddef>
#include <cstdint>

namespace rad {
namespace ucd {

/* General category, in the generator's order. */
enum Gc : uint8_t {
    Lu, Ll, Lt, Lm, Lo, Mn, Mc, Me, Nd, Nl, No,
    Pc, Pd, Ps, Pe, Pi, Pf, Po, Sm, Sc, Sk, So,
    Zs, Zl, Zp, Cc, Cf, Cs, Co, Cn,
    kGcCount
};

/* A run of one category, from `first` to the next run's `first`. */
struct GcRun { uint32_t first; uint32_t gc; };
struct CpRange { uint32_t first, last; };
struct ScriptTable { const char* name; const CpRange* ranges; size_t n; };

extern const GcRun   k_gc_runs[];
extern const size_t  k_gc_runs_n;
extern const CpRange k_white_space[];      /* \s */
extern const size_t  k_white_space_n;
extern const CpRange k_digit[];            /* \d */
extern const size_t  k_digit_n;
extern const CpRange k_word[];             /* \w as an atom, and the word test of \b */
extern const size_t  k_word_n;
extern const CpRange k_word_class[];       /* \w inside a bracket class */
extern const size_t  k_word_class_n;
extern const ScriptTable k_scripts[];
extern const size_t      k_scripts_n;
/* Case-folding orbits: each is its length followed by its members. */
extern const uint32_t k_case_orbits[];
extern const size_t   k_case_orbits_n;
/* Codepoints that (?i) also matches against a multi-codepoint sequence. */
extern const uint32_t k_case_multi[];
extern const size_t   k_case_multi_n;

}  /* namespace ucd */
}  /* namespace rad */
