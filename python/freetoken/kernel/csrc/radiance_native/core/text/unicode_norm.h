/* unicode_norm.h -- NFC / NFD / NFKC / NFKD over codepoint vectors.
 *
 * The four forms are declared steps in a HuggingFace normaliser chain; a model that declares one
 * and is tokenised without it gets silently different token boundaries (spec §12). llama.cpp has
 * no equivalent. The tables live in unicode_norm_data.cpp, which documents how they were made.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rad {

/* One codepoint's decomposition. `off_*`/`len_*` index k_decomp_pool; a zero `len` means the
 * codepoint has no decomposition of that kind and stands for itself. */
struct DecompRow { uint32_t cpt, off_c, len_c, off_k, len_k; };
struct ComposeRow { uint32_t a, b, c; };
struct CccRow { uint32_t first, last; uint32_t ccc; };

extern const uint32_t   k_decomp_pool[];
extern const size_t     k_decomp_pool_n;
extern const DecompRow  k_decomp_rows[];
extern const size_t     k_decomp_rows_n;
extern const ComposeRow k_compose[];
extern const size_t     k_compose_n;
extern const CccRow     k_ccc[];
extern const size_t     k_ccc_n;

uint32_t unorm_ccc(uint32_t cpt);

/* In place, on codepoints. `compat` selects the K forms; following with unorm_compose gives the
 * C forms. */
void unorm_decompose(std::vector<uint32_t>& cpts, bool compat);
void unorm_compose(std::vector<uint32_t>& cpts);

/* UTF-8 in and out: the form the tokeniser's string-based step chain calls. */
std::string unorm_utf8(const std::string& in, bool compat, bool compose);

}  /* namespace rad */
