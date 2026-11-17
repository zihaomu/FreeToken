/* vocab_view.h -- what the sampler needs to know about a vocabulary.
 *
 * The sampler needs the token count, each token's piece and whether it ends generation, plus
 * tokenising a short literal for DRY's sequence breakers and GBNF's `<token>` syntax. This view
 * keeps the tokeniser (core/text/) out of the sampler and its tests; a virtual call per token per
 * grammar step is host work the device does not wait on.
 *
 * text/token_view.h implements it over the real vocab; SimpleVocab over a table, for tests.
 */
#pragma once
#include "../rad_internal.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

class VocabView {
public:
    virtual ~VocabView() = default;

    virtual int32_t n_tokens() const = 0;

    /* The token's exact bytes as appended to the output stream. The grammar walks these; a
     * display form would make it admit the wrong token set. */
    virtual const std::string& token_piece(int32_t id) const = 0;

    /* End-of-generation: EOS, EOT, and whatever else this model ends on. A grammar admits one
     * only when some stack has run to completion. */
    virtual bool is_eog(int32_t id) const = 0;

    /* Tokenise a short literal. Only DRY's sequence breakers and GBNF's `<token>` form need it,
     * both optional, so a view without a tokeniser returns RAD_E_UNSUPPORTED. */
    virtual int tokenize(const std::string& text, std::vector<int32_t>* out) const {
        (void)text; (void)out;
        return RAD_E_UNSUPPORTED;
    }

    /* As tokenize, but recognising special tokens, so `<tool_call>` comes back as the one control
     * token. Grammar triggers resolve through this. */
    virtual int tokenize_special(const std::string& text, std::vector<int32_t>* out) const {
        (void)text; (void)out;
        return RAD_E_UNSUPPORTED;
    }

    /* ---------------- what a grammar backend derives from this vocabulary ----------------
     *
     * A GBNF mask works over a decoded form of the whole vocabulary (pieces as code points plus a
     * trie), which costs a full scan and is computed once. It is cached here, with the view's own
     * lifetime, because an external table could only key on the view's address, and a new view
     * built in the same storage would then be served the old model's decoded vocabulary and
     * compiled grammars. Opaque and mutable: its contents are the backend's business and filling
     * it does not change what the vocabulary says. */
    mutable std::shared_ptr<void> gbnf_cache;
};

/* A vocabulary held as a table, for the sampler and grammar tests. */
class SimpleVocab : public VocabView {
public:
    SimpleVocab() = default;
    explicit SimpleVocab(std::vector<std::string> pieces) : pieces_(std::move(pieces)) {}

    int32_t add(std::string piece, bool eog = false) {
        int32_t id = (int32_t)pieces_.size();
        pieces_.push_back(std::move(piece));
        if (eog) eog_.push_back(id);
        return id;
    }

    int32_t n_tokens() const override { return (int32_t)pieces_.size(); }

    const std::string& token_piece(int32_t id) const override {
        static const std::string empty;
        if (id < 0 || id >= (int32_t)pieces_.size()) return empty;
        return pieces_[id];
    }

    bool is_eog(int32_t id) const override {
        for (int32_t e : eog_) if (e == id) return true;
        return false;
    }

    /* A table has no special tokens of its own: every piece is matched as written. */
    int tokenize_special(const std::string& text, std::vector<int32_t>* out) const override {
        return tokenize(text, out);
    }

    /* Longest match over the table, not BPE. Enough for a DRY breaker or a `<token>` literal in a
     * test; real tokenisation lives in core/text/. */
    int tokenize(const std::string& text, std::vector<int32_t>* out) const override {
        if (!out) return RAD_E_INVAL;
        size_t pos = 0;
        while (pos < text.size()) {
            int32_t best = -1;
            size_t  best_len = 0;
            for (int32_t id = 0; id < (int32_t)pieces_.size(); ++id) {
                const std::string& p = pieces_[id];
                if (p.empty() || p.size() <= best_len) continue;
                if (text.compare(pos, p.size(), p) == 0) { best = id; best_len = p.size(); }
            }
            if (best < 0) return RAD_E_NOTFOUND;
            out->push_back(best);
            pos += best_len;
        }
        return RAD_OK;
    }

private:
    std::vector<std::string> pieces_;
    std::vector<int32_t>     eog_;
};

}  /* namespace rad */
