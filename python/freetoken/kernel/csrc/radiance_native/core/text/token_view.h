/* token_view.h -- the sampler's VocabView over a real vocabulary (sample/vocab_view.h). */
#pragma once
#include "sample/vocab_view.h"
#include "text/tokenizer.h"

#include <memory>
#include <string>
#include <vector>

namespace rad {

class TokenView final : public VocabView {
public:
    /* Pieces are built once (token_piece returns a reference), saving a string per candidate per
     * grammar step on the step path.
     *
     * They come from piece(), not Vocab::text(): text() is the vocab's own encoding (for
     * byte-level BPE the GPT-2 alphabet, where a space is 'Ġ'), while piece() runs the decoder
     * chain and yields the output bytes vocab_view.h asks for. With text() every grammar would
     * forbid tokens containing spaces or newlines. Special tokens are rendered, because an empty
     * piece would satisfy any grammar position. */
    int init(std::shared_ptr<const Vocab> v) {
        v_ = std::move(v);
        RAD_TRY(tok_.init(v_));
        pieces_.resize((size_t)v_->n_tokens());
        for (int32_t i = 0; i < v_->n_tokens(); ++i)
            pieces_[(size_t)i] = tok_.piece(i, /*render_special=*/true);
        return RAD_OK;
    }
    int32_t n_tokens() const override { return v_ ? v_->n_tokens() : 0; }
    const std::string& token_piece(int32_t id) const override {
        static const std::string kEmpty;
        return (id >= 0 && (size_t)id < pieces_.size()) ? pieces_[(size_t)id] : kEmpty;
    }
    bool is_eog(int32_t id) const override { return v_ && v_->is_eog(id); }
    int  tokenize(const std::string& s, std::vector<int32_t>* out) const override {
        return out ? tok_.encode(s, *out, /*add_special=*/false, /*parse_special=*/false)
                   : RAD_E_INVAL;
    }
    int  tokenize_special(const std::string& s, std::vector<int32_t>* out) const override {
        return out ? tok_.encode(s, *out, /*add_special=*/false, /*parse_special=*/true)
                   : RAD_E_INVAL;
    }
    const Tokenizer& tok() const { return tok_; }

private:
    std::shared_ptr<const Vocab> v_;
    Tokenizer                    tok_;
    std::vector<std::string>     pieces_;
};

}  /* namespace rad */
