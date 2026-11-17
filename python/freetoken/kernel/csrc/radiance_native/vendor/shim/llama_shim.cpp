/* vendor/shim/llama_shim.cpp -- NOT a lift. This file is ours.
 *
 * Definitions for the llama.h surface declared in vendor/shim/llama.h. See the comment there for
 * why they exist and why none of them can be reached with model == nullptr, which is the only
 * way radiance calls into the lifted chat code.
 */
#include "common.h"
#include "llama.h"

int32_t llama_chat_apply_template(const char*, const llama_chat_message*, size_t, bool,
                                  char*, int32_t) {
    /* The legacy route, refused. A hardcoded table of recognised templates is the failure mode
     * this project replaced with a declared chain; falling into it silently would undo that. */
    return -1;
}

const char*        llama_model_chat_template(const llama_model*, const char*) { return nullptr; }
const llama_vocab* llama_model_get_vocab(const llama_model*)                  { return nullptr; }
llama_token        llama_vocab_bos(const llama_vocab*)      { return LLAMA_TOKEN_NULL; }
llama_token        llama_vocab_eos(const llama_vocab*)      { return LLAMA_TOKEN_NULL; }
bool               llama_vocab_get_add_bos(const llama_vocab*) { return false; }
bool               llama_vocab_get_add_eos(const llama_vocab*) { return false; }

std::string common_token_to_piece(const llama_vocab*, llama_token, bool) { return {}; }
