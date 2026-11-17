/* vendor/shim/llama.h -- NOT a lift. This file is ours.
 *
 * llama.cpp's common.h includes llama.h, and the lifted chat sources reach through it for
 * exactly one type and one sentinel. Taking llama.h would take the engine we are replacing, so
 * the two names are declared here. They agree with radiance's own token type by construction:
 * a token id is an int32_t everywhere in this project, and -1 is not a token.
 */
#pragma once

#include <cstdint>

typedef int32_t llama_token;

#define LLAMA_TOKEN_NULL (-1)

/* ------------------------------------------------------------------ the llama_model surface
 *
 * chat.cpp's `common_chat_templates_init()` can take a llama_model and read the template, the
 * BOS/EOS spellings and the add_bos/add_eos flags off it. Radiance has no llama_model: the .rad
 * carries all four (RadVocabHeader.chat_template, bos, eos, add_bos, add_eos) and core/text/
 * chat.cpp passes them in as the overrides upstream already supports.
 *
 * So these are declared and defined (vendor/shim/llama_shim.cpp) purely so the lifted file
 * compiles unedited. Every one of them is on a branch guarded by `if (model)`, and radiance
 * always passes model == nullptr, so none of them is ever reached. The one exception is
 * llama_chat_apply_template(), which is the LEGACY non-jinja template route; it returns a
 * failure, because that route is a hardcoded table of known templates and taking it would
 * reintroduce exactly the "recognise the model" failure mode spec §12 removes. */
struct llama_model;
struct llama_vocab;

struct llama_chat_message {
    const char* role;
    const char* content;
};

int32_t     llama_chat_apply_template(const char* tmpl, const llama_chat_message* chat,
                                      size_t n_msg, bool add_ass, char* buf, int32_t length);
const char* llama_model_chat_template(const llama_model* model, const char* name);
const llama_vocab* llama_model_get_vocab(const llama_model* model);
llama_token llama_vocab_bos(const llama_vocab* vocab);
llama_token llama_vocab_eos(const llama_vocab* vocab);
bool        llama_vocab_get_add_bos(const llama_vocab* vocab);
bool        llama_vocab_get_add_eos(const llama_vocab* vocab);
