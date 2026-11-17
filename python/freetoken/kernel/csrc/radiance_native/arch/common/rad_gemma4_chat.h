/* rad_gemma4_chat.h -- Gemma 4's reply format (reasoning and tool calls) as data.
 *
 * Template analysis finds nothing in Gemma 4's chat template (llama.cpp uses a hand-written
 * handler), so without this the model would get no reasoning split and refuse tool requests.
 * This restates the model's tokenizer_config.json `response_template` in the engine's terms:
 *
 *   thinking      <|channel>thought\n ... <channel|>
 *   tool calls    <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3,flag:true}<tool_call|>, repeated
 *   the object    unquoted keys, strings between <|"|> and <|"|> with their text verbatim, every
 *                 other value as JSON writes it
 *   the turn      ends at <turn|>; after its calls the model writes <|tool_response> and stops
 *                 (generation_config.json names it beside <eos> and <turn|>)
 *
 * The model also emits an extra thought channel after the answer and a lone <channel|>.
 * `reasoning_anywhere` reads the first as reasoning and drops the second.
 *
 * Shared because two plugins serve Gemma 4 (arch/gemma4 with bf16 and fp8 experts, arch/gemma4_bf16
 * for the CT-INT4 dense release) and the format belongs to the model; the reply-format tests read
 * this same declaration. */
#pragma once
#include "rad_builder.h"

namespace rad {
namespace arch {

inline const RadChatFormat* gemma4_chat_format() {
    static const char* const kBreaks[]   = { "<|tool_call>" };
    static const char* const kTriggers[] = { "<|tool_call>" };
    static const RadChatFormat f = [] {
        RadChatFormat c{};
        c.struct_size         = (uint32_t)sizeof(RadChatFormat);
        c.name                = "plugin:gemma4";
        c.reasoning_open      = "<|channel>thought\n";
        c.reasoning_close     = "<channel|>";
        /* A call that starts inside an unclosed thought ends it there. */
        c.reasoning_breaks    = kBreaks;
        c.n_reasoning_breaks  = 1;
        /* Thinking off: the template closes an empty channel in the generation prompt. On: the
         * model opens its own, or the prompt has opened it after a tool response. */
        c.reasoning_in_prompt = RAD_CHAT_THINK_FROM_PROMPT;
        c.reasoning_anywhere  = 1;

        c.call_open           = "<|tool_call>";
        c.call_close          = "<tool_call|>";
        c.name_prefix         = "call:";
        c.name_suffix         = "";      /* the name ends at the object's `{` */
        c.call_end            = "";      /* and the object's `}` ends the call */
        c.args_form           = RAD_CHAT_ARGS_OBJECT;
        c.object_string_open  = "<|\"|>";
        c.object_string_close = "<|\"|>";
        c.object_bare_keys    = 1;

        c.triggers            = kTriggers;
        c.n_triggers          = 1;
        return c;
    }();
    return &f;
}

}  /* namespace arch */
}  /* namespace rad */
