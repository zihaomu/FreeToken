/* rad_deepseek41_chat.h -- DeepSeek-V4.1's reply format (reasoning and DSML tool calls) as data.
 *
 * The model's own encoder (encoding/encoding.py in the checkpoint) writes an assistant turn as
 *
 *   [reasoning</think>]content\n\n<｜DSML｜ calls>
 *   <｜DSML｜ invoke name="NAME">
 *   <｜DSML｜ parameter name="ARG" string="true">raw text</｜DSML｜ parameter>
 *   <｜DSML｜ parameter name="ARG" string="false">JSON</｜DSML｜ parameter>
 *   </｜DSML｜ invoke>
 *   ...
 *   </｜DSML｜ calls><｜end▁of▁sentence｜>
 *
 * Every call sits in one section, each argument's tag says whether its value is a string or JSON,
 * and the tag names carry a leading space (V4 wrote `<｜DSML｜tool_calls>`, V3.2
 * `<｜DSML｜function_calls>`). The generation prompt ends in `<think>` with thinking on and in a
 * bare `</think>` with it off, so the reply starts inside the block only when the prompt opened
 * it.
 *
 * The template analysis cannot state the typed argument tags, so the plugin declares the format;
 * the reply-format tests read this same declaration. */
#pragma once
#include "rad_builder.h"

namespace rad {
namespace arch {

inline const RadChatFormat* deepseek41_chat_format() {
    static const char* const kBreaks[]   = { "<｜DSML｜ calls>" };
    static const char* const kTriggers[] = { "<｜DSML｜ calls>" };
    static const RadChatFormat f = [] {
        RadChatFormat c{};
        c.struct_size           = (uint32_t)sizeof(RadChatFormat);
        c.name                  = "plugin:deepseek41";
        c.reasoning_open        = "<think>";
        c.reasoning_close       = "</think>";
        /* A section that starts inside an unclosed reasoning block ends it there. */
        c.reasoning_breaks      = kBreaks;
        c.n_reasoning_breaks    = 1;
        c.reasoning_in_prompt   = RAD_CHAT_THINK_FROM_PROMPT;

        c.section_open          = "<｜DSML｜ calls>";
        c.section_close         = "</｜DSML｜ calls>";
        c.name_prefix           = "<｜DSML｜ invoke name=\"";
        c.name_suffix           = "\">\n";
        c.call_end              = "</｜DSML｜ invoke>";
        c.arg_prefix            = "<｜DSML｜ parameter name=\"";
        c.arg_name_suffix       = "\"";
        c.arg_value_prefix      = " string=\"true\">";
        c.arg_json_value_prefix = " string=\"false\">";
        c.arg_value_suffix      = "</｜DSML｜ parameter>";
        c.string_value          = RAD_CHAT_VALUE_RAW;
        c.other_value           = RAD_CHAT_VALUE_JSON;
        c.value_newlines        = 0;

        c.triggers              = kTriggers;
        c.n_triggers            = 1;
        return c;
    }();
    return &f;
}

}  /* namespace arch */
}  /* namespace rad */
