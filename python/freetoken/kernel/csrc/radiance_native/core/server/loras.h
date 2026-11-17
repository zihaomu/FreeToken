/* loras.h -- which runtime LoRA adapters an image or video request asks for, and at what strength.
 *
 * Two spellings, merged:
 *
 *   `loras` in the body: a list whose entries are a name, "name:strength", or an object
 *   {"name": ..., "weight" | "strength" | "scale": ...}; or an object {name: strength}.
 *   `lora` is the same field.
 *
 *   <lora:name> or <lora:name:strength> in the prompt, as Stable Diffusion web UIs write it: the
 *   tag is taken out of the prompt, the text around it kept.
 *
 * A strength not given is 1. */
#pragma once
#include "util/json.h"

#include <string>
#include <utility>
#include <vector>

namespace rad {
namespace server {

/* RAD_OK with the adapters in `out` (possibly none) and their tags removed from *prompt;
 * RAD_E_INVAL with `why` for a malformed entry. `field` receives the field the error is about. */
int parse_loras(const Json& body, std::string* prompt,
                std::vector<std::pair<std::string, double>>* out, std::string* field,
                std::string* why);

}  /* namespace server */
}  /* namespace rad */
