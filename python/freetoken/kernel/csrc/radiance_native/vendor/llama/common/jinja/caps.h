#pragma once

#include "runtime.h"

#include <string>
#include <map>

namespace jinja {

struct caps {
    bool supports_tools = true;
    bool supports_tool_calls = true;
    bool supports_system_role = true;
    bool supports_parallel_tool_calls = true;
    bool supports_preserve_reasoning = false; // support assistant message with reasoning_content

    // one of the 2 content capabilities must be true
    bool supports_string_content = true;
    bool supports_typed_content = false;

    bool supports_object_arguments = false;

    // RADIANCE: what the first probe render that threw said, empty when none did. The probes
    // ignore exceptions, so a template calling something this engine lacks and a template
    // without tool support both come back supports_tools = false; this tells them apart.
    std::string probe_error;

    // for reporting on server
    std::map<std::string, bool> to_map() const;

    // for debugging
    std::string to_string() const;
};

caps caps_get(jinja::program & prog);

} // namespace jinja
