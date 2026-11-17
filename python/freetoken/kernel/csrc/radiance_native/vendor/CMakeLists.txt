# vendor/ -- llama.cpp lifts and third-party headers, tracked to an upstream commit and patched
# rather than edited (spec §12). UPSTREAM lists the sources and patches/ every local change;
# anything rewritten lives in core/.
#
# The public target is the INTERFACE library `rad_vendor`, which core/CMakeLists.txt links. This
# file alone decides what the rest of the tree can include from here.

# ---------------------------------------------------------------- compiled lifts
# Vendored sources compile with warnings off: they will never be fixed here, and editing upstream to
# silence them would break the patch-series rule.
add_library(rad_vendor_llama STATIC
  llama/src/unicode.cpp
  llama/src/unicode-data.cpp
  llama/common/unicode.cpp
  llama/common/jinja/lexer.cpp
  llama/common/jinja/parser.cpp
  llama/common/jinja/runtime.cpp
  llama/common/jinja/string.cpp
  llama/common/jinja/value.cpp
  llama/common/jinja/caps.cpp
  llama/common/chat.cpp
  llama/common/chat-peg-parser.cpp
  llama/common/chat-auto-parser-generator.cpp
  llama/common/chat-auto-parser-helpers.cpp
  llama/common/chat-diff-analyzer.cpp
  llama/common/peg-parser.cpp
  llama/common/json-schema-to-grammar.cpp
  llama/common/json-partial.cpp
  llama/common/regex-partial.cpp
  shim/llama_shim.cpp)
target_compile_options(rad_vendor_llama PRIVATE -w)
# The chat lift's JSON DOM is ours (core/util/json.h) since the DOM swap; its
# translation units include it by that path and link the same small library core does.
target_include_directories(rad_vendor_llama PRIVATE ${CMAKE_SOURCE_DIR}/core)
target_link_libraries(rad_vendor_llama PUBLIC rad_json)
# SYSTEM, so vendored headers included by our sources do not warn either (`-w` covers only the
# vendor's own translation units).
target_include_directories(rad_vendor_llama SYSTEM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  # shim/ first: the lifted files include "common.h", "log.h", "ggml.h" and "llama.h", which are
  # ours (vendor/shim/*.h). The path, not an edit, redirects them, so lifts stay byte-identical.
  ${CMAKE_CURRENT_SOURCE_DIR}/shim
  # llama/common, because the lifted files include each other and the jinja engine by basename.
  # Side effect: a bare "unicode.h" resolves to llama/common/unicode.h (streaming UTF-8), not
  # llama/src/unicode.h (codepoint tables), so core/ always spells the vendor path in full.
  ${CMAKE_CURRENT_SOURCE_DIR}/llama/common)

# cpp-httplib 0.64.0 is a split header/source library, so it is compiled. TLS is off: the server
# speaks plain HTTP and TLS termination belongs to a proxy in front of it.
add_library(rad_vendor_httplib STATIC cpp-httplib/httplib.cpp)
target_compile_options(rad_vendor_httplib PRIVATE -w)
target_include_directories(rad_vendor_httplib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp-httplib)
target_link_libraries(rad_vendor_httplib PRIVATE Threads::Threads)
target_compile_definitions(rad_vendor_httplib PUBLIC
  # llama.cpp's settings: long prompts in a form body or query string, many concurrent
  # connections, and no Nagle delay on streamed tokens.
  CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=1048576
  CPPHTTPLIB_LISTEN_BACKLOG=512
  CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
  CPPHTTPLIB_TCP_NODELAY=1
  # A WebSocket read waits this long for a frame. The server pings every 30 s, so a live peer --
  # one that answers the ping -- is never cut off; one that stops answering holds its socket,
  # and a server shutdown waiting on it, for a minute rather than httplib's five.
  CPPHTTPLIB_WEBSOCKET_READ_TIMEOUT_SECOND=60
  # The largest WebSocket message, its fragments together. A realtime client sends audio a few
  # hundred milliseconds a message; a larger message closes the connection.
  CPPHTTPLIB_WEBSOCKET_MAX_PAYLOAD_LENGTH=2097152)

# rapidyaml 0.16.0 ships as one amalgamated header; shim/ryml_impl.cpp compiles its implementation
# once. The include directory is SYSTEM so the header does not warn in our translation units. The
# YAML configuration reader (core/fleet/config.cpp) is its only user.
add_library(rad_vendor_ryml STATIC shim/ryml_impl.cpp)
target_compile_options(rad_vendor_ryml PRIVATE -w)
target_include_directories(rad_vendor_ryml SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rapidyaml)

# yyjson: the reader inside the tree's JSON DOM (core/util/json.cpp converts its tree into
# rad::Json, the DOM everything downstream consumes). One C file; the header rides on
# rad_vendor_yyjson's include path (<yyjson/yyjson.h>).
add_library(rad_vendor_yyjson STATIC yyjson/yyjson.c)
target_compile_options(rad_vendor_yyjson PRIVATE -w)
target_include_directories(rad_vendor_yyjson SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------- the public face
# Header-only vendors need only a path:
#   <cpp-httplib/httplib.h>      vendor/cpp-httplib/httplib.h   (or plain "httplib.h")
#   "llama/src/unicode.h"        the tokeniser's codepoint tables
#   "llama/common/jinja/*.h"     the jinja engine
#   <ryml.hpp>                   vendor/rapidyaml/ryml.hpp
add_library(rad_vendor INTERFACE)
target_include_directories(rad_vendor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rad_vendor INTERFACE rad_vendor_llama rad_vendor_httplib rad_vendor_ryml
                      rad_vendor_yyjson)
