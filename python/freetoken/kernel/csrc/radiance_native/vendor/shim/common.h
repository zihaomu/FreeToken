/* vendor/shim/common.h -- NOT a lift. This file is ours.
 *
 * llama.cpp's common/common.h is 1000 lines of CLI arguments, sampling parameters, model
 * downloading and llama.h types. The lifted chat and PEG-parser sources need eleven string
 * helpers out of it and nothing else. Providing those here, with upstream's implementations
 * (MIT, llama.cpp 06938ac12 common/common.{h,cpp}) transcribed unchanged, is what lets the lifted
 * files compile BYTE-FOR-BYTE as upstream wrote them -- which is the whole point of the
 * patch-series rule in spec §12.
 *
 * If a future refresh of vendor/ brings in a lift that needs more of common.h, the choice is to
 * add it here or to rewrite that lift into core/. It is not to start editing the lift.
 */
#pragma once

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "ggml.h"    /* GGML_ASSERT, which string_format uses */
#include "llama.h"   /* llama_token and LLAMA_TOKEN_NULL, which common_grammar_trigger uses */

#ifdef __GNUC__
#    define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#else
#    define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
inline std::string string_format(const char* fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int size = vsnprintf(nullptr, 0, fmt, ap);
    GGML_ASSERT(size >= 0);
    std::vector<char> buf((size_t)size + 1);
    const int size2 = vsnprintf(buf.data(), (size_t)size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), (size_t)size);
}

inline void string_replace_all(std::string& s, const std::string& search,
                               const std::string& replace) {
    if (search.empty()) return;
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0, last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

inline std::string regex_escape(const std::string& s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

inline std::string string_join(const std::vector<std::string>& values,
                               const std::string& separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) result << separator;
        result << values[i];
    }
    return result.str();
}

inline std::vector<std::string> string_split(const std::string& str,
                                             const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0, end = str.find(delimiter);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    parts.push_back(str.substr(start));
    return parts;
}

inline std::string string_repeat(const std::string& str, size_t n) {
    if (n == 0) return "";
    std::string result;
    result.reserve(str.length() * n);
    for (size_t i = 0; i < n; ++i) result += str;
    return result;
}

inline std::string string_strip(const std::string& str) {
    size_t start = 0, end = str.size();
    while (start < end && std::isspace((unsigned char)str[start])) ++start;
    while (end > start && std::isspace((unsigned char)str[end - 1])) --end;
    return str.substr(start, end - start);
}

template <class T>
static std::vector<T> string_split(const std::string& str, char delim) {
    static_assert(!std::is_same<T, std::string>::value,
                  "Please use the specialized version for std::string");
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}

template <>
inline std::vector<std::string> string_split<std::string>(const std::string& str, char delim) {
    std::vector<std::string> parts;
    size_t begin_pos = 0, delim_pos = str.find(delim);
    while (delim_pos != std::string::npos) {
        parts.emplace_back(str.substr(begin_pos, delim_pos - begin_pos));
        begin_pos = delim_pos + 1;
        delim_pos = str.find(delim, begin_pos);
    }
    parts.emplace_back(str.substr(begin_pos));
    return parts;
}

inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

inline bool string_starts_with(std::string_view str, char prefix) {
    return !str.empty() && str.front() == prefix;
}

inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool string_remove_suffix(std::string& str, std::string_view suffix) {
    if (string_ends_with(str, suffix)) {
        str.resize(str.size() - suffix.size());
        return true;
    }
    return false;
}

/* The longest suffix of `str` that is a prefix of `stop`, i.e. where a stop string might be
 * starting. The streaming detokeniser in core/text/tokenizer.cpp solves the same problem for
 * itself; this one is here because the lifted partial-parse code calls it. */
inline size_t string_find_partial_stop(std::string_view str, std::string_view stop) {
    if (!str.empty() && !stop.empty()) {
        const size_t max_len = std::min(str.size(), stop.size());
        const char last_char = str.back();
        for (size_t len = max_len; len > 0; --len) {
            if (stop[len - 1] == last_char && string_ends_with(str, stop.substr(0, len)))
                return str.size() - len;
        }
    }
    return std::string::npos;
}

/* ------------------------------------------------------------------ types
 *
 * Three enums and a struct that live in common.h upstream and are named in the lifted headers'
 * public signatures. They are transcribed unchanged; a value renumbered here would silently mean
 * something else in a lifted .cpp. */

enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,             /* same as deepseek, using message.reasoning_content */
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY,  /* leave inline in <think> tags when streaming */
    COMMON_REASONING_FORMAT_DEEPSEEK,         /* extract, including in streaming deltas */
};

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string                 value;
    llama_token                 token = LLAMA_TOKEN_NULL;
};

/* common.h's token renderer, on the same never-reached branch as the llama_model accessors
 * above. Radiance's own is rad::Tokenizer::piece(); this one exists so chat.cpp compiles. */
std::string common_token_to_piece(const llama_vocab* vocab, llama_token token, bool special);
