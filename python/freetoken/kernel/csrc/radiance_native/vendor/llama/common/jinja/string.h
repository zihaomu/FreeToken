#pragma once

#include <optional>
#include <string>
#include <vector>

#include "utils.h"

namespace jinja {

// allow differentiate between user input strings and template strings
// transformations should handle this information as follows:
// - one-to-one (e.g., uppercase, lowercase): preserve is_input flag
// - one-to-many (e.g., strip): if input string is marked as is_input, all resulting parts should be marked as is_input
// - many-to-one (e.g., concat): if ALL input parts are marked as is_input, resulting part should be marked as is_input
struct string_part {
    bool is_input = false; // may skip parsing special tokens if true
    std::string val;

    bool is_uppercase() const;
    bool is_lowercase() const;
};

struct string {
    std::vector<string_part> parts;
    string() = default;
    string(const std::string & v, bool user_input = false) {
        parts.push_back({user_input, v});
    }
    string(std::string && v, bool user_input = false) {
        parts.push_back({user_input, std::move(v)});
    }
    string(int v) {
        parts.push_back({false, std::to_string(v)});
    }
    string(double v) {
        parts.push_back({false, std::to_string(v)});
    }

    // mark all parts as user input
    void mark_input();

    std::string str() const;
    // Flattened equality without flattening: walk both part lists segment by segment.
    bool equals(const string & other) const;
    size_t length() const;
    void hash_update(hasher & hash) const noexcept;
    bool all_parts_are_input() const;
    bool is_uppercase() const;
    bool is_lowercase() const;

    // mark this string as input if other has ALL parts as input
    void mark_input_based_on(const string & other);

    // In place: the parts are appended to this string. Returns *this so callers can move it on.
    string & append(const string & other);

    // in-place transformations

    string uppercase();
    string lowercase();
    string capitalize();
    string titlecase();
    string strip(bool left, bool right, std::optional<const std::string_view> chars = std::nullopt);
};

} // namespace jinja
