#pragma once

#include "util/json.h"

#include <functional>
#include <memory>
#include <string>

std::string json_schema_to_grammar(const rad::Json & schema,
                                   bool force_gbnf = false);

class common_schema_converter;

// Probes a JSON schema to extract information about its structure and type constraints.
class common_schema_info {
    std::unique_ptr<common_schema_converter> impl_;

  public:
    common_schema_info();
    ~common_schema_info();

    common_schema_info(const common_schema_info &) = delete;
    common_schema_info & operator=(const common_schema_info &) = delete;
    common_schema_info(common_schema_info &&) noexcept;
    common_schema_info & operator=(common_schema_info &&) noexcept;

    void resolve_refs(rad::Json & schema);
    bool resolves_to_string(const rad::Json & schema);
    // RADIANCE: whether no value satisfies the schema -- `false`, or a `not` of a schema every
    // value satisfies (`{}`, or types naming all six JSON types).
    bool unsatisfiable(const rad::Json & schema) const;
};

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const rad::Json &)> add_schema;
    std::function<void(rad::Json &)> resolve_refs;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string gbnf_format_literal(const std::string & literal);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});
