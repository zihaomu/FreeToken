/* recipe.h -- which quantiser makes which weight, for rad-convert (spec §4.4).
 *
 * A recipe is ordered rules from a logical weight name to a quantiser and options; the first match
 * decides, and an unmatched weight keeps the checkpoint's encoding. Two spellings:
 *
 *   in a file          PATTERN  QUANTISER  key=value key=value ...      # a comment
 *   on the command     PATTERN=QUANTISER:key=value,key=value
 *
 * A pattern is a glob over the whole name: `*` matches any run (dots included), `?` one
 * character. `$NAME` or `${NAME}` in an option is expanded from the environment when the rule is
 * read, so the container records the actual values.
 */
#pragma once
#include "../rad_internal.h"
#include "rad_quant.h"

#include <string>
#include <utility>
#include <vector>

namespace rad {

class Registry;

struct RecipeRule {
    std::string pattern, quantizer;
    std::vector<std::pair<std::string, std::string>> options;   /* in the order written */
    std::string origin;                                         /* "file:12" or "--quant" */

    /* The options as one canonical string (`key=value`, comma-separated, sorted by key), as a
     * container entry records and --reuse compares them. */
    std::string options_text() const;
};

/* Does `name` match glob `pattern`? */
bool recipe_glob(const char* pattern, const char* name);

class Recipe {
public:
    /* Rules from a recipe file's text, appended. RAD_OK, or RAD_E_FORMAT with the line in `why`. */
    int add_text(const std::string& text, const std::string& origin, std::string* why);
    /* One rule in the command-line spelling, appended. */
    int add_flag(const std::string& spec, std::string* why);

    /* The rule that decides `name`, or null when none matches. */
    const RecipeRule* match(const std::string& name) const;

    const std::vector<RecipeRule>& rules() const { return rules_; }
    bool empty() const { return rules_.empty(); }
    /* Every rule in the file spelling, one per line, options expanded, as the container records. */
    std::string text() const;

private:
    std::vector<RecipeRule> rules_;
};

/* A rule with its quantiser resolved and its options typed by the quantiser's own option table,
 * so a misspelt key or a value of the wrong type is refused before declare. Integers are decimal
 * (the container records the text, and --reuse compares it, so `010` must mean what it reads as);
 * an empty value or one that is not finite is refused.
 *
 * The RadParam strings point into `keys` and `vals`, so a compiled rule is built in place and never
 * copied or moved. */
struct CompiledRule {
    const RecipeRule*        rule = nullptr;
    const RadQuantizerInfo*  q = nullptr;
    std::vector<std::string> keys, vals;
    std::vector<RadParam>    params;

    CompiledRule() = default;
    CompiledRule(const CompiledRule&) = delete;
    CompiledRule& operator=(const CompiledRule&) = delete;

    /* RAD_OK; RAD_E_NOTFOUND when no loaded quantiser has the rule's name (the message lists the
     * ones that are); RAD_E_INVAL for an option the quantiser does not take or a value its type
     * refuses. The reason, with the rule's origin, is in *why. */
    int build(const RecipeRule& r, const Registry& reg, std::string* why);
};

}  /* namespace rad */
