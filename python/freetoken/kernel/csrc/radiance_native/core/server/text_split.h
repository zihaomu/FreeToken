/* text_split.h -- text that arrives in pieces, cut into the units a speech model reads one at a
 * time: sentences, or clauses (text_split.cpp).
 *
 * The realtime speech session (speech_stream.cpp) is sent text as a client produces it, a language
 * model's tokens or a typist's keystrokes, and speaks each unit as soon as it is complete, so the
 * first audio does not wait for the whole text. A unit is complete once the text after its closing
 * punctuation shows that it ended; the end of the input ends the last one.
 */
#pragma once

#include <string>
#include <vector>

namespace rad {
namespace server {

enum class TextSplit {
    None,       /* no cutting: the whole input is one unit */
    Sentence,   /* after . ! ? and their CJK, Indic and Arabic forms */
    Clause,     /* also after , ; and their full-width and Arabic forms */
};

/* The complete units at the front of `*pending`, trimmed, in order; `*pending` keeps the rest.
 * With `final` (the input has ended) the rest is a unit too and `*pending` is left empty.
 *
 * A unit ends at its closing punctuation together with any run of punctuation and closing quotes
 * or brackets after it ("Really?!", "he said.”"). After ASCII punctuation the end needs
 * whitespace to follow, so "3.14" and "a.b" do not end one; after full-width, CJK, Indic or Arabic
 * punctuation any next character does, since those scripts put no space there. A full stop after a
 * common abbreviation (Dr., Mr., e.g., etc.) or an initial (J., J.R.) ends nothing. A unit with
 * nothing to say (only punctuation) is dropped. */
std::vector<std::string> take_text_units(std::string* pending, TextSplit mode, bool final);

}  /* namespace server */
}  /* namespace rad */
