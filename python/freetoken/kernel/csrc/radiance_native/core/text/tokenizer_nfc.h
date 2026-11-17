/* tokenizer_nfc.h -- NFC for the tokeniser's normaliser chain, applied only where it can change
 * something. The result is byte for byte unorm_utf8(text, false, true); tokenizer_nfc.cpp explains
 * why normalising only the stretches around non-boundary codepoints is equivalent. */
#pragma once
#include <string>

namespace rad {

/* Normalise `text` in place. Returns false when it was already NFC and is untouched. */
bool tok_nfc(std::string& text);

}  /* namespace rad */
