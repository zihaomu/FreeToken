#include "server/nemotron35.h"
#include "rad_plugin.h"
#include "llama/src/unicode.h"
#include <cstring>

namespace rad::server {
static bool nemotron35_meta(const RadModelMeta* meta) {
    if (!meta) return false;
    const auto is = [&](const char* key, const char* value) {
        return !std::strcmp(rad_meta_gets(meta, key, ""), value);
    };
    const auto noncausal = [&](const char* key) { return is(key, "0") || is(key, "false"); };
    return is("model_type", "mistral3") && is("text_config.model_type", "ministral3") &&
           is("vision_config.model_type", "pixtral") && noncausal("is_causal") &&
           noncausal("text_config.is_causal");
}

bool nemotron35_rerank_meta(const RadModelMeta* meta) {
    return nemotron35_meta(meta) && !std::strcmp(rad_meta_gets(meta, "architectures", ""),
                                                 "Mistral3ForSequenceClassification");
}

bool nemotron35_retrieval_meta(const RadModelMeta* meta) {
    return nemotron35_meta(meta) && !nemotron35_rerank_meta(meta);
}

int nemotron35_rerank_format(const json& messages, std::vector<Nemotron35Piece>* pieces,
                             std::string* why) {
    if (!pieces)
        return RAD_E_INVAL;
    pieces->clear();
    auto no = [&](const char* message) {
        if (why)
            *why = message;
        pieces->clear();
        return RAD_E_INVAL;
    };
    if (!messages.is_array() || messages.size() != 2 || !messages[0].is_object() ||
        !messages[1].is_object() || messages[0].value("role", json()) != "query" ||
        messages[1].value("role", json()) != "document" || !messages[0].contains("content") ||
        !messages[1].contains("content"))
        return no("reranking requires one text query followed by one document");
    std::string texts[2];
    json image;
    for (int i = 0; i < 2; ++i) {
        const auto& content = messages[i]["content"];
        if (content.is_string())
            texts[i] = content.get<std::string>();
        else if (content.is_array()) {
            bool first = true;
            for (const auto& block : content) {
                if (!block.is_object())
                    return no("content must contain typed text/image blocks");
                const json type = block.value("type", json());
                if (type == "text") {
                    if (!block.contains("text") || !block["text"].is_string())
                        return no("text blocks require string text");
                    if (!first)
                        texts[i] += "\n\n";
                    texts[i] += block["text"].get<std::string>();
                    first = false;
                } else if (i == 1 && (type == "image" || type == "image_url")) {
                    if (!image.is_null())
                        return no("at most one image per document is supported");
                    image = block;
                } else
                    return no("queries must be text; documents support text and one image");
            }
        } else
            return no("content must be a string or a list of text/image blocks");
    }
    const auto escape = [](const std::string& s) {
        std::string out;
        for (char c : s) {
            out += c;
            if (c == '[' || c == '<')
                out += "\xe2\x80\x8c";
        }
        return out;
    };
    std::string rendered = "query: " + escape(texts[0]) + "\n\npassage: " + escape(texts[1]);
    if (!image.is_null()) {
        /* Jinja's Python trim includes Unicode whitespace. The fixed image/query prefix has
         * no leading whitespace; only the rendered text's trailing edge can be trimmed. */
        auto cps = unicode_cpts_from_utf8(rendered);
        while (!cps.empty() && (unicode_cpt_flags_from_cpt(cps.back()).is_whitespace ||
                                (cps.back() >= 0x1c && cps.back() <= 0x1f)))
            cps.pop_back();
        rendered.clear();
        for (auto cp : cps)
            rendered += unicode_cpt_to_utf8(cp);
        Nemotron35Piece p;
        p.image = true;
        p.content = std::move(image);
        pieces->push_back(std::move(p));
        rendered.insert(0, " ");
    }
    Nemotron35Piece p;
    p.text = std::move(rendered);
    pieces->push_back(std::move(p));
    return RAD_OK;
}

int nemotron35_format(const json& messages, std::vector<Nemotron35Piece>* pieces,
                      std::string* why) {
    if (!pieces) return RAD_E_INVAL;
    pieces->clear();
    auto no = [&](const char* message) {
        if (why) *why = message;
        pieces->clear();
        return RAD_E_INVAL;
    };
    if (!messages.is_array() || (messages.size() != 1 && messages.size() != 2))
        return no(
            "retrieval inputs require one user message with an optional leading system prefix");
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        const char* role = i + 1 == messages.size() ? "user" : "system";
        if (!msg.is_object() || !msg.contains("role") || msg["role"] != role ||
            !msg.contains("content"))
            return no(
                "retrieval inputs require one user message with an optional leading system prefix");
    }
    auto text = [&](const std::string& s) {
        if (!pieces->empty() && !pieces->back().image)
            pieces->back().text += s;
        else {
            Nemotron35Piece p;
            p.text = s;
            pieces->push_back(std::move(p));
        }
    };
    if (messages.size() == 2) {
        const auto& content = messages[0]["content"];
        std::string prefix;
        if (content.is_string())
            prefix = content.get<std::string>();
        else if (content.is_array()) {
            bool first = true;
            for (const auto& block : content) {
                if (!block.is_object() || block.value("type", json()) != "text" ||
                    !block.contains("text") || !block["text"].is_string())
                    return no("system content may contain only text blocks");
                if (!first) prefix += "\n\n";
                prefix += block["text"].get<std::string>();
                first = false;
            }
        } else
            return no("system content must be a string or a list of text blocks");
        while (!prefix.empty() && prefix.back() == ' ')
            prefix.pop_back();
        text(prefix + " ");
    }
    const auto escape = [](const std::string& s) {
        std::string out;
        for (char c : s) {
            out += c;
            if (c == '[' || c == '<') out += "\xe2\x80\x8c";
        }
        return out;
    };
    const auto& user = messages.back()["content"];
    if (user.is_string())
        text(escape(user.get<std::string>()));
    else if (user.is_array()) {
        /* Stable image-before-text canonicalization. The image part itself moves with its
         * bytes, so image identity cannot be lost when the original list starts with text. */
        for (const auto& block : user) {
            if (!block.is_object() || !block.contains("type") || !block["type"].is_string())
                return no("user content must contain typed text and image blocks");
            const auto type = block["type"].get<std::string>();
            if (type != "text" && type != "image_url" && type != "image" && type != "input_image")
                return no("user content supports only text and image blocks");
            if (type == "text" && (!block.contains("text") || !block["text"].is_string()))
                return no("user text blocks require string text");
        }
        bool emitted = false;
        for (const auto& block : user)
            if (block["type"] != "text") {
                if (emitted) text(" ");
                Nemotron35Piece p;
                p.image = true;
                p.content = block;
                pieces->push_back(std::move(p));
                emitted = true;
            }
        for (const auto& block : user)
            if (block["type"] == "text") {
                if (emitted) text(" ");
                text(escape(block["text"].get<std::string>()));
                emitted = true;
            }
    } else
        return no("user content must be a string or a list of text and image blocks");
    bool has_image = false;
    for (const auto& p : *pieces)
        has_image |= p.image;
    if (has_image)
        for (const auto& p : *pieces)
            if (!p.image && p.text.find("[IMG]") != std::string::npos)
                return no("a retrieval system prefix cannot add image placeholders");
    return RAD_OK;
}
} // namespace rad::server
