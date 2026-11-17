/* Nemotron 3.5's retrieval template, with images kept typed until token assembly.
 * Other embedding models continue through their existing template/raw-input paths. */
#pragma once
#include "server/iface.h"
#include "util/json.h"

struct RadModelMeta;

namespace rad::server {
struct Nemotron35Piece {
    bool image = false;
    std::string text;
    json content; // the original image part, including its data URL
};

bool nemotron35_retrieval_meta(const RadModelMeta* meta);
bool nemotron35_rerank_meta(const RadModelMeta* meta);
int nemotron35_format(const json& messages, std::vector<Nemotron35Piece>* pieces, std::string* why);
int nemotron35_rerank_format(const json& messages, std::vector<Nemotron35Piece>* pieces,
                             std::string* why);
} // namespace rad::server
