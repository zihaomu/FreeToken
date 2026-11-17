/* hf_model.h -- a model source (the `model` key, or a tool's --model) that names a Hugging Face
 * repo instead of a local path.
 *
 * `model: StillDeadcode/minicpm5-2b-fp8` fetches the repo with scripts/hfget.sh into
 * the download directory and serves what it fetched: the one container at the repo's top level,
 * or the directory as a checkpoint when it holds none. Every start re-checks the hub, so a
 * changed file is fetched again before bring-up; a start that cannot reach the hub serves the
 * copy a complete earlier fetch left. */
#pragma once

#include "rad_core.h"

#include <string>

namespace rad {

/* Is `s` a repo id ("owner/name") rather than a path? A path that exists always wins. A repo id is
 * exactly two non-empty parts of [A-Za-z0-9._-], neither starting with '.' or '-', and a name that
 * does not end like a model file (.rad, .safetensors, .gguf, .json): a missing local file should
 * be reported as missing, not looked up on the hub. */
bool hf_repo_id(const std::string& s);

/* $XDG_CACHE_HOME/radiance/hf, else $HOME/.cache/radiance/hf; empty if neither is set. */
std::string hf_default_download_dir();

/* What to serve from a fetched repo directory: its one top-level .rad, or the directory itself (a
 * checkpoint) when it holds none. Several containers are ambiguous: empty, and *why names them. */
std::string hf_served_path(const std::string& dir, std::string* why);

/* If c.model is a repo id, fetch it and point c.model at what to serve. False after an error has
 * been logged. */
bool resolve_hf_model(Config& c);

}  /* namespace rad */
