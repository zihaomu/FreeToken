/* hf_model.cpp -- resolve a --model that is a Hugging Face repo id (hf_model.h).
 *
 * main.cpp calls resolve_hf_model() once, between config_parse and the Engine; bring-up then sees
 * a plain file or directory. Fetching is scripts/hfget.sh (curl and jq), run as a child process
 * from libexec/radiance/ beside the binary, installed or in the build tree.
 *
 * Downloads go to the XDG cache by default, or --download-dir (vLLM's name), not $RADIANCE_HOME,
 * which is a plugin search path whose default is not user-writable.
 *
 * An unreachable hub does not stop a restart: hfget.sh exits 3 when it changed nothing on disk,
 * and the copy a complete fetch left (marked by its success manifest) is served with a warning.
 * Any other failure may leave a partial file, so nothing is served. */
#include "hf_model.h"

#include "rad_internal.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace rad {

namespace fs = std::filesystem;

namespace {

bool repo_part(const std::string& p) {
    if (p.empty() || p.front() == '.' || p.front() == '-') return false;
    for (char ch : p) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-';
        if (!ok) return false;
    }
    return true;
}

/* <exe>/../libexec/radiance/hfget.sh, which is where the install puts it and where the build
 * copies it. */
std::string hfget_path() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return std::string();
    buf[n] = '\0';
    const fs::path p = fs::path(buf).parent_path() / ".." / "libexec" / "radiance" / "hfget.sh";
    std::error_code ec;
    return fs::is_regular_file(p, ec) ? p.lexically_normal().string() : std::string();
}

/* The script's exit status, or -1 when it could not be run. `rev` empty leaves the script's own
 * default: the repo's moving default branch. */
int run_hfget(const std::string& script, const std::string& repo, const std::string& dest,
              const std::string& rev) {
    const pid_t pid = fork();
    if (pid < 0) {
        RAD_ERR("hfget: fork: %s", std::strerror(errno));
        return -1;
    }
    if (pid == 0) {
        const char* argv[] = { "sh", script.c_str(), repo.c_str(), dest.c_str(),
                               rev.empty() ? nullptr : rev.c_str(), nullptr };
        execv("/bin/sh", const_cast<char* const*>(argv));
        std::fprintf(stderr, "hfget: /bin/sh: %s\n", std::strerror(errno));
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0)
        if (errno != EINTR) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

}  /* namespace */

bool hf_repo_id(const std::string& s) {
    std::error_code ec;
    if (s.empty() || fs::exists(s, ec)) return false;
    const size_t slash = s.find('/');
    if (slash == std::string::npos || s.find('/', slash + 1) != std::string::npos) return false;
    const std::string owner = s.substr(0, slash), name = s.substr(slash + 1);
    if (!repo_part(owner) || !repo_part(name)) return false;
    for (const char* suf : { ".rad", ".safetensors", ".gguf", ".json" })
        if (name.ends_with(suf)) return false;
    return true;
}

std::string hf_default_download_dir() {
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x)
        return std::string(x) + "/radiance/hf";
    if (const char* h = std::getenv("HOME"); h && *h)
        return std::string(h) + "/.cache/radiance/hf";
    return std::string();
}

std::string hf_served_path(const std::string& dir, std::string* why) {
    std::vector<std::string> rad;
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(dir, ec)) {
        std::error_code fe;
        if (e.is_regular_file(fe) && e.path().filename().string().ends_with(".rad"))
            rad.push_back(e.path().filename().string());
    }
    if (ec) {
        *why = dir + ": " + ec.message();
        return std::string();
    }
    if (rad.empty()) return dir;
    if (rad.size() == 1) return dir + "/" + rad[0];
    std::string names;
    for (size_t i = 0; i < rad.size(); ++i) names += (i ? ", " : "") + rad[i];
    *why = dir + " holds " + std::to_string(rad.size()) + " containers (" + names +
           "); pass the one to serve as --model " + dir + "/<file>";
    return std::string();
}

bool resolve_hf_model(Config& c) {
    if (!hf_repo_id(c.model)) return true;
    const std::string repo = c.model;
    const std::string root = c.download_dir.empty() ? hf_default_download_dir() : c.download_dir;
    if (root.empty()) {
        RAD_ERR("--model %s: no download directory -- neither XDG_CACHE_HOME nor HOME is set; "
                "name one with --download-dir", repo.c_str());
        return false;
    }
    const std::string dest = root + "/" + repo;
    std::error_code ec;
    const bool cached = fs::is_regular_file(dest + "/.hfget-manifest", ec);

    const std::string script = hfget_path();
    if (script.empty()) {
        RAD_ERR("--model %s: not a path here, and hfget.sh, which fetches a Hugging Face repo, is "
                "not installed beside this binary (libexec/radiance/)", repo.c_str());
        return false;
    }
    RAD_INFO("model: %s is not a path here; fetching it as a Hugging Face repo into %s",
             repo.c_str(), dest.c_str());
    const int rc = run_hfget(script, repo, dest, c.model_revision);
    if (rc != 0) {
        if (rc == 3 && cached) {
            RAD_WARN("model: %s could not be checked for changes (hfget.sh, above); serving the "
                     "copy fetched before", repo.c_str());
        } else {
            RAD_ERR("--model %s: not a path here, and fetching it as a Hugging Face repo failed "
                    "(hfget.sh exit %d, above)", repo.c_str(), rc);
            return false;
        }
    }
    std::string why;
    const std::string path = hf_served_path(dest, &why);
    if (path.empty()) {
        RAD_ERR("--model %s: %s", repo.c_str(), why.c_str());
        return false;
    }
    c.model = path;
    RAD_INFO("model: serving %s", c.model.c_str());
    return true;
}

}  /* namespace rad */
