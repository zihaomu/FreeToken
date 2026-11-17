/* engine_hostmem.cpp -- see engine_hostmem.h. */
#include "engine_hostmem.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>

namespace rad {

bool cgroup_room(const std::string& dir, bool v2, int64_t* room) {
    auto read_ll = [](const std::string& f, long long* v) {
        std::ifstream in(f);
        std::string t;
        if (!(in >> t)) return false;
        char* e = nullptr;
        *v = std::strtoll(t.c_str(), &e, 10);
        return e && *e == '\0';
    };
    auto stat_ll = [](const std::string& f, const char* key, long long* v) {
        std::ifstream in(f);
        std::string k;
        long long x = 0;
        while (in >> k >> x) if (k == key) { *v = x; return true; }
        return false;
    };
    long long lim = 0, used = 0, file = 0;
    if (v2) {
        if (!read_ll(dir + "/memory.max", &lim)) return false;           /* "max" fails to parse */
        if (!read_ll(dir + "/memory.current", &used)) return false;
        (void)stat_ll(dir + "/memory.stat", "file", &file);
    } else {
        if (!read_ll(dir + "/memory.limit_in_bytes", &lim) || lim >= ((long long)1 << 62))
            return false;
        if (!read_ll(dir + "/memory.usage_in_bytes", &used)) return false;
        if (!stat_ll(dir + "/memory.stat", "total_cache", &file))
            (void)stat_ll(dir + "/memory.stat", "cache", &file);
    }
    const long long hard = used > file ? used - file : 0;
    *room = lim > hard ? (int64_t)(lim - hard) : 0;
    return true;
}

void cgroup_levels(const std::string& root, std::string path, bool v2,
                   const std::function<bool(const std::string& dir, int64_t room)>& visit) {
    for (;;) {
        const std::string dir = root + (path == "/" ? std::string() : path);
        int64_t room = 0;
        if (cgroup_room(dir, v2, &room) && !visit(dir, room)) return;
        if (path.empty() || path == "/") break;
        const size_t up = path.find_last_of('/');
        path = up == 0 || up == std::string::npos ? "/" : path.substr(0, up);
    }
}

void memory_cgroups(const std::function<bool(const std::string&, int64_t)>& visit) {
    std::ifstream f("/proc/self/cgroup");
    std::string line;
    bool go = true;
    while (go && std::getline(f, line)) {
        const size_t a = line.find(':'), b = a == std::string::npos ? a : line.find(':', a + 1);
        if (b == std::string::npos) continue;
        const std::string ctrl = line.substr(a + 1, b - a - 1);
        std::string path = line.substr(b + 1);
        const bool v2 = ctrl.empty();
        bool memory = v2;
        for (size_t i = 0; !memory && i < ctrl.size();) {
            const size_t j = std::min(ctrl.find(',', i), ctrl.size());
            memory = ctrl.compare(i, j - i, "memory") == 0;
            i = j + 1;
        }
        if (!memory) continue;
        const std::string root = v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory";
        cgroup_levels(root, path, v2, [&](const std::string& dir, int64_t room) {
            return go = visit(dir, room);
        });
    }
}

int64_t cgroup_least_room() {
    int64_t least = INT64_MAX;
    memory_cgroups([&](const std::string&, int64_t room) {
        least = std::min(least, room);
        return true;
    });
    return least;
}

}  /* namespace rad */
