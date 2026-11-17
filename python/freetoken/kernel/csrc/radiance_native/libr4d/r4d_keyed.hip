/* r4d_keyed.hip -- device blocks keyed by the buffer they serve (r4d_keyed.h), and the release
 * the core calls when it frees memory (rad_plugin_freed). */
#include "r4d_keyed.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace r4d {

namespace {

struct Block { unsigned* p; int dev; };
using Key = std::tuple<int, int, const void*, const char*>;   /* row, device, owner, out */

std::mutex& mu() { static std::mutex m; return m; }
std::map<Key, Block>& blocks() { static std::map<Key, Block> b; return b; }
std::vector<FreedFn>& freed_fns() { static std::vector<FreedFn> v; return v; }

}  // namespace

unsigned* keyed_block(Keyed row, const void* owner, const void* out, size_t bytes) {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return nullptr;
    const Key key{ (int)row, dev, owner, static_cast<const char*>(out) };
    {
        std::lock_guard<std::mutex> lk(mu());
        if (auto it = blocks().find(key); it != blocks().end()) return it->second.p;
    }
    void* p = nullptr;
    if (hipMalloc(&p, bytes) != hipSuccess) return nullptr;
    /* Zero before any launch reads it. In the engine hipMemset is the AQL backend's and has
     * finished when it returns (core/device/aql.cpp); under HIP's own runtime (r4d_selftest) it
     * runs on the null stream, which a non-blocking stream does not wait for, so wait here. */
    if (hipMemset(p, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
        (void)hipFree(p);
        return nullptr;
    }
    unsigned* kept = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu());
        const auto [it, fresh] = blocks().emplace(key, Block{ (unsigned*)p, dev });
        if (fresh) return it->second.p;
        kept = it->second.p;
    }
    (void)hipFree(p);   /* another thread made this key's block in the meantime */
    return kept;
}

int on_freed(FreedFn f) {
    std::lock_guard<std::mutex> lk(mu());
    freed_fns().push_back(f);
    return 0;
}

}  // namespace r4d

/* The memory [base, base + bytes) is free and nothing launched against it runs any more: every
 * block kept for an output inside it goes. */
extern "C" void rad_plugin_freed(const void* base, int64_t bytes) {
    using namespace r4d;
    const char* lo = static_cast<const char*>(base);
    const char* hi = lo + bytes;
    std::vector<Block> gone;
    std::vector<FreedFn> fns;
    {
        std::lock_guard<std::mutex> lk(mu());
        for (auto it = blocks().begin(); it != blocks().end();) {
            const char* out = std::get<3>(it->first);
            if (out >= lo && out < hi) {
                gone.push_back(it->second);
                it = blocks().erase(it);
            } else {
                ++it;
            }
        }
        fns = freed_fns();
    }
    int was = 0;
    (void)hipGetDevice(&was);
    for (const Block& b : gone) {
        (void)hipSetDevice(b.dev);
        (void)hipFree(b.p);
    }
    (void)hipSetDevice(was);
    for (FreedFn f : fns) f(lo, hi);
}
