/* machine_key.cpp -- the machine half of the tuning cache key. Separate from tunecache.cpp, which
 * is pure file I/O and must link without a device layer. A cache from another machine gives wrong
 * answers, so the key names everything that changes the answer and nothing else.
 */
#include "tunecache.h"

#include <cstring>

namespace rad {

std::string rad_machine_key() {
    RadDeviceProps p{};
    const int n = rad_dev_count();
    return rad_machine_key_of(n, n > 0 && rad_dev_props(0, &p) >= 0 ? &p : nullptr);
}

std::string rad_machine_key_of(int n, const RadDeviceProps* dev) {
    /* No device to name at all. */
    if (n <= 0 || !dev) {
#if defined(__x86_64__)
        return "host-x86_64";
#elif defined(__aarch64__)
        return "host-aarch64";
#else
        return "host-unknown";
#endif
    }

    /* Device count, because a collective's variant depends on the peer count; card name, because
     * boards of one arch differ in clocks and memory width. The host backend's device name is the
     * CPU model, which host kernels' tuning depends on (an AVX2 part's answer is not an AVX-512
     * part's), so its key is `host-x<N>-<cpu model>`. */
    const RadDeviceProps& p = *dev;
    std::string name = p.name;
    for (char& c : name)
        if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))
            c = '_';
    while (name.size() > 1 && name.back() == '_') name.pop_back();
    if (name.empty()) name = "unknown";

    const char* arch = p.arch[0] ? p.arch : "host";
    return fmt("%s-x%d-%s", arch, n, name.c_str());
}

}  /* namespace rad */
