/* kernel_list.h -- how a kernel hierarchy is spelt in --kernels and $RADIANCE_KERNELS.
 *
 * The engine (core/config.cpp) and the tools (tools/iface.cpp) both read it here, so a list a
 * plugin was validated with in rad-kbench loads the same libraries when the engine serves. Entries
 * are separated by ',' or ':'; empty entries (a doubled or trailing separator) are dropped, since an
 * entry of "" matches no library and would warn on every start.
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace rad {

inline std::vector<std::string> rad_split_kernel_list(std::string_view s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',' || c == ':') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

}  /* namespace rad */
