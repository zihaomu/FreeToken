#pragma once
#include <cstdint>

namespace ft_native {
constexpr uint32_t magic = 0x574e5446;
constexpr uint16_t version = 2;
constexpr uint32_t max_payload = 8 * 1024 * 1024;
enum Kind : uint16_t {
    Ready = 1, Generate = 2, Admitted = 3, Tokens = 4, Finish = 5,
    Cancel = 6, Shutdown = 7, Error = 8, Stats = 9, FinalTokens = 10
};
struct Header {
    uint32_t magic_value = magic;
    uint16_t protocol_version = version;
    uint16_t kind = 0;
    uint32_t bytes = 0;
    uint32_t reserved = 0;
    uint64_t request = 0;
    uint64_t epoch = 0;
};
static_assert(sizeof(Header) == 32);
}
