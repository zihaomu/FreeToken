#pragma once
#include "mem/kv.h"

namespace rad {
struct StorageRankView {
    int device;
    KVManager* kv;
    RadStream stream;
};
}
