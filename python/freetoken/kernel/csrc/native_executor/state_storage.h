#pragma once
#include "batch_state.h"
#include "runtime/storage_view.h"
#include <map>

namespace freetoken {
// This provider copies backing only. Prefix identity, LRU and publication belong to FreeToken.
class StateStorage {
public:
    StateStorage(std::vector<rad::StorageRankView>,ExplicitBatchState&,int64_t budget);
    ~StateStorage();
    uint64_t snapshot(uint64_t sequence,const std::vector<int32_t>& prompt,uint64_t parent);
    int64_t restore(uint64_t sequence,const std::vector<int32_t>& prompt,uint64_t snapshot);
    void drop(uint64_t);
    rad::Json inspect() const;
private:
    struct Chunk {
        int group=0,rank=0;
        int state_copy=-1;
        int64_t start=0,blocks=0,bytes=0;
        void* data=nullptr;
        ~Chunk();
    };
    struct Snapshot {
        int32_t checkpoint=-1;
        ExplicitBatchState::SnapshotPosition position;
        std::vector<int32_t> tokens;
        std::vector<std::shared_ptr<Chunk>> chunks;
    };
    void require(int,const char*) const;
    std::vector<int32_t> slots(uint64_t) const;
    std::vector<rad::StorageRankView> ranks_;
    ExplicitBatchState& state_;
    int64_t budget_,checkpoint_bytes_=0;
    uint64_t next_=1,saves_=0,restores_=0,copied_out_=0,copied_in_=0;
    uint64_t cleared_state_slots_=0;
    std::map<uint64_t,Snapshot> snapshots_;
    std::vector<std::weak_ptr<Chunk>> chunks_;
    int64_t host_bytes() const;
    void audit(const Snapshot&,uint64_t);
};
}
