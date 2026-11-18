#include "state_storage.h"
#include "device/device.h"
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace freetoken {
using namespace rad;
StateStorage::Chunk::~Chunk() { if(data) rad_dev_free(data,RAD_MEM_HOST_PINNED); }
StateStorage::StateStorage(std::vector<StorageRankView> ranks,ExplicitBatchState& state,int64_t budget)
    :ranks_(std::move(ranks)),state_(state),budget_(budget) {
    for(const auto& r:ranks_) checkpoint_bytes_+=r.kv->checkpoint_bytes()*r.kv->checkpoint_slots();
    if(budget_<checkpoint_bytes_) throw std::runtime_error("prefix budget cannot hold its checkpoint backing");
}
StateStorage::~StateStorage() {
    for(const auto& r:ranks_) { (void)rad_dev_set(r.device); (void)rad_stream_sync(r.stream); }
    for(const auto& s:snapshots_) ranks_[0].kv->free_checkpoint(s.second.checkpoint);
}
void StateStorage::require(int rc,const char* message) const { if(rc<0) throw std::runtime_error(message); }
int64_t StateStorage::host_bytes() const {
    int64_t bytes=checkpoint_bytes_;
    for(const auto& weak:chunks_) if(auto chunk=weak.lock()) bytes+=chunk->bytes;
    return bytes;
}
std::vector<int32_t> StateStorage::slots(uint64_t sequence) const {
    auto& kv=*ranks_[0].kv;
    std::vector<int32_t> out(size_t(kv.n_groups()),-1);
    for(int g=0;g<kv.n_groups();++g) if(kv.plan(g)->stateful()) {
        out[size_t(g)]=kv.state_slot(sequence,g);
        if(out[size_t(g)]<0) throw std::runtime_error("state bundle has no physical slot");
    }
    return out;
}
uint64_t StateStorage::snapshot(uint64_t sequence,const std::vector<int32_t>& prompt,uint64_t parent) {
    Snapshot s;s.position=state_.snapshot_position(sequence);
    s.tokens.assign(prompt.begin(),prompt.begin()+s.position.tokens);
    int64_t previous=0;
    if(parent) {
        const auto& old=snapshots_.at(parent);
        previous=old.position.tokens;
        if(previous>s.position.tokens || !std::equal(old.tokens.begin(),old.tokens.end(),s.tokens.begin()))
            throw std::runtime_error("snapshot parent is not a prefix of this state");
        for(const auto& chunk:old.chunks)if(chunk->state_copy<0)s.chunks.push_back(chunk);
    }
    int64_t additional=0;
    for(const auto& r:ranks_) for(const auto& p:r.kv->plans()) if(p.paged() && !p.zero) {
        if(p.kind!=RAD_KV_FULL || p.streamed() || previous%p.block_size || s.position.tokens%p.block_size)
            throw std::runtime_error("snapshot requires full, non-streamed, whole KV pages");
        additional+=(s.position.tokens-previous)/p.block_size*p.bytes_per_block;
    }
    const bool full_state=std::getenv("FREETOKEN_NATIVE_SNAPSHOT_FULL_STATE")!=nullptr;
    if(full_state)for(const auto& r:ranks_)for(const auto& p:r.kv->plans())if(p.stateful() && !p.zero)
        additional+=(p.state_copies-1)*p.bytes_per_state;
    if(host_bytes()+additional>budget_ || ranks_[0].kv->free_checkpoint_slots()<1)
        throw std::runtime_error("prefix_capacity_exhausted");
    require(ranks_[0].kv->alloc_checkpoint(&s.checkpoint),"checkpoint allocation failed");
    const auto state_slots=slots(sequence);
    try {
        for(size_t rank=0;rank<ranks_.size();++rank) {
            const auto& r=ranks_[rank];require(rad_dev_set(r.device),"snapshot device selection failed");
            for(const auto& p:r.kv->plans()) if(p.paged() && !p.zero) {
                auto chunk=std::make_shared<Chunk>();chunk->rank=int(rank);chunk->group=p.index;
                chunk->start=previous/p.block_size;chunk->blocks=(s.position.tokens-previous)/p.block_size;
                chunk->bytes=chunk->blocks*p.bytes_per_block;
                if(!chunk->bytes)continue;
                chunk->data=rad_dev_alloc(chunk->bytes,RAD_MEM_HOST_PINNED);
                if(!chunk->data) throw std::runtime_error("snapshot pinned allocation failed");
                chunks_.push_back(chunk);s.chunks.push_back(chunk);
                const auto& table=ranks_[0].kv->block_table(sequence,p.index);
                if(int64_t(table.size())<chunk->start+chunk->blocks)throw std::runtime_error("short snapshot page table");
                const int64_t bytes=p.layer_bytes_per_block();
                for(int64_t l=0;l<p.n_layers;++l) for(int64_t i=0;i<chunk->blocks;++i) {
                    auto* src=static_cast<char*>(p.base)+l*p.layer_stride()+table[size_t(chunk->start+i)]*bytes;
                    auto* dst=static_cast<char*>(chunk->data)+(l*chunk->blocks+i)*bytes;
                    require(rad_memcpy_async(dst,src,bytes,r.stream),"KV snapshot copy failed");
                }
            }
            if(full_state)for(const auto& p:r.kv->plans())if(p.stateful() && !p.zero)for(int copy=1;copy<p.state_copies;++copy) {
                auto chunk=std::make_shared<Chunk>();chunk->rank=int(rank);chunk->group=p.index;chunk->state_copy=copy;
                chunk->bytes=p.bytes_per_state;chunk->data=rad_dev_alloc(chunk->bytes,RAD_MEM_HOST_PINNED);
                if(!chunk->data)throw std::runtime_error("snapshot replay allocation failed");
                chunks_.push_back(chunk);s.chunks.push_back(chunk);
                const auto slot=state_slots[size_t(p.index)]+copy*p.n_seq_states();
                for(int64_t l=0;l<p.n_layers;++l)require(rad_memcpy_async(
                    static_cast<char*>(chunk->data)+l*p.layer_bytes_per_state(),
                    r.kv->state_ptr(p.index,int32_t(l),int32_t(slot)),p.layer_bytes_per_state(),r.stream),"replay snapshot failed");
            }
            require(r.kv->checkpoint_save(state_slots.data(),int(state_slots.size()),s.checkpoint,r.stream),"state snapshot failed");
            require(rad_stream_sync(r.stream),"snapshot fence failed");
        }
    } catch(...) {
        for(const auto& r:ranks_) { (void)rad_dev_set(r.device);(void)rad_stream_sync(r.stream); }
        ranks_[0].kv->free_checkpoint(s.checkpoint);throw;
    }
    chunks_.erase(std::remove_if(chunks_.begin(),chunks_.end(),[](const auto& p){return p.expired();}),chunks_.end());
    const auto id=next_++;snapshots_.emplace(id,std::move(s));++saves_;copied_out_+=additional;return id;
}
int64_t StateStorage::restore(uint64_t sequence,const std::vector<int32_t>& prompt,uint64_t id) {
    const auto& s=snapshots_.at(id);
    if(s.tokens.size()>=prompt.size() || !std::equal(s.tokens.begin(),s.tokens.end(),prompt.begin()))
        throw std::runtime_error("snapshot does not belong to the requested token prefix");
    require(ranks_[0].kv->ensure(sequence,s.position.tokens,0),"restore page allocation failed");
    const auto state_slots=slots(sequence);
    for(const auto& r:ranks_) { require(rad_dev_set(r.device),"restore device selection failed"); require(rad_stream_sync(r.stream),"restore initial fence failed"); }
    // The snapshot owns committed copies. Recycled MTP replay copies still need fresh-lease
    // initialization; dropping their pending clears would carry the previous tenant's factors.
    for(const auto& clear:ranks_[0].kv->state_clears())for(const auto& r:ranks_) {
        const auto* p=r.kv->plan(clear.group);if(!p || !p->stateful())continue;
        require(rad_dev_set(r.device),"state initialization device selection failed");
        for(int64_t l=0;l<p->n_layers;++l)require(rad_memset_async(
            r.kv->state_ptr(clear.group,int32_t(l),clear.slot),0,p->layer_bytes_per_state(),r.stream),"state initialization failed");
        ++cleared_state_slots_;
    }
    for(const auto& chunk:s.chunks) {
        const auto& r=ranks_.at(size_t(chunk->rank));const auto& p=*r.kv->plan(chunk->group);
        require(rad_dev_set(r.device),"KV restore device selection failed");
        if(chunk->state_copy>=0) {
            const auto slot=state_slots[size_t(p.index)]+chunk->state_copy*p.n_seq_states();
            for(int64_t l=0;l<p.n_layers;++l)require(rad_memcpy_async(
                r.kv->state_ptr(p.index,int32_t(l),int32_t(slot)),
                static_cast<char*>(chunk->data)+l*p.layer_bytes_per_state(),p.layer_bytes_per_state(),r.stream),"replay restore failed");
            copied_in_+=chunk->bytes;continue;
        }
        const auto& table=ranks_[0].kv->block_table(sequence,p.index);const int64_t bytes=p.layer_bytes_per_block();
        if(int64_t(table.size())<chunk->start+chunk->blocks)throw std::runtime_error("short restore page table");
        for(int64_t l=0;l<p.n_layers;++l)for(int64_t i=0;i<chunk->blocks;++i) {
            auto* src=static_cast<char*>(chunk->data)+(l*chunk->blocks+i)*bytes;
            auto* dst=static_cast<char*>(p.base)+l*p.layer_stride()+table[size_t(chunk->start+i)]*bytes;
            require(rad_memcpy_async(dst,src,bytes,r.stream),"KV restore copy failed");
        }
        copied_in_+=chunk->bytes;
    }
    for(const auto& r:ranks_) {
        require(rad_dev_set(r.device),"state restore device selection failed");
        require(r.kv->checkpoint_restore(state_slots.data(),int(state_slots.size()),s.checkpoint,r.stream),"state restore failed");
        require(rad_stream_sync(r.stream),"restore fence failed");
    }
    // Fresh-slot clears must not overwrite the restored bundle at its first continuation.
    ranks_[0].kv->clear_state_clears();
    if(std::getenv("FREETOKEN_NATIVE_DEBUG_STATE_STORAGE"))audit(s,sequence);
    state_.restore_position(sequence,s.position);++restores_;return s.position.tokens;
}
void StateStorage::audit(const Snapshot& s,uint64_t sequence) {
    uint64_t checked=0;
    const auto state_slots=slots(sequence);
    for(const auto& chunk:s.chunks) {
        const auto& r=ranks_.at(size_t(chunk->rank));const auto& p=*r.kv->plan(chunk->group);
        require(rad_dev_set(r.device),"storage audit device selection failed");
        if(chunk->state_copy>=0) {
            std::vector<char> actual(size_t(chunk->bytes));
            const auto slot=state_slots[size_t(p.index)]+chunk->state_copy*p.n_seq_states();
            require(rad_memcpy_2d_async(actual.data(),p.layer_bytes_per_state(),r.kv->state_ptr(p.index,0,int32_t(slot)),
                p.layer_stride(),p.layer_bytes_per_state(),p.n_layers,r.stream),"replay audit failed");
            require(rad_stream_sync(r.stream),"replay audit fence failed");
            if(std::memcmp(actual.data(),chunk->data,size_t(chunk->bytes)))throw std::runtime_error("replay snapshot byte mismatch");
            checked+=chunk->bytes;continue;
        }
        const auto& table=ranks_[0].kv->block_table(sequence,p.index);
        const int64_t bytes=p.layer_bytes_per_block();
        for(int64_t l=0;l<p.n_layers;++l) {
            std::vector<char> layer(size_t(p.layer_stride()));
            require(rad_memcpy_async(layer.data(),static_cast<char*>(p.base)+l*p.layer_stride(),p.layer_stride(),r.stream),"page audit failed");
            require(rad_stream_sync(r.stream),"page audit fence failed");
            for(int64_t i=0;i<chunk->blocks;++i) {
                const auto* expected=static_cast<char*>(chunk->data)+(l*chunk->blocks+i)*bytes;
                if(std::memcmp(expected,layer.data()+table[size_t(chunk->start+i)]*bytes,size_t(bytes)))
                    throw std::runtime_error("KV snapshot byte mismatch");
                checked+=bytes;
            }
        }
    }
    RAD_INFO("FreeToken state storage audit: snapshot_tokens=%lld byte_checks=%llu exact=1",(long long)s.position.tokens,(unsigned long long)checked);
}
void StateStorage::drop(uint64_t id) {
    auto it=snapshots_.find(id);if(it==snapshots_.end())throw std::runtime_error("unknown snapshot lease");
    ranks_[0].kv->free_checkpoint(it->second.checkpoint);snapshots_.erase(it);
}
Json StateStorage::inspect() const {
    return {{"owner","freetoken"},{"snapshots",snapshots_.size()},{"host_bytes",host_bytes()},
        {"budget_bytes",budget_},{"checkpoint_backing_bytes",checkpoint_bytes_},
        {"snapshots_saved",saves_},{"snapshots_restored",restores_},
        {"restored_fresh_state_slots_cleared",cleared_state_slots_},
        {"kv_copied_out_bytes",copied_out_},{"kv_copied_in_bytes",copied_in_}};
}
}
