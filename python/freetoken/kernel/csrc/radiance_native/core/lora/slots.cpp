/* lora/slots.cpp -- see slots.h. */
#include "lora/slots.h"
#include "rad_internal.h"

namespace rad {
namespace lora {

void SlotTable::reset(int n, const std::atomic<uint64_t>* epoch) {
    std::lock_guard<std::mutex> lk(mu_);
    slots_.assign(n > 0 ? (size_t)n : 0, Slot{});
    refs_.clear();
    epoch_ = epoch;
}

int SlotTable::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return (int)slots_.size();
}

int SlotTable::reserve(const std::string& name, int* slot, bool* cooling, std::string* err) {
    std::lock_guard<std::mutex> lk(mu_);
    *slot = -1;
    *cooling = false;
    const uint64_t t = now();
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot& x = slots_[i];
        if (x.uid != 0 && x.name == name) {
            if (!x.draining) {
                *err = "an adapter named '" + name + "' is already loaded";
                return RAD_E_INVAL;
            }
            /* The old adapter's requests still run on its slot; the name loads again once they
             * are gone (POST /v1/unload_lora_adapter says how many remain). */
            const auto it = refs_.find(x.uid);
            const int n = it == refs_.end() ? 0 : it->second;
            *err = fmt("an adapter named '%s' is being unloaded: %d request%s still use%s it", name.c_str(),
                       n, n == 1 ? "" : "s", n == 1 ? "s" : "");
            return RAD_E_STATE;
        }
    }
    for (size_t i = 0; i < slots_.size(); ++i) {
        const Slot& x = slots_[i];
        if (x.uid != 0) continue;
        if (x.used && t < x.freed_at + kSlotCoolRounds) { *cooling = true; continue; }
        *slot = (int)i;
        return RAD_OK;
    }
    *err = *cooling ? std::string("the free slot was unloaded a moment ago and a step on the card may "
                                  "still read it; retry")
                    : fmt("all %zu adapter slots are in use (--max-loras)", slots_.size());
    return RAD_E_FULL;
}

uint64_t SlotTable::publish(int slot, const std::string& name, const std::string& dir) {
    std::lock_guard<std::mutex> lk(mu_);
    if (slot < 0 || (size_t)slot >= slots_.size()) return 0;
    Slot& s = slots_[(size_t)slot];
    s.name = name;
    s.dir = dir;
    s.uid = next_uid_++;
    s.used = true;
    s.draining = false;
    s.last_use = ++uses_;
    return s.uid;
}

int SlotTable::acquire(const std::string& name, int32_t* slot, uint64_t* uid,
                       std::shared_ptr<void>* hold) {
    uint64_t u = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < slots_.size() && !u; ++i) {
            Slot& x = slots_[i];
            if (x.uid == 0 || x.draining || x.name != name) continue;
            u = x.uid;
            *slot = (int32_t)i;
            x.last_use = ++uses_;
            ++refs_[u];
        }
    }
    if (!u) return RAD_E_NOTFOUND;
    *uid = u;
    /* Made outside the lock: the deleter takes it, and runs at once if the allocation fails. */
    *hold = std::shared_ptr<void>(nullptr, [this, u](void*) { release(u); });
    return RAD_OK;
}

void SlotTable::free_slot(Slot& s, const char* how) {
    RAD_INFO("lora: '%s' unloaded%s (uid %llu)", s.name.c_str(), how, (unsigned long long)s.uid);
    s.uid = 0;
    s.name.clear();
    s.dir.clear();
    s.draining = false;
    s.freed_at = now();
}

void SlotTable::release(uint64_t uid) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = refs_.find(uid);
    if (it == refs_.end()) return;
    if (--it->second > 0) return;
    refs_.erase(it);
    /* The last request of a draining adapter: the unload it was waiting for happens now. */
    for (Slot& s : slots_)
        if (s.uid == uid && s.draining) free_slot(s, " -- its last request finished");
}

int SlotTable::unload(const std::string& name, int* in_flight, std::string* err) {
    std::lock_guard<std::mutex> lk(mu_);
    *in_flight = 0;
    for (Slot& x : slots_) {
        if (x.uid == 0 || x.name != name) continue;
        const auto it = refs_.find(x.uid);
        const int n = it == refs_.end() ? 0 : it->second;
        if (n > 0) {
            if (!x.draining)
                RAD_INFO("lora: '%s' draining -- %d request%s still use%s it, new ones are refused",
                         name.c_str(), n, n == 1 ? "" : "s", n == 1 ? "s" : "");
            x.draining = true;
            *in_flight = n;
            return RAD_OK;
        }
        free_slot(x, "");
        return RAD_OK;
    }
    *err = "no adapter named '" + name + "' is loaded";
    return RAD_E_NOTFOUND;
}

int SlotTable::evict_idle(std::string* evicted) {
    std::lock_guard<std::mutex> lk(mu_);
    Slot* best = nullptr;
    for (Slot& x : slots_) {
        if (x.uid == 0 || x.draining || refs_.count(x.uid)) continue;
        if (!best || x.last_use < best->last_use) best = &x;
    }
    if (!best) return RAD_E_FULL;
    if (evicted) *evicted = best->name;
    free_slot(*best, " to make room");
    return RAD_OK;
}

std::vector<std::string> SlotTable::names() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    for (const Slot& s : slots_)
        if (s.uid != 0 && !s.draining) out.push_back(s.name);
    return out;
}

int SlotTable::in_flight(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const Slot& s : slots_) {
        if (s.uid == 0 || s.name != name) continue;
        const auto it = refs_.find(s.uid);
        return it == refs_.end() ? 0 : it->second;
    }
    return 0;
}

}  /* namespace lora */
}  /* namespace rad */
