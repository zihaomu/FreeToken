/* lora/slots.h -- which adapter each runtime-LoRA slot holds, who is using it, and when a slot may
 * be written again. Host bookkeeping only: the engine copies the adapter's bytes (engine_lora.cpp)
 * and this table decides which slot, publishes the name, hands out holds and frees slots.
 *
 * A slot's life:
 *
 *   free --load--> loaded --unload, no request uses it--> free (cooling)
 *                    |                                         ^
 *                    +--unload, requests use it--> draining ---+ the last of them finishes
 *
 * A request holds the adapter it names from admission until its reply is done or it is cancelled
 * (the server keeps the hold in the request's Generation), so an adapter is never unloaded under a
 * request. An unload of an adapter in use drains it: no new request can name it (acquire answers
 * as for a name never loaded), the requests already holding it finish, and the last hold to go
 * frees the slot. A load of the same name is refused until then.
 *
 * Cooling: a freed slot is written again only after the step loop has gone round
 * kSlotCoolRounds times (`epoch` counts its rounds). A step that read the slot may still be on
 * the card when the last hold goes, issued ahead of the readback; an idle round has no step in
 * flight and a busy round collects one, so with one step issued ahead two rounds suffice.
 *
 * Thread-safe: every call takes the table's lock. Loads are serialised by the caller (one at a
 * time between reserve and publish), which is what keeps a reserved slot from being handed out
 * twice. */
#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace lora {

class SlotTable {
public:
    /* `n` empty slots; `epoch` is the step loop's round counter, read when a slot is freed and
     * when a load looks for one. It must outlive the table and every hold the table handed out. */
    void reset(int n, const std::atomic<uint64_t>* epoch);
    int  size() const;

    /* A free slot for `name`, or why not:
     *   RAD_E_INVAL  an adapter of that name is loaded
     *   RAD_E_STATE  an adapter of that name is draining (*err says how many requests remain)
     *   RAD_E_FULL   no slot is free; *cooling says one would be after the step loop has gone
     *                round a few more times, which the caller may wait for */
    int  reserve(const std::string& name, int* slot, bool* cooling, std::string* err);
    /* The adapter in `slot` is on the card: from now on a request may name it. Returns its uid,
     * fresh for every load (never 0), which keys its prefix-cache blocks. */
    uint64_t publish(int slot, const std::string& name, const std::string& dir);

    /* The slot, uid and a hold for a loaded adapter that is not draining; RAD_E_NOTFOUND
     * otherwise. The adapter is not freed while the hold lives. */
    int  acquire(const std::string& name, int32_t* slot, uint64_t* uid, std::shared_ptr<void>* hold);

    /* Unpublish `name`. RAD_OK with *in_flight 0: the slot is free (cooling). RAD_OK with
     * *in_flight > 0: those requests still use it, so it is draining and frees itself when the
     * last of them finishes; a second unload answers the same with the count as it is then.
     * RAD_E_NOTFOUND: nothing of that name is loaded. */
    int  unload(const std::string& name, int* in_flight, std::string* err);

    /* Free the least recently acquired loaded slot that no request holds, for a load that found
     * none free; its name in *evicted. RAD_E_FULL when every loaded slot is held. */
    int  evict_idle(std::string* evicted);

    /* The loaded adapters a request may name, in slot order: not the draining ones. */
    std::vector<std::string> names() const;
    /* Requests holding the adapter `name` (draining or not), 0 for none or no such adapter. */
    int  in_flight(const std::string& name) const;

private:
    struct Slot {
        std::string name, dir;
        uint64_t    uid = 0;            /* 0: free */
        bool        used = false;       /* ever held an adapter, so `freed_at` means something */
        bool        draining = false;
        uint64_t    freed_at = 0;       /* `epoch` when it was last freed */
        uint64_t    last_use = 0;       /* acquire counter at its last acquire or publish */
    };
    void free_slot(Slot& s, const char* how);   /* under mu_ */
    void release(uint64_t uid);                 /* a hold's deleter */
    uint64_t now() const { return epoch_ ? epoch_->load(std::memory_order_acquire) : 0; }

    mutable std::mutex            mu_;
    std::vector<Slot>             slots_;
    std::map<uint64_t, int>       refs_;        /* uid -> holds */
    uint64_t                      next_uid_ = 1;
    uint64_t                      uses_ = 0;
    const std::atomic<uint64_t>*  epoch_ = nullptr;
};

/* Step-loop rounds a freed slot waits before it is written again (see Cooling above). */
inline constexpr uint64_t kSlotCoolRounds = 3;

}  /* namespace lora */
}  /* namespace rad */
