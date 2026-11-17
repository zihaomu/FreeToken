/* fleet/broker.h -- how an engine asks the fleet for its part of the cards (docs/MULTIMODEL.md §6.3).
 *
 * An engine in a fleet of several models does not split a card by itself: the other models hold
 * part of it, and the fleet decides who leaves to make room. Once declare has told the engine what
 * it costs, the engine asks its broker for a claim between what it cannot start without and what it
 * could use at most; the broker makes room (evicting by the fleet's policy) and grants a claim, and
 * the engine splits that claim as a one-model process splits its card (Config::vram_claim).
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rad {

struct VramAsk {
    std::vector<int> cards;    /* the engine's cards, rank i on cards[i]; the claim is per card */
    int64_t need = 0;          /* the fixed part: weights, arena, queues, one full-length context */
    int64_t want = 0;          /* the most the engine can use: every expert, the cache's ceiling */
    /* The cache's terms, so a broker can price a token count: kv_fixed is state no context uses,
     * (kv_ceiling - kv_fixed) pages max_seqs x max_ctx tokens, and kv_one is the cache's part of
     * `need` (kv_fixed and the pages of one full-length context). */
    int64_t kv_fixed = 0;
    int64_t kv_ceiling = 0;
    int64_t kv_one = 0;
    int64_t max_tokens = 0;    /* max_seqs x max_ctx */
    int64_t experts = 0;       /* the whole expert plane, 0 for a dense model */
    int64_t expert_floor = 0;  /* the experts the host pool cannot hold */
};

class VramBroker {
public:
    virtual ~VramBroker() = default;
    /* Grants a claim per card of at least `ask.need`, blocking until it is free on every card of
     * the ask. The claim is at most `want` unless the model's configuration fixes it. RAD_OK with
     * *grant set, or an error with *why naming what is in the way (models that may not leave, a
     * card too small). */
    virtual int grant(const VramAsk& ask, int64_t* grant, std::string* why) = 0;

    /* Room for `bytes` of host memory the engine will pin beyond the pools the fleet already counted
     * (the tables placed in host memory), made and waited for as for a load's pools, and counted
     * from here until the load ends. False when it cannot be had, and nothing is counted. */
    virtual bool host_room(int64_t bytes) { (void)bytes; return true; }
    /* All that is counted for this load, about to be pinned, once more. It was there when counted,
     * and the load may have waited minutes for its VRAM since. False when it cannot be had: pinning
     * anyway gets the process killed by the kernel. */
    virtual bool host_pools_room() { return true; }
    /* Host memory this engine must leave free beside what it pins: the host's floor and what other
     * loads have counted on and not pinned yet. */
    virtual int64_t host_margin() { return 0; }

    /* ---- flex (docs/MULTIMODEL.md §6). A granted engine's cache may grow past its claim into
     * memory no model holds, and gives it back when the fleet needs it. */
    /* For the engine being granted: the most tokens its cache may grow to, 0 for none (it stays at
     * its claim), -1 for its ceiling. */
    virtual int64_t flex_kv_max_tokens() { return 0; }
    /* May `bytes` more be backed on each of `cards` now? Never evicts and never blocks; refused
     * while the fleet is making room for a load. From an engine's step loop. */
    virtual bool    flex_grow(const std::vector<int>& cards, int64_t bytes) {
        (void)cards; (void)bytes; return false;
    }
    /* After every flex_grow that said yes, once the engine has backed the bytes or given up: a load
     * or restore starting meanwhile waits for granted growth to land before it reads the cards'
     * free memory, or it would plan room that the growth then takes. */
    virtual void    flex_done() {}
    /* The bytes a cache on `cards` could still grow by now, for admission (KVManager::set_flex_room). */
    virtual int64_t flex_room(const std::vector<int>& cards) { (void)cards; return 0; }
    /* For the engine being granted: the most VRAM its experts may take on a card, the claim's
     * share included, as its slab grows into memory no model holds (docs/MULTIMODEL.md §6.7); 0
     * for no growth, -1 for every expert it keeps in host memory. */
    virtual int64_t flex_experts_max_bytes() { return 0; }
};

}  // namespace rad
