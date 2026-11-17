/* fleet/policy.h -- which cards a model loads on and who leaves to make room (docs/MULTIMODEL.md
 * §8.3, §8.4). Pure arithmetic on a snapshot, so the rules are tested without a card.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace rad::fleet {

/* A card as the fleet sees it now: the bytes a model may still claim on it (the driver's free
 * reading less the headroom). */
struct CardRoom {
    int     card = 0;
    int64_t free = 0;
};

/* A loaded model as a candidate to leave. */
struct Resident {
    size_t           id = 0;
    std::vector<int> cards;
    int64_t          charge = 0;    /* bytes it gives back on each of its cards when it leaves */
    int              priority = 0;
    bool             pin = false;   /* never leaves */
    bool             keep = false;  /* may not leave for this load (a member of its together group) */
    bool             busy = false;  /* requests in flight or waiting */
    double           idle_s = 0;    /* since its last request */
    double           restore_s = 1; /* how long it takes to come back */
    /* Parked already: it holds little VRAM and loses its host copy if it leaves, so it is the
     * last candidate whatever its priority. */
    bool             parked = false;
    /* Bytes it gives back on each of `cards`, when they differ (a together group leaving as one,
     * whose members sit on different cards); empty takes `charge` on every card. */
    std::vector<int64_t> charges;
    /* The models that leave with it (a together group): Resident::id of each, `id` included.
     * Empty means `id` alone. */
    std::vector<size_t>  members;

    int64_t gives(int card) const;
};

struct RoomPlan {
    std::vector<int>    cards;      /* the model's cards, ascending; rank i on cards[i] */
    std::vector<size_t> victims;    /* Resident::id of every model that leaves, in eviction order;
                                     * a unit's members follow one another */
    int64_t             displaced = 0;   /* bytes the victims held, summed over their cards */
};

/* The order models leave in: resident models before parked ones; then lower priority first; at
 * equal priority idle before busy; then the one that comes back cheapest relative to how long it
 * has been idle (idle_s / restore_s, highest first); then the lower id. True when `a` leaves
 * before `b`. */
bool leaves_before(const Resident& a, const Resident& b);

/* Cards for a model of `tp` ranks that needs `need` bytes on each, and the models that leave so it
 * fits. Candidates to leave are the residents that are neither pinned nor kept, at the model's
 * `priority` or lower; `forced` leave whatever happens (the other members of an exclusive group).
 * Among the card sets that can be made to fit, the one displacing the fewest bytes wins, then the
 * one whose cards keep the fewest other models (models spread before they share a card), then the
 * one with most cards in `prefer` (where the model was before), then the lowest cards. With
 * `fixed` set only that card set is considered. A card named in `need_on` needs that many bytes
 * instead of `need`: a parked model's ranks can each have a different part still on their cards.
 *
 * True with *out filled, or false with *why naming what is in the way. */
bool plan_room(const std::vector<CardRoom>& cards, const std::vector<Resident>& residents, int tp,
               int64_t need, int priority, const std::vector<size_t>& forced,
               const std::vector<int>& prefer, const std::vector<int>* fixed, RoomPlan* out,
               std::string* why, const std::map<int, int64_t>* need_on = nullptr);

/* The claim a model gets in a fleet of several (§6.3): its need plus what its flex floor asks for,
 * never above what it can use. `kv_min_tokens` and `experts_min_mib` are the YAML flex floors
 * (-1 unset). The cache's cost of T tokens is kv_fixed + (kv_ceiling - kv_fixed) * T / max_tokens. */
int64_t flex_claim(int64_t need, int64_t want, int64_t kv_fixed, int64_t kv_ceiling,
                   int64_t kv_one, int64_t max_tokens, int64_t experts, int64_t expert_floor,
                   int64_t kv_min_tokens, int64_t experts_min_mib);

}  // namespace rad::fleet
