/* fleet/policy.cpp -- card choice and eviction order (policy.h). */
#include "fleet/policy.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace rad::fleet {

int64_t Resident::gives(int card) const {
    for (size_t i = 0; i < cards.size(); ++i)
        if (cards[i] == card) return i < charges.size() ? charges[i] : charge;
    return 0;
}

bool leaves_before(const Resident& a, const Resident& b) {
    if (a.parked != b.parked) return !a.parked;
    if (a.priority != b.priority) return a.priority < b.priority;
    if (a.busy != b.busy) return !a.busy;
    const double sa = a.idle_s / std::max(a.restore_s, 0.05);
    const double sb = b.idle_s / std::max(b.restore_s, 0.05);
    if (sa != sb) return sa > sb;
    return a.id < b.id;
}

namespace {

/* Every ascending subset of `n` indices of size k, in lexicographic order. */
void subsets(int n, int k, std::vector<std::vector<int>>* out) {
    if (k > n || k <= 0) return;
    std::vector<int> pick(k);
    for (int i = 0; i < k; ++i) pick[i] = i;
    for (;;) {
        out->push_back(pick);
        int i = k - 1;
        while (i >= 0 && pick[i] == n - k + i) --i;
        if (i < 0) return;
        ++pick[i];
        for (int j = i + 1; j < k; ++j) pick[j] = pick[j - 1] + 1;
    }
}

bool on(const Resident& r, int card) {
    return std::find(r.cards.begin(), r.cards.end(), card) != r.cards.end();
}

}  // namespace

bool plan_room(const std::vector<CardRoom>& cards, const std::vector<Resident>& residents, int tp,
               int64_t need, int priority, const std::vector<size_t>& forced,
               const std::vector<int>& prefer, const std::vector<int>* fixed, RoomPlan* out,
               std::string* why, const std::map<int, int64_t>* need_on) {
    auto need_of = [&](int card) {
        if (need_on)
            if (const auto it = need_on->find(card); it != need_on->end()) return it->second;
        return need;
    };
    if (tp <= 0 || tp > (int)cards.size()) {
        *why = "it needs " + std::to_string(tp) + " cards and " + std::to_string(cards.size()) +
               " are available";
        return false;
    }
    std::vector<const Resident*> order;
    for (const Resident& r : residents)
        if (!r.pin && !r.keep && r.priority <= priority) order.push_back(&r);
    std::sort(order.begin(), order.end(),
              [](const Resident* a, const Resident* b) { return leaves_before(*a, *b); });
    auto is_forced = [&](const Resident& r) {
        return std::find(forced.begin(), forced.end(), r.id) != forced.end();
    };

    std::vector<std::vector<int>> sets;
    if (fixed) {
        std::vector<int> idx;
        for (int c : *fixed)
            for (size_t i = 0; i < cards.size(); ++i)
                if (cards[i].card == c) idx.push_back((int)i);
        if ((int)idx.size() != tp) {
            *why = "its cards are not all available";
            return false;
        }
        std::sort(idx.begin(), idx.end());
        sets.push_back(idx);
    } else {
        subsets((int)cards.size(), tp, &sets);
    }

    bool found = false;
    std::tuple<int64_t, int, int, size_t> best{};
    int64_t short_best = INT64_MAX;
    for (size_t si = 0; si < sets.size(); ++si) {
        const std::vector<int>& set = sets[si];
        std::map<int, int64_t> room;
        for (int i : set) room[cards[(size_t)i].card] = cards[(size_t)i].free;
        std::vector<const Resident*> gone;
        auto take = [&](const Resident* r) {
            gone.push_back(r);
            for (int c : r->cards)
                if (room.count(c)) room[c] += r->gives(c);
        };
        for (const Resident& r : residents)
            if (is_forced(r)) take(&r);
        auto shortfall = [&] {
            int64_t s = 0;
            for (auto& [c, f] : room) s += std::max<int64_t>(need_of(c) - f, 0);
            return s;
        };
        for (const Resident* r : order) {
            if (shortfall() == 0) break;
            if (std::find(gone.begin(), gone.end(), r) != gone.end()) continue;
            bool helps = false;
            for (int c : r->cards)
                if (room.count(c) && room[c] < need_of(c)) helps = true;
            if (helps) take(r);
        }
        const int64_t sf = shortfall();
        if (sf > 0) {
            short_best = std::min(short_best, sf);
            continue;
        }
        int64_t displaced = 0;
        for (const Resident* r : gone)
            for (int c : r->cards) displaced += r->gives(c);
        int not_preferred = 0;
        for (int i : set)
            if (std::find(prefer.begin(), prefer.end(), cards[(size_t)i].card) == prefer.end())
                ++not_preferred;
        /* Models that stay on the set's cards: fewer is better, so models spread over the cards
         * before they share one and each computes on a card of its own while it can. */
        int crowd = 0;
        for (const Resident& r : residents) {
            if (std::find(gone.begin(), gone.end(), &r) != gone.end()) continue;
            for (int i : set)
                if (on(r, cards[(size_t)i].card)) ++crowd;
        }
        const auto key = std::make_tuple(displaced, crowd, not_preferred, si);
        if (found && !(key < best)) continue;
        found = true;
        best = key;
        out->cards.clear();
        for (int i : set) out->cards.push_back(cards[(size_t)i].card);
        out->victims.clear();
        for (const Resident* r : gone) {
            if (r->members.empty()) out->victims.push_back(r->id);
            for (size_t m : r->members) out->victims.push_back(m);
        }
        out->displaced = displaced;
    }
    if (found) return true;

    /* Name what stays in the way: every resident that may not leave for this one. */
    std::string blocking;
    for (const Resident& r : residents)
        if (r.pin || r.keep || r.priority > priority)
            blocking += (blocking.empty() ? "" : ", ") + std::to_string(r.id);
    *why = "short by " + std::to_string(short_best == INT64_MAX ? need : short_best) +
           " bytes even after every model allowed to leave has left";
    if (!blocking.empty()) *why += "; staying: #" + blocking;
    return false;
}

int64_t flex_claim(int64_t need, int64_t want, int64_t kv_fixed, int64_t kv_ceiling,
                   int64_t kv_one, int64_t max_tokens, int64_t experts, int64_t expert_floor,
                   int64_t kv_min_tokens, int64_t experts_min_mib) {
    int64_t claim = need;
    if (kv_min_tokens > 0 && max_tokens > 0 && kv_ceiling > kv_fixed) {
        const int64_t t = std::min(kv_min_tokens, max_tokens);
        const int64_t kv = kv_fixed + (int64_t)((double)(kv_ceiling - kv_fixed) * (double)t /
                                                (double)max_tokens);
        claim += std::max<int64_t>(kv - kv_one, 0);
    }
    if (experts_min_mib > 0 && experts > expert_floor) {
        const int64_t e = std::min(experts_min_mib * ((int64_t)1 << 20), experts);
        claim += std::max<int64_t>(e - expert_floor, 0);
    }
    return std::max(need, std::min(claim, want));
}

}  // namespace rad::fleet
