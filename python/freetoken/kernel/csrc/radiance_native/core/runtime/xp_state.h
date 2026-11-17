/* xp_state.h -- the expert prefetch's host state, held by the Ctx (Ctx::xp_). See xprefetch.h. */
#pragma once
#include "xprefetch.h"
#include "rad_runtime.h"
#include "../device/device.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rad {

/* A routed layer as the prefetch knows it: the next-layer description a plugin hands over, and the
 * device map from expert to its table entries, built at the first call after the layer's tables are
 * bound. `want` is the request made for it this pass, which its step patches in. */
struct XpLayer {
    RadPrefetchNext next{};
    void**   map   = nullptr;    /* device [kXpMaxNe][kXpParts]: entry addresses, null for none */
    int32_t* place = nullptr;    /* device [kXpMaxNe]: the expert's place in the routing counts */
    bool     bound = false;
    bool     want  = false;
    int      g     = 0;
    uint32_t seq   = 0;
};

/* A request: the group's mailbox the plan writes, and the mark the DMA list waits for. */
struct XpReq {
    int       g    = 0;
    uint32_t  seq  = 0;
    void*     mark = nullptr;
    RadStream ms   = nullptr;     /* the stream the mark rides */
    bool      valid = false;
};

struct XpState {
    int          device = 0;
    RadStream    stream = nullptr;     /* logits and plan, behind rides on the compute stream */
    RadEvent     tail   = nullptr;     /* the pass start waits for the stream's work of the last */
    RadEvent     sync   = nullptr;     /* the router input's readiness, where a ride is refused */
    bool         tail_live = false;
    char*        ring   = nullptr;     /* uncached: kXpGroups x kXpSlots slots of `slot` bytes */
    uint64_t     slot   = 0;
    XpPools      pools{};
    XpRec*       rec    = nullptr;     /* [kXpGroups][kXpSlots] */
    int32_t*     nrec   = nullptr;     /* [kXpGroups] */
    XpBox*       box    = nullptr;     /* [kXpGroups], pinned host */
    uint32_t*    done   = nullptr;     /* [kXpGroups], pinned host: the last request whose copies landed */
    XpStats*     stats  = nullptr;     /* device */
    XpStats*     stats_h = nullptr;    /* pinned host, read at the end */
    float*       logits = nullptr;     /* [kXpGroups][kXpMaxRows][kXpMaxNe] */
    void**       maps   = nullptr;     /* [layers][kXpMaxNe][kXpParts] */
    int32_t*     places = nullptr;     /* [layers][kXpMaxNe] */
    std::vector<XpLayer> layers;
    uint32_t     seq[kXpGroups] = {};
    int          last_layer = -1;      /* the last layer predicted for: a lower one starts a pass */
    /* Payoff: the share of copies the steps use, judged every kXpWindow predictions from the
     * counters (stats_h lags one window); below kXpMinUse the prefetch rests. */
    int64_t      passes = 0, rest_until = 0, rests = 0;
    int64_t      calls = 0;
    uint32_t     snap_copied = 0, snap_routed = 0;
    bool         have_snap = false, judge = false;
    int          restore_g  = -1;      /* the group the next step puts back */
    XpReq        pend;                 /* made at a router, waiting for its mark */

    /* The host thread: each mailbox, once the plan has written it, to the DMA queue. */
    std::thread              thr;
    std::mutex               mu;
    std::condition_variable  cv;
    std::deque<XpReq>        q;
    std::atomic<bool>        stop{false};
    std::deque<std::pair<void*, int64_t>> marks;   /* thread only: a mark and the list that polls it */
    int64_t                  lists = 0;            /* thread only: lists handed over */
    std::atomic<int64_t>     dropped{0};           /* requests the queue was too far behind for */
};

}  /* namespace rad */
