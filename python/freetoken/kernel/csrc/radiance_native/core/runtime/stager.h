/* stager.h -- the seam a step-time weight stager plugs into.
 *
 * The mover changes the residency table (runtime/residency.h) between steps; a stager changes it
 * within a step. It copies an off-card weight into VRAM ahead of the op that reads it, publishes
 * the copy for that op, and republishes the host address once the op is behind it. The run phase
 * calls the stager around the ops of a pass it armed for and re-reads the residency table when
 * told it changed.
 *
 * A pass the stager arms for is issued live, never recorded or replayed from a tape: a tape
 * replays only packets, not the host work (copies, publishes) between them.
 */
#pragma once
#include "../rad_core.h"
#include "rad_device.h"

namespace rad {

class OpStager {
public:
    virtual ~OpStager() = default;

    /* The pass is about to be issued: whether the stager takes part in it. Called before the
     * run phase brings the weight tables up to date, so a stager turning down a pass may give back
     * weights it published ahead for it (see end_pass). */
    virtual bool begin_pass(const RadBatch* b) = 0;

    /* Before op `h` is issued on stream `s`. `*resync` is set when the residency table changed,
     * so the run phase brings the op's weight tables up to date before resolving them. */
    virtual int before_op(rad_op h, RadStream s, bool* resync) = 0;

    /* After op `h` was issued on stream `s`: the point a staged weight's last reader is behind. */
    virtual int after_op(rad_op, RadStream) { return RAD_OK; }

    /* The pass has been issued, on `s`: nothing the stager published for it may outlive it. It
     * may publish weights for a later pass it expects to arm for, which that pass's begin_pass
     * keeps or gives back. */
    virtual int end_pass(RadStream s) = 0;
};

/* Two stagers on one pass, each called only on the passes it armed for, so a stager whose
 * buffers are lent out on small passes never stages because the other one armed. The call order
 * is fixed, so copies and stream waits land in the same order on every pass. */
class StagerChain final : public OpStager {
public:
    void set(OpStager* a, OpStager* b) { s_[0] = a; s_[1] = b; }

    bool begin_pass(const RadBatch* b) override {
        bool any = false;
        for (int i = 0; i < 2; ++i) {
            on_[i] = s_[i] != nullptr && s_[i]->begin_pass(b);
            any = any || on_[i];
        }
        return any;
    }
    int before_op(rad_op h, RadStream s, bool* resync) override {
        *resync = false;
        for (int i = 0; i < 2; ++i) {
            if (!on_[i]) continue;
            bool r = false;
            const int rc = s_[i]->before_op(h, s, &r);
            if (rc < 0) return rc;
            *resync = *resync || r;
        }
        return RAD_OK;
    }
    int after_op(rad_op h, RadStream s) override {
        for (int i = 0; i < 2; ++i)
            if (on_[i]) { const int rc = s_[i]->after_op(h, s); if (rc < 0) return rc; }
        return RAD_OK;
    }
    int end_pass(RadStream s) override {
        int rc = RAD_OK;
        for (int i = 0; i < 2; ++i) {
            if (!on_[i]) continue;
            const int r = s_[i]->end_pass(s);
            if (r < 0 && rc >= 0) rc = r;
            on_[i] = false;
        }
        return rc;
    }

private:
    OpStager* s_[2] = { nullptr, nullptr };
    bool      on_[2] = { false, false };
};

}  /* namespace rad */
