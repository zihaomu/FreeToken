/* pass.h -- what the engine's pass loop needs of a generator's driver.
 *
 * A generator program is driven by passes rather than token steps (abi/rad_builder.h,
 * rad_declare_generator and rad_declare_audio_generator). Each kind has its own driver, which turns
 * its requests into passes (core/gen/driver.h for images, core/gen/audio_driver.h for audio); the
 * loop that runs them on the ranks is one (Engine::gen_loop). The loop asks for the next pass,
 * runs it on every rank, copies back what the driver names and hands the pass back.
 */
#pragma once
#include "rad_core.h"

#include <vector>

namespace rad {
namespace gen {

struct IPassDriver {
    /* A device buffer the staged pass leaves for the host: `bytes` from the start of `buf` on
     * rank 0, copied to `host` behind the pass. Every rank holds the same values. */
    struct Readback {
        rad_buf buf   = RAD_NULL_HANDLE;
        int64_t bytes = 0;
        void*   host  = nullptr;
    };

    virtual ~IPassDriver() = default;

    /* The next pass, staged on every rank's stream, or null when there is nothing to run.
     * `step` is the engine's pass counter, carried into RadBatch::step. */
    virtual const RadBatch* next(int step) = 0;
    virtual const RadBatch* batch_for(int rank) const = 0;
    /* The runtime LoRA slot every row of the staged pass runs with, -1 for none. */
    virtual int32_t lora_slot() const { return -1; }
    /* What the staged pass's results are, valid until done(). */
    virtual const std::vector<Readback>& readbacks() const = 0;
    /* The pass has run on every rank, its streams are drained and the readbacks have landed. */
    virtual int  done() = 0;
    /* Fail every request queued or in flight, and refuse new ones. */
    virtual void shutdown(const char* why, int status) = 0;

    /* For /health, /stats and the dashboard. */
    virtual int64_t queued() const = 0;
    virtual int64_t running() const = 0;
};

}  /* namespace gen */
}  /* namespace rad */
