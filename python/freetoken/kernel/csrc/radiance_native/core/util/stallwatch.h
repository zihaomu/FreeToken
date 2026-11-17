/* stallwatch.h -- naming where a phase that has gone quiet is waiting.
 *
 * A hang (a driver call that never returns, a lock inside a runtime library) leaves no trace in
 * the log, often on machines that cannot be reproduced locally (a guest kernel, a passed-through
 * card), so the process reports where it waits.
 *
 * A StallWatch is held across a phase by the thread doing it. When nothing has been logged for the
 * quiet interval it reports what that thread is doing:
 *   - its scheduler state and kernel wait channel;
 *   - the system call it is in, an ioctl with its device and request, and the library that made
 *     the call;
 *   - when the thread waits in user space (a lock, a sleep), its stack as module+offset frames,
 *     which addr2line symbolises against the released binary. A thread inside an ioctl is not
 *     signalled: the call already names where it waits, and a signal could end a driver wait
 *     early;
 *   - the process's other threads that are inside an ioctl, since a lock in a runtime library is
 *     usually held by one of them.
 * It reports again each time the silence doubles, showing whether the thread moved. A phase whose
 * progress is not logged (a loop of steps) marks it with progress(). The watch reads only its own
 * process's /proc entries and wakes every quarter of the quiet interval, at most once a second,
 * while held.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <sys/types.h>

namespace rad {

class StallWatch {
public:
    /* `phase` names what is being watched in the report ("startup"). */
    explicit StallWatch(const char* phase, uint64_t quiet_ms = 60000);
    ~StallWatch();
    StallWatch(const StallWatch&) = delete;
    StallWatch& operator=(const StallWatch&) = delete;

    /* Progress the log does not show: the silence restarts as if a line had been logged. */
    void progress();

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
    std::atomic<uint64_t> beat_{0};   /* rad_mono_ns of the last progress() */
};

/* What thread `tid` of this process is doing, as the report words it: state, wait channel, system
 * call. With `stack`, a thread waiting in user space is also asked for its frames. */
std::string stall_describe_thread(pid_t tid, bool stack);

/* One line of /proc/<pid>/task/<tid>/syscall in words: "running", "not in a system call", or the
 * call with its arguments decoded where they name something (an ioctl's device and request). */
std::string stall_decode_syscall(const std::string& line);

/* `addr` as the module that maps it and its offset from the module's load base, "libfoo.so+0x1a2b":
 * the number addr2line takes for that module. */
std::string stall_module_offset(uintptr_t addr);

}  /* namespace rad */
