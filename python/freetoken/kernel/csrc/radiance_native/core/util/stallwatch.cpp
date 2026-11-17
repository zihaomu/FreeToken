/* stallwatch.cpp -- see stallwatch.h. */
#include "util/stallwatch.h"

#include "rad_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <execinfo.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rad {

namespace {

std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}

/* KFD ioctl numbers (type 'K') from <linux/kfd_ioctl.h>, listed by number because the build's
 * header can be older than the running driver. */
const char* kfd_ioctl_name(unsigned nr) {
    static const char* const k[] = {
        nullptr, "GET_VERSION", "CREATE_QUEUE", "DESTROY_QUEUE", "SET_MEMORY_POLICY",
        "GET_CLOCK_COUNTERS", "GET_PROCESS_APERTURES", "UPDATE_QUEUE", "CREATE_EVENT",
        "DESTROY_EVENT", "SET_EVENT", "RESET_EVENT", "WAIT_EVENTS", "DBG_REGISTER",
        "DBG_UNREGISTER", "DBG_ADDRESS_WATCH", "DBG_WAVE_CONTROL", "SET_SCRATCH_BACKING_VA",
        "GET_TILE_CONFIG", "SET_TRAP_HANDLER", "GET_PROCESS_APERTURES_NEW", "ACQUIRE_VM",
        "ALLOC_MEMORY_OF_GPU", "FREE_MEMORY_OF_GPU", "MAP_MEMORY_TO_GPU", "UNMAP_MEMORY_FROM_GPU",
        "SET_CU_MASK", "GET_QUEUE_WAVE_STATE", "GET_DMABUF_INFO", "IMPORT_DMABUF",
        "ALLOC_QUEUE_GWS", "SMI_EVENTS", "SVM", "SET_XNACK_MODE", "CRIU_OP", "AVAILABLE_MEMORY",
        "EXPORT_DMABUF", "RUNTIME_ENABLE", "DBG_TRAP", "CREATE_PROCESS", "PROFILER",
    };
    return nr < sizeof k / sizeof k[0] ? k[nr] : nullptr;
}

/* The system calls a waiting thread is found in, by their x86-64 numbers. */
const char* syscall_name(long nr) {
#if defined(__x86_64__)
    switch (nr) {
        case 0: return "read";         case 1: return "write";       case 2: return "open";
        case 3: return "close";        case 4: return "stat";        case 7: return "poll";
        case 9: return "mmap";         case 10: return "mprotect";   case 11: return "munmap";
        case 16: return "ioctl";       case 17: return "pread64";    case 18: return "pwrite64";
        case 23: return "select";      case 28: return "madvise";    case 35: return "nanosleep";
        case 61: return "wait4";       case 149: return "mlock";     case 202: return "futex";
        case 230: return "clock_nanosleep";                          case 232: return "epoll_wait";
        case 237: return "mbind";      case 257: return "openat";    case 262: return "newfstatat";
        case 270: return "pselect6";   case 271: return "ppoll";     case 281: return "epoll_pwait";
        case 332: return "statx";      case 426: return "io_uring_enter"; case 437: return "openat2";
        default: return nullptr;
    }
#else
    (void)nr;
    return nullptr;
#endif
}

bool is_ioctl(long nr) { return nr == SYS_ioctl; }

/* ---- the stack of a thread that waits in user space, taken by a signal it handles itself ---- */

constexpr int kMaxFrames = 48;
std::atomic<pid_t> g_want_tid{0};
std::atomic<int>   g_have{0};
void*              g_frames[kMaxFrames];
int                g_n_frames = 0;

void on_stack_signal(int) {
    if ((pid_t)syscall(SYS_gettid) != g_want_tid.load(std::memory_order_acquire)) return;
    g_n_frames = backtrace(g_frames, kMaxFrames);
    g_have.store(1, std::memory_order_release);
}

/* A real-time signal nothing else in the process uses (glibc keeps the first few for itself). */
int stack_signal() { return SIGRTMIN + 7; }

std::once_flag g_install_once;
void install_stack_handler() {
    std::call_once(g_install_once, [] {
        /* backtrace() loads its unwinder on first use; done here, outside any handler. */
        void* prime[2];
        (void)backtrace(prime, 2);
        struct sigaction sa {};
        sa.sa_handler = on_stack_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(stack_signal(), &sa, nullptr);
    });
}

std::string stack_of(pid_t tid) {
    install_stack_handler();
    g_have.store(0, std::memory_order_relaxed);
    g_want_tid.store(tid, std::memory_order_release);
    if (syscall(SYS_tgkill, getpid(), tid, stack_signal()) != 0) {
        g_want_tid.store(0);
        return "the thread could not be signalled";
    }
    for (int i = 0; i < 200 && !g_have.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    g_want_tid.store(0, std::memory_order_release);
    if (!g_have.load(std::memory_order_acquire))
        return "no stack: the thread did not take the signal within 2 s";
    std::string s;
    /* Frame 0 and 1 are the handler and the signal trampoline. */
    for (int i = 2; i < g_n_frames; ++i)
        s += (s.empty() ? "" : " ") + stall_module_offset((uintptr_t)g_frames[i]);
    return s;
}

}  /* namespace */

std::string stall_module_offset(uintptr_t addr) {
    /* Relative to the module's load base (lowest mapping start minus its file offset), which is
     * what addr2line takes for PIC code. The file offset differs where the linker places a
     * segment at an address unlike its file position, as lld does. */
    struct Map { unsigned long lo, hi, off; std::string path; };
    std::vector<Map> maps;
    {
        std::ifstream f("/proc/self/maps");
        std::string line;
        while (std::getline(f, line)) {
            Map m{};
            char path[512] = "";
            if (std::sscanf(line.c_str(), "%lx-%lx %*s %lx %*s %*s %511s", &m.lo, &m.hi, &m.off,
                            path) < 3)
                continue;
            m.path = path;
            maps.push_back(std::move(m));
        }
    }
    for (const Map& m : maps) {
        if (addr < m.lo || addr >= m.hi) continue;
        if (m.path.empty() || m.path[0] != '/') return fmt("0x%lx", (unsigned long)addr);
        unsigned long base = m.lo - m.off;
        for (const Map& o : maps)
            if (o.path == m.path && o.lo - o.off < base) base = o.lo - o.off;
        const size_t slash = m.path.rfind('/');
        return fmt("%s+0x%lx", m.path.c_str() + slash + 1, (unsigned long)(addr - base));
    }
    return fmt("0x%lx", (unsigned long)addr);
}

std::string stall_decode_syscall(const std::string& line) {
    if (line.empty()) return "unknown";
    if (line == "running") return "running";
    std::istringstream in(line);
    long nr = 0;
    if (!(in >> nr)) return "unknown";
    if (nr < 0) return "not in a system call";
    std::vector<unsigned long> a;
    std::string w;
    while (in >> w) a.push_back(std::strtoul(w.c_str(), nullptr, 16));
    /* Six arguments, then the stack pointer and the program counter. */
    const unsigned long pc = a.size() >= 8 ? a[7] : 0;
    std::string s;
    if (is_ioctl(nr) && a.size() >= 2) {
        const long fd = (long)a[0];
        const unsigned long req = a[1];
        char target[256] = "";
        const ssize_t n = readlink(fmt("/proc/self/fd/%ld", fd).c_str(), target, sizeof target - 1);
        if (n > 0) target[n] = 0; else std::snprintf(target, sizeof target, "fd %ld", fd);
        const unsigned type = (unsigned)((req >> 8) & 0xff), num = (unsigned)(req & 0xff);
        const char* kfd = type == 'K' ? kfd_ioctl_name(num) : nullptr;
        s = kfd ? fmt("ioctl(%s, AMDKFD_IOC_%s)", target, kfd)
                : fmt("ioctl(%s, request 0x%lx: type '%c' number 0x%x)", target, req,
                      type >= 0x20 && type < 0x7f ? (char)type : '?', num);
    } else if (const char* name = syscall_name(nr)) {
        s = name;
    } else {
        s = fmt("system call %ld", nr);
    }
    if (pc) s += ", called from " + stall_module_offset(pc);
    return s;
}

std::string stall_describe_thread(pid_t tid, bool stack) {
    const std::string dir = fmt("/proc/self/task/%d/", (int)tid);
    const std::string stat = read_first_line(dir + "stat");
    /* The state is the field after the parenthesised name, which may itself hold spaces. */
    char state = '?';
    const size_t rp = stat.rfind(')');
    if (rp != std::string::npos && rp + 2 < stat.size()) state = stat[rp + 2];
    const std::string wchan = read_first_line(dir + "wchan");
    const std::string sc = read_first_line(dir + "syscall");
    std::string s = fmt("tid %d (%s), state %c", (int)tid, read_first_line(dir + "comm").c_str(),
                        state);
    if (!wchan.empty() && wchan != "0") s += ", waiting in the kernel at " + wchan;
    s += ", " + stall_decode_syscall(sc);
    long nr = -1;
    std::sscanf(sc.c_str(), "%ld", &nr);
    /* A thread in a driver call is not signalled: the call already says where it waits, and the
     * signal could end the wait early instead of restarting it. */
    if (stack && !is_ioctl(nr) && state != 'D') s += "\n    stack: " + stack_of(tid);
    return s;
}

struct StallWatch::Impl {
    std::string             phase;
    uint64_t                quiet_ns = 0;
    uint64_t                poll_ns = 0;
    pid_t                   holder = 0;
    std::mutex              mu;
    std::condition_variable cv;
    bool                    stop = false;
    std::thread             th;

    void report(uint64_t silent_ns) {
        const pid_t me = (pid_t)syscall(SYS_gettid);
        std::string others;
        int n_others = 0;
        if (DIR* d = opendir("/proc/self/task")) {
            while (dirent* e = readdir(d)) {
                const pid_t t = (pid_t)std::atoi(e->d_name);
                if (t <= 0 || t == holder || t == me) continue;
                long nr = -1;
                std::sscanf(read_first_line(fmt("/proc/self/task/%d/syscall", (int)t)).c_str(),
                            "%ld", &nr);
                if (!is_ioctl(nr)) continue;
                if (n_others++ < 8) others += "\n  also in a driver call: " +
                                              stall_describe_thread(t, false);
            }
            closedir(d);
        }
        if (n_others > 8) others += fmt("\n  and %d more threads in a driver call", n_others - 8);
        RAD_WARN("%s: nothing has been logged for %.0f s since the line above. The thread doing "
                 "the %s is %s%s\n  A report of a %s that stops here should include these lines.",
                 phase.c_str(), (double)silent_ns / 1e9, phase.c_str(),
                 stall_describe_thread(holder, true).c_str(), others.c_str(), phase.c_str());
    }

    void run(const std::atomic<uint64_t>& beat) {
        uint64_t since = rad_mono_ns();       /* the start of the current silence */
        uint64_t own = 0;                     /* the time of this watch's own last report */
        uint64_t next = quiet_ns;             /* the silence that is reported next */
        std::unique_lock<std::mutex> lk(mu);
        while (!cv.wait_for(lk, std::chrono::nanoseconds(poll_ns), [&] { return stop; })) {
            const uint64_t last = std::max(log_last_line_ns(),
                                           beat.load(std::memory_order_relaxed));
            if (last > since && last != own) { since = last; next = quiet_ns; }
            const uint64_t now = rad_mono_ns();
            if (now - since < next) continue;
            lk.unlock();
            report(now - since);
            lk.lock();
            own = log_last_line_ns();
            next *= 2;
        }
    }
};

StallWatch::StallWatch(const char* phase, uint64_t quiet_ms) : p_(std::make_unique<Impl>()) {
    p_->phase = phase ? phase : "phase";
    p_->quiet_ns = quiet_ms * 1000000ull;
    /* A quarter of the quiet interval, so a report is at most that late, and at most a second. */
    p_->poll_ns = std::max<uint64_t>(1000000ull, std::min<uint64_t>(1000000000ull,
                                                                    p_->quiet_ns / 4));
    p_->holder = (pid_t)syscall(SYS_gettid);
    install_stack_handler();
    p_->th = std::thread([this] { p_->run(beat_); });
}

void StallWatch::progress() { beat_.store(rad_mono_ns(), std::memory_order_relaxed); }

StallWatch::~StallWatch() {
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->stop = true;
    }
    p_->cv.notify_all();
    if (p_->th.joinable()) p_->th.join();
}

}  /* namespace rad */
