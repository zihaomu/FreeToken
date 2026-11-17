/* rowpool.cpp -- see rowpool.h. */
#include "util/rowpool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <pthread.h>
#include <system_error>
#include <thread>

namespace rad {

namespace {

struct Loop {
    const std::function<void(int64_t)>* row = nullptr;
    int64_t rows = 0;
    std::atomic<int64_t> next{0};
    int max_help = 0;
    int help = 0;               /* workers inside the loop; under Pool::mu */
};

class Pool {
public:
    Pool() {
        const unsigned hw = std::thread::hardware_concurrency();
        cap_.store((int)std::min(15u, hw > 1 ? hw - 1 : 0u), std::memory_order_relaxed);
    }
    int cap() const { return cap_.load(std::memory_order_relaxed); }

    void run(int64_t rows, int width, const std::function<void(int64_t)>& row) {
        Loop L;
        L.row = &row;
        L.rows = rows;
        {
            std::lock_guard<std::mutex> lk(mu_);
            start(std::min(width - 1, cap()));
            L.max_help = std::min(width - 1, started_);
            if (L.max_help > 0) q_.push_back(&L);
        }
        for (int k = 0; k < L.max_help; ++k) work_cv_.notify_one();
        for (int64_t i; (i = L.next.fetch_add(1, std::memory_order_relaxed)) < rows;) row(i);
        if (L.max_help == 0) return;
        /* Every row is taken. Once the loop is off the queue no worker can join it, and each that
         * did leaves after finishing the rows it took. */
        std::unique_lock<std::mutex> lk(mu_);
        q_.erase(std::find(q_.begin(), q_.end(), &L));
        done_cv_.wait(lk, [&] { return L.help == 0; });
    }

private:
    /* Starts workers until `want` run, under mu_. A thread that cannot be started leaves the pool
     * smaller; the caller's own rows still cover the loop. */
    void start(int want) {
        while (started_ < want) {
            try {
                std::thread([this] { work(); }).detach();
            } catch (const std::system_error&) {
                cap_.store(started_, std::memory_order_relaxed);
                return;
            }
            ++started_;
        }
    }

    Loop* pick() {
        for (Loop* L : q_)
            if (L->help < L->max_help && L->next.load(std::memory_order_relaxed) < L->rows) return L;
        return nullptr;
    }

    void work() {
        pthread_setname_np(pthread_self(), "rad-rows");
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            Loop* L = nullptr;
            work_cv_.wait(lk, [&] { return (L = pick()) != nullptr; });
            ++L->help;
            lk.unlock();
            for (int64_t i; (i = L->next.fetch_add(1, std::memory_order_relaxed)) < L->rows;)
                (*L->row)(i);
            lk.lock();
            if (--L->help == 0) done_cv_.notify_all();
        }
    }

    std::mutex mu_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<Loop*> q_;
    std::atomic<int> cap_{0};
    int started_ = 0;           /* under mu_ */
};

/* Never destroyed: workers wait on it until the process ends, and a static destructor would free
 * the mutex under them. */
Pool& pool() {
    static Pool* p = new Pool;
    return *p;
}

}  // namespace

void row_pool_run(int64_t rows, int width, const std::function<void(int64_t)>& row,
                  int64_t min_rows) {
    if (rows <= 0) return;
    if (width <= 1 || rows < std::max<int64_t>(min_rows, 2) || pool().cap() == 0) {
        for (int64_t i = 0; i < rows; ++i) row(i);
        return;
    }
    pool().run(rows, (int)std::min<int64_t>(width, rows), row);
}

int row_pool_workers() { return pool().cap(); }

}  // namespace rad
