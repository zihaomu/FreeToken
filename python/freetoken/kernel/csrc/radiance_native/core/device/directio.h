/* directio.h -- the file tier's reader (spec §5.4): O_DIRECT reads into pinned buffers, used by
 * the weight tier and the KV tier.
 *
 * The page cache is avoided for two reasons:
 *   - it is a second copy of every byte already headed for a pinned buffer;
 *   - it evicts. A large tier read through the page cache throws out the mapped `.rad` pool's
 *     pages, which shows up later as an untraceable decode stall.
 *
 * O_DIRECT requires the file offset, transfer length and buffer address to be block-aligned.
 * rad_dev_alloc memory is (host.cpp) and .rad pads movement units to RAD_ALIGN_UNIT
 * (rad_format.h), so every mover read takes the unbounced path. The bounce is for other callers.
 */
#pragma once
#include <atomic>
#include <cstdint>

namespace rad {

/* Bytes the drive has returned through this reader, monotonically. The stall
 * watchdog (aql.cpp) counts this alongside the queues' read indices: a host wait
 * that moves nothing anywhere -- no queue advanced, no read came back -- is the
 * only one that reports a stall. */
inline std::atomic<uint64_t> dio_progress{0};

enum {
    RAD_DIO_READ   = 1,
    RAD_DIO_WRITE  = 2,      /* implies read: a partial block is a read-modify-write */
    RAD_DIO_CREATE = 4,
    RAD_DIO_TRUNC  = 8,

    /* Open buffered when the filesystem refuses O_DIRECT, instead of failing. Off by default: a
     * silent buffered fallback makes the weight tier ~3x slower and is invisible in a profile.
     * For tests, which run on whatever /tmp is. */
    RAD_DIO_ALLOW_BUFFERED = 16,
};

class DirectFile {
public:
    DirectFile() = default;
    ~DirectFile();
    DirectFile(const DirectFile&)            = delete;
    DirectFile& operator=(const DirectFile&) = delete;

    /* RAD_OK, or a negative RAD_E_*. `flags` is the RAD_DIO_* set above. */
    int  open(const char* path, int flags);
    void close();

    bool    is_open() const { return fd_ >= 0; }
    /* Whether the open actually got O_DIRECT. False only under RAD_DIO_ALLOW_BUFFERED. */
    bool    direct() const  { return direct_; }
    int     fd() const      { return fd_; }
    int64_t size() const;

    /* Exact reads and writes: a short transfer is RAD_E_IO, never a partial success, so callers
     * need no byte-count check. */
    int  read_at(void* dst, int64_t off, int64_t bytes);
    int  write_at(const void* src, int64_t off, int64_t bytes);

    /* The alignment every fast-path extent must satisfy. A fixed 4096 rather than the device's
     * logical block size (often 512, see STATX_DIOALIGN): it covers every real requirement, equals
     * RAD_ALIGN_UNIT, and keeps behaviour independent of the filesystem. */
    static constexpr int64_t alignment() { return 4096; }

private:
    int  bounce_ready();

    int      fd_      = -1;
    bool     direct_  = false;
    uint8_t* bounce_  = nullptr;
    int64_t  bounce_bytes_ = 0;
};

}  /* namespace rad */
