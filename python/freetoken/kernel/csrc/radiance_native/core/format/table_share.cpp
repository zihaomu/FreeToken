/* table_share.cpp -- see table_share.h. */
#include "table_share.h"

#include "../rad_internal.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>

namespace rad {
namespace {

constexpr char     kMagic[8] = { 'R', 'A', 'D', 'N', 'G', 'R', 'M', '1' };
constexpr uint32_t kVersion = 1;
constexpr int64_t  kHuge = (int64_t)2 << 20;
/* A leaving holder tries for LOCK_EX for about a second (table_share_close). */
constexpr int      kLeaveTries = 10;
constexpr unsigned kLeaveWaitUs = 100 * 1000;

/* All integers host-endian: the file lives in a tmpfs, so it never leaves the machine. */
struct Header {
    char     magic[8];
    uint32_t version;
    uint32_t ready;
    uint64_t bytes;
    uint64_t filler_pid;   /* for whoever reads the file by hand; nothing depends on it */
    char     id[24];       /* the 16 hex digits and a NUL */
};
static_assert(sizeof(Header) <= 4096, "the header must fit the first page");

enum class Seen { Fresh, Partial, Ready, Foreign };

std::string errno_text() { return std::strerror(errno); }

/* What the open data file holds. A fresh file and a half-filled one have a zero header; a header
 * that is not this protocol's or not for this table is Foreign. */
Seen classify(int fd, const std::string& id, int64_t n, std::string* why) {
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        *why = fmt("fstat: %s", errno_text().c_str());
        return Seen::Foreign;
    }
    if (st.st_size == 0) return Seen::Fresh;
    Header h{};
    const ssize_t got = ::pread(fd, &h, sizeof h, 0);
    if (got != (ssize_t)sizeof h) {
        *why = fmt("its header is %lld bytes short", (long long)(sizeof h - (got < 0 ? 0 : got)));
        return Seen::Foreign;
    }
    Header zero{};
    if (std::memcmp(&h, &zero, sizeof h) == 0) return Seen::Partial;
    if (std::memcmp(h.magic, kMagic, sizeof kMagic) != 0 || h.version != kVersion) {
        *why = "it is not a table file of this version";
        return Seen::Foreign;
    }
    if (h.bytes != (uint64_t)n || std::strncmp(h.id, id.c_str(), sizeof h.id) != 0) {
        *why = fmt("it holds %s of table '%.16s', not %s of '%s'", humanb((int64_t)h.bytes).c_str(),
                   h.id, humanb(n).c_str(), id.c_str());
        return Seen::Foreign;
    }
    if (!h.ready) return Seen::Partial;
    if (st.st_size != table_share_file_bytes(n)) {
        *why = fmt("it is marked ready but is %lld bytes long, not %lld", (long long)st.st_size,
                   (long long)table_share_file_bytes(n));
        return Seen::Foreign;
    }
    return Seen::Ready;
}

std::string file_path(const std::string& dir, const std::string& id) {
    const bool slash = !dir.empty() && dir.back() == '/';
    return dir + (slash ? "" : "/") + "ngram-" + id + ".tab";
}

void fail(TableShare* s) {
    if (s->lock_fd >= 0) ::close(s->lock_fd);
    if (s->fd >= 0) ::close(s->fd);
    *s = TableShare{};
}

}  /* namespace */

int64_t table_share_file_bytes(int64_t n) {
    return kTableShareHeader + ((n + kHuge - 1) & ~(kHuge - 1));
}

std::string table_share_id(const std::string& container_path, int64_t foff, int64_t n) {
    char real[PATH_MAX];
    const std::string path = ::realpath(container_path.c_str(), real) ? std::string(real)
                                                                     : container_path;
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) return std::string();
    uint64_t h = 1469598103934665603ull;
    auto feed = [&](const void* p, size_t len) {
        const unsigned char* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < len; ++i) h = (h ^ b[i]) * 1099511628211ull;
    };
    const int64_t parts[] = { (int64_t)st.st_size, (int64_t)st.st_mtim.tv_sec,
                              (int64_t)st.st_mtim.tv_nsec, foff, n };
    feed(path.data(), path.size());
    feed(parts, sizeof parts);
    return fmt("%016llx", (unsigned long long)h);
}

int table_share_open(const std::string& dir, const std::string& id, int64_t n, TableShare* out,
                     std::string* why) {
    if (!out || !why || dir.empty() || id.empty() || n <= 0) return RAD_E_INVAL;
    *out = TableShare{};
    out->path = file_path(dir, id);
    out->lock_path = out->path + ".lock";

    /* A holder that left last deletes the file while we open it: its unlink comes with LOCK_EX, so
     * once we hold LOCK_SH the link count says whether the file is still the one in the directory. */
    for (;;) {
        out->fd = ::open(out->path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (out->fd < 0) {
            *why = fmt("opening %s: %s", out->path.c_str(), errno_text().c_str());
            *out = TableShare{};
            return RAD_E_IO;
        }
        if (::flock(out->fd, LOCK_SH) != 0) {
            *why = fmt("locking %s: %s", out->path.c_str(), errno_text().c_str());
            fail(out);
            return RAD_E_IO;
        }
        struct stat st{};
        if (::fstat(out->fd, &st) != 0) {
            *why = fmt("fstat %s: %s", out->path.c_str(), errno_text().c_str());
            fail(out);
            return RAD_E_IO;
        }
        if (st.st_nlink > 0) {
            struct statfs fs{};
            if (::fstatfs(out->fd, &fs) == 0 && (long)fs.f_type != 0x01021994L &&
                (long)fs.f_type != 0x858458f6L)
                RAD_WARN("n-gram table share: %s is not on a tmpfs or ramfs, so the table is "
                         "written to its disk and held in the page cache", dir.c_str());
            break;
        }
        ::close(out->fd);
    }

    std::string seen_why;
    Seen seen = classify(out->fd, id, n, &seen_why);
    if (seen == Seen::Foreign) {
        *why = fmt("%s exists, and %s", out->path.c_str(), seen_why.c_str());
        fail(out);
        return RAD_E_INVAL;
    }
    if (seen == Seen::Ready) {
        out->ready = true;
        return RAD_OK;
    }

    out->lock_fd = ::open(out->lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (out->lock_fd < 0) {
        *why = fmt("opening %s: %s", out->lock_path.c_str(), errno_text().c_str());
        fail(out);
        return RAD_E_IO;
    }
    if (::flock(out->lock_fd, LOCK_EX) != 0) {
        *why = fmt("locking %s: %s", out->lock_path.c_str(), errno_text().c_str());
        fail(out);
        return RAD_E_IO;
    }
    /* The holder we waited for may have finished. */
    seen = classify(out->fd, id, n, &seen_why);
    if (seen == Seen::Foreign) {
        *why = fmt("%s exists, and %s", out->path.c_str(), seen_why.c_str());
        fail(out);
        return RAD_E_INVAL;
    }
    if (seen == Seen::Ready) {
        ::close(out->lock_fd);
        out->lock_fd = -1;
        out->ready = true;
        return RAD_OK;
    }
    if (::ftruncate(out->fd, table_share_file_bytes(n)) != 0) {
        *why = fmt("sizing %s to %s: %s", out->path.c_str(),
                   humanb(table_share_file_bytes(n)).c_str(), errno_text().c_str());
        fail(out);
        return RAD_E_NOMEM;
    }
    out->filler = true;
    return RAD_OK;
}

int table_share_mark_ready(TableShare* s, int64_t n, const std::string& id, std::string* why) {
    if (!s || !why || s->fd < 0 || !s->filler || id.size() >= sizeof(Header::id))
        return RAD_E_INVAL;
    Header h{};
    std::memcpy(h.magic, kMagic, sizeof kMagic);
    h.version = kVersion;
    h.ready = 1;
    h.bytes = (uint64_t)n;
    h.filler_pid = (uint64_t)::getpid();
    std::memcpy(h.id, id.c_str(), id.size());
    if (::pwrite(s->fd, &h, sizeof h, 0) != (ssize_t)sizeof h) {
        *why = fmt("writing the header of %s: %s", s->path.c_str(), errno_text().c_str());
        return RAD_E_IO;
    }
    if (s->lock_fd >= 0) {
        ::close(s->lock_fd);
        s->lock_fd = -1;
    }
    s->ready = true;
    return RAD_OK;
}

void table_share_close(TableShare* s) {
    if (!s) return;
    if (s->lock_fd >= 0) ::close(s->lock_fd);
    if (s->fd >= 0) {
        /* Engines stopped together (a compose recreate) leave together: one that tries while the
         * other still holds LOCK_SH, unpinning the table, is not the last, and the other may be
         * killed at the stop timeout before its own try, which leaves the table in the tmpfs. So
         * this holder lets go of LOCK_SH and tries for a while. It holds no lock from here on
         * until LOCK_EX is granted; a conversion would not keep it either, since a refused one
         * has already dropped LOCK_SH (flock(2): not atomic). LOCK_EX may then come on a file
         * another holder already deleted, with a new engine's file at its name, so only the file
         * the name still points to is deleted. The lock file goes first: nobody gets past
         * LOCK_SH on this file to use it, while a file made anew at the name the moment this one
         * is gone could already have a filler on it. */
        (void)::flock(s->fd, LOCK_UN);
        for (int i = 0; i < kLeaveTries; ++i) {
            if (::flock(s->fd, LOCK_EX | LOCK_NB) == 0) {
                struct stat mine{}, named{};
                if (::fstat(s->fd, &mine) == 0 && ::stat(s->path.c_str(), &named) == 0 &&
                    mine.st_dev == named.st_dev && mine.st_ino == named.st_ino) {
                    ::unlink(s->lock_path.c_str());
                    ::unlink(s->path.c_str());
                }
                break;
            }
            if (errno != EWOULDBLOCK && errno != EINTR) break;
            if (i + 1 < kLeaveTries) ::usleep(kLeaveWaitUs);
        }
        ::close(s->fd);
    }
    *s = TableShare{};
}

bool table_share_is_ready(const std::string& dir, const std::string& id, int64_t n) {
    if (dir.empty() || id.empty() || n <= 0) return false;
    const int fd = ::open(file_path(dir, id).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    std::string why;
    const bool ready = classify(fd, id, n, &why) == Seen::Ready;
    ::close(fd);
    return ready;
}

}  /* namespace rad */
