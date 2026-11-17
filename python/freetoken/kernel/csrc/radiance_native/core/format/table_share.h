/* table_share.h -- an n-gram table as a file in a tmpfs that engines on separate cards share
 * (--ngram-placement ram=DIR). Each engine maps the file MAP_SHARED and pins it for its own cards,
 * so the host holds one copy however many engines read it. This is the file protocol only; the
 * mapping and the pinning are the device layer's (rad_dev_map_file).
 *
 * The file is `DIR/ngram-<id>.tab`: a header in its first 2 MiB, the table from kTableShareHeader
 * on, so the table keeps the 2 MiB alignment the pinned ranges want. The id names the container
 * and the table's place in it, so a file never serves another model.
 *
 * Every holder keeps LOCK_SH on the data file for its life, which counts the readers. The first
 * engine to find the file unready takes the fill lock (LOCK_EX on `<file>.lock`, a file of its
 * own so that a later engine arriving while the table is read never waits behind the readers),
 * fills the table and marks it ready; an engine arriving meanwhile waits on the fill lock and then
 * finds it ready. The holder that gets LOCK_EX on the data file when it leaves is the last, and
 * deletes the file and the lock file; one that leaves while another still holds LOCK_SH waits a
 * moment for it, since engines stopped together leave together. A holder that died took its locks
 * with it: a half-filled file is filled again, a ready one is reused after the same sample check
 * as a fresh copy. */
#pragma once
#include <cstdint>
#include <string>

namespace rad {

/* The table's offset in the file. */
constexpr int64_t kTableShareHeader = (int64_t)2 << 20;

/* The file's length for an n-byte table: the header, then the table rounded up to 2 MiB. */
int64_t table_share_file_bytes(int64_t n);

/* The smallest table worth a shared file. A file costs the header and the table's rounding to
 * 2 MiB, up to 4 MiB beside the table, and a tmpfs entry that outlives a crash; a weight of a few
 * bytes the gather reads beside the table (its one-element scale) would cost more that way than as
 * a private copy in every engine. From 16 headers on the file's overhead is under an eighth of the
 * table; a smaller one takes the private pinned copy. */
constexpr int64_t kTableShareMin = 16 * kTableShareHeader;

struct TableShare {
    int         fd = -1;         /* the data file, LOCK_SH held until table_share_close */
    int         lock_fd = -1;    /* the fill lock, held while `filler` and not yet marked ready */
    std::string path, lock_path;
    bool        ready = false;   /* the table in the file is complete */
    bool        filler = false;  /* this holder fills it, then calls table_share_mark_ready */
};

/* FNV-1a 64 over the container's real path, size and mtime (to the nanosecond), the table's file
 * offset and its length, as 16 hex digits; empty if the container cannot be stat'ed. */
std::string table_share_id(const std::string& container_path, int64_t foff, int64_t n);

/* Open or create DIR/ngram-<id>.tab for an n-byte table and take part in the protocol: afterwards
 * `out->ready` (map it and check samples) or `out->filler` (map it, fill [kTableShareHeader, +n),
 * mark it ready). RAD_E_INVAL if a file of that name does not hold a table of this id and length
 * (another program's, or another version's; never overwritten, since a live engine may map it),
 * RAD_E_IO if the directory or file cannot be used. `why` carries the reason. */
int  table_share_open(const std::string& dir, const std::string& id, int64_t n, TableShare* out,
                      std::string* why);

/* The filler's table is complete: write the header, release the fill lock. */
int  table_share_mark_ready(TableShare* s, int64_t n, const std::string& id, std::string* why);

/* Leave. The last holder deletes the file and its lock file; while another still holds the file,
 * this waits up to about a second for it to leave first. Safe on a TableShare never opened. */
void table_share_close(TableShare* s);

/* Whether DIR/ngram-<id>.tab is a complete table of n bytes now, read from its header without
 * taking part: for the placement decision, which must know before anything is pinned. */
bool table_share_is_ready(const std::string& dir, const std::string& id, int64_t n);

}  /* namespace rad */
