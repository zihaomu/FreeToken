/* torch_zip.cpp -- the zip directory and the weights-only pickle machine (torch_zip.h). */
#include "torch_zip.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace rad {
namespace {

/* ================================================================== the zip directory */
struct Entry {
    const uint8_t* data = nullptr;
    uint64_t       bytes = 0;
};

uint16_t u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
uint64_t u64(const uint8_t* p) { return (uint64_t)u32(p) | ((uint64_t)u32(p + 4) << 32); }

/* Every stored entry, by name. torch.save writes ZIP64 records past 4 GiB and pads each local
 * header's extra field so a storage starts 64-byte aligned, which is why the data offset is read
 * from the local header rather than assumed from the central one. */
int zip_entries(const uint8_t* b, uint64_t n, std::unordered_map<std::string, Entry>* out,
                std::string* why) {
    auto no = [&](const std::string& w) {
        *why = w;
        return RAD_E_FORMAT;
    };
    if (n < 22) return no("too short to be a zip archive");
    /* The end-of-central-directory record: the last signature within a comment's reach. */
    int64_t eocd = -1;
    const uint64_t lo = n > 22 + 65535 ? n - 22 - 65535 : 0;
    for (uint64_t at = n - 22 + 1; at-- > lo;)
        if (u32(b + at) == 0x06054b50) {
            eocd = (int64_t)at;
            break;
        }
    if (eocd < 0) return no("no zip end-of-central-directory record");
    if (u16(b + eocd + 4) != 0 || u16(b + eocd + 6) != 0) return no("a multi-disk zip archive");
    uint64_t count = u16(b + eocd + 10), cd_size = u32(b + eocd + 12), cd_off = u32(b + eocd + 16);
    if (count == 0xffff || cd_size == 0xffffffffu || cd_off == 0xffffffffu) {
        if (eocd < 20 || u32(b + eocd - 20) != 0x07064b50)
            return no("a ZIP64 archive without its end-record locator");
        const uint64_t z = u64(b + eocd - 20 + 8);
        if (n < 56 || z > n - 56 || u32(b + z) != 0x06064b50)
            return no("a ZIP64 end record outside the file");
        count = u64(b + z + 32);
        cd_size = u64(b + z + 40);
        cd_off = u64(b + z + 48);
    }
    if (cd_off > n || cd_size > n - cd_off) return no("the central directory lies outside the file");
    if (count > ((uint64_t)1 << 20)) return no("more than a million zip entries");
    uint64_t at = cd_off;
    const uint64_t end = cd_off + cd_size;
    for (uint64_t i = 0; i < count; ++i) {
        if (end - at < 46 || at > end || u32(b + at) != 0x02014b50)
            return no("a malformed central directory entry");
        const uint16_t flags = u16(b + at + 8), method = u16(b + at + 10);
        uint64_t csize = u32(b + at + 20), usize = u32(b + at + 24);
        const uint64_t nl = u16(b + at + 28), el = u16(b + at + 30), cl = u16(b + at + 32);
        uint64_t loff = u32(b + at + 42);
        if (46 + nl + el + cl > end - at) return no("a central directory entry runs past its end");
        const std::string name((const char*)b + at + 46, (size_t)nl);
        /* The ZIP64 extra field: an 8-byte value for each 32-bit field at its maximum, in order. */
        const uint8_t* ex = b + at + 46 + nl;
        for (uint64_t e = 0; e + 4 <= el;) {
            const uint64_t id = u16(ex + e), sz = u16(ex + e + 2);
            if (e + 4 + sz > el) return no("a malformed extra field in '" + name + "'");
            if (id == 1) {
                uint64_t f = e + 4;
                const uint64_t fe = e + 4 + sz;
                auto take = [&](uint64_t* v) {
                    if (*v != 0xffffffffu) return true;
                    if (f + 8 > fe) return false;
                    *v = u64(ex + f);
                    f += 8;
                    return true;
                };
                if (!take(&usize) || !take(&csize) || !take(&loff))
                    return no("a short ZIP64 extra field in '" + name + "'");
            }
            e += 4 + sz;
        }
        if (flags & 1) return no("'" + name + "' is encrypted");
        if (method != 0)
            return no("'" + name + "' is compressed (method " + std::to_string(method) +
                      "); torch.save stores its entries, and a compressed storage cannot be mapped");
        if (csize != usize) return no("'" + name + "' is stored with two different sizes");
        if (n < 30 || loff > n - 30 || u32(b + loff) != 0x04034b50)
            return no("'" + name + "' has no local header where the directory places it");
        const uint64_t data = loff + 30 + u16(b + loff + 26) + u16(b + loff + 28);
        if (data > n || usize > n - data) return no("'" + name + "' runs past the end of the file");
        if (!out->emplace(name, Entry{ b + data, usize }).second)
            return no("'" + name + "' is in the archive twice");
        at += 46 + nl + el + cl;
    }
    return RAD_OK;
}

/* ================================================================== the pickle machine */
/* A value of the weights-only subset. Containers hold indices into the machine's pool, so a value
 * reached twice through the memo is one value. */
struct V {
    enum K : uint8_t { None, Bool, Int, Float, Str, Bytes, Tuple, List, Dict, Global, Storage,
                       Tensor } k = None;
    int64_t              i = 0;          /* Int, Bool; Storage: numel */
    double               f = 0;
    std::string          s;              /* Str, Bytes, Global's "module.name", Storage's key */
    std::vector<size_t>  a;              /* Tuple, List; Dict: keys and values interleaved */
    uint32_t             dtype = 0;      /* Storage */
    std::string          cls;            /* Storage: its class */
    size_t               storage = 0;    /* Tensor */
    int64_t              offset = 0;     /* Tensor, in elements */
    std::vector<int64_t> shape, stride;  /* Tensor */
};

/* The storages torch writes a state dict's tensors into, and the dtype each holds. Doubles and
 * complex storages have no RAD dtype and are refused by name where they appear. */
uint32_t storage_dtype(const std::string& g) {
    static const struct { const char* name; uint32_t dt; } kS[] = {
        { "torch.FloatStorage", RAD_F32 },  { "torch.HalfStorage", RAD_F16 },
        { "torch.BFloat16Storage", RAD_BF16 }, { "torch.LongStorage", RAD_I64 },
        { "torch.IntStorage", RAD_I32 },    { "torch.ShortStorage", RAD_I16 },
        { "torch.CharStorage", RAD_I8 },    { "torch.ByteStorage", RAD_U8 },
        { "torch.BoolStorage", RAD_BOOL },
    };
    for (const auto& s : kS)
        if (g == s.name) return s.dt;
    return RAD_DT_INVALID;
}

/* torch's weights-only allow-list, the part a state dict uses. */
bool allowed_global(const std::string& g) {
    return g == "collections.OrderedDict" || g == "torch._utils._rebuild_tensor_v2" ||
           g == "torch._utils._rebuild_parameter" || storage_dtype(g) != RAD_DT_INVALID;
}

class Machine {
public:
    Machine(const uint8_t* p, uint64_t n) : p_(p), n_(n) {}

    /* The pickle's result, or a negative status with `why`. */
    int run(size_t* result, std::string* why);
    const V& at(size_t i) const { return pool_[i]; }
    size_t   size() const { return pool_.size(); }

private:
    static constexpr size_t kMaxPool = (size_t)1 << 23;
    static constexpr size_t kMaxStack = (size_t)1 << 20;
    static constexpr uint64_t kMaxStr = (uint64_t)1 << 20;

    int no(const std::string& w) {
        why_ = w + " at byte " + std::to_string(at_) + " of the pickle";
        return RAD_E_FORMAT;
    }
    bool need(uint64_t k) const { return n_ - at_ >= k; }
    size_t add(V v) {
        pool_.push_back(std::move(v));
        return pool_.size() - 1;
    }
    bool pop(size_t* v) {
        if (stack_.size() <= (marks_.empty() ? 0 : marks_.back())) return false;
        *v = stack_.back();
        stack_.pop_back();
        return true;
    }
    /* The items above the last mark, and the mark gone. */
    bool pop_mark(std::vector<size_t>* items) {
        if (marks_.empty()) return false;
        const size_t m = marks_.back();
        marks_.pop_back();
        items->assign(stack_.begin() + (std::ptrdiff_t)m, stack_.end());
        stack_.resize(m);
        return true;
    }
    int push(size_t v) {
        if (stack_.size() >= kMaxStack || pool_.size() >= kMaxPool)
            return no("the pickle builds more than its bounds allow");
        stack_.push_back(v);
        return RAD_OK;
    }
    int str(uint64_t len, V::K k) {
        if (len > kMaxStr || !need(len)) return no("a string past its bound or the pickle's end");
        V v;
        v.k = k;
        v.s.assign((const char*)p_ + at_, (size_t)len);
        at_ += len;
        return push(add(std::move(v)));
    }
    int integer(int64_t x) {
        V v;
        v.k = V::Int;
        v.i = x;
        return push(add(std::move(v)));
    }
    int global(const std::string& mod, const std::string& name) {
        const std::string g = mod + "." + name;
        if (!allowed_global(g))
            return no("the global '" + g + "', which a weights-only checkpoint does not name; "
                      "nothing in the file is run, so it is refused");
        V v;
        v.k = V::Global;
        v.s = g;
        return push(add(std::move(v)));
    }
    int reduce();
    int persistent_id();

    const uint8_t* p_;
    uint64_t       n_;
    uint64_t       at_ = 0;
    std::vector<V> pool_;
    std::vector<size_t> stack_, marks_;
    std::unordered_map<uint32_t, size_t> memo_;
    std::string why_;
};

int Machine::persistent_id() {
    size_t pid;
    if (!pop(&pid)) return no("BINPERSID with nothing to read");
    const V& t = pool_[pid];
    /* ('storage', <storage class>, key, location, numel) */
    if (t.k != V::Tuple || t.a.size() != 5 || pool_[t.a[0]].k != V::Str ||
        pool_[t.a[0]].s != "storage" || pool_[t.a[1]].k != V::Global ||
        pool_[t.a[2]].k != V::Str || pool_[t.a[4]].k != V::Int)
        return no("a persistent id that is not a storage");
    V s;
    s.k = V::Storage;
    s.cls = pool_[t.a[1]].s;
    s.dtype = storage_dtype(s.cls);
    s.s = pool_[t.a[2]].s;
    s.i = pool_[t.a[4]].i;
    if (s.dtype == RAD_DT_INVALID) return no("the storage class '" + s.cls + "'");
    if (s.i < 0) return no("a storage of negative size");
    return push(add(std::move(s)));
}

int Machine::reduce() {
    size_t args, fn;
    if (!pop(&args) || !pop(&fn)) return no("REDUCE without its callable and arguments");
    if (pool_[fn].k != V::Global) return no("REDUCE of something that is not a global");
    if (pool_[args].k != V::Tuple) return no("REDUCE with arguments that are not a tuple");
    const std::string g = pool_[fn].s;
    const std::vector<size_t> a = pool_[args].a;
    if (g == "collections.OrderedDict") {
        if (!a.empty()) return no("an OrderedDict built from arguments");
        V d;
        d.k = V::Dict;
        return push(add(std::move(d)));
    }
    if (g == "torch._utils._rebuild_tensor_v2") {
        /* (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata]) */
        if (a.size() < 4 || pool_[a[0]].k != V::Storage || pool_[a[1]].k != V::Int ||
            pool_[a[2]].k != V::Tuple || pool_[a[3]].k != V::Tuple)
            return no("_rebuild_tensor_v2 with arguments that are not a tensor's");
        V t;
        t.k = V::Tensor;
        t.storage = a[0];
        t.offset = pool_[a[1]].i;
        for (size_t d : pool_[a[2]].a) {
            if (pool_[d].k != V::Int || pool_[d].i < 0) return no("a tensor size that is not one");
            t.shape.push_back(pool_[d].i);
        }
        for (size_t d : pool_[a[3]].a) {
            if (pool_[d].k != V::Int) return no("a tensor stride that is not one");
            t.stride.push_back(pool_[d].i);
        }
        if (t.shape.size() != t.stride.size() || t.shape.size() > RAD_MAX_RANK)
            return no("a tensor whose size and stride disagree, or of rank past " +
                      std::to_string(RAD_MAX_RANK));
        return push(add(std::move(t)));
    }
    if (g == "torch._utils._rebuild_parameter") {
        /* (data, requires_grad, backward_hooks): the parameter is its tensor. */
        if (a.empty() || pool_[a[0]].k != V::Tensor) return no("_rebuild_parameter of a non-tensor");
        return push(a[0]);
    }
    return no("a call of '" + g + "'");
}

int Machine::run(size_t* result, std::string* why) {
    auto fail = [&](int st) {
        *why = why_;
        return st;
    };
    auto line = [&](std::string* out) {
        const uint8_t* e = (const uint8_t*)std::memchr(p_ + at_, '\n', (size_t)(n_ - at_));
        if (!e || (uint64_t)(e - (p_ + at_)) > 4096) return false;
        out->assign((const char*)p_ + at_, (size_t)(e - (p_ + at_)));
        at_ = (uint64_t)(e - p_) + 1;
        return true;
    };
    int st = RAD_OK;
    while (at_ < n_) {
        const uint8_t op = p_[at_++];
        switch (op) {
        case 0x80:   /* PROTO */
            if (!need(1) || p_[at_] < 2 || p_[at_] > 5) return fail(no("a pickle protocol other than 2..5"));
            ++at_;
            break;
        case 0x95:   /* FRAME: a length hint */
            if (!need(8)) return fail(no("a truncated FRAME"));
            at_ += 8;
            break;
        case 'c': {  /* GLOBAL "module\nname\n" */
            std::string mod, name;
            if (!line(&mod) || !line(&name)) return fail(no("a malformed GLOBAL"));
            st = global(mod, name);
            break;
        }
        case 0x93: { /* STACK_GLOBAL */
            size_t nm, md;
            if (!pop(&nm) || !pop(&md) || pool_[nm].k != V::Str || pool_[md].k != V::Str)
                return fail(no("STACK_GLOBAL without a module and a name"));
            st = global(pool_[md].s, pool_[nm].s);
            break;
        }
        case 'q': case 'r': case 0x94: {   /* BINPUT, LONG_BINPUT, MEMOIZE */
            uint32_t k;
            if (op == 'q') {
                if (!need(1)) return fail(no("a truncated BINPUT"));
                k = p_[at_++];
            } else if (op == 'r') {
                if (!need(4)) return fail(no("a truncated LONG_BINPUT"));
                k = u32(p_ + at_);
                at_ += 4;
            } else {
                k = (uint32_t)memo_.size();
            }
            if (stack_.empty()) return fail(no("a memo store with nothing to store"));
            if (memo_.size() >= kMaxStack) return fail(no("a memo past its bound"));
            memo_[k] = stack_.back();
            break;
        }
        case 'h': case 'j': {   /* BINGET, LONG_BINGET */
            uint32_t k;
            if (op == 'h') {
                if (!need(1)) return fail(no("a truncated BINGET"));
                k = p_[at_++];
            } else {
                if (!need(4)) return fail(no("a truncated LONG_BINGET"));
                k = u32(p_ + at_);
                at_ += 4;
            }
            auto it = memo_.find(k);
            if (it == memo_.end()) return fail(no("a memo read of an empty slot"));
            st = push(it->second);
            break;
        }
        case '(':    /* MARK */
            if (marks_.size() >= kMaxStack) return fail(no("marks past their bound"));
            marks_.push_back(stack_.size());
            break;
        case ')': {  /* EMPTY_TUPLE */
            V t;
            t.k = V::Tuple;
            st = push(add(std::move(t)));
            break;
        }
        case 't': case 0x85: case 0x86: case 0x87: {   /* TUPLE, TUPLE1..3 */
            V t;
            t.k = V::Tuple;
            if (op == 't') {
                if (!pop_mark(&t.a)) return fail(no("TUPLE without a mark"));
            } else {
                const int k = op - 0x84;
                t.a.resize((size_t)k);
                for (int j = k - 1; j >= 0; --j)
                    if (!pop(&t.a[(size_t)j])) return fail(no("a short tuple"));
            }
            st = push(add(std::move(t)));
            break;
        }
        case ']': case '}': {   /* EMPTY_LIST, EMPTY_DICT */
            V c;
            c.k = op == ']' ? V::List : V::Dict;
            st = push(add(std::move(c)));
            break;
        }
        case 'a': case 'e': {   /* APPEND, APPENDS */
            std::vector<size_t> items(1);
            if (op == 'a') {
                if (!pop(&items[0])) return fail(no("APPEND with nothing to append"));
            } else if (!pop_mark(&items)) {
                return fail(no("APPENDS without a mark"));
            }
            if (stack_.empty() || pool_[stack_.back()].k != V::List)
                return fail(no("APPEND to something that is not a list"));
            V& l = pool_[stack_.back()];
            l.a.insert(l.a.end(), items.begin(), items.end());
            break;
        }
        case 's': case 'u': {   /* SETITEM, SETITEMS */
            std::vector<size_t> kv(2);
            if (op == 's') {
                if (!pop(&kv[1]) || !pop(&kv[0])) return fail(no("SETITEM without a pair"));
            } else if (!pop_mark(&kv) || kv.size() % 2) {
                return fail(no("SETITEMS without whole pairs"));
            }
            if (stack_.empty() || pool_[stack_.back()].k != V::Dict)
                return fail(no("a SETITEM on something that is not a dict"));
            V& d = pool_[stack_.back()];
            d.a.insert(d.a.end(), kv.begin(), kv.end());
            break;
        }
        case 'b': {  /* BUILD: an OrderedDict's state (a state dict's _metadata), ignored */
            size_t state;
            if (!pop(&state)) return fail(no("BUILD without a state"));
            if (stack_.empty() || pool_[stack_.back()].k != V::Dict)
                return fail(no("BUILD of something other than a dict"));
            break;
        }
        case 'X': case 0x8c: case 0x8d: case 'B': case 'C': case 0x8e: {
            /* BINUNICODE, SHORT_BINUNICODE, BINUNICODE8, BINBYTES, SHORT_BINBYTES, BINBYTES8 */
            uint64_t len;
            const int w = (op == 0x8c || op == 'C') ? 1 : (op == 0x8d || op == 0x8e) ? 8 : 4;
            if (!need((uint64_t)w)) return fail(no("a truncated string length"));
            len = w == 1 ? p_[at_] : w == 4 ? u32(p_ + at_) : u64(p_ + at_);
            at_ += (uint64_t)w;
            st = str(len, (op == 'B' || op == 'C' || op == 0x8e) ? V::Bytes : V::Str);
            break;
        }
        case 'J':    /* BININT */
            if (!need(4)) return fail(no("a truncated BININT"));
            st = integer((int32_t)u32(p_ + at_));
            at_ += 4;
            break;
        case 'K':    /* BININT1 */
            if (!need(1)) return fail(no("a truncated BININT1"));
            st = integer(p_[at_++]);
            break;
        case 'M':    /* BININT2 */
            if (!need(2)) return fail(no("a truncated BININT2"));
            st = integer(u16(p_ + at_));
            at_ += 2;
            break;
        case 0x8a: { /* LONG1: little-endian two's complement of up to 8 bytes here */
            if (!need(1)) return fail(no("a truncated LONG1"));
            const uint64_t k = p_[at_++];
            if (k > 8 || !need(k)) return fail(no("a LONG1 wider than 64 bits"));
            uint64_t v = 0;
            for (uint64_t j = 0; j < k; ++j) v |= (uint64_t)p_[at_ + j] << (8 * j);
            if (k > 0 && k < 8 && (p_[at_ + k - 1] & 0x80)) v |= ~(uint64_t)0 << (8 * k);
            at_ += k;
            st = integer((int64_t)v);
            break;
        }
        case 'G': {  /* BINFLOAT, big-endian */
            if (!need(8)) return fail(no("a truncated BINFLOAT"));
            uint64_t v = 0;
            for (int j = 0; j < 8; ++j) v = (v << 8) | p_[at_ + j];
            at_ += 8;
            V f;
            f.k = V::Float;
            std::memcpy(&f.f, &v, 8);
            st = push(add(std::move(f)));
            break;
        }
        case 0x88: case 0x89: {   /* NEWTRUE, NEWFALSE */
            V v;
            v.k = V::Bool;
            v.i = op == 0x88;
            st = push(add(std::move(v)));
            break;
        }
        case 'N':
            st = push(add(V{}));
            break;
        case 'Q':
            st = persistent_id();
            break;
        case 'R':
            st = reduce();
            break;
        case '.': {  /* STOP */
            size_t r;
            if (!pop(&r)) return fail(no("STOP with nothing to return"));
            *result = r;
            return RAD_OK;
        }
        default: {
            char hex[8];
            std::snprintf(hex, sizeof hex, "0x%02x", op);
            return fail(no(std::string("the opcode ") + hex +
                           ", which a weights-only checkpoint does not use"));
        }
        }
        if (st < 0) return fail(st);
    }
    return fail(no("no STOP before the pickle's end"));
}

/* A state dict's tensors, nested dicts flattened to dotted names. The memo lets a dict hold itself
 * or be shared by several parents; a state dict's dicts form a tree, so a dict reached a second
 * time is refused, which also bounds the walk by the pool. */
constexpr int kMaxDictDepth = 64;

bool walk(const Machine& m, size_t d, const std::string& prefix, int depth,
          std::vector<char>* seen, std::vector<std::pair<std::string, size_t>>* out,
          std::string* why) {
    if (depth > kMaxDictDepth) {
        *why = "dicts nested deeper than " + std::to_string(kMaxDictDepth) + " under '" + prefix +
               "'";
        return false;
    }
    if ((*seen)[d]) {
        *why = "a dict reached twice (under '" + prefix + "'); a state dict's dicts form a tree";
        return false;
    }
    (*seen)[d] = 1;
    const V& v = m.at(d);
    for (size_t j = 0; j + 1 < v.a.size(); j += 2) {
        const V& k = m.at(v.a[j]);
        if (k.k != V::Str) continue;
        const V& x = m.at(v.a[j + 1]);
        if (x.k == V::Tensor) out->emplace_back(prefix + k.s, v.a[j + 1]);
        else if (x.k == V::Dict &&
                 !walk(m, v.a[j + 1], prefix + k.s + ".", depth + 1, seen, out, why))
            return false;
    }
    return true;
}

}  /* namespace */

/* ================================================================== the file */
TorchZipFile::~TorchZipFile() { close(); }

void TorchZipFile::close() {
    if (map_) ::munmap(map_, (size_t)bytes_);
    if (fd_ >= 0) ::close(fd_);
    map_ = nullptr;
    base_ = nullptr;
    bytes_ = 0;
    fd_ = -1;
    tensors_.clear();
}

int TorchZipFile::open(const std::string& path, std::string* why) {
    close();
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        *why = std::strerror(errno);
        return RAD_E_IO;
    }
    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        *why = std::strerror(errno);
        close();
        return RAD_E_IO;
    }
    bytes_ = (uint64_t)st.st_size;
    void* m = bytes_ ? ::mmap(nullptr, (size_t)bytes_, PROT_READ, MAP_PRIVATE, fd_, 0) : MAP_FAILED;
    if (m == MAP_FAILED) {
        *why = bytes_ ? std::string("mmap: ") + std::strerror(errno) : std::string("empty file");
        close();
        return bytes_ ? RAD_E_IO : RAD_E_FORMAT;
    }
    map_ = (uint8_t*)m;
    base_ = map_;
    const int s = parse(why);
    if (s < 0) close();
    return s;
}

int TorchZipFile::open_bytes(const uint8_t* data, uint64_t bytes, std::string* why) {
    close();
    path_ = "(memory)";
    base_ = data;
    bytes_ = bytes;
    return parse(why);
}

int TorchZipFile::parse(std::string* why) {
    std::unordered_map<std::string, Entry> ent;
    RAD_TRY(zip_entries(base_, bytes_, &ent, why));
    /* The archive's one directory: the one holding data.pkl. */
    std::string dir;
    int n_pkl = 0;
    for (const auto& [name, e] : ent) {
        const size_t sl = name.find('/');
        if (sl != std::string::npos && name.compare(sl, std::string::npos, "/data.pkl") == 0) {
            dir = name.substr(0, sl + 1);
            ++n_pkl;
        }
    }
    if (n_pkl != 1) {
        *why = n_pkl ? "more than one data.pkl" : "no <archive>/data.pkl: not a torch.save checkpoint";
        return RAD_E_FORMAT;
    }
    if (auto it = ent.find(dir + "byteorder"); it != ent.end() &&
        std::string((const char*)it->second.data, (size_t)it->second.bytes) != "little") {
        *why = "a big-endian checkpoint";
        return RAD_E_FORMAT;
    }
    const Entry& pk = ent[dir + "data.pkl"];
    Machine m(pk.data, pk.bytes);
    size_t root = 0;
    RAD_TRY(m.run(&root, why));
    if (m.at(root).k != V::Dict) {
        *why = "the pickle's result is not a state dict";
        return RAD_E_FORMAT;
    }
    std::vector<std::pair<std::string, size_t>> found;
    std::vector<char> seen(m.size(), 0);
    if (!walk(m, root, "", 0, &seen, &found, why)) return RAD_E_FORMAT;
    for (const auto& [name, ti] : found) {
        const V& t = m.at(ti);
        const V& s = m.at(t.storage);
        auto it = ent.find(dir + "data/" + s.s);
        if (it == ent.end()) {
            *why = "'" + name + "' reads storage '" + s.s + "', which the archive does not hold";
            return RAD_E_FORMAT;
        }
        const int64_t esz = (int64_t)rad_dtype_bits(s.dtype) / 8;
        /* The element count in checked arithmetic: a product that wraps would pass the fit
         * check below with a shape that names far more bytes than the storage holds. */
        int64_t numel = 1;
        bool contig = true, wraps = false;
        for (size_t d = t.shape.size(); d-- > 0;) {
            if (t.shape[d] != 1 && t.stride[d] != numel) contig = false;
            wraps |= __builtin_mul_overflow(numel, t.shape[d], &numel);
        }
        int64_t end_bytes = 0;
        if (!wraps && t.offset >= 0)
            wraps = __builtin_add_overflow(t.offset, numel, &end_bytes) ||
                    __builtin_mul_overflow(end_bytes, esz, &end_bytes);
        if (wraps) {
            *why = "'" + name + "' has a size or offset past 64 bits";
            return RAD_E_FORMAT;
        }
        if (!contig) {
            *why = "'" + name + "' is not contiguous in its storage";
            return RAD_E_FORMAT;
        }
        if (t.offset < 0 || numel > s.i - t.offset || (uint64_t)end_bytes > it->second.bytes) {
            *why = "'" + name + "' does not fit its storage '" + s.s + "'";
            return RAD_E_FORMAT;
        }
        if (numel == 0) continue;
        TorchTensor tt;
        tt.name = name;
        tt.dtype = s.dtype;
        tt.storage = s.cls;
        tt.shape = t.shape;
        tt.data = it->second.data + t.offset * esz;
        tt.bytes = (uint64_t)(numel * esz);
        tensors_.push_back(std::move(tt));
    }
    return RAD_OK;
}

}  /* namespace rad */
