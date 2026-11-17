/* aql_code.cpp -- reads AMDGPU code objects: the offload bundle of a HIP fat binary, the ELF
 * inside it, and the msgpack metadata note that gives each kernel's argument layout.
 *
 * The AQL backend dispatches HIP-registered kernels itself, so it needs two things HIP does not
 * expose: the card's code object, to load into its own executable, and each kernel's argument
 * layout, to write kernarg segments. Both come from the embedded fat binary.
 *
 * Nothing here touches a device, so it builds and is tested without one.
 */
#include "aql.h"

#include <cstring>

namespace rad {
namespace aql {

/* ------------------------------------------------------------------ the offload bundle */

namespace {

const char kHsaPrefix[] = "amdgcn-amd-amdhsa--";

/* A target ID as LLVM writes it after the triple: a processor and target features, each `name+`
 * or `name-` ("gfx942:sramecc+:xnack-"). A feature absent from a code object's ID means it runs in
 * either mode; absent from a card's ID, the card has no such mode. */
struct TargetId {
    std::string proc;
    std::vector<std::pair<std::string, char>> feat;
};

TargetId parse_target_id(const std::string& s) {
    TargetId t;
    size_t i = s.find(':');
    t.proc = s.substr(0, i);
    while (i != std::string::npos) {
        const size_t j = s.find(':', i + 1);
        const std::string f = s.substr(i + 1, j == std::string::npos ? std::string::npos : j - i - 1);
        if (f.size() > 1 && (f.back() == '+' || f.back() == '-'))
            t.feat.emplace_back(f.substr(0, f.size() - 1), f.back());
        i = j;
    }
    return t;
}

/* The target ID from an ISA name ("amdgcn-amd-amdhsa--gfx1036") or a bundle entry's triple
 * ("hipv4-amdgcn-amd-amdhsa--gfx1036"); empty for non-AMDGPU targets such as the host entry. */
std::string target_id_of(const std::string& name) {
    const size_t at = name.find(kHsaPrefix);
    if (at == std::string::npos) return std::string();
    return name.substr(at + sizeof kHsaPrefix - 1);
}

/* An ISA name's target ID, or the name itself when it carries no HSA prefix. */
std::string isa_target(const std::string& isa) {
    const std::string id = target_id_of(isa);
    return id.empty() ? isa : id;
}

/* -1 if a code object for `co` cannot run on a card accepting `card`; otherwise the number of
 * features the code object pins, so one built for the card's exact mode is preferred. */
int target_fit(const TargetId& co, const TargetId& card) {
    if (co.proc != card.proc) return -1;
    for (const auto& f : co.feat) {
        bool ok = false;
        for (const auto& g : card.feat)
            if (g.first == f.first) { ok = g.second == f.second; break; }
        if (!ok) return -1;
    }
    return (int)co.feat.size();
}

}  // namespace

bool bundle_code_object(const void* bundle, const std::vector<std::string>& isas, const void** out,
                        size_t* out_size, std::string* why) {
    static const char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
    const unsigned char* b = static_cast<const unsigned char*>(bundle);
    *out = nullptr;
    *out_size = 0;
    if (!b) { *why = "null bundle"; return false; }
    if (std::memcmp(b, "CCOB", 4) == 0) {
        *why = "the fat binary is a compressed offload bundle; build the kernel library without "
               "--offload-compress";
        return false;
    }
    if (std::memcmp(b, kMagic, sizeof kMagic - 1) != 0) {
        *why = "the fat binary does not start with a clang offload bundle header";
        return false;
    }
    uint64_t n = 0;
    std::memcpy(&n, b + 24, 8);
    if (n == 0 || n > 4096) { *why = "offload bundle with an implausible entry count"; return false; }

    /* Device entries are "<kind>-amdgcn-amd-amdhsa--<target ID>"; both kinds, "hip" and "hipv4",
     * carry the same code object format for this purpose. */
    struct Entry { uint64_t off, size; TargetId id; std::string text; };
    std::vector<Entry> ents;
    const unsigned char* p = b + 32;
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t off = 0, size = 0, tl = 0;
        std::memcpy(&off, p, 8);
        std::memcpy(&size, p + 8, 8);
        std::memcpy(&tl, p + 16, 8);
        if (tl > 4096) { *why = "offload bundle entry with an implausible triple"; return false; }
        const std::string triple(reinterpret_cast<const char*>(p + 24), (size_t)tl);
        p += 24 + tl;
        const std::string id = target_id_of(triple);
        if (size == 0 || id.empty()) continue;
        ents.push_back({ off, size, parse_target_id(id), id });
    }

    /* Walk the card's ISA list in its own order: the exact processor first, then its family's
     * generic target. An exact-chip build wins where present; a family build runs on every
     * member. Within one target, a code object pinned to the card's mode wins. */
    for (const std::string& isa : isas) {
        const TargetId card = parse_target_id(isa_target(isa));
        const Entry* best = nullptr;
        int best_fit = -1;
        for (const Entry& e : ents) {
            const int f = target_fit(e.id, card);
            if (f > best_fit) { best_fit = f; best = &e; }
        }
        if (best) {
            *out = b + best->off;
            *out_size = (size_t)best->size;
            return true;
        }
    }

    std::string have, want;
    for (const Entry& e : ents) have += (have.empty() ? "" : ", ") + e.text;
    for (const std::string& isa : isas)
        want += (want.empty() ? "" : ", ") + isa_target(isa);
    *why = "the fat binary has code objects for [" + (have.empty() ? std::string("none") : have) +
           "] and this card runs [" + want + "]";
    return false;
}

/* ------------------------------------------------------------------ msgpack
 *
 * The subset code object metadata uses: maps, arrays, strings, integers, booleans and nil. Parsed
 * into a DOM: the metadata is looked up by key, is a few hundred KB at most, and is read once per
 * module at its first dispatch. */
namespace {

struct MP {
    enum T : uint8_t { Nil, Bool, Int, Str, Arr, Map, Other } t = Nil;
    int64_t i = 0;
    const char* s = nullptr;
    uint32_t n = 0;
    std::vector<MP> a;                       /* array items, or map keys and values interleaved */

    const MP* get(const char* key) const {
        if (t != Map) return nullptr;
        const size_t kl = std::strlen(key);
        for (size_t k = 0; k + 1 < a.size(); k += 2)
            if (a[k].t == Str && a[k].n == kl && std::memcmp(a[k].s, key, kl) == 0) return &a[k + 1];
        return nullptr;
    }
    std::string str() const { return t == Str ? std::string(s, n) : std::string(); }
};

struct Reader {
    const unsigned char* p;
    const unsigned char* e;
    bool ok = true;

    uint64_t be(int nb) {
        if (e - p < nb) { ok = false; return 0; }
        uint64_t v = 0;
        for (int k = 0; k < nb; ++k) v = (v << 8) | p[k];
        p += nb;
        return v;
    }
    void str(MP* m, uint32_t n) {
        if ((uint64_t)(e - p) < n) { ok = false; return; }
        m->t = MP::Str; m->s = reinterpret_cast<const char*>(p); m->n = n; p += n;
    }
    void items(MP* m, uint32_t n, int depth) {
        m->a.resize(n);
        for (uint32_t k = 0; k < n && ok; ++k) read(&m->a[k], depth + 1);
    }
    void read(MP* m, int depth) {
        if (!ok || p >= e || depth > 64) { ok = false; return; }
        const unsigned char c = *p++;
        if (c <= 0x7f)                { m->t = MP::Int; m->i = c; return; }
        if (c >= 0xe0)                { m->t = MP::Int; m->i = (int8_t)c; return; }
        if ((c & 0xf0) == 0x80)       { m->t = MP::Map; items(m, 2u * (c & 0x0f), depth); return; }
        if ((c & 0xf0) == 0x90)       { m->t = MP::Arr; items(m, c & 0x0f, depth); return; }
        if ((c & 0xe0) == 0xa0)       { str(m, c & 0x1f); return; }
        switch (c) {
            case 0xc0: m->t = MP::Nil; return;
            case 0xc2: m->t = MP::Bool; m->i = 0; return;
            case 0xc3: m->t = MP::Bool; m->i = 1; return;
            case 0xcc: m->t = MP::Int; m->i = (int64_t)be(1); return;
            case 0xcd: m->t = MP::Int; m->i = (int64_t)be(2); return;
            case 0xce: m->t = MP::Int; m->i = (int64_t)be(4); return;
            case 0xcf: m->t = MP::Int; m->i = (int64_t)be(8); return;
            case 0xd0: m->t = MP::Int; m->i = (int8_t)be(1); return;
            case 0xd1: m->t = MP::Int; m->i = (int16_t)be(2); return;
            case 0xd2: m->t = MP::Int; m->i = (int32_t)be(4); return;
            case 0xd3: m->t = MP::Int; m->i = (int64_t)be(8); return;
            case 0xca: m->t = MP::Other; be(4); return;
            case 0xcb: m->t = MP::Other; be(8); return;
            case 0xd9: str(m, (uint32_t)be(1)); return;
            case 0xda: str(m, (uint32_t)be(2)); return;
            case 0xdb: str(m, (uint32_t)be(4)); return;
            case 0xc4: case 0xc5: case 0xc6: {
                const uint32_t n = (uint32_t)be(c == 0xc4 ? 1 : c == 0xc5 ? 2 : 4);
                str(m, n); m->t = MP::Other; return;
            }
            case 0xdc: m->t = MP::Arr; items(m, (uint32_t)be(2), depth); return;
            case 0xdd: m->t = MP::Arr; items(m, (uint32_t)be(4), depth); return;
            case 0xde: m->t = MP::Map; items(m, 2u * (uint32_t)be(2), depth); return;
            case 0xdf: m->t = MP::Map; items(m, 2u * (uint32_t)be(4), depth); return;
            default: ok = false; return;   /* ext types: never in code object metadata */
        }
    }
};

Hidden hidden_kind(const std::string& k) {
    if (k.compare(0, 7, "hidden_") != 0) return Hidden::None;
    static const struct { const char* n; Hidden h; } kTab[] = {
        { "hidden_none",            Hidden::Pad },
        { "hidden_block_count_x",   Hidden::BlockCountX },
        { "hidden_block_count_y",   Hidden::BlockCountY },
        { "hidden_block_count_z",   Hidden::BlockCountZ },
        { "hidden_group_size_x",    Hidden::GroupSizeX },
        { "hidden_group_size_y",    Hidden::GroupSizeY },
        { "hidden_group_size_z",    Hidden::GroupSizeZ },
        { "hidden_remainder_x",     Hidden::RemainderX },
        { "hidden_remainder_y",     Hidden::RemainderY },
        { "hidden_remainder_z",     Hidden::RemainderZ },
        { "hidden_global_offset_x", Hidden::GlobalOffsetX },
        { "hidden_global_offset_y", Hidden::GlobalOffsetY },
        { "hidden_global_offset_z", Hidden::GlobalOffsetZ },
        { "hidden_grid_dims",       Hidden::GridDims },
        { "hidden_dynamic_lds_size", Hidden::DynamicLds },
    };
    for (const auto& t : kTab) if (k == t.n) return t.h;
    return Hidden::Unsupported;
}

}  // namespace

/* ------------------------------------------------------------------ the metadata note */

bool code_object_kernels(const void* elf, size_t size, std::vector<KernelMeta>* out,
                         std::string* why) {
    out->clear();
    const unsigned char* b = static_cast<const unsigned char*>(elf);
    if (size < 64 || std::memcmp(b, "\x7f" "ELF", 4) != 0 || b[4] != 2 /* ELFCLASS64 */) {
        *why = "not a 64-bit ELF code object";
        return false;
    }
    uint64_t phoff = 0;
    uint16_t phentsize = 0, phnum = 0;
    std::memcpy(&phoff, b + 32, 8);
    std::memcpy(&phentsize, b + 54, 2);
    std::memcpy(&phnum, b + 56, 2);
    if (phentsize < 56 || phoff + (uint64_t)phentsize * phnum > size) {
        *why = "code object program headers run past its end";
        return false;
    }
    for (uint16_t h = 0; h < phnum; ++h) {
        const unsigned char* ph = b + phoff + (uint64_t)h * phentsize;
        uint32_t type = 0;
        uint64_t off = 0, fsz = 0;
        std::memcpy(&type, ph, 4);
        std::memcpy(&off, ph + 8, 8);
        std::memcpy(&fsz, ph + 32, 8);
        if (type != 4 /* PT_NOTE */ || off + fsz > size) continue;
        const unsigned char* q = b + off;
        const unsigned char* qe = q + fsz;
        while (qe - q >= 12) {
            uint32_t namesz = 0, descsz = 0, ntype = 0;
            std::memcpy(&namesz, q, 4);
            std::memcpy(&descsz, q + 4, 4);
            std::memcpy(&ntype, q + 8, 4);
            const unsigned char* name = q + 12;
            const unsigned char* desc = name + ((namesz + 3u) & ~3u);
            const unsigned char* next = desc + ((descsz + 3u) & ~3u);
            if (next > qe) break;
            q = next;
            if (ntype != 32 /* NT_AMDGPU_METADATA */ || namesz != 7 ||
                std::memcmp(name, "AMDGPU", 6) != 0)
                continue;

            Reader r{ desc, desc + descsz };
            MP root;
            r.read(&root, 0);
            if (!r.ok || root.t != MP::Map) { *why = "malformed AMDGPU metadata"; return false; }
            const MP* ks = root.get("amdhsa.kernels");
            if (!ks || ks->t != MP::Arr) { *why = "metadata has no amdhsa.kernels"; return false; }
            for (const MP& k : ks->a) {
                KernelMeta m;
                const MP* sym = k.get(".symbol");
                if (!sym) continue;
                m.symbol = sym->str();
                if (const MP* v = k.get(".kernarg_segment_size"))       m.kernarg_size = (uint32_t)v->i;
                if (const MP* v = k.get(".group_segment_fixed_size"))   m.group_size   = (uint32_t)v->i;
                if (const MP* v = k.get(".private_segment_fixed_size")) m.private_size = (uint32_t)v->i;
                if (const MP* v = k.get(".uses_dynamic_stack"))         m.dynamic_stack = v->i != 0;
                if (const MP* as = k.get(".args"); as && as->t == MP::Arr) {
                    for (const MP& a : as->a) {
                        KernelArg ka;
                        if (const MP* v = a.get(".offset"))     ka.offset = (uint32_t)v->i;
                        if (const MP* v = a.get(".size"))       ka.size   = (uint32_t)v->i;
                        if (const MP* v = a.get(".value_kind")) ka.kind   = v->str();
                        ka.hidden = hidden_kind(ka.kind);
                        m.args.push_back(std::move(ka));
                    }
                }
                out->push_back(std::move(m));
            }
            return true;
        }
    }
    *why = "code object has no AMDGPU metadata note";
    return false;
}

}  // namespace aql
}  // namespace rad
