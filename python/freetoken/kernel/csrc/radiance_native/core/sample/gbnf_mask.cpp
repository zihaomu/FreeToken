/* gbnf_mask.cpp -- the token mask, precomputed per grammar so that no mask walks the vocabulary.
 *
 * A mask's cost adds to the decode step (gbnf.h, "Cost"). Descending the whole vocabulary trie for
 * a new state costs milliseconds on a quarter-million-piece vocabulary, so only the compile does
 * that, once per place a mask can start; a mask is assembled from the answers.
 *
 * Mask structure. A token is admissible iff some stack admits it, so the mask is the union of
 * per-stack masks. A short token either stays within the few frames nearest the top (a string's
 * content and closing quote, a free-text parameter and its end tag) or leaves them through a
 * delimiter. So a stack S is split into a key (the frames nearest the top) and the context below:
 *
 *     S = context ++ key
 *     mask(S) = accepted(key)  U  { t : t leaves the key after a prefix u, and the rest of
 *                                       t is admitted by the context }
 *
 * accepted(key) is independent of the context, so it is computed at compile time by descending
 * the vocabulary trie with the key over a bottom marker. The remainder (for every token, the text
 * after each point where it pops the key's last frame) is kept as a small trie, and only that is
 * walked at run time against the real context. This is XGrammar's adaptive token mask cache
 * (Dong et al., 2024), with stack suffixes as positions.
 *
 * Suffixes, not single elements. The parser gives `x*` and groups rules of their own, so a stack
 * top is often a one-element rule such as `[^"\\]` in a JSON string, which every multi-character
 * token leaves. Keyed on the top three frames (the class, its repetition, the closing quote) only
 * the few hundred tokens that close the string leave. Depth is adaptive: a key is extended one
 * frame down, into every frame that can sit below it, while its remainder is too large to walk.
 *
 * Keys. Two stack entries behave identically when the rest of their alternatives is the same
 * element sequence (`"</parameter>\n"` closes every tool parameter). Entries are replaced by their
 * class, the hash-consed remainder of their alternative, and keys are class sequences. Rules are
 * hash-consed first, and a repetition longer than the longest token is one class whatever its
 * count. So fifty free-text parameters share one key, and `x{0,1000}` costs what `x{0,128}` does
 * when the longest token is 128 characters.
 *
 * Mask cost. Per stack: a key lookup a few classes deep; an OR of the key's accepted set (word-wise
 * for a permissive key, a few bits for a restrictive one); and, if the key has a remainder and the
 * stack a context, a remainder-trie walk visiting only what the context keeps alive. Only the OR
 * scales with the vocabulary, and the cost is the same for new and seen states.
 *
 * Split characters. A state carrying a partial UTF-8 character admits only tokens that begin with
 * the missing bytes. Those few hundred pieces are kept in a trie per missing-byte count and walked
 * with the full stack set.
 *
 * The automaton. A set's successor depends only on which of its stacks match the code point, and
 * its character classes cut the code points into intervals where that is constant; each state
 * stores its intervals and per-interval successor, built on first use. Trie children are sorted by
 * code point, so a walk takes a run of children per interval and skips a non-matching interval
 * with one binary search, making a restrictive state cheap even at the widest node.
 *
 * Refusal. A grammar whose keys cannot be made small within kMaxKeyDepth frames (token reach
 * depends on unbounded parse depth), or whose plan exceeds kMaxCompileWork, kMaxPlanBytes or the
 * stack-set bound, is refused at compile time, naming the rule where possible, rather than served
 * with slow masks. Grammars from chat templates and the JSON Schema converter are not affected.
 */
#include "gbnf_internal.h"

#include <bit>
#include <chrono>
#include <cstring>
#include <deque>
#include <iterator>
#include <mutex>

namespace rad {
namespace gbnf {

namespace {

constexpr uint32_t kNone    = 0xFFFFFFFFu;
constexpr int32_t  kUnbuilt = -2;

/* Maximum size of a key's remainder, which bounds a mask's run-time cost per key: the remainder
 * trie walk visits every node in the worst case. Past either bound the key is extended one frame
 * down, so more is decided at compile time. */
constexpr size_t kMaxRemEntries = 1u << 13;
constexpr size_t kMaxRemNodes   = 1u << 13;

/* A walk whose remainder is already far past the bound stops early, since the key will be extended
 * anyway (otherwise a key left after one character walks the whole vocabulary). Counted as tokens
 * under nodes where the key was left, an overestimate, so it never stops a walk that would fit. */
constexpr uint64_t kAbortRemTokens = 1u << 15;

/* Maximum frames per key. Real grammars settle within three or four (a string in its repetition in
 * its quotes; whitespace in its optional). A key still too permissive this deep means masks depend
 * on unbounded stack context, and the grammar is refused. */
constexpr uint16_t kMaxKeyDepth = 12;

inline void set_bit(uint32_t* bits, int32_t id) {
    bits[(uint32_t)id >> 5] |= 1u << ((uint32_t)id & 31u);
}

inline bool test_bit(const uint32_t* bits, int32_t id) {
    return (bits[(uint32_t)id >> 5] >> ((uint32_t)id & 31u)) & 1u;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
        .count();
}

}  /* namespace */

/* ============================================================ token tries

 * A trie over code-point sequences, one terminal per token: one for the vocabulary, one per key
 * remainder, and three for pieces that resume a split character.
 *
 * Nodes are in breadth-first order, so a node's children are a contiguous id range (no child
 * pointers) and a breadth-first walk reads the node array front to back. Fields are one struct
 * per node since a visit reads all four, and terminals are in node order. The walk is
 * memory-bound; a pointer-linked layout is slower than the rescanning walk the trie replaces.
 *
 * Each node also records its subtree's range `lo..hi` in lexicographic order (`lex`). The walk
 * never reads it; the compile uses it to list what lies under a node where a key was left, as
 * do token elements (`<[id]>`, `!<[id]>`), which match a token's id wherever reached. */
struct TrieNode {
    uint32_t cp;        /* the code point on the edge into this node */
    uint32_t first;     /* its first child's node id */
    uint32_t n_child;   /* how many children follow it */
    uint32_t tbeg;      /* its first terminal slot; it ends at the next node's tbeg */
};

struct TokenTrie {
    std::vector<TrieNode>    node;     /* plus one sentinel, so node[i+1].tbeg is always valid */
    std::vector<int32_t>     tid;      /* the tokens that end at a node, in node order */
    std::vector<PartialUtf8> ttail;    /* and the incomplete sequence each one leaves behind */
    std::vector<uint32_t>    lo, hi;   /* per node: its subtree's range of `lex` */
    std::vector<int32_t>     lex;      /* token ids in lexicographic order of their sequences */
    std::vector<uint32_t>    pos;      /* token id -> its index in `lex`, when each appears once */
    uint32_t                 depth = 0;

    size_t n_nodes() const { return node.empty() ? 0 : node.size() - 1; }
    size_t bytes() const {
        return node.size() * sizeof(TrieNode) + tid.size() * 4 +
               ttail.size() * sizeof(PartialUtf8) +
               (lo.size() + hi.size() + lex.size() + pos.size()) * 4;
    }
};

struct TrieEntry {
    uint32_t    off;    /* where the sequence starts in the arena; it runs to a 0 */
    PartialUtf8 tail;
    int32_t     id;
};

/* Sorted lexicographically, every trie node is a contiguous slice of the entries (sequences
 * sharing a prefix are adjacent), so the build needs no per-node map. Sequences are 0-terminated
 * in the arena, which stops the comparison. */
static void trie_build(TokenTrie& t, const uint32_t* cps, std::vector<TrieEntry>& ents,
                       int32_t unique_over) {
    const uint32_t n = (uint32_t)ents.size();
    /* Sort by a packed key of the first three code points (21 bits each; the terminator is 0, as
     * is the padding past a shorter sequence, which is exactly the comparison's own rule), and
     * walk the strings only where the keys tie: a vocabulary's 48k entries nearly all differ
     * inside three code points, so the sort compares integers where it walked both strings. */
    {
        struct Keyed { uint64_t key; TrieEntry e; };
        std::vector<Keyed> keyed;
        keyed.reserve(n);
        for (const TrieEntry& e : ents) {
            const uint32_t* p = cps + e.off;
            uint64_t k = p[0];
            k = (k << 21) | (p[0] ? p[1] : 0u);
            k = (k << 21) | (p[0] && p[1] ? p[2] : 0u);
            keyed.push_back({ k, e });
        }
        std::sort(keyed.begin(), keyed.end(), [cps](const Keyed& a, const Keyed& b) {
            if (a.key != b.key) return a.key < b.key;
            const uint32_t* pa = cps + a.e.off;
            const uint32_t* pb = cps + b.e.off;
            while (*pa != 0 && *pa == *pb) { ++pa; ++pb; }
            return *pa < *pb;
        });
        for (uint32_t i = 0; i < n; ++i) ents[i] = keyed[i].e;
    }

    t.lex.resize(n);
    for (uint32_t i = 0; i < n; ++i) t.lex[i] = ents[i].id;
    t.tid.reserve(n);
    t.ttail.reserve(n);

    struct Range { uint32_t lo, hi; };
    std::vector<Range> cur, nxt;
    t.node.push_back({ 0u, 0u, 0u, 0u });
    t.lo.push_back(0);
    t.hi.push_back(n);
    cur.push_back({ 0u, n });

    /* Node `level_lo + i` has range cur[i]. Children are appended in the same sweep, keeping each
     * node's children and the next level contiguous, and making node ids follow processing order,
     * which the terminal slots depend on. */
    uint32_t level_lo = 0;
    for (uint32_t d = 0; !cur.empty(); ++d) {
        nxt.clear();
        const uint32_t next_lo = (uint32_t)t.node.size();
        for (size_t ni = 0; ni < cur.size(); ++ni) {
            const uint32_t node = level_lo + (uint32_t)ni;
            const Range    r    = cur[ni];
            t.node[node].tbeg   = (uint32_t)t.tid.size();

            uint32_t i = r.lo;
            while (i < r.hi && cps[ents[i].off + d] == 0) {     /* ends exactly here */
                t.tid.push_back(ents[i].id);
                t.ttail.push_back(ents[i].tail);
                ++i;
            }
            const uint32_t base = (uint32_t)t.node.size();
            t.node[node].first = base;
            while (i < r.hi) {
                const uint32_t c = cps[ents[i].off + d];
                uint32_t j = i + 1;
                while (j < r.hi && cps[ents[j].off + d] == c) ++j;
                t.node.push_back({ c, 0u, 0u, 0u });
                t.lo.push_back(i);
                t.hi.push_back(j);
                nxt.push_back({ i, j });
                i = j;
            }
            t.node[node].n_child = (uint32_t)t.node.size() - base;
        }
        level_lo = next_lo;
        cur.swap(nxt);
        t.depth = d + 1;
    }
    /* The sentinel: the last real node's terminals end here. */
    t.node.push_back({ 0u, 0u, 0u, (uint32_t)t.tid.size() });

    if (unique_over > 0) {
        t.pos.assign((size_t)unique_over, kNone);
        for (uint32_t i = 0; i < n; ++i) t.pos[(size_t)t.lex[i]] = i;
    }
}

/* ============================================================ the vocabulary index */

struct VocabIndex {
    int32_t              n_tokens = 0;
    std::vector<int32_t> eog;          /* end-of-generation: admissible once a stack completes */

    /* Every piece decoded from a clean start, 0-terminated, in one arena that opens with a pad
     * element. `off` is kNone for tokens the walk skips: end-of-generation, and control tokens
     * with no text. */
    std::vector<uint32_t>    cps;
    std::vector<uint32_t>    off;
    std::vector<PartialUtf8> tail;
    TokenTrie                trie;

    /* Pieces that decode to no characters and carry nothing out (e.g. an overlong U+0000 first,
     * which ends a decoded sequence). Every live stack admits them, a completed one included, so
     * a state with a completed stack admits these regardless of its keys. */
    std::vector<int32_t>     blank;

    /* Pieces that resume a split character, indexed by bytes still missing (r). A piece is
     * admissible only if its first r bytes are continuation bytes and the rest decodes alone; its
     * first code point is the carried bits followed by those r bytes' bits. So root edges hold
     * the r bytes' bits plus one (0 ends a sequence) and the walk ORs the carried bits in. A piece
     * of fewer than r continuation bytes only (`shorts`) leaves the character unfinished and is
     * admissible iff some stack could still take it. */
    struct Resume {
        std::vector<uint32_t> cps;
        TokenTrie             trie;
        struct Short { uint32_t bits; int32_t k; int32_t id; };
        std::vector<Short>    shorts;
    };
    Resume resume[3];

    size_t bytes() const {
        size_t b = cps.size() * 4 + off.size() * 4 + tail.size() * sizeof(PartialUtf8) +
                   (eog.size() + blank.size()) * 4 + trie.bytes();
        for (const Resume& r : resume)
            b += r.cps.size() * 4 + r.trie.bytes() + r.shorts.size() * sizeof(Resume::Short);
        return b;
    }
};

size_t vocab_index_bytes(const VocabIndex& x) { return x.bytes(); }

std::shared_ptr<const VocabIndex> vocab_index_build(const VocabView& v) {
    const auto t0 = std::chrono::steady_clock::now();
    auto x = std::make_shared<VocabIndex>();
    const int32_t n = v.n_tokens();
    x->n_tokens = n > 0 ? n : 0;
    x->off.assign((size_t)x->n_tokens, kNone);
    x->tail.assign((size_t)x->n_tokens, PartialUtf8{});
    x->cps.push_back(0);
    for (auto& r : x->resume) r.cps.push_back(0);

    std::vector<TrieEntry> ents;
    std::vector<TrieEntry> rents[3];
    ents.reserve((size_t)x->n_tokens);

    for (int32_t id = 0; id < x->n_tokens; ++id) {
        if (v.is_eog(id)) { x->eog.push_back(id); continue; }
        const std::string& piece = v.token_piece(id);
        if (piece.empty() || piece[0] == 0) continue;

        const auto dec = decode_utf8(piece, PartialUtf8{});
        x->off[(size_t)id]  = (uint32_t)x->cps.size();
        x->tail[(size_t)id] = dec.second;
        x->cps.insert(x->cps.end(), dec.first.begin(), dec.first.end());
        ents.push_back({ x->off[(size_t)id], dec.second, id });

        /* Leading continuation bytes and their bits. Like decode_utf8, the piece is read as a C
         * string, so a NUL ends it: continuation bytes followed by a NUL count as a short piece. */
        const size_t len = ::strnlen(piece.c_str(), piece.size());
        size_t k = 0;
        uint32_t bits = 0;
        while (k < len && ((uint8_t)piece[k] >> 6) == 2 && k < 3) {
            bits = (bits << 6) | ((uint8_t)piece[k] & 0x3Fu);
            ++k;
        }
        if (k == 0) continue;
        const bool more_cont = k < len && ((uint8_t)piece[k] >> 6) == 2;
        for (int r = 1; r <= 3; ++r) {
            VocabIndex::Resume& R = x->resume[r - 1];
            if ((int)k == r && !more_cont) {
                /* Finishes the character with exactly r bytes; the rest decodes from a clean
                 * start, and an invalid rest disqualifies the piece. */
                const auto rest = decode_utf8(piece.substr((size_t)r), PartialUtf8{});
                if (rest.second.n_remain < 0) continue;
                const uint32_t o = (uint32_t)R.cps.size();
                R.cps.push_back(bits + 1);
                R.cps.insert(R.cps.end(), rest.first.begin(), rest.first.end());
                rents[r - 1].push_back({ o, rest.second, id });
            } else if ((int)k < r && k == len) {
                R.shorts.push_back({ bits, (int32_t)k, id });
            }
        }
    }

    trie_build(x->trie, x->cps.data(), ents, x->n_tokens);
    for (uint32_t k = x->trie.node[0].tbeg; k < x->trie.node[1].tbeg; ++k)
        if (x->trie.ttail[k].n_remain == 0) x->blank.push_back(x->trie.tid[k]);
    for (int r = 0; r < 3; ++r)
        trie_build(x->resume[r].trie, x->resume[r].cps.data(), rents[r], x->n_tokens);

    RAD_DEBUG("gbnf: vocabulary index over %d tokens: trie of %zu nodes, depth %u; %zu/%zu/%zu "
              "pieces resume a split character; %.1f MiB, built in %.0f ms", x->n_tokens,
              x->trie.n_nodes(), x->trie.depth, x->resume[0].trie.lex.size(),
              x->resume[1].trie.lex.size(), x->resume[2].trie.lex.size(),
              (double)x->bytes() / (1024.0 * 1024.0), ms_since(t0));
    return x;
}

/* ============================================================ the stack-set automaton

 * A stack set is one state of the subset construction: walking the set gives the same answer as
 * walking each stack and OR-ing, since admissibility is the existence of a path. States are
 * interned by value, so a set reached twice is one state.
 *
 * A stack's successor does not depend on the code point: stepping a char-range top pops it and
 * pushes what follows (`match_char(top, 0).second`); the code point only decides whether the
 * stack steps. So each stack's successor closure is computed once per state, and a transition
 * depends only on which stacks fire, which is constant over the intervals the tops' character
 * classes cut. The successor is kept per interval. A state holds:
 *
 *   ctop     the char-range tops, which step over code points and test a carried partial;
 *   ttop     the token tops (`<[id]>`, `!<[id]>`), which match a token's id wherever reached;
 *   has_empty whether the set holds the empty stack: a completed parse, or in a key's walk the
 *            bottom marker, meaning the key has been left.
 *
 * States live in a deque, so a reference to one survives interning another. */
class SetDfa {
public:
    struct St {
        GbnfStacks                      stacks;
        std::vector<const GbnfElement*> ctop;
        std::vector<uint32_t>           cstk;       /* ctop[i] is stacks[cstk[i]]'s top */
        std::vector<const GbnfElement*> ttop;
        bool                            has_empty = false;
        size_t                          elems = 0;
        uint64_t                        match_cost = 0;   /* one match_char over every ctop */

        bool                            parted = false;
        std::vector<uint32_t>           lo;         /* interval starts, lo[0] == 0 */
        std::vector<int32_t>            next;       /* successor per interval */
        std::vector<int32_t>            succ;       /* per ctop: the closure after its top */
        std::vector<std::pair<uint64_t, int32_t>> fired;   /* firing pattern -> successor */
    };

    explicit SetDfa(const GbnfRuleSet* rules) : rules_(rules) {}

    /* Set the operation every step is charged to, and the mark its growth is measured from. */
    void begin(OpWork* w) { work_ = w; walk_base_ = elems_; }

    size_t elems() const { return elems_; }
    size_t n_states() const { return st_.size(); }
    const St& st(int32_t sid) const { return st_[(size_t)sid]; }

    /* Canonicalise a stack set and return its state, or -1 for the empty (dead) set, so callers
     * can prune on the sign. */
    int32_t intern(GbnfStacks s) {
        if (s.empty()) return -1;
        /* A sort is n log n stack comparisons, each as long as the common prefix; charged as every
         * element once per sort level, so a set too wide to sort within the bound is refused. Sets
         * built in canonical order only pay the adjacent checks that establish it. */
        size_t elems = 0;
        bool sorted = true;
        for (size_t i = 0; i < s.size(); ++i) {
            elems += s[i].size() + 1;
            if (sorted && i && s[i] < s[i - 1]) sorted = false;
        }
        work_->add(elems * (sorted ? 1 : 1 + (uint64_t)std::bit_width(s.size())));
        if (!sorted) std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        return intern_sorted(std::move(s));
    }

    /* The same for a set already sorted and without duplicates. */
    int32_t intern_sorted(GbnfStacks s) {
        std::vector<const GbnfStack*> ps;
        ps.reserve(s.size());
        for (const GbnfStack& k : s) ps.push_back(&k);
        return intern_ptrs(ps, &s);
    }

    /* The same, from pointers to the stacks. They are copied only for a new state, or moved out
     * of `owned` when the caller passes its own vector. */
    int32_t intern_ptrs(const std::vector<const GbnfStack*>& ps, GbnfStacks* owned = nullptr) {
        if (ps.empty()) return -1;
        uint64_t h = 1469598103934665603ull;
        size_t   elems = 0;
        for (const GbnfStack* k : ps) {
            h ^= stack_hash(*k);
            h *= 1099511628211ull;
            elems += k->size() + 1;
        }
        work_->add(elems);
        auto range = ids_.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            const GbnfStacks& have = st_[(size_t)it->second].stacks;
            work_->add(elems);
            if (have.size() != ps.size()) continue;
            size_t i = 0;
            while (i < ps.size() && have[i] == *ps[i]) ++i;
            if (i == ps.size()) return it->second;
        }

        if (ps.size() > kMaxStacks || elems > kMaxSetElems) throw_set_limit();
        work_->add(elems + ps.size() * kStackCost);
        elems_ += elems;
        if (elems_ - walk_base_ > kMaxDfaWalkElems)
            throw GbnfLimit("one operation on this grammar discovers more than " +
                            std::to_string(kMaxDfaWalkElems) + " stack elements of new states");

        GbnfStacks s;
        if (owned) {
            s = std::move(*owned);
        } else {
            s.reserve(ps.size());
            for (const GbnfStack* k : ps) s.push_back(*k);
        }
        const int32_t id = (int32_t)st_.size();
        st_.emplace_back();
        St& n = st_.back();
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i].empty()) { n.has_empty = true; continue; }
            const GbnfElement* top = s[i].back();
            if (is_token_elem(top)) {
                n.ttop.push_back(top);
            } else {
                n.ctop.push_back(top);
                n.cstk.push_back((uint32_t)i);
                n.match_cost += char_range_len(top);
            }
        }
        n.elems  = elems;
        n.stacks = std::move(s);
        ids_.emplace(h, id);
        return id;
    }

    /* advance_stack(s) as a state, memoised on the stack: many stacks, in one state or many,
     * close to the same place (every matched alternative of a wide rule returns to one
     * continuation). */
    int32_t closure(const GbnfStack& s) {
        auto it = closure_ids_.find(s);
        if (it != closure_ids_.end()) return it->second;
        GbnfStacks out;
        SetBuild   b(*work_);
        advance_stack(*rules_, s, out, b);
        const int32_t r = intern(std::move(out));
        elems_ += s.size() + 1;
        closure_ids_.emplace(s, r);
        return r;
    }

    /* The state whose stacks are the union of these states' stacks. Memoised on the input
     * states, since the same unions recur. Each state's stacks are sorted, so the union is a
     * balanced tree of pairwise pointer merges (each stack compared once per level, not n log n)
     * and stacks are copied only for a new state. Bounds are checked after every merge, so an
     * oversized union is refused before it is built. */
    int32_t unite(std::vector<int32_t>& ids) {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        while (!ids.empty() && ids[0] < 0) ids.erase(ids.begin());
        if (ids.empty()) return -1;
        if (ids.size() == 1) return ids[0];

        work_->add(ids.size() * 2);
        auto hit = unions_.find(ids);
        if (hit != unions_.end()) return hit->second;

        auto less = [](const GbnfStack* a, const GbnfStack* b) { return *a < *b; };
        std::vector<std::vector<const GbnfStack*>> parts(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            const St& S = st_[(size_t)ids[i]];
            work_->add(S.stacks.size());
            parts[i].reserve(S.stacks.size());
            for (const GbnfStack& k : S.stacks) parts[i].push_back(&k);
        }
        while (parts.size() > 1) {
            std::vector<std::vector<const GbnfStack*>> next;
            next.reserve((parts.size() + 1) / 2);
            for (size_t i = 0; i + 1 < parts.size(); i += 2) {
                std::vector<const GbnfStack*> m;
                m.reserve(parts[i].size() + parts[i + 1].size());
                size_t elems = 0;
                for (const GbnfStack* k : parts[i]) elems += k->size() + 1;
                for (const GbnfStack* k : parts[i + 1]) elems += k->size() + 1;
                work_->add(elems);
                std::set_union(parts[i].begin(), parts[i].end(), parts[i + 1].begin(),
                               parts[i + 1].end(), std::back_inserter(m), less);
                if (m.size() > kMaxStacks) throw_set_limit();
                next.push_back(std::move(m));
            }
            if (parts.size() % 2) next.push_back(std::move(parts.back()));
            parts.swap(next);
        }
        const int32_t r = intern_ptrs(parts[0]);
        elems_ += ids.size() + 1;
        unions_.emplace(ids, r);
        return r;
    }

    /* The state's intervals, built on first use from every boundary of every char-range top, so
     * each top matches either all or none of the code points in an interval. */
    const St& parted(int32_t sid) {
        St& S = st_[(size_t)sid];
        if (S.parted) return S;
        std::vector<uint32_t>& b = bounds_;
        b.clear();
        b.push_back(0);
        for (const GbnfElement* pos : S.ctop) {
            do {
                if (pos[1].type == GRE_CHAR_RNG_UPPER) {
                    b.push_back(pos->value);
                    if (pos[1].value != kNone) b.push_back(pos[1].value + 1);
                    pos += 2;
                } else if (pos->type == GRE_CHAR_ANY) {
                    pos += 1;
                } else {
                    b.push_back(pos->value);
                    if (pos->value != kNone) b.push_back(pos->value + 1);
                    pos += 1;
                }
            } while (pos->type == GRE_CHAR_ALT);
        }
        work_->add(b.size() * 4 + S.match_cost);
        std::sort(b.begin(), b.end());
        b.erase(std::unique(b.begin(), b.end()), b.end());
        S.lo = b;
        S.next.assign(b.size(), kUnbuilt);
        S.succ.assign(S.ctop.size(), kUnbuilt);
        S.parted = true;
        return S;
    }

    /* The successor over interval i of a parted state, computed on first use. Also memoised by
     * firing pattern, since many intervals fire the same stacks (in a JSON string, everything but
     * a quote or backslash). */
    int32_t next(int32_t sid, size_t i) {
        St& S = st_[(size_t)sid];
        if (S.next[i] != kUnbuilt) return S.next[i];
        const uint32_t cp = S.lo[i];
        work_->add(S.match_cost + 4);

        int32_t r;
        if (S.ctop.size() <= 64) {
            uint64_t fire = 0;
            for (size_t j = 0; j < S.ctop.size(); ++j)
                if (match_char(S.ctop[j], cp).first) fire |= 1ull << j;
            r = -1;
            bool found = fire == 0;
            for (const auto& e : S.fired)
                if (e.first == fire) { r = e.second; found = true; break; }
            if (!found) {
                std::vector<int32_t> ids;
                for (uint64_t m = fire; m; m &= m - 1)
                    ids.push_back(succ_of(sid, (size_t)__builtin_ctzll(m)));
                r = unite(ids);
                st_[(size_t)sid].fired.push_back({ fire, r });
            }
        } else {
            std::vector<int32_t> ids;
            for (size_t j = 0; j < S.ctop.size(); ++j)
                if (match_char(S.ctop[j], cp).first) ids.push_back(succ_of(sid, j));
            r = unite(ids);
        }
        st_[(size_t)sid].next[i] = r;
        return r;
    }

private:
    /* What follows ctop j once it has matched: pop the range, push the element after it, close. */
    int32_t succ_of(int32_t sid, size_t j) {
        {
            const St& S = st_[(size_t)sid];
            if (S.succ[j] != kUnbuilt) return S.succ[j];
        }
        GbnfStack sa;
        {
            const GbnfStack&   s     = st_[(size_t)sid].stacks[st_[(size_t)sid].cstk[j]];
            const GbnfElement* after = match_char(s.back(), 0).second;
            sa.assign(s.begin(), s.end() - 1);
            if (!is_end_of_sequence(after)) sa.push_back(after);
        }
        const int32_t r = closure(sa);
        st_[(size_t)sid].succ[j] = r;
        return r;
    }

    struct StackHashFn {
        size_t operator()(const GbnfStack& s) const { return (size_t)stack_hash(s); }
    };
    struct IdsHashFn {
        size_t operator()(const std::vector<int32_t>& v) const {
            uint64_t h = 1469598103934665603ull;
            for (int32_t x : v) { h ^= (uint32_t)x; h *= 1099511628211ull; }
            return (size_t)h;
        }
    };

    const GbnfRuleSet*                                   rules_ = nullptr;
    OpWork*                                              work_  = nullptr;
    std::deque<St>                                       st_;
    std::unordered_multimap<uint64_t, int32_t>           ids_;
    std::unordered_map<GbnfStack, int32_t, StackHashFn>  closure_ids_;
    std::unordered_map<std::vector<int32_t>, int32_t, IdsHashFn> unions_;
    std::vector<uint32_t>                                bounds_;
    size_t                                               elems_ = 0, walk_base_ = 0;
};

/* ============================================================ the walk

 * Descend a trie breadth-first with one automaton state per live node, setting a bit for every
 * token the state admits there (the candidate walk's rule, inverted):
 *
 *   ending here         admitted iff nothing is carried out, or some char-range top could take
 *                       some completion of what is;
 *   below, by id        a token top admits the token with its id (`!<[id]>` every other one)
 *                       among those with text left, wherever it is reached;
 *   below, by text      whatever the successor over the next code point admits.
 *
 * In a key's walk (`exits`), the empty stack is the bottom marker: a node whose state holds it is
 * where the key is left, and everything under it is recorded for the remainder. */
struct Walk {
    const TokenTrie* trie = nullptr;
    uint32_t*        bits = nullptr;

    /* A resume trie's root edges hold continuation bits plus one; the carried bits are OR'ed in.
     * `skip_first` leaves the root's first child out of the walk, for fill_resume to serve. */
    bool             resume = false;
    uint32_t         root_base = 0;
    bool             skip_first = false;

    bool                                        exits = false;
    std::vector<std::pair<uint32_t, uint32_t>>* frontier = nullptr;   /* (node, depth) */
    uint64_t                                    frontier_tokens = 0;
    uint64_t                                    frontier_limit = UINT64_MAX;
    bool                                        aborted = false;

    /* Admitted ids in first-set order, up to `list_cap`; past it only the bits are kept. */
    std::vector<int32_t>* list = nullptr;
    size_t                list_cap = 0;
    bool                  list_full = false;
};

struct WalkScratch {
    /* A live (node, state) pair, with the state itself: states live in a deque (references
     * survive interning), and carrying the pointer saves re-indexing it per visit. The id is
     * still needed to build a successor on a memo miss. */
    struct Live {
        uint32_t          n;
        int32_t           s;
        const SetDfa::St* S;
    };
    std::vector<Live> cur, nxt;
};

static inline void admit(Walk& w, int32_t id) {
    const uint32_t word = (uint32_t)id >> 5, bit = 1u << ((uint32_t)id & 31u);
    if (w.bits[word] & bit) return;
    w.bits[word] |= bit;
    if (w.list && !w.list_full) {
        if (w.list->size() < w.list_cap) w.list->push_back(id);
        else w.list_full = true;
    }
}

/* What a visit needs of its state. Read once per run of children sharing an interval (and so a
 * successor), not per child: a permissive walk visits every vocabulary trie node, where the state
 * lookup is a measurable part of a visit. */
struct VisitState {
    const SetDfa::St* S = nullptr;
    bool              expand = false;    /* it has char-range tops, so children can be stepped */
    bool              tokens = false;    /* it has token tops */
    bool              exit   = false;    /* a key's walk, and the state holds the bottom marker */
};

static inline VisitState visit_state(const SetDfa& dfa, const Walk& w, int32_t s) {
    VisitState v;
    v.S      = &dfa.st(s);
    v.expand = !v.S->ctop.empty();
    v.tokens = !v.S->ttop.empty();
    v.exit   = w.exits && v.S->has_empty;
    return v;
}

/* Tokens ending at a live node: admitted unless they carry out a partial character that no
 * char-range top can finish. */
static void visit_terminals(Walk& w, OpWork& work, const VisitState& vs, uint32_t t0,
                            uint32_t t1) {
    const TokenTrie& T = *w.trie;
    work.add(t1 - t0);
    for (uint32_t k = t0; k < t1; ++k) {
        const PartialUtf8 tail = T.ttail[k];
        bool ok = tail.n_remain == 0;
        if (!ok) {
            work.add(vs.S->match_cost);
            for (const GbnfElement* top : vs.S->ctop)
                if (match_partial_char(top, tail)) { ok = true; break; }
        }
        if (ok) admit(w, T.tid[k]);
    }
}

/* What a token top admits from below node n; `n_end` of n's subtree tokens end at n itself. */
static __attribute__((noinline)) void visit_token_tops(Walk& w, OpWork& work,
                                                       const VisitState& vs, uint32_t n,
                                                       uint32_t n_end) {
    const TokenTrie& T = *w.trie;
    uint32_t a = T.lo[n] + n_end;                                /* strictly below n */
    const uint32_t b = T.hi[n];
    if (n == 0 && w.skip_first && T.node[0].n_child) a = std::max(a, T.hi[T.node[0].first]);
    if (a >= b) return;
    for (const GbnfElement* top : vs.S->ttop) {
        const uint32_t v = top->value;
        if (top->type == GRE_TOKEN) {
            if (!T.pos.empty()) {
                if (v < T.pos.size() && T.pos[v] != kNone && T.pos[v] >= a && T.pos[v] < b)
                    admit(w, (int32_t)v);
            } else {
                work.add(b - a);
                for (uint32_t i = a; i < b; ++i)
                    if ((uint32_t)T.lex[i] == v) { admit(w, (int32_t)v); break; }
            }
        } else {
            work.add(b - a);
            for (uint32_t i = a; i < b; ++i)
                if ((uint32_t)T.lex[i] != v) admit(w, T.lex[i]);
        }
    }
}

/* Visit node n: admit what ends here and what a token top takes from below, and record an exit.
 * Returns whether the node's children should be stepped. */
static inline __attribute__((always_inline)) bool walk_visit(Walk& w, OpWork& work,
                                                             const TrieNode* nd, uint32_t n,
                                                             const VisitState& vs,
                                                             uint32_t depth) {
    const uint32_t t0 = nd[n].tbeg, t1 = nd[n + 1].tbeg;
    if (t0 != t1) visit_terminals(w, work, vs, t0, t1);
    if (vs.tokens) visit_token_tops(w, work, vs, n, t1 - t0);
    if (vs.exit) {
        const TokenTrie& T = *w.trie;
        w.frontier->push_back({ n, depth });
        w.frontier_tokens += T.hi[n] - T.lo[n];
        if (w.frontier_tokens > w.frontier_limit) { w.aborted = true; return false; }
    }
    return vs.expand && nd[n].n_child != 0;
}

static void walk(SetDfa& dfa, int32_t start, Walk& w, OpWork& work, WalkScratch& sc) {
    if (start < 0) return;
    const TrieNode* nd = w.trie->node.data();
    auto& cur = sc.cur;
    auto& nxt = sc.nxt;
    cur.clear();
    {
        const VisitState vs = visit_state(dfa, w, start);
        if (walk_visit(w, work, nd, 0, vs, 0)) cur.push_back({ 0u, start, vs.S });
        if (w.aborted) return;
    }

    for (uint32_t depth = 1; !cur.empty(); ++depth) {
        nxt.clear();
        const bool     remap = w.resume && depth == 1;
        const uint32_t base  = w.root_base;
        auto cp_of = [nd, remap, base](uint32_t c) -> uint64_t {
            return remap ? (uint64_t)(base | (nd[c].cp - 1)) : (uint64_t)nd[c].cp;
        };
        for (size_t e = 0; e < cur.size(); ++e) {
            const uint32_t n = cur[e].n;
            const int32_t  s = cur[e].s;
            /* parted() builds the intervals on first use; the walk visits states far oftener
             * than it parts them, so the built check is inlined (likewise next(), below, whose
             * memo S.next is read directly). The deque keeps S valid across the interning a
             * miss does, and S.lo/S.next are assigned once, at parting. */
            const SetDfa::St& S = *cur[e].S;
            if (!S.parted) dfa.parted(s);
            const uint32_t* lo = S.lo.data();
            const size_t    m  = S.lo.size();
            uint32_t        c  = nd[n].first + (depth == 1 && w.skip_first ? 1u : 0u);
            const uint32_t  c1 = nd[n].first + nd[n].n_child;
            size_t          i  = 0;

            /* One run of children per interval. Children are sorted, so the interval only moves
             * forward (linear among few intervals, binary search among many). A dead interval is
             * skipped by binary search over the children; a live one gives all its children the
             * same state. */
            while (c < c1) {
                const uint64_t cp = cp_of(c);
                if (m <= 16) {
                    while (i + 1 < m && lo[i + 1] <= cp) ++i;
                } else {
                    i = (size_t)(std::upper_bound(lo + i, lo + m, cp,
                                     [](uint64_t x, uint32_t y) { return x < y; }) - lo) - 1;
                }
                const uint64_t hi = i + 1 < m ? (uint64_t)lo[i + 1] : (uint64_t)1 << 33;
                int32_t        ns = S.next[i];
                if (ns == kUnbuilt) ns = dfa.next(s, i);
                if (ns < 0) {
                    uint32_t a = c, b = c1;
                    while (a < b) {
                        const uint32_t mid = a + (b - a) / 2;
                        if (cp_of(mid) < hi) a = mid + 1; else b = mid;
                    }
                    work.add(8);
                    c = a;
                    continue;
                }
                const VisitState vs = visit_state(dfa, w, ns);
                const uint32_t   c0 = c;
                for (; c < c1 && cp_of(c) < hi; ++c)
                    if (walk_visit(w, work, nd, c, vs, depth)) nxt.push_back({ c, ns, vs.S });
                work.add(8 + (c - c0));
                if (w.aborted) return;
            }
        }
        cur.swap(nxt);
    }
}

/* ============================================================ the plan */

/* A key: a suffix of stack entries by class, bottom first. Held as a tree grown downward (a node
 * is its bottom class plus the node for the entries above it); a stack is looked up from its top,
 * one entry down per level, until a node that is not extended.
 *
 * `mask` is the accepted set: for a final node, used with any context; for an extended node that
 * can be a whole stack, used only when the stack ends there. `rem` is the remainder trie, walked
 * from the context's state when there is a context. */
struct KeyNode {
    uint32_t cls      = 0;
    int32_t  up       = -1;
    uint16_t depth    = 1;
    bool     extended = false;
    bool     can_end  = false;
    int32_t  mask     = -1;
    int32_t  rem      = -1;
    std::vector<std::pair<uint32_t, int32_t>> kids;     /* by class, sorted */
};

/* An accepted set, as a bitmask when that is smaller than listing it. */
struct KeyMask {
    bool                  dense = false;
    std::vector<uint32_t> v;      /* dense: the words; otherwise the token ids, ascending */
};

struct MaskPlan {
    const GbnfRuleSet*                rules = nullptr;
    std::shared_ptr<const VocabIndex> vx;
    std::vector<uint32_t>             cls;       /* element index -> class; 0 where no entry */
    std::vector<int32_t>              top_key;   /* class -> its depth-one key, or -1 */
    std::vector<KeyNode>              keys;
    std::vector<KeyMask>              masks;
    std::vector<TokenTrie>            rems;
    PlanStats                         stats;

    /* Run-time memo: the automaton holding the states contexts close to and remainders step
     * into. It grows as states are discovered; each is a fact about the grammar, so one lock
     * covers it and every request and rank shares it. */
    mutable std::mutex              mu;
    mutable std::unique_ptr<SetDfa> dfa;
    mutable WalkScratch             ws;
    mutable std::vector<uint32_t>   mark;
    mutable uint32_t                epoch = 0;
    mutable std::vector<int32_t>    slot;
    struct Group { int32_t key; std::vector<int32_t> ctx; };
    mutable std::vector<Group>      groups;
    mutable GbnfStack               ctx;
    mutable std::vector<const GbnfStack*> step_ptrs;
    mutable uint64_t                dfa_gen = 0;   /* which automaton `dfa` is; see plan_step */
};

void MaskPlanDeleter::operator()(MaskPlan* p) const { delete p; }

namespace {

/* Everything the build derives from the rules before it walks anything. */
struct Grammar {
    const GbnfRuleSet& R;
    uint32_t           root;
    std::vector<uint32_t> erule;                 /* element index -> rule */
    std::vector<char>     reach;                 /* rule reachable from the root */
    std::vector<char>     bottom_ok;             /* rule can be the frame at a stack's bottom */
    std::vector<std::vector<uint32_t>> rev_tail;   /* rule -> rules that end an alternative in it */
    std::vector<std::vector<uint32_t>> refs_to;    /* rule -> reference elements to it */
    std::vector<uint32_t> rep;                   /* class -> a representative element */
    std::vector<std::vector<uint32_t>> tops;       /* class -> its terminal elements, reachable */
    std::vector<std::vector<uint32_t>> below;      /* rule -> entries that can sit below it */
    std::vector<char>     below_done;
    std::vector<uint32_t> stamp;                 /* rule -> the below search that last saw it */
    uint32_t              stamp_now = 0;

    explicit Grammar(const GbnfRuleSet& r, uint32_t rt) : R(r), root(rt) {}

    uint32_t next_unit(uint32_t idx) const {
        const GbnfElement* e = &R.elems[idx];
        return is_char_elem(e) ? R.index(match_char(e, 0).second) : idx + 1;
    }
};

/* Merge identical rules. Generated grammars repeat clauses (every tool's undeclared-parameter
 * clause `("<parameter=" ([^>] | ">" [^\n])* ">\n")`), and the parser gives each copy its own
 * rules, which would otherwise mean a key and a vocabulary walk per copy.
 *
 * Rules are hash-consed on their bodies, with references replaced by the target's canonical id,
 * bottom-up over the reference graph's SCCs. A self-referencing rule (a repetition) is hashed
 * with its self-references marked, so identical repetitions merge. Larger cycles are left
 * distinct (rare in generated grammars, and correct unmerged). Returns each rule's canonical id;
 * merged rules share one, and no id is a rule index. */
static std::vector<uint32_t> canonical_rules(const Grammar& G, const std::vector<char>& nullable,
                                             uint32_t max_cps, OpWork& work) {
    const GbnfRuleSet& R = G.R;
    const size_t NR = R.n_rules(), NE = R.elems.size();
    auto rule_end = [&](size_t r) { return r + 1 < NR ? R.start[r + 1] : (uint32_t)NE; };

    /* Tarjan's SCCs over references, iteratively. Components come out in reverse topological
     * order, so every rule a component refers to outside itself is already canonical. */
    std::vector<int32_t>  idx(NR, -1), low(NR, 0);
    std::vector<char>     on(NR, 0);
    std::vector<uint32_t> stk;
    std::vector<std::vector<uint32_t>> comps;
    struct Frame { uint32_t r; uint32_t e; };
    std::vector<Frame> call;
    int32_t counter = 0;
    for (size_t root = 0; root < NR; ++root) {
        if (idx[root] >= 0) continue;
        idx[root] = low[root] = counter++;
        stk.push_back((uint32_t)root);
        on[root] = 1;
        call.push_back({ (uint32_t)root, R.start[root] });
        while (!call.empty()) {
            Frame& f = call.back();
            const uint32_t v = f.r;
            if (f.e < rule_end(v)) {
                const GbnfElement& el = R.elems[f.e++];
                if (el.type != GRE_RULE_REF) continue;
                const uint32_t w = el.value;
                if (idx[w] < 0) {
                    idx[w] = low[w] = counter++;
                    stk.push_back(w);
                    on[w] = 1;
                    call.push_back({ w, R.start[w] });
                } else if (on[w]) {
                    low[v] = std::min(low[v], idx[w]);
                }
                continue;
            }
            call.pop_back();
            if (!call.empty()) low[call.back().r] = std::min(low[call.back().r], low[v]);
            if (low[v] == idx[v]) {
                std::vector<uint32_t> c;
                uint32_t w;
                do {
                    w = stk.back();
                    stk.pop_back();
                    on[w] = 0;
                    c.push_back(w);
                } while (w != v);
                comps.push_back(std::move(c));
            }
        }
    }
    work.add(NE + NR);

    constexpr uint32_t kSelf = 0xFFFFFFFEu;
    std::vector<uint32_t> canon(NR, kNone);
    std::unordered_map<std::string, uint32_t> ids;
    std::string sig;
    uint32_t unique = 0x80000000u;        /* ids no body hashes to: rules left unmerged */
    auto put = [](std::string& out, const GbnfElement& el, uint32_t v) {
        out.push_back((char)el.type);
        out.append((const char*)&v, 4);
    };

    /* A bounded repetition looks like a star to any token. The parser writes x{0,n} as a chain
     * `S(k) ::= x S(k-1) |` down to `S(1) ::= x |`, one rule per link. A token holds at most
     * `max_cps` code points, so if x always consumes one, a link with more repetitions left than
     * that is indistinguishable from `x*` and gets the star's canonical id. `depth` and `operand`
     * hold, per link, its remaining repetitions and the signature of what it repeats; a star
     * (self-referencing) is not a link. */
    std::vector<uint32_t>    depth(NR, 0);
    std::vector<std::string> operand(NR);
    auto consumes = [&](uint32_t a, uint32_t e) {
        for (uint32_t i = a; i < e; i = G.next_unit(i)) {
            const GbnfElement& el = R.elems[i];
            if (is_char_elem(&el)) return true;
            if (el.type == GRE_RULE_REF && !nullable[el.value]) return true;
        }
        return false;
    };
    auto has_token_elem = [&](uint32_t a, uint32_t e) {
        for (uint32_t i = a; i < e; ++i)
            if (is_token_elem(&R.elems[i])) return true;
        return false;
    };

    for (const auto& c : comps) {
        if (c.size() > 1) {
            for (uint32_t r : c) canon[r] = unique++;
            continue;
        }
        const uint32_t r = c[0];
        sig.clear();
        for (uint32_t i = R.start[r]; i < rule_end(r); ++i) {
            const GbnfElement& el = R.elems[i];
            put(sig, el, el.type != GRE_RULE_REF ? el.value
                       : el.value == r            ? kSelf
                                                  : canon[el.value]);
        }
        work.add(sig.size());

        /* A link: exactly `x [S(k-1)] | <empty>`. */
        const uint32_t a = R.start[r];
        uint32_t alt_end = a;
        while (!is_end_of_sequence(&R.elems[alt_end])) ++alt_end;
        const bool two_alts = R.elems[alt_end].type == GRE_ALT &&
                              R.elems[alt_end + 1].type == GRE_END;
        const bool self_ref = alt_end > a && R.elems[alt_end - 1].type == GRE_RULE_REF &&
                              R.elems[alt_end - 1].value == r;
        if (two_alts && alt_end > a && !self_ref && !has_token_elem(a, alt_end)) {
            const GbnfElement& last = R.elems[alt_end - 1];
            uint32_t op_end = alt_end, below = kNone;
            if (last.type == GRE_RULE_REF && depth[last.value] > 0) {
                op_end = alt_end - 1;
                below  = last.value;
            }
            std::string op;
            for (uint32_t i = a; i < op_end; ++i) {
                const GbnfElement& el = R.elems[i];
                put(op, el, el.type == GRE_RULE_REF ? canon[el.value] : el.value);
            }
            if (op_end > a && consumes(a, op_end) &&
                (below == kNone || operand[below] == op)) {
                depth[r]   = below == kNone ? 1 : depth[below] + 1;
                operand[r] = op;
                if (depth[r] > max_cps) {
                    std::string star = op;
                    const GbnfElement self{ GRE_RULE_REF, 0 };
                    put(star, self, kSelf);
                    put(star, GbnfElement{ GRE_ALT, 0 }, 0);
                    put(star, GbnfElement{ GRE_END, 0 }, 0);
                    sig = std::move(star);
                }
            }
        }
        canon[r] = ids.emplace(sig, (uint32_t)ids.size()).first->second;
    }
    return canon;
}

/* Which rules can match the empty string: a fixed point, as in find_left_recursion. */
static std::vector<char> nullable_rules(const GbnfRuleSet& R, OpWork& work) {
    const size_t NR = R.n_rules(), NE = R.elems.size();
    std::vector<char>     nullable(NR, 0);
    std::vector<uint32_t> alt_owner, alt_pending, todo;
    std::vector<std::vector<uint32_t>> waiting(NR);
    for (size_t r = 0; r < NR; ++r) {
        uint32_t i = R.start[r];
        for (;;) {
            bool     terminal = false;
            uint32_t refs = 0;
            uint32_t j = i;
            for (; !is_end_of_sequence(&R.elems[j]); ++j) {
                if (R.elems[j].type == GRE_RULE_REF) ++refs;
                else terminal = true;
            }
            if (!terminal) {
                if (refs == 0) {
                    if (!nullable[r]) { nullable[r] = 1; todo.push_back((uint32_t)r); }
                } else {
                    const uint32_t alt = (uint32_t)alt_owner.size();
                    alt_owner.push_back((uint32_t)r);
                    alt_pending.push_back(refs);
                    for (uint32_t k = i; k < j; ++k) waiting[R.elems[k].value].push_back(alt);
                }
            }
            if (R.elems[j].type == GRE_END) break;
            i = j + 1;
        }
    }
    while (!todo.empty()) {
        const uint32_t x = todo.back();
        todo.pop_back();
        for (uint32_t alt : waiting[x]) {
            if (--alt_pending[alt] != 0) continue;
            const uint32_t r = alt_owner[alt];
            if (!nullable[r]) { nullable[r] = 1; todo.push_back(r); }
        }
    }
    work.add(NE * 2 + NR);
    return nullable;
}

/* An entry's class is the rest of its alternative from it on, with rule references by canonical
 * rule, hash-consed backward from the end of each alternative. Entries of one class behave
 * identically anywhere. Class 0 is the end of an alternative and is never an entry.
 *
 * A run of one unit is capped like a repetition chain (canonical_rules): x{1000} is a thousand x
 * in a row, and an entry with more than `max_cps` of a code-point-consuming unit ahead of it
 * behaves, for every token, like one with one fewer, so it takes the next entry's class. */
static void compute_classes(Grammar& G, MaskPlan& P, const std::vector<char>& nullable,
                            uint32_t max_cps, OpWork& work) {
    const GbnfRuleSet& R = G.R;
    const size_t NE = R.elems.size();
    const std::vector<uint32_t> canon = canonical_rules(G, nullable, max_cps, work);
    P.cls.assign(NE, 0);
    G.rep.assign(1, 0);
    std::unordered_map<std::string, uint32_t> ids;
    std::vector<std::string> lead(1);      /* class -> its first unit's content */
    std::vector<uint32_t>    run(1, 0);    /* class -> how many of that unit lead it, capped */
    std::string key, unit;
    std::vector<uint32_t> units;

    for (size_t r = 0; r < R.n_rules(); ++r) {
        uint32_t idx = R.start[r];
        for (;;) {
            units.clear();
            while (!is_end_of_sequence(&R.elems[idx])) {
                units.push_back(idx);
                idx = G.next_unit(idx);
            }
            uint32_t nextc = 0;
            for (size_t u = units.size(); u-- > 0;) {
                unit.clear();
                for (uint32_t e = units[u], end = G.next_unit(units[u]); e < end; ++e) {
                    const GbnfElement& el = R.elems[e];
                    const uint32_t v = el.type == GRE_RULE_REF ? canon[el.value] : el.value;
                    unit.push_back((char)el.type);
                    unit.append((const char*)&v, 4);
                }
                const GbnfElement& ue = R.elems[units[u]];
                const bool consumes = is_char_elem(&ue) ||
                                      (ue.type == GRE_RULE_REF && !nullable[ue.value]);
                if (consumes && nextc != 0 && run[nextc] > max_cps && lead[nextc] == unit) {
                    P.cls[units[u]] = nextc;
                    continue;
                }
                key = unit;
                key.append((const char*)&nextc, 4);
                work.add(key.size());
                auto ins = ids.emplace(key, (uint32_t)G.rep.size());
                if (ins.second) {
                    G.rep.push_back(units[u]);
                    const uint32_t r = consumes && nextc != 0 && lead[nextc] == unit
                                           ? run[nextc] + 1 : (consumes ? 1u : 0u);
                    lead.push_back(unit);
                    run.push_back(r);
                }
                P.cls[units[u]] = ins.first->second;
                nextc = ins.first->second;
            }
            if (R.elems[idx].type == GRE_END) break;
            ++idx;
        }
    }
}

/* A blind rule can neither match anything nor be left (`r ::= r` is the shortest). A stack that
 * enters one closes to nothing, which upstream's candidate walk reads as admitting every token
 * while accepting any of them ends the parse. Suffix-based masks cannot reproduce that, and no
 * meaningful grammar depends on it, so a grammar that can reach such a rule is refused by name.
 *
 * A rule can pop past itself iff it is nullable; its closure yields a stack iff some element
 * reachable before the first non-nullable one is a terminal or a yielding rule. Both are least
 * fixed points computed with worklists in one linear pass. A rule doing neither is blind.
 * Returns a blind rule reachable from the root, or kNone. */
static uint32_t blind_rule(const Grammar& G, const std::vector<char>& nullable, OpWork& work) {
    const GbnfRuleSet& R = G.R;
    const size_t NR = R.n_rules();
    std::vector<char>                  yields(NR, 0);
    std::vector<std::vector<uint32_t>> waits(NR);     /* rule -> rules it would make yield */
    std::vector<uint32_t>              todo;
    for (size_t r = 0; r < NR; ++r) {
        uint32_t idx = R.start[r];
        for (;;) {
            bool open = true;
            for (; !is_end_of_sequence(&R.elems[idx]); idx = G.next_unit(idx)) {
                if (!open) continue;
                const GbnfElement& e = R.elems[idx];
                if (e.type == GRE_RULE_REF) {
                    waits[e.value].push_back((uint32_t)r);
                    open = nullable[e.value] != 0;
                } else {
                    if (!yields[r]) { yields[r] = 1; todo.push_back((uint32_t)r); }
                    open = false;
                }
            }
            if (R.elems[idx].type == GRE_END) break;
            ++idx;
        }
    }
    while (!todo.empty()) {
        const uint32_t t = todo.back();
        todo.pop_back();
        for (uint32_t r : waits[t])
            if (!yields[r]) { yields[r] = 1; todo.push_back(r); }
    }
    work.add(R.elems.size() * 2 + NR);
    for (size_t r = 0; r < NR; ++r)
        if (G.reach[r] && !yields[r] && !nullable[r]) return (uint32_t)r;
    return kNone;
}

static void analyse(Grammar& G, MaskPlan& P, OpWork& work) {
    const GbnfRuleSet& R = G.R;
    const size_t NE = R.elems.size(), NR = R.n_rules();

    G.erule.assign(NE, 0);
    for (size_t r = 0; r < NR; ++r) {
        const size_t end = r + 1 < NR ? R.start[r + 1] : NE;
        for (size_t i = R.start[r]; i < end; ++i) G.erule[i] = (uint32_t)r;
    }
    work.add(NE);

    /* Reachable rules. References inside unreachable rules must not count, or frames would get
     * predecessors no stack has. */
    G.reach.assign(NR, 0);
    std::vector<uint32_t> todo{ G.root };
    G.reach[G.root] = 1;
    while (!todo.empty()) {
        const uint32_t r = todo.back();
        todo.pop_back();
        const size_t end = r + 1 < NR ? R.start[r + 1] : NE;
        for (size_t i = R.start[r]; i < end; ++i) {
            const GbnfElement& e = R.elems[i];
            if (e.type == GRE_RULE_REF && !G.reach[e.value]) {
                G.reach[e.value] = 1;
                todo.push_back(e.value);
            }
        }
    }

    /* Tail references (a frame replaced rather than stacked on), and every reference by target. */
    G.rev_tail.assign(NR, {});
    G.refs_to.assign(NR, {});
    std::vector<std::vector<uint32_t>> tail_to(NR);
    for (size_t r = 0; r < NR; ++r) {
        if (!G.reach[r]) continue;
        uint32_t idx = R.start[r];
        for (;;) {
            uint32_t last = kNone;
            while (!is_end_of_sequence(&R.elems[idx])) {
                if (R.elems[idx].type == GRE_RULE_REF) G.refs_to[R.elems[idx].value].push_back(idx);
                last = idx;
                idx = G.next_unit(idx);
            }
            if (last != kNone && R.elems[last].type == GRE_RULE_REF) {
                tail_to[r].push_back(R.elems[last].value);
                G.rev_tail[R.elems[last].value].push_back((uint32_t)r);
            }
            if (R.elems[idx].type == GRE_END) break;
            ++idx;
        }
    }

    /* Frames a stack can have at its bottom: the root's, and whatever it hands over to by tail
     * reference. */
    G.bottom_ok.assign(NR, 0);
    todo.assign(1, G.root);
    G.bottom_ok[G.root] = 1;
    while (!todo.empty()) {
        const uint32_t r = todo.back();
        todo.pop_back();
        for (uint32_t t : tail_to[r])
            if (!G.bottom_ok[t]) { G.bottom_ok[t] = 1; todo.push_back(t); }
    }

    /* Terminals by class: every place a stack's top can be. */
    G.tops.assign(G.rep.size(), {});
    for (size_t r = 0; r < NR; ++r) {
        if (!G.reach[r]) continue;
        uint32_t idx = R.start[r];
        for (;;) {
            while (!is_end_of_sequence(&R.elems[idx])) {
                const GbnfElement* e = &R.elems[idx];
                if (is_char_elem(e) || is_token_elem(e)) G.tops[P.cls[idx]].push_back(idx);
                idx = G.next_unit(idx);
            }
            if (R.elems[idx].type == GRE_END) break;
            ++idx;
        }
    }
    work.add(NE * 4);

    G.below.assign(NR, {});
    G.below_done.assign(NR, 0);
    G.stamp.assign(NR, 0);
}

/* Entries that can sit directly below a frame of rule r: the element after every reference to r,
 * or to a rule that hands over to r by tail reference (where the frame below resumes).
 *
 * Kept by element, not class; a key's bottom entry carries the elements it can be. Classes merge
 * elements of different rules: the `space` ending `integer` and the `space` after `integer` in its
 * user are one class, so by class it could sit below itself without end, and a key would be
 * extended along that phantom chain until refused. By element, keys grow only as deep as real
 * stacks. */
static const std::vector<uint32_t>& below_of(Grammar& G, uint32_t r, OpWork& work) {
    if (G.below_done[r]) return G.below[r];
    std::vector<uint32_t>& out = G.below[r];
    std::vector<uint32_t>  todo{ r };
    const uint32_t mark = ++G.stamp_now;
    G.stamp[r] = mark;
    while (!todo.empty()) {
        const uint32_t b = todo.back();
        todo.pop_back();
        for (uint32_t q : G.refs_to[b]) {
            const uint32_t u = q + 1;
            if (!is_end_of_sequence(&G.R.elems[u])) out.push_back(u);
        }
        for (uint32_t t : G.rev_tail[b])
            if (G.stamp[t] != mark) { G.stamp[t] = mark; todo.push_back(t); }
        work.add(G.refs_to[b].size() + G.rev_tail[b].size() + 1);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    G.below_done[r] = 1;
    return out;
}

/* The entries that can sit directly below any of `pos`, as (class, element), sorted.
 *
 * One walk from all of `pos`'s rules, not below_of per rule. Repetition-chain links hand over by
 * tail reference, so climbing from one link visits every link above it, and one key's positions
 * can be all n links of `x{0,n}`; per rule that is n^2/2 steps. Walking from all rules at once
 * visits each rule once. A rule whose set is already known contributes it and is not climbed
 * past. A single-rule key (the common case) uses the below_of memo. */
static void preds_of(Grammar& G, const MaskPlan& P, const std::vector<uint32_t>& pos,
                     std::vector<std::pair<uint32_t, uint32_t>>* out, OpWork& work) {
    out->clear();
    const uint32_t first = pos.empty() ? kNone : G.erule[pos[0]];
    bool one_rule = true;
    for (uint32_t p : pos)
        if (G.erule[p] != first) { one_rule = false; break; }
    if (one_rule) {
        if (first != kNone)
            for (uint32_t u : below_of(G, first, work)) out->push_back({ P.cls[u], u });
    } else {
        const uint32_t mark = ++G.stamp_now;
        std::vector<uint32_t> todo;
        for (uint32_t p : pos) {
            const uint32_t r = G.erule[p];
            if (G.stamp[r] != mark) { G.stamp[r] = mark; todo.push_back(r); }
        }
        work.add(pos.size());
        while (!todo.empty()) {
            const uint32_t b = todo.back();
            todo.pop_back();
            if (G.below_done[b]) {
                for (uint32_t u : G.below[b]) out->push_back({ P.cls[u], u });
                work.add(G.below[b].size() + 1);
                continue;
            }
            for (uint32_t q : G.refs_to[b]) {
                const uint32_t u = q + 1;
                if (!is_end_of_sequence(&G.R.elems[u])) out->push_back({ P.cls[u], u });
            }
            for (uint32_t t : G.rev_tail[b])
                if (G.stamp[t] != mark) { G.stamp[t] = mark; todo.push_back(t); }
            work.add(G.refs_to[b].size() + G.rev_tail[b].size() + 1);
        }
    }
    std::sort(out->begin(), out->end());
    out->erase(std::unique(out->begin(), out->end()), out->end());
    work.add(out->size() + 1);
}

static bool can_end(const Grammar& G, const std::vector<uint32_t>& pos) {
    for (uint32_t p : pos)
        if (G.bottom_ok[G.erule[p]]) return true;
    return false;
}

static std::string rule_name_of(const std::vector<std::string>* names, uint32_t r) {
    if (names && r < names->size() && !(*names)[r].empty()) return "'" + (*names)[r] + "'";
    return "#" + std::to_string(r);
}

/* ============================================================ building the keys */

/* What one key's walk decided. A final key keeps its accepted set (valid for any context) and its
 * remainder trie. A non-final key is extended, and keeps an accepted set only if it can also be a
 * whole stack, where that set is the mask. */
struct Outcome {
    bool    final_ = false;
    int32_t mask   = -1;
    int32_t rem    = -1;      /* -1 when nothing leaves the key */
};

class KeyBuilder {
public:
    KeyBuilder(MaskPlan& P, const VocabIndex& X, SetDfa& dfa, OpWork& work)
        : P_(P), X_(X), dfa_(dfa), work_(work),
          n_words_((X.n_tokens + 31) / 32), bits_((size_t)n_words_, 0u) {}

    /* The outcome of a key whose local stack interns to s0: `exits` when something can sit below
     * it, `ends` when it can also be a whole stack. */
    Outcome outcome(int32_t s0, bool exits, bool ends) {
        Outcome o;
        Walk w = start_walk(exits);
        walk(dfa_, s0, w, work_, ws_);
        if (decide(w, o)) return o;
        if (!ends) {
            clear(w);
        } else if (!w.aborted) {
            o.mask = store(w);
        } else {
            /* The walk stopped early because the key will be extended; as a whole stack it needs
             * a full walk. */
            clear(w);
            Walk whole = start_walk(false);
            walk(dfa_, s0, whole, work_, ws_);
            o.mask = store(whole);
        }
        return o;
    }

private:
    Walk start_walk(bool exits) {
        list_.clear();
        frontier_.clear();
        Walk w;
        w.trie = &X_.trie;
        w.bits = bits_.data();
        w.list = &list_;
        w.list_cap = (size_t)n_words_;
        w.exits = exits;
        w.frontier = &frontier_;
        w.frontier_limit = kAbortRemTokens;
        return w;
    }

    /* Keep an accepted set: as a list if the walk listed it, else as a bitmask. Clears the scratch
     * bits either way. */
    int32_t store(Walk& w) {
        KeyMask m;
        if (!w.list_full) {
            m.v.assign(list_.begin(), list_.end());
            std::sort(m.v.begin(), m.v.end());
            for (uint32_t id : m.v) bits_[id >> 5] &= ~(1u << (id & 31u));
        } else {
            m.dense = true;
            m.v = bits_;
            std::fill(bits_.begin(), bits_.end(), 0u);
            ++P_.stats.dense;
        }
        P_.stats.bytes += m.v.size() * 4 + sizeof(KeyMask);
        P_.masks.push_back(std::move(m));
        return (int32_t)P_.masks.size() - 1;
    }

    void clear(Walk& w) {
        if (!w.list_full)
            for (int32_t id : list_) bits_[(uint32_t)id >> 5] &= ~(1u << ((uint32_t)id & 31u));
        else
            std::fill(bits_.begin(), bits_.end(), 0u);
    }

    /* Is the key final? If so its accepted set and remainder are kept and the scratch cleared;
     * if not, the walk's accepted set is left in the scratch for the caller. */
    bool decide(Walk& w, Outcome& o) {
        if (!w.exits) {
            o.final_ = true;
            o.mask = store(w);
            return true;
        }
        if (!w.aborted) {
            /* The remainder: tokens under each node where the key was left, from that depth on,
             * that are not already admitted. A token ending exactly there leaves an empty
             * remainder, kept only if it carries out a partial character. */
            rents_.clear();
            bool too_many = false;
            for (const auto& f : frontier_) {
                const uint32_t n = f.first, d = f.second;
                const uint32_t a = X_.trie.lo[n], b = X_.trie.hi[n];
                const uint32_t tend = X_.trie.node[n + 1].tbeg - X_.trie.node[n].tbeg;
                work_.add(b - a);
                for (uint32_t i = a; i < b; ++i) {
                    const int32_t id = X_.trie.lex[i];
                    if (test_bit(bits_.data(), id)) continue;
                    const PartialUtf8 tail = X_.tail[(size_t)id];
                    if (i < a + tend && tail.n_remain == 0) continue;
                    rents_.push_back({ X_.off[(size_t)id] + d, tail, id });
                }
                if (rents_.size() > kMaxRemEntries) { too_many = true; break; }
            }
            if (!too_many && rents_.empty()) {
                o.final_ = true;
                o.mask = store(w);
                return true;
            }
            if (!too_many) {
                TokenTrie t;
                trie_build(t, X_.cps.data(), rents_, 0);
                work_.add(rents_.size() * 16 + t.n_nodes() * 4);
                if (t.n_nodes() <= kMaxRemNodes) {
                    PlanStats& st = P_.stats;
                    st.rem_nodes += t.n_nodes();
                    st.max_rem = std::max(st.max_rem, t.n_nodes());
                    st.bytes += t.bytes();
                    P_.rems.push_back(std::move(t));
                    o.final_ = true;
                    o.rem = (int32_t)P_.rems.size() - 1;
                    o.mask = store(w);
                    return true;
                }
            }
        }
        return false;
    }

    MaskPlan&              P_;
    const VocabIndex&      X_;
    SetDfa&                dfa_;
    OpWork&                work_;
    int64_t                n_words_;
    std::vector<uint32_t>  bits_;
    std::vector<int32_t>   list_;
    std::vector<std::pair<uint32_t, uint32_t>> frontier_;
    std::vector<TrieEntry> rents_;
    WalkScratch            ws_;
};

}  /* namespace */

int plan_build(const GbnfRuleSet& R, uint32_t root, const std::shared_ptr<const VocabIndex>& vxp,
               const std::vector<std::string>* rule_names, MaskPlanPtr* out, PlanStats* stats,
               std::string* err) {
    const auto t0 = std::chrono::steady_clock::now();
    MaskPlanPtr pp(new MaskPlan());
    MaskPlan& P = *pp;
    P.rules = &R;
    P.vx    = vxp;
    const VocabIndex& X = *vxp;
    PlanStats& st = P.stats;

    OpWork work;
    work.limit = kMaxCompileWork;

    Grammar G(R, root);
    /* The longest token in code points: how far any mask can see into the grammar. */
    const uint32_t max_cps = X.trie.depth > 0 ? X.trie.depth - 1 : 0;
    const std::vector<char> nullable = nullable_rules(R, work);
    compute_classes(G, P, nullable, max_cps, work);
    analyse(G, P, work);
    st.classes = G.rep.size() - 1;

    const uint32_t blind = blind_rule(G, nullable, work);
    if (blind != kNone) {
        if (err)
            *err = "rule " + rule_name_of(rule_names, blind) + " can never match anything: every "
                   "way into it leads back into it before any character or its own end";
        return RAD_E_UNSUPPORTED;
    }

    SetDfa dfa(&R);
    dfa.begin(&work);
    KeyBuilder kb(P, X, dfa, work);

    /* The elements each key's bottom entry can be, by key; build-time only. */
    std::vector<std::vector<uint32_t>> kpos;
    std::vector<std::pair<uint32_t, uint32_t>> preds;

    P.top_key.assign(G.rep.size(), -1);
    std::vector<int32_t> todo;
    for (uint32_t c = 1; c < G.rep.size(); ++c) {
        if (G.tops[c].empty()) continue;
        KeyNode k;
        k.cls = c;
        P.keys.push_back(std::move(k));
        kpos.push_back(std::move(G.tops[c]));
        P.top_key[c] = (int32_t)P.keys.size() - 1;
        todo.push_back((int32_t)P.keys.size() - 1);
    }

    GbnfStack local;
    for (size_t qi = 0; qi < todo.size(); ++qi) {
        const int32_t ki = todo[qi];
        /* The compile automaton is a memo, dropped between keys (never during a walk) once large.
         * Each key counts as one operation for the growth bound, as each mask does at run time. */
        if (dfa.elems() > kMaxDfaElems) dfa = SetDfa(&R);
        dfa.begin(&work);

        /* A node's class is its bottom entry and `up` the suffix above it, so following `up`
         * lists the key bottom-first, in stack order. */
        local.clear();
        for (int32_t k = ki; k >= 0; k = P.keys[(size_t)k].up)
            local.push_back(&R.elems[G.rep[P.keys[(size_t)k].cls]]);
        preds_of(G, P, kpos[(size_t)ki], &preds, work);
        const bool ends = can_end(G, kpos[(size_t)ki]);
        P.keys[(size_t)ki].can_end = ends;
        st.max_depth = std::max<size_t>(st.max_depth, P.keys[(size_t)ki].depth);

        const int32_t s0 = dfa.intern(GbnfStacks{ local });
        const Outcome o = kb.outcome(s0, !preds.empty(), ends);
        if (o.final_) {
            P.keys[(size_t)ki].mask = o.mask;
            P.keys[(size_t)ki].rem  = o.rem;
        } else {
            if (P.keys[(size_t)ki].depth >= kMaxKeyDepth) {
                if (err) {
                    const uint32_t top_rule = G.erule[R.index(local.back())];
                    *err = "the tokens admissible inside rule " +
                           rule_name_of(rule_names, top_rule) + " depend on more than " +
                           std::to_string(kMaxKeyDepth) +
                           " levels of the parse stack at one position, so its masks cannot be "
                           "precomputed; restructure it so that what can follow it is decided "
                           "closer to it";
                }
                return RAD_E_UNSUPPORTED;
            }
            P.keys[(size_t)ki].extended = true;
            P.keys[(size_t)ki].mask = o.mask;
            const uint16_t depth = P.keys[(size_t)ki].depth;
            for (size_t i = 0; i < preds.size();) {
                KeyNode k;
                k.cls   = preds[i].first;
                k.up    = ki;
                k.depth = (uint16_t)(depth + 1);
                std::vector<uint32_t> at;
                for (; i < preds.size() && preds[i].first == k.cls; ++i)
                    at.push_back(preds[i].second);
                const uint32_t c = k.cls;
                P.keys.push_back(std::move(k));
                kpos.push_back(std::move(at));
                const int32_t id = (int32_t)P.keys.size() - 1;
                P.keys[(size_t)ki].kids.push_back({ c, id });
                todo.push_back(id);
            }
        }
        kpos[(size_t)ki] = {};
        st.bytes += sizeof(KeyNode);
        if (st.bytes > kMaxPlanBytes) {
            if (err)
                *err = "the grammar's precomputed masks need more than " +
                       std::to_string(kMaxPlanBytes >> 20) + " MiB";
            return RAD_E_UNSUPPORTED;
        }
    }

    st.keys  = P.keys.size();
    st.bytes += P.cls.size() * 4 + P.top_key.size() * 4;
    st.work  = work.used;
    st.ms    = ms_since(t0);
    P.mark.assign(P.keys.size(), 0);
    P.slot.assign(P.keys.size(), -1);
    if (stats) *stats = st;
    *out = std::move(pp);
    return RAD_OK;
}

/* ============================================================ a mask */

/* The key serving a stack, and how many entries it spans: from the top down, one entry per level,
 * until a non-extended node or the stack's bottom. -1 means no key covers the stack, which the
 * analysis rules out; the caller reports it. */
static int32_t key_of(const MaskPlan& P, const GbnfStack& S, size_t* depth, OpWork& work) {
    const GbnfRuleSet& R = *P.rules;
    const uint32_t c0 = P.cls[R.index(S.back())];
    int32_t k = c0 < P.top_key.size() ? P.top_key[c0] : -1;
    size_t  d = 1;
    while (k >= 0 && P.keys[(size_t)k].extended && d < S.size()) {
        const uint32_t c = P.cls[R.index(S[S.size() - 1 - d])];
        const auto& kids = P.keys[(size_t)k].kids;
        auto it = std::lower_bound(kids.begin(), kids.end(), std::make_pair(c, INT32_MIN));
        k = (it != kids.end() && it->first == c) ? it->second : -1;
        ++d;
        work.add(8);
    }
    if (k >= 0 && P.keys[(size_t)k].mask < 0) k = -1;
    *depth = d;
    return k;
}

static void or_mask(const KeyMask& m, uint32_t* bits, OpWork& work) {
    work.add(m.v.size() / (m.dense ? 8 : 1) + 1);
    if (m.dense) {
        const uint32_t* src = m.v.data();
        for (size_t w = 0, n = m.v.size(); w < n; ++w) bits[w] |= src[w];
    } else {
        for (uint32_t id : m.v) bits[id >> 5] |= 1u << (id & 31u);
    }
}

static int fill_keys(const MaskPlan& P, SetDfa& dfa, const GbnfStacks& stacks, uint32_t* bits,
                     OpWork& work) {
    if (++P.epoch == 0) {
        std::fill(P.mark.begin(), P.mark.end(), 0u);
        P.epoch = 1;
    }
    P.groups.clear();
    bool completed = false;
    for (const GbnfStack& S : stacks) {
        if (S.empty()) { completed = true; continue; }
        size_t d = 0;
        const int32_t k = key_of(P, S, &d, work);
        if (k < 0) {
            RAD_ERR("gbnf: no precomputed mask covers a stack of this grammar; the request cannot "
                    "be continued");
            return RAD_E_STATE;
        }
        const KeyNode& K = P.keys[(size_t)k];
        if (P.mark[(size_t)k] != P.epoch) {
            P.mark[(size_t)k] = P.epoch;
            P.slot[(size_t)k] = -1;
            or_mask(P.masks[(size_t)K.mask], bits, work);
        }
        if (K.rem < 0 || d >= S.size()) continue;
        P.ctx.assign(S.begin(), S.end() - (std::ptrdiff_t)d);
        work.add(P.ctx.size() + 8);
        const int32_t c = dfa.closure(P.ctx);
        if (P.slot[(size_t)k] < 0) {
            P.slot[(size_t)k] = (int32_t)P.groups.size();
            P.groups.push_back({ k, {} });
        }
        P.groups[(size_t)P.slot[(size_t)k]].ctx.push_back(c);
    }
    if (completed) {
        work.add(P.vx->blank.size() + 1);
        for (int32_t id : P.vx->blank) set_bit(bits, id);
    }
    for (auto& g : P.groups) {
        const int32_t x = dfa.unite(g.ctx);
        Walk w;
        w.trie = &P.rems[(size_t)P.keys[(size_t)g.key].rem];
        w.bits = bits;
        walk(dfa, x, w, work, P.ws);
    }
    return RAD_OK;
}

static void fill_resume(const MaskPlan& P, SetDfa& dfa, const GbnfStacks& stacks,
                        PartialUtf8 partial, uint32_t* bits, OpWork& work) {
    const int r = partial.n_remain;
    const VocabIndex::Resume& RS = P.vx->resume[r - 1];
    const int32_t s0 = dfa.intern(stacks);
    if (s0 < 0) return;
    const SetDfa::St& S = dfa.st(s0);
    const TrieNode*   nd = RS.trie.node.data();

    /* A character finished as U+0000 ends the piece. Carried bits are zero only after the head of
     * an overlong sequence; finishing it with zero bits decodes to 0, the sequence terminator, so
     * the piece has no characters whatever follows. Such a piece is admissible if it carries
     * nothing out, or if some char-range top could finish what it carries. These pieces are all
     * under the first root edge (zero bits) and are handled here, not stepped as a character. */
    const bool nul = partial.value == 0 && nd[0].n_child > 0 && nd[nd[0].first].cp == 1;
    if (nul) {
        std::vector<uint32_t> todo{ nd[0].first };
        while (!todo.empty()) {
            const uint32_t n = todo.back();
            todo.pop_back();
            for (uint32_t k = nd[n].tbeg; k < nd[n + 1].tbeg; ++k) {
                const PartialUtf8 tail = RS.trie.ttail[k];
                bool ok = tail.n_remain == 0;
                for (size_t j = 0; !ok && j < S.ctop.size(); ++j)
                    ok = match_partial_char(S.ctop[j], tail);
                if (ok) set_bit(bits, RS.trie.tid[k]);
            }
            for (uint32_t c = nd[n].first; c < nd[n].first + nd[n].n_child; ++c) todo.push_back(c);
            work.add(4 + (nd[n + 1].tbeg - nd[n].tbeg) * (S.match_cost + 1));
        }
    }

    Walk w;
    w.trie       = &RS.trie;
    w.bits       = bits;
    w.resume     = true;
    w.root_base  = partial.value << (6 * r);
    w.skip_first = nul;
    walk(dfa, s0, w, work, P.ws);

    work.add(RS.shorts.size() * (S.match_cost + 1));
    for (const auto& sh : RS.shorts) {
        const PartialUtf8 tail{ (partial.value << (6 * sh.k)) | sh.bits, r - sh.k };
        for (const GbnfElement* top : S.ctop)
            if (match_partial_char(top, tail)) { set_bit(bits, sh.id); break; }
    }
}

/* The plan's automaton, for one operation charged to `work`. Caller holds P.mu. Past its bound
 * the memo is dropped and rebuilt rather than refused, since every state can be rediscovered and
 * a grammar whose stacks deepen every token would otherwise add a state per token. */
static SetDfa& plan_dfa(const MaskPlan& P, OpWork& work) {
    if (!P.dfa || P.dfa->elems() > kMaxDfaElems) {
        if (P.dfa)
            RAD_DEBUG("gbnf: the stack-set automaton holds %zu states, %zu stack elements; "
                      "rebuilding it", P.dfa->n_states(), P.dfa->elems());
        P.dfa = std::make_unique<SetDfa>(P.rules);
        ++P.dfa_gen;
    }
    P.dfa->begin(&work);
    return *P.dfa;
}

bool plan_step(const MaskPlan& P, const std::vector<const GbnfStack*>& stacks, PlanState* at,
               const uint32_t* cps, size_t n, GbnfStacks* out, OpWork& work) {
    std::lock_guard<std::mutex> lk(P.mu);
    SetDfa& dfa = plan_dfa(P, work);

    /* The set as a state: the one the caller names, or canonicalised through pointers (sorted,
     * de-duplicated, copied only if new). The sort is charged as in intern(). */
    const bool named = at->id >= 0 && at->gen == P.dfa_gen;
    int32_t s0;
    if (named) {
        s0 = at->id;
        work.add(stacks.size() + 1);
    } else {
        std::vector<const GbnfStack*>& ps = P.step_ptrs;
        ps.clear();
        size_t elems = 0;
        bool sorted = true;
        for (const GbnfStack* k : stacks) {
            if (sorted && !ps.empty() && *k < *ps.back()) sorted = false;
            ps.push_back(k);
            elems += k->size() + 1;
        }
        work.add(elems * (sorted ? 1 : 1 + (uint64_t)std::bit_width(ps.size())));
        if (!sorted) {
            std::sort(ps.begin(), ps.end(),
                      [](const GbnfStack* a, const GbnfStack* b) { return *a < *b; });
        }
        ps.erase(std::unique(ps.begin(), ps.end(),
                             [](const GbnfStack* a, const GbnfStack* b) { return *a == *b; }),
                 ps.end());
        s0 = dfa.intern_ptrs(ps);
    }
    int32_t s = s0;

    /* Per code point: find its interval, then that interval's successor (both memoised). */
    for (size_t k = 0; k < n && s >= 0; ++k) {
        const SetDfa::St& S = dfa.parted(s);
        const size_t i = (size_t)(std::upper_bound(S.lo.begin(), S.lo.end(), cps[k]) -
                                  S.lo.begin()) - 1;
        work.add(8);
        s = dfa.next(s, i);
    }
    if (named && s == s0) return true;
    at->gen = P.dfa_gen;
    if (s < 0) {
        at->id = -1;
        out->clear();
        return false;
    }
    const SetDfa::St& S = dfa.st(s);
    at->id = S.has_empty || !S.ttop.empty() ? -1 : s;
    out->resize(S.stacks.size());
    for (size_t i = 0; i < S.stacks.size(); ++i)
        (*out)[i].assign(S.stacks[i].begin(), S.stacks[i].end());
    return false;
}

int plan_fill(const MaskPlan& P, const GbnfStacks& stacks, PartialUtf8 partial, bool allow_eog,
              uint32_t* bitmask, int64_t n_words) {
    std::lock_guard<std::mutex> lk(P.mu);
    std::fill(bitmask, bitmask + n_words, 0u);

    OpWork work;
    SetDfa& dfa = plan_dfa(P, work);

    if (allow_eog)
        for (int32_t id : P.vx->eog) set_bit(bitmask, id);

    if (partial.n_remain > 0) {
        fill_resume(P, dfa, stacks, partial, bitmask, work);
    } else {
        const int s = fill_keys(P, dfa, stacks, bitmask, work);
        if (s < 0) return s;
    }

    for (int64_t w = 0; w < n_words; ++w)
        if (bitmask[w]) return RAD_OK;
    return RAD_E_STATE;
}

}  /* namespace gbnf */
}  /* namespace rad */
