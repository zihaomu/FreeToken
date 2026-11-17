/* kld.h -- the KL divergence of this model against a recorded reference, at every scored position
 * of a fixed corpus.
 *
 * `--kld-record DIR` runs the corpus through the reference model (e.g. the bf16 checkpoint) once
 * and writes its full-vocabulary log-probabilities at every scored position. `--kld-ref DIR` runs a
 * candidate over the same tokens and compares its logits row by row, so each candidate costs one
 * prefill of the corpus.
 *
 * Reported: the distribution of per-position KL(P_ref || Q_candidate) (mean, median, 99th, 99.9th,
 * 99.99th percentile, max), top-1 agreement, and both models' perplexity on the corpus's next
 * tokens. The tail matters: a small mean KL can hide confidently wrong positions.
 *
 * The whole vocabulary by default, not a top-k: a truncated KL is a lower bound, loosest for flat
 * distributions, which are the high-KL tail. `--kld-top N` records the sparse form instead, for a
 * corpus whose full rows would not fit a disk (10M positions of a 248K vocabulary are 5 TB): the N
 * most likely tokens exactly and the mass of the rest. A candidate is then scored by the KL between
 * the two distributions coarsened to those N tokens and one bucket for every other token, which by
 * the data-processing inequality is at most the full KL and loses only what the reference spreads
 * over its tail (at N = 512 a few 1e-4 nats on a flat position, nothing on a confident one).
 *
 * Reference files (DIR):
 *   kld.json     the manifest: the model, the corpus, every document's source, token count, first
 *                scored position and first row. Written last, so an interrupted run leaves a
 *                directory no --kld-ref reads.
 *   tokens.i32   every document's tokens, concatenated, so the candidate scores the reference's
 *                tokens rather than a re-tokenisation.
 *   logp.f16     [rows, n_vocab] log-probabilities, f16: finest near log p = 0, coarse only for
 *                tail tokens with little mass.
 *   top.bin      [rows, kKldTop] {int32 id, f32 log p}: the most likely tokens exactly, where a
 *                confident position's KL lives (f16 would cost ~1e-4 nats).
 *   rows.f32     [rows, 2] the reference's own next-token NLL and entropy.
 *
 * The sparse form (`--kld-top N`, format radiance-kld-top-1) has no logp.f16; top.bin is
 * [rows, N] {int32 id, f32 log p}, most likely first, and rows.f32 is [rows, 3], the third column
 * the mass of every token outside the row's N, summed in double from the full row.
 *
 * Document d scores positions [score_from, n - 1): the logits at p predict token p + 1, and the
 * last token predicts nothing in the corpus. Its row r is row0 + p - score_from.
 */
#pragma once
#include "rad_core.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rad {

class Vocab;

/* One scoring entry's rows of a step: request `id`'s prompt positions [pos0, pos0 + n) are rows
 * [row0, row0 + n) of the block the engine copied back. */
struct KldSpan {
    uint64_t id   = 0;
    int64_t  pos0 = 0;
    int64_t  n    = 0;
    int64_t  row0 = 0;
};

/* One rank's part of that block: its rows of `width` f32 logits, the vocabulary from `vocab0`. */
struct KldShard {
    const float* rows   = nullptr;
    int64_t      vocab0 = 0;
    int64_t      width  = 0;
};

/* The exact head kept beside every f16 row. */
static constexpr int kKldTop = 16;
/* The largest --kld-top: a row of the sparse form is 8 bytes a token. */
static constexpr int kKldTopMax = 8192;

class Kld {
public:
    Kld();
    ~Kld();

    /* Reads the corpus and creates DIR (record), or reads DIR's manifest and opens its rows (ref).
     * `model` names this run's model in the manifest and the report. */
    int open(const Config& cfg, std::shared_ptr<const Vocab> vocab, int64_t n_vocab,
             const std::string& model);

    /* open() split for already-tokenised documents (tests): add_doc() each, then record_into()
     * creates DIR; score_against() is the whole ref mode. */
    int add_doc(const std::string& source, const std::vector<int32_t>& ids, int64_t score_from);
    int record_into(const std::string& dir, int64_t n_vocab, const std::string& model, int top = 0);
    int score_against(const std::string& dir, const std::string& out, int64_t n_vocab,
                      const std::string& model);

    /* The corpus alone, for a run that only drives it through prefill (--calib-corpus). */
    int load(const std::string& corpus, std::shared_ptr<const Vocab> vocab) { return load_corpus(corpus, vocab); }
    int64_t n_tokens() const { return (int64_t)tokens_.size(); }

    size_t n_docs() const { return docs_.size(); }
    const std::string& source(size_t d) const { return docs_[d].source; }
    /* Document d's tokens, and the request id the engine submits it under: d + 1. */
    std::vector<int32_t> tokens(size_t d) const;
    int64_t n_rows() const { return n_rows_; }
    int64_t rows_done() const { return done_n_.load(std::memory_order_relaxed); }

    /* A step's scored rows. Rows outside a document's scored range are skipped; a row seen twice
     * (re-prefill after preemption) is recomputed. Blocks until every row is written or scored,
     * since the engine reuses the block next step. */
    int score(const std::vector<KldSpan>& spans, const std::vector<KldShard>& shards);

    /* Record: checks every row was written and writes the manifest. Ref: the report, printed and,
     * with --kld-out, written. */
    int finish();

private:
    struct Doc {
        std::string source;
        int64_t     off = 0;          /* into tokens_ */
        int64_t     n = 0;
        int64_t     score_from = 0;
        int64_t     row0 = 0;
        int64_t     rows() const { return n - 1 - score_from; }
    };
    struct Job {
        int64_t row;                  /* the reference's row */
        int64_t at;                   /* the row in the copied block */
        int32_t next;                 /* the corpus's next token */
    };

    int  load_corpus(const std::string& path, const std::shared_ptr<const Vocab>& vocab);
    int  load_ref(const std::string& dir);
    void size_rows();
    /* A worker's row-sized scratch: the f16 row (full form), the head and the head's marks
     * (sparse form). */
    struct Scratch;
    void record_row(const Job& j, const std::vector<KldShard>& sh, Scratch& w);
    void score_row(const Job& j, const std::vector<KldShard>& sh, Scratch& w);
    void score_row_top(const Job& j, const std::vector<KldShard>& sh, Scratch& w);
    int  report();

    bool record_ = false;
    std::string dir_, out_, model_, ref_model_, corpus_;
    uint64_t corpus_hash_ = 0;
    int64_t n_vocab_ = 0;
    int     top_ = kKldTop;            /* entries a row of top.bin holds */
    bool    sparse_ = false;           /* the --kld-top form: no logp.f16, tail mass in rows.f32 */
    int64_t rows_w_ = 2;               /* rows.f32 columns */
    std::vector<int32_t> tokens_;
    std::vector<Doc> docs_;
    int64_t n_rows_ = 0;
    int fd_logp_ = -1, fd_top_ = -1, fd_rows_ = -1;

    /* Per reference row. `seen_` marks a row written or scored; the rest is the ref mode's. */
    std::vector<uint8_t> seen_;
    std::atomic<int64_t> done_n_{0};
    std::vector<float>   kl_, nll_q_, nll_p_;
    std::vector<uint8_t> top1_;
    std::atomic<int>     io_err_{0};
    float f16_lut_[65536];
};

}  /* namespace rad */
