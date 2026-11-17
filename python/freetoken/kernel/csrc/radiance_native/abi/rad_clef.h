/* rad_clef.h -- the words a clef request hands its head (RadBatch::plugin_words).
 *
 * clef's joint schema head (arch/common/rad_block_clef.h) reads more of a request than its
 * tokens: which prompt positions hold each question's instructions and each option's text, each
 * option's token ids (the lm_head rows it averages), and each question's type. The server lays
 * them out with the functions below, in the forms the head's kernels read (a cu_seqlens here, a
 * gather index there), and the engine carries them to the plugin as one range of int32 words.
 * Both sides include this header, so the layout is stated once.
 *
 * Plain C, like every abi/ header: the plugin is built apart from the engine.
 */
#ifndef RAD_CLEF_H
#define RAD_CLEF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The bounds the head's buffers are declared at. A request past one is refused by the server. */
enum {
    RAD_CLEF_MAX_QUESTIONS = 64,
    RAD_CLEF_MAX_OPTIONS   = 512,     /* every question's options together */
    RAD_CLEF_MAX_LEX       = 8192,    /* every option's tokens together */
    /* Prompt tokens. The head gathers every row of a prompt into buffers of its own, so this,
     * not the deployment's context, sizes them: at the model's own 262144 they would take
     * 4.5 GiB of the card for prompts that never pass a fifth of it. */
    RAD_CLEF_MAX_PROMPT    = 32768
};

/* The header: word 0 is the engine's (RadBatch::plugin_words), the rest the counts the layout is
 * computed from. */
enum {
    RAD_CLEF_W_ROW = 0,   /* the sequence's logits row this step, written by the engine */
    RAD_CLEF_W_N   = 1,   /* prompt tokens */
    RAD_CLEF_W_Q   = 2,   /* questions */
    RAD_CLEF_W_O   = 3,   /* options, every question together, in head order */
    RAD_CLEF_W_L   = 4,   /* option tokens, every option together */
    RAD_CLEF_W_S   = 5,   /* segments the prompt is cut into at every span's edges */
    RAD_CLEF_HDR   = 8
};

/* Where each array starts, in words from the range's start. Every array is int32. */
typedef struct RadClefWords {
    int32_t n, q, o, l, s;
    int32_t seg_cu;    /* [s + 1]: a partition of [0, n) with a boundary at every span's edges */
    int32_t q_seg;     /* [q]: the segment holding each question's instructions */
    int32_t o_seg;     /* [o]: the segment holding each option's text */
    int32_t o_q;       /* [o]: each option's question */
    int32_t opt_cu;    /* [q + 1]: each question's options, consecutive */
    int32_t type;      /* [q]: 0 noul, 1 choice, 2 score */
    int32_t lex_cu;    /* [o + 1]: each option's tokens in lex_ids, consecutive */
    int32_t lex_ids;   /* [l]: the option tokens, as the prompt holds them */
    int32_t last;      /* [1]: n - 1, the prompt's last position */
    int32_t cu_q;      /* [2]: {0, q} */
    int32_t cu_o;      /* [2]: {0, o} */
    int32_t cu_k;      /* [2]: {0, n} */
    int32_t total;     /* words in all */
} RadClefWords;

static inline RadClefWords rad_clef_words_layout(int32_t n, int32_t q, int32_t o, int32_t l,
                                                 int32_t s) {
    RadClefWords w;
    int32_t at = RAD_CLEF_HDR;
    w.n = n; w.q = q; w.o = o; w.l = l; w.s = s;
    w.seg_cu  = at; at += s + 1;
    w.q_seg   = at; at += q;
    w.o_seg   = at; at += o;
    w.o_q     = at; at += o;
    w.opt_cu  = at; at += q + 1;
    w.type    = at; at += q;
    w.lex_cu  = at; at += o + 1;
    w.lex_ids = at; at += l;
    w.last    = at; at += 1;
    w.cu_q    = at; at += 2;
    w.cu_o    = at; at += 2;
    w.cu_k    = at; at += 2;
    w.total   = at;
    return w;
}

/* The layout a range of words carries, read off its header. */
static inline RadClefWords rad_clef_words_read(const int32_t* words) {
    return rad_clef_words_layout(words[RAD_CLEF_W_N], words[RAD_CLEF_W_Q], words[RAD_CLEF_W_O],
                                 words[RAD_CLEF_W_L], words[RAD_CLEF_W_S]);
}

/* The most words a request can carry within the bounds: what the head declares
 * (rad_declare_plugin_words). A span's two edges each cut the prompt, so the segments are at most
 * two a span plus one. */
static inline int32_t rad_clef_max_words(void) {
    const int32_t spans = RAD_CLEF_MAX_QUESTIONS + RAD_CLEF_MAX_OPTIONS;
    return rad_clef_words_layout(0, RAD_CLEF_MAX_QUESTIONS, RAD_CLEF_MAX_OPTIONS,
                                 RAD_CLEF_MAX_LEX, 2 * spans + 1).total;
}

#ifdef __cplusplus
}
#endif

#endif /* RAD_CLEF_H */
