/* rad_laya.h -- the words a Laya request hands its plugin (RadBatch::plugin_words), and the pooled
 * row the plugin answers in.
 *
 * Laya (arch/laya) reads one sequence a question, `[CLS] <type> question: <instructions> [SEP]
 * [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]`, and scores each option at its [MASK]. The
 * server builds the sequence (core/server/laya.cpp) and hands the plugin what the tokens do not
 * say: the question's type, which picks the head's type embedding, and where the markers are. The
 * plugin answers in the sequence's pooled row: one logit an option, in marker order.
 *
 * Plain C, like every abi/ header: the plugin is built apart from the engine.
 */
#ifndef RAD_LAYA_H
#define RAD_LAYA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    /* Options a question, llama.cpp's bound for laya. The pooled row holds one logit an option. */
    RAD_LAYA_MAX_OPTIONS = 255,
    RAD_LAYA_POOLED      = 256
};

/* The question types in the order of the head's type embedding rows (laya's QTYPES). */
enum { RAD_LAYA_CHOICE = 0, RAD_LAYA_SCORE = 1, RAD_LAYA_NOUL = 2 };

/* A sequence's words. Word 0 is the engine's (RadBatch::plugin_words). */
enum {
    RAD_LAYA_W_ROW     = 0,   /* written by the engine */
    RAD_LAYA_W_N       = 1,   /* the sequence's tokens */
    RAD_LAYA_W_TYPE    = 2,   /* RAD_LAYA_CHOICE, RAD_LAYA_SCORE or RAD_LAYA_NOUL */
    RAD_LAYA_W_K       = 3,   /* options, so markers */
    RAD_LAYA_W_MARKERS = 4,   /* [k]: each option's [MASK] position in the sequence, in order */
    RAD_LAYA_MAX_WORDS = RAD_LAYA_W_MARKERS + RAD_LAYA_MAX_OPTIONS
};

#ifdef __cplusplus
}
#endif

#endif /* RAD_LAYA_H */
