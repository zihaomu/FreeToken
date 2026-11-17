#pragma once
// r4d_ar_shards.h: the exact two-shot's shard table. Shard s is the 16-byte words
// [off[s], off[s+1]) and is reduced by rank s. Host only and HIP-free: the launcher, the plugin
// shim (which sizes scratch and picks the block count from it) and the selftest (which prints it)
// must derive the same table from the same inputs.
//
// The even split is n16 / ws words a shard with the first n16 % ws shards one word longer. Weights
// split proportionally (floor(n16 * w_s / sum)) and hand the few remaining words, one each, to the
// weighted ranks in ascending order; equal weights therefore reproduce the even split exactly, and
// a zero weight always means an empty shard. Which rank owns a word changes only the link it
// crosses, not its fold (fp32, ascending rank), so every table gives the same bits.
#include <cstdlib>

// Below R4D_AR8_MIN_ELEMS_DEFAULT elements the weights and the pieces do not apply: a
// decode-sized message pays more for the extra handshakes and the uneven split than its link time
// saves (measured on the eight R9700: 196608 elements run fastest even and in one piece, 524288
// already gain from weights and two pieces).
// The eight-rank two-shot's form, its launch parameter `form` (r4d_ar_twoshot_Nrank_exact.hip):
// lockstep (one handshake a step), ring (the first hop sent to the ranks in ring order from the
// next one) or split (ring, and a flag after each half of a step, each awaited only where its data
// is read). Every form folds the same words in the same order, so all give the same bits. The
// engine runs lockstep; the other two exist to time a board (r4d_selftest --perfar8).
enum { R4D_AR8_FORM_LOCKSTEP = 0, R4D_AR8_FORM_RING = 1, R4D_AR8_FORM_SPLIT = 2 };
enum {
  R4D_AR_SHARD_WEIGHT_MAX = 65535,
  R4D_AR8_MIN_ELEMS_DEFAULT = 262144,
};

// The eight-rank two-shot's settings. The engine runs the defaults: weights from each card's
// slowest PCIe hop (r4d_ar_weights_from_rates, r4d_ar_links.h), the piece count from the message
// size (r4d_ar8_auto_pieces), both from R4D_AR8_MIN_ELEMS_DEFAULT elements. r4d_selftest sets
// others through r4d_ar8_test_settings (r4d.h) to check unequal and empty shards, fixed piece
// counts and the floors, and to time another board's weights.
enum { R4D_AR8_WEIGHTS_AUTO = 0, R4D_AR8_WEIGHTS_EVEN = 1, R4D_AR8_WEIGHTS_SET = 2 };
struct R4DAr8Settings {
  int weights = R4D_AR8_WEIGHTS_AUTO;
  int w[8] = { 0 };       // R4D_AR8_WEIGHTS_SET: the eight weights
  int pieces = 0;         // 0: r4d_ar8_auto_pieces
  long long weights_min = R4D_AR8_MIN_ELEMS_DEFAULT;
  long long pieces_min = R4D_AR8_MIN_ELEMS_DEFAULT;
};

// The piece count when none is set, by message size in elements: more pieces
// shorten the one-way first and last steps ((K+1)/K of the link time) but add a handshake each,
// which only a large message amortises (measured at 4,1,0,4,0,1,4,4: K2 best at 524288, K4 at
// 2097152, K8 at 8388608).
inline int r4d_ar8_auto_pieces(long long numel) {
  if (numel < 262144) return 1;
  if (numel < 1048576) return 2;
  if (numel < 4194304) return 4;
  return 8;
}

// "lockstep", "ring" or "split" as R4D_AR8_FORM_*; -1 for anything else.
inline int r4d_ar8_form_parse(const char* s) {
  if (!s) return -1;
  const char* names[3] = { "lockstep", "ring", "split" };
  for (int f = 0; f < 3; ++f) {
    const char* a = names[f];
    const char* b = s;
    while (*a && *a == *b) { ++a; ++b; }
    if (!*a && !*b) return f;
  }
  return -1;
}

// `ws` non-negative integers separated by commas, each at most R4D_AR_SHARD_WEIGHT_MAX, summing
// to more than zero; nothing else (no blanks, signs or trailing text).
inline bool r4d_ar_shard_weights_parse(const char* s, int ws, int* w) {
  if (!s) return false;
  long sum = 0;
  for (int i = 0; i < ws; ++i) {
    if (*s < '0' || *s > '9') return false;
    long v = 0;
    while (*s >= '0' && *s <= '9') {
      v = v * 10 + (*s++ - '0');
      if (v > R4D_AR_SHARD_WEIGHT_MAX) return false;
    }
    w[i] = (int)v;
    sum += v;
    if (i + 1 < ws) {
      if (*s != ',') return false;
      ++s;
    }
  }
  return *s == '\0' && sum > 0;
}

// Fills off[0..ws] and returns the longest shard. `w` null: the even split.
inline long r4d_ar_shard_table(long n16, int ws, const int* w, long* off) {
  long len[16];
  if (!w) {
    const long base = n16 / ws, rem = n16 % ws;
    for (int s = 0; s < ws; ++s) len[s] = base + (s < rem ? 1 : 0);
  } else {
    long sum = 0;
    for (int s = 0; s < ws; ++s) sum += w[s];
    long left = n16;
    for (int s = 0; s < ws; ++s) {
      len[s] = n16 * w[s] / sum;
      left -= len[s];
    }
    for (int s = 0; s < ws && left > 0; ++s)
      if (w[s] > 0) { ++len[s]; --left; }
  }
  long sh_max = 0;
  off[0] = 0;
  for (int s = 0; s < ws; ++s) {
    off[s + 1] = off[s] + len[s];
    if (len[s] > sh_max) sh_max = len[s];
  }
  return sh_max;
}

// The longest shard any message of at most n16 words can get under `w` (null: even): what the
// scratch slot must hold twice.
inline long r4d_ar_shard_max_bound(long n16, int ws, const int* w) {
  if (!w) return (n16 + ws - 1) / ws;
  long sum = 0, wmax = 0;
  for (int s = 0; s < ws; ++s) {
    sum += w[s];
    if (w[s] > wmax) wmax = w[s];
  }
  const long b = n16 * wmax / sum + 1;
  return b < n16 ? b : n16;
}

// The two-shot timeline, a diagnostic (r4d.h, `timeline`): per block R4D_AR_TL_WORDS stamps of the
// device wall clock (hipDeviceAttributeWallClockRate), [0] after the sequence step, an (arrive,
// release) pair per handshake, then the end of the block's last local work, so a launch in K
// pieces fills 2 + 2 * (K + 1). R4D_AR_TL_BLOCKS is the wide family's block ceiling.
enum { R4D_AR_TL_WORDS = 2 + 2 * (16 + 1), R4D_AR_TL_BLOCKS = 64 };
