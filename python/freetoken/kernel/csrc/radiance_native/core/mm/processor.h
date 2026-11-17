/* processor.h -- what an image, video or audio clip becomes before an encoder sees it (spec §11).
 *
 * The family fixes the arithmetic, not the numbers: every size, rate and token id comes from the
 * container, which carries the checkpoint's config.json and its two preprocessor configs
 * (rad-convert writes them under `vision_config.*`, `preprocessor.*` and `video_preprocessor.*`).
 * The Qwen-VL family (Qwen2-VL through Qwen3.x), the reference shape the others follow:
 *
 *   - A picture is resized so both sides are multiples of patch * merge and its area falls in the
 *     configured band, keeping the aspect ratio (smart_resize), with PIL's antialiased bicubic.
 *   - It is cut into patch x patch squares, `temporal` frames deep (an image is repeated to fill
 *     the depth), in merge-block order: the merge x merge patches that become one output row are
 *     consecutive. Each patch row is (channel, frame, y, x), normalised by the configured mean and
 *     deviation, in bf16.
 *   - The prompt gets `<vision_start> pad*N <vision_end>` where the template put the media, N
 *     being the encoder's output rows. A video gets one run per temporal group, each after the
 *     text `<T seconds>`; the groups share one temporal rotary component, so the text is how the
 *     model knows when a frame is.
 *   - Rotary positions have three components (temporal, height, width). Text counts up in all
 *     three; a media run holds its start in the first and adds its row and column to the other
 *     two; the text after it continues from start + max(rows, cols). See RopeLayout.
 *
 * The encoder's input is patches plus each patch's (row, col) in its segment and the segment's
 * size, which is all a learned position table or a 2-D rotary embedding needs. The processor
 * produces those too, so the core stages them without knowing what they are for.
 */
#pragma once
#include "mm/media.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct RadModelMeta;

namespace rad {
namespace mm {

enum class Kind { Image, Video, Audio };

/* The processor family, from the container's vision_config.model_type. Families differ in how a
 * picture is sized, what a patch value is, and how a video is sampled and spelled in the prompt;
 * the Item they produce and everything after it is shared.
 *
 *   QwenVL  smart_resize into a pixel band; patches (channel, frame, y, x) normalised by the
 *           configured mean and deviation; videos in temporal groups, each after a "<T seconds>".
 *   Gemma4  the largest size whose sides are multiples of patch * pooling and whose patch count
 *           fits the soft-token budget (budget * pooling^2), aspect ratio kept; patches (y, x,
 *           channel) as 2 * (u / 255) - 1, the reference model's own input scaling; a video is
 *           frames sampled at `fps` (at most `max_frames`, thinned evenly), each its own segment
 *           at the per-frame budget, each spelled <boi> placeholders <eoi> with no timestamp.
 *           Gemma 4 12B ("unified", no tower) is the same with a whole pooling window as one
 *           patch (patch = 48, merge 1) and its values u / 255 (`unit_pixels`).
 *   Lfm2    LFM2-VL's: a picture within its token budget (times a tolerance) is resized whole --
 *           sides multiples of patch * merge, pixels within the budget's band -- and one past it
 *           is cut into tile x tile crops, min_tiles..max_tiles of them in the ratio nearest its
 *           own, followed by the whole picture at the budgeted size as a thumbnail. Patches
 *           (y, x, channel) normalised by the mean and deviation, resized with torchvision's
 *           fixed-point bicubic; images only, and the tokens rotate at their positions.
 *   UnlimitedOcr
 *           Unlimited-OCR's DeepSeek-OCR shape, in the checkpoint's two modes (ImageMode): the
 *           picture padded (ImageOps.pad) to base_size x base_size on a 127 grey as the global
 *           view, and, in the gundam mode, a picture larger than one tile cut into min_tiles..
 *           max_tiles tile x tile crops at the tile in the ratio nearest its own (its
 *           dynamic_preprocess). The SAM neck downsamples by `downsample`, so patch * downsample
 *           is one output row. A view's rows are laid out as the reference's feature grid plus a
 *           column of newlines, which is why every segment carries `downsample` extra patch
 *           columns that nothing is read from: patches are raster order (Item::patch_raster), one
 *           patch (channel, y, x), each value (u / 255 - mean) / std in f32, rounded to bf16.
 *           Every placeholder is one id (the picture marker itself); the tokens rotate at their
 *           positions.
 *   Glm5Next  GLM-5.3's: a canvas whose sides are multiples of patch * merge and whose tokens fall
 *           in [min_image_tokens, max_image_tokens] (the sides rounded up, a picture under the
 *           floor grown, one over the budget fitted by the reference's search); the picture keeps
 *           its aspect ratio, is never enlarged above the floor, and is resized with torchvision's
 *           fixed-point bicubic only when its size changes; the rest of the canvas, right and
 *           bottom, is zero bytes. Patches as QwenVL's. Images only, spelled <begin_of_image>
 *           placeholders <end_of_image>, the tokens at their positions (the language model has no
 *           rotary).
 *   Pixtral  Nemotron 3.5's noncausal Mistral3 wrapper: longest-edge resize, then ceil each side
 *           to patch * merge; PIL bicubic and CLIP normalization. Raster (channel, y, x) patches;
 *           image placeholders per merged grid row, each followed by IMG_BREAK (last: IMG_END).
 *           Every token has ordinary 1-D positions, and separators retain their token embedding.
 *   Radio   NVIDIA's Nemotron-H Omni with C-RADIO: a picture becomes one tile whose patch grid
 *           falls between min_num_patches and max_num_patches, aspect ratio kept and both sides
 *           even (the pixel shuffle's factor), resized with torch's antialiased bicubic on float
 *           pixels; a video frame the same at about video_target_num_patches. Values are
 *           (u / 255 - mean) / std in f32, rounded to bf16. A patch row is the tubelet's frames
 *           (frame, channel, y, x) and then the picture's (channel, y, x), the part not in use
 *           zero, so one embedding GEMM serves both projections. Each segment carries `n_prefix`
 *           zero rows ahead of its patches, which the tower fills with its class and register
 *           tokens. Pictures are spelled <img> placeholders </img>; a video one such run a
 *           tubelet, each after its "Frame i sampled at T seconds and frame i+1 ...: " label,
 *           runs joined by a newline. Tokens at their positions (NoPE).
 *   MuseGlimmer  Meta's Muse Glimmer 30B: a picture (a video frame the same) resized so both
 *           sides are multiples of patch * merge and its merged-token count stays under the cap
 *           (max_image_tokens, a frame max_frame_rows), to the whole-merged-patch grid whose
 *           aspect is nearest its own (muse_size, no floor band: a small picture is downscaled
 *           onto the grid). Resized with torchvision's antialiased Lanczos on the uint8 pixels
 *           (resize_lanczos_torch); values (u / 255 - mean) / std in f32, rounded to bf16. A
 *           patch row is (frame, channel, y, x), the reference's (temporal, channel), a picture
 *           duplicated to fill the depth. A picture is spelled <image_start> <|patch|>*N
 *           <image_end>; a video one <vid_start>, then per temporal group the text `Time: T s`
 *           and <|video|>*N rows, the groups separated by <|vid_frame_separator|> and closed by
 *           <vid_end>, T the group's first frame time (the reference stamps a group once, after
 *           repeating the last frame to a whole group). */
enum class Family { QwenVL, Gemma4, Lfm2, UnlimitedOcr, Glm5Next, Pixtral, Radio, MuseGlimmer };

struct VisionConfig {
    Family  family = Family::QwenVL;
    int32_t patch = 0, temporal = 0, merge = 0;
    /* Gemma4: soft tokens (output rows) an image and a video frame may take. One of 70, 140,
     * 280, 560, 1120, the budgets the model was trained at. */
    int32_t image_soft_tokens = 0, video_soft_tokens = 0;
    /* Gemma4: patch values u / 255 rather than 2 * (u / 255) - 1 (the 12B's embedder). */
    bool    unit_pixels = false;
    int64_t image_min_pixels = 0, image_max_pixels = 0;
    int64_t video_min_pixels = 0, video_max_pixels = 0;
    /* Glm5Next: the band an image's canvas falls in, in output rows (tokens). */
    int32_t min_image_tokens = 0, max_image_tokens = 0;
    double  fps = 2.0;
    int32_t min_frames = 4, max_frames = 768;
    /* Per-frame ceiling on a video's resolution, in output rows (the reference processor's
     * per-frame cap). A clip with few frames does not spend the whole video budget on
     * near-native frames, so a video's cost grows with its length. 0 turns the cap off. */
    int32_t max_frame_rows = 768;
    float   mean[3] = { 0.5f, 0.5f, 0.5f };
    float   stdv[3] = { 0.5f, 0.5f, 0.5f };
    int32_t tok_image = -1, tok_video = -1, tok_start = -1, tok_end = -1;
    /* MuseGlimmer: the video run's own markers, <vid_start>, <vid_end> and the groups'
     * <|vid_frame_separator|> (a picture shares tok_start and tok_end). A container that numbers
     * them (`video_start_token_id`, `video_end_token_id`, `video_frame_separator_token_id`)
     * wins; otherwise the vocabulary resolves the spellings at Processor::init. */
    int32_t tok_video_start = -1, tok_video_end = -1, tok_video_sep = -1;
    /* Lfm2: the tiling (see Family). `max_segment_patches` is the tower's longest segment. */
    int32_t min_tiles = 0, max_tiles = 0, tile = 0;
    bool    use_thumbnail = false;
    /* Radio: a picture's patch budget [radio_min_patches, radio_max_patches] (with the global cap
     * radio_max_len tokens), the per-frame target of a video, the frames a tubelet packs and the
     * learned rows (class and register tokens) a segment carries ahead of its patches. */
    int32_t radio_min_patches = 0, radio_max_patches = 0, radio_max_len = 0;
    int32_t radio_video_patches = 0, radio_tubelet = 0, radio_prefix = 0;
    double  max_pixels_tolerance = 0.0;
    int64_t max_segment_patches = 0;

    /* UnlimitedOcr: the two modes' sizes, from the checkpoint's preprocessor config. The gundam
     * mode pads the global view to `base_size` and cuts crops at `tile`; the base mode (a page of
     * a multi-page document) pads the whole picture to `image_size`. `downsample` is the SAM
     * neck's ratio: one output row is `patch * downsample` pixels a side. `crop_mode` is the
     * deployment's mode when a request names none. */
    int32_t base_size = 0, image_size = 0, downsample = 0;
    bool    crop_mode = false;
    /* The vocabulary's spelling of the picture marker (`processor.image_token`); every
     * placeholder of a picture is that one id. */
    std::string image_token = "<image>";
    /* Pixtral: resize within this longest edge, then round each side up to patch * merge.
     * Its image rows are separated by real vocabulary tokens, without an opening marker. */
    int32_t pixtral_longest_edge = 0, tok_break = -1;
    std::string image_break_token = "[IMG_BREAK]", image_end_token = "[IMG_END]";

    /* Deployment bounds, not the checkpoint's. `max_patches` is the encoder's pass size: an image
     * is one segment and a segment is never split, so the image band's ceiling is lowered to fit
     * it. `max_source_pixels` bounds what the decoder may hold before any resize. */
    int64_t max_patches = 0;
    int64_t max_source_pixels = (int64_t)1 << 26;
    /* A whole video's patch budget, across all its passes. */
    int64_t max_video_patches = 0;

    /* Checks the pixel bands are non-empty and the frame counts and rate positive; RAD_E_INVAL
     * names the first violation in `why`. from_meta runs it on the container's values, and the
     * engine again after command-line overrides. */
    int check(std::string* why) const;

    int64_t patch_dim() const {
        /* Radio's row is a tubelet's frames and then the picture's patch (Family::Radio). */
        if (family == Family::Radio) return 3LL * (1 + radio_tubelet) * patch * patch;
        return 3LL * temporal * patch * patch;
    }
    int32_t factor() const { return patch * merge; }

    /* Read from the container; refuses one with no vision tower. Processor-only keys the
     * container omits take the family's defaults. */
    int from_meta(const RadModelMeta& m, std::string* why);
};

/* Which picture processor a request gets, for a checkpoint that has more than one: Unlimited-OCR
 * states two modes and the reference names them gundam (one picture: the global view at base_size
 * plus crops) and base (a page of a multi-page document: the global view only, at image_size).
 * `Default` is the deployment's (`VisionConfig::crop_mode`), or base when `several` says the
 * request carries more than one picture; a request may pick either mode and, with it, the sizes
 * the checkpoint states differently. `base_size` and `image_size` of 0 keep the mode's own. A
 * processor with one mode refuses a named mode and ignores `several`. */
struct ImageMode {
    enum Kind : int32_t { Default = 0, Gundam = 1, Base = 2 };
    int32_t kind = Default;
    int32_t base_size = 0, image_size = 0;
    bool    several = false;
};

/* The audio feature extractor family, from the container's audio_config.model_type. The
 * families differ in the spectrogram and in how a clip's frames are laid out for the encoder;
 * the Item they produce is shared.
 *
 *   Gemma4    Gemma4AudioFeatureExtractor: a frame_length / 2 zero lead, a periodic Hann window
 *             through an fft-point FFT, magnitudes through an HTK mel bank, log(x + mel_floor).
 *             The clip's real frames, then zero frames to whole output rows.
 *
 *   Gemma4Unified  Gemma4UnifiedAudioFeatureExtractor (the 12B's): no spectrum; a frame is
 *             `frame_length` samples as they are, the last zero-padded, one frame a row.
 *
 *   Qwen3ASR  Qwen3ASRFeatureExtractor (Whisper's): a clip shorter than `min_samples` is
 *             zero-padded to it; frames are torch.stft's, centred with a reflected n_fft / 2 at
 *             each end, a periodic Hann window and power spectra, the last frame dropped, so n
 *             samples are n / hop frames; a Slaney mel bank with Slaney area normalisation;
 *             log10(max(x, 1e-10)), floored at the clip's maximum less 8, then (x + 4) / 4.
 *             The encoder convolves each `chunk` frames (one second) on its own, zero-padded to a
 *             whole chunk, and keeps `chunk_rows(L)` rows of a chunk of L real frames. A chunk is
 *             staged as merge * chunk_rows(L) frames: its real frames, the reference's zero
 *             padding, and, where that exceeds a whole chunk, frames past it that the encoder
 *             reads as zero. Plane 1 of the coordinates marks those (RadBatch::enc_coord), which
 *             keeps one chunk's convolution from seeing the next. */
enum class AudioFamily { Gemma4, Qwen3ASR, Gemma4Unified };

/* Audio features, configured from the container: Gemma 4's framing from `preprocessor.*` (the
 * checkpoint's preprocessor_config.json), Qwen3-ASR's from `processor.feature_extractor.*`
 * (processor_config.json), and the encoder's own from audio_config.
 *
 * Gemma 4: samples are mono at `rate`, cut at `max_samples`, with `frame_length / 2` zeros in
 * front. Frame i is samples [i * hop, i * hop + frame_length), pre-emphasised when configured,
 * times a periodic Hann window, through an `fft`-point real FFT; magnitudes go through an HTK mel
 * bank (no area normalisation) and log(x + mel_floor), optionally standardised per bin.
 *
 * A frame is real when its last sample is (the reference's mask), so an n-sample clip has
 * floor((n - frame_length / 2 - 1) / hop) + 1 real frames and keeps ceil(real / merge) output
 * rows: that many placeholders between <boa> and <eoa>. The staged patches are the real frames
 * then zero frames up to a whole row, which matches the reference's zeroed frames past the end
 * for every row it keeps. */
struct AudioConfig {
    AudioFamily family = AudioFamily::Gemma4;
    int32_t rate = 0, frame_length = 0, hop = 0, fft = 0, n_mels = 0;
    double  min_freq = 0.0, max_freq = 0.0, mel_floor = 0.0;
    double  preemphasis = 0.0, input_scale = 1.0;
    bool    preemphasis_htk = true;
    std::vector<double> bin_mean, bin_std;      /* empty: no per-bin standardisation */
    /* The longest clip in samples: Gemma 4's extractor truncates at 480000 (30 s at 16 kHz).
     * Qwen3-ASR's does not truncate, and its longest clip is what one encoder pass holds. */
    int64_t max_samples = 0;
    /* Qwen3ASR: a shorter clip is zero-padded to this many samples first (the reference's
     * min_length); the frames the encoder convolves on their own (2 * n_window). */
    int64_t min_samples = 0;
    int32_t chunk = 0;
    int32_t tok_audio = -1, tok_start = -1, tok_end = -1;

    /* The encoder's: frames per output row, and the most frames a pass carries. */
    int32_t merge = 0;
    int64_t max_frames = 0;

    /* Read from the container; refuses one that states no audio tower this processor knows.
     * Qwen3-ASR's config names no token around its placeholders, so tok_start and tok_end stay
     * -1 for the caller to resolve (the architecture plugin spells them; see
     * RadAudioFormat::audio_open). */
    int from_meta(const RadModelMeta& m, std::string* why);
    /* The real frames of an n-sample clip. */
    int64_t real_frames(int64_t n_samples) const;
    /* Qwen3ASR: the encoder rows a chunk of `frames` real frames keeps -- three convolutions of
     * kernel 3, stride 2 and padding 1, so a whole chunk of 100 is 13. */
    static int64_t chunk_rows(int64_t frames);
};

/* One media part after processing: its patches, its place in the prompt's token stream and, once
 * the encoder has run, its rows. Shared between the owning request and the engine thread that
 * encodes it; the engine writes `embd` once and nothing reads it before then. */
struct Item {
    Kind    kind = Kind::Image;
    int32_t grid_t = 0, grid_h = 0, grid_w = 0;   /* in patches; grid_t temporal groups */
    int64_t n_patch = 0;
    int64_t patch_dim = 0;
    std::vector<uint16_t> pixels;                 /* [n_patch, patch_dim] bf16 */

    /* The encoder's output: n_seg segments of seg_rows rows, one segment per temporal group.
     * `rows_w` is a segment's width in output rows, which is where its rotary grid wraps. */
    int32_t n_seg = 0, seg_rows = 0, rows_w = 0, rows_h = 0;
    /* An audio clip's real feature frames; the rest of `grid_h` are zero frames padding it to
     * whole output rows. 0 for a picture or a video. */
    int32_t valid = 0;
    /* An audio clip staged in chunks (AudioFamily::Qwen3ASR): each chunk but the last is
     * `chunk_staged` frames, of which the first `chunk_read` take part in the convolution and the
     * rest read as zero. 0 for a clip staged whole. */
    int32_t chunk_staged = 0, chunk_read = 0;

    /* Per-segment sizes, when an item's segments differ (an LFM2-VL picture is its tiles plus a
     * thumbnail): each is `h` x `w` patches starting at patch `patch0`. Empty means every segment
     * is grid_h x grid_w. The rest of the engine reads segments only through the accessors below,
     * so both forms look alike. */
    struct Segment { int32_t h = 0, w = 0; int64_t patch0 = 0; bool sep = false; };
    std::vector<Segment> segments;
    /* The model rotates this item's tokens at their token positions (LFM2-VL's 1-D rotary), not
     * on the grid the Qwen-VL family's M-RoPE lays out (RopeLayout). */
    bool rope_1d = false;
    /* Patches are in raster order (row by row), not merge-block order: the coordinate planes
     * invert the one or the other (Unlimited-OCR's tower reads a picture patch by patch). */
    bool patch_raster = false;
    /* Rows ahead of each segment's patches that the encoder fills itself (Family::Radio: C-RADIO's
     * class and register tokens): zero pixels, coordinate (-1, k), counted in patch_at and
     * n_patch but never in a segment's output rows. */
    int32_t seg_prefix = 0;

    int32_t seg_h(int32_t s) const { return segments.empty() ? grid_h : segments[(size_t)s].h; }
    int32_t seg_w(int32_t s) const { return segments.empty() ? grid_w : segments[(size_t)s].w; }
    /* The one-output-row view separator (Unlimited-OCR stages it as a merge-grid cell of zeros).
     * A square segment is not enough: a chunked line of one tile is q x q too. */
    bool seg_is_sep(int32_t s) const { return !segments.empty() && segments[(size_t)s].sep; }
    /* Segment s's first patch and first output row; s == n_seg is the end of the item. */
    int64_t patch_at(int32_t s) const {
        if (segments.empty()) return (int64_t)s * (seg_prefix + (int64_t)grid_h * grid_w);
        return s >= n_seg ? n_patch : segments[(size_t)s].patch0;
    }
    int64_t row_at(int32_t s) const {
        if (segments.empty()) return (int64_t)s * seg_rows;
        const int64_t m = rows_h > 0 ? grid_h / rows_h : 1;
        return patch_at(s) / (m * m);
    }
    int64_t n_rows() const { return row_at(n_seg); }

    /* The tokens that replace the media marker, and where in them each segment's placeholder run
     * starts. Placeholders are the checkpoint's own pad ids, because an n-gram embedding hashes
     * them like any other id. */
    std::vector<int32_t> expand;
    std::vector<int32_t> seg_at;
    /* Optional explicit placeholder positions within expand, one per encoder output row.
     * Pixtral uses this because each grid row ends with a separator token that keeps its own
     * language embedding. Empty preserves the existing contiguous per-segment representation. */
    std::vector<int32_t> row_tokens;

    /* Content key over the patches and the grid, taken before the encoder runs, so a prefix-cache
     * hit over this item skips the encoder as well as the prefill. The patches already reflect
     * every preprocessing choice, and the encoder is a function of them alone. */
    uint64_t key_hi = 0, key_lo = 0;

    /* [n_rows, n_embd] bf16, written by the engine after the encoder pass. */
    std::vector<uint16_t> embd;
    bool encoded = false;
    /* Handed to the engine's encoder queue and not yet encoded. Scheduler thread only. */
    bool queued = false;

    /* The (row, col, rows, cols) of each patch in its segment, for the encoder: component-major,
     * four planes of `count` entries, patches [first, first + count). */
    void coords(int64_t first, int64_t count, int32_t* planes) const;
};

/* The rotary layout of one prompt: which tokens are media and where each run's grid sits. Built
 * once when the prompt is tokenised. The batch builder calls `at` for every row of every pass,
 * so it is a binary search over runs rather than an array the length of the prompt. */
struct RopeLayout {
    struct Run {
        int32_t tok = 0;        /* first placeholder token of the run */
        int32_t n = 0;          /* its length: rows_h * rows_w */
        int32_t w = 0;          /* its width in rows */
        int32_t base = 0;       /* the rotary position of its first token */
        int32_t shift = 0;      /* rotary minus token index for the text after the run */
        int32_t item = 0;       /* which of PromptMedia::items the run belongs to */
        int32_t row0 = 0;       /* the item's encoder row that its first token takes */
        bool    flat = false;   /* Item::rope_1d: the run's tokens rotate at their positions */
    };
    std::vector<Run> runs;      /* ascending */

    bool empty() const { return runs.empty(); }

    /* The last run starting at or before token i, or null. Inline because the batch builder
     * calls it for every row of every pass. */
    const Run* last_at(int64_t i) const {
        size_t lo = 0, hi = runs.size();
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if ((int64_t)runs[mid].tok <= i) lo = mid + 1; else hi = mid;
        }
        return lo == 0 ? nullptr : &runs[lo - 1];
    }
    /* (t, h, w) of token index i. Past the prompt, text continues from the last run's shift. */
    void at(int64_t i, int32_t out[3]) const {
        const Run* r = last_at(i);
        if (!r) { out[0] = out[1] = out[2] = (int32_t)i; return; }
        const int64_t k = i - r->tok;
        if (k < r->n && !r->flat) {
            out[0] = r->base;
            out[1] = r->base + (int32_t)(k / r->w);
            out[2] = r->base + (int32_t)(k % r->w);
            return;
        }
        out[0] = out[1] = out[2] = (int32_t)(i + r->shift);
    }
    /* The run holding token i, or null when i is text. */
    const Run* run_at(int64_t i) const {
        const Run* r = last_at(i);
        return r && i - r->tok < r->n ? r : nullptr;
    }
};

/* One media item of a prompt and where its expansion begins. */
struct Placed {
    std::shared_ptr<Item> item;
    int32_t               tok = 0;   /* index of item->expand[0] in the prompt */
};

struct PromptMedia {
    std::vector<Placed> items;
    RopeLayout          rope;
    /* Build `rope` from `items`, which must be placed and in prompt order. */
    void layout();
};

class Processor {
public:
    /* `encode_text` tokenises the timestamp text between a video's groups: plain text, no
     * special tokens added or parsed. `token_id` returns the id of the vocabulary entry spelled
     * exactly so, or -1; it finds markers the config does not number (LFM2-VL's tile and
     * thumbnail markers). */
    int init(const VisionConfig& cfg, std::function<std::vector<int32_t>(const std::string&)> encode_text,
             std::function<int32_t(const std::string&)> token_id = {});
    const VisionConfig& config() const { return cfg_; }

    /* `image_tokens` is a Gemma 4 soft-token budget for this image (0: the configured one); a
     * processor that sizes images by pixels refuses a non-zero one. `max_pixels`, when set, first
     * scales a larger picture down to that area with PIL's bicubic. `mode` picks one of the
     * checkpoint's picture modes where it has more than one (Unlimited-OCR). */
    int image(const uint8_t* bytes, size_t n, int32_t image_tokens, std::shared_ptr<Item>* out,
              std::string* why, int64_t max_pixels = 0, const ImageMode& mode = ImageMode{}) const;
    int video(const uint8_t* bytes, size_t n, std::shared_ptr<Item>* out, std::string* why) const;

    /* image() from a decoded frame: a keyframe's canvas frame (IMultimodal::prepare_frame), or a
     * test's synthetic one. */
    int image_from(const Frame& f, std::shared_ptr<Item>* out, std::string* why,
                   const ImageMode& mode = ImageMode{}) const;
    int video_from(const Video& v, std::shared_ptr<Item>* out, std::string* why) const;

    /* Media for a video generator's VAE encoders (RadVideoGenDecl), decoded under `image`'s guard.
     * A reference picture, scaled with its aspect kept to sides that are multiples of `ratio`, the
     * longer at most `max_side`. */
    int encoder_image(const uint8_t* bytes, size_t n, int32_t ratio, int32_t max_side, Frame* out,
                      std::string* why) const;

    /* A keyframe, at the video's w x h: stretched, or with `cover` scaled to cover it and cropped
     * at the centre. */
    int encoder_image_canvas(const uint8_t* bytes, size_t n, int32_t w, int32_t h, bool cover,
                             Frame* out, std::string* why) const;

    /* A reference video: frames spread over the stream, at most `max_frames` and 5 modulo 17
     * (ComfyUI's trim), each at encoder_image's sides. */
    int encoder_video(const uint8_t* bytes, size_t n, int32_t ratio, int32_t max_side,
                      int32_t max_frames, std::vector<Frame>* out, std::string* why) const;

    /* A soundtrack: its first stream, mono at `rate` Hz, at most `max_samples`. */
    int encoder_audio(const uint8_t* bytes, size_t n, int32_t rate, int64_t max_samples,
                      std::vector<float>* out, std::string* why) const;

    /* Audio, when the program declares an audio encoder; `merge` and `max_frames` are the
     * encoder's. */
    int  init_audio(const AudioConfig& cfg, std::string* why = nullptr);
    bool has_audio() const { return audio_on_; }
    const AudioConfig& audio_config() const { return acfg_; }
    int audio(const uint8_t* bytes, size_t n, std::shared_ptr<Item>* out, std::string* why) const;
    /* From samples already decoded (mono at the configured rate); for tests. */
    int audio_from(const std::vector<float>& samples, std::shared_ptr<Item>* out,
                   std::string* why) const;

private:
    int image_gemma4(const Frame& f, int32_t soft, std::shared_ptr<Item>* out,
                     std::string* why) const;
    int image_lfm2(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const;
    int image_unlimitedocr(const Frame& f, const ImageMode& mode, std::shared_ptr<Item>* out,
                           std::string* why) const;
    int image_glm5next(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const;
    int image_pixtral(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const;
    int image_radio(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const;
    int image_muse(const Frame& f, std::shared_ptr<Item>* out, std::string* why) const;
    int video_radio(const Video& v, std::shared_ptr<Item>* out, std::string* why) const;
    /* Lfm2's markers: <|img_thumbnail|>, and <|img_row_R_col_C|> at (R-1) * max_tiles + C-1. */
    int32_t              tok_thumbnail_ = -1;
    std::vector<int32_t> tok_tiles_;
    int video_gemma4(const Video& v, std::shared_ptr<Item>* out, std::string* why) const;
    int video_muse(const Video& v, std::shared_ptr<Item>* out, std::string* why) const;

    VisionConfig cfg_{};
    AudioConfig  acfg_{};
    bool         audio_on_ = false;
    int audio_qwen3asr(const std::vector<float>& samples, std::shared_ptr<Item>* out,
                       std::string* why) const;
    int audio_unified(const std::vector<float>& samples, std::shared_ptr<Item>* out,
                      std::string* why) const;

    std::vector<float>  hann_;          /* the window, float32 as the reference stores it */
    std::vector<double> mel_;           /* [n_mels, fft/2 + 1]: filter b's weight on bin k */
    std::vector<int32_t> mel_lo_, mel_hi_;  /* filter b's nonzero bins, [lo, hi) */
    std::vector<double> tw_re_, tw_im_;     /* exp(-2 pi i k / fft), k < fft/2 */
    std::vector<int32_t> brev_;             /* the FFT's bit-reversal permutation */
    std::vector<int32_t> radix_;            /* a non-power-of-two FFT's factors, smallest first */
    std::function<std::vector<int32_t>(const std::string&)> encode_text_;
};

/* smart_resize, the image form: (height, width) -> multiples of `factor` in [min, max] pixels. */
int smart_resize_image(int64_t h, int64_t w, int32_t factor, int64_t min_px, int64_t max_px,
                       int64_t* oh, int64_t* ow, std::string* why);
/* ...and the video form, which counts the frames into the budget. */
int smart_resize_video(int64_t t, int64_t h, int64_t w, int32_t tfactor, int32_t factor,
                       int64_t min_px, int64_t max_px, int64_t* oh, int64_t* ow, std::string* why);
/* Which frames a video keeps. */
std::vector<int64_t> sample_frames(const VideoInfo& info, double fps, int32_t min_frames,
                                   int32_t max_frames);
/* ...the Gemma 4 rule: one frame each 1 / fps seconds, thinned evenly to `max_frames`. */
std::vector<int64_t> sample_frames_gemma4(const VideoInfo& info, double fps, int32_t max_frames);
/* Gemma 4's target size: the largest (h, w), both multiples of `side`, whose patch count stays
 * within `max_patches`, aspect ratio kept (get_aspect_ratio_preserving_size). */
int gemma4_size(int64_t h, int64_t w, int32_t patch, int32_t pool, int64_t max_patches,
                int64_t* oh, int64_t* ow, std::string* why);

/* MuseGlimmer's smart_resize (MuseGlimmer{Image,Video}Processor.smart_resize): the whole
 * merged-patch grid (`factor` = patch * merge pixels a side, so a grid cell is one output row)
 * under `max_tokens` merged tokens whose aspect is nearest the picture's own, in Python's
 * doubles. A picture over the cap scales to the cap first. */
int muse_size(int64_t h, int64_t w, int32_t factor, int64_t max_tokens, int64_t* oh, int64_t* ow,
              std::string* why);

/* MuseGlimmerVideoProcessor.sample_frames: `total * fps / native` frames, truncated, clamped to
 * [temporal, min(num_frames, total)] and floored to a whole temporal group, then torch's
 * linspace(0, total - 1, n) truncated to integers. A stream with no stated rate is taken at
 * 24 fps. */
std::vector<int64_t> sample_frames_muse(const VideoInfo& info, double fps, int32_t temporal,
                                        int32_t num_frames);

/* Radio's tile for a picture of w x h pixels (NVIDIA's _compute_target_patches, Python's
 * round-half-even included) and for a video frame (_compute_target_patches_video), in patches. */
void radio_size(int64_t w, int64_t h, const VisionConfig& c, int32_t* wp, int32_t* hp);
void radio_video_size(int64_t w, int64_t h, const VisionConfig& c, int32_t* wp, int32_t* hp);
/* torch's F.interpolate(mode="bicubic", align_corners=False, antialias=True) of packed RGB bytes
 * taken as float, without rounding or clamping: [h, w, 3] f32 in 0..255 give or take the
 * filter's overshoot. Separable, width first, as ATen's CPU kernel; a side that keeps its size is
 * the identity. */
void resize_bicubic_aa_f32(const Frame& in, int32_t w, int32_t h, std::vector<float>* out);
/* The Muse Glimmer reference's resize filter, torchvision's antialiased Lanczos on the uint8
 * pixels (see processor.cpp). */
void resize_lanczos_torch(const Frame& in, int32_t w, int32_t h, Frame* out);
/* PIL's antialiased bicubic resample of packed RGB, bit for bit: fixed-point coefficients, a
 * horizontal pass into an 8-bit intermediate and then a vertical one. */
void resize_bicubic(const Frame& in, int32_t w, int32_t h, Frame* out);
/* ...and torchvision's on a uint8 tensor (ATen's CPU kernel), which is the same filter with the
 * coefficients at the most precision 16 bits hold, chosen per axis, rather than PIL's 22. */
void resize_bicubic_torch(const Frame& in, int32_t w, int32_t h, Frame* out);
/* GLM-5.3's smart_resize: the (oh, ow) canvas, multiples of `factor`, for an h x w picture whose
 * `temporal` frames take [min_tok, max_tok] tokens of temporal * factor^2 pixels each; and the
 * (ch, cw) the picture is resized to inside it, top left, aspect ratio kept. */
int glm5next_size(int64_t h, int64_t w, int32_t temporal, int32_t factor, int64_t min_tok,
                  int64_t max_tok, int64_t* oh, int64_t* ow, int64_t* ch, int64_t* cw,
                  std::string* why);
/* LFM2-VL's smart_resize: (height, width) to multiples of `factor`, at least one each, pixels
 * within [min_px, max_px] where the rounding left them outside. */
void lfm2_size(int64_t h, int64_t w, int32_t factor, int64_t min_px, int64_t max_px,
               int64_t* oh, int64_t* ow);
/* LFM2-VL's test for a picture to tile: its sides rounded to whole merged patches cover more than
 * `max_px` times `tolerance` pixels. */
bool lfm2_too_large(int64_t h, int64_t w, int32_t patch, int32_t factor, int64_t max_px,
                    double tolerance);
/* A picture over `max_pixels` scaled down to that area with PIL's bicubic, sides truncated; any
 * other picture as it is. */
void cap_area(const Frame& in, int64_t max_pixels, Frame* out);
/* LFM2-VL's tile grid for a w x h picture: (columns, rows) of the min..max-tile ratio nearest its
 * aspect, the larger grid of two equally near when the picture fills over half of it. */
void lfm2_grid(int64_t h, int64_t w, int32_t min_tiles, int32_t max_tiles, int32_t tile,
               int32_t* cols, int32_t* rows);
/* Whether the reference's ratio set visits the tall side (c, r), c < r, of a transposed pair after
 * the wide one: the only tie two ratios of one tile count can have. From the reference itself. */
bool uocr_tall_last(int32_t c, int32_t r);
/* Unlimited-OCR's tile grid (its dynamic_preprocess): the min..max-tile grid nearest the picture's
 * aspect, the one whose area the picture covers over half of when two are equally near, exactly
 * as the reference's set ordering resolves it. The reference calls it only for a picture larger
 * than one tile; this answers for any size. */
void uocr_grid(int64_t h, int64_t w, int32_t min_tiles, int32_t max_tiles, int32_t tile,
               int32_t* cols, int32_t* rows);
/* The placeholders a picture takes: the global view's rows plus its separator, and, when tiled,
 * one line of `cols` tiles per feature row with a newline after it. `cols` * `rows` is the tile
 * grid. */
int64_t uocr_tokens(const VisionConfig& c, const ImageMode& mode, int32_t cols, int32_t rows);
/* The mode `mode` resolves to for this deployment; RAD_E_INVAL names an unsupported size. */
int uocr_mode(const VisionConfig& c, const ImageMode& mode, bool* crop, int32_t* base_size,
              int32_t* image_size, std::string* why);
/* Runs the resample and patch staging on their scalar path even where AVX2 is available, for a
 * test that compares the two; process-wide. */
void mm_force_scalar(bool on);
/* PIL's ImageOps.pad: `in` fitted inside w x h keeping its aspect (its bicubic), centred on `fill`
 * in the channels' order (127 127 127 for the reference's 0.5 mean). */
void pad_bicubic(const Frame& in, int32_t w, int32_t h, uint8_t fill, Frame* out);

}  /* namespace mm */
}  /* namespace rad */
