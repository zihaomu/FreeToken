/* imaging.h -- image arithmetic done the way PIL does it, so a picture reaches the model exactly
 * as the reference pipeline's would.
 *
 * Resize: PIL's Image.resize with LANCZOS, bit for bit. Per-axis coefficients are computed in
 * double (support 3x the downscale, normalised to sum one) and converted to 22-bit fixed point;
 * a horizontal then a vertical pass run over 8-bit values, rounding and clipping each output.
 * RGBA is premultiplied by alpha first and divided back after, as PIL does. Diffusers'
 * VaeImageProcessor resizes this way.
 *
 * Composite: Image.paste(picture, mask=alpha) onto white, each channel
 * (c * a + 255 * (255 - a)) / 255 with PIL's rounding. The reference flattens a transparent
 * picture this way for an RGB text encoder. */
#pragma once
#include "mm/media.h"

namespace rad {
namespace mm {

/* `src` resized to w x h. */
void resize_lanczos_rgba(const FrameRGBA& src, int32_t w, int32_t h, FrameRGBA* out);
/* The same for RGB: PIL's LANCZOS on the RGB image, which is the RGBA one at alpha 255. */
void resize_lanczos_rgb(const Frame& src, int32_t w, int32_t h, Frame* out);
/* The same with PIL's BILINEAR filter (a triangle of support 1, widened when shrinking). */
void resize_bilinear_rgba(const FrameRGBA& src, int32_t w, int32_t h, FrameRGBA* out);
/* `src` over an opaque white ground. */
void composite_white(const FrameRGBA& src, Frame* out);

}  /* namespace mm */
}  /* namespace rad */
