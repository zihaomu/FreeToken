/* shape.h -- shape generation: the request, the channel its field comes back on, and the host
 * geometry around the model's passes.
 *
 * A shape generator (abi/rad_builder.h, rad_declare_shape_generator) makes a 3-D shape from one
 * picture: it denoises a latent, then decodes it into an occupancy field that is sampled on a
 * grid. core/gen/shape_driver.h turns a request into those passes; this is what a caller hands it
 * (the picture already prepared, shape_picture), what it hands back (progress as the steps and
 * the grid's levels run, then the finest grid), and the geometry both sides share: the sampler's
 * sigmas, the grid's levels and the points each level queries, and the mesh made of the grid
 * (shape_mesh) with its file formats.
 */
#pragma once
#include "rad_core.h"
#include "mm/media.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace gen {

/* What the driver knows of the declaration, for the server and the geometry below. */
struct ShapeGenInfo {
    std::string name;
    int64_t canvas = 0, enc_size = 0, enc_patch = 0;
    double  border_ratio = 0.0;
    double  pix_mean[3] = {}, pix_std[3] = {};
    int64_t default_steps = 0, max_steps = 0;
    double  default_guidance = 0.0;
    double  bounds = 1.0, level = 0.0;
    int64_t default_resolution = 0, min_resolution = 0, max_resolution = 0;
    int64_t max_lanes = 1;
};
ShapeGenInfo shape_info(const RadShapeGenDecl& d, const std::string& name);

/* The picture as an SCOND pass carries it (RadShapeGenDecl, "The picture"): `rgba` decoded with
 * its alpha, out as f32 [(enc_size / enc_patch)^2, 3 enc_patch^2]. Refuses a picture whose alpha
 * leaves nothing, or a bounding box of one row or column, as the reference does. */
int shape_picture(const ShapeGenInfo& g, const mm::FrameRGBA& rgba, std::vector<float>* patches,
                  std::string* why);

/* The sampler's grid points: numpy's linspace(0, 1, steps) taken to f32. An evaluation at each
 * but the last steps to the next. */
std::vector<float> shape_sigmas(int32_t steps);

/* The grid's levels, coarsest first: `resolution` halved while it stays at least 63, then
 * reversed (one level when it is below 63). */
std::vector<int32_t> shape_levels(int32_t resolution);

/* One axis of a level's grid: numpy's linspace(-bounds, bounds, r + 1) in f32. */
std::vector<float> shape_axis(double bounds, int32_t r);

/* The grid points of level `r` (index (i, j, k) at (i (r+1) + j) (r+1) + k) to query, given the
 * coarser level's field `prev` at resolution `rp` (NaN where it was not sampled): the coarse
 * points whose sign against `level` differs from a neighbour's, or whose value is within 0.95 of
 * zero, dilated once unless this is the last level, put at their places on this grid and dilated
 * twice there (once when the coarse set was). As FlashVDM's hierarchical decoding. */
std::vector<uint32_t> shape_refine(const std::vector<float>& prev, int32_t rp, int32_t r,
                                   double level, bool last);

/* A triangle mesh: positions xyz, and faces as three vertex indices, counter-clockwise seen from
 * outside the shape. */
struct ShapeMesh {
    std::vector<float>    v;
    std::vector<uint32_t> f;
};

/* Marching cubes over `field` ((r+1)^3 values, the layout above) at `level`, inside above it. A
 * cube with a corner not sampled (NaN) makes nothing. Vertices sit on the grid's edges where the
 * linear interpolation of their ends crosses the level, one vertex an edge, and grid index g goes
 * to g 2 bounds / (r + 1) - bounds (RadShapeGenDecl, "The field"). */
void shape_mesh(const std::vector<float>& field, int32_t r, double bounds, double level,
                ShapeMesh* out);

/* The mesh as a file: "glb" (positions, smooth normals and indices), "obj", "ply" (binary) or
 * "stl" (binary). The content type goes to `mime`. Refuses another format by name. */
int shape_export(const ShapeMesh& m, const std::string& format, std::string* bytes,
                 std::string* mime, std::string* why);

/* ================================================================== the request */
class ShapeTicket {
public:
    /* driver side */
    void progress(int32_t steps_done, int32_t points_done);
    /* The finest grid, at resolution `r`, and the end. */
    void finish(int status, const std::string& why = std::string());
    void set_field(std::vector<float>&& field, int32_t r);

    /* caller side: waits at most `ms` milliseconds for the end; true once it came. */
    bool wait(int ms);
    bool finished() const;
    int  status() const;
    std::string why() const;
    /* The field, once finished with RAD_OK; moved out. */
    std::vector<float> take_field(int32_t* r);
    void progress_of(int32_t* steps_done, int32_t* points_done) const;

    std::atomic<bool> cancel{false};

    /* Written by the driver at submit: the evaluations, the resolution and the guidance. */
    int32_t steps = 0, resolution = 0;
    double  guidance = 0.0;
    /* Usage, written by the driver before finish(): the points the decoder was queried at, and
     * per-phase times. */
    int64_t points = 0;
    int64_t t_submit_ns = 0, t_cond_ns = 0, t_denoised_ns = 0, t_done_ns = 0;

private:
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    bool                    done_ = false;
    int                     status_ = RAD_OK;
    std::string             why_;
    int32_t                 steps_done_ = 0, points_done_ = 0;
    std::vector<float>      field_;
    int32_t                 r_ = 0;
};

struct ShapeRequest {
    /* The picture, prepared (shape_picture). */
    std::vector<float> patches;
    int32_t  steps = 0;            /* sigma grid points; 0: the model's default */
    double   guidance = NAN;       /* NaN: the model's default */
    int32_t  resolution = 0;       /* the finest grid's cells a side; 0: the model's default */
    bool     dense = false;        /* sample every point of the finest grid, no coarser levels */
    uint64_t seed = 0;
    std::shared_ptr<ShapeTicket> ticket;
};

}  /* namespace gen */
}  /* namespace rad */
