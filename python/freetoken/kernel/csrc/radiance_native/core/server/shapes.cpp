/* shapes.cpp -- Hunyuan3D-2's API over a shape generator (core/gen/shape_driver.h): a 3-D shape
 * from one picture.
 *
 * The routes and fields are the reference's own API server's (hy3dgen's api_server.py), so its
 * clients (its Blender add-on, scripts written against it) work unchanged:
 *
 *   POST /generate      the shape, made while the request waits, as the file itself;
 *   POST /send          the same in the background, answered at once with {"uid"};
 *   GET  /status/{uid}  {"status": "processing"}, or "completed" with the file as
 *                       `model_base64`, or "error" with its `message`.
 *
 * The body is JSON: `image`, the picture as base64 (or a data URL), with an alpha channel for a
 * cut-out object -- the reference removes the background of an opaque picture with rembg, which
 * this server does not, so an opaque picture is taken whole; `seed` (1234 when absent, as the
 * reference), `octree_resolution`, `num_inference_steps` and `guidance_scale` (the model's own
 * when absent), and `type`, the file: glb (the default), obj, ply or stl. `volume_decoding`
 * "dense" samples every point of the grid instead of refining it coarse to fine. Texturing
 * (`texture` true) and the reference's text-to-image and re-texturing inputs (`text`, `mesh`)
 * are refused. The reference's knobs that change nothing here are taken and ignored: `face_count`
 * (texturing's face budget), `num_chunks` (its decoder's batch) and `mc_algo` "mc"; any other
 * field is refused by name. A finished background shape is kept for as long as a video, as many
 * as videos.
 */
#include "front.h"
#include "server.h"
#include "oai.h"

#include "gen/shape.h"
#include "mm/media.h"
#include "rad_internal.h"

#include "util/json.h"

#include <chrono>
#include <cmath>
#include <thread>

namespace rad {
namespace server {

namespace {

/* The largest picture decoded, as the image endpoints bound theirs. */
constexpr int64_t kMaxSourcePixels = (int64_t)1 << 26;

/* A request as the routes read it, and its file's format. */
struct ShapeAsk {
    std::unique_ptr<gen::ShapeRequest> req;
    std::string type = "glb";
};

int read_ask(const HttpRequest& q, const OaiLimits& lim, const gen::ShapeGenInfo& info,
             ShapeAsk* out, ApiError* err) {
    json b;
    if (parse_request_object(q.body, lim, &b, err) < 0) return RAD_E_INVAL;
    auto bad = [&](const std::string& p, const std::string& why) {
        *err = err_bad(p, why);
        return RAD_E_INVAL;
    };
    static const char* const kFields[] = { "image", "seed", "octree_resolution",
                                           "num_inference_steps", "guidance_scale", "type",
                                           "volume_decoding", "texture", "text", "mesh",
                                           "face_count", "num_chunks", "mc_algo" };
    for (auto it = b.begin(); it != b.end(); ++it) {
        bool known = false;
        for (const char* k : kFields) known = known || it.key() == k;
        if (!known)
            return bad(it.key(), "'" + it.key() + "' is not a field of /generate");
    }
    if (b.contains("mc_algo") && !b["mc_algo"].is_null() &&
        !(b["mc_algo"].is_string() && b["mc_algo"].get<std::string>() == "mc")) {
        *err = err_unsupported("mc_algo", "the surface is marching cubes ('mc'); differentiable "
                                          "marching cubes ('dmc') is not served");
        return RAD_E_INVAL;
    }
    for (const char* k : { "text", "mesh" })
        if (b.contains(k) && !b[k].is_null()) {
            *err = err_unsupported(k, std::string("this server makes a shape from a picture; '") +
                                          k + "' (the reference's text-to-image and re-texturing "
                                          "inputs) is not served");
            return RAD_E_INVAL;
        }
    if (b.contains("texture") && b["texture"].is_boolean() && b["texture"].get<bool>()) {
        *err = err_unsupported("texture", "texturing (Hunyuan3D-Paint) is not served: this "
                                          "server makes the untextured shape");
        return RAD_E_INVAL;
    }
    if (!b.contains("image") || !b["image"].is_string() || b["image"].get<std::string>().empty())
        return bad("image", "'image' must be the picture as base64");
    std::string img = b["image"].get<std::string>();
    if (img.rfind("data:", 0) == 0) {
        const size_t comma = img.find(',');
        if (comma == std::string::npos || img.compare(comma - 7, 7, ";base64") != 0)
            return bad("image", "a data URL must be base64 ('data:<type>;base64,...')");
        img = img.substr(comma + 1);
    }
    std::vector<uint8_t> bytes;
    if (!base64_decode(img, &bytes) || bytes.empty())
        return bad("image", "'image' is not valid base64");
    mm::FrameRGBA f;
    std::string why;
    if (mm::decode_image_rgba(bytes.data(), bytes.size(), kMaxSourcePixels, &f, &why) < 0)
        return bad("image", "the picture could not be decoded: " + why);

    auto r = std::make_unique<gen::ShapeRequest>();
    if (gen::shape_picture(info, f, &r->patches, &why) < 0) return bad("image", why);
    auto int_field = [&](const char* k, int64_t lo, int64_t hi, int32_t* v) {
        if (!b.contains(k) || b[k].is_null()) return true;
        if (!b[k].is_number_integer() || b[k].get<int64_t>() < lo || b[k].get<int64_t>() > hi) {
            bad(k, std::string("'") + k + "' must be an integer " + std::to_string(lo) + ".." +
                       std::to_string(hi));
            return false;
        }
        *v = (int32_t)b[k].get<int64_t>();
        return true;
    };
    if (!int_field("num_inference_steps", 2, info.max_steps, &r->steps) ||
        !int_field("octree_resolution", info.min_resolution, info.max_resolution,
                   &r->resolution))
        return RAD_E_INVAL;
    if (b.contains("guidance_scale") && !b["guidance_scale"].is_null()) {
        if (!b["guidance_scale"].is_number() || !(b["guidance_scale"].get<double>() >= 0.0) ||
            !std::isfinite(b["guidance_scale"].get<double>()))
            return bad("guidance_scale", "'guidance_scale' must be a finite number, not negative");
        r->guidance = b["guidance_scale"].get<double>();
    }
    r->seed = 1234;
    if (b.contains("seed") && !b["seed"].is_null()) {
        if (!b["seed"].is_number_integer())
            return bad("seed", "'seed' must be an integer");
        r->seed = b["seed"].is_number_unsigned() ? b["seed"].get<uint64_t>()
                                                 : (uint64_t)b["seed"].get<int64_t>();
    }
    if (b.contains("volume_decoding") && !b["volume_decoding"].is_null()) {
        const std::string v = b["volume_decoding"].is_string()
                                  ? b["volume_decoding"].get<std::string>() : "";
        if (v != "hierarchical" && v != "dense")
            return bad("volume_decoding", "'volume_decoding' is hierarchical or dense");
        r->dense = v == "dense";
    }
    if (b.contains("type") && !b["type"].is_null()) {
        if (!b["type"].is_string()) return bad("type", "'type' is glb, obj, ply or stl");
        out->type = b["type"].get<std::string>();
        if (out->type != "glb" && out->type != "obj" && out->type != "ply" && out->type != "stl")
            return bad("type", "type '" + out->type + "' is not one this server writes: glb, obj, "
                               "ply or stl");
    }
    r->ticket = std::make_shared<gen::ShapeTicket>();
    out->req = std::move(r);
    return RAD_OK;
}

/* The ticket's field as a mesh file, once the driver is done with it. */
int finish_file(gen::ShapeTicket& t, const gen::ShapeGenInfo& info, const std::string& type,
                std::string* file, std::string* mime, std::string* why) {
    if (t.status() < 0) {
        *why = t.why();
        return t.status();
    }
    int32_t r = 0;
    std::vector<float> field = t.take_field(&r);
    gen::ShapeMesh mesh;
    gen::shape_mesh(field, r, info.bounds, info.level, &mesh);
    std::vector<float>().swap(field);
    if (mesh.f.empty()) {
        *why = "the model's field crosses the surface level nowhere in the grid: the picture made "
               "no shape";
        return RAD_E_INVAL;
    }
    return gen::shape_export(mesh, type, file, mime, why);
}

}  /* namespace */

struct Server::ShapeJob {
    std::string uid, type;
    int64_t     created_at = 0, done_at = 0;
    std::shared_ptr<gen::ShapeTicket> ticket;
    std::thread worker;
    /* Under Server::shapes_m_ from here on. */
    bool        finished = false, failed = false;
    std::string file, message;
};

void Server::shape_caps(json& caps, const IShapeGen* gen) {
    const gen::ShapeGenInfo& info = gen->info();
    caps["shape_max_steps"] = info.max_steps;
    caps["shape_resolutions"] = { info.min_resolution, info.max_resolution };
    caps["shape_formats"] = json::array({ "glb", "obj", "ply", "stl" });
}

HttpResponse Server::handle_shape_generate(const HttpRequest& req) {
    if (!deps_.shapes)
        return error_response(err_not_implemented(
            "/generate requires a 3-D shape model; this one does not make shapes"), req.peer);
    const gen::ShapeGenInfo& info = deps_.shapes->info();
    ShapeAsk a;
    ApiError err;
    if (read_ask(req, lim_, info, &a, &err) < 0) return error_response(err, req.peer);
    std::shared_ptr<gen::ShapeTicket> t = a.req->ticket;
    std::string why;
    if (const int st = deps_.shapes->submit(std::move(a.req), &why); st < 0)
        return error_response(st == RAD_E_INVAL ? err_bad("image", why) : err_overloaded(why),
                              req.peer);
    while (!t->wait(500))
        if (stopping_.load()) {
            t->cancel.store(true);
            return error_response(err_overloaded("the server is stopping"), req.peer);
        }
    std::string file, mime;
    if (const int st = finish_file(*t, info, a.type, &file, &mime, &why); st < 0)
        return error_response(st == RAD_E_INVAL ? err_bad("image", why) : err_internal(why),
                              req.peer);
    HttpResponse res;
    res.content_type = mime;
    res.body = std::move(file);
    return res;
}

void Server::shape_work(std::shared_ptr<ShapeJob> j) {
    const gen::ShapeGenInfo& info = deps_.shapes->info();
    gen::ShapeTicket& t = *j->ticket;
    while (!t.wait(500))
        if (stopping_.load()) {
            t.cancel.store(true);
            t.wait(60000);
            break;
        }
    std::string file, mime, why;
    const int st = t.finished() ? finish_file(t, info, j->type, &file, &mime, &why)
                                : RAD_E_CANCELLED;
    std::lock_guard<std::mutex> lk(shapes_m_);
    j->finished = true;
    j->done_at = unix_now();
    if (st < 0) {
        j->failed = true;
        j->message = why.empty() ? std::string(rad_strerror(st)) : why;
    } else {
        j->file = std::move(file);
    }
    /* A finished shape can be read without this server, which may be gone when its client asks. */
    if (deps_.front)
        deps_.front->keep_job(j->uid, j->done_at + opt_.video_ttl_s, opt_.max_videos,
                              [j](const HttpRequest&) -> HttpResponse {
            return HttpResponse::json_body(200, json_dump(shape_status(*j)));
        });
}

json Server::shape_status(const ShapeJob& j) {
    json o;
    if (!j.finished) {
        o["status"] = "processing";
        int32_t steps = 0, points = 0;
        j.ticket->progress_of(&steps, &points);
        o["steps_done"] = steps;
        o["steps"] = j.ticket->steps;
    } else if (j.failed) {
        o["status"] = "error";
        o["message"] = j.message;
    } else {
        o["status"] = "completed";
        o["type"] = j.type;
        o["model_base64"] = base64_encode(j.file.data(), j.file.size());
    }
    return o;
}

void Server::sweep_shapes_locked(int64_t now) {
    for (size_t i = 0; i < shapes_.size();) {
        ShapeJob& j = *shapes_[i];
        const bool expired = j.finished && now - j.done_at > opt_.video_ttl_s;
        if (!expired) { ++i; continue; }
        if (j.worker.joinable()) j.worker.join();
        shapes_.erase(shapes_.begin() + (std::ptrdiff_t)i);
    }
    /* Past the cap the oldest finished shapes go first. */
    size_t done = 0;
    for (const auto& j : shapes_) done += j->finished;
    const size_t cap = (size_t)std::max<int64_t>(opt_.max_videos, 1);
    for (size_t i = 0; done > cap && i < shapes_.size();) {
        if (!shapes_[i]->finished) { ++i; continue; }
        if (shapes_[i]->worker.joinable()) shapes_[i]->worker.join();
        shapes_.erase(shapes_.begin() + (std::ptrdiff_t)i);
        --done;
    }
}

void Server::stop_shapes() {
    std::vector<std::shared_ptr<ShapeJob>> all;
    {
        std::lock_guard<std::mutex> lk(shapes_m_);
        all = shapes_;
    }
    for (auto& j : all) j->ticket->cancel.store(true);
    for (auto& j : all)
        if (j->worker.joinable()) j->worker.join();
}

HttpResponse Server::handle_shape_send(const HttpRequest& req) {
    if (!deps_.shapes)
        return error_response(err_not_implemented(
            "/send requires a 3-D shape model; this one does not make shapes"), req.peer);
    const gen::ShapeGenInfo& info = deps_.shapes->info();
    ShapeAsk a;
    ApiError err;
    if (read_ask(req, lim_, info, &a, &err) < 0) return error_response(err, req.peer);
    auto j = std::make_shared<ShapeJob>();
    j->uid = random_id("shape_");
    j->type = a.type;
    j->created_at = unix_now();
    j->ticket = a.req->ticket;
    std::string why;
    if (const int st = deps_.shapes->submit(std::move(a.req), &why); st < 0)
        return error_response(st == RAD_E_INVAL ? err_bad("image", why) : err_overloaded(why),
                              req.peer);
    {
        std::lock_guard<std::mutex> lk(shapes_m_);
        sweep_shapes_locked(unix_now());
        shapes_.push_back(j);
    }
    if (deps_.background) deps_.background->fetch_add(1);
    j->worker = std::thread([this, j] {
        shape_work(j);
        if (deps_.background) deps_.background->fetch_sub(1);
    });
    json o;
    o["uid"] = j->uid;
    return HttpResponse::json_body(200, json_dump(o));
}

HttpResponse Server::handle_shape_status(const HttpRequest& req) {
    if (!deps_.shapes)
        return error_response(err_not_implemented(
            "/status requires a 3-D shape model; this one does not make shapes"), req.peer);
    const std::string uid = req.param("uid");
    std::lock_guard<std::mutex> lk(shapes_m_);
    sweep_shapes_locked(unix_now());
    for (const auto& j : shapes_) {
        if (j->uid != uid) continue;
        return HttpResponse::json_body(200, json_dump(shape_status(*j)));
    }
    ApiError e = err_bad("uid", "no shape '" + uid + "'; it may have expired");
    e.status = 404;
    e.type = "not_found_error";
    return error_response(e, req.peer);
}

}  /* namespace server */
}  /* namespace rad */
