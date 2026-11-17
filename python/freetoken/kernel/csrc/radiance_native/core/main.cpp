/* main.cpp -- the entry point: turns a YAML file into engine.h's startup sequence and a status
 * into an exit code. The file's format is fleet/config.h (docs/MULTIMODEL.md §4).
 */
#include "engine.h"
#include "fleet/config.h"
#include "fleet/fleet.h"
#include "hf_model.h"
#include "util/stallwatch.h"

#include <cstring>
#include <string>

namespace {

/* The command line before 2.0 configured one model with flags. It is refused with the same
 * settings written as a file, so moving over is a copy. */
int refuse_flags(int argc, char** argv) {
    RAD_ERR("radiance takes its settings from a YAML file: radiance --config FILE "
            "(radiance --help lists the keys). This command line as a file:");
    fprintf(stderr, "\n%s\n", rad::fleet::yaml_from_args(argc, argv).c_str());
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string path;
    bool check = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            rad::fleet::print_reference(stdout);
            return 0;
        }
        if (!std::strcmp(argv[i], "--check")) { check = true; continue; }
        if (!std::strcmp(argv[i], "--config") && i + 1 < argc && path.empty()) { path = argv[++i]; continue; }
        return refuse_flags(argc, argv);
    }
    if (path.empty()) {
        if (argc > 1) return refuse_flags(argc, argv);
        rad::fleet::print_reference(stderr);
        return 2;
    }

    rad::fleet::FleetConfig fleet;
    if (rad::fleet::load(path, &fleet) != RAD_OK) return 2;
    if (fleet.log_set) rad::log_set_level(fleet.log);

    /* Keys the file may set that this build does not act on yet are refused rather than
     * ignored. */
    if (const std::string u = rad::fleet::unsupported(fleet); !u.empty()) {
        RAD_ERR("%s: this build does not act on %s yet; remove them", path.c_str(), u.c_str());
        return 2;
    }
    if (check) {
        for (const rad::fleet::ModelEntry& m : fleet.models)
            RAD_INFO("%s: model %s = %s, tp %d", path.c_str(), m.name.c_str(), m.cfg.model.c_str(),
                     m.cfg.tp);
        return 0;
    }

    /* An offline mode (KL scoring, calibration, the diagnose workload) or the graph dump runs one
     * engine to its end and serves nothing; every other configuration is served by the fleet. */
    bool offline = false;
    for (const rad::fleet::ModelEntry& m : fleet.models)
        offline |= m.cfg.kld() || m.cfg.calib() || m.cfg.diagnose_mode() || m.cfg.debug_graph;
    if (!offline) return rad::fleet::run(fleet);
    if (fleet.models.size() != 1) {
        RAD_ERR("%s: an offline mode or the graph dump runs one model; the file names %zu",
                path.c_str(), fleet.models.size());
        return 2;
    }

    rad::Config cfg = std::move(fleet.models[0].cfg);
    if (cfg.proceed_incompatible) rad::Engine::print_unsupported_banner();

    if (!rad::resolve_hf_model(cfg)) return 1;   /* a repo id: fetched, and `model` points at it */

    rad::Engine engine(std::move(cfg));

    /* A startup that stops logging reports where it waits (util/stallwatch.h). Two minutes
     * comfortably covers the slowest normal step, such as a container read from a slow disk. The
     * watch goes on into serve() and ends when the server listens. */
    auto watch = std::make_unique<rad::StallWatch>("startup", 120000);
    int s = engine.start();
    if (s < 0 || engine.config().debug_graph) watch.reset();
    if (s < 0) {
        /* start() logged the error; the report adds the machine and the decisions made so far. */
        engine.print_startup_report(s);
        return 1;
    }

    if (engine.config().debug_graph) return engine.dump();
    s = engine.serve(std::move(watch));
    if (s < 0) {
        /* serve() can also refuse before listening (e.g. a template default the template
         * rejects, an unholdable thinking budget). */
        if (!engine.startup_report_printed()) engine.print_startup_report(s);
        return 1;
    }
    return 0;
}
