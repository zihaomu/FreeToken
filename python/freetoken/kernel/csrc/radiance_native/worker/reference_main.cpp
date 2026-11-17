// Same Engine and memory owner as the worker, with Radiance's own HTTP frontend.
#include "engine.h"
#include "device/device.h"
#include "fleet/config.h"
#include "server/front.h"
#include "util/stallwatch.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {
std::atomic<bool> stopped{false};
void stop(int) { stopped.store(true, std::memory_order_relaxed); }
}

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) != "--config") {
        std::fprintf(stderr, "usage: freetoken-native-reference --config FILE\n");
        return 2;
    }
    try {
        rad::fleet::FleetConfig fleet;
        if (rad::fleet::load(argv[2], &fleet) < 0 || fleet.models.size() != 1)
            throw std::runtime_error("one model configuration is required");
        auto cfg = fleet.models[0].cfg;
        const char* expected = std::getenv("FREETOKEN_NATIVE_EXPECTED_PCIS");
        if (!expected) throw std::runtime_error("reserved GPU PCI vector is required");
        const auto pcis = rad::Json::parse(expected).get<std::vector<std::string>>();
        if (pcis.size() != size_t(cfg.tp) || rad_dev_count() != cfg.tp)
            throw std::runtime_error("visible device count differs from reservation");
        for (int i = 0; i < cfg.tp; ++i) {
            auto actual = rad::dev_pci_address(i), want = pcis[size_t(i)];
            std::transform(actual.begin(), actual.end(), actual.begin(), ::tolower);
            std::transform(want.begin(), want.end(), want.begin(), ::tolower);
            if (actual != want) throw std::runtime_error("GPU PCI differs from reservation");
        }
        rad::server::TransportOptions transport;
        transport.host = cfg.host;
        transport.port = cfg.port;
        transport.n_threads = rad::server::idle_workers(cfg.http_threads);
        rad::server::Front front(transport, 0);
        front.add(cfg.served_model_name, {}, 600000, true);
        rad::Engine engine(std::move(cfg));
        engine.set_front(&front, {});
        front.route("GET", "/freetoken/native-stats", [&](const rad::server::HttpRequest&) {
            return rad::server::HttpResponse::json_body(200, engine.headless_stats().dump());
        });
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        if (engine.start() < 0) return 1;
        std::atomic<bool> done{false};
        std::thread watcher([&] {
            while (!done.load() && !stopped.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (stopped.load()) { engine.request_stop(10000); front.stop(); }
        });
        int result = 1;
        std::thread model([&] { result = engine.serve({}); front.stop(); });
        const int listener = front.run();
        engine.request_stop(10000);
        model.join();
        done.store(true);
        watcher.join();
        return result < 0 || listener < 0 ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native reference: %s\n", error.what());
        return 1;
    }
}
