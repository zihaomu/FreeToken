#include "protocol.h"
#include "engine.h"
#include "device/device.h"
#include "fleet/config.h"
#include "format/radfile.h"
#include "server/sink.h"
#include "text/tokenizer.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <memory>
#include <poll.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

using rad::Json;
namespace ft_native {
uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

bool transfer(int fd, void* data, size_t bytes, bool sending) {
    auto* p = static_cast<char*>(data);
    while (bytes) {
        const ssize_t n = sending ? send(fd, p, bytes, MSG_NOSIGNAL) : recv(fd, p, bytes, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        p += n;
        bytes -= size_t(n);
    }
    return true;
}

bool emit(int fd, Kind k, uint64_t request, uint64_t epoch, const void* data, size_t bytes) {
    if (bytes > max_payload) return false;
    Header h;
    h.kind = k; h.request = request; h.epoch = epoch; h.bytes = uint32_t(bytes);
    return transfer(fd, &h, sizeof(h), true) && transfer(fd, const_cast<void*>(data), bytes, true);
}

bool emit_json(int fd, Kind k, uint64_t request, uint64_t epoch, const Json& j) {
    const auto s = j.dump();
    return emit(fd, k, request, epoch, s.data(), s.size());
}

Json inspect(const char* path) {
    rad::RadFile f;
    if (f.open(path) < 0) throw std::runtime_error("cannot open .rad container");
    Json j;
    j["backend"] = "radiance-native";
    j["protocol_version"] = version;
    j["model_path"] = path;
    j["model_bytes"] = f.size();
    j["metadata"] = Json::object();
    for (int64_t i = 0; i < f.meta_count(); ++i) {
        const auto& kv = f.meta_at(i);
        const std::string key(f.str(kv.key));
        if (kv.type == RAD_P_INT) j["metadata"][key] = kv.v.i;
        else if (kv.type == RAD_P_STR) j["metadata"][key] = f.str(kv.v.s);
        else if (kv.type == RAD_KV_F64) j["metadata"][key] = kv.v.f;
    }
    j["entries"] = f.entry_count();
    j["encodings"] = f.encoding_count();
    j["expert_groups"] = f.group_count();
    if (std::string(f.get_s("model_type", "")) != "qwen4_exp")
        throw std::runtime_error("native backend currently supports Qwen4Exp containers only");
    j["capabilities"] = {{"text", true}, {"kv_i8", true}, {"qwen_kv_fp8", true},
                          {"mtp_depths", std::vector<int>{0, 3}}, {"max_running_requests", 1}};
    return j;
}

Json tokenize(const char* path, const char* text_path) {
    rad::RadFile f;
    if (f.open(path) < 0) throw std::runtime_error("cannot open .rad container");
    rad::VocabBuild build;
    if (!f.vocab() || rad::vocab_deserialize(f.base(), f.size(), f.vocab().h, build) < 0)
        throw std::runtime_error("cannot read native vocabulary");
    auto vocab = std::make_shared<rad::Vocab>();
    if (vocab->load(std::move(build)) < 0) throw std::runtime_error("cannot load native vocabulary");
    rad::Tokenizer tok;
    if (tok.init(vocab) < 0) throw std::runtime_error("cannot initialize native tokenizer");
    std::ifstream file(text_path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read tokenization input");
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<int32_t> ids;
    if (tok.encode(text, ids, false, true) < 0) throw std::runtime_error("native tokenization failed");
    return {{"ids", ids}, {"n_vocab", vocab->n_tokens()}};
}

struct Active {
    Header header;
    uint64_t native_id = 0;
    std::unique_ptr<rad::server::Sink> sink;
    std::unique_ptr<rad::Detokenizer> detok;
    std::vector<int32_t> stop_ids;
    std::string matched_stop;
    uint64_t sent = 0;
    uint64_t accepted_sent = 0;
    bool admitted = false;
    bool stopping = false;
};

int run(int fd, rad::Scheduler& sched, std::shared_ptr<const rad::Vocab> vocab, const Json& facts,
        const std::function<Json()>& snapshot) {
    std::unique_ptr<Active> active;
    uint64_t next_id = 1;
    bool connected = true;
    auto error = [&](const Header& h, const std::string& why) {
        return emit_json(fd, Error, h.request, h.epoch, {{"error", why}});
    };
    try {
        if (!emit_json(fd, Ready, 0, 0, {{"engine", facts}, {"pid", int(getpid())},
                       {"eog_token_ids", vocab->eog()},
                       {"draft_accounting", "committed token flags after stop and length truncation"},
                       {"protocol_version", version}, {"clock", "CLOCK_MONOTONIC/ns"}})) return RAD_E_IO;
        while (connected) {
            pollfd pfd{fd, POLLIN, 0};
            const int pr = poll(&pfd, 1, active ? 0 : 100);
            if (pr < 0 && errno == EINTR) continue;
            if (pr < 0 || (pfd.revents & (POLLHUP | POLLERR | POLLNVAL))) break;
            if (pfd.revents & POLLIN) {
                Header h;
                if (!transfer(fd, &h, sizeof(h), false)) break;
                if (h.magic_value != magic || h.protocol_version != version ||
                    h.reserved != 0 || h.bytes > max_payload) throw std::runtime_error("invalid IPC frame");
                std::vector<char> payload(h.bytes);
                if (!transfer(fd, payload.data(), payload.size(), false)) break;
                if (h.kind == Shutdown) break;
                if (h.kind == Cancel) {
                    if (active && active->header.request == h.request && active->header.epoch == h.epoch) {
                        active->sink->request_cancel();
                        sched.cancel(active->native_id);
                    }
                } else if (h.kind == Generate) {
                    if (active) { connected = error(h, "batch-1 worker is busy"); continue; }
                    try {
                        if (payload.size() < 4) throw std::runtime_error("short generate payload");
                        uint32_t nmeta;
                        std::memcpy(&nmeta, payload.data(), 4);
                        if (nmeta > payload.size() - 4 || (payload.size() - 4 - nmeta) % 4)
                            throw std::runtime_error("invalid prompt layout");
                        const auto m = Json::parse(std::string(payload.data() + 4, nmeta));
                        auto a = std::make_unique<Active>();
                        a->header = h; a->native_id = next_id++;
                        a->stop_ids = m.value("stop_token_ids", std::vector<int32_t>{});
                        a->detok = std::make_unique<rad::Detokenizer>(vocab,
                            m.value("stop", std::vector<std::string>{}), true);
                        auto req = std::make_unique<rad::Request>();
                        req->id = a->native_id;
                        req->prompt.resize((payload.size() - 4 - nmeta) / 4);
                        std::memcpy(req->prompt.data(), payload.data() + 4 + nmeta, req->prompt.size() * 4);
                        req->max_tokens = m.value("max_tokens", 256);
                        if (req->max_tokens < 1) throw std::runtime_error("max_tokens must be positive");
                        req->ignore_eos = m.value("ignore_eos", false);
                        req->sp.temp = m.value("temperature", 0.0f);
                        req->sp.top_k = m.value("top_k", 0);
                        req->sp.top_p = m.value("top_p", 1.0f);
                        req->sp.min_p = m.value("min_p", 0.0f);
                        req->sp.rep_penalty = m.value("repetition_penalty", 1.0f);
                        req->sp.freq_penalty = m.value("frequency_penalty", 0.0f);
                        req->sp.pres_penalty = m.value("presence_penalty", 0.0f);
                        req->sp.seed = m.value("seed", uint64_t(0));
                        a->sink = std::make_unique<rad::server::Sink>();
                        req->sink = a->sink.get();
                        const int rc = sched.add(std::move(req));
                        if (rc < 0) {
                            // A rejected request never owns the sink, so release its producer wait.
                            a->sink->fail(rc, "native scheduler refused the request");
                            connected = error(h, "native scheduler refused prompt: " + std::to_string(rc));
                        } else active = std::move(a);
                    } catch (const std::exception& e) { connected = error(h, e.what()); }
                } else connected = error(h, "unsupported IPC command");
            }
            if (!active) continue;
            auto& a = *active;
            auto& sink = *a.sink;
            if (!a.admitted && sink.t_admitted_ns() >= 0) {
                connected = emit_json(fd, Admitted, a.header.request, a.header.epoch,
                    {{"admitted_ns", sink.t_start_ns() + sink.t_admitted_ns()},
                     {"prompt_tokens", sink.n_prompt()}, {"cached_tokens", sink.n_cached()}});
                a.admitted = true;
            }
            rad::server::SinkToken items[64];
            const size_t n = sink.read(items, 64);
            std::vector<int32_t> tokens;
            for (size_t i = 0; i < n && !a.stopping; ++i) {
                tokens.push_back(items[i].token);
                if (items[i].accepted_draft) ++a.accepted_sent;
                a.detok->push(items[i].token);
                const bool id_stop = std::find(a.stop_ids.begin(), a.stop_ids.end(), items[i].token) != a.stop_ids.end();
                if (id_stop || a.detok->stopped()) {
                    a.matched_stop = id_stop ? std::to_string(items[i].token) : a.detok->stop_hit();
                    a.stopping = true;
                    sink.request_cancel();
                    sched.cancel(a.native_id);
                }
            }
            if (!tokens.empty()) {
                std::vector<char> out(12 + tokens.size() * 4);
                const uint64_t stamp = now_ns(); const uint32_t count = uint32_t(tokens.size());
                std::memcpy(out.data(), &stamp, 8); std::memcpy(out.data() + 8, &count, 4);
                std::memcpy(out.data() + 12, tokens.data(), tokens.size() * 4);
                connected = connected && emit(fd, a.stopping ? FinalTokens : Tokens,
                    a.header.request, a.header.epoch, out.data(), out.size());
                a.sent += tokens.size();
            }
            if (sink.done() && n == 0) {
                const auto reason = a.stopping ? "stop" : rad::server::finish_reason_oai(sink.finish_reason());
                Json done = {{"finish_reason", reason}, {"matched_stop", a.matched_stop},
                    {"output_tokens", a.sent}, {"produced_tokens", sink.n_produced()},
                    {"prompt_tokens", sink.n_prompt()}, {"cached_tokens", sink.n_cached()},
                    {"start_ns", sink.t_start_ns()}, {"admitted_ns", sink.t_admitted_ns()},
                    {"first_token_ns", sink.t_first_token_ns()}, {"last_token_ns", sink.t_last_token_ns()},
                    {"draft_tokens", sink.n_drafted()}, {"draft_accepted", a.accepted_sent},
                    {"draft_accepted_by_sampler", sink.n_draft_accepted()},
                    {"status", sink.status()}, {"end_ns", now_ns()}};
                done["native_stats"] = snapshot();
                const auto ledger = sink.round_ledger();
                Json rounds = {{"enabled", ledger.enabled}};
                if (ledger.enabled) {
                    rounds["scope"] = "host producer timestamp groups and completed add_drafts callbacks; includes the initial commit group; not GPU kernel time or all issued-ahead passes; sampler acceptance precedes IPC/stop clipping";
                    rounds["complete"] = sink.status() == RAD_OK;
                    rounds["verified_draft_rounds"] = ledger.verifications.size();
                    rounds["native_produced_tokens"] = sink.n_produced();
                    rounds["ipc_sent_tokens"] = a.sent;
                    rounds["ipc_sent_draft_tokens"] = a.accepted_sent;
                    rounds["commits"] = Json::array();
                    for (const auto& group : ledger.commits)
                        rounds["commits"].push_back({{"since_start_ns", group.time_ns},
                            {"tokens", group.tokens}, {"accepted_draft", group.accepted_draft}});
                    rounds["verifications"] = Json::array();
                    for (const auto& verify : ledger.verifications)
                        rounds["verifications"].push_back({{"proposed", verify.proposed},
                            {"accepted_by_sampler", verify.accepted_by_sampler},
                            {"produced_total_after", verify.produced_total_after},
                            {"last_commit_since_start_ns", verify.last_commit_since_start_ns}});
                }
                done["native_stats"]["request_rounds"] = std::move(rounds);

                connected = connected && emit_json(fd, Finish, a.header.request, a.header.epoch, done);
                sched.reap(a.native_id);
                active.reset();
            } else if (n == 0) sink.wait(1);
        }
    } catch (const std::exception& e) {
        Header h;
        error(h, e.what());
        connected = false;
    }
    if (active) {
        active->sink->request_cancel();
        sched.cancel(active->native_id);
        sched.reap(active->native_id);
    }
    return connected ? RAD_OK : RAD_E_IO;
}
}

int main(int argc, char** argv) {
    const pid_t parent = getppid();
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (getppid() != parent) return 1;
    std::signal(SIGPIPE, SIG_IGN);
    int fd = -1;
    std::string config, inspect_path, tokenize_path, text_path;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
        if (key == "--inspect") inspect_path = argv[++i];
        else if (key == "--tokenize") tokenize_path = argv[++i];
        else if (key == "--text-file") text_path = argv[++i];
        else if (key == "--config") config = argv[++i];
        else if (key == "--ipc-fd") fd = std::stoi(argv[++i]);
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    try {
        if (!inspect_path.empty()) { std::puts(ft_native::inspect(inspect_path.c_str()).dump().c_str()); return 0; }
        if (!tokenize_path.empty()) {
            if (text_path.empty()) throw std::runtime_error("--tokenize requires --text-file");
            std::puts(ft_native::tokenize(tokenize_path.c_str(), text_path.c_str()).dump().c_str());
            return 0;
        }
        if (config.empty() || fd < 0) throw std::runtime_error("--config and --ipc-fd are required");
        timeval limit{5, 0};
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &limit, sizeof(limit));
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &limit, sizeof(limit));
        rad::fleet::FleetConfig fleet;
        if (rad::fleet::load(config, &fleet) < 0 || fleet.models.size() != 1)
            throw std::runtime_error("native worker requires one valid model configuration");
        auto cfg = std::move(fleet.models[0].cfg);
        if (cfg.max_seqs != 1) throw std::runtime_error("native worker requires max-num-seqs: 1");
        if (cfg.kld() || cfg.calib() || cfg.diagnose_mode() || cfg.debug_graph)
            throw std::runtime_error("offline modes are unavailable in native worker");
        const char* expected = std::getenv("FREETOKEN_NATIVE_EXPECTED_PCIS");
        if (!expected) throw std::runtime_error("FREETOKEN_NATIVE_EXPECTED_PCIS must identify the reserved GPUs");
        const auto pcis = Json::parse(expected).get<std::vector<std::string>>();
        if (pcis.size() != size_t(cfg.tp) || rad_dev_count() != cfg.tp)
            throw std::runtime_error("visible native device count differs from reservation");
        for (int i = 0; i < cfg.tp; ++i) {
            auto actual = rad::dev_pci_address(i);
            auto want = pcis[size_t(i)];
            std::transform(actual.begin(), actual.end(), actual.begin(), ::tolower);
            std::transform(want.begin(), want.end(), want.begin(), ::tolower);
            if (actual != want) throw std::runtime_error("native GPU PCI mismatch: " + actual + " != " + want);
        }
        setenv("FREETOKEN_NATIVE_MANIFEST_DIR", std::filesystem::path(config).parent_path().c_str(), 1);
        rad::Engine engine(std::move(cfg));
        const int started = engine.start();
        if (started < 0) throw std::runtime_error("native model start failed: " + std::to_string(started));
        const int rc = engine.serve_headless([&](rad::Scheduler& sched,
                          std::shared_ptr<const rad::Vocab> vocab, const Json& facts) {
            return ft_native::run(fd, sched, std::move(vocab), facts, [&] { return engine.headless_stats(); });
        });
        return rc < 0 ? 1 : 0;
    } catch (const std::exception& e) {
        if (fd >= 0) ft_native::emit_json(fd, ft_native::Error, 0, 0, {{"error", e.what()}});
        std::fprintf(stderr, "freetoken-native-worker: %s\n", e.what());
        return 1;
    }
}
