/* core/server/http.h -- the transport seam.
 *
 * Routing, OpenAI translation and metrics are written against these plain structs and never see
 * a socket, so tests/server_test.cpp drives the real handlers in-process: `Router::dispatch` is
 * the same call the transport makes. Tests need no free port and do not re-test cpp-httplib.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

/* Header names are case-insensitive per RFC 9110, and clients disagree about capitalisation of
 * Authorization and Content-Type in particular. */
struct CiLess {
    bool operator()(const std::string& a, const std::string& b) const;
};

/* One part of a multipart/form-data body: a field's value, or an uploaded file's bytes. */
struct FormPart {
    std::string name, filename, content_type, data;
};

struct HttpRequest {
    std::string method = "GET";
    std::string path;                                    /* no query string */
    std::string body;
    /* A multipart/form-data body arrives as its parts, in order, and `body` is empty. */
    std::vector<FormPart> form;
    std::map<std::string, std::string, CiLess> headers;
    std::map<std::string, std::string> query;
    std::map<std::string, std::string> params;           /* from a ":name" route segment */
    std::string peer;                                    /* "addr:port"; empty in-process. The
                                                            ATTRIBUTED caller: with --trusted-proxy
                                                            and an X-Forwarded-For, the address the
                                                            proxy chain names; the port is always
                                                            the last hop's -- the chain carries no
                                                            port. For logs and audit, never for
                                                            authority. */
    std::string raw_peer;                                /* "addr:port" of the socket peer as the
                                                            kernel saw it; empty in-process. The
                                                            only address the guards read. */
    /* Steady-clock nanoseconds at which the whole body had arrived. The server's own time for
     * the request starts here, before parsing, so network time is excluded. -1 in-process. */
    int64_t     t_received_ns = -1;

    /* True once the peer is gone; absent in-process. A dropped connection is a cancellation
     * (spec §14), so every path that waits polls this. */
    std::function<bool()> closed;

    std::string header(const std::string& k, const std::string& def = "") const;
    std::string param(const std::string& k, const std::string& def = "") const;
    /* The first form part named `k` (a field or a file), or null. */
    const FormPart* part(const std::string& k) const;
    bool        peer_gone() const { return closed && closed(); }
    /* True when raw_peer is 127.0.0.0/8, ::1, an IPv4-mapped 127.x address, or absent (the
     * in-process request). Reads the socket, never the attributed peer: a proxy an operator
     * vouched for can name any caller it likes; that changes the log line, not the guard. */
    bool        peer_is_loopback() const;
    /* True when the request comes from a browser on another origin: an `Origin` header whose
     * authority differs from `Host`. Scripts send no Origin, so this never fires for curl or an
     * SDK. Through a proxy that rewrites Host, every browser reads as cross-origin -- which is
     * exactly the case the "loopback-only" guards must not wave through. */
    bool        is_cross_origin() const;
    /* The Host header's authority, lowercased with the port stripped; "" when there is no Host.
     * The name a browser believes it is talking to — under DNS rebinding the attacker's own, and
     * through a proxy the public one, neither of which the peer's address shows. */
    std::string host_authority() const;
    /* True when the Host names this machine as the loopback: "localhost", 127.0.0.1/8, [::1].
     * A rebinding page is same-origin with the server and its victim's browser is the loopback
     * peer — only the Host still says what page asked for it. An absent Host reads as local for
     * an in-process dispatch and as remote for a socket, which always carries one. */
    bool        host_is_loopback() const;
};

/* Whether `cidr` is in the one form --trusted-proxy takes: numeric "addr/prefix", prefix within
 * its family. Config refuses anything else at startup, so a typo cannot silently unvouch a
 * proxy; ip_in_cidr treats the same rejection as simply no match. */
bool cidr_valid(const std::string& cidr);

/* True when the numeric address `addr` (no port, no zone) falls inside `cidr`. Families must
 * agree: this is not RFC 6052 translation, an IPv4 peer is not inside an IPv6 network that
 * happens to embed it. */
bool ip_in_cidr(const std::string& addr, const std::string& cidr);

/* The caller a proxy chain vouches for. `raw_addr` is the socket peer (no port), `xff` the
 * request's X-Forwarded-For value ("" when there is none), `trusted` the vouched CIDRs.
 * Nothing is read from the header unless the socket itself belongs to a vouched CIDR -- an
 * unvouched peer can name any address it likes and names none. Then the chain is walked from
 * the right (the hop nearest us, which our vouched proxy appended): the first entry that no
 * vouched CIDR covers is the caller; a chain of only vouched hops ends at its leftmost entry,
 * the original client as the outermost proxy saw it. Malformed entries stop the walk where
 * they stand -- the chain is reported, not interpreted, and the log sanitizer owns the bytes.
 * Fallback is `raw_addr`, always. */
std::string attributed_client(const std::string& raw_addr, const std::string& xff,
                              const std::vector<std::string>& trusted);

struct HttpResponse {
    int         status = 200;
    std::string content_type = "application/json; charset=utf-8";
    std::string body;
    std::vector<std::pair<std::string, std::string>> headers;

    /* Streaming. When `next` is set, `body` is ignored and the transport pulls chunks until next()
     * returns false. An empty chunk writes nothing but still checks the connection. */
    std::function<bool(std::string&)> next;

    /* Runs exactly once when the connection ends; `ok` is false if the peer left before the
     * stream finished. A dropped stream becomes a scheduler cancellation here. */
    std::function<void(bool ok)> on_close;

    bool streaming() const { return next != nullptr; }

    void set_header(std::string k, std::string v) { headers.emplace_back(std::move(k), std::move(v)); }

    static HttpResponse json_body(int status, std::string body);
    static HttpResponse text(int status, std::string body);
};

using Handler = std::function<HttpResponse(const HttpRequest&)>;

/* One WebSocket connection, as a socket route sees it: messages in, messages out, and a close.
 * The route runs for the connection's whole life on the transport's thread for it. It reads on
 * that thread and may send and close from another, so it can answer while it listens. A test
 * hands the route a connection of its own. */
class WsConn {
public:
    virtual ~WsConn() = default;
    /* Blocks for the next message: true with it in `*msg` and `*binary` saying which frame it came
     * in, false once the connection is gone. */
    virtual bool read(std::string* msg, bool* binary) = 0;
    /* A text message. Safe from any thread; false once the connection is gone. */
    virtual bool send(const std::string& text) = 0;
    /* A binary message (raw audio). Safe from any thread; false once the connection is gone, or
     * for a connection that carries text only. */
    virtual bool send_binary(const std::string& bytes) {
        (void)bytes;
        return false;
    }
    /* Ends the connection with a close code and reason (RFC 6455 §7.4); a read() blocked on it
     * returns false once the peer has answered or gone. Safe from any thread. */
    virtual void close(int code, const std::string& reason) = 0;
};

/* A socket route: the upgrade request, and the connection it became. */
using SocketHandler = std::function<void(const HttpRequest&, WsConn&)>;

/* Exact-match routing plus ":name" wildcard segments and a final "*name" that takes the rest of
 * the path, which is all the OpenAI surface needs (a model id may hold a slash, so the model
 * entry's route ends in "*id"). No regex and no priority rules. */
class Router {
public:
    void add(std::string method, std::string path, Handler h);
    void get(std::string path, Handler h)  { add("GET",  std::move(path), std::move(h)); }
    void post(std::string path, Handler h) { add("POST", std::move(path), std::move(h)); }
    /* A WebSocket route, exact path. An upgrade request to it becomes a connection the handler
     * is given; a plain GET to the same path is whatever add() registered there, if anything. */
    void socket(std::string path, SocketHandler h);
    /* The handler for an upgrade to `path`, or null. */
    const SocketHandler* socket_for(const std::string& path) const;
    std::vector<std::string> socket_routes() const;

    /* 404 with a body naming the path, 405 when the path exists under another method. Takes the
     * request by value so the transport can move a body of up to hundreds of MB into the handler. */
    HttpResponse dispatch(HttpRequest req) const;

    /* Convenience for tests. */
    HttpResponse call(const std::string& method, const std::string& path,
                      const std::string& body = "") const;

    /* Whether a route takes `method` on `path`, as dispatch would match it. */
    bool allows(const std::string& method, const std::string& path) const;

    std::vector<std::string> routes() const;

private:
    struct Route {
        std::string method;
        std::vector<std::string> segs;   /* ":name" is any one segment, a final "*name" the rest */
        Handler h;
    };
    /* The route taking `method` on the path `segs`, with its captures; null when none.
     * `*path_matched` says whether a route takes the path under any method. */
    const Route* match(const std::string& method, const std::vector<std::string>& segs,
                       std::map<std::string, std::string>* caps, bool* path_matched) const;
    std::vector<Route> routes_;
    std::vector<std::pair<std::string, SocketHandler>> sockets_;
};

/* Anthropic's clients read Anthropic's error envelope, `{"type":"error","error":{"type",
 * "message"}}`, so every refusal on one of its paths (/v1/messages and below) is written in it,
 * the transport's own included; every other path gets OpenAI's. */
bool        anthropic_path(const std::string& path);
/* A refusal's JSON body in the envelope `path` calls for (OpenAI's, or Anthropic's on its paths),
 * written without the JSON library so the error path cannot itself fail. */
std::string error_body(const std::string& path, int status, const char* type,
                       const std::string& msg);
/* Anthropic's error `type` for an HTTP status. */
const char* anthropic_error_type(int status);

/* Split "/v1/models/foo?x=1" into path and decoded query parameters. */
void parse_target(const std::string& target, std::string* path,
                  std::map<std::string, std::string>* query);
/* Percent-decode. `plus_is_space` is the form encoding: true for a query string, false for a
 * path. */
std::string url_decode(const std::string& s, bool plus_is_space = true);

/* ================================================================== the cpp-httplib binding */

/* One worker per connection for the connection's whole life: httplib reads, handles and writes
 * on one thread, and a generation's handler runs until the generation ends. The pool bounds are
 * therefore connection counts, which Server::run derives from the admission ceiling, not the
 * core count. */
struct TransportOptions {
    std::string host = "0.0.0.0";
    int         port = 8000;              /* 0 -> any free port; see HttpTransport::port */
    int         n_threads = 0;            /* kept alive while idle. 0 -> hardware_concurrency,
                                           * floor 4 */
    int64_t     max_threads = 0;          /* what the pool may grow to. 0 -> n_threads */
    int64_t     max_queued = 0;           /* accepted connections that may wait for a worker once
                                           * it has grown to max_threads; past that they are
                                           * closed. 0 -> unbounded */
    /* Longest silence allowed while the request is being read; generation is not bounded by it.
     * A peer that stalls mid-request holds a worker until this expires, so it is short. */
    int         read_timeout_s = 30;
    int         write_timeout_s = 600;
    int         keep_alive_timeout_s = 5; /* an idle connection's hold on a worker between requests */
    size_t      max_body_bytes = 512ull << 20;   /* images arrive base64 inside the JSON body */
    /* The sum of bodies being buffered across all connections at once. The per-request cap says
     * nothing about the sum, and a request's body is fully buffered before parse, admission or
     * any memory accounting sees it; over budget, a body is refused 429 rather than pushed into
     * RAM until the OOM killer chooses which process dies. 0 takes 8 concurrent bodies at the
     * per-request cap. */
    size_t      max_inflight_body_bytes = 0;
    std::string api_key;                  /* "" disables the check */
    bool        cors = true;
    /* --trusted-proxy, repeatable: CIDRs whose sockets may vouch for their caller through
     * X-Forwarded-For. Empty: no socket vouches for anything and peer always equals raw_peer.
     * Attribution rewrites the logged caller, never an authority -- see HttpRequest::peer. */
    std::vector<std::string> trusted_proxies;
};

/* Idle workers kept for TransportOptions::n_threads: that many, one per hardware thread for 0,
 * never fewer than 4. Shared so the startup report states what the transport runs. */
int idle_workers(int n_threads);

class HttpTransport {
public:
    HttpTransport();
    ~HttpTransport();
    HttpTransport(const HttpTransport&) = delete;
    HttpTransport& operator=(const HttpTransport&) = delete;

    /* Blocks until stop(). Returns RAD_OK, or negative if the bind failed. */
    int  serve(const Router& r, const TransportOptions& o);
    void stop();

    /* The port serve() bound (the only way to learn it for port 0). 0 until the server accepts
     * connections, so a non-zero answer also means stop() will take effect. */
    int  port() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  /* namespace server */
}  /* namespace rad */
