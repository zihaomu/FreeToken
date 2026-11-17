from __future__ import annotations

import json
import select
import signal
import time

import zmq

from freetoken.message import (
    AbortBackendMsg, BaseBackendMsg, BaseTokenizerMsg, BatchBackendMsg,
    CacheRebuildBackendMsg, CacheRebuildResultMsg, DetokenizeMsg, ErrorReplyMsg,
    ExitMsg, FlushCacheBackendMsg, FlushCacheResultMsg, PromptAdmittedMsg, UserMsg,
)
from freetoken.native.client import NativeClient
from freetoken.native.protocol import Kind, decode_tokens
from freetoken.utils import ZmqPullQueue, ZmqPushQueue


def native_timings(done: dict, enqueued_ns: int) -> dict | None:
    first, last, admit = (done[key] for key in ("first_token_ns", "last_token_ns", "admitted_ns"))
    if min(first, last, admit) < 0:
        return None
    return {
        "queue_ms": max(0, (done["start_ns"] + admit - enqueued_ns) / 1e6),
        "cache_n": done["cached_tokens"],
        "prompt_n": done["prompt_tokens"] - done["cached_tokens"],
        "prompt_ms": max(0, (first - admit) / 1e6),
        "predicted_n": done["output_tokens"],
        "predicted_ms": max(0.001, (last - first) / 1e6),
        "draft_n": done["draft_tokens"], "draft_n_accepted": done["draft_accepted"],
    }


def run_native_backend(args, ack_queue) -> None:
    from freetoken.server.launch import _report_startup_error

    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    receive = ZmqPullQueue(args.zmq_backend_addr, create=True, decoder=BaseBackendMsg.decoder)
    send = ZmqPushQueue(args.zmq_detokenizer_addr, create=False, encoder=BaseTokenizerMsg.encoder)
    active = None
    epoch = 0
    try:
        with NativeClient(args.native_config, args.native_work_dir) as client:
            eog_ids = set(client.ready["eog_token_ids"])
            ack_queue.put(("meta", {"execution_backend": "radiance-native", "native": client.ready}))
            ack_queue.put("Native backend is ready")
            while True:
                if client.process.poll() is not None:
                    raise RuntimeError(f"native worker exited: {client.process.returncode}")
                # select accepts ZMQ's edge-trigger FD; drain its socket fully on each wake.
                select.select([client.socket, receive.socket.getsockopt(zmq.FD)], [], [], 0.1)
                while not receive.empty():
                    msg = receive.get()
                    messages = msg.data if isinstance(msg, BatchBackendMsg) else [msg]
                    for msg in messages:
                        if isinstance(msg, ExitMsg):
                            return
                        if isinstance(msg, UserMsg):
                            if active is not None:
                                send.put(ErrorReplyMsg(msg.uid, "native backend is busy (batch 1)"))
                                continue
                            if msg.mm_items or msg.mrope_positions is not None:
                                send.put(ErrorReplyMsg(msg.uid, "native backend supports text requests only"))
                                continue
                            params = msg.sampling_params
                            token_ids = msg.input_ids.tolist()
                            if len(token_ids) >= args.native_config.max_context:
                                send.put(ErrorReplyMsg(msg.uid, "native prompt exceeds context", "context_length_exceeded"))
                                continue
                            epoch += 1
                            enqueued = time.monotonic_ns()
                            client.generate(msg.uid, epoch, token_ids, {
                                "max_tokens": params.max_tokens, "temperature": params.temperature,
                                "top_k": max(params.top_k, 0), "top_p": params.top_p,
                                "frequency_penalty": params.frequency_penalty,
                                "presence_penalty": params.presence_penalty,
                                "ignore_eos": params.ignore_eos, "stop": params.stop_strs,
                            })
                            active = {"uid": msg.uid, "epoch": epoch, "stops": params.stop_strs,
                                      "enqueued_ns": enqueued, "admitted": False,
                                      "ignore_eos": params.ignore_eos, "terminal_tokens": []}
                        elif isinstance(msg, AbortBackendMsg):
                            if active is not None and msg.uid == active["uid"]:
                                client.cancel(msg.uid, active["epoch"])
                        elif isinstance(msg, CacheRebuildBackendMsg):
                            send.put(CacheRebuildResultMsg(msg.request_id, "failed", error="native cache rebuild is unsupported"))
                        elif isinstance(msg, FlushCacheBackendMsg):
                            send.put(FlushCacheResultMsg(msg.request_id, "unsupported", "native cache flush is unsupported"))
                        else:
                            raise ValueError(f"unsupported native backend message: {type(msg).__name__}")
                while client.readable():
                    kind, uid, event_epoch, payload = client.receive()
                    if kind == Kind.ERROR and uid == 0 and event_epoch == 0:
                        raise RuntimeError(json.loads(payload)["error"])
                    if active is None or (uid, event_epoch) != (active["uid"], active["epoch"]):
                        raise RuntimeError("native event does not match the active request epoch")
                    if kind == Kind.ADMITTED:
                        if active["admitted"]:
                            raise RuntimeError("duplicate native prompt admission")
                        meta = json.loads(payload)
                        send.put(PromptAdmittedMsg(uid, meta["prompt_tokens"], meta["cached_tokens"]))
                        active["admitted"] = True
                    elif kind in (Kind.TOKENS, Kind.FINAL_TOKENS):
                        if not active["admitted"]:
                            raise RuntimeError("native tokens arrived before admission")
                        _, tokens = decode_tokens(payload)
                        if kind == Kind.FINAL_TOKENS:
                            active["terminal_tokens"].extend(tokens)
                            tokens = []
                        elif not active["ignore_eos"] and tokens and tokens[-1] in eog_ids:
                            active["terminal_tokens"] = tokens[-1:]
                            tokens = tokens[:-1]
                        if tokens:
                            send.put(DetokenizeMsg(uid, tokens, False, stop_strs=active["stops"]))
                    elif kind == Kind.FINISH:
                        done = json.loads(payload)
                        if done["status"] < 0:
                            send.put(ErrorReplyMsg(uid, f"native request failed: {done['status']}"))
                        else:
                            send.put(DetokenizeMsg(uid, active["terminal_tokens"], True, done["finish_reason"],
                                matched_stop=done["matched_stop"] or None, stop_strs=active["stops"],
                                timings=native_timings(done, active["enqueued_ns"]),
                                native_stats=done.get("native_stats")))
                        active = None
                    elif kind == Kind.ERROR:
                        send.put(ErrorReplyMsg(uid, json.loads(payload)["error"]))
                        active = None
                    else:
                        raise RuntimeError(f"unexpected native event: {kind}")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _report_startup_error(ack_queue, exc)
        raise
    finally:
        receive.socket.close(linger=0)
        receive.context.term()
        send.socket.close(linger=0)
        send.context.term()
