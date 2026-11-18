"""Bounded-batch IPC, deliberately incompatible with v1 whole-request generation."""
from __future__ import annotations

import json
import socket
import struct
from dataclasses import asdict
from enum import IntEnum

from freetoken.engine.backend import (
    Completion, Phase, RequestKey, ResourceHandle, StateCursor, StepPlan, StepTicket, checked_int,
)

MAGIC = 0x32585446
VERSION = 3
HEADER = struct.Struct("<IHHIIQQ")
MAX_PAYLOAD = 8 * 1024 * 1024


class Kind(IntEnum):
    READY = 1
    COMPILE = 2
    RESERVE = 3
    SUBMIT = 4
    COMPLETE = 5
    REVOKE = 6
    RELEASE = 7
    INSPECT = 8
    SHUTDOWN = 9
    ERROR = 10
    SNAPSHOT = 11
    RESTORE = 12
    DROP_SNAPSHOT = 13


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate executor JSON field")
        out[key] = value
    return out


def encode(value) -> bytes:
    payload = json.dumps(value, allow_nan=False, ensure_ascii=False,
                         separators=(",", ":")).encode()
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("executor payload exceeds 8 MiB")
    return payload


def decode(payload: bytes):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("executor payload exceeds 8 MiB")
    return json.loads(payload, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _ticket(value: dict) -> StepTicket:
    value = dict(value)
    value["request"] = RequestKey(**value["request"])
    return StepTicket(**value)


def encode_step(plan: StepPlan) -> bytes:
    return encode(asdict(plan))


def decode_step(payload: bytes) -> StepPlan:
    value = decode(payload)
    value["ticket"] = _ticket(value["ticket"])
    value["executable"] = ResourceHandle(**value["executable"])
    value["state"] = ResourceHandle(**value["state"])
    value["phase"] = Phase(value["phase"])
    value["input_ids"] = tuple(value.get("input_ids", ()))
    if value.get("predecessor") is not None:
        value["predecessor"] = _ticket(value["predecessor"])
    return StepPlan(**value)


def encode_completion(result: Completion) -> bytes:
    return encode(asdict(result))


def decode_completion(payload: bytes) -> Completion:
    value = decode(payload)
    value["ticket"] = _ticket(value["ticket"])
    value["state"] = ResourceHandle(**value["state"])
    value["before"] = StateCursor(**value["before"])
    if value["after"] is not None:
        value["after"] = StateCursor(**value["after"])
    value["tokens"] = tuple(value.get("tokens", ()))
    value["draft_flags"] = tuple(value.get("draft_flags", ()))
    return Completion(**value)


def send_frame(sock: socket.socket, kind: Kind, uid: int = 0, epoch: int = 0,
               payload: bytes = b"") -> None:
    checked_int(uid, "uid", maximum=(1 << 64) - 1)
    checked_int(epoch, "epoch", maximum=(1 << 64) - 1)
    if not isinstance(kind, Kind) or len(payload) > MAX_PAYLOAD:
        raise ValueError("invalid executor frame")
    sock.sendall(HEADER.pack(MAGIC, VERSION, kind, len(payload), 0, uid, epoch) + payload)


def _read(sock: socket.socket, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    while view:
        n = sock.recv_into(view)
        if not n:
            raise EOFError("executor IPC closed")
        view = view[n:]
    return bytes(data)


def recv_frame(sock: socket.socket) -> tuple[Kind, int, int, bytes]:
    magic, version, kind, size, reserved, uid, epoch = HEADER.unpack(_read(sock, HEADER.size))
    if magic != MAGIC or version != VERSION or reserved or size > MAX_PAYLOAD:
        raise ValueError("invalid executor protocol header")
    kind = Kind(kind)
    return kind, uid, epoch, _read(sock, size)
