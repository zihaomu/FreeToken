from __future__ import annotations

import json
import socket
import struct
from enum import IntEnum

HEADER = struct.Struct("<IHHIIQQ")
MAGIC = 0x574E5446
VERSION = 2
MAX_PAYLOAD = 8 * 1024 * 1024


class Kind(IntEnum):
    READY = 1
    GENERATE = 2
    ADMITTED = 3
    TOKENS = 4
    FINISH = 5
    CANCEL = 6
    SHUTDOWN = 7
    ERROR = 8
    STATS = 9
    FINAL_TOKENS = 10


def send_frame(sock: socket.socket, kind: Kind, uid: int = 0, epoch: int = 0, payload: bytes = b"") -> None:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("native payload exceeds 8 MiB")
    sock.sendall(HEADER.pack(MAGIC, VERSION, kind, len(payload), 0, uid, epoch) + payload)


def _read_exact(sock: socket.socket, count: int) -> bytes:
    parts = bytearray(count)
    view = memoryview(parts)
    while view:
        n = sock.recv_into(view)
        if n == 0:
            raise EOFError("native worker closed its IPC channel")
        view = view[n:]
    return bytes(parts)


def recv_frame(sock: socket.socket) -> tuple[Kind, int, int, bytes]:
    magic, version, kind, size, reserved, uid, epoch = HEADER.unpack(_read_exact(sock, HEADER.size))
    if magic != MAGIC or version != VERSION or reserved or size > MAX_PAYLOAD:
        raise ValueError("invalid native protocol header")
    return Kind(kind), uid, epoch, _read_exact(sock, size)


def prompt_payload(metadata: dict, token_ids: list[int]) -> bytes:
    meta = json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    size = 4 + len(meta) + 4 * len(token_ids)
    if size > MAX_PAYLOAD:
        raise ValueError("native prompt exceeds 8 MiB")
    return struct.pack("<I", len(meta)) + meta + struct.pack(f"<{len(token_ids)}i", *token_ids)


def decode_tokens(payload: bytes) -> tuple[int, list[int]]:
    if len(payload) < 12:
        raise ValueError("short native token payload")
    stamp, count = struct.unpack_from("<QI", payload)
    if len(payload) != 12 + 4 * count:
        raise ValueError("native token count does not match payload")
    return stamp, list(struct.unpack_from(f"<{count}i", payload, 12))
