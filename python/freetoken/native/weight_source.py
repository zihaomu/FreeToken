"""Read canonical .rad metadata without loading Torch, a GPU plugin, or weight data."""
from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

_HEADER = struct.Struct("<II30Q")
_ENTRY = struct.Struct("<3Q2I6q4i5Q")
_KV = struct.Struct("<QII8s")
_PLANE = struct.Struct("<QQ")
_ENCODING_HEAD = struct.Struct("<24s24sii")
_ENCODING_PLANE = struct.Struct("<24sII4q")
_ENCODING_BYTES = _ENCODING_HEAD.size + 8 * _ENCODING_PLANE.size
_SECTIONS = ("str", "meta", "dir", "plane", "enc", "vocab", "prof", "data")
_MAX_METADATA_BYTES = 128 * 1024 * 1024


def _text(value: bytes) -> str:
    end = value.find(b"\0")
    if end < 0:
        raise ValueError("unterminated .rad fixed string")
    return value[:end].decode("utf-8")


@dataclass(frozen=True)
class EncodingPlane:
    role: str
    dtype: int
    kind: int
    block: tuple[int, int]
    extent: tuple[int, int]


@dataclass(frozen=True)
class Encoding:
    scheme: str
    transform: str
    planes: tuple[EncodingPlane, ...]


@dataclass(frozen=True)
class FilePlane:
    offset: int
    nbytes: int


@dataclass(frozen=True)
class WeightRef:
    name: str
    shape: tuple[int, ...]
    encoding: Encoding
    planes: tuple[FilePlane, ...]
    layer: int
    expert: int
    slot: int
    quantizer: str
    options: str


class RadWeightSource:
    """A bounded metadata catalog; entries retain file ranges instead of materialized tensors."""

    def __init__(self, path: str | Path):
        self.path = str(Path(path).resolve())
        self.weights: dict[str, WeightRef] = {}
        self.metadata: dict[str, int | float | str] = {}
        self.typed_metadata: dict[tuple[str, int], int | float | str] = {}
        self.metadata_bytes_read = 0
        with open(self.path, "rb", buffering=0) as file:
            self._fd = file.fileno()
            before = os.fstat(self._fd)
            self.identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            self.file_bytes = before.st_size
            digest = hashlib.sha256()

            def read(offset, size):
                if not 0 <= offset <= offset + size <= self.file_bytes:
                    raise ValueError(".rad range outside file")
                if size > _MAX_METADATA_BYTES or self.metadata_bytes_read + size > _MAX_METADATA_BYTES:
                    raise ValueError(".rad metadata exceeds reader budget")
                data = os.pread(self._fd, size, offset)
                if len(data) != size:
                    raise ValueError("truncated .rad metadata")
                self.metadata_bytes_read += size
                digest.update(data)
                return data

            header = _HEADER.unpack(read(0, _HEADER.size))
            if header[:2] != (0x31444152, 2) or header[2] != self.file_bytes:
                raise ValueError("incompatible .rad header or file length")
            fields = header[3:]
            sections = {name: tuple(fields[5 + 2 * i:7 + 2 * i])
                        for i, name in enumerate(_SECTIONS)}
            item_bytes = {"str": 1, "meta": _KV.size, "dir": _ENTRY.size,
                          "plane": _PLANE.size, "enc": _ENCODING_BYTES,
                          "vocab": 1, "prof": 16, "data": 1}
            ranges = []
            for name, (offset, count) in sections.items():
                size = count * item_bytes[name]
                if not 0 <= offset <= offset + size <= self.file_bytes:
                    raise ValueError(f".rad {name} section outside file")
                if size:
                    if offset < _HEADER.size:
                        raise ValueError(".rad section overlaps its header")
                    ranges.append((offset, offset + size, name))
            for left, right in zip(sorted(ranges), sorted(ranges)[1:]):
                if left[1] > right[0]:
                    raise ValueError(f".rad sections overlap: {left[2]}, {right[2]}")
            data_start, data_bytes = sections["data"]
            if data_start % (2 * 1024 * 1024):
                raise ValueError("unaligned .rad data")
            strings = read(*sections["str"])
            if not strings or strings[-1] != 0:
                raise ValueError("invalid .rad string table")

            def string(offset):
                if not 0 <= offset < len(strings):
                    raise ValueError(".rad string offset outside table")
                end = strings.find(b"\0", offset)
                if end < 0:
                    raise ValueError("unterminated .rad string")
                return strings[offset:end].decode("utf-8")

            self.architecture, self.model_name, self.quantization, self.recipe, self.created_by = (
                string(offset) for offset in fields[:5]
            )
            offset, count = sections["meta"]
            for key, kind, pad, raw in _KV.iter_unpack(read(offset, count * _KV.size)):
                if pad:
                    raise ValueError("nonzero .rad metadata padding")
                name = string(key)
                if kind == 1:
                    value = struct.unpack("<q", raw)[0]
                elif kind == 2:
                    value = string(struct.unpack("<Q", raw)[0])
                elif kind == 100:
                    value = struct.unpack("<d", raw)[0]
                    if not math.isfinite(value):
                        raise ValueError("nonfinite .rad metadata")
                else:
                    raise ValueError("unknown .rad metadata type")
                typed_key = name, kind
                if typed_key in self.typed_metadata and self.typed_metadata[typed_key] != value:
                    raise ValueError("conflicting .rad metadata of the same type")
                self.typed_metadata.setdefault(typed_key, value)
                self.metadata[name] = value
            offset, count = sections["enc"]
            data = read(offset, count * _ENCODING_BYTES)
            encodings = []
            for i in range(count):
                start = i * _ENCODING_BYTES
                scheme, transform, nplanes, reserved = _ENCODING_HEAD.unpack_from(data, start)
                if not 1 <= nplanes <= 8 or reserved:
                    raise ValueError("invalid .rad encoding plane count")
                planes = []
                for j in range(nplanes):
                    role, dtype, kind, br, bc, er, ec = _ENCODING_PLANE.unpack_from(
                        data, start + _ENCODING_HEAD.size + j * _ENCODING_PLANE.size)
                    if kind not in (0, 1) or min(br, bc, er, ec) < 0:
                        raise ValueError("invalid .rad plane geometry")
                    planes.append(EncodingPlane(_text(role), dtype, kind, (br, bc), (er, ec)))
                if len({p.role for p in planes}) != len(planes):
                    raise ValueError("duplicate .rad encoding role")
                encodings.append(Encoding(_text(scheme), _text(transform), tuple(planes)))
            self.encodings = tuple(encodings)
            offset, count = sections["plane"]
            file_planes = tuple(FilePlane(*p) for p in _PLANE.iter_unpack(read(offset, count * _PLANE.size)))
            for plane in file_planes:
                if plane.offset % 256 or not (
                    data_start <= plane.offset <= plane.offset + plane.nbytes <= data_start + data_bytes
                ):
                    raise ValueError("invalid .rad plane range")
            offset, count = sections["dir"]
            groups = set()
            for row in _ENTRY.iter_unpack(read(offset, count * _ENTRY.size)):
                name, quantizer, options, enc, rank = row[:5]
                shape = row[5:11]
                layer, expert, slot, nplanes = row[11:15]
                first, start, nbytes, r0, r1 = row[15:]
                if not 1 <= rank <= 6 or any(x <= 0 for x in shape[:rank]) or enc >= len(encodings):
                    raise ValueError("invalid .rad weight shape or encoding")
                if (r0 or r1 or nplanes != len(encodings[enc].planes)
                        or first + nplanes > len(file_planes) or start % 4096):
                    raise ValueError("invalid .rad weight planes")
                selected = file_planes[first:first + nplanes]
                if not (data_start <= start <= start + nbytes <= data_start + data_bytes):
                    raise ValueError(".rad weight outside data")
                if any(p.offset < start or p.offset + p.nbytes > start + nbytes for p in selected):
                    raise ValueError(".rad plane outside its weight")
                name = string(name)
                if name in self.weights:
                    raise ValueError("duplicate .rad weight")
                self.weights[name] = WeightRef(name, tuple(shape[:rank]), encodings[enc], selected,
                                               layer, expert, slot, string(quantizer), string(options))
                if layer >= 0 and expert >= 0:
                    groups.add(layer)
            self.expert_groups = len(groups)
            self.metadata_sha256 = digest.hexdigest()
            after = os.fstat(self._fd)
            if self.identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError(".rad file changed while reading metadata")
        del self._fd

    def integer(self, name: str, default: int | None = None) -> int:
        value = self.typed_metadata.get((name, 1))
        if value is None:
            value = self.typed_metadata.get((name, 2), default)
        if value is None:
            raise ValueError(f"missing integer model property: {name}")
        return int(value)

    def descriptor(self) -> dict:
        return {"model_path": self.path, "model_bytes": self.file_bytes,
                "architecture": self.architecture, "model_name": self.model_name,
                "metadata": self.metadata, "entries": len(self.weights),
                "encodings": len(self.encodings), "expert_groups": self.expert_groups,
                "metadata_bytes_read": self.metadata_bytes_read,
                "metadata_sha256": self.metadata_sha256}
