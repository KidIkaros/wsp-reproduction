"""Shared safetensors/bf16 I/O for the wsp kernels (mirrors the paper's st_numpy)."""
import json
import struct

import numpy as np

DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}

L15 = "model.language_model.layers.15.mlp.down_proj.weight"
L19 = "model.language_model.layers.19.mlp.down_proj.weight"
NAMES = [L15, L19]


def st_load(path, names=None):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        off = 8 + n
        out = {}
        for name, info in head.items():
            if name == "__metadata__" or (names is not None and name not in names):
                continue
            a, b = info["data_offsets"]
            f.seek(off + a)
            arr = np.frombuffer(f.read(b - a), dtype=DT[info["dtype"]]).reshape(info["shape"])
            out[name] = (info["dtype"], arr)
        return out


def st_save(path, tensors, metadata=None):
    head, off, blobs = {}, 0, []
    for name in sorted(tensors):
        dt, arr = tensors[name]
        b = np.ascontiguousarray(arr).tobytes()
        head[name] = {"dtype": dt, "shape": list(arr.shape),
                      "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    if metadata:
        head["__metadata__"] = metadata
    h = json.dumps(head, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def bf16_bits(x32):
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)
