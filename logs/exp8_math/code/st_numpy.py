"""Minimal safetensors reader / writer in numpy (bfloat16 tensors are handled as their raw uint16 bits), so the edit
files can be applied without torch.   Used by make_edit_diff.py and apply_edit.py."""
import json
import struct

import numpy as np

DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64, "I64": np.int64, "I32": np.int32,
      "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load(path, names=None):
    """{name: (dtype string, numpy array)}; BF16 arrays are uint16 bit patterns."""
    head, off = read_header(path)
    out = {}
    with open(path, "rb") as f:
        for name, info in head.items():
            if name == "__metadata__" or (names is not None and name not in names):
                continue
            a, b = info["data_offsets"]
            f.seek(off + a)
            arr = np.frombuffer(f.read(b - a), dtype=DT[info["dtype"]]).reshape(info["shape"])
            out[name] = (info["dtype"], arr)
    return out


def save(path, tensors, metadata=None):
    """tensors: {name: (dtype string, numpy array)} written in name order."""
    head, off, blobs = {}, 0, []
    for name in sorted(tensors):
        dt, arr = tensors[name]
        b = np.ascontiguousarray(arr).tobytes()
        head[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
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
    """float32 -> bfloat16 bit pattern, round to nearest even (as torch's .to(torch.bfloat16) for finite values)."""
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)
