"""Rebuild an edited model from the base model and a run's edit (edit/edit.npz + edit/manifest.json).

    python apply_edit.py <base model.safetensors> <run's edit dir> <out>

<base model.safetensors>: for the Qwen runs, the language-model tensors of Qwen/Qwen3.5-0.8B (revision
2fc06364715b967f1860aea9cf38778875588b17; the experiment used a text-only export with the same tensor names,
model.language_model.*; the release's single file also works, since only the named tensors are read); for the Gemma
runs, google/gemma-3-270m-it (revision ac82b4e820549b854eebf28ce6dedaf9fdfa17b3) model.safetensors.

<out> ending in .safetensors: a copy of the base file with the edited tensors replaced (bfloat16).
Otherwise <out> is a folder that receives one float32 .npy per edited tensor, named like the submission Astra wrote
(the bfloat16 values; the evaluation cast the submission to bfloat16, so these give the evaluated model exactly).
Every rebuilt tensor is checked against the sha256 of the evaluated checkpoint's tensor.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_numpy as S  # noqa: E402


def rebuild(base_bits, t, z, i):
    if t["form"] == "lowrank":
        w = S.bf16_to_f32(base_bits).astype(np.float64)
        a, b = z[f"t{i}_a"], z[f"t{i}_b"]
        for k in range(a.shape[1]):
            w = w + np.outer(a[:, k], b[k])
        e = S.bf16_bits(w.astype(np.float32)).reshape(-1)
    else:
        e = base_bits.copy().reshape(-1)
    e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
    return e.reshape(base_bits.shape)


def main(base_path, edit_dir, out):
    man = json.load(open(os.path.join(edit_dir, "manifest.json")))
    z = np.load(os.path.join(edit_dir, "edit.npz"))
    names = [t["name"] for t in man["tensors"]]
    if out.endswith(".safetensors"):
        tensors = S.load(base_path)
    else:
        tensors = S.load(base_path, names)
    for i, t in enumerate(man["tensors"]):
        dt, b = tensors[t["name"]]
        assert dt == "BF16" and list(b.shape) == t["shape"], (t["name"], dt, b.shape)
        h = hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
        assert h == t["base_sha256"], f"{t['name']}: the base tensor differs from the one the edit was made on"
        e = rebuild(b, t, z, i)
        h = hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest()
        assert h == t["edited_sha256"], f"{t['name']}: rebuilt tensor does not match the evaluated checkpoint"
        tensors[t["name"]] = ("BF16", e)
        print(f"{t['name']}: rebuilt ({t['form']}, {t['changed_entries']:,} entries changed), hash OK")
    if out.endswith(".safetensors"):
        S.save(out, tensors, {"format": "pt"})
    else:
        os.makedirs(out, exist_ok=True)
        for n in names:
            np.save(os.path.join(out, n + ".npy"), S.bf16_to_f32(tensors[n][1]))
    print(f"{len(names)} tensors -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
