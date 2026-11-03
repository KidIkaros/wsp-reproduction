"""Edited checkpoint for evaluation: Qwen3-1.7B's files with every tensor in <run>/submission/<name>.npy replacing the
original (cast to bfloat16; model.embed_tokens.weight also written to the tied lm_head.weight), same shards and index.
  python build_ckpt.py <run_dir> <out_dir>
"""
import json, os, shutil, sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profiles as PF  # noqa: E402
SRC = PF.get()["model"]
COPY = tuple(f for f in sorted(os.listdir(SRC)) if not f.endswith(".safetensors") and os.path.isfile(os.path.join(SRC, f)))


def main(run, out):
    build(os.path.join(run, "submission"), out, run)


def build(sub, out, run=None):
    """Checkpoint with every <sub>/<name>.npy replacing the original tensor (also used for generation passes)."""
    edits = {f[:-4]: os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".npy")} if os.path.isdir(sub) else {}
    if "lm_head.weight" in edits:
        edits.setdefault("model.embed_tokens.weight", edits.pop("lm_head.weight"))
    os.makedirs(out, exist_ok=True)
    for f in COPY:
        shutil.copyfile(os.path.join(SRC, f), os.path.join(out, f))
    ix = os.path.join(SRC, "model.safetensors.index.json")
    wm = json.load(open(ix))["weight_map"] if os.path.exists(ix) else {"_": "model.safetensors"}
    done, changed = set(), {}
    for shard in sorted(set(wm.values())):
        sd = load_file(os.path.join(SRC, shard))
        for name in list(sd):
            src = edits.get("model.embed_tokens.weight" if name == "lm_head.weight" else name)
            if src is not None:
                a = torch.from_numpy(np.load(src)).to(sd[name].dtype)
                assert a.shape == sd[name].shape, (name, a.shape, sd[name].shape)
                changed[name] = float((a.float() - sd[name].float()).norm() / sd[name].float().norm())
                sd[name] = a.contiguous()
                done.add(name)
        save_file(sd, os.path.join(out, shard), metadata={"format": "pt"})
    missing = set(edits) - done
    assert not missing, missing
    json.dump({"run": run or sub, "edited": sorted(edits), "relative_change": changed},
              open(os.path.join(out, "edits.json"), "w"), indent=1)
    print(f"{len(edits)} tensors replaced -> {out}")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
