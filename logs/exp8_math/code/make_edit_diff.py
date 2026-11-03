"""Compact, exact record of the weight edit a run submitted: the evaluated checkpoint minus the base model.

    python make_edit_diff.py <base model.safetensors> <evaluated model.safetensors> <submission dir> <out dir>

The submission (one float32 .npy per edited tensor, written by Astra) was cast to bfloat16 to build the evaluated
checkpoint. This script compares the evaluated checkpoint with the base model tensor by tensor (bfloat16 bits),
checks that the changed tensors are exactly the submitted ones and that bf16(submission) equals the checkpoint, and
stores each changed tensor in one of two forms:

  sparse   the changed entries: flat indices (int32) and their new bfloat16 bit patterns (uint16)
  lowrank  edits that touch almost every entry (p10c's are rank 1): factors A (rows x r) and B (r x cols), float64,
           from the SVD of submission - base, applied as bf16(float32(base + A @ B summed one rank at a time)), plus
           the few entries ("fixups") where that rounding differs from the checkpoint, stored like a sparse edit

Output: <out>/edit.npz (arrays t<i>_idx, t<i>_val, t<i>_a, t<i>_b) and <out>/manifest.json (tensor names, shapes,
form, changed entries, sha256 of the base and the edited tensor's bfloat16 bytes). apply_edit.py rebuilds the edited
tensors from these and checks the hashes.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import st_numpy as S  # noqa: E402


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def lowrank_recon(base_bits, a, b):
    w = S.bf16_to_f32(base_bits).astype(np.float64)
    for k in range(a.shape[1]):
        w = w + np.outer(a[:, k], b[k])
    return S.bf16_bits(w.astype(np.float32))


def main(base_path, ckpt_path, sub_dir, out):
    base, ed = S.load(base_path), S.load(ckpt_path)
    assert set(base) == set(ed), set(base) ^ set(ed)
    changed = sorted(n for n in base if not np.array_equal(base[n][1], ed[n][1]))
    subs = sorted(f[:-4] for f in os.listdir(sub_dir) if f.endswith(".npy"))
    assert changed == subs, (changed, subs)
    arrays, tensors = {}, []
    for i, n in enumerate(changed):
        dt, b = base[n]
        e = ed[n][1]
        assert dt == "BF16"
        sub = np.load(os.path.join(sub_dir, n + ".npy"))
        assert np.array_equal(S.bf16_bits(sub), e), n
        mask = b != e
        rec = {"name": n, "shape": list(b.shape), "dtype": "bfloat16", "changed_entries": int(mask.sum()),
               "changed_columns": int(mask.any(0).sum()), "changed_rows": int(mask.any(1).sum()),
               "relative_change": float(np.linalg.norm(S.bf16_to_f32(e) - S.bf16_to_f32(b)) /
                                        np.linalg.norm(S.bf16_to_f32(b))),
               "base_sha256": sha(b), "edited_sha256": sha(e), "submitted_file": n + ".npy"}
        if mask.mean() <= 0.3:
            idx = np.flatnonzero(mask)
            arrays[f"t{i}_idx"] = idx.astype(np.int32)
            arrays[f"t{i}_val"] = e.reshape(-1)[idx]
            rec["form"] = "sparse"
        else:
            d = sub.astype(np.float64) - S.bf16_to_f32(b).astype(np.float64)
            u, s, vt = np.linalg.svd(d, full_matrices=False)
            for r in range(1, 9):
                if s[r] <= 1e-6 * s[0]:
                    break
            a, bb = u[:, :r] * s[:r], vt[:r]
            recon = lowrank_recon(b, a, bb)
            fix = np.flatnonzero(recon != e)
            arrays[f"t{i}_a"], arrays[f"t{i}_b"] = a, bb
            arrays[f"t{i}_idx"] = fix.astype(np.int32)
            arrays[f"t{i}_val"] = e.reshape(-1)[fix]
            rec.update(form="lowrank", rank=int(r), singular_values_kept=[float(x) for x in s[:r]],
                       next_singular_value=float(s[r]), fixups=int(len(fix)))
        tensors.append(rec)
        print(f"{n}: {rec['form']}, {rec['changed_entries']:,} changed entries"
              + (f", rank {rec['rank']}, {rec['fixups']} fixups" if rec["form"] == "lowrank" else ""))
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "edit.npz"), **arrays)
    json.dump({"tensors": tensors, "changed_tensors": len(tensors),
               "changed_entries_total": sum(t["changed_entries"] for t in tensors),
               "note": "changed_entries counts bfloat16 entries whose bits differ from the base model; an entry the "
                       "edit touched but that rounds back to the same bfloat16 value is not counted"},
              open(os.path.join(out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
