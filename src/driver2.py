"""Stage-2 driver: diagnose the loop direction from Stage-1 transcripts, build OUR
variant edit, and verify both our variant and the paper's exact edit on the same
18 problems.

Arms (one per GPU card, launched by launcher2.py):
  ARM_NAME=ours   -> diagnosis -> apply strength-6 rule with OUR direction to the
                     top-64 neurons -> save edited safetensors -> verify generation.
  ARM_NAME=theirs -> apply the paper's released edit (edit.npz, sha-checked) ->
                     save edited safetensors -> verify generation.

Env: OUT_DIR, DEADLINE_S, MAX_NEW (16384), PRESENCE (1.5), MAT_DIR (dataset mount).
Writes: results_<arm>.jsonl (incremental), meta_<arm>.json, model_<arm>/ (edited
safetensors + configs for HF), diag_<arm>.json (ours only).
"""
import hashlib
import json
import os
import shutil
import threading
import time

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wspio import st_load, st_save, bf16_bits, bf16_to_f32, NAMES, L15, L19

ARM = os.environ.get("ARM_NAME", "arm")
PRESENCE = float(os.environ.get("PRESENCE", "1.5"))
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working")
MAX_NEW = int(os.environ.get("MAX_NEW", "16384"))
DEADLINE = float(os.environ.get("DEADLINE_S", "7200"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
MAT_DIR = os.environ.get("MAT_DIR", ".")
RES = os.path.join(OUT_DIR, f"results_{ARM}.jsonl")
META = os.path.join(OUT_DIR, f"meta_{ARM}.json")

t_start = time.time()


def guard():
    time.sleep(DEADLINE)
    json.dump({"status": "deadline_fired", "arm": ARM,
               "elapsed_s": time.time() - t_start}, open(META, "w"))
    os._exit(0)


threading.Thread(target=guard, daemon=True).start()


def log(*a):
    print(f"[{ARM} {round(time.time()-t_start)}s]", *a, flush=True)


# ------------------------------------------------- problems: same 18 as stage 1
s1 = [json.loads(l) for l in open(os.path.join(MAT_DIR, "s1_presence1.5.jsonl"))]
PROBS = [dict(id=r["id"], source=r["source"], problem=r["problem"], gold=r["gold"],
              s1_out=r["output"], s1_looped=("</think>" not in r["output"]))
         for r in s1]
log("problems from stage-1:", len(PROBS),
    "looped:", sum(p["s1_looped"] for p in PROBS))

from huggingface_hub import snapshot_download  # noqa: E402


def base_weight_file():
    if os.path.isdir(MODEL_ID):
        snap = MODEL_ID
    else:
        snap = snapshot_download(MODEL_ID)
    return snap, [os.path.join(snap, f) for f in sorted(os.listdir(snap))
                  if f.endswith(".safetensors")][0]


def build_edited_dir(edit_fn):
    """edit_fn(name, W_f64) -> W_f64 edited. Returns dir with full bf16 safetensors."""
    snap, wf = base_weight_file()
    tensors = st_load(wf)
    for name in NAMES:
        dt, bits = tensors[name]
        assert dt == "BF16", (name, dt)
        W = bf16_to_f32(bits).astype(np.float64)
        tensors[name] = ("BF16", bf16_bits(edit_fn(name, W).astype(np.float32)))
    out_dir = os.path.join(OUT_DIR, f"model_{ARM}")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(snap):
        p = os.path.join(snap, f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copy(p, os.path.join(out_dir, f))  # incl. index.json (same layout)
    st_save(os.path.join(out_dir, "model.safetensors-00001-of-00001.safetensors"),
            tensors, {"format": "pt"})
    return out_dir


model_dir = None
if ARM == "theirs":
    man = json.load(open(os.path.join(MAT_DIR, "manifest.json")))
    z = np.load(os.path.join(MAT_DIR, "edit.npz"))
    idx_of = {t["name"]: i for i, t in enumerate(man["tensors"])}
    base_bits = st_load(base_weight_file()[1])  # ALL tensors; we rewrite the full file
    for t in man["tensors"]:
        name = t["name"]
        i = idx_of[name]
        dt, bits = base_bits[name]
        h = hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()
        assert h == t["base_sha256"], f"{name}: base tensor mismatch"
        W = bf16_to_f32(bits).astype(np.float64).reshape(-1)
        W[z[f"t{i}_idx"]] = bf16_to_f32(z[f"t{i}_val"])
        new_bits = bf16_bits(W.reshape(t["shape"]).astype(np.float32))
        h2 = hashlib.sha256(np.ascontiguousarray(new_bits).tobytes()).hexdigest()
        assert h2 == t["edited_sha256"], f"{name}: rebuilt tensor mismatch"
        log(f"{name}: sha256 OK ({t['changed_entries']:,} entries)")
        base_bits[name] = ("BF16", new_bits)
    out_dir = os.path.join(OUT_DIR, f"model_{ARM}")
    os.makedirs(out_dir, exist_ok=True)
    snap = base_weight_file()[0]
    for f in os.listdir(snap):
        p = os.path.join(snap, f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copy(p, os.path.join(out_dir, f))
    st_save(os.path.join(out_dir, "model.safetensors-00001-of-00001.safetensors"),
            base_bits, {"format": "pt"})
    model_dir = out_dir
    log("theirs: edited model saved to", model_dir)
else:
    # ------------------------------------------------- diagnosis (ours)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16,
                                                 device_map="cuda")
    model.eval()
    log("base model loaded for diagnosis")

    mods = {}
    for n, m in model.named_modules():
        if n.endswith("mlp.down_proj"):
            parts = n.split(".")
            if "layers" in parts:
                Li = parts.index("layers")
                L = int(parts[Li + 1])
                if L in (15, 19):
                    mods[L] = m
    assert sorted(mods) == [15, 19], sorted(mods)
    log("hooking layers", sorted(mods))

    captured = {}

    def make_hook(L):
        def hook(module, args, output):
            if captured.get("want") is True:
                captured[L] = args[0][0].detach().float().cpu()
        return hook

    handles = [m.register_forward_hook(make_hook(L)) for L, m in mods.items()]

    def chat_ids(problem, body):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": problem + "\nPlease reason step by step, "
              "and put your final answer within \\boxed{}."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok(prompt + body).input_ids
        # keep the END of the context: the loop zone is at the tail
        return torch.tensor(ids[-8192:], dtype=torch.long).unsqueeze(0).cuda()

    loop_texts = [(p["problem"], p["s1_out"][-16000:])
                  for p in PROBS if p["s1_looped"]][:6]
    prod_texts = [(p["problem"], p["s1_out"][:16000])
                  for p in PROBS if not p["s1_looped"]][:4]
    log("diagnosis contexts: loop", len(loop_texts), "prod", len(prod_texts))

    sums = {L: {"loop": np.zeros(3584), "prod": np.zeros(3584),
                "nloop": 0, "nprod": 0} for L in mods}
    for label, texts in (("loop", loop_texts), ("prod", prod_texts)):
        for problem, body in texts:
            captured.clear()
            captured["want"] = True
            with torch.no_grad():
                model(chat_ids(problem, body))
            captured["want"] = False
            for L in mods:
                act = captured[L]                       # [seq, 3584]
                sums[L][label] += act.sum(0).numpy()
                sums[L]["n" + label] += act.shape[0]
            log(f"captured {label}: {problem[:40]!r}")
    for h in handles:
        h.remove()

    snap, wf = base_weight_file()
    base_bits = st_load(wf, NAMES)
    diag = {}
    edits = {}
    for L in (15, 19):
        name = L15 if L == 15 else L19
        d_act = (sums[L]["loop"] / max(sums[L]["nloop"], 1)
                 - sums[L]["prod"] / max(sums[L]["nprod"], 1))
        _, bits = base_bits[name]
        W = bf16_to_f32(bits).astype(np.float64)
        d_res = W @ d_act
        d_unit = d_res / np.linalg.norm(d_res)
        neurons = np.argsort(-np.abs(d_act))[:64]
        alphas = -6.0 * (W[:, neurons].T @ d_unit)
        edits[name] = (neurons, d_unit, alphas)
        ref = np.load(os.path.join(MAT_DIR, f"loopdir_L{L}.npy"))
        ref = ref / np.linalg.norm(ref)
        cos = float(abs(d_unit @ ref))
        diag[str(L)] = dict(cos_to_theirs=cos,
                            top_neurons=neurons.tolist(),
                            n_loop_tok=int(sums[L]["nloop"]),
                            n_prod_tok=int(sums[L]["nprod"]))
        log(f"layer {L}: cos(ours, theirs) = {cos:.4f}")

    def ours_edit(name, W):
        if name not in edits:
            return W
        neurons, d_unit, alphas = edits[name]
        e = W.copy()
        e[:, neurons] = W[:, neurons] + np.outer(d_unit, alphas)
        return e

    model_dir = build_edited_dir(ours_edit)
    json.dump(diag, open(os.path.join(OUT_DIR, f"diag_{ARM}.json"), "w"), indent=1)
    log("ours: edited model saved to", model_dir)
    del model
    torch.cuda.empty_cache()

# ------------------------------------------------- verify: generation
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    LogitsProcessor, LogitsProcessorList  # noqa: E402

tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model, load_info = AutoModelForCausalLM.from_pretrained(
    model_dir, dtype=torch.float16, device_map="cuda", output_loading_info=True)
n_missing = len(load_info["missing_keys"])
log("verification model loaded from", model_dir,
    "| missing keys:", n_missing, "unexpected:", len(load_info["unexpected_keys"]))
if n_missing > 0:
    json.dump({"status": "bad_checkpoint", "arm": ARM,
               "missing": n_missing}, open(META, "w"))
    raise SystemExit(f"{ARM}: rebuilt checkpoint has {n_missing} missing keys - aborting before generation")
model.eval()
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
EOS_IDS = [IM_END, tok.eos_token_id]
log("verification model loaded from", model_dir)


class PresencePenalty(LogitsProcessor):
    def __init__(self, plens, penalty):
        self.plens = plens
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        if self.penalty <= 0:
            return scores
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.plens[b]:]
            if gen.numel():
                scores[b, torch.unique(gen)] -= self.penalty
        return scores


def gen_batch(rows):
    prompts = [
        tok.apply_chat_template(
            [{"role": "user",
              "content": r["problem"] + "\nPlease reason step by step, "
                         "and put your final answer within \\boxed{}."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for r in rows]
    enc = tok(prompts, return_tensors="pt", padding=True)
    plens = enc.attention_mask.sum(1).tolist()
    enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=MAX_NEW, do_sample=True,
            temperature=0.6, top_p=0.95, top_k=20, eos_token_id=EOS_IDS,
            logits_processor=LogitsProcessorList(
                [PresencePenalty(plens, PRESENCE)]))
    return [out[b, plens[b]:] for b in range(len(rows))]


with open(META, "w") as f:
    json.dump(dict(status="running", arm=ARM, presence=PRESENCE,
                   n_problems=len(PROBS), max_new=MAX_NEW), f)

BATCH = 6
for bi in range(0, len(PROBS), BATCH):
    rows = PROBS[bi:bi + BATCH]
    t0 = time.time()
    outs = gen_batch(rows)
    log(f"batch {bi//BATCH}: {sum(o.shape[0] for o in outs)} tokens, "
        f"{round(time.time()-t0,1)}s")
    for r, gen_ids in zip(rows, outs):
        text = tok.decode(gen_ids, skip_special_tokens=True)
        rec = dict(id=r["id"], source=r["source"], gold=r["gold"], output=text,
                   n_gen=len(gen_ids.tolist()),
                   closed_thinking=bool("</think>" in text),
                   s1_looped=r["s1_looped"])
        with open(RES, "a") as f:
            f.write(json.dumps(rec) + "\n")

recs = [json.loads(l) for l in open(RES)]
n = len(recs)
closed = sum(r["closed_thinking"] for r in recs)
summary = dict(status="complete", arm=ARM, presence=PRESENCE, n=n,
               closed=closed, closed_frac=closed / max(n, 1),
               mean_gen_tokens=sum(r["n_gen"] for r in recs) / max(n, 1),
               elapsed_s=time.time() - t_start)
json.dump(summary, open(META, "w"), indent=1)
log("SUMMARY", json.dumps(summary))
