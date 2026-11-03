"""Stage-1 driver: baseline loop diagnostic for one GPU card.

Env:
  ARM_NAME     e.g. presence1.5 / presence0
  PRESENCE     float presence penalty applied to generated tokens
  OUT_DIR      where results_<arm>.jsonl + meta_<arm>.json are written
  MAX_NEW      max new tokens per generation (default 16384)
  DEADLINE_S   wall-clock guard; os._exit after this many seconds
  N_GSM / N_MATH  problems to use (16 / 8 default)

Writes one JSON line per problem INCREMENTALLY:
  {id, source, problem, gold, output, n_gen, finish, repetitive, max_ngram_count}
"""
import json
import os
import random
import threading
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

ARM = os.environ.get("ARM_NAME", "arm")
PRESENCE = float(os.environ.get("PRESENCE", "1.5"))
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working")
MAX_NEW = int(os.environ.get("MAX_NEW", "16384"))
DEADLINE = float(os.environ.get("DEADLINE_S", "5700"))
N_GSM = int(os.environ.get("N_GSM", "16"))
N_MATH = int(os.environ.get("N_MATH", "8"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
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


log("arm", ARM, "presence", PRESENCE, "max_new", MAX_NEW,
    "cuda_devices", torch.cuda.device_count(),
    "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

# ---------------------------------------------------------------- problems
from datasets import load_dataset  # noqa: E402

gsm = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
probs = []
for i in range(N_GSM):
    r = gsm[i]
    probs.append(dict(id=f"gsm8k_train_{i}", source="gsm8k",
                      problem=r["question"],
                      gold=r["answer"].split("####")[-1].strip()))
math_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default",
                       split="train").shuffle(seed=0)
for i in range(N_MATH):
    r = math_ds[i]
    probs.append(dict(id=f"math_train_{i}", source="math",
                      problem=r["problem"], gold=r["solution"]))
log("problems:", len(probs))

# ---------------------------------------------------------------- model
tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16,
                                             device_map="cuda")
model.eval()
log("model loaded, dtype", model.dtype)
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
EOS_IDS = [IM_END, tok.eos_token_id]
log("stop ids:", EOS_IDS)


class PresencePenalty(LogitsProcessor):
    """OpenAI/vLLM semantics: flat logit subtraction for tokens already generated."""

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
    outs = []
    for b in range(len(rows)):
        gen_ids = out[b, plens[b]:]
        outs.append(gen_ids)
    return outs


def max_rep_ngram(ids, n=40):
    """Count the most repeated n-gram in the generated token ids."""
    if len(ids) < n:
        return 0
    seen = {}
    best = 0
    for i in range(0, len(ids) - n + 1):
        key = tuple(ids[i:i + n])
        seen[key] = seen.get(key, 0) + 1
        best = max(best, seen[key])
    return best


# ---------------------------------------------------------------- throughput probe
probe_rows = probs[:4]
t0 = time.time()
outs = gen_batch(probe_rows)
toks = sum(o.shape[0] for o in outs)
min_of = min(o.shape[0] for o in outs)
tps = toks / (time.time() - t0)
log(f"probe: {toks} tokens in {round(time.time()-t0,1)}s = {tps:.0f} tok/s aggregate")

# projected: each problem worst-case MAX_NEW, batch 6
n_total = len(probs)
worst_s = n_total / 6 * MAX_NEW / max(tps, 1)
log(f"worst-case projection for {n_total} problems: {worst_s/60:.0f} min; "
    f"remaining budget: {(DEADLINE-(time.time()-t_start))/60:.0f} min")
if worst_s > 0.75 * (DEADLINE - (time.time() - t_start)):
    keep = max(8, int(n_total * 0.75 * (DEADLINE - (time.time() - t_start)) / worst_s))
    log("CUTTING problems to", keep, "- recorded as deviation")
    probs = probs[:keep]

with open(META, "w") as f:
    json.dump(dict(status="running", arm=ARM, presence=PRESENCE,
                   n_problems=len(probs), probe_tps=tps, max_new=MAX_NEW), f)

# ---------------------------------------------------------------- main loop
BATCH = 6
for bi in range(0, len(probs), BATCH):
    rows = probs[bi:bi + BATCH]
    t0 = time.time()
    outs = gen_batch(rows)
    log(f"batch {bi//BATCH}: {sum(o.shape[0] for o in outs)} tokens, "
        f"{round(time.time()-t0,1)}s")
    for r, gen_ids in zip(rows, outs):
        ids = gen_ids.tolist()
        mx = max_rep_ngram(gen_ids)
        finished = IM_END in gen_ids
        rec = dict(id=r["id"], source=r["source"], problem=r["problem"],
                   gold=r["gold"], output=tok.decode(gen_ids,
                                                     skip_special_tokens=True),
                   n_gen=len(ids), finish="eos" if finished else "length",
                   max_ngram_count=mx,
                   repetitive=bool(mx >= 5),
                   hit_cap=bool(not finished and len(ids) >= MAX_NEW - 24))
        with open(RES, "a") as f:
            f.write(json.dumps(rec) + "\n")

# ---------------------------------------------------------------- summary
recs = [json.loads(l) for l in open(RES)]
n = len(recs)
hit = sum(r["hit_cap"] for r in recs)
rep = sum(r["repetitive"] for r in recs)
avg = sum(r["n_gen"] for r in recs) / max(n, 1)
summary = dict(status="complete", arm=ARM, presence=PRESENCE, n=n,
               hit_cap=hit, hit_cap_frac=hit / max(n, 1),
               repetitive=rep, repetitive_frac=rep / max(n, 1),
               mean_gen_tokens=avg, elapsed_s=time.time() - t_start)
json.dump(summary, open(META, "w"), indent=1)
log("SUMMARY", json.dumps(summary))
