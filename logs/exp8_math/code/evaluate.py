"""Math evaluation of a Qwen3-1.7B checkpoint for the weight-editing experiment (measured on AIME, MATH-500,
Minerva, OlympiadBench, AMC).

Benchmarks (bench/<name>.jsonl, built by `prepare`; never mounted in the editor's sandbox):
  aime24 (HuggingFaceH4/aime_2024, 30)   aime25 (MathArena/aime_2025, 30)   amc23 (math-ai/amc23, 40)
  math500 (HuggingFaceH4/MATH-500, 500)  minerva (math-ai/minervamath, 272)
  olympiad (math-ai/olympiadbench, 674 English text-only open-ended; multiple answers joined with ", ")
Protocol, identical for every checkpoint: chat template with enable_thinking=False, user turn = problem + "\\n" +
"Please reason step by step, and put your final answer within \\boxed{}." (weightenv/lmo/tasks/mathqa.build_prompt);
vLLM sampling at Qwen3's non-thinking setting (temperature 0.7, top_p 0.8, top_k 20), max 8,192 new tokens, seed 0;
SAMPLES per problem (16 for the AIME / AMC sets, 4 for the others); score = mean accuracy over all samples
(avg@n), graded by the last \\boxed{} answer with math-verify (mathqa.is_correct).

  python evaluate.py prepare
  python evaluate.py run <checkpoint_dir> <out_dir> [--sets aime24,aime25,amc23,math500,minerva,olympiad]
"""
import argparse, json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from weightenv.lmo.tasks import mathqa as Q  # noqa: E402

ROOT = "<data>/mathedit"
BENCH = os.path.join(ROOT, "bench")
SETS = ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad")
SAMPLES = {"aime24": 16, "aime25": 16, "amc23": 16, "math500": 4, "minerva": 4, "olympiad": 4, "gsm8k": 2,
           # beyond math (2026-10-24: p10c / p25c / p50c and the unedited model on other datasets too):
           # MMLU-Pro (a fixed random 1,000 of 12,032, the official prompt and answer extraction), GPQA Diamond (the same
           # multiple-choice format, options shuffled per question), IFEval (strict prompt-level, official checkers)
           "mmlupro1k": 1, "gpqa": 4, "ifeval": 2}
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 8192, "seed": 0}
# second experiment (2026-10-22): thinking ON, Qwen3's thinking
# setting, a 32,768-token cap for thinking + answer
SAMPLING_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768, "seed": 0}
# Qwen3.5-0.8B (2026-10-23): its model card's thinking-mode benchmark setting; evaluated on MATH-500 and GSM8K first
SAMPLING_THINK35 = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768,
                    "seed": 0}
# ... but thinking looped at 0.8B; the evaluation uses the model card's NON-thinking benchmark setting instead
SAMPLING_NOTHINK35 = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192,
                      "seed": 0}
# candidate models for the next experiment (2026-10-24: LFM2.5-350M / Gemma 3 270M IT / Qwen3-0.6B), each with
# its model card's recommended sampling
SAMPLING_LFM = {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05, "max_tokens": 8192, "seed": 0}
SAMPLING_GEMMA3 = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 8192, "seed": 0}


def prepare():
    from datasets import load_dataset
    os.makedirs(BENCH, exist_ok=True)

    def dump(name, rows):
        with open(os.path.join(BENCH, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows))

    dump("aime24", [{"id": f"aime24_{r['id']}", "problem": r["problem"], "answer": str(r["answer"])}
                    for r in load_dataset("HuggingFaceH4/aime_2024", split="train")])
    dump("aime25", [{"id": f"aime25_{r['problem_idx']}", "problem": r["problem"], "answer": str(r["answer"])}
                    for r in load_dataset("MathArena/aime_2025", split="train")])
    dump("amc23", [{"id": f"amc23_{r['id']}", "problem": r["question"], "answer": str(r["answer"])}
                   for r in load_dataset("math-ai/amc23", split="test")])
    dump("math500", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]}
                     for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
    dump("minerva", [{"id": f"minerva_{i}", "problem": r["question"], "answer": str(r["answer"])}
                     for i, r in enumerate(load_dataset("math-ai/minervamath", split="test"))])
    rows = []
    for r in load_dataset("math-ai/olympiadbench", split="test"):
        fa = r["final_answer"]
        fa = fa if isinstance(fa, list) else json.loads(fa.replace("'", '"')) if fa.startswith("[") else [fa]
        rows.append({"id": f"olympiad_{r['id']}", "problem": r["question"], "answer": ", ".join(str(x) for x in fa)})
    dump("olympiad", rows)
    dump("gsm8k", [{"id": f"gsm8k_test_{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                   for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))])


def lenient_pred(text):
    """an answer given without \\boxed{} (2026-10-24, Gemma 3 270M IT writes "Final Answer: 72"): the latest phrase after
    'Final Answer', 'the answer is' or 'Answer:' that contains a number or a formula, else the last number; used only for
    the secondary, lenient score."""
    import re
    found = []
    for pat in (r"[Ff]inal [Aa]nswer\s*(?:is)?\s*[:=]?\s*\$*([^\n$]+)", r"[Tt]he (?:final )?answer is\s*[:=]?\s*\$*([^\n$]+)",
                r"\b[Aa]nswer\s*[:=]\s*\$*([^\n$]+)"):
        found += [(m.start(1), m.group(1).strip().strip("*").rstrip(".").strip()) for m in re.finditer(pat, text)]
    for _, t in sorted(found, reverse=True):
        if re.search(r"[0-9\\]", t):
            return t
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def lenient_correct(text, pred, good, answer):
    """strict grade, or (only when no \\boxed{} answer was given) the lenient answer: as written or its last number."""
    if good or pred is not None:
        return bool(good)
    lp = lenient_pred(text)
    if lp is None:
        return False
    import re
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", lp)
    return bool(Q.is_correct(lp, answer) or (nums and Q.is_correct(nums[-1].replace(",", ""), answer)))


def grade_other(r, text, thinking):
    """(correct, pred, extra) of a non-math item: kind 'mc' (mcqa.extract, the official MMLU-Pro reading, the reasoning
    cut at the last </think>) or 'ifeval' (the answer after </think>; with thinking on and no </think> the answer is
    empty; every instruction must be followed)."""
    if r["kind"] == "mc":
        from weightenv.lmo.tasks import mcqa as M
        pred, _ = M.extract(text)
        return pred == r["answer"], pred, {}
    from weightenv.lmo.tasks.ifevalg import instructions_registry as IR
    resp = text.split("</think>")[-1].strip() if "</think>" in text else ("" if thinking else text.strip())
    follow = []
    for iid, kw in zip(r["instruction_id_list"], r["kwargs"]):
        ins = IR.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=r["prompt"])
        follow.append(bool(resp) and bool(ins.check_following(resp)))
    return all(follow), None, {"inst_ok": sum(follow), "inst_total": len(follow)}


def load_set(name):
    return [json.loads(l) for l in open(os.path.join(BENCH, f"{name}.jsonl"))]


def run(ckpt, out, sets, thinking=False, protocol="nothink", shard=None):
    sampling = {"think": SAMPLING_THINK, "think35": SAMPLING_THINK35, "nothink35": SAMPLING_NOTHINK35, "lfm": SAMPLING_LFM,
                "gemma3": SAMPLING_GEMMA3}.get(protocol, SAMPLING)
    thinking = thinking or protocol in ("think", "think35")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ckpt)
    mem = float(os.environ.get("EVAL_GPU_MEM", 0.85))
    cap = os.path.join(ROOT, "vllm_gpu_mem_cap")      # shared GPU pool: a GPU can also hold a server's model (2026-10-23)
    if os.path.exists(cap):
        mem = min(mem, float(open(cap).read()))
    llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=mem,
              max_model_len=sampling["max_tokens"] + 2048, seed=0)
    sp = os.path.join(out, "scores.json")
    job = {"protocol": protocol, "ckpt": ckpt, "ckpt_mtime": os.path.getmtime(os.path.join(ckpt, "config.json")),
           "shard": shard}
    res = json.load(open(sp)) if os.path.exists(sp) else {}
    if res and res.get("_job") != job:      # results of another checkpoint or protocol (2026-10-23 incident): start over
        os.replace(sp, sp + f".stale_{int(time.time())}")
        res = {}
    res["_job"] = job
    for name in sets:
        if name in res:
            continue
        t0 = time.time()
        rows = load_set(name)
        if shard:                                   # i/k: problems i, i+k, i+2k, ... (parallel jobs; merge.py)
            i, k = map(int, shard.split("/"))
            rows = rows[i::k]
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"] if r.get("kind") else Q.build_prompt(r["problem"])}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=thinking) for r in rows]
        sp = SamplingParams(n=SAMPLES[name], **sampling)
        outs = llm.generate(prompts, sp)
        bad = sum(1 for o in outs for x in o.outputs if x.finish_reason not in ("stop", "length"))
        if bad:      # an engine stopped from outside returns partial answers (2026-10-24: t35base gsm8k shards 0-3)
            raise RuntimeError(f"{bad} generations ended without finishing or reaching max_tokens; not saved")
        per, tot, ok, cut, lens, noans, oklen, inst_ok, inst_total = [], 0, 0, 0, [], 0, 0, 0, 0
        for r, o in zip(rows, outs):
            c = []
            for s in o.outputs:
                if r.get("kind"):                          # MMLU-Pro / GPQA / IFEval
                    good, pred, extra = grade_other(r, s.text, thinking)
                    inst_ok += extra.get("inst_ok", 0)
                    inst_total += extra.get("inst_total", 0)
                    c.append(int(good))
                    cut += s.finish_reason == "length"
                    noans += r["kind"] == "mc" and pred is None and s.finish_reason != "length"
                    oklen += good
                    lens.append(len(s.token_ids))
                    continue
                pred, _ = Q.extract(s.text)
                good = Q.is_correct(pred, r["answer"])
                c.append(int(good))
                cut += s.finish_reason == "length"
                noans += pred is None and s.finish_reason != "length"     # finished without a \\boxed{} answer
                oklen += lenient_correct(s.text, pred, good, r["answer"])
                lens.append(len(s.token_ids))
            per.append({"id": r["id"], "correct": c, "pred_first": None if r.get("kind") else Q.extract(o.outputs[0].text)[0]})
            tot += len(c)
            ok += sum(c)
        res[name] = {"acc": ok / tot, "n_problems": len(rows), "samples": SAMPLES[name], "cut_share": cut / tot,
                     "mean_tokens": sum(lens) / len(lens), "s": round(time.time() - t0),
                     "ok": ok, "total": tot, "cut": cut, "tokens_sum": sum(lens), "shard": shard, "no_answer_finished": noans,
                     "ok_lenient": oklen, **({"inst_ok": inst_ok, "inst_total": inst_total} if inst_total else {})}
        json.dump(per, open(os.path.join(out, f"{name}_per_problem.json"), "w"))
        with open(os.path.join(out, f"{name}_first_samples.jsonl"), "w") as f:
            for r, o in list(zip(rows, outs))[:20]:
                f.write(json.dumps({"id": r["id"], "answer": r.get("answer"), "text": o.outputs[0].text}) + "\n")   # IFEval rows have no answer
        json.dump(res, open(os.path.join(out, "scores.json"), "w"), indent=1)
        print(name, json.dumps(res[name]), flush=True)
    res["_protocol"] = {"name": protocol, "sampling": sampling, "samples": SAMPLES, "thinking": thinking,
                        "prompt": Q.INSTRUCTION,
                        "ckpt": ckpt}
    json.dump(res, open(os.path.join(out, "scores.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--protocol", choices=["nothink", "think", "think35", "nothink35", "lfm", "gemma3"], default="nothink")
    ap.add_argument("--shard", help="i/k: evaluate problems i::k only (one set per job; combine with merge.py)")
    ap.add_argument("--thinking", action="store_true", help="reference only: the unedited model with thinking ON "
                    "(same sampling and token cap)")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    else:
        run(os.path.abspath(a.ckpt), os.path.abspath(a.out), a.sets.split(","), a.thinking, a.protocol, a.shard)
