"""Practice problems for the Qwen3.5-0.8B runs (profile qwen35; the editor chooses which datasets
its forward passes use). Three files in PRACTICE (read-only at /practice in the sandbox):
  gsm8k_train.jsonl  GSM8K train split (openai/gsm8k main), answer = the number after ####
  math_train.jsonl   MATH train split, all levels and subjects (EleutherAI/hendrycks_math), answer = last \\boxed{}
  aime.jsonl         AIME 1983-2023 (the practice set of the Qwen3-1.7B redesign, already decontaminated)
Decontamination against the evaluation sets of this profile (MATH-500, GSM8K test) and the other four: a problem is
dropped if a 13-word window or its normalized text occurs in any evaluation problem (practice.grams / norm).
  python practice35.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from weightenv.lmo.tasks import mathqa as Q  # noqa: E402
import evaluate as EV  # noqa: E402
import practice as PR  # noqa: E402

OUT = "<data>/mathedit/practice35"


def main():
    from datasets import load_dataset
    os.makedirs(OUT, exist_ok=True)
    tg, tn = set(), set()
    for s in EV.SETS + ("gsm8k",):
        for r in EV.load_set(s):
            tg |= PR.grams(r["problem"])
            tn.add(PR.norm(r["problem"]))
    clean = lambda p: PR.norm(p) not in tn and not (PR.grams(p) & tg)
    stats = {}

    def dump(name, rows):
        kept = [r for r in rows if r["answer"] not in (None, "") and clean(r["problem"])]
        with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = {"kept": len(kept), "dropped": len(rows) - len(kept)}

    dump("gsm8k_train", [{"id": f"gsm8k_train_{i}", "source": "gsm8k", "problem": r["question"],
                          "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
                          "solution": r["answer"].split("####")[0].strip()}
                         for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train"))])
    rows = []
    for sub in PR.SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", sub, split="train")):
            rows.append({"id": f"math_train_{sub}_{i}", "source": "math", "subject": sub, "level": r["level"],
                         "problem": r["problem"], "answer": Q.gold_of_solution(r["solution"]), "solution": r["solution"]})
    dump("math_train", rows)
    dump("aime", [r for r in (json.loads(l) for l in open(os.path.join(PR.PRACTICE, "problems.jsonl"))) if r["source"] == "aime"])
    open(os.path.join(OUT, "README.txt"), "w").write(
        "Practice problems (none of them is in the evaluation). One JSON object per line with id, source, problem, answer "
        "and, for gsm8k_train and math_train, a reference solution (math_train also subject and level).\n" + "\n".join(f"{k}.jsonl: {v['kept']} problems" for k, v in stats.items()) + "\n")
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
