"""Combine a checkpoint's evaluation: <eval>/<name>/scores.json (whole sets) and <eval>/<name>/parts/*/scores.json
(one set or one shard of a set per job) -> <eval>/<name>/merged.json with acc / cut_share / mean_tokens per set and
how many problems are covered.   python merge.py <eval_dir> [names...]"""
import glob, json, os, sys

SETS = ("aime24", "aime25", "amc23", "math500", "minerva", "olympiad", "gsm8k", "mmlupro1k", "gpqa", "ifeval")
N = {"aime24": 30, "aime25": 30, "amc23": 40, "math500": 500, "minerva": 272, "olympiad": 674, "gsm8k": 1319, "mmlupro1k": 1000, "gpqa": 198, "ifeval": 541}


def merge(d):
    acc = {}
    full = os.path.join(d, "scores.json")
    if os.path.exists(full):
        for k, v in json.load(open(full)).items():
            if k in SETS and not v.get("shard"):
                acc[k] = {"ok": v.get("ok", v["acc"] * v["n_problems"] * v["samples"]),
                          "total": v.get("total", v["n_problems"] * v["samples"]),
                          "cut": v.get("cut", v["cut_share"] * v["n_problems"] * v["samples"]),
                          "tokens_sum": v.get("tokens_sum", v["mean_tokens"] * v["n_problems"] * v["samples"]),
                          "problems": v["n_problems"], "from": "full"}
    for f in glob.glob(os.path.join(d, "parts", "*", "scores.json")):
        for k, v in json.load(open(f)).items():
            if k not in SETS or k in acc and acc[k]["from"] == "full":
                continue
            a = acc.setdefault(k, {"ok": 0, "total": 0, "cut": 0, "tokens_sum": 0, "problems": 0, "from": "parts",
                                   "shards": []})
            if v.get("shard") in a["shards"]:
                continue
            a["shards"].append(v.get("shard"))
            for x in ("ok", "total", "cut", "tokens_sum"):
                a[x] += v[x]
            if "ok_lenient" in v and a.get("ok_lenient", 0) is not None:     # secondary, lenient score (evaluate.py)
                a["ok_lenient"] = a.get("ok_lenient", 0) + v["ok_lenient"]
            else:
                a["ok_lenient"] = None
            a["problems"] += v["n_problems"]
    out = {k: {"acc": a["ok"] / a["total"], "cut_share": a["cut"] / a["total"], "mean_tokens": a["tokens_sum"] / a["total"],
               "problems": a["problems"], "complete": a["problems"] == N[k],
               "acc_lenient": a["ok_lenient"] / a["total"] if a.get("ok_lenient") is not None else None}
           for k, a in acc.items()}
    json.dump(out, open(os.path.join(d, "merged.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    root = sys.argv[1]
    names = sys.argv[2:] or sorted(os.listdir(root))
    for n in names:
        d = os.path.join(root, n)
        if os.path.isdir(d):
            m = merge(d)
            print(f"{n:14s}", " ".join(f"{k} {100 * m[k]['acc']:5.1f}{'' if m[k]['complete'] else '*'}" for k in SETS if k in m))
