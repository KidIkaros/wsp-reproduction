# Weight-Space Programming — independent recreation (Qwen3.5-0.8B)

Recreation of Experiment 8 of
["Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang, Khattab — MIT CSAIL, Nov 2026), run on Kaggle T4s (~2.7 GPU-hours
total) without any frontier-LLM-in-the-loop: the diagnosis and edit were performed by
scripts, not by GPT-6 Astra.

**Models produced:**
- `hf.co/KidIkaros/qwen3.5-0.8b-loop-fix` — the variant with an **independently
  diagnosed** loop direction
- `hf.co/KidIkaros/qwen3.5-0.8b-loop-fix-reproduction` — a **bit-exact rebuild**
  of the paper's released edit (every edited tensor sha256-verifies against the authors'
  manifest)

## What we replicated

The base instruct model, thinking mode (temp 0.6, top-p 0.95, top-k 20, presence 1.5),
fails with **semantic thinking loops**: it re-derives the same step in different words
(zero exact n-gram repetition — max repeated 40-gram = 1), exhausting its token budget.
Presence penalty 1.5 vs 0 makes no difference: the failure is in the weights.

On a paired 18-problem protocol (16 GSM8K + 2 MATH train, seed-0 shuffle, 16,384-token
cap, 1 sample):

| arm | thinking closed | boxed answer | correct | rescued / regressions |
|---|---|---|---|---|
| base (stage 1, presence 1.5) | 2/18 (11%) | 2/18 | — | — |
| base (stage 1, presence 0) | 1/18 (6%) | 1/18 | — | — |
| **our edit (independent diagnosis)** | **15/18 (83%)** | 14/18 | 12/18 (67%) | 13/16, 0 reg. |
| **paper's edit (bit-exact rebuild)** | **17/18 (94%)** | 14/18 | 12/18 (67%) | 15/16, 0 reg. |

Small-n caveat: this is a mechanism replication, not a benchmark score. The paper's
MATH-500 7.9 -> 48.1 under its full protocol was not re-measured.

## The edit, recovered exactly from the released weights

The paper's `p50c` edit touches exactly two tensors (`layers.15.mlp.down_proj.weight`,
`layers.19.mlp.down_proj.weight`) and has an exact closed form:

    W[:, j] <- W[:, j] - 6 * (W[:, j] . d_hat) * d_hat     for 64 neurons j per layer

Strength 6.000 ± 0.002 across all 128 neurons; untouched columns bit-identical;
residual diffs are bf16 rounding. Astra over-corrects past orthogonality, sign-flipping
the loop-writing component. Our variant uses the same rule with a direction diagnosed
independently (mean MLP-output difference between looping and finished-thinking
transcripts, mapped through down_proj; cos 0.35 / 0.45 vs theirs — the loop subspace
is robust to direction choice at this strength).

Side-finding: the loop is *semantic*, not lexical (paraphrase loops with no exact
repetition), which is why sampling fixes (presence penalty) are inert and token-level
repetition SAE features (Neuronpedia's layer-11 set for this model) don't cover it.

## Repo layout

```
PLAN.md            decision-tree plan, outcome bands, deviations ledger (written before each run)
src/wspio.py       minimal safetensors/bf16 I/O (mirrors the authors' st_numpy)
src/driver.py      stage-1 kernel driver: baseline loop diagnostic, 2 arms (presence 1.5 / 0)
src/launcher.py    stage-1 launcher: one process per T4 (base64 driver, single-file kernel)
src/driver2.py     stage-2 kernel driver: diagnosis -> edit -> paired verification
src/launcher2.py   stage-2 launcher
build.py, build2.py  kernel builders (single-file script kernels for `kaggle kernels push`)
kernel/, kernel2/    generated kernels (also runnable as-is)
ground_truth/      loop directions + neuron sets extracted from the paper's released edit
artifacts/         per-problem results (stage 1 + stage 2), diagnosis json, kernel logs
logs/exp8_math/    sparse checkout of the authors' lm_understand-logs (their code + results)
```

## Reproducing

Requirements: a Kaggle account with GPU quota, `kaggle` CLI credentials, and ~2.5h of
T4 time (sessions bill per session, so both cards run independent jobs for free).

```bash
# stage 1: baseline loop diagnostic (~1.6h)
python3 build.py && kaggle kernels push -p kernel
# stage 2: diagnosis + our edit + their edit, paired verification (~1.1h)
python3 build2.py && kaggle kernels push -p kernel2
```

Stage 2 needs the private Kaggle dataset `mauricew/wsp-s2-materials` (edit.npz,
manifest, stage-1 transcripts, ground-truth directions); regenerate it from
`materials/` or point `MAT_DIR` at any directory with those files. Each kernel
self-terminates via a wall-clock guard, writes results incrementally, and aborts
before spending generation if a rebuilt checkpoint has missing weights.

Grading is text-based (`</think>` present + boxed answer) because HF `generate()` pads
eos-finished rows with the EOS token, making per-row finish flags unreliable.

## Credit

All credit for the discovery, the edit and the released artifacts:
Takehi, Teoh, Zhang, Khattab (MIT CSAIL). Their logs:
[rikiyat/lm_understand-logs](https://github.com/rikiyat/lm_understand-logs); their
edited model: [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
Model weights: Qwen (apache-2.0). This repo: reproduction only.
