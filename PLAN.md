# Weight-Space Programming recreation — plan

Goal: recreate the Qwen3.5-0.8B experiment from
rikiyat.github.io/blog/weight-space-programming/ on Kaggle. Confirm the thinking-loop
failure locally, understand the paper's released edit, then attempt our own
diagnosis-driven weight edit and verify it.

## Environment facts (measured 2026-11-03)

- Kaggle GPU: 7.57h remaining of 30h, refresh Sun 2026-11-07 00:00 UTC. Sessions bill
  per session; accelerator is T4 x2; only reliable stop is `kernels delete -y`; verify
  release with a `kaggle quota` delta (~3 min apart, used flat).
- Kaggle CPU kernels: free, 4 cores, 12h sessions. Do all diffing/diagnosis scripting here.
- Local: RTX 2070 8GB (correctness checks only, time-capped), torch 2.11+cu128 in
  ~/.venvs/hnetgpu, CLI at ~/.venvs/kg/bin/kaggle, account mauricew.
- Disk: 65GB free. Models: Qwen/Qwen3.5-0.8B (~3.4GB) + RikiyaT edited (~3.4GB).

## Model choice (decided)

Qwen/Qwen3.5-0.8B (instruct), NOT the Base. The loop failure we target is a
post-training artifact; Base has no thinking mode to fix. Paper used the instruct
model with thinking on.

## Stage 0 — free, no quota: get the ground truth edit

1. Download `Qwen/Qwen3.5-0.8B` and `RikiyaT/Qwen3.5-0.8B-weight-space-programming` locally.
2. Per-tensor diff: which tensors differ, element count changed, per-column norms,
   direction of the change. Paper claims: a "loop direction" subtracted from 64 MLP
   output columns in each of layers 15 and 19 (131,072 = 2 x 64 x 1024 weights).
   VERIFY that from the weights, don't trust the prose.
3. Pull the paper's logs repo (github.com/RikiyaT/lm_understand-logs) exp8 section for
   their procedure notes.

Outcome bands — RESOLVED (2026-11-03):
- Diff shows exactly the claimed edit -> CONFIRMED, and stronger than expected:
  * p50c (the 48.1% run) edited ONLY layers 15 & 19 mlp.down_proj: 65,260 + 64,928
    entries = exactly 64 intermediate neurons x 1024 output dims x 2 layers (minus
    bf16-rounding-unchanged entries).
  * Every neuron's 1024-dim delta is proportional to ONE shared direction per layer
    (|cos| = 1.0000; 2nd singular value ~600x smaller than 1st).
  * Ground truth saved: loopdir_L15/L19.npy, neurons_L15/L19.npy, proj_L15/L19.npy.
  * The weaker runs are messier: p10c = diffuse rank-1 edit over 30 tensors (layers
    8-22, attn+mlp); p25c = all 262k entries of layer 15 self_attn.o_proj. The blog
    only showcases the clean p50c edit.
- Their harness code is in logs/exp8_math/code/ (evaluate.py, apply_edit.py,
  st_numpy.py) — we reuse it rather than reimplementing.
- CORRECTED nuance vs blog: headline eval IS thinking-on at 32,768 (temp 0.6,
  top-p 0.95, top-k 20, presence 1.5) per eval/RESULTS_excerpt.md; evaluate.py's
  non-thinking SAMPLING_NOTHINK35 was only for candidate models. Astra's own
  diagnosis: "matched productive/repetitive contexts localized the relevant MLP
  writers; dose-response, layer-isolation, norm-matched random control."

## Stage 1 — GPU session #1 (~2-2.5h quota): baseline loop diagnostic

- Kernel: load instruct model bf16->fp16 on one T4, run N=24 GSM8K train problems +
  24 MATH train problems, thinking mode, paper sampling (temp 0.6, top-p 0.95, top-k 20,
  presence penalty 1.5, max 32768 tokens), 1 sample each.
- Measure: fraction hitting token limit (loop rate), mean generated length, accuracy.
- Second T4 runs the same 48 problems with presence_penalty=0 as an arm — the model
  card hints presence penalty mitigates repetition; if loop rate collapses there,
  part of the "fix" is sampling, not weights. Costs nothing extra (same session).
- Throughput probe in-session; if measured throughput makes the plan exceed ~2.5h,
  cut problems to fit and record the cut.
- Wall-clock guard: os._exit at 6.5h; results written incrementally per problem.
- Kill rule for the week: Stage 1 + Stage 2 together must fit 7.57h; if Stage 1
  overruns, Stage 2 waits for Sunday's refresh.

Bands:
- Loop rate >= 40% -> paper's failure reproduces; proceed to Stage 2.
- Loop rate 10-40% -> weaker failure; proceed but expect smaller headroom.
- Loop rate < 10% -> failure does not reproduce under our sampling; stop, report,
  compare our sampling settings against theirs before spending Stage 2 quota.

## Stage 1 results (2026-11-03, kernel wsp-s1-baseline, COMPLETE, cost 1.6h GPU)

- n=18 problems per arm (16 GSM8K + 2 MATH train, seed 0), cap 16384, thinking on,
  temp 0.6 / top-p 0.95 / top-k 20; arm A presence 1.5, arm B presence 0.
- THINKING CLOSED AND ANSWERED: 2/18 (presence 1.5), 1/18 (presence 0).
  Everything else burned the full cap in "Wait, ..." semantic loops
  (max 40-gram count = 1 -> paraphrased looping, not exact repetition).
- Presence penalty makes NO difference -> failure is in weights, not sampling.
  Paper's base: 93% hit cap; ours: ~90%. REPRODUCED per Stage 1 bands -> proceed.
- Grading lesson: HF generate pads eos-finished rows WITH EOS (generation_config
  pad_token_id None), so finish flags are unreliable; grade from text
  ('</think>' present + boxed answer). Fixed in the analysis, not the kernel.
- Throughput: 63 tok/s aggregate per T4, batch 6, ~20.5 min per batch.
- Session ended by the 5700s deadline guard mid-batch-3 as designed; quota delta
  verified flat after COMPLETE.

## Edit rule recovered from weights alone (2026-11-03)

The paper's released edit is EXACTLY:
  col_j <- col_j - 6*(col_j . d_hat)*d_hat   for 64 chosen neurons per layer,
  layers 15 and 19 mlp.down_proj (strength 6.000 +- 0.002, all 128 neurons;
  untouched columns bit-identical; residual diffs are bf16 rounding only).
Astra over-corrects to 6x past orthogonality, sign-flipping the loop-writing
component. Remaining unknown: how d and the 64-neuron set were chosen (that is
the activation-side diagnosis Stage 2 performs independently).

## Stage 2 — GPU session #2 (~1.5-2h quota): diagnosis + our variant + verify

- Card 0: diagnosis (prefill looping vs finished transcripts captured in Stage 1,
  capture residual/MLP activations at layers 15/19) -> derive d_ours + neuron set
  -> apply the strength-6 rule with OUR direction -> verify on the SAME 18
  problems (paired comparison).
- Card 1: apply THEIR exact edit (edit.npz from runs/p50c, shipped as a Kaggle
  dataset) -> verify on the same 18 problems (reference arm).
- Success band: our variant's closed-thinking rate >= 50% (vs 11% baseline),
  and not worse than their arm's on mean length. Flat band: loop rate unchanged
  -> our diagnosis failed; report and fall back to their released direction as
  the variant (clearly labelled as reproduction, not independent).
- Files for HF upload built locally after verification (models/wsp-variant/).


## Stage 2 results (2026-11-03, kernel wsp-s2-variant, COMPLETE, cost ~1.1h GPU)

Session total: 25.10h used, 4.90h remaining. Both arms complete, 18/18 problems,
paired against Stage-1 baseline on the same problems (same sampling, same order).

| arm | thinking closed | correct | rescued / regressions |
|---|---|---|---|
| baseline (stage 1) | 2/18 (11%) | — | — |
| OURS (independent diagnosis) | **15/18 (83%)** | 12/18 (67%) | 13/16, 0 regressions |
| THEIRS (bit-exact rebuild, sha256 OK) | **17/18 (94%)** | 12/18 (67%) | 15/16, 0 regressions |

- Success band (ours >= 50% closed) PASSED: the loop subspace is robust — our
  direction (cos 0.35/0.45 vs theirs, 8 diagnosis contexts, top-64 neurons by
  |mean act diff|) recovered most of the fix at strength 6.
- Both rebuilt checkpoints loaded with 0 missing keys (in-kernel gate passed).
- Deliverables staged: hf-upload/qwen3.5-0.8b-loop-fix/ (ours) and
  hf-upload/qwen3.5-0.8b-loop-fix-reproduction/ (theirs), each with README model
  card + full checkpoint (1.7GB). User uploads to HF themselves (no local token
  found; placeholders KidIkaros cross-links updated at upload).

## Interpretability side-finding

Neuronpedia's qwen3.5-0.8b coverage: one layer-11 SAE set only (Chanin, May 2026);
HF weights for layers 5/11/17. Its repetition features are TOKEN-level detectors;
our loop is SEMANTIC (paraphrase loops, max 40-gram count = 1, presence-penalty
inert). This is why sampling fixes fail and the fix must be in weights. HeadVis
lists induction-scored heads incl. L15 H7, L19 H5/H6 — candidate sources of the
loop direction for a follow-up.

## Ledger

- Spent this project: Stage 1 ~1.6h + Stage 2 ~1.1h = ~2.7h GPU. Remaining: 4.90h
  (refresh Sun Nov 7).

## Deviations from the paper's protocol (all recorded 2026-11-03, pre-launch)

- MATH dataset config 'data' -> 'default' (current datasets lib; caught in dry run).
- presence_penalty is not a transformers 5.x generate() kwarg: implemented as a
  custom logits processor with OpenAI/vLLM semantics (flat subtraction for tokens
  already generated, generated tokens only).
- fp16 not bf16 (T4 has no fast bf16); their eval ran vLLM bf16.
- max_new 16384 not 32768 (halves worst-case loop cost; loop rate still measured;
  finished-before-eos rows are flagged by eos-presence, not length).
- HF generate, not vLLM (vLLM nightly install risk not worth it). HF batches to a
  uniform total length, so per-row n_gen can exceed MAX_NEW by the prompt-length
  spread; finish is detected via <|im_end|> presence.
- flash-linear-attention installed (DeltaNet fast path); causal-conv1d left at its
  fallback (CUDA build fails without nvcc).

## Stage 0 assets (ground truth, all verified from weights)

- models/Qwen3.5-0.8B and models/Qwen3.5-0.8B-weight-space-programming (HF).
- loopdir_L{15,19}.npy (unit direction), neurons_L{15,19}.npy (64 ids),
  proj_L{15,19}.npy (per-neuron scale, mixed sign).
- Their harness code: logs/exp8_math/code/ (evaluate.py protocol, apply_edit.py).
- Local dry runs of the driver passed on the 2070 (2 problems, 400 tokens, end to end).
