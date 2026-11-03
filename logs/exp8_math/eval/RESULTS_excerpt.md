# Live results table of the math-editing experiments (excerpt)

Verbatim excerpt of the private repository's results table, final version (2026-10-27 15:06): its sections
"Experiment 6" (Qwen3.5-0.8B, thinking on; the post's runs are the prompt-v3 rows p10c / p25c / p50c, 10 / 25 / 50
passes) and "Experiment 8" (Gemma 3 270M IT; the post's runs are the rows "the lenient grading stated in the prompt",
g05b / g10b / g25b / g50b, main column MATH-500 (lenient)). The other rows are runs that are not in the post.
Accuracy in %, mean over 4 samples per problem (MATH-500, Minerva, OlympiadBench, GPQA), 2 (GSM8K, IFEval), 16 (AIME,
AMC), 1 (MMLU-Pro); bold = more than 2 points above the unedited model; (used k) = forward passes used.

## Experiment 6: Qwen3.5-0.8B, thinking ON (temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, max 32,768); same rules; prompt v3 (512 answers per generation pass, 32,768-token teacher-forced texts) with the thinking-on measurement paragraph; astra effort max; MATH-500 (4 samples) and GSM8K (2 samples)

| Model | MATH-500 | GSM8K | Edit |
|---|---|---|---|
| Unedited Qwen3.5-0.8B (thinking on) | 7.9 | 13.8 |  |
| Edited, 0 passes (prompt v2, max) | 7.9 | 13.8 | no edit made: weights identical to the unedited model |
| Edited, 5 passes (prompt v3: 512 answers / paired passes, max) (used 5) | 7.9 | 13.8 | no edit submitted (= unedited model) |
| Edited, 10 passes (prompt v3: 512 answers / paired passes, max) (used 10) | **12.3** | **25.7** |  |
| Edited, 25 passes (prompt v3: 512 answers / paired passes, max) (used 25) | **14.1** | **25.1** |  |
| Edited, 50 passes (prompt v3: 512 answers / paired passes, max) (used 50) | **48.1** | **67.1** |  |
| Edited, 10 passes (prompt v4: shorter, DEV/HOLDOUT, no failure example) (used 10) | 7.9 | 13.8 | no edit submitted (= unedited model) |
| Edited, 25 passes (prompt v4: shorter, DEV/HOLDOUT, no failure example) (used 25) | **14.1** | **25.1** |  |
|   p05c: its best candidate, which it did not submit (its own acceptance rule); evaluated at the user's request | 8.6 | **17.1** |  |
|   p10d: its best candidate, which it did not submit (its own acceptance rule); evaluated at the user's request | **10.1** | **16.1** |  |
| Edited, 5 passes (prompt v5: TF screening, fixed 1.5 SE rule) (used 0) | … | … |  |
| Edited, 10 passes (prompt v5: TF screening, fixed 1.5 SE rule) | … | … |  |
| Edited, 25 passes (prompt v5: TF screening, fixed 1.5 SE rule) | … | … |  |
| Edited, 50 passes (prompt v5: TF screening, fixed 1.5 SE rule) | … | … |  |

Other math benchmarks, same protocol (thinking on, 32,768 tokens; AIME and AMC 16 samples per problem, Minerva and
OlympiadBench 4), for the unedited model and the three edited runs with the largest gains:

| Model | AIME24 | AIME25 | AMC23 | Minerva | Olympiad | Edit |
|---|---|---|---|---|---|---|
| Unedited Qwen3.5-0.8B (thinking on) | 0.0 | 0.0 | 0.5 | 1.5 | 1.3 |  |
| p10c (v3, 10 passes) (used 10) | 0.0 | 0.0 | 1.1 | **3.9** | 2.4 |  |
| p25c (v3, 25 passes) (used 25) | 0.0 | 0.0 | 1.1 | **3.5** | **3.9** |  |
| p50c (v3, 50 passes) (used 50) | 0.6 | **2.3** | **23.4** | **25.4** | **21.8** |  |

Beyond math (thinking on, 32,768 tokens): MMLU-Pro (a fixed random 1,000 of 12,032 questions, the official prompt and
answer extraction, 1 sample), GPQA Diamond (the same multiple-choice format, 4 samples) and IFEval (strict prompt-level
accuracy with the official checkers on the answer after </think>, 2 samples):

| Model | MMLU-Pro (1k) | GPQA Diamond | IFEval | Edit |
|---|---|---|---|---|
| Unedited Qwen3.5-0.8B (thinking on) | 40.3 | 22.7 | 21.0 |  |
| p10c (v3, 10 passes) (used 10) | 40.0 | **29.4** | **49.0** |  |
| p25c (v3, 25 passes) (used 25) | **44.7** | **30.1** | **39.9** |  |
| p50c (v3, 50 passes) (used 50) | **45.5** | **36.6** | **62.3** |  |

## Experiment 8: Gemma 3 270M IT (chosen instead of LFM2.5, whose runs were paused after 1-2 passes; 54 % of its GSM8K answers finish without \boxed{}); its model card's sampling (temperature 1.0, top_k 64, top_p 0.95, max 8,192); prompt v5 with "submit the best edit"; astra effort max; evaluated on MATH-500 only (4 samples; the user: the smaller of MATH-500 and GSM8K); unedited GSM8K: 4.0

Main score for Gemma (the user, 2026-10-25): lenient grading -- the last \boxed{} answer, otherwise the answer after
'Final Answer' / 'the answer is', otherwise the last number. The runs marked "lenient grading stated in the prompt" were
told this rule (and graded their practice answers that way); the earlier runs were told the strict \boxed{} rule.

| Model | MATH-500 (lenient) | Edit |
|---|---|---|
| Unedited Gemma 3 270M IT | 6.8 |  |
| Edited, 5 passes (prompt v5, strict grading in the prompt) (used 5) | 6.7 |  |
| Edited, 10 passes (prompt v5, strict grading in the prompt) (used 10) | 6.7 |  |
| Edited, 25 passes (prompt v5, strict grading in the prompt) (used 25) | 6.1 |  |
| Edited, 50 passes (prompt v5, strict grading in the prompt) (used 50) | 6.2 |  |
| Edited, 5 passes (prompt v5, the lenient grading stated in the prompt) (used 5) | 7.0 |  |
| Edited, 10 passes (prompt v5, the lenient grading stated in the prompt) (used 10) | 7.6 |  |
| Edited, 25 passes (prompt v5, the lenient grading stated in the prompt) (used 25) | 7.7 |  |
| Edited, 50 passes (prompt v5, the lenient grading stated in the prompt) (used 50) | 7.0 |  |

Strict \boxed{} grading (the earlier main score), for reference:

| Model | MATH-500 (strict) | Edit |
|---|---|---|
| Unedited Gemma 3 270M IT | 2.8 |  |
| Edited, 5 passes (prompt v5, strict grading in the prompt) (used 5) | 2.6 |  |
| Edited, 10 passes (prompt v5, strict grading in the prompt) (used 10) | 2.9 |  |
| Edited, 25 passes (prompt v5, strict grading in the prompt) (used 25) | 3.5 |  |
| Edited, 50 passes (prompt v5, strict grading in the prompt) (used 50) | 3.2 |  |
| Edited, 5 passes (prompt v5, the lenient grading stated in the prompt) (used 5) | 3.5 |  |
| Edited, 10 passes (prompt v5, the lenient grading stated in the prompt) (used 10) | 3.1 |  |
| Edited, 25 passes (prompt v5, the lenient grading stated in the prompt) (used 25) | 3.8 |  |
| Edited, 50 passes (prompt v5, the lenient grading stated in the prompt) (used 50) | 2.6 |  |
