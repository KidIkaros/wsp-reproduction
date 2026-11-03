## Final choice

Retained the frozen edits in `/work/edited`, changing only **64 MLP output columns each in layers 15 and 19**—131,072 weight values total. The edits counteract measured repetition directions; embeddings, output head, and final norm remain untouched.

## Diagnosis and evidence

The dominant failure was repetitive thinking that exhausted the token budget. Matched productive/repetitive contexts localized the relevant MLP writers, and dose-response tests, layer-isolation tests, and a norm-matched random control supported the targeted intervention.

**Independent paired confirmation: 1,152 held-out practice problems**

| Source | Original | Edited |
|---|---:|---:|
| GSM8K practice | 11.6% | **70.8%** |
| MATH practice | 9.5% | **50.0%** |
| Combined | 10.6% | **60.4%** |

The combined gain was **49.8 percentage points, SE 1.5**, with **579 rescues and five regressions**. Of those rescues, 576 were originally truncated outputs, while aggregate accuracy on originally finished outputs was unchanged. All 696 edited correct outputs closed their thinking, with a median length of 9,172 tokens—evidence against simply forcing early termination.

These are practice results, not unseen benchmark scores; mathematical errors, missing boxes, and some degeneration remain.

**Evidence:** `/work/final_report.md`  
**Exact neuron indices and hashes:** `/work/frozen_weights_manifest.json`