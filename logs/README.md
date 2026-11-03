# Logs: LLMs can understand and write neural network weights

The complete logs of every experiment in the blog post *Sparks of Weight Space Programming: LLMs can understand and
write neural network weights*. For each experiment you get:

- everything we sent the model;
- everything it sent back, including its reasoning summaries;
- every piece of code it ran, with the output it saw;
- the evaluation behind each number in the post;
- the networks it read or wrote.

Each experiment folder has its own README. It says which run is which tab or row in the post, and which file each
number comes from.

| Folder | In the post | Runs | Tools |
|---|---|---|---|
| [`exp1_which_game/`](exp1_which_game) | Experiment 1: which game was this network trained on? | Breakout, Pong, Pac-Man, Freeway (open answer) | none |
| [`exp2_diffusion/`](exp2_diffusion) | Experiment 2: what does this diffusion model generate? | circle, square, triangle | none, no forward pass |
| [`exp3_language/`](exp3_language) | Experiment 3: which language a language model learned, from its middle blocks | Python, romaji, English, two conversations each | Python sandbox, no forward pass |
| [`exp4_fix_car/`](exp4_fix_car) | Experiment 4: fix a failing network by editing its weights | one one-shot edit | none |
| [`exp5_write_from_scratch/`](exp5_write_from_scratch) | Experiment 5: write networks from scratch | Breakout, Seaquest, Asterix (MinAtar) | none |
| [`exp6_go_from_scratch/`](exp6_go_from_scratch) | Experiment 6: write a 1.5M-weight Go network | 0, 10, 25 and 50 passes | Python sandbox |
| [`exp7_sycophancy/`](exp7_sycophancy) | Experiment 7: which language model is sycophantic? | four weights-only attempts, then the attention tool (3 pairs each) | Python sandbox |
| [`exp8_math/`](exp8_math) | Experiment 8: make a language model better at math | Qwen3.5-0.8B with 10, 25 and 50 passes, and the Gemma 3 270M follow-up | Python sandbox |

Only the runs the post reports are here. Trial runs of earlier prompt versions are left out.

## How the model was run

"Astra" in the post is OpenAI's `gpt-6-astra`.

**One turn.** Every turn is one call of the Codex CLI: `codex exec`, or `codex exec resume <thread>` to continue a
conversation. The prompt goes in on stdin. The reasoning effort is set per call and recorded in each turn's `run.json`:

| Effort | Experiments |
|---|---|
| `high` | 1, 2, 4 |
| `xhigh` | 3, 5, 6, 7 |
| `max` | 8 |

**No tools of its own.** Every built-in Codex capability that gives the model a tool is switched off: shell, file
editing, web search, image viewing, sub-agents, plugins and the rest. The configuration is in
[`harness/codex_config.toml`](harness/codex_config.toml), and the system instructions are one neutral line,
[`harness/instructions.md`](harness/instructions.md). Each call names the model and the effort itself, overriding
the defaults in that file.

**Every turn was audited.** We checked each turn's event stream. A turn counts as tool-free when the stream holds only
the model's messages, its reasoning summaries, and Codex's notice that code mode is disabled. The result is the
`audit.tool_free` field of the turn's `run.json`. All 1,125 turns in this release are tool-free.

**Where code was allowed** (Experiments 3, 6, 7, 8):
- The model wrote one Python block per turn.
- We ran it in our own sandbox ([`harness/sandbox.py`](harness/sandbox.py)): bubblewrap, no network, a working folder
  kept between runs, and a time limit per run.
- We sent the printed output back as the next message.
- Each run's code and output are in `tool_calls.jsonl`.
- What the code was allowed to do is stated in the experiment's prompt and in its README.

## Files

Not every file appears in every run.

| File | What it is |
|---|---|
| `prompt.txt`, `message.txt` | What we sent: the first message, and later ones per turn. |
| `events.jsonl` | The raw Codex event stream of one turn: the model's messages and reasoning summaries, plus token usage. |
| `answer.md`, `turn_text.md`, `final_answer.md` | The model's reply in one turn, and its final reply. |
| `explanations.md` | The model's explanation of each step, collected over the run, where the prompt asked for one. |
| `tool_calls.jsonl` | One line per sandbox run: the code, its printed output, and the time it took. |
| `run.json` | One turn's record: model, effort, tokens, wall time, and the audit. |
| `state.json` | One conversation's record: turns, runs used, and whether it finished. |
| `loop.log`, `driver.log`, `run.log` | The harness's log of the run. |
| `eval/` | The scoring, with the answer key or ground truth. |
| `networks/` | The networks the model read or wrote. |

**Placeholders.** Internal paths are replaced by placeholders:

| Placeholder | Stands for |
|---|---|
| `<data>` | our data disk |
| `<repo>` | our experiment repository |
| `<codex>` | the Codex binary |
| `<codex_home>` | Codex's state folder |
| `<venvs>` | Python environments |
| `<home>` | home directory |
| `<remote>` | a rented GPU node |

Codex thread ids are replaced by `id-` and 8 hex digits. The same thread keeps the same id in every file, so turns
of one conversation still match up.

## Not included

- **The Gemma 3 270M fine-tunes of Experiment 7** (17 GB). The data, the training script and the hashes are there, so
  they can be rebuilt.
- **The intermediate files the model wrote while working:** activations, generated-text dumps and its draft networks.
  The final networks and weight edits are included, Experiment 8's as exact diffs against the public base weights,
  with a script that applies them.
- **The text of Experiment 3's Python corpus.** It is other projects' code, so we give a manifest and a rebuild script
  instead.
- **The GPQA question texts in Experiment 8's generations.** GPQA's scores and per-question results are kept.

## Licenses

See [`LICENSE`](LICENSE).

| Content | License |
|---|---|
| Logs, data and networks | CC BY 4.0 |
| Our scripts | MIT |
| MinAtar source in Experiment 5's prompts | GPL-3.0, see `exp5_write_from_scratch/LICENSE-MinAtar` |

Third-party benchmark data keep their own licenses: MATH, GSM8K, IFEval, MMLU-Pro and the others in Experiment 8.
