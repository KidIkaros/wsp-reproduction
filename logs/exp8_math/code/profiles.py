"""Model profiles of the math weight-editing experiment; chosen by the environment variable MATHEDIT_PROFILE
(default qwen3: everything before 2026-10-23). qwen35 (2026-10-23): the text-only export of Qwen3.5-0.8B (Qwen3_5ForCausalLM, single
model.safetensors), thinking-mode benchmark sampling from its model card (temperature 0.6, top_p 0.95, top_k 20,
presence_penalty 1.5)."""
import os

ROOT = "<data>/mathedit"
PROFILES = {
    "qwen3": {"name": "Qwen3-1.7B", "model": f"{ROOT}/models/Qwen3-1.7B", "view": f"{ROOT}/model_view",
              "view_files": ("config.json", "generation_config.json", "model.safetensors.index.json",
                             "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "tokenizer.json",
                             "tokenizer_config.json", "vocab.json", "merges.txt"),
              "runs": f"{ROOT}/runs2", "practice": f"{ROOT}/practice",
              "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 32768, "seed": 0}},
    "qwen35": {"name": "Qwen3.5-0.8B", "model": f"{ROOT}/models/Qwen3.5-0.8B-text", "view": f"{ROOT}/model_view35",
               "view_files": ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                              "tokenizer_config.json", "chat_template.jinja"),
               "runs": f"{ROOT}/runs3", "practice": f"{ROOT}/practice35",
               # out of scope for the editor (the final norm, embeddings and output layer):
               # refused by save_edit, stated in the prompt
               "frozen": ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight"),
               "refuse_rescale": True, "tf_max_texts": 32,   # prompt v2 (2026-10-23): 32 texts per teacher-forced pass
               # thinking OFF (2026-10-23): with thinking on, the 0.8B looped ("Wait, let's check ...") and 92 % of
               # MATH-500 answers hit 32,768 tokens (MATH-500 9.3 %); the model card: "Qwen3.5-0.8B operates in
               # non-thinking mode by default"; its non-thinking benchmark setting
               "thinking": False,
               "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
                            "max_tokens": 8192, "seed": 0}},
}

# thinking ON for the 0.8B (2026-10-23; budgets 0, 5, 10, 25, 50): same model, rules and
# practice data; the model card's thinking benchmark setting; its own runs folder and servers
PROFILES["qwen35think"] = dict(PROFILES["qwen35"], runs=f"{ROOT}/runs4", thinking=True,
                               sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                                         "max_tokens": 32768, "seed": 0})

# prompt v3 (2026-10-23, after the v2 runs measured noise: the unedited model read 55.5 % and 50.8 % in two 128-answer
# passes): generation passes of up to 512 answers , seeded by problem and
# optionally paired ("both": 256 per model); teacher-forced passes of up to 1,024 texts (dense measurements
# on many reference solutions; cheap) and long texts (whole failing answers), at most
# tf_total_tokens per pass. The limits follow the
# run's state.json (prompt v3, mode = profile), so one server serves v2 and v3 runs of both profiles.
V3_LIMITS = {"qwen35": {"gen_max": 512, "tf_max_texts": 1024, "tf_max_tokens": 8192, "tf_total_tokens": 524288,
                        "act_max_tokens": 8192},
             "qwen35think": {"gen_max": 512, "tf_max_texts": 1024, "tf_max_tokens": 32768, "tf_total_tokens": 524288,
                             "act_max_tokens": 8192}}

# LFM2.5-350M (2026-10-24; chosen because its failures are not loops: it answers in ~124
# tokens without reasoning, GSM8K 28.0 / MATH-500 11.1 with its model card's sampling): a different architecture (gated
# short convolutions + 6 attention layers), runs in runs5, the same practice problems
PROFILES["lfm25"] = {
    "name": "LFM2.5-350M", "model": f"{ROOT}/models/LFM2.5-350M", "view": f"{ROOT}/model_view_lfm",
    "view_files": ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                   "chat_template.jinja"),
    "runs": f"{ROOT}/runs5", "practice": f"{ROOT}/practice35",
    "frozen": ("model.embedding_norm.weight", "model.embed_tokens.weight", "lm_head.weight"),
    "refuse_rescale": True, "tf_max_texts": 32, "thinking": False, "n_layers": 16,
    "arch_text": "LFM2.5-350M (16 layers of width 1,024: layers 2, 5, 8, 10, 12 and 14 are full attention (self_attn) with 16 "
                 "query and 8 key/value heads of size 64 and q/k layer norms, the other 10 are gated short-convolution layers "
                 "(conv: in_proj to 3 x 1,024 gates and values, a depthwise convolution of length 3, out_proj); SwiGLU MLPs of "
                 "width 4,608 (feed_forward.w1, w3, w2); RMSNorm before each block (operator_norm, ffn_norm) and at the end "
                 "(embedding_norm); a tied 65,536-token embedding)",
    "measure_text": "the model's chat template; the user message is the problem, a newline and \"Please reason step by step, and "
                    "put your final answer within \\boxed{{}}.\"; sampling with temperature 0.1, top_k 50, repetition penalty "
                    "1.05 and at most 8,192 new tokens;",
    "sampling": {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05, "max_tokens": 8192, "seed": 0}}
V3_LIMITS["lfm25"] = {"gen_max": 2048, "tf_max_texts": 1024, "tf_max_tokens": 8192, "tf_total_tokens": 524288,
                      "act_max_tokens": 8192}

# Gemma 3 270M IT (2026-10-24, instead of LFM2.5): 54 % of its GSM8K answers finish
# without \\boxed{} (GSM8K 4.1 / MATH-500 2.8 with its model card's sampling); runs in runs6, the same practice problems
PROFILES["gemma270"] = {
    "name": "Gemma 3 270M IT", "model": f"{ROOT}/models/gemma-3-270m-it", "view": f"{ROOT}/model_view_gemma",
    "view_files": ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                   "chat_template.jinja", "special_tokens_map.json", "added_tokens.json"),
    "runs": f"{ROOT}/runs6", "practice": f"{ROOT}/practice35",
    "frozen": ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight"),
    "refuse_rescale": True, "tf_max_texts": 32, "thinking": False, "n_layers": 18,
    "arch_text": "Gemma 3 270M IT (18 layers of width 640; every layer has attention with 4 query heads and 1 key/value head "
                 "of size 256 and q/k norms: layers 5, 11 and 17 attend to the whole context, the other 15 to a sliding window "
                 "of 512 tokens; GeGLU MLPs of width 2,048 (mlp.gate_proj, up_proj, down_proj); RMSNorm before and after "
                 "attention and MLP (input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, "
                 "post_feedforward_layernorm) and at the end (model.norm); a tied 262,144-token embedding, scaled by sqrt(640) "
                 "at the input)",
    "measure_text": "the model's chat template; the user message is the problem, a newline and \"Please reason step by step, and "
                    "put your final answer within \\boxed{{}}.\"; sampling with temperature 1.0, top_k 64, top_p 0.95 and at "
                    "most 8,192 new tokens;",
    "sampling": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 8192, "seed": 0}}
# 2026-10-25: for Gemma the format-agnostic grading is the default and the prompt says so (without telling
# the agent not to repair \\boxed{}): a \\boxed{} answer, else the stated final answer, else the last number
# (evaluate.lenient_pred); practice answers in generation passes are graded the same way; evaluated on MATH-500 only
PROFILES["gemma270"].update({
    "lenient": True, "bench_text": "MATH-500",
    "grading_text": "the final answer is graded against the reference answer: the last \\boxed{} answer if there is one, "
                    "otherwise the answer stated after 'Final Answer' or 'the answer is', otherwise the last number in the "
                    "response"})
V3_LIMITS["gemma270"] = {"gen_max": 2048, "tf_max_texts": 1024, "tf_max_tokens": 8192, "tf_total_tokens": 524288,
                         "act_max_tokens": 8192}


def get():
    return PROFILES[name()]


def name():
    return os.environ.get("MATHEDIT_PROFILE", "qwen3")


def limits(run):
    """the v3 limits of a run (None for v1 / v2 runs; every later prompt version keeps them -- 2026-10-24: v4 runs were
    served with v2 limits, no "both" and 128 answers, because this checked for "v3" only)."""
    import json
    try:
        st = json.load(open(os.path.join(run, "state.json")))
    except (OSError, ValueError):
        return None
    if st.get("prompt") in (None, "v1", "v2"):
        return None
    lim = dict(V3_LIMITS.get(st.get("mode")) or {})
    if lim and st.get("prompt") not in ("v3", "v4") and st.get("mode") == "qwen35":
        lim["gen_max"] = 2048        # prompt v5, thinking off: answers are short and v3/v4 gains sat below the noise
    return lim or None
