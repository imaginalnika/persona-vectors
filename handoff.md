# Persona vectors: handoff for mecaraman2 (and your Grok Bot)

From Josh Cho, 2026-10-13. Written so an agent can read it and pick up the work.
Status: extraction and steering are done on 3 instruct models × 7 traits. All outputs are on the shared RunPod network volume. The GPU pod used for this run was terminated afterwards (Runpod confirmed); the network volume was kept.

---

## 1. What persona vectors are (short version)
A **persona vector** is one direction in a model's residual stream (per layer) that tracks a character trait, for example "evil", "sycophantic" or "hallucinating".
- **Extraction:** answer the same questions twice, once with system prompts that push the trait and once with prompts that suppress it. Keep only pairs where an LLM judge confirms the trait showed up (or didn't). Then take the **mean difference of hidden states averaged over response tokens**, layer by layer.
- **Uses:** (a) **steering**: add `coef × vector` at one layer while generating, to turn the trait up (+) or down (−). (b) **monitoring**: project activations onto the vector to predict the trait before or during generation. (c) **preventative steering / data screening** during finetuning (from the paper; we haven't tried it yet).
- **Related:** repeng / "control vectors" (Theia Vogel) does essentially the same thing. It builds contrastive prompt pairs and takes PCA over the hidden-state differences.

## 2. Key sources
- Chen, Arditi, Sleight, Evans, Lindsey (Anthropic Fellows), "Persona Vectors: Monitoring and Controlling Character Traits in Language Models", 2025. arXiv: https://arxiv.org/abs/2507.21509
- Anthropic blog: https://www.anthropic.com/research/persona-vectors
- Official code and trait data: https://github.com/safety-research/persona_vectors (the trait JSONs live in `data_generation/trait_data_{extract,eval}/`)
- Theia Vogel (@voooooogel on X), the control-vector posts: https://x.com/voooooogel
- repeng library: https://github.com/vgel/repeng
- Blog post "Representation Engineering Mistral-7B an Acid Trip": https://vgel.me/posts/representation-engineering/
- **Our explainer site** (EN plus five KO variants): https://imaginalnika.github.io/persona-vectors/ (repo: https://github.com/imaginalnika/persona-vectors)

## 3. Models and why we chose them
| Model | Why | Layers | Seed layers we steered |
|---|---|---|---|
| Qwen2.5-7B-Instruct | The paper's main model, so we can compare directly against its results | 28 | 16, 20 |
| Hermes-3-Llama-3.1-8B | The paper's second family (Llama 3.1); Hermes is a lightly aligned roleplay tune, good for character traits | 32 | 16, 20, 24 |
| Qwen3.8-27B | Larger, recent hybrid-attention model (checkpoint is VLM `Qwen3_5ForConditionalGeneration`). We load it **text-only** via `AutoModelForCausalLM` (weights load correctly, verified) with **thinking disabled** (`enable_thinking=False`) | 64 | 32, 48 |
Also shortlisted but not run: gemma-3-12b-it (Gemma Scope SAEs exist for Gemma 2, useful for interpreting vectors).

## 4. Method (our "lean pipeline")
The official repo pins vLLM and unsloth. To fit a ~2h GPU budget we wrote a **faithful lean reimplementation** in plain transformers: `/workspace/persona-vectors-code/lean_pipeline/run_extract_steer.py`.
- **Data:** the paper's format. Each trait has 5 positive/negative instruction pairs, 20 extraction questions, a separate eval question set, and the paper's judge rubric (`eval_prompt`). Generation: 256 max tokens, batch 8.
- **Judge:** OpenRouter `openai/gpt-4.1-mini` (the paper used gpt-4.1-mini). It scores 0–100 per answer. The key is read from env `OPENROUTER_API_KEY`; it is **not** stored on the volume.
- **Filtering:** keep pairs where the trait-on answer scores high and the trait-off answer scores low (paper thresholds). The script logs "effective pairs".
- **Vectors saved for all layers:** `response_avg_diff` (main one, as in the paper), plus `prompt_avg_diff` and `prompt_last_diff`. Shape is `[n_layers+1, hidden]`; row L = output of block L (row 0 = embeddings). Steering at "layer L" adds the vector to `model.layers[L-1]` output on every generated token (the paper's ActivationSteerer convention).
- **Steering check:** 5 eval questions × coefficients {−2, −1, +1, +2} × seed layers, against two baselines: **unsteered** and a **norm-matched random vector** at +2.
- **Resumable:** the script skips any model/trait whose CSVs, vectors or `steer_summary.json` already exist. `--deadline_epoch` stops cleanly between traits.

### Traits
- Paper's 3 (their JSON data, unchanged): **evil, sycophantic, hallucinating**.
- 4 character traits Josh wrote and approved (same JSON format): **warmth, playfulness, menace, formality**. Canonical copy of the approved character-trait pack, also published in the repo: https://imaginalnika.github.io/persona-vectors/character-traits.json. The per-trait files used by the pipeline should also be at `/workspace/persona-vectors-code/lean_pipeline/trait_data/{extract,eval}/*.json` on the volume (not re-checked after the pod was terminated).
- If you want new or changed trait prompts, run them past Josh first; our rule is no new prompt text without his approval.

## 5. Results (judge score 0–100, mean over 5 questions; +2 at best layer)
| Model | Trait | Best L | Unsteered | Random +2 | **Steered +2** | Lowest −2 score |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | evil | 20 | 0 | 0 | **56** | 0 |
| Qwen2.5-7B | sycophantic | 20 | 10 | 4 | **83** | 0 |
| Qwen2.5-7B | hallucinating | 20 | 32 | 38 | **91** | 0 |
| Qwen2.5-7B | playfulness | 20 | 24 | 26 | **85** | 0 |
| Qwen2.5-7B | warmth | 16 | 87 | 84 | 85 | 34 (L20) |
| Qwen2.5-7B | menace | 20 | 82 | 81 | 88 | 6 |
| Qwen2.5-7B | formality | 16 | 90 | 89 | 89 | 0 (L20) |
| Hermes-3-8B | evil | 16 | 0 | 0 | **71** | 0 |
| Hermes-3-8B | sycophantic | 16 | 6 | 8 | **69** | 2 |
| Hermes-3-8B | hallucinating | 16 | 18 | 22 | **98** | 0 |
| Hermes-3-8B | playfulness | 24 | 12 | 32 | **76** | 2 |
| Hermes-3-8B | warmth | 16 | 84 | 80 | 87 | 32 (L20) |
| Hermes-3-8B | menace | 16 | 62 | 58 | **90** | 23 |
| Hermes-3-8B | formality | 16 | 88 | 87 | 91 | 0 (L20) |
| Qwen3.8-27B | evil | 48 | 0 | 0 | **32** | 0 |
| Qwen3.8-27B | sycophantic | 48 | 6 | 6 | **87** | 0 |
| Qwen3.8-27B | hallucinating | 48 | 0 | 18 | **98** | 16 |
| Qwen3.8-27B | playfulness | 48 | 45 | 46 | **85** | 4 |
| Qwen3.8-27B | warmth | 48 | 87 | 84 | 92 | 12 |
| Qwen3.8-27B | menace | 48 | 74 | 74 | 89 | 24 |
| Qwen3.8-27B | formality | 32 | 80 | 82 | 91 | 13 |

**Best layers:** Qwen2.5-7B → L20 (about 70% depth), Hermes-3-8B → L16 (50%), Qwen3.8-27B → L48 (75%). +1 barely moves anything and +2 does, so the useful coefficient is somewhere between them.

### Caveats (read these before trusting the numbers)
1. **n = 5 questions per condition**, so differences under ~15 points are noise. Steering was only checked on 2–3 seed layers, not a full layer sweep, even though vectors exist for all layers.
2. **Character-trait ceiling:** the approved eval questions for warmth, formality and menace already draw out the trait (e.g. villain letters for menace), so unsteered scores sit at 75–90 and +2 has no room to rise. **For those traits the evidence is the −2 direction** (e.g. warmth 87→12 on Qwen3.8, formality 90→0 on Qwen2.5).
3. **Qwen3.8 evil:** the model mostly refused the trait-pushing instructions, so only about 12 pairs survived filtering (vs ~80–99 elsewhere). The vector is weaker (+32).
4. No coherence or fluency judge was run. At +2, some outputs drift; read the examples in `steer_summary.json`.
5. The random baseline is a single random direction at +2, not a distribution.
6. Qwen3.8 ran without the fast `causal_conv1d` kernel (flash-linear-attention is installed), so it is slow: about 15 min per trait vs about 4 for the 8B models.

## 6. Where everything is: shared RunPod network volume
**Volume "persona-vector", id `25rj6x8kfl`, datacenter US-NE-1**, in the shared RunPod account. Attach it to any pod in US-NE-1 and it mounts at `/workspace`.
```
/workspace/models/                      Qwen2.5-7B-Instruct, Hermes-3-Llama-3.1-8B, Qwen3.8-27B (HF snapshots)
/workspace/persona-vectors-code/
  lean_pipeline/run_extract_steer.py    the pipeline
  lean_pipeline/trait_data/{extract,eval}/<trait>.json
  resume*.sh                            the chained-run wrappers we used
/workspace/persona-vectors-runs/
  SUMMARY.md / SUMMARY.json             results table + example outputs
  STATUS_run{2,3,4}.json, logs/run*.log
  <model>/<trait>/
    pos_instruct.csv, neg_instruct.csv  generations + judge scores
    effective_counts.json
    vectors/<trait>_response_avg_diff.pt   (also prompt_avg_diff, prompt_last_diff, meta.json)
    steer_summary.json                  all steering conditions, scores, answers
```
Load a vector: `v = torch.load(".../evil_response_avg_diff.pt")[20]  # layer 20 row`.

## 7. Running it again
1. In the RunPod console (shared account): **Pods → Deploy**, choose **US-NE-1** (Secure Cloud), GPU 1× H100 80GB (≈$4/hr; an A100 80GB is fine for the 7–8B models), a PyTorch template, and **attach network volume `25rj6x8kfl`** (mount `/workspace`). Add your own SSH public key in RunPod settings. Set a spend limit or auto-stop.
2. SSH in using the command RunPod shows for the pod, then:
```bash
pip install -U "transformers>=5.19" accelerate openai pandas tqdm flash-linear-attention   # check: model loads, versions in logs
export OPENROUTER_API_KEY=...        # your own key; don't write it to the volume
cd /root
nohup python3 -u /workspace/persona-vectors-code/lean_pipeline/run_extract_steer.py \
  --models_root /workspace/models --out_root /workspace/persona-vectors-runs \
  --models Qwen2.5-7B-Instruct Hermes-3-Llama-3.1-8B Qwen3.8-27B \
  --traits evil sycophantic hallucinating warmth playfulness menace formality \
  --deadline_epoch $(( $(date +%s) + 3*3600 )) \
  --batch_size 8 --max_tokens 256 --steer_questions 5 \
  > /workspace/persona-vectors-runs/logs/runN.log 2>&1 &
cat /workspace/persona-vectors-runs/STATUS.json   # progress
```
Finished traits are skipped. **To redo a trait, or to write to a fresh results folder, pass a new `--out_root`** (e.g. `/workspace/persona-vectors-runs-v2`) so you don't overwrite these results. STATUS.json is rewritten on each launch.
3. Stop the pod when it finishes. The volume persists and bills separately.

### Suggested next steps
- More eval questions (n≥20) and a coherence judge; a full layer sweep using the saved all-layer vectors.
- Character traits: add neutral eval questions so +steering isn't capped (needs Josh's approval for new prompt text).
- Projection monitoring (paper §4) and comparing against repeng PCA vectors on the same models.
- Gemma-3-12B-it (+ Gemma Scope) as a 4th model.

## 8. Persona mode for chat.imaginalnika.com (in progress)
chat.imaginalnika.com is our chat playground (repo `imaginalnika/basechat`). A **persona mode** that uses these vectors/traits is being built with Claude Code on branch **`persona-mode`**. That branch is **local-only on Josh's machine and not pushed yet** (so it isn't on GitHub), and it is not deployed. Ask Josh to push it before you build on it.

## 9. Secrets
This doc has no keys or private SSH material. Use your own OpenRouter key and your own SSH key on the RunPod account.
