# Compositional Memory Recall

Function/tool calling with reserved special tokens used as tool identifiers. Supports sequential training rounds across disjoint tool ranges and optional LoRA fine‑tuning of the base LM.

## Contents

- `model.py` — FunctionCallingModel with reserved‑token overrides and optional LoRA.
- `dataset.py` — NativeFunctionCallingDataset, collator, and dataloader builders.
- `training.py` — training loop and evaluation helpers.
- `eval.py` — parsing and exact/F1 style evaluation utilities.
- `main_sequential.py` — CLI for multi‑round training/eval over tool ranges.
- `lora_sequential.py` — variants for LoRA workflows.
- `icl_baseline.py` / `icl_baseline.sh` — in‑context learning baseline.
- `tool_retrieval.py`, `replay_buffer.py` — helpers/utilities.
- `xlam_datasets.py` — dataset generation helpers.
- `run_n_rounds_main.sh`, `run_n_rounds_lora.sh` — shell wrappers for sequential runs.

## Setup

```bash
pip install -r ../requirements.txt
pip install peft  # required for LoRA paths
```

## Data Layout

Expected files per round (examples):

- Train: `data/training/function_calling_train_tools{range}_{k}calls.json`
- Test:  `data/test/function_calling_test_tools{range}_{k}calls.json`

Some utilities also accept `function_calling_train.json` / `function_calling_test.json` in the CWD for single‑split workflows.

## Usage

- Quick start (sequential training with task embeddings, optional LoRA inside):
  ```bash
  bash run_n_rounds_main.sh
  ```
  - Edit variables at the top of the script to customize rounds, tools, data sizes, and training options.

- LoRA baseline (standard LoRA fine-tuning across rounds):
  ```bash
  bash run_n_rounds_lora.sh
  ```
  - This path fine-tunes with LoRA instead of training tool-specific embeddings.

- GPU selection:
  - Both scripts set `CUDA_VISIBLE_DEVICES` at the top. Change that to select your GPU.

- What the scripts do:
  - Step 1: Generate per-round train/test JSON files under `data/training` and `data/test` using `xlam_datasets.py`.
  - Step 2: Run multi-round training and evaluation, saving results under a `checkpoints*` directory.

- Key configuration variables (edit at top of the scripts):
  - Rounds and tool coverage:
    - `NUM_ROUNDS`: number of sequential rounds to train/evaluate.
    - `TOOLS_PER_ROUND`: tools included per round.
    - `START_TOOL`: first tool id; rounds cover ranges like `start-end` per round.
    - `EPOCHS_PER_ROUND`: epochs per round. In `run_n_rounds_main.sh`, can be a single value (e.g., `3`) or comma-separated per round (e.g., `1,3`). In `run_n_rounds_lora.sh`, use a single value.
  - Data generation:
    - `SAMPLES_PER_TOOL`: max samples per tool when creating synthetic datasets.
    - `TRAIN_SIZE`, `TEST_SIZE`: total train/test samples to generate per round.
    - `TRAIN_MAX_CALLS`, `TEST_MAX_CALLS`: max function calls per example; use `TRAIN_MAX_CALLS_PER_ROUND` / `TEST_MAX_CALLS_PER_ROUND` (comma-separated) to vary by round.
  - Model and optimization:
    - `MODEL`: Hugging Face model id (e.g., `meta-llama/Llama-3.2-3B-Instruct`).
    - `BATCH_SIZE`: training micro-batch size.
    - `LR`: base learning rate for embedding/adapter updates (main path).
    - `LORA_LR`: LoRA adapter learning rate (main path when `USE_LORA=true`).
  - LoRA and adapter options:
    - `USE_LORA` (main script): add `--use_lora` to train LoRA adapters alongside embeddings.
    - `FREEZE_LORA_AFTER_FIRST` (main script): freeze LoRA after round 1.
    - `REINIT_LORA` (LoRA script): when `true`, reinitializes LoRA after each round (adds `--reinit_lora_after_each_round`).
    - `REINIT_LORA`, `SAVE_CHECKPOINTS`, `EVAL_ALL_PREVIOUS` (LoRA script): control adapter resets, checkpoint saving, and evaluation on all previous test sets.
  - Curriculum and normalization (main script):
    - `CURRICULUM_LEARNING`: enable curriculum ordering.
    - `RENORM_ACTIVE_TOOLS`: renormalize active tool embeddings each round.
  - Replay buffer (LoRA script only):
    - `USE_REPLAY_BUFFER`: enable mixing prior-round samples into current training.
    - `REPLAY_BUFFER_SIZE`: total samples retained across rounds.
    - `REPLAY_RATIO`: fraction of replayed samples per batch (e.g., `0.2`).
  - Checkpoints and eval:
    - `SAVE_CHECKPOINTS`: save model states and summaries to the `CHECKPOINT_DIR` constructed in the script.
    - `EVAL_ALL_PREVIOUS`: evaluate on all earlier rounds’ test sets after each round (where supported).

- Training rounds format:
  - Scripts build `--training_rounds` internally as a comma-separated list of `start-end:epochs` (e.g., `1-50:1,51-100:3`). You generally do not need to set this manually.

- Typical modifications:
  - Increase `NUM_ROUNDS` / `TOOLS_PER_ROUND` to cover more tools.
  - Use `TRAIN_MAX_CALLS_PER_ROUND` / `TEST_MAX_CALLS_PER_ROUND` to vary complexity by round.
  - Toggle `USE_LORA` (main script) or use the pure LoRA script depending on the baseline you want to replicate.

## Notes

- Set `--dtype` to `bfloat16` or `float16` based on your GPU.
- Ensure `tokenizer.pad_token` is set (code defaults to BOS if missing).
- For deterministic curriculum or validation splits, see `dataset.py` options.

