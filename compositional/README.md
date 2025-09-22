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

- **TokMem (ours) - sequential training with memory tokens and optional LoRA**:
  ```bash
  bash run_n_rounds_main.sh
  ```
  - Supports both sequential training (multiple rounds) and single-round training.
  - Edit variables at the top of the script to customize rounds, tools, data sizes, and training options.
  - Features: memory token embeddings, optional LoRA adaptation, curriculum learning, and active tool renormalization.

- **LoRA baseline - standard LoRA fine-tuning**:
  ```bash
  bash run_n_rounds_lora.sh
  ```
  - Standard LoRA fine-tuning baseline with optional replay buffer and LoRA reinitialization.
  - Supports both sequential and single-round training.

- **Sequential vs Single-Round Training**:
  - Set `NUM_ROUNDS=1` for single-round training (non-sequential).
  - Set `NUM_ROUNDS>1` for sequential training across multiple tool ranges.
  - In sequential mode, each round trains on a disjoint set of tools, enabling continual learning evaluation.

- **TokMem Adaptation Phase**:
  - Set `FREEZE_LORA_AFTER_FIRST=true` to freeze LoRA adapters after round 1.
  - This enables an adaptation phase where only memory tokens are trainable in subsequent rounds.
  - Useful for adapting to auxiliary tools while preserving learned tool representations.

- What the scripts do:
  - Step 1: Generate per-round train/test JSON files under `data/training` and `data/test` using `xlam_datasets.py`.
  - Step 2: Run multi-round training and evaluation, saving results under a `checkpoints*` directory.

- Key configuration variables (edit at top of the scripts):

  **Common to both scripts:**
  - `NUM_ROUNDS`: number of sequential rounds (set to 1 for single-round training).
  - `TOOLS_PER_ROUND`: tools included per round.
  - `START_TOOL`: first tool id; rounds cover ranges like `start-end` per round.
  - `EPOCHS_PER_ROUND`: epochs per round (comma-separated list supported in main script).
  - `SAMPLES_PER_TOOL`: max samples per tool when creating synthetic datasets.
  - `TRAIN_SIZE`, `TEST_SIZE`: total train/test samples to generate per round.
  - `TRAIN_MAX_CALLS`, `TEST_MAX_CALLS`: max function calls per example.
  - `TRAIN_MAX_CALLS_PER_ROUND`, `TEST_MAX_CALLS_PER_ROUND`: per-round function call limits.
  - `MODEL`: Hugging Face model id.
  - `BATCH_SIZE`: training micro-batch size.

  **TokMem script (`run_n_rounds_main.sh`) specific:**
  - `LR`: base learning rate for memory token embeddings.
  - `LORA_LR`: LoRA adapter learning rate (when `USE_LORA=true`).
  - `USE_LORA`: enable LoRA adaptation alongside memory tokens.
  - `FREEZE_LORA_AFTER_FIRST`: freeze LoRA after round 1 for adaptation phase.
  - `CURRICULUM_LEARNING`: enable curriculum ordering of tools.
  - `RENORM_ACTIVE_TOOLS`: renormalize active tool embeddings each round.

  **LoRA baseline script (`run_n_rounds_lora.sh`) specific:**
  - `LR`: LoRA adapter learning rate.
  - `REINIT_LORA`: reinitialize LoRA adapters after each round.
  - `USE_REPLAY_BUFFER`: enable mixing prior-round samples into current training.
  - `REPLAY_BUFFER_SIZE`: total samples retained across rounds.
  - `REPLAY_RATIO`: fraction of replayed samples per batch.

  **Both scripts support:**
  - `SAVE_CHECKPOINTS`: save model states and summaries.
  - `EVAL_ALL_PREVIOUS`: evaluate on all previous rounds' test sets.

- Training rounds format:
  - Scripts build `--training_rounds` internally as a comma-separated list of `start-end:epochs` (e.g., `1-50:1,51-100:3`). You generally do not need to set this manually.

- Typical modifications:
  - **For sequential training**: Increase `NUM_ROUNDS` / `TOOLS_PER_ROUND` to cover more tools.
  - **For complexity variation**: Use `TRAIN_MAX_CALLS_PER_ROUND` / `TEST_MAX_CALLS_PER_ROUND` to vary function call complexity by round.
  - **For TokMem adaptation**: Set `FREEZE_LORA_AFTER_FIRST=true` and `NUM_ROUNDS>1` to enable adaptation phase on auxiliary tools.
  - **For curriculum learning**: Enable `CURRICULUM_LEARNING` in the TokMem script for ordered tool introduction.
  - **For continual learning**: Use the LoRA script with `USE_REPLAY_BUFFER=true` to mitigate catastrophic forgetting.


## **In-Context Learning (ICL) Baseline** (`compositional/icl_baseline.sh`)
Zero-shot evaluation baseline using frozen models with in-context examples and optional RAG.
- **What it does**: Evaluate tool calling performance without any training using in-context learning
- **Use case**: Establish baseline performance for compositional tasks
- **Key features**:
  - Supports RAG (Retrieval-Augmented Generation) or full tool access
  - Configurable retrieval parameters and batch sizes
  - Evaluates on specific tool ranges (default: tools 51-100)
- **Quick start**: `cd compositional && bash icl_baseline.sh`