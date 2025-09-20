# Atomic Memory Recall

Atomic task calling using reserved special tokens as task identifiers. The model overrides embedding/lm_head behavior only for the reserved tokens, keeping the base LM otherwise frozen.

## Contents

- `task_model.py` — TaskCallingModel using reserved special tokens for tasks.
- `task_dataset.py` — dataset and collation utilities for task calling.
- `task_training.py` — training loop helpers and logging.
- `natural_instructions_eval.py` — evaluation on Natural Instructions‑style data.
- `analyze_task_similarity.py` — utilities to analyze learned task token geometry.
- `test_lora_baseline.sh` — example tests for finetuning and replay memory baselines.
- `test_tokmem.sh` — example tests for TokMem and its decoupled embeddings variant.
- `test_sbert_retriever.py` — retrieval helper comparison.

## Setup

```bash
pip install -r ../requirements.txt
# Optional extras used in some scripts
pip install peft
```

## Usage

- Replicate baselines quickly by running the provided scripts:
  - LoRA + replay memory baselines:
    ```bash
    bash test_lora_baseline.sh
    ```
  - TokMem and its decoupled embeddings variant:
    ```bash
    bash test_tokmem.sh
    ```
  - Tip: adjust `CUDA_VISIBLE_DEVICES` and any paths inside the scripts as needed for your setup.

- Key arguments used by these scripts:
  - `--num_tasks`: number of tasks to sample/train.
  - `--train_size`, `--val_size`, `--test_size`: dataset sizes per task.
  - `--model_name`: Hugging Face model identifier to load.
  - `--num_epochs`: number of training epochs.
  - `--batch_size`: per-step micro-batch size. Effective batch size = `batch_size × gradient_accumulation_steps`.
  - `--gradient_accumulation_steps`: steps to accumulate before an optimizer update.
  - `--max_length`: maximum sequence length fed to the model.
  - `--max_instruction_tokens`: maximum tokens for the instruction segment.
  - `--eval_batch_size`: batch size used during evaluation.
  - `--validate_every_n_steps`: frequency (in optimizer steps) to run validation.
  - `--lr`: learning rate (LoRA baseline script).
  - `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA hyperparameters.
  - `--target_modules`: which attention/projection modules to adapt with LoRA.
  - `--save_path`: path to save checkpoints/outputs.
  - `--continual_replay`: enable replay memory baseline (flag, no value).
  - `--block_size`: size of each replay block/chunk.
  - `--continual_replay_ratio`: fraction of replayed samples mixed with current task data.


