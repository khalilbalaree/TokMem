# Embedding Capacity Ablation: TokMem vs Prefix Tuning

Embedding capacity ablation experiments comparing TokMem (trainable token embeddings) and prefix tuning approaches using frozen base models. These experiments systematically vary embedding dimensions, positions, and training sample sizes to analyze memorization capacity and generalization performance.

## Contents

- `main_memorization.py` — `SingleModelPromptTuning` module and batch generation helpers.
- `training.py`, `model.py`, `main.py` — training/eval utilities for memorization baselines.
- `run_memorization_comparison.sh` — comprehensive memorization task experiments.
- `run_training_samples_test.sh` — GSM8K task ablation with varying training sample sizes.
- `dataset.py` — small helpers for memorization datasets.

## Setup

```bash
pip install -r ../requirements.txt
pip install pandas
```

## Data

- **Memorization experiments**: Expects CSV under `memorization_data/`, e.g. `memorization_data/fanfics_1k_chunks.csv`.
- **GSM8K experiments**: Downloads GSM8K dataset automatically.

## Usage

### Memorization Capacity Experiments (`run_memorization_comparison.sh`)

Runs systematic ablation studies on synthetic memorization tasks:

```bash
bash run_memorization_comparison.sh
```

**What it does:**
- Tests multiple configurations: batch_size (8), max_length (128), prompt_length (1, 2, 5)
- Compares query positions: "before" vs "after" input text
- Runs 5 independent trials per configuration for statistical reliability
- Evaluates: test accuracy, perplexity, embedding norm changes, angle changes, cosine similarity

**Key parameters:**
- `CONFIGS`: Defines (batch_size, max_length, prompt_length) combinations
- `POSITIONS`: query placement ("before", "after")
- `NUM_RUNS`: Number of trials per configuration (default: 5)
- `ID_LENGTH`: Length of synthetic IDs to memorize (default: 1)

**Output:** Detailed logs and statistical summaries in `mem_exp_results/` directory.

### Training Sample Size Ablation (`run_training_samples_test.sh`)

Ablation study on GSM8K task with varying training sample sizes:

```bash
bash run_training_samples_test.sh
```

**What it does:**
- Tests different training sample sizes: 100, 500, 1000, 2000, 5000
- Fixed prompt length (1) with infix positioning
- Evaluates how embedding capacity affects mathematical reasoning performance
- Uses chat template for structured input formatting

**Key parameters (configurable in script):**
- `TRAINING_SAMPLES`: Array of training sizes to test
- `PROMPT_POSITION`: Embedding position ("infix", "prefix", "suffix")
- `MODEL_NAME`: Base model (default: Llama-3.2-3B-Instruct)
- `EPOCHS`: Training epochs per run
- `BATCH_SIZE`: Training batch size

**Output:** Individual logs per configuration in `results/` directory.

## Experimental Design

### Embedding Positions Compared
- **Prefix tuning**: Embeddings placed before input text
- **TokMem**: Embeddings placed after input text (infix position)

### Metrics Evaluated
- **Test Accuracy**: Memorization/generation accuracy
- **Perplexity**: Language modeling quality
- **Embedding Dynamics**: Initial/final norm, angle changes, cosine similarity
- **Statistical Reliability**: Multiple runs with different random seeds

