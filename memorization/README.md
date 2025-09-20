# Memory placement analysis

Prompt‑tuning style memorization/control experiments using frozen base models plus trainable prompt embeddings. Utilities to generate synthetic IDs and combine them with text for memorization studies.

## Contents

- `main_memorization.py` — `SingleModelPromptTuning` module and batch generation helpers.
- `training.py`, `model.py`, `main.py` — training/eval utilities for memorization baselines.
- `run_memorization_comparison.sh`, `run_training_samples_test.sh` — example runners.
- `dataset.py` — small helpers for memorization datasets.

## Setup

```bash
pip install -r ../requirements.txt
pip install pandas
```

## Data

- Expects CSV under `memorization_data/`, e.g. `memorization_data/fanfics_1k_chunks.csv`.

## Usage

- Run the comparison script:
  ```bash
  bash memorization/run_memorization_comparison.sh
  ```

## Notes

- Uses BF16 by default when available; adjust dtype/device as needed.
- All pretrained LM weights stay frozen; only prompt embeddings are trainable.

