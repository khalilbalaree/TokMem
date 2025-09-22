# TokMem: Tokenized Procedural Memory for LLMs

This repository implements **TokMem**, a method that enables large language models to acquire and recall procedural knowledges. The approach freezes the base LM while training only memory token embeddings.

## Three Experimental Tracks

### **Atomic Memory Recall** (`atomic/`)
Atomic task learning using reserved special tokens as task identifiers. Each token encodes a specific task's procedural knowledge.
- **What it does**: Train task-specific embeddings for individual tasks while keeping the base LM frozen
- **Use case**: Learning individual tasks without catastrophic forgetting
- **Quick start**: `cd atomic && bash test_tokmem.sh`

### **Compositional Memory Recall** (`compositional/`)
Multi-tool function calling with sequential training across disjoint tool ranges. Supports both TokMem embeddings and LoRA baselines for continual learning.
- **What it does**: Learn to call multiple tools/functions in sequence across training rounds
- **Use case**: Compositional reasoning and tool use without interference
- **Quick start**: `cd compositional && bash run_n_rounds_main.sh` (TokMem) or `bash run_n_rounds_lora.sh` (LoRA baseline)

### **Embedding Capacity Ablation** (`memorization/`)
Systematic ablation experiments comparing TokMem vs prefix tuning approaches. Tests embedding capacity, memorization performance, and generalization on structured tasks.
- **What it does**: Compare embedding positions, dimensions, and training dynamics between approaches
- **Use case**: Analyze memorization capacity and embedding behavior differences
- **Quick start**: `cd memorization && bash run_memorization_comparison.sh` (memorization tasks) or `bash run_training_samples_test.sh` (GSM8K)

The code targets Hugging Face Transformers models (e.g., Llama and Qwen) and PyTorch.

## Key Features

- **Frozen Base LM**: Only memory token embeddings and projections are trained
- **No Catastrophic Forgetting**: New tasks/tools don't interfere with previously learned ones  
- **Efficient**: Much smaller parameter updates compared to full fine-tuning
- **Modular**: Easy to add new tasks/tools without retraining existing ones
- **Baseline Comparisons**: Includes LoRA fine-tuning, replay memory baselines, in-context learning (ICL) baselines, and no-training baselines

**Models**: Supports Hugging Face transformers (tested on Llama-3.2-1B/3B-Instruct, qwen-2.5-0.5B-Instruct)
- GPU with BF16/FP16 support recommended
- Adjust `--dtype` and CUDA device settings in scripts as needed

## Reproducibility

- Scripts include seed settings for deterministic results
- For full reproducibility, enable CUDA determinism in your environment
- Results may vary slightly across different GPU architectures

## License & Citation

- Anonymized for review.
