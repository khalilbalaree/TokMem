# TokMem: Tokenized Procedural Memory for LLMs (Anonymous Release)

This repository implements **TokMem**, a method that enables large language models to acquire and recall procedural knowledges. The approach freezes the base LM while training only memory token embeddings.

## Three Experimental Tracks

### **Atomic Memory Recall** (`atomic/`)
Single-task memory with atomic tokens as task identifiers. Each token encodes a specific task's procedural knowledge.
- **What it does**: Train task-specific embeddings while keeping the base LM frozen
- **Use case**: Learning individual tasks without catastrophic forgetting
- **Quick start**: `cd atomic && bash test_tokmem.sh`

### **Compositional Memory Recall** (`compositional/`)  
Multi-tool function calling with function calls as procedural knowledge. Accomplishing a query requires multiple function calls. Supports both TokMem embeddings and LoRA baselines.
- **What it does**: Learn to call multiple tools/functions in sequence across training rounds
- **Use case**: Compositional reasoning and tool use without interference
- **Quick start**: `cd compositional && bash run_n_rounds_main.sh`

### **Memory Consolidation** (`memorization/`)
Prompt-tuning style memorization experiments on structured data (GSM8K, text chunks).
- **What it does**: Store and retrieve factual knowledge via memory tokens
- **Use case**: Efficient knowledge storage and retrieval
- **Quick start**: `cd memorization && bash run_memorization_comparison.sh`

The code targets Hugging Face Transformers models (e.g., Llama and Qwen) and PyTorch.

## Key Features

- **Frozen Base LM**: Only memory token embeddings and projections are trained
- **No Catastrophic Forgetting**: New tasks/tools don't interfere with previously learned ones  
- **Efficient**: Much smaller parameter updates compared to full fine-tuning
- **Modular**: Easy to add new tasks/tools without retraining existing ones
- **Baseline Comparisons**: Includes LoRA Fine-tuning, replay memory baselines, and no-training baselines


**Models**: Supports Hugging Face transformers (tested on Llama-3.2-1B/3B-Instruct, qwen-2.5-0.5B-Instruct)
- GPU with BF16/FP16 support recommended
- Adjust `--dtype` and CUDA device settings in scripts as needed

## Reproducibility

- Scripts include seed settings for deterministic results
- For full reproducibility, enable CUDA determinism in your environment
- Results may vary slightly across different GPU architectures

## License & Citation

- Anonymized for review.
