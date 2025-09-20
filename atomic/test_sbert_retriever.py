#!/usr/bin/env python3
"""
Test Sentence-BERT retriever for task demonstration retrieval.
Evaluates if the retriever can correctly match test queries to few-shot examples from the same task.
"""

import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

# Import from local modules
from task_dataset import sample_natural_instructions_tasks

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SBERTRetriever:
    """Sentence-BERT based retriever for few-shot examples"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the sentence-BERT model"""
        print(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus_metadata = []
        
    def build_corpus(self, tasks_data: Dict[str, List[Dict]], task_instructions: Dict[str, str] = None):
        """Build retrieval corpus from few-shot examples of all tasks
        
        Args:
            tasks_data: Dictionary mapping task names to their few-shot examples
            task_instructions: Optional dictionary mapping task names to their instructions
        """
        corpus_texts = []
        self.corpus_metadata = []
        
        print(f"\nBuilding retrieval corpus from {len(tasks_data)} tasks...")
        
        for task_name, examples in tasks_data.items():
            for example in examples:
                # Concatenate input and output for richer representation
                if task_instructions and task_name in task_instructions:
                    # Prepend instruction if provided
                    corpus_text = f"{task_instructions[task_name]} {example['input']}"
                else:
                    corpus_text = f"{example['input']} {example.get('output', '')}"
                corpus_texts.append(corpus_text)
                self.corpus_metadata.append({
                    'task': task_name,
                    'input': example['input'],
                    'output': example.get('output', ''),
                    'explanation': example.get('explanation', '')
                })
        
        print(f"Total corpus size: {len(corpus_texts)} few-shot examples")
        
        # Encode all corpus texts
        print("Encoding corpus with Sentence-BERT...")
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        return len(corpus_texts)
    
    def retrieve_top_k(self, query: str, k: int = 1) -> List[Dict]:
        """Retrieve top-k most similar examples for a query
        
        Args:
            query: Query text
            k: Number of top results to return
            
        Returns:
            List of dictionaries with retrieved examples and scores
        """
        # Encode query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.corpus_embeddings.cpu().numpy()
        )[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'score': float(similarities[idx]),
                'task': self.corpus_metadata[idx]['task'],
                'input': self.corpus_metadata[idx]['input'],
                'output': self.corpus_metadata[idx]['output'],
                'explanation': self.corpus_metadata[idx]['explanation']
            })
        
        return results

def evaluate_retriever(retriever: SBERTRetriever, test_data: List[Dict], 
                       verbose: bool = True) -> Dict:
    """Evaluate retriever on test data
    
    Args:
        retriever: Initialized SBERTRetriever with built corpus
        test_data: List of test examples with 'query' and 'tasks' fields
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    
    # Track per-task performance
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    
    # Track confusion matrix
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # Store examples for error analysis
    error_examples = []
    success_examples = []
    
    print(f"\nEvaluating on {len(test_data)} test samples...")
    
    for item in tqdm(test_data, desc="Evaluating"):
        # Concatenate instruction and query for full context
        query_text = f"{item['instruction']} {item['query']}"
        true_task = item['tasks'][0]  # Natural Instructions has 1 task per sample
        
        # Retrieve top-1 result
        results = retriever.retrieve_top_k(query_text, k=1)
        retrieved_task = results[0]['task']
        
        # Update metrics
        is_correct = (retrieved_task == true_task)
        if is_correct:
            correct += 1
            if len(success_examples) < 5:  # Store first 5 successes
                success_examples.append({
                    'query': item['query'][:100] + '...' if len(item['query']) > 100 else item['query'],
                    'true_task': true_task,
                    'retrieved_task': retrieved_task,
                    'score': results[0]['score']
                })
        else:
            if len(error_examples) < 10:  # Store first 10 errors
                error_examples.append({
                    'query': item['query'][:100] + '...' if len(item['query']) > 100 else item['query'],
                    'true_task': true_task,
                    'retrieved_task': retrieved_task,
                    'score': results[0]['score'],
                    'retrieved_input': results[0]['input'][:100] + '...' if len(results[0]['input']) > 100 else results[0]['input']
                })
        
        total += 1
        task_correct[true_task] += int(is_correct)
        task_total[true_task] += 1
        confusion_matrix[true_task][retrieved_task] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct / total if total > 0 else 0
    
    # Calculate per-task accuracy
    task_accuracies = {}
    for task in task_total:
        task_accuracies[task] = task_correct[task] / task_total[task] if task_total[task] > 0 else 0
    
    # Find most confused task pairs
    confused_pairs = []
    for true_task in confusion_matrix:
        for pred_task in confusion_matrix[true_task]:
            if true_task != pred_task:
                count = confusion_matrix[true_task][pred_task]
                if count > 0:
                    confused_pairs.append((true_task, pred_task, count))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    results = {
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'task_accuracies': task_accuracies,
        'confusion_matrix': dict(confusion_matrix),
        'confused_pairs': confused_pairs[:10],  # Top 10 confused pairs
        'error_examples': error_examples,
        'success_examples': success_examples
    }
    
    if verbose:
        print_results(results)
    
    return results

def print_results(results: Dict):
    """Print evaluation results in a formatted way"""
    print(f"\nOverall Top-1 Accuracy: {results['overall_accuracy']:.1%} ({results['correct']}/{results['total']} correct)")

def main():
    parser = argparse.ArgumentParser(description='Test Sentence-BERT retriever for task demonstrations')
    parser.add_argument('--num_tasks', type=int, default=10,
                       help='Number of tasks to test (default: 10)')
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Sentence-BERT model name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--test_size', type=int, default=50,
                       help='Number of test samples per task')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save results JSON')
    parser.add_argument('--tokenizer_model', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                       help='Tokenizer model for filtering')
    parser.add_argument('--include_instruction_in_demos', action='store_true',
                       help='Include task instruction in few-shot demo encoding')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize tokenizer for filtering
    print(f"Loading tokenizer: {args.tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    
    # Load data with few-shot examples
    print(f"\nLoading {args.num_tasks} tasks from Natural Instructions...")
    train_data, val_data, test_data, task_names = sample_natural_instructions_tasks(
        tasks_dir="natural-instructions-2.8/tasks",
        num_tasks=args.num_tasks,
        max_instruction_tokens=1000,
        tokenizer=tokenizer,
        stable_test_split=True,
        test_size=args.test_size,
        few_shot=True  # Enable few-shot examples
    )
    
    print(f"Loaded {len(task_names)} tasks")
    print(f"Test samples: {len(test_data)}")
    
    # Extract few-shot examples by task
    tasks_few_shot = defaultdict(list)
    task_instructions = {}
    
    # Get few-shot examples from the first training sample of each task
    # (all samples from the same task have the same few-shot examples)
    seen_tasks = set()
    for item in train_data:
        task = item['tasks'][0]
        if task not in seen_tasks:
            if 'few_shot_examples' in item:
                tasks_few_shot[task] = item['few_shot_examples']
            # Store instruction for this task
            task_instructions[task] = item.get('instruction', '')
            seen_tasks.add(task)
    
    # Verify we have few-shot examples for all tasks
    print(f"\nFew-shot examples per task:")
    total_examples = 0
    for task in sorted(task_names):
        count = len(tasks_few_shot.get(task, []))
        print(f"  {task}: {count} examples")
        total_examples += count
    
    if total_examples == 0:
        print("\nERROR: No few-shot examples found. Make sure the Natural Instructions dataset has 'Positive Examples'.")
        return
    
    print(f"\nTotal few-shot examples across all tasks: {total_examples}")
    
    # Initialize retriever
    retriever = SBERTRetriever(model_name=args.model_name)
    
    # Build corpus from few-shot examples
    # Pass task instructions if the flag is set
    if args.include_instruction_in_demos:
        print("Including task instructions in few-shot demo encoding")
        corpus_size = retriever.build_corpus(tasks_few_shot, task_instructions)
    else:
        corpus_size = retriever.build_corpus(tasks_few_shot)
    
    # Evaluate retriever
    results = evaluate_retriever(retriever, test_data, verbose=True)
    
    # Save results if requested
    if args.save_results:
        # Convert defaultdicts to regular dicts for JSON serialization
        save_data = {
            'args': vars(args),
            'results': {
                'overall_accuracy': results['overall_accuracy'],
                'correct': results['correct'],
                'total': results['total'],
                'task_accuracies': results['task_accuracies'],
                'confusion_matrix': {k: dict(v) for k, v in results['confusion_matrix'].items()},
                'confused_pairs': results['confused_pairs'],
                'error_examples': results['error_examples'][:5],  # Save only first 5
                'success_examples': results['success_examples'][:5]
            },
            'corpus_size': corpus_size,
            'task_names': task_names
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")

if __name__ == "__main__":
    main()