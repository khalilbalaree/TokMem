#!/usr/bin/env python3
"""
Natural Instructions evaluation module
Integrates the evaluation methods from Natural Instructions dataset
"""

import string
from typing import List, Dict, Union

try:
    from rouge_score import rouge_scorer
    from transformers import AutoTokenizer
    ROUGE_AVAILABLE = True
    
    class GPTTokenizer:
        def __init__(self):
            self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

        def tokenize(self, s):
            tokens = self.gpt_tokenizer.tokenize(s)
            # GPT2 uses Byte-level BPE, which will include space as part of the word. 
            # But for the first word of a sentence, there is no space before it. 
            # So, we remove all the added spaces ("Ġ"). 
            tokens = [t.lstrip("Ġ") for t in tokens]
            return tokens

    default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    xlingual_tokenizer = GPTTokenizer()
    xlingual_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    
except ImportError:
    print("Warning: rouge-score not available. Install with: pip install rouge-score")
    print("         Falling back to exact match only evaluation")
    ROUGE_AVAILABLE = False
    default_rouge_scorer = None
    xlingual_rouge_scorer = None


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, and extra whitespace.
    Adapted from Squad v1.1 evaluation, without removing articles.
    """
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction: str, ground_truth: str, xlingual: bool = False) -> bool:
    """Compute exact match after normalization"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def rouge_score(prediction: str, ground_truth: str, xlingual: bool = False) -> float:
    """Compute ROUGE-L F1 score"""
    if not ROUGE_AVAILABLE:
        # Fallback to normalized exact match
        return 1.0 if exact_match(prediction, ground_truth, xlingual) else 0.0
    
    if xlingual and xlingual_rouge_scorer:
        scorer = xlingual_rouge_scorer
    else:
        scorer = default_rouge_scorer
    
    try:
        scores = scorer.score(prediction=prediction, target=ground_truth)
        return scores["rougeL"].fmeasure
    except Exception as e:
        print(f"Warning: ROUGE scoring failed: {e}")
        return 1.0 if exact_match(prediction, ground_truth, xlingual) else 0.0


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str], xlingual: bool = False) -> float:
    """Compute metric against multiple ground truths and return the maximum score"""
    if not ground_truths:
        return 0.0
    
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_predictions(predictions: List[str], references: List[Union[str, List[str]]], 
                        task_names: List[str] = None, xlingual: bool = False) -> Dict:
    """
    Evaluate predictions using Natural Instructions metrics
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings or list of reference lists
        task_names: Optional list of task names for per-task breakdown
        xlingual: Whether to use cross-lingual evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    assert len(predictions) == len(references), f"Predictions ({len(predictions)}) and references ({len(references)}) must have same length"
    
    if task_names:
        assert len(task_names) == len(predictions), f"Task names ({len(task_names)}) must match predictions length"
    
    # Convert single references to lists
    ref_lists = []
    for ref in references:
        if isinstance(ref, str):
            ref_lists.append([ref])
        elif isinstance(ref, list):
            ref_lists.append(ref)
        else:
            ref_lists.append([str(ref)])
    
    # Compute overall metrics
    em_scores = []
    rouge_scores = []
    
    for pred, ref_list in zip(predictions, ref_lists):
        # Exact match - max over ground truths  
        def em_metric(prediction, ground_truth, xlingual=False):
            return 1.0 if exact_match(prediction, ground_truth, xlingual) else 0.0
        
        em_score = metric_max_over_ground_truths(
            em_metric, pred, ref_list, xlingual=xlingual
        )
        em_scores.append(em_score)
        
        # ROUGE-L - max over ground truths
        rouge_f1 = metric_max_over_ground_truths(
            rouge_score, pred, ref_list, xlingual=xlingual
        )
        rouge_scores.append(rouge_f1)
    
    # Overall metrics
    overall_em = sum(em_scores) / len(em_scores) * 100.0
    overall_rouge = sum(rouge_scores) / len(rouge_scores) * 100.0
    
    results = {
        "exact_match": round(overall_em, 4),
        "rougeL": round(overall_rouge, 4),
        "num_examples": len(predictions)
    }
    
    # Per-task breakdown if task names provided
    if task_names:
        task_breakdown = {}
        
        # Group by task
        task_groups = {}
        for i, task in enumerate(task_names):
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(i)
        
        # Compute per-task metrics
        for task, indices in task_groups.items():
            task_em_scores = [em_scores[i] for i in indices]
            task_rouge_scores = [rouge_scores[i] for i in indices]
            
            task_breakdown[task] = {
                "exact_match": round(sum(task_em_scores) / len(task_em_scores) * 100.0, 4),
                "rougeL": round(sum(task_rouge_scores) / len(task_rouge_scores) * 100.0, 4),
                "num_examples": len(indices)
            }
        
        results["per_task"] = task_breakdown
    
    return results


def print_evaluation_results(results: Dict, title: str = "EVALUATION RESULTS"):
    """Pretty print evaluation results"""
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)
    
    print(f"📋 Examples evaluated: {results['num_examples']}")
    print()
    
    print("🎯 OVERALL METRICS:")
    print(f"   Exact Match:  {results['exact_match']:6.2f}%")
    if ROUGE_AVAILABLE:
        print(f"   ROUGE-L F1:   {results['rougeL']:6.2f}%")
    else:
        print(f"   ROUGE-L F1:   {results['rougeL']:6.2f}% (fallback)")
    
    # Per-task breakdown
    if "per_task" in results:
        print(f"\n📊 PER-TASK BREAKDOWN:")
        print("-" * 60)
        
        for task, metrics in sorted(results["per_task"].items()):
            print(f"   {task}:")
            print(f"      Exact Match:  {metrics['exact_match']:6.2f}% ({metrics['num_examples']} examples)")
            if ROUGE_AVAILABLE:
                print(f"      ROUGE-L F1:   {metrics['rougeL']:6.2f}%")
            else:
                print(f"      ROUGE-L F1:   {metrics['rougeL']:6.2f}% (fallback)")
    
    print("=" * 60)


if __name__ == "__main__":
    # Simple test
    predictions = ["The cat sat on the mat", "Paris is the capital", "42"]
    references = [["A cat sits on a mat", "The cat sat on the mat"], ["Paris is capital of France"], ["42"]]
    task_names = ["task1", "task2", "task1"]
    
    results = evaluate_predictions(predictions, references, task_names)
    print_evaluation_results(results, "TEST EVALUATION")