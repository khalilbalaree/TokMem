#!/usr/bin/env python3
"""
In-Context Learning (ICL) Baseline for Function Calling

This script implements a baseline that directly asks a Llama model to predict function calls
based on tool descriptions and user queries, without any fine-tuning.
Uses the same generation settings as the main project for fair comparison.
"""

import json
import random
import argparse
import os
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval import compare_function_calls_advanced, calculate_tool_selection_accuracy
from tool_retrieval import ToolRetriever

def load_tool_descriptions(filepath="tool_descriptions.json") -> Dict[str, Any]:
    """Load tool descriptions from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tool descriptions file not found: {filepath}\n"
                                f"Please ensure xlam_datasets.py has been run to generate this file.")
    with open(filepath, 'r') as f:
        return json.load(f)

def load_test_data(filepath="function_calling_test.json") -> List[Dict[str, Any]]:
    """Load test data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data file not found: {filepath}\n"
                                f"Please ensure xlam_datasets.py has been run to generate this file.")
    with open(filepath, 'r') as f:
        return json.load(f)

def format_tool_descriptions(tool_descriptions: Dict[str, Any]) -> str:
    """Format tool descriptions for the prompt using original JSON format"""
    return json.dumps(tool_descriptions, indent=2)

def create_icl_prompt(user_query: str, tool_descriptions: Dict[str, Any]) -> str:
    """Create the ICL prompt for the LLM"""
    tools_text = format_tool_descriptions(tool_descriptions)
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a function calling assistant. Given a user query and available tools, determine which functions to call and with what arguments.

Available Tools:
{tools_text}

IMPORTANT: Output ONLY the function call arguments as JSON objects. For each function call needed, output the JSON arguments directly without any extra text, explanations, or formatting.

Format: Output each function call as a separate JSON object on its own line:
{{"param1": "value1", "param2": "value2"}}
{{"param1": "value3", "param2": "value4"}}

Examples:
- For one function call: {{"numbers": [5.5, 2.2, 9.9], "descending": false}}
- For multiple calls: 
{{"contingency_table": [[100, 150], [50, 100]], "significance_level": 0.01}}
{{"contingency_table": [[200, 100], [150, 50]], "significance_level": 0.01}}
{{"numbers": [5.5, 2.2, 9.9, 3.3, 7.7], "descending": false}}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

class ICLBaseline:
    """In-Context Learning baseline using Hugging Face transformers"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda", dtype=torch.bfloat16,
                 use_rag=False, retrieval_k=10):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.use_rag = use_rag
        self.retrieval_k = retrieval_k
        
        # Initialize retriever if RAG is enabled
        self.retriever = None
        if self.use_rag:
            print(f"Initializing RAG with top-{retrieval_k} retrieval...")
            self.retriever = ToolRetriever()
        
        print(f"Loading model: {model_name}")
        print(f"Device: {device}, Dtype: {dtype}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if device == "cuda" else None
        )
        
        # Configure tokenizer for decoder-only models
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Use left padding for decoder-only models
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_batch_responses(self, prompts: List[str], max_length=4096, max_new_tokens=256, temperature=0.6, do_sample=True, top_p=0.9) -> List[str]:
        """Generate responses for a batch of prompts"""
        if not prompts:
            return []
        
        # Tokenize all prompts (no truncation for long prompts)
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            truncation=False,  # Disabled truncation
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode only the new tokens for each sequence
        responses = []
        for i, generated_seq in enumerate(generated):
            input_length = inputs['input_ids'][i].shape[0]
            new_tokens = generated_seq[input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def parse_response(self, response: str) -> List[str]:
        """Parse the model response to extract function calls in {}{}... format"""
        response = response.strip()
        if not response:
            return []
        
        # Split response into lines and try to parse each as JSON
        function_calls = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse each line as a JSON object
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    function_calls.append(json.dumps(parsed))
                else:
                    # Handle non-dict responses by wrapping them
                    function_calls.append(json.dumps({"value": parsed}))
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
        
        # If no valid JSON found, try parsing the entire response as one JSON
        if not function_calls:
            try:
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            function_calls.append(json.dumps(item))
                        else:
                            function_calls.append(json.dumps({"value": item}))
                elif isinstance(parsed, dict):
                    function_calls.append(json.dumps(parsed))
            except json.JSONDecodeError:
                # Last resort: treat as raw text if it's not empty
                if response.strip():
                    function_calls.append(json.dumps({"raw_text": response, "parse_error": True}))
        
        return function_calls
    
    def _infer_tool_from_call(self, call_dict: Dict[str, Any], tool_descriptions: Dict[str, Any]) -> Optional[str]:
        """Infer which tool a function call corresponds to by matching parameter names"""
        if not call_dict or 'parse_error' in call_dict or 'raw_text' in call_dict:
            return None
        
        call_params = set(call_dict.keys())
        best_match = None
        best_score = 0
        
        for tool_name, tool_info in tool_descriptions.items():
            tool_params = set(tool_info.get('parameters', {}).keys())
            
            if not tool_params:  # Tool has no parameters
                continue
                
            # Calculate overlap score
            overlap = len(call_params & tool_params)
            total_params = len(tool_params)
            
            # Score based on how many required parameters match
            if overlap > 0:
                score = overlap / total_params
                # Bonus if all call params are valid for this tool
                if call_params.issubset(tool_params):
                    score += 0.5
                
                if score > best_score:
                    best_score = score
                    best_match = tool_name
        
        return best_match
    
    def predict_batch(self, batch_data: List[Dict[str, Any]], tool_descriptions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make predictions for a batch of samples"""
        prompts = []
        batch_info = []
        
        # Prepare batch prompts
        for sample in batch_data:
            target_tools = list(set(sample.get('tools', [])))
            
            # Use RAG to retrieve relevant tools if enabled
            if self.use_rag and self.retriever:
                relevant_tools = self.retriever.retrieve(
                    sample['user_input'], 
                    top_k=self.retrieval_k
                )
                # Track retrieval metrics
                retrieved_tool_names = set(relevant_tools.keys())
                target_tool_names = set(target_tools)
                retrieval_recall = len(retrieved_tool_names & target_tool_names) / len(target_tool_names) if target_tool_names else 1.0
                retrieval_precision = len(retrieved_tool_names & target_tool_names) / len(retrieved_tool_names) if retrieved_tool_names else 0.0
            else:
                # Use all available tools (original behavior)
                relevant_tools = tool_descriptions
                retrieval_recall = 1.0
                retrieval_precision = len(target_tools) / len(tool_descriptions) if tool_descriptions else 0.0
            
            prompt = create_icl_prompt(sample['user_input'], relevant_tools)
            prompts.append(prompt)
            batch_info.append({
                'target_tools': target_tools,
                'prompt_length': len(self.tokenizer.encode(prompt)),
                'retrieval_recall': retrieval_recall,
                'retrieval_precision': retrieval_precision,
                'num_tools_in_prompt': len(relevant_tools)
            })
        
        # Generate responses for the batch
        responses = self.generate_batch_responses(prompts, max_length=4096, max_new_tokens=256, temperature=0.6, do_sample=False, top_p=0.9)
        
        # Parse responses and combine with batch info
        results = []
        for i, (response, info) in enumerate(zip(responses, batch_info)):
            predicted_calls = self.parse_response(response)
            results.append({
                'predicted_calls': predicted_calls,
                'raw_response': response,
                'prompt_length': info['prompt_length'],
                'target_tools': info['target_tools'],
                'retrieval_recall': info.get('retrieval_recall', 1.0),
                'retrieval_precision': info.get('retrieval_precision', 0.0),
                'num_tools_in_prompt': info.get('num_tools_in_prompt', 0)
            })
        
        return results
    
    def run_evaluation(self, test_data_path="function_calling_test.json", 
                      tool_descriptions_path="tool_descriptions.json",
                      max_samples=None, output_file="icl_results.json", batch_size=8) -> Dict[str, float]:
        """Run full evaluation on test data"""
        
        print("Loading data...")
        tool_descriptions = load_tool_descriptions(tool_descriptions_path)
        test_data = load_test_data(test_data_path)
        
        # Initialize retriever with all tools if RAG is enabled
        if self.use_rag and self.retriever:
            print("Indexing tools for retrieval...")
            self.retriever.index_tools(tool_descriptions)
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        print(f"Evaluating on {len(test_data)} samples...")
        print(f"Using {len(tool_descriptions)} available tools")
        
        results = []
        total_exact_matches = 0
        total_partial_scores = 0
        
        # Process data in batches
        num_batches = (len(test_data) + batch_size - 1) // batch_size
        print(f"Processing {num_batches} batches of size {batch_size}...")
        
        for batch_idx in range(0, len(test_data), batch_size):
            batch_end = min(batch_idx + batch_size, len(test_data))
            batch_data = test_data[batch_idx:batch_end]
            current_batch_num = batch_idx // batch_size + 1
            
            print(f"Processing batch {current_batch_num}/{num_batches} (samples {batch_idx+1}-{batch_end})...")
            
            # Make batch predictions
            batch_predictions = self.predict_batch(batch_data, tool_descriptions)
            
            # Evaluate each sample in the batch
            for i, (sample, prediction) in enumerate(zip(batch_data, batch_predictions)):
                sample_id = batch_idx + i
                
                # Get target tools and function calls
                target_tools = sample.get('tools', [])
                target_calls = sample.get('function_calls', [])
                
                # Evaluate using the same function as the main project
                eval_result = compare_function_calls_advanced(
                    prediction['predicted_calls'], 
                    target_calls, 
                    ignore_order=True
                )
                
                # Calculate tool selection accuracy
                tool_selection_metrics = calculate_tool_selection_accuracy(
                    prediction['predicted_calls'],
                    target_calls
                )

                # Track metrics
                if eval_result.exact_match:
                    total_exact_matches += 1
                total_partial_scores += eval_result.f1_score
                
                # Store detailed result
                results.append({
                    'sample_id': sample_id,
                    'user_input': sample['user_input'],
                    'target_tools': target_tools,
                    'target_calls': target_calls,
                    'predicted_calls': prediction['predicted_calls'],
                    'raw_response': prediction['raw_response'],
                    'exact_match': eval_result.exact_match,
                    'f1_score': eval_result.f1_score,
                    'precision': eval_result.precision,
                    'recall': eval_result.recall,
                    'tool_selection': tool_selection_metrics,
                    'eval_details': eval_result.details,
                    'prompt_length': prediction['prompt_length'],
                    'retrieval_recall': prediction.get('retrieval_recall', 1.0),
                    'retrieval_precision': prediction.get('retrieval_precision', 0.0),
                    'num_tools_in_prompt': prediction.get('num_tools_in_prompt', 0)
                })
        
        # Calculate comprehensive metrics
        n_samples = len(test_data)
        
        # Calculate aggregated metrics
        total_precision = 0
        total_recall = 0
        total_tool_f1 = 0
        total_tool_precision = 0
        total_tool_recall = 0
        total_retrieval_recall = 0
        total_retrieval_precision = 0
        total_prompt_reduction = 0
        parse_errors = 0
        call_count_stats = {}  # Track accuracy by number of function calls
        
        for result in results:
            # Aggregate precision and recall
            total_precision += result['precision']
            total_recall += result['recall']
            
            # Aggregate tool selection metrics
            if 'tool_selection' in result:
                total_tool_f1 += result['tool_selection']['tool_f1_score']
                total_tool_precision += result['tool_selection']['tool_precision']
                total_tool_recall += result['tool_selection']['tool_recall']
            
            # Aggregate retrieval metrics if using RAG
            if self.use_rag:
                total_retrieval_recall += result.get('retrieval_recall', 0)
                total_retrieval_precision += result.get('retrieval_precision', 0)
                # Calculate prompt reduction
                original_tools = len(tool_descriptions)
                used_tools = result.get('num_tools_in_prompt', original_tools)
                reduction = 1.0 - (used_tools / original_tools) if original_tools > 0 else 0
                total_prompt_reduction += reduction
            
            # Check for parse errors
            has_parse_error = False
            for call_str in result['predicted_calls']:
                try:
                    call_dict = json.loads(call_str)
                    if 'parse_error' in call_dict:
                        has_parse_error = True
                        break
                except:
                    has_parse_error = True
                    break
            
            if has_parse_error:
                parse_errors += 1
            
            # Track by function call count
            num_calls = len(result['target_calls'])
            if num_calls not in call_count_stats:
                call_count_stats[num_calls] = {
                    'correct': 0, 
                    'total': 0,
                    'generation_f1_sum': 0,
                    'tool_f1_sum': 0
                }
            call_count_stats[num_calls]['total'] += 1
            call_count_stats[num_calls]['generation_f1_sum'] += result['f1_score']
            if 'tool_selection' in result:
                call_count_stats[num_calls]['tool_f1_sum'] += result['tool_selection']['tool_f1_score']
            if result['exact_match']:
                call_count_stats[num_calls]['correct'] += 1
        
        # Calculate averages for call count breakdown
        for num_calls, stats in call_count_stats.items():
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
                stats['avg_generation_f1'] = stats['generation_f1_sum'] / stats['total']
                stats['avg_tool_f1'] = stats['tool_f1_sum'] / stats['total']
            else:
                stats['accuracy'] = 0
                stats['avg_generation_f1'] = 0
                stats['avg_tool_f1'] = 0
        
        final_metrics = {
            'exact_match_accuracy': total_exact_matches / n_samples,
            'average_f1_score': total_partial_scores / n_samples,
            'average_precision': total_precision / n_samples,
            'average_recall': total_recall / n_samples,
            'tool_selection_f1': total_tool_f1 / n_samples,
            'tool_selection_precision': total_tool_precision / n_samples,
            'tool_selection_recall': total_tool_recall / n_samples,
            'parse_error_rate': parse_errors / n_samples,
            'total_samples': n_samples,
            'call_count_breakdown': call_count_stats
        }
        
        # Add RAG metrics if enabled
        if self.use_rag:
            final_metrics['retrieval_recall'] = total_retrieval_recall / n_samples
            final_metrics['retrieval_precision'] = total_retrieval_precision / n_samples
            final_metrics['avg_prompt_reduction'] = total_prompt_reduction / n_samples
            final_metrics['avg_tools_in_prompt'] = sum(r.get('num_tools_in_prompt', 0) for r in results) / n_samples
        
        # Save results
        output_data = {
            'metrics': final_metrics,
            'detailed_results': results,
            'config': {
                'model_name': self.model_name,
                'device': str(self.device),
                'dtype': str(self.dtype),
                'use_rag': self.use_rag,
                'retrieval_k': self.retrieval_k if self.use_rag else None,
                'batch_size': batch_size,
                'tool_descriptions_path': tool_descriptions_path,
                'test_data_path': test_data_path,
                'generation_params': {
                    'max_new_tokens': 256,
                    'temperature': 0.1,
                    'do_sample': True,
                    'top_p': 0.9
                }
            },
            'summary_stats': {
                'total_exact_matches': total_exact_matches,
                'total_parse_errors': parse_errors,
                'total_samples': n_samples,
                'average_f1': total_partial_scores / n_samples,
                'average_precision': total_precision / n_samples,
                'average_recall': total_recall / n_samples
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Print detailed results in the requested format
        print(f"\n{'='*50}")
        print(f" RESULTS:")
        print(f"   Exact Match Accuracy:     {final_metrics['exact_match_accuracy']:.3f} ({total_exact_matches}/{n_samples})")
        print(f"   Average F1 Score:         {final_metrics['average_f1_score']:.3f}")
        print(f"   Average Precision:        {final_metrics['average_precision']:.3f}")
        print(f"   Average Recall:           {final_metrics['average_recall']:.3f}")
        print(f"   Tool Selection F1:        {final_metrics['tool_selection_f1']:.3f}")
        print(f"   Tool Selection Precision: {final_metrics['tool_selection_precision']:.3f}")
        print(f"   Tool Selection Recall:    {final_metrics['tool_selection_recall']:.3f}")
        print(f"   Parse Error Rate:         {final_metrics['parse_error_rate']:.3f}")
        
        # Print RAG metrics if enabled
        if self.use_rag:
            print(f"\n   📚 RAG METRICS:")
            print(f"   Retrieval Recall:         {final_metrics['retrieval_recall']:.3f}")
            print(f"   Retrieval Precision:      {final_metrics['retrieval_precision']:.3f}")
            print(f"   Avg Prompt Reduction:     {final_metrics['avg_prompt_reduction']:.1%}")
            print(f"   Avg Tools in Prompt:      {final_metrics['avg_tools_in_prompt']:.1f}")
        
        print(f"{'='*50}")
        
        # Print breakdown by function call count
        if call_count_stats:
            print(f"\n📊 BREAKDOWN BY FUNCTION CALL COUNT:")
            print(f"{'--'*25}")
            for num_calls in sorted(call_count_stats.keys()):
                stats = call_count_stats[num_calls]
                if stats['total'] > 0:
                    print(f"\n   {num_calls} call(s) ({stats['total']} samples):")
                    print(f"      Exact Match:        {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
                    print(f"      Generation F1:      {stats['avg_generation_f1']:.3f}")
                    print(f"      Tool Selection F1:  {stats['avg_tool_f1']:.3f}")
        
        print(f"\nResults saved to: {output_file}")
        
        return final_metrics

def main():
    parser = argparse.ArgumentParser(description="ICL Baseline for Function Calling")
    parser.add_argument("--test_data", default="function_calling_test.json", 
                       help="Path to test data JSON file")
    parser.add_argument("--tool_descriptions", default="tool_descriptions.json",
                       help="Path to tool descriptions JSON file") 
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct",
                       help="Hugging Face model name")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (default: None - evaluate all)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--output", default="icl_results.json", help="Output file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # RAG arguments
    parser.add_argument("--use_rag", action="store_true",
                       help="Use RAG for tool retrieval")
    parser.add_argument("--retrieval_k", type=int, default=10,
                       help="Number of tools to retrieve when using RAG (default: 10)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Convert dtype string to torch dtype
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Create baseline and run evaluation
    baseline = ICLBaseline(
        model_name=args.model_name,
        device=args.device,
        dtype=dtype,
        use_rag=args.use_rag,
        retrieval_k=args.retrieval_k
    )
    
    # Print configuration
    if args.use_rag:
        print(f"\n🔍 RAG enabled: Retrieving top-{args.retrieval_k} tools per query")
    else:
        print("\n📦 RAG disabled: Using all tools in prompt")
    
    metrics = baseline.run_evaluation(
        test_data_path=args.test_data,
        tool_descriptions_path=args.tool_descriptions,
        max_samples=args.max_samples,
        output_file=args.output,
        batch_size=args.batch_size
    )
    
    print("\n✅ ICL baseline evaluation complete!")

if __name__ == "__main__":
    main()
