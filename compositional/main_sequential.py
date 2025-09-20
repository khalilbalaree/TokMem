#!/usr/bin/env python3
"""
Sequential training script for function calling with tool set progression.
Supports multiple rounds of training with different tool sets.
"""

import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer
import json
import os
import logging

from model import FunctionCallingModel, print_model_info
from dataset import create_native_dataloader, discover_available_tools
from training import train_native_function_calling_model, eval_native_function_calling, demo_native_function_calling


def discover_all_tool_names_from_rounds(rounds, data_dir, train_max_calls_by_round, test_max_calls_by_round):
    """Pre-discover all tool names from all rounds' data files

    Args:
        rounds: List of round specifications
        data_dir: Base directory for data files
        train_max_calls_by_round: Per-round train max call limits
        test_max_calls_by_round: Per-round test max call limits

    Returns:
        List of all tool names in order (tools 1-10, 11-20, etc.)
    """
    print("Pre-discovering all tool names from all rounds...")
    all_tool_names = []
    
    for round_idx, round_spec in enumerate(rounds):
        round_num = round_idx + 1
        tools_range = round_spec['tools']
        
        # Construct data file names
        train_data_file = os.path.join(
            data_dir,
            f"training/function_calling_train_tools{tools_range}_{train_max_calls_by_round[round_idx]}calls.json"
        )
        test_data_file = os.path.join(
            data_dir,
            f"test/function_calling_test_tools{tools_range}_{test_max_calls_by_round[round_idx]}calls.json"
        )
        
        # Check if files exist
        if not os.path.exists(train_data_file) or not os.path.exists(test_data_file):
            print(f"Warning: Data files not found for round {round_num} (tools {tools_range})")
            print(f"Expected: {train_data_file}")
            print(f"Expected: {test_data_file}")
            continue
        
        # Discover tools for this round
        round_tools = discover_available_tools(train_data_file, test_data_file)
        print(f"Round {round_num} (tools {tools_range}): {len(round_tools)} tools")
        
        # Add to complete list
        all_tool_names.extend(round_tools)
    
    print(f"Total discovered tools: {len(all_tool_names)}")
    print(f"First few tools: {all_tool_names[:5]}...")
    print(f"Last few tools: {all_tool_names[-5:]}")
    
    return all_tool_names


def freeze_lora_parameters(model):
    """Freeze all LoRA parameters in the model"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = False
            frozen_count += 1
    print(f"Frozen {frozen_count} LoRA parameters")
    return frozen_count


def unfreeze_lora_parameters(model):
    """Unfreeze all LoRA parameters in the model"""
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            unfrozen_count += 1
    print(f"Unfrozen {unfrozen_count} LoRA parameters")
    return unfrozen_count


# Active tool mapping system removed - model now uses original mappings for all 60 tools


def apply_orthogonal_init_all_tools(model, total_tools):
    """Apply orthogonal initialization to ALL tool embeddings at once
    
    Args:
        model: The model with tool embeddings
        total_tools: Total number of tools to initialize
    """
    print(f"Applying orthogonal initialization to all {total_tools} tool embeddings")
    
    with torch.no_grad():
        if model.decouple_embeddings:
            # Apply orthogonal initialization to both input and output embeddings
            input_embeddings = model.trainable_tool_input_embeddings[:total_tools]
            output_embeddings = model.trainable_tool_output_embeddings[:total_tools]
            
            # Convert to float32 for orthogonal initialization (BFloat16 not supported)
            original_dtype = input_embeddings.dtype
            input_temp = input_embeddings.float()
            output_temp = output_embeddings.float()
            
            # Initialize with orthogonal matrices
            torch.nn.init.orthogonal_(input_temp)
            torch.nn.init.orthogonal_(output_temp)
            
            # Convert back to original dtype
            input_embeddings.copy_(input_temp.to(original_dtype))
            output_embeddings.copy_(output_temp.to(original_dtype))
            
        else:
            # Apply orthogonal initialization to shared embeddings
            shared_embeddings = model.trainable_tool_embeddings[:total_tools]
            
            # Convert to float32 for orthogonal initialization (BFloat16 not supported)
            original_dtype = shared_embeddings.dtype
            shared_temp = shared_embeddings.float()
            
            # Initialize with orthogonal matrices
            torch.nn.init.orthogonal_(shared_temp)
            
            # Convert back to original dtype
            shared_embeddings.copy_(shared_temp.to(original_dtype))
    
    print(f"Orthogonal initialization applied to all {total_tools} tool embeddings - each embedding is orthogonal to all others")


def reinitialize_tool_embeddings(model, new_tool_names):
    """DEPRECATED: This function should not be used with pre-allocated embeddings.
    Use update_tool_mappings_for_round instead."""
    # Suppress unused parameter warnings
    _ = model
    _ = new_tool_names
    raise NotImplementedError(
        "reinitialize_tool_embeddings should not be used with pre-allocated embeddings. "
        "Use update_tool_mappings_for_round instead."
    )


def parse_training_rounds(rounds_str):
    """
    Parse training rounds specification.
    Format: "tools:epochs,tools:epochs,..."
    Example: "1-10:3,11-20:3,21-30:2"
    """
    rounds = []
    for round_spec in rounds_str.split(','):
        parts = round_spec.strip().split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid round specification: {round_spec}. Use 'tools:epochs' format.")
        
        tools = parts[0].strip()
        epochs = int(parts[1].strip())
        rounds.append({'tools': tools, 'epochs': epochs})
    
    return rounds


def main():
    parser = argparse.ArgumentParser(description="Sequential Function Calling Training with Tool Progression")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base model name")
    
    # Sequential training arguments
    parser.add_argument("--training_rounds", type=str, required=True,
                        help="Training rounds specification (e.g., '1-10:3,11-20:3,21-30:2' for 3 rounds)")
    parser.add_argument("--freeze_lora_after_first", action="store_true",
                        help="Freeze LoRA parameters after first round")
    
    # Training arguments  
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate for tokenized memory embeddings")
    parser.add_argument("--lora_lr", type=float, default=5e-4,
                        help="Learning rate for LoRA parameters (default: 5e-4)")
    
    # Evaluation arguments
    parser.add_argument("--eval_after_each_round", action="store_true",
                        help="Run evaluation after each training round")
    parser.add_argument("--eval_all_previous", action="store_true",
                        help="Evaluate on all previous test sets after each round")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--demo", type=int, default=None,
                        help="Number of demo examples to show after training")
    parser.add_argument("--use_ground_truth_tools", action="store_true",
                        help="Use ground truth tools during inference")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--decouple_embeddings", action="store_true",
                        help="Use separate weights for input and output tool token embeddings")
    parser.add_argument("--renorm_active_tools", action="store_true",
                        help="Renormalize active tool output-embedding norms post-step")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA fine-tuning of base model")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate (default: 0.1)")
    parser.add_argument("--lora_layer_indices", type=str, default=None,
                        help="Apply LoRA only to specific layer indices")
    parser.add_argument("--lora_target_modules", type=str, default="o_proj",
                        help="Target modules for LoRA")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--train_max_function_calls", type=int, default=4,
                        help="Maximum function calls used in data generation")
    parser.add_argument("--train_max_function_calls_per_round", type=str, default=None,
                        help="Comma-separated max function call limits per round (overrides --train_max_function_calls)")
    parser.add_argument("--test_max_function_calls", type=int, default=4,
                        help="Maximum function calls used in data generation")
    parser.add_argument("--test_max_function_calls_per_round", type=str, default=None,
                        help="Comma-separated max function call limits per round for evaluation (overrides --test_max_function_calls)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save model checkpoints after each round")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file for evaluation results")
    parser.add_argument("--curriculum_learning", action="store_true",
                        help="Enable curriculum learning")
    
    args = parser.parse_args()

    # No additional flags: default behavior keeps LoRA trainable unless freezing is requested
    
    # Setup simple logging
    import sys
    from datetime import datetime
    
    # Create log directory first
    os.makedirs("log", exist_ok=True)
    
    # Configure logging - put logs in log/ directory
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        # Ensure log file goes in log/ directory
        log_file_path = args.log_file if args.log_file.startswith('log/') else f"log/{os.path.basename(args.log_file)}"
        log_handlers.append(logging.FileHandler(log_file_path, mode='a'))
    else:
        # Create default log file in log/ directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_file = f"log/sequential_training_{timestamp}.log"
        log_handlers.append(logging.FileHandler(default_log_file, mode='a'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=log_handlers
    )
    logger = logging.getLogger(__name__)
    
    # Log start time and configuration
    logger.info(f"=== Sequential Training Started at {datetime.now()} ===")
    logger.info(f"Configuration: model={args.model_name}, rounds={args.training_rounds}, batch_size={args.batch_size}")
    
    # Parse training rounds
    try:
        rounds = parse_training_rounds(args.training_rounds)
        print(f"Parsed {len(rounds)} training rounds:")
        for i, round_spec in enumerate(rounds, 1):
            print(f"  Round {i}: Tools {round_spec['tools']}, {round_spec['epochs']} epochs")
    except ValueError as e:
        parser.error(str(e))

    num_rounds = len(rounds)

    def expand_per_round_values(values_str, fallback, arg_label):
        if values_str is None:
            return [fallback] * num_rounds

        raw_values = [part.strip() for part in values_str.split(',')]
        parsed_values = []
        for value in raw_values:
            if not value:
                continue
            try:
                parsed_values.append(int(value))
            except ValueError:
                parser.error(f"Invalid value '{value}' for {arg_label}; expected integers.")

        if not parsed_values:
            parser.error(f"No valid values provided for {arg_label}.")

        if len(parsed_values) < num_rounds:
            parsed_values.extend([parsed_values[-1]] * (num_rounds - len(parsed_values)))
        elif len(parsed_values) > num_rounds:
            print(f"Warning: {arg_label} specified {len(parsed_values)} values; using first {num_rounds}.")
            parsed_values = parsed_values[:num_rounds]

        return parsed_values

    train_max_calls_by_round = expand_per_round_values(
        args.train_max_function_calls_per_round,
        args.train_max_function_calls,
        "train_max_function_calls_per_round"
    )

    test_max_calls_by_round = expand_per_round_values(
        args.test_max_function_calls_per_round,
        args.test_max_function_calls,
        "test_max_function_calls_per_round"
    )
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    print("\n=== Sequential Function Calling Training ===")
    print(f"Model: {args.model_name}")
    print(f"Training rounds: {len(rounds)}")
    print(f"LoRA: {'Enabled' if args.use_lora else 'Disabled'}")
    if args.use_lora and args.freeze_lora_after_first:
        print("LoRA will be frozen after first round")
    print()
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    
    # Create checkpoint directory if needed
    if args.save_checkpoints:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize model (will be created in first round)
    model = None
    lora_config = None
    
    # Build LoRA config if enabled
    if args.use_lora:
        target_modules = [mod.strip() for mod in args.lora_target_modules.split(',')]
        
        if args.lora_layer_indices is not None:
            layer_indices = [int(idx.strip()) for idx in args.lora_layer_indices.split(',')]
            lora_config = {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "layer_indices": layer_indices,
                "target_modules": target_modules
            }
        else:
            lora_config = {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": target_modules
            }
    
    # Calculate total number of tools across all rounds
    total_tools = len(rounds) * 10  # Assuming each round has same number of tools
    # More robust: parse all rounds to get exact total
    all_tool_ranges = []
    for round_spec in rounds:
        tools_range = round_spec['tools']
        start, end = map(int, tools_range.split('-'))
        all_tool_ranges.append((start, end))
    total_tools = max(end for _, end in all_tool_ranges)
    print(f"Total tools to be learned: {total_tools}")
    
    # Pre-discover actual tool names from all rounds' data files
    all_tool_names = discover_all_tool_names_from_rounds(
        rounds,
        args.data_dir,
        train_max_calls_by_round,
        test_max_calls_by_round,
    )
    
    if len(all_tool_names) != total_tools:
        print(f"Warning: Expected {total_tools} tools but discovered {len(all_tool_names)}")
        print("This might cause issues with tool token mapping")
    
    # Store results for all rounds
    all_results = []
    # Store test dataloaders for cumulative evaluation
    all_test_dataloaders = []
    
    # Training loop for each round
    for round_idx, round_spec in enumerate(rounds):
        round_num = round_idx + 1
        tools_range = round_spec['tools']
        epochs = round_spec['epochs']
        
        print("\n" + "="*60)
        print(f"ROUND {round_num}/{len(rounds)}: Training on tools {tools_range}")
        print("="*60 + "\n")
        
        # Construct data file names based on tool range
        train_data_file = os.path.join(
            args.data_dir,
            f"training/function_calling_train_tools{tools_range}_{train_max_calls_by_round[round_idx]}calls.json"
        )
        test_data_file = os.path.join(
            args.data_dir,
            f"test/function_calling_test_tools{tools_range}_{test_max_calls_by_round[round_idx]}calls.json"
        )
        
        # Check if data files exist
        if not os.path.exists(train_data_file):
            print(f"Warning: Training data file not found: {train_data_file}")
            print(f"Please generate it using: python xlam_datasets.py --top_k {tools_range}")
            continue
        if not os.path.exists(test_data_file):
            print(f"Warning: Test data file not found: {test_data_file}")
            print(f"Please generate it using: python xlam_datasets.py --top_k {tools_range}")
            continue
        
        # Discover tools for this round
        print(f"Discovering tools from round {round_num} dataset...")
        round_tools = discover_available_tools(train_data_file, test_data_file)
        print(f"Found {len(round_tools)} tools for round {round_num}: {round_tools[:5]}..." if len(round_tools) > 5 else f"Found {len(round_tools)} tools: {round_tools}")
        
        # Create or update model
        if model is None:
            # First round - create new model with ALL tool slots and actual tool names
            print(f"Initializing model with {total_tools} total tool slots...")
            model = FunctionCallingModel(
                model_name=args.model_name,
                num_tools=total_tools,  # Initialize with ALL tools
                tool_names=all_tool_names,  # Use actual discovered tool names
                tokenizer=tokenizer,
                device=args.device,
                dtype=dtype,
                decouple_embeddings=args.decouple_embeddings,
                lora_config=lora_config,
            )
            print_model_info(model, f"Model with {total_tools} tool slots")
            print(f"Model initialized with all tool names: {model.tool_names[:5]}...{model.tool_names[-5:]}")
            
            # Apply orthogonal initialization to ALL tool embeddings once at the beginning
            apply_orthogonal_init_all_tools(model, total_tools)
        else:
            # Subsequent rounds - just check LoRA freezing
            if args.use_lora and args.freeze_lora_after_first and round_num == 2:
                print("\nFreezing LoRA parameters after first round...")
                freeze_lora_parameters(model)
        
        print(f"Round {round_num}: Training will focus on tools {tools_range}")
        print(f"Dataset will provide labels only for: {round_tools[:3]}...{round_tools[-3:]}")
        
        # Create datasets for this round
        print(f"Creating round {round_num} dataset...")
        train_dataloader, _, test_dataloader, _, test_examples = create_native_dataloader(
            model=model,
            train_data_path=train_data_file,
            test_data_path=test_data_file,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            eval_batch_size=args.eval_batch_size,
            curriculum_learning=args.curriculum_learning,
            validation_split=0,
            random_seed=args.seed,  # Use consistent seed across all rounds
        )
        
        # Store test dataloader for cumulative evaluation
        all_test_dataloaders.append((tools_range, test_dataloader))
        
        # Determine if LoRA should be trained this round
        train_lora = args.use_lora and (round_num == 1 or not args.freeze_lora_after_first)
        
        # Train this round
        print(f"\nTraining round {round_num} for {epochs} epochs...")
        if train_lora:
            print("Training: Embeddings + LoRA")
        else:
            print("Training: Embeddings only (LoRA frozen)" if args.use_lora else "Training: Embeddings only")
        
        # Determine active tool IDs (freeze others via grad mask in training)
        try:
            active_tool_ids = [model.tool_name_to_id[name] for name in round_tools if name in model.tool_name_to_id]
        except Exception:
            active_tool_ids = None

        round_results = train_native_function_calling_model(
            model, 
            train_dataloader, 
            num_epochs=epochs,
            lr=args.lr,
            lora_lr=args.lora_lr if train_lora else None,
            device=args.device,
            active_tool_ids=active_tool_ids,
            renorm_active_rows=(args.renorm_active_tools and round_num > 2)
        )
        
        print(f"Round {round_num} training completed! Average loss: {round_results['avg_total_loss']:.4f}")
        logger.info(f"\n[ROUND {round_num} RESULTS] Tools: {tools_range}, Epochs: {epochs}, Loss: {round_results['avg_total_loss']:.4f}")
        
        # Store results
        all_results.append({
            'round': round_num,
            'tools': tools_range,
            'epochs': epochs,
            'avg_loss': round_results['avg_total_loss'],
            'results': round_results
        })
        
        # Evaluate this round
        if args.eval_after_each_round:
            print(f"\nEvaluating round {round_num} model...")
            
            # Capture the formatted evaluation output
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Capture stdout to get the formatted evaluation results
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                eval_results = eval_native_function_calling(
                    model, tokenizer, test_dataloader, args.device,
                    use_ground_truth_tools=args.use_ground_truth_tools
                )
            
            # Get the captured formatted output
            formatted_eval_output = captured_output.getvalue()
            
            # Print the formatted output to console as well
            print(formatted_eval_output)
            
            all_results[-1]['eval_results'] = eval_results
            
            # Log the formatted evaluation results to file
            if args.log_file:
                log_file_base = args.log_file if args.log_file.startswith('log/') else f"log/{os.path.basename(args.log_file)}"
                eval_log_file = log_file_base.replace('.log', '_eval_results.log')
            else:
                eval_log_file = default_log_file.replace('.log', '_eval_results.log')
            
            with open(eval_log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"ROUND {round_num} EVALUATION - Tools: {tools_range}, Epochs: {epochs}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"{'='*60}\n")
                f.write(formatted_eval_output)
                f.write(f"\n{'='*60}\n\n")
            
            # Evaluate on all previous test sets if requested
            if args.eval_all_previous and round_num > 1:
                print(f"\nEvaluating on all previous test sets...")
                cumulative_results = {}
                # Evaluate on all previous rounds (current round is at the end, so [:-1] gives us all previous)
                for prev_tools, prev_test_dataloader in all_test_dataloaders[:-1]:
                    print(f"\nEvaluating on tools {prev_tools}...")
                    
                    # Capture the formatted evaluation output
                    captured_output = io.StringIO()
                    with redirect_stdout(captured_output):
                        prev_eval_results = eval_native_function_calling(
                            model, tokenizer, prev_test_dataloader, args.device,
                            use_ground_truth_tools=args.use_ground_truth_tools
                        )
                    
                    # Get the captured formatted output
                    prev_formatted_eval_output = captured_output.getvalue()
                    
                    # Print the formatted output to console as well
                    print(prev_formatted_eval_output)
                    
                    cumulative_results[f"tools_{prev_tools}"] = prev_eval_results
                    
                    # Also log cumulative results to eval file
                    if args.log_file:
                        log_file_base = args.log_file if args.log_file.startswith('log/') else f"log/{os.path.basename(args.log_file)}"
                        eval_log_file = log_file_base.replace('.log', '_eval_results.log')
                    else:
                        eval_log_file = default_log_file.replace('.log', '_eval_results.log')
                    
                    with open(eval_log_file, 'a') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"ROUND {round_num} CUMULATIVE EVALUATION - Previous Tools: {prev_tools}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"{'='*60}\n")
                        f.write(prev_formatted_eval_output)
                        f.write(f"\n{'='*60}\n\n")
                        
                all_results[-1]['cumulative_eval_results'] = cumulative_results
        
        # Save checkpoint if requested
        if args.save_checkpoints:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"round_{round_num}_tools_{tools_range.replace('-', '_')}.pt")
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'round': round_num,
                'tools': round_tools,
                'model_state_dict': model.state_dict(),
                'results': round_results,
            }, checkpoint_path)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"Round {result['round']} (tools {result['tools']}): "
              f"{result['epochs']} epochs, avg loss: {result['avg_loss']:.4f}")
        if 'eval_results' in result and result['eval_results']:
            print(f"  Evaluation accuracy: {result['eval_results'].get('exact_accuracy', 'N/A'):.3f}")
    
    # Run demo on last round if requested
    if args.demo is not None and test_examples is not None:
        print("\n" + "="*60)
        print(f"DEMO: Testing on final round examples")
        print("="*60 + "\n")
        demo_native_function_calling(
            model, tokenizer, test_examples[:args.demo],
            args.device, use_ground_truth_tools=args.use_ground_truth_tools
        )
    
    print("\n" + "="*60)
    print("Sequential training completed!")
    print(f"Trained {len(rounds)} rounds with different tool sets")
    if args.use_lora and args.freeze_lora_after_first:
        print("LoRA was frozen after first round")
    print("="*60)
    
    # Save final results summary
    if args.save_checkpoints:
        summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nTraining summary saved to {summary_path}")
    
    # Log completion
    logger.info(f"=== Sequential Training Completed at {datetime.now()} ===")


if __name__ == "__main__":
    main()
