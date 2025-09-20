import torch
import logging
import os
from datetime import datetime

def setup_logging(log_dir="logs"):
    """Set up logging configuration for training and evaluation"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log = os.path.join(log_dir, f"training_{timestamp}.log")
    evaluation_log = os.path.join(log_dir, f"evaluation_{timestamp}.log")
    
    # Configure training logger
    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging.INFO)
    training_handler = logging.FileHandler(training_log)
    training_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    training_logger.addHandler(training_handler)
    
    # Configure evaluation logger
    eval_logger = logging.getLogger('evaluation')
    eval_logger.setLevel(logging.INFO)
    eval_handler = logging.FileHandler(evaluation_log)
    eval_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    eval_logger.addHandler(eval_handler)
    
    return training_logger, eval_logger, training_log, evaluation_log, timestamp


def extract_trained_token_state(model):
    """Return a minimal state_dict containing only the trainable task token parameters.
    The tensors are cloned and moved to CPU to minimize GPU memory usage.
    """
    state = {}
    if getattr(model, 'decouple_embeddings', False):
        state['trainable_task_input_embeddings'] = (
            model.trainable_task_input_embeddings.detach().cpu().clone()
        )
        state['trainable_task_output_embeddings'] = (
            model.trainable_task_output_embeddings.detach().cpu().clone()
        )
    else:
        state['trainable_task_embeddings'] = (
            model.trainable_task_embeddings.detach().cpu().clone()
        )
    return state


def run_validation(model, val_dataloader, device="cuda", ignore_index=-100):
    """Run validation pass and return average loss.
    Switches model to eval() and restores previous training state when finished.
    """
    import torch
    import torch.nn.functional as F

    was_training = model.training
    model.eval()

    val_loss_total = 0.0
    val_batches = 0
    valid_losses = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            # Check if there are any valid (non-ignored) labels
            valid_mask = shift_labels != ignore_index
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss_total += loss.item()
                    valid_losses += 1
            val_batches += 1

    if valid_losses == 0:
        print(f"Warning: No valid validation losses computed ({val_batches} batches processed)")
        avg_val_loss = float('inf')
    else:
        avg_val_loss = val_loss_total / valid_losses

    if was_training:
        model.train()

    return avg_val_loss

def train_task_calling_model(model, dataloader, val_dataloader=None, num_epochs=3, lr=0.01, 
                           gradient_accumulation_steps=1, device="cuda", timestamp=None,
                           validate_every_n_steps=1000):
    """Train the task calling model using reserved tokens"""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    import torch.nn.functional as F
    
    
    # Get training logger
    training_logger = logging.getLogger('training')
    
    model.train()
    
    # Only train the trainable task embedding parameters 
    trainable_params = model.get_trainable_parameters()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    total_steps = len(dataloader) * num_epochs
    
    # Create linear learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% of steps for warmup
        num_training_steps=total_steps
    )
    
    print(f"Training for {num_epochs} epochs, {len(dataloader)} batches per epoch")
    print(f"Total steps: {total_steps}")
    print(f"Learning rate: {lr} (with linear schedule + warmup)")
    print(f"Warmup steps: {total_steps // 10}")
    if model.decouple_embeddings:
        print(f"Training mode: Decoupled embeddings")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} (input: {model.trainable_task_input_embeddings.numel()}, output: {model.trainable_task_output_embeddings.numel()})")
    else:
        print(f"Training mode: Coupled embeddings")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)} (shared: {model.trainable_task_embeddings.numel()})")
    print(f"Task token IDs to monitor: {model.reserved_token_ids}")
    print()
    
    # Log training configuration
    training_logger.info(f"TRAINING START - Epochs: {num_epochs}, Batches: {len(dataloader)}, Total steps: {total_steps}")
    training_logger.info(f"Config - LR: {lr}, Warmup: {total_steps // 10}, Mode: {'Decoupled' if model.decouple_embeddings else 'Coupled'}")
    training_logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params)}, Task tokens: {model.reserved_token_ids}")
    training_logger.info(f"PyTorch manual seed: {torch.initial_seed()}")
    
    total_loss = 0
    total_task_loss = 0
    task_token_count = 0
    task_loss_batches = 0  # Count of batches that had task tokens
    step = 0

    batch_loss = 0
    batch_task_loss = 0
    batch_task_count = 0
    batches_since_log = 0
    
    # Track best validation loss and model state
    best_val_loss = float('inf')
    best_model_state = None
    best_model_path = None
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            # Shift logits and labels for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Calculate overall loss (ignore -100 labels)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)            
            batch_loss += loss.item()
            
            # Calculate task token loss
            task_mask = torch.isin(shift_labels, model.reserved_token_tensor)
            valid_mask = shift_labels != -100
            task_token_mask = task_mask & valid_mask
            
            if task_token_mask.sum() > 0:
                task_loss = F.cross_entropy(shift_logits[task_token_mask], shift_labels[task_token_mask])
                batch_task_count += task_token_mask.sum().item()
            else:
                task_loss = torch.tensor(0.0)  # No need for device since it's just used for .item()
                batch_task_count += 0
            batch_task_loss += task_loss.item()
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            
            # Track task token loss for global statistics
            if batch_task_count > 0:
                total_task_loss += task_loss.item()
                task_token_count += batch_task_count
                task_loss_batches += 1
            
            # Increment batch counter for logging
            batches_since_log += 1

            if (batch_idx + 1) % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                # current_lr = optimizer.param_groups[0]['lr']

                # Calculate averages over the accumulated batches (100 batches or remaining)
                avg_loss = batch_loss / batches_since_log
                avg_task_loss = batch_task_loss / batches_since_log if batches_since_log > 0 else 0.0
                avg_task_count = batch_task_count / batches_since_log
                
                # Show averages over the logging window
                if batch_task_count > 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Avg Loss: {avg_loss:.4f}, Avg Task Loss: {avg_task_loss:.4f}, Avg Task Tokens: {avg_task_count:.1f}, LR: {current_lr:.6f}")    
                    training_logger.info(f"E{epoch+1}/{num_epochs} B{batch_idx+1}/{len(dataloader)} AvgLoss:{avg_loss:.4f} AvgTaskLoss:{avg_task_loss:.4f} AvgTokens:{avg_task_count:.1f} LR:{current_lr:.6f}")
                else:   
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Avg Loss: {avg_loss:.4f}, Avg Task Loss: N/A (no task tokens in window), LR: {current_lr:.6f}")
                    training_logger.info(f"E{epoch+1}/{num_epochs} B{batch_idx+1}/{len(dataloader)} AvgLoss:{avg_loss:.4f} AvgTaskLoss:N/A LR:{current_lr:.6f}")

                # Reset accumulators for next logging window
                batch_loss = 0
                batch_task_loss = 0
                batch_task_count = 0
                batches_since_log = 0
            
            step += 1

            # Step-based validation
            if val_dataloader is not None and validate_every_n_steps is not None and validate_every_n_steps > 0:
                if step % validate_every_n_steps == 0:
                    avg_val_loss = run_validation(model, val_dataloader, device=device, ignore_index=-100)
                    print(f"Step {step} - Validation Loss: {avg_val_loss:.4f}")
                    training_logger.info(f"VALIDATION STEP {step} Loss:{avg_val_loss:.4f}")
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        # Save best token state for later use
                        best_model_state = extract_trained_token_state(model)
                        best_model_path = save_trained_model(model, timestamp=timestamp, suffix='best')
                        training_logger.info(f"NEW BEST VALIDATION LOSS: {best_val_loss:.4f} | Saved: {best_model_path}")

        # After each epoch, run validation to compute average validation loss
        if val_dataloader is not None:
            avg_val_loss = run_validation(model, val_dataloader, device=device, ignore_index=-100)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")
            training_logger.info(f"VALIDATION E{epoch+1}/{num_epochs} Loss:{avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save best token state for later use
                best_model_state = extract_trained_token_state(model)
                best_model_path = save_trained_model(model, timestamp=timestamp, suffix='best')
                training_logger.info(f"NEW BEST VALIDATION LOSS: {best_val_loss:.4f} | Saved: {best_model_path}")
    
    avg_total_loss = total_loss / (len(dataloader) * num_epochs)
    
    if task_token_count > 0:
        avg_task_loss = total_task_loss / task_loss_batches  # Average task loss across batches that had task tokens
        print(f"\nTraining completed!")
        print(f"Average overall loss: {avg_total_loss:.4f}")
        print(f"Average task token loss: {avg_task_loss:.4f}")
        print(f"Total task tokens processed: {task_token_count}")
        print(f"Batches with task tokens: {task_loss_batches}/{step}")
        print(f"Task token accuracy insight: Lower task loss indicates better task selection performance")
        
        # Log training completion
        training_logger.info(f"TRAINING COMPLETE - AvgLoss:{avg_total_loss:.4f} TaskLoss:{avg_task_loss:.4f} TaskTokens:{task_token_count} TaskBatches:{task_loss_batches}/{step}")
    else:
        print(f"\nTraining completed! Average loss: {avg_total_loss:.4f}")
        print("Warning: No task tokens found in training data!")
        
        # Log training completion without task tokens
        training_logger.info(f"TRAINING COMPLETE - AvgLoss:{avg_total_loss:.4f} WARNING:NoTaskTokens")

    # Report best validation loss if validation was performed
    if val_dataloader is not None and best_val_loss < float('inf'):
        print(f"Best validation loss achieved: {best_val_loss:.4f}")
        if best_model_path:
            print(f"Best model saved to: {best_model_path}")
        training_logger.info(f"BEST VALIDATION LOSS:{best_val_loss:.4f}")
    
    # Return training results including best model state
    return {
        'avg_total_loss': avg_total_loss,
        'best_val_loss': best_val_loss if val_dataloader is not None else None,
        'best_model_state': best_model_state,
        'best_model_path': best_model_path
    }


def save_trained_model(model, save_dir="saved_models", timestamp=None, suffix=None):
    """Save the trained task tokens"""
    import os
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"task_tokens_{timestamp}.pt" if not suffix else f"task_tokens_{timestamp}_{suffix}.pt"
    filepath = os.path.join(save_dir, filename)
    
    model.save_task_tokens(filepath)
    return filepath


def demo_task_calling(model, tokenizer, test_examples, device="cuda", use_ground_truth_tasks=False):
    """Demo of task calling using held-out test examples"""
    
    model.eval()
    mode_desc = "Ground Truth Task Inference" if use_ground_truth_tasks else "Normal Task Prediction"
    print(f"\n=== Task Calling Demo ({mode_desc}) ===")
    print(f"Testing on {len(test_examples)} held-out examples")
    print(f"Available tasks: {model.task_names}")
    print()
    
    for i, example in enumerate(test_examples): 
        instruction = example['instruction']
        expected_tasks = example.get('tasks', ['unknown'])
        expected_responses = example.get('responses', [''])
        
        print(f"=== Test Example {i} ===")
        print(f"Instruction: {instruction}")
        print(f"Expected Task(s): {expected_tasks}")
        print(f"Expected Response(s): {expected_responses}")
        print()
        
        # Tokenize instruction input
        instruction_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        instruction_tokens = tokenizer(instruction_text, return_tensors="pt").to(device)
        
        # Generate with task prediction or ground truth tasks
        if use_ground_truth_tasks:
            # Use ground truth tasks for inference
            results = model.generate_with_ground_truth_tasks(
                instruction_tokens['input_ids'], 
                instruction_tokens['attention_mask'], 
                tokenizer,
                ground_truth_tasks=expected_tasks,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=False,    
            )
            print(f"Mode: Ground truth tasks used ({expected_tasks})")
        else:
            # Normal task prediction
            results = model.generate_with_task_prediction(
                instruction_tokens['input_ids'], 
                instruction_tokens['attention_mask'], 
                tokenizer,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,    
            )
            print(f"Mode: Model predicts tasks")
        
        result = results[0]
        
        # Display multi-task results
        if 'predicted_tasks' in result and result['predicted_tasks']:
            print(f"Generated {len(result['predicted_tasks'])} task(s):")
            for j, (task_info, response) in enumerate(zip(result['predicted_tasks'], result['responses'])):
                print(f"  Task {j+1}: {task_info['task_name']}")
                print(f"  Response {j+1}: {response}")
                print()
        else:
            # Backward compatibility - single task display
            print(f"Predicted Task: {result['predicted_task_name']}")
            print(f"Task Token Used: {result.get('task_token_used', 'N/A')}")
            print(f"Response: {result['response']}")
        
        print(f"Full Generated: {result['full_generated_sequence']}")
        print("-" * 50)
        print()

def eval_task_calling(model, tokenizer, test_dataloader, device="cuda", use_ground_truth_tasks=False):
    """Comprehensive evaluation of task calling model using Natural Instructions metrics"""
    import time
    from collections import defaultdict
    from natural_instructions_eval import evaluate_predictions, print_evaluation_results
    
    # Get evaluation logger
    eval_logger = logging.getLogger('evaluation')
    
    model.eval()
    
    # Calculate total examples from dataloader
    total_examples = len(test_dataloader.dataset)
    mode_desc = "Ground Truth Task Inference" if use_ground_truth_tasks else "Normal Task Prediction"
    print(f"\n=== Task Calling Evaluation ({mode_desc}) ===")
    print(f"Evaluating on {total_examples} test examples")
    print()
    
    # Log evaluation start
    eval_logger.info(f"EVALUATION START - Mode:{mode_desc} Examples:{total_examples}")
    eval_logger.info(f"PyTorch manual seed: {torch.initial_seed()}")
    
    # Collect predictions and references for Natural Instructions evaluation
    all_predictions = []
    all_references = []
    all_task_names = []
    
    # Legacy metrics for compatibility
    task_correct = 0
    task_breakdown = defaultdict(lambda: {'total': 0, 'task_correct': 0})
    
    # Evaluation loop with batches
    start_time = time.time()
    print("🔄 Running batch evaluation...")
    
    processed_examples = 0
    for batch_idx, batch in enumerate(test_dataloader):
        batch_size = len(batch['raw_data'])
        processed_examples += batch_size
        
        if batch_idx % 10 == 0 or processed_examples == total_examples:
            progress_pct = 100 * processed_examples / total_examples
            print(f"   Progress: {processed_examples}/{total_examples} ({progress_pct:.1f}%)")
            eval_logger.info(f"Progress: {processed_examples}/{total_examples} ({progress_pct:.1f}%)")
        
        try:
            # With eval mode dataset, input_ids already contain just instruction input
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Generate with task prediction or ground truth tasks
            if use_ground_truth_tasks:
                # For ground truth task inference, we need to get ground truth tasks for each example
                batch_results = []
                for i in range(batch_size):
                    example = batch['raw_data'][i]
                    expected_tasks = example.get('tasks', ['unknown'])
                    
                    # Generate for single example
                    single_input = input_ids[i:i+1]
                    single_mask = attention_mask[i:i+1]
                    
                    single_result = model.generate_with_ground_truth_tasks(
                        single_input, 
                        single_mask, 
                        tokenizer,
                        ground_truth_tasks=expected_tasks,
                        max_new_tokens=256,
                        temperature=0.6,
                        top_p=0.9,
                        do_sample=False,
                    )
                    batch_results.extend(single_result)
            else:
                # Normal task prediction for batch
                batch_results = model.generate_with_task_prediction(
                    input_ids, 
                    attention_mask, 
                    tokenizer,
                    max_new_tokens=256,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=False,
                )
            
            # Process each example in the batch
            for i in range(batch_size):
                example = batch['raw_data'][i]
                result = batch_results[i]
                
                # Extract expected data
                expected_tasks = example.get('tasks', ['unknown'])
                expected_responses = example.get('responses', [''])
                
                # Extract predicted tasks and responses
                if 'predicted_tasks' in result and result['predicted_tasks']:
                    predicted_tasks = [task_info['task_name'] for task_info in result['predicted_tasks']]
                    predicted_responses = result['responses']
                else:
                    # Backward compatibility
                    predicted_tasks = [result['predicted_task_name']] if result['predicted_task_name'] != 'none' else []
                    predicted_responses = [result['response']] if result['response'] else []
                
                # Evaluate task prediction accuracy (for legacy compatibility)
                task_match = set(predicted_tasks) == set(expected_tasks)
                if task_match:
                    task_correct += 1
                
                # Track task-level accuracy
                for task in expected_tasks:
                    task_breakdown[task]['total'] += 1
                    if task_match:
                        task_breakdown[task]['task_correct'] += 1
                
                # Collect predictions and references for Natural Instructions evaluation
                # Use the first predicted response (or empty string if none)
                pred_text = predicted_responses[0] if predicted_responses else ""
                all_predictions.append(pred_text)
                
                # Natural Instructions supports multiple references
                all_references.append(expected_responses)
                
                # Use the first expected task for per-task breakdown
                task_name = expected_tasks[0] if expected_tasks else "unknown"
                all_task_names.append(task_name)
                        
        except Exception as e:
            print(f"   Error processing batch {batch_idx + 1}: {str(e)}")
            # Add empty predictions for failed batch
            for i in range(batch_size):
                all_predictions.append("")
                all_references.append([""])
                all_task_names.append("error")
            continue
    
    eval_time = time.time() - start_time
    
    # Use Natural Instructions evaluation
    print("\n🔍 Computing Natural Instructions metrics...")
    ni_results = evaluate_predictions(
        predictions=all_predictions,
        references=all_references,
        task_names=all_task_names,
        xlingual=False
    )
    
    # Print Natural Instructions evaluation results
    print_evaluation_results(ni_results, f"NATURAL INSTRUCTIONS EVALUATION ({mode_desc})")
    
    # Legacy task accuracy calculation
    task_accuracy = task_correct / total_examples
    
    print(f"\n🎯 TASK PREDICTION ACCURACY: {task_accuracy:.3f} ({task_correct}/{total_examples})")
    print(f"⏱️  Total evaluation time: {eval_time:.2f} seconds")
    
    # Log evaluation results
    eval_logger.info(f"EVALUATION COMPLETE - TaskAcc:{task_accuracy:.3f} ExactMatch:{ni_results['exact_match']:.1f}% RougeL:{ni_results['rougeL']:.1f}% Time:{eval_time:.1f}s")
    
    # Log per-task performance
    if task_breakdown:
        for task_name in sorted(task_breakdown.keys()):
            stats = task_breakdown[task_name]
            task_acc = stats['task_correct'] / stats['total'] if stats['total'] > 0 else 0.0
            
            # Get NI metrics for this task if available
            ni_metrics = ni_results.get('per_task', {}).get(task_name, {})
            exact_match = ni_metrics.get('exact_match', 0)
            rouge_l = ni_metrics.get('rougeL', 0)
            
            eval_logger.info(f"Task:{task_name} TaskAcc:{task_acc:.3f} ExactMatch:{exact_match:.1f}% RougeL:{rouge_l:.1f}% Examples:{stats['total']}")
    
    # Per-task task accuracy breakdown (console output)
    if task_breakdown:
        print(f"\n📊 TASK PREDICTION ACCURACY BY TASK:")
        print("-" * 60)
        for task_name in sorted(task_breakdown.keys()):
            stats = task_breakdown[task_name]
            task_acc = stats['task_correct'] / stats['total'] if stats['total'] > 0 else 0.0
            print(f"   {task_name}: {task_acc:.3f} ({stats['total']} examples)")
    
    # Return results compatible with existing code
    return {
        'exact_accuracy': ni_results['exact_match'] / 100.0,  # Convert percentage to decimal
        'task_accuracy': task_accuracy,
        'avg_response_score': ni_results['rougeL'] / 100.0,  # Use ROUGE-L as response score
        'total_examples': total_examples,
        'ni_exact_match': ni_results['exact_match'],
        'ni_rouge_l': ni_results['rougeL'],
        'ni_per_task': ni_results.get('per_task', {}),
        'task_breakdown': dict(task_breakdown)
    }