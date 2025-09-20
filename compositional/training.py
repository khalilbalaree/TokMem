import torch

def train_native_function_calling_model(model, dataloader, num_epochs=3, lr=0.01, 
                                       gradient_accumulation_steps=1, device="cuda", lora_lr=None,
                                       active_tool_ids=None, renorm_active_rows=False):
    """Train the native function calling model using reserved tokens"""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    import torch.nn.functional as F
    
    model.train()
    
    # Set up optimizer with different learning rates for embeddings and LoRA
    if model.lora_config and lora_lr is not None:
        # Separate learning rates for embeddings and LoRA
        embedding_params, lora_params = model.get_trainable_parameters(separate_lora=True)
        
        # IMPORTANT: Disable weight decay for embeddings to prevent drift of unused tool rows
        param_groups = [
            {"params": embedding_params, "lr": lr, "name": "embeddings", "weight_decay": 0.0},
            {"params": lora_params, "lr": lora_lr, "name": "lora", "weight_decay": 0.01}
        ]
        optimizer = AdamW(param_groups)
        print(f"Using separate learning rates: embeddings={lr}, LoRA={lora_lr} (wd: emb=0.0, lora=0.01)")
    else:
        # Only embeddings are trainable
        embedding_params = model.get_trainable_parameters()
        # Disable weight decay for embeddings to avoid shrinking unused tools
        param_groups = [{"params": embedding_params, "lr": lr, "weight_decay": 0.0, "name": "embeddings"}]
        optimizer = AdamW(param_groups)
        print(f"Using single learning rate: {lr} (wd: emb=0.0)")
    
    total_steps = len(dataloader) * num_epochs
    
    # Create linear learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% of steps for warmup
        num_training_steps=total_steps
    )
    
    print(f"Training for {num_epochs} epochs, {len(dataloader)} batches per epoch")
    print(f"Total steps: {total_steps}")
    if model.lora_config and lora_lr is not None:
        print(f"Learning rates: embeddings={lr}, LoRA={lora_lr} (with linear schedule + warmup)")
    else:
        print(f"Learning rate: {lr} (with linear schedule + warmup)")
    print(f"Warmup steps: {total_steps // 10}")
    
    # Get all parameters for reporting
    all_trainable_params = model.get_trainable_parameters()
    total_trainable = sum(p.numel() for p in all_trainable_params)
    
    if model.decouple_embeddings:
        print(f"Training mode: Decoupled embeddings")
        print(f"Trainable parameters: {total_trainable:,} (input: {model.trainable_tool_input_embeddings.numel():,}, output: {model.trainable_tool_output_embeddings.numel():,})")
    else:
        print(f"Training mode: Coupled embeddings")
        print(f"Trainable parameters: {total_trainable:,} (shared: {model.trainable_tool_embeddings.numel():,})")
    print(f"Tool token IDs to monitor: {model.reserved_token_ids}")
    print()
    
    total_loss = 0
    total_tool_loss = 0
    tool_token_count = 0
    tool_loss_batches = 0  # Count of batches that had tool tokens
    step = 0

    batch_loss = 0
    batch_tool_loss = 0
    batch_tool_count = 0
    
    
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
            
            # Calculate tool token loss
            tool_mask = torch.isin(shift_labels, torch.tensor(model.reserved_token_ids, device=device))
            valid_mask = shift_labels != -100
            tool_token_mask = tool_mask & valid_mask
            
            if tool_token_mask.sum() > 0:
                tool_loss = F.cross_entropy(shift_logits[tool_token_mask], shift_labels[tool_token_mask])
                batch_tool_count += tool_token_mask.sum().item()
            else:
                tool_loss = torch.tensor(0.0, device=device)
                batch_tool_count += 0
            batch_tool_loss += tool_loss.item()
            
            # Backward pass
            loss.backward()

            # If we have an active tool mask, zero-out gradients for non-active rows
            if active_tool_ids is not None:
                try:
                    # Build mask once per step
                    total_tools = len(model.reserved_token_ids)
                    device_for_mask = model.trainable_tool_input_embeddings.device
                    active_mask = torch.zeros(total_tools, dtype=torch.bool, device=device_for_mask)
                    active_mask[active_tool_ids] = True

                    # Handle decoupled vs coupled embeddings
                    params_to_mask = []
                    if hasattr(model, 'trainable_tool_input_embeddings') and model.trainable_tool_input_embeddings is not None:
                        params_to_mask.append(model.trainable_tool_input_embeddings)
                    if hasattr(model, 'trainable_tool_output_embeddings') and model.trainable_tool_output_embeddings is not None:
                        # Avoid duplicating the same tensor in coupled mode
                        if model.trainable_tool_output_embeddings is not model.trainable_tool_input_embeddings:
                            params_to_mask.append(model.trainable_tool_output_embeddings)

                    for p in params_to_mask:
                        if p.grad is not None:
                            # p.grad shape: [num_tools, hidden_size]
                            inactive_rows = ~active_mask
                            p.grad[inactive_rows] = 0
                except Exception as e:
                    print(f"Warning: failed to apply active tool grad mask: {e}")
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()

                # Post-step renorm (dot-product mode only): keep active tool output-embedding
                # row norms matched to the mean norm of inactive rows to avoid dominance
                if renorm_active_rows and active_tool_ids is not None:
                    try:
                        total_tools = len(model.reserved_token_ids)
                        device_for_mask = model.trainable_tool_input_embeddings.device
                        active_mask = torch.zeros(total_tools, dtype=torch.bool, device=device_for_mask)
                        active_mask[active_tool_ids] = True
                        inactive_mask = ~active_mask

                        # Choose which parameter to renorm (prefer output embeddings)
                        params_to_norm = []
                        if hasattr(model, 'trainable_tool_output_embeddings') and model.trainable_tool_output_embeddings is not None:
                            params_to_norm.append(model.trainable_tool_output_embeddings)
                        else:
                            # Fallback for coupled mode
                            if hasattr(model, 'trainable_tool_embeddings') and model.trainable_tool_embeddings is not None:
                                params_to_norm.append(model.trainable_tool_embeddings)

                        with torch.no_grad():
                            for p in params_to_norm:
                                # Compute target norm as mean norm of inactive rows (or overall mean if none)
                                if inactive_mask.any():
                                    target_norm = p.data[inactive_mask].norm(dim=1).mean().clamp(min=1e-6)
                                else:
                                    target_norm = p.data.norm(dim=1).mean().clamp(min=1e-6)

                                act_idx = active_mask.nonzero(as_tuple=False).squeeze(-1)
                                if act_idx.numel() > 0:
                                    act_norms = p.data[act_idx].norm(dim=1, keepdim=True).clamp(min=1e-6)
                                    p.data[act_idx] = p.data[act_idx] * (target_norm / act_norms)
                    except Exception as e:
                        print(f"Warning: failed to apply post-step renorm: {e}")

                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            
            # Track tool token loss
            if batch_tool_count > 0:
                total_tool_loss += tool_loss.item()
                tool_token_count += batch_tool_count
                tool_loss_batches += 1

            if (batch_idx + 1) % 10 == 0:
                # Handle learning rate reporting for multiple parameter groups
                if model.lora_config and lora_lr is not None:
                    lr_info = f"LR(emb/lora): {scheduler.get_last_lr()[0]:.6f}/{scheduler.get_last_lr()[1]:.6f}"
                else:
                    lr_info = f"LR: {scheduler.get_last_lr()[0]:.6f}"

                batch_size = len(input_ids)
                avg_loss = batch_loss / batch_size
                avg_tool_loss = batch_tool_loss / batch_size
                avg_tool_count = batch_tool_count / batch_size

                # Show current batch loss only
                if batch_tool_count > 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}, Tool Loss: {avg_tool_loss:.4f}, Tool Tokens: {avg_tool_count}, {lr_info}")    
                else:   
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}, Tool Loss: N/A (no tool tokens), {lr_info}")

                batch_loss = 0
                batch_tool_loss = 0
                batch_tool_count = 0
            
            step += 1
            
        
    
    avg_total_loss = total_loss / (len(dataloader) * num_epochs)
    
    if tool_token_count > 0:
        avg_tool_loss = total_tool_loss / tool_loss_batches  # Average tool loss across batches that had tool tokens
        print(f"\nTraining completed!")
        print(f"Average overall loss: {avg_total_loss:.4f}")
        print(f"Average tool token loss: {avg_tool_loss:.4f}")
        print(f"Total tool tokens processed: {tool_token_count}")
        print(f"Batches with tool tokens: {tool_loss_batches}/{step}")
        print(f"Tool token accuracy insight: Lower tool loss indicates better tool selection performance")
    else:
        print(f"\nTraining completed! Average loss: {avg_total_loss:.4f}")
        print("Warning: No tool tokens found in training data!")
    
    return {
        'avg_total_loss': avg_total_loss
    }


def demo_native_function_calling(model, tokenizer, test_examples, device="cuda", use_ground_truth_tools=False):
    """Demo of native function calling using held-out test examples"""
    
    model.eval()
    mode_desc = "Ground Truth Tool Inference" if use_ground_truth_tools else "Normal Tool Prediction"
    print(f"\n=== Native Function Calling Demo ({mode_desc}) ===")
    print(f"Testing on {len(test_examples)} held-out examples")
    print(f"Available tools: {model.tool_names}")
    print()
    
    for i, example in enumerate(test_examples): 
        user_input = example['user_input']
        expected_tools = example.get('tools', [example.get('tool_name', 'unknown')])
        expected_calls = example.get('function_calls', [example.get('function_call', '{}')])
        
        print(f"=== Test Example {i} ===")
        print(f"User Query: {user_input}")
        print(f"Expected Tool(s): {expected_tools}")
        print(f"Expected Call(s): {expected_calls}")
        print()
        
        # Tokenize user input
        user_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        user_tokens = tokenizer(user_text, return_tensors="pt").to(device)
        
        # Generate with tool prediction or ground truth tools
        if use_ground_truth_tools:
            # Use ground truth tools for inference
            results = model.generate_with_ground_truth_tools(
                user_tokens['input_ids'], 
                user_tokens['attention_mask'], 
                tokenizer,
                ground_truth_tools=expected_tools,
                max_new_tokens=150,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,    
            )
            print(f"Mode: Ground truth tools used ({expected_tools})")
        else:
            # Normal tool prediction
            results = model.generate_with_tool_prediction(
                user_tokens['input_ids'], 
                user_tokens['attention_mask'], 
                tokenizer,
                max_new_tokens=150,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,    
            )
            print(f"Mode: Model predicts tools")
        
        result = results[0]
        
        # Display multi-tool results
        if 'predicted_tools' in result and result['predicted_tools']:
            print(f"Generated {len(result['predicted_tools'])} tool(s):")
            for j, (tool_info, func_call) in enumerate(zip(result['predicted_tools'], result['function_calls'])):
                print(f"  Tool {j+1}: {tool_info['tool_name']}")
                print(f"  Function Call {j+1}: {func_call}")
                
                # Parse function call
                parsed = model.parse_function_call(func_call)
                print(f"  Parsed {j+1}: {parsed}")
                print()
        else:
            # Backward compatibility - single tool display
            print(f"Predicted Tool: {result['predicted_tool_name']}")
            print(f"Tool Token Used: {result.get('tool_token_used', 'N/A')}")
            print(f"Function Call: {result['function_call']}")
            
            # Parse function call
            parsed = model.parse_function_call(result['function_call'])
            print(f"Parsed: {parsed}")
        
        print(f"Full Generated: {result['full_generated_sequence']}")
        print("-" * 50)
        print()

def eval_native_function_calling(model, tokenizer, test_dataloader, device="cuda", use_ground_truth_tools=False):
    """Comprehensive evaluation of native function calling model using batch processing"""
    from eval import compare_function_calls_advanced
    import time
    from collections import defaultdict
    
    model.eval()
    
    # Calculate total examples from dataloader
    total_examples = len(test_dataloader.dataset)
    mode_desc = "Ground Truth Tool Inference" if use_ground_truth_tools else "Normal Tool Prediction"
    print(f"\n=== Native Function Calling Evaluation ({mode_desc}) ===")
    print(f"Evaluating on {total_examples} test examples")
    print()
    
    # Evaluation metrics
    exact_matches = 0
    tool_correct = 0
    f1_scores = []
    precision_scores = []
    recall_scores = []
    tool_f1_scores = []
    tool_precision_scores = []
    tool_recall_scores = []
    parse_errors = 0
    
    # Breakdown by function call count
    call_count_breakdown = defaultdict(lambda: {
        'total': 0, 
        'exact_matches': 0, 
        'tool_correct': 0, 
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'tool_f1_scores': [],
        'tool_precision_scores': [],
        'tool_recall_scores': [],
        'parse_errors': 0
    })
    
    # Evaluation loop with batches
    start_time = time.time()
    print("🔄 Running batch evaluation...")
    
    processed_examples = 0
    for batch_idx, batch in enumerate(test_dataloader):
        batch_size = len(batch['raw_data'])
        processed_examples += batch_size
        
        if batch_idx % 10 == 0 or processed_examples == total_examples:
            print(f"   Progress: {processed_examples}/{total_examples} ({100 * processed_examples / total_examples:.1f}%)")
        
        try:
            # With eval mode dataset, input_ids already contain just user input
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Generate with tool prediction or ground truth tools
            if use_ground_truth_tools:
                # For ground truth tool inference, we need to get ground truth tools for each example
                batch_results = []
                for i in range(batch_size):
                    example = batch['raw_data'][i]
                    expected_tools = example.get('tools', [example.get('tool_name', 'unknown')])
                    
                    # Generate for single example
                    single_input = input_ids[i:i+1]
                    single_mask = attention_mask[i:i+1]
                    
                    single_result = model.generate_with_ground_truth_tools(
                        single_input, 
                        single_mask, 
                        tokenizer,
                        ground_truth_tools=expected_tools,
                        max_new_tokens=256,
                        temperature=0.6,
                        top_p=0.9,
                        do_sample=False,
                    )
                    batch_results.extend(single_result)
            else:
                # Normal tool prediction for batch
                batch_results = model.generate_with_tool_prediction(
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
                expected_tools = example.get('tools', [example.get('tool_name', 'unknown')])
                expected_calls = example.get('function_calls', [example.get('function_call', '{}')])
                expected_call_count = len(expected_calls)
                
                # Extract predicted tools and function calls
                if 'predicted_tools' in result and result['predicted_tools']:
                    predicted_tools = [tool_info['tool_name'] for tool_info in result['predicted_tools']]
                    predicted_calls = result['function_calls']
                else:
                    # Backward compatibility
                    predicted_tools = [result['predicted_tool_name']] if result['predicted_tool_name'] != 'none' else []
                    predicted_calls = [result['function_call']] if result['function_call'] else []
                
                # Evaluate tool prediction accuracy - calculate F1 scores
                from collections import Counter
                from eval import calculate_f1_score
                
                # Calculate tool-level F1 scores
                tool_f1_result = calculate_f1_score(predicted_tools, expected_tools)
                tool_f1_scores.append(tool_f1_result['f1_score'])
                tool_precision_scores.append(tool_f1_result['precision'])
                tool_recall_scores.append(tool_f1_result['recall'])
                
                # Keep binary tool matching for backward compatibility
                tool_match = Counter(predicted_tools) == Counter(expected_tools)
                if tool_match:
                    tool_correct += 1
                
                # Evaluate function call accuracy using improved metric
                eval_result = compare_function_calls_advanced(
                    predicted_calls,
                    expected_calls,
                    ignore_order=True
                )
                
                if eval_result.exact_match:
                    exact_matches += 1
                
                f1_scores.append(eval_result.f1_score)
                precision_scores.append(eval_result.precision)
                recall_scores.append(eval_result.recall)
                
                # Track breakdown by call count
                call_count_breakdown[expected_call_count]['total'] += 1
                if eval_result.exact_match:
                    call_count_breakdown[expected_call_count]['exact_matches'] += 1
                if tool_match:
                    call_count_breakdown[expected_call_count]['tool_correct'] += 1
                call_count_breakdown[expected_call_count]['f1_scores'].append(eval_result.f1_score)
                call_count_breakdown[expected_call_count]['precision_scores'].append(eval_result.precision)
                call_count_breakdown[expected_call_count]['recall_scores'].append(eval_result.recall)
                call_count_breakdown[expected_call_count]['tool_f1_scores'].append(tool_f1_result['f1_score'])
                call_count_breakdown[expected_call_count]['tool_precision_scores'].append(tool_f1_result['precision'])
                call_count_breakdown[expected_call_count]['tool_recall_scores'].append(tool_f1_result['recall'])
                
                # Track parse errors (both global and per call count)
                if 'parse_errors' in eval_result.details:
                    current_parse_errors = eval_result.details['parse_errors']['outputs']
                    parse_errors += current_parse_errors
                    call_count_breakdown[expected_call_count]['parse_errors'] += current_parse_errors
                        
        except Exception as e:
            print(f"   Error processing batch {batch_idx + 1}: {str(e)}")
            # Add zero scores for failed batch
            for i in range(batch_size):
                f1_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                tool_f1_scores.append(0.0)
                tool_precision_scores.append(0.0)
                tool_recall_scores.append(0.0)
            continue
    
    eval_time = time.time() - start_time
    
    # Calculate final metrics
    exact_accuracy = exact_matches / total_examples
    tool_accuracy = tool_correct / total_examples
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_tool_f1_score = sum(tool_f1_scores) / len(tool_f1_scores) if tool_f1_scores else 0.0
    avg_tool_precision = sum(tool_precision_scores) / len(tool_precision_scores) if tool_precision_scores else 0.0
    avg_tool_recall = sum(tool_recall_scores) / len(tool_recall_scores) if tool_recall_scores else 0.0
    parse_error_rate = parse_errors / total_examples
    
    # Print simplified results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"📋 Dataset: {total_examples} examples")
    print(f"⏱️  Evaluation time: {eval_time:.2f} seconds")
    print(f"🔧 Mode: {mode_desc}")
    print()
    
    print("🎯 RESULTS:")
    print(f"   Exact Match Accuracy:     {exact_accuracy:.3f} ({exact_matches}/{total_examples})")
    print(f"   Tool Prediction Accuracy: {tool_accuracy:.3f} ({tool_correct}/{total_examples})")
    print(f"   Average F1 Score:         {avg_f1_score:.3f}")
    print(f"   Average Precision:        {avg_precision:.3f}")
    print(f"   Average Recall:           {avg_recall:.3f}")
    print(f"   Average Tool F1 Score:    {avg_tool_f1_score:.3f}")
    print(f"   Average Tool Precision:   {avg_tool_precision:.3f}")
    print(f"   Average Tool Recall:      {avg_tool_recall:.3f}")
    print(f"   Parse Error Rate:         {parse_error_rate:.3f}")
    print("=" * 50)
    
    # Breakdown by function call count
    print("\n📊 EXACT MATCH ACCURACY:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        accuracy = stats['exact_matches'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"   {call_count} call(s): {accuracy:.3f} ({stats['exact_matches']}/{stats['total']})")
    print("=" * 50)
    
    print("\n📊 TOOL PREDICTION ACCURACY:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        tool_accuracy = stats['tool_correct'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"   {call_count} call(s): {tool_accuracy:.3f} ({stats['tool_correct']}/{stats['total']})")
    print("=" * 50)
    
    print("\n📊 AVERAGE F1 SCORE (Function Calls):")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        avg_f1 = sum(stats['f1_scores']) / len(stats['f1_scores']) if stats['f1_scores'] else 0.0
        avg_prec = sum(stats['precision_scores']) / len(stats['precision_scores']) if stats['precision_scores'] else 0.0
        avg_rec = sum(stats['recall_scores']) / len(stats['recall_scores']) if stats['recall_scores'] else 0.0
        print(f"   {call_count} call(s): F1={avg_f1:.3f}, P={avg_prec:.3f}, R={avg_rec:.3f}")
    print("=" * 50)
    
    print("\n📊 AVERAGE TOOL F1 SCORE:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        avg_tool_f1 = sum(stats['tool_f1_scores']) / len(stats['tool_f1_scores']) if stats['tool_f1_scores'] else 0.0
        avg_tool_prec = sum(stats['tool_precision_scores']) / len(stats['tool_precision_scores']) if stats['tool_precision_scores'] else 0.0
        avg_tool_rec = sum(stats['tool_recall_scores']) / len(stats['tool_recall_scores']) if stats['tool_recall_scores'] else 0.0
        print(f"   {call_count} call(s): Tool F1={avg_tool_f1:.3f}, P={avg_tool_prec:.3f}, R={avg_tool_rec:.3f}")
    print("=" * 50)
    
    print("\n📊 PARSE ERROR RATE:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        error_rate = stats['parse_errors'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"   {call_count} call(s): {error_rate:.3f} ({stats['parse_errors']}/{stats['total']})")
    print("=" * 50)
    
    return {
        'exact_accuracy': exact_accuracy,
        'tool_accuracy': tool_accuracy,
        'avg_f1_score': avg_f1_score,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_tool_f1_score': avg_tool_f1_score,
        'avg_tool_precision': avg_tool_precision,
        'avg_tool_recall': avg_tool_recall,
        'parse_error_rate': parse_error_rate,
        'total_examples': total_examples,
        'call_count_breakdown': dict(call_count_breakdown)
    }
