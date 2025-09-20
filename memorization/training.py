import torch
import torch.nn as nn
import torch.optim as optim
import re
from tqdm import tqdm
from dataset import create_batch

def set_seed(seed=42):
    """Fix seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_answer(text):
    """
    Extract numerical answer from GSM8K solution text.
    STRICT MODE: Only accepts answers with the '#### ' format.
    Answers without '#### ' marker are ignored and return None.
    Returns tuple: (cleaned numerical string or None, format_compliant boolean).
    """
    try:
        # Check if format is followed (contains '#### ')
        format_compliant = '#### ' in text
        
        # Look for the LAST occurrence of '#### ' (GSM8K standard) - NO FALLBACK
        parts = text.rsplit('#### ', 1)
        if len(parts) == 2:
            answer_str = parts[1].strip()
            # Extract first number from answer string (handles commas and text after number)
            match = re.search(r'([-+]?[\d,]+(?:\.[\d,]+)?)', answer_str)
            if match:
                answer_str = match.group(1)
                
                # Clean the answer: remove commas, dollar signs, and whitespace
                cleaned_answer = answer_str.replace(',', '').replace('$', '').strip()
                
                # Validate it's actually a number
                try:
                    float(cleaned_answer)  # This will raise ValueError if not a valid number
                    return cleaned_answer, format_compliant
                except ValueError:
                    return None, format_compliant
        
        # No '#### ' found - return None (strict mode)
        return None, format_compliant
            
    except Exception:
        return None, False

def train_qa_model(model, tokenizer, train_dataset, val_dataset, batch_size=4, max_length=512, epochs=3, 
                   lr=0.01, gradient_accumulation_steps=1, validation_steps=1000, device="cuda", dtype=torch.bfloat16,
                   system_prompt="You are a helpful assistant."):
    """Train QA model using pre-split train and validation datasets with chat template and flexible prompt positioning"""
    
    print(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation")
    
    optimizer = optim.AdamW([model.prompt_embeddings], lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    losses = []
    best_accuracy = 0.0
    best_loss = float('inf')
    best_epoch = 0
    global_step = 0
    best_model_state = None  # Store best model state in memory
    
    # Calculate number of batches per epoch using training dataset
    num_batches = (len(train_dataset) + batch_size - 1) // batch_size
    
    # Adjust learning rate for gradient accumulation
    effective_lr = lr / gradient_accumulation_steps
    for param_group in optimizer.param_groups:
        param_group['lr'] = effective_lr
    
    # Training loop with progress bars
    for epoch in tqdm(range(epochs), desc="Epochs", position=0):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        # Shuffle training dataset for each epoch (improves training dynamics)
        import random
        train_indices = list(range(len(train_dataset)))
        random.shuffle(train_indices)
        # Use shuffled indices to access the training dataset
        
        # Progress bar for batches within each epoch
        batch_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", 
                         position=1, leave=False)
        
        # Iterate through all batches in the dataset
        for batch_idx in batch_pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_dataset))
            current_batch_size = end_idx - start_idx
            
            # Get shuffled indices for this batch
            batch_indices = train_indices[start_idx:end_idx]
            batch_dataset = train_dataset.select(batch_indices)
            
            # Create batch
            chat_data = create_batch(
                tokenizer, batch_dataset, current_batch_size, max_length, 
                0, model.prompt_position, model.use_chat_template, device, system_prompt  # start_idx=0 since we're using selected dataset
            )
            
            prefix_tokens = chat_data["prefix_tokens"]
            prefix_masks = chat_data["prefix_masks"]
            postfix_tokens = chat_data["postfix_tokens"]
            postfix_masks = chat_data["postfix_masks"]
            assistant_header_tokens = chat_data["assistant_header_tokens"]
            assistant_header_masks = chat_data["assistant_header_masks"]
            
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(postfix_tokens, postfix_mask=postfix_masks, 
                             prefix_tokens=prefix_tokens, prefix_mask=prefix_masks,
                             assistant_header_tokens=assistant_header_tokens, assistant_header_mask=assistant_header_masks)
                
                # Calculate loss only on assistant tokens (after user and prompt)
                if model.prompt_position == "prefix":
                    # [P] [User] [Assistant]
                    prefix_length = model.prompt_length + prefix_tokens.shape[1] + assistant_header_tokens.shape[1]
                else:  # infix
                    # [User] [P] [Assistant]
                    prefix_length = prefix_tokens.shape[1] + model.prompt_length + assistant_header_tokens.shape[1]
                
                shift_logits = logits[..., prefix_length:-1, :].contiguous()
                shift_labels = postfix_tokens[..., 1:].contiguous()
                shift_attention = postfix_masks[..., 1:].contiguous()
                
                # Ignore padding tokens
                shift_labels = shift_labels.clone()
                shift_labels[shift_attention == 0] = -100
                
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
            
            loss.backward()
            epoch_loss += loss.item()
            batch_count += 1
            
            # Update batch progress bar with current loss
            current_loss = (epoch_loss * gradient_accumulation_steps) / batch_count
            batch_pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
            
            # Update parameters after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Run validation every validation_steps
                if global_step % validation_steps == 0:
                    tqdm.write(f"\nRunning validation at step {global_step}...")
                    val_loss, val_accuracy, val_format_rate = evaluate_on_dataset_subset(model, tokenizer, val_dataset, batch_size, max_length, device, dtype, system_prompt)
                    tqdm.write(f"  Step {global_step} Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Format: {val_format_rate:.4f}")
                    
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_loss = val_loss
                        best_epoch = epoch
                        # Save the current best model state
                        best_model_state = model.prompt_embeddings.clone().detach()
                        tqdm.write(f"  New best accuracy: {best_accuracy:.4f}")
                    
                    model.train()  # Return to training mode
        
        # Close batch progress bar
        batch_pbar.close()
        
        # Average loss over all batches
        avg_loss = (epoch_loss * gradient_accumulation_steps) / batch_count
        losses.append(avg_loss)
        
        # Update epoch progress bar
        tqdm.write(f"Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}, Global Step: {global_step}")
        
        # Final validation at end of epoch if no validation happened during epoch
        if global_step % validation_steps != 0:
            tqdm.write(f"\nRunning end-of-epoch validation...")
            val_loss, val_accuracy, val_format_rate = evaluate_on_dataset_subset(model, tokenizer, val_dataset, batch_size, max_length, device, dtype, system_prompt)
            tqdm.write(f"  Epoch {epoch+1} End Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Format: {val_format_rate:.4f}")
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_loss = val_loss
                best_epoch = epoch
                # Save the current best model state
                best_model_state = model.prompt_embeddings.clone().detach()
                tqdm.write(f"  New best accuracy: {best_accuracy:.4f}")
            
            # Early stopping
            if val_accuracy > 0.95:
                tqdm.write(f"  Early stopping at epoch {epoch+1} (accuracy > 0.95)")
                break
    
    # Restore the best model state for testing
    if best_model_state is not None:
        model.prompt_embeddings.data = best_model_state
        tqdm.write(f"Restored model to best validation state (accuracy: {best_accuracy:.4f}, epoch: {best_epoch + 1})")
    
    return losses, best_loss, best_accuracy, best_epoch

def evaluate_on_dataset_subset(model, tokenizer, dataset, batch_size, max_length, device="cuda", dtype=torch.bfloat16,
                              system_prompt="You are a helpful assistant that solves math word problems step by step."):
    """Evaluate model on a subset of dataset for validation during training using actual accuracy"""
    model.eval()
    total_correct = 0
    total_examples = 0
    total_loss = 0.0
    total_batches = 0
    total_format_compliant = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        val_pbar = tqdm(range(num_batches), desc="Validation", leave=False)
        for batch_idx in val_pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            current_batch_size = end_idx - start_idx
            
            # Create batch
            chat_data = create_batch(
                tokenizer, dataset, current_batch_size, max_length, 
                start_idx, model.prompt_position, model.use_chat_template, device, system_prompt
            )
            
            prefix_tokens = chat_data["prefix_tokens"]
            prefix_masks = chat_data["prefix_masks"]
            postfix_tokens = chat_data["postfix_tokens"]
            postfix_masks = chat_data["postfix_masks"]
            assistant_header_tokens = chat_data["assistant_header_tokens"]
            assistant_header_masks = chat_data["assistant_header_masks"]
            
            # Calculate loss for monitoring
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(postfix_tokens, postfix_mask=postfix_masks,
                             prefix_tokens=prefix_tokens, prefix_mask=prefix_masks,
                             assistant_header_tokens=assistant_header_tokens, assistant_header_mask=assistant_header_masks)
                
                if model.prompt_position == "prefix":
                    prefix_length = model.prompt_length + prefix_tokens.shape[1] + assistant_header_tokens.shape[1]
                else:  # infix
                    prefix_length = prefix_tokens.shape[1] + model.prompt_length + assistant_header_tokens.shape[1]
                
                shift_logits = logits[..., prefix_length:-1, :].contiguous()
                shift_labels = postfix_tokens[..., 1:].contiguous()
                shift_attention = postfix_masks[..., 1:].contiguous()
                
                shift_labels = shift_labels.clone()
                shift_labels[shift_attention == 0] = -100
                
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
                
                total_loss += loss.item()
                total_batches += 1
            
            # Generate predictions for actual accuracy
            predictions = model.generate_answers(
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_masks,
                tokenizer=tokenizer,
                max_new_tokens=256,
                assistant_header_tokens=assistant_header_tokens,
                assistant_header_mask=assistant_header_masks
            )
            
            # Calculate accuracy and format compliance
            for i in range(current_batch_size):
                predicted_answer, pred_format_ok = extract_answer(predictions[i])
                ground_truth_answer, _ = extract_answer(chat_data["examples"][i]["answer"])
                
                # Count format compliance
                if pred_format_ok:
                    total_format_compliant += 1
                
                # Only count as correct if both extractions succeeded and match
                if (predicted_answer is not None and ground_truth_answer is not None and 
                    predicted_answer == ground_truth_answer):
                    total_correct += 1
                total_examples += 1
            
            # Update validation progress
            current_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
            current_format_rate = total_format_compliant / total_examples if total_examples > 0 else 0.0
            current_avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
            val_pbar.set_postfix({'Acc': f'{current_accuracy:.3f}', 'Fmt': f'{current_format_rate:.3f}', 'Loss': f'{current_avg_loss:.4f}'})
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    format_rate = total_format_compliant / total_examples if total_examples > 0 else 0.0
    
    return avg_loss, accuracy, format_rate


def evaluate_on_test_set(model, tokenizer, test_dataset, batch_size=4, max_length=512, device="cuda",
                        system_prompt="You are a helpful assistant."):
    """Evaluate model on test set with answer accuracy and format compliance"""
    model.eval()
    total_correct = 0
    total_examples = 0
    total_format_compliant = 0
    
    num_batches = (len(test_dataset) + batch_size - 1) // batch_size
    
    # Progress bar for test evaluation
    test_pbar = tqdm(range(num_batches), desc="Evaluating on test set")
    
    for batch_idx in test_pbar:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_dataset))
        current_batch_size = end_idx - start_idx
        
        # Create batch
        test_batch = create_batch(tokenizer, test_dataset, current_batch_size, max_length, 
                                start_idx, model.prompt_position, model.use_chat_template, device, system_prompt)
        
        # Generate predictions
        predictions = model.generate_answers(
            prefix_tokens=test_batch["prefix_tokens"],
            prefix_mask=test_batch["prefix_masks"],
            tokenizer=tokenizer,
            max_new_tokens=256,
            assistant_header_tokens=test_batch["assistant_header_tokens"],
            assistant_header_mask=test_batch["assistant_header_masks"]
        )
        
        # Calculate accuracy and format compliance
        for i in range(current_batch_size):
            predicted_answer, pred_format_ok = extract_answer(predictions[i])
            ground_truth_answer, _ = extract_answer(test_batch["examples"][i]["answer"])
            
            # Count format compliance
            if pred_format_ok:
                total_format_compliant += 1
            
            # Debug: Print first few predictions to understand the issue
            if total_examples < 3:
                print(f"Example {total_examples + 1}:")
                print(f"  Generated: {predictions[i]}")
                print(f"  Predicted answer: {predicted_answer}")
                print(f"  Ground truth answer: {ground_truth_answer}")
                print(f"  Format compliant: {pred_format_ok}")
                print(f"  Match: {predicted_answer == ground_truth_answer if predicted_answer is not None and ground_truth_answer is not None else False}")
                print()
            
            # Only count as correct if both extractions succeeded and match
            if (predicted_answer is not None and ground_truth_answer is not None and 
                predicted_answer == ground_truth_answer):
                total_correct += 1
            total_examples += 1
        
        # Update progress bar with current accuracy and format compliance
        current_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        current_format_rate = total_format_compliant / total_examples if total_examples > 0 else 0.0
        test_pbar.set_postfix({'Acc': f'{current_accuracy:.3f}', 
                              'Fmt': f'{current_format_rate:.3f}',
                              'Correct': f'{total_correct}/{total_examples}'})
    
    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    format_rate = total_format_compliant / total_examples if total_examples > 0 else 0.0
    return accuracy, total_correct, total_examples, format_rate, total_format_compliant