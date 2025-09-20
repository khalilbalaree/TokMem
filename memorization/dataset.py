import torch
from datasets import load_dataset

def load_gsm8k_data(split="train", max_samples=None):
    """Load GSM8K dataset from HuggingFace"""
    dataset = load_dataset("gsm8k", "main", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset

def load_gsm8k_with_split(train_size=None, val_size=100, seed=42):
    """
    Load GSM8K training data and split it into train and validation sets
    
    Args:
        train_size: Number of training samples (None for all available after reserving val_size)
        val_size: Number of validation samples
        seed: Random seed for reproducible splits
    
    Returns:
        train_dataset, val_dataset: Split datasets
    """
    # Calculate total samples needed
    if train_size is None:
        # Load all available data, then split
        full_dataset = load_gsm8k_data("train", None)
        total_available = len(full_dataset)
        # Reserve val_size for validation, rest for training
        train_size = total_available - val_size
        total_samples_needed = total_available
    else:
        # Load train_size + val_size total samples
        total_samples_needed = train_size + val_size
    
    # Load the full dataset with total samples needed
    full_dataset = load_gsm8k_data("train", total_samples_needed)
    
    # Ensure we have enough data
    actual_total = len(full_dataset)
    if actual_total < val_size + 1:
        raise ValueError(f"Not enough data: need at least {val_size + 1} samples, got {actual_total}")
    
    # If we couldn't load enough, adjust train_size
    if actual_total < total_samples_needed:
        train_size = actual_total - val_size
        print(f"Warning: Only {actual_total} samples available, using {train_size} for training")
    
    # Create train/validation split to get exactly the desired sizes
    split_datasets = full_dataset.train_test_split(
        train_size=train_size,
        test_size=val_size,
        shuffle=True, 
        seed=seed
    )
    
    train_dataset = split_datasets["train"]
    val_dataset = split_datasets["test"]  # Note: train_test_split calls it "test" but it's our validation set
    
    return train_dataset, val_dataset

def create_chat_batch(tokenizer, dataset, batch_size=4, max_length=512, start_index=0, 
                     prompt_position="infix", use_chat_template=True, device="cuda", 
                     system_prompt="You are a helpful assistant."):
    """Create batch of GSM8K QA examples with chat template formatting"""
    
    examples = []
    all_prefix_tokens = []
    all_postfix_tokens = []
    all_assistant_header_tokens = []
    
    for i in range(batch_size):
        data_index = start_index + i
        if data_index >= len(dataset):
            data_index = data_index % len(dataset)
        
        example = dataset[data_index]
        question_text = example["question"]
        answer_text = example["answer"]
        
        if use_chat_template:
            # Manual format for better control and efficiency (similar to util_gsm8k.py)
            # system_text = (
            #     f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            #     f"{system_prompt}<|eot_id|>"
            # )

            system_text = (
                f"<|begin_of_text|>"
            )
            
            
            prefix_text = (
                f"{system_text}<|start_header_id|>user<|end_header_id|>\n\n"
                f"question: {question_text}<|eot_id|>"
            )
            
            assistant_header_text = f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            postfix_text = f"answer: {answer_text}<|eot_id|>"
            
        else:
            # Simple concatenation without chat template
            assistant_header_text = None
            prefix_text = f"<|begin_of_text|>Question: {question_text}"
            assistant_header_text = "\n\n"
            postfix_text = f"Answer: {answer_text}<|end_of_text|>"
        
        examples.append({
            "question": question_text,
            "answer": answer_text,
            "prefix_text": prefix_text,
            "postfix_text": postfix_text,
            "assistant_header_text": assistant_header_text
        })
  
        prefix_tokens = tokenizer.encode(prefix_text, 
                                     truncation=True,
                                     max_length=max_length//2,
                                     add_special_tokens=False,
                                     return_tensors="pt")[0]
        all_prefix_tokens.append(prefix_tokens)

        postfix_tokens = tokenizer.encode(postfix_text,
                                          truncation=True,
                                          max_length=max_length//2,
                                          add_special_tokens=False,
                                          return_tensors="pt")[0]
        all_postfix_tokens.append(postfix_tokens)

        assistant_header_tokens = tokenizer.encode(assistant_header_text,
                                                    add_special_tokens=False,
                                                    return_tensors="pt")[0]
        all_assistant_header_tokens.append(assistant_header_tokens)

    
    # left padding for prefix tokens
    max_user_len = max(len(tokens) for tokens in all_prefix_tokens)
    padded_prefix_tokens = []
    prefix_masks = []
    
    for tokens in all_prefix_tokens:
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        pad_length = max_user_len - len(tokens)
        if pad_length > 0:
            tokens = torch.cat([torch.full((pad_length,), tokenizer.pad_token_id, dtype=tokens.dtype), tokens])
            attention_mask = torch.cat([torch.zeros(pad_length, dtype=torch.long), attention_mask])
        
        padded_prefix_tokens.append(tokens)
        prefix_masks.append(attention_mask)
    
    # right padding for postfix tokens
    max_postfix_len = max(len(tokens) for tokens in all_postfix_tokens)
    padded_postfix_tokens = []
    postfix_masks = []
    
    for tokens in all_postfix_tokens:
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        pad_length = max_postfix_len - len(tokens)
        if pad_length > 0:
            tokens = torch.cat([tokens, torch.full((pad_length,), tokenizer.pad_token_id, dtype=tokens.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        padded_postfix_tokens.append(tokens)
        postfix_masks.append(attention_mask)
    
    
    max_assistant_header_len = max(len(tokens) for tokens in all_assistant_header_tokens)
    assistant_header_tokens = []
    assistant_header_masks = []
    
    for tokens in all_assistant_header_tokens:
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        pad_length = max_assistant_header_len - len(tokens)
        assert pad_length == 0, "Assistant header tokens should be empty"
        
        assistant_header_tokens.append(tokens)
        assistant_header_masks.append(attention_mask)
    

    # Stack into batch tensors
    batch_prefix_tokens = torch.stack(padded_prefix_tokens).to(device)
    batch_prefix_masks = torch.stack(prefix_masks).to(device)
    batch_postfix_tokens = torch.stack(padded_postfix_tokens).to(device)
    batch_postfix_masks = torch.stack(postfix_masks).to(device)
    batch_assistant_header_tokens = torch.stack(assistant_header_tokens).to(device)
    batch_assistant_header_masks = torch.stack(assistant_header_masks).to(device)
    
    what_to_return = {
        "prefix_tokens": batch_prefix_tokens,
        "prefix_masks": batch_prefix_masks,
        "postfix_tokens": batch_postfix_tokens,
        "postfix_masks": batch_postfix_masks,
        "assistant_header_tokens": batch_assistant_header_tokens,
        "assistant_header_masks": batch_assistant_header_masks,
        "examples": examples,
        "prompt_position": prompt_position,
        "use_chat_template": use_chat_template,
    }

    
    return what_to_return

# Main batch creation function
def create_batch(tokenizer, dataset, batch_size=4, max_length=512, start_index=0, 
                prompt_position="infix", use_chat_template=True, device="cuda",
                system_prompt="You are a helpful assistant."):
    """Main batch creation function with all options"""
    return create_chat_batch(tokenizer, dataset, batch_size, max_length, start_index, 
                           prompt_position, use_chat_template, device, system_prompt)

