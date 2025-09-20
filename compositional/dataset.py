import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import os
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer

def discover_available_tools(train_data_path="function_calling_train.json", test_data_path="function_calling_test.json"):   
    """Discover available tool names from split train/test files"""

    train_file = train_data_path
    test_file = test_data_path
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        try:
            # Load both train and test files
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            # Extract tools from both datasets
            train_tools = set()
            test_tools = set()
            
            for sample in train_data:
                tools = sample.get('tools', [])
                train_tools.update(tools)
                
            for sample in test_data:
                tools = sample.get('tools', [])
                test_tools.update(tools)
            
            # Verify train and test have the same tools
            if train_tools != test_tools:
                print(f"Warning: Train tools ({len(train_tools)}) != Test tools ({len(test_tools)})")
                print(f"  Train only: {train_tools - test_tools}")
                print(f"  Test only: {test_tools - train_tools}")
            else:
                print(f"✅ Train and test datasets have the same {len(train_tools)} tools")
            
            # Return union of all tools
            all_tools = train_tools.union(test_tools)
            available_tools = sorted(list(all_tools))
            print(f"Discovered {len(available_tools)} available tools: {available_tools}")
            return available_tools
            
        except Exception as e:
            print(f"Error reading split files: {e}")
    
    # If no files found, return empty list instead of None
    print("Warning: No function calling data files found. Returning empty tool list.")
    return []
   
class NativeFunctionCallingDataset(Dataset):
    """Dataset for function calling using native reserved special tokens as tool tokens"""
    
    def __init__(self, data_path=None, tokenizer=None, max_length=512, model=None, mode="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = model  # Need model for reserved token mappings
        self.mode = mode  # "train" or "eval"
        
        if data_path:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            # Data will be set later (for validation splits)
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # sys_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful function calling assistant.<|eot_id|>"
        
        # Create sequence: [User] [Reserved_Tool_Token1] [Function_Call1] [Reserved_Tool_Token2] [Function_Call2] ... <|eot_id|>
        user_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{item['user_input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"     
        
        # Tokenize user input
        user_tokens = self.tokenizer(user_text, add_special_tokens=False)['input_ids']
        
        # If eval mode, return just the user input
        if self.mode == "eval":
            input_ids = torch.tensor(user_tokens, dtype=torch.long)
            attention_mask = torch.ones(len(user_tokens), dtype=torch.long)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor([-100] * len(user_tokens), dtype=torch.long),  # Not used in eval
                'user_length': len(user_tokens),
                'raw_data': item
            }
        
        # Training mode: build full sequence with tools and function calls
        full_sequence = user_tokens.copy()
        labels = [-100] * len(user_tokens)  # Ignore user tokens in loss
        
        # Use multi-tool format only
        if 'tools' not in item or 'function_calls' not in item:
            raise ValueError("Data must contain 'tools' and 'function_calls' fields")
            
        tools = item['tools'] if isinstance(item['tools'], list) else [item['tools']]
        function_calls = item['function_calls'] if isinstance(item['function_calls'], list) else [item['function_calls']]
        
        # Ensure tools and function_calls have the same length
        if len(tools) != len(function_calls):
            raise ValueError(f"Mismatch between tools ({len(tools)}) and function_calls ({len(function_calls)})")
        
        for tool_name, function_call in zip(tools, function_calls):
            # Get reserved token ID for tool (from model)
            tool_token_id = self.model.get_tool_token_id(tool_name)
            if tool_token_id is None:
                raise ValueError(f"Tool '{tool_name}' not found in model tool mappings")
            
            # Tokenize function call
            function_call_tokens = self.tokenizer(function_call, add_special_tokens=False)['input_ids']
            
            # Add tool token and function call to sequence
            full_sequence.append(tool_token_id)
            full_sequence.extend(function_call_tokens)
            
            # Add to labels (learn to predict tool token and function call tokens)
            labels.append(tool_token_id)
            labels.extend(function_call_tokens)
        
        # Add end-of-turn token
        eot_token = self.tokenizer('<|eot_id|>', add_special_tokens=False)['input_ids']
        full_sequence.extend(eot_token)
        labels.extend(eot_token)
        
        # Create attention mask
        attention_mask = [1] * len(full_sequence)
        
        # Convert to tensors
        input_ids = torch.tensor(full_sequence, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'user_length': len(user_tokens),
            'raw_data': item
        }

def collate_fn(batch, tokenizer):
    """Custom collate function for batching variable-length sequences"""
    
    def pad_sequence(sequences, padding_value=0):
        max_len = max(seq.size(0) for seq in sequences)
        padded = torch.full((len(sequences), max_len), padding_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            # Left padding: place sequence at the end
            start_idx = max_len - seq.size(0)
            padded[i, start_idx:] = seq
        return padded
    
    # Pad sequences with proper token IDs
    input_ids = pad_sequence([item['input_ids'] for item in batch], padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], padding_value=-100)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'user_lengths': torch.tensor([item['user_length'] for item in batch]),
        'raw_data': [item['raw_data'] for item in batch]
    }

def create_native_dataloader(model, train_data_path=None, test_data_path=None, tokenizer=None, 
                           batch_size=4, max_length=512, eval_batch_size=32, curriculum_learning=False,
                           validation_split=0.05, random_seed=42):
    """Create separate DataLoaders for training, validation and testing using pre-split data files
    
    Args:
        model: Model instance with tool token mappings
        train_data_path: Path to training data JSON file
        test_data_path: Path to test data JSON file  
        tokenizer_name: Name of tokenizer to use
        batch_size: Batch size for training dataloader
        max_length: Maximum sequence length
        eval_batch_size: Batch size for test dataloader (for efficient evaluation)
        curriculum_learning: If True, sort training data by number of function calls (ascending)
        validation_split: Ratio of training data to use for validation (default: 0.05)
        random_seed: Random seed for validation split (default: 42)
    
    Returns:
        train_dataloader: Training DataLoader
        val_dataloader: Validation DataLoader
        test_dataloader: Test DataLoader with configurable batch size for evaluation
        tokenizer: Tokenizer used
        test_examples: List of raw test examples for demo
    """
    
    # Create training dataset and dataloader
    train_dataloader = None
    val_dataloader = None
    if train_data_path and os.path.exists(train_data_path):
        # Load full training data
        full_train_dataset = NativeFunctionCallingDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            model=model,
        )
        
        # Split training data for validation
        if validation_split > 0:
            # Set random seed for reproducible splits
            random.seed(random_seed)
            
            # Shuffle data indices
            full_data = full_train_dataset.data.copy()
            random.shuffle(full_data)
            
            # Calculate split point
            total_samples = len(full_data)
            val_samples = int(total_samples * validation_split)
            train_samples = total_samples - val_samples
            
            # Split data
            val_data = full_data[:val_samples]
            train_data = full_data[val_samples:]
            
            # Create separate datasets
            train_dataset = NativeFunctionCallingDataset(
                data_path=None,  # Will set data directly
                tokenizer=tokenizer,
                max_length=max_length,
                model=model,
                mode="train"
            )
            train_dataset.data = train_data
            
            val_dataset = NativeFunctionCallingDataset(
                data_path=None,  # Will set data directly
                tokenizer=tokenizer,
                max_length=max_length,
                model=model,
                mode="train"
            )
            val_dataset.data = val_data
            
            print(f"Dataset split: {train_samples} training, {val_samples} validation ({validation_split:.1%} split)")
        else:
            train_dataset = full_train_dataset
            val_dataset = None
            print(f"No validation split: Using full training dataset ({len(train_dataset)} samples)")
        
        # Apply curriculum learning if requested (only on training data)
        if curriculum_learning:
            def get_num_function_calls(item):
                """Get number of function calls from a data item"""
                if 'function_calls' not in item:
                    raise ValueError("Data must contain 'function_calls' field for curriculum learning")
                return len(item['function_calls']) if isinstance(item['function_calls'], list) else 1
            
            # Sort training data by number of function calls (ascending)
            train_dataset.data.sort(key=get_num_function_calls)
            
            # Print curriculum learning statistics
            num_calls_dist = {}
            for item in train_dataset.data:
                num_calls = get_num_function_calls(item)
                num_calls_dist[num_calls] = num_calls_dist.get(num_calls, 0) + 1
            
            print(f"✅ Curriculum learning enabled: Training data sorted by function call complexity")
            print(f"   Distribution: {dict(sorted(num_calls_dist.items()))}")
        
        # Create training dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False if curriculum_learning else True,  # Don't shuffle if using curriculum learning
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )
        
        # Create validation dataloader if validation data exists
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=eval_batch_size,  # Use eval batch size for validation
                shuffle=False,  # Don't shuffle validation data
                collate_fn=lambda batch: collate_fn(batch, tokenizer)
            )
        
        print(f"Training dataset loaded: {len(train_dataset)} samples from {train_data_path}")
        if val_dataset is not None:
            print(f"Validation dataset created: {len(val_dataset)} samples")
    else:
        print(f"Warning: Training data path '{train_data_path}' not found or not provided")
    
    # Create test dataset and dataloader
    test_dataloader = None
    test_examples = []
    if test_data_path and os.path.exists(test_data_path):
        test_dataset = NativeFunctionCallingDataset(
            data_path=test_data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            model=model,
            mode="eval"  # Use eval mode for test dataset
        )
        
        # Use larger batch size for test dataloader for efficient batch evaluation
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,  # Use specified eval batch size
            shuffle=False,  # Don't shuffle test data for consistent evaluation
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )
        
        # Keep raw test examples for demo purposes
        test_examples = test_dataset.data.copy()
        
        print(f"Test dataset loaded: {len(test_dataset)} samples from {test_data_path}")
    else:
        print(f"Warning: Test data path '{test_data_path}' not found or not provided")
    
    return train_dataloader, val_dataloader, test_dataloader, tokenizer, test_examples 