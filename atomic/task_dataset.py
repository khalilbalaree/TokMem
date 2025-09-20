import torch
from torch.utils.data import Dataset, DataLoader
import json
import hashlib
import random
import os
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer

def load_natural_instructions_from_splits(tasks_dir="natural-instructions-2.8/tasks", 
                                          train_tasks_file="natural-instructions-2.8/splits/default/train_tasks.txt",
                                          test_tasks_file="natural-instructions-2.8/splits/default/test_tasks.txt",
                                          val_ratio=0.1, max_instances_per_task=500, max_instances_per_task_test=100, max_instruction_tokens=1024, tokenizer=None):
    """Load tasks from Natural Instructions dataset using predefined train/test splits
    
    Args:
        tasks_dir: Directory containing Natural Instructions task files
        train_tasks_file: File containing training task names (one per line)
        test_tasks_file: File containing test task names (one per line)
        val_ratio: Ratio of training data to use for validation (default 0.1)
        max_instances_per_task: Maximum instances per task for balance (default 500)    
        max_instances_per_task_test: Maximum instances per task for balance (default 100)
        max_instruction_tokens: Maximum token length for instructions (default 1000)
        tokenizer: Tokenizer to use for token length filtering (required if max_instruction_tokens is used)
        
    Returns:
        train_data, val_data, test_data: Lists of formatted samples
        num_train_tasks: Number of unique training tasks successfully processed
        num_test_tasks: Number of unique test tasks successfully processed
    """
    if not os.path.exists(tasks_dir):
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    
    if not os.path.exists(train_tasks_file):
        raise FileNotFoundError(f"Train tasks file not found: {train_tasks_file}")
        
    if not os.path.exists(test_tasks_file):
        raise FileNotFoundError(f"Test tasks file not found: {test_tasks_file}")
    
    # Read train and test task names
    with open(train_tasks_file, 'r') as f:
        train_task_names = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(test_tasks_file, 'r') as f:
        test_task_names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📋 Loading {len(train_task_names)} training tasks and {len(test_task_names)} test tasks from splits")
    
    # Process training tasks
    all_train_data = []
    all_val_data = []
    train_filtered = 0
    
    print(f"\n🔄 Processing training tasks...")
    for i, task_name in enumerate(train_task_names):
        task_file = f"{task_name}.json"
        task_path = os.path.join(tasks_dir, task_file)
        
        if not os.path.exists(task_path):
            print(f"Warning: Task file not found: {task_file}, skipping")
            continue
            
        try:
            with open(task_path, 'r') as f:
                task_data = json.load(f)
            
            # Check if task is English-only
            input_lang = task_data.get('Input_language', [])
            output_lang = task_data.get('Output_language', [])
            
            if not (('English' in input_lang or input_lang == ['English']) and 
                   ('English' in output_lang or output_lang == ['English'])):
                print(f"Skipping non-English task: {task_name}")
                continue
            
            # Extract task information
            definition = ' '.join(task_data.get('Definition', []))
            instances = task_data.get('Instances', [])
            
            if not instances:
                print(f"Warning: No instances found in {task_file}, skipping")
                continue
            
            if i % 50 == 0:  # Progress indicator
                print(f"  Processed {i}/{len(train_task_names)} training tasks...")
            
            # Pre-randomly select instances (more efficient than processing all)
            pre_select_size = min(len(instances), max_instances_per_task * 2)  # Select 3x the final target to account for filtering
            if len(instances) > pre_select_size:
                instances = random.sample(instances, pre_select_size)
            
            # Shuffle instances
            random.shuffle(instances)
            
            # Filter instances by instruction token length first
            def is_instruction_valid(instruction):
                """Check if instruction is within token limit"""
                if tokenizer is None:
                    return True  # No filtering if no tokenizer provided
                token_count = len(tokenizer.encode(instruction, add_special_tokens=False))
                return token_count <= max_instruction_tokens
            
            # Filter all instances upfront
            filtered_instances = []
            task_filtered = 0
            
            for instance in instances:
                instruction = f"{definition}\n\n{instance['input']}"
                if is_instruction_valid(instruction):
                    filtered_instances.append(instance)
                else:
                    task_filtered += 1
            
            # Cap each task at max_instances_per_task for balance after filtering
            if len(filtered_instances) > max_instances_per_task:
                filtered_instances = filtered_instances[:max_instances_per_task]
            
            # Split training instances into train/val
            val_size = int(len(filtered_instances) * val_ratio)
            train_instances = filtered_instances[val_size:]
            val_instances = filtered_instances[:val_size]
            
            # Process training instances
            for instance in train_instances:
                instruction = f"{definition}\n\n{instance['input']}"
                sample = {
                    'instruction': definition,
                    'query': instance['input'],
                    'tasks': [task_name],
                    'responses': [instance['output'][0] if isinstance(instance['output'], list) else instance['output']]
                }
                all_train_data.append(sample)
            
            # Process validation instances
            for instance in val_instances:
                instruction = f"{definition}\n\n{instance['input']}"
                sample = {
                    'instruction': definition,
                    'query': instance['input'],
                    'tasks': [task_name],
                    'responses': [instance['output'][0] if isinstance(instance['output'], list) else instance['output']]
                }
                all_val_data.append(sample)
            
            train_filtered += task_filtered
            
        except Exception as e:
            print(f"Error processing {task_file}: {e}")
            continue
    
    # Process test tasks
    all_test_data = []
    test_filtered = 0
    
    print(f"\n🔄 Processing test tasks...")
    for i, task_name in enumerate(test_task_names):
        task_file = f"{task_name}.json"
        task_path = os.path.join(tasks_dir, task_file)
        
        if not os.path.exists(task_path):
            print(f"Warning: Task file not found: {task_file}, skipping")
            continue
            
        try:
            with open(task_path, 'r') as f:
                task_data = json.load(f)
            
            # Check if task is English-only
            input_lang = task_data.get('Input_language', [])
            output_lang = task_data.get('Output_language', [])
            
            if not (('English' in input_lang or input_lang == ['English']) and 
                   ('English' in output_lang or output_lang == ['English'])):
                print(f"Skipping non-English task: {task_name}")
                continue
            
            # Extract task information
            definition = ' '.join(task_data.get('Definition', []))
            instances = task_data.get('Instances', [])
            
            if not instances:
                print(f"Warning: No instances found in {task_file}, skipping")
                continue
            
            if i % 20 == 0:  # Progress indicator
                print(f"  Processed {i}/{len(test_task_names)} test tasks...")
            
            # Pre-randomly select instances (more efficient than processing all)
            pre_select_size = min(len(instances), max_instances_per_task_test * 2)  # Select 3x the final target to account for filtering
            if len(instances) > pre_select_size:
                instances = random.sample(instances, pre_select_size)
            
            # Shuffle instances
            random.shuffle(instances)
            
            # Filter instances by instruction token length first
            def is_instruction_valid(instruction):
                """Check if instruction is within token limit"""
                if tokenizer is None:
                    return True  # No filtering if no tokenizer provided
                token_count = len(tokenizer.encode(instruction, add_special_tokens=False))
                return token_count <= max_instruction_tokens
            
            # Filter all instances upfront
            filtered_instances = []
            task_filtered = 0
            
            for instance in instances:
                instruction = f"{definition}\n\n{instance['input']}"
                if is_instruction_valid(instruction):
                    filtered_instances.append(instance)
                else:
                    task_filtered += 1
            
            # Cap each task at max_instances_per_task for balance after filtering
            if len(filtered_instances) > max_instances_per_task_test:
                filtered_instances = filtered_instances[:max_instances_per_task_test]
            
            # Process all test instances
            for instance in filtered_instances:
                instruction = f"{definition}\n\n{instance['input']}"
                sample = {
                    'instruction': definition,
                    'query': instance['input'],
                    'tasks': [task_name],
                    'responses': [instance['output'][0] if isinstance(instance['output'], list) else instance['output']]
                }
                all_test_data.append(sample)
            
            test_filtered += task_filtered
            
        except Exception as e:
            print(f"Error processing {task_file}: {e}")
            continue
    
    if train_filtered > 0:
        print(f"🔧 Filtered out {train_filtered} training samples with instructions longer than {max_instruction_tokens} tokens")
    if test_filtered > 0:
        print(f"🔧 Filtered out {test_filtered} test samples with instructions longer than {max_instruction_tokens} tokens")
    
    return all_train_data, all_val_data, all_test_data


def count_training_tasks(train_tasks_file="natural-instructions-2.8/splits/default/train_tasks.txt"):
    """Count the number of tasks in the training split file
    
    Args:
        train_tasks_file: File containing training task names (one per line)
        
    Returns:
        int: Number of training tasks in the file
    """
    if not os.path.exists(train_tasks_file):
        print(f"Warning: Train tasks file not found: {train_tasks_file}")
        return 0
    
    with open(train_tasks_file, 'r') as f:
        num_tasks = len([line.strip() for line in f.readlines() if line.strip()])
    
    return num_tasks


def sample_natural_instructions_tasks(
    tasks_dir="natural-instructions-2.8/tasks",
    num_tasks=5,
    max_instruction_tokens=1000,
    tokenizer=None,
    stable_test_split: bool = True,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    few_shot: bool = False,
):
    """Sample a few tasks from Natural Instructions dataset and split with absolute sizes per task
    
    Args:
        tasks_dir: Directory containing Natural Instructions task files
        num_tasks: Number of tasks to sample
        max_instruction_tokens: Maximum token length for instructions (default 1000)
        tokenizer: Tokenizer to use for token length filtering (required if max_instruction_tokens is used)
        stable_test_split: If True, select test examples first deterministically so test remains stable when train/val sizes change
        train_size: Absolute number of training samples per task
        val_size: Absolute number of validation samples per task
        test_size: Absolute number of test samples per task
        few_shot: If True, use few-shot learning
    Returns:
        train_data, val_data, test_data: Lists of formatted samples
        task_names: Sorted list of verified task names present in the data
    """
    if not os.path.exists(tasks_dir):
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    
    # Get all task files
    all_task_files = [f for f in os.listdir(tasks_dir) if f.startswith('task') and f.endswith('.json')]
    
    # Filter for English-only tasks
    print(f"🔍 Filtering {len(all_task_files)} tasks for English input/output...")
    english_task_files = []
    
    for task_file in all_task_files:
        task_path = os.path.join(tasks_dir, task_file)
        try:
            with open(task_path, 'r') as f:
                task_data = json.load(f)
            
            # Check if both input and output languages are English
            input_lang = task_data.get('Input_language', [])
            output_lang = task_data.get('Output_language', [])
            
            # Keep tasks where English is in both input and output languages
            if ('English' in input_lang or input_lang == ['English']) and \
               ('English' in output_lang or output_lang == ['English']):
                english_task_files.append(task_file)
                
        except Exception as e:
            print(f"Warning: Could not read {task_file}: {e}")
            continue
    
    print(f"✅ Found {len(english_task_files)} English-only tasks out of {len(all_task_files)} total")
    
    if len(english_task_files) < num_tasks:
        print(f"Warning: Only {len(english_task_files)} English tasks found, using all of them")
        num_tasks = len(english_task_files)
        task_files = english_task_files
    else:
        task_files = english_task_files
    
    # Use deterministic sampling to ensure smaller samples are subsets of larger ones
    # Sort task files to ensure reproducible ordering
    task_files_sorted = sorted(task_files)
    
    # Shuffle the sorted list using the global seed already set in main
    # This preserves the global determinism while ensuring hierarchical sampling
    random.shuffle(task_files_sorted)
    
    # Take the first num_tasks from the shuffled list
    # This ensures that tasks 1-5 are the same whether you ask for 5 or 10 tasks
    sampled_files = task_files_sorted[:num_tasks]
    
    print(f"Sampled {num_tasks} English-only tasks: {sampled_files}")
    
    all_train_data = []
    all_val_data = []
    all_test_data = []
    
    # Track filtered samples
    total_filtered = 0
    
    for task_file in sampled_files:
        task_path = os.path.join(tasks_dir, task_file)
        
        try:
            with open(task_path, 'r') as f:
                task_data = json.load(f)
            
            # Extract task information
            task_name = task_file.replace('.json', '')
            definition = ' '.join(task_data.get('Definition', []))
            instances = task_data.get('Instances', [])

            if few_shot:
                few_shot_instances = task_data.get('Positive Examples', [])
            else:
                few_shot_instances = []

            if not instances:
                print(f"Warning: No instances found in {task_file}, skipping")
                continue
            
            print(f"Processing {task_name}: {len(instances)} instances")

            # NOTE: For stability of the test split, we first filter, then derive test/train pools.
            # Filter instances by instruction token length first
            def is_instruction_valid(instruction):
                """Check if instruction is within token limit"""
                if tokenizer is None:
                    return True  # No filtering if no tokenizer provided
                token_count = len(tokenizer.encode(instruction, add_special_tokens=False))
                return token_count <= max_instruction_tokens
            
            # Filter all instances upfront
            filtered_instances = []
            task_filtered = 0
            for instance in instances:
                instruction = f"{definition}\n\n{instance['input']}" # coarse filtering which does not consider chat format
                if is_instruction_valid(instruction):
                    filtered_instances.append(instance)
                else:
                    task_filtered += 1

            # Skip task if all samples were filtered out
            if len(filtered_instances) == 0:
                print(f"  Skipping {task_name}: all instances filtered out")
                continue
            
            # Calculate total after filtering
            total = len(filtered_instances)

            # Derive a stable test split first if requested
            def instance_hash_info(inst):
                output_text = inst['output'][0] if isinstance(inst.get('output', ''), list) else inst.get('output', '')
                key = (inst.get('id', '') or '') + '||' + inst.get('input', '') + '||' + str(output_text)
                h = hashlib.md5(key.encode('utf-8')).hexdigest()
                rank = int(h, 16)
                return rank

            ranked_all = sorted(filtered_instances, key=lambda inst: instance_hash_info(inst))

            # Determine test set size default if not provided
            effective_test_size = test_size if test_size is not None else max(0, min(100, total // 10))
            if stable_test_split:
                test_instances = ranked_all[:min(effective_test_size, len(ranked_all))]
                remaining_instances = ranked_all[len(test_instances):]
            else:
                # Non-stable fallback: shuffle before slicing
                tmp = filtered_instances.copy()
                random.shuffle(tmp)
                test_instances = tmp[:min(effective_test_size, len(tmp))]
                remaining_instances = [inst for inst in tmp if inst not in test_instances]

            # Train/Val sizes defaults
            # If neither provided, put all remaining into train
            effective_train_size = train_size if train_size is not None else len(remaining_instances)
            effective_val_size = val_size if val_size is not None else 0

            ranked_remaining = sorted(remaining_instances, key=lambda inst: instance_hash_info(inst))
            train_instances = ranked_remaining[:min(effective_train_size, len(ranked_remaining))]
            after_train = ranked_remaining[len(train_instances):]
            val_instances = after_train[:min(effective_val_size, len(after_train))]

            # Convert to our format
            for instance in train_instances:
                instruction = f"{definition}\n\n{instance['input']}"
                sample = {
                    'instruction': definition,
                    'query': instance['input'],
                    'tasks': [task_name],
                    'responses': [instance['output'][0] if isinstance(instance['output'], list) else instance['output']],
                    'few_shot_examples': few_shot_instances,
                }
                all_train_data.append(sample)
            
            for instance in val_instances:
                instruction = f"{definition}\n\n{instance['input']}"
                sample = {
                    'instruction': definition,
                    'query': instance['input'],
                    'tasks': [task_name],
                    'responses': [instance['output'][0] if isinstance(instance['output'], list) else instance['output']],
                    'few_shot_examples': few_shot_instances,
                }
                all_val_data.append(sample)
                
            for instance in test_instances:
                instruction = f"{definition}\n\n{instance['input']}"
                sample = {
                    'instruction': definition,
                    'query': instance['input'],
                    'tasks': [task_name], 
                    'responses': [instance['output'][0] if isinstance(instance['output'], list) else instance['output']],
                    'few_shot_examples': few_shot_instances,
                }
                all_test_data.append(sample)
            
            total_filtered += task_filtered
            if task_filtered > 0:
                print(f"  Filtered {task_filtered} samples with instructions > {max_instruction_tokens} tokens")
            
            print(f"  Split - Train: {len(train_instances)}, Val: {len(val_instances)}, Test: {len(test_instances)}")
            
        except Exception as e:
            print(f"Error processing {task_file}: {e}")
            continue
    
    print(f"\nInitial samples - Train: {len(all_train_data)}, Val: {len(all_val_data)}, Test: {len(all_test_data)}")
    if total_filtered > 0:
        print(f"🔧 Filtered out {total_filtered} samples with instructions longer than {max_instruction_tokens} tokens")
    
    # Extract train tasks as the authoritative task list
    train_tasks = set()
    for item in all_train_data:
        train_tasks.update(item['tasks'])
    
    print(f"📋 Found {len(train_tasks)} training tasks")
    
    # Filter val data to only keep samples with training tasks
    filtered_val_data = []
    removed_val_samples = 0
    
    for item in all_val_data:
        # Keep samples that only contain training tasks
        if all(task in train_tasks for task in item['tasks']):
            filtered_val_data.append(item)
        else:
            removed_val_samples += 1
    
    all_val_data = filtered_val_data
    if removed_val_samples > 0:
        print(f"🔧 Removed {removed_val_samples} validation samples containing non-training tasks")
    
    # Filter test data to only keep samples with training tasks
    filtered_test_data = []
    removed_test_samples = 0
    
    for item in all_test_data:
        # Keep samples that only contain training tasks
        if all(task in train_tasks for task in item['tasks']):
            filtered_test_data.append(item)
        else:
            removed_test_samples += 1
    
    all_test_data = filtered_test_data
    if removed_test_samples > 0:
        print(f"🔧 Removed {removed_test_samples} test samples containing non-training tasks")
    
    # Use training tasks as the final task list
    task_names = sorted(list(train_tasks))
    actual_num_tasks = len(task_names)
    
    print(f"✅ Using {actual_num_tasks} training tasks: {task_names[:3]}{'...' if len(task_names) > 3 else ''}")
    print(f"\nFinal samples - Train: {len(all_train_data)}, Val: {len(all_val_data)}, Test: {len(all_test_data)}")
    
    return all_train_data, all_val_data, all_test_data, task_names



class NaturalInstructionsTaskDataset(Dataset):
    """Dataset for Natural Instructions using native reserved special tokens as task tokens"""
    
    def __init__(self, data_path=None, data=None, tokenizer=None, max_length=512, model=None, mode="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = model  # Need model for reserved token mappings
        self.mode = mode  # "train" or "eval"
        
        if data is not None:
            self.data = data
        elif data_path:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError("Either data_path or data must be provided")
        
        # Convert data to our format if needed
        self._convert_data_format()
    
    def _convert_data_format(self):
        """Ensure data is in the correct format for Natural Instructions"""
        # Data should already be in the correct format from sample_natural_instructions_tasks
        # Just validate the format
        for i, item in enumerate(self.data):
            if not all(key in item for key in ['instruction', 'tasks', 'responses']):
                print(f"Warning: Item {i} missing required keys: {item}")
        
        print(f"Loaded {len(self.data)} examples in Natural Instructions format")
    
    def __len__(self):
        return len(self.data)
    
    def _format_instruction(self, item):
        """Format few-shot examples into a single instruction"""
        # Build multiturn conversation with positive examples
        conversation_parts = []
        
        # Detect model type for chat format
        is_qwen = 'qwen' in self.tokenizer.name_or_path.lower()
        
        if is_qwen:
            # Qwen chat format (no system prompt to match Llama setup)
            # Add positive examples in multiturn format
            if 'few_shot_examples' in item and len(item['few_shot_examples']) > 0:
                for i, example in enumerate(item['few_shot_examples']):
                    # User turn with example input
                    if i == 0:
                        conversation_parts.append(f"<|im_start|>user\n{item['instruction']}\n\n{example.get('input', '')}<|im_end|>\n")
                    else:
                        conversation_parts.append(f"<|im_start|>user\n{example.get('input', '')}<|im_end|>\n")
                    # Assistant turn with example output  
                    conversation_parts.append(f"<|im_start|>assistant\n{example.get('output', '')}<|im_end|>\n")
            
                # Finally add the actual query in user turn and start assistant response
                conversation_parts.append(f"<|im_start|>user\n{item['query']}<|im_end|>\n")
                conversation_parts.append("<|im_start|>assistant\n")    
            
            else:
                conversation_parts.append(f"<|im_start|>user\n{item['instruction']}\n\n{item['query']}<|im_end|>\n")
                conversation_parts.append("<|im_start|>assistant\n")
        else:
            # Llama chat format (original)
            # Add positive examples in multiturn format
            conversation_parts.append("<|begin_of_text|>")
            
            if 'few_shot_examples' in item and len(item['few_shot_examples']) > 0:
                for i, example in enumerate(item['few_shot_examples']):
                    # User turn with example input
                    if i == 0:
                        conversation_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{item['instruction']}\n\n{example.get('input', '')}<|eot_id|>")
                    else:
                        conversation_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{example.get('input', '')}<|eot_id|>")
                    # Assistant turn with example output  
                    conversation_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{example.get('output', '')}<|eot_id|>")
            
                # Finally add the actual query in user turn and start assistant response
                conversation_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{item['query']}<|eot_id|>")
                conversation_parts.append(f"<|start_header_id|>assistant<|end_header_id|>")    
            
            else:
                conversation_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{item['instruction']}\n\n{item['query']}<|eot_id|>")
                conversation_parts.append(f"<|start_header_id|>assistant<|end_header_id|>")
            
        instruction_text = "".join(conversation_parts)
        return instruction_text
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create sequence: [Instruction] [Reserved_Task_Token] [Response] <|eot_id|>
        instruction_text = self._format_instruction(item)
        
        # Tokenize instruction input
        instruction_tokens = self.tokenizer(instruction_text, add_special_tokens=False)['input_ids']
        
        # If eval mode, return just the instruction input
        if self.mode == "eval":
            instruction_tokens = instruction_tokens[:self.max_length]
            
            input_ids = torch.tensor(instruction_tokens, dtype=torch.long)
            attention_mask = torch.ones(len(instruction_tokens), dtype=torch.long)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor([-100] * len(instruction_tokens), dtype=torch.long),  # Not used in eval
                'instruction_length': len(instruction_tokens),
                'raw_data': item
            }
        
        # Training mode: build full sequence with tasks and responses
        full_sequence = instruction_tokens.copy()
        labels = [-100] * len(instruction_tokens)  # Ignore instruction tokens in loss
        
        # Use multi-task format only
        if 'tasks' not in item or 'responses' not in item:
            raise ValueError("Data must contain 'tasks' and 'responses' fields")
            
        # Extract first task and response (Natural Instructions has 1 task per sample)
        task_name = item['tasks'][0]           
        response = item['responses'][0]
        
        # Get reserved token ID for task (from model)
        task_token_id = self.model.get_task_token_id(task_name)
        if task_token_id is None:
            raise ValueError(f"Task '{task_name}' not found in model task mappings. Skipping.")
            
        # Tokenize response
        response_tokens = self.tokenizer(str(response), add_special_tokens=False)['input_ids']
        
        # Add task token and response to sequence
        full_sequence.append(task_token_id)
        full_sequence.extend(response_tokens)
        
        # Add to labels (learn to predict task token and response tokens)
        labels.append(task_token_id)
        labels.extend(response_tokens)
        
        # Add end-of-turn token
        eot_token = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False)['input_ids']
        full_sequence.extend(eot_token)
        labels.extend(eot_token)
        
        # Truncate to max_length
        full_sequence = full_sequence[:self.max_length]
        labels = labels[:self.max_length]
        
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
            'instruction_length': len(instruction_tokens),
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
        'instruction_lengths': torch.tensor([item['instruction_length'] for item in batch]),
        'raw_data': [item['raw_data'] for item in batch]
    }


def create_natural_instructions_dataloader(model, train_data=None, val_data=None, test_data=None,
                                         tokenizer=None, batch_size=4, max_length=512, eval_batch_size=32):
    """Create DataLoaders for Natural Instructions
    
    Args:
        model: Model instance with task token mappings
        train_data: Training data samples
        val_data: Validation data samples
        test_data: Test data samples
        tokenizer: Tokenizer to use
        batch_size: Batch size for training dataloader
        max_length: Maximum sequence length
        eval_batch_size: Batch size for test dataloader
    
    Returns:
        train_dataloader: Training DataLoader
        val_dataloader: Validation DataLoader (if val_data provided)
        test_dataloader: Test DataLoader
        tokenizer: Tokenizer used
        test_examples: List of raw test examples for demo
    """
    
    # Create training dataset and dataloader
    train_dataloader = None
    if train_data is not None:
        train_dataset = NaturalInstructionsTaskDataset(
            data=train_data,
            tokenizer=tokenizer,
            max_length=max_length,
            model=model,
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )
        
        print(f"Training dataset created: {len(train_dataset)} samples")
    else:
        print(f"Warning: No training data provided")
    
    # Create validation dataset and dataloader
    val_dataloader = None
    if val_data is not None:
        val_dataset = NaturalInstructionsTaskDataset(
            data=val_data,
            tokenizer=tokenizer,
            max_length=max_length,
            model=model,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )
        
        print(f"Validation dataset created: {len(val_dataset)} samples")
    
    # Create test dataset and dataloader
    test_dataloader = None
    test_examples = []
    if test_data is not None:
        test_dataset = NaturalInstructionsTaskDataset(
            data=test_data,
            tokenizer=tokenizer,
            max_length=max_length,
            model=model,
            mode="eval"
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer)
        )
        
        test_examples = test_dataset.data.copy()
        print(f"Test dataset created: {len(test_dataset)} samples")
    else:
        print(f"Warning: No test data provided")
    
    return train_dataloader, val_dataloader, test_dataloader, tokenizer, test_examples