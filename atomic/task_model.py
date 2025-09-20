import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import json

def count_parameters(model):
    """Count trainable and total parameters in a model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def print_model_info(model, model_name):
    """Print model information including parameter counts"""
    trainable_params, total_params = count_parameters(model)
    print(f"  {model_name}:")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable ratio: {trainable_params/total_params*100:.4f}%")

class TaskCallingModel(nn.Module):
    """
    Task calling using Llama's native reserved special tokens as task tokens.
    
    This model transparently overrides the embedding and lm_head layers to use 
    trainable embeddings for reserved tokens, with optional coupling/decoupling 
    of input and output layer weights. This enables the use of native generation 
    methods while maintaining custom task token behavior.
    
    Args:
        model_name: HuggingFace model name/path (default: "meta-llama/Llama-3.2-1B-Instruct")
        num_tasks: Number of task tokens to use (max 248, default: 100)
        task_names: List of task names (default: auto-generated)
        tokenizer: Pre-loaded tokenizer (required)
        device: Device to load model on (default: "cuda")
        dtype: Model data type (default: torch.bfloat16)
        decouple_embeddings: If True, use separate weights for input/output layers.
                           If False, share weights between input/output (default: True)
    """
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct",
                 num_tasks=100, task_names=None, tokenizer=None, device="cuda", dtype=torch.bfloat16, 
                 decouple_embeddings=False, is_extended=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        # self.num_tasks = min(num_tasks, 248)  # Max 248 reserved tokens available
        self.num_tasks = num_tasks
        self.device = device
        self.dtype = dtype
        self.decouple_embeddings = decouple_embeddings
        
        # Load tokenizer to get reserved token mappings
        self.tokenizer = tokenizer
        
        # Get reserved special token mappings
        self._setup_reserved_tokens()
        
        # Load frozen base model (all parameters frozen)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        if is_extended:
            self.model.resize_token_embeddings(len(tokenizer))
            
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get original embedding parameters for initialization
        original_embeddings = self.model.model.embed_tokens.weight.data
        
        # Create trainable parameters for task tokens (coupled or decoupled)
        # Clone the embeddings for the reserved tokens and make them parameters
        # Pre-create tensor for efficiency and reuse
        self.reserved_token_tensor = torch.tensor(self.reserved_token_ids, device=device)
        
        if self.decouple_embeddings:
            # Separate parameters for input and output layers
            self.trainable_task_input_embeddings = nn.Parameter(
                original_embeddings[self.reserved_token_tensor].clone()
            )
            self.trainable_task_output_embeddings = nn.Parameter(
                original_embeddings[self.reserved_token_tensor].clone()
            )
        else:
            # Shared parameter for both input and output layers
            self.trainable_task_embeddings = nn.Parameter(
                original_embeddings[self.reserved_token_tensor].clone()
            )
            # Create aliases for backward compatibility
            self.trainable_task_input_embeddings = self.trainable_task_embeddings
            self.trainable_task_output_embeddings = self.trainable_task_embeddings
        
        # Task name mapping
        self.task_names = task_names or [f"task_{i}" for i in range(self.num_tasks)]
        self.task_name_to_id = {name: i for i, name in enumerate(self.task_names)}
        self.task_id_to_name = {i: name for i, name in enumerate(self.task_names)}
        
        # Task token mappings (using reserved tokens)
        self.task_id_to_token_id = {i: self.reserved_token_ids[i] for i in range(self.num_tasks)}
        self.token_id_to_task_id = {self.reserved_token_ids[i]: i for i in range(self.num_tasks)} 

        # Override the model's forward method to use our custom embeddings and logits
        self._setup_model_override()
    
    def _setup_model_override(self):
        """Set up model override to make embedding/logit modifications transparent"""
        # Store original methods
        self.original_embed_forward = self.model.model.embed_tokens.forward
        self.original_lm_head_forward = self.model.lm_head.forward
        
        # Override embedding layer
        def custom_embed_forward(input_ids):
            # Get standard embeddings
            embeddings = self.original_embed_forward(input_ids)
            
            # Create mask for any reserved tokens in input_ids
            is_reserved = torch.isin(input_ids, self.reserved_token_tensor)
            if is_reserved.any():
                # Find which reserved token each position corresponds to
                for i, reserved_token_id in enumerate(self.reserved_token_ids):
                    mask = (input_ids == reserved_token_id)
                    if mask.any():
                        embeddings[mask] = self.trainable_task_input_embeddings[i]
            
            return embeddings
        
        # Override lm_head  
        def custom_lm_head_forward(hidden_states):
            # Get standard logits
            logits = self.original_lm_head_forward(hidden_states)
            
            # Efficiently replace reserved token logits using batch matmul
            # Shape: hidden_states (..., hidden_dim), task_embeddings (num_tasks, hidden_dim)
            task_logits = torch.matmul(hidden_states, self.trainable_task_output_embeddings.T)  # (..., num_tasks)
            
            # Replace the specific reserved token positions with our computed logits
            logits[..., self.reserved_token_tensor] = task_logits
            
            return logits
        
        # Apply overrides
        self.model.model.embed_tokens.forward = custom_embed_forward
        self.model.lm_head.forward = custom_lm_head_forward
    
    def restore_original_model(self):
        """Restore the original model methods (useful for cleanup or debugging)"""
        if hasattr(self, 'original_embed_forward'):
            self.model.model.embed_tokens.forward = self.original_embed_forward
        if hasattr(self, 'original_lm_head_forward'):
            self.model.lm_head.forward = self.original_lm_head_forward

    def _setup_reserved_tokens(self):
        """Extract reserved special token IDs from tokenizer"""
        vocab = self.tokenizer.get_vocab()
        reserved_tokens = {k: v for k, v in vocab.items() if 'reserved_special_token_' in k}
        
        # Sort by token ID to get consistent ordering
        sorted_reserved = sorted(reserved_tokens.items(), key=lambda x: x[1])
        
        # Take first num_tasks reserved tokens
        self.reserved_token_names = [token for token, _ in sorted_reserved[:self.num_tasks]]
        self.reserved_token_ids = [token_id for _, token_id in sorted_reserved[:self.num_tasks]]
        
        print(f"Using {len(self.reserved_token_ids)} reserved tokens as task tokens:")
        for i, (name, token_id) in enumerate(zip(self.reserved_token_names[:5], self.reserved_token_ids[:5])):
            print(f"  {name}: {token_id}")
        if len(self.reserved_token_ids) > 5:
            print(f"  ... and {len(self.reserved_token_ids) - 5} more")
        print(f"Embedding coupling mode: {'Decoupled' if self.decouple_embeddings else 'Coupled'}")
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for task calling training
        
        Sequence: [Instruction] [Reserved_Task_Token] [Response] <|eot_id|>
        """
        # Use the model's forward method directly (our overrides handle the custom embeddings/logits)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    
    def generate_with_task_prediction(self, instruction_tokens, instruction_mask, tokenizer, 
                                     max_new_tokens=256, temperature=0.6, top_p=0.9, do_sample=False):
        """Generate complete sequence: predict task, then generate response"""
        self.eval()
        
        with torch.no_grad():
            # Use native generation - our overrides make the custom embeddings/logits transparent
            generated = self.model.generate(
                input_ids=instruction_tokens,
                attention_mask=instruction_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Parse the generated sequences
            return self._parse_generated_sequences(generated, instruction_tokens, tokenizer)

    def generate_with_ground_truth_tasks(self, instruction_tokens, instruction_mask, tokenizer, ground_truth_tasks,
                                       max_new_tokens=256, temperature=0.6, top_p=0.9, do_sample=False):
        """Generate sequence where predicted task tokens are replaced with ground truth task tokens"""
        self.eval()
        
        # Convert ground truth task names to token IDs
        gt_task_token_ids = []
        for task_name in ground_truth_tasks:
            if task_name in self.task_name_to_id:
                task_id = self.task_name_to_id[task_name]
                token_id = self.task_id_to_token_id[task_id]
                gt_task_token_ids.append(token_id)
            else:
                print(f"Warning: Unknown task name '{task_name}', skipping")
        
        with torch.no_grad():
            batch_size = instruction_tokens.shape[0]
            device = instruction_tokens.device
            
            # Initialize generation state
            input_ids = instruction_tokens.clone()
            attention_mask = instruction_mask.clone()
            task_replacement_count = [0] * batch_size  # Track how many tasks replaced per example
            
            for step in range(max_new_tokens):
                # Get model predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Get logits for last position
                
                # Sample/select next tokens
                if do_sample:
                    # Apply temperature
                    logits = logits / temperature
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        for i in range(batch_size):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            logits[i][indices_to_remove] = -float('inf')
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = torch.argmax(logits, dim=-1)
                
                # Check if any predicted tokens are task tokens and replace them
                for i in range(batch_size):
                    predicted_token = next_tokens[i].item()
                    
                    # If predicted token is a task token and we have ground truth tasks left
                    if (predicted_token in self.token_id_to_task_id and 
                        task_replacement_count[i] < len(gt_task_token_ids)):
                        # Replace with ground truth task token
                        next_tokens[i] = gt_task_token_ids[task_replacement_count[i]]
                        task_replacement_count[i] += 1
                
                # Check for EOS tokens
                if (next_tokens == tokenizer.eos_token_id).all():
                    break
                
                # Append new tokens to input_ids and update attention_mask
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype),
                    ],
                    dim=-1,
                )
            
            # Parse the generated sequences
            return self._parse_generated_sequences(input_ids, instruction_tokens, tokenizer)
    
    def _parse_generated_sequences(self, generated_tokens, input_tokens, tokenizer):
        """Parse generated sequences to extract task predictions and responses"""
        results = []
        batch_size = generated_tokens.shape[0]
        input_length = input_tokens.shape[1]
        
        for i in range(batch_size):
            # Extract only the newly generated part
            generated_seq = generated_tokens[i, input_length:]
            
            # Remove EOS tokens and convert to list
            valid_tokens = []
            for token_id in generated_seq:
                if token_id.item() == tokenizer.eos_token_id:
                    break
                valid_tokens.append(token_id.item())
            
            if not valid_tokens:
                results.append({
                    'predicted_tasks': [],
                    'responses': [],
                    'full_generated_sequence': '',
                    # Backward compatibility
                    'predicted_task_id': None,
                    'predicted_task_name': 'none',
                    'response': ''
                })
                continue
            
            # Find all task tokens and their positions
            task_positions = []
            for j, token_id in enumerate(valid_tokens):
                if token_id in self.token_id_to_task_id:
                    task_id = self.token_id_to_task_id[token_id]
                    task_name = self.task_id_to_name[task_id]
                    task_positions.append({
                        'position': j,
                        'token_id': token_id,
                        'task_id': task_id,
                        'task_name': task_name
                    })
            
            predicted_tasks = []
            responses = []
            
            # Extract response for each task
            for idx, task_info in enumerate(task_positions):
                start_pos = task_info['position'] + 1  # Position after task token
                
                # Find end position (next task token or end of sequence)
                if idx + 1 < len(task_positions):
                    end_pos = task_positions[idx + 1]['position']
                else:
                    # Last task - response goes to end of sequence (excluding EOT if present)
                    end_pos = len(valid_tokens)
                    # Check if last tokens are EOT tokens and exclude them
                    eot_tokens = tokenizer(tokenizer.eos_token, add_special_tokens=False)['input_ids']
                    if len(eot_tokens) > 0 and end_pos >= len(eot_tokens):
                        # Check if the last tokens match EOT
                        if valid_tokens[-len(eot_tokens):] == eot_tokens:
                            end_pos -= len(eot_tokens)
                
                # Extract response tokens
                if start_pos < end_pos:
                    response_tokens = valid_tokens[start_pos:end_pos]
                    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                else:
                    response = ""
                
                predicted_tasks.append({
                    'task_id': task_info['task_id'],
                    'task_name': task_info['task_name'],
                    'token_id': task_info['token_id']
                })
                responses.append(response)
            
            # If no task tokens found, treat entire sequence as response
            if not task_positions:
                response = tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
                predicted_tasks = []
                responses = [response] if response else []
            
            # Prepare result with both new multi-task format and backward compatibility
            result = {
                'predicted_tasks': predicted_tasks,
                'responses': responses,
                'full_generated_sequence': tokenizer.decode(valid_tokens, skip_special_tokens=True),
                # Backward compatibility - use first task if available
                'predicted_task_id': predicted_tasks[0]['task_id'] if predicted_tasks else None,
                'predicted_task_name': predicted_tasks[0]['task_name'] if predicted_tasks else 'none',
                'response': responses[0] if responses else '',
                'task_token_used': tokenizer.decode([predicted_tasks[0]['token_id']]) if predicted_tasks else None
            }
            
            results.append(result)
        
        return results
    
    def get_task_token_id(self, task_name):
        """Get the reserved token ID for a task name"""
        if task_name in self.task_name_to_id:
            task_id = self.task_name_to_id[task_name]
            return self.task_id_to_token_id[task_id]
        return None
    
    def get_task_name_from_token_id(self, token_id):
        """Get task name from reserved token ID"""
        if token_id in self.token_id_to_task_id:
            task_id = self.token_id_to_task_id[token_id]
            return self.task_id_to_name[task_id]
        return None
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters for optimizer initialization"""
        if self.decouple_embeddings:
            return [self.trainable_task_input_embeddings, self.trainable_task_output_embeddings]
        else:
            return [self.trainable_task_embeddings]
    
    def save_task_tokens(self, filepath):
        """Save trained task token embeddings to file"""
        save_data = {
            'task_names': self.task_names,
            'num_tasks': self.num_tasks,
            'decouple_embeddings': self.decouple_embeddings,
            'task_name_to_id': self.task_name_to_id,
            'reserved_token_ids': self.reserved_token_ids,
        }
        
        if self.decouple_embeddings:
            save_data['input_embeddings'] = self.trainable_task_input_embeddings.detach().cpu()
            save_data['output_embeddings'] = self.trainable_task_output_embeddings.detach().cpu()
        else:
            save_data['embeddings'] = self.trainable_task_embeddings.detach().cpu()
        
        torch.save(save_data, filepath)
        print(f"Task tokens saved to {filepath}")
    
    def load_task_tokens(self, filepath):
        """Load task token embeddings from file"""
        data = torch.load(filepath, map_location='cpu')
        
        # Basic checks
        if data['task_names'] != self.task_names:
            raise ValueError(f"Task names don't match saved file")
        
        if data['decouple_embeddings'] != self.decouple_embeddings:
            raise ValueError(f"Embedding mode doesn't match")
        
        # Load embeddings
        if self.decouple_embeddings:
            self.trainable_task_input_embeddings.data = data['input_embeddings'].to(self.device)
            self.trainable_task_output_embeddings.data = data['output_embeddings'].to(self.device)
        else:
            self.trainable_task_embeddings.data = data['embeddings'].to(self.device)
        
        print(f"Task tokens loaded from {filepath}")
        return data