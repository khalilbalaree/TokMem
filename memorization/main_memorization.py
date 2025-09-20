import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random

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

# Fix seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Set device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16
torch.set_default_dtype(dtype)
print(f"Using device: {device}")
print(f"Using dtype: {dtype}")
print(f"Seed fixed at: 42")

class SingleModelPromptTuning(nn.Module):
    """Prompt tuning with a single model"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1b", prompt_length=10, before_prompt=False, pad_token_id=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.prompt_length = prompt_length
        self.before_prompt = before_prompt
        self.pad_token_id = pad_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        
        # Freeze all model parameters (including pretrained lm_head)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Trainable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, self.config.hidden_size, device=device, dtype=dtype) * 0.1
        )
    
    def forward(self, input_ids, attention_mask=None, id_tokens=None):
        batch_size = input_ids.shape[0]
        
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        input_embeds = self.model.model.embed_tokens(input_ids)
        
        if id_tokens is not None:
            # ID tokens and prompt ordering depends on before_prompt flag
            # If before_prompt=True: [ID] [P] [Text], else: [P] [ID] [Text]
            id_embeds = self.model.model.embed_tokens(id_tokens)
            if self.before_prompt:
                combined_embeds = torch.cat([id_embeds, prompt_embeds, input_embeds], dim=1)
            else:
                combined_embeds = torch.cat([prompt_embeds, id_embeds, input_embeds], dim=1)
        else:
            # No ID tokens: [P] [Text]
            combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        if attention_mask is not None:
            # Create attention mask for prompt embeddings (always 1s)
            prompt_mask = torch.ones(batch_size, self.prompt_length, 
                                   dtype=attention_mask.dtype, device=attention_mask.device)
            
            if id_tokens is not None:
                # Create attention mask for ID tokens (1 for real tokens, 0 for padding)
                id_mask = (id_tokens != self.pad_token_id).to(dtype=attention_mask.dtype)
                
                # Combine masks based on ordering
                if self.before_prompt:
                    prefix_mask = torch.cat([id_mask, prompt_mask], dim=1)
                else:
                    prefix_mask = torch.cat([prompt_mask, id_mask], dim=1)
            else:
                prefix_mask = prompt_mask
            
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        outputs = self.model(inputs_embeds=combined_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        
        return logits

def create_memorization_batch(tokenizer, batch_size=4, max_length=512, start_index=0, use_ids=False, id_length=1, id_method="random_words", before_prompt=False):
    """Create batch of texts for memorization"""
    import pandas as pd
    import random
    import hashlib
    import string
    
    df = pd.read_csv('./memorization_data/fanfics_1k_chunks.csv')
    
    texts = []
    all_tokens = []
    all_id_tokens = []
    all_id_texts = []  # Store generated ID texts for display
    
    # Track used words/combinations across all texts in the batch for random_words method
    used_words = set()
    used_combinations = set()
    
    # Generate ID tokens based on method
    def generate_id(index, text, method, length, used_words_set, used_combinations_set):
        if method == "random_words":
            # Random English words - generate multiple words to reach target token length
            word_list = ["apple", "brave", "cloud", "dance", "eagle", "flame", "grace", "house", "ice", "joy",
                        "kind", "light", "magic", "night", "ocean", "peace", "queen", "river", "storm", "tree",
                        "unity", "voice", "wind", "xenon", "youth", "zebra", "anchor", "bloom", "craft", "dream",
                        "ember", "frost", "giant", "honor", "ivory", "jewel", "knight", "lunar", "mist", "noble"]
            
            # Deterministic but unique seed per index
            random.seed(42 + index)
            
            if length == 1:
                # For single words, maintain unique words across all texts
                available_words = [word for word in word_list if word not in used_words_set]
                
                if len(available_words) == 0:
                    print(f"Warning: All words used, falling back to full word list for text {index}")
                    available_words = word_list
                
                word = random.choice(available_words)
                used_words_set.add(word)
                return word
            else:
                # For multiple words, allow word reuse but ensure unique combinations
                max_attempts = 1000  # Prevent infinite loops
                attempts = 0
                
                while attempts < max_attempts:
                    words = []
                    current_tokens = 0
                    
                    # Generate words until we reach approximately the target token length
                    while current_tokens < length:
                        word = random.choice(word_list)
                        words.append(word)
                        # Rough estimate: most words are 1-2 tokens
                        current_tokens += 1 + (len(word) // 4)  # Approximation
                    
                    # Check if this combination has been used
                    combination = " ".join(words)
                    if combination not in used_combinations_set:
                        used_combinations_set.add(combination)
                        return combination
                    
                    attempts += 1
                    # Re-seed with different value for next attempt
                    random.seed(42 + index + attempts)
                
                # If we couldn't find a unique combination, return the last generated one with warning
                print(f"Warning: Could not find unique combination for text {index} after {max_attempts} attempts")
                return combination
        
        elif method == "content_hash":
            # Hash-based IDs from text content - repeat hash if needed for long sequences
            hash_obj = hashlib.md5(text.encode())
            hash_str = hash_obj.hexdigest()
            
            # If we need more characters than a single hash provides, repeat it
            if length > len(hash_str):
                repeat_count = (length // len(hash_str)) + 1
                hash_str = hash_str * repeat_count
            
            return hash_str[:length]
        
        else:
            raise ValueError(f"Unknown ID method: {method}")
    
    for i in range(batch_size):
        text_index = start_index + i
        if text_index >= len(df):
            text_index = text_index % len(df)  # Wrap around if we exceed dataset size
            
        text = df['text'][text_index]
        texts.append(text)
        
        # Tokenize the original text (without ID)
        tokens = tokenizer.encode(text, 
                                  truncation=True,
                                  max_length=max_length,
                                  add_special_tokens=False,
                                  return_tensors="pt")[0]
        all_tokens.append(tokens)
        
        # Generate and tokenize ID separately if using IDs
        if use_ids:
            id_text = generate_id(i, text, id_method, id_length, used_words, used_combinations)
            all_id_texts.append(id_text)  # Store for display
            id_tokens = tokenizer.encode(id_text,
                                       add_special_tokens=False,
                                       return_tensors="pt")[0]
            all_id_tokens.append(id_tokens)
    
    # Pad text sequences to same length
    max_seq_len = max(len(tokens) for tokens in all_tokens)
    padded_tokens = []
    attention_masks = []
    
    for tokens in all_tokens:
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones(len(tokens), dtype=torch.long)
        
        # Pad with tokenizer.pad_token_id
        pad_length = max_seq_len - len(tokens)
        if pad_length > 0:
            tokens = torch.cat([tokens, torch.full((pad_length,), tokenizer.pad_token_id, dtype=tokens.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        padded_tokens.append(tokens)
        attention_masks.append(attention_mask)
    
    # Stack into batch tensors
    batch_tokens = torch.stack(padded_tokens).to(device)
    batch_attention_masks = torch.stack(attention_masks).to(device)
    
    # Handle ID tokens if using IDs
    batch_id_tokens = None
    if use_ids:
        # Print original ID token lengths
        print("ID Token Lengths (before padding):")
        original_lengths = [len(id_tokens) for id_tokens in all_id_tokens]
        for i, length in enumerate(original_lengths):
            print(f"  Batch item {i}: {length} tokens")
        
        # Pad ID tokens to same length
        max_id_len = max(len(id_tokens) for id_tokens in all_id_tokens)
        print(f"Maximum ID token length: {max_id_len}")
        
        padded_id_tokens = []
        
        for i, id_tokens in enumerate(all_id_tokens):
            original_len = len(id_tokens)
            pad_length = max_id_len - original_len
            if pad_length > 0:
                id_tokens = torch.cat([torch.full((pad_length,), tokenizer.pad_token_id, dtype=id_tokens.dtype), id_tokens])
            padded_id_tokens.append(id_tokens)
            print(f"  Batch item {i}: {original_len} -> {len(id_tokens)} tokens (padded +{pad_length})")
        
        batch_id_tokens = torch.stack(padded_id_tokens).to(device)
        print(f"Final batch ID tokens shape: {batch_id_tokens.shape}")
        print()
        
        # Update texts to show the combined format for display - ordering depends on before_prompt
        display_texts = []
        for i, text in enumerate(texts):
            id_display = all_id_texts[i]  # Use the exact same ID text that was tokenized
            if before_prompt:
                display_texts.append(f"{id_display} [P] {text}")
            else:
                display_texts.append(f"[P] {id_display} {text}")
        texts = display_texts
    
    return {
        "tokens": batch_tokens,
        "attention_masks": batch_attention_masks,
        "id_tokens": batch_id_tokens,
        "texts": texts,
        "batch_size": batch_size
    }

def train_model(model, text_data, epochs=5000, lr=0.01):
    """Train model to memorize batch of texts with early stopping"""
    optimizer = optim.AdamW([model.prompt_embeddings], lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens
    # Note: No GradScaler needed for bfloat16 - it has wider dynamic range than float16
    
    tokens = text_data["tokens"]  # Already batched [batch_size, seq_len]
    attention_masks = text_data["attention_masks"]  # [batch_size, seq_len]
    id_tokens = text_data.get("id_tokens", None)  # [batch_size, id_len] or None
    batch_size = text_data["batch_size"]
    losses = []
    best_accuracy = 0.0
    best_loss = float('inf')
    best_perplexity = float('inf')
    best_epoch = 0
    early_stopped = False
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with autocast for bfloat16
        with torch.amp.autocast('cuda', dtype=dtype):
            logits = model(tokens, attention_mask=attention_masks, id_tokens=id_tokens)
            
            # Loss calculation - predict next token from current token
            # Only calculate loss on the non-prompt tokens
            id_length = id_tokens.shape[1] if id_tokens is not None else 0
            prefix_length = id_length + model.prompt_length
            shift_logits = logits[..., prefix_length:-1, :].contiguous()  # Skip prefix tokens, [batch, seq-1, vocab]
            shift_labels = tokens[..., 1:].contiguous()  # [batch, seq-1]
            shift_attention = attention_masks[..., 1:].contiguous()  # [batch, seq-1]
            
            # Set padding tokens to -100 so they're ignored by CrossEntropyLoss
            shift_labels = shift_labels.clone()
            shift_labels[shift_attention == 0] = -100
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
        
        # Simple backward pass (no gradient scaling needed for bfloat16)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Validate accuracy every 100 epochs
        if epoch % 100 == 0 and epoch > 0:
            val_loss, val_accuracy, val_perplexity = evaluate_memorization(model, text_data)
            print(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Perplexity: {val_perplexity:.4f}")
            
            # Track best perplexity (lower is better)
            if val_perplexity < best_perplexity:
                best_perplexity = val_perplexity
                best_loss = val_loss
                best_accuracy = val_accuracy
                best_epoch = epoch
            
            # Early stopping if perplexity < 1.1 (very good performance)
            if val_perplexity < 1.1:
                print(f"  Early stopping at epoch {epoch} (perplexity < 1.1)")
                early_stopped = True
                break
    
    # Return best performance achieved during training
    return losses, best_loss, best_accuracy, best_perplexity, best_epoch

def evaluate_memorization(model, text_data):
    """Evaluate memorization performance on batch of texts"""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')  # Ignore padding tokens, no reduction
    
    tokens = text_data["tokens"]  # Already batched [batch_size, seq_len]
    attention_masks = text_data["attention_masks"]  # [batch_size, seq_len]
    id_tokens = text_data.get("id_tokens", None)  # [batch_size, id_len] or None
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            logits = model(tokens, attention_mask=attention_masks, id_tokens=id_tokens)
            
            # Loss - only calculate on non-prompt tokens
            id_length = id_tokens.shape[1] if id_tokens is not None else 0
            prefix_length = id_length + model.prompt_length
            shift_logits = logits[..., prefix_length:-1, :].contiguous()  # Skip prefix tokens, [batch, seq-1, vocab]
            shift_labels = tokens[..., 1:].contiguous()  # [batch, seq-1]
            shift_attention = attention_masks[..., 1:].contiguous()  # [batch, seq-1]
            
            # Set padding tokens to -100 so they're ignored by CrossEntropyLoss
            shift_labels = shift_labels.clone()
            shift_labels[shift_attention == 0] = -100
            
            # Calculate loss per token (no reduction)
            loss_per_token = criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                     shift_labels.view(-1))
            
            # Calculate average loss (for backward compatibility)
            valid_mask = shift_attention.view(-1) == 1
            valid_losses = loss_per_token[valid_mask]
            loss = valid_losses.mean().item() if len(valid_losses) > 0 else float('inf')
            
            # Calculate perplexity
            perplexity = torch.exp(valid_losses.mean()).item() if len(valid_losses) > 0 else float('inf')
            
            # Accuracy - only count non-padding tokens
            predictions = torch.argmax(shift_logits, dim=-1)
            # Only calculate accuracy on non-padding tokens
            valid_mask = shift_attention == 1
            correct_tokens = ((predictions == tokens[..., 1:]) & valid_mask).sum().item()
            total_tokens = valid_mask.sum().item()
            accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return loss, accuracy, perplexity



def run_batch_text_memorization(batch_size=4, max_length=512, start_index=0, prompt_length=1, use_ids=False, id_length=1, id_method="random_words", before_prompt=False):
    """Memorize a batch of texts using prompt tuning"""
    set_seed(42)
    
    if use_ids:
        position_desc = "before prompt" if before_prompt else "after prompt"
        id_desc = f" (with {id_method} IDs, length {id_length}, {position_desc})"
    else:
        id_desc = ""
    print(f"Batch Text Memorization{id_desc}")
    print(f"Batch size: {batch_size}, Max length: {max_length}, Start index: {start_index}, Prompt length: {prompt_length}")
    print("=" * 60)
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text_data = create_memorization_batch(tokenizer, batch_size=batch_size, 
                                        max_length=max_length, start_index=start_index, 
                                        use_ids=use_ids, id_length=id_length, id_method=id_method, 
                                        before_prompt=before_prompt)
    
    print("BATCH INFORMATION:")
    print("-" * 40)
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {text_data['tokens'].shape[1]}")
    print(f"Total tokens to memorize: {text_data['tokens'].numel():,}")
    if use_ids:
        method_desc = {
            "random_words": f"Random English words (approx {id_length} tokens)",
            "content_hash": f"Content hash (length {id_length})"
        }
        position_desc = "before prompt [ID] [P] [Text]" if before_prompt else "after prompt [P] [ID] [Text]"
        print(f"ID method: {method_desc.get(id_method, id_method)}")
        print(f"ID position: {position_desc}")
    for i, text in enumerate(text_data['texts']):
        display_text = text[:50] + "..." if len(text) > 50 else text
        print(f"  Text {start_index + i}: '{display_text}'")
    print()
    
    # Create single model
    set_seed(42)
    single_model = SingleModelPromptTuning(prompt_length=prompt_length, before_prompt=before_prompt, pad_token_id=tokenizer.pad_token_id).to(device).to(dtype)
    
    print("MODEL INFORMATION:")
    print("-" * 40)
    print_model_info(single_model, "Single Model")
    print("=" * 60)
    
    # Train models
    print("TRAINING:")
    print("-" * 40)
    
    print("\nTraining Single Model...")
    set_seed(42)
    
    # Save initial prompt embeddings before training
    initial_embeddings = single_model.prompt_embeddings.clone().detach()
    initial_norm = torch.norm(initial_embeddings).item()
    
    single_losses, single_loss, single_acc, single_perplexity, single_epoch = train_model(single_model, text_data)
    
    # Calculate prompt embeddings norm after training
    final_embeddings = single_model.prompt_embeddings.clone().detach()
    final_norm = torch.norm(final_embeddings).item()
    
    # Calculate angle change between initial and final embeddings
    # Flatten embeddings for angle calculation
    initial_flat = initial_embeddings.view(-1)
    final_flat = final_embeddings.view(-1)
    
    # Calculate cosine similarity
    dot_product = torch.dot(initial_flat, final_flat).item()
    cosine_similarity = dot_product / (initial_norm * final_norm)
    
    # Clamp to avoid numerical issues with arccos
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    
    # Calculate angle in radians and degrees
    angle_radians = torch.acos(torch.tensor(cosine_similarity)).item()
    angle_degrees = angle_radians * 180.0 / torch.pi
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Single Model:  Loss: {single_loss:.4f}, Acc: {single_acc:.4f}, Perplexity: {single_perplexity:.4f}, Best at epoch: {single_epoch}")
    print(f"Prompt Embeddings - Initial Norm: {initial_norm:.4f}, Final Norm: {final_norm:.4f}")
    print(f"Prompt Embedding Angle Change: {angle_degrees:.2f}° ({angle_radians:.4f} radians)")
    print(f"Cosine Similarity (initial vs final): {cosine_similarity:.4f}")
    
    return {
        'mode': 'batch_text',
        'batch_size': batch_size,
        'max_length': max_length,
        'sequence_length': text_data['tokens'].shape[1],
        'use_ids': use_ids,
        'id_length': id_length if use_ids else None,
        'id_method': id_method if use_ids else None,
        'accuracy': single_acc,
        'loss': single_loss,
        'perplexity': single_perplexity,
        'initial_norm': initial_norm,
        'final_norm': final_norm,
        'angle_degrees': angle_degrees,
        'angle_radians': angle_radians,
        'cosine_similarity': cosine_similarity
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python memorize_batch.py [batch_size] [max_length] [start_index] [prompt_length] [ids] [id_length] [id_method] [before|after]")
        print()
        print("ID Methods:")
        print("  random_words     - Random English words: apple storm magic brave... (default)")
        print("  content_hash     - Hash from text content: 3f2a8d7b1e...")
        print()
        print("ID Position:")
        print("  before           - ID tokens come before prompt: [ID] [P] [Text]")
        print("  after (default)  - ID tokens come after prompt: [P] [ID] [Text]")
        print()
        print("Examples:")
        print("  python memorize_batch.py 4 512 0 1 ids 50 random_words       # Long word sequences")
        print("  python memorize_batch.py 4 512 0 1 ids 200 content_hash      # Long hash-based IDs")
        print("  python memorize_batch.py 4 512 0 1 ids 500 random_words      # Very long word sequences")
        print("  python memorize_batch.py 4 512 0 1 ids 100 content_hash      # Medium hash-based IDs")
        print("  python memorize_batch.py 4 512 0 1 ids 50 random_words before  # ID before prompt")
        print("  python memorize_batch.py 4 512 0 1 ids 100 content_hash after   # ID after prompt")
        sys.exit(1)
    
    # Parse arguments
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    max_length = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    start_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    prompt_length = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    
    # Parse optional flags
    use_ids = False
    id_length = 1
    id_method = "random_words"
    before_prompt = False
    
    i = 5
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.lower() == 'ids':
            use_ids = True
            # Check if next argument is id_length
            if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                i += 1
                id_length = int(sys.argv[i])
            # Check if argument after that is id_method
            if i + 1 < len(sys.argv) and sys.argv[i + 1] in ["random_words", "content_hash"]:
                i += 1
                id_method = sys.argv[i]
        elif arg in ["random_words", "content_hash"]:
            id_method = arg
        elif arg.lower() == 'before':
            before_prompt = True
        elif arg.lower() == 'after':
            before_prompt = False
        i += 1
    
    results = run_batch_text_memorization(
        batch_size=batch_size,
        max_length=max_length,
        start_index=start_index,
        prompt_length=prompt_length,
        use_ids=use_ids,
        id_length=id_length,
        id_method=id_method,
        before_prompt=before_prompt
    )