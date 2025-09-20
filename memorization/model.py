import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

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

class QAPromptTuning(nn.Module):
    """Prompt tuning for QA tasks with chat templates and flexible prompt positioning"""
    
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", prompt_length=10, 
                 prompt_position="infix", use_chat_template=False, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.prompt_length = prompt_length
        self.prompt_position = prompt_position  # "prefix" or "infix"
        self.use_chat_template = use_chat_template
        self.device = device
        self.dtype = dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Trainable prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(prompt_length, self.config.hidden_size, device=device, dtype=dtype) * 0.1
        )
    
    def forward(self, postfix_tokens=None, postfix_mask=None, prefix_tokens=None, prefix_mask=None, 
                assistant_header_tokens=None, assistant_header_mask=None):
        """
        Forward pass with chat template and flexible prompt positioning
        
        Prefix mode: [P] [User] [Assistant Answer]
        Infix mode:  [User] [P] [Assistant Answer]
        """
        batch_size = postfix_tokens.shape[0]
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_mask = torch.ones(batch_size, self.prompt_length, 
                               dtype=postfix_mask.dtype, device=postfix_mask.device)
        
        prefix_embeds = self.model.model.embed_tokens(prefix_tokens)
        postfix_embeds = self.model.model.embed_tokens(postfix_tokens)
        
        if self.prompt_position == "prefix":
            # Format: [P] [User] [Assistant Answer]

            assistant_header_embeds = self.model.model.embed_tokens(assistant_header_tokens)
            combined_embeds = torch.cat([prompt_embeds, prefix_embeds, assistant_header_embeds, postfix_embeds], dim=1)
            combined_mask = torch.cat([prompt_mask, prefix_mask, assistant_header_mask, postfix_mask], dim=1)

        else:  # infix mode

            assistant_header_embeds = self.model.model.embed_tokens(assistant_header_tokens)
            combined_embeds = torch.cat([prefix_embeds, prompt_embeds, assistant_header_embeds, postfix_embeds], dim=1)
            combined_mask = torch.cat([prefix_mask, prompt_mask, assistant_header_mask, postfix_mask], dim=1)
            
        
        outputs = self.model(inputs_embeds=combined_embeds, attention_mask=combined_mask)
        return outputs.logits

    def generate_answers(self, prefix_tokens, prefix_mask, tokenizer, max_new_tokens=256, 
                        assistant_header_tokens=None, assistant_header_mask=None):
        """Generate answers for a batch of user messages using chat template"""
        self.eval()
        
        batch_size = prefix_tokens.shape[0]
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        prompt_mask = torch.ones(batch_size, self.prompt_length, 
                               dtype=prefix_mask.dtype, device=prefix_mask.device)
        
        prefix_embeds = self.model.model.embed_tokens(prefix_tokens)
        
        # Combine embeddings based on prompt position
        if self.prompt_position == "prefix":
            # [P] [User] [Assistant]
            
            assistant_header_embeds = self.model.model.embed_tokens(assistant_header_tokens)
            input_embeds = torch.cat([prompt_embeds, prefix_embeds, assistant_header_embeds], dim=1)
            attention_mask = torch.cat([prompt_mask, prefix_mask, assistant_header_mask], dim=1)
       
        else:  # infix
           
            assistant_header_embeds = self.model.model.embed_tokens(assistant_header_tokens)
            input_embeds = torch.cat([prefix_embeds, prompt_embeds, assistant_header_embeds], dim=1)
            attention_mask = torch.cat([prefix_mask, prompt_mask, assistant_header_mask], dim=1)

        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
    
        predictions = []
        for i in range(batch_size):
            prediction = tokenizer.decode(outputs[i], skip_special_tokens=True)
            predictions.append(prediction)
        
        return predictions