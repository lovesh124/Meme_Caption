"""
MemeCrafter Model Architecture using BLIP-2
"""
import torch
import torch.nn as nn
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.config import config


class MemeCrafterModel(nn.Module):
    """
    MemeCrafter model using BLIP-2 for meme caption generation
    """
    def __init__(self, model_name=None, use_lora=True, freeze_vision=True):
        super().__init__()
        
        self.model_name = model_name or config.model_name
        self.use_lora = use_lora
        
        print(f"Loading BLIP-2 model: {self.model_name}")
        
        # Load processor and model
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if config.device != "cpu" else torch.float32,
            device_map=config.device if config.device != "mps" else None
        )
        
        # Move to device if using MPS (Mac M1/M2)
        if config.device == "mps":
            self.model = self.model.to(config.device)
        
        # Freeze vision encoder if specified
        if freeze_vision:
            print("Freezing vision encoder...")
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
        
        # Apply LoRA for efficient fine-tuning
        if use_lora:
            print("Applying LoRA for efficient fine-tuning...")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj"],  # Apply to attention layers
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        """
        Forward pass through the model
        
        Args:
            pixel_values: Preprocessed image tensors
            input_ids: Tokenized text input (for conditional generation)
            attention_mask: Attention mask for text
            labels: Target captions for training
        
        Returns:
            Model outputs including loss and generated sequences
        """
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
    
    def generate(self, pixel_values, prompt=None, **generation_kwargs):
        """
        Generate captions for images
        
        Args:
            pixel_values: Preprocessed image tensors
            prompt: Optional text prompt to guide generation
            **generation_kwargs: Additional generation parameters
        
        Returns:
            Generated caption text
        """
        # Set default generation parameters
        gen_kwargs = {
            "max_length": config.max_length,
            "min_length": config.min_length,
            "num_beams": config.num_beams,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
            "length_penalty": config.length_penalty,
            "do_sample": True,
        }
        gen_kwargs.update(generation_kwargs)
        
        # Generate caption
        if prompt:
            # Tokenize prompt
            prompt_ids = self.processor(
                text=prompt,
                return_tensors="pt"
            ).input_ids.to(self.model.device)
            
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=prompt_ids,
                **gen_kwargs
            )
        else:
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                **gen_kwargs
            )
        
        # Decode generated tokens
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def save_model(self, save_path):
        """Save model and processor"""
        print(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
    
    @classmethod
    def load_model(cls, load_path):
        """Load model and processor from path"""
        print(f"Loading model from {load_path}")
        model = cls(model_name=load_path, use_lora=False)
        return model


class MemeCrafterPromptModel(MemeCrafterModel):
    """
    Enhanced MemeCrafter with prompt engineering for better meme understanding
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Define meme-specific prompts
        self.meme_prompts = [
            "Generate a funny meme caption: ",
            "What humorous text fits this image? ",
            "Create a witty meme caption for: ",
            "Write a comedic caption: ",
        ]
    
    def generate_with_prompt(self, pixel_values, prompt_type="default", **generation_kwargs):
        """
        Generate captions with meme-specific prompts
        
        Args:
            pixel_values: Preprocessed image tensors
            prompt_type: Type of prompt to use ("default", "funny", "witty", "comedic")
        """
        if prompt_type == "default":
            prompt = self.meme_prompts[0]
        elif prompt_type == "funny":
            prompt = self.meme_prompts[0]
        elif prompt_type == "witty":
            prompt = self.meme_prompts[2]
        elif prompt_type == "comedic":
            prompt = self.meme_prompts[3]
        else:
            prompt = None
        
        return self.generate(pixel_values, prompt=prompt, **generation_kwargs)


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    # Test model initialization
    print("Testing MemeCrafter model initialization...")
    model = MemeCrafterModel()
    
    trainable, total = count_parameters(model.model)
    print(f"\nModel parameters:")
    print(f"  Trainable: {trainable:,}")
    print(f"  Total: {total:,}")
    print(f"  Trainable %: {100 * trainable / total:.2f}%")
