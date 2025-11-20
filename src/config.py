"""
Configuration file for MemeCrafter model
Contains all hyperparameters and paths
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Paths
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(project_root, "archive (4)")
    images_dir: str = os.path.join(data_dir, "images/images")
    labels_file: str = os.path.join(data_dir, "labels.csv")
    processed_data_dir: str = os.path.join(project_root, "data/processed")
    models_dir: str = os.path.join(project_root, "models")
    results_dir: str = os.path.join(project_root, "results")
    
    # Model parameters - Using BLIP-2 for better vision-language alignment
    model_name: str = "Salesforce/blip2-opt-2.7b"  # BLIP-2 with OPT-2.7B language model
    # Alternative smaller options:
    # "Salesforce/blip2-opt-2.7b" - Good balance of performance and size
    # "Salesforce/blip2-flan-t5-xl" - Better instruction following
    # "Salesforce/blip-image-captioning-base" - Original BLIP (lighter)
    
    max_length: int = 50  # Maximum caption length
    min_length: int = 10  # Minimum caption length
    image_size: int = 224  # Image input size
    
    # Training parameters
    batch_size: int = 8  # Reduced for BLIP-2 due to larger model size
    num_epochs: int = 10
    learning_rate: float = 1e-5  # Lower LR for fine-tuning large model
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4  # Increased to simulate larger batch
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    
    # Generation parameters
    num_beams: int = 5  # For beam search during generation
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.8
    repetition_penalty: float = 1.2  # Avoid repetitive captions
    length_penalty: float = 1.0
    
    # Training optimization
    use_8bit: bool = False  # Set to True for 8-bit quantization (saves memory)
    freeze_vision_encoder: bool = True  # Freeze vision weights, only train Q-Former and LLM
    use_lora: bool = True  # Use LoRA for efficient fine-tuning
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.05
    
    # Logging
    log_interval: int = 10
    save_interval: int = 1  # Save every N epochs
    use_wandb: bool = False  # Set to True to use Weights & Biases
    wandb_project: str = "memecrafter"
    
    # Device
    device: str = "cuda"  # For Mac M1/M2, use "cuda" for NVIDIA, "cpu" for CPU
    
    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

# Create a global config instance
config = Config()
