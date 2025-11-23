"""
Fine-tune GPT-2 for meme caption generation
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json

from src.config import config


class MemeGPT2Dataset(Dataset):
    """Dataset for GPT-2 training"""
    
    def __init__(self, txt_file, tokenizer, max_length=150):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load training examples
        print(f"Loading training data from {txt_file}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            self.examples = [line.strip() for line in f if line.strip()]
        
        print(f"✓ Loaded {len(self.examples)} training examples")
        
        # Show sample
        if self.examples:
            print(f"\n📝 Sample example:")
            print(self.examples[0][:200] + "..." if len(self.examples[0]) > 200 else self.examples[0])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize
        encodings = self.tokenizer(
            example,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for language modeling)
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def train_gpt2():
    """Fine-tune GPT-2 on meme captions"""
    
    print("="*80)
    print("GPT-2 Meme Caption Fine-tuning")
    print("="*80)
    
    # Check if training data exists
    train_file = os.path.join(config.processed_data_dir, 'gpt2_training.txt')
    if not os.path.exists(train_file):
        print(f"\n❌ Error: Training data not found at {train_file}")
        print("Please run: python -m src.prepare_gpt2_data")
        return
    
    # Load tokenizer and model
    print("\n📦 Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.eos_token_id
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ GPT-2 loaded: {trainable_params:,} trainable parameters")
    
    # Load dataset
    print("\n📊 Loading training data...")
    train_dataset = MemeGPT2Dataset(train_file, tokenizer, max_length=150)
    
    # Create output directory
    output_dir = os.path.join(config.models_dir, "gpt2_meme")
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    print("\n⚙️ Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # 3 epochs is usually enough for GPT-2
        per_device_train_batch_size=4,  # Adjust based on your GPU memory
        gradient_accumulation_steps=8,  # Effective batch size = 4 * 8 = 32
        learning_rate=5e-5,
        warmup_steps=500,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="no",
        fp16=config.device == "cuda",  # Use mixed precision on GPU
        logging_dir=os.path.join(config.results_dir, "gpt2_logs"),
        report_to="none",  # Disable wandb
        load_best_model_at_end=False,
    )
    
    print(f"✓ Configuration:")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Device: {config.device}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not masked language modeling (we're doing causal LM)
    )
    
    # Trainer
    print("\n🚀 Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "="*80)
    print("🎯 Starting training...")
    print("="*80)
    print("This will take approximately:")
    print("  - On GPU: 1-2 hours")
    print("  - On M1/M2: 3-4 hours")
    print("  - On CPU: 8-12 hours")
    print("="*80 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Saving current checkpoint...")
    
    # Save final model
    final_path = os.path.join(config.models_dir, "gpt2_meme_final")
    print(f"\n💾 Saving final model to {final_path}...")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save training info
    training_info = {
        'model': 'gpt2',
        'num_epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'learning_rate': training_args.learning_rate,
        'num_examples': len(train_dataset),
        'trainable_parameters': trainable_params,
    }
    
    info_path = os.path.join(final_path, 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80)
    print(f"📁 Model saved to: {final_path}")
    print(f"📊 Training info: {info_path}")
    print("\n🎉 Next step: Update your demo to use the fine-tuned model!")
    print("="*80)


def main():
    """Main training function"""
    train_gpt2()


if __name__ == "__main__":
    main()

