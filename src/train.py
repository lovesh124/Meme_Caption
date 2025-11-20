"""
Training script for MemeCrafter using BLIP-2
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from datetime import datetime

from src.config import config
from src.model import MemeCrafterModel
from src.dataset import MemeDataset, collate_fn
from src.preprocess import load_and_preprocess_data


class Trainer:
    """Trainer class for MemeCrafter model"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device) if 'input_ids' in batch else None
            attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass with gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), 
                    config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * config.gradient_accumulation_steps,
                'lr': self.scheduler.get_last_lr()[0]
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device) if 'input_ids' in batch else None
                attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(config.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(config.models_dir, "best_model")
                self.model.save_model(save_path)
                print(f"✓ Saved best model with val_loss: {val_loss:.4f}")
            
            # Save checkpoint every save_interval epochs
            if (epoch + 1) % config.save_interval == 0:
                save_path = os.path.join(config.models_dir, f"checkpoint_epoch_{epoch+1}")
                self.model.save_model(save_path)
                print(f"✓ Saved checkpoint at epoch {epoch+1}")
        
        # Save final model
        final_path = os.path.join(config.models_dir, "final_model")
        self.model.save_model(final_path)
        print(f"✓ Saved final model")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': {
                'model_name': config.model_name,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'num_epochs': config.num_epochs,
            }
        }
        
        history_path = os.path.join(config.results_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✓ Saved training history to {history_path}")


def main():
    """Main training function"""
    print("=" * 60)
    print("MemeCrafter Training Pipeline")
    print("=" * 60)
    
    # Set device
    device = torch.device(config.device)
    print(f"\nUsing device: {device}")
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    train_df, val_df, test_df = load_and_preprocess_data()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Initialize model
    print("\nInitializing MemeCrafter model...")
    model = MemeCrafterModel(
        use_lora=config.use_lora,
        freeze_vision=config.freeze_vision_encoder
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MemeDataset(train_df, model.processor, max_length=config.max_length)
    val_dataset = MemeDataset(val_df, model.processor, max_length=config.max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Mac M1/M2
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    
    # Setup optimizer
    print("\nSetting up optimizer and scheduler...")
    optimizer = AdamW(
        model.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    # Start training
    print("\n" + "=" * 60)
    trainer.train()
    print("=" * 60)
    print("Training complete!")


if __name__ == "__main__":
    main()
