"""
PyTorch Dataset class for MemeCrafter with BLIP-2
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class MemeDataset(Dataset):
    def __init__(self, csv_path_or_df, processor, max_length=50):
        """
        Args:
            csv_path_or_df: Path to CSV file or pandas DataFrame
            processor: BLIP-2 processor
            max_length: Maximum length for text tokenization
        """
        # Load data
        if isinstance(csv_path_or_df, str):
            self.df = pd.read_csv(csv_path_or_df)
        else:
            self.df = csv_path_or_df
            
        self.processor = processor
        self.max_length = max_length
        
        # Get images directory from config
        from src.config import config
        self.images_dir = config.images_dir
        
        print(f"Loaded {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name and caption
        img_name = self.df.iloc[idx]['image_name']
        caption = self.df.iloc[idx]['caption']
        
        # Load and preprocess image
        img_path = os.path.join(self.images_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        # Process image and text with BLIP-2 processor
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Create labels from input_ids for language modeling
        if 'input_ids' in encoding:
            labels = encoding['input_ids'].clone()
            # Set padding tokens to -100 so they're ignored in loss calculation
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            encoding['labels'] = labels
        
        # Add metadata
        encoding['caption'] = caption
        encoding['image_name'] = img_name
        
        return encoding


def collate_fn(batch):
    """Custom collate function for batching"""
    # Collect all keys except metadata
    keys_to_stack = ['pixel_values', 'input_ids', 'attention_mask', 'labels']
    collated = {}
    
    for key in keys_to_stack:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
    
    # Collect metadata as lists
    collated['captions'] = [item['caption'] for item in batch]
    collated['image_names'] = [item['image_name'] for item in batch]
    
    return collated
