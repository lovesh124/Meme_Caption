"""
Data preprocessing script for MemeCrafter
Handles loading, cleaning, and splitting the meme dataset
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

class MemeDataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.labels_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_data(self):
        """Load the labels CSV file"""
        print("Loading data...")
        self.labels_df = pd.read_csv(self.config.labels_file)
        print(f"Loaded {len(self.labels_df)} samples")
        return self.labels_df
    
    def clean_text(self, text):
        """Clean and preprocess text captions"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def verify_images(self):
        """Verify that all images exist and are valid"""
        print("Verifying images...")
        valid_indices = []
        
        for idx, row in tqdm(self.labels_df.iterrows(), total=len(self.labels_df)):
            image_path = os.path.join(self.config.images_dir, row['image_name'])
            
            # Check if image exists
            if not os.path.exists(image_path):
                continue
                
            # Try to open and verify the image
            try:
                img = Image.open(image_path)
                img.verify()
                valid_indices.append(idx)
            except Exception as e:
                continue
        
        # Keep only valid samples
        self.labels_df = self.labels_df.iloc[valid_indices].reset_index(drop=True)
        print(f"Valid samples: {len(self.labels_df)}")
        
    def preprocess_captions(self):
        """Clean and preprocess all captions"""
        print("Preprocessing captions...")
        
        # Use text_corrected as the primary caption source
        self.labels_df['caption'] = self.labels_df['text_corrected'].apply(self.clean_text)
        
        # If caption is empty, fall back to text_ocr
        empty_mask = self.labels_df['caption'].str.len() == 0
        self.labels_df.loc[empty_mask, 'caption'] = self.labels_df.loc[empty_mask, 'text_ocr'].apply(self.clean_text)
        
        # Filter out samples with empty captions or very short captions
        min_length = 5
        valid_mask = self.labels_df['caption'].str.len() >= min_length
        self.labels_df = self.labels_df[valid_mask].reset_index(drop=True)
        
        print(f"Samples after caption filtering: {len(self.labels_df)}")
        
    def create_splits(self):
        """Split data into train, validation, and test sets"""
        print("Creating data splits...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            self.labels_df,
            test_size=self.config.test_ratio,
            random_state=self.config.random_seed,
            stratify=self.labels_df['overall_sentiment'] if 'overall_sentiment' in self.labels_df.columns else None
        )
        
        # Second split: train vs val
        val_ratio_adjusted = self.config.val_ratio / (1 - self.config.test_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=self.config.random_seed,
            stratify=train_val_df['overall_sentiment'] if 'overall_sentiment' in train_val_df.columns else None
        )
        
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        
        print(f"Train samples: {len(self.train_df)}")
        print(f"Validation samples: {len(self.val_df)}")
        print(f"Test samples: {len(self.test_df)}")
        
    def save_processed_data(self):
        """Save processed data to disk"""
        print("Saving processed data...")
        
        # Save dataframes
        self.train_df.to_csv(os.path.join(self.config.processed_data_dir, 'train.csv'), index=False)
        self.val_df.to_csv(os.path.join(self.config.processed_data_dir, 'val.csv'), index=False)
        self.test_df.to_csv(os.path.join(self.config.processed_data_dir, 'test.csv'), index=False)
        
        # Save statistics
        stats = {
            'total_samples': len(self.labels_df),
            'train_samples': len(self.train_df),
            'val_samples': len(self.val_df),
            'test_samples': len(self.test_df),
            'avg_caption_length': float(self.labels_df['caption'].str.len().mean()),
            'sentiment_distribution': self.labels_df['overall_sentiment'].value_counts().to_dict() if 'overall_sentiment' in self.labels_df.columns else {}
        }
        
        with open(os.path.join(self.config.processed_data_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("Processed data saved successfully!")
        
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        self.load_data()
        self.verify_images()
        self.preprocess_captions()
        self.create_splits()
        self.save_processed_data()
        
        return self.train_df, self.val_df, self.test_df


def load_and_preprocess_data():
    """Helper function to load preprocessed data or run preprocessing"""
    from src.config import config
    
    # Check if processed data exists
    train_path = os.path.join(config.processed_data_dir, 'train.csv')
    val_path = os.path.join(config.processed_data_dir, 'val.csv')
    test_path = os.path.join(config.processed_data_dir, 'test.csv')
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Loading existing preprocessed data...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df
    else:
        print("Preprocessed data not found. Running preprocessing...")
        preprocessor = MemeDataPreprocessor(config)
        return preprocessor.run_preprocessing()


if __name__ == "__main__":
    from src.config import config
    
    preprocessor = MemeDataPreprocessor(config)
    train_df, val_df, test_df = preprocessor.run_preprocessing()
    
    print("\nPreprocessing complete!")
