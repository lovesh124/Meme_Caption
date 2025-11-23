"""
Prepare training data for GPT-2 meme caption generation
Run BLIP v1 on all images to get visual descriptions, then combine with OCR
"""
import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from src.config import config


def generate_visual_descriptions():
    """Generate visual descriptions for all images using BLIP v1"""
    
    # Load BLIP v1
    print("Loading BLIP v1 for visual descriptions...")
    device = torch.device(config.device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if config.device != "cpu" else torch.float32,
    )
    model.to(device)
    model.eval()
    
    # Load training data
    train_df = pd.read_csv(os.path.join(config.processed_data_dir, 'train.csv'))
    
    print(f"\nGenerating visual descriptions for {len(train_df)} images...")
    print("This may take a while (~1 hour for 5,500 images)...")
    
    visual_descriptions = []
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_path = os.path.join(config.images_dir, row['image_name'])
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Generate visual description
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values=pixel_values,
                    max_length=50,
                    num_beams=5
                )
            
            visual_desc = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            visual_descriptions.append(visual_desc)
            
        except Exception as e:
            print(f"\nError processing {row['image_name']}: {e}")
            visual_descriptions.append("")
        
        # Save checkpoint every 1000 images
        if (idx + 1) % 1000 == 0:
            temp_df = train_df.iloc[:idx+1].copy()
            temp_df['visual_description'] = visual_descriptions[:idx+1]
            checkpoint_path = os.path.join(config.processed_data_dir, f'train_visual_checkpoint_{idx+1}.csv')
            temp_df.to_csv(checkpoint_path, index=False)
            print(f"\n✓ Checkpoint saved at {idx+1} images")
    
    # Add to dataframe
    train_df['visual_description'] = visual_descriptions
    
    # Save
    output_path = os.path.join(config.processed_data_dir, 'train_with_visual.csv')
    train_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")
    
    return train_df


def create_gpt2_training_data(train_df=None):
    """
    Create training examples in the format:
    Input: "Meme image shows: [visual]. Text says: [ocr]. Meme caption:"
    Output: [actual caption]
    """
    
    # Load dataframe if not provided
    if train_df is None:
        visual_data_path = os.path.join(config.processed_data_dir, 'train_with_visual.csv')
        if not os.path.exists(visual_data_path):
            print(f"Error: {visual_data_path} not found!")
            print("Run generate_visual_descriptions() first.")
            return None
        train_df = pd.read_csv(visual_data_path)
    
    print(f"\nCreating GPT-2 training data from {len(train_df)} examples...")
    
    training_examples = []
    skipped = 0
    
    for idx, row in train_df.iterrows():
        visual_desc = str(row.get('visual_description', '')).strip()
        ocr_text = str(row.get('text_corrected', '') or row.get('text_ocr', '')).strip()
        caption = str(row.get('caption', '')).strip()
        
        # Skip if no caption
        if not caption or len(caption) < 5:
            skipped += 1
            continue
        
        # Create prompt based on available information
        if visual_desc and ocr_text:
            prompt = f"Meme image shows: {visual_desc}. Text says: {ocr_text}. Meme caption:"
        elif ocr_text:
            prompt = f"Meme text: {ocr_text}. Meme caption:"
        elif visual_desc:
            prompt = f"Image shows: {visual_desc}. Meme caption:"
        else:
            skipped += 1
            continue  # Skip if no context
        
        # Create training example with end token
        training_example = f"{prompt} {caption}<|endoftext|>"
        training_examples.append(training_example)
    
    print(f"Created {len(training_examples)} training examples")
    print(f"Skipped {skipped} examples (no caption or context)")
    
    # Save to text file
    output_path = os.path.join(config.processed_data_dir, 'gpt2_training.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(example + '\n')
    
    print(f"✅ Saved to {output_path}")
    
    # Show some examples
    print("\n📝 Sample training examples:")
    print("="*80)
    for i in range(min(3, len(training_examples))):
        print(f"\nExample {i+1}:")
        print(training_examples[i][:200] + "..." if len(training_examples[i]) > 200 else training_examples[i])
    print("="*80)
    
    return training_examples


def main():
    """Main function to prepare all GPT-2 training data"""
    print("="*80)
    print("GPT-2 Meme Caption Training Data Preparation")
    print("="*80)
    
    # Check if visual descriptions already exist
    visual_data_path = os.path.join(config.processed_data_dir, 'train_with_visual.csv')
    
    if os.path.exists(visual_data_path):
        print(f"\n✓ Found existing visual descriptions at {visual_data_path}")
        response = input("Use existing visual descriptions? (y/n): ")
        if response.lower() == 'y':
            train_df = pd.read_csv(visual_data_path)
        else:
            print("\nRegenerating visual descriptions...")
            train_df = generate_visual_descriptions()
    else:
        print("\nGenerating visual descriptions (Step 1/2)...")
        train_df = generate_visual_descriptions()
    
    # Create GPT-2 training format
    print("\nCreating GPT-2 training format (Step 2/2)...")
    create_gpt2_training_data(train_df)
    
    print("\n" + "="*80)
    print("✅ Data preparation complete!")
    print("="*80)
    print("\nNext step: Train GPT-2")
    print("Run: python -m src.train_gpt2")
    print("="*80)


if __name__ == "__main__":
    main()

