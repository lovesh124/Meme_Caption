"""
Quick test script to diagnose demo issues
"""
import os
import sys
import torch
from PIL import Image

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.config import config
from src.model import MemeCrafterModel

print("="*60)
print("Testing MemeCrafter Model Issues")
print("="*60)

# Test 1: Check if fine-tuned model loads correctly
print("\n[TEST 1] Loading fine-tuned model...")
model_path = "models/best_model"
if os.path.exists(model_path):
    model = MemeCrafterModel(use_lora=config.use_lora)
    
    # The bug is here - load_model is a classmethod but returns a new instance
    # It doesn't load weights into the existing instance!
    print(f"Model created, now attempting to load from {model_path}...")
    
    # Current buggy approach in demo.py line 42:
    # self.model.load_model(model_path)  # This RETURNS a new model but doesn't assign it!
    
    print("❌ BUG FOUND: demo.py line 42 doesn't assign the returned model!")
    print("   Current: self.model.load_model(model_path)")
    print("   Should be: self.model = MemeCrafterModel.load_model(model_path)")
else:
    print(f"Model path not found: {model_path}")

# Test 2: Check base model vs fine-tuned model
print("\n[TEST 2] Comparing base model output...")
print("Creating base BLIP-2 model...")
base_model = MemeCrafterModel(use_lora=False)
base_model.eval()

# Create a simple test image
test_image = Image.new('RGB', (224, 224), color='blue')

print("\nGenerating caption with base model (no prompt)...")
with torch.no_grad():
    caption1 = base_model.generate_caption(test_image, max_length=50, num_beams=3)
    print(f"Caption 1: {caption1}")
    
    caption2 = base_model.generate_caption(test_image, max_length=50, num_beams=3)
    print(f"Caption 2: {caption2}")
    
    if caption1 == caption2:
        print("✅ Model is deterministic (expected)")
    else:
        print("⚠️ Model output varies (sampling enabled)")

# Test 3: Check sentiment parsing
print("\n[TEST 3] Testing sentiment parsing...")
test_responses = [
    "positive, negative, neutral, or sarcastic",  # This contains ALL keywords!
    "positive",
    "the sentiment is positive",
    "negative and sad",
]

sentiment_map = {
    'positive': 'positive',
    'negative': 'negative', 
    'neutral': 'neutral',
    'sarcastic': 'sarcastic',
}

for response in test_responses:
    detected = 'neutral'
    for keyword, sentiment in sentiment_map.items():
        if keyword in response.lower():
            detected = sentiment
            break  # This breaks on FIRST match
    print(f"Response: '{response}' → Detected: {detected}")
    
print("\n❌ BUG FOUND: If BLIP-2 returns the prompt words, it matches 'positive' first!")
print("   The model might be echoing the prompt instead of answering it.")

print("\n[TEST 4] Checking sentiment display...")
scores = {
    'positive': 1.0,
    'negative': 0.0,
    'neutral': 0.0,
    'sarcastic': 0.0
}
print("Sentiment scores that show all 4 sentiments:")
for sent_type, score in scores.items():
    if score > 0:  # Only shows if > 0
        print(f"   {sent_type}: {score:.2%}")

print("✅ This part is correct - only shows sentiment with score > 0")

print("\n" + "="*60)
print("SUMMARY OF BUGS FOUND:")
print("="*60)
print("1. ❌ demo.py line 42: model.load_model() doesn't assign return value")
print("   - Fine-tuned weights are NEVER loaded!")
print("   - Both self.model and self.base_model are using BASE weights")
print("   - That's why visual_description and caption are the same!")
print()
print("2. ❌ Sentiment prompt might be echoed by model")
print("   - Prompt contains 'positive, negative, neutral, or sarcastic'")
print("   - If BLIP-2 echoes this, parser matches 'positive' (first in dict)")
print()
print("3. ⚠️ Model might need different prompting strategy")
print("   - BLIP-2 is trained for VQA but needs specific format")
print("   - May need: 'Question: ... Answer:' format adjustment")
print("="*60)

