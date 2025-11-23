# 🎯 GPT-2 Fine-tuning Guide for Meme Captions

## Overview

This guide explains how to fine-tune GPT-2 to generate funnier, more meme-appropriate captions by learning from your dataset of 5,500+ memes.

---

## 🔄 Training Pipeline

```
Step 1: Prepare Data
├── Run BLIP v1 on all training images
├── Generate visual descriptions
├── Combine: OCR text + Visual description → Meme caption
└── Save training examples

Step 2: Fine-tune GPT-2
├── Load base GPT-2 (124M parameters)
├── Train on meme caption examples
├── Learn meme-style language patterns
└── Save fine-tuned model

Step 3: Use in Demo
├── Load fine-tuned GPT-2
├── Generate captions from OCR + Visual
└── Get funnier, meme-aware outputs!
```

---

## 📋 Prerequisites

Make sure you've already:
- ✅ Preprocessed your data: `python -m src.preprocess`
- ✅ Have `data/processed/train.csv` with 5,578 examples

---

## 🚀 Step-by-Step Instructions

### **Step 1: Prepare Training Data**

This step generates visual descriptions for all training images and creates GPT-2 training format.

```bash
cd "/Users/loveshkumar/Documents/GEN AI/final "
python -m src.prepare_gpt2_data
```

**What it does:**
1. Loads BLIP v1 model
2. Processes all 5,578 training images
3. Generates visual descriptions (e.g., "a person sitting at a desk")
4. Creates training examples in format:
   ```
   Meme image shows: [visual]. Text says: [OCR]. Meme caption: [actual caption]
   ```
5. Saves to `data/processed/gpt2_training.txt`

**Time:** ~1-2 hours (depending on your hardware)

**Output files:**
- `data/processed/train_with_visual.csv` - Training data with visual descriptions
- `data/processed/gpt2_training.txt` - GPT-2 training format (~5,500 examples)

---

### **Step 2: Fine-tune GPT-2**

Train GPT-2 on your meme captions.

```bash
python -m src.train_gpt2
```

**What it does:**
1. Loads base GPT-2 (124M parameters)
2. Loads training data (5,500+ examples)
3. Fine-tunes for 3 epochs
4. Saves checkpoints every 500 steps
5. Saves final model to `models/gpt2_meme_final/`

**Training parameters:**
- Epochs: 3
- Batch size: 4 (effective: 32 with gradient accumulation)
- Learning rate: 5e-5
- Optimizer: AdamW
- Warmup steps: 500

**Time estimates:**
- GPU (NVIDIA): 1-2 hours
- M1/M2 Mac: 3-4 hours
- CPU: 8-12 hours

**Output files:**
- `models/gpt2_meme_final/` - Fine-tuned model
- `models/gpt2_meme/` - Training checkpoints
- `results/gpt2_logs/` - Training logs

---

### **Step 3: Use Fine-tuned Model**

The demo will automatically detect and use the fine-tuned model!

```bash
python demo.py --model_path models/best_model --share
```

**What changes:**
- Caption generation now uses your fine-tuned GPT-2
- Captions will be more meme-appropriate
- Better understanding of meme humor patterns

---

## 📊 Training Data Format

### **Input Format:**
```
Meme image shows: a person sitting at a desk looking confused at a computer screen. 
Text says: When you code works but you don't know why. 
Meme caption: WHEN YOUR CODE WORKS BUT YOU DON'T KNOW WHY<|endoftext|>
```

### **What GPT-2 Learns:**
1. **Meme language patterns:**
   - ALL CAPS for emphasis
   - Short, punchy phrases
   - Internet slang and abbreviations

2. **Visual-text relationships:**
   - How image content relates to text
   - Context from both modalities

3. **Humor patterns:**
   - Sarcasm and irony
   - Relatable situations
   - Exaggeration for comedic effect

---

## 📈 Expected Improvements

### **Before Fine-tuning (Base GPT-2):**
```
Input: "Image shows: person at desk. Text: when your code works"
Output: "This is an image of a person working on a computer. 
         The text suggests they are programming."
```
❌ Generic, not meme-like

### **After Fine-tuning:**
```
Input: "Image shows: person at desk. Text: when your code works"
Output: "WHEN YOUR CODE WORKS BUT YOU DON'T KNOW WHY"
```
✅ Meme-style, captures humor, concise!

---

## 🔧 Configuration Options

### **Adjust Training in `src/train_gpt2.py`:**

```python
# More epochs for better learning (but risk overfitting)
num_train_epochs=5

# Larger batch size (if you have more GPU memory)
per_device_train_batch_size=8

# Lower learning rate for more stable training
learning_rate=3e-5
```

### **Adjust Data Format in `src/prepare_gpt2_data.py`:**

```python
# Include sentiment in prompt
prompt = f"Sentiment: {sentiment}. Image: {visual}. Text: {ocr}. Caption:"

# Different prompt templates
prompt = f"Meme: {ocr}. Scene: {visual}. Funny caption:"
```

---

## 💾 Disk Space Requirements

- Visual descriptions CSV: ~150 MB
- Training text file: ~50 MB
- Base GPT-2 model: ~500 MB
- Fine-tuned model: ~500 MB
- Training checkpoints: ~1.5 GB
- **Total: ~2.7 GB**

---

## 🐛 Troubleshooting

### **Problem: Out of memory during training**
**Solution:**
```python
# In src/train_gpt2.py, reduce batch size:
per_device_train_batch_size=2
gradient_accumulation_steps=16  # Keep effective batch size = 32
```

### **Problem: Training too slow**
**Solution:**
```python
# Reduce max_length in MemeGPT2Dataset:
max_length=100  # Default is 150
```

### **Problem: Checkpoints taking too much space**
**Solution:**
```python
# In TrainingArguments:
save_total_limit=2  # Keep only 2 checkpoints
```

### **Problem: GPT-2 generates repetitive text**
**Solution:**
```python
# In demo.py, adjust generation parameters:
no_repeat_ngram_size=3  # Already set
repetition_penalty=1.2  # Add this parameter
```

---

## 📊 Monitoring Training

### **Watch training progress:**
```bash
# In another terminal, monitor the log file:
tail -f results/gpt2_logs/train.log
```

### **Check GPU usage (if using GPU):**
```bash
nvidia-smi -l 1
```

### **Check training loss:**
Look for decreasing loss in the output:
```
{'loss': 2.5, 'learning_rate': 5e-05, 'epoch': 0.5}
{'loss': 2.1, 'learning_rate': 4.8e-05, 'epoch': 1.0}  ← Loss decreasing = good!
{'loss': 1.8, 'learning_rate': 4.5e-05, 'epoch': 1.5}
```

---

## 🎓 Advanced: Using LoRA for Efficient Training

For even faster training with less memory, you can use LoRA (Low-Rank Adaptation):

```python
# Add to src/train_gpt2.py (after loading model):
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.237
```

**Benefits:**
- ✅ Train only 0.24% of parameters
- ✅ 4x faster training
- ✅ Use less memory
- ✅ Smaller saved models (~2 MB vs 500 MB)

---

## 📝 Evaluation

After training, evaluate your model:

```python
# Create src/evaluate_gpt2.py
from src.prepare_gpt2_data import create_gpt2_training_data
import pandas as pd

# Generate captions for test set
test_df = pd.read_csv('data/processed/test.csv')
# ... generate captions with your fine-tuned model
# ... compare to ground truth captions
# ... calculate BLEU, ROUGE, etc.
```

---

## 🎉 Summary

**Complete training pipeline:**
```bash
# Step 1: Prepare data (1-2 hours)
python -m src.prepare_gpt2_data

# Step 2: Train GPT-2 (2-4 hours)
python -m src.train_gpt2

# Step 3: Test in demo
python demo.py --model_path models/best_model --share
```

**Total time:** 3-6 hours
**Result:** GPT-2 that generates meme-style captions!

---

## 📚 Files Created

```
src/
├── prepare_gpt2_data.py   # Data preparation script
└── train_gpt2.py          # Training script

data/processed/
├── train_with_visual.csv  # Training data + visual descriptions
└── gpt2_training.txt      # GPT-2 format training file

models/
├── gpt2_meme/            # Training checkpoints
└── gpt2_meme_final/      # Final fine-tuned model

results/
└── gpt2_logs/            # Training logs
```

---

## 🤝 Questions?

If training fails or you need help, check:
1. GPU memory usage
2. Training logs in `results/gpt2_logs/`
3. Checkpoint files in `models/gpt2_meme/`

Good luck with your training! 🚀

