#!/bin/bash

# GPT-2 Fine-tuning Pipeline
# Complete script to prepare data and train GPT-2 for meme captions

echo "========================================================================"
echo "  GPT-2 Meme Caption Fine-tuning Pipeline"
echo "========================================================================"
echo ""

# Navigate to project directory
cd "/Users/loveshkumar/Documents/GEN AI/final "

echo "📁 Current directory: $(pwd)"
echo ""

# Check if processed data exists
if [ ! -f "data/processed/train.csv" ]; then
    echo "❌ Error: Preprocessed data not found!"
    echo "Please run: python -m src.preprocess"
    exit 1
fi

echo "✓ Found preprocessed training data"
echo ""

# Step 1: Prepare GPT-2 training data
echo "========================================================================"
echo "Step 1/2: Preparing GPT-2 Training Data"
echo "========================================================================"
echo "This will:"
echo "  - Generate visual descriptions for all training images (~1-2 hours)"
echo "  - Create GPT-2 training format"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

python3 -m src.prepare_gpt2_data

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Data preparation failed!"
    exit 1
fi

echo ""
echo "✅ Data preparation complete!"
echo ""

# Step 2: Train GPT-2
echo "========================================================================"
echo "Step 2/2: Training GPT-2"
echo "========================================================================"
echo "This will fine-tune GPT-2 on your meme dataset (~2-4 hours)"
echo ""
read -p "Continue with training? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

python3 -m src.train_gpt2

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "✅ Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Your fine-tuned GPT-2 model is ready!"
echo ""
echo "📁 Model location: models/gpt2_meme_final/"
echo ""
echo "🚀 Next step: Test your model"
echo "   python demo.py --model_path models/best_model --share"
echo ""
echo "========================================================================"

