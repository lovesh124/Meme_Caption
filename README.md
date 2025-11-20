# 🎭 MemeCrafter - Meme Analysis with BLIP-2 and LoRA

Advanced meme understanding system that combines OCR text extraction with fine-tuned BLIP-2 for semantic analysis and sentiment understanding.

## 🌟 Features

- **OCR Text Extraction**: Extracts text from memes using EasyOCR
- **Fine-tuned BLIP-2**: Image captioning and understanding specifically trained on memes
- **Sentiment Analysis**: Understands positive, negative, neutral, and sarcastic sentiments
- **Context-Aware Analysis**: Combines visual and textual elements for deeper understanding
- **Model Comparison**: Compare base BLIP-2 vs fine-tuned model performance
- **Interactive Demo**: Beautiful Gradio UI for easy testing

## 📋 Requirements

- Python 3.8+
- macOS (optimized for M1/M2) or Linux
- 16GB+ RAM recommended
- GPU optional (works on CPU/MPS)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract (for OCR)

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### 3. Prepare Your Data

Ensure your data structure looks like this:
```
archive (4)/
├── labels.csv          # Contains columns: image_name, text_corrected, text_ocr, overall_sentiment
└── images/
    └── images/
        ├── image_1.jpg
        ├── image_2.png
        └── ...
```

### 4. Preprocess Data

```bash
python -m src.preprocess
```

This will:
- Clean and validate images
- Process captions
- Split data into train/val/test sets
- Save to `data/processed/`

### 5. Train the Model

```bash
python -m src.train
```

Training parameters can be configured in `src/config.py`:
- `num_epochs`: Number of training epochs (default: 5)
- `batch_size`: Batch size for training (default: 4)
- `learning_rate`: Learning rate (default: 5e-5)
- `use_lora`: Use LoRA for efficient fine-tuning (default: True)

The best model will be saved to `models/best_model/`

### 6. Run the Demo

**With fine-tuned model:**
```bash
python demo.py --model_path models/best_model
```

**With base model (no fine-tuning):**
```bash
python demo.py
```

**Create a public link:**
```bash
python demo.py --model_path models/best_model --share
```

The demo will be available at `http://localhost:7860`

## 🎯 Demo Features

### 🔍 Analyze Meme Tab
Upload a meme and get:
- **OCR Results**: Extracted text with confidence scores
- **Generated Caption**: Visual description from BLIP-2
- **Context Analysis**: Semantic understanding combining text and visuals
- **Sentiment Analysis**: Sentiment breakdown with confidence scores
- **Complete Summary**: Comprehensive analysis report

### ⚖️ Compare Models Tab
Upload a meme and compare:
- Base BLIP-2 model vs Your fine-tuned model
- See how fine-tuning improves understanding
- Side-by-side sentiment comparison
- Caption quality comparison

## 📁 Project Structure

```
.
├── src/
│   ├── config.py          # Configuration settings
│   ├── model.py           # BLIP-2 model with LoRA
│   ├── dataset.py         # PyTorch Dataset
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation metrics
│   └── ocr_utils.py       # OCR utilities
├── demo.py                # Gradio demo app
├── requirements.txt       # Dependencies
├── data/                  # Processed data
├── models/                # Saved models
├── results/               # Training results
└── notebooks/             # Jupyter notebooks
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model settings
model_name = "Salesforce/blip2-opt-2.7b"
use_lora = True
lora_r = 16
lora_alpha = 32

# Training settings
num_epochs = 5
batch_size = 4
learning_rate = 5e-5
max_length = 50

# Data settings
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
```

## 📊 Training Tips

1. **Start Small**: Begin with a smaller dataset to test the pipeline
2. **Monitor Loss**: Check `results/training_history.json` for loss curves
3. **GPU Memory**: Reduce `batch_size` if you run out of memory
4. **LoRA Benefits**: LoRA reduces memory usage and training time significantly
5. **Evaluation**: Run evaluation after training to measure performance

## 🎓 How It Works

### Fine-tuning Process
1. **Base Model**: Starts with pre-trained BLIP-2 (Salesforce)
2. **LoRA Adaptation**: Adds small trainable adapters to the model
3. **Dataset Training**: Trains on your meme dataset with captions
4. **Optimization**: Uses AdamW optimizer with linear warmup

### Demo Analysis Pipeline
1. **OCR Extraction**: EasyOCR extracts text from the meme
2. **Visual Captioning**: BLIP-2 generates image description
3. **Context Analysis**: Combines OCR text with visual understanding
4. **Sentiment Detection**: Analyzes sentiment using keyword matching and context
5. **Summary Generation**: Creates comprehensive analysis report

## 🚀 Advanced Usage

### Evaluate Model Performance

```bash
python -m src.evaluate --model_path models/best_model
```

### Train with Custom Settings

```python
# Modify src/config.py first, then:
python -m src.train
```

### Export Model for Production

```python
from src.model import MemeCrafterModel

model = MemeCrafterModel()
model.load_model("models/best_model")
model.save_model("models/production")
```

## 📈 Expected Results

With fine-tuning, you should see:
- ✅ Better understanding of meme-specific language
- ✅ Improved sentiment detection (especially sarcasm)
- ✅ More accurate context-aware analysis
- ✅ Better handling of text-image relationships
- ✅ Understanding of meme formats and cultural references

## 🐛 Troubleshooting

**Issue**: Out of memory during training
- **Solution**: Reduce `batch_size` in `config.py`

**Issue**: OCR not working
- **Solution**: Install Tesseract: `brew install tesseract`

**Issue**: Slow training
- **Solution**: Ensure you're using GPU/MPS acceleration

**Issue**: Model not loading
- **Solution**: Check the model path exists and contains all required files

## 📚 References

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://github.com/huggingface/peft)

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to open issues or submit pull requests!

---

Made with ❤️ using PyTorch, Hugging Face, and Gradio
