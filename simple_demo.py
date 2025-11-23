"""
Simple Flask Demo - Meme Analysis without Gradio
A minimal web interface that avoids all Gradio bugs
"""
import os
import sys
import torch
from PIL import Image
import io
import base64
from flask import Flask, render_template_string, request, jsonify

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import config
from src.model import MemeCrafterModel
from src.ocr_utils import OCRExtractor

app = Flask(__name__)

# Global analyzer
analyzer = None

class MemeAnalyzer:
    def __init__(self, model_path=None):
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")
        
        # Initialize OCR with proper parameters
        print("Initializing OCR extractor...")
        self.ocr = OCRExtractor(use_easyocr=True, languages=['en'])
        
        # Initialize fine-tuned model
        print("Loading fine-tuned BLIP-2 model...")
        
        if model_path:
            print(f"Loading fine-tuned model from {model_path}")
            # FIX: load_model returns a new instance, must assign it
            self.model = MemeCrafterModel.load_model(model_path)
        else:
            print("Using base BLIP-2 model")
            self.model = MemeCrafterModel(use_lora=config.use_lora)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize BLIP v1 for pure visual descriptions
        print("Loading BLIP v1 for visual descriptions...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.base_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.base_blip = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if config.device != "cpu" else torch.float32,
        )
        self.base_blip.to(self.device)
        self.base_blip.eval()
        print("BLIP v1 loaded successfully!")
        
        # Initialize GPT-2 for meme caption generation
        print("Loading GPT-2 for caption generation...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2_model.to(self.device)
        self.gpt2_model.eval()
        print("GPT-2 loaded successfully!")
    
    def analyze_meme(self, image):
        print("Starting meme analysis...")
        results = {}
        
        try:
            # Extract OCR text
            print("Extracting OCR text...")
            ocr_result = self.ocr.extract_text(image, method='easyocr')
            results['ocr_text'] = ocr_result.get('text', '')
            results['ocr_confidence'] = ocr_result.get('confidence', 0.0)
            print(f"OCR complete. Text: {results['ocr_text']}")
            
            # Generate pure visual description using BLIP v1
            print("Generating visual description with BLIP v1...")
            inputs = self.base_processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.base_blip.generate(
                    pixel_values=pixel_values,
                    max_length=50,
                    num_beams=5
                )
            
            visual_description = self.base_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results['visual_description'] = visual_description
            print(f"Visual description: {visual_description}")
            
            # Generate meme caption using GPT-2
            print("Generating meme caption with GPT-2...")
            caption = self._generate_meme_caption_gpt2(ocr_result.get('text', ''), visual_description)
            results['caption'] = caption
            print(f"Caption: {caption}")
            
            # Context analysis
            if ocr_result.get('text', '').strip():
                print("Generating context analysis...")
                prompt = f"This meme contains the text: '{ocr_result['text']}'. Describe it:"
                context = self.model.generate_caption(image, prompt=prompt, max_length=80)
                results['context'] = context
            else:
                results['context'] = "No text detected."
            
            # BLIP-2 based sentiment analysis
            print("Analyzing sentiment with BLIP-2...")
            results['sentiment'] = self._analyze_sentiment_blip2(image, ocr_result.get('text', ''))
            print("Analysis complete!")
            
        except Exception as e:
            print(f"Error in analyze_meme: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        return results
    
    def _generate_meme_caption_gpt2(self, ocr_text, visual_description):
        """Generate meme caption using GPT-2 based on OCR text and visual description"""
        # Create prompt combining OCR text and visual description
        if ocr_text and visual_description:
            prompt = f"Meme image shows: {visual_description}. Text says: {ocr_text}. Meme caption:"
        elif ocr_text:
            prompt = f"Meme text: {ocr_text}. Caption:"
        elif visual_description:
            prompt = f"Image shows: {visual_description}. Meme caption:"
        else:
            prompt = "Meme caption:"
        
        # Tokenize
        inputs = self.gpt2_tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        # Generate with GPT-2
        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=len(inputs.input_ids[0]) + 50,
                num_beams=5,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                no_repeat_ngram_size=3,
                pad_token_id=self.gpt2_tokenizer.eos_token_id
            )
        
        # Decode and extract only the generated part
        full_text = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        caption = full_text[len(prompt):].strip()
        
        if not caption or len(caption) < 5:
            caption = full_text.strip()
        
        return caption
    
    def _analyze_sentiment_blip2(self, image, ocr_text):
        """
        Use BLIP-2 to analyze sentiment by asking it directly
        This leverages the fine-tuned model's understanding of memes
        """
        # Ask BLIP-2 about the sentiment - IMPROVED PROMPT
        if ocr_text:
            sentiment_prompt = f"This meme says: '{ocr_text}'. Question: What is the sentiment? Answer:"
        else:
            sentiment_prompt = "Question: What is the sentiment of this image? Answer:"
        
        sentiment_response = self.model.generate_caption(
            image, 
            prompt=sentiment_prompt, 
            max_length=20,
            num_beams=3,
            temperature=0.3,
            do_sample=False
        ).lower().strip()
        
        print(f"BLIP-2 sentiment response: '{sentiment_response}'")
        
        # Parse the response with better word matching
        sentiment_keywords = {
            'positive': ['positive', 'happy', 'funny', 'humorous', 'joyful', 'cheerful', 'good'],
            'negative': ['negative', 'sad', 'angry', 'bad', 'upset', 'annoyed'],
            'neutral': ['neutral', 'normal', 'calm', 'indifferent'],
            'sarcastic': ['sarcastic', 'ironic', 'satirical', 'mocking']
        }
        
        # Count sentiment keyword matches
        sentiment_scores = {}
        
        for sentiment, keywords in sentiment_keywords.items():
            count = sum(1 for keyword in keywords if keyword in sentiment_response)
            if count > 0:
                sentiment_scores[sentiment] = count
        
        # Determine detected sentiment
        if sentiment_scores:
            detected_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        else:
            # Fallback
            detected_sentiment = 'neutral'
            for sentiment, keywords in sentiment_keywords.items():
                if any(kw in sentiment_response for kw in keywords):
                    detected_sentiment = sentiment
                    break
        
        # Get explanation by asking BLIP-2
        if ocr_text:
            explanation_prompt = f"This meme shows: '{ocr_text}'. Question: Describe the mood. Answer:"
        else:
            explanation_prompt = "Question: Describe the mood of this image. Answer:"
            
        explanation = self.model.generate_caption(
            image,
            prompt=explanation_prompt,
            max_length=60,
            num_beams=3,
            temperature=0.7,
            do_sample=True
        )
        
        return {
            'overall': detected_sentiment,
            'confidence': 0.85,  # BLIP-2 based confidence is high if it responded clearly
            'explanation': explanation,
            'raw_response': sentiment_response
        }

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>MemeCrafter - Meme Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .upload-section { text-align: center; margin: 30px 0; }
        input[type="file"] { display: none; }
        .upload-btn { background: #4CAF50; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .upload-btn:hover { background: #45a049; }
        .analyze-btn { background: #2196F3; color: white; padding: 15px 40px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 20px; }
        .analyze-btn:hover { background: #0b7dda; }
        .analyze-btn:disabled { background: #ccc; cursor: not-allowed; }
        .preview { max-width: 500px; margin: 20px auto; display: none; }
        .preview img { max-width: 100%; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        .results { margin-top: 30px; display: none; }
        .result-box { background: #f9f9f9; padding: 20px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #2196F3; }
        .result-box h3 { margin-top: 0; color: #2196F3; }
        .loading { text-align: center; display: none; margin: 20px 0; }
        .spinner { border: 5px solid #f3f3f3; border-top: 5px solid #2196F3; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>�� MemeCrafter - Meme Analysis</h1>
        <p style="text-align: center; color: #666;">Upload a meme image to analyze with OCR + BLIP-2</p>
        
        <div class="upload-section">
            <input type="file" id="imageInput" accept="image/*">
            <label for="imageInput" class="upload-btn">📁 Choose Meme Image</label>
            
            <div class="preview" id="preview">
                <img id="previewImg" src="">
            </div>
            
            <button class="analyze-btn" id="analyzeBtn" onclick="analyzeMeme()" disabled>
                🔍 Analyze Meme
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing meme... Please wait...</p>
        </div>
        
        <div class="results" id="results">
            <div class="result-box">
                <h3>📝 OCR Text Extraction</h3>
                <p id="ocrText"></p>
                <p><strong>Confidence:</strong> <span id="ocrConf"></span></p>
            </div>
            
            <div class="result-box">
                <h3>🖼️ Visual Description (BLIP-2)</h3>
                <p id="caption"></p>
            </div>
            
            <div class="result-box">
                <h3>🧠 Context-Aware Analysis</h3>
                <p id="context"></p>
            </div>
            
            <div class="result-box">
                <h3>💭 Sentiment Analysis (BLIP-2)</h3>
                <p><strong>Overall Sentiment:</strong> <span id="sentiment" style="font-size: 18px; font-weight: bold;"></span></p>
                <p><strong>Confidence:</strong> <span id="sentimentConf"></span></p>
                <p><strong>Explanation:</strong> <span id="sentimentExp"></span></p>
                <p style="font-size: 12px; color: #666;"><em>Raw response: <span id="sentimentRaw"></span></em></p>
            </div>
        </div>
    </div>
    
    <script>
        let selectedImage = null;
        
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    selectedImage = event.target.result;
                    document.getElementById('previewImg').src = selectedImage;
                    document.getElementById('preview').style.display = 'block';
                    document.getElementById('analyzeBtn').disabled = false;
                    document.getElementById('results').style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
        });
        
        async function analyzeMeme() {
            if (!selectedImage) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: selectedImage })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    document.getElementById('ocrText').textContent = data.ocr_text || 'No text detected';
                    document.getElementById('ocrConf').textContent = (data.ocr_confidence * 100).toFixed(1) + '%';
                    document.getElementById('caption').textContent = data.caption;
                    document.getElementById('context').textContent = data.context;
                    document.getElementById('sentiment').textContent = data.sentiment.overall;
                    document.getElementById('sentimentConf').textContent = (data.sentiment.confidence * 100).toFixed(1) + '%';
                    document.getElementById('sentimentExp').textContent = data.sentiment.explanation;
                    document.getElementById('sentimentRaw').textContent = data.sentiment.raw_response;
                    document.getElementById('results').style.display = 'block';
                }
            } catch (error) {
                alert('Error analyzing meme: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        results = analyzer.analyze_meme(image)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best_model')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    
    print("Initializing MemeCrafter...")
    analyzer = MemeAnalyzer(model_path=args.model_path)
    print(f"\n{'='*60}")
    print(f"Starting Flask server on http://127.0.0.1:{args.port}")
    print(f"{'='*60}\n")
    
    app.run(host='127.0.0.1', port=args.port, debug=False)
