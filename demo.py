"""
MemeCrafter Demo - Simple UI Version
Combines OCR text extraction with semantic analysis and sentiment understanding
"""
import os
import torch
import gradio as gr
from PIL import Image
import json
from datetime import datetime
import re

from src.config import config
from src.model import MemeCrafterModel
from src.ocr_utils import OCRExtractor


class MemeAnalyzer:
    """Main class for meme analysis combining OCR and BLIP-2"""
    
    def __init__(self, model_path=None, use_base_model=False):
        """
        Initialize the analyzer
        
        Args:
            model_path: Path to fine-tuned model (if None, uses base model)
            use_base_model: If True, load base BLIP-2 without fine-tuning
        """
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")
        
        # Initialize OCR
        print("Initializing OCR extractor...")
        self.ocr = OCRExtractor()
        
        # Initialize fine-tuned model
        print("Loading fine-tuned BLIP-2 model...")
        
        if model_path and not use_base_model:
            print(f"Loading fine-tuned model from {model_path}")
            self.model = MemeCrafterModel.load_model(model_path)
            self.is_finetuned = True
        else:
            print("Using base BLIP-2 model")
            self.model = MemeCrafterModel(use_lora=config.use_lora)
            self.is_finetuned = False
        
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
        
        # Initialize GPT-2 for meme caption generation
        print("Loading GPT-2 for caption generation...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        gpt2_finetuned_path = os.path.join(config.models_dir, "gpt2_meme_final")
        if os.path.exists(gpt2_finetuned_path):
            print(f"✓ Loading fine-tuned GPT-2 from {gpt2_finetuned_path}")
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_finetuned_path)
            self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_finetuned_path)
            self.is_gpt2_finetuned = True
        else:
            print("Fine-tuned GPT-2 not found, using base GPT-2")
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.is_gpt2_finetuned = False
        
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        self.gpt2_model.to(self.device)
        self.gpt2_model.eval()
    
    def analyze_meme(self, image):
        """Complete meme analysis pipeline"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        results = {}
        
        # 1. Extract OCR text
        ocr_result = self.ocr.extract_text(image, method='easyocr')
        results['ocr_text'] = ocr_result['text']
        results['ocr_confidence'] = ocr_result['confidence']
        
        # 2. Generate pure visual description
        inputs = self.base_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.base_blip.generate(
                pixel_values=pixel_values,
                max_length=100,
                num_beams=5
            )
        visual_description = self.base_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results['visual_description'] = visual_description
        
        # 3. Generate meme caption
        caption = self._generate_meme_caption_gpt2(ocr_result['text'], visual_description)
        results['generated_caption'] = caption
        
        # 4. Generate context-aware analysis
        if ocr_result['text']:
            prompt = f"This meme contains the text: '{ocr_result['text']}'. Analyze the sentiment and meaning:"
            context_analysis = self.model.generate_caption(
                image,
                prompt=prompt,
                max_length=100,
                num_beams=5,
                temperature=0.7
            )
            results['context_analysis'] = context_analysis
        else:
            results['context_analysis'] = "No text detected to analyze specifically."
        
        # 5. Analyze sentiment
        sentiment_scores = self._analyze_sentiment_blip2(image, ocr_result['text'])
        results['sentiment'] = sentiment_scores
        
        return results
    
    def _generate_meme_caption_gpt2(self, ocr_text, visual_description):
        """Generate meme caption using GPT-2"""
        if ocr_text and visual_description:
            prompt = f"Meme image shows: {visual_description}. Text says: {ocr_text}. Meme caption:"
        elif ocr_text:
            prompt = f"Meme text: {ocr_text}. Caption:"
        elif visual_description:
            prompt = f"Image shows: {visual_description}. Meme caption:"
        else:
            prompt = "Meme caption:"
        
        inputs = self.gpt2_tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens = 40,
                max_length=len(inputs.input_ids[0]) + 50,
                num_beams=3,
                temperature=0.7,
                top_k=40,
                top_p=0.90,
                do_sample=True,
                no_repeat_ngram_size=2,
                pad_token_id=self.gpt2_tokenizer.eos_token_id
            )
        
        full_text = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        caption = full_text[len(prompt):].strip()
        
        if not caption or len(caption) < 5:
            caption = full_text.strip()
        
        # Clean up caption
        website_patterns = [
            r'\b(memegenerator|memecenter|memebox|9gag|imgflip|knowyourmeme)\.?(com|net)?\b',
            r'http[s]?://\S+',
        ]
        for pattern in website_patterns:
            caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
        caption = re.sub(r'\s+', ' ', caption).strip()

        return caption
    
    def _analyze_sentiment_blip2(self, image, ocr_text):
        """Use BLIP-2 to analyze sentiment"""
        if ocr_text:
            sentiment_prompt = f"This meme says: '{ocr_text}'. Question: What is the sentiment? Answer:"
        else:
            sentiment_prompt = "Question: What is the sentiment of this image? Answer:"
        
        sentiment_response = self.model.generate_caption(
            image, 
            prompt=sentiment_prompt, 
            max_length=100,
            num_beams=3,
            temperature=0.3,
            do_sample=False
        ).lower().strip()
        
        # Basic sentiment mapping
        sentiment_keywords = {
            'positive': ['positive', 'happy', 'funny', 'humorous', 'good', 'joy'],
            'negative': ['negative', 'sad', 'angry', 'bad', 'upset'],
            'neutral': ['neutral', 'normal', 'calm'],
            'sarcastic': ['sarcastic', 'ironic', 'satirical']
        }
        
        detected_sentiment = 'neutral'
        for sent, kws in sentiment_keywords.items():
            if any(kw in sentiment_response for kw in kws):
                detected_sentiment = sent
                break
                
        if ocr_text:
            explanation_prompt = f"This meme shows: '{ocr_text}'. Question: Describe the mood. Answer:"
        else:
            explanation_prompt = "Question: Describe the mood of this image. Answer:"
            
        explanation = self.model.generate_caption(
            image, prompt=explanation_prompt, max_length=100, num_beams=3
        )
        
        return {
            'overall': detected_sentiment,
            'explanation': explanation,
            'raw_response': sentiment_response
        }

# ==========================================
# SIMPLIFIED GRADIO INTERFACE
# ==========================================

def create_demo(model_path=None):
    """Create a simplified Gradio demo interface"""
    
    # Initialize analyzer
    analyzer = MemeAnalyzer(model_path=model_path)
    
    def simple_analyze(image):
        """Analyze uploaded image and return a single formatted Markdown string"""
        if image is None:
            return "Please upload an image first."
        
        try:
            # Run analysis
            r = analyzer.analyze_meme(image)
            
            # Formatting helpers
            emoji_map = {
                'positive': '😂 Positive/Funny',
                'negative': '😤 Negative/Sad',
                'neutral': '😐 Neutral',
                'sarcastic': '😏 Sarcastic/Ironic'
            }
            sentiment_display = emoji_map.get(r['sentiment']['overall'], r['sentiment']['overall'].title())
            
            ocr_text = r['ocr_text'] if r['ocr_text'] else "*(No text detected)*"
            
            # Construct Markdown Report
            markdown_output = f"""
# Analysis Results

### **Generated Caption**
> **"{r['generated_caption']}"**

---

### **Deep Dive**

**1. What the meme says (OCR)**
`{ocr_text}`

**2. What we see (Visuals)**
{r['visual_description']}

**3. Meaning & Context**
{r['context_analysis']}

---

### **Vibe Check**
**Mood:** {sentiment_display}

**Explanation:** _{r['sentiment']['explanation']}_
            """
            return markdown_output
        
        except Exception as e:
            return f"### ⚠️ Error \nAn error occurred during analysis: {str(e)}"

    # Create Simple Interface
    with gr.Blocks(title="MemeCrafter", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(
            """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                <h1>MemeCrafter</h1>
                <p style="font-size: 1.1em;">
                    Upload a meme below. The AI will read the text, look at the image, 
                    and explain the joke and sentiment.
                </p>
            </div>
            """
        )
        
        with gr.Row():
            # Left Column: Input
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil", 
                    label="Upload Meme", 
                    height=400
                )
                analyze_btn = gr.Button("Analyze Meme", variant="primary", size="lg")
            
            # Right Column: Output
            with gr.Column(scale=1):
                output_box = gr.Markdown(label="Results")
        
        # Footer
        gr.Markdown(
            """
            <div style="text-align: center; color: #888; margin-top: 20px;">
                Model: Fine-tuned BLIP-2 | Text Extraction: EasyOCR | Captioning: GPT-2
            </div>
            """
        )

        # Connect button to function
        analyze_btn.click(
            fn=simple_analyze,
            inputs=image_input,
            outputs=output_box
        )
    
    return demo


def main():
    """Main function to launch demo"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    
    if args.model_path and not os.path.exists(args.model_path):
        print("Warning: Model path not found. Using base model.")
        args.model_path = None
    
    demo = create_demo(model_path=args.model_path)
    demo.launch(server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()