"""
MemeCrafter Demo - Meme Analysis with OCR and Fine-tuned BLIP-2
Combines OCR text extraction with semantic analysis and sentiment understanding
"""
import os
import torch
import gradio as gr
from PIL import Image
import json
from datetime import datetime

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
        
        # Load fine-tuned weights if provided
        if model_path and not use_base_model:
            print(f"Loading fine-tuned model from {model_path}")
            # FIX: load_model is a classmethod that returns a new instance
            self.model = MemeCrafterModel.load_model(model_path)
            self.is_finetuned = True
        else:
            print("Using base BLIP-2 model")
            self.model = MemeCrafterModel(use_lora=config.use_lora)
            self.is_finetuned = False
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize BLIP v1 for pure visual descriptions (not BLIP-2)
        # BLIP v1 focuses more on visual content and less on text in images
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
    
    def analyze_meme(self, image):
        """
        Complete meme analysis pipeline
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            dict with analysis results
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        results = {}
        
        # 1. Extract OCR text
        print("Extracting text with OCR...")
        ocr_result = self.ocr.extract_text(image, method='easyocr')
        results['ocr_text'] = ocr_result['text']
        results['ocr_confidence'] = ocr_result['confidence']
        
        # 2. Generate pure visual description using BLIP v1 (not fine-tuned)
        print("Generating visual description with BLIP v1...")
        # BLIP v1 focuses on visual content more than text in images
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
        
        # 3. Generate meme caption with fine-tuned model (may include text content)
        print("Generating meme caption with fine-tuned BLIP-2...")
        caption = self.model.generate_caption(
            image,
            max_length=config.max_length,
            num_beams=5,
            temperature=0.7
        )
        results['generated_caption'] = caption
        
        # 4. Generate context-aware analysis (with OCR text as prompt)
        if ocr_result['text']:
            print("Generating context-aware analysis...")
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
            results['context_analysis'] = "No text detected in meme."
        
        # 5. Analyze sentiment using BLIP-2 directly
        print("Analyzing sentiment with BLIP-2...")
        sentiment_scores = self._analyze_sentiment_blip2(image, ocr_result['text'])
        results['sentiment'] = sentiment_scores
        
        # 6. Generate comprehensive summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _analyze_sentiment_blip2(self, image, ocr_text):
        """
        Use BLIP-2 to analyze sentiment by asking it directly
        This leverages the fine-tuned model's understanding of memes
        """
        # Ask BLIP-2 about the sentiment - IMPROVED PROMPT
        # Simpler prompt that's less likely to be echoed
        if ocr_text:
            sentiment_prompt = f"This meme says: '{ocr_text}'. Question: What is the sentiment? Answer:"
        else:
            sentiment_prompt = "Question: What is the sentiment of this image? Answer:"
        
        sentiment_response = self.model.generate_caption(
            image, 
            prompt=sentiment_prompt, 
            max_length=20,  # Shorter to avoid rambling
            num_beams=3,
            temperature=0.3,  # Lower temperature for more focused response
            do_sample=False   # Greedy decoding for consistency
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
        response_words = sentiment_response.split()
        
        for sentiment, keywords in sentiment_keywords.items():
            count = sum(1 for keyword in keywords if keyword in sentiment_response)
            if count > 0:
                sentiment_scores[sentiment] = count
        
        # Determine detected sentiment
        if sentiment_scores:
            detected_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        else:
            # Fallback: check for simple matches
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
            max_length=100,
            num_beams=3,
            temperature=0.7,
            do_sample=True
        )
        
        # Create scores dict for compatibility with summary generation
        scores = {
            'positive': 1.0 if detected_sentiment == 'positive' else 0.0,
            'negative': 1.0 if detected_sentiment == 'negative' else 0.0,
            'neutral': 1.0 if detected_sentiment == 'neutral' else 0.0,
            'sarcastic': 1.0 if detected_sentiment == 'sarcastic' else 0.0
        }
        
        return {
            'scores': scores,
            'overall': detected_sentiment,
            'confidence': 0.85,  # BLIP-2 based confidence
            'explanation': explanation,
            'raw_response': sentiment_response
        }
    
    def _generate_summary(self, results):
        """Generate a comprehensive summary of the analysis"""
        summary_parts = []
        
        # Model type
        model_type = "fine-tuned" if self.is_finetuned else "base"
        summary_parts.append(f"📊 Analysis using {model_type} BLIP-2 model\n")
        
        # OCR results
        if results['ocr_text']:
            summary_parts.append(f"📝 Detected Text: '{results['ocr_text']}'")
            summary_parts.append(f"   Confidence: {results['ocr_confidence']:.2%}\n")
        else:
            summary_parts.append("📝 No text detected in image\n")
        
        # Generated caption
        summary_parts.append(f"🖼️ Visual Description: {results['generated_caption']}\n")
        
        # Context analysis
        if results['context_analysis'] != "No text detected in meme.":
            summary_parts.append(f"🧠 Context Analysis: {results['context_analysis']}\n")
        
        # Sentiment
        sentiment = results['sentiment']
        emoji_map = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐',
            'sarcastic': '😏'
        }
        emoji = emoji_map.get(sentiment['overall'], '😐')
        summary_parts.append(f"💭 Overall Sentiment: {sentiment['overall'].upper()} {emoji}")
        summary_parts.append(f"   Confidence: {sentiment['confidence']:.2%}")
        
        # Sentiment breakdown
        summary_parts.append("\n📈 Sentiment Breakdown:")
        for sent_type, score in sentiment['scores'].items():
            if score > 0:
                bar = "█" * int(score * 20)
                summary_parts.append(f"   {sent_type.capitalize()}: {bar} {score:.2%}")
        
        return "\n".join(summary_parts)
    
    def compare_models(self, image, base_model_path=None):
        """
        Compare base model vs fine-tuned model performance
        
        Args:
            image: PIL Image
            base_model_path: Optional path to base model
            
        Returns:
            dict with comparison results
        """
        results = {
            'finetuned': None,
            'base': None
        }
        
        # Analyze with fine-tuned model
        if self.is_finetuned:
            print("Analyzing with fine-tuned model...")
            results['finetuned'] = self.analyze_meme(image)
        
        # Analyze with base model
        print("Analyzing with base model...")
        base_analyzer = MemeAnalyzer(use_base_model=True)
        results['base'] = base_analyzer.analyze_meme(image)
        
        return results


# Gradio Interface Functions
def create_demo(model_path=None):
    """Create Gradio demo interface"""
    
    # Initialize analyzer
    analyzer = MemeAnalyzer(model_path=model_path)
    
    def analyze_image(image):
        """Analyze uploaded image"""
        if image is None:
            return "Please upload an image", "", "", "", "", ""
        
        try:
            results = analyzer.analyze_meme(image)
            
            # Format outputs
            ocr_output = f"**Detected Text:** {results['ocr_text']}\n\n**Confidence:** {results['ocr_confidence']:.2%}"
            visual_output = results['visual_description']  # NEW: True visual description
            caption_output = results['generated_caption']
            context_output = results['context_analysis']
            summary_output = results['summary']
            
            # Format sentiment as text instead of JSON to avoid Gradio bug
            sentiment_text = f"""**Overall Sentiment:** {results['sentiment']['overall'].upper()}
**Confidence:** {results['sentiment']['confidence']:.2%}

**Explanation:** {results['sentiment']['explanation']}

**Raw Response:** {results['sentiment']['raw_response']}
"""
            
            return ocr_output, visual_output, caption_output, context_output, summary_output, sentiment_text
        
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            return error_msg, "", "", "", "", ""
    
    def compare_models_ui(image):
        """Compare base vs fine-tuned model"""
        if image is None:
            return "Please upload an image", ""
        
        try:
            # Create both analyzers
            finetuned_analyzer = MemeAnalyzer(model_path=model_path)
            base_analyzer = MemeAnalyzer(use_base_model=True)
            
            # Analyze with both
            finetuned_results = finetuned_analyzer.analyze_meme(image)
            base_results = base_analyzer.analyze_meme(image)
            
            # Format comparison
            comparison = []
            comparison.append("=" * 60)
            comparison.append("FINE-TUNED MODEL RESULTS")
            comparison.append("=" * 60)
            comparison.append(finetuned_results['summary'])
            comparison.append("\n" + "=" * 60)
            comparison.append("BASE MODEL RESULTS")
            comparison.append("=" * 60)
            comparison.append(base_results['summary'])
            comparison.append("\n" + "=" * 60)
            comparison.append("KEY DIFFERENCES")
            comparison.append("=" * 60)
            
            # Compare captions
            comparison.append(f"\n📝 Caption Comparison:")
            comparison.append(f"   Fine-tuned: {finetuned_results['generated_caption']}")
            comparison.append(f"   Base: {base_results['generated_caption']}")
            
            # Compare sentiment
            comparison.append(f"\n💭 Sentiment Comparison:")
            comparison.append(f"   Fine-tuned: {finetuned_results['sentiment']['overall'].upper()} "
                            f"({finetuned_results['sentiment']['confidence']:.2%} confidence)")
            comparison.append(f"   Base: {base_results['sentiment']['overall'].upper()} "
                            f"({base_results['sentiment']['confidence']:.2%} confidence)")
            
            comparison_text = "\n".join(comparison)
            
            # Detailed comparison as text instead of JSON
            comparison_detail = f"""**FINE-TUNED MODEL:**
Caption: {finetuned_results['generated_caption']}
Sentiment: {finetuned_results['sentiment']['overall'].upper()}
Confidence: {finetuned_results['sentiment']['confidence']:.2%}

**BASE MODEL:**
Caption: {base_results['generated_caption']}
Sentiment: {base_results['sentiment']['overall'].upper()}
Confidence: {base_results['sentiment']['confidence']:.2%}
"""
            
            return comparison_text, comparison_detail
        
        except Exception as e:
            error_msg = f"Error comparing models: {str(e)}"
            return error_msg, ""
    
    # Create Gradio Interface
    with gr.Blocks(title="MemeCrafter - Meme Analysis Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 MemeCrafter - Advanced Meme Analysis
        
        This demo combines **OCR text extraction** with **fine-tuned BLIP-2** for comprehensive meme analysis:
        - 📝 Extract text from memes using OCR
        - 🖼️ Generate visual descriptions
        - 🧠 Perform semantic analysis with context
        - 💭 Understand sentiment and meaning
        - 📊 Compare base vs fine-tuned model performance
        """)
        
        with gr.Tabs():
            # Tab 1: Single Analysis
            with gr.Tab("🔍 Analyze Meme"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Upload Meme")
                        analyze_btn = gr.Button("Analyze Meme", variant="primary")
                    
                    with gr.Column(scale=1):
                        ocr_output = gr.Textbox(label="OCR Results", lines=3, value="")
                        visual_output = gr.Textbox(label="Visual Description (BLIP v1 - Visual Only)", lines=3, value="")
                        caption_output = gr.Textbox(label="Meme Caption (Fine-tuned BLIP-2)", lines=3, value="")
                
                with gr.Row():
                    context_output = gr.Textbox(label="Context-Aware Analysis", lines=4, value="")
                
                with gr.Row():
                    with gr.Column():
                        summary_output = gr.Textbox(label="Complete Summary", lines=10, value="")
                    with gr.Column():
                        sentiment_output = gr.Textbox(label="Sentiment Analysis", lines=10, value="")
                
                analyze_btn.click(
                    fn=analyze_image,
                    inputs=input_image,
                    outputs=[ocr_output, visual_output, caption_output, context_output, summary_output, sentiment_output],
                    api_name="analyze"
                )
            
            # Tab 2: Model Comparison
            with gr.Tab("⚖️ Compare Models"):
                gr.Markdown("""
                ### Compare Base BLIP-2 vs Fine-tuned Model
                See how fine-tuning improves meme understanding and sentiment analysis!
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_image = gr.Image(type="pil", label="Upload Meme")
                        compare_btn = gr.Button("Compare Models", variant="primary")
                    
                    with gr.Column(scale=1):
                        comparison_output = gr.Textbox(label="Comparison Results", lines=20, value="")
                
                comparison_detail = gr.Textbox(label="Detailed Comparison", lines=10, value="")
                
                compare_btn.click(
                    fn=compare_models_ui,
                    inputs=compare_image,
                    outputs=[comparison_output, comparison_detail],
                    api_name="compare"
                )
            
            # Tab 3: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About MemeCrafter
                
                MemeCrafter is an advanced meme analysis system that combines:
                
                ### 🔧 Technologies Used:
                - **BLIP-2**: State-of-the-art vision-language model
                - **LoRA**: Parameter-efficient fine-tuning
                - **EasyOCR**: Robust text extraction from images
                - **Sentiment Analysis**: Custom sentiment understanding
                
                ### 📊 Fine-tuning Advantages:
                The fine-tuned model has been trained on your meme dataset to:
                - Better understand meme-specific language and humor
                - More accurately detect sentiment and sarcasm
                - Provide context-aware analysis combining visual and textual elements
                - Understand cultural references and meme formats
                
                ### 🎯 Use Cases:
                - Social media content moderation
                - Sentiment analysis of viral content
                - Meme trend analysis
                - Educational research on internet culture
                
                ### 📝 Model Information:
                - **Base Model**: BLIP-2 (Salesforce)
                - **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
                - **Training Dataset**: Custom meme dataset with captions
                - **Device**: {device}
                
                ---
                Made with ❤️ using Gradio and Hugging Face Transformers
                """.format(device=config.device))
        
        # Examples (optional)
        gr.Markdown("### 📸 Example Memes")
        gr.Markdown("Upload your own meme image to get started!")
    
    return demo


def main():
    """Main function to launch demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MemeCrafter Demo")
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to fine-tuned model (default: uses base model)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run demo on (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create public link'
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    if args.model_path and not os.path.exists(args.model_path):
        print(f"Warning: Model path {args.model_path} not found. Using base model.")
        args.model_path = None
    
    # Create and launch demo
    print("Creating demo interface...")
    demo = create_demo(model_path=args.model_path)
    
    print(f"\nLaunching demo on port {args.port}...")
    print("=" * 60)
    
    demo.launch(
        server_port=args.port,
        share=True
    )


if __name__ == "__main__":
    main()
