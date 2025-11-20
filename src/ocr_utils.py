"""
OCR utilities for extracting text from meme images
Supports multiple OCR backends: EasyOCR and Tesseract
"""
import cv2
import numpy as np
from PIL import Image
import easyocr
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MemeOCR:
    """OCR processor for meme images with text extraction and preprocessing"""
    
    def __init__(self, use_easyocr=True, languages=['en']):
        """
        Initialize OCR processor
        
        Args:
            use_easyocr: If True, use EasyOCR (more accurate), else use Tesseract
            languages: List of language codes for OCR
        """
        self.use_easyocr = use_easyocr
        self.languages = languages
        
        if use_easyocr:
            print("Initializing EasyOCR...")
            self.reader = easyocr.Reader(languages, gpu=False)  # Set gpu=True if available
        else:
            try:
                import pytesseract
                self.pytesseract = pytesseract
            except ImportError:
                raise ImportError("Tesseract not installed. Install with: pip install pytesseract")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply thresholding to get binary image
        # Use adaptive thresholding for better results with varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised
    
    def extract_text_easyocr(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        """
        Extract text using EasyOCR
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (combined_text, detailed_results)
        """
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Run OCR
        results = self.reader.readtext(img_array)
        
        # Extract text and confidence scores
        detailed_results = []
        texts = []
        
        for bbox, text, confidence in results:
            if confidence > 0.3:  # Filter low confidence detections
                texts.append(text)
                detailed_results.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': bbox
                })
        
        # Combine all text
        combined_text = ' '.join(texts)
        
        return combined_text, detailed_results
    
    def extract_text_tesseract(self, image: Image.Image) -> Tuple[str, List[Dict]]:
        """
        Extract text using Tesseract OCR
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (combined_text, detailed_results)
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Run OCR
        data = self.pytesseract.image_to_data(
            preprocessed, 
            output_type=self.pytesseract.Output.DICT
        )
        
        # Extract text with confidence > threshold
        texts = []
        detailed_results = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            confidence = int(data['conf'][i])
            text = data['text'][i].strip()
            
            if confidence > 30 and text:  # Filter low confidence and empty
                texts.append(text)
                detailed_results.append({
                    'text': text,
                    'confidence': confidence / 100.0,
                    'bbox': (data['left'][i], data['top'][i], 
                            data['width'][i], data['height'][i])
                })
        
        combined_text = ' '.join(texts)
        
        return combined_text, detailed_results
    
    def extract_text(self, image: Image.Image) -> Dict:
        """
        Main method to extract text from image
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with extracted text and metadata
        """
        if self.use_easyocr:
            text, details = self.extract_text_easyocr(image)
        else:
            text, details = self.extract_text_tesseract(image)
        
        return {
            'text': text,
            'cleaned_text': self.clean_text(text),
            'details': details,
            'word_count': len(text.split()),
            'has_text': len(text.strip()) > 0
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted OCR text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        import re
        text = re.sub(r'[^\w\s\.,!?;\-\'\"]', '', text)
        
        return text.strip()
    
    def visualize_ocr_results(self, image: Image.Image, ocr_results: Dict) -> Image.Image:
        """
        Draw bounding boxes on image showing detected text regions
        
        Args:
            image: PIL Image
            ocr_results: Results from extract_text method
            
        Returns:
            Annotated PIL Image
        """
        img_array = np.array(image)
        img_copy = img_array.copy()
        
        # Draw bounding boxes
        for detail in ocr_results['details']:
            bbox = detail['bbox']
            confidence = detail['confidence']
            
            if self.use_easyocr:
                # EasyOCR bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(img_copy, [pts], True, (0, 255, 0), 2)
            else:
                # Tesseract bbox format: (x, y, w, h)
                x, y, w, h = bbox
                cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add confidence score
            text_pos = (int(bbox[0][0]), int(bbox[0][1]) - 10) if self.use_easyocr else (x, y-10)
            cv2.putText(img_copy, f"{confidence:.2f}", text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return Image.fromarray(img_copy)


# Convenience function
def extract_text_from_image(image: Image.Image, use_easyocr=True) -> Dict:
    """
    Quick function to extract text from an image
    
    Args:
        image: PIL Image
        use_easyocr: Whether to use EasyOCR (True) or Tesseract (False)
        
    Returns:
        Dictionary with OCR results
    """
    ocr = MemeOCR(use_easyocr=use_easyocr)
    return ocr.extract_text(image)
