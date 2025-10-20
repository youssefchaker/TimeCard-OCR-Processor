import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Dict

class ImagePreprocessor:
    def __init__(self):
        self.config = {
            'dpi': 300,
            'denoise_strength': 10,
            'threshold_block_size': 11,
            'threshold_constant': 2
        }
    
    def pdf_to_images(self, pdf_path: str) -> list:
        """Convert PDF to list of images using PyMuPDF"""
        try:
            import fitz
            
            pdf_document = fitz.open(pdf_path)
            images = []
            
            # Convert each page to image
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Create matrix for high-quality conversion
                zoom_factor = self.config['dpi'] / 72
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                
                # Get pixmap (image) from page
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to bytes then to PIL Image
                img_data = pix.tobytes("ppm")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Convert to OpenCV format (BGR)
                opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                images.append(opencv_image)
            
            pdf_document.close()
            return images
            
        except ImportError:
            raise ImportError("PyMuPDF (fitz) is required. Please install it: pip install PyMuPDF")
        except Exception as e:
            raise Exception(f"PDF conversion failed: {str(e)}")
    
    def try_orientations(self, image, ocr_engine, max_orientations=4):
        """Try different orientations and return the best one based on OCR quality"""
        orientations = [
            (0, "original", image),
            (1, "90° clockwise", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
            (2, "180°", cv2.rotate(image, cv2.ROTATE_180)),
            (3, "90° counter-clockwise", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
        ]
        
        best_orientation = None
        best_score = -1
        best_blocks = []
        best_processed_image = None
        
        for orientation_id, orientation_name, oriented_image in orientations[:max_orientations]:
            try:
                # Preprocess the oriented image
                processed_image = self.preprocess_image(oriented_image)
                
                # Extract text with OCR
                extracted_data = ocr_engine.extract_text_with_confidence(processed_image)
                
                # Group into logical blocks
                text_blocks = ocr_engine.group_text_by_blocks(extracted_data)
                
                # Calculate quality score for this orientation
                score = self._calculate_orientation_score(text_blocks)
                
                # Update best orientation if this one is better
                if score > best_score:
                    best_score = score
                    best_orientation = orientation_name
                    best_blocks = text_blocks
                    best_processed_image = processed_image
                    
            except Exception:
                continue
        
        if best_orientation and best_score > 0:
            return best_processed_image, best_blocks, best_orientation
        else:
            processed_image = self.preprocess_image(image)
            extracted_data = ocr_engine.extract_text_with_confidence(processed_image)
            text_blocks = ocr_engine.group_text_by_blocks(extracted_data)
            return processed_image, text_blocks, "original"
    
    def _calculate_orientation_score(self, text_blocks: List[Dict]) -> float:
        """Calculate a quality score for OCR results to determine best orientation"""
        if not text_blocks:
            return 0.0
        
        score = 0.0
        
        # Factor 1: Number of text blocks
        block_count_score = min(len(text_blocks) / 10, 1.0)
        
        # Factor 2: Average confidence
        avg_confidence = np.mean([block['confidence'] for block in text_blocks])
        confidence_score = avg_confidence
        
        # Factor 3: Text quality
        meaningful_text_score = self._calculate_text_quality(text_blocks)
        
        # Combined score
        score = (
            block_count_score * 0.2 +
            confidence_score * 0.3 +
            meaningful_text_score * 0.5
        )
        
        return score
    
    def _calculate_text_quality(self, text_blocks: List[Dict]) -> float:
        """Calculate how meaningful the extracted text is"""
        if not text_blocks:
            return 0.0
        
        meaningful_count = 0
        total_blocks = len(text_blocks)
        
        for block in text_blocks:
            text = block['text'].strip()
            
            if len(text) < 2:
                continue
                
            if self._is_garbage_text(text):
                continue
                
            if self._looks_like_meaningful_text(text):
                meaningful_count += 1
        
        return meaningful_count / total_blocks if total_blocks > 0 else 0.0
    
    def _is_garbage_text(self, text: str) -> bool:
        """Check if text looks like garbage/artifacts"""
        if len(text) < 2:
            return True
            
        special_chars = set('!@#$%^&*()_+-=[]{}|;:,.<>?/~`')
        special_count = sum(1 for char in text if char in special_chars)
        if special_count / len(text) > 0.7:
            return True
        
        if len(set(text)) == 1 and len(text) > 3:
            return True
            
        import re
        garbage_patterns = [
            r'^[^a-zA-Z0-9]{3,}$',
            r'^[|]{2,}$',
            r'^[\.]{3,}$',
            r'^[\-]{3,}$',
        ]
        
        for pattern in garbage_patterns:
            if re.match(pattern, text):
                return True
                
        return False
    
    def _looks_like_meaningful_text(self, text: str) -> bool:
        """Check if text looks like meaningful content"""
        if not any(char.isalnum() for char in text):
            return False
            
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 2 <= avg_word_length <= 20:
                return True
                
        return False
    
    def preprocess_image(self, image):
        """Enhance image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=self.config['denoise_strength'])
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config['threshold_block_size'],
            self.config['threshold_constant']
        )
        
        return thresh