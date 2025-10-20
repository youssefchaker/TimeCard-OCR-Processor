import cv2
import numpy as np
from PIL import Image
import os
import io

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
            import fitz  # PyMuPDF
            print(f"📄 Converting PDF using PyMuPDF: {pdf_path}")
            
            # Open the PDF
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
                
                print(f"   ✓ Page {page_num + 1} converted to image: {opencv_image.shape[1]}x{opencv_image.shape[0]}")
            
            pdf_document.close()
            print(f"✅ Successfully converted PDF to {len(images)} high-quality images")
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
        
        print(f"   🔄 Testing {max_orientations} orientations for optimal text detection...")
        
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
                
                print(f"      {orientation_name}: {len(text_blocks)} blocks, score: {score:.2f}")
                
                # Update best orientation if this one is better
                if score > best_score:
                    best_score = score
                    best_orientation = orientation_name
                    best_blocks = text_blocks
                    best_processed_image = processed_image
                    
            except Exception as e:
                print(f"      ❌ {orientation_name} failed: {e}")
                continue
        
        if best_orientation and best_score > 0:
            print(f"   ✅ Best orientation: {best_orientation} (score: {best_score:.2f})")
            return best_processed_image, best_blocks, best_orientation
        else:
            print(f"   ⚠ No good orientation found, using original")
            processed_image = self.preprocess_image(image)
            extracted_data = ocr_engine.extract_text_with_confidence(processed_image)
            text_blocks = ocr_engine.group_text_by_blocks(extracted_data)
            return processed_image, text_blocks, "original"
    
    def _calculate_orientation_score(self, text_blocks: list[dict]) -> float:
        """Calculate a quality score for OCR results to determine best orientation"""
        if not text_blocks:
            return 0.0
        
        score = 0.0
        
        # Factor 1: Number of text blocks (more is generally better)
        block_count_score = min(len(text_blocks) / 10, 1.0)  # Normalize to 0-1
        
        # Factor 2: Average confidence
        avg_confidence = np.mean([block['confidence'] for block in text_blocks])
        confidence_score = avg_confidence  # Already 0-1
        
        # Factor 3: Text quality (ratio of meaningful text vs garbage)
        meaningful_text_score = self._calculate_text_quality(text_blocks)
        
        # Factor 4: Text distribution (good orientations have more horizontal text)
        distribution_score = self._calculate_text_distribution(text_blocks)
        
        # Combined score (weighted)
        score = (
            block_count_score * 0.2 +
            confidence_score * 0.3 +
            meaningful_text_score * 0.4 +
            distribution_score * 0.1
        )
        
        return score
    
    def _calculate_text_quality(self, text_blocks: list[dict]) -> float:
        """Calculate how meaningful the extracted text is"""
        if not text_blocks:
            return 0.0
        
        meaningful_count = 0
        total_blocks = len(text_blocks)
        
        for block in text_blocks:
            text = block['text'].strip()
            
            # Skip if text is too short
            if len(text) < 2:
                continue
                
            # Check if text looks like garbage
            if self._is_garbage_text(text):
                continue
                
            # Check if text has reasonable word-like structure
            if self._looks_like_meaningful_text(text):
                meaningful_count += 1
        
        return meaningful_count / total_blocks if total_blocks > 0 else 0.0
    
    def _is_garbage_text(self, text: str) -> bool:
        """Check if text looks like garbage/artifacts"""
        # Very short text
        if len(text) < 2:
            return True
            
        # Mostly special characters
        special_chars = set('!@#$%^&*()_+-=[]{}|;:,.<>?/~`')
        special_count = sum(1 for char in text if char in special_chars)
        if special_count / len(text) > 0.7:  # More than 70% special chars
            return True
        
        # Repeated characters (like "aaaa", "======")
        if len(set(text)) == 1 and len(text) > 3:
            return True
            
        # Common OCR garbage patterns
        garbage_patterns = [
            r'^[^a-zA-Z0-9]{3,}$',  # Only special chars
            r'^[|]{2,}$',  # Multiple pipes
            r'^[\.]{3,}$',  # Multiple dots
            r'^[\-]{3,}$',  # Multiple dashes
        ]
        
        import re
        for pattern in garbage_patterns:
            if re.match(pattern, text):
                return True
                
        return False
    
    def _looks_like_meaningful_text(self, text: str) -> bool:
        """Check if text looks like meaningful content"""
        # Contains at least one letter or number
        if not any(char.isalnum() for char in text):
            return False
            
        # Reasonable length for words
        words = text.split()
        if words:
            # Check if average word length is reasonable
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 2 <= avg_word_length <= 20:
                return True
                
        return False
    
    def _calculate_text_distribution(self, text_blocks: list[dict]) -> float:
        """Calculate how well text is distributed (good for horizontal reading)"""
        if len(text_blocks) < 2:
            return 0.5  # Neutral score for small amounts of text
        
        # Calculate variance in y-positions (less variance = more horizontal)
        y_positions = [block['bbox']['top'] for block in text_blocks]
        y_variance = np.var(y_positions)
        
        # Calculate average block width vs height (wider blocks = more horizontal)
        width_height_ratios = []
        for block in text_blocks:
            bbox = block['bbox']
            if bbox['height'] > 0:
                ratio = bbox['width'] / bbox['height']
                width_height_ratios.append(ratio)
        
        if width_height_ratios:
            avg_ratio = np.mean(width_height_ratios)
            # Good ratio is > 2 (wider than tall)
            ratio_score = min(avg_ratio / 3, 1.0)
        else:
            ratio_score = 0.5
            
        # Combine scores
        # Lower y-variance and higher width ratio = better horizontal distribution
        variance_score = 1.0 - min(y_variance / 10000, 1.0)  # Normalize variance
        
        return (variance_score + ratio_score) / 2
    
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
    
    def save_temp_image(self, image, filename):
        """Save temporary image for debugging"""
        temp_dir = 'data/temp'
        os.makedirs(temp_dir, exist_ok=True)
        cv2.imwrite(os.path.join(temp_dir, filename), image)
        print(f"💾 Saved temp image: {filename}")