import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json

class OCREngine:
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.setup_tesseract()
    
    def setup_tesseract(self):
        """Configure Tesseract paths"""
        try:
            # Windows typical path
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        except:
            # Linux/Mac - usually in PATH
            pass
    
    def extract_text_with_confidence(self, image) -> List[Dict]:
        """Extract text with confidence scores using Tesseract"""
        try:
            # Convert OpenCV image to PIL
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Run OCR with confidence data
            data = pytesseract.image_to_data(
                pil_image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6 -c preserve_interword_spaces=1'
            )
            
            extracted_data = []
            
            for i in range(len(data['text'])):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                # Only include high-confidence results
                if text and confidence > self.confidence_threshold * 100:
                    extracted_data.append({
                        'text': text,
                        'confidence': confidence / 100,
                        'bbox': {
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        },
                        'page_num': data.get('page_num', 1)
                    })
            
            return extracted_data
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return []
    
    def group_text_by_blocks(self, extracted_data: List[Dict]) -> List[Dict]:
        """Group text into logical blocks based on spatial proximity"""
        if not extracted_data:
            return []
        
        # Sort by vertical position
        sorted_data = sorted(extracted_data, key=lambda x: x['bbox']['top'])
        
        blocks = []
        current_block = []
        line_height_threshold = 20  # pixels
        
        for i, item in enumerate(sorted_data):
            if not current_block:
                current_block.append(item)
                continue
            
            # Check if this item is on the same line as previous
            last_item = current_block[-1]
            vertical_diff = abs(item['bbox']['top'] - last_item['bbox']['top'])
            
            if vertical_diff < line_height_threshold:
                current_block.append(item)
            else:
                # New line/block
                blocks.append(self._merge_block_text(current_block))
                current_block = [item]
        
        if current_block:
            blocks.append(self._merge_block_text(current_block))
        
        return blocks
    
    def _merge_block_text(self, block_items: List[Dict]) -> Dict:
        """Merge items in a block into a single text entry"""
        if not block_items:
            return {}
        
        # Sort horizontally
        block_items.sort(key=lambda x: x['bbox']['left'])
        
        merged_text = ' '.join(item['text'] for item in block_items)
        avg_confidence = sum(item['confidence'] for item in block_items) / len(block_items)
        
        return {
            'text': merged_text,
            'confidence': avg_confidence,
            'bbox': block_items[0]['bbox'],
            'item_count': len(block_items)
        }