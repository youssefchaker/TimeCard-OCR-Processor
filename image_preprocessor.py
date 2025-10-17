import cv2
import numpy as np
from PIL import Image
import os
import io

class ImagePreprocessor:
    def __init__(self):
        self.config = {
            'dpi': 300,  # High DPI for better OCR quality
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
                zoom_factor = self.config['dpi'] / 72  # Convert DPI to zoom
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
    
    def preprocess_image(self, image):
        """Enhance image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=self.config['denoise_strength'])
        
        # Adaptive thresholding for better text contrast
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