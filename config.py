import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Tesseract path (adjust for your system)
    TESSERACT_PATH = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')
    
    # Confidence threshold (0.0 - 1.0)
    CONFIDENCE_THRESHOLD = 0.8
    
    # Output formats
    OUTPUT_FORMATS = ['json', 'csv']
    
    # Image settings
    IMAGE_DPI = 300  # High DPI for better OCR accuracy
    
    # File paths
    INPUT_DIR = 'data/input'
    OUTPUT_DIR = 'data/output'
    TEMP_DIR = 'data/temp'