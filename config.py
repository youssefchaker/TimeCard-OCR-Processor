import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Tesseract path (adjust for your system)
    TESSERACT_PATH = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')
    
    # Confidence threshold (0.0 - 1.0)
    CONFIDENCE_THRESHOLD = 0.8
    
    # File paths
    INPUT_DIR = 'data/input'
    OUTPUT_DIR = 'data/output'
