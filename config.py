import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    TESSERACT_PATH = os.getenv('TESSERACT_PATH', '/usr/bin/tesseract')
    
    CONFIDENCE_THRESHOLD = 0.8
    
    INPUT_DIR = 'data/input'
    OUTPUT_DIR = 'data/output'
