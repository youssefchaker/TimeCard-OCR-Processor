import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any

from ocr_engine import OCREngine
from document_parser import DocumentParser
from image_preprocessor import ImagePreprocessor
from config import Config

class TimeCardOCRApp:
    def __init__(self, confidence_threshold: float = 0.8):
        self.config = Config()
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine(confidence_threshold)
        self.parser = DocumentParser()
        
        # Create directories
        os.makedirs(self.config.INPUT_DIR, exist_ok=True)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.config.TEMP_DIR, exist_ok=True)
        
        print("🚀 TimeCard OCR Application Initialized")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Input directory: {self.config.INPUT_DIR}")
        print(f"   Output directory: {self.config.OUTPUT_DIR}")
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        print(f"\n📋 Processing: {pdf_path}")
        
        try:
            # Convert PDF to images using PyMuPDF
            images = self.preprocessor.pdf_to_images(pdf_path)
            
            all_results = {}
            
            for page_num, image in enumerate(images):
                print(f"   📄 Processing page {page_num + 1}/{len(images)}")
                
                # Preprocess image
                processed_image = self.preprocessor.preprocess_image(image)
                
                # Save temp image for debugging (optional)
                self.preprocessor.save_temp_image(
                    processed_image, 
                    f"page_{page_num + 1}_processed.png"
                )
                
                # Extract text with OCR
                extracted_data = self.ocr_engine.extract_text_with_confidence(processed_image)
                
                # Group into logical blocks
                text_blocks = self.ocr_engine.group_text_by_blocks(extracted_data)
                
                # Parse document structure
                if page_num == 0:  # First page is usually work order
                    page_result = self.parser.parse_work_order(text_blocks)
                elif page_num >= 6:  # Timesheet pages
                    page_result = self.parser.parse_timesheet(text_blocks)
                else:
                    page_result = {
                        'document_type': 'unknown',
                        'text_blocks': [block['text'] for block in text_blocks],
                        'confidence_score': sum(block['confidence'] for block in text_blocks) / len(text_blocks) if text_blocks else 0
                    }
                
                all_results[f'page_{page_num + 1}'] = page_result
            
            # Save results
            output_filename = self._save_results(pdf_path, all_results)
            
            return {
                'success': True,
                'file_path': pdf_path,
                'output_file': output_filename,
                'pages_processed': len(images),
                'results': all_results
            }
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
            return {
                'success': False,
                'file_path': pdf_path,
                'error': str(e)
            }
    
    def process_directory(self, directory_path: str = None) -> list:
        """Process all PDFs in a directory"""
        if directory_path is None:
            directory_path = self.config.INPUT_DIR
        
        results = []
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        print(f"\n🔍 Found {len(pdf_files)} PDF files in {directory_path}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            result = self.process_pdf(pdf_path)
            results.append(result)
        
        return results
    
    def _save_results(self, pdf_path: str, results: Dict) -> str:
        """Save results to JSON file"""
        filename = os.path.basename(pdf_path).replace('.pdf', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            self.config.OUTPUT_DIR, 
            f"{filename}_{timestamp}.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description='Time Card OCR Application')
    parser.add_argument('--file', '-f', help='Process a single PDF file')
    parser.add_argument('--dir', '-d', help='Process all PDFs in directory')
    parser.add_argument('--confidence', '-c', type=float, default=0.8, 
                       help='Confidence threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    app = TimeCardOCRApp(confidence_threshold=args.confidence)
    
    if args.file:
        result = app.process_pdf(args.file)
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"\n{status} - Processed: {args.file}")
        
    elif args.dir:
        results = app.process_directory(args.dir)
        successful = sum(1 for r in results if r['success'])
        print(f"\n📊 Processing Complete:")
        print(f"   Total files: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        
    else:
        # Process default input directory
        results = app.process_directory()
        successful = sum(1 for r in results if r['success'])
        print(f"\n📊 Processing Complete:")
        print(f"   Total files: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")

if __name__ == "__main__":
    main()