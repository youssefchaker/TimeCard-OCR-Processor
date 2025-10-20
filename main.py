import os
import pandas as pd
import argparse
from typing import Dict, Any, List

from ocr_engine import OCREngine
from document_parser import DocumentParser
from image_preprocessor import ImagePreprocessor
from config import Config

class TimeCardOCRApp:
    def __init__(self, confidence_threshold: float = 0.8, auto_orient: bool = True):
        self.config = Config()
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine(confidence_threshold)
        self.parser = DocumentParser()
        self.auto_orient = auto_orient
        
        # Create directories
        os.makedirs(self.config.INPUT_DIR, exist_ok=True)
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        print(f"Processing: {os.path.basename(pdf_path)}")
        
        try:
            # Convert PDF to images using PyMuPDF
            images = self.preprocessor.pdf_to_images(pdf_path)
            
            all_data_frames = []
            
            for page_num, image in enumerate(images):
                if self.auto_orient:
                    # Try different orientations to find the best one
                    processed_image, text_blocks, _ = self.preprocessor.try_orientations(
                        image, self.ocr_engine
                    )
                else:
                    # Use original orientation only
                    processed_image = self.preprocessor.preprocess_image(image)
                    extracted_data = self.ocr_engine.extract_text_with_confidence(processed_image)
                    text_blocks = self.ocr_engine.group_text_by_blocks(extracted_data)
                
                # Parse document
                page_result = self.parser.parse_document(text_blocks)
                
                # Collect all data frames from this page
                for data_frame in page_result.get('structured_data', []):
                    data_info = {
                        'page': page_num + 1,
                        'dataframe': data_frame
                    }
                    all_data_frames.append(data_info)
                
                print(f"  Page {page_num + 1} done")
            
            # Save CSV files
            output_files = self._save_csv_files(pdf_path, all_data_frames)
            
            return {
                'success': True,
                'file_path': pdf_path,
                'output_files': output_files,
                'pages_processed': len(images),
                'data_sets_found': len(all_data_frames)
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {
                'success': False,
                'file_path': pdf_path,
                'error': str(e)
            }
    
    def _save_csv_files(self, pdf_path: str, data_sets: List[Dict]) -> List[str]:
        """Save all data as CSV files"""
        filename = os.path.basename(pdf_path).replace('.pdf', '')
        output_files = []
        
        # Save each data set as a separate CSV file
        for data_info in data_sets:
            page_num = data_info['page']
            
            # Simple filename: {input_filename}_page{page_number}.csv
            csv_filename = f"{filename}_page{page_num}.csv"
            csv_path = os.path.join(self.config.OUTPUT_DIR, csv_filename)
            
            try:
                df = data_info['dataframe']
                
                if df.empty:
                    # Create empty file
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.write("No data extracted")
                else:
                    # Save with proper CSV formatting
                    # Include headers for tables, exclude for sequential data
                    if len(df.columns) > 1:  # Likely a table
                        df.to_csv(csv_path, index=False, encoding='utf-8')
                    else:  # Sequential data
                        df.to_csv(csv_path, index=False, header=False, encoding='utf-8')
                
                output_files.append(csv_path)
                
            except Exception as e:
                print(f"Failed to save page {page_num}: {e}")
        
        print(f"Saved {len(output_files)} CSV files")
        return output_files
    
    def process_directory(self, directory_path: str = None) -> list:
        """Process all PDFs in a directory"""
        if directory_path is None:
            directory_path = self.config.INPUT_DIR
        
        results = []
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            result = self.process_pdf(pdf_path)
            results.append(result)
            print()
        
        return results

def main():
    parser = argparse.ArgumentParser(description='PDF to CSV Converter')
    parser.add_argument('--file', '-f', help='Process a single PDF file')
    parser.add_argument('--dir', '-d', help='Process all PDFs in directory')
    parser.add_argument('--confidence', '-c', type=float, default=0.8, 
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--no-orient', action='store_true', 
                       help='Disable auto-orientation detection')
    
    args = parser.parse_args()
    
    app = TimeCardOCRApp(
        confidence_threshold=args.confidence,
        auto_orient=not args.no_orient
    )
    
    if args.file:
        result = app.process_pdf(args.file)
        if result['success']:
            print(f"Completed: {result['pages_processed']} pages processed")
            print(f"Generated: {len(result['output_files'])} CSV files")
        else:
            print("Failed to process file")
        
    else:
        # Process default input directory
        results = app.process_directory()
        successful = sum(1 for r in results if r['success'])
        print(f"\nProcessing complete: {successful}/{len(results)} files successful")

if __name__ == "__main__":
    main()