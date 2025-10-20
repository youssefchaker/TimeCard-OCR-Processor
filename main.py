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
                
                # Parse document - ALWAYS get data, even if minimal
                page_result = self.parser.parse_document(text_blocks)
                
                # Ensure we always have output for this page
                page_data = self._ensure_page_output(page_result, text_blocks, page_num + 1)
                
                # Collect data frame for this page
                data_info = {
                    'page': page_num + 1,
                    'dataframe': page_data
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
    
    def _ensure_page_output(self, page_result: Dict, text_blocks: List[Dict], page_num: int) -> pd.DataFrame:
        """Ensure we always have output for every page, even if minimal"""
        # Check if we have structured data
        structured_data = page_result.get('structured_data', [])
        
        if structured_data:
            # Use the first structured data frame found
            for data_frame in structured_data:
                if not data_frame.empty:
                    return data_frame
        
        # If no structured data or all empty, create basic output
        return self._create_basic_output(text_blocks, page_num)
    
    def _create_basic_output(self, text_blocks: List[Dict], page_num: int) -> pd.DataFrame:
        """Create basic output when no structured data is found"""
        if not text_blocks:
            # Create empty dataframe with one row indicating no data
            return pd.DataFrame([["No text detected on page"]])
        
        # Create simple output with all text blocks in order
        data = []
        for block in text_blocks:
            data.append([block['text']])
        
        return pd.DataFrame(data)
    
    def _save_csv_files(self, pdf_path: str, data_sets: List[Dict]) -> List[str]:
        """Save all data as CSV files - ensure proper comma separation"""
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
                
                # Ensure proper CSV formatting with commas
                if df.empty:
                    # Create empty CSV with just headers if needed
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.write("")
                else:
                    # Save with proper CSV formatting
                    # Use header=False to remove column names, but ensure commas between columns
                    df.to_csv(csv_path, index=False, header=False, encoding='utf-8')
                    
                    # Verify the CSV has proper comma separation
                    self._verify_csv_format(csv_path)
                
                output_files.append(csv_path)
                
            except Exception as e:
                print(f"Failed to save page {page_num}: {e}")
        
        print(f"Saved {len(output_files)} CSV files")
        return output_files
    
    def _verify_csv_format(self, csv_path: str):
        """Verify that the CSV file has proper comma separation"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                lines = content.split('\n')
                for line in lines:
                    # Check if this looks like a multi-column row that should have commas
                    if len(line) > 50 and ',' not in line:  # Long line without commas
                        print(f"Warning: Potential formatting issue in {os.path.basename(csv_path)}")
                        break
        except Exception:
            pass
    
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
            print()  # Empty line between files
        
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