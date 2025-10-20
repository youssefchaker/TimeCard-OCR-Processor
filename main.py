import os
import pandas as pd
import argparse
from datetime import datetime
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
        os.makedirs(self.config.TEMP_DIR, exist_ok=True)
        
        print("🚀 Advanced OCR with Auto-Orientation Detection")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Auto-orientation: {'ENABLED' if auto_orient else 'DISABLED'}")
        print(f"   Input directory: {self.config.INPUT_DIR}")
        print(f"   Output directory: {self.config.OUTPUT_DIR}")
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        print(f"\n📋 Processing: {pdf_path}")
        
        try:
            # Convert PDF to images using PyMuPDF
            images = self.preprocessor.pdf_to_images(pdf_path)
            
            all_data_frames = []
            orientation_info = []
            
            for page_num, image in enumerate(images):
                print(f"   📄 Processing page {page_num + 1}/{len(images)}")
                
                if self.auto_orient:
                    # Try different orientations to find the best one
                    processed_image, text_blocks, orientation = self.preprocessor.try_orientations(
                        image, self.ocr_engine
                    )
                    orientation_info.append({
                        'page': page_num + 1,
                        'orientation': orientation,
                        'blocks_found': len(text_blocks)
                    })
                else:
                    # Use original orientation only
                    processed_image = self.preprocessor.preprocess_image(image)
                    extracted_data = self.ocr_engine.extract_text_with_confidence(processed_image)
                    text_blocks = self.ocr_engine.group_text_by_blocks(extracted_data)
                    orientation_info.append({
                        'page': page_num + 1,
                        'orientation': 'original',
                        'blocks_found': len(text_blocks)
                    })
                
                # Save processed image for debugging
                self.preprocessor.save_temp_image(
                    processed_image, 
                    f"page_{page_num + 1}_processed.png"
                )
                
                # Parse document using advanced spatial analysis
                page_result = self.parser.parse_document(text_blocks)
                
                # Check if we need to try alternative orientations due to poor results
                if self.auto_orient and self._needs_reorientation(page_result, text_blocks):
                    print(f"   ⚠ Poor results detected, trying alternative orientations...")
                    # Try the orientations we haven't tried yet
                    for orientation_id in [1, 2, 3]:  # Skip original (0)
                        try:
                            oriented_image = self._rotate_image(image, orientation_id)
                            alt_processed = self.preprocessor.preprocess_image(oriented_image)
                            alt_extracted = self.ocr_engine.extract_text_with_confidence(alt_processed)
                            alt_blocks = self.ocr_engine.group_text_by_blocks(alt_extracted)
                            alt_result = self.parser.parse_document(alt_blocks)
                            
                            if self._is_better_result(alt_result, page_result):
                                print(f"   ✅ Better results with orientation {orientation_id}")
                                page_result = alt_result
                                text_blocks = alt_blocks
                                orientation_info[-1]['orientation'] = f'alt_{orientation_id}'
                                break
                        except Exception as e:
                            continue
                
                # Collect all data frames from this page
                for data_frame in page_result.get('structured_data', []):
                    data_info = {
                        'page': page_num + 1,
                        'data_type': self._classify_data_type(data_frame),
                        'dataframe': data_frame,
                        'block_count': len(text_blocks),
                        'confidence': page_result.get('confidence_score', 0),
                        'orientation': orientation_info[-1]['orientation']
                    }
                    all_data_frames.append(data_info)
            
            # Save CSV files
            output_files = self._save_csv_files(pdf_path, all_data_frames, orientation_info)
            
            return {
                'success': True,
                'file_path': pdf_path,
                'output_files': output_files,
                'pages_processed': len(images),
                'data_sets_found': len(all_data_frames),
                'orientations': orientation_info
            }
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'file_path': pdf_path,
                'error': str(e)
            }
    
    def _rotate_image(self, image, orientation_id):
        """Rotate image based on orientation ID"""
        if orientation_id == 1:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation_id == 2:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif orientation_id == 3:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
    
    def _needs_reorientation(self, page_result: Dict, text_blocks: List[Dict]) -> bool:
        """Check if the results are poor enough to warrant trying other orientations"""
        # If no structured data found
        if not page_result.get('structured_data'):
            return True
        
        # If structured data is empty
        for data_frame in page_result.get('structured_data', []):
            if data_frame.empty:
                return True
        
        # If very few text blocks found
        if len(text_blocks) < 3:
            return True
        
        # If average confidence is very low
        if page_result.get('confidence_score', 0) < 0.3:
            return True
        
        return False
    
    def _is_better_result(self, new_result: Dict, old_result: Dict) -> bool:
        """Check if new results are better than old results"""
        # Compare number of structured data sets
        new_count = len(new_result.get('structured_data', []))
        old_count = len(old_result.get('structured_data', []))
        
        if new_count > old_count:
            return True
        
        # Compare confidence scores
        new_confidence = new_result.get('confidence_score', 0)
        old_confidence = old_result.get('confidence_score', 0)
        
        if new_confidence > old_confidence + 0.1:  # At least 10% better
            return True
        
        return False
    
    def _classify_data_type(self, df: pd.DataFrame) -> str:
        """Classify the type of data structure"""
        if df.empty:
            return 'empty'
        
        shape = df.shape
        if shape[1] > 1:
            return f'table_{shape[1]}col'
        elif 'Sequence' in df.columns:
            return 'sequential'
        else:
            return f'data_{shape[1]}col'
    
    def _save_csv_files(self, pdf_path: str, data_sets: List[Dict], orientation_info: List[Dict]) -> Dict[str, str]:
        """Save all data as CSV files"""
        filename = os.path.basename(pdf_path).replace('.pdf', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output = os.path.join(self.config.OUTPUT_DIR, f"{filename}_{timestamp}")
        
        output_files = {'csv': []}
        
        # Save orientation info
        orient_df = pd.DataFrame(orientation_info)
        orient_csv = f"{base_output}_orientation_info.csv"
        orient_df.to_csv(orient_csv, index=False)
        print(f"🧭 Orientation info saved to: {orient_csv}")
        
        # Save each data set as a separate CSV file
        for i, data_info in enumerate(data_sets):
            page_num = data_info['page']
            data_type = data_info['data_type']
            orientation = data_info['orientation']
            
            csv_filename = f"{base_output}_page{page_num}_{data_type}_{orientation}.csv"
            
            try:
                data_info['dataframe'].to_csv(csv_filename, index=False, encoding='utf-8')
                output_files['csv'].append(csv_filename)
                
                df = data_info['dataframe']
                print(f"💾 Saved {data_type} from page {page_num} ({orientation}): {csv_filename}")
                print(f"   📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show detailed preview for tables
                if 'table' in data_type and not df.empty:
                    print(f"   📋 Columns: {list(df.columns)}")
                    print(f"   👀 Data preview:")
                    print(df.head().to_string(index=False))
                    print()
                
            except Exception as e:
                print(f"❌ Failed to save {data_type}: {e}")
        
        # Create a master CSV with all data
        if data_sets:
            master_csv = f"{base_output}_MASTER_ALL_DATA.csv"
            try:
                master_dfs = []
                for data_info in data_sets:
                    df = data_info['dataframe'].copy()
                    df.insert(0, 'Source_Page', data_info['page'])
                    df.insert(1, 'Data_Type', data_info['data_type'])
                    df.insert(2, 'Orientation', data_info['orientation'])
                    df.insert(3, 'Block_Count', data_info['block_count'])
                    master_dfs.append(df)
                
                if master_dfs:
                    master_df = pd.concat(master_dfs, ignore_index=True)
                    master_df.to_csv(master_csv, index=False, encoding='utf-8')
                    output_files['master_csv'] = master_csv
                    print(f"📚 Master CSV with all data saved to: {master_csv}")
            except Exception as e:
                print(f"❌ Failed to create master CSV: {e}")
        
        return output_files
    
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

def main():
    parser = argparse.ArgumentParser(description='Advanced OCR with Auto-Orientation')
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
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"\n{status} - Processed: {args.file}")
        if result['success']:
            print(f"📄 Pages processed: {result['pages_processed']}")
            print(f"📊 Data sets found: {result['data_sets_found']}")
            print(f"💾 CSV files created: {len(result['output_files'].get('csv', []))}")
            if result.get('orientations'):
                print(f"🧭 Orientations used: {[o['orientation'] for o in result['orientations']]}")
        
    elif args.dir:
        results = app.process_directory(args.dir)
        successful = sum(1 for r in results if r['success'])
        total_data_sets = sum(r.get('data_sets_found', 0) for r in results if r['success'])
        print(f"\n📊 Processing Complete:")
        print(f"   Total files: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        print(f"   Total data sets extracted: {total_data_sets}")
        
    else:
        # Process default input directory
        results = app.process_directory()
        successful = sum(1 for r in results if r['success'])
        total_data_sets = sum(r.get('data_sets_found', 0) for r in results if r['success'])
        print(f"\n📊 Processing Complete:")
        print(f"   Total files: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        print(f"   Total data sets extracted: {total_data_sets}")

if __name__ == "__main__":
    main()