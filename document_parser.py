import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.cluster import DBSCAN

class DocumentParser:
    def __init__(self):
        # No patterns - completely spatial analysis only
        pass
    
    def extract_structured_data(self, ocr_blocks: List[Dict]) -> List[pd.DataFrame]:
        """Extract any structured data using pure spatial analysis"""
        data_frames = []
        
        if not ocr_blocks:
            return data_frames
        
        print(f"   🔍 Analyzing {len(ocr_blocks)} text blocks for spatial structure...")
        
        # Method 1: Advanced table detection using spatial clustering
        tables = self._detect_tables_advanced(ocr_blocks)
        for i, table in enumerate(tables):
            if not table.empty:
                data_frames.append(table)
                print(f"   📊 Found table {i+1}: {table.shape[1]} columns × {table.shape[0]} rows")
        
        # Method 2: If no tables found, create a sequential text export
        if not data_frames:
            sequential_data = self._create_sequential_export(ocr_blocks)
            data_frames.append(sequential_data)
            print(f"   📝 Created sequential text export: {sequential_data.shape[0]} lines")
        
        return data_frames
    
    def _detect_tables_advanced(self, ocr_blocks: List[Dict]) -> List[pd.DataFrame]:
        """Advanced table detection using multi-level clustering"""
        tables = []
        
        if len(ocr_blocks) < 2:
            return tables
        
        # Step 1: Cluster into rows using robust DBSCAN
        rows = self._cluster_rows_robust(ocr_blocks)
        
        if len(rows) < 2:
            return tables
        
        # Step 2: Detect column structure using multiple methods
        column_structure = self._detect_column_structure(rows)
        
        if not column_structure:
            return tables
        
        # Step 3: Create table data
        for table_data in column_structure:
            if len(table_data) >= 1:
                df = self._create_dataframe_from_table_data(table_data)
                if df is not None and not df.empty:
                    tables.append(df)
        
        return tables
    
    def _cluster_rows_robust(self, ocr_blocks: List[Dict]) -> List[List[Dict]]:
        """Robust row clustering using multiple techniques"""
        if not ocr_blocks:
            return []
        
        # Method 1: Use y-centers for clustering
        y_centers = np.array([
            block['bbox']['top'] + block['bbox']['height'] / 2 
            for block in ocr_blocks
        ]).reshape(-1, 1)
        
        # Adaptive epsilon based on text height statistics
        heights = [block['bbox']['height'] for block in ocr_blocks]
        median_height = np.median(heights)
        eps = median_height * 0.8  # More aggressive clustering
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=1)
        row_labels = dbscan.fit_predict(y_centers)
        
        # Group blocks by row
        rows_dict = {}
        for i, label in enumerate(row_labels):
            if label not in rows_dict:
                rows_dict[label] = []
            rows_dict[label].append(ocr_blocks[i])
        
        # Convert to list and sort rows by y-position
        rows = []
        for label in sorted(rows_dict.keys()):
            row_blocks = rows_dict[label]
            # Sort blocks in row by x-position (left to right)
            row_blocks.sort(key=lambda x: x['bbox']['left'])
            rows.append(row_blocks)
        
        # Sort rows by average y-position (top to bottom)
        rows.sort(key=lambda row: np.mean([block['bbox']['top'] for block in row]))
        
        return rows
    
    def _detect_column_structure(self, rows: List[List[Dict]]) -> List[List[List[str]]]:
        """Detect column structure using horizontal alignment analysis"""
        if not rows:
            return []
        
        # Analyze horizontal distribution across all rows
        all_x_positions = []
        for row in rows:
            for block in row:
                all_x_positions.append(block['bbox']['left'])
        
        if not all_x_positions:
            return []
        
        # Use histogram-based column detection
        column_boundaries = self._histogram_column_detection(rows)
        
        if not column_boundaries:
            # Fallback: dynamic column detection per row group
            return self._dynamic_column_detection(rows)
        
        # Assign text to columns based on boundaries
        table_data = []
        for row in rows:
            row_data = [''] * (len(column_boundaries) - 1)
            for block in row:
                col_idx = self._find_column_index(block, column_boundaries)
                if col_idx < len(row_data):
                    if row_data[col_idx]:
                        row_data[col_idx] += ' ' + block['text']
                    else:
                        row_data[col_idx] = block['text']
            table_data.append(row_data)
        
        return [table_data]
    
    def _histogram_column_detection(self, rows: List[List[Dict]]) -> List[float]:
        """Use histogram of x-positions to detect column boundaries"""
        all_x = []
        for row in rows:
            for block in row:
                all_x.extend([
                    block['bbox']['left'],  # Left edge
                    block['bbox']['left'] + block['bbox']['width'] / 2,  # Center
                    block['bbox']['left'] + block['bbox']['width']  # Right edge
                ])
        
        if not all_x:
            return []
        
        # Create histogram of x-positions
        hist, bin_edges = np.histogram(all_x, bins=min(50, len(all_x)))
        
        # Find peaks in the histogram (common x-positions)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                peaks.append(bin_edges[i])
        
        # Use peaks as column centers, convert to boundaries
        if peaks:
            peaks.sort()
            boundaries = []
            # Add left boundary
            boundaries.append(min(all_x) - 10)
            # Add boundaries between peaks
            for i in range(len(peaks) - 1):
                boundary = (peaks[i] + peaks[i+1]) / 2
                boundaries.append(boundary)
            # Add right boundary
            boundaries.append(max(all_x) + 10)
            return boundaries
        
        return []
    
    def _dynamic_column_detection(self, rows: List[List[Dict]]) -> List[List[List[str]]]:
        """Dynamic column detection that adapts to row structure"""
        if not rows:
            return []
        
        # Group rows by similar column count
        row_groups = {}
        for row in rows:
            col_count = len(row)
            if col_count not in row_groups:
                row_groups[col_count] = []
            row_groups[col_count].append(row)
        
        # Find the most common column count
        if not row_groups:
            return []
        
        max_count = max(row_groups.keys(), key=lambda k: len(row_groups[k]))
        main_rows = row_groups[max_count]
        
        # Create table data using fixed columns for the main group
        table_data = []
        for row in main_rows:
            row_data = [block['text'] for block in row]
            # Pad with empty strings if needed
            while len(row_data) < max_count:
                row_data.append('')
            table_data.append(row_data)
        
        return [table_data]
    
    def _find_column_index(self, block: Dict, boundaries: List[float]) -> int:
        """Find which column a block belongs to"""
        block_center = block['bbox']['left'] + block['bbox']['width'] / 2
        
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= block_center < boundaries[i + 1]:
                return i
        
        return len(boundaries) - 2  # Last column
    
    def _create_dataframe_from_table_data(self, table_data: List[List[str]]) -> pd.DataFrame:
        """Create DataFrame from table data with intelligent header detection"""
        if not table_data:
            return pd.DataFrame()
        
        # Try to detect if first row is a header
        if len(table_data) > 1 and self._is_potential_header(table_data[0], table_data[1:]):
            headers = [str(cell).strip() for cell in table_data[0]]
            data_rows = table_data[1:]
        else:
            # Use generic column names
            num_cols = len(table_data[0])
            headers = [f'Col_{i+1}' for i in range(num_cols)]
            data_rows = table_data
        
        # Create DataFrame
        if data_rows:
            df = pd.DataFrame(data_rows, columns=headers)
        else:
            df = pd.DataFrame([headers], columns=headers)
        
        # Clean the DataFrame
        df = self._clean_dataframe(df)
        return df
    
    def _is_potential_header(self, header_row: List[str], data_rows: List[List[str]]) -> bool:
        """Check if first row could be a header based on content characteristics"""
        if not header_row or not data_rows:
            return False
        
        # Header cells are typically shorter and more concise
        header_avg_len = np.mean([len(str(cell)) for cell in header_row if cell])
        data_avg_len = np.mean([len(str(cell)) for row in data_rows[:3] for cell in row if cell])
        
        # Header rows often have more text-like content vs data-like content
        header_text_ratio = sum(1 for cell in header_row if cell and any(c.isalpha() for c in str(cell))) / len(header_row)
        data_text_ratio = sum(1 for row in data_rows[:3] for cell in row if cell and any(c.isalpha() for c in str(cell))) / (len(data_rows[:3]) * len(header_row))
        
        return header_avg_len < data_avg_len * 2 and header_text_ratio > data_text_ratio * 0.8
    
    def _create_sequential_export(self, ocr_blocks: List[Dict]) -> pd.DataFrame:
        """Create a sequential export when no table structure is found"""
        sequential_data = []
        
        # Sort blocks by reading order (top to bottom, left to right)
        sorted_blocks = sorted(ocr_blocks, key=lambda x: (x['bbox']['top'], x['bbox']['left']))
        
        for i, block in enumerate(sorted_blocks):
            sequential_data.append({
                'Sequence': i + 1,
                'Text': block['text'],
                'Confidence': block['confidence'],
                'X': block['bbox']['left'],
                'Y': block['bbox']['top'],
                'Width': block['bbox']['width'],
                'Height': block['bbox']['height']
            })
        
        return pd.DataFrame(sequential_data)
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the DataFrame"""
        # Remove completely empty columns
        df = df.loc[:, ~df.isna().all()]
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Fill NaN with empty strings
        df = df.fillna('')
        
        # Clean string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: str(x).strip() if x else '')
        
        return df
    
    def parse_document(self, ocr_blocks: List[Dict]) -> Dict[str, Any]:
        """Generic document parser"""
        result = {
            'document_type': 'generic',
            'structured_data': [],
            'confidence_score': 0.0,
            'text_blocks_count': len(ocr_blocks)
        }
        
        try:
            # Extract structured data
            result['structured_data'] = self.extract_structured_data(ocr_blocks)
            
            # Calculate overall confidence
            if ocr_blocks:
                result['confidence_score'] = np.mean([block['confidence'] for block in ocr_blocks])
                
        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Parser error: {e}")
        
        return result