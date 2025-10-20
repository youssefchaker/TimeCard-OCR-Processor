import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.cluster import DBSCAN

class DocumentParser:
    def __init__(self):
        pass
    
    def extract_structured_data(self, ocr_blocks: List[Dict]) -> List[pd.DataFrame]:
        """Extract any structured data - always return at least one DataFrame"""
        data_frames = []
        
        if not ocr_blocks:
            # Return empty dataframe instead of empty list
            return [pd.DataFrame([["No text detected"]])]
        
        # Method 1: Try to detect tables with multiple columns
        tables = self._detect_tables_advanced(ocr_blocks)
        for table in tables:
            if not table.empty:
                data_frames.append(table)
        
        # Method 2: If no multi-column tables found, create sequential export
        if not data_frames:
            sequential_data = self._create_sequential_export(ocr_blocks)
            data_frames.append(sequential_data)
        
        # Always return at least one DataFrame, even if empty
        if not data_frames:
            data_frames.append(pd.DataFrame([["No structured data found"]]))
        
        return data_frames
    
    def _detect_tables_advanced(self, ocr_blocks: List[Dict]) -> List[pd.DataFrame]:
        """Advanced table detection - focus on multi-column data"""
        tables = []
        
        # Try table detection even with minimal blocks
        if len(ocr_blocks) < 2:
            # For single blocks, create a simple row
            return self._create_simple_output(ocr_blocks)
        
        # Step 1: Cluster into rows
        rows = self._cluster_rows_robust(ocr_blocks)
        
        if len(rows) < 1:
            return []
        
        # Step 2: Try multiple column detection strategies
        column_structures = []
        
        # Strategy 1: Histogram-based (best for real tables)
        hist_structure = self._detect_columns_histogram(rows)
        if hist_structure and self._has_multiple_columns(hist_structure):
            column_structures.append(hist_structure)
        
        # Strategy 2: Fixed column approach
        fixed_structure = self._detect_columns_fixed(rows)
        if fixed_structure and self._has_multiple_columns(fixed_structure):
            column_structures.append(fixed_structure)
        
        # Strategy 3: Group by common patterns (for semi-structured data)
        pattern_structure = self._detect_columns_by_patterns(rows)
        if pattern_structure and self._has_multiple_columns(pattern_structure):
            column_structures.append(pattern_structure)
        
        # Create tables from all successful strategies
        for column_structure in column_structures:
            if column_structure and len(column_structure) >= 1:
                df = self._create_dataframe_from_table_data(column_structure)
                if df is not None and not df.empty:
                    tables.append(df)
        
        return tables
    
    def _has_multiple_columns(self, table_data: List[List[str]]) -> bool:
        """Check if the table data has multiple non-empty columns"""
        if not table_data:
            return False
        
        for row in table_data:
            non_empty_cols = sum(1 for cell in row if cell and str(cell).strip())
            if non_empty_cols > 1:
                return True
        
        return False
    
    def _create_simple_output(self, ocr_blocks: List[Dict]) -> List[pd.DataFrame]:
        """Create simple output for minimal blocks"""
        if not ocr_blocks:
            return [pd.DataFrame([["No text detected"]])]
        
        # Try to split single blocks into multiple columns if they contain commas or spaces
        data = []
        for block in ocr_blocks:
            text = block['text'].strip()
            
            # If text contains commas, split by commas
            if ',' in text:
                parts = [part.strip() for part in text.split(',')]
                data.append(parts)
            # If text contains multiple words, split by spaces (limit to 3 columns max)
            elif ' ' in text and len(text) > 10:
                words = text.split()
                if len(words) > 2:
                    # Try to create 2-3 columns
                    mid_point = len(words) // 2
                    col1 = ' '.join(words[:mid_point])
                    col2 = ' '.join(words[mid_point:])
                    data.append([col1, col2])
                else:
                    data.append([text])
            else:
                data.append([text])
        
        return [pd.DataFrame(data)]
    
    def _cluster_rows_robust(self, ocr_blocks: List[Dict]) -> List[List[Dict]]:
        """Robust row clustering - more permissive settings"""
        if not ocr_blocks:
            return []
        
        # For very few blocks, just put each in its own row
        if len(ocr_blocks) <= 3:
            return [[block] for block in ocr_blocks]
        
        # Use y-centers for clustering
        y_centers = np.array([
            block['bbox']['top'] + block['bbox']['height'] / 2 
            for block in ocr_blocks
        ]).reshape(-1, 1)
        
        # More permissive epsilon
        heights = [block['bbox']['height'] for block in ocr_blocks]
        median_height = np.median(heights) if heights else 20
        eps = median_height * 1.2  # More permissive
        
        # DBSCAN clustering with minimal samples
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
    
    def _detect_columns_histogram(self, rows: List[List[Dict]]) -> List[List[str]]:
        """Histogram-based column detection"""
        if not rows:
            return []
        
        all_x = []
        for row in rows:
            for block in row:
                all_x.extend([
                    block['bbox']['left'],
                    block['bbox']['left'] + block['bbox']['width'] / 2,
                ])
        
        if not all_x:
            return []
        
        # Create histogram
        hist, bin_edges = np.histogram(all_x, bins=min(30, len(rows) * 2))
        
        # Find peaks (common x-positions)
        peaks = []
        peak_threshold = np.mean(hist) * 0.7  # Lower threshold
        for i in range(1, len(hist) - 1):
            if hist[i] > peak_threshold:
                peaks.append(bin_edges[i])
        
        if len(peaks) >= 2:  # Only use if we found multiple columns
            peaks.sort()
            # Create boundaries around peaks
            boundaries = [min(all_x) - 10]
            for i in range(len(peaks) - 1):
                boundary = (peaks[i] + peaks[i+1]) / 2
                boundaries.append(boundary)
            boundaries.append(max(all_x) + 10)
            
            # Assign text to columns
            table_data = []
            for row in rows:
                row_data = [''] * (len(boundaries) - 1)
                for block in row:
                    col_idx = self._find_column_index(block, boundaries)
                    if col_idx < len(row_data):
                        if row_data[col_idx]:
                            row_data[col_idx] += ' ' + block['text']
                        else:
                            row_data[col_idx] = block['text']
                table_data.append(row_data)
            
            return table_data
        
        return []
    
    def _detect_columns_fixed(self, rows: List[List[Dict]]) -> List[List[str]]:
        """Fixed column approach based on most common column count"""
        if not rows:
            return []
        
        # Find most common column count
        col_counts = [len(row) for row in rows]
        if not col_counts:
            return []
        
        # Use the maximum column count found (but at least 2)
        max_cols = max(max(col_counts), 2)
        
        table_data = []
        for row in rows:
            row_data = [block['text'] for block in row]
            # Pad with empty strings if needed
            while len(row_data) < max_cols:
                row_data.append('')
            table_data.append(row_data)
        
        return table_data
    
    def _detect_columns_by_patterns(self, rows: List[List[Dict]]) -> List[List[str]]:
        """Detect columns based on common patterns in the data"""
        if not rows:
            return []
        
        # Look for rows that have clear separators or patterns
        table_data = []
        
        for row in rows:
            row_texts = [block['text'] for block in row]
            
            # If we have multiple blocks, use them as columns
            if len(row_texts) >= 2:
                table_data.append(row_texts)
            else:
                # For single-block rows, try to split by common patterns
                text = row_texts[0] if row_texts else ""
                if text:
                    # Try to split by commas, spaces, or other patterns
                    if ',' in text and len(text) > 10:
                        parts = [part.strip() for part in text.split(',')]
                        table_data.append(parts)
                    elif '  ' in text:  # Double space as separator
                        parts = [part.strip() for part in text.split('  ') if part.strip()]
                        if len(parts) >= 2:
                            table_data.append(parts)
                        else:
                            table_data.append([text])
                    else:
                        table_data.append([text])
        
        return table_data
    
    def _find_column_index(self, block: Dict, boundaries: List[float]) -> int:
        """Find which column a block belongs to"""
        block_center = block['bbox']['left'] + block['bbox']['width'] / 2
        
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= block_center < boundaries[i + 1]:
                return i
        
        return len(boundaries) - 2
    
    def _create_dataframe_from_table_data(self, table_data: List[List[str]]) -> pd.DataFrame:
        """Create DataFrame from table data"""
        if not table_data:
            return pd.DataFrame()
        
        # Remove empty rows
        non_empty_data = [row for row in table_data if any(cell.strip() for cell in row)]
        if not non_empty_data:
            return pd.DataFrame()
        
        # Create DataFrame without column names
        df = pd.DataFrame(non_empty_data)
        
        # Clean the DataFrame
        df = self._clean_dataframe(df)
        return df
    
    def _create_sequential_export(self, ocr_blocks: List[Dict]) -> pd.DataFrame:
        """Create sequential export with potential multi-column data"""
        if not ocr_blocks:
            return pd.DataFrame([["No text detected"]])
        
        # Sort blocks by reading order
        sorted_blocks = sorted(ocr_blocks, key=lambda x: (x['bbox']['top'], x['bbox']['left']))
        
        # Try to create multi-column data when possible
        data = []
        current_row = []
        row_threshold = 20
        
        for i, block in enumerate(sorted_blocks):
            if not current_row:
                current_row.append(block)
                continue
            
            # Check if this block is on the same row as previous
            last_block = current_row[-1]
            last_bottom = last_block['bbox']['top'] + last_block['bbox']['height']
            current_top = block['bbox']['top']
            
            if current_top <= last_bottom + row_threshold:
                current_row.append(block)
            else:
                # New row - add current row to data
                row_texts = [b['text'] for b in current_row]
                data.append(row_texts)
                current_row = [block]
        
        # Add the last row
        if current_row:
            row_texts = [b['text'] for b in current_row]
            data.append(row_texts)
        
        # If we only have single-column data, try to split some rows
        if all(len(row) == 1 for row in data) and len(data) > 3:
            # Try to find rows that can be split
            enhanced_data = []
            for row in data:
                text = row[0]
                if ',' in text and len(text) > 15:
                    parts = [part.strip() for part in text.split(',')]
                    enhanced_data.append(parts)
                else:
                    enhanced_data.append(row)
            data = enhanced_data
        
        return pd.DataFrame(data)
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the DataFrame - ensure proper data types and remove completely empty columns"""
        if df.empty:
            return df
        
        # Remove completely empty columns
        non_empty_cols = []
        for col in df.columns:
            if not df[col].isna().all() and not (df[col] == '').all():
                non_empty_cols.append(col)
        
        if non_empty_cols:
            df = df[non_empty_cols]
        else:
            return pd.DataFrame()
        
        # Fill NaN with empty strings
        df = df.fillna('')
        
        # Clean string columns
        for col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip() if x else '')
        
        return df
    
    def parse_document(self, ocr_blocks: List[Dict]) -> Dict[str, Any]:
        """Generic document parser - always returns structured data"""
        result = {
            'structured_data': []
        }
        
        try:
            # Extract structured data - this now always returns at least one DataFrame
            result['structured_data'] = self.extract_structured_data(ocr_blocks)
        except Exception as e:
            # Even on error, return empty DataFrame
            result['structured_data'] = [pd.DataFrame([["Error processing page"]])]
        
        return result