import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.cluster import DBSCAN

class DocumentParser:
    def __init__(self):
        pass
    
    def extract_structured_data(self, ocr_blocks: List[Dict]) -> List[pd.DataFrame]:
        """Extract structured data using spatial gap analysis"""
        data_frames = []
        
        if not ocr_blocks:
            return [pd.DataFrame([["No text detected"]])]
        
        # Method 1: Spatial table detection using actual gaps between elements
        tables = self._detect_spatial_tables(ocr_blocks)
        for table in tables:
            if not table.empty:
                data_frames.append(table)
        
        # Method 2: Fallback to sequential if no spatial structure found
        if not data_frames:
            sequential_data = self._create_spatial_sequential(ocr_blocks)
            data_frames.append(sequential_data)
        
        return data_frames
    
    def _detect_spatial_tables(self, ocr_blocks: List[Dict]) -> List[pd.DataFrame]:
        """Detect tables based on actual spatial gaps between elements"""
        tables = []
        
        if len(ocr_blocks) < 2:
            return [self._create_single_column(ocr_blocks)]
        
        # Step 1: Cluster into rows based on vertical alignment
        rows = self._cluster_rows_robust(ocr_blocks)
        
        if len(rows) < 1:
            return []
        
        # Step 2: For each row, detect columns based on horizontal gaps
        table_data = self._detect_columns_by_gaps(rows)
        
        if table_data:
            df = self._create_dataframe_from_table_data(table_data)
            if df is not None and not df.empty:
                tables.append(df)
        
        return tables
    
    def _detect_columns_by_gaps(self, rows: List[List[Dict]]) -> List[List[str]]:
        """Detect columns based on actual horizontal gaps between text elements"""
        if not rows:
            return []
        
        # Analyze gaps across all rows to find common column boundaries
        all_gaps = self._analyze_horizontal_gaps(rows)
        
        if not all_gaps:
            # Fallback: use per-row gap detection
            return self._detect_columns_per_row(rows)
        
        # Use the most common gaps as column separators
        column_boundaries = self._find_optimal_column_boundaries(all_gaps, rows)
        
        if not column_boundaries:
            return self._detect_columns_per_row(rows)
        
        # Assign text to columns based on boundaries
        table_data = []
        for row in rows:
            row_data = self._assign_row_to_columns(row, column_boundaries)
            table_data.append(row_data)
        
        return table_data
    
    def _analyze_horizontal_gaps(self, rows: List[List[Dict]]) -> List[float]:
        """Analyze horizontal gaps between text elements across all rows"""
        all_gaps = []
        
        for row in rows:
            if len(row) < 2:
                continue
                
            # Sort blocks by x-position
            sorted_blocks = sorted(row, key=lambda x: x['bbox']['left'])
            
            # Calculate gaps between consecutive blocks
            for i in range(len(sorted_blocks) - 1):
                current_block = sorted_blocks[i]
                next_block = sorted_blocks[i + 1]
                
                # Calculate gap: start of next block minus end of current block
                current_end = current_block['bbox']['left'] + current_block['bbox']['width']
                next_start = next_block['bbox']['left']
                gap = next_start - current_end
                
                # Only consider significant gaps (not overlapping or very close)
                if gap > 5:
                    all_gaps.append(gap)
        
        return all_gaps
    
    def _find_optimal_column_boundaries(self, all_gaps: List[float], rows: List[List[Dict]]) -> List[float]:
        """Find optimal column boundaries based on gap analysis"""
        if not all_gaps or not rows:
            return []
        
        # Use clustering to find common gap sizes
        gap_array = np.array(all_gaps).reshape(-1, 1)
        
        # Use DBSCAN to cluster gap sizes
        dbscan = DBSCAN(eps=10, min_samples=2)
        gap_labels = dbscan.fit_predict(gap_array)
        
        # Find the most common gap size (largest cluster)
        cluster_gaps = {}
        for i, label in enumerate(gap_labels):
            if label == -1: 
                continue
            if label not in cluster_gaps:
                cluster_gaps[label] = []
            cluster_gaps[label].append(all_gaps[i])
        
        if not cluster_gaps:
            return []
        
        # Find the cluster with the most gaps
        largest_cluster = max(cluster_gaps.values(), key=len)
        typical_gap = np.median(largest_cluster)
        
        # Now use this typical gap to establish column boundaries
        # Collect all x-positions to establish common columns
        all_x_positions = []
        for row in rows:
            for block in row:
                all_x_positions.append(block['bbox']['left'])
                all_x_positions.append(block['bbox']['left'] + block['bbox']['width'])
        
        if not all_x_positions:
            return []
        
        all_x_positions.sort()
        
        # Group x-positions that are close together
        boundaries = []
        current_group = []
        grouping_threshold = typical_gap / 2
        
        for x in all_x_positions:
            if not current_group:
                current_group.append(x)
            elif x - current_group[-1] < grouping_threshold:
                current_group.append(x)
            else:
                # New group - add boundary at average of group
                boundaries.append(sum(current_group) / len(current_group))
                current_group = [x]
        
        if current_group:
            boundaries.append(sum(current_group) / len(current_group))
        
        # Ensure boundaries cover the entire width
        min_x = min(all_x_positions)
        max_x = max(all_x_positions)
        if boundaries and boundaries[0] > min_x:
            boundaries.insert(0, min_x - 10)
        if boundaries and boundaries[-1] < max_x:
            boundaries.append(max_x + 10)
        
        return boundaries
    
    def _detect_columns_per_row(self, rows: List[List[Dict]]) -> List[List[str]]:
        """Detect columns for each row individually based on gaps"""
        table_data = []
        
        for row in rows:
            if len(row) == 1:
                # Single block - try to split by internal patterns
                text = row[0]['text']
                if ', ' in text:
                    parts = [part.strip() for part in text.split(', ')]
                    table_data.append(parts)
                elif '  ' in text:
                    parts = [part.strip() for part in text.split('  ') if part.strip()]
                    table_data.append(parts)
                else:
                    table_data.append([text])
            else:
                # Multiple blocks - use them as separate columns
                row_data = [block['text'] for block in row]
                table_data.append(row_data)
        
        return table_data
    
    def _assign_row_to_columns(self, row: List[Dict], boundaries: List[float]) -> List[str]:
        """Assign blocks in a row to columns based on spatial boundaries"""
        if not row or not boundaries:
            return []
        
        # Initialize row with empty columns
        row_data = [''] * (len(boundaries) - 1)
        
        for block in row:
            block_center = block['bbox']['left'] + block['bbox']['width'] / 2
            
            # Find which column this block belongs to
            col_index = self._find_spatial_column(block_center, boundaries)
            
            if col_index < len(row_data):
                if row_data[col_index]:
                    # If column already has text, append with space
                    row_data[col_index] += ' ' + block['text']
                else:
                    row_data[col_index] = block['text']
        
        return row_data
    
    def _find_spatial_column(self, x_position: float, boundaries: List[float]) -> int:
        """Find which column a position belongs to based on boundaries"""
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= x_position < boundaries[i + 1]:
                return i
        
        return len(boundaries) - 2  # Last column
    
    def _cluster_rows_robust(self, ocr_blocks: List[Dict]) -> List[List[Dict]]:
        """Robust row clustering based on vertical alignment"""
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
        
        # Adaptive epsilon based on text height
        heights = [block['bbox']['height'] for block in ocr_blocks]
        median_height = np.median(heights) if heights else 20
        eps = median_height * 1.2
        
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
    
    def _create_single_column(self, ocr_blocks: List[Dict]) -> pd.DataFrame:
        """Create single column output"""
        if not ocr_blocks:
            return pd.DataFrame([["No text detected"]])
        
        data = []
        for block in ocr_blocks:
            data.append([block['text']])
        
        return pd.DataFrame(data)
    
    def _create_spatial_sequential(self, ocr_blocks: List[Dict]) -> pd.DataFrame:
        """Create sequential output with spatial awareness"""
        if not ocr_blocks:
            return pd.DataFrame([["No text detected"]])
        
        # Sort by reading order (top to bottom, left to right)
        sorted_blocks = sorted(ocr_blocks, key=lambda x: (x['bbox']['top'], x['bbox']['left']))
        
        # Try to group into logical rows based on vertical proximity
        data = []
        current_row = []
        row_threshold = 25  # pixels
        
        for block in sorted_blocks:
            if not current_row:
                current_row.append(block)
                continue
            
            # Check vertical alignment
            last_block = current_row[-1]
            last_bottom = last_block['bbox']['top'] + last_block['bbox']['height']
            current_top = block['bbox']['top']
            
            if current_top <= last_bottom + row_threshold:
                current_row.append(block)
            else:
                # New row
                row_texts = [b['text'] for b in current_row]
                data.append(row_texts)
                current_row = [block]
        
        # Add final row
        if current_row:
            row_texts = [b['text'] for b in current_row]
            data.append(row_texts)
        
        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in data) if data else 1
        for i in range(len(data)):
            if len(data[i]) < max_cols:
                data[i].extend([''] * (max_cols - len(data[i])))
        
        return pd.DataFrame(data)
    
    def _create_dataframe_from_table_data(self, table_data: List[List[str]]) -> pd.DataFrame:
        """Create DataFrame from table data"""
        if not table_data:
            return pd.DataFrame()
        
        # Remove completely empty rows
        non_empty_data = [row for row in table_data if any(cell.strip() for cell in row)]
        if not non_empty_data:
            return pd.DataFrame()
        
        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in non_empty_data)
        clean_data = []
        for row in non_empty_data:
            if len(row) < max_cols:
                row.extend([''] * (max_cols - len(row)))
            clean_data.append(row)
        
        # Create DataFrame without column headers
        df = pd.DataFrame(clean_data)
        
        # Clean the data
        for col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).strip() if x else '')
        
        return df
    
    def parse_document(self, ocr_blocks: List[Dict]) -> Dict[str, Any]:
        """Generic document parser"""
        result = {
            'structured_data': []
        }
        
        try:
            result['structured_data'] = self.extract_structured_data(ocr_blocks)
        except Exception as e:
            result['structured_data'] = [pd.DataFrame([["Error processing page"]])]
        
        return result