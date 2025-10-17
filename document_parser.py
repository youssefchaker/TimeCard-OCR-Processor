import re
from typing import Dict, List, Any
from datetime import datetime

class DocumentParser:
    def __init__(self):
        self.patterns = {
            'work_order_number': r'Number:\s*(\S+)',
            'date': r'(\d{1,2}/\d{1,2}/\d{4})',
            'time': r'(\d{1,2}:\d{2}\s*[AP]M)',
            'phone': r'x?(\d{3,})',
            'studio': r'Studio\s*(\d+(?:/\d+)?)'
        }
    
    def parse_work_order(self, ocr_blocks: List[Dict]) -> Dict[str, Any]:
        """Parse work order document structure"""
        result = {
            'document_type': 'work_order',
            'work_order_number': '',
            'dates': {},
            'client': {},
            'contact': {},
            'resources': [],
            'confidence_score': 0.0,
            'parse_errors': []
        }
        
        text_lines = [block['text'] for block in ocr_blocks]
        confidence_scores = []
        
        try:
            # Extract work order number
            wo_match = self._find_pattern(text_lines, self.patterns['work_order_number'])
            if wo_match:
                result['work_order_number'] = wo_match.group(1)
            
            # Extract dates and times
            result['dates'] = self._extract_dates_times(text_lines)
            
            # Extract client information
            result['client'] = self._extract_client_info(text_lines)
            
            # Extract contact information
            result['contact'] = self._extract_contact_info(text_lines)
            
            # Extract resources table
            result['resources'] = self._extract_resources(ocr_blocks)
            
            # Calculate overall confidence
            if ocr_blocks:
                result['confidence_score'] = sum(block['confidence'] for block in ocr_blocks) / len(ocr_blocks)
                
        except Exception as e:
            result['parse_errors'].append(str(e))
        
        return result
    
    def parse_timesheet(self, ocr_blocks: List[Dict]) -> Dict[str, Any]:
        """Parse timesheet document structure"""
        result = {
            'document_type': 'timesheet',
            'date': '',
            'employees': [],
            'confidence_score': 0.0,
            'parse_errors': []
        }
        
        try:
            # Extract date
            date_match = self._find_pattern([block['text'] for block in ocr_blocks], self.patterns['date'])
            if date_match:
                result['date'] = date_match.group(1)
            
            # Extract employee data (simplified - you'll need to customize based on actual structure)
            result['employees'] = self._extract_employee_data(ocr_blocks)
            
            if ocr_blocks:
                result['confidence_score'] = sum(block['confidence'] for block in ocr_blocks) / len(ocr_blocks)
                
        except Exception as e:
            result['parse_errors'].append(str(e))
        
        return result
    
    def _find_pattern(self, text_lines: List[str], pattern: str):
        """Find pattern in text lines"""
        for line in text_lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match
        return None
    
    def _extract_dates_times(self, text_lines: List[str]) -> Dict:
        """Extract date and time information"""
        dates = {}
        
        # Find begin/end times
        for i, line in enumerate(text_lines):
            if 'begin time:' in line.lower() and i + 1 < len(text_lines):
                dates['begin_time'] = text_lines[i + 1].strip()
            elif 'end time:' in line.lower() and i + 1 < len(text_lines):
                dates['end_time'] = text_lines[i + 1].strip()
        
        return dates
    
    def _extract_client_info(self, text_lines: List[str]) -> Dict:
        """Extract client information"""
        client = {}
        client_section = False
        
        for line in text_lines:
            if 'client' in line.lower():
                client_section = True
                continue
            
            if client_section:
                if 'sony' in line.lower():
                    client['name'] = line.strip()
                elif 'attn:' in line.lower():
                    client['contact_person'] = line.replace('Attn:', '').strip()
                elif any(keyword in line.lower() for keyword in ['contact', 'phone']):
                    break
        
        return client
    
    def _extract_contact_info(self, text_lines: List[str]) -> Dict:
        """Extract contact information"""
        contact = {}
        contact_section = False
        
        for line in text_lines:
            if 'contact:' in line.lower():
                contact_section = True
                contact['name'] = line.replace('Contact:', '').strip()
            elif contact_section and 'phone:' in line.lower():
                phone_match = re.search(self.patterns['phone'], line)
                if phone_match:
                    contact['phone'] = phone_match.group(1)
                break
        
        return contact
    
    def _extract_resources(self, ocr_blocks: List[Dict]) -> List[Dict]:
        """Extract resources from table structure"""
        resources = []
        
        # Look for table structure (this is simplified)
        table_started = False
        
        for block in ocr_blocks:
            text = block['text']
            
            # Detect table start
            if any(keyword in text.lower() for keyword in ['qty', 'group', 'resource']):
                table_started = True
                continue
            
            if table_started:
                # Simple table parsing - you'll need to enhance this
                if re.match(r'^\d+\s+[A-Za-z]', text):
                    parts = text.split()
                    if len(parts) >= 2:
                        resource = {
                            'quantity': parts[0],
                            'group': parts[1],
                            'name': ' '.join(parts[2:]) if len(parts) > 2 else '',
                            'confidence': block['confidence']
                        }
                        resources.append(resource)
        
        return resources
    
    def _extract_employee_data(self, ocr_blocks: List[Dict]) -> List[Dict]:
        """Extract employee timesheet data"""
        employees = []
        # Implement based on your timesheet structure
        return employees