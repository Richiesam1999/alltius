import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import camelot.io as camelot
import cv2
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TableExtractor:
    """A class to extract tables from PDFs and images for RAG systems"""
    
    def __init__(self, tesseract_path=None):
        """Initialize the extractor with optional tesseract path"""
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def extract_from_pdf(self, pdf_path, pages='all'):
        """Extract tables from PDF using Camelot"""
        logger.info(f"Extracting tables from PDF: {pdf_path}")
        try:
            # Use Camelot's lattice method for structured tables with lines
            tables_lattice = camelot.read_pdf(
                pdf_path, 
                pages=pages, 
                flavor='lattice',
                suppress_stdout=True
            )
            
            # Use Camelot's stream method for tables without clear boundaries
            tables_stream = camelot.read_pdf(
                pdf_path, 
                pages=pages, 
                flavor='stream',
                suppress_stdout=True
            )
            
            # Combine results, prioritizing higher accuracy tables
            all_tables = []
            
            # Process lattice tables
            for i, table in enumerate(tables_lattice):
                if table.accuracy > 80:  # Only accept reasonably accurate tables
                    df = table.df
                    all_tables.append({
                        'table_id': f'lattice_{i}',
                        'page': table.page,
                        'accuracy': table.accuracy,
                        'data': df,
                        'text': self._dataframe_to_text(df)
                    })
            
            # Process stream tables
            for i, table in enumerate(tables_stream):
                if table.accuracy > 80:  # Only accept reasonably accurate tables
                    df = table.df
                    all_tables.append({
                        'table_id': f'stream_{i}',
                        'page': table.page,
                        'accuracy': table.accuracy,
                        'data': df,
                        'text': self._dataframe_to_text(df)
                    })
            
            logger.info(f"Successfully extracted {len(all_tables)} tables from PDF")
            return all_tables
        
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {str(e)}")
            return []
    
    def extract_from_image(self, image_path):
        """Extract tables from images using OpenCV and Tesseract"""
        logger.info(f"Extracting tables from image: {image_path}")
        try:
            # Read the image
            img = cv2.imread(image_path)
            
            # Preprocess the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Find contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Find table contours - typically large rectangles
            potential_tables = []
            for c in contours[:10]:  # Check top 10 largest contours
                # Approximate the contour to a polygon
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                
                # If it's a rectangle and large enough
                if len(approx) == 4 and cv2.contourArea(c) > img.shape[0] * img.shape[1] * 0.05:
                    potential_tables.append(c)
            
            # Extract all text from the image as fallback
            full_text = pytesseract.image_to_string(Image.open(image_path))
            
            # Try to extract structured table data
            tables = []
            
            # If we found potential tables
            if potential_tables:
                # Process each potential table
                for i, table_contour in enumerate(potential_tables):
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(table_contour)
                    
                    # Extract table region
                    table_img = img[y:y+h, x:x+w]
                    
                    # Extract text using Tesseract with table structure recognition
                    table_text = pytesseract.image_to_string(
                        Image.fromarray(table_img),
                        config='--psm 6'  # Assume a single uniform block of text
                    )
                    
                    # Extract table data using image_to_data for structured data
                    table_data = pytesseract.image_to_data(
                        Image.fromarray(table_img),
                        output_type=pytesseract.Output.DATAFRAME
                    )
                    
                    # Filter rows with text
                    table_data = table_data[table_data['text'].notnull() & (table_data['text'].str.strip() != '')]
                    
                    tables.append({
                        'table_id': f'image_table_{i}',
                        'bbox': (x, y, w, h),
                        'text': table_text,
                        'data': table_data,
                        'structured_text': self._process_table_data(table_data)
                    })
            
            # If no tables were detected, use full text
            if not tables:
                tables.append({
                    'table_id': 'full_image',
                    'bbox': None,
                    'text': full_text,
                    'data': None,
                    'structured_text': self._extract_table_structure_from_text(full_text)
                })
            
            logger.info(f"Successfully extracted {len(tables)} tables from image")
            return tables
        
        except Exception as e:
            logger.error(f"Error extracting tables from image: {str(e)}")
            return []
    
    def _process_table_data(self, data):
        """Process OCR data to create structured table text"""
        try:
            # Group by top coordinate (row)
            data['line_num'] = data['top'].map(lambda t: int(t / 10) * 10)
            
            # Sort by line number and left position
            data = data.sort_values(['line_num', 'left'])
            
            # Construct rows
            rows = []
            for line_num, group in data.groupby('line_num'):
                row = ' | '.join(group['text'].astype(str))
                rows.append(row)
            
            return '\n'.join(rows)
        except Exception as e:
            logger.error(f"Error processing table data: {str(e)}")
            return ""
    
    def _extract_table_structure_from_text(self, text):
        """Try to extract table structure from plain text"""
        # Simple heuristic: look for consistent patterns of spaces or pipe characters
        lines = text.split('\n')
        
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        
        # Look for delimiter patterns
        if any('|' in line for line in lines):
            # Process as pipe-delimited
            return '\n'.join(lines)
        
        # Look for consistent spacing that might indicate columns
        col_positions = []
        for line in lines[:min(5, len(lines))]:  # Check first few lines
            positions = [m.start() for m in re.finditer(r'\s{2,}', line)]
            if positions:
                col_positions.extend(positions)
        
        # Count most common positions
        if col_positions:
            # Create bins for position clusters
            bins = {}
            for pos in col_positions:
                assigned = False
                for bin_center in list(bins.keys()):
                    if abs(pos - bin_center) < 5:  # Within 5 chars, consider same column
                        bins[bin_center] = bins[bin_center] + 1
                        assigned = True
                        break
                if not assigned:
                    bins[pos] = 1
            
            # Sort positions
            sorted_positions = sorted([pos for pos, count in bins.items() if count > 1])
            
            if sorted_positions:
                # Convert text to table using positions
                structured_lines = []
                for line in lines:
                    if not line.strip():
                        continue
                    
                    # Insert pipe separators at column positions
                    new_line = list(line)
                    for pos in sorted(sorted_positions, reverse=True):
                        if pos < len(new_line):
                            new_line.insert(pos, ' | ')
                    structured_lines.append(''.join(new_line))
                
                return '\n'.join(structured_lines)
        
        # If no clear structure, return as is
        return text
    
    def _dataframe_to_text(self, df):
        """Convert DataFrame to structured text representation"""
        # Replace NaN with empty strings
        df = df.fillna('')
        
        # Convert to string representation
        rows = []
        for _, row in df.iterrows():
            rows.append(' | '.join(str(cell).strip() for cell in row))
        
        return '\n'.join(rows)
    
    def extract_from_file(self, file_path):
        """Extract tables from file (detects file type and uses appropriate method)"""
        _, ext = os.path.splitext(file_path.lower())
        
        if ext in ['.pdf']:
            return self.extract_from_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return self.extract_from_image(file_path)
        else:
            logger.error(f"Unsupported file type: {ext}")
            return []
    
    def save_results(self, tables, output_dir):
        """Save extracted tables to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each table
        for i, table in enumerate(tables):
            # Save as CSV if DataFrame is available
            if isinstance(table.get('data', None), pd.DataFrame):
                output_path = os.path.join(output_dir, f"table_{i}.csv")
                table['data'].to_csv(output_path, index=False)
            
            # Save text representation
            output_text_path = os.path.join(output_dir, f"table_{i}.txt")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(table.get('text', '') or table.get('structured_text', ''))
        
        # Save combined text for RAG
        combined_text = "\n\n".join(table.get('text', '') or table.get('structured_text', '') 
                                   for table in tables)
        
        combined_path = os.path.join(output_dir, "combined_tables.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        logger.info(f"Saved {len(tables)} tables to {output_dir}")
        return combined_path

def main():
    parser = argparse.ArgumentParser(description='Extract tables from PDFs and images for RAG systems')
    parser.add_argument('input_file', help='Path to input file (PDF or image)')
    parser.add_argument('--output_dir', default='extracted_tables', help='Output directory for extracted tables')
    parser.add_argument('--tesseract_path', help='Path to Tesseract executable (if not in PATH)')
    
    args = parser.parse_args()
    
    extractor = TableExtractor(tesseract_path=args.tesseract_path)
    tables = extractor.extract_from_file(args.input_file)
    
    if tables:
        combined_path = extractor.save_results(tables, args.output_dir)
        print(f"Extraction complete. Combined output saved to: {combined_path}")
    else:
        print("No tables found or extraction failed.")

if __name__ == "__main__":
    main()