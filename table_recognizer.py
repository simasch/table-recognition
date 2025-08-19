#!/usr/bin/env python3

import cv2
import numpy as np
import pytesseract
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


class TableRecognizer:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    
    def find_table_region(self, binary: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find the main table region in the image, excluding window borders"""
        h, w = binary.shape
        
        # Find horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 5))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines to find table structure
        table_structure = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find rectangular contours that look like tables
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too close to image edges (likely window border)
            if x < 10 and y < 10 and x + w > binary.shape[1] - 10 and y + h > binary.shape[0] - 10:
                continue
            
            # Filter based on size and shape
            if area > 3000 and w > 100 and h > 50:
                # Check aspect ratio (tables are usually wider than tall)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 10:
                    candidates.append((x, y, w, h, area))
        
        # Choose the best candidate (not necessarily the largest)
        if candidates:
            # Sort by distance from center and area
            center_x, center_y = w // 2, h // 2
            candidates.sort(key=lambda c: abs(c[0] + c[2]//2 - center_x) + abs(c[1] + c[3]//2 - center_y))
            
            # Return the most centered large contour
            for x, y, w, h, area in candidates:
                if area > max(3000, binary.shape[0] * binary.shape[1] * 0.05):
                    return (x, y, w, h)
            
            # If no good centered one, return the largest
            candidates.sort(key=lambda c: c[4], reverse=True)
            return candidates[0][:4]
        
        return None
    
    def detect_table_lines(self, binary: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[List, List]:
        h, w = binary.shape
        
        # Don't crop if no good region found - work with full image
        x, y = 0, 0
        
        # Detect lines with morphology - use smaller kernel for better line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 8))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        # Find line positions with lower threshold for better detection
        h_lines = []
        for row in range(h):
            if np.sum(horizontal_lines[row, :]) > w * 255 * 0.2:
                h_lines.append(row)
        
        v_lines = []
        for col in range(w):
            if np.sum(vertical_lines[:, col]) > h * 255 * 0.2:
                v_lines.append(col)
        
        # Merge close lines
        h_lines = self.merge_close_lines(h_lines, threshold=8)
        v_lines = self.merge_close_lines(v_lines, threshold=8)
        
        # Filter lines that are likely table boundaries
        # Remove lines too close to edges if we have interior lines
        if len(h_lines) > 6:
            h_lines = [hline for hline in h_lines if 20 < hline < h - 20]
        if len(v_lines) > 5:
            v_lines = [vline for vline in v_lines if 20 < vline < w - 20]
        
        return h_lines, v_lines
    
    def merge_close_lines(self, lines: List[int], threshold: int = 10) -> List[int]:
        if not lines:
            return []
        
        lines = sorted(lines)
        merged = [lines[0]]
        for line in lines[1:]:
            if line - merged[-1] > threshold:
                merged.append(line)
        return merged
    
    def extract_cells_from_grid(self, image: np.ndarray, h_lines: List[int], v_lines: List[int]) -> List[List[str]]:
        if len(h_lines) < 2 or len(v_lines) < 2:
            return []
        
        rows = []
        for i in range(len(h_lines) - 1):
            row = []
            for j in range(len(v_lines) - 1):
                y1, y2 = h_lines[i], h_lines[i + 1]
                x1, x2 = v_lines[j], v_lines[j + 1]
                
                # Add small margin
                margin = 3
                y1 = max(0, y1 + margin)
                y2 = min(image.shape[0], y2 - margin)
                x1 = max(0, x1 + margin)
                x2 = min(image.shape[1], x2 - margin)
                
                if y2 > y1 and x2 > x1:
                    cell_img = image[y1:y2, x1:x2]
                    text = self.extract_text_from_cell(cell_img)
                    row.append(text)
                else:
                    row.append("")
            
            # Only keep rows with content
            if any(cell.strip() for cell in row):
                rows.append(row)
        
        return rows
    
    def extract_text_from_cell(self, cell_image: np.ndarray) -> str:
        try:
            if len(cell_image.shape) == 3:
                gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cell_image
            
            # Resize if too small
            if gray.shape[0] < 20 or gray.shape[1] < 20:
                scale = max(30 / gray.shape[0], 30 / gray.shape[1])
                new_h = int(gray.shape[0] * scale)
                new_w = int(gray.shape[1] * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Threshold
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Check if we need to invert (text should be white on black for tesseract)
            white_pixels = np.sum(binary == 255)
            total_pixels = binary.shape[0] * binary.shape[1]
            if white_pixels > total_pixels * 0.5:
                binary = cv2.bitwise_not(binary)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Add padding
            padding = 15
            padded = cv2.copyMakeBorder(
                binary, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            
            # Try different OCR modes
            configs = [
                r'--oem 3 --psm 7',  # Single text line
                r'--oem 3 --psm 8',  # Single word
                r'--oem 3 --psm 6',  # Uniform block
            ]
            
            best_text = ""
            best_conf = 0
            
            for config in configs:
                try:
                    data = pytesseract.image_to_data(padded, config=config, output_type=pytesseract.Output.DICT)
                    for i, conf in enumerate(data['conf']):
                        if int(conf) > best_conf and data['text'][i].strip():
                            best_text = data['text'][i]
                            best_conf = int(conf)
                except:
                    pass
            
            if not best_text:
                # Fallback to simple OCR
                text = pytesseract.image_to_string(padded, config=configs[0]).strip()
            else:
                text = best_text
            
            # Clean text
            text = ' '.join(text.split())
            
            # Remove artifacts
            import re
            text = re.sub(r'[|\[\]{}\\/_«»]', '', text)
            text = text.strip()
            
            return text
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting text: {e}")
            return ""
    
    def recognize_table(self, image_path: str) -> List[List[str]]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        binary = self.preprocess_image(image)
        
        # Try to find table region first
        region = self.find_table_region(binary)
        
        if self.debug_mode and region:
            print(f"Found table region at: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
        
        # Detect lines
        h_lines, v_lines = self.detect_table_lines(binary, region)
        
        if self.debug_mode:
            print(f"Detected {len(h_lines)} horizontal lines and {len(v_lines)} vertical lines")
            
            if self.debug_mode and len(h_lines) > 0 and len(v_lines) > 0:
                # Save debug visualization
                vis_image = image.copy()
                for h in h_lines:
                    cv2.line(vis_image, (0, h), (image.shape[1], h), (0, 255, 0), 2)
                for v in v_lines:
                    cv2.line(vis_image, (v, 0), (v, image.shape[0]), (255, 0, 0), 2)
                cv2.imwrite("debug_lines.png", vis_image)
                print("Debug visualization saved to debug_lines.png")
        
        # Extract cells
        table_data = self.extract_cells_from_grid(image, h_lines, v_lines)
        
        # Post-process to clean up data
        table_data = self.clean_table_data(table_data)
        
        return table_data
    
    def clean_table_data(self, table_data: List[List[str]]) -> List[List[str]]:
        """Remove empty rows/columns and clean up the data"""
        if not table_data:
            return table_data
        
        # Remove empty rows
        cleaned = []
        for row in table_data:
            if any(cell.strip() for cell in row):
                cleaned.append(row)
        
        # Remove empty columns if all cells in column are empty
        if cleaned:
            num_cols = len(cleaned[0])
            cols_to_keep = []
            for col_idx in range(num_cols):
                if any(row[col_idx].strip() for row in cleaned if col_idx < len(row)):
                    cols_to_keep.append(col_idx)
            
            if cols_to_keep:
                cleaned = [[row[i] if i < len(row) else "" for i in cols_to_keep] for row in cleaned]
        
        return cleaned


def export_to_csv(table_data: List[List[str]], output_path: str):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(table_data)


def main():
    parser = argparse.ArgumentParser(description='Recognize tables in images and export to CSV')
    parser.add_argument('image', help='Path to the image containing a table')
    parser.add_argument('-o', '--output', default='output.csv', help='Output CSV file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image file '{args.image}' not found")
        return 1
    
    recognizer = TableRecognizer(debug_mode=args.debug)
    
    try:
        print(f"Processing image: {args.image}")
        table_data = recognizer.recognize_table(args.image)
        
        if not table_data:
            print("No table detected in the image")
            return 0
        
        print(f"Detected table with {len(table_data)} rows and {len(table_data[0]) if table_data else 0} columns")
        
        export_to_csv(table_data, args.output)
        print(f"CSV saved to {args.output}")
        
        print("\nTable preview:")
        for row in table_data[:5]:
            print(" | ".join(row))
        if len(table_data) > 5:
            print("...")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())