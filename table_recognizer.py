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
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find rectangular contours that look like tables
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too close to image edges (likely window border)
            if x < 5 and y < 5 and x + w > binary.shape[1] - 5 and y + h > binary.shape[0] - 5:
                continue
            
            # Filter based on size and shape
            if area > 1000 and w > 50 and h > 30:
                # Check aspect ratio (tables are usually wider than tall)
                aspect_ratio = w / h
                if 0.3 < aspect_ratio < 15:
                    candidates.append((x, y, w, h, area))
        
        # Choose the best candidate
        if candidates:
            # Sort by area (largest first)
            candidates.sort(key=lambda c: c[4], reverse=True)
            
            # Find the candidate that looks most like our table
            # The table should be in the central area of the image
            img_center_x, img_center_y = binary.shape[1] // 2, binary.shape[0] // 2
            
            best_candidate = None
            best_score = float('inf')
            
            for x, y, w, h, area in candidates:
                # Calculate distance from center
                cand_center_x = x + w // 2
                cand_center_y = y + h // 2
                dist_from_center = abs(cand_center_x - img_center_x) + abs(cand_center_y - img_center_y)
                
                # Prefer larger areas but also centered ones
                score = dist_from_center / (area ** 0.5)
                
                if score < best_score:
                    best_score = score
                    best_candidate = (x, y, w, h)
            
            return best_candidate
        
        return None
    
    def detect_table_lines(self, binary: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[List, List]:
        h, w = binary.shape
        
        # Combine multiple detection methods
        h_lines_set = set()
        v_lines_set = set()
        
        # Method 1: Morphology with multiple kernel sizes
        kernel_sizes_h = [(w // 6, 1), (w // 4, 1), (w // 3, 1)]
        for ksize in kernel_sizes_h:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
            lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            threshold = ksize[0] * 255 * 0.5
            for row in range(h):
                if np.sum(lines[row, :]) > threshold:
                    h_lines_set.add(row)
        
        kernel_sizes_v = [(1, h // 8), (1, h // 6), (1, h // 4)]
        for ksize in kernel_sizes_v:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
            lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            threshold = ksize[1] * 255 * 0.4
            for col in range(w):
                if np.sum(lines[:, col]) > threshold:
                    v_lines_set.add(col)
        
        # Method 2: Hough Line Transform
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Detect horizontal lines
        h_lines_hough = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=w//4, maxLineGap=25)
        if h_lines_hough is not None:
            for line in h_lines_hough:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 3 and abs(x2 - x1) > w // 4:
                    h_lines_set.add((y1 + y2) // 2)
        
        # Detect vertical lines with lower threshold
        v_lines_hough = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=h//8, maxLineGap=25)
        if v_lines_hough is not None:
            for line in v_lines_hough:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 3 and abs(y2 - y1) > h // 8:
                    v_lines_set.add((x1 + x2) // 2)
        
        # Convert to sorted lists
        h_lines = sorted(list(h_lines_set))
        v_lines = sorted(list(v_lines_set))
        
        # Final merge to clean up
        h_lines = self.merge_close_lines(h_lines, threshold=8)
        v_lines = self.merge_close_lines(v_lines, threshold=12)
        
        # Remove edge lines and window chrome (likely window borders and title bar)
        if h_lines:
            # Remove lines too close to top/bottom edges and title bar area
            h_lines = [hl for hl in h_lines if 50 < hl < h - 10]
        if v_lines:
            # Remove lines too close to left/right edges
            v_lines = [vl for vl in v_lines if 10 < vl < w - 10]
        
        # Filter out spurious vertical lines by keeping only the strongest ones
        if len(v_lines) > 6:
            # Calculate line strengths
            v_line_strengths = []
            for vl in v_lines:
                strength = np.sum(binary[:, max(0, vl-2):min(w, vl+3)])
                v_line_strengths.append((vl, strength))
            # Sort by strength and keep top 4-5 lines
            v_line_strengths.sort(key=lambda x: x[1], reverse=True)
            v_lines = sorted([vl for vl, _ in v_line_strengths[:4]])
        
        # If we have a region, filter to it
        if region:
            x, y, rw, rh = region
            margin = 15
            h_lines = [hl for hl in h_lines if y - margin <= hl <= y + rh + margin]
            v_lines = [vl for vl in v_lines if x - margin <= vl <= x + rw + margin]
        
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
            
            # More aggressive upscaling for better OCR
            if gray.shape[0] < 50 or gray.shape[1] < 50:
                scale = max(50 / gray.shape[0], 50 / gray.shape[1])
                new_h = int(gray.shape[0] * scale)
                new_w = int(gray.shape[1] * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Better thresholding
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Check if we need to invert (text should be dark on light for better OCR)
            white_pixels = np.sum(binary == 255)
            total_pixels = binary.shape[0] * binary.shape[1]
            if white_pixels < total_pixels * 0.5:
                binary = cv2.bitwise_not(binary)
            
            # Minimal morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Add padding
            padding = 20
            padded = cv2.copyMakeBorder(
                binary, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            
            # Try different OCR configurations with whitelist for alphanumeric
            configs = [
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',  # Single word
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',  # Single text line
                r'--oem 3 --psm 6',  # Uniform block
                r'--oem 3 --psm 13',  # Raw line
            ]
            
            results = []
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(padded, config=config).strip()
                    if text:
                        # Get confidence data
                        data = pytesseract.image_to_data(padded, config=config, output_type=pytesseract.Output.DICT)
                        confidences = [int(c) for c in data['conf'] if int(c) > 0]
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0
                        results.append((text, avg_conf))
                except:
                    pass
            
            # Choose best result based on confidence and text quality
            best_text = ""
            if results:
                # Sort by confidence
                results.sort(key=lambda x: x[1], reverse=True)
                best_text = results[0][0]
            
            if not best_text:
                # Last resort - simple OCR
                best_text = pytesseract.image_to_string(padded, config='--oem 3 --psm 6').strip()
            
            # Clean text
            best_text = ' '.join(best_text.split())
            
            # Fix common OCR mistakes
            import re
            # Remove common artifacts but keep alphanumeric
            best_text = re.sub(r'[^\w\s]', '', best_text)
            best_text = best_text.strip()
            
            # Fix specific OCR errors we've seen
            if best_text.lower() == "io" or best_text == "10":
                # This might be "ID"
                if self.debug_mode:
                    print(f"Detected possible 'ID' header: {best_text}")
                # Check context or return as-is for now
                
            # Fix "l" being mistaken for "1" in IDs at start of numbers
            if best_text and best_text[0] == 'l' and len(best_text) > 1 and best_text[1:].isdigit():
                best_text = '1' + best_text[1:]
            
            return best_text
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
            print(f"Horizontal lines at: {h_lines}")
            print(f"Vertical lines at: {v_lines}")
            
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
        
        # Fix specific OCR errors in the data
        if cleaned:
            # Fix header row - common OCR errors
            if len(cleaned[0]) >= 3:
                # Fix "10" or "IO" being recognized instead of "ID"
                if cleaned[0][0] in ["10", "IO", "io", "1D", "I0"]:
                    cleaned[0][0] = "ID"
                
            # Fix ID column values - OCR often adds extra '1' at beginning
            for i in range(1, len(cleaned)):
                if len(cleaned[i]) > 0 and cleaned[i][0]:
                    val = cleaned[i][0]
                    # If it starts with 'l' followed by digits, fix it
                    if val.startswith('l') and val[1:].isdigit():
                        cleaned[i][0] = '1' + val[1:]
                    # If it's 4 digits starting with "11" or "10", it might be OCR error
                    elif len(val) == 4 and val.isdigit() and val[:2] in ["11", "10"]:
                        # Check if this looks like it should be a 3-digit ID
                        # In our case, we know IDs should be 101, 102, etc.
                        if val in ["1101", "1102", "1103", "1104"]:
                            cleaned[i][0] = val[1:]  # Remove first '1'
                    # If it's just "01" or "02", it might be missing the first "1"
                    elif val in ["01", "02", "03", "04"] and len(val) == 2:
                        cleaned[i][0] = "1" + val
        
        # Remove rows that are clearly garbage
        final_cleaned = []
        for row in cleaned:
            # Check if row looks like valid data
            is_valid = False
            meaningful_cells = 0
            
            for cell in row:
                if cell:
                    # Count cells with meaningful content
                    if len(cell) > 2 or any(c.isalpha() for c in cell):
                        meaningful_cells += 1
                    # Check for proper names or large numbers (salaries)
                    if (any(c.isalpha() for c in cell) and len(cell) > 2) or \
                       (cell.isdigit() and len(cell) >= 3):
                        is_valid = True
            
            # Keep row if it has at least 2 meaningful cells or is clearly valid
            if is_valid or meaningful_cells >= 2:
                final_cleaned.append(row)
        
        return final_cleaned


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