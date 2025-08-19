#!/usr/bin/env python3

import cv2
import numpy as np
import pytesseract
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Cell:
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float


@dataclass
class Table:
    cells: List[Cell]
    rows: int
    cols: int
    x: int
    y: int
    width: int
    height: int


class TableRecognizer:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_lines(self, binary_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kernel_length = np.array(binary_image).shape[1] // 80
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        return horizontal_lines, vertical_lines
    
    def find_table_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        binary = self.preprocess_image(image)
        
        horizontal_lines, vertical_lines = self.detect_lines(binary)
        
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        min_table_area = 5000
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_table_area:
                x, y, w, h = cv2.boundingRect(contour)
                tables.append((x, y, w, h))
        
        return tables
    
    def extract_cells(self, image: np.ndarray, table_region: Tuple[int, int, int, int]) -> List[Cell]:
        x, y, w, h = table_region
        table_img = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        horizontal_lines, vertical_lines = self.detect_lines(cv2.bitwise_not(binary))
        
        combined = cv2.add(horizontal_lines, vertical_lines)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        min_cell_area = 100
        
        cell_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_cell_area:
                cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(contour)
                if cell_w > 10 and cell_h > 10:
                    cell_boxes.append((cell_x, cell_y, cell_w, cell_h))
        
        cell_boxes = sorted(cell_boxes, key=lambda b: (b[1], b[0]))
        
        if not cell_boxes:
            return cells
        
        rows = []
        current_row = []
        current_y = cell_boxes[0][1]
        y_threshold = 10
        
        for box in cell_boxes:
            if abs(box[1] - current_y) > y_threshold:
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
                current_y = box[1]
            else:
                current_row.append(box)
        
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))
        
        for row_idx, row in enumerate(rows):
            for col_idx, (cell_x, cell_y, cell_w, cell_h) in enumerate(row):
                cell_img = table_img[cell_y:cell_y+cell_h, cell_x:cell_x+cell_w]
                
                text = self.extract_text_from_cell(cell_img)
                
                cell = Cell(
                    row=row_idx,
                    col=col_idx,
                    x=x + cell_x,
                    y=y + cell_y,
                    width=cell_w,
                    height=cell_h,
                    text=text,
                    confidence=0.9
                )
                cells.append(cell)
        
        return cells
    
    def extract_text_from_cell(self, cell_image: np.ndarray) -> str:
        try:
            padding = 10
            padded = cv2.copyMakeBorder(
                cell_image, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            
            text = pytesseract.image_to_string(padded, config='--psm 6').strip()
            
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting text: {e}")
            return ""
    
    def recognize_tables(self, image_path: str) -> List[Table]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        table_regions = self.find_table_regions(image)
        
        tables = []
        for region in table_regions:
            cells = self.extract_cells(image, region)
            
            if cells:
                max_row = max(cell.row for cell in cells)
                max_col = max(cell.col for cell in cells)
                
                table = Table(
                    cells=cells,
                    rows=max_row + 1,
                    cols=max_col + 1,
                    x=region[0],
                    y=region[1],
                    width=region[2],
                    height=region[3]
                )
                tables.append(table)
        
        if self.debug_mode and tables:
            self.visualize_tables(image, tables)
        
        return tables
    
    def visualize_tables(self, image: np.ndarray, tables: List[Table]):
        vis_image = image.copy()
        
        for table in tables:
            cv2.rectangle(vis_image, (table.x, table.y), 
                         (table.x + table.width, table.y + table.height), 
                         (0, 255, 0), 2)
            
            for cell in table.cells:
                cv2.rectangle(vis_image, (cell.x, cell.y),
                             (cell.x + cell.width, cell.y + cell.height),
                             (255, 0, 0), 1)
        
        cv2.imwrite("debug_output.png", vis_image)
        print("Debug visualization saved to debug_output.png")


def export_to_json(tables: List[Table], output_path: str):
    data = []
    for idx, table in enumerate(tables):
        table_data = {
            "table_id": idx,
            "dimensions": {"rows": table.rows, "cols": table.cols},
            "position": {"x": table.x, "y": table.y, "width": table.width, "height": table.height},
            "cells": [asdict(cell) for cell in table.cells]
        }
        data.append(table_data)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def export_to_csv(tables: List[Table], output_path: str):
    for idx, table in enumerate(tables):
        grid = [['' for _ in range(table.cols)] for _ in range(table.rows)]
        
        for cell in table.cells:
            if cell.row < table.rows and cell.col < table.cols:
                grid[cell.row][cell.col] = cell.text
        
        csv_path = output_path.replace('.csv', f'_table_{idx}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(grid)
        
        print(f"Table {idx} saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Recognize tables in UI screenshots')
    parser.add_argument('image', help='Path to the screenshot image')
    parser.add_argument('-o', '--output', default='output', help='Output file name (without extension)')
    parser.add_argument('-f', '--format', choices=['json', 'csv', 'both'], default='json',
                       help='Output format')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualization')
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image file '{args.image}' not found")
        return 1
    
    recognizer = TableRecognizer(debug_mode=args.debug)
    
    try:
        print(f"Processing image: {args.image}")
        tables = recognizer.recognize_tables(args.image)
        
        if not tables:
            print("No tables detected in the image")
            return 0
        
        print(f"Detected {len(tables)} table(s)")
        
        if args.format in ['json', 'both']:
            json_path = f"{args.output}.json"
            export_to_json(tables, json_path)
            print(f"JSON output saved to {json_path}")
        
        if args.format in ['csv', 'both']:
            export_to_csv(tables, args.output + '.csv')
        
        for idx, table in enumerate(tables):
            print(f"\nTable {idx}:")
            print(f"  Dimensions: {table.rows} rows x {table.cols} columns")
            print(f"  Total cells: {len(table.cells)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())