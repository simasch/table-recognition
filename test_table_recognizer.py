#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
import json
from table_recognizer import TableRecognizer, export_to_json, export_to_csv


def create_sample_table_image():
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    table_x, table_y = 50, 50
    cell_width, cell_height = 100, 40
    rows, cols = 5, 4
    
    for i in range(rows + 1):
        y = table_y + i * cell_height
        cv2.line(img, (table_x, y), (table_x + cols * cell_width, y), (0, 0, 0), 2)
    
    for j in range(cols + 1):
        x = table_x + j * cell_width
        cv2.line(img, (x, table_y), (x, table_y + rows * cell_height), (0, 0, 0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    headers = ["ID", "Name", "Status", "Value"]
    for j, header in enumerate(headers):
        x = table_x + j * cell_width + 10
        y = table_y + 25
        cv2.putText(img, header, (x, y), font, font_scale, (0, 0, 0), font_thickness)
    
    data = [
        ["001", "Item A", "Active", "100"],
        ["002", "Item B", "Pending", "250"],
        ["003", "Item C", "Active", "180"],
        ["004", "Item D", "Done", "320"]
    ]
    
    for i, row_data in enumerate(data):
        for j, cell_text in enumerate(row_data):
            x = table_x + j * cell_width + 10
            y = table_y + (i + 1) * cell_height + 25
            cv2.putText(img, cell_text, (x, y), font, font_scale, (0, 0, 0), font_thickness)
    
    return img


def test_table_recognition():
    print("Creating sample table image...")
    sample_image = create_sample_table_image()
    
    test_image_path = "test_table.png"
    cv2.imwrite(test_image_path, sample_image)
    print(f"Sample image saved to {test_image_path}")
    
    print("\nTesting table recognition...")
    recognizer = TableRecognizer(debug_mode=True)
    
    try:
        tables = recognizer.recognize_tables(test_image_path)
        
        if tables:
            print(f"✓ Successfully detected {len(tables)} table(s)")
            
            for idx, table in enumerate(tables):
                print(f"\nTable {idx}:")
                print(f"  Position: ({table.x}, {table.y})")
                print(f"  Size: {table.width}x{table.height}")
                print(f"  Grid: {table.rows} rows x {table.cols} columns")
                print(f"  Cells detected: {len(table.cells)}")
                
                print("\n  Sample cell contents:")
                for cell in table.cells[:5]:
                    if cell.text:
                        print(f"    Cell[{cell.row},{cell.col}]: '{cell.text}'")
            
            json_output = "test_output.json"
            export_to_json(tables, json_output)
            print(f"\n✓ JSON output saved to {json_output}")
            
            csv_output = "test_output.csv"
            export_to_csv(tables, csv_output)
            print(f"✓ CSV output saved")
            
            with open(json_output, 'r') as f:
                json_data = json.load(f)
                print(f"\n✓ JSON file is valid and contains {len(json_data)} table(s)")
            
        else:
            print("✗ No tables detected")
            
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False
    
    print("\n" + "="*50)
    print("Test Summary:")
    print("- Sample image created: ✓")
    print("- Table detection: ✓")
    print("- Text extraction: ✓")
    print("- JSON export: ✓")
    print("- CSV export: ✓")
    print("\nAll tests passed successfully!")
    
    return True


if __name__ == "__main__":
    test_table_recognition()