# UI Table Recognition Utility

A Python utility that recognizes and extracts tables from UI screenshots using computer vision and OCR.

## Features

- Detects tables in UI screenshots automatically
- Extracts text from table cells using OCR
- Preserves table structure (rows and columns)
- Exports results in JSON or CSV format
- Debug mode with visual output
- Handles various table styles

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Usage

Basic usage:
```bash
python table_recognizer.py screenshot.png
```

With options:
```bash
# Export as CSV
python table_recognizer.py screenshot.png -f csv -o results

# Export both JSON and CSV
python table_recognizer.py screenshot.png -f both -o results

# Enable debug mode with visualization
python table_recognizer.py screenshot.png --debug
```

## Command Line Arguments

- `image`: Path to the screenshot image (required)
- `-o, --output`: Output file name without extension (default: "output")
- `-f, --format`: Output format - json, csv, or both (default: "json")
- `--debug`: Enable debug mode with visualization

## Output Formats

### JSON Format
Contains structured data with:
- Table dimensions and position
- Cell-by-cell information with text and coordinates
- Confidence scores

### CSV Format
Each table is saved as a separate CSV file with the extracted text arranged in rows and columns.

## Example

```bash
# Process a screenshot and save as JSON
python table_recognizer.py ui_screenshot.png -o extracted_tables

# Process with debug visualization
python table_recognizer.py ui_screenshot.png --debug
```

## How It Works

1. **Image Preprocessing**: Converts to grayscale and applies noise reduction
2. **Line Detection**: Identifies horizontal and vertical lines using morphological operations
3. **Table Region Detection**: Finds rectangular regions that form tables
4. **Cell Extraction**: Segments tables into individual cells
5. **OCR Processing**: Extracts text from each cell using Tesseract
6. **Structure Preservation**: Maintains row/column relationships
7. **Export**: Formats results as JSON or CSV

## Requirements

- Python 3.7+
- OpenCV
- Tesseract OCR
- NumPy
- pytesseract