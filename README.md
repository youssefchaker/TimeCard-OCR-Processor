# Timecard OCR App

This application extracts structured data from timecard PDF files using Optical Character Recognition (OCR) and saves the output as CSV files.

## Features

*   **PDF to CSV Conversion**: Processes PDF timecards and converts them into structured CSV data.
*   **OCR Engine**: Utilizes the Tesseract OCR engine to extract text from images.
*   **Image Preprocessing**: Enhances images for better OCR accuracy, including grayscale conversion, denoising, and thresholding.
*   **Automatic Orientation Detection**: Automatically detects and corrects the orientation of the document for accurate text extraction.
*   **Table Detection**: Intelligently detects tables within the document to extract structured data.
*   **Command-Line Interface**: Provides a CLI to process a single file or a directory of files.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd timecard_ocr_app
    ```

2.  **Install Tesseract:**
    This application requires Tesseract OCR engine to be installed on your system. You can download it from [here](https://github.com/tesseract-ocr/tesseract).
    Make sure to add the Tesseract installation directory to your system's PATH or set the `TESSERACT_PATH` in a `.env` file.

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application can be run from the command line.

### Processing a Single File

To process a single PDF file, use the `--file` or `-f` argument:
```bash
python main.py --file /path/to/your/timecard.pdf
```

### Processing a Directory

To process all PDF files in a directory, use the `--dir` or `-d` argument. If no directory is specified, it will use the default `data/input` directory.
```bash
python main.py --dir /path/to/your/pdf_directory
```

### Command-line Arguments

*   `--file`, `-f`: Path to a single PDF file to process.
*   `--dir`, `-d`: Path to a directory containing PDF files to process.
*   `--confidence`, `-c`: Sets the OCR confidence threshold (a float between 0.0 and 1.0). Default is 0.8.
*   `--no-orient`: Disables the automatic orientation detection.

## Configuration

You can configure the application using a `.env` file in the root directory.

*   `TESSERACT_PATH`: The path to the Tesseract executable.
*   `INPUT_DIR`: The default input directory for PDF files (default: `data/input`).
*   `OUTPUT_DIR`: The directory where the output CSV files will be saved (default: `data/output`).

Example `.env` file:
```
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Dependencies

The application uses the following Python libraries:

*   `pytesseract`
*   `PyMuPDF`
*   `opencv-python`
*   `pillow`
*   `pandas`
*   `numpy`
*   `python-dotenv`
*   `scikit-learn`
*   `fastapi`
*   `uvicorn`
*   `python-multipart`

These can be installed by running `pip install -r requirements.txt`.
