# Financial Data Extractor from PDFs

## Overview

This project is a sophisticated pipeline designed to extract information about underwriters and guarantors from financial PDF documents. It leverages a two-stage process involving precise table extraction using `PyMuPDF` and advanced data structuring using OpenAI's `gpt-4o` model.

The primary goal is to convert unstructured tables within PDFs into a clean, validated, and structured JSON format.

## Features

- **High-Precision Table Extraction**: Scans PDFs to find relevant pages based on financial keywords and extracts tables with high fidelity.
- **AI-Powered Data Structuring**: Uses `gpt-4o` to interpret Markdown tables and convert them into a structured JSON format based on a detailed set of rules.
- **Data Validation**: Employs `Pydantic` models to ensure the final output is correctly typed and conforms to a predefined schema.
- **Detailed Metadata**: Includes metadata in the output, such as the source file, page numbers, and a confidence score for the extraction.
- **Efficiency Reporting**: A `--verbose` flag provides an analysis of processing times for both the local table scan and the AI API call.

## How It Works

The application operates using a two-stage Retrieval-Augmented Generation (RAG) pipeline:

1.  **Stage 1: Retrieval (Local Processing)**
    -   The PDF is scanned for a set of keywords (e.g., "underwriter", "guarantor", "teckningsåtagare") to identify candidate pages.
    -   The tool creates a context window of pages (one page before and after the candidate page) to ensure no relevant data is missed.
    -   `PyMuPDF` (`fitz`) is used to find and extract all tables from these pages.
    -   The extracted tables are converted into a clean Markdown format.

2.  **Stage 2: Generation (AI Processing)**
    -   The Markdown tables are sent to the `gpt-4o` API with a detailed prompt.
    -   The prompt includes definitions, investor level mappings, and strict data cleaning rules (e.g., converting currency strings to integers and percentages to floats).
    -   The AI processes the tables and returns a structured JSON object.

Finally, the application validates this JSON response against Pydantic models and prints the final, clean output to the console.

## Setup & Installation

Follow these steps to set up the project environment.

**1. Clone the Repository**

```bash
git clone <repository-url>
cd finance-lab-project
```

**2. Create a Virtual Environment**

It is recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**

Install the required Python packages using `requirements.txt`.

```bash
pip install -r requirements.txt
```

**4. Set Up Environment Variables**

The application requires an OpenAI API key. Create a `.env` file in the root of the project:

```
touch .env
```

Then, add your API key to the `.env` file:

```
OPENAI_API_KEY="your-openai-api-key-here"
```

## Usage

You can run the extraction process using the provided shell script or by directly invoking the Python script. The script takes the path to a PDF file as an argument.

**Using the Shell Script (Recommended)**

The `extract_underwriters.sh` script handles the activation of the virtual environment automatically.

```bash
./extract_underwriters.sh path/to/your/document.pdf
```

### Command-Line Arguments

-   `pdf_path` (Required): The file path to the PDF document.
-   `--verbose`: (Optional) Prints a detailed efficiency analysis report, including document page count and processing durations.
-   `--debug`: (Optional) Prints the raw Markdown tables extracted by `PyMuPDF` before they are sent to the AI for processing.

### Example

```bash
./extract_underwriters.sh pdfs/ADVT_2025_08_14_Memorandum.pdf --verbose
```

## Output Structure

The script outputs a JSON object to standard output. The structure is defined by the Pydantic models in `src/models.py`.

```json
{
  "meta": {
    "source": "ADVT_2025_08_14_Memorandum.pdf",
    "extracted_at": "2025-09-13T12:00:00.000000Z",
    "source_page": "11,12",
    "confidence": "high"
  },
  "investors": [
    {
      "name": "Investor Name One",
      "commitment": {
        "amount": 500000,
        "percent": 5.1
      },
      "investor_level": 0
    },
    {
      "name": "Investor Name Two",
      "commitment": {
        "amount": 250000,
        "percent": 2.5
      },
      "investor_level": 1
    }
  ]
}
```

-   `meta`: Contains metadata about the extraction process.
-   `investors`: A list of investor objects, each containing their name, commitment details, and investor level (0 for underwriters, 1+ for guarantors).

