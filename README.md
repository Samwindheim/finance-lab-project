# Financial Data Extractor from PDFs

## Overview

This project is a pipeline designed to extract information about underwriters and guarantors from financial PDF documents. It leverages a two-stage process involving precise, context-aware table extraction using `PyMuPDF` and advanced data structuring using OpenAI's `gpt-5-mini` model.

The primary goal is to convert unstructured tables within PDFs into a clean, validated, and structured JSON format.

## Features

- **High-Precision Table Extraction**: Scans PDFs to find relevant pages based on financial keywords and extracts tables with high fidelity.
- **AI-Powered Data Structuring**: Uses `gpt-5-mini` to interpret Markdown tables and convert them into a structured JSON format based on a detailed set of rules.
- **Data Validation**: Employs `Pydantic` models to ensure the final output is correctly typed and conforms to a predefined schema.
- **Detailed Metadata**: Includes metadata in the output, such as the source file, page numbers, and a confidence score for the extraction.
- **Efficiency Reporting**: A `--verbose` flag provides an analysis of processing times for both the local table scan and the AI API call.

## How It Works

The application operates using a two-stage Retrieval-Augmented Generation (RAG) pipeline:

1.  **Stage 1: Retrieval (Local Processing)**
    -   The PDF is scanned for keywords (e.g., "underwriter", "guarantor") to identify candidate pages.
    -   A context window of pages (one before, one after) is created to ensure no relevant data is missed.
    -   `PyMuPDF` (`fitz`) extracts both the tables and the full text from these pages.
    -   This combined data (text + tables) is formatted and prepared for the AI.

2.  **Stage 2: Generation (AI Processing)**
    -   The contextual data is sent to the `gpt-5-mini` API with a detailed prompt.
    -   The prompt instructs the AI to use the page text to understand headers, units, and other context, then apply strict data cleaning rules.
    -   The AI returns a structured JSON object.

Finally, the application sorts the extracted investors by their `investor_level`, validates the data against Pydantic models, and prints the final, clean output.

## Setup & Installation

**1. Clone the Repository**
```bash
git clone <repository-url>
cd finance-lab-project
```

**2. Create a Virtual Environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set Up Environment Variables**
The application requires an OpenAI API key. Create a `.env` file:

Then, add your API key to it:
```
OPENAI_API_KEY="your-openai-api-key-here"
```

## Usage

Run the extraction process using the provided shell script, passing the path to a PDF file.

```bash
./extract_underwriters.sh path/to/your/document.pdf
```

### Command-Line Arguments
-   `pdf_path` (Required): The file path to the PDF document.
-   `--verbose`: (Optional) Prints an efficiency analysis report.
-   `--debug`: (Optional) Prints the raw contextual data sent to the AI.

## Testing for Accuracy

The project includes a testing framework to verify the accuracy of the extraction against "ground truth" data.

**1. Create a Ground Truth File**
For a PDF you want to test (e.g., `mydoc.pdf`), manually create a correct JSON file and save it as `tests/ground_truth/mydoc.json`. This file should contain the exact `investors` array you expect the script to produce.

**2. Run the Test**
Execute the test script, passing the path to the PDF:
```bash
./test.sh path/to/your/document.pdf
```
The script will compare the output from your program against the ground truth data and report any differences.

## Output Structure

The script outputs a JSON object to standard output. The structure is defined by the Pydantic models in `src/models.py`.

```json
{
  "meta": {
    "source": "example.pdf",
    "extracted_at": "2025-09-13T12:00:00.000000Z",
    "source_page": "2,3",
    "confidence": "high"
  },
  "investors": [
    {
      "name": "Investor Name One",
      "commitment": {
        "amount": 10000,
        "percent": 5.1
      },
      "investor_level": 0
    },
    {
      "name": "Investor Name Two",
      "commitment": {
        "amount": 20000,
        "percent": 2.5
      },
      "investor_level": 1
    }
  ]
}
```

-   `meta`: Contains metadata about the extraction process.
-   `investors`: A list of investor objects, each containing their name, commitment details, and investor level (0 for underwriters, 1+ for guarantors).

