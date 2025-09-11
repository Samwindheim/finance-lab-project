"""
This script serves as the main entry point for the application.

It handles command-line argument parsing, orchestrates the PDF parsing and
data extraction process, and prints the final structured JSON output to the console.
"""

import argparse
import os
from dotenv import load_dotenv

from src.data_extractor import extract_investor_data
from src.models import ExtractionResult, Meta


def main():
    """
    Main function to run the data extraction process.
    """
    load_dotenv()  # Load environment variables from .env file

    parser = argparse.ArgumentParser(description="Extract underwriter and guarantor data from a PDF document.")
    parser.add_argument("pdf_path", type=str, help="The file path or URL to the PDF document.")
    parser.add_argument("--verbose", action="store_true", help="Print a detailed efficiency analysis report.")
    parser.add_argument("--debug", action="store_true", help="Print the raw text extracted by fitz before AI processing.")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: The file '{args.pdf_path}' was not found.")
        return

    # Extract the raw data using the core logic
    extracted_data = extract_investor_data(args.pdf_path, verbose=args.verbose, debug=args.debug)

    # Create the metadata object
    meta_info = Meta(
        source=os.path.basename(args.pdf_path),
        source_page=extracted_data.get("source_page"),
        confidence=extracted_data.get("confidence"),
    )

    # Combine meta and investor data and validate with Pydantic
    full_result = ExtractionResult(
        meta=meta_info,
        investors=extracted_data.get("investors", [])
    )

    # Print the final, validated JSON output
    print(full_result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
