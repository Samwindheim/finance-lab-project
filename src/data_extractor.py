"""
This module contains the core logic for AI-powered data extraction.

It orchestrates the main two-stage RAG pipeline:
1.  **Retrieval**: Invokes the `pdf_parser` to find and extract relevant tables
    from the source PDF, which are returned as clean Markdown.
2.  **Generation**: Constructs a detailed prompt with specific mapping and cleaning
    rules, then sends the Markdown table(s) to the OpenAI API (gpt-4o) to be
    converted into a structured JSON format.

It also calculates a final confidence score based on the completeness of the
extracted data.
"""
import time
import os
import json
from openai import OpenAI
from typing import Dict, Any, List

from src.pdf_parser import find_and_extract_tables_as_markdown, get_page_count


SEARCH_KEYWORDS = [
    "underwriter", "guarantor", "teckningsåtagare", "garant",
    "commitment", "åtaganden", "garantiåtaganden", "teckningsåtaganden"
]


def get_openai_client() -> OpenAI:
    """
    Initializes and returns the OpenAI client.

    Expects the OPENAI_API_KEY environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)


def _calculate_confidence(investors: List[Dict[str, Any]]) -> str:
    """
    Calculates a confidence score based on the completeness of the extracted data.
    """
    if not investors:
        return "none"

    has_medium_confidence = False
    for investor in investors:
        commitment = investor.get("commitment", {})
        amount = commitment.get("amount")
        percent = commitment.get("percent")

        if amount is None and percent is None:
            return "low"  # Found an investor with no commitment data at all
        if amount is None or percent is None:
            has_medium_confidence = True

    return "medium" if has_medium_confidence else "high"


def create_extraction_prompt(markdown_tables: str) -> str:
    """
    Creates the prompt for the OpenAI API to convert a Markdown table to JSON.
    """
    return f"""
    You are an expert AI assistant specializing in financial data processing. Your task is to convert the following financial data into a structured JSON format.

    The input contains text and Markdown table(s) extracted from a PDF page. You must use the **Page Text** to understand the context, especially for column headers which might be missing from the Markdown table itself.

    **Definitions:**
    - **Underwriter (Swedish: "teckningsåtagare")**: A person or entity that has committed to underwriting an issue. They are not compensated. Their investor level is always 0.
    - **Guarantor (Swedish: "Garant")**: A person or entity that agrees to sign up for shares if the issue is not filled. They are compensated for this service. Their investor level starts at 1 for the first/lowest guarantee level, 2 for the next, and so on.
    - **Amount (Swedish: "Belopp")**: The amount in SEK (or other currency) that the investor has committed to underwrite or guarantee.
    - **Share of the rights issue percentage (Swedish: "Andel av Företrädesemissionen")**: The percentage of the issue that the investor has committed to underwrite or guarantee.

    **Investor Level Mapping:**
    - Commitments from a "Tecknings-förbindelser", "Teckningsåtaganden", or similar column are **investor_level 0**.
    - Commitments from a general "Garantiåtaganden", "Garantier", or similar column are **investor_level 1**.
    - Commitments from a "Botten-garantier" column are **investor_level 1**.
    - Commitments from a "Toppgarantier" column are **investor_level 2**.
    - If a single person has commitments in multiple columns (e.g., both underwriting and guaranteeing), create a separate investor object for each commitment.

    **Data Cleaning Rules (VERY IMPORTANT):**
    - **Use Page Text for Context**: The Markdown tables might be missing headers. Use the accompanying **Page Text** to find the correct column headers and understand the table's structure.
    - **For `amount`**:
        - Check the **Page Text** or table headers for currency units like "TSEK", "KSEK" (thousands), or "MSEK" (millions). If found, you MUST multiply the numerical amount by the correct factor (1,000 for TSEK/KSEK, 1,000,000 for MSEK).
        - This is also true for range of amounts. Make sure to multiply the range by the correct factor.
        - After accounting for units, remove all spaces, currency symbols (e.g., "SEK"), and text. 
        - If an amount is expressed as a range (e.g., "2 - 100"), it MUST be stored as a string "2 - 100".
        - You MUST represent the amount as a float if it has a decimal represented as a comma. (e.g., "123 456,10" should be represented as 123456.10)
        - The final value must be an integer, a float, or a string if it is a range. Do not process columns that represent totals of other columns.
    - **For `percent`**: Remove the percentage sign (`%`). Use a period (`.`) as the decimal separator. The final value must be a float.
    - **Ignore Summary Rows**: Do not extract data from rows that are clearly totals or summaries (e.g., starting with "Totalt", "Summa").

    **Instructions:**
    1.  Carefully read the **Page Text** to understand the context of the table(s).
    2.  Parse the Markdown table(s). Use the context to assign correct headers if they are missing.
    3.  For each row representing an investor, create one or more JSON objects for each commitment found.
    4.  Extract the `name`, `commitment` (`amount`, `percent`), and determine the `investor_level`.
    5.  Apply the **Data Cleaning Rules** meticulously to all numerical values.
    6.  The `source_pages` are indicated by the `--- Page X ---` markers. Your JSON response must include a `source_pages` key containing a list of these page numbers.

    **Example Output Format:**
    ```json
    {{
      "source_pages": [11],
      "investors": [
        {{
          "name": "Tuvedalen Ltd.",
          "commitment": {{
            "amount": 1652367,
            "percent": 9.4
          }},
          "investor_level": 0
        }}
      ]
    }}
    ```

    **Financial Data for Analysis:**
    ---
    {markdown_tables}
    ---
    """


def _extract_data_from_markdown(markdown: str) -> Dict[str, Any]:
    """
    Uses OpenAI to convert a Markdown table string into structured investor data.
    """
    client = get_openai_client()
    prompt = create_extraction_prompt(markdown)

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=1.0,
        )
        response_content = response.choices[0].message.content
        if response_content:
            return json.loads(response_content)
        return {}
    except Exception as e:
        print(f"An error occurred with the OpenAI API call: {e}")
        return {}


def extract_investor_data(pdf_path: str, verbose: bool = False, debug: bool = False) -> Dict[str, Any]:
    """
    Orchestrates the two-stage process: find and extract tables, then convert them to JSON.
    """
    metrics = {}

    # --- Stage 1: Fitz Table Scan ---
    scan_start_time = time.time()
    markdown_tables = find_and_extract_tables_as_markdown(pdf_path, SEARCH_KEYWORDS)
    metrics["scan_duration"] = time.time() - scan_start_time

    if not markdown_tables:
        if verbose:
            print("--- Efficiency Analysis ---")
            print(f"Total pages in document:        {get_page_count(pdf_path)}")
            print(f"Fitz table scan duration:       {metrics['scan_duration']:.2f}s")
            print("No relevant tables found.")
            print("---------------------------")
        return {"investors": [], "source_page": "", "confidence": "none"}

    combined_markdown = "\n\n".join(markdown_tables)

    # Debug: Show the raw markdown that will be sent to AI
    if debug:
        print("=" * 80)
        print("DEBUG: Markdown table(s) extracted by fitz (before AI processing)")
        print("=" * 80)
        print(combined_markdown)
        print("=" * 80)
        print(f"Total characters: {len(combined_markdown)}")
        print("=" * 80)

    # --- Stage 2: LLM Conversion ---
    api_start_time = time.time()
    extracted_data = _extract_data_from_markdown(combined_markdown)
    metrics["api_call_duration"] = time.time() - api_start_time

    investors = extracted_data.get("investors", [])
    source_pages = extracted_data.get("source_pages", [])

    # Verbose: print the efficiency analysis
    if verbose:
        print("--- Efficiency Analysis ---")
        print(f"Total pages in document:        {get_page_count(pdf_path)}")
        print(f"Fitz table scan duration:       {metrics['scan_duration']:.2f}s")
        print(f"LLM payload character count:    {len(combined_markdown)}")
        print(f"LLM API call duration:          {metrics['api_call_duration']:.2f}s")
        print("---------------------------")

    # Pydantic doesn't have a clean way to format a list of numbers, so do it manually
    page_str = ",".join(map(str, sorted(source_pages)))

    return {
        "investors": investors,
        "source_page": page_str,
        "confidence": _calculate_confidence(investors),
    }