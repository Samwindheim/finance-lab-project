"""
This module contains the core logic for AI-powered data extraction.

It orchestrates the main two-stage RAG pipeline:
1.  **Retrieval**: Invokes the `pdf_parser` to find and extract relevant tables
    from the source PDF, which are returned as clean Markdown.
2.  **Generation**: Constructs a detailed prompt with specific mapping and cleaning
    rules, then sends the Markdown table(s) to the OpenAI API (gpt-5-mini) to be
    converted into a structured JSON format.

It also calculates a final confidence score based on the completeness of the extracted data.
"""
import time
import os
import json
from openai import OpenAI
from typing import Dict, Any
import fitz

from .pdf_parser import (
    get_page_count,
    find_pages_to_scan,
    extract_tables_with_pymupdf,
    extract_tables_with_camelot,
    extract_text_only
)

LLM_MODEL = "gpt-5-mini"

SEARCH_KEYWORDS = [
    "förbin", "garant", "SEK", "totalt"
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


def create_extraction_prompt(text_input: str) -> str:
    """
    Creates the prompt for the OpenAI API to convert financial data (tables or text) to JSON.
    """
    return f"""
    You are an expert AI assistant specializing in Swedish financial data processing. 
    You are lookin for Teckningsförbindelser ochgarantiåtaganden
    Convert financial data (from Markdown tables or plain paragraph text) into structured JSON.
    If both tables and text are present, prioritize the markdown tables over the page text. 

    **Key Rules:**
    - Use Page Text for missing headers/context.
    - Each investor row → one JSON object.
    - Include: 
    - `name`
    - `commitment`: {{ "amount", "percent" }}
    - `investor_level`
    - Add `source_pages` from `--- Page X ---`.

    **Investor Levels:**
    - Underwriter (“Teckningsåtagare” or "Teckningsförbindelser" or similar) → level 0
    - Guarantor (“Garantier”, “Botten-garantier”, or "Bottom-up garantiåtaganden" or similar) → level 1
    - “Toppgarantier” or "Top-down garantiåtaganden" or similar → level 2
    - If same person has multiple roles → create separate objects.
    - If the level is not specified or ambiguous, default to `investor_level: 0`.

    **Data Cleaning:**
    - **Amount:**
    - Units: "TSEK" multiply by 1,000. E.g. "100" → "1000000".
    - Remove spaces, currency text, symbols.
    - Ranges → string (e.g., "2000 - 10000"). (apply units multiplication to ranges if necessary)
    - Decimals: The input text represents decimal points (1.5) as commas (1,5). Make sure to replace the comma with a period (e.g., "123 456,10" → 123456.10).
    - Spaces in numbers → remove (e.g., "123 456" → 123456).

    - **Percent:** remove `%`, use `.` decimal, store as float.

    - **Ignore totals/summaries** (rows like “Totalt”, “Summa”).

    - **Names Swedish: Namn:**
    - Do not alter spelling, punctuation, capitalization, or formatting (e.g., “Grimborg Consulting AB (Michael Grimborg)” → “Grimborg Consulting AB (Michael Grimborg)”).
    - Remove numbers at the end of names (e.g., “Jim Joe1” → “Jim Joe”).
    - If the name spans a new line, replace the new line with a space. "Svanberg & Co\nInvest AB" → "Svanberg & Co Invest AB"
    - Remove ending asterisk(s) from names (e.g., “Jim Joe*” → “Jim Joe”).

    - Important: Do not make up any data. It is important that there are no hallucinations.
    - Make sure to not grab the wrong data from the wrong tables or pages.

    **Output Format Example:**
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
    {text_input}
    ---
    """


def _extract_data_from_markdown(text_input: str) -> Dict[str, Any]:
    """
    Uses OpenAI to convert a Markdown table string into structured investor data.
    """
    client = get_openai_client()
    prompt = create_extraction_prompt(text_input)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            reasoning_effort="medium"
        )
        response_content = response.choices[0].message.content
        if response_content:
            return json.loads(response_content)
        return {}
    except Exception as e:
        print(f"An error occurred with the OpenAI API call: {e}")
        return {}


def extract_investor_data(pdf_path: str, verbose: bool = False, debug: bool = False, mode: str = "tables", gridless: bool = False) -> Dict[str, Any]:
    """
    Orchestrates the two-stage process: find and extract tables/text, then convert them to JSON.
    
    Args:
        pdf_path: The file path to the PDF document.
        verbose: If True, prints a detailed efficiency analysis report.
        debug: If True, prints the raw extracted content before AI processing.
        mode: The extraction mode - either "tables" or "text".
        gridless: If True, uses Camelot exclusively for table extraction.
    """
    metrics = {}
    investors = []
    source_pages = []
    extraction_method = "N/A"
    combined_markdown = ""

    # --- 1. Open PDF and Find Candidate Pages ---
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return {"investors": [], "source_page": ""}

    scan_start_time = time.time()
    pages_to_scan = find_pages_to_scan(doc, SEARCH_KEYWORDS)

    if not pages_to_scan:
        if verbose:
            print("--- Efficiency Analysis ---")
            print(f"Total pages in document:        {get_page_count(pdf_path)}")
            print("No candidate pages found based on keywords.")
            print("---------------------------")
        doc.close()
        return {"investors": [], "source_page": ""}

    # --- 2. Extract Data based on Mode (Text or Tables) ---
    # --- Mode: Plain Text Extraction (no table parsing) ---
    if mode == "text":
        extracted_text = extract_text_only(doc, pages_to_scan)
        metrics["scan_duration_text"] = time.time() - scan_start_time
        
        if extracted_text:
            extraction_method = "Plain Text"
            combined_markdown = "\n\n".join(extracted_text)

            if debug:
                print("=" * 80)
                print(f"DEBUG: Attempting LLM extraction with plain text ({len(combined_markdown)} chars)")
                print("=" * 80)
                print(combined_markdown)
                print("=" * 80)

            api_start_time = time.time()
            extracted_data = _extract_data_from_markdown(combined_markdown)
            metrics["api_call_duration_text"] = time.time() - api_start_time
            investors = extracted_data.get("investors", [])
            source_pages = extracted_data.get("source_pages", [])
    
    # --- Mode: Table Extraction (default) ---
    else:
        # If --gridless is flagged, use Camelot only.
        if gridless:
            camelot_start_time = time.time()
            markdown_tables = extract_tables_with_camelot(doc, pages_to_scan, pdf_path)
            metrics["scan_duration_camelot"] = time.time() - camelot_start_time

            if markdown_tables:
                extraction_method = "Camelot"
                combined_markdown = "\n\n".join(markdown_tables)

                if debug:
                    print("=" * 80)
                    print(f"DEBUG: Attempting LLM extraction with Camelot output ({len(combined_markdown)} chars)")
                    print("=" * 80)
                    print(combined_markdown)
                    print("=" * 80)

                api_start_time = time.time()
                extracted_data = _extract_data_from_markdown(combined_markdown)
                metrics["api_call_duration_camelot"] = time.time() - api_start_time
                investors = extracted_data.get("investors", [])
                source_pages = extracted_data.get("source_pages", [])
        
        # Default behavior: Try PyMuPDF first, then fall back to Camelot.
        else:
            # --- Pass 1: Extract tables using PyMuPDF ---
            markdown_tables = extract_tables_with_pymupdf(doc, pages_to_scan)
            metrics["scan_duration_pymupdf"] = time.time() - scan_start_time

            if markdown_tables:
                extraction_method = "PyMuPDF"
                combined_markdown = "\n\n".join(markdown_tables)

                if debug:
                    print("=" * 80)
                    print(f"DEBUG: Attempting LLM extraction with PyMuPDF output ({len(combined_markdown)} chars)")
                    print("=" * 80)
                    print(combined_markdown)
                    print("=" * 80)

                api_start_time = time.time()
                extracted_data = _extract_data_from_markdown(combined_markdown)
                metrics["api_call_duration_pymupdf"] = time.time() - api_start_time
                investors = extracted_data.get("investors", [])
                source_pages = extracted_data.get("source_pages", [])

            # --- Pass 2: Fallback to Camelot if no investors were found ---
            if not investors:
                if verbose and markdown_tables:
                    print("PyMuPDF found tables, but no investors were extracted. Falling back to Camelot.")

                camelot_start_time = time.time()
                markdown_tables = extract_tables_with_camelot(doc, pages_to_scan, pdf_path)
                metrics["scan_duration_camelot"] = time.time() - camelot_start_time

                if markdown_tables:
                    extraction_method = "Camelot"
                    combined_markdown = "\n\n".join(markdown_tables)

                    if debug:
                        print("=" * 80)
                        print(f"DEBUG: Attempting LLM extraction with Camelot fallback output ({len(combined_markdown)} chars)")
                        print("=" * 80)
                        print(combined_markdown)
                        print("=" * 80)

                    api_start_time = time.time()
                    extracted_data = _extract_data_from_markdown(combined_markdown)
                    metrics["api_call_duration_camelot"] = time.time() - api_start_time
                    investors = extracted_data.get("investors", [])
                    source_pages = extracted_data.get("source_pages", [])

    doc.close()

    # --- 3. Final Validation and Post-Processing ---
    if not investors:
        if verbose:
            print("--- Efficiency Analysis ---")
            print(f"Total pages in document:        {get_page_count(pdf_path)}")
            if "scan_duration_pymupdf" in metrics:
                print(f"PyMuPDF scan duration:          {metrics['scan_duration_pymupdf']:.2f}s")
            if "scan_duration_camelot" in metrics:
                print(f"Camelot scan duration:          {metrics['scan_duration_camelot']:.2f}s")
            print("No investors extracted after all attempts.")
            print("---------------------------")
        return {"investors": [], "source_page": ""}

    # Post-process to ensure investor_level is always an integer
    for investor in investors:
        if investor.get("investor_level") is None:
            investor["investor_level"] = 0  # Default to underwriter

    # Sort investors by investor level for consistent output
    investors.sort(key=lambda x: x.get("investor_level", 0))

    # --- 4. Reporting and Output ---
    # Verbose: print the efficiency analysis
    if verbose:
        print("--- Efficiency Analysis ---")
        print(f"Total pages in document:        {get_page_count(pdf_path)}")
        print(f"Extraction mode:                {mode}")
        if "scan_duration_text" in metrics:
            print(f"Text scan duration:             {metrics['scan_duration_text']:.2f}s")
        if "api_call_duration_text" in metrics:
            print(f"LLM API call duration (Text):   {metrics['api_call_duration_text']:.2f}s")
        if "scan_duration_pymupdf" in metrics:
            print(f"Fitz table scan duration:       {metrics['scan_duration_pymupdf']:.2f}s")
        if "api_call_duration_pymupdf" in metrics:
            print(f"LLM API call duration (Fitz):   {metrics.get('api_call_duration_pymupdf', 0):.2f}s")
        if "scan_duration_camelot" in metrics:
            print(f"Camelot table scan duration:    {metrics['scan_duration_camelot']:.2f}s")
        if "api_call_duration_camelot" in metrics:
            print(f"LLM API call duration (Camelot): {metrics.get('api_call_duration_camelot', 0):.2f}s")
        print(f"Final pdf parsing method:        {extraction_method}")
        print(f"LLM payload character count:    {len(combined_markdown)}")
        print("---------------------------")

    # Pydantic doesn't have a clean way to format a list of numbers, so do it manually
    page_str = ",".join(map(str, sorted(source_pages)))

    return {
        "investors": investors,
        "source_page": page_str,
    }