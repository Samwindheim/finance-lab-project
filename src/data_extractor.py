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

from .pdf_parser import find_and_extract_tables_as_markdown, get_page_count

LLM_MODEL = "gpt-5-mini"

SEARCH_KEYWORDS = [
    "förbin", "garant", "SEK"
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
    You are an expert AI assistant specializing in Swedish financial data processing.  
    Convert financial data (Page Text + Markdown tables) into structured JSON.
    Prioritize the markdown tables over the page text. 

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

    # Sort investors by investor level for consistent output
    investors.sort(key=lambda x: x.get("investor_level", 0))

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