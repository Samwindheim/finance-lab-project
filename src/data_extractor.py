"""
This module contains the core logic for extracting investor data from text.

It implements a two-stage process:
1.  **Search**: Scans the document text for relevant keywords to find candidate pages.
2.  **Extract**: Sends the relevant text snippet to the OpenAI API with a detailed
    prompt to get structured data back.

It also handles communication with the OpenAI API and shapes the final data output.
"""

import time
import os
import json
from openai import OpenAI
from typing import Dict, Any, List, Set, Tuple

from src.pdf_parser import extract_text_by_page, get_page_count


SEARCH_KEYWORDS = [
    "teckningsåtagare", "garant", 
    "åtaganden", "garantiåtaganden", "teckningsåtaganden"
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


def _format_page_numbers(page_numbers: List[int]) -> str:
    """
    Formats a list of page numbers into a condensed string (e.g., "11-12").
    """
    if not page_numbers:
        return ""
    if len(page_numbers) == 1:
        return str(page_numbers[0])
    return f"{min(page_numbers)}-{max(page_numbers)}"


def _calculate_confidence(investors: List[Dict[str, Any]]) -> str:
    """
    Calculates a confidence score based on the completeness of the extracted data.

    - high: All investors have both amount and percent.
    - medium: Some investors are missing either an amount or a percent.
    - low: Some investors are missing both amount and percent.
    - none: No investors were found.

    Args:
        investors: The list of extracted investor dictionaries.

    Returns:
        A string representing the confidence level.
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


def _find_relevant_text(pdf_path: str) -> Tuple[str, Set[int]]:
    """
    Scans a PDF for keywords, identifies relevant pages, and returns their combined text
    with a context window (one page before and one page after).

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A tuple containing the combined text and a set of the original candidate page numbers.
    """
    pages_text: List[Tuple[int, str]] = list(extract_text_by_page(pdf_path))
    if not pages_text:
        return "", set()

    candidate_pages = set()
    for page_num, text in pages_text:
        found_keywords = {keyword for keyword in SEARCH_KEYWORDS if keyword.lower() in text.lower()}
        if len(found_keywords) >= 2:
            candidate_pages.add(page_num)

    if not candidate_pages:
        # Fallback to the single-keyword search if the stricter search finds nothing
        for page_num, text in pages_text:
            if any(keyword.lower() in text.lower() for keyword in SEARCH_KEYWORDS):
                candidate_pages.add(page_num)

    if not candidate_pages:
        return "", set()

    # Create a full context window of relevant pages (including one page before and one after the matches)
    relevant_pages_indices = set()
    for page_num in sorted(list(candidate_pages)):
        page_idx = page_num - 1  # convert 1-based page number to 0-based index
        if page_idx > 0:
            relevant_pages_indices.add(page_idx - 1)
        relevant_pages_indices.add(page_idx)
        if page_idx < len(pages_text) - 1:
            relevant_pages_indices.add(page_idx + 1)

    # Combine the text from the relevant pages
    full_text = []
    for i in sorted(list(relevant_pages_indices)):
        page_num, text = pages_text[i]
        full_text.append(f"--- Page {page_num} ---\n{text}")

    return "\n\n".join(full_text), candidate_pages


def create_extraction_prompt(text: str) -> str:
    """
    Creates the prompt for the OpenAI API to extract investor information.
    """
    return f"""
    You are an expert AI assistant specializing in financial document analysis. Your task is to extract information about underwriters and guarantors from the provided text of a financial memorandum.

    **Definitions:**
    - **Underwriter (Swedish: "teckningsåtagare")**: A person or entity that has committed to underwriting an issue. They are not compensated. Their investor level is always 0.
    - **Guarantor (Swedish: "Garant")**: A person or entity that agrees to sign up for shares if the issue is not filled. They are compensated for this service. Their investor level starts at 1 for the first/lowest guarantee level, 2 for the next, and so on.

    **Instructions:**
    1.  Carefully read the text provided below.
    2.  Identify all underwriters and guarantors.
    3.  Extract the following information for each one:
        - `name` Swedish: "Namn": The full name of the person or entity.
        - `commitment` Swedish: "Åtagande": The amount in SEK (or other currency) and/or the percentage of the issue. Both should be included if available.
        - `investor_level`: Assign `0` for underwriters. For guarantors, assign `1` for the primary or "bottom" guarantee level, and increment the number for any subsequent "top" or additional levels.
    4.  Format the output as a single JSON object. The root of the object should contain a list called `investors`.
    5.  **Crucially, do not hallucinate or invent any information.** If a piece of data (like a commitment amount or percentage) is not present for an investor, set its value to `null`. If no underwriters or guarantors are found in the text, return an empty `investors` list.
    6.  The JSON output should only contain the `investors` list. Do not include any other keys at the root level.
    7. In your final JSON response, also include a key named `source_pages` which is a list of the page numbers (as integers) from which you extracted the investor data. You can identify the page numbers from the `--- Page X ---` markers in the input text.

    **Example Output Format:**
    ```json
    {{
      "source_pages": [7, 8],
      "investors": [
        {{
          "name": "Investor Name AB",
          "commitment": {{
            "amount": 500000,
            "percent": 3.5
          }},
          "investor_level": 1
        }},
        {{
          "name": "Another Investor",
          "commitment": {{
            "amount": null,
            "percent": 7.6
          }},
          "investor_level": 0
        }}
      ]
    }}
    ```

    **Text for Analysis:**
    ---
    {text}
    ---
    """


def _extract_data_from_text(text: str) -> Dict[str, Any]:
    """
    Uses OpenAI to extract investor data from a given text snippet.

    Args:
        text: The text content to analyze.

    Returns:
        A dictionary containing the extracted data. Returns an empty dict if extraction fails.
    """
    client = get_openai_client()
    prompt = create_extraction_prompt(text)

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
    Orchestrates the two-stage process: find relevant text in the PDF, then extract data from it.

    Args:
        pdf_path: The path to the PDF file.
        verbose: If True, prints a detailed efficiency analysis.
        debug: If True, prints the raw text that will be sent to the AI.

    Returns:
        A dictionary with the extracted data, including investors, source page, and confidence.
    """
    metrics = {}

    # --- Stage 1: Fitz Scan ---
    scan_start_time = time.time()
    relevant_text, source_pages_set = _find_relevant_text(pdf_path)
    metrics["scan_duration"] = time.time() - scan_start_time

    if not relevant_text:
        if verbose:
            print("--- Efficiency Analysis ---")
            print(f"Total pages in document:        {get_page_count(pdf_path)}")
            print(f"Fitz keyword scan duration:     {metrics['scan_duration']:.2f}s")
            print("No relevant keywords found.")
            print("---------------------------")
        return {"investors": [], "source_page": "", "confidence": "none"}

    # Debug: Show the raw text that will be sent to AI
    if debug:
        print("=" * 80)
        print("DEBUG: Raw text extracted by fitz (before AI processing)")
        print("=" * 80)
        print(relevant_text)
        print("=" * 80)
        print(f"Total characters: {len(relevant_text)}")
        print("=" * 80)

    # --- Stage 2: LLM Extraction ---
    api_start_time = time.time()
    extracted_data = _extract_data_from_text(relevant_text)
    metrics["api_call_duration"] = time.time() - api_start_time

    investors = extracted_data.get("investors", [])
    source_pages = extracted_data.get("source_pages", [])

    if verbose:
        print("--- Efficiency Analysis ---")
        print(f"Total pages in document:        {get_page_count(pdf_path)}")
        print(f"Fitz keyword scan duration:     {metrics['scan_duration']:.2f}s")
        print(f"Candidate pages found:          {sorted(list(source_pages_set))}")
        print(f"LLM payload character count:    {len(relevant_text)}")
        print(f"LLM API call duration:          {metrics['api_call_duration']:.2f}s")
        print("---------------------------")

    return {
        "investors": investors,
        "source_page": _format_page_numbers(source_pages),
        "confidence": _calculate_confidence(investors),
    }
