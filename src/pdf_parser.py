"""
This module is responsible for all direct interactions with PDF files.

It uses the PyMuPDF library (fitz) to perform two key tasks:
1.  **Keyword Scanning**: Efficiently searching through the document's text to
    find pages that are candidates for containing relevant data.
2.  **Table Extraction**: Using fitz's advanced layout-aware features to find
    and extract structured table data from the candidate pages.

The final output is a clean, Markdown-formatted string for each table,
ready for AI processing.
"""
from typing import List, Any

import fitz  # PyMuPDF


def _table_to_markdown(table: List[List[Any]]) -> str:
    """Converts a list of lists into a Markdown table string."""
    markdown = "| " + " | ".join(map(str, table[0])) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
    for row in table[1:]:
        markdown += "| " + " | ".join(map(str, row)) + " |\n"
    return markdown


def extract_tables_from_page(page: fitz.Page) -> List[str]:
    """
    Finds and extracts all tables on a given page and returns them as Markdown strings.

    Args:
        page: A fitz.Page object.

    Returns:
        A list of strings, where each string is a Markdown representation of a table.
    """
    tables = page.find_tables()
    markdown_tables = []
    for table in tables:
        # The .extract() method returns a list of lists, perfect for our conversion
        table_data = table.extract()
        if table_data:
            markdown_tables.append(_table_to_markdown(table_data))
    return markdown_tables


def find_and_extract_tables_as_markdown(pdf_path: str, search_keywords: List[str]) -> List[str]:
    """
    Scans a PDF for keywords, then extracts tables from the pages where keywords
    are found, including a context window of one page before and after.

    Args:
        pdf_path: The path to the PDF file.
        search_keywords: A list of keywords to identify candidate pages.

    Returns:
        A list of Markdown-formatted tables found on the candidate pages and their context.
    """
    found_tables = []
    try:
        doc = fitz.open(pdf_path)

        # First pass: Find all pages that contain our keywords
        candidate_pages = set()
        for page in doc:
            text = page.get_text("text", sort=True).lower()
            found_kws = {kw for kw in search_keywords if kw.lower() in text}
            if len(found_kws) >= 2:
                candidate_pages.add(page.number + 1)

        if not candidate_pages:
            # Fallback to single keyword search
            for page in doc:
                text = page.get_text("text", sort=True).lower()
                if any(keyword.lower() in text for keyword in search_keywords):
                    candidate_pages.add(page.number + 1)
        
        if not candidate_pages:
            doc.close()
            return []

        # Create a context window of pages to scan for tables (before, during, after)
        pages_to_scan_for_tables = set()
        for page_num in sorted(list(candidate_pages)):
            page_idx = page_num - 1
            if page_idx > 0:
                pages_to_scan_for_tables.add(page_idx - 1)  # Page before
            pages_to_scan_for_tables.add(page_idx)             # The candidate page
            if page_idx < doc.page_count - 1:
                pages_to_scan_for_tables.add(page_idx + 1)  # Page after

        # Second pass: Extract tables from the context window pages
        for page_idx in sorted(list(pages_to_scan_for_tables)):
            page = doc[page_idx]
            page_tables = extract_tables_from_page(page)
            if page_tables:
                page_num = page.number + 1
                page_text = page.get_text("text", sort=True)
                
                page_content = f"--- Page {page_num} ---\n"
                page_content += f"**Page Text:**\n{page_text}\n---\n"
                page_content += "**Extracted Table(s):**\n"
                page_content += "\n".join(page_tables)
                found_tables.append(page_content)

        doc.close()
    except Exception as e:
        print(f"Error processing PDF for table extraction: {e}")
    return found_tables


def get_page_count(pdf_path: str) -> int:
    """
    Gets the total number of pages in a PDF document.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        The total number of pages, or 0 if the file cannot be read.
    """
    try:
        doc = fitz.open(pdf_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return 0
