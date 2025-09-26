"""
This module is responsible for all direct interactions with PDF files.

It uses the PyMuPDF library (fitz) to extract tables from a PDF file. 
It also uses the Camelot library to extract tables from a PDF file.
"""
from typing import List, Any, Set

import fitz  # PyMuPDF
import camelot


def _table_to_markdown(table: List[List[Any]]) -> str:
    """Converts a list of lists into a Markdown table string."""
    markdown = "| " + " | ".join(map(str, table[0])) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
    for row in table[1:]:
        markdown += "| " + " | ".join(map(str, row)) + " |\n"
    return markdown


def _format_page_content(page: fitz.Page, tables: List[str]) -> str:
    """Formats the extracted page content with tables into a structured string."""
    page_num = page.number + 1
    page_text = page.get_text("text", sort=True)

    page_content = f"--- Page {page_num} ---\n"
    page_content += f"**Page Text:**\n{page_text}\n---\n"
    page_content += "**Extracted Table(s):**\n"
    page_content += "\n".join(tables)
    return page_content


def _extract_tables_with_camelot(pdf_path: str, page_num: int) -> List[List[List[Any]]]:
    """
    Fallback function to extract tables using Camelot-py for a single page.

    Args:
        pdf_path: The file path to the PDF document.
        page_num: The page number to extract tables from (1-based).

    Returns:
        A list of tables, where each table is a list of lists.
    """
    try:
        # Use stream for gridless tables, pages is 1-based
        tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')
        return [table.data for table in tables]
    except Exception as e:
        # Camelot can be noisy with errors, so we'll often ignore them
        # print(f"Camelot error on page {page_num}: {e}")
        return []


def extract_tables_from_page(page: fitz.Page) -> List[str]:
    """
    Finds and extracts all tables on a given page using PyMuPDF and returns them
    as Markdown strings.

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


def find_pages_to_scan(doc: fitz.Document, search_keywords: List[str]) -> Set[int]:
    """
    Finds pages containing keywords and returns a set of page indices to scan,
    including a context window of one page before and after.
    """
    candidate_pages_indices = set()
    for page in doc:
        text = page.get_text("text", sort=True).lower()
        found_kws = {kw for kw in search_keywords if kw.lower() in text}
        if len(found_kws) >= 2:
            candidate_pages_indices.add(page.number)

    if not candidate_pages_indices:
        # Fallback to single keyword search
        for page in doc:
            text = page.get_text("text", sort=True).lower()
            if any(keyword.lower() in text for keyword in search_keywords):
                candidate_pages_indices.add(page.number)

    if not candidate_pages_indices:
        return set()

    # Create a context window of pages to scan for tables (before, during, after)
    pages_to_scan_for_tables = set()
    for page_idx in sorted(list(candidate_pages_indices)):
        if page_idx > 0:
            pages_to_scan_for_tables.add(page_idx - 1)  # Page before
        pages_to_scan_for_tables.add(page_idx)             # The candidate page
        if page_idx < doc.page_count - 1:
            pages_to_scan_for_tables.add(page_idx + 1)  # Page after

    return pages_to_scan_for_tables


def extract_tables_with_pymupdf(doc: fitz.Document, pages_to_scan: Set[int]) -> List[str]:
    """Extracts tables from a set of pages using PyMuPDF."""
    found_tables = []
    for page_idx in sorted(list(pages_to_scan)):
        page = doc[page_idx]
        page_tables = extract_tables_from_page(page)
        if page_tables:
            found_tables.append(_format_page_content(page, page_tables))
    return found_tables


def extract_tables_with_camelot(doc: fitz.Document, pages_to_scan: Set[int], pdf_path: str) -> List[str]:
    """Extracts tables from a set of pages using Camelot."""
    found_tables = []
    for page_idx in sorted(list(pages_to_scan)):
        page = doc[page_idx]
        page_num = page.number + 1

        camelot_tables_data = _extract_tables_with_camelot(pdf_path, page_num)

        if camelot_tables_data:
            page_tables_markdown = [_table_to_markdown(table) for table in camelot_tables_data if table]

            if page_tables_markdown:
                found_tables.append(_format_page_content(page, page_tables_markdown))
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
