"""
This module is responsible for handling all interactions with PDF files.

It uses the PyMuPDF library (fitz) to extract text content from PDF documents,
providing functions to get text from the entire document at once or page by page.
"""
from typing import Generator

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from all pages of a PDF document.

    Args:
        pdf_path: The file path to the PDF document.

    Returns:
        A single string containing all the text from the PDF.
    """
    full_text = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            full_text.append(page.get_text())
        doc.close()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""
    return "\n".join(full_text)


def extract_text_by_page(pdf_path: str) -> Generator[tuple[int, str], None, None]:
    """
    Extracts text from a PDF document, yielding one page at a time.

    Args:
        pdf_path: The file path to the PDF document.

    Yields:
        A tuple containing the page number (1-based) and the text content of that page.
    """
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            yield i + 1, page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return


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
