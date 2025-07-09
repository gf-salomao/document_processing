"""OCR Extractor using Tesseract"""

from typing import Optional
import os
import pytesseract
from pdf2image import convert_from_path


def extract_text(file_path: str) -> Optional[str]:
    """
    Extract text from a PDF or image file using Tesseract OCR.

    Parameters
    ----------
    file_path : str
        Path to the document file.

    Returns
    -------
    Optional[str]
        Extracted text, or None if extraction fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = ""
    try:
        if file_path.lower().endswith(".pdf"):
            images = convert_from_path(file_path)
            for img in images:
                text += pytesseract.image_to_string(img)
        else:
            text = pytesseract.image_to_string(file_path)
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return None

    return text.strip()
