"""OCR Extractor using Tesseract"""

from typing import Optional
import os
import pytesseract
from pdf2image import convert_from_path
import time

from logger import get_logger

logger = get_logger(__name__)


def extract_text(file_or_image) -> Optional[str]:
    """
    Extract text from a PDF, image file path, or in-memory OpenCV image using Tesseract OCR.

    Parameters
    ----------
    file_or_image : str or numpy.ndarray
        Path to the document file or in-memory OpenCV image.

    Returns
    -------
    Optional[str]
        Extracted text, or None if extraction fails.
    """
    logger.info("Starting OCR extraction")
    start_time = time.time()
    text = ""
    try:
        if isinstance(file_or_image, str):
            if not os.path.exists(file_or_image):
                raise FileNotFoundError(f"File not found: {file_or_image}")
            if file_or_image.lower().endswith(".pdf"):
                images = convert_from_path(file_or_image)
                for img in images:
                    text += pytesseract.image_to_string(img)
            else:
                text = pytesseract.image_to_string(file_or_image)
        else:
            # Assume it's an OpenCV image (numpy array)
            text = pytesseract.image_to_string(file_or_image)
    except Exception as e:
        logger.exception(f"OCR extraction failed: {e}")
        return None

    elapsed = time.time() - start_time
    logger.info(f"OCR extraction completed in {elapsed:.2f} seconds")
    return text.strip()
