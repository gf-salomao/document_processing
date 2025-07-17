"""Test OCR function"""

import os
import difflib
from ocr.text_extractor import extract_text
from ocr.preprocessing import preprocess_image


def test_ocr_on_sample_files(folder_path: str):
    """
    Test OCR extraction on all files in the given folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing documents (PDFs/images).
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Only process PDFs or common image formats
        if filename.lower().endswith(
            (".pdf", ".png", ".jpg", ".jpeg", ".tiff")
        ):
            print(f"\n=== Processing: {filename} ===")

            # Extract text from original
            original_text = extract_text(file_path) or ""

            # Extract text from preprocessed
            preprocessed = preprocess_image(file_path)
            processed_text = extract_text(preprocessed) or ""

            print("---original")
            print(original_text)

            print("+++preprocessed")
            print(processed_text)


if __name__ == "__main__":
    folder = "data/raw/email"
    test_ocr_on_sample_files(folder)
