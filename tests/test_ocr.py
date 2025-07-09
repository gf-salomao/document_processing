"""Test OCR function"""

import os
from ocr.text_extractor import extract_text


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

            text = extract_text(file_path)

            if text:
                print(f"Extracted text (first 500 chars):\n{text[:500]}...")
            else:
                print("Failed to extract text or got empty result.")


if __name__ == "__main__":
    folder = "data/raw/email"
    test_ocr_on_sample_files(folder)
