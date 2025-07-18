"""Test Index Searching"""

from vector_db.indexer import DocumentIndexer
from ocr.text_extractor import extract_text


import os

def main():
    indexer = DocumentIndexer()

    folder = "data/raw/email"
    for root, _, files in os.walk(folder):
        for name in files:
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")):
                continue

            path = os.path.join(root, name)
            text = extract_text(path)
            doc_type, confidence = indexer.search_document_type(text)
            print(f"{name} → Predicted type: {doc_type}, confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()
