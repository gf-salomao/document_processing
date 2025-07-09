"""Test LLM Entity Extraction"""

import os
from llm.entities_extractor import extract_entities
from ocr.text_extractor import extract_text


def main():
    # Choose a document type from your 16 classes, e.g., 'invoice'
    document_type = "letter"

    test_folder = "data/testing/letter"
    for filename in os.listdir(test_folder):
        file_path = os.path.join(test_folder, filename)
        text = extract_text(file_path)

        print(
            f"\nTesting file: {filename} for document type: '{document_type}'"
        )
        print("-" * 60)

        entities = extract_entities(text, document_type)

        print("Extracted entities:")
        print(entities)


if __name__ == "__main__":
    main()
