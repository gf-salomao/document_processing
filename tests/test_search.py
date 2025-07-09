"""Test Index Searching"""

from vector_db.indexer import DocumentIndexer
from ocr.text_extractor import extract_text


def main():
    indexer = DocumentIndexer()
    indexer.load_index(
        "data/index/doc_type_index.index", "data/index/doc_type_index_meta.pkl"
    )

    # pick a test file
    file = "data/testing/news_article/0000092722.jpg"
    text = extract_text(file)

    doc_type, confidence = indexer.search_document_type(text)
    print(f"Predicted type: {doc_type}, confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()
