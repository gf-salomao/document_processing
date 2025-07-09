"""Test Doc Type Indexing"""

from vector_db.indexer import DocumentIndexer


def main():
    indexer = DocumentIndexer()
    folder = "data/raw"
    indexer.index_documents(folder, output_path="data/index/doc_type_index")

    print("Done indexing!")


if __name__ == "__main__":
    main()
