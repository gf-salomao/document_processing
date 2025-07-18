"""Test Doc Type Indexing"""

from vector_db.indexer import DocumentIndexer


def main():
    indexer = DocumentIndexer()
    folder = "data/raw/email"
    indexer.index_documents(folder)

    print("Done indexing!")


if __name__ == "__main__":
    main()
