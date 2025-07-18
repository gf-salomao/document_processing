import os
import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from ocr.text_extractor import extract_text
from llm.entities_extractor import extract_entities
from logger import get_logger

logger = get_logger(__name__)


class DocumentIndexer:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        db_path: str = "data/chromadb",
    ):
        self.model_name = embedding_model
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.chroma = PersistentClient(path=db_path)
        self.collection = self.chroma.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn,  # type: ignore
        )

    def index_documents(self, folder_path: str):
        logger.info(f"Starting indexing for folder: {folder_path}")
        start_time = time.time()

        doc_ids, documents, metadatas = [], [], []
        current_id = 0

        for root, _, files in os.walk(folder_path):
            doc_type = os.path.basename(root)

            for filename in files:
                if filename.lower().endswith(
                    (".pdf", ".png", ".jpg", ".jpeg")
                ):
                    file_path = os.path.join(root, filename)
                    logger.info(f"Indexing {filename} (type: {doc_type}) ...")

                    text = extract_text(file_path)
                    if not text:
                        logger.warning(
                            f"Skipped {filename}: no text extracted"
                        )
                        continue

                    entities = extract_entities(text, doc_type)
                    doc_ids.append(f"doc_{current_id}")
                    documents.append(text)
                    metadatas.append(
                        {
                            "filename": filename,
                            "document_type": doc_type,
                            "entities": entities,
                        }
                    )
                    current_id += 1

        if not documents:
            logger.warning("No documents indexed!")
            return

        self.collection.upsert(
            documents=documents, metadatas=metadatas, ids=doc_ids
        )

        logger.info(f"Indexed {len(documents)} documents.")
        elapsed = time.time() - start_time
        logger.info(f"Indexing completed in {elapsed:.2f} seconds")

    def search_document_type(self, text: str, top_k: int = 1):
        results = self.collection.query(query_texts=[text], n_results=top_k)
        if results["metadatas"] and results["metadatas"][0]:
            doc_type = results["metadatas"][0][0].get("document_type")
            distance = results["distances"][0][0]
            confidence = float(1 / (1 + distance)) if distance else 1.0
            return doc_type, confidence
        return None, 0.0
