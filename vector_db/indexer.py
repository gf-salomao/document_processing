"""Document Indexer to generate an Vector Database"""

# vector_db/indexer.py

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ocr.text_extractor import extract_text

# Disabling parallelism to skip warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DocumentIndexer:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the indexer with embedding model.
        """
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.id_to_metadata = {}  # map index id to document type (for search)

    def index_documents(self, folder_path: str, output_path: str):
        """
        Index all documents in folder and save FAISS index and metadata.

        Parameters
        ----------
        folder_path : str
            Path to documents.
        output_path : str
            Path to save index and metadata (without extension).
        """
        embeddings = []
        current_id = 0

        for root, _, files in os.walk(folder_path):
            doc_type = os.path.basename(root)

            for filename in files:
                if filename.lower().endswith(
                    (".pdf", ".png", ".jpg", ".jpeg")
                ):
                    file_path = os.path.join(root, filename)
                    print(f"Indexing {filename} (type: {doc_type}) ...")

                    text = extract_text(file_path)
                    if not text:
                        print("Skipped: no text extracted")
                        continue

                    emb = self.model.encode(text)
                    embeddings.append(emb)

                    self.id_to_metadata[current_id] = {
                        "filename": filename,
                        "document_type": doc_type,
                    }
                    current_id += 1

        if not embeddings:
            print("No documents indexed!")
            return

        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))

        # save index and metadata
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        faiss.write_index(self.index, f"{output_path}.index")
        with open(f"{output_path}_meta.pkl", "wb") as f:
            pickle.dump(self.id_to_metadata, f)

        print(f"Indexed {len(embeddings)} documents.")

    def load_index(self, index_path: str, metadata_path: str):
        """
        Load index and metadata from disk.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.id_to_metadata = pickle.load(f)

    def search_document_type(self, text: str, top_k: int = 1):
        """
        Search document type given extracted text.

        Returns
        -------
        tuple: (document_type, confidence_score)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")

        emb = self.model.encode(text)
        emb = emb.reshape(1, -1).astype("float32")

        distances, indices = self.index.search(emb, top_k)
        idx = indices[0][0]
        distance = distances[0][0]

        metadata = self.id_to_metadata.get(idx)
        if metadata:
            doc_type = metadata["document_type"]
            confidence = float(1 / (1 + distance))  # simple scoring
            return doc_type, confidence
        else:
            return None, 0.0
