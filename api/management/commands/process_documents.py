import os
import time
import json

from django.core.management.base import BaseCommand

from vector_db.indexer import DocumentIndexer
from ocr.text_extractor import extract_text
from llm.entities_extractor import extract_entities
from logger import get_logger

logger = get_logger(__name__)


class Command(BaseCommand):
    help = "Process and index all documents in a given folder recursively."

    def add_arguments(self, parser):
        parser.add_argument(
            "folder", type=str, help="Path to the folder containing documents"
        )

    def handle(self, *args, **options):
        folder = options["folder"]
        indexer = DocumentIndexer()

        if not os.path.exists(folder):
            logger.error(f"Provided folder does not exist: {folder}")
            return

        logger.info(f"Processing documents in: {folder}")
        start_time = time.time()

        for root, _, files in os.walk(folder):
            for filename in files:
                if not filename.lower().endswith(
                    (".pdf", ".png", ".jpg", ".jpeg")
                ):
                    continue

                path = os.path.join(root, filename)
                logger.info(f"Processing file: {filename}")

                try:
                    text = extract_text(path)

                    if not text:
                        logger.warning(f"No text extracted from {filename}")
                        continue

                    doc_type = os.path.basename(folder)
                    confidence = 1.0  # Placeholder confidence for folder-based classification
                    entities = extract_entities(text, doc_type)

                    indexer.collection.upsert(
                        documents=[text],
                        metadatas=[
                            {
                                "filename": filename,
                                "document_type": doc_type,
                                "confidence": confidence,
                                "entities": json.dumps(entities),
                            }
                        ],
                        ids=[f"doc_{time.time_ns()}"],
                    )

                    logger.info(f"Successfully processed: {filename}")

                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")

        elapsed = time.time() - start_time
        logger.info(f"Finished processing. Total time: {elapsed:.2f} seconds")
