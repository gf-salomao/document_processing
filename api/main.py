"""FastAPI API Implementation"""

# api/main.py


import os
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from llm.entities_extractor import extract_entities
from vector_db.indexer import DocumentIndexer
from ocr.text_extractor import extract_text

app = FastAPI(title="Document Understanding API")

# Load index once at startup
indexer = DocumentIndexer()
indexer.load_index(
    "data/index/doc_type_index.index", "data/index/doc_type_index_meta.pkl"
)


@app.post("/extract_entities/")
async def extract_entities_endpoint(file: UploadFile = File(...)):
    """
    Accepts a document file, extracts text, predicts document type,
    and returns JSON response.
    """
    start_time = time.time()

    # Validate file type
    if not file.filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Step 1: OCR
    text = extract_text(temp_path)
    os.remove(temp_path)

    if not text:
        raise HTTPException(status_code=500, detail="Failed to extract text")

    # Step 2: Predict document type
    document_type, confidence = indexer.search_document_type(text)

    # Step 3: Dummy entities (next step: call LLM)
    entities = extract_entities(text, document_type)

    processing_time = round(time.time() - start_time, 2)

    response = {
        "document_type": document_type,
        "confidence": confidence,
        "entities": entities,
        "processing_time": f"{processing_time}s",
    }

    return JSONResponse(content=response)
