# 🧠 Intelligent Document Understanding API

An end-to-end FastAPI application to extract structured information from unstructured documents using OCR, semantic search (vector DB), and local LLaMA models.

---

## 🚀 **Features**
✅ Upload scanned PDFs & images  
✅ OCR text extraction (Tesseract)  
✅ Document type detection via FAISS & embeddings  
✅ Entity extraction using local LLaMA model  
✅ Standardized JSON output:
```json
{
  "document_type": "Invoice",
  "confidence": 0.92,
  "entities": {
    "invoice_number": "INV-12345",
    "date": "2024-01-01",
    "total_amount": "$450.00",
    "vendor_name": "ABC Corp"
  },
  "processing_time": "1.25s"
}
```

---

## ⚙️ **Project structure**
```plaintext
.
├── api/                # FastAPI application
├── ocr/                # OCR module
├── vector_db/          # FAISS index & semantic search
├── llm/                # Entity extraction with LLaMA
├── data/               # Raw documents & indexes
├── tests/              # Unit & integration tests
└── README.md
```

---

## 🛠 **Installation**
Install dependencies:
```bash
pip install -r requirements.txt
```

> ⚠ **Tesseract required:**  
Sßee instructions: https://tesseract-ocr.github.io/tessdoc/Installation.html

---
## 🔗 Setup
- Use setup.sh for a fast and easy setup on a new project

### **Prepare dataset**
- Dowloand the dataset using `dataset_download.sh`
- Copy all files to `data/raw`

- Note: The first two files of each document type were used as testing files out of the indexing process

- Alternatively, you can run `setup.sh` to automatically download and extract the dataset.

---

### Download the model
Download the `.gguf` model from [this link](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main).
Then place it in the `models/` folder and add the model name to the `llm/text_extractor.py` file

- Alternatively, you can run `setup.sh` to automatically download the model into the models folder.

---

## 🧩 **Prepare vector index**

Run indexing script (adjust folder to your dataset):
```bash
python tests/test_index.py
```

---

## 🤖 **Run the API**
```bash
uvicorn api.main:app --reload
```

Visit docs:
```
http://127.0.0.1:8000/docs
```

---

## 📦 **Example cURL request**
```bash
curl -X POST "http://127.0.0.1:8000/extract_entities/" \
  -F "file=@data/raw/invoices/invoice_001.pdf"
```

---

## ✏ **Configuration**
- Change embedding model in `vector_db/indexer.py`
- Change LLaMA model & path in `llm/entities_extractor.py`

---

## 📍 **Future / Bonus ideas**
- Docker File
- Better models for Embbeding and LLM
- Handle low-quality OCR with preprocessing
- Add field-level confidence scores
- Add web UI for testing
- Add retry & fallback for JSON parsing

---

## ✏ **Testing files**
- Module testing files (development) are available on `tests` directory
- If you're using VSCode feel free to use the tests implemented on `.vscode/launch.json`

---

## 🧑‍💻 **Author**
Gabriel Salomão