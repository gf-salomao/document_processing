# 🧠 Intelligent Document Understanding API

An end-to-end Django application to extract structured information from unstructured documents using OCR, semantic search (ChromaDB), and local LLaMA models.

---

## 🚀 **Features**
✅ Upload scanned PDFs & images  
✅ OCR text extraction (Tesseract)  
✅ Document type detection via ChromaDB & embeddings  
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
├── api/                # Django views and routes
├── core/                # Django core
├── ocr/                # OCR module
├── vector_db/          # ChromaDB index & semantic search
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

## 🤖 **Run the Django API**
```bash
python manage.py runserver
```

Test endpoint:
```
http://127.0.0.1:8000/api/extract_entities/
```

---

## 📦 **Example cURL request**
```bash
curl -X POST "http://127.0.0.1:8000/api/extract_entities/" \
  -F "file=@data/raw/invoices/invoice_001.pdf"
```

---

## 📂 **Batch Document Indexing**

To process an entire dataset folder:

```bash
python manage.py process_documents data/raw/
```

This command will:
- OCR each file
- Detect document type
- Extract entities
- Store everything in ChromaDB
- Log results to console and logs/app.log

---

## ✏ **Configuration**
- Change embedding model in `vector_db/indexer.py`
- Change LLaMA model & path in `llm/entities_extractor.py`

---

## 📍 **Future / Bonus ideas**
- Docker File
- Better models for Embbeding and LLM
- Add field-level confidence scores
- Add web UI for testing
- Add retry & fallback for JSON parsing

---

## ✏ **Testing files**
- Module testing files (development) are available on `tests` directory
- If you're using VSCode feel free to use the tests implemented on `.vscode/launch.json`

---

## 🧱 Architecture Overview

This project was designed with the goal of being a **100% free solution**, with no external paid APIs or keys required.

### 🧠 LLM Choice: Local & Free
To avoid unpredictable API costs (e.g., OpenAI, Azure), the pipeline uses local LLaMA models running on the user’s own machine. These models are smaller to ensure they work on personal devices, sacrificing some performance in exchange for accessibility and cost control.

### 🖼️ OCR & Image Processing
- **OCR Engine:** Tesseract
- **Preprocessing:** OpenCV
- **Text extraction model:** TinyLLaMA for entity recognition

The image preprocessing step is intentionally gentle to avoid over-sanitizing images:
- Tesseract relies on surrounding gray pixels as part of its inference features.
- Over-filtering could remove small text, which is common in forms and receipts.

Post-OCR, a simple postprocessing step removes non-meaningful characters and normalizes the output text.

### 📁 Dataset
The first two samples of each document type in the dataset were reserved for testing and excluded from the ChromaDB index.

---

## 🧑‍💻 **Author**
Gabriel Salomão