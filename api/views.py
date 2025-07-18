import os
import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile

from llm.entities_extractor import extract_entities
from vector_db.indexer import DocumentIndexer
from ocr.text_extractor import extract_text
from ocr.preprocessing import preprocess_image

indexer = DocumentIndexer()


@csrf_exempt
def extract_entities_view(request):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Only POST method is allowed"}, status=405
        )

    start_time = time.time()

    upload = request.FILES.get("file")
    if not upload or not upload.name.lower().endswith(
        (".pdf", ".png", ".jpg", ".jpeg")
    ):
        return JsonResponse({"error": "Invalid or missing file"}, status=400)

    # Save file temporarily
    temp_path = f"temp_{upload.name}"
    with open(temp_path, "wb") as f:
        for chunk in upload.chunks():
            f.write(chunk)

    try:
        preprocessed = preprocess_image(temp_path)
        os.remove(temp_path)

        text = extract_text(preprocessed)
        if not text:
            raise ValueError("Text extraction failed")

        document_type, confidence = indexer.search_document_type(text)
        entities = extract_entities(text, document_type)

        response = {
            "document_type": document_type,
            "confidence": confidence,
            "entities": entities,
            "processing_time": f"{round(time.time() - start_time, 2)}s",
        }

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
