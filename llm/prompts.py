"""Prompts per document type for LLM extraction.

This dictionary maps each document type to its expected fields and the LLM prompt to retrieve them.
"""

custom_prompts = {
    "specification": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "specification_id": "",\n'
        '  "title": "",\n'
        '  "version": "",\n'
        '  "date": "",\n'
        '  "author": ""\n'
        "}"
    ),
    "email": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "from": "",\n'
        '  "to": "",\n'
        '  "date": "",\n'
        '  "subject": "",\n'
        '  "body": ""\n'
        "}"
    ),
    "advertisement": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "campaign_name": "",\n'
        '  "start_date": "",\n'
        '  "end_date": "",\n'
        '  "company": "",\n'
        '  "target_audience": ""\n'
        "}"
    ),
    "handwritten": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "writer_name": "",\n'
        '  "date": "",\n'
        '  "content_summary": ""\n'
        "}"
    ),
    "scientific report": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "report_title": "",\n'
        '  "authors": "",\n'
        '  "publication_date": "",\n'
        '  "summary": "",\n'
        '  "keywords": ""\n'
        "}"
    ),
    "budget": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "budget_id": "",\n'
        '  "fiscal_year": "",\n'
        '  "total_amount": "",\n'
        '  "department": "",\n'
        '  "approval_date": ""\n'
        "}"
    ),
    "scientific publication": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "title": "",\n'
        '  "authors": "",\n'
        '  "journal": "",\n'
        '  "publication_date": "",\n'
        '  "doi": ""\n'
        "}"
    ),
    "presentation": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "presentation_title": "",\n'
        '  "presenter": "",\n'
        '  "date": "",\n'
        '  "event_name": "",\n'
        '  "slide_count": ""\n'
        "}"
    ),
    "file folder": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "folder_name": "",\n'
        '  "creation_date": "",\n'
        '  "owner": "",\n'
        '  "number_of_documents": ""\n'
        "}"
    ),
    "memo": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "memo_id": "",\n'
        '  "author": "",\n'
        '  "recipient": "",\n'
        '  "date": "",\n'
        '  "subject": ""\n'
        "}"
    ),
    "resume": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "candidate_name": "",\n'
        '  "email": "",\n'
        '  "phone": "",\n'
        '  "education": "",\n'
        '  "skills": ""\n'
        "}"
    ),
    "invoice": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "invoice_number": "",\n'
        '  "date": "",\n'
        '  "total_amount": "",\n'
        '  "vendor_name": ""\n'
        "}"
    ),
    "letter": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "sender": "",\n'
        '  "recipient": "",\n'
        '  "date": "",\n'
        '  "subject": "",\n'
        '  "signature": ""\n'
        "}"
    ),
    "questionnaire": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "questionnaire_id": "",\n'
        '  "title": "",\n'
        '  "creation_date": "",\n'
        '  "number_of_questions": ""\n'
        "}"
    ),
    "form": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "form_id": "",\n'
        '  "submission_date": "",\n'
        '  "applicant_name": "",\n'
        '  "purpose": ""\n'
        "}"
    ),
    "news article": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "headline": "",\n'
        '  "author": "",\n'
        '  "publication_date": "",\n'
        '  "source": "",\n'
        '  "summary": ""\n'
        "}"
    ),
    "default": (
        "Fill in the missing values in the following JSON based on the document text. Only use information present in the text. Leave any unknown fields as empty strings. Return only the completed JSON.\n"
        "{\n"
        '  "field1": "",\n'
        '  "field2": ""\n'
        "}"
    ),
}
