"""Implemets LLM (LLaMa) to extract entities"""

import json
import re
from llama_cpp import Llama
from .fields import fields_per_type
from .prompts import custom_prompts

import os
import time
from logger import get_logger

logger = get_logger(__name__)

n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", "0"))

llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=n_gpu_layers,
)


def extract_json_block(text: str) -> str:
    """
    Extracts the first JSON object found in the text.

    Parameters
    ----------
    text : str
        Input text containing JSON and other text.

    Returns
    -------
    str
        JSON string or empty string if not found.
    """
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return ""


# Helper function to validate extracted entities against schema and confidence values
def validate_entities(entities: dict, document_type: str) -> dict:
    """
    Validate extracted entities against schema for the document type.

    Parameters
    ----------
    entities : dict
        The extracted entities with values and confidence.
    document_type : str
        Document type name.

    Returns
    -------
    dict
        Dictionary with missing fields, invalid confidence values, and the original entities.
    """
    expected_fields = fields_per_type.get(document_type.lower(), [])
    missing_fields = []
    invalid_confidences = []

    for field in expected_fields:
        if field not in entities:
            missing_fields.append(field)
        else:
            field_value = entities[field]
            if not isinstance(field_value, dict):
                invalid_confidences.append(field)
            else:
                conf = field_value.get("confidence", None)
                if not isinstance(conf, (float, int)) or not (0 <= conf <= 1):
                    invalid_confidences.append(field)

    return {
        "missing_fields": missing_fields,
        "invalid_confidences": invalid_confidences,
        "entities": entities,
    }


def extract_entities(text: str, document_type: str) -> dict:
    """
    Use local LLaMA to extract entities from text based on document type.
    """
    logger.info(f"Extracting entities for document type: {document_type}")
    start_time = time.time()

    max_ctx = 2048
    reserved_prompt = 150
    reserved_output = 150
    max_doc_tokens = max_ctx - reserved_prompt - reserved_output

    text_tokens = llm.tokenize(text.encode("utf-8"))
    if len(text_tokens) > max_doc_tokens:
        text = llm.detokenize(text_tokens[:max_doc_tokens]).decode(
            "utf-8", errors="ignore"
        )

    prompt_template = custom_prompts.get(
        document_type.lower(), custom_prompts["default"]
    )
    prompt = f"{prompt_template}\n\nDocument Text:\n\n{text}"

    logger.debug("--- Prompt ---\n%s", prompt)

    output = llm(prompt, max_tokens=512, stop=["</s>"])

    # output['choices'][0]['text'] for older llama-cpp-python
    generated_text = (
        output["choices"][0]["text"]
        if "choices" in output
        else output["generation"]
    )

    logger.debug("--- Response ---\n%s", generated_text)

    try:
        json_text = extract_json_block(generated_text).replace("'", '"')
        entities = json.loads(json_text)
    except json.JSONDecodeError as error:
        logger.error(
            "Failed to parse LLM response as JSON: %s\nError: %s",
            generated_text[:100],
            error,
        )
        entities = {"error": "Failed to parse LLM response as JSON"}

    validation = validate_entities(entities, document_type)
    elapsed = time.time() - start_time
    logger.info(f"Entity extraction completed in {elapsed:.2f} seconds")
    return validation
