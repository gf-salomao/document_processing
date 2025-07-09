"""Implemets LLM (LLaMa) to extract entities"""

import json
import re
from llama_cpp import Llama
from .fields import fields_per_type

# load model once (adjust path)
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=1024,
    n_gpu_layers=40,
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


def extract_entities(text: str, document_type: str) -> dict:
    """
    Use local LLaMA to extract entities from text based on document type.
    """
    field_list = fields_per_type.get(
        document_type.lower(), ["field1", "field2"]
    )

    prompt = f"""
        You are an information extraction assistant.
        Extract these fields: {field_list}.
        Return your answer as JSON, wrapped between <json> and </json> tags, with NO extra text.

        Document Text:
        {text}
    """

    output = llm(prompt, max_tokens=512, stop=["</s>"])

    # output['choices'][0]['text'] for older llama-cpp-python
    generated_text = (
        output["choices"][0]["text"]
        if "choices" in output
        else output["generation"]
    )

    try:
        json_text = extract_json_block(generated_text).replace("'", '"')
        entities = json.loads(json_text)
    except json.JSONDecodeError as error:
        print(
            "Failed to parse LLM response as JSON", generated_text[:100], error
        )
        entities = {"error": "Failed to parse LLM response as JSON"}

    return entities
