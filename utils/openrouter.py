import json
import requests
from decouple import config
from llama_index import Document
from litellm import acompletion


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-small"


async def completion(document: Document):
    api_key = config("OPENROUTER_API_KEY")
    content = _generate_content(document)

    headers = _generate_headers(api_key)
    data = _generate_data(content)

    response = requests.post(
        url=OPENROUTER_API_URL,
        headers=headers,
        data=json.dumps(data),
    )

    docs = response.json()
    print(docs)


def _generate_content(document: Document) -> str:
    return f"""Summarize the block of text below.

Text:
------------------------------------------
{document.get_content()}
------------------------------------------

Your summary:"""


def _generate_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def _generate_data(content: str) -> dict:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
    }
