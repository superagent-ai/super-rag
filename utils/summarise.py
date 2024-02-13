from decouple import config
from llama_index import Document
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=config("OPENAI_API_KEY"),
)


def _generate_content(document: Document) -> str:
    return f"""Make an in depth summary the block of text below.

Text:
------------------------------------------
{document.get_content()}
------------------------------------------

Your summary:"""


SUMMARY_SUFFIX = "summary"


async def completion(document: Document):
    content = _generate_content(document)
    completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model="gpt-3.5-turbo-16k",
    )

    return completion.choices[0].message.content
