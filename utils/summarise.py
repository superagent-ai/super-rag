from decouple import config
from openai import AsyncOpenAI

from models.document import BaseDocument

client = AsyncOpenAI(
    api_key=config("OPENAI_API_KEY"),
)


def _generate_content(document: BaseDocument) -> str:
    return f"""Make an in depth summary the block of text below.

Text:
------------------------------------------
{document.text}
------------------------------------------

Your summary:"""


async def completion(document: BaseDocument):
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
