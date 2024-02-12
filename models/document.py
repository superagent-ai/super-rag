from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str
    file_url: str
    metadata: dict | None = None
