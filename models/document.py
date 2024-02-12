from pydantic import BaseModel


class BaseDocument(BaseModel):
    id: str
    content: str
    doc_url: str
    metadata: dict | None = None


class BaseDocumentChunk(BaseDocument):
    document_id: str
    page_number: str = ""
    dense_embedding: list[float] | None = None
