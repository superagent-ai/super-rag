import uuid
from typing import List, Optional

from pydantic import BaseModel, validator


class BaseDocument(BaseModel):
    id: str
    content: str
    doc_url: str
    metadata: dict | None = None


class BaseDocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    doc_url: str
    metadata: dict | None = None
    page_number: str = ""
    dense_embedding: Optional[List[float]] = None

    @validator("id")
    def id_must_be_valid_uuid(_cls, v):
        try:
            uuid_obj = uuid.UUID(v, version=4)
            return str(uuid_obj)
        except ValueError:
            raise ValueError("id must be a valid UUID")

    @validator("dense_embedding")
    def embeddings_must_be_list_of_floats(_cls, v):
        if v is None:
            return v  # Allow None to pass through
        if not all(isinstance(item, float) for item in v):
            raise ValueError("embeddings must be a list of floats")
        return v
