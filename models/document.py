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
    doc_url: str
    document_id: str
    content: str
    source: str
    source_type: str
    chunk_index: int | None = None
    title: str | None = None
    token_count: int | None = None
    page_number: int | None = None
    metadata: dict | None = None
    dense_embedding: Optional[List[float]] = None

    @classmethod
    def from_metadata(cls, metadata: dict):
        exclude_keys = {
            "chunk_id",
            "chunk_index",
            "document_id",
            "doc_url",
            "content",
            "source",
            "source_type",
            "title",
            "token_count",
            "page_number",
        }
        filtered_metadata = {k: v for k, v in metadata.items() if k not in exclude_keys}
        return cls(
            id=metadata.get("chunk_id", ""),
            **metadata,
            metadata=filtered_metadata,
            dense_embedding=metadata.get("values"),
        )

    @validator("id")
    def id_must_be_valid_uuid(cls, v):
        try:
            uuid_obj = uuid.UUID(v, version=4)
            return str(uuid_obj)
        except ValueError:
            raise ValueError(f"id must be a valid UUID, got {v}")

    @validator("dense_embedding")
    def embeddings_must_be_list_of_floats(cls, v):
        if v is None:
            return v  # Allow None to pass through
        if not all(isinstance(item, float) for item in v):
            raise ValueError(f"embeddings must be a list of floats, got {v}")
        return v

    def to_vector_db(self):
        metadata = {
            "chunk_id": self.id,
            "chunk_index": self.chunk_index or "",
            "document_id": self.document_id,
            "doc_url": self.doc_url,
            "content": self.content,
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title or "",
            "token_count": self.token_count,
            **(self.metadata or {}),
        }
        result = {
            "id": self.id,
            "values": self.dense_embedding,
            "metadata": metadata,
        }
        return result
