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
    doc_url: str | None = None
    document_id: str
    content: str
    source: str | None = None
    source_type: str | None = None
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
        # Prepare metadata for the constructor and for embedding into the object
        constructor_metadata = {
            k: v for k, v in metadata.items() if k not in exclude_keys
        }
        filtered_metadata = {
            k: v for k, v in metadata.items() if k in exclude_keys and k != "chunk_id"
        }

        def to_int(value):
            try:
                return int(value) if str(value).isdigit() else None
            except (TypeError, ValueError):
                return None

        chunk_index = to_int(metadata.get("chunk_index"))
        token_count = to_int(metadata.get("token_count"))

        # Remove explicitly passed keys from filtered_metadata to avoid duplication
        for key in ["chunk_index", "token_count"]:
            filtered_metadata.pop(key, None)

        return cls(
            id=metadata.get("chunk_id", ""),
            chunk_index=chunk_index,
            token_count=token_count,
            **filtered_metadata,  # Pass filtered metadata for constructor
            metadata=constructor_metadata,  # Pass the rest as part of the metadata
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
