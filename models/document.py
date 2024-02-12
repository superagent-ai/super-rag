from pydantic import BaseModel


class BaseDocument(BaseModel):
    id: str
    text: str
    doc_url: str
    metadata: dict | None = None


class BaseDocumentChunk(BaseDocument):
    document_id: str
    dense_embedding: list[float] | None = None

    def to_pinecone(self):
        metadata = {
            "document_id": self.document_id,
            "text": self.text,
            "doc_url": self.doc_url,
            **(self.metadata or {}),
        }
        result = {
            "id": self.id,
            "values": self.dense_embedding,
            "metadata": metadata,
        }
        return result
