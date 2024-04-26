from pydantic import BaseModel
from typing import List, Optional

from models.document import BaseDocumentChunk
from models.ingest import EncoderConfig
from models.vector_database import VectorDatabase
from qdrant_client.http.models import Filter


class RequestPayload(BaseModel):
    input: str
    vector_database: VectorDatabase
    index_name: str
    encoder: EncoderConfig = EncoderConfig()
    session_id: Optional[str] = None
    interpreter_mode: Optional[bool] = False
    exclude_fields: List[str] = None
    # TODO: use our own Filter model
    filter: Optional[Filter] = None


class ResponseData(BaseModel):
    content: str
    doc_url: str
    page_number: Optional[int]
    metadata: Optional[dict] = None


class ResponsePayload(BaseModel):
    success: bool
    data: List[BaseDocumentChunk]

    def model_dump(self, exclude: set = None):
        return {
            "success": self.success,
            "data": [chunk.dict(exclude=exclude) for chunk in self.data],
        }
