from typing import List, Optional

from pydantic import BaseModel

from models.document import BaseDocumentChunk
from models.ingest import Encoder
from models.vector_database import VectorDatabase


class RequestPayload(BaseModel):
    input: str
    vector_database: VectorDatabase
    index_name: str
    encoder: Encoder
    session_id: Optional[str] = None


# class ResponseData(BaseModel):
#     content: str
#     doc_url: str
#     page_number: Optional[int]


class ResponsePayload(BaseModel):
    success: bool
    data: List[BaseDocumentChunk]
