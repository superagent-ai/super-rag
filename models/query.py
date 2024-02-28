from typing import List, Optional

from pydantic import BaseModel

from models.document import BaseDocumentChunk
from models.ingest import EncoderConfig
from models.vector_database import VectorDatabase


class RequestPayload(BaseModel):
    input: str
    vector_database: VectorDatabase
    index_name: str
    encoder: EncoderConfig = EncoderConfig()
    session_id: Optional[str] = None


class ResponsePayload(BaseModel):
    success: bool
    data: List[BaseDocumentChunk]
