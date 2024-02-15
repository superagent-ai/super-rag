from pydantic import BaseModel

from models.ingest import Encoder
from models.vector_database import VectorDatabase


class RequestPayload(BaseModel):
    index_name: str
    file_url: str
    vector_database: VectorDatabase
    encoder: Encoder


class DeleteResponse(BaseModel):
    num_of_deleted_chunks: int


class ResponsePayload(BaseModel):
    success: bool
    data: DeleteResponse
