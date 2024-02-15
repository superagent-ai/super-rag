from pydantic import BaseModel

from models.ingest import EncoderEnum
from models.vector_database import VectorDatabase


class File(BaseModel):
    url: str


class RequestPayload(BaseModel):
    index_name: str
    files: list[File]
    vector_database: VectorDatabase
    encoder: EncoderEnum


class DeleteResponse(BaseModel):
    num_of_deleted_chunks: int


class ResponsePayload(BaseModel):
    success: bool
    data: DeleteResponse
