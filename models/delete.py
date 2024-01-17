from pydantic import BaseModel
from models.vector_database import VectorDatabase


class RequestPayload(BaseModel):
    index_name: str
    file_url: str
    vector_database: VectorDatabase


class ResponsePayload(BaseModel):
    success: bool
