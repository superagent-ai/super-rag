from pydantic import BaseModel
from typing import List
from models.vector_database import VectorDatabase


class RequestPayload(BaseModel):
    input: str
    vector_database: VectorDatabase
    index_name: str


class ResponsePayload(BaseModel):
    success: bool
    data: List
