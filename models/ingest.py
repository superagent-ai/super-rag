from typing import List
from pydantic import BaseModel
from models.file import File
from models.vector_database import VectorDatabase


class RequestPayload(BaseModel):
    files: List[File]
    vector_database: VectorDatabase
    index_name: str
