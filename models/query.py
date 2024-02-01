from typing import List, Optional

from pydantic import BaseModel

from models.vector_database import VectorDatabase


class RequestPayload(BaseModel):
    input: str
    vector_database: VectorDatabase
    index_name: str


class ResponseData(BaseModel):
    content: str
    file_url: str
    page_label: Optional[str]


class ResponsePayload(BaseModel):
    success: bool
    data: List[ResponseData]
