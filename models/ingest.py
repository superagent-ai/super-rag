from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from models.file import File
from models.vector_database import VectorDatabase


class EncoderEnum(str, Enum):
    cohere = "cohere"
    openai = "openai"
    huggingface = "huggingface"
    fastembed = "fastembed"


class RequestPayload(BaseModel):
    files: List[File]
    encoder: EncoderEnum
    vector_database: VectorDatabase
    index_name: str
    webhook_url: Optional[str] = None
