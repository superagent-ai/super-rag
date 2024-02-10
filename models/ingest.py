from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from models.file import File
from models.vector_database import VectorDatabase


# Step 1: Define the Encoder Enum
class EncoderEnum(str, Enum):
    cohere = "cohere"
    openai = "openai"
    huggingface = "huggingface"
    fastembed = "fastembed"


# Step 2: Use the Enum in RequestPayload
class RequestPayload(BaseModel):
    files: List[File]
    encoder: EncoderEnum
    vector_database: VectorDatabase
    index_name: str
    webhook_url: Optional[str] = None
