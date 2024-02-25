from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from models.file import File
from models.google_drive import GoogleDrive
from models.vector_database import VectorDatabase


class ChunkConfig(BaseModel):
    partition_strategy: Literal["auto", "hi_res"] = "auto"
    split_method: Literal["by_title", "semantic"] = "by_title"
    min_chunk_tokens: int = Field(50, description="Only for `semantic` method")
    max_token_size: int = Field(300, description="Only for `semantic` method")
    rolling_window_size: int = Field(
        1,
        description=(
            "Only for `semantic` method. Compares each element with the previous one"
        ),
    )


class EncoderEnum(str, Enum):
    cohere = "cohere"
    openai = "openai"


class Encoder(BaseModel):
    name: str
    provider: str
    dimensions: Optional[int] = None


class RequestPayload(BaseModel):
    index_name: str
    encoder: Encoder
    vector_database: VectorDatabase
    chunk_config: ChunkConfig
    files: Optional[List[File]] = None
    google_drive: Optional[GoogleDrive] = None
    webhook_url: Optional[str] = None
