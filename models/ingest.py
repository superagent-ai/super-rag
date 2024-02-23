from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

from models.file import File
from models.vector_database import VectorDatabase
from models.google_drive import GoogleDrive
from models.aws_s3 import AwsS3


class EncoderEnum(str, Enum):
    cohere = "cohere"
    openai = "openai"


class Encoder(BaseModel):
    name: str
    type: str
    dimensions: Optional[int] = None


class RequestPayload(BaseModel):
    files: Optional[List[File]] = None
    google_drive: Optional[GoogleDrive] = None
    aws_s3: Optional[AwsS3] = None
    encoder: Encoder
    vector_database: VectorDatabase
    index_name: str
    webhook_url: Optional[str] = None
