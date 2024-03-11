from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from semantic_router.encoders import BaseEncoder, CohereEncoder, OpenAIEncoder

from models.file import File
from models.google_drive import GoogleDrive
from models.vector_database import VectorDatabase
from models.api import ApiError


class EncoderProvider(str, Enum):
    cohere = "cohere"
    openai = "openai"


class EncoderConfig(BaseModel):
    provider: EncoderProvider = Field(
        default=EncoderProvider.cohere, description="Embedding provider"
    )
    model_name: str = Field(
        default="embed-multilingual-light-v3.0",
        description="Model name for the encoder",
    )
    dimensions: int = Field(default=384, description="Dimension of the encoder output")

    _encoder_config = {
        EncoderProvider.cohere: {
            "class": CohereEncoder,
            "default_model_name": "embed-multilingual-light-v3.0",
            "default_dimensions": 384,
        },
        EncoderProvider.openai: {
            "class": OpenAIEncoder,
            "default_model_name": "text-embedding-3-small",
            "default_dimensions": 1536,
        },
    }

    def get_encoder(self) -> BaseEncoder:
        config = self._encoder_config.get(self.provider)
        if not config:
            raise ValueError(f"Encoder '{self.provider}' not found.")
        model_name = self.model_name or config["default_model_name"]
        encoder_class = config["class"]
        return encoder_class(name=model_name)


class UnstructuredConfig(BaseModel):
    partition_strategy: Literal["auto", "hi_res"] = Field(default="auto")
    hi_res_model_name: Literal["detectron2_onnx", "chipper"] = Field(
        default="detectron2_onnx", description="Only for `hi_res` strategy"
    )
    process_tables: bool = Field(
        default=False, description="Only for `hi_res` strategy"
    )


class SplitterConfig(BaseModel):
    name: Literal["semantic", "by_title"] = Field(
        default="semantic", description="Splitter name, `semantic` or `by_title`"
    )
    min_tokens: int = Field(default=30, description="Only for `semantic` method")
    max_tokens: int = Field(
        default=400, description="Only for `semantic` and `recursive` methods"
    )
    rolling_window_size: int = Field(
        default=1,
        description="Only for `semantic` method, cumulative window size "
        "for comparing similarity between elements",
    )
    prefix_title: bool = Field(
        default=True, description="Add to split titles, headers, only `semantic` method"
    )
    prefix_summary: bool = Field(
        default=True, description="Add to split sub-document summary"
    )


class DocumentProcessorConfig(BaseModel):
    encoder: EncoderConfig = EncoderConfig()
    unstructured: UnstructuredConfig = UnstructuredConfig()
    splitter: SplitterConfig = SplitterConfig()


class RequestPayload(BaseModel):
    index_name: str
    vector_database: VectorDatabase
    document_processor: DocumentProcessorConfig = DocumentProcessorConfig()
    files: Optional[List[File]] = None
    google_drive: Optional[GoogleDrive] = None
    webhook_url: Optional[str] = None


class TaskStatus(str, Enum):
    DONE = "DONE"
    PENDING = "PENDING"
    FAILED = "FAILED"


class IngestTaskResponse(BaseModel):
    status: TaskStatus
    error: Optional[ApiError] = None
