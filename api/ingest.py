from typing import Dict, List
from enum import Enum
from pydantic import BaseModel
from fastapi import APIRouter


router = APIRouter()


class DatabaseType(Enum):
    qdrant = "qdrant"
    pinecone = "pinecone"
    weaviate = "weaviate"
    astra = "astra"


class VectorDatabase(BaseModel):
    type: DatabaseType
    config: Dict


class RequestPayload(BaseModel):
    files: List[str]
    vector_database: VectorDatabase


@router.post("/ingest")
async def ingest(payload: RequestPayload) -> Dict:
    return payload.model_dump()
