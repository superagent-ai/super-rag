from typing import Dict, List
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from decouple import config

app = FastAPI(
    title="SuperRag",
    docs_url="/",
    description="The superagent RAG pipeline",
    version="0.0.1",
    servers=[{"url": config("API_BASE_URL")}],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DatabaseType(Enum):
    qdrant = "qdrant"
    pinecone = "pinecone"
    weaviate = "weaviate"
    astra = "astra"


class VectorDatabase(BaseModel):
    type: DatabaseType
    config: Dict


class RequestPayload(BaseModel):
    files: List
    vector_database: VectorDatabase


@app.post("/ingest")
async def ingest(payload: RequestPayload) -> Dict:
    return payload.model_dump()
