from enum import Enum
from typing import Dict

from pydantic import BaseModel


class DatabaseType(Enum):
    qdrant = "qdrant"
    pinecone = "pinecone"
    weaviate = "weaviate"
    astra = "astra"
    pgvector = "pgvector"


class VectorDatabase(BaseModel):
    type: DatabaseType
    config: Dict
