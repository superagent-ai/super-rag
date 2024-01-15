from typing import Dict
from fastapi import APIRouter
from models.ingest import RequestPayload
from service.embedding import EmbeddingService

router = APIRouter()


@router.post("/ingest")
async def ingest(payload: RequestPayload) -> Dict:
    embeddings = EmbeddingService(files=payload.files, index_name=payload.index_name)
    documents = await embeddings.generate_documents()
    return {"success": True, "data": documents}
