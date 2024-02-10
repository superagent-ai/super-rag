import asyncio
from typing import Dict

import aiohttp
from fastapi import APIRouter

from models.ingest import RequestPayload
from service.embedding import EmbeddingService

router = APIRouter()


@router.post("/ingest")
async def ingest(payload: RequestPayload) -> Dict:
    embedding_service = EmbeddingService(
        files=payload.files,
        index_name=payload.index_name,
        vector_credentials=payload.vector_database,
    )
    documents = await embedding_service.generate_documents()
    summary_documents = await embedding_service.generate_summary_documents(
        documents=documents
    )
    chunks, summary_chunks = await asyncio.gather(
        embedding_service.generate_chunks(documents=documents),
        embedding_service.generate_chunks(documents=summary_documents),
    )

    await asyncio.gather(
        embedding_service.generate_embeddings(nodes=chunks),
        embedding_service.generate_embeddings(
            nodes=summary_chunks, index_name=f"{payload.index_name}summary"
        ),
    )

    if payload.webhook_url:
        async with aiohttp.ClientSession() as session:
            await session.post(
                url=payload.webhook_url,
                json={"index_name": payload.index_name, "status": "completed"},
            )

    return {"success": True, "index_name": payload.index_name}
