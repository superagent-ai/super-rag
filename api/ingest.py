import asyncio
from typing import Dict

import aiohttp
from fastapi import APIRouter

from models.ingest import RequestPayload
from service.embedding import EmbeddingService, get_encoder
from utils.summarise import SUMMARY_SUFFIX

router = APIRouter()


@router.post("/ingest")
async def ingest(payload: RequestPayload) -> Dict:
    embedding_service = EmbeddingService(
        files=payload.files,
        index_name=payload.index_name,
        vector_credentials=payload.vector_database,
    )
    documents = await embedding_service.generate_documents()
    chunks = await embedding_service.generate_chunks(documents=documents)

    encoder = get_encoder(encoder_type=payload.encoder)

    summary_documents = await embedding_service.generate_summary_documents(
        documents=documents
    )
    chunks, summary_chunks = await asyncio.gather(
        embedding_service.generate_chunks(documents=documents),
        embedding_service.generate_chunks(documents=summary_documents),
    )

    await asyncio.gather(
        embedding_service.generate_embeddings(
            nodes=chunks, encoder=encoder, index_name=payload.index_name
        ),
        embedding_service.generate_embeddings(
            nodes=summary_chunks,
            encoder=encoder,
            index_name=f"{payload.index_name}-{SUMMARY_SUFFIX}",
        ),
    )

    if payload.webhook_url:
        async with aiohttp.ClientSession() as session:
            await session.post(
                url=payload.webhook_url,
                json={"index_name": payload.index_name, "status": "completed"},
            )

    return {"success": True, "index_name": payload.index_name}
