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
        dimensions=payload.encoder.dimensions,
    )
    chunks = await embedding_service.generate_chunks()
    encoder = get_encoder(encoder_config=payload.encoder)
    # summary_documents = await embedding_service.generate_summary_documents(
    #     documents=chunks
    # )

    await asyncio.gather(
        embedding_service.generate_and_upsert_embeddings(
            documents=chunks, encoder=encoder, index_name=payload.index_name
        ),
        # embedding_service.generate_and_upsert_embeddings(
        #     documents=summary_documents,
        #     encoder=encoder,
        #     index_name=f"{payload.index_name}{SUMMARY_SUFFIX}",
        # ),
    )

    if payload.webhook_url:
        async with aiohttp.ClientSession() as session:
            await session.post(
                url=payload.webhook_url,
                json={"index_name": payload.index_name, "status": "completed"},
            )

    return {"success": True, "index_name": payload.index_name}
