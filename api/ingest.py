import asyncio
from typing import Dict

import aiohttp
from fastapi import APIRouter

from models.ingest import RequestPayload
from service.embedding import EmbeddingService, get_encoder
from service.ingest import handle_google_drive, handle_urls
from utils.summarise import SUMMARY_SUFFIX

router = APIRouter()


@router.post("/ingest")
async def ingest(payload: RequestPayload) -> Dict:
    encoder = get_encoder(encoder_config=payload.encoder)
    embedding_service = EmbeddingService(
        encoder=encoder,
        index_name=payload.index_name,
        vector_credentials=payload.vector_database,
        dimensions=payload.encoder.dimensions,
    )
    chunks = []
    summary_documents = []
    if payload.files:
        chunks, summary_documents = await handle_urls(
            embedding_service, payload.files, payload.chunk_config
        )

    elif payload.google_drive:
        chunks, summary_documents = await handle_google_drive(
            embedding_service, payload.google_drive
        )  # type: ignore TODO: Fix typing

    await asyncio.gather(
        embedding_service.embed_and_upsert(
            chunks=chunks, encoder=encoder, index_name=payload.index_name
        ),
        embedding_service.embed_and_upsert(
            chunks=summary_documents,
            encoder=encoder,
            index_name=f"{payload.index_name}{SUMMARY_SUFFIX}",
        ),
    )

    if payload.webhook_url:
        async with aiohttp.ClientSession() as session:
            await session.post(
                url=payload.webhook_url,
                json={"index_name": payload.index_name, "status": "completed"},
            )

    return {"success": True, "index_name": payload.index_name}
