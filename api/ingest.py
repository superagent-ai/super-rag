import asyncio
from typing import Dict

import aiohttp
from fastapi import APIRouter

import encoders
from models.ingest import EncoderEnum, RequestPayload
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
    chunks = await embedding_service.generate_chunks(documents=documents)

    encoder_mapping = {
        EncoderEnum.cohere: encoders.CohereEncoder,
        EncoderEnum.openai: encoders.OpenAIEncoder,
        EncoderEnum.huggingface: encoders.HuggingFaceEncoder,
        EncoderEnum.fastembed: encoders.FastEmbedEncoder,
    }

    encoder_class = encoder_mapping.get(payload.encoder)
    if encoder_class is None:
        raise ValueError(f"Unsupported encoder: {payload.encoder}")
    encoder = encoder_class()

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
            index_name=f"{payload.index_name}-summary",
        ),
    )

    if payload.webhook_url:
        async with aiohttp.ClientSession() as session:
            await session.post(
                url=payload.webhook_url,
                json={"index_name": payload.index_name, "status": "completed"},
            )

    return {"success": True, "index_name": payload.index_name}
