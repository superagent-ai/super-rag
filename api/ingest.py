from typing import Dict

import requests
from fastapi import APIRouter

import encoders
from models.ingest import EncoderEnum, RequestPayload
from service.embedding import EmbeddingService

router = APIRouter()


# Ensure you import the encoders module or specific encoder classes


@router.post("/ingest")
async def ingest(payload: RequestPayload) -> Dict:
    embedding_service = EmbeddingService(
        files=payload.files,
        index_name=payload.index_name,
        vector_credentials=payload.vector_database,
    )
    documents = await embedding_service.generate_documents()
    chunks = await embedding_service.generate_chunks(documents=documents)

    # Encoder selection based on the payload's encoder value
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

    await embedding_service.generate_embeddings(nodes=chunks, encoder=encoder)

    if payload.webhook_url:
        requests.post(
            url=payload.webhook_url,
            json={"index_name": payload.index_name, "status": "completed"},
        )
    return {"success": True, "index_name": payload.index_name}
