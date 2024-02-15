from fastapi import APIRouter

from models.delete import RequestPayload, ResponsePayload
from service.embedding import get_encoder
from service.vector_database import VectorService, get_vector_service

router = APIRouter()


@router.delete("/delete", response_model=ResponsePayload)
async def delete(payload: RequestPayload):
    encoder = get_encoder(encoder_type=payload.encoder)
    vector_service: VectorService = get_vector_service(
        index_name=payload.index_name,
        credentials=payload.vector_database,
        encoder=encoder,
    )

    for file in payload.files:
        data = await vector_service.delete(file_url=file.url)

    return ResponsePayload(success=True, data=data)
