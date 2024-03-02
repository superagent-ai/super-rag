from fastapi import APIRouter

from models.delete import RequestPayload, ResponsePayload
from vectordbs import get_vector_service
from vectordbs.base import BaseVectorDatabase

router = APIRouter()


@router.delete("/delete", response_model=ResponsePayload)
async def delete(payload: RequestPayload):
    encoder = payload.encoder.get_encoder()
    vector_service: BaseVectorDatabase = get_vector_service(
        index_name=payload.index_name,
        credentials=payload.vector_database,
        encoder=encoder,
        dimensions=payload.encoder.dimensions,
    )

    for file in payload.files:
        data = await vector_service.delete(file_url=file.url)

    return ResponsePayload(success=True, data=data)
