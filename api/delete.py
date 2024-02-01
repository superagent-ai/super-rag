from fastapi import APIRouter, Depends

from auth.user import get_current_api_user
from models.delete import RequestPayload, ResponsePayload
from service.vector_database import VectorService, get_vector_service

router = APIRouter()


@router.delete("/delete", response_model=ResponsePayload)
async def delete(payload: RequestPayload, _api_user=Depends(get_current_api_user)):
    vector_service: VectorService = get_vector_service(
        index_name=payload.index_name, credentials=payload.vector_database
    )
    await vector_service.delete(file_url=payload.file_url)
    return {"success": True}
