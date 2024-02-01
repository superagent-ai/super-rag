from fastapi import APIRouter, Depends

from auth.user import get_current_api_user
from models.query import RequestPayload, ResponsePayload
from service.vector_database import VectorService, get_vector_service

router = APIRouter()


@router.post("/query", response_model=ResponsePayload)
async def query(payload: RequestPayload, _api_user=Depends(get_current_api_user)):
    vector_service: VectorService = get_vector_service(
        index_name=payload.index_name, credentials=payload.vector_database
    )
    chunks = await vector_service.query(input=payload.input, top_k=4)
    documents = await vector_service.convert_to_rerank_format(chunks=chunks)
    if len(documents):
        documents = await vector_service.rerank(
            query=payload.input, documents=documents
        )
    return {"success": True, "data": documents}
