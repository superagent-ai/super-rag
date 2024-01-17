from fastapi import APIRouter
from models.query import RequestPayload, ResponsePayload
from service.vector_database import get_vector_service, VectorService

router = APIRouter()


@router.post("/query", response_model=ResponsePayload)
async def query(payload: RequestPayload):
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
