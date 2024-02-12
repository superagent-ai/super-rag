from fastapi import APIRouter

from models.query import RequestPayload, ResponseData, ResponsePayload
from service.router import query as _query

router = APIRouter()


@router.post("/query", response_model=ResponsePayload)
async def query(payload: RequestPayload):
    chunks = await _query(payload=payload)
    response_data = [
        ResponseData(
            content=chunk.content, doc_url=chunk.doc_url, page_label=chunk.page_number
        )
        for chunk in chunks
    ]
    return {"success": True, "data": response_data}
