from fastapi import APIRouter

from models.query import RequestPayload, ResponsePayload
from service.router import query as _query

router = APIRouter()


@router.post("/query", response_model=ResponsePayload)
async def query(payload: RequestPayload):
    chunks = await _query(payload=payload)
    # NOTE: Filter out fields before given to LLM
    return ResponsePayload(success=True, data=chunks)
