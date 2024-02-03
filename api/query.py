from fastapi import APIRouter

from models.query import RequestPayload, ResponsePayload
from service.router import query as _query

router = APIRouter()


@router.post("/query", response_model=ResponsePayload)
async def query(payload: RequestPayload):
    output = await _query(payload=payload)
    return {"success": True, "data": output}
