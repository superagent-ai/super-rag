from fastapi import APIRouter

from api import ingest

router = APIRouter()
api_prefix = "/api/v1"

router.include_router(ingest.router, tags=["Ingest"], prefix=api_prefix)
