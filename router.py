from fastapi import APIRouter

from api import ingest, query

router = APIRouter()
api_prefix = "/api/v1"

router.include_router(ingest.router, tags=["Ingest"], prefix=api_prefix)
router.include_router(query.router, tags=["Query"], prefix=api_prefix)
