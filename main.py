from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from decouple import config
from router import router


app = FastAPI(
    title="SuperRag",
    docs_url="/",
    description="The superagent RAG pipeline",
    version="0.0.1",
    servers=[{"url": config("API_BASE_URL")}],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
