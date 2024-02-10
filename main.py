from decouple import config
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import router

load_dotenv()

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
