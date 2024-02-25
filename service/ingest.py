from typing import List

from models.file import File
from models.google_drive import GoogleDrive
from models.ingest import ChunkConfig
from service.embedding import EmbeddingService


async def handle_urls(
    embedding_service: EmbeddingService, files: List[File], config: ChunkConfig
):
    embedding_service.files = files
    chunks = await embedding_service.generate_chunks(
        partition_strategy=config.partition_strategy,
        split_method=config.split_method,
        min_chunk_tokens=config.min_chunk_tokens,
        max_token_size=config.max_token_size,
        rolling_window_size=config.rolling_window_size,
    )
    summary_documents = await embedding_service.generate_summary_documents(
        documents=chunks
    )
    return chunks, summary_documents


async def handle_google_drive(
    _embedding_service: EmbeddingService, _google_drive: GoogleDrive
):
    pass
