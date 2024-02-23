from typing import List

from models.file import File
from models.google_drive import GoogleDrive
from models.aws_s3 import AwsS3
from service.embedding import EmbeddingService


async def handle_urls(
    embedding_service: EmbeddingService,
    files: List[File],
):
    embedding_service.files = files
    chunks = await embedding_service.generate_chunks()
    summary_documents = await embedding_service.generate_summary_documents(
        documents=chunks
    )
    return chunks, summary_documents


async def handle_google_drive(
    _embedding_service: EmbeddingService, _google_drive: GoogleDrive
):
    pass


async def handle_s3(_embedding_service: EmbeddingService, _aws_S3: AwsS3):
    print("INGESTION")
    pass
