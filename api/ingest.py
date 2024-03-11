import asyncio
from typing import Dict

import aiohttp
from fastapi import APIRouter

from models.ingest import RequestPayload, TaskStatus
from models.api import ApiError
from service.embedding import EmbeddingService
from service.ingest import handle_google_drive, handle_urls
from utils.summarise import SUMMARY_SUFFIX

from service.redis.ingest_task_manager import (
    IngestTaskManager,
    CreateTaskDto,
    UpdateTaskDto,
)
from service.redis.client import redis_client

from fastapi.responses import JSONResponse
from fastapi import status
from service.kafka.config import kafka_bootstrap_servers, ingest_topic
from service.kafka.producer import kafka_producer

router = APIRouter()


class IngestPayload(RequestPayload):
    task_id: str


@router.post("/ingest")
async def add_ingest_queue(payload: RequestPayload):
    try:
        task_manager = IngestTaskManager(redis_client)
        task_id = task_manager.create(CreateTaskDto(status=TaskStatus.PENDING))
        print("Task ID: ", task_id)

        message = IngestPayload(**payload.model_dump(), task_id=str(task_id))

        msg = message.model_dump_json().encode()

        kafka_producer.send(ingest_topic, msg)
        kafka_producer.flush()

        return {"success": True, "task_id": task_id}

    except Exception as err:
        print(f"error: {err}")


@router.get("/ingest/tasks/{task_id}")
async def get_task(task_id: str):
    task_manager = IngestTaskManager(redis_client)

    task = task_manager.get(task_id)

    if task:
        return task

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"sucess": False, "error": {"message": "Task not found"}},
    )


async def ingest(payload: IngestPayload, task_manager: IngestTaskManager) -> Dict:
    try:
        encoder = payload.document_processor.encoder.get_encoder()
        embedding_service = EmbeddingService(
            encoder=encoder,
            index_name=payload.index_name,
            vector_credentials=payload.vector_database,
            dimensions=payload.document_processor.encoder.dimensions,
        )
        chunks = []
        summary_documents = []
        if payload.files:
            chunks, summary_documents = await handle_urls(
                embedding_service=embedding_service,
                files=payload.files,
                config=payload.document_processor,
            )

        elif payload.google_drive:
            chunks, summary_documents = await handle_google_drive(
                embedding_service, payload.google_drive
            )  # type: ignore TODO: Fix typing

        tasks = [
            embedding_service.embed_and_upsert(
                chunks=chunks, encoder=encoder, index_name=payload.index_name
            ),
        ]

        if summary_documents and all(item is not None for item in summary_documents):
            tasks.append(
                embedding_service.embed_and_upsert(
                    chunks=summary_documents,
                    encoder=encoder,
                    index_name=f"{payload.index_name}{SUMMARY_SUFFIX}",
                )
            )

        await asyncio.gather(*tasks)

        if payload.webhook_url:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    url=payload.webhook_url,
                    json={"index_name": payload.index_name, "status": "completed"},
                )

        task_manager.update(
            task_id=payload.task_id,
            task=UpdateTaskDto(
                status=TaskStatus.DONE,
            ),
        )
    except Exception as e:
        print("Marking task as failed...", e)
        task_manager.update(
            task_id=payload.task_id,
            task=UpdateTaskDto(
                status=TaskStatus.FAILED,
                error=ApiError(message=str(e)),
            ),
        )
